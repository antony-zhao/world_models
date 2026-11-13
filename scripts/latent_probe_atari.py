"""What does a trained world model's latent know about the sprites, and how fast does
open-loop imagination drift?

Rolls out the trained agent, reading sprite positions from Atari RAM through a shadow
`ALEInterface` kept in lockstep with the vector env (same trick as
ram_periodicity_analysis.py; needs noop_max=0), then reports:

  1. latent statistics: per-dim variance, participation ratio, mean squared difference
     of z by lag, jitter under the training augmentation
  2. one-step prior error against the copy-z_t baseline
  3. open-loop imagination error by horizon, rolling the prior forward on the *true*
     actions (z has ~unit variance per dim under SIGReg, so an MSE of ~1 means the
     prediction is no better than the mean latent)
  4. saliency: share of ||dz/dpixels||^2 inside a box around each sprite, against the
     box's share of the image area (games with a `pixel_map` only)
  5. position probes (ridge and MLP) on the latent z and the conv embedding e. The MLP
     probe overfits at these data sizes (it did so from raw pixels too), so read the
     numbers relative to a `--random-init` run of the same script, not as absolutes.

Usage:
    python scripts/latent_probe_atari.py --run runs/atari100k/<run_dir>
    python scripts/latent_probe_atari.py --run runs/atari100k/<run_dir> --random-init
"""

import argparse
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from ale_py import ALEInterface, LoggerMode, roms
from omegaconf import OmegaConf
from torch import nn

from world_models.torch.agents.agent import Agent
from world_models.torch.common.augments import random_shift
from world_models.torch.common.distributions import DiscreteLatentDist
from world_models.utils.config import preprocess_config

gym.register_envs(ale_py)

# RAM offsets from mila-iqia/atari-representation-learning's ram_annotations.py.
# ms_pacman player_x/player_y were checked against the centroid of Pacman's yellow pixels
# in the 64x64 frame (corr 0.997); `pixel_map` is the linear fit from that check,
# (slope, intercept) per axis, residual ~0.5 px. The ghosts are assumed to share it.
GAMES = {
    "ms_pacman": {
        "sprites": {
            "pacman": (10, 16),
            "ghost_sue": (6, 12),
            "ghost_inky": (7, 13),
            "ghost_pinky": (8, 14),
            "ghost_blinky": (9, 15),
        },
        "pixel_map": ((0.399, -3.6), (0.303, 1.7)),
    },
    "breakout": {"sprites": {"ball": (99, 101), "paddle": (72, None)}, "pixel_map": None},
}


def load_agent(run_dir, checkpoint, device, random_init=False):
    run_dir = Path(run_dir)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval, use_cache=False)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.num_eval_envs = 1
    cfg.device = device
    env = build_env(cfg.env.game)
    cfg.action_dim = int(env.single_action_space.n)
    env.close()
    cfg.obs_shape = (3, 64, 64)
    cfg = preprocess_config(cfg)
    agent = Agent(cfg).to(device)
    if random_init:
        print("random init (no checkpoint loaded)")
    else:
        ckpt = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
        agent.world_model.load_state_dict(ckpt["world_model"])
        agent.actor.load_state_dict(ckpt["actor"])
        print(f"loaded {run_dir.name}/{checkpoint} at env_step={ckpt['env_step']}")
    agent.world_model.eval()
    agent.actor.eval()
    return agent, cfg


def build_env(game):
    # Same preprocessing as train.build_envs, except noop_max=0: with sticky actions already
    # off, that makes the env deterministic given the actions, which the shadow emulator needs.
    return ale_py.AtariVectorEnv(
        game=game, num_envs=1, img_height=64, img_width=64, grayscale=False, stack_num=1,
        frameskip=4, maxpool=True, noop_max=0, episodic_life=False, life_loss_info=False,
        reward_clipping=False, repeat_action_probability=0.0, use_fire_reset=False,
        full_action_space=False, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )  # fmt: skip


@torch.no_grad()
def collect(agent, game, steps, seed=0):
    """Roll out the agent's stochastic policy. Returns obs (N, 3, 64, 64) uint8, actions (N,),
    ram (N, 128) read at the same step as each obs, and boundary (N,), true where a life loss
    or game over happens between step t and t + 1."""
    env = build_env(game)
    shadow = ALEInterface()
    shadow.setLoggerMode(LoggerMode.Error)
    shadow.setFloat("repeat_action_probability", 0.0)
    shadow.loadROM(roms.get_rom_path(game))
    action_set = shadow.getMinimalActionSet()

    obs, _ = env.reset(seed=seed)
    shadow.reset_game()
    agent.reset_eval_state()
    obs_list, actions, rams, boundary, returns = [], [], [], [], []
    episode_return = 0.0
    for _ in range(steps):
        frame = np.transpose(obs[:, 0], (0, 3, 1, 2))
        obs_list.append(frame[0].copy())
        rams.append(shadow.getRAM().copy())
        lives = shadow.lives()
        action = agent.eval_act(frame.astype(np.float32), det=False)
        obs, reward, term, trunc, _ = env.step(action)
        for _ in range(4):
            shadow.act(action_set[int(action[0])])
        game_over = shadow.game_over()
        assert game_over == bool(term[0] or trunc[0]), "shadow emulator lost lockstep"
        boundary.append(game_over or shadow.lives() < lives)
        actions.append(int(action[0]))
        episode_return += float(reward[0])
        if game_over:
            shadow.reset_game()
            returns.append(episode_return)
            episode_return = 0.0
    env.close()
    mean_return = f"{np.mean(returns):.0f}" if returns else "n/a"
    print(f"collected {steps} steps: {len(returns)} full games, mean return {mean_return}")
    return {
        "obs": np.array(obs_list),
        "actions": np.array(actions),
        "ram": np.array(rams).astype(np.float32),
        "boundary": np.array(boundary),
    }


@torch.no_grad()
def encode(wm, obs, device, batch=2048):
    """obs (N, 3, 64, 64) uint8 -> conv embedding e (N, E), posterior-mode latent z (N, D)."""
    embeds, latents = [], []
    for i in range(0, len(obs), batch):
        x = torch.from_numpy(obs[i : i + batch]).float().to(device) / 255.0 - 0.5
        e = wm.encoder(x)
        embeds.append(e)
        latents.append(wm.posterior(e).mode.flatten(1))
    return torch.cat(embeds), torch.cat(latents)


def imagination_windows(boundary, context, horizon, stride, start=0):
    """Start indices of windows of context + horizon + 1 steps with no life loss / game over."""
    span = context + horizon
    starts = [s for s in range(start, len(boundary) - span - 1, stride)]
    return [s for s in starts if not boundary[s : s + span].any()]


@torch.no_grad()
def open_loop(wm, z, actions_onehot, starts, context, horizon):
    """Feed `context` real latents, then roll the prior forward for `horizon` steps on the
    true actions. Returns imagined and real latents for steps context+1 .. context+horizon,
    both (W, horizon, D)."""
    device = z.device
    idx = torch.tensor(starts, device=device)[:, None] + torch.arange(
        context + horizon + 1, device=device
    )
    zs, acts = z[idx], actions_onehot[idx]
    state = wm.sequence_model.initial_state(len(starts), device)
    _, state = wm.sequence_model.parallel_forward(zs[:, :context], acts[:, :context], state=state)
    latent, imagined = zs[:, context], []
    for k in range(horizon):
        latent, _, state = wm.sequence_model.imagine_step(
            latent, acts[:, context + k], state, wm.prior
        )
        imagined.append(latent.flatten(1))
    return torch.stack(imagined, 1), zs[:, context + 1 :]


def sprite_boxes(ram, spec, half=2.5):
    """Boolean masks (N, 64, 64), one per sprite, of a (2*half)^2 px box around the sprite."""
    (ax, bx), (ay, by) = spec["pixel_map"]
    yy, xx = np.mgrid[0:64, 0:64]
    boxes = {}
    for name, (ix, iy) in spec["sprites"].items():
        cx = (ax * ram[:, ix] + bx)[:, None, None]
        cy = (ay * ram[:, iy] + by)[:, None, None]
        boxes[name] = (np.abs(xx[None] - cx) <= half) & (np.abs(yy[None] - cy) <= half)
    return boxes


def position_targets(ram, spec):
    names, cols = [], []
    for name, (ix, iy) in spec["sprites"].items():
        for axis, index in (("x", ix), ("y", iy)):
            if index is not None:
                names.append(f"{name}_{axis}")
                cols.append(ram[:, index])
    return names, np.stack(cols, 1)


def block_split(n, device, block=400, seed=0):
    """Train / val / test masks (70 / 10 / 20%) over contiguous blocks, so neighbouring
    frames never straddle the split."""
    gen = torch.Generator().manual_seed(seed)
    block_id = torch.arange(n) // block
    role = (torch.randperm(int(block_id.max()) + 1, generator=gen)[block_id] % 10).to(device)
    return role < 7, role == 7, role > 7


def ridge_probe(x, y, train, val, test):
    """Ridge regression, penalty picked on val per target. Returns test R^2 per target."""
    x, y = x.double(), y.double()
    x = (x - x[train].mean(0)) / (x[train].std(0) + 1e-6)
    y = (y - y[train].mean(0)) / (y[train].std(0) + 1e-6)
    gram = x[train] @ x[train].T
    eye = torch.eye(len(gram), device=x.device, dtype=x.dtype)

    def r2(weights, mask):
        return 1 - ((x[mask] @ weights - y[mask]) ** 2).mean(0) / y[mask].var(0)

    best_val, best_test = None, None
    for penalty in (1e-1, 1, 10, 100, 1e3, 1e4, 1e5):
        weights = x[train].T @ torch.linalg.solve(gram + penalty * eye, y[train])
        val_r2, test_r2 = r2(weights, val), r2(weights, test)
        if best_val is None:
            best_val, best_test = val_r2, test_r2
        else:
            better = val_r2 > best_val
            best_val = torch.where(better, val_r2, best_val)
            best_test = torch.where(better, test_r2, best_test)
    return best_test.cpu().numpy()


def mlp_probe(x, y, train, val, test, steps=6000, batch=256, lr=1e-3):
    """Two-hidden-layer MLP regression, early-stopped on val. Returns test and train MAE per
    target, in the targets' own units."""
    x, y = x.float(), y.float()
    x = (x - x[train].mean(0)) / (x[train].std(0) + 1e-6)
    y_mean, y_std = y[train].mean(0), y[train].std(0) + 1e-6
    net = nn.Sequential(
        nn.Linear(x.shape[1], 1024), nn.LayerNorm(1024), nn.SiLU(),
        nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.SiLU(),
        nn.Linear(1024, y.shape[1]),
    ).to(x.device)  # fmt: skip
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    train_idx = torch.nonzero(train).squeeze(1)

    @torch.no_grad()
    def mae(mask):
        net.eval()
        error = ((net(x[mask]) * y_std + y_mean) - y[mask]).abs().mean(0)
        net.train()
        return error

    best = (float("inf"), None, None)
    for step in range(steps):
        idx = train_idx[torch.randint(0, len(train_idx), (batch,), device=x.device)]
        loss = F.smooth_l1_loss(net(x[idx]), (y[idx] - y_mean) / y_std)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            val_mae = mae(val).mean().item()
            if val_mae < best[0]:
                best = (val_mae, mae(test).cpu().numpy(), mae(train).cpu().numpy())
    return best[1], best[2]


def saliency_share(wm, obs, boxes, device, directions=16):
    """Mean share of ||d(v . z)/d pixels||^2, over random unit v, falling inside each mask."""
    x = (torch.from_numpy(obs).float().to(device) / 255.0 - 0.5).requires_grad_(True)
    energy = torch.zeros(len(obs), 64, 64, device=device)
    for _ in range(directions):
        dist = wm.posterior(wm.encoder(x))
        # a categorical mode is an argmax with no gradient, so differentiate its probabilities
        z = (dist.probs if isinstance(dist, DiscreteLatentDist) else dist.mode).flatten(1)
        v = torch.randn(z.shape[1], device=device)
        (grad,) = torch.autograd.grad((z @ (v / v.norm())).sum(), x)
        energy += (grad**2).sum(1)
    total = energy.sum((1, 2))
    shares = {}
    for name, mask in boxes.items():
        mask = torch.from_numpy(mask).to(device)
        shares[name] = (
            ((energy * mask).sum((1, 2)) / total).mean().item(),
            mask.float().mean().item(),
        )
    return shares


def fmt(values, digits=3):
    return " ".join(f"{v:.{digits}f}" for v in values)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    agent, cfg = load_agent(args.run, args.checkpoint, device, args.random_init)
    wm = agent.world_model
    game = cfg.env.game
    spec = GAMES.get(game)

    data = collect(agent, game, args.steps, args.seed)
    obs, boundary, ram = data["obs"], data["boundary"], data["ram"]
    n = len(obs)
    e, z = encode(wm, obs, device)
    actions = F.one_hot(torch.from_numpy(data["actions"]).to(device), cfg.action_dim).float()
    valid = torch.from_numpy(~boundary).to(device)

    # ---- 1. latent statistics
    eig = torch.linalg.eigvalsh(torch.cov(z.T)).clamp(min=0)
    print("\n-- latent statistics")
    print(
        f"variance per dim {z.var(0).mean():.3f}   "
        f"participation ratio {(eig.sum() ** 2 / (eig**2).sum()).item():.0f}/{z.shape[1]}"
    )
    lags = [1, 2, 3, 4, 8, 16, 64]
    lag_diffs = [((z[lag:] - z[:-lag]) ** 2).mean().item() for lag in lags]
    print(f"mean sq diff of z at lags {lags} (2 * var = {2 * z.var(0).mean():.3f}): {fmt(lag_diffs)}")
    with torch.no_grad():
        x = torch.from_numpy(obs[:2048]).float().to(device) / 255.0 - 0.5
        z_aug = wm.posterior(wm.encoder(random_shift(1)(x)[0])).mode.flatten(1)
    print(f"augmentation jitter mse(z(aug x), z(x)): {((z_aug - z[:2048]) ** 2).mean():.4f}")

    # ---- 2. one-step prediction, teacher-forced over training-length chunks
    chunk = cfg.train.wm_seq_len
    num_chunks = n // chunk
    zc = z[: num_chunks * chunk].reshape(num_chunks, chunk, -1)
    ac = actions[: num_chunks * chunk].reshape(num_chunks, chunk, -1)
    mask = valid[: num_chunks * chunk].reshape(num_chunks, chunk)[:, :-1]
    with torch.no_grad():
        seq_states, _ = wm.sequence_model.parallel_forward(zc, ac)
        pred = wm.prior(seq_states).mode.flatten(2)
    prior_mse = ((pred[:, :-1] - zc[:, 1:]) ** 2).mean(-1)[mask].mean().item()
    copy_mse = ((zc[:, :-1] - zc[:, 1:]) ** 2).mean(-1)[mask].mean().item()
    print("\n-- one-step prediction")
    print(f"prior mse {prior_mse:.4f}   copy z_t baseline {copy_mse:.4f}")

    # ---- 3. open-loop imagination on the true actions
    context, horizon = cfg.train.context_window, cfg.train.horizon
    starts = imagination_windows(boundary, context, horizon, stride=5)
    imagined, real = open_loop(wm, z, actions, starts, context, horizon)
    ks = [k for k in (1, 2, 4, 8, 12, 16) if k <= horizon]
    index = [k - 1 for k in ks]
    held = z[torch.tensor(starts, device=device) + context][:, None]
    print(f"\n-- open-loop imagination, true actions, {len(starts)} windows, context {context}")
    print(f"horizon k                {ks}")
    print(f"latent mse, imagined     {fmt(((imagined - real) ** 2).mean((0, 2))[index])}")
    print(f"latent mse, hold z_ctx   {fmt(((held - real) ** 2).mean((0, 2))[index])}")
    print(f"imagined variance / dim  {fmt(imagined.var(0).mean(-1)[index])}")

    if spec is None:
        print(f"\nno RAM annotations for {game}; skipping saliency and position probes")
        return

    # ---- 4. saliency
    if spec["pixel_map"] is not None:
        sample = np.random.choice(n, min(512, n), replace=False)
        boxes = {k: v[sample] for k, v in sprite_boxes(ram, spec).items()}
        print("\n-- saliency: share of ||dz/dpixels||^2 in a 5x5 box (share of image area)")
        for name, (share, area) in saliency_share(wm, obs[sample], boxes, device).items():
            print(f"{name:14s} {share:.3f}  ({area:.4f})  = {share / area:.1f}x uniform")

    # ---- 5. position probes
    names, targets = position_targets(ram, spec)
    y = torch.from_numpy(targets).to(device)
    train, val, test = block_split(n, device, seed=args.seed)
    baseline = (y[test] - y[train].mean(0)).abs().mean(0).cpu().numpy()
    print("\n-- position probes (RAM units; rerun with --random-init for the reference)")
    print(f"{'':30s}" + " ".join(f"{name:>13s}" for name in names))
    print(f"{'mean-predictor test MAE':30s}" + " ".join(f"{v:13.1f}" for v in baseline))
    for label, feats in (("z", z), ("conv e", e)):
        r2 = ridge_probe(feats, y, train, val, test)
        test_mae, train_mae = mlp_probe(feats, y, train, val, test, steps=args.probe_steps)
        print(f"{label + ': ridge test R^2':30s}" + " ".join(f"{v:13.2f}" for v in r2))
        print(f"{label + ': MLP test MAE':30s}" + " ".join(f"{v:13.1f}" for v in test_mae))
        print(f"{label + ': MLP train MAE':30s}" + " ".join(f"{v:13.1f}" for v in train_mae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory holding config.yaml")
    parser.add_argument("--checkpoint", default="checkpoint_latest.pt")
    parser.add_argument("--steps", type=int, default=24000, help="env steps to collect")
    parser.add_argument("--probe-steps", type=int, default=6000)
    parser.add_argument("--random-init", action="store_true", help="skip loading the checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
