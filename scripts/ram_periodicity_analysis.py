"""Is the ~25-step damped oscillation in the RSSM h_t lag-similarity curve
(see embedding_similarity_by_lag.py) genuine ball-bounce periodicity, or an artifact?

Extends that analysis with Atari RAM ground truth:

  1. Ball (x, y) and paddle x position are read from RAM every step, in lockstep with
     the same rollout used for obs_embedding/h_t, via a shadow `ALEInterface` (the
     vectorized `AtariVectorEnv` used for stepping has no RAM accessor -- confirmed by
     inspecting `ale_py._ale_py.ALEVectorInterface`, whose only methods are
     get_action_set/get_num_envs/get_observation_shape/handle/recv/reset/send). The RAM
     byte offsets (ball_x=99, ball_y=101, player_x=72) are taken from the Breakout entry
     of mila-iqia/atari-representation-learning's ram_annotations.py (fetched directly
     from GitHub, not from memory) and were separately confirmed by hand: player_x moves
     smoothly with LEFT/RIGHT and clamps at the walls, ball_x/ball_y move by ~1/step
     along a straight line between bounces (see conversation for the raw trace).
     Lockstep between the shadow ALEInterface and the vector env was validated by
     replaying 800 random-action steps through both with noop_max=0 (required -- it's
     the only source of nondeterminism given repeat_action_probability=0) and comparing
     `info["lives"]` at every step, including several life-losses and game-overs: 0
     mismatches. On a vector-env game-over, AtariVectorEnv's NEXT_STEP autoreset mode
     discards the given action and resets instead, so the shadow mirrors that by calling
     `reset_game()` (instead of stepping) whenever the previous macro-step ended with
     `ale.game_over()`.

  2. Ball-position autocorrelation vs. lag, overlaid (dashed) on the obs_embedding/h_t
     curves.

  3. Cross-correlation between the (per-episode, then averaged) obs_embedding and h_t
     lag-curves -- both raw and after removing each curve's local trend, since a raw
     correlation would mostly just pick up that both curves decay with lag rather than
     whether their *oscillations* line up in phase.

  4. A peak-lag vs. trough-lag frame-pair grid for visual inspection.

Usage:
    python scripts/ram_periodicity_analysis.py \
        --run "runs/atari100k/Good DreamerCDP Breakout" --steps 2000 --episodes 5
"""

import argparse
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ale_py import ALEInterface, LoggerMode, roms
from gymnasium.wrappers.vector import TransformObservation
from omegaconf import OmegaConf

from world_models.torch.agents.agent import Agent
from world_models.utils.config import preprocess_config

gym.register_envs(ale_py)

# Verified against mila-iqia/atari-representation-learning's ram_annotations.py
# (breakout entry) and empirically (see module docstring).
RAM_BALL_X, RAM_BALL_Y, RAM_PLAYER_X = 99, 101, 72


def build_env(cfg):
    # noop_max=0: makes the vector env deterministic given the action sequence (the only
    # other stochasticity, repeat_action_probability, is already 0), which is required
    # for the shadow ALEInterface below to stay in exact lockstep for RAM reads.
    common = dict(
        img_height=64, img_width=64, grayscale=False, stack_num=1, frameskip=4,
        maxpool=True, noop_max=0, episodic_life=False, life_loss_info=True,
        reward_clipping=False, repeat_action_probability=0.0, use_fire_reset=False,
        full_action_space=True,
    )
    env = ale_py.AtariVectorEnv(game=cfg.env.game, num_envs=1, **common)
    space = gym.spaces.Box(0, 255, (3, 64, 64), np.uint8)
    wrapped = TransformObservation(
        env, lambda o: np.transpose(o[:, 0], (0, 3, 1, 2)),
        observation_space=gym.spaces.Box(0, 255, (1, 3, 64, 64), np.uint8),
        single_observation_space=space,
    )
    return wrapped, env.ale.get_action_set()


def build_shadow_ale(game):
    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.loadROM(roms.get_rom_path(game))
    return ale


def rollout_episode(env, action_set, shadow, agent, steps, seed, det):
    agent.reset_eval_state()
    obs, _ = env.reset(seed=seed)
    shadow.reset_game()
    was_game_over = False

    obs_list, action_list, done_list = [], [], []
    ball_x, ball_y, player_x = [], [], []
    prev_done = np.zeros(1, dtype=bool)
    for _ in range(steps):
        obs_list.append(obs[0].copy())
        ram = shadow.getRAM()
        ball_x.append(int(ram[RAM_BALL_X]))
        ball_y.append(int(ram[RAM_BALL_Y]))
        player_x.append(int(ram[RAM_PLAYER_X]))

        action = agent.eval_act(obs.astype(np.float32), dones=prev_done, det=det)
        obs, _, term, trunc, _ = env.step(action)
        prev_done = term | trunc
        action_list.append(int(action[0]))
        done_list.append(bool(prev_done[0]))

        if was_game_over:
            shadow.reset_game()
        else:
            for _ in range(4):
                shadow.act(action_set[int(action[0])])
        was_game_over = shadow.game_over()

    ram_traj = {"ball_x": ball_x, "ball_y": ball_y, "player_x": player_x}
    return obs_list, action_list, done_list, ram_traj


def sim_by_lag(x, max_lag):
    # x: (T, D). cosine similarity, averaged over all valid (t, t+lag) pairs.
    x = F.normalize(x, dim=-1)
    T = x.shape[0]
    return {lag: (x[: T - lag] * x[lag:]).sum(-1).mean().item() for lag in range(min(max_lag, T - 1) + 1)}


def autocorr_by_lag(xy, max_lag):
    # xy: (T, 2) ball (x, y), centered per-episode then normalized so lag 0 == 1.
    # This is a plain (multivariate) autocorrelation, not cosine similarity: ball
    # position is always positive, so uncentered cosine similarity would just measure
    # closeness to the mean and hide the oscillation.
    xy = xy - xy.mean(0, keepdim=True)
    T = xy.shape[0]
    var0 = (xy * xy).sum(-1).mean().item()
    out = {}
    for lag in range(min(max_lag, T - 1) + 1):
        cov = (xy[: T - lag] * xy[lag:]).sum(-1).mean().item()
        out[lag] = cov / var0
    return out


def detrend(curve, window):
    # subtract a centered moving average to isolate the oscillatory ripple from the
    # overall decay trend, so cross-correlation reflects phase, not just "both decay".
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(curve, pad, mode="edge")
    trend = np.convolve(padded, kernel, mode="valid")[: len(curve)]
    return curve - trend


def find_peak(curve, lags, lo, hi):
    idx = [i for i, lag in enumerate(lags) if lo <= lag <= hi]
    best = max(idx, key=lambda i: curve[i])
    return lags[best]


def find_trough(curve, lags, lo, hi):
    idx = [i for i, lag in enumerate(lags) if lo <= lag <= hi]
    best = min(idx, key=lambda i: curve[i])
    return lags[best]


def main(run_dir, steps, max_lag, episodes, det, checkpoint, out_plot, out_grid):
    run_dir = Path(run_dir)
    OmegaConf.register_new_resolver("eval", eval, use_cache=False)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.num_eval_envs = 1

    env, action_set = build_env(cfg)
    cfg.action_dim = int(env.single_action_space.n)
    cfg.obs_shape = tuple(int(x) for x in env.single_observation_space.shape)
    cfg = preprocess_config(cfg)
    shadow = build_shadow_ale(cfg.env.game)

    agent = Agent(cfg).to(cfg.device)
    ckpt_path = run_dir / checkpoint
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    agent.world_model.load_state_dict(ckpt["world_model"])
    agent.world_model.eval()
    print(f"Loaded {ckpt_path.name} at env_step={ckpt['env_step']}")

    emb_curves, h_curves, ball_curves = [], [], []
    all_obs, all_ram, all_dones = [], [], []
    total_boundaries = 0
    for ep in range(episodes):
        obs_list, action_list, done_list, ram_traj = rollout_episode(
            env, action_set, shadow, agent, steps, seed=ep, det=det
        )
        total_boundaries += sum(done_list)
        all_obs.append(obs_list)
        all_ram.append(ram_traj)
        all_dones.append(done_list)

        obs_t = torch.from_numpy(np.stack(obs_list)).float().to(cfg.device).unsqueeze(0)
        actions_t = (
            F.one_hot(torch.tensor(action_list), cfg.action_dim).float().to(cfg.device).unsqueeze(0)
        )
        dones_t = torch.tensor(done_list, dtype=torch.float32, device=cfg.device).unsqueeze(0)

        with torch.no_grad():
            transformed = obs_t / 255.0 - 0.5
            obs_embeddings = agent.world_model.encoder(transformed)  # (1,T,E)
            _, hiddens, _, _ = agent.world_model.sequence_model.rollout(
                obs_embeddings, actions_t, agent.world_model.posterior, agent.world_model.prior, dones_t
            )  # hiddens: (1,T,D) = h_t

        emb_curves.append(sim_by_lag(obs_embeddings[0].cpu(), max_lag))
        h_curves.append(sim_by_lag(hiddens[0].cpu(), max_lag))

        ball_xy = torch.tensor(
            list(zip(ram_traj["ball_x"], ram_traj["ball_y"])), dtype=torch.float32
        )
        ball_curves.append(autocorr_by_lag(ball_xy, max_lag))
        print(f"  episode {ep+1}/{episodes} done ({sum(done_list)} boundaries)")
    env.close()

    lags = sorted(emb_curves[0])
    emb_mean = np.array([np.mean([c[lag] for c in emb_curves]) for lag in lags])
    h_mean = np.array([np.mean([c[lag] for c in h_curves]) for lag in lags])
    ball_mean = np.array([np.mean([c[lag] for c in ball_curves]) for lag in lags])

    print(f"\n{episodes} episodes x {steps} steps ({total_boundaries} total episode boundaries)")
    print(f"{'lag':>5}  {'obs_embedding':>14}  {'rssm hidden h_t':>16}  {'ball autocorr':>14}")
    for lag, e, h, b in zip(lags, emb_mean, h_mean, ball_mean):
        print(f"{lag:>5}  {e:>14.4f}  {h:>16.4f}  {b:>14.4f}")

    # ---- cross-correlation: h_t vs obs_embedding lag-curves (excluding trivial lag 0)
    sl = slice(1, None)
    raw_corr = np.corrcoef(emb_mean[sl], h_mean[sl])[0, 1]
    window = 13  # ~ half the observed period, smooths out the oscillation itself
    emb_resid = detrend(emb_mean, window)
    h_resid = detrend(h_mean, window)
    ball_resid = detrend(ball_mean, window)
    resid_corr = np.corrcoef(emb_resid[sl], h_resid[sl])[0, 1]
    ball_h_corr = np.corrcoef(ball_resid[sl], h_resid[sl])[0, 1]
    ball_emb_corr = np.corrcoef(ball_resid[sl], emb_resid[sl])[0, 1]

    peak_h = find_peak(h_mean, lags, 18, 32)
    trough_h = find_trough(h_mean, lags, 5, 18)
    peak_emb = find_peak(emb_resid, lags, 18, 32)
    peak_ball = find_peak(ball_mean, lags, 18, 32)

    print("\n--- cross-correlation of the lag-curves themselves (lags 1..max_lag) ---")
    print(f"corr(obs_embedding, h_t), raw curves          : {raw_corr:+.3f}")
    print(f"corr(obs_embedding, h_t), detrended (ripple)   : {resid_corr:+.3f}")
    print(f"corr(ball autocorr, h_t), detrended            : {ball_h_corr:+.3f}")
    print(f"corr(ball autocorr, obs_embedding), detrended  : {ball_emb_corr:+.3f}")
    print(f"h_t          : peak near lag {peak_h}, trough near lag {trough_h}")
    print(f"obs_embedding: ripple peak (detrended) near lag {peak_emb}")
    print(f"ball autocorr: peak near lag {peak_ball}")

    # ---- extended plot: obs_embedding, h_t, ball position autocorrelation
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lags, emb_mean, label="obs_embedding", color="tab:blue")
    ax.plot(lags, h_mean, label="rssm hidden h_t", color="tab:orange")
    ax.plot(lags, ball_mean, label="ball (x,y) RAM autocorr", color="tab:green", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("lag (env steps)")
    ax.set_ylabel("mean similarity / autocorrelation")
    ax.set_title(
        f"Similarity vs. lag — {cfg.env.game} ({ckpt_path.name}, "
        f"avg over {episodes} eps x {steps} steps)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    print(f"\nSaved plot to {out_plot}")

    # ---- visual check: peak-lag vs trough-lag frame pairs, from episode 0
    obs_list, ram_traj, done_list = all_obs[0], all_ram[0], all_dones[0]
    T = len(obs_list)
    max_needed = max(peak_h, trough_h)
    # only consider windows that don't cross an episode boundary (so the frame pairs
    # compare states within the same life/game rather than across a reset), and where
    # the ball has actually been launched at every endpoint (ball_y == 0 is the
    # "resting on the paddle, not yet fired" sentinel -- comparing against that is
    # meaningless).
    ball_y = ram_traj["ball_y"]
    candidates = [
        t0
        for t0 in range(T - max_needed)
        if not any(done_list[t0 : t0 + max_needed])
        and ball_y[t0] > 0
        and ball_y[t0 + peak_h] > 0
        and ball_y[t0 + trough_h] > 0
    ]
    starts = np.array(candidates)[np.linspace(0, len(candidates) - 1, 5, dtype=int)]

    fig2, axes = plt.subplots(len(starts), 3, figsize=(9, 3 * len(starts)))
    for row, t0 in enumerate(starts):
        for col, (dt, tag) in enumerate([(0, "t"), (peak_h, f"t+{peak_h} (peak)"), (trough_h, f"t+{trough_h} (trough)")]):
            t = t0 + dt
            frame = np.transpose(obs_list[t], (1, 2, 0))  # CHW -> HWC
            ax_ = axes[row, col]
            ax_.imshow(frame)
            prefix = f"t0={t0}  " if col == 0 else ""
            ax_.set_title(
                f"{prefix}{tag}\nball=({ram_traj['ball_x'][t]},{ram_traj['ball_y'][t]}) "
                f"paddle={ram_traj['player_x'][t]}",
                fontsize=8,
            )
            ax_.axis("off")
    fig2.suptitle(
        f"Peak-lag (Δ={peak_h}) vs. trough-lag (Δ={trough_h}) frame pairs — episode 0"
    )
    fig2.tight_layout()
    fig2.savefig(out_grid, dpi=150)
    print(f"Saved frame-pair grid to {out_grid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/atari100k/Good DreamerCDP Breakout")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-lag", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=5, help="independent rollouts to average over")
    parser.add_argument("--det", action="store_true", help="use deterministic (mode) posterior")
    parser.add_argument("--checkpoint", default="checkpoint_latest.pt")
    parser.add_argument("--out-plot", default="scripts/ram_periodicity_analysis.png")
    parser.add_argument("--out-grid", default="scripts/ram_periodicity_frame_pairs.png")
    args = parser.parse_args()
    main(
        args.run, args.steps, args.max_lag, args.episodes, args.det, args.checkpoint,
        args.out_plot, args.out_grid,
    )
