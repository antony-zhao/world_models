"""How similar are obs_embeddings / RSSM hidden states across time, as a function of lag?

Rolls out the trained agent for several episodes, encodes every observed frame, runs it
through the RSSM (teacher-forced on the real obs), then reports mean cosine similarity
between (obs_embedding_t, obs_embedding_{t+lag}) and (h_t, h_{t+lag}) averaged over all
valid t and all episodes, for a range of lags. Plots the result.

Usage:
    python scripts/embedding_similarity_by_lag.py \
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
from gymnasium.wrappers.vector import TransformObservation
from omegaconf import OmegaConf

from world_models.torch.agents.agent import Agent
from world_models.utils.config import preprocess_config

gym.register_envs(ale_py)


def build_env(cfg):
    common = dict(
        img_height=64, img_width=64, grayscale=False, stack_num=1, frameskip=4,
        maxpool=True, noop_max=30, episodic_life=False, life_loss_info=True,
        reward_clipping=False, repeat_action_probability=0.0, use_fire_reset=False,
        full_action_space=True,
    )
    env = ale_py.AtariVectorEnv(game=cfg.env.game, num_envs=1, **common)
    space = gym.spaces.Box(0, 255, (3, 64, 64), np.uint8)
    return TransformObservation(
        env, lambda o: np.transpose(o[:, 0], (0, 3, 1, 2)),
        observation_space=gym.spaces.Box(0, 255, (1, 3, 64, 64), np.uint8),
        single_observation_space=space,
    )


def rollout_episode(env, agent, steps, seed, det):
    agent.reset_eval_state()
    obs, _ = env.reset(seed=seed)
    obs_list, action_list, done_list = [], [], []
    prev_done = np.zeros(1, dtype=bool)
    for _ in range(steps):
        obs_list.append(obs[0].copy())
        action = agent.eval_act(obs.astype(np.float32), dones=prev_done, det=det)
        obs, _, term, trunc, _ = env.step(action)
        prev_done = term | trunc
        action_list.append(int(action[0]))
        done_list.append(bool(prev_done[0]))
    return obs_list, action_list, done_list


def sim_by_lag(x, max_lag):
    # x: (T, D). returns dict lag -> mean cosine sim over all valid (t, t+lag) pairs
    x = F.normalize(x, dim=-1)
    T = x.shape[0]
    out = {}
    for lag in range(0, min(max_lag, T - 1) + 1):
        sims = (x[: T - lag] * x[lag:]).sum(-1)
        out[lag] = sims.mean().item()
    return out


def main(run_dir, steps, max_lag, episodes, det, checkpoint, out_path):
    run_dir = Path(run_dir)
    OmegaConf.register_new_resolver("eval", eval, use_cache=False)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.num_eval_envs = 1

    env = build_env(cfg)
    cfg.action_dim = int(env.single_action_space.n)
    cfg.obs_shape = tuple(int(x) for x in env.single_observation_space.shape)
    cfg = preprocess_config(cfg)

    agent = Agent(cfg).to(cfg.device)
    ckpt_path = run_dir / checkpoint
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    agent.world_model.load_state_dict(ckpt["world_model"])
    agent.world_model.eval()
    print(f"Loaded {ckpt_path.name} at env_step={ckpt['env_step']}")

    emb_curves, h_curves = [], []
    total_boundaries = 0
    for ep in range(episodes):
        obs_list, action_list, done_list = rollout_episode(env, agent, steps, seed=ep, det=det)
        total_boundaries += sum(done_list)

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
        print(f"  episode {ep+1}/{episodes} done ({sum(done_list)} boundaries)")
    env.close()

    lags = sorted(emb_curves[0])
    emb_mean = np.array([np.mean([c[lag] for c in emb_curves]) for lag in lags])
    h_mean = np.array([np.mean([c[lag] for c in h_curves]) for lag in lags])

    print(
        f"\n{episodes} episodes x {steps} steps ({total_boundaries} total episode boundaries)"
    )
    print(f"{'lag':>5}  {'obs_embedding':>14}  {'rssm hidden h_t':>16}")
    for lag, e, h in zip(lags, emb_mean, h_mean):
        print(f"{lag:>5}  {e:>14.4f}  {h:>16.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lags, emb_mean, label="obs_embedding", color="tab:blue")
    ax.plot(lags, h_mean, label="rssm hidden h_t", color="tab:orange")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("lag (env steps)")
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(
        f"Embedding/hidden-state similarity vs. lag — {cfg.env.game} "
        f"({ckpt_path.name}, avg over {episodes} eps x {steps} steps)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/atari100k/Good DreamerCDP Breakout")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-lag", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=5, help="independent rollouts to average over")
    parser.add_argument("--det", action="store_true", help="use deterministic (mode) posterior")
    parser.add_argument("--checkpoint", default="checkpoint_latest.pt")
    parser.add_argument(
        "--out",
        default="/tmp/claude-1000/-home-antony-world-models/6104ab2e-c7a5-4999-8238-a6c3470059b2/scratchpad/embedding_similarity_by_lag.png",
    )
    args = parser.parse_args()
    main(args.run, args.steps, args.max_lag, args.episodes, args.det, args.checkpoint, args.out)
