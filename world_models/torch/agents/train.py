"""Training script for world-model RL on Atari.

Usage:
    python -m world_models.torch.agents.train --config configs/atari_pong.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from world_models.torch.agents.agent import Agent
from world_models.utils.logger import Logger

gym.register_envs(ale_py)


# --------------------------------------------------------------------- env


def make_env_fn(env_id: str, seed: int, full_action_space: bool = True):
    def thunk():
        env = gym.make(env_id, frameskip=1, full_action_space=full_action_space)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=64,
            grayscale_obs=False,
            scale_obs=False,
            terminal_on_life_loss=False,
        )
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.transpose(obs, (2, 0, 1)),
            observation_space=gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
        )
        env.action_space.seed(seed)
        return env

    return thunk


def build_envs(cfg):
    train_envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(cfg.env.id, cfg.seed + i) for i in range(cfg.num_envs)]
    )
    eval_env = make_env_fn(cfg.env.id, cfg.seed + 10000)()
    return train_envs, eval_env


# ----------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(agent, env, num_episodes: int) -> tuple[float, float]:
    """Run greedy eval episodes on a single env, return (mean_return, mean_length)."""
    returns, lengths = [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        agent.reset_eval_state()
        ep_return, ep_len, done = 0.0, 0, False
        while not done:
            obs_batched = np.expand_dims(obs.astype(np.float32), 0)  # (1, C, H, W)
            action = agent.eval_act(obs_batched, det=True)[0]
            obs, reward, term, trunc, _ = env.step(int(action))
            ep_return += float(reward)
            ep_len += 1
            done = term or trunc
        returns.append(ep_return)
        lengths.append(ep_len)
    return float(np.mean(returns)), float(np.mean(lengths))


# --------------------------------------------------------------- training


def prefill_buffer(agent, envs, cfg):
    obs, _ = envs.reset(seed=cfg.seed)
    episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    completed = []

    print(f"Prefilling buffer with {cfg.train.prefill_steps} random steps...")
    for _ in range(cfg.train.prefill_steps):
        actions = np.array([envs.single_action_space.sample() for _ in range(cfg.num_envs)])
        next_obs, rewards, terms, truncs, _ = envs.step(actions)
        agent.add_transition(obs, actions, rewards, terms, truncs)
        episode_returns += rewards
        dones = terms | truncs
        for i in np.where(dones)[0]:
            completed.append(float(episode_returns[i]))
            episode_returns[i] = 0.0
        obs = next_obs
    print(
        f"  prefill complete. mean random-policy return: "
        f"{np.mean(completed) if completed else 0:.2f}"
    )
    return obs, episode_returns


def save_checkpoint(agent, path: Path, env_step: int):
    state = {
        "env_step": env_step,
        "step_count": agent.step_count,
        "world_model": agent.world_model.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "optim_wm": agent.optim_wm.state_dict(),
        "optim_actor": agent.optim_actor.state_dict(),
        "optim_critic": agent.optim_critic.state_dict(),
        "returns_range_ema": agent.returns_range_ema,
    }
    torch.save(state, path)


def main(cfg_path: str, overrides: list[str] = None):
    OmegaConf.register_new_resolver("eval", eval)
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg.merge_with_dotlist(overrides)

    # ---- output dir + logger
    run_name = f"{cfg.env.id.replace('/', '-')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path(cfg.train.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config.yaml")
    logger = Logger(str(log_dir))
    print(f"Logging to: {log_dir}")

    # ---- seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- envs
    train_envs, eval_env = build_envs(cfg)
    cfg.action_dim = int(train_envs.single_action_space.n)
    cfg.obs_shape = tuple(int(x) for x in train_envs.single_observation_space.shape)
    print(f"Env: {cfg.env.id}  action_dim={cfg.action_dim}  obs_shape={cfg.obs_shape}")

    # ---- agent
    agent = Agent(cfg).to(cfg.device)
    wm_params = sum(p.numel() for p in agent.world_model.parameters())
    ac_params = sum(p.numel() for p in agent.actor.parameters()) + sum(
        p.numel() for p in agent.critic.parameters()
    )
    print(f"Agent built. WM params: {wm_params:,}  AC params: {ac_params:,}")

    # ---- prefill
    obs, episode_returns = prefill_buffer(agent, train_envs, cfg)
    completed_returns = []

    # ---- main loop
    print(f"Training for {cfg.train.total_env_steps} env steps...")
    env_step = cfg.train.prefill_steps

    while env_step < cfg.train.total_env_steps:
        # Interact
        actions = agent.act(obs.astype(np.float32))
        next_obs, rewards, terms, truncs, _ = train_envs.step(actions)
        agent.add_transition(obs, actions, rewards, terms, truncs)
        episode_returns += rewards
        dones = terms | truncs
        for i in np.where(dones)[0]:
            completed_returns.append(float(episode_returns[i]))
            episode_returns[i] = 0.0
        obs = next_obs
        env_step += cfg.num_envs

        # Train
        if env_step % cfg.train.train_every == 0:
            for _ in range(cfg.train.train_ratio):
                _, metrics = agent.train_step()
                logger.add_metrics(metrics)

        # Periodic scalar logging
        if env_step % cfg.train.log_every == 0:
            if completed_returns:
                recent = completed_returns[-100:]
                logger.add_scalar("train/episode_return", float(np.mean(recent)))
                logger.add_scalar("train/episode_return_max", float(np.max(recent)))
            logger.add_scalar("train/buffer_size", agent.buffer.size)
            logger.add_scalar("train/env_step", env_step)
            logger.write(env_step)

        # Evaluation
        if env_step % cfg.train.eval_every == 0:
            eval_return, eval_length = evaluate(agent, eval_env, cfg.train.eval_episodes)
            logger.add_scalar("eval/episode_return", eval_return)
            logger.add_scalar("eval/episode_length", eval_length)
            logger.write(env_step)
            print(
                f"  step {env_step:>9,}  "
                f"train_return={np.mean(completed_returns[-100:]) if completed_returns else 0:7.2f}  "
                f"eval_return={eval_return:7.2f}"
            )

        # Checkpoint
        if env_step % cfg.train.checkpoint_every == 0:
            ckpt_path = log_dir / f"checkpoint_{env_step}.pt"
            save_checkpoint(agent, ckpt_path, env_step)
            save_checkpoint(agent, log_dir / "checkpoint_latest.pt", env_step)

    save_checkpoint(agent, log_dir / "checkpoint_final.pt", env_step)
    train_envs.close()
    eval_env.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="Config overrides like env.id=ALE/Pong-v5")
    args = parser.parse_args()
    main(args.config, args.overrides)
