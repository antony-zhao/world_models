"""Training script for world-model RL on Atari.

Usage:
    python -m world_models.torch.agents.train --config configs/atari100k.yaml
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.vector import TransformObservation
from omegaconf import OmegaConf

from world_models.torch.agents.agent import Agent
from world_models.utils.config import preprocess_config
from world_models.utils.logger import Logger

gym.register_envs(ale_py)

# Offset so eval env seeds never overlap the training env seed stream.
EVAL_SEED_OFFSET = 10_000


def _to_cxhxw(obs):
    # AtariVectorEnv gives (N, stack=1, H, W, C); model wants (N, C, H, W).
    return np.transpose(obs[:, 0], (0, 3, 1, 2))


class ChannelFirst(gym.vector.VectorWrapper):
    # TransformObservation asserts NEXT_STEP autoreset; this does the same transform under SameStep.
    # info["final_obs"] is left untransformed (unused).
    def __init__(self, env, observation_space, single_observation_space):
        super().__init__(env)
        self.observation_space = observation_space
        self.single_observation_space = single_observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return _to_cxhxw(obs), info

    def step(self, actions):
        obs, rewards, terms, truncs, info = self.env.step(actions)
        return _to_cxhxw(obs), rewards, terms, truncs, info


def build_envs(cfg):
    common = dict(
        img_height=64,
        img_width=64,
        grayscale=False,
        stack_num=1,
        frameskip=4,
        maxpool=True,
        noop_max=30,
        episodic_life=False,
        life_loss_info=False,  # `terminated` only on game over (DreamerV3 `lives: unused`)
        reward_clipping=False,
        repeat_action_probability=0.0,
        use_fire_reset=False,
        full_action_space=False,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    single_space = gym.spaces.Box(0, 255, (3, 64, 64), np.uint8)

    def make(num):
        env = ale_py.AtariVectorEnv(game=cfg.env.game, num_envs=num, **common)
        batch_space = gym.spaces.Box(0, 255, (num, 3, 64, 64), np.uint8)
        return ChannelFirst(env, batch_space, single_space)
        # return TransformObservation(
        #     env,
        #     _to_cxhxw,
        #     observation_space=batch_space,
        #     single_observation_space=single_space,
        # )

    train_envs = make(cfg.num_envs)
    eval_envs = make(cfg.num_eval_envs)
    # Seed the vector action-space RNG so prefill / any .sample() is reproducible.
    train_envs.action_space.seed(cfg.seed)
    eval_envs.action_space.seed(cfg.seed + EVAL_SEED_OFFSET)
    return train_envs, eval_envs


# ----------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(agent, env, num_episodes, eval_seed=None):
    """Evaluate using full-game returns.

    With life_loss_info=False, `terminated` only fires on game over. Under SameStep
    autoreset, info["lives"] at a done step already belongs to the reset game, so
    game over is detected from `terminated | truncated` rather than `lives == 0`.
    """
    num_envs = env.num_envs
    returns_buffer = []
    lengths_buffer = []

    obs, _ = env.reset(seed=eval_seed)
    agent.reset_eval_state()

    ep_returns = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    prev_dones = np.zeros(num_envs, dtype=bool)
    while len(returns_buffer) < num_episodes:
        actions = agent.eval_act(obs.astype(np.float32), dones=prev_dones, det=False)
        obs, rewards, terms, truncs, infos = env.step(actions)
        # prev_dones feeds the agent's context reset; life-loss terminals DO reset
        # the recurrent/transformer context (consistent with training), so use the
        # full terminal signal here.
        prev_dones = terms | truncs
        game_over = terms | truncs

        ep_returns += rewards
        ep_lengths += 1

        for i in np.where(game_over)[0]:
            if len(returns_buffer) < num_episodes:
                returns_buffer.append(float(ep_returns[i]))
                lengths_buffer.append(int(ep_lengths[i]))
            ep_returns[i] = 0
            ep_lengths[i] = 0
    return float(np.mean(returns_buffer)), float(np.mean(lengths_buffer))


# --------------------------------------------------------------- training


def prefill_buffer(agent, envs, cfg, initial_obs):
    obs = initial_obs
    episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    completed = []

    print(f"Prefilling buffer with {cfg.train.prefill_steps} random steps...")
    for _ in range(cfg.train.prefill_steps):
        actions = envs.action_space.sample()
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        # life_loss_info=False => `terms` is game over only.
        terms_for_buffer = terms
        agent.add_transition(obs, actions, rewards, terms_for_buffer, truncs)
        episode_returns += rewards
        # Full-game boundary for return tracking.
        game_over = terms | truncs
        for i in np.where(game_over)[0]:
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
    # Use safe filename: replace / in env IDs like ALE/Pong-v5
    game_safe = cfg.env.game.replace("/", "-")
    config_stem = Path(cfg_path).stem  # "dreamer"
    config_parent = Path(cfg_path).parent.name  # "rssm"
    run_name = (
        f"{config_parent}_{config_stem}_{game_safe}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_dir = Path(cfg.train.log_dir) / run_name
    log_dir = Path(cfg.train.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config.yaml")
    logger = Logger(str(log_dir))
    print(f"Logging to: {log_dir}")

    # ---- seed (env stream is seeded separately at reset / action_space.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if getattr(cfg, "deterministic", False):
        # Full run-to-run determinism for debugging (costs some speed). See
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---- envs
    train_envs, eval_envs = build_envs(cfg)
    cfg.action_dim = int(train_envs.single_action_space.n)
    cfg.obs_shape = tuple(int(x) for x in train_envs.single_observation_space.shape)
    cfg = preprocess_config(cfg)
    print(f"Env: {cfg.env.game}  action_dim={cfg.action_dim}  obs_shape={cfg.obs_shape}")

    # ---- agent
    agent = Agent(cfg).to(cfg.device)
    wm_params = sum(p.numel() for p in agent.world_model.parameters())
    ac_params = sum(p.numel() for p in agent.actor.parameters()) + sum(
        p.numel() for p in agent.critic.parameters()
    )
    print(f"Agent built. WM params: {wm_params:,}  AC params: {ac_params:,}")

    # ---- initial reset (seeds the training env stream deterministically)
    initial_obs, _ = train_envs.reset(seed=cfg.seed)

    # ---- prefill
    obs, episode_returns = prefill_buffer(agent, train_envs, cfg, initial_obs)
    completed_returns = []

    # ---- main loop
    print(f"Training for {cfg.train.total_env_steps} env steps...")
    env_step = cfg.train.prefill_steps

    while env_step < cfg.train.total_env_steps:
        # Interact
        actions = agent.act(obs.astype(np.float32))
        next_obs, rewards, terms, truncs, infos = train_envs.step(actions)
        # life_loss_info=False => `terms` is game over only,
        # which is exactly what we want in the buffer (STORM convention).
        terms_for_buffer = terms
        agent.add_transition(obs, actions, rewards, terms_for_buffer, truncs)

        episode_returns += rewards
        # logging: a full game ends on terminated or truncated
        game_over = terms | truncs
        for i in np.where(game_over)[0]:
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
            eval_return, eval_length = evaluate(
                agent,
                eval_envs,
                cfg.train.eval_episodes,
                eval_seed=cfg.seed + EVAL_SEED_OFFSET,
            )
            logger.add_scalar("eval/episode_return", eval_return)
            logger.add_scalar("eval/episode_length", eval_length)
            logger.write(env_step)
            train_return = np.mean(completed_returns[-100:]) if completed_returns else 0
            print(
                f"  step {env_step:>9,}  "
                f"train_return={train_return:7.2f}  "
                f"eval_return={eval_return:7.2f}"
            )

        # Checkpoint
        if env_step % cfg.train.checkpoint_every == 0:
            ckpt_path = log_dir / f"checkpoint_{env_step}.pt"
            save_checkpoint(agent, ckpt_path, env_step)
            save_checkpoint(agent, log_dir / "checkpoint_latest.pt", env_step)

    save_checkpoint(agent, log_dir / "checkpoint_final.pt", env_step)
    train_envs.close()
    eval_envs.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="Config overrides like env.game=ALE/Pong-v5")
    args = parser.parse_args()
    main(args.config, args.overrides)
