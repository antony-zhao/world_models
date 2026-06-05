import collections

import torch
from torch import nn, optim
from torch.nn import functional as F

from world_models.torch.agents.builder import (
    build_actor,
    build_buffer,
    build_critic,
    build_world_model,
)
from world_models.torch.common.models import TargetNetwork
from world_models.torch.common.sequence_models import RSSM
from world_models.torch.common.utils import make_state, to_numpy, transform_obs


class Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # build components
        self.world_model = build_world_model(cfg.wm)
        self.actor = build_actor(cfg.actor)
        self.critic = build_critic(cfg.critic)
        self.critic_target = TargetNetwork(self.critic, cfg.train.critic_tau)
        self.buffer = build_buffer(cfg.buffer)

        self.optim_wm = optim.Adam(
            self.world_model.parameters(), cfg.optim.wm_lr, eps=cfg.optim.eps
        )
        self.optim_actor = optim.Adam(self.actor.parameters(), cfg.optim.ac_lr, eps=cfg.optim.eps)
        self.optim_critic = optim.Adam(self.critic.parameters(), cfg.optim.ac_lr, eps=cfg.optim.eps)

        self.is_rssm = isinstance(self.world_model.sequence_model, RSSM)
        if self.is_rssm:
            self.model_state = self.world_model.initial_state(cfg.num_envs, cfg.device)
            self.history = None
        else:
            self.model_state = None
            self.history = collections.deque(maxlen=cfg.train.context_window)

        self.returns_range_ema = None

        self.device = cfg.device
        self.num_envs = cfg.num_envs
        self.action_type = cfg.action_type
        self.action_dim = cfg.action_dim
        self.horizon = cfg.train.horizon
        self.gamma = cfg.train.gamma
        self.lambda_ = cfg.train.lambda_
        self.entropy_coef = cfg.train.entropy_coef
        self.wm_batch_size = cfg.train.wm_batch_size
        self.wm_seq_len = cfg.train.wm_seq_len
        self.ac_batch_size = cfg.train.ac_batch_size
        self.ac_seq_len = cfg.train.ac_seq_len
        self.train_ac_after = cfg.train.train_ac_after
        self.wm_clip = cfg.optim.wm_clip
        self.ac_clip = cfg.optim.ac_clip
        self.returns_pct_lo = cfg.train.returns_percentile_lo
        self.returns_pct_hi = cfg.train.returns_percentile_hi
        self.returns_ema_tau = cfg.train.returns_ema_tau

        self.step_count = 0

    # ---- env interaction ----------------------------------------------------

    def act(self, obs, det=False):
        """
        Args:
            obs: (num_envs, *obs_shape) tensor on device
            det: greedy if True
        Returns:
            action: (num_envs,) int (discrete) or (num_envs, action_dim) tensor (continuous)
                    on CPU, raw form for env.step()
        """
        # 1. get (latent, seq_state) — branches on RSSM vs non-RSSM
        # 2. state = make_state(latent, seq_state)
        # 3. action = self.actor.policy_fn(state, det)
        # 4. update online state (RSSM: model_state; non-RSSM: append to history)
        # 5. return action.cpu()
        ...

    def add_transition(self, obs, action, reward, terminated, truncated):
        """
        Args:
            obs:        (num_envs, *obs_shape)
            action:     (num_envs,) int or (num_envs, action_dim) tensor
            reward:     (num_envs,) float
            terminated: (num_envs,) bool
            truncated:  (num_envs,) bool
        """
        # 1. store in buffer (per-env)
        # 2. for non-RSSM: append (obs, action) to history
        # 3. on done (per env): reset model_state (RSSM) or clear history (non-RSSM)
        ...

    # ---- training -----------------------------------------------------------

    def train_step(self):
        """One WM update; one AC update if past warmup. Returns merged metrics dict."""
        # 1. sample WM batch
        # 2. WM update
        # 3. if buffer.size < train_ac_after: return WM metrics
        # 4. sample AC batch
        # 5. AC update
        # 6. merge metrics, return
        ...

    def _train_world_model(self, obs, actions_oh, rewards, terminated, dones):
        """
        Returns: loss_dict
        Steps: forward → backward → clip → step
        """
        ...

    def _train_actor_critic(self, context_obs, context_actions_oh, context_dones):
        """
        Returns: loss_dict with actor/critic/entropy losses and advantage stats.
        Steps:
          1. imagine_actor closure (samples + format_action)
          2. self.world_model.imagine(...) → (latents, actor_seqs, head_seqs, actions_oh)
          3. predict rewards, continues from head_seqs
          4. critic on actor_states, target_critic for bootstrap
          5. compute_lambda_returns → returns (length H)
          6. recompute actor dist on actor_states for log_prob and entropy
          7. advantages = returns - values[:, :-1].detach(); normalize via _normalize_advantages
          8. actor_loss = -(adv.detach() * log_probs[:, :-1]).mean() - entropy_coef * entropy[:, :-1].mean()
          9. critic_loss = -critic.make_dist(value_logits[:, :-1]).log_prob(returns.detach()).mean()
         10. critic backward + step; actor backward + step
         11. critic_target.update()
        """
        ...

    def _normalize_advantages(self, returns, advantages):
        """Percentile range of returns, EMA-smoothed, divides advantages. Returns normalized adv."""
        ...

    def _actions_to_model_input(self, actions):
        """Discrete: int → one-hot. Continuous: passthrough."""
        ...

    # ---- checkpoint ---------------------------------------------------------

    def state_dict_full(self): ...
    def load_state_dict_full(
        self, state
    ): ...  # ---- checkpoint ---------------------------------------------------------

    def state_dict_full(self): ...
    def load_state_dict_full(self, state): ...
