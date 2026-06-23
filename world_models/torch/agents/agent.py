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
from world_models.torch.common.utils import compute_lambda_returns, make_state, to_numpy


class Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.world_model = build_world_model(cfg.wm, cfg.action_dim)
        self.is_rssm = isinstance(self.world_model.sequence_model, RSSM)
        self.latent_size = cfg.wm.latent.num_categories * cfg.wm.latent.num_codes

        input_dim = self.latent_size + self.world_model.sequence_model.output_dim

        self.actor = build_actor(cfg.actor, input_dim, cfg.action_dim, cfg.action_type)
        self.critic = build_critic(cfg.critic, input_dim)
        self.critic_target = TargetNetwork(self.critic, cfg.train.critic_tau)
        self.buffer = build_buffer(cfg.buffer)

        self.optim_wm = optim.Adam(
            self.world_model.parameters(), cfg.optim.wm_lr, eps=cfg.optim.eps
        )
        self.optim_actor = optim.Adam(self.actor.parameters(), cfg.optim.ac_lr, eps=cfg.optim.eps)
        self.optim_critic = optim.Adam(self.critic.parameters(), cfg.optim.ac_lr, eps=cfg.optim.eps)

        if self.is_rssm:
            self.model_state = self.world_model.initial_state(cfg.num_envs, cfg.device)
            self.eval_model_state = self.world_model.initial_state(cfg.num_eval_envs, cfg.device)
            self.prev_latent = torch.zeros((cfg.num_envs, self.latent_size), device=cfg.device)
            self.prev_action = torch.zeros((cfg.num_envs, cfg.action_dim), device=cfg.device)
            self.prev_done = torch.zeros(cfg.num_envs, dtype=torch.bool, device=cfg.device)
            self.eval_latent = torch.zeros((cfg.num_eval_envs, self.latent_size), device=cfg.device)
            self.eval_action = torch.zeros((cfg.num_eval_envs, cfg.action_dim), device=cfg.device)
        else:
            self.obs_history = collections.deque(maxlen=cfg.train.context_window + 1)
            self.action_history = collections.deque(maxlen=cfg.train.context_window)
            self.obs_history.append(torch.zeros((cfg.num_envs, *cfg.obs_shape)).to(cfg.device))
            self.action_history.append(torch.zeros((cfg.num_envs, cfg.action_dim)).to(cfg.device))
            self.eval_obs = collections.deque(maxlen=cfg.train.context_window + 1)
            self.eval_action = collections.deque(maxlen=cfg.train.context_window)
            self.eval_obs.append(torch.zeros((cfg.num_eval_envs, *cfg.obs_shape)).to(cfg.device))
            self.eval_action.append(torch.zeros((cfg.num_eval_envs, cfg.action_dim)).to(cfg.device))

        self.returns_range_ema = None

        self.device = cfg.device
        self.num_envs = cfg.num_envs
        self.num_eval_envs = cfg.num_eval_envs
        self.action_type = cfg.action_type
        self.action_dim = cfg.action_dim
        self.obs_shape = cfg.obs_shape
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

    def act(self, obs, det=False):
        if self.is_rssm:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            done_tensor = self.prev_done.float().unsqueeze(-1)
            latent, seq_state = self.world_model.step_obs_rssm(
                obs_tensor, self.prev_action, self.prev_latent, done_tensor, self.model_state, det
            )
            self.prev_latent = latent.detach()
            self.model_state = seq_state.detach()
        else:
            self.obs_history.append(torch.from_numpy(obs).float().to(self.device))
            obs_tensor = torch.stack(list(self.obs_history), 1)
            action_tensor = torch.stack(list(self.action_history), 1)
            # action_tensor = self._actions_to_model_input(action_tensor)
            latent, seq_state = self.world_model.step_obs_window(obs_tensor, action_tensor, det=det)
        state = make_state(latent, seq_state)
        action = self.actor.policy_fn(state, det)
        processed_actions = self._actions_to_model_input(action.detach())
        if self.is_rssm:
            self.prev_action = processed_actions
        else:
            self.action_history.append(processed_actions)
        action = to_numpy(action)

        return action

    def reset_eval_state(self):
        if self.is_rssm:
            self.eval_model_state = self.world_model.initial_state(self.num_eval_envs, self.device)
            self.eval_latent.zero_()
            self.eval_action.zero_()
        else:
            self.eval_obs.clear()
            self.eval_action.clear()
            self.eval_obs.append(
                torch.zeros((self.num_eval_envs, *self.obs_shape), device=self.device)
            )
            self.eval_action.append(
                torch.zeros((self.num_eval_envs, self.action_dim), device=self.device)
            )

    def eval_act(self, obs, det=True):
        if self.is_rssm:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            done_tensor = torch.zeros(self.num_eval_envs, dtype=torch.bool, device=self.device)
            latent, seq_state = self.world_model.step_obs_rssm(
                obs_tensor,
                self.eval_action,
                self.eval_latent,
                done_tensor,
                self.eval_model_state,
                det,
            )
            self.eval_latent = latent.detach()
            self.eval_model_state = seq_state.detach()
        else:
            self.eval_obs.append(torch.from_numpy(obs).float().to(self.device))
            obs_tensor = torch.stack(list(self.eval_obs), 1)
            action_tensor = torch.stack(list(self.eval_action), 1)
            latent, seq_state = self.world_model.step_obs_window(obs_tensor, action_tensor, det=det)
        state = make_state(latent, seq_state)
        action = self.actor.policy_fn(state, det)
        processed_actions = self._actions_to_model_input(action.detach())
        if self.is_rssm:
            self.eval_action = processed_actions
        else:
            self.eval_action.append(processed_actions)
        action = to_numpy(action)

        return action

    def add_transition(self, obs, action, reward, terminated, truncated):
        self.buffer.add_sample([obs, action, reward, terminated, truncated])
        self.step_count += self.num_envs
        if self.is_rssm:
            self.prev_done = torch.tensor(terminated | truncated).to(self.device)

    def train_step(self):
        obs, actions, rewards, terms, truncs = self.buffer.sample_as_tensors(
            self.device, self.wm_batch_size, self.wm_seq_len
        )
        dones = torch.logical_or(terms, truncs).float()
        processed_actions = self._actions_to_model_input(actions)
        wm_loss, wm_metrics, old_latent, old_seq = self._train_world_model(
            obs, processed_actions, rewards, terms, dones
        )
        if self.step_count > self.train_ac_after:
            if self.is_rssm:
                initial_latent = old_latent
                initial_seq = old_seq
                ac_metrics = self._train_actor_critic((initial_latent, initial_seq))
            else:
                context_obs, context_actions, _, context_terms, context_truncs = (
                    self.buffer.sample_as_tensors(self.device, self.ac_batch_size, self.ac_seq_len)
                )
                context_dones = torch.logical_or(context_terms, context_truncs)
                ac_metrics = self._train_actor_critic((context_obs, context_actions, context_dones))
            return wm_loss, wm_metrics | ac_metrics
        else:
            return wm_loss, wm_metrics

    def _train_world_model(self, obs, actions, rewards, terminated, dones):
        wm_loss, wm_loss_dict, latents, seq_states = self.world_model.world_model_loss(
            obs, actions, rewards, terminated, dones
        )
        if self.step_count < 50:
            print(f"step {self.step_count}: total={wm_loss.item():.4f}, dict={wm_loss_dict}")
            print(f"  decoder is None: {self.world_model.decoder is None}")
            print(
                f"  obj_coef={self.world_model.obj_coef}, "
                f"dyn_coef={self.world_model.dyn_coef}, "
                f"repr_coef={self.world_model.repr_coef}"
            )
            # check decoder is getting gradient
            for n, p in self.world_model.decoder.named_parameters():
                if p.grad is not None:
                    print(f"  decoder {n}: grad_norm={p.grad.norm().item():.4f}")
                break
        self.optim_wm.zero_grad(set_to_none=True)
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.wm_clip)
        self.optim_wm.step()
        return wm_loss, wm_loss_dict, latents, seq_states

    def _train_actor_critic(self, context):
        if self.is_rssm:
            initial_latent, initial_seq = context
            initial_latent = initial_latent.flatten(0, 1)
            initial_seq = initial_seq.flatten(0, 1)
            latents, actor_seq, head_seq, actions = self.world_model.imagine(
                None, None, self.horizon, self.actor, None, initial_latent, initial_seq
            )
        else:
            context_obs, context_actions, context_dones = context
            context_actions = self._actions_to_model_input(context_actions)
            latents, actor_seq, head_seq, actions = self.world_model.imagine(
                context_obs, context_actions, self.horizon, self.actor, context_dones
            )
        head_state = (
            make_state(latents, head_seq) if self.world_model.use_combined_state else head_seq
        )
        reward_pred = self.world_model.reward_predictor(head_state).mean().detach()
        continue_pred = self.world_model.continue_predictor(head_state).mean.squeeze(-1).detach()
        actor_state = make_state(latents, actor_seq).detach()
        values, value_logits = self.critic(actor_state)
        value_targets, _ = self.critic_target(actor_state)
        returns = compute_lambda_returns(
            values.detach(), reward_pred[:, :-1], continue_pred[:, :-1], self.gamma, self.lambda_
        )
        returns_targets = compute_lambda_returns(
            value_targets.detach(),
            reward_pred[:, :-1],
            continue_pred[:, :-1],
            self.gamma,
            self.lambda_,
        )
        action_dist = self.actor.policy_dist(actor_state[:, :-1])
        adv = returns - values.detach()[:, :-1]
        adv = self._normalize_advantages(returns, adv)
        if self.action_type == "discrete":
            actions = actions.argmax(-1)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        if log_probs.dim() > adv.dim():
            log_probs = log_probs.sum(-1)
            entropy = entropy.sum(-1)
        actor_loss = -(adv * log_probs + self.entropy_coef * entropy).mean()
        critic_loss = (
            # -self.critic.make_dist(value_logits).log_prob(value_targets.detach()).mean()
            -self.critic.make_dist(value_logits[:, :-1]).log_prob(returns_targets.detach())
            - self.critic.make_dist(value_logits[:, :-1]).log_prob(returns.detach())
        ).mean()
        self.optim_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.ac_clip)
        self.optim_critic.step()
        self.optim_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.ac_clip)
        self.optim_actor.step()
        self.critic_target.update()
        return {
            "ac/loss/actor": to_numpy(actor_loss),
            "ac/loss/critic": to_numpy(critic_loss),
            "ac/loss/entropy": to_numpy(action_dist.entropy().mean()),
            "ac/stats/advantage_mean": to_numpy(adv.mean()),
            "ac/stats/advantage_std": to_numpy(adv.std()),
            "ac/stats/returns_mean": to_numpy(returns.mean()),
            "ac/stats/value_mean": to_numpy(values.mean()),
            "ac/stats/returns_range_ema": to_numpy(self.returns_range_ema),
        }

    def _normalize_advantages(self, returns, advantages):
        q_hi = torch.quantile(returns, self.returns_pct_hi)
        q_lo = torch.quantile(returns, self.returns_pct_lo)
        range_ = (q_hi - q_lo).detach()
        if self.returns_range_ema is None:
            self.returns_range_ema = range_
        else:
            tau = self.returns_ema_tau
            self.returns_range_ema = (tau * range_ + (1 - tau) * self.returns_range_ema).detach()
        scale = torch.clamp(self.returns_range_ema, min=1.0)
        return (advantages / scale).detach()

    def _actions_to_model_input(self, actions):
        if self.action_type == "discrete":
            actions = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        return actions

    def state_dict_full(self): ...
    def load_state_dict_full(self, state): ...
