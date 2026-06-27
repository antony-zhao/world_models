import torch
from torch import nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough, kl_divergence

from world_models.torch.common.heads import BernoulliHead, TwoHotHead
from world_models.torch.common.models import Posterior, Prior
from world_models.torch.common.sequence_models import RSSM, SequenceModel
from world_models.torch.common.utils import make_state, to_numpy, transform_obs


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder,
        sequence_model: SequenceModel,
        posterior: Posterior,
        prior: Prior,
        continue_predictor: BernoulliHead,
        reward_predictor: TwoHotHead,
        objective,
        obs_type,
        obj_coef=1.0,
        dyn_coef=0.5,
        repr_coef=0.1,
        head_coef=1.0,
        free_nats=1.0,
        use_combined_state=False,
        decoder=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sequence_model = sequence_model
        self.posterior = posterior
        self.prior = prior
        self.continue_predictor = continue_predictor
        self.reward_predictor = reward_predictor
        self.objective = objective
        self.is_image = obs_type == "image"
        self.use_combined_state = use_combined_state

        self.obj_coef = obj_coef
        self.dyn_coef = dyn_coef
        self.repr_coef = repr_coef
        self.head_coef = head_coef
        self.free_nats = free_nats

    def initial_state(self, batch_size, device):
        return self.sequence_model.initial_state(batch_size, device)

    @torch.no_grad()
    def step_obs_rssm(self, obs, prev_action, prev_latent, prev_done, model_state, det=False):
        # action should be preprocessed into one_hot before being passed here
        # takes in x_t, a_{t-1}, z_{t-1}
        # returns the latent (z_t), seq_state (h_t)
        if not isinstance(self.sequence_model, RSSM):
            raise TypeError(
                f"step_obs_rssm requires RSSM, got {type(self.sequence_model).__name__}"
            )
        transformed_obs = transform_obs(obs, self.is_image)
        obs_embedding = self.encoder(transformed_obs)
        # Passes in single obs and action
        seq_state, _ = self.sequence_model.step(prev_latent, prev_action, model_state)
        seq_state = (
            seq_state * (1 - prev_done)
            + self.sequence_model.initial_state_from_reference(prev_done) * prev_done
        )
        post_dist = self.posterior(obs_embedding, seq_state)
        latent = post_dist.mode if det else post_dist.sample()

        return latent, seq_state

    @torch.no_grad()
    def step_obs_window(self, obs_window, action_window, det=False):
        # Passes in a sequence of obs (o_{t-l}-o_t)
        # and actions (a_{t-l}-a_{t-1})
        if isinstance(self.sequence_model, RSSM):
            raise TypeError("step_obs_window is for Mamba/Transformer; use step_obs_rssm for RSSM")
        transformed_obs = transform_obs(obs_window, self.is_image)
        obs_embedding = self.encoder(transformed_obs)
        post_dist = self.posterior(obs_embedding)
        latent = post_dist.mode if det else post_dist.sample()
        seq_state, _ = self.sequence_model.parallel_forward(latent[:, :-1], action_window)
        return latent[:, -1], seq_state[:, -1]

    @torch.no_grad()
    def imagine(
        self,
        context_obs,
        context_actions,
        horizon,
        actor,
        dones=None,
        initial_latent=None,
        initial_seq=None,
    ):
        # The only reason to pass initial latent and initial sequence is to match SheepRL
        # (and other similar Dreamer implementations) that pass the values from the
        # world model loss/rollout. Don't ever use them for Mamba or Transformer
        self.eval()
        actor.eval()

        if isinstance(self.sequence_model, RSSM):
            if initial_latent is None or initial_seq is None:
                transformed = transform_obs(context_obs, self.is_image)
                embeddings = self.encoder(transformed)
                latents, seq_states, _ = self.sequence_model.step_through(
                    embeddings, context_actions, self.posterior, dones
                )
                initial_latent = latents[:, -1]
                initial_seq = seq_states[:, -1]
            model_state = initial_seq
        else:
            B, T, _ = context_actions.shape
            transformed = transform_obs(context_obs, self.is_image)
            embeddings = self.encoder(transformed)
            latents = self.posterior(embeddings).sample()
            model_state = self.sequence_model.initial_state(B, context_actions.device)
            seq_outputs, model_state = self.sequence_model.parallel_forward(
                latents[:, :-1], context_actions[:, :-1], state=model_state
            )
            initial_latent = latents[:, -1]
            initial_seq = seq_outputs[:, -1]

        latents, actor_seq, head_seq, actions = self.sequence_model.imagine_rollout(
            initial_latent, initial_seq, model_state, horizon, actor, self.prior
        )
        self.train()
        actor.train()
        return latents, actor_seq, head_seq, actions

    def world_model_loss(self, obs, actions, rewards, terminated, dones):
        transformed_obs = transform_obs(obs, self.is_image)
        obs_embeddings = self.encoder(transformed_obs)
        latents, seq_states, post_logits, prior_logits = self.sequence_model.rollout(
            obs_embeddings, actions, self.posterior, self.prior, dones
        )
        if self.use_combined_state:
            states = make_state(latents, seq_states)
            continue_dist = self.continue_predictor(states)
            reward_twohot = self.reward_predictor(states)
            if self.decoder is None:
                reconstructions = None
            else:
                reconstructions = self.decoder(states)
        else:
            if latents.ndim > seq_states.ndim:
                decoder_latents = latents.flatten(-2)
            else:
                decoder_latents = latents
            continue_dist = self.continue_predictor(seq_states)
            reward_twohot = self.reward_predictor(seq_states)
            if self.decoder is None:
                reconstructions = None
            else:
                reconstructions = self.decoder(decoder_latents)
        context = {
            "obs": transformed_obs,
            "reconstructions": reconstructions,
            "rewards": rewards,
            "terminated": terminated,
            "dones": dones,
            "latents": latents,
            "seq_states": seq_states,
            "obs_embeddings": obs_embeddings,
            "post_logits": post_logits,
            "prior_logits": prior_logits,
        }
        obj_loss, loss_dict = self.objective(context)
        head_losses, head_loss_dict = self.head_losses(
            rewards, reward_twohot, terminated, continue_dist
        )

        dyn_loss, unclipped_kl = self.dynamics_loss(post_logits, prior_logits)
        repr_loss = self.representation_loss(post_logits, prior_logits)
        loss = (
            obj_loss * self.obj_coef
            + head_losses * self.head_coef
            + dyn_loss * self.dyn_coef
            + repr_loss * self.repr_coef
        )
        loss_dict["wm/loss/KL_div"] = to_numpy(dyn_loss)
        loss_dict["wm/loss/unclipped_KL"] = to_numpy(unclipped_kl)
        loss_dict = loss_dict | head_loss_dict
        return (
            loss,
            loss_dict,
            latents.detach(),
            seq_states.detach(),
        )

    def head_losses(self, reward, reward_twohot, terminated, continue_dist):
        reward_error = -reward_twohot.log_prob(reward, aggregate=False).mean()
        continue_error = -continue_dist.log_prob(1 - terminated.unsqueeze(-1)).mean()
        total_loss = reward_error + continue_error
        return total_loss, {
            "wm/loss/reward_loss": to_numpy(reward_error),
            "wm/loss/continue_loss": to_numpy(continue_error),
        }

    def dynamics_loss(self, post_logits, prior_logits):
        latent_post = Independent(OneHotCategoricalStraightThrough(logits=post_logits.detach()), 1)
        latent_prior = Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1)
        kl_div = kl_divergence(latent_post, latent_prior)
        return torch.clip(kl_div, min=self.free_nats).mean(), kl_div.mean()

    def representation_loss(self, post_logits, prior_logits):
        latent_post = Independent(OneHotCategoricalStraightThrough(logits=post_logits), 1)
        latent_prior = Independent(
            OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1
        )
        kl_div = kl_divergence(latent_post, latent_prior)
        return torch.clip(kl_div, min=self.free_nats).mean()
