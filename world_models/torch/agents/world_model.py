import torch
from torch import nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough, kl_divergence

from world_models.torch.common.heads import BernoulliHead, TwoHotHead
from world_models.torch.common.models import Posterior, Prior
from world_models.torch.common.sequence_models import RSSM, SequenceModel
from world_models.torch.common.utils import to_numpy, transform_obs


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

        self.obj_coef = obj_coef
        self.dyn_coef = dyn_coef
        self.repr_coef = repr_coef
        self.head_coef = head_coef
        self.free_nats = free_nats

    def initial_state(self, batch_size, device):
        return self.sequence_model.initial_state(batch_size, device)

    def step_obs(self, obs, action, model_state=None, det=False):
        # action should be preprocessed into one_hot before being passed here
        # RSSM and Transformer/Mamba have pretty big differences, RSSM handles the model state
        # Whereas the others take in a sequence of observations and actions to produce the
        # output state/latents.
        # Returns the latent, state, and model_state (for rssm to reuse)
        transformed_obs = transform_obs(obs, self.is_image)
        obs_embedding = self.encoder(transformed_obs)
        if isinstance(self.sequence_model, RSSM):
            # Passes in single obs and action
            post_dist = self.posterior(obs_embedding, model_state)
            latent = post_dist.mode() if det else post_dist.sample()
            seq_state = model_state
            _, model_state = self.sequence_model.step(latent, action, model_state)
        else:
            # Passes in a sequence of obs (o_{t-l}-o_t)
            # and actions (a_{t-l}-a_{t-1})
            post_dist = self.posterior(obs_embedding)
            latent = post_dist.mode() if det else post_dist.sample()
            seq_state, _ = self.sequence_model.parallel_forward(latent[:, :-1], action)
            seq_state = seq_state[:, -1]
            latent = latent[:, -1]
        return latent, seq_state, model_state

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
        B, T, _ = context_actions.shape

        transformed = transform_obs(context_obs, self.is_image)
        embeddings = self.encoder(transformed)
        if isinstance(self.sequence_model, RSSM):
            if initial_latent is None or initial_seq is None:
                latents, seq_states, _ = self.sequence_model.step_through(
                    embeddings, context_actions, self.posterior, dones
                )
                initial_latent = latents[:, -1]
                initial_seq = seq_states[:, -1]
            model_state = initial_seq
        else:
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
        latents, states, post_logits, prior_logits = self.sequence_model.rollout(
            obs_embeddings, actions, self.posterior, self.prior, dones
        )
        continue_dist = self.continue_predictor(states)
        reward_twohot = self.reward_predictor(states)
        if self.decoder is None:
            reconstructions = None
        else:
            reconstructions = self.decoder(states)
        obj_loss, loss_dict = self.objective(  # placeholder for now
            transformed_obs, reconstructions, rewards, dones
        )
        head_losses, head_loss_dict = self.head_losses(
            rewards, reward_twohot, terminated, continue_dist
        )

        dyn_loss = self.dynamics_loss(post_logits, prior_logits)
        repr_loss = self.representation_loss(post_logits, prior_logits)
        loss = (
            obj_loss * self.obj_coef
            + head_losses * self.head_coef
            + dyn_loss * self.dyn_coef
            + repr_loss * self.repr_coef
        )
        loss_dict["loss/KL_div"] = to_numpy(dyn_loss)
        loss_dict = loss_dict | head_loss_dict
        return (
            loss,
            loss_dict,
            states.detach(),
            obs_embeddings.detach(),
        )

    def head_losses(self, reward, reward_twohot, terminated, continue_dist):
        reward_error = -reward_twohot.log_prob(reward, aggregate=False).mean()
        continue_error = -continue_dist.log_prob(1 - terminated.unsqueeze(-1)).mean()
        total_loss = reward_error + continue_error
        return total_loss, {
            "loss/reward loss": to_numpy(reward_error),
            "loss/continue loss": to_numpy(continue_error),
        }

    def dynamics_loss(self, post_logits, prior_logits):
        latent_post = Independent(OneHotCategoricalStraightThrough(logits=post_logits.detach()), 1)
        latent_prior = Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1)
        kl_div = kl_divergence(latent_post, latent_prior)
        return torch.clip(kl_div, min=self.free_nats).mean()

    def representation_loss(self, post_logits, prior_logits):
        latent_post = Independent(OneHotCategoricalStraightThrough(logits=post_logits), 1)
        latent_prior = Independent(
            OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1
        )
        kl_div = kl_divergence(latent_post, latent_prior)
        return torch.clip(kl_div, min=self.free_nats).mean()
