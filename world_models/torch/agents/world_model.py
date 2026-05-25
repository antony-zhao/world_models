import numpy as np
import torch
from torch import nn
from torch.distributions import (
    Bernoulli,
    Categorical,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    kl_divergence,
)
from torch.distributions.utils import probs_to_logits
from torch.nn import functional as F

from world_models.torch.agents.models import DreamerDecoder, DreamerEncoder, Prior
from world_models.torch.common.distributions import DiscreteLatentDist
from world_models.torch.common.models import MLP
from world_models.torch.common.sequence_models import RSSM
from world_models.torch.common.utils import (
    TwoHotEncoding,
    make_state,
    to_numpy,
    transform_obs,
)


class DreamerWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = DreamerEncoder(config)
        self.decoder = DreamerDecoder(config)
        self.rssm = RSSM(config)
        self.dynamics_predictor = DreamerDynamics(config)
        self.reward_predictor = DreamerMLP(
            config.state_size, config.num_bins, config.hidden_dim, config.num_hiddens_world_model
        )
        self.continue_predictor = DreamerMLP(
            config.state_size, 1, config.hidden_dim, config.num_hiddens_world_model
        )
        self.initial_hidden = nn.Parameter(torch.zeros(config.hidden_state_size).float())
        self.rollout_length = config.rollout_length
        self.config = config
        self.is_image = self.config.obs_type == "image"
        self.bins = torch.linspace(config.bin_low, config.bin_high, config.num_bins).to(
            config.device
        )
        self.h_initial = None

    def compute_latents(self, obs_embedding, hidden):
        probs_enc = self.encoder.compute_latent(obs_embedding, hidden)
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1).rsample()
        probs_dyn = self.dynamics_predictor(hidden)
        return probs_enc, latent_enc, probs_dyn

    def imagine_step(self, latent, hidden, action):
        # treat this like how you would a normal environment step
        # state is {z, h}
        with torch.no_grad():
            state = torch.cat([latent, hidden], -1)
            continue_prob = F.sigmoid(self.continue_predictor(state))
            reward_logits = self.reward_predictor(state)
            reward = WeightedAverageOverBins(self.bins, reward_logits).weighted_average()
            next_hidden = self.rssm(latent, hidden, action)
            probs_dyn = self.dynamics_predictor(next_hidden)
            next_latent = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1).rsample()
        return (next_latent, next_hidden), reward, continue_prob

    def dynamic_step(self, obs_embedding, action, done, hidden, is_first):
        h_initial = self._get_hidden(action.shape[0])

        if hidden is None:
            hidden = h_initial
        else:
            hidden = hidden * (1 - is_first) + h_initial * is_first

        probs_enc, latent_enc, probs_dyn = self.compute_latents(obs_embedding, hidden)

        state = make_state(latent_enc, hidden)
        done = done.unsqueeze(-1)
        next_hidden = self.rssm(
            latent_enc.flatten(-2), hidden, action
        )  # * (1 - done) + done * h_initial
        return probs_enc, probs_dyn, state, next_hidden

    def _get_hidden(self, batch_size):
        return torch.tanh(self.initial_hidden.expand(batch_size, -1))

    def recurrent_step(self, hidden, latent, action):
        if self.config.action_type == "discrete":
            action = F.one_hot(action.long(), self.config.action_dim).float()
        next_hidden = self.rssm(latent.flatten(-2), hidden, action)
        return next_hidden

    def world_model_loss(self, obs, actions, rewards, dones):
        enc_probs = []
        dyn_probs = []
        states = []
        transformed_obs = transform_obs(obs, self.is_image).transpose(0, 1).contiguous()
        if self.is_image:
            T, B, C, H, W = transformed_obs.shape
            flat_transform_obs = transformed_obs.reshape(B * T, C, H, W)
        else:
            T, B, dim = transformed_obs.shape
            flat_transform_obs = transformed_obs.reshape(B * T, dim)
        obs_embeddings = self.encoder.embed_observations(flat_transform_obs)
        obs_embeddings = obs_embeddings.reshape(T, B, -1)
        actions = actions.transpose(0, 1).contiguous()
        rewards = rewards.transpose(0, 1).contiguous()
        dones = dones.transpose(0, 1).contiguous()
        is_first = torch.zeros_like(dones)
        is_first[0] = 1
        is_first[1:] = dones[:-1]
        is_first = is_first.unsqueeze(-1).float()
        hidden = None
        for i in range(T):
            probs_enc, probs_dyn, state, hidden = self.dynamic_step(
                obs_embeddings[i],
                actions[i],
                dones[i],
                hidden,
                is_first[i],  # i == 0
            )
            enc_probs.append(probs_enc)
            dyn_probs.append(probs_dyn)
            states.append(state)

        states = torch.stack(states)
        enc_probs = torch.stack(enc_probs)
        dyn_probs = torch.stack(dyn_probs)
        continue_preds = self.continue_predictor(states)
        reward_logits = self.reward_predictor(states)
        reconstructions = self.decoder.from_state(states.reshape(T * B, -1)).reshape(
            transformed_obs.shape
        )
        pred_loss, loss_dict = self.prediction_loss(
            transformed_obs, reconstructions, rewards, reward_logits, dones, continue_preds
        )
        dyn_loss = self.dynamics_loss(enc_probs, dyn_probs)
        rep_loss = self.representation_loss(enc_probs, dyn_probs)
        loss = (
            pred_loss * self.config.prediction_loss_coef
            + dyn_loss * self.config.dynamics_loss_coef
            + rep_loss * self.config.representation_loss_coef
        )
        loss_dict["loss/KL divergence"] = to_numpy(dyn_loss)
        return (
            loss,
            loss_dict,
            states.transpose(0, 1).detach(),
            obs_embeddings.transpose(0, 1).detach(),
        )

    def prediction_loss(self, obs, reconstruction, reward, reward_logits, dones, continue_logits):
        if self.is_image:
            obs = obs.flatten(2)
            reconstruction = reconstruction.flatten(2)
        reconstruction_error = ((obs - reconstruction) ** 2).sum(-1).mean()
        reward_prediction = WeightedAverageOverBins(self.bins, reward_logits)
        reward_error = -reward_prediction.log_prob(reward, aggregate=False).mean()
        continue_dist = Independent(Bernoulli(logits=continue_logits), 1)
        continue_error = -continue_dist.log_prob(1 - dones.unsqueeze(-1)).mean()
        total_loss = reconstruction_error + reward_error + continue_error * 10
        return total_loss, {
            "loss/reconstruction loss": to_numpy(reconstruction_error),
            "loss/reward loss": to_numpy(reward_error),
            "loss/continue loss": to_numpy(continue_error),
        }

    def dynamics_loss(self, probs_enc, probs_dyn):
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc.detach()), 1)
        latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn), 1)
        kl_div = kl_divergence(latent_enc, latent_dyn)
        return torch.clip(kl_div, min=self.config.free_nats).mean()

    def representation_loss(self, probs_enc, probs_dyn):
        latent_enc = Independent(OneHotCategoricalStraightThrough(probs_enc), 1)
        latent_dyn = Independent(OneHotCategoricalStraightThrough(probs_dyn.detach()), 1)
        kl_div = kl_divergence(latent_enc, latent_dyn)
        return torch.clip(kl_div, min=self.config.free_nats).mean()
