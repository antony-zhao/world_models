import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from world_models.torch.common.distributions import DreamerLatentDist
from world_models.torch.common.models import DreamerMLP
from world_models.torch.common.sequence_models import RSSM
from world_models.torch.common.utils import (
    TwoHotEncoding,
    make_state,
    to_numpy,
    transform_obs,
)
from world_models.torch.dreamer.models import DreamerDecoder, DreamerEncoder, Prior


class DreamerWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = DreamerEncoder(config)
        self.decoder = DreamerDecoder(config)
        self.rssm = RSSM(config)
        self.prior = Prior(config)
        self.reward_predictor = DreamerMLP(
            config.state_size,
            config.num_bins,
            config.hidden_dim,
            config.num_hiddens_world_model,
        )
        self.continue_predictor = DreamerMLP(
            config.state_size, 1, config.hidden_dim, config.num_hiddens_world_model
        )
        self.initial_hidden = nn.Parameter(torch.zeros(config.hidden_state_size).float())
        self.rollout_length = config.rollout_length
        self.obs_type = config.obs_type
        self.is_image = config.obs_type == "image"
        self.bins = torch.linspace(config.bin_low, config.bin_high, config.num_bins).to(
            config.device
        )
        self.config = config

    def compute_latents(self, obs_embedding, hidden):
        probs_enc = self.encoder.compute_latent(obs_embedding, hidden)
        latent_enc = DreamerLatentDist(probs=probs_enc).rsample()
        probs_dyn = self.dynamics_predictor(hidden)
        return probs_enc, latent_enc, probs_dyn

    def imagine_step(self, latent, hidden, action):
        # treat this like how you would a normal environment step
        # state is {z, h}
        with torch.no_grad():
            state = torch.cat([latent, hidden], -1)
            continue_prob = None
            if self.config.cont_pred:
                continue_prob = F.sigmoid(self.continue_predictor(state))
            reward = None
            if self.config.reward_pred:
                reward_logits = self.reward_predictor(state)
                reward = TwoHotEncoding(self.bins, reward_logits).weighted_average()
            next_hidden = self.rssm(latent, hidden, action)
            probs_dyn = self.dynamics_predictor(next_hidden)
            next_latent = DreamerLatentDist(probs_dyn).rsample()
        return (next_latent, next_hidden), reward, continue_prob

    def dynamic_step(self, obs_embedding, action, done, hidden):
        h_initial = self._get_hidden(action.shape[0])

        if hidden is None:
            hidden = h_initial

        probs_enc, latent_enc, probs_dyn = self.compute_latents(obs_embedding, hidden)

        state = make_state(latent_enc, hidden)
        done = done.unsqueeze(-1)
        next_hidden = (
            self.rssm(latent_enc.flatten(-2), hidden, action) * (1 - done) + done * h_initial
        )
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
        if self.obs_type != "multi":
            transformed_obs = transform_obs(obs, self.is_image).transpose(0, 1).contiguous()
            if self.is_image:
                T, B, C, H, W = transformed_obs.shape
                flat_transform_obs = transformed_obs.reshape(B * T, C, H, W)
            else:
                T, B, dim = transformed_obs.shape
                flat_transform_obs = transformed_obs.reshape(B * T, dim)
        else:
            img, vec = obs
            transformed_img = transform_obs(img, is_image=True).transpose(0, 1).contiguous()
            transformed_vec = transform_obs(vec, is_image=False).transpose(0, 1).contiguous()
            transformed_obs = (transformed_img, transformed_vec)
            T, B, C, H, W = transformed_img.shape
            flat_transform_img = transformed_img.reshape(B * T, C, H, W)
            flat_transform_vec = transformed_vec.reshape(B * T, -1)
            flat_transform_obs = (flat_transform_img, flat_transform_vec)

        obs_embeddings = self.encoder.embed_observations(flat_transform_obs)
        obs_embeddings = obs_embeddings.reshape(T, B, -1)
        actions = actions.transpose(0, 1).contiguous()
        rewards = rewards.transpose(0, 1).contiguous()
        dones = dones.transpose(0, 1).contiguous()
        hidden = None

        for i in range(T):
            probs_enc, probs_dyn, state, hidden = self.dynamic_step(
                obs_embeddings[i], actions[i], dones[i], hidden
            )
            enc_probs.append(probs_enc)
            dyn_probs.append(probs_dyn)
            states.append(state)

        states = torch.stack(states)
        enc_probs = torch.stack(enc_probs)
        dyn_probs = torch.stack(dyn_probs)
        continue_preds = self.continue_predictor(states) if self.config.cont_pred else None
        reward_logits = self.reward_predictor(states) if self.config.reward_pred else None
        if self.obs_type == "multi":
            img_recon, vec_recon = self.decoder.from_state(states.reshape(T * B, -1))
            reconstructions = (
                img_recon.reshape(transformed_img.shape),
                vec_recon.reshape(transformed_vec.shape),
            )
        else:
            reconstructions = self.decoder.from_state(states.reshape(T * B, -1)).reshape(
                transformed_obs.shape
            )

        pred_loss, loss_dict = self.prediction_loss(
            transformed_obs,
            reconstructions,
            rewards,
            reward_logits,
            dones,
            continue_preds,
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
