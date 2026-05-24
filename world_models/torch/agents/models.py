import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from world_models.torch.common.decoders import ConvDecoder
from world_models.torch.common.distributions import DreamerLatentDist
from world_models.torch.common.encoders import ConvEncoder
from world_models.torch.common.models import MLP
from world_models.torch.common.sequence_models import RSSM
from world_models.torch.common.utils import (
    TwoHotEncoding,
    make_state,
    to_numpy,
    transform_obs,
    unimix,
)


class DreamerEncoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    # hidden_state_size is h_t (the hidden state of the recurrent network)
    # categories is number of rows, codes is the number of columns (softmaxed over codes)
    # hidden dim is just the hidden dim of linear layers
    def __init__(self, config):
        super().__init__()
        if config.obs_type == "image":
            self.encoder = ConvEncoder(
                config.filter_base,
                config.num_convs,
                config.kernel_size,
                config.num_channels,
                config.image_size,
                config.act,
            )
            output_dim = np.prod(self.encoder.output_size)
        elif config.obs_type == "vector":
            self.encoder = nn.Linear(config.obs_dim, config.hidden_dim)
            output_dim = config.hidden_dim
        elif config.obs_type == "multi":  # For handling stuff like image + proprio
            self.conv_encoder = ConvEncoder(
                config.filter_base,
                config.num_convs,
                config.kernel_size,
                config.num_channels,
                config.image_size,
                config.act,
            )
            self.encoder = nn.Linear(
                np.prod(self.conv_encoder.output_size) + config.obs_dim,
                config.hidden_dim,
            )
            output_dim = config.hidden_dim
        self.posterior = MLP(
            output_dim + config.hidden_state_size,
            config.latent_size,
            config.hidden_dim,
            num_hiddens=config.num_hiddens_world_model,
        )
        self.num_categoricals = config.num_categoricals
        self.num_codes = config.num_codes
        self.obs_type = config.obs_type
        self.is_image = config.obs_type == "image"
        self.config = config

    def forward(self, x, h):
        # x as in the observation specifically, h is the same hidden state
        embedded = self.embed_observations(x)
        return self.compute_latent(embedded, h)

    def embed_observations(self, x):
        # since the encoder computation can be batched without
        # needing to incorporate the hidden state
        if self.obs_type != "multi":
            encoded = self.encoder(x)
            if self.is_image:
                encoded = torch.flatten(encoded, -3)
        else:
            img, vec = x
            encoded_img = self.conv_encoder(img).flatten(-3)
            encoded = self.encoder(torch.cat([encoded_img, vec], dim=-1))
        return encoded

    def compute_latent(self, embedded, hidden):
        states = torch.cat([embedded, hidden], 1)
        logits = self.posterior(states)
        probs = torch.softmax(
            logits.reshape(logits.shape[0], self.num_categoricals, self.num_codes), -1
        )
        unimixed_probs = unimix(probs, self.num_codes, self.config.latent_unimix)
        return unimixed_probs


class DreamerDecoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    def __init__(self, config):
        super().__init__()
        self.obs_type = config.obs_type
        if config.obs_type == "image":
            self.input_dim = config.output_dim
            self._in = nn.Linear(config.state_size, np.prod(self.input_dim))
            self.decoder = ConvDecoder(
                config.filter_base,
                config.num_convs,
                config.kernel_size,
                config.num_channels,
                config.act,
            )
        elif self.obs_type == "vector":
            self._in = None
            self.decoder = MLP(
                config.state_size,
                config.obs_dim,
                config.hidden_dim,
                num_hiddens=config.num_hiddens_world_model,
            )
        elif self.obs_type == "multi":
            self.input_dim = config.output_dim
            self._in = nn.Linear(config.state_size, np.prod(self.input_dim))
            self.conv_decoder = ConvDecoder(
                config.filter_base,
                config.num_convs,
                config.kernel_size,
                config.num_channels,
                config.act,
            )
            self.vec_decoder = MLP(
                config.state_size,
                config.obs_dim,
                config.hidden_dim,
                num_hiddens=config.num_hiddens_world_model,
            )

    def from_state(self, x):
        if self.obs_type != "multi":
            if self._in:
                x = self._in(x)
            if self.obs_type == "image":
                x = x.reshape(-1, *self.input_dim)
            reconstruction = self.decoder(x)
        else:
            conv_in = self._in(x).reshape(-1, *self.input_dim)
            img_recon = self.conv_decoder(conv_in)
            vec_recon = self.vec_decoder(x)
            reconstruction = (img_recon, vec_recon)
        return reconstruction

    def forward(self, z, h):
        x = make_state(z, h)
        reconstruction = self.from_state(x)
        return reconstruction


class Posterior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dynamics = MLP(
            config.state_size,
            config.latent_size,
            config.hidden_dim,
            config.num_hiddens_world_model,
        )

    def forward(self, latent, hidden):
        state = make_state(latent, hidden)
        logits_dyn = self.dynamics(state)
        logits_dyn = logits_dyn.reshape(-1, self.config.num_categoricals, self.config.num_codes)
        probs_dyn = torch.softmax(logits_dyn, -1)
        probs_dyn = unimix(probs_dyn, self.config.num_codes, self.config.latent_unimix)
        return probs_dyn


class Prior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dynamics = MLP(
            config.hidden_state_size,
            config.latent_size,
            config.hidden_dim,
            config.num_hiddens_world_model,
        )

    def forward(self, hidden):
        logits_dyn = self.dynamics(hidden)
        logits_dyn = logits_dyn.reshape(-1, self.config.num_categoricals, self.config.num_codes)
        probs_dyn = torch.softmax(logits_dyn, -1)
        probs_dyn = unimix(probs_dyn, self.config.num_codes, self.config.latent_unimix)
        return probs_dyn
