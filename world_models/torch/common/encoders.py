import numpy as np
import torch
from torch import nn

from world_models.torch.common.models import MLP, ChannelNorm
from world_models.torch.common.utils import compute_pad


class ConvEncoder(nn.Module):
    # built for 64x64 observations and downscales them to 4x4
    # Generally follows the Dreamer convolutional encoder
    def __init__(
        self,
        filter_base=8,
        num_convs=4,
        kernel_size=4,
        image_channels=3,
        input_size=64,
        act=nn.SiLU,
        norm=ChannelNorm,
    ):
        super().__init__()
        filters = [filter_base * 2**i for i in range(num_convs)]
        layers = []
        for i, filter in enumerate(filters):
            layers.append(
                nn.Conv2d(
                    image_channels if i == 0 else filters[i - 1],
                    filter,
                    kernel_size,
                    stride=2,
                    padding=compute_pad(kernel_size, 2),
                    bias=i == len(filters) - 1,
                )
            )
            if i < len(filters) - 1:
                layers.append(norm(filter))
                layers.append(act())
        self.layers = nn.Sequential(*layers)
        size = input_size // (2**num_convs)
        self.conv_dim = (filters[-1], size, size)
        self.output_dim = np.prod(self.conv_dim)

    def forward(self, x):
        *batch, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        x = self.layers(x)
        return x.reshape(*batch, -1)


class VectorEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass


class ViTEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass


# Old code, don't look at it too hard, just need it for reference if I end up needing it
class MultiEncoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    # hidden_state_size is h_t (the hidden state of the recurrent network)
    # categories is number of rows, codes is the number of columns (softmaxed over codes)
    # hidden dim is just the hidden dim of linear layers
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError
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
