import numpy as np
from torch import nn

from world_models.torch.common.models import MLP, ChannelNorm
from world_models.torch.common.utils import compute_pad


class ConvDecoder(nn.Module):
    # does the reverse of the encoder conv, pass in reversed filters
    def __init__(
        self,
        in_dim,
        conv_dim,
        filter_base=8,
        num_convs=4,
        kernel_size=4,
        image_channels=3,
        act=nn.SiLU,
        norm=ChannelNorm,
    ):
        super().__init__()
        filters = [filter_base * 2**i for i in reversed(range(num_convs))]
        layers = []
        self.conv_dim = conv_dim
        layers.append(nn.Linear(in_dim, np.prod(conv_dim)))
        layers.append(nn.Unflatten(-1, self.conv_dim))
        for i, filter in enumerate(filters):
            layers.append(
                nn.ConvTranspose2d(
                    filter,
                    image_channels if i == len(filters) - 1 else filters[i + 1],
                    kernel_size,
                    stride=2,
                    padding=compute_pad(kernel_size, 2),
                    bias=i == len(filters) - 1,
                )
            )
            if i < len(filters) - 1:
                layers.append(norm(filters[i + 1]))
                layers.append(act())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # allows it to handle B, dim or B, T, dim
        *batch, d = x.shape
        x = x.reshape(-1, d)
        x = self.layers(x)
        return x.reshape(*batch, *x.shape[1:])


class VectorDecoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass


class ViTDecoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass


# Old code, don't look at it too hard, just need it for reference if I end up needing it
class MultiDecoder(nn.Module):
    # if obs_type isn't image then input_dim should be specified
    def __init__(self, config):
        raise NotImplementedError
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
