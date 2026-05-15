import torch
from torch import nn

from world_models.torch.common.models import ChannelNorm, NormAndAct
from world_models.torch.common.utils import compute_pad


class ConvDecoder(nn.Module):
    # does the reverse of the encoder conv, pass in reversed filters
    def __init__(
        self,
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
                layers.append(
                    NormAndAct(
                        image_channels if i == len(filters) - 1 else filters[i + 1], norm, act
                    )
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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


class MultiDecoder(nn.Module):
    def __init__(self, img_decoder=ConvDecoder):
        super().__init__()
        pass
