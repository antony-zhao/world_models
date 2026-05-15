from torch import nn

from world_models.torch.common.models import ChannelNorm
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
        self.output_size = (filters[-1], size, size)

    def forward(self, x):
        return self.layers(x)


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


class MultiEncoder(nn.Module):
    def __init__(self, img_encoder=ConvEncoder):
        super().__init__()
        pass
