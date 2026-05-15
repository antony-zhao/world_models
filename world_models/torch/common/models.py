import copy

import torch
from torch import nn


class MLP(nn.Module):
    # if hidden dims is specified then doesn't use skip connections
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        num_hiddens=2,
        act=nn.GELU,
        hidden_dims=None,
        final_act=None,
        skip_connections=False,
    ):
        super().__init__()
        if hidden_dims is not None:
            assert len(hidden_dims) + 1 == num_hiddens
            hidden_dim = hidden_dims[0]
            self.skip_connections = False
        else:
            self.skip_connections = True
        if skip_connections is not None:
            self.skip_connections = skip_connections
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hiddens = []
        for i in range(num_hiddens):
            if hidden_dims is None:
                self.hiddens.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.hiddens.append(nn.Linear(hidden_dim, hidden_dims[i + 1]))
                hidden_dim = hidden_dims[i + 1]
        self.hiddens = nn.ModuleList(self.hiddens)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = act()
        self.final_act = final_act

    def forward(self, x):
        x = self.act(self.input_layer(x))
        for i in range(len(self.hiddens)):
            if self.skip_connections:
                x = self.act(self.hiddens[i](x)) + x
            else:
                x = self.act(self.hiddens[i](x))
        logits = self.output_layer(x)
        if self.final_act is not None:
            return self.final_act(logits)
        return logits


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=nn.GELU):
        super().__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.act = act()

    def forward(self, x):
        x_skip = x.clone()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x + x_skip


class IMPALABlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.GELU, num_blocks=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res = nn.Sequential(
            *[ResBlock(out_channels, out_channels, 3, act) for _ in range(num_blocks)]
        )
        self.act = act()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.res(x)
        return x


class IMPALACNN(nn.Module):
    def __init__(self, image_size, num_blocks, image_channels=3, channel_base=16, act=nn.GELU):
        channels = [image_channels] + [channel_base * 2**i for i in range(num_blocks)]
        self.image_size = image_size
        self.image_channels = image_channels
        super().__init__()
        self.layers = nn.Sequential(
            *[IMPALABlock(channels[i], channels[i + 1], act) for i in range(num_blocks)]
        )
        self.output_dim = self.compute_output_dim()

    def compute_output_dim(self):
        x = torch.zeros(1, self.image_channels, self.image_size, self.image_size)
        x = self.layers(x)
        return x.view(-1).shape[0]

    def forward(self, obs):
        return self.layers(obs)


class ChannelNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DreamerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hiddens, act=nn.SiLU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True))
        layers.append(act())
        for _ in range(num_hiddens):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True))
            layers.append(act())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ContDreamerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hiddens, act=nn.SiLU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True))
        layers.append(act())
        for _ in range(num_hiddens):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True))
            layers.append(act())
        self.layers = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.mean_head(x), self.log_std_head(x)


class BlockLinear(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=8):
        super().__init__()
        self.networks = nn.ModuleList(
            [
                nn.Linear(input_size // num_blocks, output_size // num_blocks)
                for _ in range(num_blocks)
            ]
        )
        self.num_blocks = num_blocks

    def forward(self, x):
        output = []
        x_chunks = torch.split(x, x.shape[-1] // self.num_blocks, dim=-1)
        for i in range(self.num_blocks):
            output.append(self.networks[i](x_chunks[i]))
        return torch.cat(output, -1)


class TargetNetwork(nn.Module):
    def __init__(self, original_network, tau=None, update_freq=None):
        super().__init__()
        self.network = copy.deepcopy(original_network)
        self.original = original_network
        for param in self.network.parameters():
            param.requires_grad = False
        if tau is None and update_freq is None:
            raise RuntimeError("At least one of tau or update frequency should be specified")
        self.tau = tau  # esentially the moving average, slowly updates every time
        self.update_freq = update_freq
        self.i = 0

    def update(self):
        target_net_state_dict = self.network.state_dict()
        original_net_state_dict = self.original.state_dict()
        if self.tau is not None:
            for key in original_net_state_dict:
                target_net_state_dict[key] = original_net_state_dict[
                    key
                ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.network.load_state_dict(target_net_state_dict)
        else:
            self.i += 1
            if (self.i % self.update_freq) == 0:
                self.network.load_state_dict(original_net_state_dict)

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    @property
    def net(self):
        return self.network


def reparameterize_normal(mu, sigma):
    return torch.randn_like(mu) * sigma + mu
