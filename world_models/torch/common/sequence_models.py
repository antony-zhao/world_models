import torch
from torch import nn
from torch.nn import functional as F

from world_models.torch.common.models import BlockLinear


class DreamerGRU(nn.Module):
    def __init__(self, hidden_state_size, use_block_linear=True):
        super().__init__()
        if use_block_linear:
            self.layer = BlockLinear(hidden_state_size, hidden_state_size * 3)
        else:
            self.layer = nn.Linear(hidden_state_size, hidden_state_size * 3)
        self.hidden_state_size = hidden_state_size

    def forward(self, x, h):
        x = self.layer(x)
        reset, cand, update = torch.split(x, self.hidden_state_size, -1)
        reset = F.sigmoid(reset)
        cand = F.tanh(reset * cand)
        update = F.sigmoid(update - 1)
        h_new = update * cand + (1 - update) * h
        return h_new


class RSSM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError


class MambaSequenceModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError


class TransformerSequenceModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError
