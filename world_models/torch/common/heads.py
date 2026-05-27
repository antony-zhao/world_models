import torch
from torch import nn
from torch.distributions import Bernoulli, Distribution, Independent

from world_models.torch.common.distributions import DiscreteLatentDist, TwoHotEncoding
from world_models.torch.common.models import MLP
from world_models.torch.common.utils import symexp, symlog, unimix


class CategoricalHead(nn.Module):
    def __init__(self, num_categories, num_codes, unimix_prob=0.01):
        super().__init__()
        self.num_categories = num_categories
        self.num_codes = num_codes
        self.unimix_prob = unimix_prob

    def forward(self, logits):
        *batch_dims, _ = logits.shape
        reshaped_logits = logits.reshape(*batch_dims, self.num_categories, self.num_codes)
        probs = torch.softmax(reshaped_logits, -1)
        unimixed_probs = unimix(probs, self.num_codes, self.unimix_prob)
        return DiscreteLatentDist(probs=unimixed_probs)


class GaussianHead(nn.Module):
    def __init__(self):
        raise NotImplementedError


class TwoHotHead(nn.Module):
    def __init__(
        self,
        in_dim,
        num_bins,
        bin_low,
        bin_high,
        hidden_dim=256,
        n_layers=2,
        to_value=symexp,
        to_bin=symlog,
    ):
        super().__init__()
        bins = torch.linspace(bin_low, bin_high, num_bins)
        self.register_buffer("bins", bins)
        self.proj = MLP(in_dim, num_bins, hidden_dim, n_layers)
        self.to_value = to_value
        self.to_bin = to_bin

    def forward(self, x) -> TwoHotEncoding:
        logits = self.proj(x)
        return TwoHotEncoding(self.bins, logits, self.to_value, self.to_bin)


class BernoulliHead(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=256, n_layers=2):
        super().__init__()
        self.proj = MLP(in_dim, out_dim, hidden_dim, n_layers)

    def forward(self, x) -> Distribution:
        logits = self.proj(x)
        return Independent(Bernoulli(logits=logits), 1)


class DiscreteActionHead(nn.Module):
    pass


class ContinuousActionHead(nn.Module):
    pass
