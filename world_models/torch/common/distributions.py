import torch
import torch.nn.functional as F
from torch import autograd, distributions
from torch.distributions import Independent, OneHotCategoricalStraightThrough

from world_models.torch.common.utils import symexp, symlog


class DreamerLatentDist:
    def __init__(self, probs=None, logits=None):
        self.dist = Independent(OneHotCategoricalStraightThrough(probs=probs, logits=logits), 1)

    @property
    def probs(self):
        return self.dist.base_dist.probs

    @property
    def logits(self):
        return self.dist.base_dist.logits

    def sample(self):
        return self.dist.sample()

    def rsample(self):
        return self.dist.rsample()

    def mode(self):
        return self.dist.mode()


class TwoHotEncoding:
    def __init__(self, bins, logits=None, to_value=symexp, to_bin=symlog):
        self.logits = logits
        self.probs = torch.softmax(self.logits, -1)
        self.bins = bins
        self.to_value = to_value
        self.to_bin = to_bin

    def mean(self):
        weighted_average = self.probs @ self.bins
        return self.to_value(weighted_average)

    def two_hot(self, vals):
        index_1 = torch.bucketize(vals, self.bins) - 1
        index_1 = index_1.clamp(0, len(self.bins) - 2)
        index_2 = index_1 + 1
        b_k = self.bins[index_1]
        b_k2 = self.bins[index_2]
        proportion_2 = torch.abs(b_k - vals) / torch.abs(b_k2 - b_k)
        proportion_1 = torch.abs(b_k2 - vals) / torch.abs(b_k2 - b_k)
        one_hot_1 = F.one_hot(index_1, len(self.bins))
        one_hot_2 = F.one_hot(index_2, len(self.bins))
        two_hot_encoded = (
            proportion_1.unsqueeze(-1) * one_hot_1 + proportion_2.unsqueeze(-1) * one_hot_2
        )
        return two_hot_encoded

    def log_prob(self, vals, aggregate=True):
        # basically just the loss
        target = self.two_hot(self.to_bin(vals)).detach()
        log_probs = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        loss = (target * log_probs).sum(-1)
        return loss.mean() if aggregate else loss


# safetanh code comes from torchrl


def _safetanh(x, eps):  # noqa: D103
    lim = 1.0 - eps
    y = x.tanh()
    return y.clamp(-lim, lim)


def _safeatanh(y, eps):  # noqa: D103
    lim = 1.0 - eps
    return y.clamp(-lim, lim).atanh()


class _SafeaTanhNoEps(autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(tanh_val):
        eps = torch.finfo(tanh_val.dtype).resolution
        lim = 1.0 - eps
        output = tanh_val.clamp(-lim, lim)
        # ctx.save_for_backward(output)
        output = output.atanh()
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        tanh_val = inputs[0]
        eps = torch.finfo(tanh_val.dtype).resolution

        # ctx.mark_non_differentiable(ind, ind_inv)
        # # Tensors must be saved via ctx.save_for_backward. Please do not
        # # assign them directly onto the ctx object.
        ctx.save_for_backward(tanh_val)
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (tanh_val,) = ctx.saved_tensors
        eps = ctx.eps
        lim = 1.0 - eps
        output = tanh_val.clamp(-lim, lim)
        return (grad / (1 - output.pow(2)),)


class _SafeTanhNoEps(autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(input):
        output = input.tanh()
        eps = torch.finfo(input.dtype).resolution
        lim = 1.0 - eps
        output = output.clamp(-lim, lim)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (output,) = ctx.saved_tensors
        return (grad * (1 - output.pow(2)),)


safetanh_noeps = _SafeTanhNoEps.apply
safeatanh_noeps = _SafeaTanhNoEps.apply


class SafeTanhTransform(distributions.TanhTransform):
    def _call(self, x):
        return safetanh_noeps(x)

    def _inverse(self, y):
        return safeatanh_noeps(y)


class SafeTanhNormal(distributions.TransformedDistribution):
    def __init__(self, loc, scale, lower_bound=-1.0, upper_bound=1.0):
        base_dist = distributions.Normal(loc, scale)

        affine_scale = (upper_bound - lower_bound) / 2.0
        affine_loc = (upper_bound + lower_bound) / 2.0

        transforms = [
            SafeTanhTransform(),
            distributions.AffineTransform(loc=affine_loc, scale=affine_scale),
        ]

        super().__init__(base_dist, transforms)
