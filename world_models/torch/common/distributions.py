import torch
from torch import autograd, distributions
from torch.distributions import Independent, OneHotCategoricalStraightThrough


class DreamerLatentDist:
    def __init__(self, probs=None, logits=None):
        self.dist = Independent(OneHotCategoricalStraightThrough(probs=probs, logits=logits), 1)

    def sample(self):
        return self.dist.sample()

    def deterministic(self):
        return self.dist.mode()


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
