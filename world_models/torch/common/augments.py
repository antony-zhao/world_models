"""
For torchvision-style transforms, or just outright using some of them
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import v2


def random_resize_crop(size, views, scale=(0.8, 1.0)):
    raise NotImplementedError
    transform = v2.Compose(
        [
            v2.RandomResizedCrop(size, scale=scale, antialias=True),
        ]
    )

    def transform_obs(obs):
        return torch.stack([transform(obs) for _ in range(views)])

    return transform_obs


def _shift_images(x, pad):
    # DrQ-v2 RandomShiftsAug: replicate-pad by `pad`, then crop back to the original size at an
    # independent integer offset per image, i.e. a shift of up to +-pad px with edge pixels filling
    # the border. x: (N, C, H, W) with H == W
    n, _, h, w = x.shape
    assert h == w, f"random shift expects square images, got {h}x{w}"
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device).to(x.dtype)
    shift = shift * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base_grid + shift, padding_mode="zeros", align_corners=False)


def random_shift(views, pad=1, intensity=0.05):
    """DrQ / SPR-style Atari augmentation: per-image random shift, then per-image intensity.

    Returns a function mapping obs (..., C, H, W) to views (V, ..., C, H, W), each image with its
    own shift and intensity. obs is expected centered (transform_obs: pixels / 255 - 0.5), so the
    intensity factor 1 + intensity * clip(N(0, 1), -2, 2) is applied on the [0, 1] scale, i.e. as a
    brightness change. intensity=0 turns it off.
    """

    def augment(obs):
        *lead, c, h, w = obs.shape
        x = obs.reshape(-1, c, h, w).repeat(views, 1, 1, 1)  # (V * N, C, H, W)
        x = _shift_images(x, pad)
        if intensity > 0:
            noise = torch.randn(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
            x = (x + 0.5) * (1.0 + intensity * noise.clamp(-2.0, 2.0)) - 0.5
        return x.reshape(views, *lead, c, h, w)

    return augment
