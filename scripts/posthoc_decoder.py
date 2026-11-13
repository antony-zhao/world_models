"""Do the sprites survive in a reconstruction-free latent, and in imagined latents?

Trains a ConvDecoder on the *frozen* posterior latents of a saved checkpoint (nothing in
the world model is updated, so this costs no training-time compute), then:

  - reports held-out reconstruction error over the whole frame and inside a box around
    each sprite, for latents from real frames and for open-loop imagined latents at
    several horizons (prior rolled forward on the true actions). The "mean frame" row is
    the error of predicting the average training frame, i.e. what a decoder gets with no
    information about the sprite.
  - saves a PNG: real frames on top, decoded posterior latents below, decoded imagined
    latents at the bottom, one column per horizon step.

Usage:
    python scripts/posthoc_decoder.py --run runs/atari100k/<run_dir> --out decoded.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from latent_probe_atari import (
    GAMES,
    collect,
    encode,
    imagination_windows,
    load_agent,
    open_loop,
    sprite_boxes,
)

from world_models.torch.common.decoders import ConvDecoder


def train_decoder(decoder, z, obs, steps, batch, lr):
    opt = torch.optim.Adam(decoder.parameters(), lr)
    for step in range(steps):
        idx = torch.randint(0, len(z), (batch,), device=z.device)
        target = obs[idx].float() / 255.0 - 0.5
        # summed over pixels, as in ReconstructionObjective
        loss = ((decoder(z[idx]) - target) ** 2).flatten(1).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  decoder step {step + 1}/{steps}  loss {loss.item():.2f}")


@torch.no_grad()
def decode(decoder, z, batch=2048):
    return torch.cat([decoder(z[i : i + batch]) for i in range(0, len(z), batch)])


def region_errors(pred, target, boxes):
    """Mean squared error over the whole frame and inside each box. pred / target (N, 3, 64, 64),
    boxes {name: (N, 64, 64) bool tensor}."""
    error = ((pred - target) ** 2).mean(1)
    out = {"frame": error.mean().item()}
    for name, mask in boxes.items():
        out[name] = (error * mask).sum().item() / mask.sum().item()
    return out


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    agent, cfg = load_agent(args.run, args.checkpoint, device)
    wm = agent.world_model
    spec = GAMES.get(cfg.env.game)
    has_boxes = spec is not None and spec["pixel_map"] is not None

    data = collect(agent, cfg.env.game, args.steps, args.seed)
    n = len(data["obs"])
    obs = torch.from_numpy(data["obs"]).to(device)
    _, z = encode(wm, data["obs"], device)
    actions = F.one_hot(torch.from_numpy(data["actions"]).to(device), cfg.action_dim).float()
    boxes = sprite_boxes(data["ram"], spec) if has_boxes else {}
    boxes = {name: torch.from_numpy(mask).to(device) for name, mask in boxes.items()}

    split = int(n * 0.9)  # train on the first 90%, evaluate on the last 10%
    enc = cfg.wm.encoder
    decoder = ConvDecoder(
        z.shape[1], cfg.wm.encoder_conv_dim, enc.filter_base, enc.num_convs, enc.kernel_size, 3
    ).to(device)
    print(f"training decoder on {split} frozen latents...")
    train_decoder(decoder, z[:split], obs[:split], args.decoder_steps, args.batch, args.lr)
    decoder.eval()

    context, horizon = cfg.train.context_window, cfg.train.horizon
    starts = imagination_windows(data["boundary"], context, horizon, stride=3, start=split)
    imagined, real = open_loop(wm, z, actions, starts, context, horizon)
    # frame index of imagined step k (0-based) in window w: starts[w] + context + 1 + k
    frames = torch.tensor(starts, device=device)[:, None] + context + 1 + torch.arange(
        horizon, device=device
    )
    targets = obs[frames].float() / 255.0 - 0.5  # (W, H, 3, 64, 64)
    dec_real = decode(decoder, real.flatten(0, 1)).reshape(targets.shape)
    dec_imagined = decode(decoder, imagined.flatten(0, 1)).reshape(targets.shape)
    mean_frame = (obs[:split].float() / 255.0 - 0.5).mean(0)

    regions = ["frame"] + list(boxes)
    print(f"\nheld-out reconstruction mse x 1e3, {len(starts)} windows, context {context}")
    print(f"{'':28s}" + " ".join(f"{name:>13s}" for name in regions))

    def report(label, pred, k=None):
        sel = slice(None) if k is None else k
        window_boxes = {name: mask[frames][:, sel] for name, mask in boxes.items()}
        errors = region_errors(pred[:, sel].flatten(0, -4), targets[:, sel].flatten(0, -4), {
            name: mask.flatten(0, -3) for name, mask in window_boxes.items()
        })
        print(f"{label:28s}" + " ".join(f"{errors[name] * 1e3:13.2f}" for name in regions))

    report("mean frame (no information)", mean_frame.expand_as(targets))
    report("decoded posterior z", dec_real)
    for k in [k for k in (1, 2, 4, 8, 12, 16) if k <= horizon]:
        report(f"decoded imagined, k={k}", dec_imagined, k - 1)

    # ---- picture: a few windows, real / decoded posterior / decoded imagined per horizon step
    cols = [k - 1 for k in (1, 2, 4, 8, 12, 16) if k <= horizon]
    picks = np.linspace(0, len(starts) - 1, args.windows).astype(int)

    def to_image(x):
        return (x + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(
        3 * len(picks), len(cols), figsize=(2 * len(cols), 6 * len(picks)), squeeze=False
    )
    rows = (("real", targets), ("posterior z", dec_real), ("imagined", dec_imagined))
    for p, w in enumerate(picks):
        for r, (label, source) in enumerate(rows):
            for c, k in enumerate(cols):
                ax = axes[3 * p + r, c]
                ax.imshow(to_image(source[w, k]), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(label)
                if r == 0:
                    ax.set_title(f"k={k + 1}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory holding config.yaml")
    parser.add_argument("--checkpoint", default="checkpoint_latest.pt")
    parser.add_argument("--out", default="posthoc_decoder.png")
    parser.add_argument("--steps", type=int, default=30000, help="env steps to collect")
    parser.add_argument("--decoder-steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--windows", type=int, default=3, help="windows shown in the picture")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
