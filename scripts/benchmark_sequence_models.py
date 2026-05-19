"""
Benchmark for sequence models (RSSM, Mamba, Transformer) at world-model
training shapes.

Configs benchmarked:
- Drama published: d_model=512, n=2 (matches Drama paper, ~7M agent params)
- Drama deep: d_model=512, n=8 (~13M, scale-up via depth)
- STORM-sized: d_model=256, n=2 (matches STORM transformer baseline)
- Mid Transformer: d_model=512, n=4 (transformer at Drama deep param count)
- DreamerV3 S/XL: original DreamerV3 model sizes for comparison

Run from project root:
    python scripts/benchmark_sequence_models.py
"""

# testing accuracy, compute time, and vram requirements, written by Claude.
import time

import torch
from torch import nn

from world_models.torch.common.sequence_models import (
    RSSM,
    MambaSequenceModel,
    TransformerSequenceModel,
)


def get_param_memory(model):
    """Memory used by parameters and persistent buffers (bytes)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def fmt_bytes(b):
    """Format bytes as MB or GB."""
    if b < 1024**2:
        return f"{b / 1024:.1f} KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def _shape_and_timing_test(model, latent_size, action_dim, B=16, T=64, n_iters=100, atol=1e-5):
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    output_dim = model.output_dim

    latents = torch.randn(B, T, latent_size, device=device)
    actions = torch.randn(B, T, action_dim, device=device)

    # --- Shape checks ---
    outputs, _ = model(latents, actions)
    assert outputs.shape == (B, T, output_dim), (
        f"outputs shape {outputs.shape} != expected ({B}, {T}, {output_dim})"
    )

    state = model.initial_state(B, device)
    outputs, _ = model(latents, actions, state)
    assert outputs.shape == (B, T, output_dim)

    dones = torch.zeros(B, T, device=device)
    dones[2, 8] = 1.0
    outputs, _ = model(latents, actions, dones=dones)
    assert outputs.shape == (B, T, output_dim)

    # step vs forward consistency
    state = model.initial_state(B, device)
    step_outputs = []
    for t in range(T):
        out, state = model.step(latents[:, t], actions[:, t], state)
        step_outputs.append(out)
    step_outputs = torch.stack(step_outputs, dim=1)

    forward_outputs, _ = model(latents, actions)
    max_diff = (step_outputs - forward_outputs).abs().max().item()
    assert torch.allclose(step_outputs, forward_outputs, atol=atol), (
        f"step and forward differ (max diff: {max_diff}, atol: {atol})"
    )

    print(
        f"{model.__class__.__name__} shape and consistency: OK "
        f"(max step/forward diff: {max_diff:.2e})"
    )

    # --- Timing ---
    for _ in range(10):
        _ = model(latents, actions)
    if is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model(latents, actions)
    if is_cuda:
        torch.cuda.synchronize()
    forward_time = (time.perf_counter() - start) / n_iters * 1000

    s = model.initial_state(B, device)
    for _ in range(10):
        _, s = model.step(latents[:, 0], actions[:, 0], s)
    if is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        s = model.initial_state(B, device)
        for t in range(T):
            _, s = model.step(latents[:, t], actions[:, t], s)
    if is_cuda:
        torch.cuda.synchronize()
    step_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"  forward (B={B}, T={T}): {forward_time:.2f} ms ({forward_time / T:.3f} ms/step)")
    print(f"  step    (B={B}, T={T}): {step_time:.2f} ms ({step_time / T:.3f} ms/step)")


def run_workload(model, latents, actions, mode, B, T, n_warmup, n_iters, track_memory=False):
    """Run a single workload and return (time_ms, peak_memory_bytes_above_baseline)."""
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    if mode == "forward":

        def run():
            return model(latents, actions)
    elif mode == "forward+backward":

        def run():
            out, _ = model(latents, actions)
            out.sum().backward()
            model.zero_grad()
    elif mode == "step_loop":

        def run():
            s = model.initial_state(B, device)
            for t_idx in range(T):
                _, s = model.step(latents[:, t_idx], actions[:, t_idx], s)
    elif mode == "step":
        s = model.initial_state(B, device)

        def run():
            nonlocal s
            _, s = model.step(latents[:, 0], actions[:, 0], s)

    for _ in range(n_warmup):
        run()
    sync()

    if track_memory and is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)

    start = time.perf_counter()
    for _ in range(n_iters):
        run()
    sync()
    elapsed_ms = (time.perf_counter() - start) / n_iters * 1000

    peak_bytes = 0
    if track_memory and is_cuda:
        peak_bytes = torch.cuda.max_memory_allocated(device) - baseline

    return elapsed_ms, peak_bytes


def benchmark_config(name, model_factory, latent_size, action_dim, track_memory=True):
    """Run the full workload suite on a single model configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = model_factory(latent_size=latent_size, action_dim=action_dim).to(device)
    except Exception as e:
        print(f"\n=== {name} ===")
        print(f"  FAILED to construct: {e}")
        return

    n_params = sum(p.numel() for p in model.parameters())
    d_model = model.output_dim
    param_mem = get_param_memory(model)

    print(f"\n=== {name} ===")
    print(f"  d_model: {d_model}, parameters: {n_params:,} ({fmt_bytes(param_mem)})")

    workloads = [
        (16, 128, "forward+backward", "WM training (Drama config)", 5, 20),
        (1024, 16, "step_loop", "Imagination (Drama config)", 3, 10),
        (256, 16, "step_loop", "Imagination (smaller batch fallback)", 3, 10),
        (1, 16, "forward", "Online inference (single env, context)", 5, 20),
        (16, 16, "forward", "Online inference (vectorized 16 envs)", 5, 20),
        (1, 1, "step", "Per-step inference (B=1)", 10, 50),
    ]

    for B, T, mode, desc, n_warmup, n_iters in workloads:
        try:
            latents = torch.randn(B, T, latent_size, device=device)
            actions = torch.randn(B, T, action_dim, device=device)
            t, peak_mem = run_workload(
                model,
                latents,
                actions,
                mode,
                B,
                T,
                n_warmup,
                n_iters,
                track_memory=track_memory,
            )

            mem_str = f"  ({fmt_bytes(peak_mem)})" if track_memory and device == "cuda" else ""

            if mode == "step_loop":
                print(
                    f"  B={B:>4} T={T:>3} {mode:<18} {t:>9.2f} ms \
                            ({t / T:.2f} ms/step){mem_str} [{desc}]"
                )
            else:
                print(f"  B={B:>4} T={T:>3} {mode:<18} {t:>9.2f} ms{mem_str} [{desc}]")

            del latents, actions
            if device == "cuda":
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  B={B:>4} T={T:>3} {mode:<18}       OOM  [{desc}]")
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  B={B:>4} T={T:>3} {mode:<18}    FAILED: {str(e)[:60]}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


cfg = dict(latent_size=32 * 32, action_dim=18)
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# === Shape and consistency tests on small configs first ===
print("=" * 70)
print("Shape and consistency tests (small configs)")
print("=" * 70)

rssm = RSSM(
    d_model=512, hidden_dim=512, num_hiddens=2, use_block_linear=True, act=nn.SiLU, **cfg
).to(device)
_shape_and_timing_test(rssm, **cfg, atol=1e-5)
del rssm
if device == "cuda":
    torch.cuda.empty_cache()

if device == "cuda":
    mamba = MambaSequenceModel(
        d_model=512, n_layers=2, d_state=16, d_conv=4, expand=2, headdim=64, **cfg
    ).to(device)
    _shape_and_timing_test(mamba, **cfg, atol=1e-3)
    del mamba
    torch.cuda.empty_cache()

transformer = TransformerSequenceModel(
    d_model=512,
    num_heads=8,
    n_layers=2,
    max_seq_len=256,
    expand=4,
    dropout_p=0.0,
    act=nn.SiLU,
    **cfg,
).to(device)
_shape_and_timing_test(transformer, **cfg, atol=1e-3)
del transformer
if device == "cuda":
    torch.cuda.empty_cache()

# === Production benchmark across configs ===
print()
print("=" * 70)
print("Production benchmark across configurations")
print("=" * 70)

configs = [
    (
        "Drama published (d_model=512, n=2)",
        lambda latent_size, action_dim: MambaSequenceModel(
            d_model=512,
            n_layers=2,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=64,
            latent_size=latent_size,
            action_dim=action_dim,
        ),
    ),
    (
        "Drama deep (d_model=512, n=8)",
        lambda latent_size, action_dim: MambaSequenceModel(
            d_model=512,
            n_layers=8,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=64,
            latent_size=latent_size,
            action_dim=action_dim,
        ),
    ),
    (
        "STORM-sized Transformer (d_model=256, n=2)",
        lambda latent_size, action_dim: TransformerSequenceModel(
            d_model=256,
            num_heads=8,
            n_layers=2,
            max_seq_len=256,
            expand=4,
            latent_size=latent_size,
            action_dim=action_dim,
        ),
    ),
    (
        "Mid Transformer (d_model=512, n=4)",
        lambda latent_size, action_dim: TransformerSequenceModel(
            d_model=512,
            num_heads=8,
            n_layers=4,
            max_seq_len=256,
            expand=4,
            latent_size=latent_size,
            action_dim=action_dim,
        ),
    ),
    (
        "DreamerV3 S (d_model=512, ~18M)",
        lambda latent_size, action_dim: RSSM(
            d_model=512,
            hidden_dim=512,
            num_hiddens=2,
            use_block_linear=True,
            act=nn.SiLU,
            latent_size=latent_size,
            action_dim=action_dim,
        ),
    ),
    (
        "DreamerV3 XL (d_model=4096, ~200M)",
        lambda latent_size, action_dim: RSSM(
            d_model=4096,
            hidden_dim=1024,
            num_hiddens=5,
            use_block_linear=True,
            act=nn.SiLU,
            latent_size=latent_size,
            action_dim=action_dim,
        ),
    ),
]

for name, factory in configs:
    benchmark_config(name, factory, latent_size=cfg["latent_size"], action_dim=cfg["action_dim"])

# === SDPA vs manual attention scaling sweep ===
print()
print("=" * 70)
print("Flash/SDPA vs Manual attention scaling sweep")
print("=" * 70)
print("Goal: identify where Flash (SDPA) becomes worth enabling.")
print("Step mode always uses manual; this sweep tests parallel forward only.")
print()


def _sdpa_sweep(latent_size, action_dim, d_model_values, T_values, B=16, n_iters=20):
    """Compare SDPA-on vs SDPA-off for parallel forward at varying (d_model, T)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Skipping (requires CUDA)")
        return

    print(f"  {'d_model':>8} {'T':>6} {'manual (ms)':>14} {'sdpa (ms)':>14} {'speedup':>10}")

    for d_model in d_model_values:
        for T in T_values:
            results = {}
            for use_sdpa in (False, True):
                try:
                    model = TransformerSequenceModel(
                        d_model=d_model,
                        num_heads=8,
                        n_layers=2,
                        max_seq_len=max(T + 16, 256),
                        expand=4,
                        latent_size=latent_size,
                        action_dim=action_dim,
                        use_sdpa=use_sdpa,
                    ).to(device)

                    latents = torch.randn(B, T, latent_size, device=device)
                    actions = torch.randn(B, T, action_dim, device=device)

                    # Warmup
                    for _ in range(5):
                        _ = model(latents, actions)
                    torch.cuda.synchronize()

                    # Time
                    start = time.perf_counter()
                    for _ in range(n_iters):
                        _ = model(latents, actions)
                    torch.cuda.synchronize()
                    t = (time.perf_counter() - start) / n_iters * 1000
                    results[use_sdpa] = t

                    del model, latents, actions
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    results[use_sdpa] = None
                    torch.cuda.empty_cache()

            manual_t = results[False]
            sdpa_t = results[True]
            if manual_t is None or sdpa_t is None:
                manual_s = f"{manual_t:>14.2f}" if manual_t else "          OOM"
                sdpa_s = f"{sdpa_t:>14.2f}" if sdpa_t else "          OOM"
                ratio_s = "          --"
            else:
                manual_s = f"{manual_t:>14.2f}"
                sdpa_s = f"{sdpa_t:>14.2f}"
                ratio = manual_t / sdpa_t
                marker = " *" if ratio > 1.1 else ""  # * if SDPA is meaningfully faster
                ratio_s = f"{ratio:>9.2f}x{marker}"

            print(f"  {d_model:>8} {T:>6} {manual_s} {sdpa_s} {ratio_s}")


# Vary d_model and T independently
_sdpa_sweep(
    latent_size=cfg["latent_size"],
    action_dim=cfg["action_dim"],
    d_model_values=(256, 512, 1024, 2048),
    T_values=(64, 128, 256, 512, 1024),
)
