"""Correctness and speed checks for TransformerSequenceModel's attention paths.

Correctness (eval mode, fp32), each one compares two code paths on the same weights:
  1. SDPA vs manual attention, parallel forward.
  2. Causality: changing inputs at t >= k must not change outputs at t < k.
  3. step() with the KV cache (manual attention, T=1) vs parallel forward (SDPA).
  4. Chunked parallel forward with the KV cache: the second chunk runs at seq_offset > 0,
     which is the only place the rectangular-mask SDPA branch is used.

Speed: forward + backward at world-model training shape for manual attention and each SDPA
backend (math / efficient / flash / cudnn), in fp32 and under bf16 autocast. FlashAttention
only runs in fp16/bf16, so it shows as unavailable in fp32.

Run from the project root:
    python tests/test_transformer_attention.py
"""

from __future__ import annotations

import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from world_models.torch.common.sequence_models import TransformerSequenceModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# storm.yaml sizes
LATENT, ACTION, D_MODEL, HEADS, LAYERS, MAX_LEN, EXPAND = 1024, 4, 512, 8, 2, 256, 4
B, T = 16, 64


def build(dropout_p=0.0):
    torch.manual_seed(0)
    model = TransformerSequenceModel(
        LATENT, ACTION, D_MODEL, num_heads=HEADS, n_layers=LAYERS, max_seq_len=MAX_LEN,
        expand=EXPAND, dropout_p=dropout_p, use_sdpa=True,
    ).to(DEVICE)
    return model


def set_sdpa(model, on):
    for block in model.blocks:
        block.use_sdpa = on


def inputs(b=B, t=T, seed=1):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return (
        torch.randn(b, t, LATENT, device=DEVICE, generator=g),
        torch.randn(b, t, ACTION, device=DEVICE, generator=g),
    )


# ------------------------------------------------------------------ correctness


@torch.no_grad()
def test_sdpa_vs_manual(model):
    lat, act = inputs()
    set_sdpa(model, True)
    out_sdpa, _ = model.parallel_forward(lat, act)
    set_sdpa(model, False)
    out_manual, _ = model.parallel_forward(lat, act)
    set_sdpa(model, True)
    diff = (out_sdpa - out_manual).abs().max().item()
    print(f"  1. sdpa vs manual (parallel): max abs diff {diff:.2e}")
    assert diff < 1e-4, diff


@torch.no_grad()
def test_causality(model, k=40):
    lat, act = inputs()
    lat2, act2 = lat.clone(), act.clone()
    lat2[:, k:] += torch.randn_like(lat2[:, k:])
    act2[:, k:] += torch.randn_like(act2[:, k:])
    for on in (True, False):
        set_sdpa(model, on)
        a, _ = model.parallel_forward(lat, act)
        b, _ = model.parallel_forward(lat2, act2)
        past = (a[:, :k] - b[:, :k]).abs().max().item()
        future = (a[:, k:] - b[:, k:]).abs().max().item()
        name = "sdpa" if on else "manual"
        print(f"  2. causality ({name}): change at t>={k} -> past diff {past:.2e}, future diff {future:.2e}")
        assert past < 1e-5 and future > 1e-3, (past, future)
    set_sdpa(model, True)


@torch.no_grad()
def test_step_vs_parallel(model, t=32):
    lat, act = inputs(t=t)
    parallel, _ = model.parallel_forward(lat, act)
    state = model.initial_state(B, DEVICE)
    steps = []
    for i in range(t):
        d, state = model.step(lat[:, i], act[:, i], state)
        steps.append(d)
    diff = (parallel - torch.stack(steps, 1)).abs().max().item()
    print(f"  3. step (KV cache, manual T=1) vs parallel (sdpa): max abs diff {diff:.2e}")
    assert diff < 1e-4, diff


@torch.no_grad()
def test_chunked_cache(model, split=24):
    lat, act = inputs()
    full, _ = model.parallel_forward(lat, act)
    for on in (True, False):
        set_sdpa(model, on)
        state = model.initial_state(B, DEVICE)
        first, state = model.parallel_forward(lat[:, :split], act[:, :split], state)
        second, state = model.parallel_forward(lat[:, split:], act[:, split:], state)
        diff = (full - torch.cat([first, second], 1)).abs().max().item()
        name = "sdpa rect-mask" if on else "manual"
        print(f"  4. chunked cache, 2nd chunk at seq_offset={split} ({name}): max abs diff {diff:.2e}")
        assert diff < 1e-4, diff
    set_sdpa(model, True)


# ------------------------------------------------------------------------ speed


def time_fwd_bwd(model, lat, act, dtype, n_warmup=5, n_iters=20):
    def run():
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            out, _ = model.parallel_forward(lat, act)
        out.float().sum().backward()

    for _ in range(n_warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(n_iters):
        run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / n_iters * 1e3
    return ms, torch.cuda.max_memory_allocated() / 2**20


def speed(t=T):
    if DEVICE != "cuda":
        print("  skipped (needs CUDA)")
        return
    model = build()
    model.train()
    lat, act = inputs(t=t)
    lat.requires_grad_(True)
    configs = [("manual", None)] + [
        (f"sdpa:{b.name.lower()}", b)
        for b in (SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION,
                  SDPBackend.CUDNN_ATTENTION)
    ] + [("sdpa:auto", "auto")]
    print(f"  B={B} T={t} d_model={D_MODEL} heads={HEADS} layers={LAYERS}, forward+backward")
    print(f"  {'path':22s} {'fp32 ms':>9s} {'fp32 MB':>9s} {'bf16 ms':>9s} {'bf16 MB':>9s}")
    for name, backend in configs:
        row = []
        for dtype in (torch.float32, torch.bfloat16):
            set_sdpa(model, backend is not None)
            try:
                if backend in (None, "auto"):
                    ms, mb = time_fwd_bwd(model, lat, act, dtype)
                else:
                    with sdpa_kernel(backend):
                        ms, mb = time_fwd_bwd(model, lat, act, dtype)
                row += [f"{ms:9.2f}", f"{mb:9.0f}"]
            except RuntimeError:  # backend unsupported for this dtype / device
                row += [f"{'n/a':>9s}", f"{'':>9s}"]
            model.zero_grad(set_to_none=True)
            lat.grad = None
        print(f"  {name:22s} " + " ".join(row))
    set_sdpa(model, True)


def main():
    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    model = build().eval()
    print("\ncorrectness (eval, fp32)")
    test_sdpa_vs_manual(model)
    test_causality(model)
    test_step_vs_parallel(model)
    test_chunked_cache(model)
    print("\nspeed")
    speed()
    print("\nall correctness checks passed")


if __name__ == "__main__":
    main()
