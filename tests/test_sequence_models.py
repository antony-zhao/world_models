"""Shape, gradient, and timing tests for sequence models.

Run as:  python -m world_models.torch.common.test_sequence_models
or directly: python world_models/torch/common/test_sequence_models.py

Covers:
  - rollout return shapes (latents, seq_outputs, post/prior logits)
  - imagine_rollout return shapes (latents, actor_seqs, head_seqs, actions)
  - backward through rollout
  - step-vs-parallel-forward consistency (Mamba/Transformer): a sequence run via
    repeated step() should produce the same features as parallel_forward on the
    same (latents, actions). This is the silent-bug check for the cache/offset.
  - timing with warmup + CUDA events + averaging
"""

from __future__ import annotations

import statistics
import time

import torch

from world_models.torch.common.heads import CategoricalHead
from world_models.torch.common.models import Posterior, Prior
from world_models.torch.common.sequence_models import (
    RSSM,
    MambaSequenceModel,
    TransformerSequenceModel,
)
from world_models.torch.common.utils import make_state

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- helpers


def build_heads(embed_dim, d_model, num_cat, num_codes, includes_h, device):
    """Posterior + Prior with categorical dist heads. Posterior conditions on the
    sequence state only for RSSM-style (includes_h=True)."""
    out_dim = num_cat * num_codes
    posterior = Posterior(
        embed_dim,
        out_dim,
        CategoricalHead(num_cat, num_codes),
        d_model=d_model,
        n_layers=1,
        includes_sequence_state=includes_h,
    ).to(device)
    prior = Prior(
        d_model,
        out_dim,
        CategoricalHead(num_cat, num_codes),
        n_layers=1,
    ).to(device)
    return posterior, prior


def benchmark(fn, n_warmup=5, n_iters=20, backward=False):
    """Time a zero-arg callable with CUDA-synced warmup + averaging.
    If backward=True, fn must return a scalar to backprop.
    Returns (mean_ms, std_ms, peak_mem_mb)."""
    is_cuda = DEVICE == "cuda"

    for _ in range(n_warmup):
        out = fn()
        if backward:
            out.backward()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_iters):
        if is_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            t0 = time.perf_counter()

        out = fn()
        if backward:
            out.backward()

        if is_cuda:
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            times.append((time.perf_counter() - t0) * 1000.0)

    peak_mem = (torch.cuda.max_memory_allocated() / 1e6) if is_cuda else 0.0
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, peak_mem


# ---------------------------------------------------------- shape + grad


def test_rollout(model, embed_dim, action_dim, num_cat, num_codes, includes_h, B=16, T=64):
    d_model = model.output_dim
    posterior, prior = build_heads(embed_dim, d_model, num_cat, num_codes, includes_h, DEVICE)

    embeddings = torch.randn(B, T, embed_dim, device=DEVICE)
    actions = torch.randn(B, T, action_dim, device=DEVICE)
    dones = torch.zeros(B, T, device=DEVICE)
    dones[1, T // 2] = 1.0  # exercise the done-reset path (RSSM)

    latents, seq_outputs, post_logits, prior_logits = model.rollout(
        embeddings, actions, posterior, prior, dones=dones
    )

    name = model.__class__.__name__
    print(f"\n=== {name} rollout (B={B}, T={T}) ===")
    print(f"  latents:      {tuple(latents.shape)}")
    print(f"  seq_outputs:  {tuple(seq_outputs.shape)}")
    print(f"  post_logits:  {tuple(post_logits.shape)}")
    print(f"  prior_logits: {tuple(prior_logits.shape)}")

    assert latents.shape[:2] == (B, T), f"{name}: latents {latents.shape}"
    assert seq_outputs.shape[:2] == (B, T), f"{name}: seq_outputs {seq_outputs.shape}"
    assert seq_outputs.shape[-1] == d_model, (
        f"{name}: seq_outputs width {seq_outputs.shape[-1]} != d_model={d_model}"
    )
    # latent shape: (B, T, num_cat, num_codes) — sampled from categorical
    assert latents.shape[-2:] == (num_cat, num_codes), (
        f"{name}: latents shape {latents.shape}, expected (..., {num_cat}, {num_codes})"
    )
    # KL logit lengths match each other; either T or T-1 depending on architecture
    assert post_logits.shape[1] == prior_logits.shape[1], (
        f"{name}: post/prior logit time dims differ"
    )
    assert post_logits.shape[1] in (T, T - 1), (
        f"{name}: KL logit time dim {post_logits.shape[1]} not in (T, T-1)"
    )
    # confirm make_state(latents, seq_outputs) gives the expected joint width
    joint = make_state(latents, seq_outputs)
    expected_joint_width = num_cat * num_codes + d_model
    assert joint.shape[-1] == expected_joint_width, (
        f"{name}: make_state width {joint.shape[-1]} != {expected_joint_width}"
    )

    # backward through the returned tensors
    (latents.sum() + post_logits.sum() + prior_logits.sum()).backward()
    print("  shapes OK, backward OK")


def test_imagine(model, action_dim, num_cat, num_codes, B=8, H=15):
    """Verify the 4-tuple return shapes and that all four tensors have time dim H+1."""
    d_model = model.output_dim
    _, prior = build_heads(8, d_model, num_cat, num_codes, includes_h=False, device=DEVICE)

    def actor(latent, seq_state):
        return torch.randn(B, action_dim, device=DEVICE)

    model_state = model.initial_state(B, DEVICE)
    init_latent = torch.randn(B, num_cat, num_codes, device=DEVICE)
    init_seq = torch.randn(B, d_model, device=DEVICE)

    with torch.no_grad():
        latents, actor_seqs, head_seqs, actions = model.imagine_rollout(
            init_latent, init_seq, model_state, H, actor, prior
        )

    name = model.__class__.__name__
    print(
        f"  imagine: latents {tuple(latents.shape)}, "
        f"actor_seqs {tuple(actor_seqs.shape)}, "
        f"head_seqs {tuple(head_seqs.shape)}, "
        f"actions {tuple(actions.shape)}"
    )
    assert latents.shape[1] == H + 1, f"{name}: latents time dim {latents.shape[1]} != {H + 1}"
    assert actor_seqs.shape[1] == H + 1, f"{name}: actor_seqs {actor_seqs.shape[1]} != {H + 1}"
    assert head_seqs.shape[1] == H + 1, f"{name}: head_seqs {head_seqs.shape[1]} != {H + 1}"
    assert actions.shape[1] == H + 1, f"{name}: actions {actions.shape[1]} != {H + 1}"
    print("  imagine shapes OK")


def test_step_vs_parallel(model, action_dim, num_cat, num_codes, B=4, T=8):
    """For Mamba/Transformer: verify step() repeated T times produces the same
    features as parallel_forward on the same (latents, actions). Catches cache /
    seqlen_offset bugs (e.g. parallel_forward not advancing the offset)."""
    if isinstance(model, RSSM):
        print(f"  skipping step-vs-parallel for RSSM (no parallel_forward)")
        return

    d_model = model.output_dim
    name = model.__class__.__name__

    latents = torch.randn(B, T, num_cat, num_codes, device=DEVICE)
    actions = torch.randn(B, T, action_dim, device=DEVICE)

    with torch.no_grad():
        # parallel pass
        parallel_out, _ = model.parallel_forward(latents, actions)

        # incremental pass using step()
        state = model.initial_state(B, DEVICE)
        step_outs = []
        for t in range(T):
            d_t, state = model.step(latents[:, t], actions[:, t], state)
            step_outs.append(d_t)
        step_out = torch.stack(step_outs, dim=1)

    diff = (parallel_out - step_out).abs().max().item()
    print(f"  step-vs-parallel max abs diff: {diff:.6e}")
    # Numerical tolerance: attention/SSM can differ slightly due to floating point.
    # Anything under ~1e-3 in fp32 indicates the cache/offset is being managed correctly.
    assert diff < 1e-3, f"{name}: step and parallel_forward disagree (diff={diff})"
    print("  step-vs-parallel OK")


# ------------------------------------------------------------- timing


def time_model(
    name,
    model,
    embed_dim,
    action_dim,
    num_cat,
    num_codes,
    includes_h,
    B_train=16,
    T_train=64,
    B_imag=1024,
    H=15,
):
    d_model = model.output_dim
    posterior, prior = build_heads(embed_dim, d_model, num_cat, num_codes, includes_h, DEVICE)

    emb = torch.randn(B_train, T_train, embed_dim, device=DEVICE)
    act = torch.randn(B_train, T_train, action_dim, device=DEVICE)
    dones = torch.zeros(B_train, T_train, device=DEVICE)

    def rollout_fn(posterior=posterior, prior=prior):
        model.zero_grad(set_to_none=True)
        posterior.zero_grad(set_to_none=True)
        prior.zero_grad(set_to_none=True)
        latents, seq_outputs, pl, prl = model.rollout(emb, act, posterior, prior, dones=dones)
        return latents.sum() + pl.sum() + prl.sum()

    r_mean, r_std, r_mem = benchmark(rollout_fn, backward=True)

    def actor(latent, seq_state):
        return torch.randn(B_imag, action_dim, device=DEVICE)

    init_latent = torch.randn(B_imag, num_cat, num_codes, device=DEVICE)
    init_seq = torch.randn(B_imag, d_model, device=DEVICE)

    def imagine_fn(prior=prior):
        with torch.no_grad():
            # fresh model_state each iter — cache-based models mutate in place
            model_state = model.initial_state(B_imag, DEVICE)
            out = model.imagine_rollout(init_latent, init_seq, model_state, H, actor, prior)
            return out[0].sum()

    i_mean, i_std, i_mem = benchmark(imagine_fn, backward=False)

    peak = max(r_mem, i_mem)
    print(f"{name:<24} {r_mean:>8.2f}±{r_std:<5.2f} {i_mean:>8.2f}±{i_std:<5.2f} {peak:>9.0f}")

    del posterior, prior


# ---------------------------------------------------------------- run


def main():
    embed_dim, action_dim = 256, 18
    num_cat, num_codes = 32, 32
    latent_size = num_cat * num_codes

    # (name, ctor, includes_h, needs_cuda)
    factories = [
        (
            "RSSM-d512",
            lambda: RSSM(latent_size, action_dim, d_model=512, hidden_dim=512),
            True,
            False,
        ),
        (
            "RSSM-d8192",
            lambda: RSSM(latent_size, action_dim, d_model=8192, hidden_dim=8192),
            True,
            False,
        ),
        (
            "Mamba-d512-n2",
            lambda: MambaSequenceModel(
                latent_size, action_dim, d_model=512, n_layers=2, d_state=16
            ),
            False,
            True,
        ),
        (
            "Transformer-d512-n2",
            lambda: TransformerSequenceModel(
                latent_size, action_dim, d_model=512, num_heads=8, n_layers=2, max_seq_len=128
            ),
            False,
            False,
        ),
    ]

    print("=" * 72)
    print("SHAPE + GRAD + CONSISTENCY TESTS")
    print("=" * 72)
    for name, ctor, includes_h, needs_cuda in factories:
        if needs_cuda and DEVICE != "cuda":
            print(f"\n(skipping {name} — requires CUDA)")
            continue
        model = ctor().to(DEVICE)
        test_rollout(model, embed_dim, action_dim, num_cat, num_codes, includes_h)
        test_imagine(model, action_dim, num_cat, num_codes)
        test_step_vs_parallel(model, action_dim, num_cat, num_codes)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("TIMING (warmup=5, iters=20)  —  rollout fwd+bwd / imagine, in ms")
    print("=" * 72)
    print(f"{'config':<24} {'rollout (ms)':>14} {'imagine (ms)':>14} {'peak MB':>10}")
    print("-" * 72)
    for name, ctor, includes_h, needs_cuda in factories:
        if needs_cuda and DEVICE != "cuda":
            continue
        model = ctor().to(DEVICE)
        time_model(name, model, embed_dim, action_dim, num_cat, num_codes, includes_h)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
