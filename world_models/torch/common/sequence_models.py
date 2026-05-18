from abc import ABC, abstractmethod

import torch
from mamba_ssm.utils.generation import InferenceParams
from torch import nn

from world_models.torch.common.models import DreamerGRU, DreamerMLP, MambaBlock


class SequenceModel(nn.Module, ABC):
    """
    General sequence model, forward does an entire sequence at once,
    step does a single step.
    The state is meant to handle things like hidden state for recurrent, kv cache, etc.
    """

    @abstractmethod
    def forward(self, latents, actions, state=None, dones=None): ...

    @abstractmethod
    def step(self, latent, action, state, done=None): ...

    @property
    @abstractmethod
    def output_dim(self): ...


class RSSM(SequenceModel):
    # Also called the sequence model
    def __init__(
        self,
        latent_size,
        action_dim,
        d_model,
        hidden_dim=512,
        num_hiddens=2,
        act=nn.SiLU,
        use_block_linear=False,
    ):
        super().__init__()
        in_proj = nn.Linear(d_model + latent_size + action_dim, hidden_dim * 3, bias=False)
        norm = nn.RMSNorm(hidden_dim * 3, eps=1e-6)
        act_fn = act()
        mlp = DreamerMLP(
            hidden_dim * 3,
            d_model,
            hidden_dim,
            num_hiddens=num_hiddens,
        )
        self.proj = nn.Sequential(in_proj, norm, act_fn, mlp)
        self.gru = DreamerGRU(d_model, use_block_linear)
        self.initial_hidden = nn.Parameter(torch.zeros(d_model))
        self.d_model = d_model

    @property
    def output_dim(self):
        return self.d_model

    def initial_state(self, batch_size, device):
        return torch.tanh(self.initial_hidden).expand(batch_size, -1).to(device)

    def initial_state_from_reference(self, reference_tensor):
        batch_size, device = reference_tensor.shape[0], reference_tensor.device

        return self.initial_state(batch_size, device)

    def forward(self, latents, actions, hidden=None, dones=None):
        B, T, _ = latents.shape

        if hidden is None:
            hidden = self.initial_state_from_reference(latents)
        if dones is not None:
            dones = dones.unsqueeze(-1).float()

        hiddens = []
        for i in range(T):
            hidden, _ = self.step(
                latents[:, i], actions[:, i], hidden, dones[:, i] if dones is not None else None
            )
            hiddens.append(hidden)
        return torch.stack(hiddens, dim=1), hidden

    def step(self, latent, action, hidden, done=None):
        # the actual model output itself is the same as the state
        x = torch.cat([latent, action, hidden], -1)
        x = self.proj(x)
        h_new = self.gru(x, hidden)
        if done is not None:
            h_new = h_new * (1 - done) + done * self.initial_state_from_reference(latent)
        return h_new, h_new


class MambaSequenceModel(SequenceModel):
    def __init__(
        self,
        latent_size,
        action_dim,
        d_model,
        act=nn.SiLU,
        n_layers=2,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
    ):
        super().__init__()
        in_proj = nn.Linear(latent_size + action_dim, d_model, bias=False)
        act_fn = act()
        self.proj = nn.Sequential(in_proj, act_fn)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, i, d_state, d_conv, expand, headdim) for i in range(n_layers)]
        )
        self.norm_f = nn.RMSNorm(d_model, eps=1e-6)
        self.d_model = d_model

    @property
    def output_dim(self):
        return self.d_model

    def initial_state(self, batch_size, device, max_seq_len=1024):
        inference_params = InferenceParams(
            max_seqlen=max_seq_len,
            max_batch_size=batch_size,
        )

        inference_params.key_value_memory_dict = {
            i: block.allocate_inference_cache(batch_size, max_seq_len)
            for i, block in enumerate(self.blocks)
        }
        return inference_params

    def forward(self, latents, actions, state=None, dones=None):
        x = torch.cat([latents, actions], -1)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, inference_params=state)
        x = self.norm_f(x)
        return x, state

    def step(self, latent, action, state, done=None):
        latent_t = latent.unsqueeze(1)
        action_t = action.unsqueeze(1)
        output, state = self.forward(latent_t, action_t, state)
        state.seqlen_offset += 1
        return output.squeeze(1), state


class TransformerSequenceModel(SequenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError


if __name__ == "__main__":
    import time

    def _shape_and_timing_test(model, latent_size, action_dim, B=16, T=64, n_iters=100, atol=1e-5):
        device = next(model.parameters()).device
        is_cuda = device.type == "cuda"
        output_dim = model.output_dim

        latents = torch.randn(B, T, latent_size, device=device)
        actions = torch.randn(B, T, action_dim, device=device)

        # --- Shape checks ---
        outputs, final_state = model(latents, actions)
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
        forward_time = (time.perf_counter() - start) / n_iters

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
        step_time = (time.perf_counter() - start) / n_iters

        print(
            f"  forward (B={B}, T={T}): {forward_time * 1000:.2f} ms "
            f"({forward_time / T * 1000:.3f} ms/step)"
        )
        print(
            f"  step    (B={B}, T={T}): {step_time * 1000:.2f} ms "
            f"({step_time / T * 1000:.3f} ms/step)"
        )

        # Add to your test
        s = model.initial_state(B, device)
        torch.cuda.synchronize()

        # First step
        start = time.perf_counter()
        _, s = model.step(latents[:, 0], actions[:, 0], s)
        torch.cuda.synchronize()
        first_step_time = time.perf_counter() - start

        # Subsequent steps
        n = 50
        start = time.perf_counter()
        for t in range(1, n + 1):
            _, s = model.step(latents[:, t % T], actions[:, t % T], s)
        torch.cuda.synchronize()
        subsequent_step_time = (time.perf_counter() - start) / n

        print(f"  first step: {first_step_time * 1000:.2f} ms")
        print(f"  subsequent step: {subsequent_step_time * 1000:.2f} ms")

    cfg = dict(
        latent_size=32 * 32,
        action_dim=18,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # RSSM
    rssm = RSSM(
        d_model=512,
        hidden_dim=512,
        num_hiddens=2,
        use_block_linear=False,
        act=nn.SiLU,
        **cfg,
    ).to(device)
    _shape_and_timing_test(rssm, **cfg, atol=1e-5)

    # Mamba
    if device == "cuda":
        mamba = MambaSequenceModel(
            d_model=512,
            n_layers=2,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64,
            **cfg,
        ).to(device)
        _shape_and_timing_test(mamba, **cfg, atol=1e-3)
    else:
        print("Skipping Mamba test (requires CUDA)")

    def _scaling_sweep(model_factory, cfg, B=16, T_values=(16, 64, 256, 1024), n_iters=50):
        """Measure forward and step time as T grows. Verifies parallel scan benefit."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n{model_factory.__name__} scaling sweep (B={B}, varying T):")
        print(f"  {'T':>6} {'forward (ms)':>14} {'step (ms)':>14} {'fwd/step':>10}")

        for T in T_values:
            model = model_factory(**cfg).to(device)
            is_cuda = device == "cuda"

            latents = torch.randn(B, T, cfg["latent_size"], device=device)
            actions = torch.randn(B, T, cfg["action_dim"], device=device)

            # Warmup
            for _ in range(5):
                _ = model(latents, actions)
            if is_cuda:
                torch.cuda.synchronize()

            # Time forward
            start = time.perf_counter()
            for _ in range(n_iters):
                _ = model(latents, actions)
            if is_cuda:
                torch.cuda.synchronize()
            forward_time = (time.perf_counter() - start) / n_iters

            # Time step
            max_seq = max(T_values) + 16  # ensure we have headroom
            s = (
                model.initial_state(B, device, max_seq_len=max_seq)
                if "Mamba" in model_factory.__name__
                else model.initial_state(B, device)
            )
            for _ in range(5):
                _, s = model.step(latents[:, 0], actions[:, 0], s)
            if is_cuda:
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(n_iters):
                s = (
                    model.initial_state(B, device, max_seq_len=max_seq)
                    if "Mamba" in model_factory.__name__
                    else model.initial_state(B, device)
                )
                for t in range(T):
                    _, s = model.step(latents[:, t], actions[:, t], s)
            if is_cuda:
                torch.cuda.synchronize()
            step_time = (time.perf_counter() - start) / n_iters

            ratio = step_time / forward_time
            print(
                f"  {T:>6} {forward_time * 1000:>14.2f} {step_time * 1000:>14.2f} {ratio:>10.2f}x"
            )

            # Cleanup to free GPU memory
            del model, latents, actions
            if is_cuda:
                torch.cuda.empty_cache()

    # Then in __main__:
    def make_rssm(**cfg):
        return RSSM(
            d_model=512, hidden_dim=512, num_hiddens=2, use_block_linear=False, act=nn.SiLU, **cfg
        )

    make_rssm.__name__ = "RSSM"

    def make_mamba(**cfg):
        return MambaSequenceModel(
            d_model=512, n_layers=2, d_state=128, d_conv=4, expand=2, headdim=64, **cfg
        )

    make_mamba.__name__ = "MambaSequenceModel"

    # # After your existing tests:
    # _scaling_sweep(make_rssm, cfg)
    # if device == "cuda":
    #     _scaling_sweep(make_mamba, cfg)
    def _production_benchmark(model, latent_size, action_dim, name):
        """Benchmark at the actual shapes used during training."""
        device = next(model.parameters()).device
        print(f"\n=== {name} production benchmark ===")

        workloads = [
            # (B, T, mode, description)
            (16, 128, "forward+backward", "WM training step (Drama config)"),
            (1024, 16, "step_loop", "Imagination (Drama config)"),
            (1, 16, "forward", "Online inference (Drama, single env)"),
            (16, 16, "forward", "Online inference (vectorized 16 envs)"),
            (1, 1, "step", "Per-step inference (B=1)"),
        ]

        for B, T, mode, desc in workloads:
            latents = torch.randn(B, T, latent_size, device=device)
            actions = torch.randn(B, T, action_dim, device=device)

            if mode == "forward":
                # Warmup
                for _ in range(5):
                    _ = model(latents, actions)
                torch.cuda.synchronize()
                # Time
                start = time.perf_counter()
                for _ in range(20):
                    _ = model(latents, actions)
                torch.cuda.synchronize()
                t = (time.perf_counter() - start) / 20 * 1000
                print(f"  B={B:>4} T={T:>3} {mode:<18} {t:>8.2f} ms  [{desc}]")

            elif mode == "forward+backward":
                for _ in range(5):
                    out, _ = model(latents, actions)
                    out.sum().backward()
                    model.zero_grad()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    out, _ = model(latents, actions)
                    out.sum().backward()
                    model.zero_grad()
                torch.cuda.synchronize()
                t = (time.perf_counter() - start) / 20 * 1000
                print(f"  B={B:>4} T={T:>3} {mode:<18} {t:>8.2f} ms  [{desc}]")

            elif mode == "step_loop":
                # Test full imagination rollout
                for _ in range(3):
                    s = model.initial_state(B, device)
                    for t_idx in range(T):
                        _, s = model.step(latents[:, t_idx], actions[:, t_idx], s)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    s = model.initial_state(B, device)
                    for t_idx in range(T):
                        _, s = model.step(latents[:, t_idx], actions[:, t_idx], s)
                torch.cuda.synchronize()
                t = (time.perf_counter() - start) / 10 * 1000
                print(
                    f"  B={B:>4} T={T:>3} {mode:<18} {t:>8.2f} ms  ({t / T:.2f} ms/step) [{desc}]"
                )

            elif mode == "step":
                s = model.initial_state(B, device)
                for _ in range(10):
                    _, s = model.step(latents[:, 0], actions[:, 0], s)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(50):
                    _, s = model.step(latents[:, 0], actions[:, 0], s)
                torch.cuda.synchronize()
                t = (time.perf_counter() - start) / 50 * 1000
                print(f"  B={B:>4} T={T:>3} {mode:<18} {t:>8.2f} ms  [{desc}]")

            del latents, actions
            torch.cuda.empty_cache()

    _production_benchmark(rssm, 1024, 18, "rssm")
    _production_benchmark(mamba, 1024, 18, "mamba")
