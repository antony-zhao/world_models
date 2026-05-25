from abc import ABC, abstractmethod

import torch
from mamba_ssm.utils.generation import InferenceParams
from torch import nn

from world_models.torch.common.models import (
    MLP,
    DreamerGRU,
    MambaBlock,
    TransformerBlock,
)
from world_models.torch.common.utils import make_state


class SequenceModel(nn.Module, ABC):
    """
    General sequence model, forward does an entire sequence at once,
    step does a single step.
    The model_state is meant to handle things like hidden state for recurrent, kv cache, etc.
    """

    @abstractmethod
    def step(self, latent, action, model_state): ...

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use rollout() for the full training pass or step() for inference."
        )

    @property
    @abstractmethod
    def output_dim(self): ...

    def rollout(self, embeddings, actions, posterior, prior, dones=None):
        # returns (latents, states, posterior_dists, prior_dists),
        # default is for mamba/transformer since those are the same,
        # as they can run through all the data in parallel
        # RSSM has an override
        post_dist = posterior(embeddings, None)
        latents = post_dist.rsample()
        seq_state, _ = self.parallel_forward(latents, actions)
        prior_dist = prior(seq_state)
        return (
            latents,
            make_state(latents, seq_state),
            post_dist.logits[:, 1:],
            prior_dist.logits[:, :-1],
        )

    def imagine_step(self, latent, action, model_state, prior):
        # returns next_latent, next_state, new_model_state
        seq_state, model_state = self.step(latent, action, model_state)
        prior_dist = prior(seq_state)
        next_latent = prior_dist.sample()
        next_state = make_state(next_latent, seq_state)
        return next_latent, next_state, model_state

    def imagine_rollout(self, initial_latent, initial_state, model_state, horizon, actor, prior):
        # returns imagined_states, imagined_latents, actions
        latent, state = initial_latent, initial_state
        latents, states, actions = [], [], []

        for t in range(horizon + 1):
            action = actor(latent, state)
            latents.append(latent)
            states.append(state)
            actions.append(action)
            if t < horizon:
                latent, state, model_state = self.imagine_step(latent, action, model_state, prior)

        latents = torch.stack(latents, 1)
        states = torch.stack(states, 1)
        actions = torch.stack(actions, 1)
        return latents, states, actions


class RSSM(SequenceModel):
    def __init__(
        self,
        latent_size,
        action_dim,
        d_model,
        hidden_dim=512,
        n_layers=2,
        act=nn.SiLU,
        use_block_linear=True,
    ):
        super().__init__()
        in_proj = nn.Linear(d_model + latent_size + action_dim, hidden_dim * 3, bias=False)
        norm = nn.RMSNorm(hidden_dim * 3, eps=1e-6)
        act_fn = act()
        mlp = MLP(
            hidden_dim * 3,
            d_model,
            hidden_dim,
            n_layers=n_layers,
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

    def step(self, latent, action, hidden):
        # the actual model output itself is the same as the state
        state = make_state(latent, hidden)
        x = torch.cat([state, action], -1)
        x = self.proj(x)
        h_new = self.gru(x, hidden)
        return h_new, h_new

    def rollout(self, embeddings, actions, posterior, prior, dones=None):
        # returns (latents, states, posterior_dists, prior_dists),
        B, T, _ = embeddings.shape
        latents, states = [], []
        post_logits, prior_logits = [], []
        hidden = initial_hidden = self.initial_state_from_reference(embeddings)

        for i in range(T):
            post_dist = posterior(embeddings[:, i], hidden)
            latent = post_dist.rsample()

            if i > 0:
                prior_dist = prior(hidden)
                post_logits.append(post_dist.logits)
                prior_logits.append(prior_dist.logits)

            state = make_state(latent, hidden)
            latents.append(latent)
            states.append(state)

            _, hidden = self.step(latent, actions[:, i], hidden)

            done = dones[:, i].unsqueeze(-1)
            hidden = (1 - done) * hidden + done * initial_hidden

        return (
            torch.stack(latents, 1),
            torch.stack(states, 1),
            torch.stack(post_logits, 1),
            torch.stack(prior_logits, 1),
        )


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

    def parallel_forward(self, latents, actions, state=None, dones=None):
        if latents.ndim > actions.ndim:
            latents = latents.flatten(-2)
        x = torch.cat([latents, actions], -1)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, inference_params=state)
        x = self.norm_f(x)
        return x, state

    def step(self, latent, action, state):
        latent_t = latent.unsqueeze(1)
        action_t = action.unsqueeze(1)
        output, state = self.parallel_forward(latent_t, action_t, state)
        state.seqlen_offset += 1
        return output.squeeze(1), state


class TransformerSequenceModel(SequenceModel):
    def __init__(
        self,
        latent_size,
        action_dim,
        d_model,
        num_heads=8,
        n_layers=2,
        max_seq_len=1024,
        expand=4,
        dropout_p=0.0,
        act=nn.SiLU,
        use_sdpa=True,
    ):
        super().__init__()
        in_proj = nn.Linear(latent_size + action_dim, d_model, bias=False)
        act_fn = act()
        self.proj = nn.Sequential(in_proj, act_fn)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, max_seq_len, expand, dropout_p, act, use_sdpa=use_sdpa
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

    @property
    def output_dim(self):
        return self.d_model

    def initial_state(self, batch_size, device):
        caches = [
            self.blocks[i].allocate_kv_cache(batch_size, device) for i in range(self.n_layers)
        ]
        return (caches, 0)  # (per-block caches, seq_offset)

    def parallel_forward(self, latents, actions, state=None, dones=None):
        if latents.ndim > actions.ndim:
            latents = latents.flatten(-2)
        x = torch.cat([latents, actions], -1)
        x = self.proj(x)
        B, T, _ = x.shape

        if state is not None:
            block_caches, seq_offset = state
        else:
            block_caches, seq_offset = [None] * self.n_layers, 0

        new_caches = []
        for block, cache in zip(self.blocks, block_caches):
            x, new_cache = block(x, cache=cache, seq_offset=seq_offset)
            new_caches.append(new_cache)

        x = self.norm_f(x)

        if state is not None:
            new_state = (new_caches, seq_offset + T)
        else:
            new_state = None

        return x, new_state

    def step(self, latent, action, state):
        latent_t = latent.unsqueeze(1)
        action_t = action.unsqueeze(1)
        output, state = self.parallel_forward(latent_t, action_t, state)
        return output.squeeze(1), state


if __name__ == "__main__":
    import statistics
    import time

    from world_models.torch.common.heads import CategoricalHead
    from world_models.torch.common.models import Posterior, Prior
    from world_models.torch.common.utils import make_state

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------- helpers

    def build_heads(embed_dim, d_model, num_cat, num_codes, includes_h, device):
        """Posterior + Prior with categorical dist heads. Posterior conditions
        on the sequence state only for RSSM-style models (includes_h=True)."""
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
        Returns (mean_ms, std_ms, peak_mem_mb).
        """
        is_cuda = device == "cuda"

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
        posterior, prior = build_heads(embed_dim, d_model, num_cat, num_codes, includes_h, device)

        embeddings = torch.randn(B, T, embed_dim, device=device)
        actions = torch.randn(B, T, action_dim, device=device)
        dones = torch.zeros(B, T, device=device)
        dones[1, T // 2] = 1.0  # exercise the done-reset path

        latents, states, post_logits, prior_logits = model.rollout(
            embeddings, actions, posterior, prior, dones=dones
        )

        name = model.__class__.__name__
        print(f"\n=== {name} rollout (B={B}, T={T}) ===")
        print(f"  latents:      {tuple(latents.shape)}")
        print(f"  states:       {tuple(states.shape)}")
        print(f"  post_logits:  {tuple(post_logits.shape)}")
        print(f"  prior_logits: {tuple(prior_logits.shape)}")

        assert latents.shape[:2] == (B, T), f"{name}: latents {latents.shape}"
        assert states.shape[:2] == (B, T), f"{name}: states {states.shape}"
        expected_width = num_cat * num_codes + d_model
        assert states.shape[-1] == expected_width, (
            f"{name}: state width {states.shape[-1]} != {expected_width}"
        )
        assert post_logits.shape[0] == B and prior_logits.shape[0] == B
        assert post_logits.shape[1] == prior_logits.shape[1], (
            f"{name}: post/prior logit lengths differ"
        )
        assert post_logits.shape[1] in (T, T - 1), (
            f"{name}: KL logit time dim {post_logits.shape[1]} not in (T, T-1)"
        )
        (latents.sum() + post_logits.sum() + prior_logits.sum()).backward()
        print("  shapes OK, backward OK")

    def test_imagine(model, action_dim, num_cat, num_codes, B=1024, H=15):
        d_model = model.output_dim
        _, prior = build_heads(8, d_model, num_cat, num_codes, includes_h=False, device=device)

        def actor(latent, state):
            return torch.randn(B, action_dim, device=device)

        model_state = model.initial_state(B, device)
        init_latent = torch.randn(B, num_cat, num_codes, device=device)
        init_seq = torch.randn(B, d_model, device=device)
        init_state = make_state(init_latent, init_seq)

        latents, states, actions = model.imagine_rollout(
            init_latent, init_state, model_state, H, actor, prior
        )
        name = model.__class__.__name__
        print(
            f"  imagine: latents {tuple(latents.shape)}, "
            f"states {tuple(states.shape)}, actions {tuple(actions.shape)}"
        )
        assert latents.shape[1] == H + 1, f"{name}: imagine horizon {latents.shape[1]} != {H + 1}"
        assert actions.shape[1] == H + 1
        print("  imagine shapes OK")

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
        posterior, prior = build_heads(embed_dim, d_model, num_cat, num_codes, includes_h, device)

        emb = torch.randn(B_train, T_train, embed_dim, device=device)
        act = torch.randn(B_train, T_train, action_dim, device=device)
        dones = torch.zeros(B_train, T_train, device=device)

        def rollout_fn():
            model.zero_grad(set_to_none=True)
            posterior.zero_grad(set_to_none=True)
            prior.zero_grad(set_to_none=True)
            latents, states, pl, prl = model.rollout(emb, act, posterior, prior, dones=dones)
            return latents.sum() + pl.sum() + prl.sum()

        r_mean, r_std, r_mem = benchmark(rollout_fn, backward=True)

        def actor(latent, state):
            return torch.randn(B_imag, action_dim, device=device)

        init_latent = torch.randn(B_imag, num_cat, num_codes, device=device)
        init_seq = torch.randn(B_imag, d_model, device=device)
        init_state = make_state(init_latent, init_seq)

        def imagine_fn():
            with torch.no_grad():
                # fresh state each iter — cache-based models (Mamba/Transformer)
                # mutate state in place, so reusing it across iters is wrong.
                model_state = model.initial_state(B_imag, device)
                out = model.imagine_rollout(init_latent, init_state, model_state, H, actor, prior)
                return out[0].sum()

        i_mean, i_std, i_mem = benchmark(imagine_fn, backward=False)

        peak = max(r_mem, i_mem)
        print(f"{name:<24} {r_mean:>8.2f}±{r_std:<5.2f} {i_mean:>8.2f}±{i_std:<5.2f} {peak:>9.0f}")

        del posterior, prior

    # ---------------------------------------------------------------- run

    embed_dim, action_dim = 256, 18
    num_cat, num_codes = 32, 32
    latent_size = num_cat * num_codes

    # model factory list: (name, ctor, includes_h, needs_cuda)
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

    print("=" * 70)
    print("SHAPE + GRAD TESTS")
    print("=" * 70)
    for name, ctor, includes_h, needs_cuda in factories:
        if needs_cuda and device != "cuda":
            print(f"\n(skipping {name} — requires CUDA)")
            continue
        model = ctor().to(device)
        test_rollout(model, embed_dim, action_dim, num_cat, num_codes, includes_h)
        test_imagine(model, action_dim, num_cat, num_codes)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("TIMING (warmup=5, iters=20)  —  rollout fwd+bwd / imagine, in ms")
    print("=" * 70)
    print(f"{'config':<24} {'rollout (ms)':>14} {'imagine (ms)':>14} {'peak MB':>10}")
    print("-" * 70)
    for name, ctor, includes_h, needs_cuda in factories:
        if needs_cuda and device != "cuda":
            continue
        model = ctor().to(device)
        time_model(name, model, embed_dim, action_dim, num_cat, num_codes, includes_h)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.")
