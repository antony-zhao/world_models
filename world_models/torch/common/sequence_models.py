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
    import time

    from world_models.torch.common.heads import CategoricalHead
    from world_models.torch.common.models import Posterior, Prior
    from world_models.torch.common.utils import make_state

    def build_heads(embed_dim, d_model, num_cat, num_codes, includes_h, device):
        """Posterior + Prior with a categorical dist head. Posterior includes
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

    def test_rollout(
        model, embed_dim, action_dim, num_cat, num_codes, includes_h, B=4, T=8, device="cpu"
    ):
        d_model = model.output_dim
        posterior, prior = build_heads(embed_dim, d_model, num_cat, num_codes, includes_h, device)

        embeddings = torch.randn(B, T, embed_dim, device=device)
        actions = torch.randn(B, T, action_dim, device=device)
        dones = torch.zeros(B, T, device=device)
        dones[1, 4] = 1.0  # exercise the done-reset path (RSSM)

        latents, states, post_logits, prior_logits = model.rollout(
            embeddings, actions, posterior, prior, dones=dones
        )

        name = model.__class__.__name__
        print(f"\n=== {name} rollout (B={B}, T={T}) ===")
        print(f"  latents:      {tuple(latents.shape)}")
        print(f"  states:       {tuple(states.shape)}")
        print(f"  post_logits:  {tuple(post_logits.shape)}")
        print(f"  prior_logits: {tuple(prior_logits.shape)}")

        # latents/states must keep the full time dim
        assert latents.shape[0] == B and latents.shape[1] == T, (
            f"{name}: latents shape {latents.shape}, expected ({B}, {T}, ...)"
        )
        assert states.shape[0] == B and states.shape[1] == T, (
            f"{name}: states shape {states.shape}, expected ({B}, {T}, ...)"
        )
        # state width = flattened latent + d_model
        expected_width = num_cat * num_codes + d_model
        assert states.shape[-1] == expected_width, (
            f"{name}: state width {states.shape[-1]} != {expected_width}"
        )
        # KL logits: post and prior must align (same length), B preserved
        assert post_logits.shape[0] == B and prior_logits.shape[0] == B
        assert post_logits.shape[1] == prior_logits.shape[1], (
            f"{name}: post/prior logit time dims differ "
            f"({post_logits.shape[1]} vs {prior_logits.shape[1]})"
        )
        # logit time dim should be T (parallel, sliced) or T-1 (RSSM skip-first)
        assert post_logits.shape[1] in (T, T - 1), (
            f"{name}: KL logit time dim {post_logits.shape[1]} not in (T, T-1)"
        )
        # gradient check: rollout outputs should be differentiable
        loss = latents.sum() + post_logits.sum() + prior_logits.sum()
        loss.backward()
        print("  shapes OK, backward OK")

    def test_imagine(model, action_dim, num_cat, num_codes, B=4, H=5, device="cpu"):
        d_model = model.output_dim
        _, prior = build_heads(0, d_model, num_cat, num_codes, includes_h=False, device=device)

        # actor stub: (latent, state) -> action of shape (B, action_dim)
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
        assert latents.shape[1] == H + 1, (
            f"{name}: imagine latents time dim {latents.shape[1]} != H+1={H + 1}"
        )
        assert actions.shape[1] == H + 1
        print("  imagine shapes OK")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_dim, action_dim = 256, 18
    latent_size = 32 * 32  # flattened categorical, used as latent_size in constructors
    num_cat, num_codes = 32, 32

    print("=" * 60)
    print("Rollout + imagine shape/grad tests")
    print("=" * 60)

    # --- RSSM: posterior conditions on sequence state ---
    rssm = RSSM(
        latent_size=latent_size,
        action_dim=action_dim,
        d_model=512,
        hidden_dim=512,
        n_layers=2,
    ).to(device)
    test_rollout(rssm, embed_dim, action_dim, num_cat, num_codes, includes_h=True, device=device)
    test_imagine(rssm, action_dim, num_cat, num_codes, device=device)
    del rssm
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Mamba: posterior embedding-only (needs CUDA) ---
    if device == "cuda":
        mamba = MambaSequenceModel(
            latent_size=latent_size,
            action_dim=action_dim,
            d_model=512,
            n_layers=2,
            d_state=16,
        ).to(device)
        test_rollout(
            mamba, embed_dim, action_dim, num_cat, num_codes, includes_h=False, device=device
        )
        test_imagine(mamba, action_dim, num_cat, num_codes, device=device)
        del mamba
        torch.cuda.empty_cache()
    else:
        print("\n(skipping Mamba — requires CUDA)")

    # --- Transformer: posterior embedding-only ---
    transformer = TransformerSequenceModel(
        latent_size=latent_size,
        action_dim=action_dim,
        d_model=512,
        num_heads=8,
        n_layers=2,
        max_seq_len=64,
    ).to(device)
    test_rollout(
        transformer, embed_dim, action_dim, num_cat, num_codes, includes_h=False, device=device
    )
    test_imagine(transformer, action_dim, num_cat, num_codes, device=device)
    del transformer
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nAll tests passed.")
