from abc import ABC, abstractmethod

import torch
from mamba_ssm.utils.generation import InferenceParams
from torch import Tensor, nn

from world_models.torch.common.models import MLP, DreamerGRU, MambaBlock, TransformerBlock
from world_models.torch.common.utils import make_state


class SequenceModel(nn.Module, ABC):
    """
    General sequence model, forward does an entire sequence at once,
    step does a single step.
    The model_state is meant to handle things like hidden state for recurrent, kv cache, etc.
    """

    @abstractmethod
    def initial_state(self, batch_size, device): ...

    @abstractmethod
    def step(self, latent, action, model_state): ...

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use rollout() for the full training pass or step() for inference."
        )

    @property
    @abstractmethod
    def output_dim(self): ...

    def rollout(
        self, embeddings, actions, posterior, prior, dones=None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # returns (latents, states, posterior_dists, prior_dists),
        # default is for mamba/transformer since those are the same,
        # as they can run through all the data in parallel
        # RSSM has an override
        post_dist = posterior(embeddings, None)
        latents = post_dist.rsample()  # z_{0:T} = p(x_{0:T})
        seq_state, _ = self.parallel_forward(latents, actions)  # d_{0:T} = f(z_{0:T}, a_{0:T})
        prior_dist = prior(seq_state)
        return (
            latents,
            seq_state,
            post_dist.logits[:, 1:],
            prior_dist.logits[:, :-1],
        )

    def imagine_step(self, latent, action, model_state, prior) -> tuple[Tensor, Tensor, Tensor]:
        # returns next_latent, seq_state, new_model_state
        seq_state, model_state = self.step(
            latent, action, model_state
        )  # d_t or h_{t+1} = f(z_t, a_t, (h_t))
        prior_dist = prior(seq_state)
        next_latent = prior_dist.sample()  # z_{t+1}
        return next_latent, seq_state, model_state  # returns {{z^hat}_{t+1}, d_t/h_{t+1}}

    def imagine_rollout(
        self, initial_latent, initial_seq, model_state, horizon, actor, prior
    ) -> tuple[Tensor, Tensor, Tensor]:
        # returns
        # imagined_latents, imagined_seq_states (actor), imagined_sequence_states (heads), actions
        latent, seq_state = initial_latent, initial_seq  # z_0, d_initial
        latents, seq_states, actions = [], [], []

        for t in range(horizon + 1):
            # a_t = pi(z_t, d_{t-1})
            action = actor(latent, seq_state)
            latents.append(latent)
            seq_states.append(seq_state)
            actions.append(action)
            if t < horizon:
                latent, seq_state, model_state = self.imagine_step(
                    latent, action, model_state, prior
                )
            else:
                # compute final d_t
                seq_state, model_state = self.step(latent, action, model_state)
                seq_states.append(seq_state)

        latents = torch.stack(latents, 1)  # z_{0:T}
        seq_states = torch.stack(seq_states, 1)
        # For Mamba/Transformer this is d_{init:T}
        actions = torch.stack(actions, 1)
        return latents, seq_states[:, :-1], seq_states[:, 1:], actions


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

    def step_through(self, embeddings, actions, posterior, dones=None):
        B, T, _ = embeddings.shape
        latents, hiddens = [], []
        post_logits = []

        # starting from an initial h_0
        hidden = initial_hidden = self.initial_state_from_reference(embeddings)

        for i in range(T):
            post_dist = posterior(embeddings[:, i], hidden)  # z_t = q(x_t, h_t)
            latent = post_dist.rsample()

            if i > 0:
                post_logits.append(post_dist.logits)

            latents.append(latent)
            hiddens.append(hidden)

            _, hidden = self.step(latent, actions[:, i], hidden)  # h_{t+1} = f(z_t, a_t, h_t)
            if dones is not None:
                done = dones[:, i].unsqueeze(-1)
                hidden = (
                    1 - done
                ) * hidden + done * initial_hidden  # update for episode boundaries

        return (
            torch.stack(latents, 1),
            torch.stack(hiddens, 1),
            torch.stack(post_logits, 1),
        )

    def rollout(self, embeddings, actions, posterior, prior, dones=None):
        # returns (latents, states, posterior_dists, prior_dists),
        latents, hiddens, post_logits = self.step_through(embeddings, actions, posterior, dones)
        prior_logits = prior(hiddens).logits

        return latents, hiddens, post_logits, prior_logits

    def imagine_rollout(
        self, initial_latent, initial_seq, model_state, horizon, actor, prior
    ) -> tuple[Tensor, Tensor, Tensor]:
        # returns
        # imagined_latents, imagined_seq_states (actor), imagined_sequence_states (heads), actions
        latent, seq_state = initial_latent, initial_seq  # z_0, h_0
        latents, seq_states, actions = [], [], []

        for t in range(horizon + 1):
            # a_t = pi(z_t, h_0)
            action = actor(latent, seq_state)
            latents.append(latent)
            seq_states.append(seq_state)
            actions.append(action)
            if t < horizon:
                latent, seq_state, model_state = self.imagine_step(
                    latent, action, model_state, prior
                )

        latents = torch.stack(latents, 1)  # z_{0:T}
        seq_states = torch.stack(seq_states, 1)  # h_{0:T}
        actions = torch.stack(actions, 1)
        return latents, seq_states, seq_states, actions


class MambaSequenceModel(SequenceModel):
    def __init__(
        self,
        latent_size,
        action_dim,
        d_model,
        n_layers=2,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        act=nn.SiLU,
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

    def parallel_forward(self, latents, actions, state=None):
        # same as the diagram, concatenates all of the z_t, a_t for s_t
        # d_{0:T} = f(s_{0:T})
        if latents.ndim > actions.ndim:
            latents = latents.flatten(-2)
        x = torch.cat([latents, actions], -1)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, inference_params=state)
        x = self.norm_f(x)
        if state is not None:
            state.seqlen_offset += x.shape[1]
        return x, state

    def step(self, latent, action, state):
        latent_t = latent.unsqueeze(1)
        action_t = action.unsqueeze(1)
        output, state = self.parallel_forward(
            latent_t, action_t, state
        )  # notation is a bit awkward right now but returns d_t given z_t, a_t
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

    def parallel_forward(self, latents, actions, state=None):
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
