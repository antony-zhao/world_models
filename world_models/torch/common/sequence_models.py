from abc import ABC, abstractmethod

import torch
from mamba_ssm.utils.generation import InferenceParams
from torch import nn

from world_models.torch.common.models import (
    DreamerGRU,
    DreamerMLP,
    MambaBlock,
    TransformerBlock,
)


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
        use_block_linear=True,
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

    def forward(self, latents, actions, state=None, dones=None):
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

    def step(self, latent, action, state, done=None):
        latent_t = latent.unsqueeze(1)
        action_t = action.unsqueeze(1)
        output, state = self.forward(latent_t, action_t, state)
        return output.squeeze(1), state
