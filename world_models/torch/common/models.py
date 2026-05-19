import copy

import torch
from mamba_ssm import Mamba2
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    # if hidden dims is specified then doesn't use skip connections
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        num_hiddens=2,
        act=nn.GELU,
        hidden_dims=None,
        final_act=None,
        skip_connections=False,
    ):
        super().__init__()
        if hidden_dims is not None:
            assert len(hidden_dims) + 1 == num_hiddens
            hidden_dim = hidden_dims[0]
            self.skip_connections = False
        else:
            self.skip_connections = skip_connections
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hiddens = []
        for i in range(num_hiddens):
            if hidden_dims is None:
                self.hiddens.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.hiddens.append(nn.Linear(hidden_dim, hidden_dims[i + 1]))
                hidden_dim = hidden_dims[i + 1]
        self.hiddens = nn.ModuleList(self.hiddens)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = act()
        self.final_act = final_act

    def forward(self, x):
        x = self.act(self.input_layer(x))
        for i in range(len(self.hiddens)):
            if self.skip_connections:
                x = self.act(self.hiddens[i](x)) + x
            else:
                x = self.act(self.hiddens[i](x))
        logits = self.output_layer(x)
        if self.final_act is not None:
            return self.final_act(logits)
        return logits


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=nn.GELU):
        super().__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.act = act()

    def forward(self, x):
        x_skip = x.clone()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x + x_skip


class IMPALABlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.GELU, num_blocks=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res = nn.Sequential(
            *[ResBlock(out_channels, out_channels, 3, act) for _ in range(num_blocks)]
        )
        self.act = act()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.res(x)
        return x


class IMPALACNN(nn.Module):
    def __init__(self, image_size, num_blocks, image_channels=3, channel_base=16, act=nn.GELU):
        super().__init__()
        channels = [image_channels] + [channel_base * 2**i for i in range(num_blocks)]
        self.image_size = image_size
        self.image_channels = image_channels
        self.layers = nn.Sequential(
            *[IMPALABlock(channels[i], channels[i + 1], act) for i in range(num_blocks)]
        )
        self.output_dim = self.compute_output_dim()

    def compute_output_dim(self):
        x = torch.zeros(1, self.image_channels, self.image_size, self.image_size)
        x = self.layers(x)
        return x.view(-1).shape[0]

    def forward(self, obs):
        return self.layers(obs)


class ChannelNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(num_channels, eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DreamerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hiddens, act=nn.SiLU):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.RMSNorm(hidden_dim, eps=1e-6))
        layers.append(act())
        for _ in range(num_hiddens):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.RMSNorm(hidden_dim, eps=1e-6))
            layers.append(act())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ContDreamerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hiddens, act=nn.SiLU):
        super().__init__()
        self.layers = DreamerMLP(input_dim, output_dim, hidden_dim, num_hiddens, act)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.mean_head(x), self.log_std_head(x)


class BlockLinear(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=8):
        super().__init__()
        self.networks = nn.ModuleList(
            [
                nn.Linear(input_size // num_blocks, output_size // num_blocks)
                for _ in range(num_blocks)
            ]
        )
        self.num_blocks = num_blocks

    def forward(self, x):
        output = []
        x_chunks = torch.split(x, x.shape[-1] // self.num_blocks, dim=-1)
        for i in range(self.num_blocks):
            output.append(self.networks[i](x_chunks[i]))
        return torch.cat(output, -1)


class DreamerGRU(nn.Module):
    def __init__(self, hidden_state_size, use_block_linear=True):
        super().__init__()
        if use_block_linear:
            self.layer = BlockLinear(hidden_state_size, hidden_state_size * 3)
        else:
            self.layer = nn.Linear(hidden_state_size, hidden_state_size * 3)
        self.hidden_state_size = hidden_state_size

    def forward(self, x, h):
        x = self.layer(x)
        reset, cand, update = torch.split(x, self.hidden_state_size, -1)
        reset = F.sigmoid(reset)
        cand = F.tanh(reset * cand)
        update = F.sigmoid(update - 1)
        h_new = update * cand + (1 - update) * h
        return h_new


class TargetNetwork(nn.Module):
    def __init__(self, original_network, tau=None, update_freq=None):
        super().__init__()
        self.network = copy.deepcopy(original_network)
        self.original = original_network
        for param in self.network.parameters():
            param.requires_grad = False
        if tau is None and update_freq is None:
            raise RuntimeError("At least one of tau or update frequency should be specified")
        self.tau = tau  # esentially the moving average, slowly updates every time
        self.update_freq = update_freq
        self.i = 0

    def update(self):
        target_net_state_dict = self.network.state_dict()
        original_net_state_dict = self.original.state_dict()
        if self.tau is not None:
            for key in original_net_state_dict:
                target_net_state_dict[key] = original_net_state_dict[
                    key
                ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.network.load_state_dict(target_net_state_dict)
        else:
            self.i += 1
            if (self.i % self.update_freq) == 0:
                self.network.load_state_dict(original_net_state_dict)

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    @property
    def net(self):
        return self.network


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        layer_idx,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            layer_idx=layer_idx,
        )

    def forward(self, x, inference_params=None):
        return self.mamba(self.norm(x), inference_params=inference_params) + x

    def allocate_inference_cache(self, batch_size, max_seq_len, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seq_len, **kwargs)


class RotaryEmbedding(nn.Module):
    # From Claude
    """RoPE — rotary position embeddings.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding" (https://arxiv.org/abs/2104.09864)

    Applied to Q and K before attention. Each pair of dims rotates at
    frequency 1 / base^(2i/dim).
    """

    def __init__(self, dim, base=10000.0, max_seq_len=4096):
        super().__init__()
        assert dim % 2 == 0, f"RoPE requires even dim, got {dim}"
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cache_len = seq_len

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, seq_offset=0):
        """Apply RoPE to q and k.

        q, k: (B, num_heads, T, head_dim)
        seq_offset: starting position (nonzero during KV-cache step inference)
        Returns: q_rot, k_rot of the same shape as inputs.
        """
        T = q.shape[-2]
        end = seq_offset + T
        if end > self._cache_len:
            self._build_cache(end)

        cos = self.cos_cached[seq_offset:end].to(dtype=q.dtype)
        sin = self.sin_cached[seq_offset:end].to(dtype=q.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class TransformerFFNetwork(nn.Module):
    def __init__(self, d_model, expand=2, dropout=0.0, act=nn.SiLU):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model * expand, bias=False)
        self.act = act()
        self.lin2 = nn.Linear(d_model * expand, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len=4096,
        expand=2,
        dropout=0.0,
        act=nn.SiLU,
        is_causal=True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.causal = is_causal

        self.rms1 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rms2 = nn.RMSNorm(d_model)
        self.ff = TransformerFFNetwork(d_model, expand, dropout, act)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def allocate_kv_cache(self, batch_size, device, dtype=torch.float32):
        shape = (batch_size, self.num_heads, self.max_seq_len, self.head_dim)
        cache_k = torch.zeros(shape, device=device, dtype=dtype)
        cache_v = torch.zeros(shape, device=device, dtype=dtype)
        return (cache_k, cache_v)

    def forward(self, x, cache=None, seq_offset=0):
        h = self.rms1(x)
        B, T, _ = h.shape

        # Compute Q, K, V
        qkv = self.qkv(h).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv.unbind(dim=0)  # each: (B, num_heads, T, head_dim)

        # Apply RoPE at the correct absolute positions
        q, k = self.rope(q, k, seq_offset=seq_offset)

        if cache is not None:
            # Write new K, V into the cache at positions [seq_offset:seq_offset+T]
            assert seq_offset + T <= self.max_seq_len, (
                f"KV cache overflow: seq_offset={seq_offset}, T={T}, max_seq_len={self.max_seq_len}"
            )
            cache_k, cache_v = cache
            cache_k[:, :, seq_offset : seq_offset + T] = k
            cache_v[:, :, seq_offset : seq_offset + T] = v

            # Use the filled portion only for attention
            k_full = cache_k[:, :, : seq_offset + T]
            v_full = cache_v[:, :, : seq_offset + T]
            new_cache = (cache_k, cache_v)  # return the full buffer for next step
        else:
            k_full = k
            v_full = v
            new_cache = None

        # Causal mask only meaningful when T > 1 (parallel training mode)
        # In step mode (T=1), the new query attends to all cached keys, no future to mask
        is_causal = self.causal and T > 1

        attn = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn = self.out_proj(attn)
        x = x + attn
        x = x + self.ff(self.rms2(x))

        return x, new_cache


def reparameterize_normal(mu, sigma):
    return torch.randn_like(mu) * sigma + mu
