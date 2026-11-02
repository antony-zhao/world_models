import numpy as np
import torch
from torch import nn

ACTIVATION_MAP = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def get_activation(name):
    return ACTIVATION_MAP[name.lower()]


class BatchNorm1dReshape(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        x = self.bn(x)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        return x


def get_norm(name, dim):
    if name is None or name.lower() == "none":
        return nn.Identity()
    elif name.lower() == "layer":
        return nn.LayerNorm(dim, eps=1e-6)
    elif name.lower() == "rms":
        return nn.RMSNorm(dim, eps=1e-6)
    elif name.lower() == "batch":
        return BatchNorm1dReshape(dim)
    else:
        raise ValueError(f"Unknown norm: {name}")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_pad(kernel_size, stride):
    return int(np.ceil((kernel_size - stride) / 2))


def to_numpy(x):
    return x.detach().cpu().numpy()


def unimix(x, num_codes, proportion=0.01):
    uniform = torch.ones_like(x) / num_codes
    return x * (1 - proportion) + uniform * proportion


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def symlog_squared_error(y, y_hat):
    return ((symlog(y) - symlog(y_hat)) ** 2).sum(-1).mean()


def make_state(latent, seq_state):
    if latent.ndim > seq_state.ndim:
        latent = latent.flatten(-2)
    return torch.cat([latent, seq_state], -1)


def transform_obs(obs, is_image):
    if is_image:
        transformed_obs = obs / 255.0 - 0.5
    else:
        transformed_obs = symlog(obs)
    return transformed_obs


# the initializers are old/unused but are still here for now, might delete later
def init_last_layer(model, init_func):
    for module in model.modules():
        if isinstance(module, (nn.Linear)):
            last_layer = module

    if isinstance(last_layer, nn.Linear):
        init_func(last_layer.weight.data)
        if hasattr(last_layer.bias, "data"):
            last_layer.bias.data.fill_(0.0)
    elif isinstance(last_layer, nn.LayerNorm):
        last_layer.weight.data.fill_(1.0)
        if hasattr(last_layer.bias, "data"):
            last_layer.bias.data.fill_(0.0)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        if hasattr(module.bias, "data"):
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        if hasattr(module.bias, "data"):
            module.bias.data.fill_(0.0)


def compute_lambda_returns(values, rewards, continues, gamma, lambda_):
    T = rewards.shape[1]
    returns = torch.empty_like(rewards)
    next_return = values[:, -1]
    for t in reversed(range(T)):
        next_return = rewards[:, t] + gamma * continues[:, t] * (
            (1 - lambda_) * values[:, t + 1] + lambda_ * next_return
        )
        returns[:, t] = next_return
    return returns


def compute_gae(rewards, values, dones, discount=0.99, lam=0.95):
    # from here https://github.com/zplizzi/pytorch-ppo/blob/master/gae.py
    # values is T+1, since it needs to include the value for the next_obs, also purely numpy
    B, T = rewards.shape
    next_value = values[:, -1]
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - dones[:, t]
        if t == T - 1:
            nextvalues = next_value
        else:
            nextvalues = values[t + 1]
        delta = rewards[:, t] + discount * nextvalues * nextnonterminal - values[:, t]
        advantages[:, t] = lastgaelam = delta + discount * lam * nextnonterminal * lastgaelam
    returns = advantages + values[:, :-1]
    return advantages, returns


import numpy as np
import torch


def make_quadrature(lam=1.0, T=16, t_min=0.2, t_max=4.0):
    """Build quadrature knots and weights for the Epps-Pulley integral.

    Returns:
        t_knots : (T,) frequency knots, uniformly spaced in [t_min, t_max]
        weights : (T,) trapezoidal weights * Gaussian frequency weight w(t_k)
    """
    t_knots = np.linspace(t_min, t_max, T)
    dt = t_knots[1] - t_knots[0]
    alpha = np.full(T, dt)  # trapezoidal weights
    alpha[0] *= 0.5  # endpoints get half weight
    alpha[-1] *= 0.5
    return t_knots, alpha * np.exp(-(t_knots**2) / (2 * lam**2))


def random_unit_vectors(M, D, rng):
    """Sample M points uniformly from S^{D-1}."""
    u = rng.standard_normal((M, D))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u


def sigreg(Z, M=1024, lam=1.0, T=16, rng=None):
    # Numpy implementations from: https://rezabyt.github.io/blogposts/sigreg-tutorial.html
    """SIGReg = average Epps-Pulley statistic over M random projections.

    Z   : (N, D) batch of embeddings
    M   : number of random projection directions
    lam : Gaussian-weight bandwidth
    T   : number of quadrature knots
    """
    if rng is None:
        rng = np.random.default_rng()

    t_knots, weights = make_quadrature(lam=lam, T=T)
    U = random_unit_vectors(M, Z.shape[1], rng)  # (M, D)
    H = U @ Z.T  # (M, N): M projections

    # Vectorized Epps-Pulley across all M projections
    angles = t_knots[None, :, None] * H[:, None, :]  # (M, T, N)
    C = np.cos(angles).mean(axis=2)  # (M, T)
    S = np.sin(angles).mean(axis=2)  # (M, T)
    G = np.exp(-(t_knots**2) / 2)  # (T,)
    diff_sq = (C - G[None, :]) ** 2 + S**2  # (M, T)
    return Z.shape[0] * (weights * diff_sq).sum(axis=1).mean()  # scalar


class SIGReg(torch.nn.Module):
    # From claude based on above.
    """Differentiable torch version of `sigreg`.

    forward(Z): Z is (..., N, D); the statistic is computed per leading index
    (e.g. per view) over the N samples, then averaged.
    """

    def __init__(self, M=1024, lam=1.0, T=16, t_min=0.2, t_max=4.0):
        super().__init__()
        self.M = M
        t_knots, weights = make_quadrature(lam=lam, T=T, t_min=t_min, t_max=t_max)
        t_knots = torch.tensor(t_knots, dtype=torch.float32)
        self.register_buffer("t", t_knots)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("G", torch.exp(-(t_knots**2) / 2))

    def forward(self, Z):
        Z = Z.float()
        U = torch.randn(Z.shape[-1], self.M, device=Z.device)
        U = U / U.norm(dim=0, keepdim=True)  # (D, M) random unit directions
        H = Z @ U  # (..., N, M)
        angles = H.unsqueeze(-1) * self.t  # (..., N, M, T)
        C = angles.cos().mean(-3)  # (..., M, T)
        S = angles.sin().mean(-3)
        diff_sq = (C - self.G) ** 2 + S**2
        return Z.shape[-2] * (diff_sq @ self.weights).mean()
