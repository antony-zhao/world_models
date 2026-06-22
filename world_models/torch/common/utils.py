import numpy as np
import torch
from torch import nn


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
