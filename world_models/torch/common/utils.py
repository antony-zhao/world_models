import numpy as np
import torch
import torch.nn.functional as F
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


def make_state(latent, hidden):
    return torch.cat([latent.flatten(-2), hidden], -1)


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
    T = rewards.shape[0]
    returns = torch.empty_like(rewards)

    next_return = values[-1]
    for t in reversed(range(T)):
        next_return = rewards[t] + gamma * continues[t] * (
            (1 - lambda_) * values[t + 1] + lambda_ * next_return
        )
        returns[t] = next_return

    return returns


class TwoHotEncoding:
    def __init__(self, bins, logits=None, forward=symexp, backward=symlog):
        self.logits = logits
        self.probs = torch.softmax(self.logits, -1)
        self.bins = bins
        self.forward = forward
        self.backward = backward

    def weighted_average(self):
        weighted_average = self.probs @ self.bins
        return self.forward(weighted_average)

    def two_hot(self, vals):
        index_1 = torch.bucketize(vals, self.bins) - 1
        index_1 = index_1.clamp(0, len(self.bins) - 2)
        index_2 = index_1 + 1
        b_k = self.bins[index_1]
        b_k2 = self.bins[index_2]
        proportion_2 = torch.abs(b_k - vals) / torch.abs(b_k2 - b_k)
        proportion_1 = torch.abs(b_k2 - vals) / torch.abs(b_k2 - b_k)
        one_hot_1 = F.one_hot(index_1, len(self.bins))
        one_hot_2 = F.one_hot(index_2, len(self.bins))
        two_hot_encoded = (
            proportion_1.unsqueeze(-1) * one_hot_1 + proportion_2.unsqueeze(-1) * one_hot_2
        )
        return two_hot_encoded

    def log_prob(self, vals, aggregate=True):
        # basically just the loss
        target = self.two_hot(self.backward(vals)).detach()
        log_probs = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        loss = (target * log_probs).sum(-1)
        return loss.mean() if aggregate else loss
