import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from world_models.torch.common.distributions import SafeTanhNormal
from world_models.torch.common.heads import TwoHotHead
from world_models.torch.common.models import MLP, ContMLP
from world_models.torch.common.utils import make_state, unimix


class Actor(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim,
        num_hiddens,
        action_type,
        act=nn.SiLU,
        actor_unimix=None,
        log_std_min=None,
        log_std_max=None,
        std_min=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_type = action_type
        if action_type == "discrete":
            self.mlp = MLP(input_dim, action_dim, hidden_dim, num_hiddens, act)
            self.actor_unimix = actor_unimix
        else:
            self.mlp = ContMLP(
                input_dim,
                action_dim,
                hidden_dim,
                num_hiddens,
                act,
                log_std_min,
                log_std_max,
                std_min,
            )

    def policy_dist(self, x):
        if self.action_type == "discrete":
            logits = self.mlp(x)
            probs = torch.softmax(logits, -1)
            unimixed_probs = unimix(probs, self.action_dim, self.actor_unimix)
            action_dist = Categorical(probs=unimixed_probs)
        else:
            mean, std = self.mlp(x)
            action_dist = SafeTanhNormal(loc=mean, scale=std)
        return action_dist

    def sample_action(self, dist):
        if self.action_type == "discrete":
            action = dist.sample()
        else:
            action = dist.rsample()
        return action

    def policy_fn(self, x, det=False):
        # actually choosing the action, returns the actual action
        action_dist = self.policy_dist(x)
        if not det:
            action = self.sample_action(action_dist)
        else:
            action = action_dist.mode
        return action.detach()

    def format_action(self, action):
        if self.action_type == "discrete":
            return F.one_hot(action, self.action_dim).float()
        return action

    def forward(self, latent, seq_state):
        x = make_state(latent, seq_state)
        dist = self.policy_dist(x)
        action = self.sample_action(dist)
        return self.format_action(action)


class Critic(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_hiddens, num_bins, bin_low, bin_high, act=nn.SiLU
    ):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, hidden_dim, num_hiddens - 1, act)
        self.twohot = TwoHotHead(hidden_dim, num_bins, bin_low, bin_high, hidden_dim, None)
        # self.twohot.zero_init()

    def forward(self, x):
        logits = self.mlp(x)
        twohot = self.twohot(logits)
        return twohot.mean(), twohot.logits

    def make_dist(self, logits):
        return self.twohot.make_dist_from_logits(logits)
