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
        actor_unimix=None,
        log_std_min=None,
        log_std_max=None,
        std_min=None,
        act=nn.SiLU,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_type = action_type
        if action_type == "discrete":
            self.mlp = MLP(input_dim, action_dim, hidden_dim, num_hiddens, act=act)
            self.actor_unimix = actor_unimix
        else:
            self.mlp = ContMLP(
                input_dim,
                action_dim,
                hidden_dim,
                num_hiddens,
                log_std_min,
                log_std_max,
                std_min,
                act=act,
            )

    def policy_dist(self, x):
        if self.action_type == "discrete":
            logits = self.mlp(x)
            probs = torch.softmax(logits, -1)
            unimixed_probs = unimix(probs, self.action_dim, self.actor_unimix)
            try:
                action_dist = Categorical(probs=unimixed_probs)
            except ValueError:
                self._report_action_dist_failure(
                    x=x, logits=logits, probs=probs, unimixed_probs=unimixed_probs
                )
                raise
        else:
            mean, std = self.mlp(x)
            try:
                action_dist = SafeTanhNormal(loc=mean, scale=std)
            except ValueError:
                self._report_action_dist_failure(x=x, mean=mean, std=std)
                raise
        return action_dist

    def _report_action_dist_failure(self, **tensors):
        # x being non-finite implicates the world model/upstream state; logits/probs/etc
        # being non-finite while x is clean implicates this actor's own weights instead.
        print("\n[Actor.policy_dist] failed to build the action distribution - diagnostics:")
        for name, t in tensors.items():
            finite = torch.isfinite(t)
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            finite_vals = t[finite]
            lo = finite_vals.min().item() if finite_vals.numel() else float("nan")
            hi = finite_vals.max().item() if finite_vals.numel() else float("nan")
            print(
                f"  {name}: shape={tuple(t.shape)} nan={n_nan} inf={n_inf} "
                f"finite_range=[{lo:.4g}, {hi:.4g}]"
            )

        bad_params = [n for n, p in self.named_parameters() if not torch.isfinite(p).all()]
        if bad_params:
            print(f"  actor parameters containing nan/inf: {bad_params}")
        else:
            print(
                "  actor parameters: all finite -> the bad values came in through the "
                "input `x`, i.e. upstream from the world model/RSSM/encoder, not this "
                "actor's own weights"
            )
        print()

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
        self.twohot = TwoHotHead(
            input_dim, num_bins, bin_low, bin_high, hidden_dim, num_hiddens, act=act
        )
        self.twohot.zero_init()

    def forward(self, x):
        twohot = self.twohot(x)
        return twohot.mean(), twohot.logits

    def make_dist(self, logits):
        return self.twohot.make_dist_from_logits(logits)
