class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.mlp = DreamerMLP(
            config.state_size, config.action_dim, config.hidden_dim, config.num_hiddens_actor_critic
        )
        self.action_type = config.action_type
        if config.action_type == "continuous":
            self.log_std = nn.Parameter(-torch.ones(config.action_dim))
        else:
            self.actor_unimix = config.actor_unimix

    def policy_dist(self, x):
        logits = self.mlp(x)
        if self.action_type == "discrete":
            probs = torch.softmax(logits, -1)
            unimixed_probs = unimix(probs, self.action_dim, self.actor_unimix)
            logits = probs_to_logits(unimixed_probs)
            action_dist = Categorical(logits=logits)
        else:
            action_dist = Normal(loc=logits, scale=torch.exp(self.log_std))
        return action_dist

    def sample_action(self, dist):
        if self.action_type == "discrete":
            action = dist.sample()
        else:
            action = dist.rsample()
        return action

    def policy_fn(self, x, det=False):
        # actually choosing the action, returns the actual action as well as the log prob of the action and entropy
        action_dist = self.policy_dist(x)
        if not det:
            action = self.sample_action(action_dist)
        else:
            return action_dist.mode
        return action.detach()


class Critic(nn.Module):
    def __init__(self, config, bins):
        super().__init__()
        self.mlp = DreamerMLP(
            config.state_size, config.num_bins, config.hidden_dim, config.num_hiddens_actor_critic
        )
        self.bins = bins

    def forward(self, x):
        logits = self.mlp(x)
        weighted_average = WeightedAverageOverBins(self.bins, logits)
        return weighted_average.weighted_average(), logits
