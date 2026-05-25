class DreamerV3:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.world_model = torch.compile(DreamerWorldModel(config).to(self.device))
        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config, self.world_model.bins).to(self.device)
        self.init_models()
        self.critic_target = TargetNetwork(self.critic, config.critic_tau)
        self.optim_wm = Adam(self.world_model.parameters(), config.wm_lr, eps=1e-5)
        self.optim_actor = Adam(self.actor.parameters(), config.reinforce_lr, eps=1e-5)
        self.optim_critic = Adam(self.critic.parameters(), config.reinforce_lr, eps=1e-5)

        act_dim = () if config.action_type == "discrete" else (config.action_dim,)
        if config.obs_type == "image":
            self.is_image = True
            self.buffer = PerEnvBuffer(
                config.num_envs,
                [config.image_dim, act_dim, (), ()],
                dtypes=[np.uint8, np.int32, np.float32, np.bool],
                buffer_size=1_000_000,
            )
        elif config.obs_type == "vector":
            self.is_image = False
            self.buffer = PerEnvBuffer(
                config.num_envs, [(config.obs_dim,), act_dim, (), ()], buffer_size=1_000_000
            )
        else:
            raise NotImplementedError
        # buffer needs to account for order in episodes
        self.active_hidden = torch.zeros(config.num_envs, config.hidden_state_size).to(self.device)
        self.eval_hidden = torch.zeros(1, config.hidden_state_size).to(self.device)
        # the history for the environment itself, keeping track of it in here since
        # it would be a bit weird to have this be in the main part of the program

        self.range_ema = None
        self.return_range_tau = config.return_range_tau
        # Used for calculating the range of returns to help normalize the reinforce gradient

        self.gamma = config.gamma
        self.lambda_ = config.lambda_
        self.percentiles = config.percentiles
        self.action_type = config.action_type
        self.num_actions = config.action_dim

    def choose_action(self, obs, det=False):
        state, latent = self.obs_to_state(obs, self.active_hidden)
        action = self.actor.policy_fn(state, det)
        return action, latent

    def eval_action(self, obs, det=True, reset=False):
        if reset:
            self.eval_hidden = self.world_model._get_hidden(1)
        state, latent = self.obs_to_state(obs, self.eval_hidden)
        action = self.actor.policy_fn(state, det)
        self.eval_hidden = self.world_model.recurrent_step(
            self.eval_hidden, latent, action
        ).detach()
        return to_numpy(action)

    def process_sample(self, obs, latent, action, reward, done):
        # do a step in RSSM and store stuff in buffer
        self.buffer.add_sample([obs, to_numpy(action), reward, done])

        continue_ = torch.tensor(1 - done).unsqueeze(1).to(self.device)
        self.active_hidden = (
            continue_ * self.world_model.recurrent_step(self.active_hidden, latent, action)
            + (1 - continue_) * self.world_model._get_hidden(self.config.num_envs)
        ).detach()

    def imagine_rollout(self, state, steps=None):
        states = []
        actions = []
        action_log_probs = []
        action_entropies = []
        rewards = []
        continues = []
        for _ in range(self.config.rollout_length if steps is None else steps):
            action_dist = self.actor.policy_dist(state)
            action = self.actor.sample_action(action_dist).detach()
            action_prob = action_dist.log_prob(action)
            action_log_probs.append(action_prob)
            action_entropy = action_dist.entropy()
            action_entropies.append(action_entropy)
            if self.config.action_type == "discrete":
                action = F.one_hot(action.long(), self.config.action_dim).float()
            (next_latent, next_hidden), reward, continue_ = self.world_model.imagine_step(
                state[:, : self.config.latent_size], state[:, self.config.latent_size :], action
            )
            states.append(state.detach())
            actions.append(action.detach())
            rewards.append(reward.detach())
            continues.append(continue_.squeeze().detach())
            state = torch.concatenate([next_latent.flatten(-2), next_hidden], 1)
        states.append(state.detach())
        return (
            torch.stack(states),
            torch.stack(rewards),
            torch.stack(continues),
            torch.stack(action_log_probs),
            torch.stack(action_entropies),
        )

    def train(self):
        obs, actions, rewards, dones = self.buffer.sample_as_tensors(
            self.config.device, self.config.sample_batch_size, self.config.sample_seq_len
        )
        if self.config.action_type == "discrete":
            actions = F.one_hot(actions.long(), self.config.action_dim).float()
        with torch.amp.autocast(device_type="cuda"):
            loss_wm, loss_dict, new_states, _ = self.world_model.world_model_loss(
                obs, actions, rewards, dones
            )
        self.optim_wm.zero_grad(set_to_none=True)
        loss_wm.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 5)
        self.optim_wm.step()

        if self.buffer.size < self.config.train_reinforce_after:
            return loss_dict

        states, rewards, continues, log_probs, entropy = self.imagine_rollout(
            new_states.reshape(-1, self.config.state_size)
        )
        continues[0] = 1 - dones.flatten()

        with torch.amp.autocast(device_type="cuda"):
            loss_critic, returns, values = self.reinforce_critic_loss(states, rewards, continues)
            loss_actor, actor_ent = self.reinforce_actor_loss(returns, values, log_probs, entropy)

        self.optim_critic.zero_grad(set_to_none=True)
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.optim_critic.step()
        self.optim_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.optim_actor.step()

        loss_dict["loss/actor loss"] = to_numpy(loss_actor)
        loss_dict["loss/critic loss"] = to_numpy(loss_critic)
        loss_dict["loss/actor entropy"] = to_numpy(actor_ent)
        return loss_dict

    def reinforce_actor_loss(self, returns, values, action_log_probs, entropy):
        range_ = torch.quantile(returns, 1 - self.percentiles) - torch.quantile(
            returns, self.percentiles
        )
        if self.range_ema is not None:
            self.range_ema = (
                range_ * self.return_range_tau + self.range_ema * (1 - self.return_range_tau)
            ).detach()
        else:
            self.range_ema = range_

        adv = ((returns - values) / torch.clip(self.range_ema, min=1)).detach()
        actor_loss = -(adv * action_log_probs + entropy * self.config.entropy_coef)
        actor_loss = actor_loss.mean()
        return actor_loss, entropy.mean()

    def reinforce_critic_loss(self, states, rewards, continues):
        self.critic_target.update()
        values, value_logits = self.critic(states)
        value_target, _ = self.critic_target(states)
        returns = compute_lambda_returns(values, rewards, continues, self.gamma, self.lambda_)
        value_bins = WeightedAverageOverBins(self.world_model.bins, value_logits[:-1])
        loss = -value_bins.log_prob(returns.detach(), aggregate=False)
        loss -= value_bins.log_prob(value_target.detach()[:-1], aggregate=False)
        loss = loss.mean()
        return (
            loss,
            returns.detach(),
            values.detach()[:-1],
        )  # returning returns and values for the actor to reuse later

    def obs_to_state(self, obs, hidden=None):
        if hidden is None:
            hidden = self.world_model._get_hidden(obs.shape[0])
        transformed_obs = transform_obs(obs, self.is_image)
        latent_prob = self.world_model.encoder(transformed_obs, hidden)
        latent = Independent(OneHotCategoricalStraightThrough(latent_prob), 1).sample()
        state = make_state(latent, hidden)
        return state, latent
