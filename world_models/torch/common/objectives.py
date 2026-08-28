from typing import List

import torch
from torch import nn

from world_models.torch.agents.actor_critic import Actor, Critic
from world_models.torch.common.decoders import ConvDecoder
from world_models.torch.common.models import TargetNetwork
from world_models.torch.common.utils import compute_lambda_returns, make_state, symlog_squared_error


class RepresentationObjective(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def step(self): ...


class CompoundObjective(nn.Module):
    def __init__(self, objectives: List):
        super().__init__()
        self.objectives = nn.ModuleList(objectives)

    def forward(self, context):
        total_loss = 0.0
        merged_dict = {}
        for obj in self.objectives:
            loss, metrics = obj(context)
            total_loss = total_loss + loss
            merged_dict |= metrics
        return total_loss, merged_dict

    def step(self):
        for obj in self.objectives:
            obj.step()


class ReconstructionObjective(RepresentationObjective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight = cfg.weight
        self.obs_type = cfg.obs_type
        self.decoder = self.build_decoder(cfg, cfg.encoder_conv_dim)
        if hasattr(cfg, "stop_grad"):
            self.stop_grad = cfg.stop_grad
        else:
            self.stop_grad = False
        # useful for reconstructionless objectives just to verify that the model is learning
        self.use_combined_state = cfg.use_combined_state

    def build_decoder(self, cfg, encoder_output_dim):
        if cfg.type == "conv":
            return ConvDecoder(
                cfg.latent_size + cfg.state_dim if cfg.use_combined_state else cfg.latent_size,
                encoder_output_dim,
                cfg.filter_base,
                cfg.num_convs,
                cfg.kernel_size,
                cfg.image_channels,
            )
        elif cfg.type == "vector":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, context):
        obs = context["obs"]
        decoder_in = (
            context["states"] if self.use_combined_state else context["latents"].flatten(-2)
        )
        if self.stop_grad:
            decoder_in = decoder_in.detach()
        reconstructions = self.decoder(decoder_in)

        if self.obs_type == "image":
            loss = ((obs - reconstructions) ** 2).flatten(2).sum(-1).mean()
        else:
            loss = symlog_squared_error(obs, reconstructions)

        return self.weight * loss, {"wm/loss/reconstruction_loss": loss.item()}


class ActionObjective(RepresentationObjective):
    def __init__(self, cfg):
        # Original MuDreamer uses x_t and s_{t-1} for rssm,
        # for the transformer/mamba we use x_t and x_{t-1}
        # This specifically does mean to not use seq_state for transformer/mamba,
        # as it provides action information to the prediction
        super().__init__(cfg)
        self.weight = cfg.weight
        self.action_type = cfg.action_type
        input_dim = cfg.embedding_dim + cfg.joint_state_dim
        self.action_predictor = Actor(
            input_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.num_hiddens,
            cfg.action_type,
            cfg.actor_unimix,
            cfg.log_std_min,
            cfg.log_std_max,
            cfg.std_min,
        )  # repurposing actor since it shares most of the same architecture

    def forward(self, context):
        # latent = context["latents"][:, 1:-1]
        # seq_state = context["seq_states"][:, :-2]
        latent = context["latents"][:, :-1]
        seq_state = context["seq_states"][:, :-1]
        state = make_state(latent, seq_state)
        next_in = context["obs_embeddings"][:, 1:]  # [:, 2:]
        combined_in = torch.cat([state, next_in], -1)
        action_dist = self.action_predictor.policy_dist(combined_in)
        # mask = 1 - context["terminated"][:, 1:-1].float()
        # targets = context["actions"][:, 1:-1]
        mask = 1 - context["terminated"][:, :-1].float()
        targets = context["actions"][:, :-1]
        if self.action_type == "discrete":
            targets = targets.argmax(-1)
        prediction_loss = -(action_dist.log_prob(targets) * mask).sum() / mask.sum()

        return self.weight * prediction_loss, {
            "wm/loss/action_prediction_loss": prediction_loss.item(),
        }


class ValueObjective(RepresentationObjective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight = cfg.weight
        input_dim = cfg.latent_size + cfg.state_dim
        self.value_predictor = Critic(
            input_dim,
            cfg.hidden_dim,
            cfg.num_hiddens,
            cfg.num_bins,
            cfg.bin_low,
            cfg.bin_high,
        )  # similarly repurposing the Critic since it has the same architecture
        self.target = TargetNetwork(self.value_predictor, cfg.tau)
        self.gamma = cfg.gamma
        self.lambda_ = cfg.lambda_

    def forward(self, context):
        states = make_state(
            context["latents"], context["seq_states"]
        )  # make_state(context["latents"][:, 1:], context["seq_states"][:, :-1])
        _, value_logits = self.value_predictor(states)
        value_targets, _ = self.target(states)
        returns = compute_lambda_returns(
            value_targets.detach(),
            context["rewards"][:, :-1],  # [:, 1:-1],
            1 - context["terminated"][:, :-1].float(),  # [:, 1:-1].float(),
            self.gamma,
            self.lambda_,
        )
        mask = 1 - context["terminated"][:, :-1].float()
        value_loss = (
            -self.value_predictor.make_dist(value_logits[:, :-1]).log_prob(
                returns.detach()
            )  # * mask
        ).mean()  # .sum() / mask.sum()
        return value_loss * self.weight, {"wm/loss/value_prediction_loss": value_loss.item()}

    def step(self):
        self.target.update()
