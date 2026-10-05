from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from world_models.torch.agents.actor_critic import Actor, Critic
from world_models.torch.common.decoders import ConvDecoder
from world_models.torch.common.models import MLP, TargetNetwork
from world_models.torch.common.utils import (
    compute_lambda_returns,
    symlog_squared_error,
)


class RepresentationObjective(nn.Module):
    def __init__(self, cfg):
        self.weight = cfg.weight
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
                cfg.latent_size + cfg.d_model if cfg.use_combined_state else cfg.latent_size,
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


class CDPObjective(RepresentationObjective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rssm = cfg.type == "rssm"
        self.cdp_predictor = MLP(
            cfg.d_model, cfg.embedding_dim, cfg.hidden_dim, cfg.num_hiddens, cfg.norm
        )

    def forward(self, context):
        if self.rssm:
            seq_states = context["seq_states"][:, 1:]
            embedding_targets = context["obs_embeddings"][:, 1:].detach()
        else:
            seq_states = context["seq_states"][:, :-1]
            embedding_targets = context["obs_embeddings"][:, 1:].detach()
        embedding_predictions = self.cdp_predictor(seq_states)
        target_norm = embedding_targets.norm(dim=-1)
        pred_norm = embedding_predictions.norm(dim=-1)
        cos_sim = -F.cosine_similarity(
            embedding_targets, embedding_predictions, dim=-1
        ).mean()  # no dim=-1 worked for some reason, something to look at
        return self.weight * cos_sim, {
            "wm/loss/cdp_loss": cos_sim.item(),
            "wm/stats/cdp_target_norm_min": target_norm.min().item(),
            "wm/stats/cdp_target_norm_mean": target_norm.mean().item(),
            "wm/stats/cdp_pred_norm_min": pred_norm.min().item(),
            "wm/stats/cdp_pred_norm_mean": pred_norm.mean().item(),
        }


class ActionObjective(RepresentationObjective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_type = cfg.action_type
        self.rssm = cfg.type == "rssm"
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
        states = context["states"][:, :-1]  # rssm has (B, T-1), transformer has (B, 1:T-1)
        if self.rssm:
            next_in = context["obs_embeddings"][:, 1:]
            mask = 1 - context["terminated"][:, :-1].float()
            targets = context["actions"][:, :-1]
        else:
            next_in = context["obs_embeddings"][:, 2:]
            mask = 1 - context["terminated"][:, 1:-1].float()
            targets = context["actions"][:, 1:-1]
        combined_in = torch.cat([states, next_in], -1)
        action_dist = self.action_predictor.policy_dist(combined_in)
        if self.action_type == "discrete":
            targets = targets.argmax(-1)
        prediction_loss = -(action_dist.log_prob(targets) * mask).sum() / mask.sum()

        return self.weight * prediction_loss, {
            "wm/loss/action_prediction_loss": prediction_loss.item(),
        }


class ValueObjective(RepresentationObjective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rssm = cfg.type == "rssm"
        input_dim = cfg.latent_size + cfg.d_model
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
        states = context["states"]
        if self.rssm:
            rewards = context["rewards"][:, :-1]
            terminated = context["terminated"][:, :-1]
        else:
            rewards = context["rewards"][:, 1:-1]
            terminated = context["terminated"][:, 1:-1]
        _, value_logits = self.value_predictor(states)
        value_targets, _ = self.target(states)
        returns = compute_lambda_returns(
            value_targets.detach(),
            rewards,
            1 - terminated.float(),
            self.gamma,
            self.lambda_,
        )
        # mask = 1 - context["terminated"][:, :-1].float()
        value_loss = (
            -self.value_predictor.make_dist(value_logits[:, :-1]).log_prob(
                returns.detach()
            )  # * mask
        ).mean()  # .sum() / mask.sum()
        return value_loss * self.weight, {"wm/loss/value_prediction_loss": value_loss.item()}

    def step(self):
        self.target.update()


class ContrastiveTrajectoryObjective(RepresentationObjective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rssm = cfg.type == "rssm"
        self.gamma = cfg.gamma
        self.beta = cfg.beta

        self.state_proj = MLP(cfg.d_model, cfg.embed_dim, cfg.hidden_dim, cfg.num_hiddens, cfg.norm)
        self.obs_proj = MLP(
            cfg.embedding_dim, cfg.embed_dim, cfg.hidden_dim, cfg.num_hiddens, cfg.norm
        )

        B = cfg.batch_size
        T = cfg.seq_len - 1
        weight_block = self.discount_matrix(T, self.gamma) - self.beta
        weight = torch.block_diag(*([weight_block] * B)) + self.beta
        self.register_buffer("weight_matrix", weight)
        target_block = torch.ones((T, T))
        target = torch.block_diag(*([target_block] * B))
        self.register_buffer("target_matrix", target)
        t_idx = torch.arange(T)
        lag_block = (t_idx[None, :] - t_idx[:, None]).float()  # entry j - i
        self.register_buffer("lag_matrix", torch.block_diag(*([lag_block] * B)))

    def discount_matrix(self, n, gamma, normalize=False):
        idx = torch.arange(n)
        exps = (idx[None, :] - idx[:, None]).clamp(min=0)
        mat = torch.triu(torch.full((n, n), gamma) ** exps)
        if normalize:
            mat /= mat.sum(-1, keepdim=True)
        return mat

    def forward(self, context):
        if self.rssm:
            seq_states = context["seq_states"][:, 1:]
            embedding_targets = context["obs_embeddings"][:, 1:]
        else:
            seq_states = context["seq_states"][:, :-1]
            embedding_targets = context["obs_embeddings"][:, 1:]

        B, T, _ = seq_states.shape
        seq_proj = self.state_proj(seq_states)
        obs_proj = self.obs_proj(embedding_targets)
        dones = context["dones"].float()
        segments = (dones.cumsum(1) - dones)[:, 1:].flatten()
        same_segment = segments[:, None] == segments[None, :]
        target = self.target_matrix * same_segment

        in_block = self.target_matrix.bool()
        weight = torch.where(in_block & ~same_segment, self.beta, self.weight_matrix)

        dot_prod = torch.einsum("ijx,klx->ijkl", seq_proj, obs_proj).reshape(B * T, B * T)
        loss = F.binary_cross_entropy_with_logits(
            dot_prod, target, weight=weight, reduction="sum"
        ) / (B * T)

        stats = {
            "wm/loss/contrastive_loss": loss.item(),
            "wm/stats/state_embed_norm": seq_proj.norm(dim=-1).mean().item(),
            "wm/stats/obs_embed_norm": obs_proj.norm(dim=-1).mean().item(),
        }
        return self.weight * loss, stats


class TemporalConsistencyObjective(RepresentationObjective):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rssm = cfg.type == "rssm"
        self.predictor = MLP(
            cfg.d_model, cfg.embedding_dim, cfg.hidden_dim, cfg.num_hiddens, cfg.norm
        )

    # def forward(self, context):
    #     if self.rssm:
    #         seq_states = context["seq_states"][:, 1:]
    #         embedding_targets = context["obs_embeddings"][:, 1:].detach()
    #     else:
    #         seq_states = context["seq_states"][:, :-1]
    #         embedding_targets = context["obs_embeddings"][:, 1:].detach()
    #     embedding_predictions = self.cdp_predictor(seq_states)
    #     target_norm = embedding_targets.norm(dim=1)
    #     pred_norm = embedding_predictions.norm(dim=1)
    #     cos_sim = -F.cosine_similarity(
    #         embedding_targets, embedding_predictions, dim=1, eps=1e-5
    #     ).mean()  # no dim=-1 worked for some reason, something to look at
    #     return self.weight * cos_sim, {
    #         "wm/loss/cdp_loss": cos_sim.item(),
    #         "wm/stats/cdp_target_norm_min": target_norm.min().item(),
    #         "wm/stats/cdp_target_norm_mean": target_norm.mean().item(),
    #         "wm/stats/cdp_pred_norm_min": pred_norm.min().item(),
    #         "wm/stats/cdp_pred_norm_mean": pred_norm.mean().item(),
    #     }


class Plan2Explore(RepresentationObjective):
    def __init__(self):
        pass

    def forward(self):
        pass

    def compute_intrinsic_reward(self):
        pass


class LatentDisagreement(RepresentationObjective):
    pass
