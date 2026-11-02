from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from world_models.torch.agents.actor_critic import Actor, Critic
from world_models.torch.common.decoders import ConvDecoder
from world_models.torch.common.models import MLP, TargetNetwork
from world_models.torch.common.utils import (
    compute_lambda_returns,
    init_last_layer,
    symlog_squared_error,
    transform_obs,
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
        self.num_negatives = cfg.num_negatives
        self.loss_type = cfg.get("loss_type", "bce")  # "bce" (CRL binary NCE) or "infonce"
        self.logsumexp_penalty = cfg.get("logsumexp_penalty", 0.1)  # infonce only
        # "obs": anchor is (u_t, a_t), CRL's phi(s, a), as a separate head on the shared encoder;
        # goals are u_{t+1}, u_{t+2}, ...; "state": anchor is the sequence-model state h_t
        self.anchor = cfg.get("anchor", "obs")
        # infonce logit: "l2" -||phi - psi|| (JaxGCRL / scaling-crl) or "dot" phi^T psi (CRL)
        self.energy = self._neg_l2 if cfg.get("energy", "l2") == "l2" else self._dot
        # optional prefix for logged keys, so several contrastive terms can be logged side by side
        # ("wm/loss/contrastive_loss" -> "wm/loss/<log_name>/contrastive_loss"); None keeps the
        # plain keys used by earlier runs
        self.log_name = cfg.get("log_name", None)

        anchor_dim = {
            "obs": cfg.embedding_dim + cfg.action_dim,
            "posterior": cfg.latent_size + cfg.action_dim,
            "state": cfg.d_model,
        }[self.anchor]
        self.anchor_proj = MLP(anchor_dim, cfg.embed_dim, cfg.hidden_dim, cfg.num_hiddens, cfg.norm)

        self.goal_proj = MLP(
            cfg.embedding_dim, cfg.embed_dim, cfg.hidden_dim, cfg.num_hiddens, cfg.norm
        )
        for proj in (self.anchor_proj, self.goal_proj):
            init_last_layer(proj, lambda w: nn.init.uniform_(w, -1e-12, 1e-12))

        B = cfg.batch_size
        T = cfg.seq_len - 1
        weight_block = self.discount_matrix(T, self.gamma)
        weight = torch.stack(([weight_block] * B)).reshape(B * T, T)
        self.register_buffer("weight_matrix", weight)
        target_block = torch.ones((T, T))
        target = torch.stack(([target_block] * B)).reshape(B * T, T)
        self.register_buffer("pos_target_matrix", target)

    def discount_matrix(self, n, gamma, normalize=False):
        idx = torch.arange(n)
        exps = (idx[None, :] - idx[:, None]).clamp(min=0)
        mat = torch.triu(torch.full((n, n), gamma) ** exps)
        if normalize:
            mat /= mat.sum(-1, keepdim=True)
        return mat

    @staticmethod
    def _neg_l2(x, y):
        # -||x - y|| between every row of x and every row of y (batched over leading dims);
        # the 1e-6 keeps the sqrt gradient finite at distance 0
        return -torch.sqrt(((x[..., :, None, :] - y[..., None, :, :]) ** 2).sum(-1) + 1e-6)

    @staticmethod
    def _dot(x, y):
        # x^T y between every row of x and every row of y (batched over leading dims)
        return x @ y.transpose(-1, -2)

    def _infonce_loss(self, anchor_repr, goal_repr, negative_repr, same_segment):
        # Expected InfoNCE: CRL samples one goal j ~ p_i per anchor; this averages over p_i.
        B, T, _ = anchor_repr.shape

        # l_ij = energy(phi(anchor_i), psi(u_j)): positives from the anchor's own window,
        # negatives from the buffer
        pos_logits = self.energy(anchor_repr, goal_repr).reshape(B * T, T)  # (B*T, T)
        neg_logits = self.energy(anchor_repr.reshape(B * T, -1), negative_repr)  # (B*T, K)

        # p_ij = gamma^(t_j - t_i) over future frames of the same episode, sums to 1 per anchor.
        # With anchor="obs" the last frame before a done has no same-episode goal, so its row is
        # all zero; the clamp gives that anchor no positive term instead of 0/0
        pos_weight = self.weight_matrix * same_segment
        pos_weight = pos_weight / pos_weight.sum(-1, keepdim=True).clamp(min=1e-8)

        # log S_i = log sum_k exp(l_ik)
        log_s = torch.logsumexp(neg_logits, -1, keepdim=True)

        # L_i = sum_j p_ij * -log(e^l_ij / (e^l_ij + S_i)) = sum_j p_ij * softplus(log S_i - l_ij)
        infonce = (pos_weight * F.softplus(log_s - pos_logits)).sum(-1).mean()
        # InfoNCE ignores a per-row shift of the logits; this pins their level
        penalty = self.logsumexp_penalty * (log_s**2).mean()

        parts = dict(
            pos_logits=pos_logits,
            neg_logits=neg_logits,
            pos_weight=pos_weight,
            infonce=infonce,
            penalty=penalty,
        )
        return infonce + penalty, parts

    @torch.no_grad()
    def _infonce_stats(self, parts):
        pos_logits, neg_logits = parts["pos_logits"], parts["neg_logits"]
        rows = torch.arange(pos_logits.shape[0], device=pos_logits.device)
        lag0 = pos_logits[rows, rows % pos_logits.shape[1]]  # row b*T + t, column t
        pos_vs_neg_acc = (lag0 > neg_logits.max(-1).values).float().mean()
        return {
            "wm/loss/infonce_term": parts["infonce"].item(),
            "wm/loss/logsumexp_penalty_term": parts["penalty"].item(),
            "wm/stats/pos_logit": (parts["pos_weight"] * pos_logits).sum(-1).mean().item(),
            "wm/stats/neg_logit": neg_logits.mean().item(),
            # does the lag-0 positive beat every buffer negative? (CRL's categorical accuracy)
            "wm/stats/pos_vs_neg_acc": pos_vs_neg_acc.item(),
        }

    def forward(self, context):
        goals = context["obs_embeddings"][:, 1:]
        if self.anchor == "obs":
            # (u_t, a_t) against u_{t+1}, u_{t+2}, ...: anchor rows are steps 0..T-1, goal columns
            # 1..T. a_t is the action taken at o_t, the one leading to o_{t+1}
            anchors = torch.cat([context["obs_embeddings"][:, :-1], context["actions"][:, :-1]], -1)
            anchor_offset = 0
        elif self.anchor == "posterior":
            # (q(z_t | h_t, u_t), a_t) against u_{t+1}, ...: categorical probabilities, flattened, to
            # avoid the one-hot sample. z_t is computed from u_t, so goals start one step ahead,
            # same alignment as "obs". The gradient reaches h_t (and the RSSM) through q
            post_probs = context["post_logits"].softmax(-1).flatten(2)  # (B, T+1, latent_size)
            anchors = torch.cat([post_probs[:, :-1], context["actions"][:, :-1]], -1)
            anchor_offset = 0
        elif self.rssm:
            # h_t is computed before x_t, so h_t against u_t is already a prediction
            anchors = context["seq_states"][:, 1:]
            anchor_offset = 1
        else:
            # STORM's h_t depends on z_t, so it needs the one-step shift
            anchors = context["seq_states"][:, :-1]
            anchor_offset = 1  # segments taken from the goal index, as before

        B, T, _ = anchors.shape
        anchor_repr = self.anchor_proj(anchors)
        goal_repr = self.goal_proj(goals)

        # dones[t] = 1 means obs t+1 starts a new episode, so segment t = number of dones before t;
        # each side uses its own time index
        dones = context["dones"].float()
        segments = dones.cumsum(1) - dones
        anchor_segments = segments[:, anchor_offset : anchor_offset + T]
        same_segment = anchor_segments[:, :, None] == segments[:, None, 1 : 1 + T]
        same_segment = same_segment.reshape(B * T, T)

        wm = context["world_model"]
        negative_obs = context["buffer"].sample_as_tensors(
            anchors.device, self.num_negatives, seq_len=1
        )[0]
        negative_repr = self.goal_proj(wm.encoder(transform_obs(negative_obs, wm.is_image)))

        if self.loss_type == "infonce":
            loss, parts = self._infonce_loss(anchor_repr, goal_repr, negative_repr, same_segment)
            extra_stats = self._infonce_stats(parts)
        else:
            pos_dot_prod = torch.einsum("btd,bsd->bts", anchor_repr, goal_repr).reshape(B * T, T)
            neg_dot_prod = torch.einsum("btd,cd->btc", anchor_repr, negative_repr).reshape(
                B * T, -1
            )

            neg_target = torch.zeros_like(neg_dot_prod)

            loss = F.binary_cross_entropy_with_logits(
                pos_dot_prod,
                self.pos_target_matrix,
                self.weight_matrix * same_segment,
                reduction="mean",
            ) * 2 + F.binary_cross_entropy_with_logits(neg_dot_prod, neg_target, reduction="mean")
            extra_stats = {}

        stats = {
            "wm/loss/contrastive_loss": loss.item(),
            # keys kept from earlier runs so TensorBoard curves stay comparable
            "wm/stats/state_embed_norm": anchor_repr.norm(dim=-1).mean().item(),
            "wm/stats/obs_embed_norm": goal_repr.norm(dim=-1).mean().item(),
        } | extra_stats
        if self.log_name is not None:
            stats = {
                "/".join(k.split("/", 2)[:2] + [self.log_name, k.split("/", 2)[2]]): v
                for k, v in stats.items()
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
