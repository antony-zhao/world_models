from torch import optim

from world_models.buffers.buffers import PerEnvBuffer
from world_models.torch.agents.actor_critic import Actor, Critic
from world_models.torch.agents.world_model import WorldModel
from world_models.torch.common.encoders import ConvEncoder
from world_models.torch.common.heads import BernoulliHead, CategoricalHead, TwoHotHead
from world_models.torch.common.models import Posterior, Prior
from world_models.torch.common.objectives import (
    ActionObjective,
    CDPObjective,
    CompoundObjective,
    ContrastiveTrajectoryObjective,
    ReconstructionObjective,
    ValueObjective,
)
from world_models.torch.common.sequence_models import (
    RSSM,
    MambaSequenceModel,
    TransformerSequenceModel,
)


def build_encoder(cfg):
    if cfg.type == "conv":
        return ConvEncoder(
            cfg.filter_base,
            cfg.num_convs,
            cfg.kernel_size,
            cfg.image_channels,
            cfg.obs_shape[-1],
        )
    elif cfg.type == "vector":
        raise NotImplementedError
    else:
        raise NotImplementedError


def build_sequence_model(cfg, latent_size, action_dim):
    if cfg.type == "rssm":
        return RSSM(
            latent_size,
            action_dim,
            cfg.d_model,
            cfg.hidden_dim,
            cfg.n_layers,
            cfg.use_block_linear,
            norm=cfg.norm,
        )
    elif cfg.type == "mamba":
        return MambaSequenceModel(
            latent_size,
            action_dim,
            cfg.d_model,
            cfg.n_layers,
            cfg.d_state,
            cfg.d_conv,
            cfg.expand,
            cfg.headdim,
            norm=cfg.norm,
        )
    elif cfg.type == "transformer":
        return TransformerSequenceModel(
            latent_size,
            action_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            n_layers=cfg.n_layers,
            max_seq_len=cfg.max_seq_len,
            expand=cfg.expand,
            dropout_p=cfg.dropout_p,
            use_sdpa=cfg.use_sdpa,
            norm=cfg.norm,
        )
    else:
        raise NotImplementedError


def build_posterior(cfg, embed_dim, latent_size, d_model, includes_sequence_state):
    dist_head = CategoricalHead(cfg.num_categories, cfg.num_codes, cfg.unimix_prob)
    return Posterior(
        embed_dim,
        latent_size,
        dist_head,
        d_model,
        cfg.num_hiddens,
        cfg.hidden_dim,
        cfg.norm,
        includes_sequence_state,
    )


def build_prior(cfg, d_model, latent_size):
    dist_head = CategoricalHead(cfg.num_categories, cfg.num_codes, cfg.unimix_prob)
    return Prior(
        d_model,
        latent_size,
        dist_head,
        cfg.num_hiddens,
        cfg.hidden_dim,
    )


def build_heads(cfg, in_dim):
    reward_predictor = TwoHotHead(
        in_dim,
        cfg.num_bins,
        cfg.bin_low,
        cfg.bin_high,
        cfg.hidden_dim,
        cfg.num_hiddens,
    )
    continue_predictor = BernoulliHead(
        in_dim,
        1,
        cfg.hidden_dim,
        cfg.num_hiddens,
    )
    reward_predictor.zero_init()
    return reward_predictor, continue_predictor


def build_objective(cfg):
    objectives = []
    for obj_cfg in cfg.objectives:
        if obj_cfg.name == "reconstruction":
            objectives.append(ReconstructionObjective(obj_cfg))
        elif obj_cfg.name == "action_prediction":
            objectives.append(ActionObjective(obj_cfg))
        elif obj_cfg.name == "value_prediction":
            objectives.append(ValueObjective(obj_cfg))
        elif obj_cfg.name == "cdp":
            objectives.append(CDPObjective(obj_cfg))
        elif obj_cfg.name == "contrastive":
            objectives.append(ContrastiveTrajectoryObjective(obj_cfg))
        else:
            raise ValueError(f"Unknown objective: {obj_cfg.name}")

    if len(objectives) == 1:
        return objectives[0]
    return CompoundObjective(objectives)


def build_wm_optimizer(cfg, world_model):
    default_lr = cfg.wm_lr
    encoder_lr = cfg.get("encoder_lr", default_lr)
    seq_lr = cfg.get("sequence_model_lr", default_lr)
    obj_lr = cfg.get("objective_lr", default_lr)

    encoder_params = list(world_model.encoder.parameters())
    seq_params = list(world_model.sequence_model.parameters())
    obj_params = list(world_model.objective.parameters())

    seen = set(id(p) for p in encoder_params + seq_params + obj_params)
    other_params = [p for p in world_model.parameters() if id(p) not in seen]

    return optim.Adam(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": seq_params, "lr": seq_lr},
            {"params": obj_params, "lr": obj_lr},
            {"params": other_params, "lr": default_lr},
        ],
        eps=cfg.wm_eps,
    )


def build_world_model(cfg, action_dim) -> WorldModel:
    latent_size = cfg.latent.num_categories * cfg.latent.num_codes
    encoder = build_encoder(cfg.encoder)
    sequence_model = build_sequence_model(cfg.sequence_model, latent_size, action_dim)
    d_model = cfg.sequence_model.d_model
    embed_dim = encoder.output_dim
    includes_h = isinstance(sequence_model, RSSM)
    posterior = build_posterior(cfg.posterior, embed_dim, latent_size, d_model, includes_h)
    prior = build_prior(cfg.prior, d_model, latent_size)
    objective = build_objective(cfg)
    head_in_dim = latent_size + d_model if cfg.use_combined_state else d_model
    reward_predictor, continue_predictor = build_heads(cfg.heads, head_in_dim)
    return WorldModel(
        encoder,
        sequence_model,
        posterior,
        prior,
        continue_predictor,
        reward_predictor,
        objective,
        cfg.obs_type,
        cfg.obj_coef,
        cfg.dyn_coef,
        cfg.repr_coef,
        cfg.head_coef,
        cfg.reward_coef,
        cfg.continue_coef,
        cfg.free_nats,
        cfg.use_combined_state,
    )


def build_actor(cfg, input_dim, action_dim, action_type) -> Actor:
    return Actor(
        input_dim,
        action_dim,
        cfg.hidden_dim,
        cfg.num_hiddens,
        action_type,
        cfg.actor_unimix,
        cfg.log_std_min,
        cfg.log_std_max,
        cfg.std_min,
    )


def build_critic(cfg, input_dim) -> Critic:
    return Critic(
        input_dim,
        cfg.hidden_dim,
        cfg.num_hiddens,
        cfg.num_bins,
        cfg.bin_low,
        cfg.bin_high,
    )


def build_buffer(cfg):
    if cfg.prioritized:
        # need to add sumtree weight in somewhere, probably owned by the agent
        # since there should be separate weights for world model and for actor-critic
        raise NotImplementedError
    return PerEnvBuffer(
        cfg.num_envs, buffer_size=cfg.buffer_size
    )  # buffer has lazy init for first sample
