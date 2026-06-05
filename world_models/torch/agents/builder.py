from world_models.buffers.buffers import PerEnvBuffer
from world_models.torch.agents.actor_critic import Actor, Critic
from world_models.torch.agents.world_model import WorldModel
from world_models.torch.common.decoders import ConvDecoder, VectorDecoder
from world_models.torch.common.encoders import ConvEncoder, MultiEncoder, VectorEncoder
from world_models.torch.common.heads import BernoulliHead, CategoricalHead, TwoHotHead
from world_models.torch.common.models import Posterior, Prior
from world_models.torch.common.objectives import CompoundObjective, ReconstructionObjective
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
            cfg.act,
        )
    elif cfg.type == "vector":
        raise NotImplementedError
    else:
        raise NotImplementedError


def build_decoder(cfg, encoder_output_size):
    if cfg.type == "conv":
        return ConvDecoder(
            cfg.in_dim,
            encoder_output_size,
            cfg.filter_base,
            cfg.num_convs,
            cfg.kernel_size,
            cfg.image_channels,
            cfg.act,
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
            cfg.act,
            cfg.use_block_linear,
        )
    elif cfg.type == "mamba":
        return MambaSequenceModel(
            latent_size,
            action_dim,
            cfg.d_model,
            cfg.act,
            cfg.n_layers,
            cfg.d_state,
            cfg.d_conv,
            cfg.expand,
            cfg.headdim,
        )
    elif cfg.type == "transformer":
        return TransformerSequenceModel(
            latent_size,
            action_dim,
            cfg.d_model,
            cfg.n_layers,
            cfg.num_heads,
            cfg.max_seq_len,
            cfg.expand,
            cfg.dropout_p,
            cfg.act,
            cfg.use_sdpa,
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
        cfg.n_layers,
        cfg.hidden_dim,
        cfg.act,
        includes_sequence_state,
    )


def build_prior(cfg, d_model, latent_size):
    dist_head = CategoricalHead(cfg.num_categories, cfg.num_codes, cfg.unimix_prob)
    return Prior(d_model, latent_size, dist_head, cfg.n_layers, cfg.hidden_dim, cfg.act)


def build_heads(cfg, in_dim):
    reward_predictor = TwoHotHead(
        in_dim, cfg.num_bins, cfg.bin_low, cfg.bin_high, cfg.hidden_dim, cfg.n_layers, cfg.act
    )
    continue_predictor = BernoulliHead(in_dim, 1, cfg.hidden_dim, cfg.n_layers, cfg.act)
    return reward_predictor, continue_predictor


def build_objective(cfg, obs_type):
    objectives = []
    for obj_cfg in cfg.objectives:
        if obj_cfg.name == "reconstruction":
            objectives.append(ReconstructionObjective(obj_cfg.weight, obs_type))
        elif obj_cfg.name == "infonce":
            raise NotImplementedError
        elif obj_cfg.name == "sigreg":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown objective: {obj_cfg.name}")

    if len(objectives) == 1:
        return objectives[0]
    return CompoundObjective(objectives)


def build_world_model(cfg, action_dim):
    latent_size = cfg.latent.num_categories * cfg.latent.num_codes
    encoder = build_encoder(cfg.encoder)
    sequence_model = build_sequence_model(cfg.sequence_model, latent_size, action_dim)
    d_model = sequence_model.output_dim
    embed_dim = encoder.output_dim
    includes_h = isinstance(sequence_model, RSSM)
    posterior = build_posterior(cfg.posterior, embed_dim, latent_size, d_model, includes_h)
    prior = build_prior(cfg.prior, d_model, latent_size)
    objective = build_objective(cfg, cfg.obs_type)
    head_in_dim = latent_size + d_model if cfg.use_combined_state else d_model
    reward_predictor, continue_predictor = build_heads(cfg.heads, head_in_dim)
    decoder = (
        build_decoder(cfg.decoder, encoder.output_size) if objective.requires_decoder else None
    )
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
        cfg.free_nats,
        cfg.use_combined_state,
        decoder,
    )


def build_actor(cfg, input_dim, action_dim, action_type):
    return Actor(
        input_dim,
        action_dim,
        cfg.hidden_dim,
        cfg.num_hiddens,
        action_type,
        cfg.act,
        cfg.actor_unimix,
        cfg.log_std_min,
        cfg.log_std_max,
        cfg.std_min,
    )


def build_critic(cfg, input_dim):
    return Critic(
        input_dim,
        cfg.hidden_dim,
        cfg.num_hiddens,
        cfg.num_bins,
        cfg.bin_low,
        cfg.bin_high,
        cfg.act,
    )


def build_buffer(cfg):
    if cfg.prioritized:
        # need to add sumtree weight in somewhere, probably owned by the agent
        # since there should be separate weights for world model and for actor-critic
        raise NotImplementedError
    return PerEnvBuffer(cfg.num_envs, cfg.buffer_size)  # buffer has lazy init for first sample
