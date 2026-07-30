"""Config preprocessing: resolve string activations to classes, propagate shared
values from top-level sections into per-module configs.

Runs once after config load + CLI overrides, before any modules are built.
"""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

ACTIVATION_MAP = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def get_activation(name_or_cls):
    """Convert an activation name (string) to an nn.Module class."""
    if isinstance(name_or_cls, str):
        return ACTIVATION_MAP[name_or_cls.lower()]
    return name_or_cls


def _set_if_missing(node, key, value):
    """Set node[key] = value only if the key is not already present."""
    if key not in node:
        node[key] = value


def _get_nested(cfg, path):
    """Safely traverse a nested path; return None if any step is missing."""
    node = cfg
    for k in path:
        if not hasattr(node, "__contains__") or k not in node:
            return None
        node = node[k]
    return node


def preprocess_config(cfg: DictConfig) -> DictConfig:
    """Fill in cross-field defaults from shared sections.

    After this runs, every per-module config section has all fields needed
    for construction. Encoder-derived dims (encoder_output_dim, latent_dim,
    etc.) are NOT filled here — those are injected by WorldModel.__init__
    since they depend on constructed modules.
    """
    OmegaConf.set_struct(cfg, False)

    # ---- Propagate latent dims to posterior/prior ----
    for section_name in ["posterior", "prior"]:
        section = cfg.wm[section_name]
        _set_if_missing(section, "num_categories", cfg.wm.latent.num_categories)
        _set_if_missing(section, "num_codes", cfg.wm.latent.num_codes)

    cfg.wm.latent_size = cfg.wm.latent.num_categories * cfg.wm.latent.num_codes

    # ---- Propagate hidden_dim from model.hidden_dim ----
    hidden_dim_targets = [
        ("wm", "posterior"),
        ("wm", "prior"),
        ("wm", "heads"),
        ("actor",),
        ("critic",),
    ]
    for path in hidden_dim_targets:
        node = _get_nested(cfg, path)
        if node is not None:
            _set_if_missing(node, "hidden_dim", cfg.model.hidden_dim)

    # ---- Propagate args to each objective in the list ----
    for obj_cfg in cfg.wm.objectives:
        _set_if_missing(obj_cfg, "hidden_dim", cfg.model.hidden_dim)
        _set_if_missing(obj_cfg, "act", cfg.model.act)
        _set_if_missing(obj_cfg, "latent_size", cfg.wm.latent_size)
        _set_if_missing(obj_cfg, "state_dim", cfg.wm.sequence_model.d_model)
        _set_if_missing(obj_cfg, "use_combined_state", cfg.wm.use_combined_state)

    # ---- Propagate activation from model.act ----
    act_targets = [
        ("wm", "encoder"),
        ("wm", "sequence_model"),
        ("wm", "posterior"),
        ("wm", "prior"),
        ("wm", "heads"),
        ("actor",),
        ("critic",),
    ]
    for path in act_targets:
        node = _get_nested(cfg, path)
        if node is not None:
            _set_if_missing(node, "act", cfg.model.act)

    # ---- Propagate activation to each objective in the list ----
    for obj_cfg in cfg.wm.objectives:
        _set_if_missing(obj_cfg, "act", cfg.model.act)

    # ---- Propagate symlog bins to heads and critic ----
    for section in [cfg.wm.heads, cfg.critic]:
        _set_if_missing(section, "num_bins", cfg.symlog.num_bins)
        _set_if_missing(section, "bin_low", cfg.symlog.low)
        _set_if_missing(section, "bin_high", cfg.symlog.high)

    # ---- Set encoder image_channels and obs_shape from cfg.obs_shape ----
    if cfg.obs_shape is not None:
        obs_shape = list(cfg.obs_shape)
        if cfg.wm.encoder.type == "conv":
            _set_if_missing(cfg.wm.encoder, "image_channels", obs_shape[0])
            _set_if_missing(cfg.wm.encoder, "obs_shape", obs_shape)

    if cfg.wm.encoder.type == "conv":
        filter = cfg.wm.encoder.filter_base * 2 ** (cfg.wm.encoder.num_convs - 1)
        spatial = cfg.obs_shape[1] // (2**cfg.wm.encoder.num_convs)
        cfg.wm.embedding_dim = filter * spatial * spatial
        cfg.wm.encoder_conv_dim = (filter, spatial, spatial)

    # ---- Set posterior/prior norm_type defaults ----
    _set_if_missing(cfg.wm.posterior, "norm_type", "layer")
    _set_if_missing(cfg.wm.prior, "norm_type", "layer")

    return cfg
