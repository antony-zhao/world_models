"""Smoke test: instantiate WM + Actor + Critic via builders, run one of each
forward+backward, and report what works/fails. Catches integration bugs
(arg-order mismatches, missing fields, dtype/device issues) before the agent
training loop adds more layers.

Run from the project root:
    python tests/test_smoke.py

This uses SimpleNamespace as a stand-in for OmegaConf — drop in your real cfg
loader once configs land.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from world_models.torch.agents.builder import build_actor, build_critic, build_world_model
from world_models.torch.common.sequence_models import RSSM
from world_models.torch.common.utils import make_state

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- config


def make_cfg(sequence_model_type: str = "transformer") -> SimpleNamespace:
    """Build a tiny-but-valid cfg tree. Small dims for fast iteration."""
    num_cat, num_codes = 8, 8
    latent_size = num_cat * num_codes  # 64
    d_model = 64
    action_dim = 4
    action_type = "discrete"

    obs_shape = (3, 64, 64)
    image_channels = obs_shape[0]
    input_size = obs_shape[-1]

    encoder = SimpleNamespace(
        type="conv",
        filter_base=8,
        num_convs=4,
        kernel_size=4,
        image_channels=image_channels,
        obs_shape=obs_shape,
        act=nn.SiLU,
    )

    decoder = SimpleNamespace(
        type="conv",
        in_dim=latent_size,  # STORM-style: decoder takes z_t only
        filter_base=8,
        num_convs=4,
        kernel_size=4,
        image_channels=image_channels,
        act=nn.SiLU,
    )

    if sequence_model_type == "rssm":
        sequence_model = SimpleNamespace(
            type="rssm",
            d_model=d_model,
            hidden_dim=d_model,
            n_layers=2,
            act=nn.SiLU,
            use_block_linear=True,
        )
    elif sequence_model_type == "mamba":
        sequence_model = SimpleNamespace(
            type="mamba",
            d_model=d_model,
            n_layers=2,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=16,
            act=nn.SiLU,
        )
    elif sequence_model_type == "transformer":
        sequence_model = SimpleNamespace(
            type="transformer",
            d_model=d_model,
            n_layers=2,
            num_heads=4,
            max_seq_len=128,
            expand=2,
            dropout_p=0.0,
            use_sdpa=True,
            act=nn.SiLU,
        )
    else:
        raise ValueError(sequence_model_type)

    posterior = SimpleNamespace(
        n_layers=1,
        hidden_dim=64,
        num_categories=num_cat,
        num_codes=num_codes,
        unimix_prob=0.01,
        act=nn.SiLU,
    )
    prior = SimpleNamespace(
        n_layers=1,
        hidden_dim=64,
        num_categories=num_cat,
        num_codes=num_codes,
        unimix_prob=0.01,
        act=nn.SiLU,
    )
    heads = SimpleNamespace(
        num_bins=51,
        bin_low=-20.0,
        bin_high=20.0,
        hidden_dim=64,
        n_layers=2,
        act=nn.SiLU,
    )
    objectives = SimpleNamespace(
        objectives=[SimpleNamespace(name="reconstruction", weight=1.0)],
    )

    wm = SimpleNamespace(
        obs_type="image",
        use_combined_state=False,
        latent=SimpleNamespace(num_categories=num_cat, num_codes=num_codes),
        encoder=encoder,
        decoder=decoder,
        sequence_model=sequence_model,
        posterior=posterior,
        prior=prior,
        heads=heads,
        objectives=objectives.objectives,  # build_objective reads cfg.objectives as a list
        obj_coef=1.0,
        dyn_coef=0.5,
        repr_coef=0.1,
        head_coef=1.0,
        free_nats=1.0,
    )

    actor = SimpleNamespace(
        hidden_dim=64,
        num_hiddens=2,
        actor_unimix=0.01,
        log_std_min=-5.0,
        log_std_max=2.0,
        std_min=0.1,
        act=nn.SiLU,
    )
    critic = SimpleNamespace(
        hidden_dim=64,
        num_hiddens=2,
        num_bins=51,
        bin_low=-20.0,
        bin_high=20.0,
        act=nn.SiLU,
    )

    return SimpleNamespace(
        action_dim=action_dim,
        action_type=action_type,
        wm=wm,
        actor=actor,
        critic=critic,
        latent_size=latent_size,
        d_model=d_model,
    )


# ---------------------------------------------------------------- helpers


def make_fake_batch(B: int, T: int, action_dim: int, device: str):
    """Fake batch matching the WM's expected interface."""
    obs = (torch.rand(B, T, 3, 64, 64, device=device) * 255).to(torch.uint8)
    # one-hot actions
    action_idx = torch.randint(0, action_dim, (B, T), device=device)
    actions = torch.zeros(B, T, action_dim, device=device)
    actions.scatter_(-1, action_idx.unsqueeze(-1), 1.0)
    rewards = torch.randn(B, T, device=device)
    terminated = torch.zeros(B, T, device=device)
    # mark one done midway through one sequence to exercise reset path (RSSM)
    dones = torch.zeros(B, T, device=device)
    dones[0, T // 2] = 1.0
    return obs.float(), actions, rewards, terminated, dones


def build_actor_and_critic(cfg):
    """Build the real Actor + Critic. Shared across imagine and AC tests so we
    only construct once and so imagine can call .eval()/.train() on a real
    nn.Module."""
    input_dim = cfg.latent_size + cfg.d_model
    actor = build_actor(cfg.actor, input_dim, cfg.action_dim, cfg.action_type).to(DEVICE)
    critic = build_critic(cfg.critic, input_dim).to(DEVICE)
    return actor, critic


# ---------------------------------------------------------------- tests


def test_wm_construction_and_loss(cfg, B: int = 2, T: int = 8):
    """Build WM, run world_model_loss, backward."""
    print("\n--- test_wm_construction_and_loss ---")
    wm = build_world_model(cfg.wm, cfg.action_dim).to(DEVICE)
    print(f"  built {type(wm.sequence_model).__name__}")

    obs, actions, rewards, terminated, dones = make_fake_batch(B, T, cfg.action_dim, DEVICE)

    loss, loss_dict, seq_states_detached, embs_detached = wm.world_model_loss(
        obs, actions, rewards, terminated, dones
    )
    print(f"  loss = {loss.item():.4f}")
    print(f"  loss_dict keys: {sorted(loss_dict.keys())}")
    print(f"  seq_states.shape = {tuple(seq_states_detached.shape)}")
    print(f"  embeddings.shape = {tuple(embs_detached.shape)}")

    loss.backward()
    grad_count = sum(1 for p in wm.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in wm.parameters() if p.requires_grad)
    print(f"  parameters with non-zero grad: {grad_count} / {total}")
    assert loss.item() == loss.item(), "NaN loss"
    return wm


def test_imagine(cfg, wm, actor, B: int = 2, T_context: int = 8, horizon: int = 4):
    """Run imagine end-to-end with a real Actor and check return shapes."""
    print("\n--- test_imagine ---")
    obs, actions, _, _, dones = make_fake_batch(B, T_context, cfg.action_dim, DEVICE)

    latents, actor_seqs, head_seqs, imag_actions = wm.imagine(
        obs, actions, horizon, actor, dones=dones
    )
    print(f"  latents.shape:     {tuple(latents.shape)}")
    print(f"  actor_seqs.shape:  {tuple(actor_seqs.shape)}")
    print(f"  head_seqs.shape:   {tuple(head_seqs.shape)}")
    print(f"  imag_actions.shape:{tuple(imag_actions.shape)}")

    expected_T = horizon + 1
    assert latents.shape[1] == expected_T, f"latents T={latents.shape[1]} != {expected_T}"
    assert actor_seqs.shape[1] == expected_T
    assert head_seqs.shape[1] == expected_T
    assert imag_actions.shape[1] == expected_T
    return latents, actor_seqs, head_seqs, imag_actions


def test_actor_critic(cfg, actor, critic, B: int = 2, T: int = 4):
    """Run forward through Actor/Critic on imagined-like states + backward."""
    print("\n--- test_actor_critic ---")
    print(f"  actor: {sum(p.numel() for p in actor.parameters())} params")
    print(f"  critic: {sum(p.numel() for p in critic.parameters())} params")

    # Fake agent state: [z_t (num_cat * num_codes), seq (d_model)]
    num_cat = cfg.wm.latent.num_categories
    num_codes = cfg.wm.latent.num_codes
    latents = torch.zeros(B, T, num_cat, num_codes, device=DEVICE)
    latents.scatter_(-1, torch.randint(0, num_codes, (B, T, num_cat, 1), device=DEVICE), 1.0)
    seq_states = torch.randn(B, T, cfg.d_model, device=DEVICE)
    states = make_state(latents, seq_states)
    print(f"  state shape: {tuple(states.shape)}")

    # Actor forward
    dist = actor.policy_dist(states)
    action = actor.sample_action(dist)
    formatted = actor.format_action(action)
    print(f"  raw action shape: {tuple(action.shape)}")
    print(f"  formatted (one-hot) action shape: {tuple(formatted.shape)}")
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    print(f"  log_prob.shape: {tuple(log_prob.shape)}, entropy.shape: {tuple(entropy.shape)}")

    # Actor backward through a fake objective (does grad flow?)
    fake_advantage = torch.randn_like(log_prob)
    actor_loss = -(fake_advantage.detach() * log_prob).mean()
    actor_loss.backward()
    actor_grads = sum(
        1 for p in actor.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  actor backward OK, params w/ grad: {actor_grads}")

    # Critic forward and backward
    values, value_logits = critic(states)
    print(f"  values shape: {tuple(values.shape)}, value_logits shape: {tuple(value_logits.shape)}")
    fake_targets = torch.randn_like(values)
    dist = critic.make_dist(value_logits)
    critic_loss = -dist.log_prob(fake_targets).mean()
    critic_loss.backward()
    critic_grads = sum(
        1 for p in critic.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  critic backward OK, params w/ grad: {critic_grads}")


def test_greedy_acting(cfg, actor):
    """policy_fn(det=True) should not crash for discrete or continuous."""
    print("\n--- test_greedy_acting ---")
    input_dim = cfg.latent_size + cfg.d_model
    state = torch.randn(1, input_dim, device=DEVICE)

    sampled = actor.policy_fn(state, det=False)
    greedy = actor.policy_fn(state, det=True)
    print(f"  sampled action: {sampled.item() if sampled.numel() == 1 else sampled.shape}")
    print(f"  greedy  action: {greedy.item() if greedy.numel() == 1 else greedy.shape}")


# ---------------------------------------------------------------- main


def run_for(sequence_model_type: str):
    print(f"\n{'=' * 60}\nSMOKE TEST :: {sequence_model_type.upper()}\n{'=' * 60}")
    cfg = make_cfg(sequence_model_type)
    wm = test_wm_construction_and_loss(cfg)
    actor, critic = build_actor_and_critic(cfg)
    test_imagine(cfg, wm, actor)
    test_actor_critic(cfg, actor, critic)
    test_greedy_acting(cfg, actor)
    print(f"\n{sequence_model_type}: all smoke tests passed.")


def main():
    print(f"Device: {DEVICE}")
    for sm_type in ["transformer", "rssm"]:  # add "mamba" if CUDA available
        try:
            run_for(sm_type)
        except Exception as e:
            print(f"\n{sm_type}: FAILED with {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    if DEVICE == "cuda":
        try:
            run_for("mamba")
        except Exception as e:
            print(f"\nmamba: FAILED with {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
