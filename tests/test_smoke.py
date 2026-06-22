"""Smoke test: instantiate the full Agent via Agent(cfg), feed the buffer with
fake transitions, and exercise the entire train_step pipeline end-to-end.

This is the integration test that should run cleanly before launching any real
training. It exercises:

  1. Agent construction via builders (every component instantiated).
  2. agent.act() with a fake observation (online inference path).
  3. agent.add_transition() to populate the buffer.
  4. agent.train_step() pre-warmup (WM-only training).
  5. agent.train_step() post-warmup (WM + AC training, including imagine).
  6. agent.act() after some training (sanity check that online state survives).

Run from the project root:
    python tests/test_smoke.py
"""

from __future__ import annotations

import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from world_models.torch.agents.agent import Agent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B_envs = 2  # vectorized env count for the smoke test


# --------------------------------------------------------------------- config


def make_cfg(sequence_model_type: str) -> SimpleNamespace:
    """Minimal but complete cfg tree for Agent."""
    num_cat, num_codes = 8, 8
    d_model = 64
    action_dim = 4
    action_type = "discrete"
    obs_shape = (3, 64, 64)
    num_envs = B_envs

    # ----- WM sub-configs
    encoder = SimpleNamespace(
        type="conv",
        filter_base=8,
        num_convs=4,
        kernel_size=4,
        image_channels=obs_shape[0],
        obs_shape=obs_shape,
        act=nn.SiLU,
    )
    decoder = SimpleNamespace(
        type="conv",
        in_dim=num_cat * num_codes,
        filter_base=8,
        num_convs=4,
        kernel_size=4,
        image_channels=obs_shape[0],
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
        num_bins=51, bin_low=-20.0, bin_high=20.0, hidden_dim=64, n_layers=2, act=nn.SiLU
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
        objectives=[SimpleNamespace(name="reconstruction", weight=1.0)],
        obj_coef=1.0,
        dyn_coef=0.5,
        repr_coef=0.1,
        head_coef=1.0,
        free_nats=1.0,
    )

    # ----- AC sub-configs
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
    buffer = SimpleNamespace(
        num_envs=num_envs,
        buffer_size=1024,
        prioritized=False,
    )
    train = SimpleNamespace(
        critic_tau=0.02,
        horizon=4,
        gamma=0.99,
        lambda_=0.95,
        entropy_coef=3e-4,
        wm_batch_size=4,
        wm_seq_len=4,
        ac_batch_size=4,
        ac_seq_len=4,
        train_ac_after=20,
        returns_percentile_lo=0.05,
        returns_percentile_hi=0.95,
        returns_ema_tau=0.02,
        context_window=4,
    )
    optim_cfg = SimpleNamespace(
        wm_lr=1e-4,
        ac_lr=3e-5,
        eps=1e-8,
        wm_clip=1000.0,
        ac_clip=100.0,
    )

    return SimpleNamespace(
        action_dim=action_dim,
        action_type=action_type,
        num_envs=num_envs,
        device=DEVICE,
        obs_shape=obs_shape,
        wm=wm,
        actor=actor,
        critic=critic,
        buffer=buffer,
        train=train,
        optim=optim_cfg,
    )


# ------------------------------------------------------------------- helpers


def fake_obs(cfg):
    """Random uint8 image obs for env interaction."""
    return np.random.randint(0, 255, (cfg.num_envs, *cfg.obs_shape), dtype=np.uint8)


def populate_buffer(agent, cfg, n_steps: int):
    """Add fake transitions to the buffer by simulating env interaction."""
    for _ in range(n_steps):
        obs = fake_obs(cfg).astype(np.float32)
        action = agent.act(obs)
        reward = np.random.randn(cfg.num_envs).astype(np.float32)
        # mark one transition as terminated to exercise the reset path
        terminated = np.zeros(cfg.num_envs, dtype=bool)
        if np.random.rand() < 0.1:
            terminated[0] = True
        truncated = np.zeros(cfg.num_envs, dtype=bool)
        agent.add_transition(obs, action, reward, terminated, truncated)


# --------------------------------------------------------------------- tests


def test_construction(cfg):
    """Build the agent from cfg. Catches arg-order, missing-field, dtype issues."""
    print("\n--- test_construction ---")
    agent = Agent(cfg).to(DEVICE)
    wm_params = sum(p.numel() for p in agent.world_model.parameters())
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"  WM:     {wm_params:>10,} params")
    print(f"  Actor:  {actor_params:>10,} params")
    print(f"  Critic: {critic_params:>10,} params")
    print(f"  is_rssm: {agent.is_rssm}")
    return agent


def test_act_initial(agent, cfg):
    """Online act on fresh agent (no buffer data, just placeholders)."""
    print("\n--- test_act_initial ---")
    obs = fake_obs(cfg).astype(np.float32)
    action = agent.act(obs)
    print(f"  action shape: {action.shape}, dtype: {action.dtype}")
    print(f"  action values: {action.tolist()}")
    assert action.shape == (cfg.num_envs,), f"expected ({cfg.num_envs},) got {action.shape}"


def test_populate_buffer(agent, cfg, n_steps):
    """Populate buffer by repeatedly acting and adding transitions."""
    print(f"\n--- test_populate_buffer (n_steps={n_steps}) ---")
    populate_buffer(agent, cfg, n_steps)
    print(f"  buffer.size: {agent.buffer.size}")
    print(f"  agent.step_count: {agent.step_count}")
    return agent.buffer.size


def test_train_step_wm_only(agent, cfg, n_iters=3):
    """Pre-warmup: only WM training. Should always succeed."""
    print(f"\n--- test_train_step_wm_only (n_iters={n_iters}) ---")
    assert agent.step_count <= agent.train_ac_after, (
        f"step_count={agent.step_count} already exceeds train_ac_after={agent.train_ac_after}, "
        "AC will run too — adjust the test or buffer-populate count."
    )
    losses = []
    for i in range(n_iters):
        loss, metrics = agent.train_step()
        losses.append(loss.item())
        print(f"  iter {i}: wm_loss={loss.item():.4f}, metric_keys={sorted(metrics.keys())}")
        assert torch.isfinite(loss), f"non-finite loss at iter {i}: {loss}"
    print(f"  loss progression: {losses}")


def test_train_step_full(agent, cfg, n_iters=3):
    """Post-warmup: WM + AC. This is where RSSM-imagine bugs surface."""
    print(f"\n--- test_train_step_full (n_iters={n_iters}) ---")
    # Make sure we're past warmup
    while agent.step_count <= agent.train_ac_after:
        populate_buffer(agent, cfg, 5)
    print(f"  step_count={agent.step_count}, train_ac_after={agent.train_ac_after}")

    for i in range(n_iters):
        loss, metrics = agent.train_step()
        actor_loss = metrics.get("ac/loss/actor")
        critic_loss = metrics.get("ac/loss/critic")
        entropy = metrics.get("ac/loss/entropy")
        adv_mean = metrics.get("ac/stats/advantage_mean")
        print(
            f"  iter {i}: wm={loss.item():.3f}  "
            f"actor={actor_loss:.3f}  critic={critic_loss:.3f}  "
            f"H={entropy:.3f}  adv_mean={adv_mean:.3f}"
        )
        assert torch.isfinite(loss), f"non-finite wm_loss at iter {i}"
        assert "ac/loss/actor" in metrics, "AC metrics missing — AC didn't run"


def test_act_after_training(agent, cfg):
    """Online act after some training — make sure state isn't corrupted."""
    print("\n--- test_act_after_training ---")
    obs = fake_obs(cfg).astype(np.float32)
    action = agent.act(obs)
    print(f"  action shape: {action.shape}, values: {action.tolist()}")


def test_parameter_updates(agent, cfg):
    """Confirm parameters actually change across train_steps."""
    print("\n--- test_parameter_updates ---")
    wm_params_before = [p.clone() for p in agent.world_model.parameters() if p.requires_grad]
    populate_buffer(agent, cfg, 10)
    for _ in range(3):
        agent.train_step()
    wm_params_after = list(agent.world_model.parameters())
    changed = sum(
        1
        for b, a in zip(wm_params_before, wm_params_after)
        if a.requires_grad and not torch.allclose(b, a)
    )
    print(f"  WM params changed: {changed} / {len(wm_params_before)}")


# ------------------------------------------------------------------ main


def run_for(seq_type: str):
    print(f"\n{'=' * 60}\nAGENT SMOKE TEST :: {seq_type.upper()}\n{'=' * 60}")
    cfg = make_cfg(seq_type)
    torch.manual_seed(0)
    np.random.seed(0)

    agent = test_construction(cfg)
    test_act_initial(agent, cfg)
    test_populate_buffer(agent, cfg, n_steps=10)  # below train_ac_after
    test_train_step_wm_only(agent, cfg, n_iters=3)
    test_train_step_full(agent, cfg, n_iters=3)
    test_act_after_training(agent, cfg)
    test_parameter_updates(agent, cfg)
    print(f"\n{seq_type}: all smoke tests passed.")


def main():
    print(f"Device: {DEVICE}")
    archs = ["transformer", "rssm"]
    if DEVICE == "cuda":
        archs.append("mamba")
    results = {}
    for sm in archs:
        try:
            run_for(sm)
            results[sm] = "passed"
        except Exception as e:
            print(f"\n{sm}: FAILED with {type(e).__name__}: {e}")
            traceback.print_exc()
            results[sm] = f"FAILED ({type(e).__name__})"

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for sm, r in results.items():
        print(f"  {sm:>12s}: {r}")


if __name__ == "__main__":
    main()
