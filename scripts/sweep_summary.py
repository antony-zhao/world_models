"""One-line-per-run summary of a sweep folder, for spotting representation collapse.

    python scripts/sweep_summary.py runs/atari100k/lejepa_storm_full

Finds every run (a folder with config.yaml) under the given roots and prints the final eval
(mean of the last 3 evals, since single 10-game evals are noisy), STORM's published score, and
the SIGReg / view-loss / entropy trajectory. SIGReg's absolute level differs between games, so
read sig_ratio (final / 5k): healthy runs keep falling (Breakout seed 1: ~0.4), the collapsed
Breakout seed 2 went flat (~0.7) with an unusually low view loss.
"""

import argparse
import glob
import os

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# STORM (Zhang et al., 2023) Atari100k means over 5 seeds, quoted from memory: check against
# the paper's table before putting them in a figure.
STORM = {
    "alien": 984, "amidar": 205, "assault": 801, "asterix": 1028, "bank_heist": 641,
    "battle_zone": 13540, "boxing": 80, "breakout": 16, "chopper_command": 1888,
    "crazy_climber": 66776, "demon_attack": 165, "freeway": 34, "frostbite": 1316,
    "gopher": 8240, "hero": 11044, "jamesbond": 509, "kangaroo": 4208, "krull": 8413,
    "kung_fu_master": 26182, "ms_pacman": 2673, "pong": 11, "private_eye": 7781,
    "qbert": 4522, "road_runner": 17564, "seaquest": 525, "up_n_down": 7985,
}


def scalars(run):
    files = sorted(glob.glob(os.path.join(run, "**", "events.out*"), recursive=True))
    if not files:
        return {}
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return {t: {e.step: e.value for e in ea.Scalars(t)} for t in ea.Tags()["scalars"]}


def at(series, step, tol=600):
    if not series:
        return float("nan")
    k = min(series, key=lambda s: abs(s - step))
    return series[k] if abs(k - step) <= tol else float("nan")


def last(series):
    return series[max(series)] if series else float("nan")


def main(roots):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    runs = sorted(
        {os.path.dirname(p) for r in roots for p in glob.glob(os.path.join(r, "**", "config.yaml"), recursive=True)}
    )
    hdr = (f"{'game':16s} {'seed':>4s} {'step':>7s} {'eval':>8s} {'STORM':>7s} {'ratio':>6s} | "
           f"{'sig@5k':>7s} {'sig@20k':>7s} {'sig_end':>7s} {'sig_ratio':>9s} | "
           f"{'view_end':>8s} {'entropy':>7s} {'cont':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for run in runs:
        try:
            cfg = OmegaConf.load(os.path.join(run, "config.yaml"))
            s = scalars(run)
        except Exception as e:  # unreadable or still-starting run
            print(f"{run}: skipped ({type(e).__name__})")
            continue
        ev = s.get("eval/episode_return", {})
        if not ev:
            continue
        game = cfg.env.game
        tail = [ev[k] for k in sorted(ev)[-3:]]
        final = sum(tail) / len(tail)
        storm = STORM.get(game, float("nan"))
        sig = s.get("wm/loss/sigreg_loss", {})
        sig5, sig_end = at(sig, 5000), last(sig)
        print(
            f"{game:16s} {cfg.seed:4d} {max(ev):7d} {final:8.1f} {storm:7.0f} {final / storm:6.2f} | "
            f"{sig5:7.2f} {at(sig, 20000):7.2f} {sig_end:7.2f} {sig_end / sig5:9.2f} | "
            f"{last(s.get('wm/loss/view_loss', {})):8.4f} {last(s.get('ac/loss/entropy', {})):7.3f} "
            f"{last(s.get('wm/loss/continue_loss', {})):8.2g}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", help="sweep folders (searched recursively for runs)")
    main(p.parse_args().roots)
