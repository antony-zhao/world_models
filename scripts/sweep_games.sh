#!/usr/bin/env bash
# Run one config across a diverse set of Atari100k games.
#
# Usage:
#   ./scripts/sweep_games.sh configs/atari100k/rssm/contrastive_dreamer.yaml
#   ./scripts/sweep_games.sh configs/atari100k/rssm/dreamer.yaml --seeds 3 --jobs 2
#   ./scripts/sweep_games.sh <config> --games "pong breakout"
#
# Default games, picked for different dynamics:
#   pong      simple, near-solved sanity check
#   breakout  periodic ball dynamics
#   boxing    18 actions, dense reward
#   freeway   sparse reward / exploration
#   ms_pacman maze, static per-level context
#   krull     large reward scale
#
# Each run logs to: $LOG_ROOT/<game>_seed<N>/<train.py run name>/
# Tensorboard:  tensorboard --logdir $LOG_ROOT

set -euo pipefail

CONFIG="${1:?usage: $0 <config.yaml> [--games \"...\"] [--seeds N] [--jobs N] [--log-root DIR]}"
shift
GAMES="${GAMES:-pong breakout boxing freeway ms_pacman krull}"
SEEDS="${SEEDS:-1}"
JOBS="${JOBS:-1}"
LOG_ROOT="${LOG_ROOT:-./runs/atari100k/$(basename "$CONFIG" .yaml)_games_$(date +%Y%m%d-%H%M%S)}"
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --games) GAMES="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_ROOT"
cp "$CONFIG" "$LOG_ROOT/"  # keep the exact config the sweep was launched with
echo "Sweep config:"
echo "  CONFIG:   $CONFIG"
echo "  GAMES:    $GAMES"
echo "  SEEDS:    $SEEDS"
echo "  JOBS:     $JOBS"
echo "  LOG_ROOT: $LOG_ROOT"
echo ""

run_one() {
  local game="$1" seed="$2"
  local run_dir="${LOG_ROOT}/${game}_seed${seed}"
  mkdir -p "$run_dir"
  echo "Starting: ${game}_seed${seed} at $(date)"
  # train.py creates its own timestamped run folder inside train.log_dir
  $PYTHON -m world_models.torch.agents.train \
    --config "$CONFIG" \
    env.game="$game" \
    seed="$seed" \
    train.log_dir="$run_dir" \
    > "${run_dir}/train.log" 2>&1 \
    || echo "FAILED: ${game}_seed${seed} (see ${run_dir}/train.log)"
  echo "Finished: ${game}_seed${seed} at $(date)"
}

for seed in $(seq 1 $((SEEDS))); do
  for game in $GAMES; do
    run_one "$game" "$seed" &
    # keep at most $JOBS runs going
    while (( $(jobs -rp | wc -l) >= JOBS )); do
      wait -n
    done
  done
done
wait

echo "Sweep complete. View results with:"
echo "  tensorboard --logdir $LOG_ROOT"
