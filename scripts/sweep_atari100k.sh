#!/usr/bin/env bash
# Launch training across a few Atari100k games sequentially.
#
# Usage:
#   ./scripts/sweep_atari100k.sh                 # default 3 games, 1 seed each
#   ./scripts/sweep_atari100k.sh --seeds 3       # 3 seeds per game
#   GAMES="Pong Boxing" ./scripts/sweep_atari100k.sh
#
# Each run logs to: $LOG_ROOT/<game>_seed<N>/
# Tensorboard:  tensorboard --logdir $LOG_ROOT

set -euo pipefail

# ---------- defaults ----------
CONFIG="${CONFIG:-configs/atari100k.yaml}"
GAMES="${GAMES:-Pong Boxing Breakout}"
SEEDS="${SEEDS:-1}"
LOG_ROOT="${LOG_ROOT:-./runs/sweep_$(date +%Y%m%d-%H%M%S)}"
PYTHON="${PYTHON:-python}"

# parse --seeds flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --games) GAMES="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_ROOT"
echo "Sweep config:"
echo "  CONFIG:   $CONFIG"
echo "  GAMES:    $GAMES"
echo "  SEEDS:    $SEEDS"
echo "  LOG_ROOT: $LOG_ROOT"
echo ""

# ---------- run loop ----------
for game in $GAMES; do
  for seed in $(seq 0 $((SEEDS - 1))); do
    run_name="${game}_seed${seed}"
    run_dir="${LOG_ROOT}/${run_name}"
    mkdir -p "$run_dir"

    echo "=========================================="
    echo "Starting: $run_name"
    echo "Log dir:  $run_dir"
    echo "Start at: $(date)"
    echo "=========================================="

    # tee stdout/stderr to a log file, also keep on screen
    $PYTHON -m world_models.torch.agents.train \
      --config "$CONFIG" \
      env.id="ALE/${game}-v5" \
      seed="$seed" \
      train.log_dir="$LOG_ROOT" \
      2>&1 | tee "${run_dir}/train.log"

    echo "Finished: $run_name at $(date)"
    echo ""
  done
done

echo "Sweep complete. View results with:"
echo "  tensorboard --logdir $LOG_ROOT"
