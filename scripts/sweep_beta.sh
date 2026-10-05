#!/usr/bin/env bash
# Sweep the contrastive objective's beta (cross-trajectory negative weight).
#
# Usage:
#   ./scripts/sweep_beta.sh                              # default betas, 1 seed, sequential
#   ./scripts/sweep_beta.sh --betas "0.05 0.2" --seeds 2
#   ./scripts/sweep_beta.sh --jobs 2                     # run 2 at a time on the same GPU
#   GAME=pong ./scripts/sweep_beta.sh
#
# Each run logs to: $LOG_ROOT/beta_<value>_seed<N>/<train.py run name>/
# so runs show up in tensorboard grouped by beta.
# Tensorboard:  tensorboard --logdir $LOG_ROOT

set -euo pipefail

# ---------- defaults ----------
CONFIG="${CONFIG:-configs/atari100k/rssm/contrastive_dreamer.yaml}"
BETAS="${BETAS:-0.02 0.05 0.1 0.3 1.0}"
SEEDS="${SEEDS:-1}"
GAME="${GAME:-breakout}"
JOBS="${JOBS:-1}"
OBJ_INDEX="${OBJ_INDEX:-0}"  # position of the contrastive objective in wm.objectives
LOG_ROOT="${LOG_ROOT:-./runs/atari100k/beta_sweep_$(date +%Y%m%d-%H%M%S)}"
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --betas) BETAS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --game) GAME="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_ROOT"
echo "Sweep config:"
echo "  CONFIG:   $CONFIG"
echo "  GAME:     $GAME"
echo "  BETAS:    $BETAS"
echo "  SEEDS:    $SEEDS"
echo "  JOBS:     $JOBS"
echo "  LOG_ROOT: $LOG_ROOT"
echo ""

run_one() {
  local beta="$1" seed="$2"
  local run_name="beta_${beta}_seed${seed}"
  local run_dir="${LOG_ROOT}/${run_name}"
  mkdir -p "$run_dir"

  echo "=========================================="
  echo "Starting: $run_name"
  echo "Log dir:  $run_dir"
  echo "Start at: $(date)"
  echo "=========================================="

  # train.py creates its own timestamped run folder inside train.log_dir
  $PYTHON -m world_models.torch.agents.train \
    --config "$CONFIG" \
    env.game="$GAME" \
    seed="$seed" \
    "wm.objectives.${OBJ_INDEX}.beta=${beta}" \
    train.log_dir="$run_dir" \
    > "${run_dir}/train.log" 2>&1 \
    || echo "FAILED: $run_name (see ${run_dir}/train.log)"

  echo "Finished: $run_name at $(date)"
}

# ---------- run loop ----------
for seed in $(seq 1 $((SEEDS))); do
  for beta in $BETAS; do
    run_one "$beta" "$seed" &
    # keep at most $JOBS runs going
    while (( $(jobs -rp | wc -l) >= JOBS )); do
      wait -n
    done
  done
done
wait

echo "Sweep complete. View results with:"
echo "  tensorboard --logdir $LOG_ROOT"
