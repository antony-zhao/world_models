#!/bin/bash
# One Atari100k game per array task. Submit from the repo root:
#
#   mkdir -p slurm_logs
#   sbatch scripts/slurm_atari100k.sbatch                      # all 26 games, 3 at a time
#   sbatch --array=19 scripts/slurm_atari100k.sbatch           # one game (19 = ms_pacman)
#   sbatch --array=0-8%3 scripts/slurm_atari100k.sbatch        # a subset
#   sbatch --export=ALL,CONFIG=configs/atari100k/transformer/storm.yaml scripts/slurm_atari100k.sbatch
#   sbatch --export=ALL,SEED=2,OVERRIDES="optim.ac_eps=1e-5" scripts/slurm_atari100k.sbatch
#
# Resubmitting skips games that already finished.
# Runs land in:  runs/atari100k/<config>_full/<game>_seed<N>/<train.py run name>/
# Only checkpoint_final.pt is saved; pass OVERRIDES="train.checkpoint_every=25000" for intermediates.
#
# Index -> game:
#    0 alien            7 breakout        14 hero            21 private_eye
#    1 amidar           8 chopper_command 15 jamesbond       22 qbert
#    2 assault          9 crazy_climber   16 kangaroo        23 road_runner
#    3 asterix         10 demon_attack    17 krull           24 seaquest
#    4 bank_heist      11 freeway         18 kung_fu_master  25 up_n_down
#    5 battle_zone     12 frostbite       19 ms_pacman
#    6 boxing          13 gopher          20 pong

#SBATCH --job-name=a100k
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal           # 24 h max; gpu-long allows 7 days at lower priority
#SBATCH --gres=gpu:a100-40gb:1     # Alpine wants the type; the other is gpu:a100_80gb:1
#SBATCH --nodes=1                  # Alpine rejects GPU jobs without it
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00            # ~6 h alone on an A6000; margin for a slower node
#SBATCH --array=0-25%3
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --account=ucb871_asc1

set -eo pipefail

GAMES=(alien amidar assault asterix bank_heist battle_zone boxing breakout
  chopper_command crazy_climber demon_attack freeway frostbite gopher hero jamesbond
  kangaroo krull kung_fu_master ms_pacman pong private_eye qbert road_runner seaquest
  up_n_down)

CONFIG="${CONFIG:-configs/atari100k/transformer/lejepa_storm.yaml}"
SEED="${SEED:-2}"
OVERRIDES="${OVERRIDES:-}"
LOG_ROOT="${LOG_ROOT:-./runs/atari100k/$(basename "$CONFIG" .yaml)_full}"

# Personal Miniconda under /projects (2 GB home quota is too small for it) if present,
# otherwise CURC's anaconda module. CONDA_ENV overrides the environment name.
module purge
CONDA_BASE="${CONDA_BASE:-/projects/$USER/miniconda3}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  module load anaconda
fi
conda activate "${CONDA_ENV:-world-models}"
set -u  # after activation: conda's activate scripts trip on unset variables

cd "$SLURM_SUBMIT_DIR"
game="${GAMES[$SLURM_ARRAY_TASK_ID]}"
run_dir="${LOG_ROOT}/${game}_seed${SEED}"

# train.py writes checkpoint_final.pt only when a run completes
if compgen -G "${run_dir}/*/checkpoint_final.pt" > /dev/null; then
  echo "skip (done): ${game}"
  exit 0
fi
mkdir -p "$run_dir"

echo "game=${game} seed=${SEED} config=${CONFIG} node=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHONUNBUFFERED=1 python -m world_models.torch.agents.train \
  --config "$CONFIG" \
  env.game="$game" \
  seed="$SEED" \
  train.log_dir="$run_dir" \
  train.checkpoint_every=0 \
  $OVERRIDES

echo "finished ${game} at $(date)"
