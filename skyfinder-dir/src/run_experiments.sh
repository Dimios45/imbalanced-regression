#!/usr/bin/env bash
# Run all 5 DIR experiments for SkyFinder temperature regression.
# Each experiment appends one row to results/results_table.csv.
# Usage: bash src/run_experiments.sh [--epochs N] [--batch_size B]

set -euo pipefail

PYTHON=/mnt/data/mritunjoyh/miniforge3/envs/graspmas/bin/python
SCRIPT=src/train.py
DATA=data/
STORE=results/checkpoints
EPOCHS=${EPOCHS:-30}
BS=${BATCH_SIZE:-256}    # MI300X has 192 GB HBM — large batch is free
WANDB_FLAG=${USE_WANDB:+--wandb}   # set USE_WANDB=1 to enable
WORKERS=${WORKERS:-8}

echo "===== SkyFinder DIR Experiments (epochs=$EPOCHS, batch=$BS) ====="

run() {
  local name="$1"; shift
  echo ""
  echo ">>> [$name] $*"
  $PYTHON $SCRIPT --data_dir $DATA --store_root $STORE \
    --epoch $EPOCHS --batch_size $BS --workers $WORKERS \
    --store_name "$name" ${WANDB_FLAG} "$@" 2>&1
}

# 1. Vanilla baseline
run "01_baseline"

# 2. LDS only (label distribution smoothing, sqrt-inv reweight)
run "02_lds"            --reweight sqrt_inv --lds

# 3. LDS + FDS (full DIR method)
run "03_lds_fds"        --reweight sqrt_inv --lds --fds

echo ""
echo "===== All experiments done. Results → results/results_table.csv ====="
