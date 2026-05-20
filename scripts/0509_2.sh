#!/usr/bin/env bash
set -euo pipefail

export WANDB_CACHE_DIR=/mnt/shared-storage-user/beam/wuhao/wandb_cache
export WANDB_ARTIFACT_CACHE=10GB
export WANDB_BASE_URL="http://100.96.30.112:8080"
export WANDB_API_KEY="local-f401d2b9276fb4a6dd1db4c1efee0512723ce6fb"


EXP_PREFIX="${EXP_PREFIX:-20260511_double_pert_pair_5fold}" \
LOG_TO_WANDB="${LOG_TO_WANDB:-1}" \
FOLDS="${FOLDS:-0 1 2 3 4}" \
MAX_EPOCHS="${MAX_EPOCHS:-100}" \
LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
bash scripts/exp_06_double_pert_pair_5fold.sh
