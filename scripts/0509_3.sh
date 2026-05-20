#!/usr/bin/env bash
set -euo pipefail

export WANDB_CACHE_DIR=/mnt/shared-storage-user/beam/wuhao/wandb_cache
export WANDB_ARTIFACT_CACHE=10GB
export WANDB_BASE_URL="http://100.96.30.112:8080"
export WANDB_API_KEY="local-f401d2b9276fb4a6dd1db4c1efee0512723ce6fb"


EXP_PREFIX="${EXP_PREFIX:-20260510_extra_single_all_train_infer}" \
LOG_TO_WANDB="${LOG_TO_WANDB:-1}" \
RUN_INFERENCE="${RUN_INFERENCE:-1}" \
MAX_EPOCHS="${MAX_EPOCHS:-100}" \
LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}" \
BATCH_SIZE="${BATCH_SIZE:-128}" \
REFERENCE_5FOLD_CKPT_PATH="${REFERENCE_5FOLD_CKPT_PATH:-checkpoints/20260511_single_pert_stratified_5fold}" \
REFERENCE_EPOCH_AGG="${REFERENCE_EPOCH_AGG:-median}" \
REFERENCE_EPOCH_MIN_COUNT="${REFERENCE_EPOCH_MIN_COUNT:-5}" \
SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}" \
SCHEDULER_NAME="${SCHEDULER_NAME:-}" \
bash scripts/exp_07_extra_single_all_train_infer.sh
