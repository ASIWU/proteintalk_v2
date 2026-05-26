#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_mse_gap_ckpt_strategy}"
GPU_IDS="${GPU_IDS:-0,1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
STRATEGIES="${STRATEGIES:-valid_auprc valid_auroc loss2 total_loss last10 last20}"
VARIANTS="${VARIANTS:-mse nomse}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 2
fi

"${PYTHON_BIN}" -m py_compile \
  train.py \
  infer.py \
  dataset/training_ready_fast_dataset.py \
  model/fast_delta_model.py \
  model/fast_lightning.py \
  model/graph_feature_utils.py

read -r -a FOLD_ARRAY <<< "${FOLDS}"

run_one() {
  local gpu_id="$1"
  local strategy="$2"
  local variant="$3"
  local fold="$4"
  local script="scripts/exp_01_single_pert_stratified_5fold.sh"
  local best_metric="${strategy}"
  local monitor=""
  local save_top_k="${SAVE_TOP_K:-1}"
  local save_last_ckpt="${SAVE_LAST_CKPT:-1}"
  local max_epochs="${MAX_EPOCHS:-50}"
  if [[ "${strategy}" =~ ^last([0-9]+)$ ]]; then
      best_metric="valid_auprc"
      monitor="none"
      save_top_k=0
      save_last_ckpt=0
      max_epochs="${BASH_REMATCH[1]}"
  fi
  if [[ "${variant}" == "nomse" ]]; then
    script="scripts/exp_04_single_no_mse_5fold.sh"
  fi

  local exp_prefix="${BASE_EXP_PREFIX}_${strategy}_${variant}"
  local log_path="${LOG_DIR}/${exp_prefix}_fold${fold}.log"
  echo "[run][gpu=${gpu_id}] strategy=${strategy} variant=${variant} fold=${fold}"
  GPU_IDS="${gpu_id}" \
  EXP_PREFIX="${exp_prefix}" \
  FOLDS="${fold}" \
  BEST_CKPT_METRIC="${best_metric}" \
  MONITOR="${monitor}" \
  SAVE_TOP_K="${save_top_k}" \
  SAVE_LAST_CKPT="${save_last_ckpt}" \
  RUN_PREFLIGHT=0 \
  RUN_INFERENCE=0 \
  LOGGER_BACKEND="${LOGGER_BACKEND:-none}" \
  LOG_TO_WANDB="${LOG_TO_WANDB:-0}" \
  PROGRESS_BAR="${PROGRESS_BAR:-0}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  MAX_EPOCHS="${max_epochs}" \
  MSE_WEIGHT="${MSE_WEIGHT:-0.25}" \
  GRAPH_FEATURE_MODE=real \
  GRAPH_STRUCTURAL_RP=1 \
  GRAPH_DRUG_CONCAT=1 \
  GRAPH_LOGIT_SCALE="${GRAPH_LOGIT_SCALE:-2.0}" \
  PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-pcep}" \
  ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}" \
  bash "${script}" > "${log_path}" 2>&1
  echo "[done][gpu=${gpu_id}] strategy=${strategy} variant=${variant} fold=${fold}"
}

jobs=()
for strategy in ${STRATEGIES}; do
  for variant in ${VARIANTS}; do
    for fold in "${FOLD_ARRAY[@]}"; do
      jobs+=("${strategy}|${variant}|${fold}")
    done
  done
done

worker() {
  local slot="$1"
  local gpu_id="${GPU_ARRAY[$slot]}"
  local i
  for ((i = slot; i < ${#jobs[@]}; i += ${#GPU_ARRAY[@]})); do
    IFS='|' read -r strategy variant fold <<< "${jobs[$i]}"
    run_one "${gpu_id}" "${strategy}" "${variant}" "${fold}"
  done
}

pids=()
for ((slot = 0; slot < ${#GPU_ARRAY[@]}; slot += 1)); do
  worker "${slot}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "[error] at least one checkpoint-strategy job failed; inspect ${LOG_DIR}/${BASE_EXP_PREFIX}_*.log" >&2
  exit "${status}"
fi

echo "[done] checkpoint-strategy MSE-gap screening complete; EXP_PREFIX=${BASE_EXP_PREFIX}"
