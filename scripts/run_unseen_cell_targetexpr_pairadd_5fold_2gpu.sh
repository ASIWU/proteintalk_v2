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
BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_targetexpr_pairadd_cell_5fold}"
GPU_IDS="${GPU_IDS:-0,1}"
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

read -r -a FOLD_ARRAY <<< "${FOLDS:-0 1 2 3 4}"

run_fold() {
  local gpu_id="$1"
  local fold="$2"
  echo "[run][gpu=${gpu_id}] fold=${fold}"
  GPU_IDS="${gpu_id}" \
  EXP_PREFIX="${BASE_EXP_PREFIX}" \
  FOLDS="${fold}" \
  RUN_PREFLIGHT=0 \
  RUN_INFERENCE=0 \
  LOGGER_BACKEND="${LOGGER_BACKEND:-none}" \
  LOG_TO_WANDB="${LOG_TO_WANDB:-0}" \
  PROGRESS_BAR="${PROGRESS_BAR:-0}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  MAX_EPOCHS="${MAX_EPOCHS:-50}" \
  MSE_WEIGHT="${MSE_WEIGHT:-0.075}" \
  GRAPH_FEATURE_MODE=real \
  GRAPH_STRUCTURAL_RP=1 \
  GRAPH_DRUG_CONCAT=1 \
  PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-pcep}" \
  COVARIATE_UNK_FOR_UNSEEN=1 \
  COVARIATE_UNK_DROPOUT="${COVARIATE_UNK_DROPOUT:-0.15}" \
  TARGET_EXPRESSION_MODE=pdi_ppi \
  TARGET_EXPRESSION_FUSION_MODE=pair_add \
  TARGET_EXPRESSION_TOPK="${TARGET_EXPRESSION_TOPK:-256}" \
  TARGET_EXPRESSION_PPI_TOPK="${TARGET_EXPRESSION_PPI_TOPK:-32}" \
  TARGET_EXPRESSION_PPI_ALPHA="${TARGET_EXPRESSION_PPI_ALPHA:-0.5}" \
  TARGET_EXPRESSION_PPI_NORM="${TARGET_EXPRESSION_PPI_NORM:-raw}" \
  TARGET_EXPRESSION_DEGREE_PENALTY="${TARGET_EXPRESSION_DEGREE_PENALTY:-0.0}" \
  TARGET_EXPRESSION_INIT_SCALE="${TARGET_EXPRESSION_INIT_SCALE:-0.5}" \
  TARGET_EXPRESSION_CELL_GATE_MODE="${TARGET_EXPRESSION_CELL_GATE_MODE:-off}" \
  TARGET_EXPRESSION_CELL_GATE_SCALE="${TARGET_EXPRESSION_CELL_GATE_SCALE:-0.0}" \
  TARGET_EXPRESSION_CELL_GATE_TEMPERATURE="${TARGET_EXPRESSION_CELL_GATE_TEMPERATURE:-1.0}" \
  CELL_PAIR_FILM_SCALE=0.0 \
  ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}" \
  bash scripts/exp_03_single_cell_5fold.sh > "${LOG_DIR}/${BASE_EXP_PREFIX}_fold${fold}.log" 2>&1
  echo "[done][gpu=${gpu_id}] fold=${fold}"
}

worker() {
  local slot="$1"
  local gpu_id="${GPU_ARRAY[$slot]}"
  local i
  for ((i = slot; i < ${#FOLD_ARRAY[@]}; i += ${#GPU_ARRAY[@]})); do
    run_fold "${gpu_id}" "${FOLD_ARRAY[$i]}"
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
  echo "[error] at least one fold failed; inspect ${LOG_DIR}/${BASE_EXP_PREFIX}_fold*.log" >&2
  exit "${status}"
fi

echo "[done] target-expression pair_add 5-fold complete; EXP_PREFIX=${BASE_EXP_PREFIX}"
