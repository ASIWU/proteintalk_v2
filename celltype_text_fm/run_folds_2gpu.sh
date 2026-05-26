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
EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_celltype_text}"
GPU_IDS="${GPU_IDS:-0,1}"
FOLDS="${FOLDS:-0 1 2 3 4}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
read -r -a FOLD_ARRAY <<< "${FOLDS}"

common_args=(
  --task-name ptv3_main_singledrug
  --task-head response
  --batch-size "${BATCH_SIZE:-256}"
  --max-epochs "${MAX_EPOCHS:-50}"
  --mse-weight "${MSE_WEIGHT:-0.075}"
  --logger-backend "${LOGGER_BACKEND:-none}"
  --checkpoint-dir "${CKPT_DIR:-checkpoints}"
  --log-dir "${LOG_DIR:-logs}"
  --cell-type-text-mode sapbert
  --cell-type-text-cache "${CELL_TYPE_TEXT_CACHE:-celltype_text_fm/artifacts/cell_type_sapbert_features.npz}"
  --cell-type-text-logit-scale "${CELL_TYPE_TEXT_LOGIT_SCALE:-0.0}"
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  common_args+=("${extra_args[@]}")
fi

run_fold() {
  local gpu_id="$1"
  local fold="$2"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u celltype_text_fm/train_text_celltype.py \
    "${common_args[@]}" \
    --experiment-name "${EXP_PREFIX}_fold${fold}" \
    --split-strategy "cell_5fold_fold${fold}"
}

pids=()
for idx in "${!FOLD_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
  run_fold "${gpu}" "${FOLD_ARRAY[$idx]}" &
  pids+=("$!")
  if (( ${#pids[@]} >= ${#GPU_ARRAY[@]} )); then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

