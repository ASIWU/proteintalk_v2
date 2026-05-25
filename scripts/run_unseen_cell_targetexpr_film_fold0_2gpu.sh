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
BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_targetexpr_film_cell_fold0}"
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

echo "[prebuild] target-expression cache"
CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "${PYTHON_BIN}" -u train.py \
  --dataset-group ptv3 \
  --model-type fast_delta \
  --task-name ptv3_main_singledrug \
  --split-strategy cell_5fold_fold0 \
  --task-head response \
  --batch-size 16 \
  --max-epochs 1 \
  --devices 1 \
  --accelerator gpu \
  --precision bf16-mixed \
  --num-workers 0 \
  --logger-backend none \
  --no-save-last-ckpt \
  --save-top-k 0 \
  --dry-run-batches 1 \
  --graph-feature-mode real \
  --graph-structural-rp \
  --graph-drug-concat \
  --protein-concat-mode pcep \
  --mse-weight 0.075 \
  --covariate-unk-for-unseen \
  --covariate-unk-dropout 0.15 \
  --target-expression-mode pdi_ppi \
  --target-expression-topk 256 \
  --target-expression-ppi-topk 32 \
  --target-expression-ppi-alpha 0.5

VARIANTS=(
  "baseline:off:0.0:256:32:0.5"
  "film025:off:0.25:256:32:0.5"
  "pdi_ppi:pdi_ppi:0.0:256:32:0.5"
  "pdi_ppi_film025:pdi_ppi:0.25:256:32:0.5"
  "pdi_only:pdi:0.0:256:32:0.0"
  "pdi_ppi_top512_film025:pdi_ppi:0.25:512:32:0.5"
)

run_variant() {
  local gpu_id="$1"
  local spec="$2"
  IFS=':' read -r name target_mode film_scale target_topk ppi_topk ppi_alpha <<< "${spec}"
  local exp_prefix="${BASE_EXP_PREFIX}_${name}"
  local log_path="${LOG_DIR}/${exp_prefix}.log"
  echo "[run][gpu=${gpu_id}] ${name} target=${target_mode} film=${film_scale} topk=${target_topk} ppi_topk=${ppi_topk} alpha=${ppi_alpha}"
  GPU_IDS="${gpu_id}" \
  EXP_PREFIX="${exp_prefix}" \
  FOLDS="0" \
  RUN_PREFLIGHT=0 \
  RUN_INFERENCE=0 \
  LOGGER_BACKEND=none \
  LOG_TO_WANDB=0 \
  PROGRESS_BAR=0 \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  MAX_EPOCHS="${MAX_EPOCHS:-50}" \
  MSE_WEIGHT="${MSE_WEIGHT:-0.075}" \
  GRAPH_FEATURE_MODE=real \
  GRAPH_STRUCTURAL_RP=1 \
  GRAPH_DRUG_CONCAT=1 \
  PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-pcep}" \
  COVARIATE_UNK_FOR_UNSEEN=1 \
  COVARIATE_UNK_DROPOUT="${COVARIATE_UNK_DROPOUT:-0.15}" \
  TARGET_EXPRESSION_MODE="${target_mode}" \
  TARGET_EXPRESSION_TOPK="${target_topk}" \
  TARGET_EXPRESSION_PPI_TOPK="${ppi_topk}" \
  TARGET_EXPRESSION_PPI_ALPHA="${ppi_alpha}" \
  CELL_PAIR_FILM_SCALE="${film_scale}" \
  ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}" \
  bash scripts/exp_03_single_cell_5fold.sh > "${log_path}" 2>&1
  echo "[done][gpu=${gpu_id}] ${name}; log=${log_path}"
}

worker() {
  local slot="$1"
  local gpu_id="${GPU_ARRAY[$slot]}"
  local i
  for ((i = slot; i < ${#VARIANTS[@]}; i += ${#GPU_ARRAY[@]})); do
    run_variant "${gpu_id}" "${VARIANTS[$i]}"
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
  echo "[error] at least one variant failed; inspect ${LOG_DIR}/${BASE_EXP_PREFIX}_*.log" >&2
  exit "${status}"
fi

echo "[done] variants complete; EXP_PREFIX=${BASE_EXP_PREFIX}"
