#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

GPU_IDS="${GPU_IDS:-0,1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
METHODS="${METHODS:-baseline4 baseline4_zero baseline4_no_mse}"
SPLIT_PREFIX="${SPLIT_PREFIX:-pert_stratified_5fold_fold}"
EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_baseline4_1gpu}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MSE_WEIGHT="${MSE_WEIGHT:-0.25}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
CKPT_DIR="${CKPT_DIR:-new_version/checkpoints}"
LOG_ROOT="${LOG_ROOT:-new_version/runtime_logs/${EXP_PREFIX}}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"

method_args() {
  local method="$1"
  case "${method}" in
    baseline4)
      printf '%s\n' \
        --mse-weight "${MSE_WEIGHT}" \
        --positive-weight none \
        --target-protein-max-length 32 \
        --graph-feature-mode real \
        --graph-feature-dim 128 \
        --graph-structural-rp \
        --graph-drug-concat \
        --graph-logit-scale 2.0 \
        --protein-concat-mode pcep \
        --protein-concat-dim 64 \
        --protein-concat-topk 512 \
        --protein-concat-init-scale 0.1
      ;;
    baseline4_zero)
      printf '%s\n' \
        --mse-weight "${MSE_WEIGHT}" \
        --positive-weight none \
        --target-protein-max-length 32 \
        --graph-feature-mode zero \
        --graph-feature-dim 128 \
        --graph-structural-rp \
        --graph-drug-concat \
        --graph-logit-scale 2.0 \
        --protein-concat-mode pcep \
        --protein-concat-dim 64 \
        --protein-concat-topk 512 \
        --protein-concat-init-scale 0.1
      ;;
    baseline4_no_mse)
      printf '%s\n' \
        --no-mse-loss \
        --positive-weight none \
        --target-protein-max-length 32 \
        --graph-feature-mode real \
        --graph-feature-dim 128 \
        --graph-structural-rp \
        --graph-drug-concat \
        --graph-logit-scale 2.0 \
        --protein-concat-mode pcep \
        --protein-concat-dim 64 \
        --protein-concat-topk 512 \
        --protein-concat-init-scale 0.1
      ;;
    baseline4_mse010)
      printf '%s\n' \
        --mse-weight 0.10 \
        --positive-weight none \
        --target-protein-max-length 32 \
        --graph-feature-mode real \
        --graph-feature-dim 128 \
        --graph-structural-rp \
        --graph-drug-concat \
        --graph-logit-scale 2.0 \
        --protein-concat-mode pcep \
        --protein-concat-dim 64 \
        --protein-concat-topk 512 \
        --protein-concat-init-scale 0.1
      ;;
    baseline4_mse005)
      printf '%s\n' \
        --mse-weight 0.05 \
        --positive-weight none \
        --target-protein-max-length 32 \
        --graph-feature-mode real \
        --graph-feature-dim 128 \
        --graph-structural-rp \
        --graph-drug-concat \
        --graph-logit-scale 2.0 \
        --protein-concat-mode pcep \
        --protein-concat-dim 64 \
        --protein-concat-topk 512 \
        --protein-concat-init-scale 0.1
      ;;
    baseline4_mse050)
      printf '%s\n' \
        --mse-weight 0.50 \
        --positive-weight none \
        --target-protein-max-length 32 \
        --graph-feature-mode real \
        --graph-feature-dim 128 \
        --graph-structural-rp \
        --graph-drug-concat \
        --graph-logit-scale 2.0 \
        --protein-concat-mode pcep \
        --protein-concat-dim 64 \
        --protein-concat-topk 512 \
        --protein-concat-init-scale 0.1
      ;;
    *)
      echo "[error] unknown method: ${method}" >&2
      return 2
      ;;
  esac
}

task_specs=()
for method in ${METHODS}; do
  for fold in ${FOLDS}; do
    task_specs+=("${method} ${fold}")
  done
done

run_task() {
  local gpu_id="$1"
  local method="$2"
  local fold="$3"
  local experiment="${EXP_PREFIX}_${method}_fold${fold}"
  local log_file="${LOG_ROOT}/${experiment}.log"
  mapfile -t extra_args < <(method_args "${method}")
  echo "[run][gpu=${gpu_id}] ${experiment}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" python new_version/train.py \
    --dataset-group ptv3 \
    --task-name ptv3_main_singledrug \
    --split-strategy "${SPLIT_PREFIX}${fold}" \
    --experiment-name "${experiment}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --accelerator gpu \
    --devices 1 \
    --strategy auto \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}" \
    --logger-backend "${LOGGER_BACKEND}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --save-top-k 1 \
    --no-save-last-ckpt \
    --monitor val/task_auprc \
    --monitor-mode max \
    --log-every-n-steps 49 \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1
  echo "[done][gpu=${gpu_id}] ${experiment} log=${log_file}"
}

worker() {
  local slot="$1"
  local gpu_id="${GPU_ARRAY[$slot]}"
  local i spec method fold
  for ((i = slot; i < ${#task_specs[@]}; i += ${#GPU_ARRAY[@]})); do
    spec="${task_specs[$i]}"
    method="${spec% *}"
    fold="${spec#* }"
    run_task "${gpu_id}" "${method}" "${fold}"
  done
}

echo "[baseline4] EXP_PREFIX=${EXP_PREFIX}"
echo "[baseline4] METHODS=${METHODS}"
echo "[baseline4] FOLDS=${FOLDS}"
echo "[baseline4] GPU_IDS=${GPU_IDS}; workers=${#GPU_ARRAY[@]}; BATCH_SIZE=${BATCH_SIZE}; MAX_EPOCHS=${MAX_EPOCHS}"

pids=()
for ((slot = 0; slot < ${#GPU_ARRAY[@]}; slot += 1)); do
  worker "${slot}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python new_version/summarize_runs.py \
  --checkpoint-dir "${CKPT_DIR}" \
  --prefix "${EXP_PREFIX}" \
  --output "${LOG_ROOT}/summary.tsv"

echo "[summary] ${LOG_ROOT}/summary.tsv"
