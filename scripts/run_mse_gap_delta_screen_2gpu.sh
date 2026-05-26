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
BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_mse_gap_delta_screen}"
GPU_IDS="${GPU_IDS:-0,1}"
FOLDS="${FOLDS:-0 2}"
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
METHODS="${METHODS:-mse_target_pdi mse_target_pdi_w05 mse_target_pdippi_w05 delta_summary_all01_dim16 delta_summary_all025_dim16 delta_summary_all025_dim32 delta_summary_all025 delta_summary_all05 delta_gate_all05 delta_summary_pdi delta_gate_pdi delta_summary_pdippi_sched topvar_pdi_sched}"

run_one() {
  local gpu_id="$1"
  local method="$2"
  local fold="$3"
  local variant="$4"
  local script="scripts/exp_01_single_pert_stratified_5fold.sh"
  if [[ "${variant}" == "nomse" ]]; then
    script="scripts/exp_04_single_no_mse_5fold.sh"
  fi

  local method_env=()
  case "${method}" in
    mse_target_pdi)
      method_env=(
        RESPONSE_DELTA_MODE=off
        DELTA_LOGIT_SCALE=0.0
        MSE_TARGET_MODE=pdi
        MSE_TARGET_TOPK=512
        MSE_WEIGHT=0.25
      )
      ;;
    mse_target_pdi_w05)
      method_env=(
        RESPONSE_DELTA_MODE=off
        DELTA_LOGIT_SCALE=0.0
        MSE_TARGET_MODE=pdi
        MSE_TARGET_TOPK=512
        MSE_WEIGHT=0.5
      )
      ;;
    mse_target_pdippi_w05)
      method_env=(
        RESPONSE_DELTA_MODE=off
        DELTA_LOGIT_SCALE=0.0
        MSE_TARGET_MODE=pdi_ppi
        MSE_TARGET_TOPK=512
        MSE_TARGET_PPI_TOPK=32
        MSE_TARGET_PPI_ALPHA=0.5
        MSE_WEIGHT=0.5
      )
      ;;
    delta_summary_all01_dim16)
      method_env=(
        RESPONSE_DELTA_MODE=summary
        RESPONSE_DELTA_DIM=16
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=0.1
        MSE_TARGET_MODE=all
        MSE_WEIGHT=0.25
      )
      ;;
    delta_summary_all025_dim16)
      method_env=(
        RESPONSE_DELTA_MODE=summary
        RESPONSE_DELTA_DIM=16
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=0.25
        MSE_TARGET_MODE=all
        MSE_WEIGHT=0.25
      )
      ;;
    delta_summary_all025_dim32)
      method_env=(
        RESPONSE_DELTA_MODE=summary
        RESPONSE_DELTA_DIM=32
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=0.25
        MSE_TARGET_MODE=all
        MSE_WEIGHT=0.25
      )
      ;;
    delta_summary_all025)
      method_env=(
        RESPONSE_DELTA_MODE=summary
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=0.25
        MSE_TARGET_MODE=all
        MSE_WEIGHT=0.25
      )
      ;;
    delta_summary_all05)
      method_env=(
        RESPONSE_DELTA_MODE=summary
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=0.5
        MSE_TARGET_MODE=all
        MSE_WEIGHT=0.25
      )
      ;;
    delta_gate_all05)
      method_env=(
        RESPONSE_DELTA_MODE=gate
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=0.5
        MSE_TARGET_MODE=all
        MSE_WEIGHT=0.25
      )
      ;;
    delta_summary_pdi)
      method_env=(
        RESPONSE_DELTA_MODE=summary
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=0.5
        MSE_TARGET_MODE=pdi
        MSE_TARGET_TOPK=512
        MSE_WEIGHT=0.25
      )
      ;;
    delta_gate_pdi)
      method_env=(
        RESPONSE_DELTA_MODE=gate
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=1.0
        MSE_TARGET_MODE=pdi
        MSE_TARGET_TOPK=512
        MSE_WEIGHT=0.25
      )
      ;;
    delta_summary_pdippi_sched)
      method_env=(
        RESPONSE_DELTA_MODE=summary
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=1.0
        MSE_TARGET_MODE=pdi_ppi
        MSE_TARGET_TOPK=512
        MSE_TARGET_PPI_TOPK=32
        MSE_TARGET_PPI_ALPHA=0.5
        MSE_WEIGHT=0.35
        MSE_WEIGHT_SCHEDULE=warmup_decay
        MSE_DECAY_START_EPOCH_FRAC=0.4
        MSE_FINAL_WEIGHT_MULTIPLIER=0.35
      )
      ;;
    topvar_pdi_sched)
      method_env=(
        RESPONSE_DELTA_MODE=summary
        RESPONSE_DELTA_DETACH=1
        DELTA_LOGIT_SCALE=1.0
        MSE_TARGET_MODE=topvar_pdi
        MSE_TARGET_TOPK=768
        MSE_TARGET_VARIANCE_TOPK=4096
        MSE_TARGET_VARIANCE_SCALE=2.0
        MSE_WEIGHT=0.35
        MSE_WEIGHT_SCHEDULE=warmup_decay
        MSE_DECAY_START_EPOCH_FRAC=0.4
        MSE_FINAL_WEIGHT_MULTIPLIER=0.35
      )
      ;;
    *)
      echo "[error] unknown method: ${method}" >&2
      exit 2
      ;;
  esac

  local exp_prefix="${BASE_EXP_PREFIX}_${method}_${variant}"
  local log_path="${LOG_DIR}/${exp_prefix}_fold${fold}.log"
  echo "[run][gpu=${gpu_id}] method=${method} variant=${variant} fold=${fold}"
  env \
    GPU_IDS="${gpu_id}" \
    EXP_PREFIX="${exp_prefix}" \
    FOLDS="${fold}" \
    RUN_PREFLIGHT=0 \
    RUN_INFERENCE=0 \
    LOGGER_BACKEND="${LOGGER_BACKEND:-none}" \
    LOG_TO_WANDB="${LOG_TO_WANDB:-0}" \
    PROGRESS_BAR="${PROGRESS_BAR:-0}" \
    BATCH_SIZE="${BATCH_SIZE:-256}" \
    MAX_EPOCHS="${MAX_EPOCHS:-50}" \
    GRAPH_FEATURE_MODE=real \
    GRAPH_STRUCTURAL_RP=1 \
    GRAPH_DRUG_CONCAT=1 \
    GRAPH_LOGIT_SCALE="${GRAPH_LOGIT_SCALE:-2.0}" \
    PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-pcep}" \
    ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}" \
    "${method_env[@]}" \
    bash "${script}" > "${log_path}" 2>&1
  echo "[done][gpu=${gpu_id}] method=${method} variant=${variant} fold=${fold}"
}

jobs=()
for method in ${METHODS}; do
  for variant in mse nomse; do
    for fold in "${FOLD_ARRAY[@]}"; do
      jobs+=("${method}|${variant}|${fold}")
    done
  done
done

worker() {
  local slot="$1"
  local gpu_id="${GPU_ARRAY[$slot]}"
  local i
  for ((i = slot; i < ${#jobs[@]}; i += ${#GPU_ARRAY[@]})); do
    IFS='|' read -r method variant fold <<< "${jobs[$i]}"
    run_one "${gpu_id}" "${method}" "${fold}" "${variant}"
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
  echo "[error] at least one screening job failed; inspect ${LOG_DIR}/${BASE_EXP_PREFIX}_*.log" >&2
  exit "${status}"
fi

echo "[done] MSE-gap delta screening complete; EXP_PREFIX=${BASE_EXP_PREFIX}"
