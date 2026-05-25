#!/usr/bin/env bash
set -euo pipefail

# Fold-0 covariate ablation for the current fast_delta baseline.
# Runs unseen-drug fold0 and unseen-cell fold0 across two GPUs by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_covariate_fold0}"
GPU_IDS="${GPU_IDS:-0,1}"
PROFILES="${PROFILES:-full full_covunk015 no_cov drop_machine drop_plate drop_cell drop_cell_type drop_batch drop_pert_time bio_only tech_only cell_identity_only no_plate_batch no_highcard_ids cell_type_time_only cell_type_only cell_only cell_identity_covunk015 bio_covunk015}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${REPO_ROOT}/logs/${BASE_EXP_PREFIX}_runtime_summary.tsv}"
LOG_DIR="${LOG_DIR:-logs}"

export MODEL_TYPE="${MODEL_TYPE:-fast_delta}"
export DEVICES="${DEVICES:-1}"
export STRATEGY="${STRATEGY:-auto}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
export MAX_EPOCHS="${MAX_EPOCHS:-50}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export MSE_WEIGHT="${MSE_WEIGHT:-0.25}"
export LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
export LOG_TO_WANDB="${LOG_TO_WANDB:-0}"
export PRECISION="${PRECISION:-bf16-mixed}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}"
export RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
export RUN_DATA_VALIDATION="${RUN_DATA_VALIDATION:-0}"
export RUN_INFERENCE=0
export PROGRESS_BAR="${PROGRESS_BAR:-0}"
export GRAPH_FEATURE_MODE="${GRAPH_FEATURE_MODE:-real}"
export GRAPH_FEATURE_DIM="${GRAPH_FEATURE_DIM:-128}"
export GRAPH_FEATURE_SEED="${GRAPH_FEATURE_SEED:-17}"
export GRAPH_STRUCTURAL_RP="${GRAPH_STRUCTURAL_RP:-1}"
export GRAPH_DRUG_CONCAT="${GRAPH_DRUG_CONCAT:-1}"
export GRAPH_PAIR_ADD_SCALE="${GRAPH_PAIR_ADD_SCALE:-0.0}"
export GRAPH_LOGIT_SCALE="${GRAPH_LOGIT_SCALE:-2.0}"
export PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-pcep}"
export PROTEIN_CONCAT_DIM="${PROTEIN_CONCAT_DIM:-64}"
export PROTEIN_CONCAT_TOPK="${PROTEIN_CONCAT_TOPK:-512}"
export PAIR_FUSION_MODE="${PAIR_FUSION_MODE:-symmetric}"
export PAIR_TYPE_FEATURES="${PAIR_TYPE_FEATURES:-0}"
export MSE_INACTIVE_LABEL_WEIGHT="${MSE_INACTIVE_LABEL_WEIGHT:-1.0}"
export USE_DDI="${USE_DDI:-0}"
export CKPT_DIR="${CKPT_DIR:-checkpoints}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
export LOG_DIR
export TIME_SUMMARY_PATH

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m py_compile \
  train.py \
  dataset/training_ready_fast_dataset.py \
  model/fast_delta_model.py \
  model/fast_lightning.py \
  scripts/prebuild_graph_cache.py \
  scripts/covariate_analysis_report.py

if [[ "${MODEL_TYPE}" == "fast_delta" && "${GRAPH_FEATURE_MODE}" != "off" ]]; then
  prebuild_args=(
    --dataset-group ptv3
    --graph-cache-dir "${GRAPH_CACHE_DIR:-graph_cache}"
    --graph-feature-dim "${GRAPH_FEATURE_DIM}"
    --graph-feature-seed "${GRAPH_FEATURE_SEED}"
  )
  if [[ "${GRAPH_STRUCTURAL_RP}" == "1" ]]; then
    prebuild_args+=(--graph-structural-rp)
  fi
  if [[ "${GRAPH_MULTIHOP:-0}" == "1" ]]; then
    prebuild_args+=(--graph-multihop)
  fi
  "${PYTHON_BIN}" scripts/prebuild_graph_cache.py "${prebuild_args[@]}"
fi

profile_covariates() {
  case "$1" in
    full|full_covunk015) echo "machineID_new Cell_plate Cell cell_type batch pert_time" ;;
    no_cov) echo "__none__" ;;
    drop_machine) echo "Cell_plate Cell cell_type batch pert_time" ;;
    drop_plate) echo "machineID_new Cell cell_type batch pert_time" ;;
    drop_cell) echo "machineID_new Cell_plate cell_type batch pert_time" ;;
    drop_cell_type) echo "machineID_new Cell_plate Cell batch pert_time" ;;
    drop_batch) echo "machineID_new Cell_plate Cell cell_type pert_time" ;;
    drop_pert_time) echo "machineID_new Cell_plate Cell cell_type batch" ;;
    bio_only) echo "Cell cell_type pert_time" ;;
    tech_only) echo "machineID_new Cell_plate batch" ;;
    cell_identity_only) echo "Cell cell_type" ;;
    no_plate_batch) echo "machineID_new Cell cell_type pert_time" ;;
    no_highcard_ids) echo "machineID_new cell_type pert_time" ;;
    cell_type_time_only) echo "cell_type pert_time" ;;
    cell_type_only) echo "cell_type" ;;
    cell_only) echo "Cell" ;;
    cell_identity_covunk015) echo "Cell cell_type" ;;
    bio_covunk015) echo "Cell cell_type pert_time" ;;
    *) echo "[error] unknown covariate profile: $1" >&2; return 2 ;;
  esac
}

profile_unk_fields() {
  case "$1" in
    full_covunk015) echo "machineID_new Cell_plate Cell cell_type batch pert_time" ;;
    cell_identity_covunk015) echo "Cell cell_type" ;;
    bio_covunk015) echo "Cell cell_type pert_time" ;;
    *) echo "" ;;
  esac
}

profile_report_arg() {
  local profile="$1"
  local covs
  covs="$(profile_covariates "${profile}")"
  if [[ "${covs}" == "__none__" ]]; then
    echo "${profile}:__none__"
  else
    echo "${profile}:${covs// /,}"
  fi
}

run_one() {
  local gpu_id="$1"
  local task_label="$2"
  local profile="$3"
  local script_path
  local covariates
  local log_path

  covariates="$(profile_covariates "${profile}")"
  local unk_fields
  unk_fields="$(profile_unk_fields "${profile}")"
  if [[ "${task_label}" == "unseen_drug_fold0" ]]; then
    script_path="scripts/exp_01_single_pert_stratified_5fold.sh"
  elif [[ "${task_label}" == "unseen_cell_fold0" ]]; then
    script_path="scripts/exp_03_single_cell_5fold.sh"
  else
    echo "[error] unknown task_label: ${task_label}" >&2
    return 2
  fi

  log_path="${LOG_DIR}/${BASE_EXP_PREFIX}_${task_label}_${profile}.log"
  echo "[run][gpu=${gpu_id}] task=${task_label} profile=${profile} covariates=${covariates}"
  if [[ -n "${unk_fields}" ]]; then
    GPU_IDS="${gpu_id}" \
    EXP_PREFIX="${BASE_EXP_PREFIX}_${task_label}_${profile}" \
    FOLDS="0" \
    BATCH_COV_LIST="${covariates}" \
    COVARIATE_UNK_FOR_UNSEEN=1 \
    COVARIATE_UNK_FIELDS="${unk_fields}" \
    COVARIATE_UNK_DROPOUT=0.15 \
    bash "${script_path}" >"${log_path}" 2>&1
  else
    GPU_IDS="${gpu_id}" \
    EXP_PREFIX="${BASE_EXP_PREFIX}_${task_label}_${profile}" \
    FOLDS="0" \
    BATCH_COV_LIST="${covariates}" \
    COVARIATE_UNK_FOR_UNSEEN=0 \
    COVARIATE_UNK_FIELDS="" \
    COVARIATE_UNK_DROPOUT=0.0 \
    bash "${script_path}" >"${log_path}" 2>&1
  fi
  echo "[done][gpu=${gpu_id}] task=${task_label} profile=${profile} log=${log_path}"
}

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
read -r -a PROFILE_ARRAY <<< "${PROFILES}"
if (( ${#GPU_ARRAY[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 2
fi

task_specs=()
for profile in "${PROFILE_ARRAY[@]}"; do
  task_specs+=("unseen_drug_fold0 ${profile}")
  task_specs+=("unseen_cell_fold0 ${profile}")
done

worker() {
  local slot="$1"
  local gpu_id="${GPU_ARRAY[$slot]}"
  local i spec task_label profile
  for ((i = slot; i < ${#task_specs[@]}; i += ${#GPU_ARRAY[@]})); do
    spec="${task_specs[$i]}"
    task_label="$(awk '{print $1}' <<< "${spec}")"
    profile="$(awk '{print $2}' <<< "${spec}")"
    run_one "${gpu_id}" "${task_label}" "${profile}"
  done
}

echo "[covariate-analysis] EXP_PREFIX=${BASE_EXP_PREFIX}"
echo "[covariate-analysis] PROFILES=${PROFILES}"
echo "[covariate-analysis] GPU_IDS=${GPU_IDS}"

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
  echo "[error] at least one covariate ablation failed; inspect ${LOG_DIR}/${BASE_EXP_PREFIX}_*.log" >&2
  exit "${status}"
fi

report_profiles=()
for profile in "${PROFILE_ARRAY[@]}"; do
  report_profiles+=("$(profile_report_arg "${profile}")")
done

"${PYTHON_BIN}" scripts/covariate_analysis_report.py \
  --exp-prefix "${BASE_EXP_PREFIX}" \
  --runtime-summary "${TIME_SUMMARY_PATH}" \
  --profiles "${report_profiles[@]}" \
  --output-json "${LOG_DIR}/${BASE_EXP_PREFIX}_covariate_analysis.json" \
  --output-md "${LOG_DIR}/${BASE_EXP_PREFIX}_covariate_analysis.md"

echo "[done] covariate analysis report: ${LOG_DIR}/${BASE_EXP_PREFIX}_covariate_analysis.md"
