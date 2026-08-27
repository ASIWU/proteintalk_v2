#!/usr/bin/env bash
set -euo pipefail

# Fold0 feature-attribution matrix for ptv3_main_singledrug fast_delta response.
# Run this inside the gpu2 tmux session; CPU artifact generation/reporting stays local.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d)_feature_attr}"
TASK_NAME="${TASK_NAME:-ptv3_main_singledrug}"
TASK_HEAD="${TASK_HEAD:-response}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${REPO_ROOT}/logs/${BASE_EXP_PREFIX}_runtime_summary.tsv}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-data/training_ready}"
RANDOM_CONTROL_EXPRESSION_PATH="${RANDOM_CONTROL_EXPRESSION_PATH:-data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_seed42.npy}"
ZERO_CONTROL_EXPRESSION_PATH="${ZERO_CONTROL_EXPRESSION_PATH:-data/training_ready/ptv3/tasks/ptv3_main_singledrug/zero_control_expression.npy}"
ZERO_DRUG_EMBEDDING_PATH="${ZERO_DRUG_EMBEDDING_PATH:-data/training_ready/ptv3/derived/drug_embedding_morgan_2048_zero.pkl}"
ZERO_MORGAN_GRAPH_CACHE_DIR="${ZERO_MORGAN_GRAPH_CACHE_DIR:-graph_cache/${BASE_EXP_PREFIX}_zero_morgan}"

VARIANTS=(
  full_mse
  full_nomse
  random_control
  no_control_pcep
  pcep_off
  cov_none
  graph_zero
  target_zero
  morgan_zero
  morgan_only
  graph_only
  target_only
  cov_only
  control_only
)

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}" "${ZERO_MORGAN_GRAPH_CACHE_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing required file: ${path}" >&2
    exit 2
  fi
}

require_file "${RANDOM_CONTROL_EXPRESSION_PATH}"
require_file "${ZERO_CONTROL_EXPRESSION_PATH}"
require_file "${ZERO_DRUG_EMBEDDING_PATH}"

run_one() {
  local panel="$1"
  local split_strategy="$2"
  local variant="$3"
  local exp_name="${BASE_EXP_PREFIX}_${panel}_f0_${variant}"
  local log_path="${LOG_DIR}/${exp_name}.log"
  local control_expression_mode="real"
  local random_control_path=""
  local protein_concat_mode="pcep"
  local graph_feature_mode="real"
  local target_protein_max_length="32"
  local batch_cov_list=""
  local drug_embedding_path=""
  local graph_cache_dir="${GRAPH_CACHE_DIR:-graph_cache}"
  local -a extra_train_args=(--no-mse-loss)

  case "${variant}" in
    full_mse)
      extra_train_args=()
      ;;
    full_nomse)
      ;;
    random_control)
      control_expression_mode="random_saved"
      random_control_path="${RANDOM_CONTROL_EXPRESSION_PATH}"
      ;;
    no_control_pcep)
      control_expression_mode="random_saved"
      random_control_path="${ZERO_CONTROL_EXPRESSION_PATH}"
      protein_concat_mode="off"
      ;;
    pcep_off)
      protein_concat_mode="off"
      ;;
    cov_none)
      batch_cov_list="__none__"
      ;;
    graph_zero)
      graph_feature_mode="zero"
      ;;
    target_zero)
      target_protein_max_length="0"
      ;;
    morgan_zero)
      drug_embedding_path="${ZERO_DRUG_EMBEDDING_PATH}"
      graph_cache_dir="${ZERO_MORGAN_GRAPH_CACHE_DIR}"
      ;;
    morgan_only)
      control_expression_mode="random_saved"
      random_control_path="${ZERO_CONTROL_EXPRESSION_PATH}"
      protein_concat_mode="off"
      graph_feature_mode="zero"
      target_protein_max_length="0"
      batch_cov_list="__none__"
      ;;
    graph_only)
      control_expression_mode="random_saved"
      random_control_path="${ZERO_CONTROL_EXPRESSION_PATH}"
      protein_concat_mode="off"
      target_protein_max_length="0"
      batch_cov_list="__none__"
      drug_embedding_path="${ZERO_DRUG_EMBEDDING_PATH}"
      graph_cache_dir="${ZERO_MORGAN_GRAPH_CACHE_DIR}"
      ;;
    target_only)
      control_expression_mode="random_saved"
      random_control_path="${ZERO_CONTROL_EXPRESSION_PATH}"
      protein_concat_mode="off"
      graph_feature_mode="zero"
      batch_cov_list="__none__"
      drug_embedding_path="${ZERO_DRUG_EMBEDDING_PATH}"
      ;;
    cov_only)
      control_expression_mode="random_saved"
      random_control_path="${ZERO_CONTROL_EXPRESSION_PATH}"
      protein_concat_mode="off"
      graph_feature_mode="zero"
      target_protein_max_length="0"
      drug_embedding_path="${ZERO_DRUG_EMBEDDING_PATH}"
      ;;
    control_only)
      graph_feature_mode="zero"
      target_protein_max_length="0"
      batch_cov_list="__none__"
      drug_embedding_path="${ZERO_DRUG_EMBEDDING_PATH}"
      ;;
    *)
      echo "[error] unknown variant: ${variant}" >&2
      return 2
      ;;
  esac

  if [[ "${ALLOW_EXISTING_RUN:-0}" != "1" && -e "${log_path}" ]]; then
    echo "[error] log file already exists: ${log_path}" >&2
    echo "[error] choose a new EXP_PREFIX or set ALLOW_EXISTING_RUN=1 intentionally" >&2
    return 1
  fi

  echo "[run] panel=${panel} split=${split_strategy} variant=${variant} exp=${exp_name} log=${log_path}"
  (
    export EXP_PREFIX="${BASE_EXP_PREFIX}"
    export GPU_IDS=0
    export DEVICES=1
    export LOGGER_BACKEND=none
    export LOG_TO_WANDB=0
    export WANDB_MODE=disabled
    export FOLDS=0
    export RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
    export RUN_DATA_VALIDATION="${RUN_DATA_VALIDATION:-0}"
    export RUN_INFERENCE=0
    export PROGRESS_BAR="${PROGRESS_BAR:-0}"
    export TASK_NAME
    export TRAINING_READY_ROOT
    export TIME_SUMMARY_PATH
    export LOG_DIR
    export CKPT_DIR
    export OUTPUT_DIR
    export MODEL_TYPE="${MODEL_TYPE:-fast_delta}"
    export CONTROL_EXPRESSION_MODE="${control_expression_mode}"
    export RANDOM_CONTROL_EXPRESSION_PATH="${random_control_path}"
    export PROTEIN_CONCAT_MODE="${protein_concat_mode}"
    export GRAPH_FEATURE_MODE="${graph_feature_mode}"
    export TARGET_PROTEIN_MAX_LENGTH="${target_protein_max_length}"
    export BATCH_COV_LIST="${batch_cov_list}"
    export DRUG_EMBEDDING_PATH="${drug_embedding_path}"
    export GRAPH_CACHE_DIR="${graph_cache_dir}"
    source "${SCRIPT_DIR}/ptv3_experiment_common.sh"
    ptv3_print_settings "feature attribution ${panel} fold0 ${variant}"
    ptv3_run_preflight
    ptv3_train "${exp_name}" "${TASK_NAME}" "${split_strategy}" "${TASK_HEAD}" "${extra_train_args[@]}"
    ptv3_done
  ) >"${log_path}" 2>&1
  echo "[done] panel=${panel} variant=${variant} exp=${exp_name}"
}

echo "[feature-attr] BASE_EXP_PREFIX=${BASE_EXP_PREFIX}"
echo "[feature-attr] TASK_NAME=${TASK_NAME}"
echo "[feature-attr] logs=${LOG_DIR}"
echo "[feature-attr] time_summary=${TIME_SUMMARY_PATH}"

for variant in "${VARIANTS[@]}"; do
  run_one "exp01" "pert_stratified_5fold_fold0" "${variant}"
done

for variant in "${VARIANTS[@]}"; do
  run_one "exp03" "cell_5fold_fold0" "${variant}"
done

echo "[feature-attr] all requested fold0 variants completed"
