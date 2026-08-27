#!/usr/bin/env bash
set -euo pipefail

# Fold0 screen for random_saved control-expression artifacts in exp_04_v2.
# Run this inside the gpu2 tmux session; CPU artifact generation/reporting stays local.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d)_exp04_v2_random_expr_screen}"
TASK_NAME="${TASK_NAME:-ptv3_main_singledrug}"
TASK_HEAD="${TASK_HEAD:-response}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-pert_stratified_5fold_fold0}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${REPO_ROOT}/logs/${BASE_EXP_PREFIX}_runtime_summary.tsv}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-data/training_ready}"
TASK_DIR="${TASK_DIR:-${TRAINING_READY_ROOT}/ptv3/tasks/${TASK_NAME}}"

POLICIES=(
  real_control_full_nomse
  per_protein_normal_clip
  global_normal_clip
  global_value_bootstrap
  fixed_gene_permutation
  per_row_gene_permutation
  cross_cell_real_control
  zero_control
)

declare -A ARTIFACT_PATHS=(
  [per_protein_normal_clip]="${TASK_DIR}/random_control_expression_per_protein_normal_clip_seed42.npy"
  [global_normal_clip]="${TASK_DIR}/random_control_expression_global_normal_clip_seed42.npy"
  [global_value_bootstrap]="${TASK_DIR}/random_control_expression_global_value_bootstrap_seed42.npy"
  [fixed_gene_permutation]="${TASK_DIR}/random_control_expression_fixed_gene_permutation_seed42.npy"
  [per_row_gene_permutation]="${TASK_DIR}/random_control_expression_per_row_gene_permutation_seed42.npy"
  [cross_cell_real_control]="${TASK_DIR}/random_control_expression_cross_cell_real_control_seed42.npy"
  [zero_control]="${TASK_DIR}/random_control_expression_zero_control_seed42.npy"
)

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing required file: ${path}" >&2
    echo "[error] generate random-expression artifacts on CPU before launching this GPU screen" >&2
    exit 2
  fi
}

for policy in "${POLICIES[@]}"; do
  if [[ "${policy}" == "real_control_full_nomse" ]]; then
    continue
  fi
  require_file "${ARTIFACT_PATHS[${policy}]}"
done

run_one() {
  local policy="$1"
  local exp_name="${BASE_EXP_PREFIX}_f0_${policy}"
  local log_path="${LOG_DIR}/${exp_name}.log"
  local control_expression_mode="real"
  local random_control_path=""

  if [[ "${policy}" != "real_control_full_nomse" ]]; then
    control_expression_mode="random_saved"
    random_control_path="${ARTIFACT_PATHS[${policy}]}"
  fi

  if [[ "${ALLOW_EXISTING_RUN:-0}" != "1" && -e "${log_path}" ]]; then
    echo "[error] log file already exists: ${log_path}" >&2
    echo "[error] choose a new EXP_PREFIX or set ALLOW_EXISTING_RUN=1 intentionally" >&2
    return 1
  fi

  echo "[run] split=${SPLIT_STRATEGY} policy=${policy} exp=${exp_name} log=${log_path}"
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
    export PROTEIN_CONCAT_MODE=pcep
    export GRAPH_FEATURE_MODE=real
    export TARGET_PROTEIN_MAX_LENGTH=32
    export BATCH_COV_LIST=""
    export DRUG_EMBEDDING_PATH=""
    source "${SCRIPT_DIR}/ptv3_experiment_common.sh"
    ptv3_print_settings "exp04_v2 random-expression fold0 screen ${policy}"
    ptv3_run_preflight
    ptv3_train "${exp_name}" "${TASK_NAME}" "${SPLIT_STRATEGY}" "${TASK_HEAD}" --no-mse-loss
    ptv3_done
  ) >"${log_path}" 2>&1
  echo "[done] policy=${policy} exp=${exp_name}"
}

echo "[random-expr-screen] BASE_EXP_PREFIX=${BASE_EXP_PREFIX}"
echo "[random-expr-screen] TASK_NAME=${TASK_NAME}"
echo "[random-expr-screen] SPLIT_STRATEGY=${SPLIT_STRATEGY}"
echo "[random-expr-screen] logs=${LOG_DIR}"
echo "[random-expr-screen] time_summary=${TIME_SUMMARY_PATH}"

for policy in "${POLICIES[@]}"; do
  run_one "${policy}"
done

echo "[random-expr-screen] all requested fold0 policies completed"
