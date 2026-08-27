#!/usr/bin/env bash

# PTV1 experiment wrapper. It reuses the maintained PTV3 mechanics after
# forcing PTV1-specific defaults and dataset-group arguments.

PTV1_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MSE_WEIGHT="${MSE_WEIGHT:-0.50}"
USE_DOSE_COVARIATE="${USE_DOSE_COVARIATE:-1}"
CELL_LLM_MODE="${CELL_LLM_MODE:-frozen}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz}"
CELL_TYPE_LLM_MODE="${CELL_TYPE_LLM_MODE:-frozen}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz}"
GRAPH_FEATURE_MODE="${GRAPH_FEATURE_MODE:-real}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/ptv1}"
MODEL_TYPE="${MODEL_TYPE:-fast_delta}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DROPOUT="${DROPOUT:-0.15}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# shellcheck source=../ptv3_experiment_common.sh
source "${PTV1_SCRIPT_DIR}/../ptv3_experiment_common.sh"

ptv1_patch_dataset_group_array() {
  local array_name="$1"
  local -n args_ref="${array_name}"
  local idx
  for ((idx = 0; idx < ${#args_ref[@]} - 1; idx++)); do
    if [[ "${args_ref[$idx]}" == "--dataset-group" ]]; then
      args_ref[$((idx + 1))]="ptv1"
    fi
  done
}

ptv1_patch_dataset_group_array COMMON_TRAIN_ARGS
ptv1_patch_dataset_group_array COMMON_INFER_ARGS

ptv1_print_settings() {
  ptv3_print_settings "$@"
  echo "[settings] DATASET_GROUP=ptv1; GRAPH_CACHE_DIR=${GRAPH_CACHE_DIR}; CELL_LLM_MODE=${CELL_LLM_MODE}; CELL_TYPE_LLM_MODE=${CELL_TYPE_LLM_MODE}"
}

ptv1_run_preflight() {
  if [[ "${RUN_PREFLIGHT}" != "1" ]]; then
    return
  fi
  "${PYTHON_BIN}" -m py_compile \
    train.py \
    infer.py \
    dataset/training_ready_dataset.py \
    dataset/training_ready_fast_dataset.py \
    model/fast_delta_model.py \
    model/fast_lightning.py \
    model/graph_feature_utils.py \
    model/training_ready_models.py \
    model/training_ready_lightning.py \
    scripts/check_wandb_auth.py \
    scripts/select_reference_epoch.py \
    scripts/ptv1/report_ptv1_exp_results.py \
    scripts/ptv1/report_ptv1_fine_tune_results.py \
    utils/ptv1/00_standardize_ptv1_rawdata.py \
    utils/ptv1/01_validate_ptv1_standardized.py \
    utils/ptv1/02_build_ptv1_training_ready.py \
    utils/ptv1/03_validate_ptv1_training_ready.py \
    utils/ptv1/04_build_ptv1_splits.py \
    utils/ptv1/05_build_ptv1_embeddings.py \
    utils/ptv1/06_build_ptv1_graph_matrices.py \
    utils/ptv1/07_build_ptv1_cell_llm_embeddings.py \
    utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py \
    utils/ptv1/09_bootstrap_ptv1_llm_embeddings_from_local.py \
    utils/11_build_cell_llm_embeddings.py \
    utils/11_build_cell_type_llm_embeddings.py
  if [[ "${RUN_DATA_VALIDATION}" == "1" ]]; then
    "${PYTHON_BIN}" utils/ptv1/01_validate_ptv1_standardized.py
    "${PYTHON_BIN}" utils/ptv1/03_validate_ptv1_training_ready.py
    "${PYTHON_BIN}" utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only
    "${PYTHON_BIN}" utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py --validate-only
  fi
  "${PYTHON_BIN}" scripts/check_wandb_auth.py \
    --logger-backend "${LOGGER_BACKEND}" \
    --log-to-wandb "${LOG_TO_WANDB}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-env-file "${WANDB_ENV_FILE}"
}

ptv1_train() {
  ptv3_train "$@"
}

ptv1_best_checkpoint() {
  ptv3_best_checkpoint "$@"
}

ptv1_last_checkpoint() {
  ptv3_last_checkpoint "$@"
}

ptv1_reference_epoch() {
  local reference_path="$1"
  local task_name="$2"
  local task_head="$3"
  local split_strategy_regex="$4"
  local summary_json="$5"
  local -a reference_args=(
    "${reference_path}"
    --task-name "${task_name}"
    --expect-task-head "${task_head}"
    --expect-model-type "${MODEL_TYPE}"
    --expect-dataset-group ptv1
    --method "${REFERENCE_EPOCH_AGG}"
    --rounding "${REFERENCE_EPOCH_ROUNDING}"
    --min-count "${REFERENCE_EPOCH_MIN_COUNT}"
    --summary-json "${summary_json}"
  )
  if [[ -n "${split_strategy_regex}" ]]; then
    reference_args+=(--split-strategy-regex "${split_strategy_regex}")
  fi
  if [[ "${REFERENCE_REQUIRE_TEST_COMPLETED}" == "1" ]]; then
    reference_args+=(--require-test-completed)
  fi
  if [[ "${REFERENCE_ALLOW_MIXED_CONFIG}" == "1" ]]; then
    reference_args+=(--allow-mixed-reference-config)
  fi
  if [[ "${REFERENCE_ALLOW_DUPLICATE_SPLITS}" == "1" ]]; then
    reference_args+=(--allow-duplicate-split-strategies)
  fi
  "${PYTHON_BIN}" scripts/select_reference_epoch.py "${reference_args[@]}"
}

ptv1_record_reference_epoch_policy() {
  ptv3_record_reference_epoch_policy "$@"
}

ptv1_infer() {
  ptv3_infer "$@"
}

ptv1_done() {
  ptv3_done
}
