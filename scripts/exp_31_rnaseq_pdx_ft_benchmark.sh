#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="exp31_rnaseq_pdx_ft_benchmark"

# Exp31 fine-tunes the selected exp09 unified single+double checkpoint on
# BRCA PDX RNA-seq baseline expression, then evaluates zero-shot on non-BRCA
# tumor types. Keep these defaults aligned with exp09 unless explicitly
# overridden by the launch command.
PAIR_FUSION_MODE="${PAIR_FUSION_MODE:-dual}"
PAIR_TYPE_FEATURES="${PAIR_TYPE_FEATURES:-1}"
MSE_INACTIVE_LABEL_WEIGHT="${MSE_INACTIVE_LABEL_WEIGHT:-0.2}"
USE_DDI="${USE_DDI:-1}"
GRAPH_PAIR_ADD_SCALE="${GRAPH_PAIR_ADD_SCALE:-0.5}"
SAVE_TOP_K="${SAVE_TOP_K:-1}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}"
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}"
MONITOR="${MONITOR:-val/task_auprc}"
MONITOR_MODE="${MONITOR_MODE:-max}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
USE_DOSE_COVARIATE="${USE_DOSE_COVARIATE:-1}"
BATCH_COV_LIST="${BATCH_COV_LIST:-machineID_new Cell_plate Cell cell_type batch pert_time pert_dose1 pert_dose2}"
CELL_LLM_MODE="${CELL_LLM_MODE:-frozen}"
CELL_LLM_FUSION_MODE="${CELL_LLM_FUSION_MODE:-covariate}"
CELL_TYPE_LLM_MODE="${CELL_TYPE_LLM_MODE:-frozen}"
CELL_TYPE_LLM_FUSION_MODE="${CELL_TYPE_LLM_FUSION_MODE:-covariate}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/exp31_rnaseq}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"

EXP31_INIT_CKPT="${EXP31_INIT_CKPT:-checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/last.ckpt}"
TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-data/training_ready_exp31_rnaseq}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-${TRAINING_READY_ROOT}/ptv3/derived/cell_llm_embedding_exp31_rnaseq_qwen3_4096.npz}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-${TRAINING_READY_ROOT}/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

RUN_ZEROSHOT="${RUN_ZEROSHOT:-1}"
RUN_REPORT="${RUN_REPORT:-0}"
EXP31_SPLIT_STRATEGY="${EXP31_SPLIT_STRATEGY:-brca_ft_valid_nonbrca_test}"
EXP31_CELL_LLM_INDEX_COLUMN="${EXP31_CELL_LLM_INDEX_COLUMN:-cell_llm_index}"

EXP31_SUFFIXES=(
  sensitive_early
  sensitive_late
  disease_control_early
  disease_control_late
)
EXP31_TASKS=(
  ptv3_exp31_rnaseq_sensitive_early
  ptv3_exp31_rnaseq_sensitive_late
  ptv3_exp31_rnaseq_disease_control_early
  ptv3_exp31_rnaseq_disease_control_late
)

ptv3_print_settings "Exp31 RNA-seq PDX BRCA fine-tune and non-BRCA benchmark"

if [[ ! -f "${EXP31_INIT_CKPT}" ]]; then
  echo "[error] missing exp31 init checkpoint: ${EXP31_INIT_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${CELL_LLM_EMBEDDING_PATH}" ]]; then
  echo "[error] missing exp31 cell LLM embedding: ${CELL_LLM_EMBEDDING_PATH}" >&2
  echo "[hint] build it on CPU with utils/31_build_exp31_cell_llm_embeddings.py" >&2
  exit 1
fi
if [[ ! -f "${TRAINING_READY_ROOT}/ptv3/global_meta.json" ]]; then
  echo "[error] missing exp31 training-ready root: ${TRAINING_READY_ROOT}/ptv3" >&2
  echo "[hint] build it on CPU with utils/31_build_exp31_rnaseq_training_ready.py" >&2
  exit 1
fi

ptv3_run_preflight
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON_BIN}" -m py_compile scripts/report_exp31_rnaseq_pdx_ft_benchmark.py
fi

exp31_infer() {
  local checkpoint_path="$1"
  local task_name="$2"
  local exp_name="$3"
  local -a limit_args=()
  if [[ -n "${INFER_LIMIT_BATCHES}" ]]; then
    limit_args=(--limit-batches "${INFER_LIMIT_BATCHES}")
  fi

  ptv3_ensure_clean_path "${OUTPUT_DIR}/${exp_name}/${task_name}" "inference output directory"

  local start_utc
  local end_utc
  local start_sec
  local end_sec
  local status
  start_utc="$(ptv3_utc_now)"
  start_sec="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u infer.py \
    "${COMMON_INFER_ARGS[@]}" \
    --task-name "${task_name}" \
    --split-strategy "${EXP31_SPLIT_STRATEGY}" \
    --split-name test \
    --task-head unified \
    --checkpoint-path "${checkpoint_path}" \
    --output-dir "${OUTPUT_DIR}/${exp_name}/${task_name}" \
    --batch-size "${INFER_BATCH_SIZE}" \
    --device "${INFER_DEVICE}" \
    --cell-llm-index-column "${EXP31_CELL_LLM_INDEX_COLUMN}" \
    --allow-checkpoint-config-mismatch \
    "${limit_args[@]}"
  status="$?"
  set -e
  end_utc="$(ptv3_utc_now)"
  end_sec="$(date +%s)"

  ptv3_record_time \
    "infer" \
    "${exp_name}" \
    "${task_name}" \
    "${EXP31_SPLIT_STRATEGY}" \
    "test" \
    "${status}" \
    "${start_utc}" \
    "${end_utc}" \
    "$((end_sec - start_sec))" \
    "${OUTPUT_DIR}/${exp_name}/${task_name}"

  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

for idx in "${!EXP31_SUFFIXES[@]}"; do
  suffix="${EXP31_SUFFIXES[$idx]}"
  task_name="${EXP31_TASKS[$idx]}"
  train_exp="${EXP_PREFIX}_${suffix}_ft"
  echo "[exp31] fine-tune ${task_name} -> ${CKPT_DIR}/${train_exp}"
  ptv3_train "${train_exp}" \
    "${task_name}" "${EXP31_SPLIT_STRATEGY}" unified \
    --checkpoint-path "${EXP31_INIT_CKPT}" \
    --no-mse-loss \
    --cell-llm-index-column "${EXP31_CELL_LLM_INDEX_COLUMN}"

  if [[ "${RUN_INFERENCE}" == "1" ]]; then
    ft_ckpt="$(ptv3_best_checkpoint "${train_exp}")"
    exp31_infer "${ft_ckpt}" "${task_name}" "${EXP_PREFIX}_${suffix}_ft_infer"
    if [[ "${RUN_ZEROSHOT}" == "1" ]]; then
      exp31_infer "${EXP31_INIT_CKPT}" "${task_name}" "${EXP_PREFIX}_${suffix}_exp09_zeroshot"
    fi
  fi
done

if [[ "${RUN_REPORT}" == "1" && "${RUN_INFERENCE}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/report_exp31_rnaseq_pdx_ft_benchmark.py \
    --exp-prefix "${EXP_PREFIX}" \
    --training-ready-root "${TRAINING_READY_ROOT}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --init-checkpoint "${EXP31_INIT_CKPT}" \
    --output-markdown "docs/${OUTPUT_DATE}_exp31_rnaseq_pdx_ft_benchmark_results.md" \
    --output-csv "${OUTPUT_DIR}/${EXP_PREFIX}_pdx_ft_benchmark_summary.csv" \
    --output-json "${OUTPUT_DIR}/${EXP_PREFIX}_pdx_ft_benchmark_summary.json"
fi

ptv3_done
