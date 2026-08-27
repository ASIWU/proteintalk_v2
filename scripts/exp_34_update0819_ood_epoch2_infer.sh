#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="${EXPERIMENT_SET_NAME:-exp34_update0819_ood_epoch2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Architecture is fixed to the selected exp09 epoch-2 manifest.
MODEL_TYPE="fast_delta"
HIDDEN_DIM="512"
EXPRESSION_LATENT_DIM="768"
COVARIATE_EMBEDDING_DIM="96"
NUM_HEADS="8"
NUM_LAYERS="4"
DROPOUT="0.15"
CONTROL_LAYERS="2"
FUSION_LAYERS="3"
TARGET_LAYERS="2"
TARGET_PROTEIN_MAX_LENGTH="32"

GRAPH_FEATURE_MODE="real"
GRAPH_FEATURE_DIM="128"
GRAPH_FEATURE_SEED="17"
GRAPH_STRUCTURAL_RP="1"
GRAPH_MULTIHOP="0"
GRAPH_DRUG_CONCAT="1"
GRAPH_LAYERS="2"
GRAPH_INIT_SCALE="0.1"
GRAPH_PAIR_ADD_SCALE="0.5"
GRAPH_LOGIT_SCALE="2.0"
GRAPH_JUMP_FUSION="concat"
GRAPH_JUMP_GATE="softmax"
GRAPH_JUMP_TEMPERATURE="1.0"
PAIR_FUSION_MODE="dual"
PAIR_TYPE_FEATURES="1"
CELL_PAIR_FILM_SCALE="0.0"

TARGET_EXPRESSION_MODE="off"
TARGET_EXPRESSION_DIM="64"
TARGET_EXPRESSION_TOPK="256"
TARGET_EXPRESSION_PPI_TOPK="32"
TARGET_EXPRESSION_PPI_ALPHA="0.5"
TARGET_EXPRESSION_PPI_NORM="raw"
TARGET_EXPRESSION_DEGREE_PENALTY="0.0"
TARGET_EXPRESSION_INIT_SCALE="0.1"
TARGET_EXPRESSION_SEED="29"
TARGET_EXPRESSION_FUSION_MODE="piece"
TARGET_EXPRESSION_CELL_GATE_MODE="off"
TARGET_EXPRESSION_CELL_GATE_SCALE="0.0"
TARGET_EXPRESSION_CELL_GATE_TEMPERATURE="1.0"

PROTEIN_CONCAT_MODE="pcep"
PROTEIN_CONCAT_DIM="64"
PROTEIN_CONCAT_TOPK="512"
PROTEIN_CONCAT_INIT_SCALE="0.1"
PROTEIN_CONCAT_SEED="23"
PROTEIN_CONCAT_SCORE_MODE="multiply"
PROTEIN_CONCAT_EXPR_SCALE="1.0"

CONTROL_LOGIT_SCALE="0.0"
PAIR_LOGIT_SCALE="0.0"
TARGET_LOGIT_SCALE="0.0"
COVARIATE_LOGIT_SCALE="0.0"
RESPONSE_BASE_LOGIT_SCALE="1.0"
RESPONSE_DELTA_MODE="off"
RESPONSE_DELTA_DIM="64"
RESPONSE_DELTA_SEED="31"
RESPONSE_DELTA_DETACH="0"
DELTA_LOGIT_SCALE="0.0"
DELTA_LOGIT_LEARNABLE="0"
RESPONSE_TRAJECTORY_MODE="off"
RESPONSE_TRAJECTORY_DIM="64"
RESPONSE_TRAJECTORY_SEED="37"
RESPONSE_TRAJECTORY_DETACH="0"
TRAJECTORY_LOGIT_SCALE="0.0"
TRAJECTORY_LOGIT_LEARNABLE="0"

CELL_LLM_MODE="frozen"
CELL_LLM_FUSION_MODE="covariate"
CELL_LLM_CONDITION_SCALE="0.0"
CELL_LLM_LOGIT_SCALE="0.0"
CELL_LLM_DROPOUT="0.0"
CELL_TYPE_LLM_MODE="frozen"
CELL_TYPE_LLM_FUSION_MODE="covariate"
CELL_TYPE_LLM_CONDITION_SCALE="0.0"
CELL_TYPE_LLM_LOGIT_SCALE="0.0"
CELL_TYPE_LLM_DROPOUT="0.0"

CONTROL_DRUG_INTERACTION_MODE="off"
CONTROL_DRUG_INTERACTION_SCALE="0.0"
CONTROL_DRUG_LOGIT_SCALE="0.0"
OBSERVED_PERTURB_EXPRESSION_MODE="off"
OBSERVED_PERTURB_EXPRESSION_SCALE="0.0"
OBSERVED_PERTURB_LOGIT_SCALE="0.0"
CELL_PRIOR_MODE="off"
CELL_PRIOR_K="8"
CELL_PRIOR_TEMPERATURE="0.2"
CELL_PRIOR_LOGIT_SCALE="0.0"
CELL_PRIOR_FIXED_LOGIT_SCALE="0.0"

BATCH_COV_LIST="machineID_new Cell_plate Cell cell_type batch pert_time pert_dose1 pert_dose2"
USE_DOSE_COVARIATE="1"
DOSE_COVARIATE_FIELDS="pert_dose1 pert_dose2"
USE_DDI="1"
ZERO_INIT_DELTA_HEAD="0"
CONTROL_EXPRESSION_DROPOUT="0.0"
CONTROL_EXPRESSION_MODE="real"
RANDOM_CONTROL_EXPRESSION_PATH=""

TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-/tmp/proteintalk_exp34_update0819_ood_runtime}"
DERIVED_ROOT="${TRAINING_READY_ROOT}/ptv3/derived"
PROTEIN_EMBEDDING_PATH="${DERIVED_ROOT}/protein_embedding_esm.pkl"
DRUG_EMBEDDING_PATH="${DERIVED_ROOT}/drug_embedding_morgan_2048.pkl"
PPI_MATRIX_PATH="${DERIVED_ROOT}/ppi_matrix.npy"
PDI_MATRIX_PATH="${DERIVED_ROOT}/pdi_matrix.npy"
DDI_MATRIX_PATH="${DERIVED_ROOT}/ddi_matrix.npy"
CELL_LLM_EMBEDDING_PATH="${DERIVED_ROOT}/cell_llm_embedding_qwen3_4096.npz"
CELL_TYPE_LLM_EMBEDDING_PATH="${DERIVED_ROOT}/cell_type_llm_embedding_qwen3_4096_v2.npz"
GRAPH_CACHE_DIR="${TRAINING_READY_ROOT}/ptv3/graph_cache"

EXP34_CHECKPOINT="${EXP34_CHECKPOINT:-${REPO_ROOT}/checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt}"
EXPECTED_CHECKPOINT_SHA256="7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076"
EXP_PREFIX="${EXP_PREFIX:-20260819_exp34_update0819_ood_epoch2}"
GPU_IDS="${GPU_IDS:-0}"
INFER_DEVICE="${INFER_DEVICE:-cuda:0}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
NUM_WORKERS="0"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
LOGGER_BACKEND="none"
LOG_TO_WANDB="0"
WANDB_MODE="disabled"
PROGRESS_BAR="${PROGRESS_BAR:-0}"
RUN_DATA_VALIDATION="0"
RUN_BUILD="${RUN_BUILD:-1}"
REQUIRE_GPU_TMUX="${REQUIRE_GPU_TMUX:-1}"
REQUIRED_TMUX_TARGET="${REQUIRED_TMUX_TARGET:-gpu1_deep:0}"
EXP34_BUILDER_SCRIPT="${EXP34_BUILDER_SCRIPT:-utils/34_build_update0819_ood_training_ready.py}"
EXP34_BUILD_SUMMARY_REL="${EXP34_BUILD_SUMMARY_REL:-ptv3/exp34_update0819_ood_build_summary.json}"
EXP34_TASKS_STRING="${EXP34_TASKS_STRING:-ptv3_exp34_update0819_ood_target_none ptv3_exp34_update0819_ood_target_mechanism}"

if [[ -z "${OUTPUT_DATE:-}" ]]; then
  if [[ "${EXP_PREFIX}" =~ ^([0-9]{4})([0-9]{2})([0-9]{2}) ]]; then
    OUTPUT_DATE="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-${BASH_REMATCH[3]}"
  else
    OUTPUT_DATE="$(date +%F)"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${OUTPUT_DATE:0:7}/${OUTPUT_DATE}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${LOG_DIR}/${EXP_PREFIX}_runtime_summary.tsv}"
RUN_LOG_PATH="${RUN_LOG_PATH:-${LOG_DIR}/${EXP_PREFIX}.log}"

mkdir -p "${LOG_DIR}"
if [[ "${EXP34_LOG_INITIALIZED:-0}" != "1" ]]; then
  export EXP34_LOG_INITIALIZED=1
  exec > >(tee -a "${RUN_LOG_PATH}") 2>&1
fi

# shellcheck source=scripts/ptv3_experiment_common.sh
source "${SCRIPT_DIR}/ptv3_experiment_common.sh"

if [[ "${REQUIRE_GPU_TMUX}" == "1" ]]; then
  # rlaunch worker shells do not inherit the outer tmux socket.  The launcher
  # therefore passes EXP34_TMUX_TARGET after resolving the outer target.
  current_tmux_target="${EXP34_TMUX_TARGET:-$(tmux display-message -p '#S:#I' 2>/dev/null || true)}"
  if [[ "${current_tmux_target}" != "${REQUIRED_TMUX_TARGET}" ]]; then
    echo "[error] ${EXPERIMENT_SET_NAME} GPU inference must run in tmux ${REQUIRED_TMUX_TARGET}; current=${current_tmux_target:-none}" >&2
    exit 1
  fi
fi

if [[ "${INFER_DEVICE}" == cuda:* ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[error] ${EXPERIMENT_SET_NAME} requested ${INFER_DEVICE}, but no NVIDIA GPU/driver is visible" >&2
    exit 1
  fi
  if ! "${PYTHON_BIN}" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[error] ${EXPERIMENT_SET_NAME} requested ${INFER_DEVICE}, but torch.cuda.is_available() is false" >&2
    exit 1
  fi
fi

actual_checkpoint_sha256="$(sha256sum "${EXP34_CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_checkpoint_sha256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
  echo "[error] checkpoint SHA-256 mismatch: ${actual_checkpoint_sha256}" >&2
  exit 1
fi

if [[ "${RUN_BUILD}" == "1" && ! -f "${TRAINING_READY_ROOT}/${EXP34_BUILD_SUMMARY_REL}" ]]; then
  echo "[${EXPERIMENT_SET_NAME}] build runtime copy-on-write root: ${TRAINING_READY_ROOT}"
  "${PYTHON_BIN}" "${EXP34_BUILDER_SCRIPT}" \
    --output-root "${TRAINING_READY_ROOT}" \
    --checkpoint "${EXP34_CHECKPOINT}"
fi
"${PYTHON_BIN}" "${EXP34_BUILDER_SCRIPT}" \
  --output-root "${TRAINING_READY_ROOT}" \
  --checkpoint "${EXP34_CHECKPOINT}" \
  --preflight-only

read -r -a EXP34_TASKS <<< "${EXP34_TASKS_STRING}"
if [[ "${#EXP34_TASKS[@]}" -eq 0 ]]; then
  echo "[error] no inference task configured" >&2
  exit 1
fi

ptv3_print_settings "${EXPERIMENT_SET_NAME} single-drug inference"
echo "[${EXPERIMENT_SET_NAME}] checkpoint=${EXP34_CHECKPOINT}"
echo "[${EXPERIMENT_SET_NAME}] checkpoint_sha256=${actual_checkpoint_sha256}"
echo "[${EXPERIMENT_SET_NAME}] runtime_training_ready=${TRAINING_READY_ROOT}"
echo "[${EXPERIMENT_SET_NAME}] output=${OUTPUT_DIR}/${EXP_PREFIX}"
echo "[${EXPERIMENT_SET_NAME}] required_tmux=${REQUIRED_TMUX_TARGET}"

ptv3_ensure_clean_path "${OUTPUT_DIR}/${EXP_PREFIX}" "${EXPERIMENT_SET_NAME} inference output directory"

required_files=(
  "${EXP34_CHECKPOINT}"
  "$(dirname "${EXP34_CHECKPOINT}")/run_manifest.json"
  "${TRAINING_READY_ROOT}/ptv3/global_meta.json"
  "${PROTEIN_EMBEDDING_PATH}"
  "${DRUG_EMBEDDING_PATH}"
  "${PPI_MATRIX_PATH}"
  "${PDI_MATRIX_PATH}"
  "${DDI_MATRIX_PATH}"
  "${CELL_LLM_EMBEDDING_PATH}"
  "${CELL_TYPE_LLM_EMBEDDING_PATH}"
  "${GRAPH_CACHE_DIR}/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy"
  "${GRAPH_CACHE_DIR}/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.meta.json"
)
for task_name in "${EXP34_TASKS[@]}"; do
  task_dir="${TRAINING_READY_ROOT}/ptv3/tasks/${task_name}"
  split_dir="${TRAINING_READY_ROOT}/ptv3/splits/${task_name}"
  required_files+=(
    "${task_dir}/feature_table.parquet"
    "${task_dir}/feature_expression_matrix.npy"
    "${task_dir}/feature_ordered_protein_index.json"
    "${task_dir}/feature_ordered_protein_uniprot.json"
    "${task_dir}/feature_sample_ids.json"
    "${split_dir}/test_indices_test_only.pkl"
    "${split_dir}/test_set_info_test_only.pkl"
    "${split_dir}/row_to_set_index.pkl"
    "${split_dir}/set_info.pkl"
    "${split_dir}/split_manifest.json"
  )
done
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
      echo "[error] missing required ${EXPERIMENT_SET_NAME} artifact: ${required_file}" >&2
    exit 1
  fi
done

ptv3_run_preflight

exp34_infer() {
  local task_name="$1"
  local output_path="${OUTPUT_DIR}/${EXP_PREFIX}/${task_name}"
  local -a limit_args=()
  if [[ -n "${INFER_LIMIT_BATCHES}" ]]; then
    limit_args=(--limit-batches "${INFER_LIMIT_BATCHES}")
  fi
  ptv3_ensure_clean_path "${output_path}" "${EXPERIMENT_SET_NAME} task output directory"
  local start_utc end_utc start_sec end_sec status
  start_utc="$(ptv3_utc_now)"
  start_sec="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u infer.py \
    "${COMMON_INFER_ARGS[@]}" \
    --task-name "${task_name}" \
    --split-strategy test_only \
    --split-name test \
    --task-head unified \
    --checkpoint-path "${EXP34_CHECKPOINT}" \
    --output-dir "${output_path}" \
    --batch-size "${INFER_BATCH_SIZE}" \
    --device "${INFER_DEVICE}" \
    --num-workers 0 \
    --protein-embedding-path "${PROTEIN_EMBEDDING_PATH}" \
    --drug-embedding-path "${DRUG_EMBEDDING_PATH}" \
    --ppi-matrix-path "${PPI_MATRIX_PATH}" \
    --pdi-matrix-path "${PDI_MATRIX_PATH}" \
    --ddi-matrix-path "${DDI_MATRIX_PATH}" \
    --cell-llm-embedding-path "${CELL_LLM_EMBEDDING_PATH}" \
    --cell-llm-index-column cell_llm_index \
    --cell-type-llm-embedding-path "${CELL_TYPE_LLM_EMBEDDING_PATH}" \
    --cell-type-llm-index-column cell_type_llm_index \
    --use-target \
    --init-delta-scale 0.1 \
    --save-expression-pred \
    --allow-checkpoint-config-mismatch \
    "${limit_args[@]}"
  status="$?"
  set -e
  end_utc="$(ptv3_utc_now)"
  end_sec="$(date +%s)"
  ptv3_record_time \
    "infer" "${EXP_PREFIX}" "${task_name}" "test_only" "test" "${status}" \
    "${start_utc}" "${end_utc}" "$((end_sec - start_sec))" "${output_path}"
  if [[ "${status}" -ne 0 ]]; then
    echo "[error] ${task_name} inference failed" >&2
    exit "${status}"
  fi
}

for task_name in "${EXP34_TASKS[@]}"; do
  echo "[${EXPERIMENT_SET_NAME}] infer ${task_name}"
  exp34_infer "${task_name}"
done

ptv3_done
echo "[done] run log: ${RUN_LOG_PATH}"
