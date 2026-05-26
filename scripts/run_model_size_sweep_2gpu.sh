#!/usr/bin/env bash
set -euo pipefail

# Explore fast_delta model capacity on unseen-drug and unseen-cell 5-fold splits.
# Default task settings are intentionally task-specific:
#   - unseen_drug: baseline4 single-drug setting, MSE_WEIGHT=0.25
#   - unseen_cell: current stronger covariate-UNK setting, MSE_WEIGHT=0.075

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_model_size_sweep}"
GPU_IDS="${GPU_IDS:-0,1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
TASKS="${TASKS:-unseen_drug unseen_cell}"
SIZE_PROFILES="${SIZE_PROFILES:-h192:192:256:32 h256:256:384:48 h384:384:512:64 h512:512:768:96 h768:768:1024:128}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

export MODEL_TYPE="${MODEL_TYPE:-fast_delta}"
export DEVICES="${DEVICES:-1}"
export STRATEGY="${STRATEGY:-auto}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
export MAX_EPOCHS="${MAX_EPOCHS:-50}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export PRECISION="${PRECISION:-bf16-mixed}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}"
export LOGGER_BACKEND="${LOGGER_BACKEND:-none}"
export LOG_TO_WANDB="${LOG_TO_WANDB:-0}"
export PROGRESS_BAR="${PROGRESS_BAR:-0}"
export RUN_PREFLIGHT=0
export RUN_INFERENCE=0
export SAVE_TOP_K="${SAVE_TOP_K:-1}"
export SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}"
export ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
export CKPT_DIR
export LOG_DIR
export OUTPUT_DIR

export GRAPH_FEATURE_MODE="${GRAPH_FEATURE_MODE:-real}"
export GRAPH_FEATURE_DIM="${GRAPH_FEATURE_DIM:-128}"
export GRAPH_FEATURE_SEED="${GRAPH_FEATURE_SEED:-17}"
export GRAPH_STRUCTURAL_RP="${GRAPH_STRUCTURAL_RP:-1}"
export GRAPH_MULTIHOP="${GRAPH_MULTIHOP:-0}"
export GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache}"
export GRAPH_DRUG_CONCAT="${GRAPH_DRUG_CONCAT:-1}"
export GRAPH_PAIR_ADD_SCALE="${GRAPH_PAIR_ADD_SCALE:-0.0}"
export GRAPH_LOGIT_SCALE="${GRAPH_LOGIT_SCALE:-2.0}"
export GRAPH_JUMP_FUSION="${GRAPH_JUMP_FUSION:-concat}"
export GRAPH_JUMP_GATE="${GRAPH_JUMP_GATE:-softmax}"
export GRAPH_JUMP_TEMPERATURE="${GRAPH_JUMP_TEMPERATURE:-1.0}"
export PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-pcep}"
export PROTEIN_CONCAT_DIM="${PROTEIN_CONCAT_DIM:-64}"
export PROTEIN_CONCAT_TOPK="${PROTEIN_CONCAT_TOPK:-512}"
export PAIR_FUSION_MODE="${PAIR_FUSION_MODE:-symmetric}"
export PAIR_TYPE_FEATURES="${PAIR_TYPE_FEATURES:-0}"
export MSE_INACTIVE_LABEL_WEIGHT="${MSE_INACTIVE_LABEL_WEIGHT:-1.0}"
export USE_DDI="${USE_DDI:-0}"

DRUG_MSE_WEIGHT="${DRUG_MSE_WEIGHT:-0.25}"
DRUG_COVARIATE_UNK_FOR_UNSEEN="${DRUG_COVARIATE_UNK_FOR_UNSEEN:-0}"
DRUG_COVARIATE_UNK_DROPOUT="${DRUG_COVARIATE_UNK_DROPOUT:-0.0}"
DRUG_COVARIATE_UNK_FIELDS="${DRUG_COVARIATE_UNK_FIELDS:-}"
DRUG_BATCH_COV_LIST="${DRUG_BATCH_COV_LIST:-}"

CELL_MSE_WEIGHT="${CELL_MSE_WEIGHT:-0.075}"
CELL_COVARIATE_UNK_FOR_UNSEEN="${CELL_COVARIATE_UNK_FOR_UNSEEN:-1}"
CELL_COVARIATE_UNK_DROPOUT="${CELL_COVARIATE_UNK_DROPOUT:-0.15}"
CELL_COVARIATE_UNK_FIELDS="${CELL_COVARIATE_UNK_FIELDS:-}"
CELL_BATCH_COV_LIST="${CELL_BATCH_COV_LIST:-}"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 2
fi
read -r -a FOLD_ARRAY <<< "${FOLDS}"
read -r -a TASK_ARRAY <<< "${TASKS}"
read -r -a PROFILE_ARRAY <<< "${SIZE_PROFILES}"

"${PYTHON_BIN}" -m py_compile \
  train.py \
  infer.py \
  dataset/training_ready_fast_dataset.py \
  model/fast_delta_model.py \
  model/fast_lightning.py \
  model/graph_feature_utils.py \
  scripts/prebuild_graph_cache.py \
  scripts/model_size_sweep_report.py

if [[ "${MODEL_TYPE}" == "fast_delta" && "${GRAPH_FEATURE_MODE}" != "off" ]]; then
  prebuild_args=(
    --dataset-group ptv3
    --graph-cache-dir "${GRAPH_CACHE_DIR}"
    --graph-feature-dim "${GRAPH_FEATURE_DIM}"
    --graph-feature-seed "${GRAPH_FEATURE_SEED}"
    --summary-json "${LOG_DIR}/${BASE_EXP_PREFIX}_graph_cache_summary.json"
  )
  if [[ "${GRAPH_STRUCTURAL_RP}" == "1" ]]; then
    prebuild_args+=(--graph-structural-rp)
  fi
  if [[ "${GRAPH_MULTIHOP}" == "1" ]]; then
    prebuild_args+=(--graph-multihop)
  fi
  echo "[prebuild] graph cache"
  "${PYTHON_BIN}" scripts/prebuild_graph_cache.py "${prebuild_args[@]}"
fi

task_specs=()
for profile in "${PROFILE_ARRAY[@]}"; do
  IFS=':' read -r profile_name hidden_dim expression_latent_dim covariate_embedding_dim <<< "${profile}"
  if [[ -z "${profile_name}" || -z "${hidden_dim}" || -z "${expression_latent_dim}" || -z "${covariate_embedding_dim}" ]]; then
    echo "[error] bad SIZE_PROFILES entry: ${profile}" >&2
    exit 2
  fi
  for task_name in "${TASK_ARRAY[@]}"; do
    for fold in "${FOLD_ARRAY[@]}"; do
      task_specs+=("${task_name}:${profile_name}:${hidden_dim}:${expression_latent_dim}:${covariate_embedding_dim}:${fold}")
    done
  done
done

run_one() {
  local gpu_id="$1"
  local spec="$2"
  local task_name profile_name hidden_dim expression_latent_dim covariate_embedding_dim fold
  IFS=':' read -r task_name profile_name hidden_dim expression_latent_dim covariate_embedding_dim fold <<< "${spec}"

  local script_path exp_prefix log_path mse_weight cov_unk cov_dropout cov_fields cov_list
  if [[ "${task_name}" == "unseen_drug" ]]; then
    script_path="scripts/exp_01_single_pert_stratified_5fold.sh"
    mse_weight="${DRUG_MSE_WEIGHT}"
    cov_unk="${DRUG_COVARIATE_UNK_FOR_UNSEEN}"
    cov_dropout="${DRUG_COVARIATE_UNK_DROPOUT}"
    cov_fields="${DRUG_COVARIATE_UNK_FIELDS}"
    cov_list="${DRUG_BATCH_COV_LIST}"
  elif [[ "${task_name}" == "unseen_cell" ]]; then
    script_path="scripts/exp_03_single_cell_5fold.sh"
    mse_weight="${CELL_MSE_WEIGHT}"
    cov_unk="${CELL_COVARIATE_UNK_FOR_UNSEEN}"
    cov_dropout="${CELL_COVARIATE_UNK_DROPOUT}"
    cov_fields="${CELL_COVARIATE_UNK_FIELDS}"
    cov_list="${CELL_BATCH_COV_LIST}"
  else
    echo "[error] unknown task: ${task_name}" >&2
    return 2
  fi

  exp_prefix="${BASE_EXP_PREFIX}_${task_name}_${profile_name}_fold${fold}"
  log_path="${LOG_DIR}/${exp_prefix}.log"
  echo "[run][gpu=${gpu_id}] task=${task_name} profile=${profile_name} hidden=${hidden_dim} expr=${expression_latent_dim} cov=${covariate_embedding_dim} fold=${fold}"

  GPU_IDS="${gpu_id}" \
  EXP_PREFIX="${exp_prefix}" \
  FOLDS="${fold}" \
  HIDDEN_DIM="${hidden_dim}" \
  EXPRESSION_LATENT_DIM="${expression_latent_dim}" \
  COVARIATE_EMBEDDING_DIM="${covariate_embedding_dim}" \
  MSE_WEIGHT="${mse_weight}" \
  COVARIATE_UNK_FOR_UNSEEN="${cov_unk}" \
  COVARIATE_UNK_DROPOUT="${cov_dropout}" \
  COVARIATE_UNK_FIELDS="${cov_fields}" \
  BATCH_COV_LIST="${cov_list}" \
  bash "${script_path}" > "${log_path}" 2>&1

  echo "[done][gpu=${gpu_id}] task=${task_name} profile=${profile_name} fold=${fold}; log=${log_path}"
}

worker() {
  local slot="$1"
  local gpu_id="${GPU_ARRAY[$slot]}"
  local i
  for ((i = slot; i < ${#task_specs[@]}; i += ${#GPU_ARRAY[@]})); do
    run_one "${gpu_id}" "${task_specs[$i]}"
  done
}

echo "[stage] running ${#task_specs[@]} jobs across ${#GPU_ARRAY[@]} GPU workers"
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
  echo "[error] at least one model-size sweep job failed; inspect ${LOG_DIR}/${BASE_EXP_PREFIX}_*.log" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" scripts/model_size_sweep_report.py \
  --exp-prefix "${BASE_EXP_PREFIX}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --markdown-out "${LOG_DIR}/${BASE_EXP_PREFIX}_model_size_report.md" \
  --json-out "${LOG_DIR}/${BASE_EXP_PREFIX}_model_size_report.json"

echo "[done] model-size sweep complete; report=${LOG_DIR}/${BASE_EXP_PREFIX}_model_size_report.md"
