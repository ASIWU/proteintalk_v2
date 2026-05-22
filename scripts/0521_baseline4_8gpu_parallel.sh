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

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${REPO_ROOT}/scripts/wandb_env.local}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${WANDB_ENV_FILE}"
  set +a
fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://100.96.30.112:8080}"
export WANDB_API_KEY="${WANDB_API_KEY:-local-f401d2b9276fb4a6dd1db4c1efee0512723ce6fb}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/wuhao/beam_wuhao/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/shared-storage-user/beam/wuhao/hf_cache}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/shared-storage-user/beam/wuhao/wandb_cache}"
export WANDB_ARTIFACT_CACHE="${WANDB_ARTIFACT_CACHE:-10GB}"

BASE_EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_baseline4_all}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
FOLDS="${FOLDS:-0 1 2 3 4}"
RUN_EXTRA_TASKS="${RUN_EXTRA_TASKS:-1}"

export MODEL_TYPE="${MODEL_TYPE:-fast_delta}"
export DEVICES="${DEVICES:-1}"
export STRATEGY="${STRATEGY:-auto}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-256}"
export MAX_EPOCHS="${MAX_EPOCHS:-50}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export MSE_WEIGHT="${MSE_WEIGHT:-0.25}"
export POSITIVE_WEIGHT="${POSITIVE_WEIGHT:-none}"
export LOGGER_BACKEND="${LOGGER_BACKEND:-wandb}"
export LOG_TO_WANDB="${LOG_TO_WANDB:-1}"
export PRECISION="${PRECISION:-bf16-mixed}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}"
export SAVE_TOP_K="${SAVE_TOP_K:-1}"
export SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}"
export RUN_PREFLIGHT=0
export RUN_DATA_VALIDATION="${RUN_DATA_VALIDATION:-0}"
export CKPT_DIR="${CKPT_DIR:-checkpoints}"
export LOG_DIR="${LOG_DIR:-logs}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
export TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${LOG_DIR}/${BASE_EXP_PREFIX}_runtime_summary.tsv}"

export GRAPH_FEATURE_MODE="${GRAPH_FEATURE_MODE:-real}"
export GRAPH_FEATURE_DIM="${GRAPH_FEATURE_DIM:-128}"
export GRAPH_FEATURE_SEED="${GRAPH_FEATURE_SEED:-17}"
export GRAPH_STRUCTURAL_RP="${GRAPH_STRUCTURAL_RP:-1}"
export GRAPH_MULTIHOP="${GRAPH_MULTIHOP:-0}"
export GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache}"
export FORCE_GRAPH_CACHE_REBUILD="${FORCE_GRAPH_CACHE_REBUILD:-0}"
export GRAPH_DRUG_CONCAT="${GRAPH_DRUG_CONCAT:-1}"
export GRAPH_PAIR_ADD_SCALE="${GRAPH_PAIR_ADD_SCALE:-0.0}"
export GRAPH_LOGIT_SCALE="${GRAPH_LOGIT_SCALE:-2.0}"
export GRAPH_JUMP_FUSION="${GRAPH_JUMP_FUSION:-concat}"
export GRAPH_JUMP_GATE="${GRAPH_JUMP_GATE:-softmax}"
export GRAPH_JUMP_TEMPERATURE="${GRAPH_JUMP_TEMPERATURE:-1.0}"
export PROTEIN_CONCAT_MODE="${PROTEIN_CONCAT_MODE:-pcep}"
export PROTEIN_CONCAT_DIM="${PROTEIN_CONCAT_DIM:-64}"
export PROTEIN_CONCAT_TOPK="${PROTEIN_CONCAT_TOPK:-512}"
export PROTEIN_CONCAT_INIT_SCALE="${PROTEIN_CONCAT_INIT_SCALE:-0.1}"
export PROTEIN_CONCAT_SEED="${PROTEIN_CONCAT_SEED:-23}"

SINGLE_PAIR_FUSION_MODE="${SINGLE_PAIR_FUSION_MODE:-symmetric}"
SINGLE_PAIR_TYPE_FEATURES="${SINGLE_PAIR_TYPE_FEATURES:-0}"
SINGLE_MSE_INACTIVE_LABEL_WEIGHT="${SINGLE_MSE_INACTIVE_LABEL_WEIGHT:-1.0}"
SINGLE_USE_DDI="${SINGLE_USE_DDI:-0}"
SINGLE_GRAPH_PAIR_ADD_SCALE="${SINGLE_GRAPH_PAIR_ADD_SCALE:-0.0}"

DOUBLE_PAIR_FUSION_MODE="${DOUBLE_PAIR_FUSION_MODE:-dual}"
DOUBLE_PAIR_TYPE_FEATURES="${DOUBLE_PAIR_TYPE_FEATURES:-1}"
DOUBLE_MSE_INACTIVE_LABEL_WEIGHT="${DOUBLE_MSE_INACTIVE_LABEL_WEIGHT:-0.2}"
DOUBLE_USE_DDI="${DOUBLE_USE_DDI:-1}"
DOUBLE_GRAPH_PAIR_ADD_SCALE="${DOUBLE_GRAPH_PAIR_ADD_SCALE:-0.5}"

echo "[task-config] single: PAIR_FUSION_MODE=${SINGLE_PAIR_FUSION_MODE}; PAIR_TYPE_FEATURES=${SINGLE_PAIR_TYPE_FEATURES}; MSE_INACTIVE_LABEL_WEIGHT=${SINGLE_MSE_INACTIVE_LABEL_WEIGHT}; USE_DDI=${SINGLE_USE_DDI}; GRAPH_PAIR_ADD_SCALE=${SINGLE_GRAPH_PAIR_ADD_SCALE}"
echo "[task-config] double: PAIR_FUSION_MODE=${DOUBLE_PAIR_FUSION_MODE}; PAIR_TYPE_FEATURES=${DOUBLE_PAIR_TYPE_FEATURES}; MSE_INACTIVE_LABEL_WEIGHT=${DOUBLE_MSE_INACTIVE_LABEL_WEIGHT}; USE_DDI=${DOUBLE_USE_DDI}; GRAPH_PAIR_ADD_SCALE=${DOUBLE_GRAPH_PAIR_ADD_SCALE}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < 1 )); then
  echo "[error] GPU_IDS is empty" >&2
  exit 2
fi

echo "[preflight] compiling train/infer/model scripts"
"${PYTHON_BIN}" -m py_compile \
  train.py \
  infer.py \
  dataset/training_ready_fast_dataset.py \
  model/fast_delta_model.py \
  model/fast_lightning.py \
  model/graph_feature_utils.py \
  model/training_ready_models.py \
  model/training_ready_lightning.py \
  scripts/check_wandb_auth.py \
  scripts/prebuild_graph_cache.py \
  scripts/select_reference_epoch.py

"${PYTHON_BIN}" scripts/check_wandb_auth.py \
  --logger-backend "${LOGGER_BACKEND}" \
  --log-to-wandb "${LOG_TO_WANDB}" \
  --wandb-mode "${WANDB_MODE:-}" \
  --wandb-env-file "${WANDB_ENV_FILE}"

if [[ "${RUN_DATA_VALIDATION}" == "1" ]]; then
  "${PYTHON_BIN}" utils/01_validate_standardized_outputs.py
  "${PYTHON_BIN}" utils/03_validate_training_ready_outputs.py
fi

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

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
  if [[ "${FORCE_GRAPH_CACHE_REBUILD}" == "1" ]]; then
    prebuild_args+=(--force-graph-cache-rebuild)
  fi
  echo "[preflight] prebuilding graph cache before parallel workers"
  "${PYTHON_BIN}" scripts/prebuild_graph_cache.py "${prebuild_args[@]}"
else
  echo "[preflight] skipping graph cache prebuild for MODEL_TYPE=${MODEL_TYPE}, GRAPH_FEATURE_MODE=${GRAPH_FEATURE_MODE}"
fi

task_specs=()
for fold in ${FOLDS}; do
  task_specs+=("scripts/exp_01_single_pert_stratified_5fold.sh ${BASE_EXP_PREFIX}_exp01_single_pert_stratified_5fold ${fold}")
  task_specs+=("scripts/exp_02_single_cell_type_5fold.sh ${BASE_EXP_PREFIX}_exp02_single_cell_type_5fold ${fold}")
  task_specs+=("scripts/exp_03_single_cell_5fold.sh ${BASE_EXP_PREFIX}_exp03_single_cell_5fold ${fold}")
  task_specs+=("scripts/exp_04_single_no_mse_5fold.sh ${BASE_EXP_PREFIX}_exp04_single_no_mse_5fold ${fold}")
  task_specs+=("scripts/exp_05_single_no_pdi_5fold.sh ${BASE_EXP_PREFIX}_exp05_single_no_graph_5fold ${fold}")
  task_specs+=("scripts/exp_06_double_pert_pair_5fold.sh ${BASE_EXP_PREFIX}_exp06_double_pert_pair_5fold ${fold}")
done

run_fold_task() {
  local gpu_id="$1"
  local script_path="$2"
  local exp_prefix="$3"
  local fold="$4"
  echo "[run][gpu=${gpu_id}] ${script_path} EXP_PREFIX=${exp_prefix} FOLD=${fold}"
  if [[ "${script_path}" == *"exp_06_double_pert_pair_5fold.sh" ]]; then
    GPU_IDS="${gpu_id}" \
    EXP_PREFIX="${exp_prefix}" \
    FOLDS="${fold}" \
    PAIR_FUSION_MODE="${DOUBLE_PAIR_FUSION_MODE}" \
    PAIR_TYPE_FEATURES="${DOUBLE_PAIR_TYPE_FEATURES}" \
    MSE_INACTIVE_LABEL_WEIGHT="${DOUBLE_MSE_INACTIVE_LABEL_WEIGHT}" \
    USE_DDI="${DOUBLE_USE_DDI}" \
    GRAPH_PAIR_ADD_SCALE="${DOUBLE_GRAPH_PAIR_ADD_SCALE}" \
    bash "${script_path}"
  else
    GPU_IDS="${gpu_id}" \
    EXP_PREFIX="${exp_prefix}" \
    FOLDS="${fold}" \
    PAIR_FUSION_MODE="${SINGLE_PAIR_FUSION_MODE}" \
    PAIR_TYPE_FEATURES="${SINGLE_PAIR_TYPE_FEATURES}" \
    MSE_INACTIVE_LABEL_WEIGHT="${SINGLE_MSE_INACTIVE_LABEL_WEIGHT}" \
    USE_DDI="${SINGLE_USE_DDI}" \
    GRAPH_PAIR_ADD_SCALE="${SINGLE_GRAPH_PAIR_ADD_SCALE}" \
    bash "${script_path}"
  fi
  echo "[done][gpu=${gpu_id}] ${script_path} EXP_PREFIX=${exp_prefix} FOLD=${fold}"
}

worker() {
  local slot="$1"
  local gpu_id="${GPU_ARRAY[$slot]}"
  local i spec script_path exp_prefix fold
  for ((i = slot; i < ${#task_specs[@]}; i += ${#GPU_ARRAY[@]})); do
    spec="${task_specs[$i]}"
    script_path="$(awk '{print $1}' <<< "${spec}")"
    exp_prefix="$(awk '{print $2}' <<< "${spec}")"
    fold="$(awk '{print $3}' <<< "${spec}")"
    run_fold_task "${gpu_id}" "${script_path}" "${exp_prefix}" "${fold}"
  done
}

echo "[stage1] running ${#task_specs[@]} fold tasks across ${#GPU_ARRAY[@]} GPU workers"
pids=()
for ((slot = 0; slot < ${#GPU_ARRAY[@]}; slot += 1)); do
  worker "${slot}" &
  pids+=("$!")
done
stage1_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "[error] stage1 worker failed: pid=${pid}" >&2
    stage1_status=1
  fi
done
if [[ "${stage1_status}" -ne 0 ]]; then
  echo "[error] one or more stage1 workers failed; runtime summary: ${TIME_SUMMARY_PATH}" >&2
  exit "${stage1_status}"
fi
echo "[stage1] completed fold tasks"

if [[ "${RUN_EXTRA_TASKS}" != "1" ]]; then
  echo "[done] RUN_EXTRA_TASKS=0; runtime summary: ${TIME_SUMMARY_PATH}"
  exit 0
fi

single_ref="${CKPT_DIR}/${BASE_EXP_PREFIX}_exp01_single_pert_stratified_5fold"
double_ref="${CKPT_DIR}/${BASE_EXP_PREFIX}_exp06_double_pert_pair_5fold"

echo "[stage2] running extra all-train/infer tasks"
GPU_IDS="${GPU_ARRAY[0]}" \
EXP_PREFIX="${BASE_EXP_PREFIX}_exp07_extra_single_all_train_infer" \
REFERENCE_5FOLD_CKPT_PATH="${single_ref}" \
REFERENCE_SPLIT_STRATEGY_REGEX='^pert_stratified_5fold_fold[0-9]+$' \
PAIR_FUSION_MODE="${SINGLE_PAIR_FUSION_MODE}" \
PAIR_TYPE_FEATURES="${SINGLE_PAIR_TYPE_FEATURES}" \
MSE_INACTIVE_LABEL_WEIGHT="${SINGLE_MSE_INACTIVE_LABEL_WEIGHT}" \
USE_DDI="${SINGLE_USE_DDI}" \
GRAPH_PAIR_ADD_SCALE="${SINGLE_GRAPH_PAIR_ADD_SCALE}" \
bash scripts/exp_07_extra_single_all_train_infer.sh &
pid_single="$!"

second_gpu="${GPU_ARRAY[0]}"
if (( ${#GPU_ARRAY[@]} > 1 )); then
  second_gpu="${GPU_ARRAY[1]}"
fi
GPU_IDS="${second_gpu}" \
EXP_PREFIX="${BASE_EXP_PREFIX}_exp08_extra_double_all_train_infer" \
REFERENCE_5FOLD_CKPT_PATH="${double_ref}" \
REFERENCE_SPLIT_STRATEGY_REGEX='^pert_id_5fold_fold[0-9]+$' \
PAIR_FUSION_MODE="${DOUBLE_PAIR_FUSION_MODE}" \
PAIR_TYPE_FEATURES="${DOUBLE_PAIR_TYPE_FEATURES}" \
MSE_INACTIVE_LABEL_WEIGHT="${DOUBLE_MSE_INACTIVE_LABEL_WEIGHT}" \
USE_DDI="${DOUBLE_USE_DDI}" \
GRAPH_PAIR_ADD_SCALE="${DOUBLE_GRAPH_PAIR_ADD_SCALE}" \
bash scripts/exp_08_extra_double_all_train_infer.sh &
pid_double="$!"

stage2_status=0
if ! wait "${pid_single}"; then
  echo "[error] stage2 extra single task failed: pid=${pid_single}" >&2
  stage2_status=1
fi
if ! wait "${pid_double}"; then
  echo "[error] stage2 extra double task failed: pid=${pid_double}" >&2
  stage2_status=1
fi
if [[ "${stage2_status}" -ne 0 ]]; then
  echo "[error] one or more stage2 tasks failed; runtime summary: ${TIME_SUMMARY_PATH}" >&2
  exit "${stage2_status}"
fi

echo "[done] runtime summary: ${TIME_SUMMARY_PATH}"
