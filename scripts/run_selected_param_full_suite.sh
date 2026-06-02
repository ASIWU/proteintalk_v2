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
BASE_PREFIX="${BASE_PREFIX:-$(date +%Y%m%d)_selected_param_full_suite}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${LOG_DIR}/${BASE_PREFIX}_runtime_summary.tsv}"
RUN_REPORT="${RUN_REPORT:-1}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda:0}"
REPORT_INFER_BATCH_SIZE="${REPORT_INFER_BATCH_SIZE:-512}"
REPORT_NUM_WORKERS="${REPORT_NUM_WORKERS:-4}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

COMMON_ENV=(
  "USE_DOSE_COVARIATE=1"
  "DOSE_COVARIATE_FIELDS=pert_dose1 pert_dose2"
  "GPU_IDS=${GPU_IDS}"
  "FOLDS=${FOLDS}"
  "DEVICES=1"
  "STRATEGY=auto"
  "LOGGER_BACKEND=none"
  "LOG_TO_WANDB=0"
  "PROGRESS_BAR=${PROGRESS_BAR:-0}"
  "RUN_DATA_VALIDATION=${RUN_DATA_VALIDATION:-0}"
  "RUN_INFERENCE=1"
  "RUN_PREFLIGHT=${RUN_PREFLIGHT:-1}"
  "MODEL_TYPE=fast_delta"
  "HIDDEN_DIM=512"
  "EXPRESSION_LATENT_DIM=768"
  "COVARIATE_EMBEDDING_DIM=96"
  "INFER_BATCH_SIZE=256"
  "PRECISION=${PRECISION:-bf16-mixed}"
  "NUM_WORKERS=${NUM_WORKERS:-4}"
  "MAX_EPOCHS=${MAX_EPOCHS:-50}"
  "SAVE_TOP_K=1"
  "SAVE_LAST_CKPT=1"
  "BEST_CKPT_METRIC=valid_auprc"
  "GRAPH_FEATURE_MODE=real"
  "GRAPH_STRUCTURAL_RP=1"
  "GRAPH_DRUG_CONCAT=1"
  "GRAPH_LOGIT_SCALE=2.0"
  "PROTEIN_CONCAT_MODE=pcep"
  "PROTEIN_CONCAT_TOPK=512"
  "CKPT_DIR=${CKPT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}"
)

SINGLE_MSE075_DROP010=(
  "LEARNING_RATE=2e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.10"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.75"
)

CELLTYPE_CTRL_DROP020=(
  "LEARNING_RATE=2e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.20"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.25"
  "CONTROL_EXPRESSION_DROPOUT=0.10"
)

CELL_LR1E4=(
  "LEARNING_RATE=1e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.15"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.25"
)

DOUBLE_DBL_MSE010=(
  "LEARNING_RATE=2e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.15"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.25"
  "PAIR_FUSION_MODE=dual"
  "PAIR_TYPE_FEATURES=1"
  "MSE_INACTIVE_LABEL_WEIGHT=0.10"
  "USE_DDI=1"
  "GRAPH_PAIR_ADD_SCALE=0.5"
)

EXTRA_SINGLE_MSE050_LR1E4=(
  "LEARNING_RATE=1e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.15"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.50"
)

EXTRA_DOUBLE_LR3E4=(
  "LEARNING_RATE=3e-4"
  "BATCH_SIZE=256"
  "DROPOUT=0.15"
  "WEIGHT_DECAY=1e-4"
  "MSE_WEIGHT=0.25"
  "PAIR_FUSION_MODE=dual"
  "PAIR_TYPE_FEATURES=1"
  "MSE_INACTIVE_LABEL_WEIGHT=0.2"
  "USE_DDI=1"
  "GRAPH_PAIR_ADD_SCALE=0.5"
)

run_exp() {
  local exp_suffix="$1"
  local script_path="$2"
  local config_name="$3"
  shift 3
  local exp_prefix="${BASE_PREFIX}_${exp_suffix}"
  echo "[run] ${exp_suffix} config=${config_name} folds=${FOLDS}"
  env "${COMMON_ENV[@]}" "$@" "EXP_PREFIX=${exp_prefix}" bash "${script_path}"
}

echo "[settings] BASE_PREFIX=${BASE_PREFIX}; FOLDS=${FOLDS}; GPU_IDS=${GPU_IDS}"
"${PYTHON_BIN}" -m py_compile train.py infer.py scripts/report_cell_drug_time_eval.py scripts/report_ptv3_exp_results.py

run_exp exp01_single_pert_stratified_5fold scripts/exp_01_single_pert_stratified_5fold.sh mse075_drop010 "${SINGLE_MSE075_DROP010[@]}"
run_exp exp04_single_no_mse_5fold scripts/exp_04_single_no_mse_5fold.sh mse075_drop010 "${SINGLE_MSE075_DROP010[@]}"
run_exp exp05_single_no_graph_5fold scripts/exp_05_single_no_pdi_5fold.sh mse075_drop010 "${SINGLE_MSE075_DROP010[@]}"
run_exp exp02_single_cell_type_5fold scripts/exp_02_single_cell_type_5fold.sh ctrl_drop020 "${CELLTYPE_CTRL_DROP020[@]}"
run_exp exp03_single_cell_5fold scripts/exp_03_single_cell_5fold.sh lr1e4 "${CELL_LR1E4[@]}"
run_exp exp06_double_pert_pair_5fold scripts/exp_06_double_pert_pair_5fold.sh dbl_mse010 "${DOUBLE_DBL_MSE010[@]}"
run_exp exp07_extra_single_all_train_infer scripts/exp_07_extra_single_all_train_infer.sh mse050_lr1e4 "${EXTRA_SINGLE_MSE050_LR1E4[@]}"
run_exp exp08_extra_double_all_train_infer scripts/exp_08_extra_double_all_train_infer.sh lr3e4 "${EXTRA_DOUBLE_LR3E4[@]}"

if [[ "${RUN_REPORT}" == "1" ]]; then
  report_markdown="${LOG_DIR}/${BASE_PREFIX}_cell_drug_dose_time_eval.md"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" scripts/report_cell_drug_time_eval.py \
    --prefix "${BASE_PREFIX}" \
    --checkpoint-root "${CKPT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --materialize-fold-predictions \
    --device "${REPORT_DEVICE}" \
    --infer-batch-size "${REPORT_INFER_BATCH_SIZE}" \
    --num-workers "${REPORT_NUM_WORKERS}" \
    --csv-out "${OUTPUT_DIR}/${BASE_PREFIX}_cell_drug_dose_time_eval.csv" \
    --json-out "${OUTPUT_DIR}/${BASE_PREFIX}_cell_drug_dose_time_eval.json" \
    --format markdown | tee "${report_markdown}"
  echo "[report] ${report_markdown}"
fi

echo "[done] selected-parameter full suite completed for BASE_PREFIX=${BASE_PREFIX}"
