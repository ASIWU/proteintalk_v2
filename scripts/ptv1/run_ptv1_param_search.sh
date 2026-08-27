#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
BASE_PREFIX="${BASE_PREFIX:-$(date +%Y%m%d_%H%M)_ptv1_param}"
STAGES="${STAGES:-stage1}"
SEARCH_FOLDS="${SEARCH_FOLDS:-0 2 4}"
FULL_FOLDS="${FULL_FOLDS:-0 1 2 3 4}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/ptv1}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

COMMON_ENV=(
  "GPU_IDS=${GPU_IDS}"
  "DEVICES=1"
  "STRATEGY=auto"
  "LOGGER_BACKEND=none"
  "LOG_TO_WANDB=0"
  "PROGRESS_BAR=${PROGRESS_BAR:-0}"
  "RUN_DATA_VALIDATION=${RUN_DATA_VALIDATION:-0}"
  "RUN_INFERENCE=1"
  "RUN_PREFLIGHT=${RUN_PREFLIGHT:-0}"
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
  "GRAPH_CACHE_DIR=${GRAPH_CACHE_DIR}"
  "PROTEIN_CONCAT_MODE=pcep"
  "PROTEIN_CONCAT_TOPK=512"
  "USE_DOSE_COVARIATE=1"
  "DOSE_COVARIATE_FIELDS=pert_dose1 pert_dose2"
  "CELL_LLM_MODE=frozen"
  "CELL_LLM_EMBEDDING_PATH=data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz"
  "CKPT_DIR=${CKPT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
)

CONFIG_ENV=()
set_config_env() {
  local name="$1"
  CONFIG_ENV=()
  case "${name}" in
    baseline)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse025)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    mse010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.10")
      ;;
    mse000)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.00")
      ;;
    mse025_pos_auto)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=auto")
      ;;
    mse025_focal)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "FOCAL_LOSS=1" "FOCAL_GAMMA=2.0")
      ;;
    mse075)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.75")
      ;;
    mse075_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.75")
      ;;
    lr1e4_mse050)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    lr3e4_mse050)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_target_pdi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_TARGET_MODE=pdi")
      ;;
    mse050_target_ppi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_TARGET_MODE=pdi_ppi")
      ;;
    mse050_topvar_pdi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_TARGET_MODE=topvar_pdi")
      ;;
    mse025_inactive010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10")
      ;;
    mse025_rank005)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "RANKING_LOSS_WEIGHT=0.05")
      ;;
    mse025_graph_off)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "GRAPH_FEATURE_MODE=off" "GRAPH_STRUCTURAL_RP=0" "GRAPH_DRUG_CONCAT=0")
      ;;
    mse025_graph_logit1)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "GRAPH_LOGIT_SCALE=1.0")
      ;;
    mse025_graph_logit4)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "GRAPH_LOGIT_SCALE=4.0")
      ;;
    mse025_pcep_off)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "PROTEIN_CONCAT_MODE=off")
      ;;
    mse025_pcep_additive)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "PROTEIN_CONCAT_SCORE_MODE=additive")
      ;;
    mse025_no_plate_cov)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "BATCH_COV_LIST=Cell cell_type batch pert_time")
      ;;
    mse025_ctrl_drop005)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "CONTROL_EXPRESSION_DROPOUT=0.05")
      ;;
    mse025_target_expr_pdi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "TARGET_EXPRESSION_MODE=pdi")
      ;;
    pos_auto_mse050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "POSITIVE_WEIGHT=auto")
      ;;
    focal_mse050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "FOCAL_LOSS=1" "FOCAL_GAMMA=2.0")
      ;;
    *)
      echo "[error] unknown PTV1 config: ${name}" >&2
      exit 2
      ;;
  esac
}

run_exp12() {
  local stage="$1"
  local config="$2"
  local folds="$3"
  local run_prefix="${BASE_PREFIX}_${stage}_${config}"
  local time_summary="${LOG_DIR}/${run_prefix}_runtime_summary.tsv"
  set_config_env "${config}"
  echo "[run] stage=${stage} config=${config} folds=${folds}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "FOLDS=${folds}" \
    "EXP_PREFIX=${run_prefix}" \
    "TIME_SUMMARY_PATH=${time_summary}" \
    bash scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh
}

stage1() {
  local configs="${STAGE1_CONFIGS:-mse075_drop010 mse025 mse075 lr1e4_mse050 lr3e4_mse050 mse050_drop010 mse050_drop020 mse050_target_pdi mse050_target_ppi mse025_inactive010 mse025_rank005 pos_auto_mse050 focal_mse050}"
  local config
  for config in ${configs}; do
    run_exp12 stage1 "${config}" "${SEARCH_FOLDS}"
  done
}

stage2() {
  local configs="${STAGE2_CONFIGS:-mse075_drop010}"
  local config
  for config in ${configs}; do
    run_exp12 stage2 "${config}" "${FULL_FOLDS}"
  done
}

echo "[settings] BASE_PREFIX=${BASE_PREFIX}; STAGES=${STAGES}; SEARCH_FOLDS=${SEARCH_FOLDS}; FULL_FOLDS=${FULL_FOLDS}; GPU_IDS=${GPU_IDS}"
"${PYTHON_BIN}" -m py_compile train.py infer.py scripts/ptv1/report_ptv1_exp_results.py scripts/ptv1/report_ptv1_param_search.py
bash -n scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh scripts/ptv1/ptv1_experiment_common.sh

for stage in ${STAGES}; do
  case "${stage}" in
    stage1) stage1 ;;
    stage2) stage2 ;;
    all)
      stage1
      stage2
      ;;
    *)
      echo "[error] unknown stage: ${stage}" >&2
      exit 2
      ;;
  esac
done

echo "[done] PTV1 parameter search completed for BASE_PREFIX=${BASE_PREFIX}"
