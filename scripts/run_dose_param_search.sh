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

BASE_PREFIX="${BASE_PREFIX:-$(date +%Y%m%d)_dose_param_search}"
STAGES="${STAGES:-stage1}"
SEARCH_FOLDS="${SEARCH_FOLDS:-0 2 4}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

COMMON_ENV=(
  "USE_DOSE_COVARIATE=1"
  "DOSE_COVARIATE_FIELDS=pert_dose1 pert_dose2"
  "GPU_IDS=${GPU_IDS}"
  "FOLDS=${SEARCH_FOLDS}"
  "DEVICES=1"
  "STRATEGY=auto"
  "LOGGER_BACKEND=none"
  "LOG_TO_WANDB=0"
  "PROGRESS_BAR=0"
  "RUN_DATA_VALIDATION=0"
  "RUN_INFERENCE=1"
  "RUN_PREFLIGHT=0"
  "MODEL_TYPE=fast_delta"
  "HIDDEN_DIM=512"
  "EXPRESSION_LATENT_DIM=768"
  "COVARIATE_EMBEDDING_DIM=96"
  "INFER_BATCH_SIZE=256"
  "PRECISION=bf16-mixed"
  "NUM_WORKERS=4"
  "MAX_EPOCHS=50"
  "SAVE_TOP_K=1"
  "SAVE_LAST_CKPT=1"
  "BEST_CKPT_METRIC=valid_auprc"
  "GRAPH_FEATURE_MODE=real"
  "GRAPH_STRUCTURAL_RP=1"
  "GRAPH_DRUG_CONCAT=1"
  "GRAPH_LOGIT_SCALE=2.0"
  "PROTEIN_CONCAT_MODE=pcep"
  "PROTEIN_CONCAT_TOPK=512"
)

CONFIG_ENV=()
set_config_env() {
  local name="$1"
  CONFIG_ENV=()
  case "${name}" in
    base)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr1e4)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr1e4_drop010)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr1e4_drop020)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr1e4_mse050)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    lr1e4_mse075)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.75")
      ;;
    lr1e4_ctrl_drop010)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "CONTROL_EXPRESSION_DROPOUT=0.10")
      ;;
    lr1e4_covunk_cell)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_FOR_UNSEEN=1" "COVARIATE_UNK_FIELDS=Cell")
      ;;
    lr1e4_covunk_celltype)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_FOR_UNSEEN=1" "COVARIATE_UNK_FIELDS=cell_type")
      ;;
    lr1e4_covdrop010)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    lr3e4)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr3e4_drop010)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr3e4_drop020)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    lr3e4_bs128)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=128" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    bs128)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=128" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    bs128_drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=128" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    bs128_ctrl_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=128" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "CONTROL_EXPRESSION_DROPOUT=0.10")
      ;;
    drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25")
      ;;
    wd5e4)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=5e-4" "MSE_WEIGHT=0.25")
      ;;
    smooth002)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "LABEL_SMOOTHING=0.02")
      ;;
    mse050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse075)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.75")
      ;;
    mse075_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.75")
      ;;
    mse100)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=1.00")
      ;;
    mse100_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=1.00")
      ;;
    mse050_lr1e4)
      CONFIG_ENV=("LEARNING_RATE=1e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_lr3e4)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50")
      ;;
    mse050_gl3)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "GRAPH_LOGIT_SCALE=3.0")
      ;;
    mse050_warmdecay)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_WEIGHT_SCHEDULE=warmup_decay" "MSE_FINAL_WEIGHT_MULTIPLIER=0.25")
      ;;
    mse050_pre1)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_PRETRAIN_EPOCHS=1" "MSE_PRETRAIN_BCE_WEIGHT=0.0")
      ;;
    mse050_pre2)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_PRETRAIN_EPOCHS=2" "MSE_PRETRAIN_BCE_WEIGHT=0.0")
      ;;
    mse050_drop010_pre1)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_PRETRAIN_EPOCHS=1" "MSE_PRETRAIN_BCE_WEIGHT=0.0")
      ;;
    mse050_drop010_pre2)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_PRETRAIN_EPOCHS=2" "MSE_PRETRAIN_BCE_WEIGHT=0.0")
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
    mse050_drop010_target_pdi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_TARGET_MODE=pdi")
      ;;
    mse050_drop010_target_ppi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_TARGET_MODE=pdi_ppi")
      ;;
    mse050_drop010_topvar_pdi)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.10" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.50" "MSE_TARGET_MODE=topvar_pdi")
      ;;
    ctrl_drop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "CONTROL_EXPRESSION_DROPOUT=0.10")
      ;;
    ctrl_drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "CONTROL_EXPRESSION_DROPOUT=0.10")
      ;;
    covunk_cell)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_FOR_UNSEEN=1" "COVARIATE_UNK_FIELDS=Cell")
      ;;
    covunk_celltype)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_FOR_UNSEEN=1" "COVARIATE_UNK_FIELDS=cell_type")
      ;;
    covdrop010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "COVARIATE_UNK_DROPOUT=0.10")
      ;;
    dbl_mse010)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10")
      ;;
    dbl_mse010_lr3e4)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10")
      ;;
    lr3e4_dbl_mse010)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10")
      ;;
    lr3e4_dbl_mse050)
      CONFIG_ENV=("LEARNING_RATE=3e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.50")
      ;;
    dbl_mse010_bs128)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=128" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10")
      ;;
    dbl_mse010_drop020)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.20" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10")
      ;;
    dbl_mse010_rank005)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.10" "RANKING_LOSS_WEIGHT=0.05")
      ;;
    dbl_mse050)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "MSE_INACTIVE_LABEL_WEIGHT=0.50")
      ;;
    rank005)
      CONFIG_ENV=("LEARNING_RATE=2e-4" "BATCH_SIZE=256" "DROPOUT=0.15" "WEIGHT_DECAY=1e-4" "MSE_WEIGHT=0.25" "RANKING_LOSS_WEIGHT=0.05")
      ;;
    *)
      echo "[error] unknown config: ${name}" >&2
      exit 2
      ;;
  esac
}

run_exp() {
  local stage="$1"
  local config="$2"
  local script_path="$3"
  local exp_suffix="$4"
  shift 4
  local run_prefix="${BASE_PREFIX}_${stage}_${config}"
  local exp_prefix="${run_prefix}_${exp_suffix}"
  local time_summary="${LOG_DIR}/${run_prefix}_runtime_summary.tsv"
  set_config_env "${config}"
  echo "[run] stage=${stage} config=${config} script=${script_path} folds=${SEARCH_FOLDS}"
  env "${COMMON_ENV[@]}" "$@" "${CONFIG_ENV[@]}" \
    "EXP_PREFIX=${exp_prefix}" \
    "TIME_SUMMARY_PATH=${time_summary}" \
    bash "${script_path}"
}

stage1() {
  local configs="${STAGE1_CONFIGS:-base mse050 mse075 mse050_lr1e4 mse050_lr3e4 mse050_drop010 mse050_drop020 mse050_gl3 mse050_warmdecay ctrl_drop010}"
  local mode="${STAGE1_MODE:-full}"
  local config
  if [[ "${mode}" != "full" && "${mode}" != "screen" && "${mode}" != "ablations" ]]; then
    echo "[error] unknown STAGE1_MODE=${mode}; expected full, screen, or ablations" >&2
    exit 2
  fi
  for config in ${configs}; do
    if [[ "${mode}" == "full" || "${mode}" == "screen" ]]; then
      run_exp stage1 "${config}" scripts/exp_01_single_pert_stratified_5fold.sh exp01_single_pert_stratified_5fold
    fi
    if [[ "${mode}" == "full" || "${mode}" == "ablations" ]]; then
      run_exp stage1 "${config}" scripts/exp_04_single_no_mse_5fold.sh exp04_single_no_mse_5fold
      run_exp stage1 "${config}" scripts/exp_05_single_no_pdi_5fold.sh exp05_single_no_graph_5fold
    fi
  done
}

stage2() {
  local configs="${STAGE2_CONFIGS:-base lr1e4 lr3e4 bs128 drop010 drop020 covunk_cell covdrop010 ctrl_drop010}"
  local config
  for config in ${configs}; do
    run_exp stage2 "${config}" scripts/exp_03_single_cell_5fold.sh exp03_single_cell_5fold
  done
}

stage3() {
  local configs="${STAGE3_CONFIGS:-base lr1e4 lr3e4 bs128 drop010 drop020 covunk_celltype covdrop010 ctrl_drop010}"
  local config
  for config in ${configs}; do
    run_exp stage3 "${config}" scripts/exp_02_single_cell_type_5fold.sh exp02_single_cell_type_5fold
  done
}

stage4() {
  local configs="${STAGE4_CONFIGS:-base lr1e4 lr3e4 bs128 drop010 drop020 dbl_mse010 dbl_mse050 rank005}"
  local config
  for config in ${configs}; do
    run_exp stage4 "${config}" scripts/exp_06_double_pert_pair_5fold.sh exp06_double_pert_pair_5fold \
      "PAIR_FUSION_MODE=dual" "PAIR_TYPE_FEATURES=1" "MSE_INACTIVE_LABEL_WEIGHT=0.2" "USE_DDI=1" "GRAPH_PAIR_ADD_SCALE=0.5"
  done
}

stage5() {
  local configs="${STAGE5_CONFIGS:-base lr1e4 lr3e4 bs128 drop010 drop020 mse050 smooth002}"
  local config
  for config in ${configs}; do
    run_exp stage5 "${config}" scripts/exp_07_extra_single_all_train_infer.sh exp07_extra_single_all_train_infer
  done
}

stage6() {
  local configs="${STAGE6_CONFIGS:-base lr1e4 lr3e4 bs128 drop010 drop020 dbl_mse010 dbl_mse050 rank005}"
  local config
  for config in ${configs}; do
    run_exp stage6 "${config}" scripts/exp_08_extra_double_all_train_infer.sh exp08_extra_double_all_train_infer \
      "PAIR_FUSION_MODE=dual" "PAIR_TYPE_FEATURES=1" "MSE_INACTIVE_LABEL_WEIGHT=0.2" "USE_DDI=1" "GRAPH_PAIR_ADD_SCALE=0.5"
  done
}

echo "[settings] BASE_PREFIX=${BASE_PREFIX}; STAGES=${STAGES}; SEARCH_FOLDS=${SEARCH_FOLDS}; GPU_IDS=${GPU_IDS}"
"${PYTHON_BIN}" -m py_compile train.py infer.py scripts/report_ptv3_exp_results.py

for stage in ${STAGES}; do
  case "${stage}" in
    stage1) stage1 ;;
    stage2) stage2 ;;
    stage3) stage3 ;;
    stage4) stage4 ;;
    stage5) stage5 ;;
    stage6) stage6 ;;
    all)
      stage1
      stage2
      stage3
      stage4
      stage5
      stage6
      ;;
    *)
      echo "[error] unknown stage: ${stage}" >&2
      exit 2
      ;;
  esac
done

echo "[done] dose parameter search stages completed for BASE_PREFIX=${BASE_PREFIX}"
