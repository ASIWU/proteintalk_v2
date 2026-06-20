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
BASE_PREFIX="${BASE_PREFIX:-20260609_ptv1_cell_celltype_llm_tune_v1}"
EXP11_CANDIDATES="${EXP11_CANDIDATES:-baseline_mse050 mse025 mse075 mse050_drop010 mse050_lr1e4 mse050_lr3e4 mse025_drop010 mse075_drop010 mse050_target_pdi mse050_target_ppi pos_auto_mse050 focal_mse050 mse050_label_smooth005}"
EXP12_CANDIDATES="${EXP12_CANDIDATES:-baseline_mse050 mse025 mse075 mse050_drop010 mse050_lr1e4 mse050_lr3e4 mse025_drop010 mse075_drop010 mse050_target_pdi mse050_target_ppi pos_auto_mse050 focal_mse050 mse050_label_smooth005}"
EXP12_FOLDS="${EXP12_FOLDS:-0 1 2 3 4}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/ptv1_cell_celltype}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz}"
RUN_EXP11="${RUN_EXP11:-1}"
RUN_EXP12="${RUN_EXP12:-1}"
RUN_EXP13_VALID="${RUN_EXP13_VALID:-1}"
RUN_EXP13_ORACLE="${RUN_EXP13_ORACLE:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
REPORT_AFTER="${REPORT_AFTER:-1}"
RUN_DATA_VALIDATION="${RUN_DATA_VALIDATION:-1}"
EXPECTED_EXTRA_ROWS="${EXPECTED_EXTRA_ROWS:-218}"
EXP11_SAVE_TOP_K="${EXP11_SAVE_TOP_K:--1}"
EXP12_SAVE_TOP_K="${EXP12_SAVE_TOP_K:-1}"
REPORT_MD_PATH="${REPORT_MD_PATH:-logs/${BASE_PREFIX}_cell_celltype_tune_results.md}"
REPORT_TSV_PATH="${REPORT_TSV_PATH:-outputs/${BASE_PREFIX}_cell_celltype_tune_results.tsv}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

COMMON_ENV=(
  "GPU_IDS=${GPU_IDS}"
  "DEVICES=1"
  "STRATEGY=auto"
  "LOGGER_BACKEND=none"
  "LOG_TO_WANDB=0"
  "PROGRESS_BAR=${PROGRESS_BAR:-0}"
  "RUN_DATA_VALIDATION=0"
  "RUN_INFERENCE=1"
  "RUN_PREFLIGHT=0"
  "MODEL_TYPE=fast_delta"
  "HIDDEN_DIM=512"
  "EXPRESSION_LATENT_DIM=768"
  "COVARIATE_EMBEDDING_DIM=96"
  "BATCH_SIZE=${BATCH_SIZE:-256}"
  "INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-256}"
  "PRECISION=${PRECISION:-bf16-mixed}"
  "NUM_WORKERS=${NUM_WORKERS:-4}"
  "MAX_EPOCHS=${MAX_EPOCHS:-50}"
  "SAVE_LAST_CKPT=1"
  "BEST_CKPT_METRIC=valid_auprc"
  "LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-1.0}"
  "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-1.0}"
  "LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES:-1.0}"
  "INFER_LIMIT_BATCHES=${INFER_LIMIT_BATCHES:-}"
  "GRAPH_FEATURE_MODE=real"
  "GRAPH_STRUCTURAL_RP=1"
  "GRAPH_DRUG_CONCAT=1"
  "GRAPH_LOGIT_SCALE=2.0"
  "GRAPH_CACHE_DIR=${GRAPH_CACHE_DIR}"
  "PROTEIN_CONCAT_MODE=pcep"
  "PROTEIN_CONCAT_TOPK=512"
  "PROTEIN_CONCAT_SCORE_MODE=multiply"
  "USE_DOSE_COVARIATE=1"
  "DOSE_COVARIATE_FIELDS=pert_dose1 pert_dose2"
  "CELL_LLM_MODE=frozen"
  "CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
  "CELL_LLM_FUSION_MODE=${CELL_LLM_FUSION_MODE:-covariate}"
  "CELL_LLM_CONDITION_SCALE=${CELL_LLM_CONDITION_SCALE:-0.0}"
  "CELL_LLM_LOGIT_SCALE=${CELL_LLM_LOGIT_SCALE:-0.0}"
  "CELL_LLM_DROPOUT=${CELL_LLM_DROPOUT:-0.0}"
  "CELL_TYPE_LLM_MODE=frozen"
  "CELL_TYPE_LLM_EMBEDDING_PATH=${CELL_TYPE_LLM_EMBEDDING_PATH}"
  "CELL_TYPE_LLM_FUSION_MODE=${CELL_TYPE_LLM_FUSION_MODE:-covariate}"
  "CELL_TYPE_LLM_CONDITION_SCALE=${CELL_TYPE_LLM_CONDITION_SCALE:-0.0}"
  "CELL_TYPE_LLM_LOGIT_SCALE=${CELL_TYPE_LLM_LOGIT_SCALE:-0.0}"
  "CELL_TYPE_LLM_DROPOUT=${CELL_TYPE_LLM_DROPOUT:-0.0}"
  "CKPT_DIR=${CKPT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
)

CONFIG_ENV=()
set_config_env() {
  local name="$1"
  CONFIG_ENV=(
    "LEARNING_RATE=2e-4"
    "BATCH_SIZE=${BATCH_SIZE:-256}"
    "DROPOUT=0.15"
    "WEIGHT_DECAY=1e-4"
    "MSE_WEIGHT=0.50"
    "MSE_TARGET_MODE=all"
    "POSITIVE_WEIGHT=none"
    "POSITIVE_LABEL_SAMPLING_WEIGHT=1.0"
    "FOCAL_LOSS=0"
    "LABEL_SMOOTHING=0.0"
    "GRAPH_FEATURE_MODE=real"
    "GRAPH_STRUCTURAL_RP=1"
    "GRAPH_DRUG_CONCAT=1"
    "GRAPH_LOGIT_SCALE=2.0"
    "PROTEIN_CONCAT_MODE=pcep"
    "PROTEIN_CONCAT_SCORE_MODE=multiply"
    "TARGET_EXPRESSION_MODE=off"
    "CONTROL_EXPRESSION_DROPOUT=0.0"
    "BATCH_COV_LIST="
  )
  case "${name}" in
    baseline_mse050|mse050) ;;
    mse000) CONFIG_ENV+=("MSE_WEIGHT=0.00") ;;
    mse010) CONFIG_ENV+=("MSE_WEIGHT=0.10") ;;
    mse025) CONFIG_ENV+=("MSE_WEIGHT=0.25") ;;
    mse075) CONFIG_ENV+=("MSE_WEIGHT=0.75") ;;
    mse100) CONFIG_ENV+=("MSE_WEIGHT=1.00") ;;
    mse025_lr1e4) CONFIG_ENV+=("MSE_WEIGHT=0.25" "LEARNING_RATE=1e-4") ;;
    mse025_lr3e4) CONFIG_ENV+=("MSE_WEIGHT=0.25" "LEARNING_RATE=3e-4") ;;
    mse050_lr1e4) CONFIG_ENV+=("LEARNING_RATE=1e-4") ;;
    mse050_lr3e4) CONFIG_ENV+=("LEARNING_RATE=3e-4") ;;
    mse075_lr1e4) CONFIG_ENV+=("MSE_WEIGHT=0.75" "LEARNING_RATE=1e-4") ;;
    mse075_lr3e4) CONFIG_ENV+=("MSE_WEIGHT=0.75" "LEARNING_RATE=3e-4") ;;
    mse025_drop010) CONFIG_ENV+=("MSE_WEIGHT=0.25" "DROPOUT=0.10") ;;
    mse025_drop020) CONFIG_ENV+=("MSE_WEIGHT=0.25" "DROPOUT=0.20") ;;
    mse050_drop010) CONFIG_ENV+=("DROPOUT=0.10") ;;
    mse050_drop020) CONFIG_ENV+=("DROPOUT=0.20") ;;
    mse075_drop010) CONFIG_ENV+=("MSE_WEIGHT=0.75" "DROPOUT=0.10") ;;
    mse075_drop020) CONFIG_ENV+=("MSE_WEIGHT=0.75" "DROPOUT=0.20") ;;
    mse025_pos_auto) CONFIG_ENV+=("MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=auto") ;;
    pos_auto_mse050) CONFIG_ENV+=("POSITIVE_WEIGHT=auto") ;;
    mse075_pos_auto) CONFIG_ENV+=("MSE_WEIGHT=0.75" "POSITIVE_WEIGHT=auto") ;;
    mse025_focal) CONFIG_ENV+=("MSE_WEIGHT=0.25" "FOCAL_LOSS=1" "FOCAL_GAMMA=2.0") ;;
    focal_mse050) CONFIG_ENV+=("FOCAL_LOSS=1" "FOCAL_GAMMA=2.0") ;;
    mse075_focal) CONFIG_ENV+=("MSE_WEIGHT=0.75" "FOCAL_LOSS=1" "FOCAL_GAMMA=2.0") ;;
    mse025_label_smooth005) CONFIG_ENV+=("MSE_WEIGHT=0.25" "LABEL_SMOOTHING=0.05") ;;
    mse050_label_smooth005) CONFIG_ENV+=("LABEL_SMOOTHING=0.05") ;;
    mse075_label_smooth005) CONFIG_ENV+=("MSE_WEIGHT=0.75" "LABEL_SMOOTHING=0.05") ;;
    mse050_graph_logit1) CONFIG_ENV+=("GRAPH_LOGIT_SCALE=1.0") ;;
    mse050_graph_logit4) CONFIG_ENV+=("GRAPH_LOGIT_SCALE=4.0") ;;
    mse025_pcep_off) CONFIG_ENV+=("MSE_WEIGHT=0.25" "PROTEIN_CONCAT_MODE=off") ;;
    mse050_pcep_off) CONFIG_ENV+=("PROTEIN_CONCAT_MODE=off") ;;
    mse050_pcep_additive) CONFIG_ENV+=("PROTEIN_CONCAT_SCORE_MODE=additive") ;;
    mse025_target_pdi) CONFIG_ENV+=("MSE_WEIGHT=0.25" "MSE_TARGET_MODE=pdi") ;;
    mse050_target_pdi) CONFIG_ENV+=("MSE_TARGET_MODE=pdi") ;;
    mse050_target_ppi) CONFIG_ENV+=("MSE_TARGET_MODE=pdi_ppi") ;;
    mse075_target_pdi) CONFIG_ENV+=("MSE_WEIGHT=0.75" "MSE_TARGET_MODE=pdi") ;;
    mse075_target_ppi) CONFIG_ENV+=("MSE_WEIGHT=0.75" "MSE_TARGET_MODE=pdi_ppi") ;;
    mse050_ctrl_drop005) CONFIG_ENV+=("CONTROL_EXPRESSION_DROPOUT=0.05") ;;
    mse050_no_plate_cov) CONFIG_ENV+=("BATCH_COV_LIST=Cell cell_type batch pert_time") ;;
    *)
      echo "[error] unknown PTV1 cell+cell_type tuning candidate: ${name}" >&2
      exit 2
      ;;
  esac
}

train_completed() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
manifest = Path(sys.argv[1]) / "run_manifest.json"
if not manifest.exists():
    raise SystemExit(1)
with manifest.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
raise SystemExit(0 if payload.get("run_status") == "fit_completed" else 1)' "${CKPT_DIR}/${exp_name}"
}

extra_completed() {
  local exp_name="$1"
  if [[ "${SKIP_COMPLETED}" != "1" ]]; then
    return 1
  fi
  "${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
root = Path(sys.argv[1]) / "ptv1_extra_singledrug"
expected = int(sys.argv[2])
manifest_path = root / "run_manifest.json"
metrics_path = root / "metrics.json"
if not manifest_path.exists() or not metrics_path.exists():
    raise SystemExit(1)
with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("dataset_group") != "ptv1" or manifest.get("task_name") != "ptv1_extra_singledrug":
    raise SystemExit(1)
if int(manifest.get("n_predictions") or -1) != expected:
    raise SystemExit(1)
raise SystemExit(0)' "${OUTPUT_DIR}/${exp_name}" "${EXPECTED_EXTRA_ROWS}"
}

validate_required_artifacts() {
  "${PYTHON_BIN}" utils/ptv1/03_validate_ptv1_training_ready.py
  "${PYTHON_BIN}" utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only --output "${CELL_LLM_EMBEDDING_PATH}"
  "${PYTHON_BIN}" utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py --validate-only --output "${CELL_TYPE_LLM_EMBEDDING_PATH}"
}

run_exp11() {
  local config="$1"
  local run_prefix="${BASE_PREFIX}_${config}"
  local exp_name="${run_prefix}_ptv1_random_split"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && train_completed "${exp_name}"; then
    echo "[skip] exp11 completed: ${exp_name}"
    return
  fi
  set_config_env "${config}"
  echo "[run] exp11 config=${config}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "SAVE_TOP_K=${EXP11_SAVE_TOP_K}" \
    "EXP_PREFIX=${run_prefix}" \
    "TIME_SUMMARY_PATH=${LOG_DIR}/${run_prefix}_runtime_summary.tsv" \
    bash scripts/ptv1/exp_11_ptv1_random_split.sh
}

run_exp12_fold() {
  local config="$1"
  local fold="$2"
  local run_prefix="${BASE_PREFIX}_${config}"
  local exp_name="${run_prefix}_ptv1_unseen_drug_fold${fold}"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && train_completed "${exp_name}"; then
    echo "[skip] exp12 completed: ${exp_name}"
    return
  fi
  set_config_env "${config}"
  echo "[run] exp12 config=${config} fold=${fold}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "SAVE_TOP_K=${EXP12_SAVE_TOP_K}" \
    "FOLDS=${fold}" \
    "EXP_PREFIX=${run_prefix}" \
    "TIME_SUMMARY_PATH=${LOG_DIR}/${run_prefix}_runtime_summary.tsv" \
    bash scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh
}

run_exp13_valid() {
  local config="$1"
  local run_prefix="${BASE_PREFIX}_${config}"
  local exp11_name="${run_prefix}_ptv1_random_split"
  local exp13_name="${run_prefix}_extra_valid_from_exp11"
  if extra_completed "${exp13_name}"; then
    echo "[skip] exp13 valid completed: ${exp13_name}"
    return
  fi
  if ! train_completed "${exp11_name}"; then
    echo "[warn] exp13 valid skipped because exp11 is incomplete: ${exp11_name}" >&2
    return
  fi
  set_config_env "${config}"
  echo "[run] exp13 valid config=${config}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "SAVE_TOP_K=${EXP11_SAVE_TOP_K}" \
    "EXP_PREFIX=${run_prefix}" \
    "EXP11_EXP_NAME=${exp11_name}" \
    "EXP13_EXP_NAME=${exp13_name}" \
    "TIME_SUMMARY_PATH=${LOG_DIR}/${run_prefix}_runtime_summary.tsv" \
    bash scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh direct
}

ckpt_epoch() {
  local ckpt="$1"
  local name
  name="$(basename "${ckpt}")"
  if [[ "${name}" =~ epoch=([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    return 1
  fi
}

run_exp13_oracle() {
  local config="$1"
  local run_prefix="${BASE_PREFIX}_${config}"
  local exp11_name="${run_prefix}_ptv1_random_split"
  local exp11_dir="${CKPT_DIR}/${exp11_name}"
  if ! train_completed "${exp11_name}"; then
    echo "[warn] exp13 oracle skipped because exp11 is incomplete: ${exp11_name}" >&2
    return
  fi
  set_config_env "${config}"
  local found=0
  while IFS= read -r ckpt; do
    [[ -n "${ckpt}" ]] || continue
    found=1
    local epoch
    epoch="$(ckpt_epoch "${ckpt}")"
    local exp13_name="${run_prefix}_extra_oracle_epoch${epoch}_from_exp11"
    if extra_completed "${exp13_name}"; then
      echo "[skip] exp13 oracle completed: ${exp13_name}"
      continue
    fi
    echo "[run] exp13 oracle config=${config} epoch=${epoch}"
    env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
      "SAVE_TOP_K=${EXP11_SAVE_TOP_K}" \
      "EXP_PREFIX=${run_prefix}" \
      "EXP11_EXP_NAME=${exp11_name}" \
      "EXP11_CKPT_PATH=${ckpt}" \
      "EXP13_EXP_NAME=${exp13_name}" \
      "TIME_SUMMARY_PATH=${LOG_DIR}/${run_prefix}_runtime_summary.tsv" \
      bash scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh direct
  done < <(find "${exp11_dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' | sort -V)
  if [[ "${found}" == "0" ]]; then
    echo "[warn] no saved epoch checkpoints found for oracle scan: ${exp11_dir}" >&2
  fi
}

run_reports() {
  mkdir -p "$(dirname "${REPORT_MD_PATH}")" "$(dirname "${REPORT_TSV_PATH}")"
  "${PYTHON_BIN}" scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py \
    --base-prefix "${BASE_PREFIX}" \
    --expected-extra-rows "${EXPECTED_EXTRA_ROWS}" \
    --format markdown \
    > "${REPORT_MD_PATH}"
  "${PYTHON_BIN}" scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py \
    --base-prefix "${BASE_PREFIX}" \
    --expected-extra-rows "${EXPECTED_EXTRA_ROWS}" \
    --format tsv \
    > "${REPORT_TSV_PATH}"
  echo "[report] ${REPORT_MD_PATH}"
  echo "[report] ${REPORT_TSV_PATH}"
}

echo "[settings] BASE_PREFIX=${BASE_PREFIX}; GPU_IDS=${GPU_IDS}; EXP12_FOLDS=${EXP12_FOLDS}"
echo "[settings] EXP11_CANDIDATES=${EXP11_CANDIDATES}"
echo "[settings] EXP12_CANDIDATES=${EXP12_CANDIDATES}"
echo "[settings] RUN_EXP11=${RUN_EXP11}; RUN_EXP12=${RUN_EXP12}; RUN_EXP13_VALID=${RUN_EXP13_VALID}; RUN_EXP13_ORACLE=${RUN_EXP13_ORACLE}"
echo "[settings] CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
echo "[settings] CELL_TYPE_LLM_EMBEDDING_PATH=${CELL_TYPE_LLM_EMBEDDING_PATH}"
"${PYTHON_BIN}" -m py_compile \
  train.py \
  infer.py \
  scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py \
  utils/ptv1/07_build_ptv1_cell_llm_embeddings.py \
  utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py \
  utils/ptv1/09_bootstrap_ptv1_llm_embeddings_from_local.py \
  utils/11_build_cell_llm_embeddings.py \
  utils/11_build_cell_type_llm_embeddings.py
bash -n \
  scripts/ptv1/exp_11_ptv1_random_split.sh \
  scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh \
  scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh \
  scripts/ptv1/run_ptv1_cell_celltype_llm_fine_tune_search.sh \
  scripts/ptv1/ptv1_experiment_common.sh \
  scripts/ptv3_experiment_common.sh

if [[ "${RUN_DATA_VALIDATION}" == "1" ]]; then
  validate_required_artifacts
fi

for config in ${EXP11_CANDIDATES}; do
  if [[ "${RUN_EXP11}" == "1" ]]; then
    run_exp11 "${config}"
  fi
  if [[ "${RUN_EXP13_VALID}" == "1" ]]; then
    run_exp13_valid "${config}"
  fi
  if [[ "${RUN_EXP13_ORACLE}" == "1" ]]; then
    run_exp13_oracle "${config}"
  fi
done

for config in ${EXP12_CANDIDATES}; do
  if [[ "${RUN_EXP12}" == "1" ]]; then
    for fold in ${EXP12_FOLDS}; do
      run_exp12_fold "${config}" "${fold}"
    done
  fi
done

if [[ "${REPORT_AFTER}" == "1" ]]; then
  run_reports
fi

echo "[done] PTV1 graph + Cell/cell_type LLM tuning search completed for BASE_PREFIX=${BASE_PREFIX}"
