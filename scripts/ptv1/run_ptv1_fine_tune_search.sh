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
BASE_PREFIX="${BASE_PREFIX:-20260608_ptv1_cell_llm_tune_v1}"
FINE_TUNE_CANDIDATES="${FINE_TUNE_CANDIDATES:-baseline_mse050 mse000 mse010 mse025 mse075 mse100 mse025_lr1e4 mse025_lr3e4 mse050_lr1e4 mse050_lr3e4 mse025_drop010 mse025_drop020 mse050_drop010 mse050_drop020 mse025_pos_auto pos_auto_mse050 mse025_focal focal_mse050 mse025_label_smooth005 mse050_label_smooth005 mse025_pos_sample2 mse025_graph_logit1 mse025_graph_logit4 mse025_graph_off mse025_pcep_off mse025_pcep_additive mse025_target_pdi mse050_target_pdi mse050_target_ppi mse025_ctrl_drop005 mse025_no_plate_cov mse025_target_expr_pdi}"
EXP12_FOLDS="${EXP12_FOLDS:-0 1 2 3 4}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/ptv1}"
RUN_EXP11="${RUN_EXP11:-1}"
RUN_EXP12="${RUN_EXP12:-1}"
RUN_EXP13_DIRECT="${RUN_EXP13_DIRECT:-1}"
RUN_EXP13_ALL_TRAIN="${RUN_EXP13_ALL_TRAIN:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
REPORT_AFTER="${REPORT_AFTER:-1}"
EXPECTED_EXTRA_ROWS="${EXPECTED_EXTRA_ROWS:-218}"
EXP11_AUPRC_THRESHOLD="${EXP11_AUPRC_THRESHOLD:-0.883452}"
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
  "SAVE_TOP_K=1"
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
  "CELL_LLM_EMBEDDING_PATH=data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz"
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
    baseline_mse050) ;;
    mse000) CONFIG_ENV+=("MSE_WEIGHT=0.00") ;;
    mse010) CONFIG_ENV+=("MSE_WEIGHT=0.10") ;;
    mse025) CONFIG_ENV+=("MSE_WEIGHT=0.25") ;;
    mse075) CONFIG_ENV+=("MSE_WEIGHT=0.75") ;;
    mse100) CONFIG_ENV+=("MSE_WEIGHT=1.00") ;;
    mse025_lr1e4) CONFIG_ENV+=("MSE_WEIGHT=0.25" "LEARNING_RATE=1e-4") ;;
    mse025_lr3e4) CONFIG_ENV+=("MSE_WEIGHT=0.25" "LEARNING_RATE=3e-4") ;;
    mse050_lr1e4) CONFIG_ENV+=("LEARNING_RATE=1e-4") ;;
    mse050_lr3e4) CONFIG_ENV+=("LEARNING_RATE=3e-4") ;;
    mse025_drop010) CONFIG_ENV+=("MSE_WEIGHT=0.25" "DROPOUT=0.10") ;;
    mse025_drop020) CONFIG_ENV+=("MSE_WEIGHT=0.25" "DROPOUT=0.20") ;;
    mse050_drop010) CONFIG_ENV+=("DROPOUT=0.10") ;;
    mse050_drop020) CONFIG_ENV+=("DROPOUT=0.20") ;;
    mse025_pos_auto) CONFIG_ENV+=("MSE_WEIGHT=0.25" "POSITIVE_WEIGHT=auto") ;;
    pos_auto_mse050) CONFIG_ENV+=("POSITIVE_WEIGHT=auto") ;;
    mse025_focal) CONFIG_ENV+=("MSE_WEIGHT=0.25" "FOCAL_LOSS=1" "FOCAL_GAMMA=2.0") ;;
    focal_mse050) CONFIG_ENV+=("FOCAL_LOSS=1" "FOCAL_GAMMA=2.0") ;;
    mse025_label_smooth005) CONFIG_ENV+=("MSE_WEIGHT=0.25" "LABEL_SMOOTHING=0.05") ;;
    mse050_label_smooth005) CONFIG_ENV+=("LABEL_SMOOTHING=0.05") ;;
    mse025_pos_sample2) CONFIG_ENV+=("MSE_WEIGHT=0.25" "POSITIVE_LABEL_SAMPLING_WEIGHT=2.0") ;;
    mse025_graph_logit1) CONFIG_ENV+=("MSE_WEIGHT=0.25" "GRAPH_LOGIT_SCALE=1.0") ;;
    mse025_graph_logit4) CONFIG_ENV+=("MSE_WEIGHT=0.25" "GRAPH_LOGIT_SCALE=4.0") ;;
    mse025_graph_off) CONFIG_ENV+=("MSE_WEIGHT=0.25" "GRAPH_FEATURE_MODE=off" "GRAPH_STRUCTURAL_RP=0" "GRAPH_DRUG_CONCAT=0" "GRAPH_LOGIT_SCALE=0.0") ;;
    mse025_pcep_off) CONFIG_ENV+=("MSE_WEIGHT=0.25" "PROTEIN_CONCAT_MODE=off") ;;
    mse025_pcep_additive) CONFIG_ENV+=("MSE_WEIGHT=0.25" "PROTEIN_CONCAT_SCORE_MODE=additive") ;;
    mse025_target_pdi) CONFIG_ENV+=("MSE_WEIGHT=0.25" "MSE_TARGET_MODE=pdi") ;;
    mse050_target_pdi) CONFIG_ENV+=("MSE_TARGET_MODE=pdi") ;;
    mse050_target_ppi) CONFIG_ENV+=("MSE_TARGET_MODE=pdi_ppi") ;;
    mse025_ctrl_drop005) CONFIG_ENV+=("MSE_WEIGHT=0.25" "CONTROL_EXPRESSION_DROPOUT=0.05") ;;
    mse025_no_plate_cov) CONFIG_ENV+=("MSE_WEIGHT=0.25" "BATCH_COV_LIST=Cell cell_type batch pert_time") ;;
    mse025_target_expr_pdi) CONFIG_ENV+=("MSE_WEIGHT=0.25" "TARGET_EXPRESSION_MODE=pdi") ;;
    *)
      echo "[error] unknown PTV1 fine-tune candidate: ${name}" >&2
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
    "FOLDS=${fold}" \
    "EXP_PREFIX=${run_prefix}" \
    "TIME_SUMMARY_PATH=${LOG_DIR}/${run_prefix}_runtime_summary.tsv" \
    bash scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh
}

run_exp13_direct() {
  local config="$1"
  local run_prefix="${BASE_PREFIX}_${config}"
  local exp11_name="${run_prefix}_ptv1_random_split"
  local exp13_name="${run_prefix}_extra_direct_from_exp11"
  if extra_completed "${exp13_name}"; then
    echo "[skip] exp13 direct completed: ${exp13_name}"
    return
  fi
  if ! train_completed "${exp11_name}"; then
    echo "[warn] exp13 direct skipped because exp11 is incomplete: ${exp11_name}" >&2
    return
  fi
  set_config_env "${config}"
  echo "[run] exp13 direct config=${config}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "EXP_PREFIX=${run_prefix}" \
    "EXP11_EXP_NAME=${exp11_name}" \
    "EXP13_EXP_NAME=${exp13_name}" \
    "TIME_SUMMARY_PATH=${LOG_DIR}/${run_prefix}_runtime_summary.tsv" \
    bash scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh direct
}

run_exp13_all_train() {
  local config="$1"
  local run_prefix="${BASE_PREFIX}_${config}"
  local exp11_name="${run_prefix}_ptv1_random_split"
  local exp13_name="${run_prefix}_all_ptv1_for_extra_from_exp11"
  if extra_completed "${exp13_name}"; then
    echo "[skip] exp13 all_train completed: ${exp13_name}"
    return
  fi
  if ! train_completed "${exp11_name}"; then
    echo "[warn] exp13 all_train skipped because exp11 is incomplete: ${exp11_name}" >&2
    return
  fi
  set_config_env "${config}"
  echo "[run] exp13 all_train config=${config}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "EXP_PREFIX=${run_prefix}" \
    "EXP11_EXP_NAME=${exp11_name}" \
    "EXP13_EXP_NAME=${exp13_name}" \
    "TIME_SUMMARY_PATH=${LOG_DIR}/${run_prefix}_runtime_summary.tsv" \
    bash scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh all_train
}

echo "[settings] BASE_PREFIX=${BASE_PREFIX}; GPU_IDS=${GPU_IDS}; EXP12_FOLDS=${EXP12_FOLDS}"
echo "[settings] RUN_EXP11=${RUN_EXP11}; RUN_EXP12=${RUN_EXP12}; RUN_EXP13_DIRECT=${RUN_EXP13_DIRECT}; RUN_EXP13_ALL_TRAIN=${RUN_EXP13_ALL_TRAIN}"
"${PYTHON_BIN}" -m py_compile \
  train.py \
  infer.py \
  scripts/ptv1/report_ptv1_exp_results.py \
  scripts/ptv1/report_ptv1_param_search.py \
  scripts/ptv1/report_ptv1_fine_tune_results.py \
  utils/ptv1/01_validate_ptv1_standardized.py \
  utils/ptv1/03_validate_ptv1_training_ready.py \
  utils/ptv1/07_build_ptv1_cell_llm_embeddings.py \
  utils/11_build_cell_llm_embeddings.py
bash -n \
  scripts/ptv1/exp_11_ptv1_random_split.sh \
  scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh \
  scripts/ptv1/exp_13_ptv1_extra_single_all_train_infer.sh \
  scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh \
  scripts/ptv1/run_ptv1_fine_tune_search.sh \
  scripts/ptv1/ptv1_experiment_common.sh \
  scripts/ptv3_experiment_common.sh
if [[ "${RUN_DATA_VALIDATION:-0}" == "1" ]]; then
  "${PYTHON_BIN}" utils/ptv1/01_validate_ptv1_standardized.py
  "${PYTHON_BIN}" utils/ptv1/03_validate_ptv1_training_ready.py
  "${PYTHON_BIN}" utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only
fi

for config in ${FINE_TUNE_CANDIDATES}; do
  if [[ "${RUN_EXP11}" == "1" ]]; then
    run_exp11 "${config}"
  fi
  if [[ "${RUN_EXP12}" == "1" ]]; then
    for fold in ${EXP12_FOLDS}; do
      run_exp12_fold "${config}" "${fold}"
    done
  fi
  if [[ "${RUN_EXP13_DIRECT}" == "1" ]]; then
    run_exp13_direct "${config}"
  fi
  if [[ "${RUN_EXP13_ALL_TRAIN}" == "1" ]]; then
    run_exp13_all_train "${config}"
  fi
done

if [[ "${REPORT_AFTER}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/ptv1/report_ptv1_fine_tune_results.py \
    --base-prefix "${BASE_PREFIX}" \
    --exp11-auprc-threshold "${EXP11_AUPRC_THRESHOLD}" \
    --format markdown \
    > "${LOG_DIR}/${BASE_PREFIX}_fine_tune_results.md"
  "${PYTHON_BIN}" scripts/ptv1/report_ptv1_fine_tune_results.py \
    --base-prefix "${BASE_PREFIX}" \
    --exp11-auprc-threshold "${EXP11_AUPRC_THRESHOLD}" \
    --format tsv \
    > "${OUTPUT_DIR}/${BASE_PREFIX}_fine_tune_results.tsv"
  echo "[report] ${LOG_DIR}/${BASE_PREFIX}_fine_tune_results.md"
  echo "[report] ${OUTPUT_DIR}/${BASE_PREFIX}_fine_tune_results.tsv"
fi

echo "[done] PTV1 fine-tune search completed for BASE_PREFIX=${BASE_PREFIX}"
