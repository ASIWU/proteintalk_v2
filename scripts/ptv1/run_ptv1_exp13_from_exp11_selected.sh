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

CANDIDATE="${CANDIDATE:-mse050_drop010}"
SOURCE_EXP11_NAME="${SOURCE_EXP11_NAME:-20260608_ptv1_cell_llm_tune_v1_mse050_drop010_ptv1_random_split}"
OUTPUT_PREFIX_BASE="${OUTPUT_PREFIX_BASE:-20260608_ptv1_cell_llm_exp13_from_exp11_graphon}"
OUTPUT_PREFIX_VERSION="${OUTPUT_PREFIX_VERSION:-1}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${OUTPUT_PREFIX_BASE}_v${OUTPUT_PREFIX_VERSION}_${CANDIDATE}}"
ALLOW_EXISTING_EXP13_PREFIX="${ALLOW_EXISTING_EXP13_PREFIX:-0}"
RUN_DIRECT="${RUN_DIRECT:-1}"
RUN_ALL_TRAIN="${RUN_ALL_TRAIN:-1}"
REPORT_AFTER="${REPORT_AFTER:-1}"
STATIC_CHECKS="${STATIC_CHECKS:-1}"
EXPECTED_EXTRA_ROWS="${EXPECTED_EXTRA_ROWS:-218}"
GPU_IDS="${GPU_IDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-graph_cache/ptv1}"
REPORT_PATH="${REPORT_PATH:-docs/2026-06-08_ptv1_exp13_from_exp11_graphon_rerun_report.md}"
REPORT_TSV_PATH="${REPORT_TSV_PATH:-}"

if [[ "${CANDIDATE}" != "mse050_drop010" ]]; then
  echo "[error] this official exp_13 rerun is pinned to CANDIDATE=mse050_drop010" >&2
  exit 2
fi

prefix_has_artifacts() {
  local prefix="$1"
  [[ -e "${OUTPUT_DIR}/${prefix}_extra_direct_from_exp11" ]] && return 0
  [[ -e "${OUTPUT_DIR}/${prefix}_all_ptv1_for_extra_from_exp11" ]] && return 0
  [[ -e "${CKPT_DIR}/${prefix}_all_ptv1_for_extra_from_exp11" ]] && return 0
  [[ -e "${LOG_DIR}/${prefix}_runtime_summary.tsv" ]] && return 0
  return 1
}

choose_output_prefix() {
  if [[ "${ALLOW_EXISTING_EXP13_PREFIX}" == "1" ]]; then
    printf "%s\n" "${OUTPUT_PREFIX}"
    return
  fi

  local version
  local candidate_prefix
  for ((version = OUTPUT_PREFIX_VERSION; version <= OUTPUT_PREFIX_VERSION + 99; version++)); do
    candidate_prefix="${OUTPUT_PREFIX_BASE}_v${version}_${CANDIDATE}"
    if ! prefix_has_artifacts "${candidate_prefix}"; then
      printf "%s\n" "${candidate_prefix}"
      return
    fi
  done

  echo "[error] could not find an unused exp_13 output prefix after 100 versions" >&2
  exit 2
}

precheck_source_exp11() {
  "${PYTHON_BIN}" -c 'import json, math, re, sys; from pathlib import Path
name = sys.argv[1]
root = Path(sys.argv[2])
manifest_path = root / name / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing source exp_11 manifest: {manifest_path}")
with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
errors = []
expected = {
    "dataset_group": "ptv1",
    "task_name": "ptv1_aivc",
    "split_strategy": "fixed_experiment_type",
    "run_status": "fit_completed",
    "mse_weight": 0.50,
    "dropout": 0.10,
    "graph_feature_mode": "real",
    "graph_structural_rp": True,
    "graph_drug_concat": True,
    "graph_logit_scale": 2.0,
    "mse_target_mode": "all",
    "cell_llm_mode": "frozen",
}
def value(key):
    if key in manifest:
        return manifest[key]
    args = manifest.get("args")
    return args.get(key) if isinstance(args, dict) else None
def same(actual, wanted):
    if isinstance(wanted, bool):
        if isinstance(actual, str):
            return actual.lower() in {"1", "true", "yes"} if wanted else actual.lower() in {"0", "false", "no"}
        return bool(actual) is wanted
    if isinstance(wanted, float):
        try:
            return math.isclose(float(actual), wanted, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(wanted)
for key, wanted in expected.items():
    actual = value(key)
    if not same(actual, wanted):
        errors.append(f"{key}: expected {wanted!r}, found {actual!r}")
checkpoint = manifest.get("best_model_path") or manifest.get("test_checkpoint_path")
if not checkpoint:
    errors.append("missing best_model_path/test_checkpoint_path")
else:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        errors.append(f"missing source best checkpoint: {checkpoint_path}")
    match = re.search(r"epoch=(\d+)", str(checkpoint))
    if match is None:
        errors.append(f"could not parse selected epoch from checkpoint: {checkpoint}")
if "mse025_graph_off" in name or value("graph_feature_mode") == "off":
    errors.append("graph-off source is forbidden for this official exp_13 rerun")
if errors:
    raise SystemExit("[source precheck failed]\n" + "\n".join(f"- {item}" for item in errors))
selected_epoch = re.search(r"epoch=(\d+)", str(checkpoint)).group(1)
print(f"[source] {manifest_path}")
print(f"[source] checkpoint={checkpoint}")
print(f"[source] selected_epoch={selected_epoch}")' "${SOURCE_EXP11_NAME}" "${CKPT_DIR}"
}

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
  "ALLOW_EXISTING_RUN=0"
)

CONFIG_ENV=(
  "LEARNING_RATE=2e-4"
  "BATCH_SIZE=${BATCH_SIZE:-256}"
  "DROPOUT=0.10"
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

if [[ "${STATIC_CHECKS}" == "1" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    train.py \
    infer.py \
    scripts/ptv1/report_ptv1_exp13_from_exp11_selected.py
  bash -n \
    scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh \
    scripts/ptv1/run_ptv1_exp13_from_exp11_selected.sh \
    scripts/ptv1/ptv1_experiment_common.sh \
    scripts/ptv3_experiment_common.sh
fi

precheck_source_exp11

SELECTED_OUTPUT_PREFIX="$(choose_output_prefix)"
if [[ "${SELECTED_OUTPUT_PREFIX}" != "${OUTPUT_PREFIX}" ]]; then
  echo "[prefix] requested prefix has artifacts; using ${SELECTED_OUTPUT_PREFIX}"
fi
OUTPUT_PREFIX="${SELECTED_OUTPUT_PREFIX}"
DIRECT_EXP_NAME="${OUTPUT_PREFIX}_extra_direct_from_exp11"
ALL_TRAIN_EXP_NAME="${OUTPUT_PREFIX}_all_ptv1_for_extra_from_exp11"
TIME_SUMMARY_PATH="${LOG_DIR}/${OUTPUT_PREFIX}_runtime_summary.tsv"
if [[ -z "${REPORT_TSV_PATH}" ]]; then
  REPORT_TSV_PATH="outputs/${OUTPUT_PREFIX}_exp13_from_exp11_selected_report.tsv"
fi

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}" "$(dirname "${REPORT_PATH}")" "$(dirname "${REPORT_TSV_PATH}")"

echo "[settings] source_exp11=${SOURCE_EXP11_NAME}"
echo "[settings] output_prefix=${OUTPUT_PREFIX}"
echo "[settings] run_direct=${RUN_DIRECT}; run_all_train=${RUN_ALL_TRAIN}; expected_extra_rows=${EXPECTED_EXTRA_ROWS}"

if [[ "${RUN_DIRECT}" == "1" ]]; then
  echo "[run] exp13 direct: ${DIRECT_EXP_NAME}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "EXP_PREFIX=${OUTPUT_PREFIX}" \
    "EXP11_EXP_NAME=${SOURCE_EXP11_NAME}" \
    "EXP13_EXP_NAME=${DIRECT_EXP_NAME}" \
    "TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}" \
    bash scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh direct
fi

if [[ "${RUN_ALL_TRAIN}" == "1" ]]; then
  echo "[run] exp13 all_train: ${ALL_TRAIN_EXP_NAME}"
  env "${COMMON_ENV[@]}" "${CONFIG_ENV[@]}" \
    "EXP_PREFIX=${OUTPUT_PREFIX}" \
    "EXP11_EXP_NAME=${SOURCE_EXP11_NAME}" \
    "EXP13_EXP_NAME=${ALL_TRAIN_EXP_NAME}" \
    "TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}" \
    bash scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh all_train
fi

if [[ "${REPORT_AFTER}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/ptv1/report_ptv1_exp13_from_exp11_selected.py \
    --source-exp11-name "${SOURCE_EXP11_NAME}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --expected-extra-rows "${EXPECTED_EXTRA_ROWS}" \
    --format markdown \
    > "${REPORT_PATH}"
  "${PYTHON_BIN}" scripts/ptv1/report_ptv1_exp13_from_exp11_selected.py \
    --source-exp11-name "${SOURCE_EXP11_NAME}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --expected-extra-rows "${EXPECTED_EXTRA_ROWS}" \
    --format tsv \
    > "${REPORT_TSV_PATH}"
  echo "[report] ${REPORT_PATH}"
  echo "[report] ${REPORT_TSV_PATH}"
fi

echo "[done] official exp_13-from-exp_11 rerun completed for OUTPUT_PREFIX=${OUTPUT_PREFIX}"
