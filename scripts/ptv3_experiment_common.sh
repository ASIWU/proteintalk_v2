#!/usr/bin/env bash

# Shared mechanics for the small PTV3 experiment scripts.
# User-facing experiment choices should live in exp_*.sh, not here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DEVICES="${DEVICES:-8}"
PRECISION="${PRECISION:-32-true}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"
SCHEDULER_NAME="${SCHEDULER_NAME:-}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
BCE_WEIGHT="${BCE_WEIGHT:-}"
POSITIVE_WEIGHT="${POSITIVE_WEIGHT:-}"
FOCAL_LOSS="${FOCAL_LOSS:-0}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1.0}"
INFER_LIMIT_BATCHES="${INFER_LIMIT_BATCHES:-}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d_%H%M)_${EXPERIMENT_SET_NAME:-ptv3}}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
INFER_DEVICE="${INFER_DEVICE:-cuda:0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
SAVE_EVERY_N_TRAIN_STEPS="${SAVE_EVERY_N_TRAIN_STEPS:-}"
SAVE_TOP_K="${SAVE_TOP_K:--1}"
SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-}"
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}"
REFERENCE_5FOLD_CKPT_PATH="${REFERENCE_5FOLD_CKPT_PATH:-}"
REFERENCE_EPOCH_AGG="${REFERENCE_EPOCH_AGG:-median}"
REFERENCE_EPOCH_ROUNDING="${REFERENCE_EPOCH_ROUNDING:-nearest}"
REFERENCE_EPOCH_MIN_COUNT="${REFERENCE_EPOCH_MIN_COUNT:-5}"
REFERENCE_REQUIRE_TEST_COMPLETED="${REFERENCE_REQUIRE_TEST_COMPLETED:-1}"
REFERENCE_SPLIT_STRATEGY_REGEX="${REFERENCE_SPLIT_STRATEGY_REGEX:-}"
REFERENCE_ALLOW_MIXED_CONFIG="${REFERENCE_ALLOW_MIXED_CONFIG:-0}"
REFERENCE_ALLOW_DUPLICATE_SPLITS="${REFERENCE_ALLOW_DUPLICATE_SPLITS:-0}"
USER_SET_MONITOR="${MONITOR+x}"
MONITOR="${MONITOR:-}"
if [[ -n "${SAVE_EVERY_N_TRAIN_STEPS}" && -z "${USER_SET_MONITOR}" ]]; then
  MONITOR="none"
fi
MONITOR_MODE="${MONITOR_MODE:-}"
LOGGER_BACKEND="${LOGGER_BACKEND:-tensorboard}"
LOG_TO_WANDB="${LOG_TO_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-aivc_proteintalk}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
ALLOW_NONFINITE_MONITOR="${ALLOW_NONFINITE_MONITOR:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${LOG_DIR}/${EXP_PREFIX}_runtime_summary.tsv}"

read -r -a FOLD_LIST <<< "${FOLDS}"

COMMON_TRAIN_ARGS=(
  --dataset-group ptv3
  --model-type attention_v10_hetero_cls_ee
  --batch-size "${BATCH_SIZE}"
  --max-epochs "${MAX_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-heads "${NUM_HEADS}"
  --num-layers "${NUM_LAYERS}"
  --dropout "${DROPOUT}"
  --mse-weight "${MSE_WEIGHT}"
  --optimizer-name "${OPTIMIZER_NAME}"
  --accelerator gpu
  --devices "${DEVICES}"
  --strategy ddp_find_unused_parameters_true
  --precision "${PRECISION}"
  --num-workers "${NUM_WORKERS}"
  --gradient-clip-val "${GRADIENT_CLIP_VAL}"
  --checkpoint-dir "${CKPT_DIR}"
  --log-dir "${LOG_DIR}"
  --save-every-n-epochs "${SAVE_EVERY_N_EPOCHS}"
  --save-top-k "${SAVE_TOP_K}"
  --best-ckpt-metric "${BEST_CKPT_METRIC}"
  --logger-backend "${LOGGER_BACKEND}"
  --wandb-project "${WANDB_PROJECT}"
  --log-every-n-steps "${LOG_EVERY_N_STEPS}"
  --check-val-every-n-epoch "${CHECK_VAL_EVERY_N_EPOCH}"
  --limit-train-batches "${LIMIT_TRAIN_BATCHES}"
  --limit-val-batches "${LIMIT_VAL_BATCHES}"
  --limit-test-batches "${LIMIT_TEST_BATCHES}"
)

if [[ -n "${SAVE_EVERY_N_TRAIN_STEPS}" ]]; then
  COMMON_TRAIN_ARGS+=(--save-every-n-train-steps "${SAVE_EVERY_N_TRAIN_STEPS}")
fi
if [[ "${SAVE_LAST_CKPT}" != "1" ]]; then
  COMMON_TRAIN_ARGS+=(--no-save-last-ckpt)
fi
if [[ -n "${CHECKPOINT_FILENAME}" ]]; then
  COMMON_TRAIN_ARGS+=(--checkpoint-filename "${CHECKPOINT_FILENAME}")
fi
if [[ -n "${MONITOR}" ]]; then
  COMMON_TRAIN_ARGS+=(--monitor "${MONITOR}")
fi
if [[ -n "${MONITOR_MODE}" ]]; then
  COMMON_TRAIN_ARGS+=(--monitor-mode "${MONITOR_MODE}")
fi
if [[ -n "${SCHEDULER_NAME}" ]]; then
  COMMON_TRAIN_ARGS+=(--scheduler-name "${SCHEDULER_NAME}")
fi
if [[ -n "${BCE_WEIGHT}" ]]; then
  COMMON_TRAIN_ARGS+=(--bce-weight "${BCE_WEIGHT}")
fi
if [[ -n "${POSITIVE_WEIGHT}" ]]; then
  COMMON_TRAIN_ARGS+=(--positive-weight "${POSITIVE_WEIGHT}")
fi
if [[ "${FOCAL_LOSS}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--focal-loss)
fi
if [[ "${LOG_TO_WANDB}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--log-to-wandb)
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_TRAIN_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  COMMON_TRAIN_ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  COMMON_TRAIN_ARGS+=(--wandb-mode "${WANDB_MODE}")
fi
if [[ "${WANDB_LOG_MODEL}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--wandb-log-model)
fi
if [[ "${ALLOW_NONFINITE_MONITOR}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--allow-nonfinite-monitor)
fi
if [[ -n "${WANDB_TAGS}" ]]; then
  read -r -a WANDB_TAG_LIST <<< "${WANDB_TAGS}"
  COMMON_TRAIN_ARGS+=(--wandb-tags "${WANDB_TAG_LIST[@]}")
fi

ptv3_utc_now() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

ptv3_init_time_summary() {
  mkdir -p "$(dirname "${TIME_SUMMARY_PATH}")"
  if [[ ! -f "${TIME_SUMMARY_PATH}" ]]; then
    printf "kind\texperiment\ttask_name\tsplit_strategy\tsplit_name\tstatus\tstart_utc\tend_utc\tduration_sec\tartifact\n" > "${TIME_SUMMARY_PATH}"
  fi
}

ptv3_record_time() {
  local kind="$1"
  local experiment="$2"
  local task_name="$3"
  local split_strategy="$4"
  local split_name="$5"
  local status="$6"
  local start_utc="$7"
  local end_utc="$8"
  local duration_sec="$9"
  local artifact="${10}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${kind}" \
    "${experiment}" \
    "${task_name}" \
    "${split_strategy}" \
    "${split_name}" \
    "${status}" \
    "${start_utc}" \
    "${end_utc}" \
    "${duration_sec}" \
    "${artifact}" >> "${TIME_SUMMARY_PATH}"
}

ptv3_ensure_clean_path() {
  local path="$1"
  local kind="$2"
  if [[ "${ALLOW_EXISTING_RUN}" == "1" ]]; then
    return
  fi
  if [[ -e "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "[error] ${kind} already exists and is not empty: ${path}" >&2
    echo "[error] choose a new EXP_PREFIX or set ALLOW_EXISTING_RUN=1 intentionally" >&2
    exit 1
  fi
}

ptv3_print_settings() {
  local title="$1"
  ptv3_init_time_summary
  echo "[experiment] ${title}"
  echo "[settings] EXP_PREFIX=${EXP_PREFIX}"
  echo "[settings] FOLDS=${FOLDS}"
  echo "[settings] MAX_EPOCHS=${MAX_EPOCHS}"
  echo "[settings] BATCH_SIZE=${BATCH_SIZE} per GPU; DEVICES=${DEVICES}; GPU_IDS=${GPU_IDS}"
  echo "[settings] LEARNING_RATE=${LEARNING_RATE}; OPTIMIZER_NAME=${OPTIMIZER_NAME}; SCHEDULER_NAME=${SCHEDULER_NAME:-none}"
  echo "[settings] LOGGER_BACKEND=${LOGGER_BACKEND}; LOG_TO_WANDB=${LOG_TO_WANDB}; WANDB_PROJECT=${WANDB_PROJECT}"
  echo "[settings] LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS}; CHECK_VAL_EVERY_N_EPOCH=${CHECK_VAL_EVERY_N_EPOCH}"
  echo "[settings] BEST_CKPT_METRIC=${BEST_CKPT_METRIC}; MONITOR=${MONITOR:-auto}; MONITOR_MODE=${MONITOR_MODE:-auto}; ALLOW_NONFINITE_MONITOR=${ALLOW_NONFINITE_MONITOR}"
  echo "[settings] REFERENCE_5FOLD_CKPT_PATH=${REFERENCE_5FOLD_CKPT_PATH:-none}; REFERENCE_EPOCH_AGG=${REFERENCE_EPOCH_AGG}; REFERENCE_EPOCH_ROUNDING=${REFERENCE_EPOCH_ROUNDING}; REFERENCE_EPOCH_MIN_COUNT=${REFERENCE_EPOCH_MIN_COUNT}"
  echo "[settings] REFERENCE_REQUIRE_TEST_COMPLETED=${REFERENCE_REQUIRE_TEST_COMPLETED}; REFERENCE_SPLIT_STRATEGY_REGEX=${REFERENCE_SPLIT_STRATEGY_REGEX:-script-default}; REFERENCE_ALLOW_MIXED_CONFIG=${REFERENCE_ALLOW_MIXED_CONFIG}; REFERENCE_ALLOW_DUPLICATE_SPLITS=${REFERENCE_ALLOW_DUPLICATE_SPLITS}"
  echo "[settings] SAVE_EVERY_N_EPOCHS=${SAVE_EVERY_N_EPOCHS}; SAVE_EVERY_N_TRAIN_STEPS=${SAVE_EVERY_N_TRAIN_STEPS:-none}"
  echo "[settings] RUN_PREFLIGHT=${RUN_PREFLIGHT}; RUN_INFERENCE=${RUN_INFERENCE}"
  echo "[settings] TIME_SUMMARY_PATH=${TIME_SUMMARY_PATH}"
}

ptv3_run_preflight() {
  if [[ "${RUN_PREFLIGHT}" != "1" ]]; then
    return
  fi
  "${PYTHON_BIN}" -m py_compile \
    train.py \
    infer.py \
    dataset/training_ready_dataset.py \
    model/training_ready_models.py \
    model/training_ready_lightning.py \
    scripts/select_reference_epoch.py \
    utils/00_standardize_rawdata.py \
    utils/01_validate_standardized_outputs.py \
    utils/02_build_training_ready_data.py \
    utils/03_validate_training_ready_outputs.py \
    utils/09_build_data_splits.py
  "${PYTHON_BIN}" utils/01_validate_standardized_outputs.py
  "${PYTHON_BIN}" utils/03_validate_training_ready_outputs.py
}

ptv3_train() {
  local exp_name="$1"
  local task_name="$2"
  local split_strategy="$3"
  local task_head="$4"
  shift 4

  ptv3_ensure_clean_path "${CKPT_DIR}/${exp_name}" "checkpoint directory"
  ptv3_ensure_clean_path "${LOG_DIR}/${exp_name}" "log directory"

  local start_utc
  local end_utc
  local start_sec
  local end_sec
  local status
  start_utc="$(ptv3_utc_now)"
  start_sec="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u train.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    --experiment-name "${exp_name}" \
    --task-name "${task_name}" \
    --split-strategy "${split_strategy}" \
    --task-head "${task_head}" \
    "$@"
  status="$?"
  set -e
  end_utc="$(ptv3_utc_now)"
  end_sec="$(date +%s)"

  ptv3_record_time \
    "train" \
    "${exp_name}" \
    "${task_name}" \
    "${split_strategy}" \
    "-" \
    "${status}" \
    "${start_utc}" \
    "${end_utc}" \
    "$((end_sec - start_sec))" \
    "${CKPT_DIR}/${exp_name}"

  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

ptv3_best_checkpoint() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
exp = Path(sys.argv[1])
manifest_path = exp / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing run manifest: {manifest_path}")
manifest = json.load(manifest_path.open())
if manifest.get("run_status") != "fit_completed":
    raise SystemExit(f"run is not fit_completed: {manifest_path}")
checkpoint = manifest.get("best_model_path") or str(exp / "last.ckpt")
if not Path(checkpoint).exists():
    raise SystemExit(f"checkpoint does not exist: {checkpoint}")
print(checkpoint)' "${CKPT_DIR}/${exp_name}"
}

ptv3_last_checkpoint() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import sys; from pathlib import Path
checkpoint = Path(sys.argv[1]) / "last.ckpt"
if not checkpoint.exists():
    raise SystemExit(f"checkpoint does not exist: {checkpoint}")
print(checkpoint)' "${CKPT_DIR}/${exp_name}"
}

ptv3_reference_epoch() {
  local reference_path="$1"
  local task_name="$2"
  local task_head="$3"
  local split_strategy_regex="$4"
  local summary_json="$5"
  local -a reference_args=(
    "${reference_path}"
    --task-name "${task_name}"
    --expect-task-head "${task_head}"
    --expect-model-type attention_v10_hetero_cls_ee
    --expect-dataset-group ptv3
    --method "${REFERENCE_EPOCH_AGG}"
    --rounding "${REFERENCE_EPOCH_ROUNDING}"
    --min-count "${REFERENCE_EPOCH_MIN_COUNT}"
    --summary-json "${summary_json}"
  )
  if [[ -n "${split_strategy_regex}" ]]; then
    reference_args+=(--split-strategy-regex "${split_strategy_regex}")
  fi
  if [[ "${REFERENCE_REQUIRE_TEST_COMPLETED}" == "1" ]]; then
    reference_args+=(--require-test-completed)
  fi
  if [[ "${REFERENCE_ALLOW_MIXED_CONFIG}" == "1" ]]; then
    reference_args+=(--allow-mixed-reference-config)
  fi
  if [[ "${REFERENCE_ALLOW_DUPLICATE_SPLITS}" == "1" ]]; then
    reference_args+=(--allow-duplicate-split-strategies)
  fi
  "${PYTHON_BIN}" scripts/select_reference_epoch.py \
    "${reference_args[@]}"
}

ptv3_record_reference_epoch_policy() {
  local exp_name="$1"
  local reference_path="$2"
  local reference_task_name="$3"
  local reference_epoch="$4"
  local applied_max_epochs="$5"
  local reference_split_strategy_regex="$6"
  local reference_summary_json="$7"
  "${PYTHON_BIN}" -c 'import json, sys; from datetime import datetime, timezone; from pathlib import Path
run_dir = Path(sys.argv[1])
manifest_path = run_dir / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing run manifest: {manifest_path}")
selected_checkpoint_path = run_dir / "last.ckpt"
if not selected_checkpoint_path.exists():
    raise SystemExit(f"reference epoch policy requires last.ckpt, but it does not exist: {selected_checkpoint_path}")
reference_summary_path = Path(sys.argv[9])
reference_summary = None
if str(reference_summary_path):
    if not reference_summary_path.exists():
        raise SystemExit(f"missing reference epoch summary: {reference_summary_path}")
    with reference_summary_path.open("r", encoding="utf-8") as handle:
        reference_summary = json.load(handle)
with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
manifest["reference_epoch_policy"] = {
    "enabled": True,
    "reference_path": sys.argv[2],
    "reference_task_name": sys.argv[3],
    "selected_epoch": int(sys.argv[4]),
    "applied_max_epochs": int(sys.argv[5]),
    "aggregation": sys.argv[6],
    "rounding": sys.argv[7],
    "min_count": int(sys.argv[8]),
    "require_test_completed": sys.argv[10] == "1",
    "split_strategy_regex": sys.argv[11] or None,
    "allow_mixed_reference_config": sys.argv[12] == "1",
    "allow_duplicate_split_strategies": sys.argv[13] == "1",
    "checkpoint_policy": "fixed_reference_epoch_last_ckpt",
    "selected_checkpoint_path": str(selected_checkpoint_path.resolve()),
    "reference_summary": reference_summary,
    "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
' "${CKPT_DIR}/${exp_name}" \
    "${reference_path}" \
    "${reference_task_name}" \
    "${reference_epoch}" \
    "${applied_max_epochs}" \
    "${REFERENCE_EPOCH_AGG}" \
    "${REFERENCE_EPOCH_ROUNDING}" \
    "${REFERENCE_EPOCH_MIN_COUNT}" \
    "${reference_summary_json}" \
    "${REFERENCE_REQUIRE_TEST_COMPLETED}" \
    "${reference_split_strategy_regex}" \
    "${REFERENCE_ALLOW_MIXED_CONFIG}" \
    "${REFERENCE_ALLOW_DUPLICATE_SPLITS}"
}

ptv3_infer() {
  local checkpoint_path="$1"
  local task_name="$2"
  local task_head="$3"
  local exp_name="$4"
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
    --dataset-group ptv3 \
    --model-type attention_v10_hetero_cls_ee \
    --task-name "${task_name}" \
    --split-strategy test_only \
    --split-name test \
    --task-head "${task_head}" \
    --checkpoint-path "${checkpoint_path}" \
    --output-dir "${OUTPUT_DIR}/${exp_name}/${task_name}" \
    --batch-size "${INFER_BATCH_SIZE}" \
    --device "${INFER_DEVICE}" \
    "${limit_args[@]}"
  status="$?"
  set -e
  end_utc="$(ptv3_utc_now)"
  end_sec="$(date +%s)"

  ptv3_record_time \
    "infer" \
    "${exp_name}" \
    "${task_name}" \
    "test_only" \
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

ptv3_done() {
  echo "[done] runtime summary: ${TIME_SUMMARY_PATH}"
}
