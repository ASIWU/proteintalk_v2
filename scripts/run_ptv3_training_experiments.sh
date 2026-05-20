#!/usr/bin/env bash
set -euo pipefail

# Legacy all-in-one runner.  For human-readable experiment scripts, use:
#   scripts/exp_01_single_pert_stratified_5fold.sh
#   scripts/exp_02_single_cell_type_5fold.sh
#   scripts/exp_03_single_cell_5fold.sh
#   scripts/exp_04_single_no_mse_5fold.sh
#   scripts/exp_05_single_no_pdi_5fold.sh
#   scripts/exp_06_double_pert_pair_5fold.sh
#   scripts/exp_07_extra_single_all_train_infer.sh
#   scripts/exp_08_extra_double_all_train_infer.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
BATCH_SIZE="${BATCH_SIZE:-2}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1.0}"
INFER_LIMIT_BATCHES="${INFER_LIMIT_BATCHES:-}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXP_PREFIX="${EXP_PREFIX:-$(date +%Y%m%d)_ptv3}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
INFER_DEVICE="${INFER_DEVICE:-cuda:0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
BEST_CKPT_METRIC="${BEST_CKPT_METRIC:-valid_auprc}"
USER_SET_MONITOR="${MONITOR+x}"
MONITOR="${MONITOR:-}"
if [[ -n "${SAVE_EVERY_N_TRAIN_STEPS:-}" && -z "${USER_SET_MONITOR}" ]]; then
  MONITOR="none"
fi
MONITOR_MODE="${MONITOR_MODE:-}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
ALLOW_NONFINITE_MONITOR="${ALLOW_NONFINITE_MONITOR:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
TIME_SUMMARY_PATH="${TIME_SUMMARY_PATH:-${LOG_DIR}/${EXP_PREFIX}_runtime_summary.tsv}"

read -r -a FOLD_LIST <<< "${FOLDS}"

common_train_args=(
  --dataset-group ptv3
  --model-type attention_v10_hetero_cls_ee
  --batch-size "${BATCH_SIZE}"
  --max-epochs "${MAX_EPOCHS}"
  --accelerator gpu
  --devices "${DEVICES}"
  --strategy ddp_find_unused_parameters_true
  --precision "${PRECISION}"
  --num-workers "${NUM_WORKERS}"
  --checkpoint-dir "${CKPT_DIR}"
  --log-dir "${LOG_DIR}"
  --save-every-n-epochs "${SAVE_EVERY_N_EPOCHS}"
  --best-ckpt-metric "${BEST_CKPT_METRIC}"
  --log-every-n-steps "${LOG_EVERY_N_STEPS}"
  --limit-train-batches "${LIMIT_TRAIN_BATCHES}"
  --limit-val-batches "${LIMIT_VAL_BATCHES}"
  --limit-test-batches "${LIMIT_TEST_BATCHES}"
)

if [[ -n "${MONITOR}" ]]; then
  common_train_args+=(--monitor "${MONITOR}")
fi
if [[ -n "${MONITOR_MODE}" ]]; then
  common_train_args+=(--monitor-mode "${MONITOR_MODE}")
fi
if [[ "${ALLOW_NONFINITE_MONITOR}" == "1" ]]; then
  common_train_args+=(--allow-nonfinite-monitor)
fi

ensure_clean_path() {
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

utc_now() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

init_time_summary() {
  mkdir -p "$(dirname "${TIME_SUMMARY_PATH}")"
  printf "kind\texperiment\ttask_name\tsplit_strategy\tsplit_name\tstatus\tstart_utc\tend_utc\tduration_sec\tartifact\n" > "${TIME_SUMMARY_PATH}"
}

record_time_summary() {
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

run_preflight() {
  "${PYTHON_BIN}" -m py_compile \
    train.py \
    infer.py \
    dataset/training_ready_dataset.py \
    model/training_ready_models.py \
    model/training_ready_lightning.py \
    utils/00_standardize_rawdata.py \
    utils/01_validate_standardized_outputs.py \
    utils/02_build_training_ready_data.py \
    utils/03_validate_training_ready_outputs.py \
    utils/09_build_data_splits.py
  "${PYTHON_BIN}" utils/01_validate_standardized_outputs.py
  "${PYTHON_BIN}" utils/03_validate_training_ready_outputs.py
}

run_train() {
  local exp_name="$1"
  local task_name="$2"
  local split_strategy="$3"
  local task_head="$4"
  shift 4
  ensure_clean_path "${CKPT_DIR}/${exp_name}" "checkpoint directory"
  ensure_clean_path "${LOG_DIR}/${exp_name}" "log directory"
  local start_utc
  local end_utc
  local start_sec
  local end_sec
  local status
  start_utc="$(utc_now)"
  start_sec="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u train.py \
    "${common_train_args[@]}" \
    --experiment-name "${exp_name}" \
    --task-name "${task_name}" \
    --split-strategy "${split_strategy}" \
    --task-head "${task_head}" \
    "$@"
  status="$?"
  set -e
  end_utc="$(utc_now)"
  end_sec="$(date +%s)"
  record_time_summary \
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

best_checkpoint() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
exp=Path(sys.argv[1])
manifest_path=exp/"run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing run manifest: {manifest_path}")
manifest=json.load(manifest_path.open())
if manifest.get("run_status") != "fit_completed":
    raise SystemExit(f"run is not fit_completed: {manifest_path}")
checkpoint=manifest.get("best_model_path") or str(exp/"last.ckpt")
if not Path(checkpoint).exists():
    raise SystemExit(f"checkpoint does not exist: {checkpoint}")
print(checkpoint)' "${CKPT_DIR}/${exp_name}"
}

run_infer() {
  local checkpoint_path="$1"
  local task_name="$2"
  local task_head="$3"
  local exp_name="$4"
  local -a limit_args=()
  if [[ -n "${INFER_LIMIT_BATCHES}" ]]; then
    limit_args=(--limit-batches "${INFER_LIMIT_BATCHES}")
  fi
  ensure_clean_path "${OUTPUT_DIR}/${exp_name}/${task_name}" "inference output directory"
  local start_utc
  local end_utc
  local start_sec
  local end_sec
  local status
  start_utc="$(utc_now)"
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
  end_utc="$(utc_now)"
  end_sec="$(date +%s)"
  record_time_summary \
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

init_time_summary

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  run_preflight
fi

for fold in "${FOLD_LIST[@]}"; do
  run_train "${EXP_PREFIX}_single_pert_stratified_fold${fold}" \
    ptv3_main_singledrug "pert_stratified_5fold_fold${fold}" response

  run_train "${EXP_PREFIX}_single_cell_type_fold${fold}" \
    ptv3_main_singledrug "cell_type_5fold_fold${fold}" response

  run_train "${EXP_PREFIX}_single_cell_fold${fold}" \
    ptv3_main_singledrug "cell_5fold_fold${fold}" response

  run_train "${EXP_PREFIX}_single_no_mse_fold${fold}" \
    ptv3_main_singledrug "pert_stratified_5fold_fold${fold}" response \
    --no-mse-loss

  run_train "${EXP_PREFIX}_single_no_pdi_fold${fold}" \
    ptv3_main_singledrug "pert_stratified_5fold_fold${fold}" response \
    --pdi-mode zero

  run_train "${EXP_PREFIX}_double_pert_pair_fold${fold}" \
    ptv3_main_doubledrug "pert_id_5fold_fold${fold}" synergy
done

all_single_exp="${EXP_PREFIX}_all_single_for_extra"
run_train "${all_single_exp}" \
  ptv3_main_singledrug all_train_subset_test response \
  --skip-test

all_single_double_exp="${EXP_PREFIX}_all_single_double_for_extra"
run_train "${all_single_double_exp}" \
  ptv3_main_doubledrug all_train_subset_test synergy \
  --skip-test

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  all_single_ckpt="$(best_checkpoint "${all_single_exp}")"
  all_single_double_ckpt="$(best_checkpoint "${all_single_double_exp}")"

  for task_name in \
    ptv3_extra_singledrug_mat1_480_faims \
    ptv3_extra_singledrug_mat1_qe \
    ptv3_extra_singledrug_mat2_480_faims \
    ptv3_extra_singledrug_mat2_qe \
    ptv3_extra_singledrug_mat3_qe \
    ptv3_extra_singledrug_mat4_qe; do
    run_infer "${all_single_ckpt}" "${task_name}" response "${all_single_exp}"
  done

  for task_name in \
    ptv3_extra_doubledrug_nature \
    ptv3_extra_doubledrug_nc \
    ptv3_extra_doubledrug_guomics; do
    run_infer "${all_single_double_ckpt}" "${task_name}" synergy "${all_single_double_exp}"
  done
fi
