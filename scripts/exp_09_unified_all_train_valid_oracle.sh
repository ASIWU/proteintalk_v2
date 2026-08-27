#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="exp09_unified_all_train_valid_oracle"

# Match exp08's validated double-drug architecture defaults, but train one
# unified response head on both merged single-drug and native double-drug rows.
PAIR_FUSION_MODE="${PAIR_FUSION_MODE:-dual}"
PAIR_TYPE_FEATURES="${PAIR_TYPE_FEATURES:-1}"
MSE_INACTIVE_LABEL_WEIGHT="${MSE_INACTIVE_LABEL_WEIGHT:-0.2}"
USE_DDI="${USE_DDI:-1}"
GRAPH_PAIR_ADD_SCALE="${GRAPH_PAIR_ADD_SCALE:-0.5}"
SAVE_TOP_K="${SAVE_TOP_K:--1}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
SAVE_LAST_CKPT="${SAVE_LAST_CKPT:-1}"
MONITOR="${MONITOR:-none}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-{epoch}}"
REFERENCE_EPOCH_AGG="${EXP09_REFERENCE_EPOCH_AGG:-mean}"
REFERENCE_EPOCH_ROUNDING="${EXP09_REFERENCE_EPOCH_ROUNDING:-nearest}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

EXP09_EXP01_REFERENCE_PATH="${EXP09_EXP01_REFERENCE_PATH:-${REFERENCE_EXP01_5FOLD_CKPT_PATH:-}}"
EXP09_EXP06_REFERENCE_PATH="${EXP09_EXP06_REFERENCE_PATH:-${REFERENCE_EXP06_5FOLD_CKPT_PATH:-}}"
EXP09_VALID_SINGLE_EPOCH="${EXP09_VALID_SINGLE_EPOCH:-}"
EXP09_VALID_DOUBLE_EPOCH="${EXP09_VALID_DOUBLE_EPOCH:-}"
RUN_VALID="${RUN_VALID:-1}"
RUN_ORACLE="${RUN_ORACLE:-1}"
RUN_REPORT="${RUN_REPORT:-1}"
REPORT_DEVICE="${REPORT_DEVICE:-${INFER_DEVICE}}"
REPORT_INFER_BATCH_SIZE="${REPORT_INFER_BATCH_SIZE:-512}"
REPORT_NUM_WORKERS="${REPORT_NUM_WORKERS:-${NUM_WORKERS}}"

ptv3_print_settings "Exp09 unified-head all single+double training, valid reference-epoch extra eval, and oracle all-epoch extra eval"

if [[ "${SAVE_TOP_K}" != "-1" ]]; then
  echo "[error] exp09 oracle requires SAVE_TOP_K=-1 so every epoch checkpoint is retained" >&2
  exit 1
fi
if [[ "${SAVE_EVERY_N_EPOCHS}" != "1" ]]; then
  echo "[error] exp09 oracle requires SAVE_EVERY_N_EPOCHS=1" >&2
  exit 1
fi
if [[ "${SAVE_LAST_CKPT}" != "1" ]]; then
  echo "[error] exp09 requires SAVE_LAST_CKPT=1" >&2
  exit 1
fi
if [[ "${SCHEDULER_NAME}" == "plateau" ]]; then
  echo "[error] exp09 rejects SCHEDULER_NAME=plateau because all-data validation would affect the learning-rate schedule" >&2
  exit 1
fi

ptv3_run_preflight
"${PYTHON_BIN}" -m py_compile scripts/report_exp09_valid_oracle.py

resolve_exp09_reference_epoch() {
  local override_epoch="$1"
  local reference_path="$2"
  local task_name="$3"
  local task_head="$4"
  local split_regex="$5"
  local summary_json="$6"
  local label="$7"
  if [[ -n "${override_epoch}" ]]; then
    echo "${override_epoch}"
    return
  fi
  if [[ -z "${reference_path}" ]]; then
    echo "[error] ${label} requires a reference path or explicit epoch override" >&2
    echo "[error] set EXP09_EXP01_REFERENCE_PATH / EXP09_EXP06_REFERENCE_PATH or EXP09_VALID_SINGLE_EPOCH / EXP09_VALID_DOUBLE_EPOCH" >&2
    exit 1
  fi
  ptv3_reference_epoch \
    "${reference_path}" \
    "${task_name}" \
    "${task_head}" \
    "${split_regex}" \
    "${summary_json}"
}

ptv3_checkpoint_for_epoch() {
  local exp_name="$1"
  local epoch="$2"
  "${PYTHON_BIN}" -c 'import re, sys; from pathlib import Path
run_dir = Path(sys.argv[1])
target = int(sys.argv[2])
matches = []
for path in run_dir.glob("epoch=*.ckpt"):
    match = re.search(r"epoch=(\d+)", path.name)
    if match and int(match.group(1)) == target:
        matches.append(path)
if not matches:
    raise SystemExit(f"missing checkpoint for epoch={target}: {run_dir}")
matches.sort(key=lambda item: (item.stat().st_mtime, item.name))
print(matches[-1])' "${CKPT_DIR}/${exp_name}" "${epoch}"
}

ptv3_epoch_checkpoints() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import re, sys; from pathlib import Path
run_dir = Path(sys.argv[1])
items = []
for path in run_dir.glob("epoch=*.ckpt"):
    match = re.search(r"epoch=(\d+)", path.name)
    if match:
        items.append((int(match.group(1)), path))
for epoch, path in sorted(items):
    print(f"{epoch}\t{path}")' "${CKPT_DIR}/${exp_name}"
}

infer_exp09_single_extra() {
  local checkpoint_path="$1"
  local output_exp_name="$2"
  for task_name in \
    ptv3_extra_singledrug_mat1_480_faims \
    ptv3_extra_singledrug_mat1_qe \
    ptv3_extra_singledrug_mat2_480_faims \
    ptv3_extra_singledrug_mat2_qe \
    ptv3_extra_singledrug_mat3_qe \
    ptv3_extra_singledrug_mat4_qe; do
    ptv3_infer "${checkpoint_path}" "${task_name}" unified "${output_exp_name}"
  done
}

infer_exp09_double_extra() {
  local checkpoint_path="$1"
  local output_exp_name="$2"
  for task_name in \
    ptv3_extra_doubledrug_nature \
    ptv3_extra_doubledrug_nc \
    ptv3_extra_doubledrug_guomics; do
    ptv3_infer "${checkpoint_path}" "${task_name}" unified "${output_exp_name}"
  done
}

single_reference_summary_json="${LOG_DIR}/${EXP_PREFIX}_exp09_single_reference_epoch_summary.json"
double_reference_summary_json="${LOG_DIR}/${EXP_PREFIX}_exp09_double_reference_epoch_summary.json"
single_valid_epoch="$(resolve_exp09_reference_epoch \
  "${EXP09_VALID_SINGLE_EPOCH}" \
  "${EXP09_EXP01_REFERENCE_PATH}" \
  ptv3_main_singledrug \
  response \
  "^pert_stratified_5fold_fold[0-9]+$" \
  "${single_reference_summary_json}" \
  "exp09 valid single")"
double_valid_epoch="$(resolve_exp09_reference_epoch \
  "${EXP09_VALID_DOUBLE_EPOCH}" \
  "${EXP09_EXP06_REFERENCE_PATH}" \
  ptv3_main_doubledrug \
  synergy \
  "^pert_id_5fold_fold[0-9]+$" \
  "${double_reference_summary_json}" \
  "exp09 valid double")"

max_valid_epoch="${single_valid_epoch}"
if (( double_valid_epoch > max_valid_epoch )); then
  max_valid_epoch="${double_valid_epoch}"
fi
if (( MAX_EPOCHS <= max_valid_epoch )); then
  echo "[error] MAX_EPOCHS=${MAX_EPOCHS} must be greater than max valid epoch ${max_valid_epoch}" >&2
  exit 1
fi

all_unified_exp="${EXP_PREFIX}_unified_all_single_double_for_extra"
ptv3_train "${all_unified_exp}" \
  ptv3_main_doubledrug all_train_subset_test unified \
  --skip-test

valid_output_exp="${all_unified_exp}_valid"
if [[ "${RUN_INFERENCE}" == "1" && "${RUN_VALID}" == "1" ]]; then
  single_valid_ckpt="$(ptv3_checkpoint_for_epoch "${all_unified_exp}" "${single_valid_epoch}")"
  double_valid_ckpt="$(ptv3_checkpoint_for_epoch "${all_unified_exp}" "${double_valid_epoch}")"
  echo "[valid] exp07 epoch=${single_valid_epoch} checkpoint=${single_valid_ckpt}"
  echo "[valid] exp08 epoch=${double_valid_epoch} checkpoint=${double_valid_ckpt}"
  infer_exp09_single_extra "${single_valid_ckpt}" "${valid_output_exp}"
  infer_exp09_double_extra "${double_valid_ckpt}" "${valid_output_exp}"
fi

if [[ "${RUN_INFERENCE}" == "1" && "${RUN_ORACLE}" == "1" ]]; then
  while IFS=$'\t' read -r epoch checkpoint_path; do
    [[ -n "${epoch}" ]] || continue
    oracle_output_exp="$(printf "%s_oracle_epoch%03d" "${all_unified_exp}" "${epoch}")"
    echo "[oracle] epoch=${epoch} checkpoint=${checkpoint_path} output=${OUTPUT_DIR}/${oracle_output_exp}"
    infer_exp09_single_extra "${checkpoint_path}" "${oracle_output_exp}"
    infer_exp09_double_extra "${checkpoint_path}" "${oracle_output_exp}"
  done < <(ptv3_epoch_checkpoints "${all_unified_exp}")
fi

if [[ "${RUN_REPORT}" == "1" && "${RUN_INFERENCE}" == "1" && "${RUN_VALID}" == "1" && "${RUN_ORACLE}" == "1" ]]; then
  report_markdown="${LOG_DIR}/${all_unified_exp}_valid_oracle_eval.md"
  "${PYTHON_BIN}" scripts/report_exp09_valid_oracle.py \
    --valid-root "${OUTPUT_DIR}/${valid_output_exp}" \
    --oracle-root-glob "${OUTPUT_DIR}/${all_unified_exp}_oracle_epoch*" \
    --csv-out "${OUTPUT_DIR}/${all_unified_exp}_valid_oracle_eval.csv" \
    --json-out "${OUTPUT_DIR}/${all_unified_exp}_valid_oracle_eval.json" \
    --format markdown | tee "${report_markdown}"
  echo "[report] ${report_markdown}"
fi

ptv3_done
