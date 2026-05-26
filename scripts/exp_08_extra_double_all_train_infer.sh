#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="extra_double_all_train_infer"

# Match the validated double-drug 5-fold default used by exp_06.
PAIR_FUSION_MODE="${PAIR_FUSION_MODE:-dual}"
PAIR_TYPE_FEATURES="${PAIR_TYPE_FEATURES:-1}"
MSE_INACTIVE_LABEL_WEIGHT="${MSE_INACTIVE_LABEL_WEIGHT:-0.2}"
USE_DDI="${USE_DDI:-1}"
GRAPH_PAIR_ADD_SCALE="${GRAPH_PAIR_ADD_SCALE:-0.5}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

ptv3_print_settings "Baseline4 train on all single+double data, then infer extra double-drug datasets"
ptv3_run_preflight

all_single_double_exp="${EXP_PREFIX}_all_single_double_for_extra"
train_args=(--skip-test)
checkpoint_policy="best_validation"
if [[ -n "${REFERENCE_5FOLD_CKPT_PATH}" ]]; then
  if [[ "${SAVE_LAST_CKPT}" != "1" ]]; then
    echo "[error] reference epoch policy requires SAVE_LAST_CKPT=1 so last.ckpt can be used for extra inference" >&2
    exit 1
  fi
  if [[ "${SCHEDULER_NAME}" == "plateau" ]]; then
    echo "[error] reference epoch policy rejects SCHEDULER_NAME=plateau because all-data validation would affect the learning-rate schedule" >&2
    exit 1
  fi
  reference_split_strategy_regex="${REFERENCE_SPLIT_STRATEGY_REGEX:-^pert_id_5fold_fold[0-9]+$}"
  reference_summary_json="${LOG_DIR}/${all_single_double_exp}_reference_epoch_summary.json"
  reference_epoch="$(ptv3_reference_epoch \
    "${REFERENCE_5FOLD_CKPT_PATH}" \
    ptv3_main_doubledrug \
    synergy \
    "${reference_split_strategy_regex}" \
    "${reference_summary_json}")"
  reference_max_epochs="$((reference_epoch + 1))"
  checkpoint_policy="reference_epoch"
  echo "[checkpoint-policy] extra double uses reference ${REFERENCE_EPOCH_AGG} epoch=${reference_epoch}; training all-single-double for max_epochs=${reference_max_epochs} and using last.ckpt"
  train_args+=(--max-epochs "${reference_max_epochs}" --monitor none)
fi

ptv3_train "${all_single_double_exp}" \
  ptv3_main_doubledrug all_train_subset_test synergy \
  "${train_args[@]}"
if [[ "${checkpoint_policy}" == "reference_epoch" ]]; then
  ptv3_record_reference_epoch_policy \
    "${all_single_double_exp}" \
    "${REFERENCE_5FOLD_CKPT_PATH}" \
    ptv3_main_doubledrug \
    "${reference_epoch}" \
    "${reference_max_epochs}" \
    "${reference_split_strategy_regex}" \
    "${reference_summary_json}"
fi

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  if [[ "${checkpoint_policy}" == "reference_epoch" ]]; then
    all_single_double_ckpt="$(ptv3_last_checkpoint "${all_single_double_exp}")"
  else
    all_single_double_ckpt="$(ptv3_best_checkpoint "${all_single_double_exp}")"
  fi
  for task_name in \
    ptv3_extra_doubledrug_nature \
    ptv3_extra_doubledrug_nc \
    ptv3_extra_doubledrug_guomics; do
    ptv3_infer "${all_single_double_ckpt}" "${task_name}" synergy "${all_single_double_exp}"
  done

  "${PYTHON_BIN}" scripts/report_extra_doubledrug_test_label_auprc.py \
    "${OUTPUT_DIR}/${all_single_double_exp}" \
    --allow-partial \
    --csv-out "${OUTPUT_DIR}/${all_single_double_exp}/extra_doubledrug_test_label_auprc.csv" \
    --json-out "${OUTPUT_DIR}/${all_single_double_exp}/extra_doubledrug_test_label_auprc.json" \
    --format markdown
fi

ptv3_done
