#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="exp13_ptv1_extra_single_all_train_infer"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv1_experiment_common.sh"

ptv1_latest_exp12_reference_glob() {
  local newest_name=""
  local newest_base=""
  while read -r _mtime name; do
    local base
    base="$(sed -E 's/fold[0-4]$//' <<< "${name}")"
    if [[ -n "${base}" ]] && compgen -G "${CKPT_DIR}/${base}fold*" >/dev/null; then
      newest_name="${name}"
      newest_base="${base}"
      break
    fi
  done < <(find "${CKPT_DIR}" -maxdepth 1 -type d -name '*ptv1_unseen_drug_fold[0-4]' -printf '%T@ %f\n' 2>/dev/null | sort -nr)

  if [[ -n "${newest_name}" ]]; then
    printf "%s/%sfold*" "${CKPT_DIR}" "${newest_base}"
  fi
}

ptv1_print_settings "PTV1 all-train, then test-only infer PTV1 extra single-drug"
ptv1_run_preflight

reference_path="${REFERENCE_5FOLD_CKPT_PATH:-}"
if [[ -z "${reference_path}" ]]; then
  reference_path="$(ptv1_latest_exp12_reference_glob)"
fi
if [[ -z "${reference_path}" ]]; then
  echo "[error] exp13 needs exp12 reference folds. Set REFERENCE_5FOLD_CKPT_PATH or run exp_12 first." >&2
  exit 1
fi
if [[ "${SAVE_LAST_CKPT}" != "1" ]]; then
  echo "[error] reference epoch policy requires SAVE_LAST_CKPT=1 so last.ckpt can be used for extra inference" >&2
  exit 1
fi
if [[ "${SCHEDULER_NAME}" == "plateau" ]]; then
  echo "[error] reference epoch policy rejects SCHEDULER_NAME=plateau because all-data validation would affect the learning-rate schedule" >&2
  exit 1
fi

all_ptv1_exp="${EXP_PREFIX}_all_ptv1_for_extra"
reference_split_strategy_regex="${REFERENCE_SPLIT_STRATEGY_REGEX:-^pert_id_5fold_fold[0-9]+$}"
reference_summary_json="${LOG_DIR}/${all_ptv1_exp}_reference_epoch_summary.json"
reference_epoch="$(ptv1_reference_epoch \
  "${reference_path}" \
  ptv1_aivc \
  response \
  "${reference_split_strategy_regex}" \
  "${reference_summary_json}")"
reference_max_epochs="$((reference_epoch + 1))"
echo "[checkpoint-policy] extra PTV1 uses ${REFERENCE_EPOCH_AGG} epoch=${reference_epoch}; training all-PTV1 for max_epochs=${reference_max_epochs} and using last.ckpt"

ptv1_train "${all_ptv1_exp}" \
  ptv1_aivc all_train_subset_test response \
  --skip-test \
  --max-epochs "${reference_max_epochs}" \
  --monitor none

ptv1_record_reference_epoch_policy \
  "${all_ptv1_exp}" \
  "${reference_path}" \
  ptv1_aivc \
  "${reference_epoch}" \
  "${reference_max_epochs}" \
  "${reference_split_strategy_regex}" \
  "${reference_summary_json}"

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  all_ptv1_ckpt="$(ptv1_last_checkpoint "${all_ptv1_exp}")"
  ptv1_infer "${all_ptv1_ckpt}" ptv1_extra_singledrug response "${all_ptv1_exp}"
fi

ptv1_done
