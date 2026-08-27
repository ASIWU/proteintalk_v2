#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="exp13_ptv1_extra_single_from_exp11"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv1_experiment_common.sh"

MODE="${1:-${EXP13_MODE:-direct}}"
EXP11_EXP_NAME="${EXP11_EXP_NAME:-${EXP_PREFIX}_ptv1_random_split}"

ptv1_exp11_checkpoint() {
  if [[ -n "${EXP11_CKPT_PATH:-}" ]]; then
    printf "%s\n" "${EXP11_CKPT_PATH}"
  else
    ptv1_best_checkpoint "${EXP11_EXP_NAME}"
  fi
}

ptv1_exp11_manifest() {
  local checkpoint_path="$1"
  "${PYTHON_BIN}" -c 'import sys; from pathlib import Path
checkpoint = Path(sys.argv[1])
manifest = checkpoint.resolve().parent / "run_manifest.json"
if not manifest.exists():
    raise SystemExit(f"missing exp11 run_manifest.json next to checkpoint: {manifest}")
print(manifest)' "${checkpoint_path}"
}

ptv1_exp11_best_epoch() {
  local manifest_path="$1"
  "${PYTHON_BIN}" -c 'import json, re, sys; from pathlib import Path
manifest_path = Path(sys.argv[1])
with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("dataset_group") != "ptv1":
    raise SystemExit(f"expected ptv1 manifest: {manifest_path}")
if manifest.get("task_name") != "ptv1_aivc":
    raise SystemExit(f"expected ptv1_aivc manifest: {manifest_path}")
if manifest.get("split_strategy") != "fixed_experiment_type":
    raise SystemExit(f"expected fixed_experiment_type exp11 manifest: {manifest_path}")
if manifest.get("run_status") != "fit_completed":
    raise SystemExit(f"exp11 manifest is not fit_completed: {manifest_path}")
checkpoint = manifest.get("best_model_path") or manifest.get("test_checkpoint_path")
if not checkpoint:
    raise SystemExit(f"exp11 manifest has no selected checkpoint path: {manifest_path}")
match = re.search(r"epoch=(\d+)", str(checkpoint))
if match is None:
    raise SystemExit(f"could not parse epoch from exp11 checkpoint path: {checkpoint}")
print(int(match.group(1)))' "${manifest_path}"
}

ptv1_train_completed() {
  local exp_name="$1"
  "${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
manifest_path = Path(sys.argv[1]) / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(1)
with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
raise SystemExit(0 if manifest.get("run_status") == "fit_completed" else 1)' "${CKPT_DIR}/${exp_name}"
}

ptv1_record_exp11_reference_policy() {
  local exp_name="$1"
  local exp11_manifest_path="$2"
  local exp11_checkpoint_path="$3"
  local selected_epoch="$4"
  local applied_max_epochs="$5"
  "${PYTHON_BIN}" -c 'import json, sys; from datetime import datetime, timezone; from pathlib import Path
run_dir = Path(sys.argv[1])
manifest_path = run_dir / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing all-train manifest: {manifest_path}")
selected_checkpoint_path = run_dir / "last.ckpt"
if not selected_checkpoint_path.exists():
    raise SystemExit(f"exp11 reference policy requires last.ckpt, but it does not exist: {selected_checkpoint_path}")
with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
manifest["exp13_from_exp11_policy"] = {
    "enabled": True,
    "reference_manifest_path": str(Path(sys.argv[2]).resolve()),
    "reference_checkpoint_path": str(Path(sys.argv[3]).resolve()),
    "reference_task_name": "ptv1_aivc",
    "reference_split_strategy": "fixed_experiment_type",
    "selected_epoch": int(sys.argv[4]),
    "applied_max_epochs": int(sys.argv[5]),
    "checkpoint_policy": "exp11_best_epoch_all_train_last_ckpt",
    "selected_checkpoint_path": str(selected_checkpoint_path.resolve()),
    "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
' "${CKPT_DIR}/${exp_name}" "${exp11_manifest_path}" "${exp11_checkpoint_path}" "${selected_epoch}" "${applied_max_epochs}"
}

ptv1_print_settings "PTV1 extra single-drug infer from exp11 (${MODE})"
ptv1_run_preflight

exp11_ckpt="$(ptv1_exp11_checkpoint)"
exp11_manifest="$(ptv1_exp11_manifest "${exp11_ckpt}")"

case "${MODE}" in
  direct)
    exp13_exp="${EXP13_EXP_NAME:-${EXP_PREFIX}_extra_direct_from_exp11}"
    echo "[checkpoint-policy] exp13 direct uses exp11 best checkpoint: ${exp11_ckpt}"
    if [[ "${RUN_INFERENCE}" == "1" ]]; then
      ptv1_infer "${exp11_ckpt}" ptv1_extra_singledrug response "${exp13_exp}"
    fi
    ;;
  all_train)
    if [[ "${SAVE_LAST_CKPT}" != "1" ]]; then
      echo "[error] exp13 all_train requires SAVE_LAST_CKPT=1 so last.ckpt can be used for extra inference" >&2
      exit 1
    fi
    if [[ "${SCHEDULER_NAME}" == "plateau" ]]; then
      echo "[error] exp13 all_train rejects SCHEDULER_NAME=plateau because all-data validation would affect the learning-rate schedule" >&2
      exit 1
    fi
    selected_epoch="$(ptv1_exp11_best_epoch "${exp11_manifest}")"
    reference_max_epochs="$((selected_epoch + 1))"
    all_ptv1_exp="${EXP13_EXP_NAME:-${EXP_PREFIX}_all_ptv1_for_extra_from_exp11}"
    echo "[checkpoint-policy] exp13 all_train uses exp11 best epoch=${selected_epoch}; training all-PTV1 for max_epochs=${reference_max_epochs} and using last.ckpt"

    if ptv1_train_completed "${all_ptv1_exp}"; then
      echo "[skip] all-train checkpoint already fit_completed: ${CKPT_DIR}/${all_ptv1_exp}"
    else
      ptv1_train "${all_ptv1_exp}" \
        ptv1_aivc all_train_subset_test response \
        --skip-test \
        --max-epochs "${reference_max_epochs}" \
        --monitor none
    fi

    ptv1_record_exp11_reference_policy \
      "${all_ptv1_exp}" \
      "${exp11_manifest}" \
      "${exp11_ckpt}" \
      "${selected_epoch}" \
      "${reference_max_epochs}"

    if [[ "${RUN_INFERENCE}" == "1" ]]; then
      all_ptv1_ckpt="$(ptv1_last_checkpoint "${all_ptv1_exp}")"
      ptv1_infer "${all_ptv1_ckpt}" ptv1_extra_singledrug response "${all_ptv1_exp}"
    fi
    ;;
  *)
    echo "[error] unknown EXP13_MODE: ${MODE}; expected direct or all_train" >&2
    exit 2
    ;;
esac

ptv1_done
