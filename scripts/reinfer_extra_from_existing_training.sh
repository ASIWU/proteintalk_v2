#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SET_NAME="reinfer_extra_from_existing_training"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ptv3_experiment_common.sh"

CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-best}"
EXTRA_MODE="${EXTRA_MODE:-single}"
SOURCE_EXP_NAME="${SOURCE_EXP_NAME:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
SHOW_RESULTS="${SHOW_RESULTS:-1}"

if [[ -z "${SOURCE_EXP_NAME}" ]]; then
  case "${EXTRA_MODE}" in
    single)
      SOURCE_EXP_NAME="20260510_extra_single_all_train_infer_all_single_for_extra"
      ;;
    double)
      SOURCE_EXP_NAME="20260510_extra_double_all_train_infer_all_single_double_for_extra"
      ;;
    *)
      echo "[error] SOURCE_EXP_NAME is required when EXTRA_MODE is not single or double" >&2
      exit 1
      ;;
  esac
fi

if [[ "${EXTRA_MODE}" == "auto" ]]; then
  EXTRA_MODE="$("${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
manifest_path = Path(sys.argv[1]) / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing source run manifest: {manifest_path}")
manifest = json.load(manifest_path.open())
task_name = manifest.get("task_name")
task_head = manifest.get("task_head")
if task_name == "ptv3_main_singledrug" and task_head == "response":
    print("single")
elif task_name == "ptv3_main_doubledrug" and task_head == "synergy":
    print("double")
else:
    raise SystemExit(f"cannot infer EXTRA_MODE from task_name={task_name!r}, task_head={task_head!r}")' \
    "${CKPT_DIR}/${SOURCE_EXP_NAME}")"
fi

case "${CHECKPOINT_POLICY}" in
  best)
    resolved_checkpoint="$(ptv3_best_checkpoint "${SOURCE_EXP_NAME}")"
    ;;
  last)
    resolved_checkpoint="$(ptv3_last_checkpoint "${SOURCE_EXP_NAME}")"
    ;;
  reference|reference_policy)
    resolved_checkpoint="$("${PYTHON_BIN}" -c 'import json, sys; from pathlib import Path
manifest_path = Path(sys.argv[1]) / "run_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing source run manifest: {manifest_path}")
manifest = json.load(manifest_path.open())
policy = manifest.get("reference_epoch_policy")
if not isinstance(policy, dict):
    raise SystemExit(f"source run does not contain reference_epoch_policy: {manifest_path}")
checkpoint = policy.get("selected_checkpoint_path")
if not checkpoint:
    raise SystemExit(f"reference_epoch_policy.selected_checkpoint_path is missing: {manifest_path}")
checkpoint_path = Path(checkpoint)
if not checkpoint_path.exists():
    raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
print(checkpoint_path)' "${CKPT_DIR}/${SOURCE_EXP_NAME}")"
    ;;
  explicit)
    if [[ -z "${CHECKPOINT_PATH}" ]]; then
      echo "[error] CHECKPOINT_PATH is required when CHECKPOINT_POLICY=explicit" >&2
      exit 1
    fi
    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
      echo "[error] checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
      exit 1
    fi
    resolved_checkpoint="${CHECKPOINT_PATH}"
    ;;
  *)
    echo "[error] unsupported CHECKPOINT_POLICY=${CHECKPOINT_POLICY}; use best, last, reference, or explicit" >&2
    exit 1
    ;;
esac

OUTPUT_EXP_NAME="${OUTPUT_EXP_NAME:-${SOURCE_EXP_NAME}_${CHECKPOINT_POLICY}_reinfer}"

case "${EXTRA_MODE}" in
  single)
    infer_head="response"
    infer_tasks=(
      ptv3_extra_singledrug_mat1_480_faims
      ptv3_extra_singledrug_mat1_qe
      ptv3_extra_singledrug_mat2_480_faims
      ptv3_extra_singledrug_mat2_qe
      ptv3_extra_singledrug_mat3_qe
      ptv3_extra_singledrug_mat4_qe
    )
    ;;
  double)
    infer_head="synergy"
    infer_tasks=(
      ptv3_extra_doubledrug_nature
      ptv3_extra_doubledrug_nc
      ptv3_extra_doubledrug_guomics
    )
    ;;
  *)
    echo "[error] unsupported EXTRA_MODE=${EXTRA_MODE}; use single, double, or auto" >&2
    exit 1
    ;;
esac

ptv3_init_time_summary
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON_BIN}" -m py_compile infer.py scripts/show_extra_results.py
fi

echo "[reuse] no training will be run"
echo "[reuse] source run: ${CKPT_DIR}/${SOURCE_EXP_NAME}"
echo "[reuse] checkpoint policy: ${CHECKPOINT_POLICY}"
echo "[reuse] checkpoint: ${resolved_checkpoint}"
echo "[reuse] extra mode: ${EXTRA_MODE}; head=${infer_head}; tasks=${#infer_tasks[@]}"
echo "[reuse] output: ${OUTPUT_DIR}/${OUTPUT_EXP_NAME}"

for task_name in "${infer_tasks[@]}"; do
  ptv3_infer "${resolved_checkpoint}" "${task_name}" "${infer_head}" "${OUTPUT_EXP_NAME}"
done

if [[ "${SHOW_RESULTS}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/show_extra_results.py "${OUTPUT_DIR}/${OUTPUT_EXP_NAME}" --no-aggregate
fi

ptv3_done
