#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

usage() {
  cat <<'EOF'
Usage:
  RUN_MODE=screen   bash scripts/run_cell_celltype_llm_clip10_param_search.sh
  RUN_MODE=full     bash scripts/run_cell_celltype_llm_clip10_param_search.sh
  RUN_MODE=selected bash scripts/run_cell_celltype_llm_clip10_param_search.sh
  RUN_MODE=all      bash scripts/run_cell_celltype_llm_clip10_param_search.sh

Runs the legacy clip10 hyperparameter search while requiring both frozen Cell LLM
and frozen cell_type LLM embeddings.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
RUN_MODE="${RUN_MODE:-screen}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
BASE_PREFIX_ROOT="${BASE_PREFIX_ROOT:-20260608_cell_celltype_llm_clip10_tune}"
BASE_PREFIX="${BASE_PREFIX:-}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

selected_prefix_for_screen() {
  local screen="$1"
  echo "${screen/cell_celltype_llm_clip10_tune/cell_celltype_llm_clip10_tuned_selected}"
}

select_screen_prefix() {
  if [[ -n "${SCREEN_PREFIX:-}" ]]; then
    echo "${SCREEN_PREFIX}"
    return
  fi
  if [[ -n "${BASE_PREFIX}" ]]; then
    echo "${BASE_PREFIX}"
    return
  fi
  local idx=1
  local candidate selected
  while true; do
    candidate="${BASE_PREFIX_ROOT}_v${idx}"
    selected="$(selected_prefix_for_screen "${candidate}")"
    if [[ "${ALLOW_EXISTING_RUN}" == "1" || ! -f "${OUTPUT_DIR}/${selected}_cell_drug_dose_time_eval.csv" ]]; then
      echo "${candidate}"
      return
    fi
    idx=$((idx + 1))
  done
}

SCREEN_PREFIX="$(select_screen_prefix)"
FULL_PREFIX="${FULL_PREFIX:-${SCREEN_PREFIX}_full}"
SELECTED_PREFIX="${SELECTED_PREFIX:-$(selected_prefix_for_screen "${SCREEN_PREFIX}")}"

validate_inputs() {
  "${PYTHON_BIN}" - "${CELL_LLM_EMBEDDING_PATH}" "${CELL_TYPE_LLM_EMBEDDING_PATH}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cell_path = Path(sys.argv[1])
cell_type_path = Path(sys.argv[2])


def load_matrix(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise SystemExit(f"missing embedding artifact: {path}")
    payload = np.load(path, allow_pickle=False)
    if "embedding_matrix" not in payload.files:
        raise SystemExit(f"{path} must contain embedding_matrix")
    meta = {}
    if "meta_json" in payload.files:
        meta = json.loads(str(payload["meta_json"].item()))
    return payload["embedding_matrix"], meta


cell_matrix, cell_meta = load_matrix(cell_path)
cell_type_matrix, cell_type_meta = load_matrix(cell_type_path)
if cell_matrix.shape != (74, 4096):
    raise SystemExit(f"{cell_path} expected shape (74, 4096), got {cell_matrix.shape}")
if cell_meta.get("kind") != "cell_llm_embedding" or cell_meta.get("index_column") != "Cell_index":
    raise SystemExit(f"{cell_path} has invalid metadata: {cell_meta}")
if cell_type_matrix.shape != (14, 4096):
    raise SystemExit(f"{cell_type_path} expected shape (14, 4096), got {cell_type_matrix.shape}")
if cell_type_meta.get("kind") != "cell_type_llm_embedding" or cell_type_meta.get("index_column") != "cell_type_index":
    raise SystemExit(f"{cell_type_path} has invalid metadata: {cell_type_meta}")
for name, matrix in (("Cell", cell_matrix), ("cell_type", cell_type_matrix)):
    if not np.allclose(matrix[0], 0.0):
        raise SystemExit(f"{name} embedding row 0 must be zero")
    if not np.isfinite(matrix[1:]).all() or np.any(np.all(np.isclose(matrix[1:], 0.0), axis=1)):
        raise SystemExit(f"{name} non-no rows must be finite and nonzero")

root = Path("data/training_ready/ptv3")
meta = json.loads((root / "global_meta.json").read_text())
if len(meta["value_to_index"]["Cell"]) != 74 or int(meta["value_to_index"]["Cell"].get("no", -1)) != 0:
    raise SystemExit("global Cell mapping must contain 74 values with no=0")
if len(meta["value_to_index"]["cell_type"]) != 14 or int(meta["value_to_index"]["cell_type"].get("no", -1)) != 0:
    raise SystemExit("global cell_type mapping must contain 14 values with no=0")

for task in meta.get("task_names", []):
    table_path = root / "tasks" / str(task) / "feature_table.parquet"
    csv_path = table_path.with_suffix(".csv")
    columns = ["Cell_index", "cell_type_index", "pert_dose1", "pert_dose2"]
    if table_path.exists():
        df = pd.read_parquet(table_path, columns=columns)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, usecols=columns, low_memory=False)
    else:
        raise SystemExit(f"missing feature_table for {task}")
    if int(pd.to_numeric(df["Cell_index"], errors="coerce").fillna(0).astype(int).max()) >= cell_matrix.shape[0]:
        raise SystemExit(f"{task}: Cell_index exceeds Cell embedding rows")
    if int(pd.to_numeric(df["cell_type_index"], errors="coerce").fillna(0).astype(int).max()) >= cell_type_matrix.shape[0]:
        raise SystemExit(f"{task}: cell_type_index exceeds cell_type embedding rows")
    for field in ("pert_dose1", "pert_dose2"):
        numeric = pd.to_numeric(df[field], errors="coerce").dropna()
        if not numeric.empty and (float(numeric.min()) < 0.0 or float(numeric.max()) > 10.0):
            raise SystemExit(f"{task}: {field} outside clip10 range")
print("[validate] Cell and cell_type LLM artifacts passed")
PY
}

echo "[settings] RUN_MODE=${RUN_MODE}; SCREEN_PREFIX=${SCREEN_PREFIX}; FULL_PREFIX=${FULL_PREFIX}; SELECTED_PREFIX=${SELECTED_PREFIX}; ALLOW_EXISTING_RUN=${ALLOW_EXISTING_RUN}"
echo "[settings] CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
echo "[settings] CELL_TYPE_LLM_EMBEDDING_PATH=${CELL_TYPE_LLM_EMBEDDING_PATH}"

"${PYTHON_BIN}" -m py_compile \
  train.py infer.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py \
  scripts/report_cell_drug_time_eval.py scripts/report_cell_celltype_llm_clip10_param_search.py
bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_param_search.sh
validate_inputs

env \
  RUN_MODE="${RUN_MODE}" \
  SCREEN_PREFIX="${SCREEN_PREFIX}" \
  FULL_PREFIX="${FULL_PREFIX}" \
  SELECTED_PREFIX="${SELECTED_PREFIX}" \
  ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN}" \
  TUNING_REPORT_SCRIPT=scripts/report_cell_celltype_llm_clip10_param_search.py \
  CELL_LLM_MODE=frozen \
  CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH}" \
  CELL_LLM_FUSION_MODE="${CELL_LLM_FUSION_MODE:-covariate}" \
  CELL_LLM_CONDITION_SCALE="${CELL_LLM_CONDITION_SCALE:-0.0}" \
  CELL_LLM_LOGIT_SCALE="${CELL_LLM_LOGIT_SCALE:-0.0}" \
  CELL_LLM_DROPOUT="${CELL_LLM_DROPOUT:-0.0}" \
  CELL_TYPE_LLM_MODE=frozen \
  CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH}" \
  CELL_TYPE_LLM_FUSION_MODE="${CELL_TYPE_LLM_FUSION_MODE:-covariate}" \
  CELL_TYPE_LLM_CONDITION_SCALE="${CELL_TYPE_LLM_CONDITION_SCALE:-0.0}" \
  CELL_TYPE_LLM_LOGIT_SCALE="${CELL_TYPE_LLM_LOGIT_SCALE:-0.0}" \
  CELL_TYPE_LLM_DROPOUT="${CELL_TYPE_LLM_DROPOUT:-0.0}" \
  bash scripts/run_cell_llm_clip10_param_search.sh

echo "[done] Cell + cell_type LLM clip10 parameter search finished; RUN_MODE=${RUN_MODE}; SCREEN_PREFIX=${SCREEN_PREFIX}"
