#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
CELL_LLM_EMBEDDING_PATH="${CELL_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz}"
CELL_TYPE_LLM_EMBEDDING_PATH="${CELL_TYPE_LLM_EMBEDDING_PATH:-data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz}"
BASE_PREFIX_ROOT="${BASE_PREFIX_ROOT:-20260608_cell_celltype_llm_clip10_selected}"
BASE_PREFIX="${BASE_PREFIX:-}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-0}"
RUN_REPORT="${RUN_REPORT:-1}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${OUTPUT_DIR}"

select_prefix() {
  if [[ -n "${BASE_PREFIX}" ]]; then
    echo "${BASE_PREFIX}"
    return
  fi
  local idx=1
  local candidate
  while true; do
    candidate="${BASE_PREFIX_ROOT}_v${idx}"
    if [[ "${ALLOW_EXISTING_RUN}" == "1" ]]; then
      echo "${candidate}"
      return
    fi
    if [[ ! -f "${OUTPUT_DIR}/${candidate}_cell_drug_dose_time_eval.csv" ]]; then
      echo "${candidate}"
      return
    fi
    idx=$((idx + 1))
  done
}

validate_inputs() {
  "${PYTHON_BIN}" - "${CELL_LLM_EMBEDDING_PATH}" "${CELL_TYPE_LLM_EMBEDDING_PATH}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cell_path = Path(sys.argv[1])
cell_type_path = Path(sys.argv[2])


def load_meta(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise SystemExit(f"missing embedding artifact: {path}")
    payload = np.load(path, allow_pickle=False)
    if "embedding_matrix" not in payload.files:
        raise SystemExit(f"{path} must contain embedding_matrix")
    matrix = payload["embedding_matrix"]
    if not path.with_suffix(".json").exists():
        raise SystemExit(f"missing sidecar: {path.with_suffix('.json')}")
    meta = {}
    if "meta_json" in payload.files:
        meta = json.loads(str(payload["meta_json"].item()))
    return matrix, meta


cell_matrix, cell_meta = load_meta(cell_path)
if cell_matrix.shape != (74, 4096):
    raise SystemExit(f"{cell_path} expected shape (74, 4096), got {cell_matrix.shape}")
if cell_meta.get("kind") != "cell_llm_embedding" or cell_meta.get("index_column") != "Cell_index":
    raise SystemExit(f"{cell_path} has invalid Cell LLM metadata: {cell_meta}")

cell_type_matrix, cell_type_meta = load_meta(cell_type_path)
if cell_type_matrix.shape != (14, 4096):
    raise SystemExit(f"{cell_type_path} expected shape (14, 4096), got {cell_type_matrix.shape}")
if cell_type_meta.get("kind") != "cell_type_llm_embedding" or cell_type_meta.get("index_column") != "cell_type_index":
    raise SystemExit(f"{cell_type_path} has invalid cell_type LLM metadata: {cell_type_meta}")

for name, matrix in (("Cell", cell_matrix), ("cell_type", cell_type_matrix)):
    if not np.allclose(matrix[0], 0.0):
        raise SystemExit(f"{name} row 0 must be zero")
    nonzero = matrix[1:]
    if not np.isfinite(nonzero).all() or np.any(np.all(np.isclose(nonzero, 0.0), axis=1)):
        raise SystemExit(f"{name} non-no rows must be finite and nonzero")

root = Path("data/training_ready/ptv3")
meta = json.load((root / "global_meta.json").open())
if len(meta["value_to_index"]["Cell"]) != 74 or int(meta["value_to_index"]["Cell"].get("no", -1)) != 0:
    raise SystemExit("global Cell mapping must contain 74 values with no=0")
if len(meta["value_to_index"]["cell_type"]) != 14 or int(meta["value_to_index"]["cell_type"].get("no", -1)) != 0:
    raise SystemExit("global cell_type mapping must contain 14 values with no=0")

for task in meta.get("task_names", []):
    task_dir = root / "tasks" / str(task)
    table_path = task_dir / "feature_table.parquet"
    if table_path.exists():
        df = pd.read_parquet(table_path, columns=["Cell_index", "cell_type_index", "pert_dose1", "pert_dose2"])
    else:
        csv_path = task_dir / "feature_table.csv"
        if not csv_path.exists():
            raise SystemExit(f"missing feature_table for {task}")
        df = pd.read_csv(csv_path, usecols=["Cell_index", "cell_type_index", "pert_dose1", "pert_dose2"], low_memory=False)
    max_cell = pd.to_numeric(df["Cell_index"], errors="coerce").fillna(0).astype(int).max()
    max_cell_type = pd.to_numeric(df["cell_type_index"], errors="coerce").fillna(0).astype(int).max()
    if int(max_cell) >= cell_matrix.shape[0]:
        raise SystemExit(f"{task}: max Cell_index {max_cell} exceeds Cell embedding rows {cell_matrix.shape[0]}")
    if int(max_cell_type) >= cell_type_matrix.shape[0]:
        raise SystemExit(
            f"{task}: max cell_type_index {max_cell_type} exceeds cell_type embedding rows {cell_type_matrix.shape[0]}"
        )
    for field in ("pert_dose1", "pert_dose2"):
        numeric = pd.to_numeric(df[field], errors="coerce").dropna()
        if not numeric.empty and (float(numeric.max()) > 10.0 or float(numeric.min()) < 0.0):
            raise SystemExit(f"{task}: {field} outside clip10 range")

print("[validate] Cell and cell_type LLM artifacts passed")
PY
}

SELECTED_PREFIX="$(select_prefix)"
echo "[settings] SELECTED_PREFIX=${SELECTED_PREFIX}"
echo "[settings] CELL_LLM_EMBEDDING_PATH=${CELL_LLM_EMBEDDING_PATH}"
echo "[settings] CELL_TYPE_LLM_EMBEDDING_PATH=${CELL_TYPE_LLM_EMBEDDING_PATH}"

"${PYTHON_BIN}" -m py_compile \
  train.py infer.py dataset/training_ready_fast_dataset.py model/fast_delta_model.py \
  scripts/report_cell_drug_time_eval.py utils/11_build_cell_llm_embeddings.py \
  utils/11_build_cell_type_llm_embeddings.py
bash -n scripts/ptv3_experiment_common.sh scripts/run_cell_llm_clip10_param_search.sh
validate_inputs

env \
  RUN_MODE=selected \
  SELECTED_PREFIX="${SELECTED_PREFIX}" \
  SELECTED_STAGE1_CONFIG=mse050_target_pdi \
  SELECTED_STAGE2_CONFIG=covdrop010_drop010 \
  SELECTED_STAGE3_CONFIG=covdrop010_lr1e4 \
  SELECTED_STAGE4_CONFIG=drop020_mseinactive010 \
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
  RUN_REPORT="${RUN_REPORT}" \
  bash scripts/run_cell_llm_clip10_param_search.sh

echo "[done] Cell + cell_type LLM selected suite completed for SELECTED_PREFIX=${SELECTED_PREFIX}"
