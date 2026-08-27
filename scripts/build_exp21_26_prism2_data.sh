#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
if ! conda activate flow_v2; then
  conda activate /mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python}"
TRAINING_READY_ROOT="${TRAINING_READY_ROOT:-data/training_ready_prism2_main}"
BASE_TRAINING_READY_ROOT="${BASE_TRAINING_READY_ROOT:-data/training_ready}"
GPU_TMUX_SESSION="${GPU_TMUX_SESSION:-gpu2}"
AUTO_RUN_GRAPH_REBUILD="${AUTO_RUN_GRAPH_REBUILD:-1}"

"${PYTHON_BIN}" -m py_compile \
  utils/npy_io.py \
  utils/02_build_training_ready_data.py \
  utils/03_validate_training_ready_outputs.py \
  utils/09_build_data_splits.py \
  train.py \
  infer.py \
  dataset/training_ready_fast_dataset.py \
  model/graph_feature_utils.py

"${PYTHON_BIN}" utils/02_build_training_ready_data.py \
  --dataset-group ptv3 \
  --output-root "${TRAINING_READY_ROOT}" \
  --include-prism2-main-tasks

set +e
"${PYTHON_BIN}" - "${BASE_TRAINING_READY_ROOT}" "${TRAINING_READY_ROOT}" <<'PY'
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from numpy.lib import format as np_format

base_root = Path(sys.argv[1])
new_root = Path(sys.argv[2])
base_meta_path = base_root / "ptv3" / "global_meta.json"
new_meta_path = new_root / "ptv3" / "global_meta.json"
base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
new_meta = json.loads(new_meta_path.read_text(encoding="utf-8"))


def mapping_hash(mapping: dict[str, int]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


same_pert = mapping_hash(base_meta["pert_index"]) == mapping_hash(new_meta["pert_index"])
same_protein = mapping_hash(base_meta["protein_index"]) == mapping_hash(new_meta["protein_index"])
summary = {
    "base_global_meta": str(base_meta_path),
    "new_global_meta": str(new_meta_path),
    "same_pert_index": same_pert,
    "same_protein_index": same_protein,
    "base_pert_index_size": len(base_meta["pert_index"]),
    "new_pert_index_size": len(new_meta["pert_index"]),
    "base_protein_index_size": len(base_meta["protein_index"]),
    "new_protein_index_size": len(new_meta["protein_index"]),
}

log_path = new_root / "ptv3" / "derived_graph_reuse_check.json"
log_path.parent.mkdir(parents=True, exist_ok=True)

if not (same_pert and same_protein):
    summary["status"] = "rebuild_required"
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(20)

base_derived = base_root / "ptv3" / "derived"
new_derived = new_root / "ptv3" / "derived"
if not new_derived.exists():
    rel_target = os.path.relpath(base_derived.resolve(), new_derived.parent.resolve())
    os.symlink(rel_target, new_derived, target_is_directory=True)
elif new_derived.is_symlink():
    if new_derived.resolve() != base_derived.resolve():
        raise SystemExit(f"{new_derived} already points to {new_derived.resolve()}, expected {base_derived.resolve()}")
elif not new_derived.is_dir():
    raise SystemExit(f"{new_derived} exists and is not a directory/symlink")


def npy_shape(path: Path) -> tuple[tuple[int, ...], str]:
    with path.open("rb") as handle:
        version = np_format.read_magic(handle)
        shape, _fortran, dtype = np_format._read_array_header(handle, version)
    return tuple(int(value) for value in shape), str(dtype)


def embedding_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    matrix = payload["embedding_matrix"] if isinstance(payload, dict) else payload
    return tuple(int(value) for value in np.asarray(matrix).shape)

pert_count = len(new_meta["pert_index"])
protein_count = len(new_meta["protein_index"])
checks = {
    "ddi_matrix": npy_shape(new_derived / "ddi_matrix.npy"),
    "pdi_matrix": npy_shape(new_derived / "pdi_matrix.npy"),
    "ppi_matrix": npy_shape(new_derived / "ppi_matrix.npy"),
    "drug_embedding": embedding_shape(new_derived / "drug_embedding_morgan_2048.pkl"),
    "protein_embedding": embedding_shape(new_derived / "protein_embedding_esm.pkl"),
}
expected = {
    "ddi_matrix": ((pert_count, pert_count), "float32"),
    "pdi_matrix": ((pert_count, protein_count), "float32"),
    "ppi_matrix": ((protein_count, protein_count), "float32"),
}
for name, (shape, dtype) in checks.items():
    if name in expected:
        expected_shape, expected_dtype = expected[name]
        if shape != expected_shape or dtype != expected_dtype:
            raise SystemExit(f"{name} mismatch: got shape={shape} dtype={dtype}, expected shape={expected_shape} dtype={expected_dtype}")
if checks["drug_embedding"][0] != pert_count:
    raise SystemExit(f"drug embedding rows {checks['drug_embedding'][0]} != pert_index size {pert_count}")
if checks["protein_embedding"][0] != protein_count:
    raise SystemExit(f"protein embedding rows {checks['protein_embedding'][0]} != protein_index size {protein_count}")

summary["status"] = "reused_existing_derived"
summary["derived_path"] = str(new_derived)
summary["checks"] = {key: list(value) if isinstance(value, tuple) else value for key, value in checks.items()}
log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
graph_status="$?"
set -e

if [[ "${graph_status}" == "20" ]]; then
  echo "[graph] PRISM2 global_meta index changed; graph rebuild is required."
  if [[ "${AUTO_RUN_GRAPH_REBUILD}" == "1" ]]; then
    tmux send-keys -t "${GPU_TMUX_SESSION}" \
      "cd ${REPO_ROOT} && source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh && conda activate flow_v2 && TRAINING_READY_ROOT=${TRAINING_READY_ROOT} bash scripts/exp21_26_rebuild_prism2_graphs.sh" C-m
    echo "[graph] dispatched rebuild to tmux session ${GPU_TMUX_SESSION}"
  else
    echo "[graph] run manually: TRAINING_READY_ROOT=${TRAINING_READY_ROOT} bash scripts/exp21_26_rebuild_prism2_graphs.sh"
  fi
  exit 20
elif [[ "${graph_status}" != "0" ]]; then
  exit "${graph_status}"
fi

"${PYTHON_BIN}" utils/09_build_data_splits.py \
  --training-ready-root "${TRAINING_READY_ROOT}" \
  --dataset-group ptv3

"${PYTHON_BIN}" utils/03_validate_training_ready_outputs.py \
  --output-root "${TRAINING_READY_ROOT}"

"${PYTHON_BIN}" - "${TRAINING_READY_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
task_dir = root / "ptv3" / "tasks" / "ptv3_main_singledrug_prism2"
df = pd.read_csv(task_dir / "feature_table.csv", low_memory=False)
control = df["is_control"].astype(bool)
non = df.loc[~control]
p1 = non["PRISM1st_label_total"].astype("string").fillna("").str.strip()
p2 = non["PRISM2nd_label_total"].astype("string").fillna("").str.strip()
summary = {
    "task": "ptv3_main_singledrug_prism2",
    "feature_rows": int(len(df)),
    "control_rows": int(control.sum()),
    "non_control_rows": int(len(non)),
    "prism2_nonempty": int(p2.ne("").sum()),
    "prism2_only_rows": int((p1.eq("") & p2.ne("")).sum()),
    "prism1_only_rows": int((p1.ne("") & p2.eq("")).sum()),
    "prism2_counts": {str(k): int(v) for k, v in p2.value_counts(dropna=False).items()},
}
path = root / "ptv3" / "exp21_26_prism2_data_summary.json"
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[done] exp_21-26 PRISM2 data is ready under ${TRAINING_READY_ROOT}"
