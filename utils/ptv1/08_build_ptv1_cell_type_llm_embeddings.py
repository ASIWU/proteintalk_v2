#!/usr/bin/env python3
"""Build and validate PTV1 cell_type LLM embeddings aligned by cell_type_index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _shared import REPO_ROOT, load_json, load_repo_module


cell_type_llm = load_repo_module("utils/11_build_cell_type_llm_embeddings.py", "ptv_cell_type_llm_shared")

DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
DEFAULT_DATASET_GROUP = "ptv1"
DEFAULT_OUTPUT = (
    DEFAULT_TRAINING_READY_ROOT
    / DEFAULT_DATASET_GROUP
    / "derived"
    / "cell_type_llm_embedding_qwen3_4096_v2.npz"
)
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def parse_meta_json(payload: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta_json" not in payload.files:
        raise ValueError("cell_type LLM artifact must contain meta_json")
    raw = payload["meta_json"].item()
    meta = json.loads(str(raw))
    if not isinstance(meta, dict):
        raise ValueError("cell_type LLM meta_json must decode to an object")
    return meta


def expected_cell_type_names(global_meta: dict[str, Any]) -> list[str]:
    mapping = global_meta.get("value_to_index", {}).get("cell_type")
    if not isinstance(mapping, dict):
        raise ValueError("global_meta value_to_index.cell_type is missing")
    if int(mapping.get("no", -1)) != 0:
        raise ValueError('global_meta value_to_index["cell_type"] must reserve index 0 for "no"')
    by_index: dict[int, str] = {}
    for name, index in mapping.items():
        by_index[int(index)] = str(name)
    if sorted(by_index) != list(range(max(by_index) + 1)):
        raise ValueError("global_meta cell_type indices must be contiguous from 0")
    return [by_index[index] for index in range(max(by_index) + 1)]


def read_feature_table(task_dir: Path) -> pd.DataFrame:
    return cell_type_llm.read_feature_table(task_dir)


def validate_feature_tables(
    *,
    training_ready_root: Path,
    dataset_group: str,
    global_meta: dict[str, Any],
    embedding_matrix: np.ndarray,
) -> dict[str, Any]:
    task_root = training_ready_root / dataset_group / "tasks"
    task_names = list(global_meta.get("task_names") or [])
    if not task_names:
        raise ValueError("global_meta task_names is empty")

    summaries: dict[str, Any] = {}
    observed_indices: set[int] = set()
    for task_name in task_names:
        task_dir = task_root / str(task_name)
        if not task_dir.exists():
            raise FileNotFoundError(f"missing task directory: {task_dir}")
        df = read_feature_table(task_dir)
        for column in ("cell_type", "cell_type_index"):
            if column not in df.columns:
                raise KeyError(f"{task_dir}/feature_table is missing {column}")
        indices = pd.to_numeric(df["cell_type_index"], errors="raise").astype(int).to_numpy()
        if indices.size and (indices.min() < 0 or indices.max() >= embedding_matrix.shape[0]):
            raise ValueError(
                f"{task_name} cell_type_index range [{indices.min()}, {indices.max()}] "
                f"exceeds embedding rows={embedding_matrix.shape[0]}"
            )
        unique_indices = sorted(int(value) for value in np.unique(indices))
        observed_indices.update(unique_indices)
        summaries[str(task_name)] = {
            "rows": int(len(df)),
            "unique_cell_type_indices": unique_indices,
            "unique_cell_type_count": int(len(unique_indices)),
            "unique_cell_type_values": sorted(str(value) for value in df["cell_type"].dropna().unique().tolist()),
        }

    nonzero_observed = sorted(index for index in observed_indices if index > 0)
    if nonzero_observed:
        row_norms = np.linalg.norm(embedding_matrix[nonzero_observed], axis=1)
        if not np.all(row_norms > 0):
            bad = [index for index, norm in zip(nonzero_observed, row_norms) if norm <= 0]
            raise ValueError(f"observed nonzero cell_type_index rows have zero embeddings: {bad}")
    return summaries


def validate_ptv1_cell_type_llm_artifact(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output)
    sidecar_path = output_path.with_suffix(".json")
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(sidecar_path)

    payload = np.load(output_path, allow_pickle=False)
    if "embedding_matrix" not in payload.files:
        raise ValueError(f"{output_path} must contain embedding_matrix")
    embedding_matrix = payload["embedding_matrix"].astype(np.float32, copy=False)
    if embedding_matrix.ndim != 2:
        raise ValueError(f"embedding_matrix must be 2D, got shape={embedding_matrix.shape}")

    meta = parse_meta_json(payload)
    expected_meta = {
        "dataset_group": "ptv1",
        "kind": "cell_type_llm_embedding",
        "input_field": "cell_type",
        "index_column": "cell_type_index",
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise ValueError(f"meta {key} must be {expected!r}, got {meta.get(key)!r}")

    global_meta_path = Path(args.training_ready_root) / "ptv1" / "global_meta.json"
    global_meta = load_json(global_meta_path)
    if not isinstance(global_meta, dict) or global_meta.get("dataset_group") != "ptv1":
        raise ValueError(f"expected PTV1 global_meta at {global_meta_path}")
    names = expected_cell_type_names(global_meta)
    expected_shape = (len(names), int(args.expected_dim))
    if args.expected_rows is not None:
        expected_shape = (int(args.expected_rows), int(args.expected_dim))
    if tuple(embedding_matrix.shape) != expected_shape:
        raise ValueError(f"expected embedding shape {expected_shape}, got {tuple(embedding_matrix.shape)}")
    if len(names) != embedding_matrix.shape[0]:
        raise ValueError(f"global_meta has {len(names)} cell_type indices, artifact has {embedding_matrix.shape[0]} rows")

    if "cell_type_indices" in payload.files:
        indices = payload["cell_type_indices"].astype(int)
        expected_indices = np.arange(embedding_matrix.shape[0], dtype=int)
        if not np.array_equal(indices, expected_indices):
            raise ValueError("cell_type_indices must be contiguous and aligned to row order")
    if "cell_type_names" in payload.files:
        artifact_names = [str(value) for value in payload["cell_type_names"].tolist()]
        if artifact_names != names:
            raise ValueError("cell_type_names are not aligned with global_meta cell_type indices")

    if not np.allclose(embedding_matrix[0], 0.0):
        raise ValueError("row 0 must be the reserved zero vector")
    nonzero_rows = embedding_matrix[1:]
    if not np.isfinite(nonzero_rows).all():
        raise ValueError("rows 1..N must contain only finite values")
    zero_norm_rows = np.where(np.linalg.norm(nonzero_rows, axis=1) <= 0)[0] + 1
    if len(zero_norm_rows):
        raise ValueError(f"rows 1..N must be nonzero; zero rows={zero_norm_rows.tolist()}")

    task_summaries = validate_feature_tables(
        training_ready_root=Path(args.training_ready_root),
        dataset_group=args.dataset_group,
        global_meta=global_meta,
        embedding_matrix=embedding_matrix,
    )
    summary = {
        "artifact": str(output_path),
        "sidecar": str(sidecar_path),
        "shape": list(embedding_matrix.shape),
        "dataset_group": meta.get("dataset_group"),
        "kind": meta.get("kind"),
        "input_field": meta.get("input_field"),
        "index_column": meta.get("index_column"),
        "task_summaries": task_summaries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", default=DEFAULT_DATASET_GROUP)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--description-model", default=cell_type_llm.DEFAULT_DESCRIPTION_MODEL)
    parser.add_argument("--embedding-model", default=cell_type_llm.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-normalize-embeddings", action="store_false", dest="normalize_embeddings")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=2)
    parser.add_argument("--expected-dim", type=int, default=4096)
    parser.set_defaults(normalize_embeddings=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_group != "ptv1":
        raise ValueError("--dataset-group must remain ptv1 for this wrapper")
    if not args.validate_only:
        cell_type_llm.build_embeddings(args)
    if not args.skip_validation:
        validate_ptv1_cell_type_llm_artifact(args)


if __name__ == "__main__":
    main()
