#!/usr/bin/env python3
"""Bootstrap PTV1 LLM embedding artifacts from existing local LLM artifacts.

This utility is intentionally network-free. It is used when fresh external API
embedding generation is not approved but local LLM-derived artifacts already
exist and can be re-indexed for PTV1.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from _shared import REPO_ROOT, load_json


DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
DEFAULT_PTV1_CELL_SOURCE = DEFAULT_TRAINING_READY_ROOT / "ptv1" / "derived" / "cell_llm_embedding_qwen3_4096.npz"
DEFAULT_PTV1_CELL_OUTPUT = DEFAULT_TRAINING_READY_ROOT / "ptv1" / "derived" / "cell_llm_embedding_qwen3_4096_v2.npz"
DEFAULT_PTV3_CELL_TYPE_SOURCE = DEFAULT_TRAINING_READY_ROOT / "ptv3" / "derived" / "cell_type_llm_embedding_qwen3_4096_v2.npz"
DEFAULT_PTV1_CELL_TYPE_OUTPUT = DEFAULT_TRAINING_READY_ROOT / "ptv1" / "derived" / "cell_type_llm_embedding_qwen3_4096_v2.npz"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_npz_json(payload: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in payload.files:
        return None
    return json.loads(str(payload[key].item()))


def ordered_mapping_names(meta: dict[str, Any], key: str) -> list[str]:
    mapping = meta.get("value_to_index", {}).get(key)
    if not isinstance(mapping, dict):
        raise ValueError(f"missing global_meta value_to_index.{key}")
    by_index = {int(index): str(name) for name, index in mapping.items()}
    if int(mapping.get("no", -1)) != 0:
        raise ValueError(f'global_meta value_to_index["{key}"] must reserve no=0')
    if sorted(by_index) != list(range(max(by_index) + 1)):
        raise ValueError(f"global_meta {key} indices must be contiguous from 0")
    return [by_index[index] for index in range(max(by_index) + 1)]


def load_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    payload = load_json(sidecar)
    return payload if isinstance(payload, dict) else {}


def bootstrap_cell(source: Path, output: Path, ptv1_meta: dict[str, Any], *, force: bool) -> dict[str, Any]:
    if output.exists() and output.with_suffix(".json").exists() and not force:
        return {"artifact": str(output), "kept_existing": True}
    z = np.load(source, allow_pickle=False)
    matrix = z["embedding_matrix"].astype(np.float32, copy=False)
    expected_names = ordered_mapping_names(ptv1_meta, "Cell")
    source_names = [str(value) for value in z["cell_names"].tolist()]
    if source_names != expected_names:
        raise ValueError(f"Cell names in {source} are not aligned with current PTV1 global_meta")
    source_meta = parse_npz_json(z, "meta_json") or {}
    meta = dict(source_meta)
    meta.update(
        {
            "kind": "cell_llm_embedding",
            "dataset_group": "ptv1",
            "input_field": "Cell",
            "index_column": "Cell_index",
            "bootstrap_method": "copied_from_existing_ptv1_cell_llm_artifact",
            "bootstrap_source_artifact": str(source.resolve()),
            "bootstrap_recorded_at": iso_now(),
        }
    )
    descriptions_json = z["descriptions_json"] if "descriptions_json" in z.files else np.asarray(json.dumps({}))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embedding_matrix=matrix,
        cell_names=np.asarray(expected_names),
        cell_indices=np.arange(len(expected_names), dtype=np.int64),
        descriptions_json=descriptions_json,
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    sidecar = load_sidecar(source)
    sidecar.setdefault("meta", {})
    sidecar["meta"].update(meta)
    dump_json(output.with_suffix(".json"), sidecar)
    return {"artifact": str(output), "source": str(source), "shape": list(matrix.shape)}


def bootstrap_cell_type(source: Path, output: Path, ptv1_meta: dict[str, Any], *, force: bool) -> dict[str, Any]:
    if output.exists() and output.with_suffix(".json").exists() and not force:
        return {"artifact": str(output), "kept_existing": True}
    names = ordered_mapping_names(ptv1_meta, "cell_type")
    if names != ["no", "BREAST"]:
        raise ValueError(f"PTV1 cell_type bootstrap expects ['no', 'BREAST'], got {names}")
    z = np.load(source, allow_pickle=False)
    source_names = [str(value) for value in z["cell_type_names"].tolist()]
    if "BREAST" not in source_names:
        raise ValueError(f"source cell_type artifact lacks BREAST row: {source}")
    breast_index = source_names.index("BREAST")
    source_matrix = z["embedding_matrix"].astype(np.float32, copy=False)
    matrix = np.zeros((2, source_matrix.shape[1]), dtype=np.float32)
    matrix[1] = source_matrix[breast_index]
    source_meta = parse_npz_json(z, "meta_json") or {}
    meta = dict(source_meta)
    meta.update(
        {
            "kind": "cell_type_llm_embedding",
            "dataset_group": "ptv1",
            "cell_type_count": 2,
            "nonzero_cell_type_count": 1,
            "input_field": "cell_type",
            "index_column": "cell_type_index",
            "bootstrap_method": "reindexed_breast_row_from_existing_ptv3_cell_type_llm_artifact",
            "bootstrap_source_artifact": str(source.resolve()),
            "bootstrap_source_cell_type_index": int(breast_index),
            "bootstrap_recorded_at": iso_now(),
        }
    )

    source_sidecar = load_sidecar(source)
    source_descriptions = source_sidecar.get("cell_types") if isinstance(source_sidecar.get("cell_types"), dict) else {}
    breast_description = dict(source_descriptions.get(str(breast_index), {})) if isinstance(source_descriptions, dict) else {}
    breast_description.update({"index": 1, "canonical_name": "BREAST", "raw_values": {"BREAST": 15224}, "row_count": 15224})
    descriptions = {
        "0": {"index": 0, "canonical_name": "no", "raw_values": {}, "row_count": 0, "description": "Missing or unspecified cell type. Reserved zero-vector row.", "prompt": None},
        "1": breast_description,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embedding_matrix=matrix,
        cell_type_names=np.asarray(names),
        cell_type_indices=np.arange(len(names), dtype=np.int64),
        descriptions_json=np.asarray(json.dumps(descriptions, ensure_ascii=False)),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    dump_json(output.with_suffix(".json"), {"meta": meta, "cell_types": descriptions})
    return {"artifact": str(output), "source": str(source), "shape": list(matrix.shape), "source_breast_index": breast_index}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ready-root", type=Path, default=DEFAULT_TRAINING_READY_ROOT)
    parser.add_argument("--ptv1-cell-source", type=Path, default=DEFAULT_PTV1_CELL_SOURCE)
    parser.add_argument("--ptv1-cell-output", type=Path, default=DEFAULT_PTV1_CELL_OUTPUT)
    parser.add_argument("--ptv3-cell-type-source", type=Path, default=DEFAULT_PTV3_CELL_TYPE_SOURCE)
    parser.add_argument("--ptv1-cell-type-output", type=Path, default=DEFAULT_PTV1_CELL_TYPE_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.training_ready_root / "ptv1" / "global_meta.json"
    ptv1_meta = load_json(meta_path)
    if not isinstance(ptv1_meta, dict) or ptv1_meta.get("dataset_group") != "ptv1":
        raise ValueError(f"expected PTV1 global_meta at {meta_path}")
    summary = {
        "cell": bootstrap_cell(args.ptv1_cell_source, args.ptv1_cell_output, ptv1_meta, force=args.force),
        "cell_type": bootstrap_cell_type(args.ptv3_cell_type_source, args.ptv1_cell_type_output, ptv1_meta, force=args.force),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
