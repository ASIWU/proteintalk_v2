#!/usr/bin/env python3
"""Build zero-input artifacts for the fold0 feature attribution experiments."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_DIR = REPO_ROOT / "data/training_ready/ptv3/tasks/ptv3_main_singledrug"
DEFAULT_DERIVED_DIR = REPO_ROOT / "data/training_ready/ptv3/derived"
ZERO_CHUNK_VALUES = 8_388_608


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype, bool]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
    return tuple(int(dim) for dim in shape), np.dtype(dtype), bool(fortran_order)


def write_zero_npy(path: Path, shape: tuple[int, int], dtype: np.dtype = np.dtype("float32")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    total_values = int(shape[0]) * int(shape[1])
    chunk = np.zeros(min(ZERO_CHUNK_VALUES, total_values), dtype=dtype)
    with tmp_path.open("wb") as handle:
        np.lib.format.write_array_header_2_0(
            handle,
            {
                "descr": dtype.str,
                "fortran_order": False,
                "shape": shape,
            },
        )
        remaining = total_values
        while remaining > 0:
            take = min(int(chunk.size), remaining)
            handle.write(chunk[:take].tobytes(order="C"))
            remaining -= take
    os.replace(tmp_path, path)


def validate_zero_npy(path: Path, expected_shape: tuple[int, int]) -> dict[str, Any]:
    shape, dtype, fortran_order = npy_header(path)
    if shape != expected_shape:
        raise ValueError(f"{path} shape {shape} != expected {expected_shape}")
    if dtype != np.dtype("float32"):
        raise ValueError(f"{path} dtype {dtype} != float32")
    if fortran_order:
        raise ValueError(f"{path} must be C-order")
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        np.lib.format._read_array_header(handle, version)
        while True:
            raw = handle.read(ZERO_CHUNK_VALUES * dtype.itemsize)
            if not raw:
                break
            values = np.frombuffer(raw, dtype=dtype)
            if values.size and np.any(values != 0.0):
                raise ValueError(f"{path} is not all-zero")
    return {
        "path": str(path),
        "shape": [shape[0], shape[1]],
        "dtype": str(dtype),
        "all_zero": True,
    }


def build_zero_control(expression_path: Path, output_path: Path, *, overwrite: bool) -> dict[str, Any]:
    expression_shape, expression_dtype, _ = npy_header(expression_path)
    if len(expression_shape) != 2:
        raise ValueError(f"{expression_path} must be 2D; got {expression_shape}")
    expected_shape = (int(expression_shape[0]), int(expression_shape[1]))
    kept_existing = False
    if output_path.exists() and not overwrite:
        kept_existing = True
    else:
        write_zero_npy(output_path, expected_shape)
        meta = {
            "kind": "zero_control_expression",
            "generated_at": iso_now(),
            "source_expression_path": str(expression_path.resolve()),
            "output_path": str(output_path.resolve()),
            "shape": [expected_shape[0], expected_shape[1]],
            "dtype": "float32",
            "source_dtype": str(expression_dtype),
            "policy": "all feature-expression rows replaced by zeros for control-expression ablation",
            "seed": None,
            "control_row_count": 0,
            "fallback_protein_count": 0,
            "mean_fallback_protein_count": 0.0,
            "std_fallback_protein_count": 0.0,
            "clipped_negative_count": 0,
            "per_protein_stats_policy": "not_applicable_zero_matrix",
        }
        dump_json(output_path.with_suffix(".meta.json"), meta)
    summary = validate_zero_npy(output_path, expected_shape)
    summary["kept_existing"] = kept_existing
    summary["meta_path"] = str(output_path.with_suffix(".meta.json"))
    return summary


def embedding_matrix_from_payload(path: Path, payload: object) -> np.ndarray:
    if isinstance(payload, dict) and "embedding_matrix" in payload:
        return np.asarray(payload["embedding_matrix"], dtype=np.float32)
    if isinstance(payload, np.ndarray):
        return np.asarray(payload, dtype=np.float32)
    raise ValueError(f"{path} must be a dict with embedding_matrix or an ndarray")


def validate_zero_embedding(path: Path, expected_shape: tuple[int, int]) -> dict[str, Any]:
    payload = load_pickle(path)
    matrix = embedding_matrix_from_payload(path, payload)
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"{path} embedding shape {matrix.shape} != expected {expected_shape}")
    if matrix.dtype != np.float32:
        raise ValueError(f"{path} embedding dtype {matrix.dtype} != float32")
    if matrix.size and not np.all(matrix == 0.0):
        raise ValueError(f"{path} embedding_matrix is not all-zero")
    return {
        "path": str(path),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "dtype": str(matrix.dtype),
        "all_zero": True,
    }


def build_zero_drug_embedding(source_path: Path, output_path: Path, *, overwrite: bool) -> dict[str, Any]:
    source_payload = load_pickle(source_path)
    source_matrix = embedding_matrix_from_payload(source_path, source_payload)
    if source_matrix.ndim != 2:
        raise ValueError(f"{source_path} embedding_matrix must be 2D; got {source_matrix.shape}")
    expected_shape = (int(source_matrix.shape[0]), int(source_matrix.shape[1]))
    if expected_shape[1] != 2048:
        raise ValueError(f"{source_path} expected Morgan width 2048, got {expected_shape[1]}")

    kept_existing = False
    if output_path.exists() and not overwrite:
        kept_existing = True
    else:
        zero_matrix = np.zeros(expected_shape, dtype=np.float32)
        if isinstance(source_payload, dict):
            output_payload = dict(source_payload)
            output_payload["embedding_matrix"] = zero_matrix
        else:
            output_payload = zero_matrix
        dump_pickle_atomic(output_path, output_payload)
        dump_json(
            output_path.with_suffix(".meta.json"),
            {
                "kind": "zero_drug_embedding",
                "generated_at": iso_now(),
                "source_embedding_path": str(source_path.resolve()),
                "output_path": str(output_path.resolve()),
                "shape": [expected_shape[0], expected_shape[1]],
                "dtype": "float32",
                "policy": "preserve source pickle payload fields and replace embedding_matrix with zeros",
            },
        )
    summary = validate_zero_embedding(output_path, expected_shape)
    summary["kept_existing"] = kept_existing
    summary["source_shape"] = [expected_shape[0], expected_shape[1]]
    summary["meta_path"] = str(output_path.with_suffix(".meta.json"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-path", type=Path, default=DEFAULT_TASK_DIR / "feature_expression_matrix.npy")
    parser.add_argument("--zero-control-output", type=Path, default=DEFAULT_TASK_DIR / "zero_control_expression.npy")
    parser.add_argument("--drug-embedding-path", type=Path, default=DEFAULT_DERIVED_DIR / "drug_embedding_morgan_2048.pkl")
    parser.add_argument("--zero-drug-output", type=Path, default=DEFAULT_DERIVED_DIR / "drug_embedding_morgan_2048_zero.pkl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.expression_path.exists():
        raise FileNotFoundError(args.expression_path)
    if not args.drug_embedding_path.exists():
        raise FileNotFoundError(args.drug_embedding_path)

    summary = {
        "generated_at": iso_now(),
        "zero_control": build_zero_control(args.expression_path, args.zero_control_output, overwrite=args.overwrite),
        "zero_drug_embedding": build_zero_drug_embedding(
            args.drug_embedding_path,
            args.zero_drug_output,
            overwrite=args.overwrite,
        ),
    }
    if args.summary_json is not None:
        dump_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
