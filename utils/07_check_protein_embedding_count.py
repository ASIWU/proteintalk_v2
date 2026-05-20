#!/usr/bin/env python3
"""Check embedding row count against a global_meta.json index."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOBAL_META = REPO_ROOT / "data" / "training_ready" / "ptv3" / "global_meta.json"
DEFAULT_EMBEDDING_PKL = (
    REPO_ROOT / "data" / "training_ready" / "ptv3" / "derived" / "protein_embedding_esm.pkl"
)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def ordered_ids(index_mapping: dict[str, Any]) -> list[str]:
    try:
        return [item for item, _ in sorted(index_mapping.items(), key=lambda pair: int(pair[1]))]
    except Exception as exc:
        raise ValueError("global_meta.json `protein_index` values must be integer-like.") from exc


def matrix_row_count(embedding_matrix: object) -> int:
    shape = getattr(embedding_matrix, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    try:
        return len(embedding_matrix)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("embedding_matrix must be an array-like object with rows.") from exc


def matrix_shape(embedding_matrix: object) -> tuple[int, ...]:
    shape = getattr(embedding_matrix, "shape", None)
    if shape is not None:
        return tuple(int(value) for value in shape)
    return (matrix_row_count(embedding_matrix),)


def validate_payload(meta: dict[str, Any], payload: dict[str, Any], *, index_key: str) -> list[str]:
    failures: list[str] = []
    entity_index = meta.get(index_key)
    if not isinstance(entity_index, dict):
        raise ValueError(f"global_meta.json must contain object field `{index_key}`.")

    embedding_matrix = payload.get("embedding_matrix")
    if embedding_matrix is None:
        raise ValueError("embedding pickle payload must contain `embedding_matrix`.")

    meta_count = len(entity_index)
    embedding_count = matrix_row_count(embedding_matrix)
    if embedding_count != meta_count:
        failures.append(f"embedding row count {embedding_count} != {index_key} count {meta_count}")

    index_to_item = payload.get("index_to_item")
    if index_to_item is not None:
        if len(index_to_item) != meta_count:
            failures.append(f"index_to_item count {len(index_to_item)} != {index_key} count {meta_count}")
        else:
            expected_order = ordered_ids(entity_index)
            observed_order = [str(item) for item in index_to_item]
            if observed_order != expected_order:
                failures.append(f"index_to_item order does not match global_meta {index_key} order")

    item_to_index = payload.get("item_to_index")
    if item_to_index is not None and len(item_to_index) != meta_count:
        failures.append(f"item_to_index count {len(item_to_index)} != {index_key} count {meta_count}")

    payload_embedding_dim = payload.get("embedding_dim")
    shape = matrix_shape(embedding_matrix)
    if payload_embedding_dim is not None and len(shape) >= 2 and int(payload_embedding_dim) != shape[1]:
        failures.append(f"payload embedding_dim {payload_embedding_dim} != embedding_matrix column count {shape[1]}")

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether protein_embedding_esm.pkl rows match global_meta.json protein_index count."
    )
    parser.add_argument(
        "--index-key",
        default="protein_index",
        help="Index field in global_meta.json to compare against, for example protein_index or pert_index.",
    )
    parser.add_argument(
        "--global-meta",
        type=Path,
        default=DEFAULT_GLOBAL_META,
        help="Path to data/training_ready/<dataset>/global_meta.json.",
    )
    parser.add_argument(
        "--embedding-pkl",
        type=Path,
        default=DEFAULT_EMBEDDING_PKL,
        help="Path to protein embedding pickle payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_json(args.global_meta)
    payload = load_pickle(args.embedding_pkl)
    if not isinstance(meta, dict):
        raise ValueError("global_meta.json root must be a JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("embedding pickle root must be a dict payload.")

    failures = validate_payload(meta, payload, index_key=args.index_key)
    entity_count = len(meta[args.index_key])
    embedding_shape = matrix_shape(payload["embedding_matrix"])
    embedding_count = embedding_shape[0]
    print(f"global_meta {args.index_key} count: {entity_count}")
    print(f"embedding_matrix shape:         {embedding_shape}")
    if len(embedding_shape) >= 2:
        print(f"embedding feature dimension:    {embedding_shape[1]}")
    if "embedding_dim" in payload:
        print(f"payload embedding_dim:          {payload['embedding_dim']}")
    if "max_length" in payload:
        print(f"payload max_length:             {payload['max_length']} (sequence tokenizer limit)")

    if failures:
        print("FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PASSED: embedding count matches metadata entity count.")


if __name__ == "__main__":
    main()
