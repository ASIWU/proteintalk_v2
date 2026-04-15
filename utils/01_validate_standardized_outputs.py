#!/usr/bin/env python3
"""Lightweight validator for standardized ProteinTalk outputs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "standardized"


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported npy version {version} for {path}")
    return shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate standardized ProteinTalk outputs")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    audit = load_json(output_root / "file_audit.json")

    failures: list[str] = []
    summaries: list[str] = []

    for task_name, payload in audit["tasks"].items():
        info_path = Path(payload["info_path"])
        info_df = pd.read_csv(info_path, low_memory=False)
        expr = payload["expression"]
        matrix_shape = read_npy_shape(Path(expr["matrix_path"]))
        protein_order = load_json(Path(expr["protein_order_path"]))
        sample_ids = load_json(Path(expr["sample_ids_path"]))
        index_map = load_json(Path(expr["sample_id_to_row_index_path"]))

        if len(info_df) != matrix_shape[0]:
            failures.append(f"{task_name}: info rows {len(info_df)} != matrix rows {matrix_shape[0]}")
        if len(protein_order) != matrix_shape[1]:
            failures.append(f"{task_name}: protein order {len(protein_order)} != matrix cols {matrix_shape[1]}")
        if len(sample_ids) != matrix_shape[0]:
            failures.append(f"{task_name}: sample_ids {len(sample_ids)} != matrix rows {matrix_shape[0]}")
        if len(index_map) != matrix_shape[0]:
            failures.append(f"{task_name}: index map {len(index_map)} != matrix rows {matrix_shape[0]}")

        if expr.get("expression_dict_materialized"):
            dict_path = Path(expr["expression_dict_path"])
            with dict_path.open("rb") as handle:
                expr_dict = pickle.load(handle)
            if len(expr_dict) != matrix_shape[0]:
                failures.append(f"{task_name}: expression dict {len(expr_dict)} != matrix rows {matrix_shape[0]}")

        summaries.append(
            f"{task_name}\trows={matrix_shape[0]}\tproteins={matrix_shape[1]}\t"
            f"dict={'yes' if expr.get('expression_dict_materialized') else 'no'}"
        )

    for summary in summaries:
        print(summary)

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    print("\nValidation passed.")


if __name__ == "__main__":
    main()
