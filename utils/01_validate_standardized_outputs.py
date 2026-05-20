#!/usr/bin/env python3
"""Lightweight validator for standardized ProteinTalk outputs."""

from __future__ import annotations

import argparse
import json
import pickle
import re
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


def parse_uniprot_token(token: str) -> bool:
    token = token.strip()
    return bool(
        re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", token)
        or re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{3}[0-9]", token)
        or re.fullmatch(r"[A-NR-Z][0-9](?:[A-Z0-9]{3}[0-9]){2}", token)
        or re.fullmatch(r"A0A[A-Z0-9]{7}", token)
    )


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
        if info_df["sample_id"].duplicated(keep=False).any():
            duplicated = info_df.loc[info_df["sample_id"].duplicated(keep=False), "sample_id"].astype(str).unique().tolist()
            failures.append(f"{task_name}: duplicated sample_id values in info.csv: {duplicated[:10]}")
        if len(protein_order) != matrix_shape[1]:
            failures.append(f"{task_name}: protein order {len(protein_order)} != matrix cols {matrix_shape[1]}")
        if len(protein_order) != len(set(protein_order)):
            duplicated = [protein for protein in protein_order if protein_order.count(protein) > 1]
            failures.append(f"{task_name}: duplicated UniProt IDs in protein order: {sorted(set(duplicated))[:10]}")
        invalid_proteins = [protein for protein in protein_order if not parse_uniprot_token(protein)]
        if invalid_proteins:
            failures.append(f"{task_name}: invalid UniProt IDs in protein order: {invalid_proteins[:10]}")
        if len(sample_ids) != matrix_shape[0]:
            failures.append(f"{task_name}: sample_ids {len(sample_ids)} != matrix rows {matrix_shape[0]}")
        if len(sample_ids) != len(set(sample_ids)):
            duplicates = [sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1]
            failures.append(f"{task_name}: duplicated sample_ids in sample_ids.json: {sorted(set(duplicates))[:10]}")
        if len(index_map) != matrix_shape[0]:
            failures.append(f"{task_name}: index map {len(index_map)} != matrix rows {matrix_shape[0]}")
        if "doubledrug" in task_name:
            if "synergy" not in info_df.columns:
                failures.append(f"{task_name}: missing synergy column in info.csv")
            else:
                synergy = info_df["synergy"]
                synergy_non_empty = synergy.notna()
                if synergy.dtype == object:
                    synergy_non_empty = synergy.fillna("").astype(str).str.strip() != ""
                if not synergy_non_empty.any():
                    failures.append(f"{task_name}: synergy column is present but all values are empty")

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
