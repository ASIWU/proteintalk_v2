#!/usr/bin/env python3
"""Validate isolated PTV1 standardized artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _shared import REPO_ROOT, load_json


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "standardized"
EXPECTED_TASKS = ("ptv1_aivc", "ptv1_extra_singledrug")


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


def parse_json_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else []


def nonempty(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().ne("")


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def validate_main_controls(info: pd.DataFrame, failures: list[str]) -> dict[str, int]:
    controls = info.loc[info["control"].astype("string").fillna("").str.strip().eq(info["sample_id"].astype(str))]
    lookup = controls.set_index("sample_id")[["BioRep", "Cell_plate"]].to_dict("index")
    pert_time = pd.to_numeric(info["pert_time"], errors="coerce")
    self_control_bad = pert_time.eq(0) & ~info["control"].astype(str).eq(info["sample_id"].astype(str))
    if self_control_bad.any():
        failures.append(f"ptv1_aivc: pert_time==0 rows must be self-controls; bad={int(self_control_bad.sum())}")

    bad_group = 0
    missing = 0
    for row in info.loc[~pert_time.eq(0)].itertuples(index=False):
        control = clean_text(getattr(row, "control", ""))
        if not control:
            missing += 1
            continue
        control_meta = lookup.get(control)
        if not control_meta:
            bad_group += 1
            continue
        if str(control_meta["BioRep"]) != str(getattr(row, "BioRep")) or str(control_meta["Cell_plate"]) != str(getattr(row, "Cell_plate")):
            bad_group += 1
    if bad_group:
        failures.append(f"ptv1_aivc: matched controls must share (BioRep, Cell_plate); bad={bad_group}")
    return {"missing_control_count": missing, "self_control_count": int(len(controls))}


def validate_task(task_name: str, payload: dict[str, Any], failures: list[str]) -> str:
    info_path = Path(payload["info_path"])
    info = pd.read_csv(info_path, low_memory=False)
    expr = payload["expression"]
    matrix_shape = read_npy_shape(Path(expr["matrix_path"]))
    protein_order = load_json(Path(expr["protein_order_path"]))
    sample_ids = load_json(Path(expr["sample_ids_path"]))
    index_map = load_json(Path(expr["sample_id_to_row_index_path"]))

    if len(info) != matrix_shape[0]:
        failures.append(f"{task_name}: info rows {len(info)} != matrix rows {matrix_shape[0]}")
    if len(protein_order) != matrix_shape[1]:
        failures.append(f"{task_name}: protein_order length {len(protein_order)} != matrix cols {matrix_shape[1]}")
    if len(sample_ids) != matrix_shape[0] or len(index_map) != matrix_shape[0]:
        failures.append(f"{task_name}: sample id artifacts are not aligned to matrix rows")
    if info["sample_id"].duplicated(keep=False).any():
        failures.append(f"{task_name}: duplicated sample_id values")
    invalid = [item for item in protein_order if not parse_uniprot_token(str(item))]
    if invalid:
        failures.append(f"{task_name}: invalid protein_order UniProt IDs: {invalid[:5]}")
    if expr.get("expression_dict_materialized"):
        with Path(expr["expression_dict_path"]).open("rb") as handle:
            expr_dict = pickle.load(handle)
        if len(expr_dict) != matrix_shape[0]:
            failures.append(f"{task_name}: expression_dict row count mismatch")

    if task_name == "ptv1_aivc":
        control_summary = validate_main_controls(info, failures)
        labels = set(info["PRISM1st_label_total"].astype("string").fillna("").str.strip().unique())
        if not {"Y", "N"}.issubset(labels):
            failures.append(f"ptv1_aivc: PRISM1st_label_total should contain Y and N labels; found={sorted(labels)}")
        return (
            f"{task_name}\trows={matrix_shape[0]}\tproteins={matrix_shape[1]}"
            f"\tself_controls={control_summary['self_control_count']}"
            f"\tmissing_controls={control_summary['missing_control_count']}"
        )

    pert1 = info["pert_id1"].astype("string").fillna("").str.strip()
    pert2 = info["pert_id2"].astype("string").fillna("").str.strip()
    if not pert1.eq(pert2).all():
        failures.append("ptv1_extra_singledrug: pert_id2 must equal pert_id1")
    labels = set(info["PRISM2nd_label_total"].astype("string").fillna("").str.strip().unique())
    if not labels.issubset({"0", "1"}) or labels != {"0", "1"}:
        failures.append(f"ptv1_extra_singledrug: PRISM2nd_label_total should contain 0/1; found={sorted(labels)}")
    empty_targets = 0
    bad_targets = 0
    for value in info["target_protein_list"].tolist():
        parsed = parse_json_list(value)
        empty_targets += int(len(parsed) == 0)
        bad_targets += int(any(not isinstance(item, str) for item in parsed))
    if bad_targets:
        failures.append("ptv1_extra_singledrug: standardized target_protein_list must contain UniProt strings before stage 2")
    missing_controls = int(~nonempty(info["control"]).any()) if len(info) else 0
    missing_controls = int(info["control"].astype("string").fillna("").str.strip().eq("").sum())
    return (
        f"{task_name}\trows={matrix_shape[0]}\tproteins={matrix_shape[1]}"
        f"\tmissing_controls={missing_controls}\tempty_target_lists={empty_targets}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    ptv1_root = args.output_root / "ptv1"
    audit_path = ptv1_root / "file_audit.json"
    if not audit_path.exists():
        audit_path = args.output_root / "file_audit.json"
    audit = load_json(audit_path)
    if not isinstance(audit, dict):
        raise SystemExit(f"invalid audit JSON: {audit_path}")

    failures: list[str] = []
    tasks = audit.get("tasks", {})
    for task_name in EXPECTED_TASKS:
        if task_name not in tasks:
            failures.append(f"missing task audit entry: {task_name}")
    summaries = [
        validate_task(task_name, tasks[task_name], failures)
        for task_name in EXPECTED_TASKS
        if task_name in tasks
    ]

    meta_path = ptv1_root / "global_meta.json"
    meta = load_json(meta_path)
    if not isinstance(meta, dict) or meta.get("dataset_group") != "ptv1":
        failures.append(f"invalid PTV1 global_meta: {meta_path}")
    elif meta.get("task_names") != list(EXPECTED_TASKS):
        failures.append(f"PTV1 global_meta task_names mismatch: {meta.get('task_names')}")

    for summary in summaries:
        print(summary)
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print("\nPTV1 standardized validation passed.")


if __name__ == "__main__":
    main()
