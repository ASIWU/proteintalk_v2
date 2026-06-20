#!/usr/bin/env python3
"""Validate isolated PTV1 training-ready artifacts and split contracts."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _shared import REPO_ROOT, load_json


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "training_ready"
MAIN_TASK = "ptv1_aivc"
EXTRA_TASK = "ptv1_extra_singledrug"
MAIN_STRATEGIES = ["fixed_experiment_type", "random"] + [f"pert_id_5fold_fold{idx}" for idx in range(5)] + [
    "all_train_subset_test"
]


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_feature_table(task_dir: Path) -> pd.DataFrame:
    if (task_dir / "feature_table.parquet").exists():
        return pd.read_parquet(task_dir / "feature_table.parquet")
    if (task_dir / "feature_table.pkl").exists():
        return pd.read_pickle(task_dir / "feature_table.pkl")
    return pd.read_csv(task_dir / "feature_table.csv", low_memory=False)


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


def is_control_frame(df: pd.DataFrame) -> pd.Series:
    control = df["control"].astype("string").fillna("").str.strip()
    sample_id = df["sample_id"].astype("string").fillna("").str.strip()
    return control.eq(sample_id) | control.str.lower().eq("control")


def encode_binary_label(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None
    if text in {"y", "yes", "1", "true", "sensitive", "responsive"}:
        return 1
    if text in {"n", "no", "0", "false", "non-responsive", "nonresponsive"}:
        return 0
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isfinite(number):
        if number == 1.0:
            return 1
        if number == 0.0:
            return 0
    return None


def primary_non_control_indices(df: pd.DataFrame) -> list[int]:
    source_role = df.get("source_row_role", pd.Series(["self"] * len(df))).astype("string").fillna("").str.strip()
    membership = df.get("feature_membership", pd.Series(["primary"] * len(df))).astype("string").fillna("").str.strip()
    mask = (~is_control_frame(df)) & source_role.eq("self") & membership.eq("primary")
    return [int(idx) for idx in df.index[mask]]


def validate_task_outputs(
    *,
    output_root: Path,
    task_name: str,
    meta: dict[str, Any],
    failures: list[str],
) -> pd.DataFrame:
    task_dir = output_root / "ptv1" / "tasks" / task_name
    processed = pd.read_csv(task_dir / "processed.csv", low_memory=False)
    feature_csv = pd.read_csv(task_dir / "feature_table.csv", low_memory=False)
    feature = load_feature_table(task_dir)
    processed_matrix = np.load(task_dir / "processed_expression_matrix.npy")
    feature_matrix = np.load(task_dir / "feature_expression_matrix.npy")
    processed_order = load_json(task_dir / "processed_ordered_protein_index.json")
    feature_order = load_json(task_dir / "feature_ordered_protein_index.json")
    processed_ids = load_json(task_dir / "processed_sample_ids.json")
    feature_ids = load_json(task_dir / "feature_sample_ids.json")

    if len(processed) != processed_matrix.shape[0]:
        failures.append(f"{task_name}: processed rows != matrix rows")
    if len(feature) != feature_matrix.shape[0]:
        failures.append(f"{task_name}: feature rows != matrix rows")
    if len(feature_csv) != len(feature) or feature_csv.columns.tolist() != feature.columns.tolist():
        failures.append(f"{task_name}: feature_table.csv is not aligned with native feature table")
    if len(processed_order) != processed_matrix.shape[1] or len(feature_order) != feature_matrix.shape[1]:
        failures.append(f"{task_name}: ordered protein index length does not match matrix cols")
    if processed_matrix.shape[1] == 2000 or feature_matrix.shape[1] == 2000:
        failures.append(f"{task_name}: protein axis is exactly 2000; truncation is forbidden")
    if list(map(str, processed_ids)) != processed["sample_id"].astype(str).tolist():
        failures.append(f"{task_name}: processed_sample_ids.json is not aligned")
    if list(map(str, feature_ids)) != feature["sample_id"].astype(str).tolist():
        failures.append(f"{task_name}: feature_sample_ids.json is not aligned")
    if processed["sample_id"].duplicated().any() or feature["sample_id"].duplicated().any():
        failures.append(f"{task_name}: duplicated sample_id in training-ready outputs")
    if processed["is_control"].astype(bool).tolist() != is_control_frame(processed).astype(bool).tolist():
        failures.append(f"{task_name}: processed is_control mismatch")
    if feature["is_control"].astype(bool).tolist() != is_control_frame(feature).astype(bool).tolist():
        failures.append(f"{task_name}: feature is_control mismatch")

    protein_count = len(meta["protein_index"])
    for frame_name, frame in (("processed", processed), ("feature", feature)):
        for row_idx, value in enumerate(frame["target_protein_list"].tolist()):
            parsed = parse_json_list(value)
            if any((not isinstance(item, int)) or item < 0 or item >= protein_count for item in parsed):
                failures.append(f"{task_name} {frame_name}: invalid target_protein_list at row {row_idx}")
                break
        for column in ("pert_index1", "pert_index2"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or (values < 0).any() or (values >= len(meta["pert_index"])).any():
                failures.append(f"{task_name} {frame_name}: invalid {column}")

    if task_name == EXTRA_TASK:
        for frame_name, frame in (("processed", processed), ("feature", feature)):
            non_control = ~frame["is_control"].astype(bool)
            pert1 = frame["pert_id1"].astype("string").fillna("").str.strip()
            pert2 = frame["pert_id2"].astype("string").fillna("").str.strip()
            if (non_control & pert1.ne(pert2)).any():
                failures.append(f"{task_name} {frame_name}: non-control rows must have pert_id2 == pert_id1")
        if not feature["source_row_role"].astype("string").fillna("").eq("matched_control").any():
            failures.append(f"{task_name}: feature table has no appended matched controls")

    return feature


def load_indices(split_dir: Path, split_name: str, strategy: str, failures: list[str], task_name: str) -> list[int]:
    path = split_dir / f"{split_name}_indices_{strategy}.pkl"
    if not path.exists():
        failures.append(f"{task_name}: missing {path.name}")
        return []
    return [int(item) for item in load_pickle(path)]


def label_column(task_name: str) -> str:
    return "PRISM2nd_label_total" if task_name == EXTRA_TASK else "PRISM1st_label_total"


def validate_label_balance(
    df: pd.DataFrame,
    indices: list[int],
    *,
    task_name: str,
    split_name: str,
    strategy: str,
    failures: list[str],
) -> None:
    labels = [encode_binary_label(value) for value in df.iloc[indices][label_column(task_name)].tolist()]
    known = [value for value in labels if value is not None]
    if len(known) != len(labels):
        failures.append(f"{task_name} {strategy}/{split_name}: missing active labels")
    if not known:
        failures.append(f"{task_name} {strategy}/{split_name}: no known labels")
        return
    pos = sum(value == 1 for value in known)
    neg = sum(value == 0 for value in known)
    if split_name in {"valid", "test"} and (pos == 0 or neg == 0):
        failures.append(f"{task_name} {strategy}/{split_name}: needs both label classes; pos={pos} neg={neg}")


def validate_split_family(
    *,
    output_root: Path,
    task_name: str,
    feature_df: pd.DataFrame,
    strategies: list[str],
    allow_all_train_subset_overlap: bool,
    failures: list[str],
) -> None:
    split_dir = output_root / "ptv1" / "splits" / task_name
    for strategy in strategies:
        split_lists: dict[str, list[int]] = {}
        split_sets: dict[str, set[int]] = {}
        for split_name in ("train", "valid", "test"):
            indices = load_indices(split_dir, split_name, strategy, failures, task_name)
            split_lists[split_name] = indices
            split_sets[split_name] = set(indices)
            if strategy != "test_only" and not indices:
                failures.append(f"{task_name} {strategy}/{split_name}: split must be non-empty")
            if any(index < 0 or index >= len(feature_df) for index in indices):
                failures.append(f"{task_name} {strategy}/{split_name}: out-of-range indices")
        if strategy == "test_only":
            if split_lists["train"] or split_lists["valid"]:
                failures.append(f"{task_name} test_only: train/valid must be empty")
            expected = set(primary_non_control_indices(feature_df))
            if split_sets["test"] != expected:
                failures.append(f"{task_name} test_only: test must cover all primary non-control anchors")
            validate_label_balance(feature_df, split_lists["test"], task_name=task_name, split_name="test", strategy=strategy, failures=failures)
            continue

        train_valid = split_sets["train"] & split_sets["valid"]
        train_test = split_sets["train"] & split_sets["test"]
        valid_test = split_sets["valid"] & split_sets["test"]
        if strategy == "all_train_subset_test" and allow_all_train_subset_overlap:
            if valid_test:
                failures.append(f"{task_name} {strategy}: valid/test overlap={len(valid_test)}")
        elif train_valid or train_test or valid_test:
            failures.append(
                f"{task_name} {strategy}: unexpected overlap train_valid={len(train_valid)} train_test={len(train_test)} valid_test={len(valid_test)}"
            )
        for split_name, indices in split_lists.items():
            validate_label_balance(feature_df, indices, task_name=task_name, split_name=split_name, strategy=strategy, failures=failures)


def validate_pert_id_fold_disjoint(
    *,
    output_root: Path,
    feature_df: pd.DataFrame,
    failures: list[str],
) -> None:
    split_dir = output_root / "ptv1" / "splits" / MAIN_TASK
    pert_ids = feature_df["pert_id1"].astype("string").fillna("").str.strip()
    anchor_set = set(primary_non_control_indices(feature_df))
    for fold in range(5):
        strategy = f"pert_id_5fold_fold{fold}"
        groups: dict[str, set[str]] = {}
        for split_name in ("train", "valid", "test"):
            indices = [idx for idx in load_indices(split_dir, split_name, strategy, failures, MAIN_TASK) if idx in anchor_set]
            groups[split_name] = set(pert_ids.iloc[indices].tolist())
        if groups["train"] & groups["valid"] or groups["train"] & groups["test"] or groups["valid"] & groups["test"]:
            failures.append(f"{MAIN_TASK} {strategy}: pert_id1 appears in multiple split sets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    ptv1_root = args.output_root / "ptv1"
    audit_path = ptv1_root / "file_audit.json"
    if not audit_path.exists():
        audit_path = args.output_root / "file_audit.json"
    audit = load_json(audit_path)
    meta = load_json(ptv1_root / "global_meta.json")
    failures: list[str] = []

    if not isinstance(meta, dict) or meta.get("dataset_group") != "ptv1":
        failures.append("global_meta.json is not a PTV1 meta file")
    if not isinstance(audit, dict):
        failures.append("file_audit.json is not a JSON object")

    main_feature = validate_task_outputs(output_root=args.output_root, task_name=MAIN_TASK, meta=meta, failures=failures)
    extra_feature = validate_task_outputs(output_root=args.output_root, task_name=EXTRA_TASK, meta=meta, failures=failures)
    validate_split_family(
        output_root=args.output_root,
        task_name=MAIN_TASK,
        feature_df=main_feature,
        strategies=MAIN_STRATEGIES,
        allow_all_train_subset_overlap=True,
        failures=failures,
    )
    validate_pert_id_fold_disjoint(output_root=args.output_root, feature_df=main_feature, failures=failures)
    validate_split_family(
        output_root=args.output_root,
        task_name=EXTRA_TASK,
        feature_df=extra_feature,
        strategies=["test_only"],
        allow_all_train_subset_overlap=False,
        failures=failures,
    )

    print(
        f"{MAIN_TASK}\tfeature={len(main_feature)}x{np.load(ptv1_root / 'tasks' / MAIN_TASK / 'feature_expression_matrix.npy').shape[1]}"
    )
    print(
        f"{EXTRA_TASK}\tfeature={len(extra_feature)}x{np.load(ptv1_root / 'tasks' / EXTRA_TASK / 'feature_expression_matrix.npy').shape[1]}"
    )
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print("\nPTV1 training-ready validation passed.")


if __name__ == "__main__":
    main()
