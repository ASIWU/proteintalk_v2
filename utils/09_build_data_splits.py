#!/usr/bin/env python3
"""Build train/valid/test split artifacts for training-ready ProteinTalk data.

This script implements the Step 3 workflow described in
`docs/Data_Process_3.md` against the current `data/training_ready` layout.

The generated indices always refer to rows in each task's `feature_table`,
because the feature table contains both prediction anchors and matched control
rows. Primary self, non-control rows are used as normal split anchors. For the
PTV3 main double-drug task, merged single-drug rows are also indexed as
train-only auxiliary anchors so the double-drug training split can include all
single-drug perturbation rows without leaking them into double-drug validation
or test splits.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
DEFAULT_PTV1_EXPERIMENT_TYPE_DIR = REPO_ROOT / "data" / "rawdata" / "ptv1" / "experiment_type_list"

PTV3_MAIN_SINGLE = "ptv3_main_singledrug"
PTV3_MAIN_DOUBLE = "ptv3_main_doubledrug"
PTV3_EXTRA_TEST_TASKS = {
    "ptv3_extra_singledrug_mat1_480_faims",
    "ptv3_extra_singledrug_mat1_qe",
    "ptv3_extra_singledrug_mat2_480_faims",
    "ptv3_extra_singledrug_mat2_qe",
    "ptv3_extra_singledrug_mat3_qe",
    "ptv3_extra_singledrug_mat4_qe",
    "ptv3_extra_doubledrug_guomics",
    "ptv3_extra_doubledrug_nc",
    "ptv3_extra_doubledrug_nature",
}
PTV1_MAIN = "ptv1_aivc"
PTV1_EXTRA_TEST = "ptv1_extra_singledrug"


@dataclass(frozen=True)
class SplitPayload:
    strategy: str
    train: list[int]
    valid: list[int]
    test: list[int]
    policy: str
    allow_train_test_overlap: bool = False


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def read_feature_table(task_dir: Path) -> pd.DataFrame:
    parquet_path = task_dir / "feature_table.parquet"
    pickle_path = task_dir / "feature_table.pkl"
    csv_path = task_dir / "feature_table.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"missing feature_table parquet/pickle/csv under {task_dir}")


def normalize_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def normalized_series(df: pd.DataFrame, column: str, default: str = "no") -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    values = df[column].astype("string").fillna("").str.strip()
    return values.mask(values.eq(""), default).astype("object")


def as_bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=bool)
    values = df[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    lowered = values.astype("string").fillna("").str.lower().str.strip()
    return lowered.isin({"true", "1", "yes", "y"})


def unique_sorted(indices: Iterable[int]) -> list[int]:
    return sorted({int(index) for index in indices})


def shuffled(items: list[Any], rng: np.random.Generator) -> list[Any]:
    values = list(items)
    rng.shuffle(values)
    return values


def validation_item_count(n_items: int, valid_ratio: float) -> int:
    """Keep validation non-empty when a split has enough non-test groups/rows."""

    if n_items <= 1 or valid_ratio <= 0:
        return 0
    return min(n_items - 1, max(1, int(n_items * valid_ratio)))


def split_items(
    items: list[Any],
    *,
    rng: np.random.Generator,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[list[Any], list[Any], list[Any]]:
    items = shuffled(items, rng)
    n_total = len(items)
    n_test = max(0, n_total - int(n_total * train_ratio))
    test_items = items[:n_test]
    train_valid_items = items[n_test:]
    n_valid = validation_item_count(len(train_valid_items), valid_ratio)
    valid_items = train_valid_items[:n_valid]
    train_items = train_valid_items[n_valid:]
    return train_items, valid_items, test_items


def split_anchor_indices(
    indices: list[int],
    *,
    rng: np.random.Generator,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[list[int], list[int], list[int]]:
    train, valid, test = split_items(indices, rng=rng, train_ratio=train_ratio, valid_ratio=valid_ratio)
    return unique_sorted(train), unique_sorted(valid), unique_sorted(test)


def build_group_map(df: pd.DataFrame, indices: list[int], column: str) -> dict[str, list[int]]:
    values = normalized_series(df, column)
    group_map: dict[str, list[int]] = {}
    for row_index in indices:
        key = normalize_text(values.iloc[row_index], default="no")
        group_map.setdefault(key, []).append(int(row_index))
    return group_map


def indices_from_groups(group_map: dict[str, list[int]], keys: Iterable[str]) -> list[int]:
    rows: list[int] = []
    for key in keys:
        rows.extend(group_map.get(key, []))
    return unique_sorted(rows)


def build_pert_pair_column(df: pd.DataFrame) -> pd.Series:
    pert1 = normalized_series(df, "pert_id1")
    pert2 = normalized_series(df, "pert_id2")
    pairs: list[str] = []
    for left, right in zip(pert1, pert2):
        left_text = str(left)
        right_text = str(right)
        if right_text == "no":
            pairs.append(left_text)
        else:
            first, second = sorted([left_text, right_text])
            pairs.append(f"{first}+{second}")
    return pd.Series(pairs, index=df.index, dtype="object")


def add_internal_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_split_pert_id"] = normalized_series(df, "pert_id1")
    df["_split_pert_pair"] = build_pert_pair_column(df)
    df["_split_cell"] = normalized_series(df, "Cell")
    df["_split_cell_type"] = normalized_series(df, "cell_type")
    return df


def create_group_split(
    df: pd.DataFrame,
    indices: list[int],
    *,
    column: str,
    strategy: str,
    rng: np.random.Generator,
    train_ratio: float,
    valid_ratio: float,
    policy: str,
) -> SplitPayload:
    group_map = build_group_map(df, indices, column)
    train_keys, valid_keys, test_keys = split_items(
        list(group_map), rng=rng, train_ratio=train_ratio, valid_ratio=valid_ratio
    )
    return SplitPayload(
        strategy=strategy,
        train=indices_from_groups(group_map, train_keys),
        valid=indices_from_groups(group_map, valid_keys),
        test=indices_from_groups(group_map, test_keys),
        policy=policy,
    )


def create_group_folds(
    df: pd.DataFrame,
    indices: list[int],
    *,
    column: str,
    strategy_prefix: str,
    rng: np.random.Generator,
    n_folds: int,
    valid_ratio: float,
    policy: str,
) -> list[SplitPayload]:
    group_map = build_group_map(df, indices, column)
    keys = shuffled(list(group_map), rng)
    folds = [list(part) for part in np.array_split(np.asarray(keys, dtype=object), n_folds)]
    payloads: list[SplitPayload] = []
    for fold_idx in range(n_folds):
        test_keys = folds[fold_idx]
        remaining_keys = [key for idx, fold in enumerate(folds) if idx != fold_idx for key in fold]
        valid_count = validation_item_count(len(remaining_keys), valid_ratio)
        valid_keys = remaining_keys[:valid_count]
        train_keys = remaining_keys[valid_count:]
        payloads.append(
            SplitPayload(
                strategy=f"{strategy_prefix}_5fold_fold{fold_idx}",
                train=indices_from_groups(group_map, train_keys),
                valid=indices_from_groups(group_map, valid_keys),
                test=indices_from_groups(group_map, test_keys),
                policy=policy,
            )
        )
    return payloads


def create_stratified_pert_folds(
    df: pd.DataFrame,
    indices: list[int],
    *,
    rng: np.random.Generator,
    n_folds: int,
    valid_ratio: float,
    top_n: int,
) -> list[SplitPayload]:
    group_map = build_group_map(df, indices, "_split_pert_id")
    rows = []
    cell_values = normalized_series(df, "_split_cell")
    for pert_id, pert_indices in group_map.items():
        unique_cells = len({normalize_text(cell_values.iloc[idx], default="no") for idx in pert_indices})
        rows.append((pert_id, unique_cells))
    rows.sort(key=lambda item: (-item[1], item[0]))
    if len(rows) >= top_n * 2:
        effective_top_n = top_n
    else:
        effective_top_n = len(rows) // 2
    top_keys = [item[0] for item in rows[:effective_top_n]]
    rest_keys = [item[0] for item in rows[effective_top_n:]]
    top_folds = [list(part) for part in np.array_split(np.asarray(shuffled(top_keys, rng), dtype=object), n_folds)]
    rest_folds = [list(part) for part in np.array_split(np.asarray(shuffled(rest_keys, rng), dtype=object), n_folds)]
    payloads: list[SplitPayload] = []
    for fold_idx in range(n_folds):
        test_keys = top_folds[fold_idx] + rest_folds[fold_idx]
        remaining_keys = [
            key
            for idx in range(n_folds)
            if idx != fold_idx
            for key in (top_folds[idx] + rest_folds[idx])
        ]
        valid_count = validation_item_count(len(remaining_keys), valid_ratio)
        valid_keys = remaining_keys[:valid_count]
        train_keys = remaining_keys[valid_count:]
        payloads.append(
            SplitPayload(
                strategy=f"pert_stratified_5fold_fold{fold_idx}",
                train=indices_from_groups(group_map, train_keys),
                valid=indices_from_groups(group_map, valid_keys),
                test=indices_from_groups(group_map, test_keys),
                policy=(
                    "5-fold pert_id split stratified by the number of unique Cell values; "
                    f"top_n={effective_top_n}"
                ),
            )
        )
    return payloads


def create_stratified_pert_split(
    df: pd.DataFrame,
    indices: list[int],
    *,
    rng: np.random.Generator,
    train_ratio: float,
    valid_ratio: float,
    top_n: int,
) -> SplitPayload:
    group_map = build_group_map(df, indices, "_split_pert_id")
    cell_values = normalized_series(df, "_split_cell")
    rows = []
    for pert_id, pert_indices in group_map.items():
        unique_cells = len({normalize_text(cell_values.iloc[idx], default="no") for idx in pert_indices})
        rows.append((pert_id, unique_cells))
    rows.sort(key=lambda item: (-item[1], item[0]))
    if len(rows) >= top_n * 2:
        effective_top_n = top_n
    else:
        effective_top_n = len(rows) // 2
    top_keys = [item[0] for item in rows[:effective_top_n]]
    rest_keys = [item[0] for item in rows[effective_top_n:]]
    train_keys: list[str] = []
    valid_keys: list[str] = []
    test_keys: list[str] = []
    for key_group in (top_keys, rest_keys):
        group_train, group_valid, group_test = split_items(
            key_group, rng=rng, train_ratio=train_ratio, valid_ratio=valid_ratio
        )
        train_keys.extend(group_train)
        valid_keys.extend(group_valid)
        test_keys.extend(group_test)
    return SplitPayload(
        strategy="pert_stratified",
        train=indices_from_groups(group_map, train_keys),
        valid=indices_from_groups(group_map, valid_keys),
        test=indices_from_groups(group_map, test_keys),
        policy=(
            "one-shot pert_id split stratified by the number of unique Cell values; "
            f"top_n={effective_top_n}"
        ),
    )


def subset_set_info(set_info: dict[int, dict[str, list[int]]], indices: list[int]) -> dict[int, dict[str, list[int]]]:
    index_set = set(indices)
    subset: dict[int, dict[str, list[int]]] = {}
    for set_idx, info in set_info.items():
        perturb = [idx for idx in info["perturb"] if idx in index_set]
        if perturb:
            subset[int(set_idx)] = {
                "control": list(info["control"]),
                "perturb": perturb,
            }
    return subset


def write_split(output_dir: Path, set_info: dict[int, dict[str, list[int]]], payload: SplitPayload) -> dict[str, Any]:
    train = unique_sorted(payload.train)
    valid = unique_sorted(payload.valid)
    test = unique_sorted(payload.test)
    for split_name, split_indices in (("train", train), ("valid", valid), ("test", test)):
        dump_pickle(output_dir / f"{split_name}_indices_{payload.strategy}.pkl", split_indices)
        dump_pickle(output_dir / f"{split_name}_set_info_{payload.strategy}.pkl", subset_set_info(set_info, split_indices))
    # Compatibility with older scripts that used `val`.
    dump_pickle(output_dir / f"val_indices_{payload.strategy}.pkl", valid)
    dump_pickle(output_dir / f"val_set_info_{payload.strategy}.pkl", subset_set_info(set_info, valid))

    train_set = set(train)
    valid_set = set(valid)
    test_set = set(test)
    overlaps = {
        "train_valid": len(train_set & valid_set),
        "train_test": len(train_set & test_set),
        "valid_test": len(valid_set & test_set),
    }
    if not payload.allow_train_test_overlap and any(overlaps.values()):
        raise ValueError(f"{payload.strategy}: unexpected split overlap {overlaps}")
    if payload.allow_train_test_overlap and overlaps["valid_test"]:
        raise ValueError(f"{payload.strategy}: valid split overlaps test {overlaps}")
    return {
        "strategy": payload.strategy,
        "policy": payload.policy,
        "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "overlaps": overlaps,
        "allow_train_test_overlap": payload.allow_train_test_overlap,
    }


def build_pairing_metadata(
    df: pd.DataFrame,
    *,
    anchor_memberships: set[str] | None = None,
) -> tuple[dict[int, dict[str, list[int]]], dict[int, int], dict[int, dict[str, str]], list[int], dict[str, Any]]:
    if anchor_memberships is None:
        anchor_memberships = {"primary"}
    is_control = as_bool_series(df, "is_control")
    source_role = normalized_series(df, "source_row_role", default="self")
    membership = normalized_series(df, "feature_membership", default="primary")
    sample_ids = normalized_series(df, "sample_id", default="")
    controls = normalized_series(df, "control", default="")

    anchor_mask = (~is_control) & source_role.eq("self") & membership.isin(anchor_memberships)
    candidate_anchor_indices = [int(idx) for idx in df.index[anchor_mask]]

    control_sample_ids = {controls.iloc[idx] for idx in candidate_anchor_indices if controls.iloc[idx]}
    sample_id_to_control_index: dict[str, int] = {}
    duplicate_control_sample_ids: list[str] = []
    for row_index in df.index:
        sample_id = sample_ids.iloc[row_index]
        if sample_id not in control_sample_ids:
            continue
        if sample_id in sample_id_to_control_index:
            duplicate_control_sample_ids.append(sample_id)
            continue
        sample_id_to_control_index[sample_id] = int(row_index)

    valid_anchor_indices: list[int] = []
    skipped_missing_control: list[int] = []
    control_to_perturb: dict[str, list[int]] = {}
    for row_index in candidate_anchor_indices:
        control_id = controls.iloc[row_index]
        if control_id not in sample_id_to_control_index:
            skipped_missing_control.append(row_index)
            continue
        valid_anchor_indices.append(row_index)
        control_to_perturb.setdefault(control_id, []).append(row_index)

    set_info: dict[int, dict[str, list[int]]] = {}
    row_to_set_index: dict[int, int] = {}
    set_to_grouping: dict[int, dict[str, str]] = {}
    for set_idx, control_id in enumerate(sorted(control_to_perturb)):
        control_index = sample_id_to_control_index[control_id]
        perturb_indices = unique_sorted(control_to_perturb[control_id])
        set_info[set_idx] = {"control": [control_index], "perturb": perturb_indices}
        set_to_grouping[set_idx] = {"control_sample_id": control_id}
        row_to_set_index[control_index] = set_idx
        for row_index in perturb_indices:
            row_to_set_index[row_index] = set_idx

    audit = {
        "feature_rows": int(len(df)),
        "candidate_anchor_count": len(candidate_anchor_indices),
        "valid_anchor_count": len(valid_anchor_indices),
        "set_count": len(set_info),
        "skipped_missing_control_count": len(skipped_missing_control),
        "skipped_missing_control_example_indices": skipped_missing_control[:20],
        "duplicate_control_sample_id_count": len(set(duplicate_control_sample_ids)),
        "duplicate_control_sample_id_examples": sorted(set(duplicate_control_sample_ids))[:20],
        "anchor_rule": (
            "not is_control and source_row_role == self and "
            f"feature_membership in {sorted(anchor_memberships)}"
        ),
    }
    return set_info, row_to_set_index, set_to_grouping, unique_sorted(valid_anchor_indices), audit


def anchors_with_membership(
    df: pd.DataFrame,
    anchor_indices: list[int],
    *,
    memberships: set[str],
    source_task: str | None = None,
) -> list[int]:
    membership = normalized_series(df, "feature_membership", default="primary")
    source_tasks = normalized_series(df, "source_task", default="")
    selected: list[int] = []
    for idx in anchor_indices:
        if membership.iloc[idx] not in memberships:
            continue
        if source_task is not None and source_tasks.iloc[idx] != source_task:
            continue
        selected.append(int(idx))
    return unique_sorted(selected)


def append_train_only_anchors(payload: SplitPayload, train_only_indices: list[int], *, reason: str) -> SplitPayload:
    if not train_only_indices:
        return payload
    return SplitPayload(
        strategy=payload.strategy,
        train=unique_sorted(payload.train + train_only_indices),
        valid=payload.valid,
        test=payload.test,
        policy=f"{payload.policy}; {reason}",
        allow_train_test_overlap=payload.allow_train_test_overlap,
    )


def task_label_columns(task_name: str) -> list[str]:
    if task_name in {PTV3_MAIN_DOUBLE, "ptv3_extra_doubledrug_guomics", "ptv3_extra_doubledrug_nc", "ptv3_extra_doubledrug_nature"}:
        return ["synergy"]
    if task_name in PTV3_EXTRA_TEST_TASKS or task_name == PTV1_EXTRA_TEST:
        return ["PRISM2nd_label_total"]
    return ["PRISM1st_label_total"]


def check_label_coverage(df: pd.DataFrame, anchor_indices: list[int], task_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    checked_anchor_count = len(anchor_indices)
    for column in task_label_columns(task_name):
        if column not in df.columns:
            warning = (
                f"{task_name}: checked label column {column!r} is missing for all "
                f"{checked_anchor_count} checked anchors"
            )
            warnings.warn(warning, RuntimeWarning)
            result[column] = {
                "status": "missing_column",
                "checked_anchor_count": checked_anchor_count,
                "missing_anchor_count": checked_anchor_count,
                "missing_anchor_fraction": 1.0 if checked_anchor_count else 0.0,
                "all_labels_missing": bool(checked_anchor_count),
                "warning": warning,
                "missing_anchor_examples": anchor_indices[:20],
            }
            continue
        values = df[column].astype("string").fillna("").str.strip()
        missing = [idx for idx in anchor_indices if values.iloc[idx] == ""]
        all_missing = checked_anchor_count > 0 and len(missing) == checked_anchor_count
        warning = None
        if all_missing:
            warning = (
                f"{task_name}: checked label column {column!r} is empty for all "
                f"{checked_anchor_count} checked anchors"
            )
            warnings.warn(warning, RuntimeWarning)
        result[column] = {
            "status": "all_missing_labels" if all_missing else ("ok" if not missing else "missing_labels"),
            "checked_anchor_count": checked_anchor_count,
            "missing_anchor_count": len(missing),
            "missing_anchor_fraction": (len(missing) / checked_anchor_count) if checked_anchor_count else 0.0,
            "all_labels_missing": all_missing,
            "warning": warning,
            "missing_anchor_examples": missing[:20],
        }
    return result


def make_random_payload(
    indices: list[int],
    *,
    rng: np.random.Generator,
    train_ratio: float,
    valid_ratio: float,
) -> SplitPayload:
    train, valid, test = split_anchor_indices(indices, rng=rng, train_ratio=train_ratio, valid_ratio=valid_ratio)
    return SplitPayload(
        strategy="random",
        train=train,
        valid=valid,
        test=test,
        policy="row-level random split; valid is drawn from the train side before final train write",
    )


def make_all_train_subset_test(
    indices: list[int],
    *,
    rng: np.random.Generator,
    valid_ratio: float,
    subset_test_ratio: float,
) -> SplitPayload:
    shuffled_indices = shuffled(indices, rng)
    test_count = int(len(shuffled_indices) * subset_test_ratio)
    test = unique_sorted(shuffled_indices[:test_count])
    valid_count = validation_item_count(len(shuffled_indices), valid_ratio)
    valid = unique_sorted(shuffled_indices[test_count:test_count + valid_count])
    train = unique_sorted(indices)
    return SplitPayload(
        strategy="all_train_subset_test",
        train=train,
        valid=valid,
        test=test,
        policy="all anchors are train; test is a train subset; valid is a disjoint train subset held out for monitoring",
        allow_train_test_overlap=True,
    )


def make_test_only(indices: list[int]) -> SplitPayload:
    return SplitPayload(
        strategy="test_only",
        train=[],
        valid=[],
        test=unique_sorted(indices),
        policy="all valid anchors are written to test; train and valid are empty",
    )


def parse_ptv1_experiment_type_file(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            first_cell, first_pert = parts[0].split("_", 1)
            pairs.add((normalize_text(first_cell), normalize_text(first_pert)))
            for pert_id in parts[1:]:
                pairs.add((normalize_text(first_cell), normalize_text(pert_id)))
    return pairs


def load_ptv1_experiment_type_pairs(split_dir: Path) -> dict[str, set[tuple[str, str]]]:
    return {
        "train": parse_ptv1_experiment_type_file(split_dir / "train_experiment_type_list.txt"),
        "valid": parse_ptv1_experiment_type_file(split_dir / "val_experiment_type_list.txt"),
        "test": parse_ptv1_experiment_type_file(split_dir / "test_experiment_type_list.txt"),
    }


def make_ptv1_experiment_type_split(
    df: pd.DataFrame,
    indices: list[int],
    *,
    experiment_type_dir: Path,
) -> tuple[SplitPayload, dict[str, Any]]:
    split_pairs = load_ptv1_experiment_type_pairs(experiment_type_dir)
    split_rows: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    unmatched: list[int] = []
    unmatched_examples: list[dict[str, str | int]] = []

    cell_plate = normalized_series(df, "Cell_plate", default="")
    pert_id1 = normalized_series(df, "pert_id1", default="")
    for idx in indices:
        key = (normalize_text(cell_plate.iloc[idx]), normalize_text(pert_id1.iloc[idx]))
        assigned = False
        for split_name, pairs in split_pairs.items():
            if key in pairs:
                split_rows[split_name].append(int(idx))
                assigned = True
                break
        if not assigned:
            unmatched.append(int(idx))
            if len(unmatched_examples) < 20:
                unmatched_examples.append(
                    {
                        "row_index": int(idx),
                        "Cell_plate": key[0],
                        "pert_id1": key[1],
                    }
                )

    audit = {
        "experiment_type_dir": str(experiment_type_dir),
        "pair_counts": {split_name: len(pairs) for split_name, pairs in split_pairs.items()},
        "matched_anchor_count": sum(len(rows) for rows in split_rows.values()),
        "unmatched_anchor_count": len(unmatched),
        "unmatched_anchor_examples": unmatched_examples,
        "matching_key": ["Cell_plate", "pert_id1"],
    }
    return SplitPayload(
        strategy="fixed_experiment_type",
        train=unique_sorted(split_rows["train"]),
        valid=unique_sorted(split_rows["valid"]),
        test=unique_sorted(split_rows["test"]),
        policy="PTV1 fixed split parsed directly from data/rawdata/ptv1/experiment_type_list using (Cell_plate, pert_id1)",
    ), audit


def build_task_splits(
    *,
    dataset_group: str,
    task_name: str,
    task_dir: Path,
    output_dir: Path,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    n_folds: int,
    subset_test_ratio: float,
    stratified_top_n: int,
) -> dict[str, Any]:
    df = add_internal_group_columns(read_feature_table(task_dir).reset_index(drop=True))
    anchor_memberships = {"primary"}
    if dataset_group == "ptv3" and task_name == PTV3_MAIN_DOUBLE:
        anchor_memberships = {"primary", "merged_single_drug"}
    set_info, row_to_set_index, set_to_grouping, anchor_indices, pairing_audit = build_pairing_metadata(
        df,
        anchor_memberships=anchor_memberships,
    )
    primary_anchor_indices = anchors_with_membership(df, anchor_indices, memberships={"primary"})
    auxiliary_single_train_indices: list[int] = []
    label_coverage_anchor_indices = anchor_indices
    if dataset_group == "ptv3" and task_name == PTV3_MAIN_DOUBLE:
        auxiliary_single_train_indices = anchors_with_membership(
            df,
            anchor_indices,
            memberships={"merged_single_drug"},
            source_task=PTV3_MAIN_SINGLE,
        )
        label_coverage_anchor_indices = primary_anchor_indices
        pairing_audit["primary_valid_anchor_count"] = len(primary_anchor_indices)
        pairing_audit["auxiliary_train_anchor_count"] = len(auxiliary_single_train_indices)
        pairing_audit["auxiliary_train_rule"] = (
            "`ptv3_main_singledrug` rows merged into the double-drug feature table "
            "are appended to every double-drug train split only. They are excluded "
            "from double-drug valid/test splits because they do not have synergy labels."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_pickle(output_dir / "set_info.pkl", set_info)
    dump_pickle(output_dir / "row_to_set_index.pkl", row_to_set_index)
    dump_pickle(output_dir / "set_to_grouping.pkl", set_to_grouping)

    rng = np.random.default_rng(seed)
    payloads: list[SplitPayload] = []
    implementation_notes: list[str] = []

    if dataset_group == "ptv3" and task_name == PTV3_MAIN_SINGLE:
        payloads.append(make_random_payload(anchor_indices, rng=rng, train_ratio=train_ratio, valid_ratio=valid_ratio))
        payloads.append(
            create_group_split(
                df,
                anchor_indices,
                column="_split_cell",
                strategy="cell",
                rng=rng,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                policy="group split by Cell",
            )
        )
        payloads.append(
            create_group_split(
                df,
                anchor_indices,
                column="_split_cell_type",
                strategy="cell_type",
                rng=rng,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                policy="group split by cell_type",
            )
        )
        payloads.append(
            create_stratified_pert_split(
                df,
                anchor_indices,
                rng=rng,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                top_n=stratified_top_n,
            )
        )
        payloads.extend(
            create_group_folds(
                df,
                anchor_indices,
                column="_split_pert_id",
                strategy_prefix="pert_id",
                rng=rng,
                n_folds=n_folds,
                valid_ratio=valid_ratio,
                policy="5-fold group split by pert_id1",
            )
        )
        payloads.extend(
            create_group_folds(
                df,
                anchor_indices,
                column="_split_cell",
                strategy_prefix="cell",
                rng=rng,
                n_folds=n_folds,
                valid_ratio=valid_ratio,
                policy="5-fold group split by Cell",
            )
        )
        payloads.extend(
            create_group_folds(
                df,
                anchor_indices,
                column="_split_cell_type",
                strategy_prefix="cell_type",
                rng=rng,
                n_folds=n_folds,
                valid_ratio=valid_ratio,
                policy="5-fold group split by cell_type",
            )
        )
        payloads.extend(
            create_stratified_pert_folds(
                df,
                anchor_indices,
                rng=rng,
                n_folds=n_folds,
                valid_ratio=valid_ratio,
                top_n=stratified_top_n,
            )
        )
        payloads.append(
            make_all_train_subset_test(
                anchor_indices,
                rng=rng,
                valid_ratio=valid_ratio,
                subset_test_ratio=subset_test_ratio,
            )
        )
    elif dataset_group == "ptv3" and task_name == PTV3_MAIN_DOUBLE:
        payloads.extend(
            create_group_folds(
                df,
                primary_anchor_indices,
                column="_split_pert_pair",
                strategy_prefix="pert_id",
                rng=rng,
                n_folds=n_folds,
                valid_ratio=valid_ratio,
                policy="5-fold group split by canonical unordered double-drug pert_id pair",
            )
        )
        if auxiliary_single_train_indices:
            payloads = [
                append_train_only_anchors(
                    payload,
                    auxiliary_single_train_indices,
                    reason="all merged `ptv3_main_singledrug` anchors are added to train only",
                )
                for payload in payloads
            ]
        payloads.append(
            append_train_only_anchors(
                make_all_train_subset_test(
                    primary_anchor_indices,
                    rng=rng,
                    valid_ratio=valid_ratio,
                    subset_test_ratio=subset_test_ratio,
                ),
                auxiliary_single_train_indices,
                reason="all merged `ptv3_main_singledrug` anchors are added to train only",
            )
        )
        implementation_notes.append(
            "Data_Process_3 says double_drug needs pure pert_id 5-fold. The current format has pert_id1 and pert_id2, so this script treats the canonical unordered pair as the fold family to avoid reversed-pair leakage."
        )
        implementation_notes.append(
            "For PTV3 main double-drug training, merged main single-drug anchors are appended to train only for every strategy. Native double-drug anchors still define the pert-pair folds and the valid/test splits."
        )
    elif dataset_group == "ptv3" and task_name in PTV3_EXTRA_TEST_TASKS:
        payloads.append(make_test_only(anchor_indices))
        implementation_notes.append(
            "The Step 3 doc explicitly marks extra_guomics, nc, and nature as test-only. The extra_singledrug tasks are also written as test_only because Step 4 lists extra_singledrug as inference input."
        )
    elif dataset_group == "ptv1" and task_name == PTV1_MAIN:
        fixed_payload, ptv1_split_audit = make_ptv1_experiment_type_split(
            df,
            anchor_indices,
            experiment_type_dir=DEFAULT_PTV1_EXPERIMENT_TYPE_DIR,
        )
        payloads.append(fixed_payload)
        payloads.append(make_random_payload(anchor_indices, rng=rng, train_ratio=train_ratio, valid_ratio=valid_ratio))
        payloads.extend(
            create_group_folds(
                df,
                anchor_indices,
                column="_split_pert_id",
                strategy_prefix="pert_id",
                rng=rng,
                n_folds=n_folds,
                valid_ratio=valid_ratio,
                policy="PTV1 5-fold group split by pert_id1",
            )
        )
        payloads.append(
            make_all_train_subset_test(
                anchor_indices,
                rng=rng,
                valid_ratio=valid_ratio,
                subset_test_ratio=subset_test_ratio,
            )
        )
        implementation_notes.append(
            "PTV1 fixed_experiment_type is parsed directly from rawdata/ptv1/experiment_type_list; random split is also generated for PTV1."
        )
    elif dataset_group == "ptv1" and task_name == PTV1_EXTRA_TEST:
        payloads.append(make_test_only(anchor_indices))
    else:
        payloads.append(make_test_only(anchor_indices))
        implementation_notes.append(f"No explicit split policy found for {task_name}; wrote test_only by default.")

    split_summaries = [write_split(output_dir, set_info, payload) for payload in payloads]
    manifest = {
        "dataset_group": dataset_group,
        "task_name": task_name,
        "generated_at": iso_now(),
        "feature_table": str(task_dir / "feature_table.parquet"),
        "output_dir": str(output_dir),
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "n_folds": n_folds,
        "subset_test_ratio": subset_test_ratio,
        "pairing": pairing_audit,
        "label_coverage": check_label_coverage(df, label_coverage_anchor_indices, task_name),
        "label_coverage_anchor_rule": (
            "primary double-drug anchors only"
            if dataset_group == "ptv3" and task_name == PTV3_MAIN_DOUBLE
            else "all valid anchors"
        ),
        "splits": split_summaries,
        "implementation_notes": implementation_notes,
    }
    if dataset_group == "ptv1" and task_name == PTV1_MAIN:
        manifest["ptv1_experiment_type_split"] = ptv1_split_audit
    dump_json(output_dir / "split_manifest.json", manifest)
    return manifest


def select_tasks(training_ready_root: Path, dataset_group: str, task: str | None) -> list[tuple[str, str]]:
    selected_groups = ["ptv1", "ptv3"] if dataset_group == "all" else [dataset_group]
    selected: list[tuple[str, str]] = []
    for group in selected_groups:
        meta_path = training_ready_root / group / "global_meta.json"
        meta = load_json(meta_path)
        for task_name in meta["task_names"]:
            if task is not None and task_name != task:
                continue
            selected.append((group, task_name))
    if task is not None and not selected:
        raise ValueError(f"task {task!r} not found under dataset_group={dataset_group}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ProteinTalk Step 3 split artifacts")
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", choices=["ptv1", "ptv3", "all"], default="all")
    parser.add_argument("--task", default=None, help="Optional single task name to process")
    parser.add_argument("--output-subdir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--subset-test-ratio", type=float, default=0.2)
    parser.add_argument("--stratified-top-n", type=int, default=100)
    args = parser.parse_args()

    training_ready_root = Path(args.training_ready_root)
    all_manifests: dict[str, Any] = {
        "generated_at": iso_now(),
        "training_ready_root": str(training_ready_root),
        "tasks": {},
    }
    for group, task_name in select_tasks(training_ready_root, args.dataset_group, args.task):
        task_dir = training_ready_root / group / "tasks" / task_name
        output_dir = training_ready_root / group / args.output_subdir / task_name
        print(f"[split] {group}/{task_name} -> {output_dir}")
        manifest = build_task_splits(
            dataset_group=group,
            task_name=task_name,
            task_dir=task_dir,
            output_dir=output_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            n_folds=args.n_folds,
            subset_test_ratio=args.subset_test_ratio,
            stratified_top_n=args.stratified_top_n,
        )
        all_manifests["tasks"][f"{group}/{task_name}"] = {
            "output_dir": manifest["output_dir"],
            "valid_anchor_count": manifest["pairing"]["valid_anchor_count"],
            "strategies": [item["strategy"] for item in manifest["splits"]],
        }

    output_path = training_ready_root / "split_build_manifest.json"
    dump_json(output_path, all_manifests)
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
