#!/usr/bin/env python3
"""Validate training-ready ProteinTalk artifacts."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "training_ready"
DISCRETE_FIELDS = [
    "machineID_new",
    "Cell_plate",
    "Cell",
    "cell_type",
    "batch",
    "pert_time",
    "pert_dose1",
    "pert_dose2",
]


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_json_list_cell(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    return []


def is_control_frame(df: pd.DataFrame) -> pd.Series:
    control = df["control"].astype("string").fillna("").str.strip()
    sample_id = df["sample_id"].astype("string").fillna("").str.strip()
    return control.eq(sample_id) | control.str.lower().eq("control")


def load_feature_table(task_dir: Path) -> pd.DataFrame:
    parquet_path = task_dir / "feature_table.parquet"
    pickle_path = task_dir / "feature_table.pkl"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    raise FileNotFoundError(f"missing feature table parquet/pickle under {task_dir}")


def parse_mapping_index(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid index")
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty index")
    number = float(text)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"non-integer index {value!r}")
    return int(number)


def validate_target_lists(df: pd.DataFrame, *, protein_index_size: int, failures: list[str], task_name: str, frame_name: str) -> None:
    for row_number, value in enumerate(df["target_protein_list"].tolist()):
        try:
            parsed = parse_json_list_cell(value)
        except Exception as exc:
            failures.append(f"{task_name} {frame_name}: target_protein_list row {row_number} is not valid JSON list ({exc})")
            continue
        for item in parsed:
            if not isinstance(item, int):
                failures.append(f"{task_name} {frame_name}: target_protein_list row {row_number} contains non-int value {item!r}")
                break
            if item < 0 or item >= protein_index_size:
                failures.append(f"{task_name} {frame_name}: target_protein_list row {row_number} contains out-of-range protein index {item}")
                break


def validate_filter_rule(df: pd.DataFrame, *, task_kind: str, task_name: str, failures: list[str]) -> None:
    control = df["is_control"].astype(bool)
    if task_kind == "single":
        mask = (~control) & (~df["PRISM1st_label_total"].astype("string").fillna("").str.strip().ne(""))
        if mask.any():
            failures.append(f"{task_name}: found non-control rows with empty PRISM1st_label_total after filtering")
    elif task_kind == "double":
        mask = (~control) & (~df["synergy"].astype("string").fillna("").str.strip().ne(""))
        if mask.any():
            failures.append(f"{task_name}: found non-control rows with empty synergy after filtering")
    elif task_kind == "extra":
        if task_name == "ptv1_extra_singledrug":
            return
        if "_extra_singledrug" in task_name:
            mask = (~control) & (~df["PRISM2nd_label_total"].astype("string").fillna("").str.strip().ne(""))
            if mask.any():
                failures.append(f"{task_name}: found non-control rows with empty PRISM2nd_label_total after filtering")
        elif "_extra_doubledrug" in task_name:
            mask = (~control) & (~df["PRISM1st_label_total"].astype("string").fillna("").str.strip().ne(""))
            if mask.any():
                failures.append(f"{task_name}: found non-control rows with empty PRISM1st_label_total after filtering")
        else:
            failures.append(f"{task_name}: unsupported extra-task filter rule")


def validate_full_protein_axis(
    *,
    task_name: str,
    frame_name: str,
    matrix_shape: tuple[int, ...],
    source_protein_counts: dict[str, Any],
    failures: list[str],
) -> None:
    protein_axis = int(matrix_shape[1])
    if protein_axis == 2000:
        failures.append(
            f"{task_name} {frame_name}: protein axis is exactly 2000; top-2000 protein truncation is forbidden"
        )
    positive_source_counts = [
        int(value)
        for value in source_protein_counts.values()
        if pd.notna(value) and int(value) > 0
    ]
    if positive_source_counts and protein_axis < max(positive_source_counts):
        failures.append(
            f"{task_name} {frame_name}: protein axis {protein_axis} is smaller than max source protein count "
            f"{max(positive_source_counts)}; task matrices must use the full source protein union"
        )


def validate_single_drug_second_slot(
    df: pd.DataFrame,
    *,
    task_name: str,
    frame_name: str,
    failures: list[str],
) -> None:
    if "pert_id1" not in df.columns or "pert_id2" not in df.columns:
        failures.append(f"{task_name} {frame_name}: missing pert_id1/pert_id2 columns")
        return
    control = df["is_control"].astype(bool) if "is_control" in df.columns else is_control_frame(df)
    pert1 = df["pert_id1"].astype("string").fillna("").str.strip()
    pert2 = df["pert_id2"].astype("string").fillna("").str.strip()
    non_control = ~control
    mismatch = non_control & pert1.ne("") & pert2.ne(pert1)
    if mismatch.any():
        failures.append(
            f"{task_name} {frame_name}: single-drug non-control rows must satisfy pert_id2 == pert_id1; "
            f"mismatches={int(mismatch.sum())}"
        )
    if "pert_index1" in df.columns and "pert_index2" in df.columns:
        idx1 = pd.to_numeric(df["pert_index1"], errors="coerce")
        idx2 = pd.to_numeric(df["pert_index2"], errors="coerce")
        index_mismatch = non_control & idx1.notna() & idx2.notna() & idx1.ne(idx2)
        if index_mismatch.any():
            failures.append(
                f"{task_name} {frame_name}: single-drug non-control rows must satisfy pert_index2 == pert_index1; "
                f"mismatches={int(index_mismatch.sum())}"
            )


def validate_single_drug_contracts(
    processed_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    task_name: str,
    task_kind: str,
    failures: list[str],
) -> None:
    is_single_task = (
        task_kind == "single"
        or "_extra_singledrug" in task_name
        or task_name == "ptv1_extra_singledrug"
    )
    if is_single_task:
        validate_single_drug_second_slot(processed_df, task_name=task_name, frame_name="processed", failures=failures)
        validate_single_drug_second_slot(feature_df, task_name=task_name, frame_name="feature", failures=failures)
    if task_name == "ptv3_main_doubledrug":
        membership = feature_df["feature_membership"].astype("string").fillna("").str.strip()
        source_task = feature_df["source_task"].astype("string").fillna("").str.strip()
        auxiliary = feature_df.loc[membership.eq("merged_single_drug") & source_task.eq("ptv3_main_singledrug")]
        validate_single_drug_second_slot(
            auxiliary,
            task_name=task_name,
            frame_name="merged_single_drug_feature",
            failures=failures,
        )


def validate_double_drug_auxiliary_labels(feature_df: pd.DataFrame, *, task_name: str, failures: list[str]) -> None:
    if task_name != "ptv3_main_doubledrug":
        return
    membership = feature_df["feature_membership"].astype("string").fillna("").str.strip()
    source_task = feature_df["source_task"].astype("string").fillna("").str.strip()
    auxiliary = membership.eq("merged_single_drug") & source_task.eq("ptv3_main_singledrug")
    if not auxiliary.any():
        failures.append("ptv3_main_doubledrug: no merged single-drug auxiliary rows found in feature table")
        return
    if "training_label_scope" not in feature_df.columns:
        failures.append("ptv3_main_doubledrug: missing training_label_scope column")
    else:
        bad_scope = auxiliary & ~feature_df["training_label_scope"].astype("string").fillna("").str.strip().eq(
            "single_drug_auxiliary_synergy_masked"
        )
        if bad_scope.any():
            failures.append(
                "ptv3_main_doubledrug: merged single-drug rows are not marked "
                "single_drug_auxiliary_synergy_masked"
            )

    if "synergy" in feature_df.columns:
        non_empty_synergy = auxiliary & feature_df["synergy"].astype("string").fillna("").str.strip().ne("")
        if non_empty_synergy.any():
            failures.append(
                "ptv3_main_doubledrug: merged single-drug auxiliary rows have non-empty synergy; "
                "the synergy head must only consume native double-drug labels"
            )

    if "PRISM1st_label_total" in feature_df.columns:
        non_control_auxiliary = auxiliary & ~feature_df["is_control"].astype(bool)
        empty_active_prism = (
            non_control_auxiliary
            & feature_df["PRISM1st_label_total"].astype("string").fillna("").str.strip().eq("")
        )
        if empty_active_prism.any():
            failures.append(
                "ptv3_main_doubledrug: non-control auxiliary rows are missing active "
                "PRISM1st_label_total response labels"
            )
    source_label = "auxiliary_source_PRISM1st_label_total"
    if source_label not in feature_df.columns:
        failures.append(f"ptv3_main_doubledrug: missing {source_label} audit column")
    else:
        non_control_auxiliary = auxiliary & ~feature_df["is_control"].astype(bool)
        empty_source = non_control_auxiliary & feature_df[source_label].astype("string").fillna("").str.strip().eq("")
        if empty_source.any():
            failures.append(f"ptv3_main_doubledrug: auxiliary rows missing preserved {source_label}")


def encode_binary_label(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None
    positive = {"sensitive", "responsive", "syn", "synergy", "synergistic", "y", "yes", "1", "true"}
    negative = {
        "non-responsive",
        "nonresponsive",
        "non-syn",
        "nonsyn",
        "non-synergy",
        "non_synergy",
        "n",
        "no",
        "0",
        "false",
    }
    if text in positive:
        return 1
    if text in negative:
        return 0
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    if numeric == 1.0:
        return 1
    if numeric == 0.0:
        return 0
    return None


def split_label_column(task_name: str) -> str:
    if task_name == "ptv3_main_doubledrug" or "_extra_doubledrug" in task_name:
        return "synergy"
    if "_extra_singledrug" in task_name or task_name == "ptv1_extra_singledrug":
        return "PRISM2nd_label_total"
    return "PRISM1st_label_total"


def load_split_indices(split_dir: Path, split_name: str, strategy: str, failures: list[str], task_name: str) -> list[int]:
    path = split_dir / f"{split_name}_indices_{strategy}.pkl"
    if not path.exists():
        failures.append(f"{task_name}: missing split indices {path}")
        return []
    return [int(item) for item in load_pickle(path)]


def validate_split_label_balance(
    df: pd.DataFrame,
    *,
    indices: list[int],
    label_column: str,
    task_name: str,
    split_name: str,
    strategy: str,
    failures: list[str],
) -> None:
    if label_column not in df.columns:
        failures.append(f"{task_name} {strategy}/{split_name}: missing active label column {label_column}")
        return
    values = [encode_binary_label(value) for value in df.iloc[indices][label_column].tolist()]
    known = [value for value in values if value is not None]
    missing_count = len(values) - len(known)
    if task_name == "ptv3_main_doubledrug" and split_name == "train":
        subset = df.iloc[indices]
        merged_single_count = int(subset["feature_membership"].astype("string").fillna("").eq("merged_single_drug").sum())
        if missing_count != merged_single_count:
            failures.append(
                f"{task_name} {strategy}/train: missing active synergy labels must correspond exactly to "
                f"merged single-drug rows; missing={missing_count} merged_single={merged_single_count}"
            )
    elif missing_count:
        failures.append(f"{task_name} {strategy}/{split_name}: found {missing_count} missing active labels")
    if not known:
        failures.append(f"{task_name} {strategy}/{split_name}: no known active labels")
        return
    positive_count = sum(value == 1 for value in known)
    negative_count = sum(value == 0 for value in known)
    if split_name in {"valid", "test"} and (positive_count == 0 or negative_count == 0):
        failures.append(
            f"{task_name} {strategy}/{split_name}: active labels need both classes; "
            f"positive={positive_count} negative={negative_count}"
        )


def validate_split_family(
    *,
    output_root: Path,
    dataset_group: str,
    task_name: str,
    feature_df: pd.DataFrame,
    strategies: list[str],
    allow_all_train_subset_overlap: bool,
    failures: list[str],
) -> None:
    split_dir = output_root / dataset_group / "splits" / task_name
    if not split_dir.exists():
        failures.append(f"{task_name}: missing split directory {split_dir}")
        return
    label_column = split_label_column(task_name)
    for strategy in strategies:
        split_sets: dict[str, set[int]] = {}
        split_lists: dict[str, list[int]] = {}
        for split_name in ("train", "valid", "test"):
            indices = load_split_indices(split_dir, split_name, strategy, failures, task_name)
            split_lists[split_name] = indices
            split_sets[split_name] = set(indices)
            if strategy != "test_only" and not indices:
                failures.append(f"{task_name} {strategy}/{split_name}: split must be non-empty")
            if any(index < 0 or index >= len(feature_df) for index in indices):
                failures.append(f"{task_name} {strategy}/{split_name}: contains out-of-range feature row indices")
        if strategy == "test_only":
            if split_lists["train"] or split_lists["valid"]:
                failures.append(f"{task_name} test_only: train/valid must be empty")
            anchors = default_primary_anchor_indices(feature_df)
            if set(split_lists["test"]) != set(anchors):
                failures.append(
                    f"{task_name} test_only: test indices must equal primary non-control anchors; "
                    f"test={len(split_lists['test'])} anchors={len(anchors)}"
                )
            validate_split_label_balance(
                feature_df,
                indices=split_lists["test"],
                label_column=label_column,
                task_name=task_name,
                split_name="test",
                strategy=strategy,
                failures=failures,
            )
            continue
        train_valid = split_sets["train"] & split_sets["valid"]
        train_test = split_sets["train"] & split_sets["test"]
        valid_test = split_sets["valid"] & split_sets["test"]
        if strategy == "all_train_subset_test" and allow_all_train_subset_overlap:
            if valid_test:
                failures.append(f"{task_name} {strategy}: valid/test overlap={len(valid_test)}")
        elif train_valid or train_test or valid_test:
            failures.append(
                f"{task_name} {strategy}: unexpected row overlap "
                f"train_valid={len(train_valid)} train_test={len(train_test)} valid_test={len(valid_test)}"
            )
        for split_name in ("train", "valid", "test"):
            validate_split_label_balance(
                feature_df,
                indices=split_lists[split_name],
                label_column=label_column,
                task_name=task_name,
                split_name=split_name,
                strategy=strategy,
                failures=failures,
            )


def default_primary_anchor_indices(df: pd.DataFrame) -> list[int]:
    is_control = df["is_control"].astype(bool) if "is_control" in df.columns else is_control_frame(df)
    source_role = df.get("source_row_role", pd.Series(["self"] * len(df), index=df.index)).astype("string").fillna("").str.strip()
    membership = df.get("feature_membership", pd.Series(["primary"] * len(df), index=df.index)).astype("string").fillna("").str.strip()
    return [int(idx) for idx in df.index[(~is_control) & source_role.eq("self") & membership.eq("primary")]]


def validate_double_drug_unordered_pair_leakage(
    feature_df: pd.DataFrame,
    *,
    output_root: Path,
    failures: list[str],
) -> None:
    task_name = "ptv3_main_doubledrug"
    split_dir = output_root / "ptv3" / "splits" / task_name
    primary = feature_df["feature_membership"].astype("string").fillna("").eq("primary")
    non_control = ~feature_df["is_control"].astype(bool)
    native_rows = set(feature_df.index[primary & non_control])
    unordered_pair = feature_df.apply(
        lambda row: "+".join(sorted([str(row["pert_id1"]), str(row["pert_id2"])])),
        axis=1,
    )
    for fold_idx in range(5):
        strategy = f"pert_id_5fold_fold{fold_idx}"
        pair_sets: dict[str, set[str]] = {}
        for split_name in ("train", "valid", "test"):
            indices = [
                index
                for index in load_split_indices(split_dir, split_name, strategy, failures, task_name)
                if index in native_rows
            ]
            pair_sets[split_name] = set(unordered_pair.iloc[indices].tolist())
        train_valid = pair_sets["train"] & pair_sets["valid"]
        train_test = pair_sets["train"] & pair_sets["test"]
        valid_test = pair_sets["valid"] & pair_sets["test"]
        if train_valid or train_test or valid_test:
            failures.append(
                f"{task_name} {strategy}: unordered pair leakage "
                f"train_valid={len(train_valid)} train_test={len(train_test)} valid_test={len(valid_test)}"
            )


def validate_required_split_artifacts(
    *,
    output_root: Path,
    audit: dict[str, Any],
    feature_tables: dict[tuple[str, str], pd.DataFrame],
    failures: list[str],
) -> None:
    split_build_manifest_path = output_root / "split_build_manifest.json"
    if not split_build_manifest_path.exists():
        failures.append(f"missing global split build manifest: {split_build_manifest_path}")
    else:
        split_build_manifest = load_json(split_build_manifest_path)
        manifest_tasks = set(split_build_manifest.get("tasks", {}))
        expected_tasks = {f"{payload['dataset_group']}/{task_name}" for task_name, payload in audit["tasks"].items()}
        missing = sorted(expected_tasks - manifest_tasks)
        if missing:
            failures.append(f"split_build_manifest is missing tasks: {missing}")

    single_strategies = (
        [f"pert_stratified_5fold_fold{idx}" for idx in range(5)]
        + [f"cell_type_5fold_fold{idx}" for idx in range(5)]
        + [f"cell_5fold_fold{idx}" for idx in range(5)]
        + ["all_train_subset_test"]
    )
    validate_split_family(
        output_root=output_root,
        dataset_group="ptv3",
        task_name="ptv3_main_singledrug",
        feature_df=feature_tables[("ptv3", "ptv3_main_singledrug")],
        strategies=single_strategies,
        allow_all_train_subset_overlap=True,
        failures=failures,
    )
    double_strategies = [f"pert_id_5fold_fold{idx}" for idx in range(5)] + ["all_train_subset_test"]
    double_df = feature_tables[("ptv3", "ptv3_main_doubledrug")]
    validate_split_family(
        output_root=output_root,
        dataset_group="ptv3",
        task_name="ptv3_main_doubledrug",
        feature_df=double_df,
        strategies=double_strategies,
        allow_all_train_subset_overlap=True,
        failures=failures,
    )
    validate_double_drug_unordered_pair_leakage(double_df, output_root=output_root, failures=failures)
    for task_name in sorted(task for task in audit["tasks"] if task.startswith("ptv3_extra_")):
        validate_split_family(
            output_root=output_root,
            dataset_group="ptv3",
            task_name=task_name,
            feature_df=feature_tables[("ptv3", task_name)],
            strategies=["test_only"],
            allow_all_train_subset_overlap=False,
            failures=failures,
        )


def validate_index_columns(
    df: pd.DataFrame,
    *,
    meta: dict[str, Any],
    task_name: str,
    frame_name: str,
    failures: list[str],
) -> None:
    mappings = meta["value_to_index"]
    for field in DISCRETE_FIELDS:
        index_column = f"{field}_index"
        if index_column not in df.columns:
            failures.append(f"{task_name} {frame_name}: missing {index_column}")
            continue
        mapping_key = "pert_dose" if field in {"pert_dose1", "pert_dose2"} else field
        try:
            allowed_indices = {parse_mapping_index(value) for value in mappings[mapping_key].values()}
        except ValueError as exc:
            failures.append(f"{task_name} {frame_name}: invalid mapping values for {mapping_key} ({exc})")
            continue
        values = pd.to_numeric(df[index_column], errors="coerce")
        if values.isna().any():
            failures.append(f"{task_name} {frame_name}: {index_column} contains non-numeric values")
            continue
        integer_values = values.astype(np.int64)
        if not np.allclose(values.to_numpy(dtype=np.float64), integer_values.to_numpy(dtype=np.float64)):
            failures.append(f"{task_name} {frame_name}: {index_column} contains non-integer values")
            continue
        if not pd.Series(integer_values).isin(allowed_indices).all():
            failures.append(f"{task_name} {frame_name}: {index_column} contains values outside value_to_index[{mapping_key}]")

    for column_name, upper_bound in (
        ("pert_index1", len(meta["pert_index"])),
        ("pert_index2", len(meta["pert_index"])),
    ):
        if column_name not in df.columns:
            failures.append(f"{task_name} {frame_name}: missing {column_name}")
            continue
        values = pd.to_numeric(df[column_name], errors="coerce")
        if values.isna().any():
            failures.append(f"{task_name} {frame_name}: {column_name} contains non-numeric values")
            continue
        if ((values < 0) | (values >= upper_bound)).any():
            failures.append(f"{task_name} {frame_name}: {column_name} contains out-of-range values")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate training-ready ProteinTalk outputs")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    audit = load_json(output_root / "file_audit.json")

    failures: list[str] = []
    summaries: list[str] = []
    feature_tables: dict[tuple[str, str], pd.DataFrame] = {}
    dataset_meta = {
        dataset_group: load_json(Path(payload["global_meta_path"]))
        for dataset_group, payload in audit["dataset_groups"].items()
    }

    for dataset_group, meta in dataset_meta.items():
        protein_special = meta["special_values"]["protein_index"]
        pert_special = meta["special_values"]["pert_index"]
        if "control" not in protein_special or "no" not in protein_special:
            failures.append(f"{dataset_group}: protein_index special values are incomplete")
        if "no" not in pert_special:
            failures.append(f"{dataset_group}: pert_index special values are incomplete")
        for field in ("machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time", "pert_dose1", "pert_dose2"):
            if field not in meta["value_to_index"]:
                failures.append(f"{dataset_group}: missing value_to_index mapping for {field}")
            elif "no" not in meta["value_to_index"][field]:
                failures.append(f"{dataset_group}: value_to_index[{field}] is missing `no`")
        for field in ("pert_dose1", "pert_dose2"):
            mapping = meta["value_to_index"][field]
            if any(not isinstance(value, str) for value in mapping.values()):
                failures.append(f"{dataset_group}: value_to_index[{field}] must store string index values")
                continue
            try:
                no_value = parse_mapping_index(mapping["no"])
                numeric_values = [parse_mapping_index(value) for key, value in mapping.items() if key != "no"]
            except ValueError as exc:
                failures.append(f"{dataset_group}: value_to_index[{field}] contains invalid string indices ({exc})")
                continue
            if numeric_values and no_value != (max(numeric_values) + 1):
                failures.append(f"{dataset_group}: value_to_index[{field}]['no'] must equal max(numeric_index) + 1")

    for task_name, payload in audit["tasks"].items():
        dataset_group = payload["dataset_group"]
        meta = dataset_meta[dataset_group]
        protein_index_size = len(meta["protein_index"])
        task_dir = output_root / dataset_group / "tasks" / task_name

        processed_df = pd.read_csv(task_dir / "processed.csv", low_memory=False)
        processed_matrix = np.load(task_dir / "processed_expression_matrix.npy")
        processed_ordered_protein_index = load_json(task_dir / "processed_ordered_protein_index.json")
        processed_sample_ids = load_json(task_dir / "processed_sample_ids.json")

        feature_df_csv = pd.read_csv(task_dir / "feature_table.csv", low_memory=False)
        feature_df = load_feature_table(task_dir)
        feature_tables[(dataset_group, task_name)] = feature_df
        feature_matrix = np.load(task_dir / "feature_expression_matrix.npy")
        feature_ordered_protein_index = load_json(task_dir / "feature_ordered_protein_index.json")
        feature_sample_ids = load_json(task_dir / "feature_sample_ids.json")
        feature_loading_manifest = load_json(task_dir / "feature_loading_manifest.json")

        if len(processed_df) != processed_matrix.shape[0]:
            failures.append(f"{task_name}: processed rows {len(processed_df)} != processed matrix rows {processed_matrix.shape[0]}")
        if len(feature_df) != feature_matrix.shape[0]:
            failures.append(f"{task_name}: feature rows {len(feature_df)} != feature matrix rows {feature_matrix.shape[0]}")
        if len(feature_df_csv) != len(feature_df):
            failures.append(f"{task_name}: feature_table.csv row count != native feature table row count")
        if feature_df_csv.columns.tolist() != feature_df.columns.tolist():
            failures.append(f"{task_name}: feature_table.csv columns != native feature table columns")
        if len(processed_ordered_protein_index) != processed_matrix.shape[1]:
            failures.append(f"{task_name}: processed ordered protein indices {len(processed_ordered_protein_index)} != processed matrix cols {processed_matrix.shape[1]}")
        if len(feature_ordered_protein_index) != feature_matrix.shape[1]:
            failures.append(f"{task_name}: feature ordered protein indices {len(feature_ordered_protein_index)} != feature matrix cols {feature_matrix.shape[1]}")
        validate_full_protein_axis(
            task_name=task_name,
            frame_name="processed",
            matrix_shape=processed_matrix.shape,
            source_protein_counts=payload.get("processed_source_protein_counts", {}),
            failures=failures,
        )
        validate_full_protein_axis(
            task_name=task_name,
            frame_name="feature",
            matrix_shape=feature_matrix.shape,
            source_protein_counts=payload.get("feature_source_protein_counts", {}),
            failures=failures,
        )
        if len(processed_sample_ids) != len(processed_df):
            failures.append(f"{task_name}: processed sample_ids length mismatch")
        if len(feature_sample_ids) != len(feature_df):
            failures.append(f"{task_name}: feature sample_ids length mismatch")
        if processed_df["sample_id"].astype(str).tolist() != [str(item) for item in processed_sample_ids]:
            failures.append(f"{task_name}: processed sample_ids.json is not aligned with processed.csv")
        if feature_df["sample_id"].astype(str).tolist() != [str(item) for item in feature_sample_ids]:
            failures.append(f"{task_name}: feature sample_ids.json is not aligned with feature table")
        if feature_df_csv["sample_id"].astype(str).tolist() != feature_df["sample_id"].astype(str).tolist():
            failures.append(f"{task_name}: feature_table.csv sample_id order != native feature table")
        if processed_df["sample_id"].duplicated(keep=False).any():
            failures.append(f"{task_name}: duplicated sample_ids found in processed.csv")
        if feature_df["sample_id"].duplicated(keep=False).any():
            failures.append(f"{task_name}: duplicated sample_ids found in feature table")
        if len(processed_ordered_protein_index) != len(set(processed_ordered_protein_index)):
            failures.append(f"{task_name}: processed ordered protein indices are not unique")
        if len(feature_ordered_protein_index) != len(set(feature_ordered_protein_index)):
            failures.append(f"{task_name}: feature ordered protein indices are not unique")
        if any((not isinstance(item, int)) or item < 0 or item >= protein_index_size for item in processed_ordered_protein_index):
            failures.append(f"{task_name}: processed ordered protein index list contains invalid values")
        if any((not isinstance(item, int)) or item < 0 or item >= protein_index_size for item in feature_ordered_protein_index):
            failures.append(f"{task_name}: feature ordered protein index list contains invalid values")

        validate_target_lists(processed_df, protein_index_size=protein_index_size, failures=failures, task_name=task_name, frame_name="processed")
        validate_target_lists(feature_df, protein_index_size=protein_index_size, failures=failures, task_name=task_name, frame_name="feature")
        validate_index_columns(processed_df, meta=meta, task_name=task_name, frame_name="processed", failures=failures)
        validate_index_columns(feature_df, meta=meta, task_name=task_name, frame_name="feature", failures=failures)
        validate_filter_rule(processed_df, task_kind=payload["task_kind"], task_name=task_name, failures=failures)
        validate_filter_rule(feature_df.loc[feature_df["feature_membership"] == "primary"], task_kind=payload["task_kind"], task_name=task_name, failures=failures)
        validate_single_drug_contracts(
            processed_df,
            feature_df,
            task_name=task_name,
            task_kind=payload["task_kind"],
            failures=failures,
        )
        validate_double_drug_auxiliary_labels(feature_df, task_name=task_name, failures=failures)
        if processed_df["is_control"].astype(bool).tolist() != is_control_frame(processed_df).astype(bool).tolist():
            failures.append(f"{task_name}: processed is_control does not match the control rule")
        if feature_df["is_control"].astype(bool).tolist() != is_control_frame(feature_df).astype(bool).tolist():
            failures.append(f"{task_name}: feature is_control does not match the control rule")

        if processed_df["processed_row_index"].astype(int).tolist() != list(range(len(processed_df))):
            failures.append(f"{task_name}: processed_row_index is not a contiguous 0..N-1 range")
        if feature_df["feature_row_index"].astype(int).tolist() != list(range(len(feature_df))):
            failures.append(f"{task_name}: feature_row_index is not a contiguous 0..N-1 range")
        if processed_df["expression_row_index"].astype(int).tolist() != list(range(len(processed_df))):
            failures.append(f"{task_name}: processed expression_row_index is not aligned with processed rows")
        if feature_df["expression_row_index"].astype(int).tolist() != list(range(len(feature_df))):
            failures.append(f"{task_name}: feature expression_row_index is not aligned with feature rows")
        if feature_df_csv["expression_row_index"].astype(int).tolist() != feature_df["expression_row_index"].astype(int).tolist():
            failures.append(f"{task_name}: feature_table.csv expression_row_index != native feature table")

        expected_manifest = {
            "row_key_column": "sample_id",
            "expression_row_index_column": "expression_row_index",
            "expression_matrix_path": str(task_dir / "feature_expression_matrix.npy"),
            "ordered_protein_index_path": str(task_dir / "feature_ordered_protein_index.json"),
            "ordered_protein_uniprot_path": str(task_dir / "feature_ordered_protein_uniprot.json"),
            "sample_ids_path": str(task_dir / "feature_sample_ids.json"),
            "feature_table_csv_path": str(task_dir / "feature_table.csv"),
        }
        for key, expected_value in expected_manifest.items():
            if feature_loading_manifest.get(key) != expected_value:
                failures.append(f"{task_name}: feature_loading_manifest[{key}] is not aligned with task outputs")
        native_table_path = feature_loading_manifest.get("feature_table_native_path", "")
        if native_table_path not in {
            str(task_dir / "feature_table.parquet"),
            str(task_dir / "feature_table.pkl"),
        }:
            failures.append(f"{task_name}: feature_loading_manifest[feature_table_native_path] is not aligned with task outputs")

        summaries.append(
            f"{task_name}\tprocessed={len(processed_df)}x{processed_matrix.shape[1]}\tfeature={len(feature_df)}x{feature_matrix.shape[1]}"
        )

    validate_required_split_artifacts(
        output_root=output_root,
        audit=audit,
        feature_tables=feature_tables,
        failures=failures,
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
