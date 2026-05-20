#!/usr/bin/env python3
"""Build training-ready ProteinTalk artifacts from stage-1 standardized outputs.

This script implements the workflow in `docs/Data_Process_2.md`.

Outputs are written under `data/training_ready/` and include:
1. task-level `processed.csv`
2. task-level processed log1p expression matrices aligned to processed rows
3. task-level feature tables plus aligned expression matrices
4. updated dataset-level `global_meta.json`
5. a build audit describing filters, merges, and implementation assumptions

Notes
-----
- Stage-1 outputs under `data/standardized/` are treated as immutable inputs.
- Feature tables are written as CSV plus a dataframe-native file. Parquet is
  preferred; pickle is used as a fallback when parquet dependencies are absent.
- Rows without native expression values are expanded to all-NaN vectors.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "standardized"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "training_ready"

PTV3_FINAL_TASK_ORDER = [
    "ptv3_main_singledrug",
    "ptv3_main_doubledrug",
    "ptv3_extra_singledrug_mat1_480_faims",
    "ptv3_extra_singledrug_mat1_qe",
    "ptv3_extra_singledrug_mat2_480_faims",
    "ptv3_extra_singledrug_mat2_qe",
    "ptv3_extra_singledrug_mat3_qe",
    "ptv3_extra_singledrug_mat4_qe",
    "ptv3_extra_doubledrug_guomics",
    "ptv3_extra_doubledrug_nc",
    "ptv3_extra_doubledrug_nature",
]

PTV3_SOURCE_TASKS = PTV3_FINAL_TASK_ORDER + ["ptv3_extra_baseline"]
PTV1_FINAL_TASK_ORDER = [
    "ptv1_aivc",
    "ptv1_extra_singledrug",
]

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
TEXT_CANONICAL_FIELDS = {"machineID_new", "Cell_plate", "Cell", "cell_type", "batch"}
DENSE_PROCESSED_FIELDS = [
    "sample_id",
    "source_task",
    "source_row_role",
    "task_context",
    "feature_membership",
]


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    dataset_group: str
    task_kind: str
    merge_main_single_into_feature: bool = False


TASK_SPECS = [
    TaskSpec("ptv3_main_singledrug", "ptv3", "single"),
    TaskSpec("ptv3_main_doubledrug", "ptv3", "double", merge_main_single_into_feature=True),
    TaskSpec("ptv3_extra_singledrug_mat1_480_faims", "ptv3", "extra"),
    TaskSpec("ptv3_extra_singledrug_mat1_qe", "ptv3", "extra"),
    TaskSpec("ptv3_extra_singledrug_mat2_480_faims", "ptv3", "extra"),
    TaskSpec("ptv3_extra_singledrug_mat2_qe", "ptv3", "extra"),
    TaskSpec("ptv3_extra_singledrug_mat3_qe", "ptv3", "extra"),
    TaskSpec("ptv3_extra_singledrug_mat4_qe", "ptv3", "extra"),
    TaskSpec("ptv3_extra_doubledrug_guomics", "ptv3", "extra"),
    TaskSpec("ptv3_extra_doubledrug_nc", "ptv3", "extra"),
    TaskSpec("ptv3_extra_doubledrug_nature", "ptv3", "extra"),
    TaskSpec("ptv1_aivc", "ptv1", "ptv1"),
    TaskSpec("ptv1_extra_singledrug", "ptv1", "extra"),
]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_free_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_json_list_cell(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    text = normalize_free_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return parsed
    return []


def json_list_string(items: list[Any]) -> str:
    return json.dumps(items, ensure_ascii=False)


def unique_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def canonicalize_text_value(value: object) -> str:
    text = normalize_free_text(value)
    if not text:
        return "no"
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "no"


def canonicalize_numeric_value(value: object) -> str:
    text = normalize_free_text(value)
    if not text:
        return "no"
    try:
        number = float(text)
    except ValueError:
        return canonicalize_text_value(text)
    if not math.isfinite(number):
        return "no"
    return f"{number:g}"


def canonicalize_discrete_value(field: str, value: object) -> str:
    if field in TEXT_CANONICAL_FIELDS:
        return canonicalize_text_value(value)
    return canonicalize_numeric_value(value)


def build_value_to_index(values: list[str]) -> dict[str, int]:
    ordered = {"no": 0}
    for value in sorted({value for value in values if value and value != "no"}):
        ordered[value] = len(ordered)
    return ordered


def build_pert_dose_value_to_index(values: list[str]) -> dict[str, str]:
    ordered_numeric_values = sorted(
        {value for value in values if value and value != "no"},
        key=lambda item: (float(item), item),
    )
    mapping: dict[str, str] = {}
    max_index = -1
    for value in ordered_numeric_values:
        index_value = int(math.ceil(float(value)))
        mapping[value] = str(index_value)
        max_index = max(max_index, index_value)
    mapping["no"] = str(max_index + 1 if max_index >= 0 else 0)
    return mapping


def is_non_empty_series(series: pd.Series) -> pd.Series:
    return (~series.isna()) & (series.astype("string").fillna("").str.strip() != "")


def is_control_frame(df: pd.DataFrame) -> pd.Series:
    control = df["control"].astype("string").fillna("").str.strip()
    sample_id = df["sample_id"].astype("string").fillna("").str.strip()
    return control.eq(sample_id) | control.str.lower().eq("control")


def inverse_index(mapping: dict[str, int]) -> list[str]:
    result = [""] * len(mapping)
    for key, index in mapping.items():
        result[index] = key
    return result


class Stage1Cache:
    """Lazy loader for stage-1 standardized task artifacts."""

    def __init__(self, input_root: Path, dataset_group: str) -> None:
        self.input_root = input_root
        self.dataset_group = dataset_group
        self.group_root = input_root / dataset_group
        self.task_root = self.group_root / "tasks"
        self.info_cache: dict[str, pd.DataFrame] = {}
        self.protein_order_cache: dict[str, list[str]] = {}
        self.row_index_cache: dict[str, dict[str, int]] = {}
        self.matrix_cache: dict[str, np.ndarray] = {}

    def task_dir(self, task_name: str) -> Path:
        return self.task_root / task_name

    def load_info(self, task_name: str) -> pd.DataFrame:
        if task_name not in self.info_cache:
            self.info_cache[task_name] = pd.read_csv(
                self.task_dir(task_name) / "info.csv",
                low_memory=False,
            )
        return self.info_cache[task_name].copy()

    def load_protein_order(self, task_name: str) -> list[str]:
        if task_name not in self.protein_order_cache:
            self.protein_order_cache[task_name] = list(load_json(self.task_dir(task_name) / "protein_order.json"))
        return list(self.protein_order_cache[task_name])

    def load_row_index_map(self, task_name: str) -> dict[str, int]:
        if task_name not in self.row_index_cache:
            payload = load_json(self.task_dir(task_name) / "sample_id_to_row_index.json")
            self.row_index_cache[task_name] = {str(key): int(value) for key, value in payload.items()}
        return dict(self.row_index_cache[task_name])

    def load_expression_matrix(self, task_name: str) -> np.ndarray:
        if task_name not in self.matrix_cache:
            matrix = np.load(self.task_dir(task_name) / "expression_matrix.npy")
            matrix = np.asarray(matrix, dtype=np.float32)
            finite_mask = np.isfinite(matrix)
            matrix[finite_mask] = np.log1p(matrix[finite_mask])
            self.matrix_cache[task_name] = matrix
        return self.matrix_cache[task_name]

    def release(self) -> None:
        self.info_cache.clear()
        self.protein_order_cache.clear()
        self.row_index_cache.clear()
        self.matrix_cache.clear()
        gc.collect()


def collect_stage1_task_names(input_root: Path, dataset_group: str) -> list[str]:
    task_root = input_root / dataset_group / "tasks"
    return sorted(path.name for path in task_root.iterdir() if path.is_dir())


def build_stage2_global_meta(
    *,
    dataset_group: str,
    input_root: Path,
    cache: Stage1Cache,
) -> dict[str, Any]:
    stage1_meta = load_json(input_root / dataset_group / "global_meta.json")
    task_names = collect_stage1_task_names(input_root, dataset_group)

    protein_index = {str(key): int(value) for key, value in stage1_meta["protein_index"].items()}
    for special_value in ("control", "no"):
        if special_value not in protein_index:
            protein_index[special_value] = len(protein_index)

    pert_index = {str(key): int(value) for key, value in stage1_meta["pert_index"].items()}
    if "no" not in pert_index:
        pert_index["no"] = len(pert_index)

    pertid_to_smiles = {str(key): normalize_free_text(value) for key, value in stage1_meta.get("pertid_to_smiles", {}).items()}
    for pert_id in pert_index:
        pertid_to_smiles.setdefault(pert_id, "")

    raw_target_map = {
        str(key): list(value) if isinstance(value, list) else []
        for key, value in stage1_meta.get("pertid_to_target_protein_list", {}).items()
    }
    pertid_to_target_index_list: dict[str, list[int]] = {}
    pertid_to_target_uniprot_list: dict[str, list[str]] = {}
    pertid_to_missing_target_uniprot: dict[str, list[str]] = {}
    for pert_id in pert_index:
        raw_targets = [normalize_free_text(item) for item in raw_target_map.get(pert_id, []) if normalize_free_text(item)]
        mapped_indices: list[int] = []
        missing_targets: list[str] = []
        for protein in raw_targets:
            if protein in protein_index:
                mapped_indices.append(protein_index[protein])
            else:
                missing_targets.append(protein)
        pertid_to_target_index_list[pert_id] = unique_preserve_order(mapped_indices)
        pertid_to_target_uniprot_list[pert_id] = unique_preserve_order(raw_targets)
        if missing_targets:
            pertid_to_missing_target_uniprot[pert_id] = unique_preserve_order(missing_targets)

    normalized_values: dict[str, list[str]] = {field: [] for field in DISCRETE_FIELDS}
    for task_name in task_names:
        info_df = cache.load_info(task_name)
        for field in DISCRETE_FIELDS:
            if field not in info_df.columns:
                normalized_values[field].append("no")
                continue
            normalized_values[field].extend(
                canonicalize_discrete_value(field, value) for value in info_df[field].tolist()
            )

    shared_dose_mapping = build_pert_dose_value_to_index(
        normalized_values["pert_dose1"] + normalized_values["pert_dose2"]
    )
    value_to_index = {
        "machineID_new": build_value_to_index(normalized_values["machineID_new"]),
        "Cell_plate": build_value_to_index(normalized_values["Cell_plate"]),
        "Cell": build_value_to_index(normalized_values["Cell"]),
        "cell_type": build_value_to_index(normalized_values["cell_type"]),
        "batch": build_value_to_index(normalized_values["batch"]),
        "pert_time": build_value_to_index(normalized_values["pert_time"]),
        "pert_dose": shared_dose_mapping,
        "pert_dose1": shared_dose_mapping,
        "pert_dose2": shared_dose_mapping,
    }

    return {
        "dataset_group": dataset_group,
        "generated_at": iso_now(),
        "source_root": str(input_root / dataset_group),
        "source_task_names": task_names,
        "task_names": PTV3_FINAL_TASK_ORDER if dataset_group == "ptv3" else PTV1_FINAL_TASK_ORDER,
        "protein_index": protein_index,
        "protein_index_to_id": inverse_index(protein_index),
        "pert_index": pert_index,
        "pert_index_to_id": inverse_index(pert_index),
        "pertid_to_smiles": pertid_to_smiles,
        "pertid_to_target_protein_list": pertid_to_target_index_list,
        "pertid_to_target_uniprot_list": pertid_to_target_uniprot_list,
        "pertid_to_missing_target_uniprot": pertid_to_missing_target_uniprot,
        "value_to_index": value_to_index,
        "categorical_normalization_rules": {
            "machineID_new": "uppercase, replace punctuation/space/hyphen with `_`, collapse repeated `_`",
            "Cell_plate": "uppercase, replace punctuation/space/hyphen with `_`, collapse repeated `_`",
            "Cell": "uppercase, replace punctuation/space/hyphen with `_`, collapse repeated `_`",
            "cell_type": "uppercase, replace punctuation/space/hyphen with `_`, collapse repeated `_`",
            "batch": "uppercase, replace punctuation/space/hyphen with `_`, collapse repeated `_`",
            "pert_time": "numeric values use compact float text; non-numeric values fall back to uppercase canonical text",
            "pert_dose1": "shared with pert_dose2; numeric values use compact float text and map to stringified ceil(dose); missing values map to `no`, whose index is string(max_numeric_index + 1)",
            "pert_dose2": "shared with pert_dose1; numeric values use compact float text and map to stringified ceil(dose); missing values map to `no`, whose index is string(max_numeric_index + 1)",
        },
        "special_values": {
            "protein_index": {"control": protein_index["control"], "no": protein_index["no"]},
            "pert_index": {"no": pert_index["no"]},
        },
    }


def append_target_index_columns(
    df: pd.DataFrame,
    *,
    protein_index: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    raw_strings: list[str] = []
    mapped_strings: list[str] = []
    missing_strings: list[str] = []
    total_missing_targets = 0
    rows_with_missing_targets = 0

    for value in df.get("target_protein_list", pd.Series(["[]"] * len(df))).tolist():
        raw_targets = [normalize_free_text(item) for item in parse_json_list_cell(value) if normalize_free_text(item)]
        mapped_indices: list[int] = []
        missing_targets: list[str] = []
        for protein in raw_targets:
            if protein in protein_index:
                mapped_indices.append(protein_index[protein])
            else:
                missing_targets.append(protein)
        mapped_indices = unique_preserve_order(mapped_indices)
        missing_targets = unique_preserve_order(missing_targets)
        raw_strings.append(json_list_string(raw_targets))
        mapped_strings.append(json_list_string(mapped_indices))
        missing_strings.append(json_list_string(missing_targets))
        if missing_targets:
            rows_with_missing_targets += 1
            total_missing_targets += len(missing_targets)

    df["target_protein_uniprot_list"] = raw_strings
    df["target_protein_list"] = mapped_strings
    df["target_protein_missing_from_index"] = missing_strings
    df["target_protein_count"] = df["target_protein_list"].map(lambda text: len(parse_json_list_cell(text)))
    return df, {
        "rows_with_missing_target_uniprot": rows_with_missing_targets,
        "total_missing_target_uniprot": total_missing_targets,
    }


def add_common_columns(df: pd.DataFrame, *, task_context: str) -> pd.DataFrame:
    df = df.copy()
    df["task_context"] = task_context
    if "source_task" not in df.columns:
        df["source_task"] = task_context
    if "source_row_role" not in df.columns:
        df["source_row_role"] = "self"
    if "feature_membership" not in df.columns:
        df["feature_membership"] = "primary"
    return df


def task_uses_single_drug_second_slot_copy(task_name: str, task_kind: str) -> bool:
    return (
        task_kind == "single"
        or "_extra_singledrug" in task_name
        or task_name in {"ptv1_aivc", "ptv1_extra_singledrug", "ptv3_extra_baseline"}
    )


def enforce_single_drug_second_slot(
    df: pd.DataFrame,
    *,
    task_name: str,
    task_kind: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """For single-drug rows, use the same perturbation id in both model input slots."""

    if not task_uses_single_drug_second_slot_copy(task_name, task_kind):
        return df, {"applied": False}
    df = df.copy()
    if "pert_id1" not in df.columns:
        return df, {"applied": False, "reason": "missing pert_id1"}
    if "pert_id2" not in df.columns:
        df["pert_id2"] = ""
    pert1 = df["pert_id1"].astype("string").fillna("").str.strip()
    pert2 = df["pert_id2"].astype("string").fillna("").str.strip()
    copy_mask = pert1.ne("") & (pert2.eq("") | pert2.str.lower().eq("no"))
    df.loc[copy_mask, "pert_id2"] = df.loc[copy_mask, "pert_id1"]
    return df, {
        "applied": True,
        "rule": "single-drug rows use `pert_id2 == pert_id1`; blank/`no` second slots are copied before index encoding",
        "copied_rows": int(copy_mask.sum()),
    }


def sanitize_double_drug_auxiliary_rows(feature_df: pd.DataFrame, *, task_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mask synergy labels on merged single-drug rows inside the main double-drug task."""
    feature_df = feature_df.copy()
    if task_name != "ptv3_main_doubledrug":
        return feature_df, {"applied": False}

    auxiliary_mask = (
        feature_df["feature_membership"].astype("string").fillna("").str.strip().eq("merged_single_drug")
        & feature_df["source_task"].astype("string").fillna("").str.strip().eq("ptv3_main_singledrug")
    )
    if "training_label_scope" not in feature_df.columns:
        feature_df["training_label_scope"] = "native_task"
    feature_df.loc[auxiliary_mask, "training_label_scope"] = "single_drug_auxiliary_synergy_masked"

    preserved_counts: dict[str, int] = {}
    for label_column in ("PRISM1st_label_total", "PRISM2nd_label_total", "synergy"):
        if label_column not in feature_df.columns:
            continue
        original_column = f"auxiliary_source_{label_column}"
        if original_column not in feature_df.columns:
            feature_df[original_column] = pd.NA
        valid_mask = auxiliary_mask & is_non_empty_series(feature_df[label_column])
        feature_df.loc[valid_mask, original_column] = feature_df.loc[valid_mask, label_column]
        preserved_counts[label_column] = int(valid_mask.sum())

    if "synergy" in feature_df.columns:
        feature_df.loc[auxiliary_mask, "synergy"] = pd.NA

    return feature_df, {
        "applied": True,
        "auxiliary_single_drug_rows": int(auxiliary_mask.sum()),
        "rule": (
            "Merged `ptv3_main_singledrug` rows in `ptv3_main_doubledrug` are "
            "single-drug auxiliary rows: PRISM response labels are preserved "
            "for audit and non-synergy uses, while active synergy labels are cleared so "
            "double-drug loss2 is trained only on native double-drug synergy rows. "
            "Original auxiliary labels are retained in `auxiliary_source_*` columns."
        ),
        "cleared_active_label_columns": ["synergy"],
        "preserved_source_label_counts": preserved_counts,
    }


def add_index_columns(df: pd.DataFrame, *, meta: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    value_to_index = meta["value_to_index"]
    for field in DISCRETE_FIELDS:
        mapping_key = "pert_dose" if field in {"pert_dose1", "pert_dose2"} else field
        mapping = value_to_index[mapping_key]
        normalized = [canonicalize_discrete_value(field, value) for value in df.get(field, pd.Series([""] * len(df))).tolist()]
        df[f"{field}_norm"] = normalized
        df[f"{field}_index"] = [mapping.get(value, mapping["no"]) for value in normalized]

    pert_index = meta["pert_index"]
    df["pert_index1"] = [
        pert_index.get(normalize_free_text(value) or "no", pert_index["no"])
        for value in df.get("pert_id1", pd.Series([""] * len(df))).tolist()
    ]
    df["pert_index2"] = [
        pert_index.get(normalize_free_text(value) or "no", pert_index["no"])
        for value in df.get("pert_id2", pd.Series([""] * len(df))).tolist()
    ]
    df["is_control"] = is_control_frame(df)
    return df


def ensure_unique_sample_ids(df: pd.DataFrame, *, task_name: str, frame_name: str) -> None:
    duplicated = df["sample_id"].astype("string").fillna("").duplicated(keep=False)
    if duplicated.any():
        sample_ids = df.loc[duplicated, "sample_id"].astype(str).unique().tolist()
        raise ValueError(f"{task_name} {frame_name}: duplicated sample_ids detected: {sample_ids[:10]}")


def fetch_control_rows_for_extra_task(task_name: str, own_df: pd.DataFrame, cache: Stage1Cache) -> pd.DataFrame:
    required_controls = own_df.loc[
        own_df["control"].astype("string").fillna("").str.strip() != "",
        ["control", "control_match_source_task"],
    ].copy()
    required_controls["control"] = required_controls["control"].astype("string").fillna("").str.strip()
    required_controls["control_match_source_task"] = required_controls["control_match_source_task"].astype("string").fillna("").str.strip()
    required_controls = required_controls[
        (required_controls["control"] != "") & (required_controls["control_match_source_task"] != "")
    ].drop_duplicates()

    control_rows: list[pd.DataFrame] = []
    for control_sample_id, source_task in required_controls.itertuples(index=False):
        source_df = cache.load_info(source_task)
        matched = source_df.loc[source_df["sample_id"].astype("string").fillna("").str.strip() == control_sample_id].copy()
        if matched.empty:
            raise ValueError(f"{task_name}: matched control {control_sample_id} was not found in {source_task}")
        matched["source_task"] = source_task
        matched["source_row_role"] = "matched_control"
        matched["feature_membership"] = "primary"
        matched["matched_into_task"] = task_name
        control_rows.append(matched.head(1))

    if not control_rows:
        return pd.DataFrame(columns=list(own_df.columns) + ["matched_into_task"])
    return pd.concat(control_rows, ignore_index=True, sort=False)


def apply_processed_filter(df: pd.DataFrame, *, task_name: str, task_kind: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    control_mask = is_control_frame(df)
    filter_rule = "none"
    kept_mask = pd.Series(True, index=df.index)
    if task_kind == "single":
        filter_rule = "non-control rows require non-empty PRISM1st_label_total"
        kept_mask = control_mask | is_non_empty_series(df["PRISM1st_label_total"])
    elif task_kind == "double":
        filter_rule = "non-control rows require non-empty synergy"
        kept_mask = control_mask | is_non_empty_series(df["synergy"])
    elif task_kind == "extra":
        if task_name == "ptv1_extra_singledrug":
            filter_rule = "none (ptv1 extra rows are kept after stage-1 unique (cell, E115_id) selection)"
            kept_mask = pd.Series(True, index=df.index)
        elif "_extra_singledrug" in task_name:
            filter_rule = "non-control rows require non-empty PRISM2nd_label_total"
            kept_mask = control_mask | is_non_empty_series(df["PRISM2nd_label_total"])
        elif "_extra_doubledrug" in task_name:
            filter_rule = "non-control rows require non-empty PRISM1st_label_total"
            kept_mask = control_mask | is_non_empty_series(df["PRISM1st_label_total"])
        else:
            raise ValueError(f"{task_name}: unsupported extra-task filter rule")

    kept_df = df.loc[kept_mask].reset_index(drop=True)
    removed_df = df.loc[~kept_mask]
    return kept_df, {
        "control_rule": "control row when control == sample_id or control == `control`",
        "filter_rule": filter_rule,
        "rows_before_filter": int(len(df)),
        "rows_after_filter": int(len(kept_df)),
        "removed_non_control_rows": int(len(removed_df)),
        "control_rows_before_filter": int(control_mask.sum()),
        "control_rows_after_filter": int(is_control_frame(kept_df).sum()),
    }


def collect_ordered_protein_indices(
    source_tasks: list[str],
    *,
    cache: Stage1Cache,
    meta: dict[str, Any],
) -> tuple[list[int], list[str], dict[str, int]]:
    protein_index = meta["protein_index"]
    collected: set[int] = set()
    per_source_counts: dict[str, int] = {}
    for source_task in source_tasks:
        source_proteins = cache.load_protein_order(source_task)
        per_source_counts[source_task] = len(source_proteins)
        for protein in source_proteins:
            if protein not in protein_index:
                raise ValueError(f"{source_task}: protein {protein} is missing from {meta['dataset_group']} global_meta protein_index")
            collected.add(protein_index[protein])
    ordered_indices = sorted(collected)
    ordered_uniprot = [meta["protein_index_to_id"][index] for index in ordered_indices]
    return ordered_indices, ordered_uniprot, per_source_counts


def build_aligned_expression_matrix(
    df: pd.DataFrame,
    *,
    ordered_protein_indices: list[int],
    cache: Stage1Cache,
    meta: dict[str, Any],
) -> np.ndarray:
    matrix = np.full((len(df), len(ordered_protein_indices)), np.nan, dtype=np.float32)
    if not len(df) or not ordered_protein_indices:
        return matrix

    ordered_index_position = {protein_index: position for position, protein_index in enumerate(ordered_protein_indices)}
    for source_task, source_rows in df.groupby("source_task", sort=False):
        source_protein_order = cache.load_protein_order(source_task)
        if not source_protein_order:
            continue

        source_matrix = cache.load_expression_matrix(source_task)
        source_row_index = cache.load_row_index_map(source_task)
        source_sample_ids = source_rows["sample_id"].astype(str).tolist()
        source_positions = [int(position) for position in source_rows.index.tolist()]
        stage1_rows = [source_row_index[sample_id] for sample_id in source_sample_ids]

        source_pairs = [
            (source_position, ordered_index_position[meta["protein_index"][protein]])
            for source_position, protein in enumerate(source_protein_order)
            if meta["protein_index"][protein] in ordered_index_position
        ]
        if source_pairs:
            source_matrix_positions = [source_position for source_position, _ in source_pairs]
            source_target_positions = [target_position for _, target_position in source_pairs]
            selected = np.asarray(source_matrix[np.ix_(stage1_rows, source_matrix_positions)], dtype=np.float32)
            matrix[np.ix_(source_positions, source_target_positions)] = selected
    return matrix


def save_feature_dataframe(df: pd.DataFrame, output_dir: Path) -> tuple[str, str]:
    csv_path = output_dir / "feature_table.csv"
    df.to_csv(csv_path, index=False)

    try:
        parquet_path = output_dir / "feature_table.parquet"
        df.to_parquet(parquet_path, index=False)
        return str(csv_path), str(parquet_path)
    except Exception:
        pickle_path = output_dir / "feature_table.pkl"
        df.to_pickle(pickle_path)
        return str(csv_path), str(pickle_path)


def write_feature_loading_manifest(
    *,
    output_dir: Path,
    feature_table_csv_path: str,
    feature_table_native_path: str,
) -> str:
    manifest_path = output_dir / "feature_loading_manifest.json"
    dump_json(
        manifest_path,
        {
            "row_key_column": "sample_id",
            "expression_row_index_column": "expression_row_index",
            "expression_matrix_path": str(output_dir / "feature_expression_matrix.npy"),
            "ordered_protein_index_path": str(output_dir / "feature_ordered_protein_index.json"),
            "ordered_protein_uniprot_path": str(output_dir / "feature_ordered_protein_uniprot.json"),
            "sample_ids_path": str(output_dir / "feature_sample_ids.json"),
            "feature_table_csv_path": feature_table_csv_path,
            "feature_table_native_path": feature_table_native_path,
            "loading_contract": (
                "For any row in the feature table, use `expression_row_index` to index "
                "`feature_expression_matrix.npy`; that row vector is ready to feed to the model."
            ),
        },
    )
    return str(manifest_path)


def write_processed_outputs(
    *,
    output_dir: Path,
    df: pd.DataFrame,
    matrix: np.ndarray,
    ordered_protein_indices: list[int],
    ordered_protein_uniprot: list[str],
) -> dict[str, str]:
    processed_csv_path = output_dir / "processed.csv"
    df.to_csv(processed_csv_path, index=False)
    np.save(output_dir / "processed_expression_matrix.npy", matrix)
    dump_json(output_dir / "processed_ordered_protein_index.json", ordered_protein_indices)
    dump_json(output_dir / "processed_ordered_protein_uniprot.json", ordered_protein_uniprot)
    dump_json(output_dir / "processed_sample_ids.json", df["sample_id"].astype(str).tolist())
    return {
        "processed_csv_path": str(processed_csv_path),
        "processed_expression_matrix_path": str(output_dir / "processed_expression_matrix.npy"),
        "processed_ordered_protein_index_path": str(output_dir / "processed_ordered_protein_index.json"),
        "processed_ordered_protein_uniprot_path": str(output_dir / "processed_ordered_protein_uniprot.json"),
        "processed_sample_ids_path": str(output_dir / "processed_sample_ids.json"),
    }


def write_feature_outputs(
    *,
    output_dir: Path,
    df: pd.DataFrame,
    matrix: np.ndarray,
    ordered_protein_indices: list[int],
    ordered_protein_uniprot: list[str],
) -> dict[str, str]:
    feature_table_csv_path, feature_table_native_path = save_feature_dataframe(df, output_dir)
    np.save(output_dir / "feature_expression_matrix.npy", matrix)
    dump_json(output_dir / "feature_ordered_protein_index.json", ordered_protein_indices)
    dump_json(output_dir / "feature_ordered_protein_uniprot.json", ordered_protein_uniprot)
    dump_json(output_dir / "feature_sample_ids.json", df["sample_id"].astype(str).tolist())
    feature_loading_manifest_path = write_feature_loading_manifest(
        output_dir=output_dir,
        feature_table_csv_path=feature_table_csv_path,
        feature_table_native_path=feature_table_native_path,
    )
    return {
        "feature_table_csv_path": feature_table_csv_path,
        "feature_table_native_path": feature_table_native_path,
        "feature_expression_matrix_path": str(output_dir / "feature_expression_matrix.npy"),
        "feature_ordered_protein_index_path": str(output_dir / "feature_ordered_protein_index.json"),
        "feature_ordered_protein_uniprot_path": str(output_dir / "feature_ordered_protein_uniprot.json"),
        "feature_sample_ids_path": str(output_dir / "feature_sample_ids.json"),
        "feature_loading_manifest_path": feature_loading_manifest_path,
    }


def build_task_outputs(
    *,
    spec: TaskSpec,
    cache: Stage1Cache,
    meta: dict[str, Any],
    output_root: Path,
    processed_registry: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    own_df = cache.load_info(spec.task_name)
    own_df["source_task"] = spec.task_name
    own_df["source_row_role"] = "self"
    own_df["feature_membership"] = "primary"

    appended_control_df = pd.DataFrame()
    if spec.task_kind == "extra":
        appended_control_df = fetch_control_rows_for_extra_task(spec.task_name, own_df, cache)

    combined_processed_df = pd.concat([own_df, appended_control_df], ignore_index=True, sort=False)
    combined_processed_df = add_common_columns(combined_processed_df, task_context=spec.task_name)
    combined_processed_df, single_slot_audit = enforce_single_drug_second_slot(
        combined_processed_df,
        task_name=spec.task_name,
        task_kind=spec.task_kind,
    )
    processed_df, filter_audit = apply_processed_filter(
        combined_processed_df,
        task_name=spec.task_name,
        task_kind=spec.task_kind,
    )
    processed_df, target_audit = append_target_index_columns(processed_df, protein_index=meta["protein_index"])
    processed_df = add_index_columns(processed_df, meta=meta)
    processed_df["processed_row_index"] = np.arange(len(processed_df), dtype=np.int32)
    processed_df["expression_row_index"] = processed_df["processed_row_index"]
    processed_df["feature_membership"] = processed_df["feature_membership"].fillna("primary")
    ensure_unique_sample_ids(processed_df, task_name=spec.task_name, frame_name="processed")

    processed_source_tasks = unique_preserve_order(processed_df["source_task"].astype(str).tolist())
    processed_ordered_protein_indices, processed_ordered_uniprot, processed_source_counts = collect_ordered_protein_indices(
        processed_source_tasks,
        cache=cache,
        meta=meta,
    )
    processed_matrix = build_aligned_expression_matrix(
        processed_df,
        ordered_protein_indices=processed_ordered_protein_indices,
        cache=cache,
        meta=meta,
    )

    feature_df = processed_df.copy()
    auxiliary_label_audit: dict[str, Any] = {"applied": False}
    if spec.merge_main_single_into_feature:
        merged_single = processed_registry["ptv3_main_singledrug"].copy()
        merged_single["feature_membership"] = "merged_single_drug"
        feature_df = pd.concat([feature_df, merged_single], ignore_index=True, sort=False)
    feature_df = add_common_columns(feature_df, task_context=spec.task_name)
    feature_df, auxiliary_label_audit = sanitize_double_drug_auxiliary_rows(feature_df, task_name=spec.task_name)
    feature_df["feature_row_index"] = np.arange(len(feature_df), dtype=np.int32)
    feature_df["expression_row_index"] = feature_df["feature_row_index"]
    ensure_unique_sample_ids(feature_df, task_name=spec.task_name, frame_name="feature")

    feature_source_tasks = unique_preserve_order(feature_df["source_task"].astype(str).tolist())
    feature_ordered_protein_indices, feature_ordered_uniprot, feature_source_counts = collect_ordered_protein_indices(
        feature_source_tasks,
        cache=cache,
        meta=meta,
    )
    feature_matrix = build_aligned_expression_matrix(
        feature_df,
        ordered_protein_indices=feature_ordered_protein_indices,
        cache=cache,
        meta=meta,
    )

    task_output_dir = ensure_dir(output_root / spec.dataset_group / "tasks" / spec.task_name)
    processed_paths = write_processed_outputs(
        output_dir=task_output_dir,
        df=processed_df,
        matrix=processed_matrix,
        ordered_protein_indices=processed_ordered_protein_indices,
        ordered_protein_uniprot=processed_ordered_uniprot,
    )
    feature_paths = write_feature_outputs(
        output_dir=task_output_dir,
        df=feature_df,
        matrix=feature_matrix,
        ordered_protein_indices=feature_ordered_protein_indices,
        ordered_protein_uniprot=feature_ordered_uniprot,
    )

    processed_registry[spec.task_name] = processed_df.copy()

    task_manifest = {
        "task_name": spec.task_name,
        "dataset_group": spec.dataset_group,
        "task_kind": spec.task_kind,
        "rows": {
            "own_rows_before_append": int(len(own_df)),
            "appended_control_rows_before_filter": int(len(appended_control_df)),
            "processed_rows": int(len(processed_df)),
            "feature_rows": int(len(feature_df)),
        },
        "controls": {
            "processed_control_rows": int(is_control_frame(processed_df).sum()),
            "feature_control_rows": int(is_control_frame(feature_df).sum()),
        },
        "filter_audit": filter_audit,
        "target_audit": target_audit,
        "single_drug_second_slot_audit": single_slot_audit,
        "auxiliary_label_audit": auxiliary_label_audit,
        "processed_sources": processed_source_tasks,
        "feature_sources": feature_source_tasks,
        "processed_source_protein_counts": processed_source_counts,
        "feature_source_protein_counts": feature_source_counts,
        "processed_matrix_shape": [int(value) for value in processed_matrix.shape],
        "feature_matrix_shape": [int(value) for value in feature_matrix.shape],
    }
    task_manifest.update(processed_paths)
    task_manifest.update(feature_paths)
    return task_manifest


def build_dataset_group(
    *,
    dataset_group: str,
    input_root: Path,
    output_root: Path,
    task_specs: list[TaskSpec],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache = Stage1Cache(input_root, dataset_group)
    meta = build_stage2_global_meta(dataset_group=dataset_group, input_root=input_root, cache=cache)
    ensure_dir(output_root / dataset_group)
    dump_json(output_root / dataset_group / "global_meta.json", meta)

    processed_registry: dict[str, pd.DataFrame] = {}
    task_manifests: dict[str, Any] = {}
    for spec in task_specs:
        if spec.dataset_group != dataset_group:
            continue
        task_manifests[spec.task_name] = build_task_outputs(
            spec=spec,
            cache=cache,
            meta=meta,
            output_root=output_root,
            processed_registry=processed_registry,
        )
        gc.collect()

    cache.release()
    return meta, task_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training-ready ProteinTalk artifacts from stage-1 outputs")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Stage-1 standardized root")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Training-ready output root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    ptv3_meta, ptv3_tasks = build_dataset_group(
        dataset_group="ptv3",
        input_root=input_root,
        output_root=output_root,
        task_specs=TASK_SPECS,
    )
    ptv1_meta, ptv1_tasks = build_dataset_group(
        dataset_group="ptv1",
        input_root=input_root,
        output_root=output_root,
        task_specs=TASK_SPECS,
    )

    audit = {
        "generated_at": iso_now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "implementation_notes": [
            "Stage-2 outputs are written to `data/training_ready/` so stage-1 standardized artifacts remain untouched.",
            "Stage-2 control rows use the explicit rule `control == sample_id` or `control == control`.",
            "Ordered protein index lists are the union of source-task proteins, sorted by dataset-level global protein index order.",
            "Expression matrices are log1p transformed in-memory and retain NaN values at their original positions.",
            "`batch` is tokenized with the other batch covariates and written as `batch_index` in processed/feature tables.",
            "Single-drug rows use `pert_id2 == pert_id1` before perturbation-index encoding; `group_size` is permanently unsupported in the new no-group batch contract.",
            "Rows without native expression values are expanded to all-NaN vectors in the aligned task matrices.",
            "Extra-task matched control rows are appended before filtering, so appended controls are never removed by label-based filters.",
            "Feature tables are written as CSV plus parquet when available; pickle is used only as a fallback for the dataframe-native artifact.",
            "Each task also writes `feature_loading_manifest.json`, which records the aligned expression matrix path and the row-index contract used by dataloaders.",
            "For `ptv1_extra_singledrug`, no additional stage-2 label filter is applied after the stage-1 unique `(cell, E115_id)` selection. For other `*_extra_singledrug*` tasks, the non-control row filter uses PRISM2nd_label_total. For any `*_extra_doubledrug*` task, the non-control row filter uses PRISM1st_label_total.",
            "PTV1 is kept in its own index space and does not share global metadata with PTV3.",
        ],
        "dataset_groups": {
            "ptv3": {
                "global_meta_path": str(output_root / "ptv3" / "global_meta.json"),
                "task_names": PTV3_FINAL_TASK_ORDER,
                "protein_index_size": len(ptv3_meta["protein_index"]),
                "pert_index_size": len(ptv3_meta["pert_index"]),
            },
            "ptv1": {
                "global_meta_path": str(output_root / "ptv1" / "global_meta.json"),
                "task_names": PTV1_FINAL_TASK_ORDER,
                "protein_index_size": len(ptv1_meta["protein_index"]),
                "pert_index_size": len(ptv1_meta["pert_index"]),
            },
        },
        "tasks": {},
    }
    audit["tasks"].update(ptv3_tasks)
    audit["tasks"].update(ptv1_tasks)
    dump_json(output_root / "file_audit.json", audit)


if __name__ == "__main__":
    main()
