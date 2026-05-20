#!/usr/bin/env python3
"""Export extra-data raw CSV copies annotated with generated sample_id and control.

The original PTV3 extra single/double raw files do not carry sample ids.  Stage 1
generates deterministic sample ids, and Stage 2 filters rows before training.
This exporter joins those artifacts back to the raw row order:

- sample_id comes from ``data/standardized/ptv3/tasks/<task>/info.csv``.
- stage-2 ``processed.csv`` decides which raw rows survived filtering.
- control is rematched from original single-drug or extra-baseline sample ids.
- raw rows absent from the processed self rows receive an empty control value.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STANDARDIZED_ROOT = REPO_ROOT / "data" / "standardized"
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
DEFAULT_OUTPUT_ROOT = DEFAULT_STANDARDIZED_ROOT / "ptv3" / "extra_raw_with_controls"

PTV3_EXTRA_TASKS = [
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

ALLOWED_CONTROL_SOURCE_TASKS = [
    "ptv3_main_singledrug",
    "ptv3_extra_baseline",
]

PROCESSED_AUDIT_COLUMNS = [
    "control_match_level",
    "control_match_source_task",
    "control_match_pool_kind",
    "control_match_score",
    "control_match_machine",
    "control_match_type",
    "control_match_batch",
    "control_match_plate",
]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def normalize_lookup_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return text.replace("-", "").replace("_", "")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_row_index(series: pd.Series, *, column: str, path: Path) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        examples = series.loc[missing].head(10).astype(str).tolist()
        raise ValueError(f"{path}: {column} contains non-numeric row indices: {examples}")

    rounded = np.rint(numeric.to_numpy(dtype=float))
    non_integral = np.abs(numeric.to_numpy(dtype=float) - rounded) > 1e-9
    if non_integral.any():
        examples = series.loc[non_integral].head(10).astype(str).tolist()
        raise ValueError(f"{path}: {column} contains non-integer row indices: {examples}")

    return pd.Series(rounded.astype(np.int64), index=series.index)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def stage1_info_path(standardized_root: Path, task_name: str) -> Path:
    return standardized_root / "ptv3" / "tasks" / task_name / "info.csv"


def processed_csv_path(training_ready_root: Path, task_name: str) -> Path:
    return training_ready_root / "ptv3" / "tasks" / task_name / "processed.csv"


def load_allowed_control_pool(standardized_root: Path) -> pd.DataFrame:
    pool_frames: list[pd.DataFrame] = []
    for task_name in ALLOWED_CONTROL_SOURCE_TASKS:
        path = stage1_info_path(standardized_root, task_name)
        df = read_csv(path)
        required = {"sample_id", "machineID_new", "Cell", "cell_type", "batch", "Cell_plate", "control"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")

        df = df.copy()
        df["sample_id"] = clean_string_series(df["sample_id"])
        df["control"] = clean_string_series(df["control"])
        if task_name == "ptv3_extra_baseline":
            controls = df.copy()
            controls["control_pool_kind"] = "extra_baseline"
        else:
            controls = df.loc[df["control"].eq(df["sample_id"])].copy()
            controls["control_pool_kind"] = "main_single_self_control"

        controls["control_source_task"] = task_name
        pool_frames.append(
            controls[
                [
                    "sample_id",
                    "machineID_new",
                    "Cell",
                    "cell_type",
                    "batch",
                    "Cell_plate",
                    "control_source_task",
                    "control_pool_kind",
                ]
            ]
        )

    control_pool = pd.concat(pool_frames, ignore_index=True, sort=False)
    if control_pool["sample_id"].duplicated(keep=False).any():
        examples = control_pool.loc[control_pool["sample_id"].duplicated(keep=False), "sample_id"].head(10).tolist()
        raise ValueError(f"duplicate allowed control sample_ids: {examples}")
    return control_pool


def label_match_level(row: pd.Series) -> str:
    if not str(row.get("sample_id_ctrl", "") or "").strip():
        return "no_cell_match"
    for label, field in (
        ("machine", "match_machine"),
        ("type", "match_type"),
        ("batch", "match_batch"),
        ("plate", "match_plate"),
    ):
        if bool(row[field]):
            return label
    return "cell_only"


def match_allowed_controls(control_pool: pd.DataFrame, query_df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=query_df.index)
    for column in ["control", *PROCESSED_AUDIT_COLUMNS]:
        result[column] = ""
    if query_df.empty:
        return result

    control = control_pool.copy()
    query = query_df.copy().reset_index(drop=False).rename(columns={"index": "__row_index"})
    for frame in (control, query):
        for column in ("machineID_new", "Cell", "cell_type", "batch", "Cell_plate"):
            frame[f"__norm_{column}"] = frame[column].map(normalize_lookup_text)

    merged = query.merge(control, on="__norm_Cell", how="left", suffixes=("", "_ctrl"))
    merged["sample_id_ctrl"] = clean_string_series(merged.get("sample_id_ctrl", pd.Series([""] * len(merged))))
    merged["match_machine"] = merged["__norm_machineID_new"].eq(merged["__norm_machineID_new_ctrl"])
    merged["match_type"] = merged["__norm_cell_type"].eq(merged["__norm_cell_type_ctrl"])
    merged["match_batch"] = merged["__norm_batch"].eq(merged["__norm_batch_ctrl"])
    merged["match_plate"] = merged["__norm_Cell_plate"].eq(merged["__norm_Cell_plate_ctrl"])

    merged["control_match_score"] = (
        merged["match_machine"].astype(bool).astype(int) * 8
        + merged["match_type"].astype(bool).astype(int) * 4
        + merged["match_batch"].astype(bool).astype(int) * 2
        + merged["match_plate"].astype(bool).astype(int)
    )
    merged.loc[merged["sample_id_ctrl"].eq(""), "control_match_score"] = 0
    merged["control_match_level"] = merged.apply(label_match_level, axis=1)
    merged.sort_values(
        by=[
            "__row_index",
            "match_machine",
            "match_type",
            "match_batch",
            "match_plate",
            "sample_id_ctrl",
        ],
        ascending=[True, False, False, False, False, True],
        inplace=True,
    )
    best = merged.drop_duplicates(subset="__row_index", keep="first").set_index("__row_index")

    result["control"] = result.index.map(best["sample_id_ctrl"].fillna(""))
    result["control_match_level"] = result.index.map(best["control_match_level"].fillna("no_cell_match"))
    result["control_match_source_task"] = result.index.map(best["control_source_task"].fillna(""))
    result["control_match_pool_kind"] = result.index.map(best["control_pool_kind"].fillna(""))
    result["control_match_score"] = result.index.map(best["control_match_score"].fillna(0)).astype(int)
    result["control_match_machine"] = result.index.map(best["match_machine"].fillna(False)).astype(bool)
    result["control_match_type"] = result.index.map(best["match_type"].fillna(False)).astype(bool)
    result["control_match_batch"] = result.index.map(best["match_batch"].fillna(False)).astype(bool)
    result["control_match_plate"] = result.index.map(best["match_plate"].fillna(False)).astype(bool)
    no_match = result["control"].astype("string").fillna("").str.strip().eq("")
    result.loc[no_match, ["control_match_source_task", "control_match_pool_kind"]] = ""
    return result


def single_source_file(stage1_df: pd.DataFrame, *, task_name: str, path: Path) -> str:
    if "source_file_info" not in stage1_df.columns:
        raise ValueError(f"{path}: missing source_file_info")
    source_files = [
        value
        for value in clean_string_series(stage1_df["source_file_info"]).drop_duplicates().tolist()
        if value
    ]
    if len(source_files) != 1:
        raise ValueError(f"{task_name}: expected exactly one source_file_info, found {source_files}")
    return source_files[0]


def load_stage1_info(standardized_root: Path, task_name: str) -> tuple[pd.DataFrame, str]:
    path = stage1_info_path(standardized_root, task_name)
    df = read_csv(path)
    required = {"sample_id", "source_row_index", "source_file_info"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    df = df.copy()
    df["__source_row_index_int"] = normalize_row_index(df["source_row_index"], column="source_row_index", path=path)
    duplicated = df["__source_row_index_int"].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "__source_row_index_int"].head(10).astype(int).tolist()
        raise ValueError(f"{path}: duplicate source_row_index values: {examples}")
    return df, single_source_file(df, task_name=task_name, path=path)


def load_processed_self_rows(training_ready_root: Path, task_name: str) -> pd.DataFrame:
    path = processed_csv_path(training_ready_root, task_name)
    df = read_csv(path)
    required = {"sample_id", "control", "source_row_index", "source_task", "source_row_role"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    source_task = clean_string_series(df["source_task"])
    source_role = clean_string_series(df["source_row_role"])
    self_rows = df.loc[source_task.eq(task_name) & source_role.eq("self")].copy()
    self_rows["__source_row_index_int"] = normalize_row_index(
        self_rows["source_row_index"],
        column="source_row_index",
        path=path,
    )
    duplicated = self_rows["__source_row_index_int"].duplicated(keep=False)
    if duplicated.any():
        examples = self_rows.loc[duplicated, "__source_row_index_int"].head(10).astype(int).tolist()
        raise ValueError(f"{path}: duplicate processed self source_row_index values: {examples}")
    return self_rows


def stringify_mapping(frame: pd.DataFrame, index_column: str, value_column: str) -> dict[int, str]:
    if value_column not in frame.columns:
        return {}
    subset = frame[[index_column, value_column]].copy()
    subset[value_column] = clean_string_series(subset[value_column])
    return dict(zip(subset[index_column].astype(int).tolist(), subset[value_column].tolist()))


def rename_conflicting_raw_columns(raw: pd.DataFrame, annotation_columns: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    renamed = raw.copy()
    rename_map: dict[str, str] = {}
    existing = set(renamed.columns)
    for column in annotation_columns:
        if column not in existing:
            continue
        candidate = f"raw_{column}"
        suffix = 2
        while candidate in existing or candidate in annotation_columns:
            candidate = f"raw_{column}_{suffix}"
            suffix += 1
        rename_map[column] = candidate
        existing.remove(column)
        existing.add(candidate)
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed, rename_map


def build_task_export(
    *,
    task_name: str,
    standardized_root: Path,
    training_ready_root: Path,
    output_root: Path,
    control_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage1_df, source_file = load_stage1_info(standardized_root, task_name)
    processed_self = load_processed_self_rows(training_ready_root, task_name)

    source_path = Path(source_file)
    raw_path = source_path if source_path.is_absolute() else REPO_ROOT / source_path
    raw = read_csv(raw_path)
    raw_indices = pd.Index(range(len(raw)))

    stage1_by_row = stage1_df.set_index("__source_row_index_int", drop=False)
    processed_by_row = processed_self.set_index("__source_row_index_int", drop=False)

    sample_id_map = stringify_mapping(stage1_by_row, "__source_row_index_int", "sample_id")
    kept_indices = raw_indices.intersection(processed_by_row.index)
    matched_controls = match_allowed_controls(control_pool, stage1_by_row.loc[kept_indices].copy())

    annotation = pd.DataFrame(index=raw_indices)
    annotation["sample_id"] = [sample_id_map.get(int(idx), "") for idx in raw_indices]
    annotation["control"] = ""
    annotation["source_row_index"] = raw_indices.astype(int)
    annotation["kept_after_filter"] = raw_indices.isin(processed_by_row.index).tolist()
    annotation["stage2_filter_status"] = np.where(
        annotation["kept_after_filter"],
        "kept_after_stage2_filter",
        "filtered_out_before_stage2_processed",
    )
    annotation["control_export_status"] = np.where(
        annotation["kept_after_filter"],
        "pending_allowed_control_match",
        "filtered_out_blank_control",
    )

    for column in PROCESSED_AUDIT_COLUMNS:
        annotation[column] = ""
    for column in ["control", *PROCESSED_AUDIT_COLUMNS]:
        annotation.loc[matched_controls.index, column] = matched_controls[column].tolist()
    matched_control_mask = annotation["control"].astype("string").fillna("").str.strip().ne("")
    annotation.loc[
        annotation["kept_after_filter"] & matched_control_mask,
        "control_export_status",
    ] = "matched_allowed_control"
    annotation.loc[
        annotation["kept_after_filter"] & ~matched_control_mask,
        "control_export_status",
    ] = "kept_but_no_allowed_control_match"

    raw_for_output, renamed_raw_columns = rename_conflicting_raw_columns(raw, annotation.columns.tolist())
    annotated_raw = pd.concat([annotation.reset_index(drop=True), raw_for_output.reset_index(drop=True)], axis=1)

    output_name = f"{raw_path.stem}_with_sample_id_control{raw_path.suffix}"
    output_path = output_root / output_name
    annotated_raw.to_csv(output_path, index=False)

    missing_stage1_indices = sorted(set(raw_indices.astype(int).tolist()) - set(stage1_by_row.index.astype(int).tolist()))
    processed_indices = set(processed_by_row.index.astype(int).tolist())
    control_filled = annotation["control"].astype("string").fillna("").str.strip().ne("")

    manifest = {
        "task_name": task_name,
        "source_file": source_file,
        "output_file": display_path(output_path),
        "raw_rows": int(len(raw)),
        "stage1_rows": int(len(stage1_df)),
        "processed_self_rows": int(len(processed_self)),
        "rows_with_sample_id": int(annotation["sample_id"].astype("string").fillna("").str.strip().ne("").sum()),
        "rows_kept_after_filter": int(annotation["kept_after_filter"].sum()),
        "rows_blank_control_due_to_filter": int((~annotation["kept_after_filter"]).sum()),
        "rows_with_control": int(control_filled.sum()),
        "rows_kept_without_control": int((annotation["kept_after_filter"] & ~control_filled).sum()),
        "stage2_filter_status_counts": {
            str(key): int(value)
            for key, value in annotation["stage2_filter_status"].value_counts(dropna=False).sort_index().items()
        },
        "control_export_status_counts": {
            str(key): int(value)
            for key, value in annotation["control_export_status"].value_counts(dropna=False).sort_index().items()
        },
        "allowed_control_source_tasks": ALLOWED_CONTROL_SOURCE_TASKS,
        "missing_stage1_source_row_indices": missing_stage1_indices[:50],
        "missing_stage1_source_row_count": int(len(missing_stage1_indices)),
        "processed_source_row_indices_outside_raw": sorted(processed_indices - set(raw_indices.astype(int).tolist()))[:50],
        "renamed_raw_columns": renamed_raw_columns,
    }
    return annotated_raw, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write PTV3 extra single/double raw CSV copies with generated sample_id "
            "and post-filter control columns."
        )
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=DEFAULT_STANDARDIZED_ROOT,
        help="Stage-1 standardized root containing ptv3/tasks/<task>/info.csv.",
    )
    parser.add_argument(
        "--training-ready-root",
        type=Path,
        default=DEFAULT_TRAINING_READY_ROOT,
        help="Stage-2 training-ready root containing ptv3/tasks/<task>/processed.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where annotated raw CSVs and mapping files are written.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=PTV3_EXTRA_TASKS,
        help="Task to export. Repeat to export multiple tasks. Defaults to all PTV3 extra single/double tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = args.task or PTV3_EXTRA_TASKS
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    control_pool = load_allowed_control_pool(args.standardized_root)

    manifests: list[dict[str, Any]] = []
    mapping_frames: list[pd.DataFrame] = []
    for task_name in tasks:
        annotated_raw, manifest = build_task_export(
            task_name=task_name,
            standardized_root=args.standardized_root,
            training_ready_root=args.training_ready_root,
            output_root=output_root,
            control_pool=control_pool,
        )
        manifests.append(manifest)

        mapping_columns = [
            "sample_id",
            "control",
            "source_row_index",
            "kept_after_filter",
            "stage2_filter_status",
            "control_export_status",
            *PROCESSED_AUDIT_COLUMNS,
        ]
        mapping = annotated_raw[mapping_columns].copy()
        mapping.insert(0, "task_name", task_name)
        mapping.insert(1, "source_file", manifest["source_file"])
        mapping_frames.append(mapping)

    mapping_df = pd.concat(mapping_frames, ignore_index=True, sort=False)
    mapping_path = output_root / "extra_sample_id_control_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)

    nonempty_path = output_root / "extra_sample_id_control_mapping_nonempty.csv"
    nonempty = mapping_df.loc[mapping_df["control"].astype("string").fillna("").str.strip().ne("")]
    nonempty.to_csv(nonempty_path, index=False)

    manifest = {
        "created_at": iso_now(),
        "standardized_root": str(args.standardized_root),
        "training_ready_root": str(args.training_ready_root),
        "output_root": str(output_root),
        "control_rule": (
            "Stage-2 processed self rows decide which raw rows are kept. Control is rematched "
            "only from original ptv3_main_singledrug self-controls and ptv3_extra_baseline "
            "sample ids. Raw rows filtered out before processed.csv keep their generated "
            "sample_id but receive blank control."
        ),
        "allowed_control_source_tasks": ALLOWED_CONTROL_SOURCE_TASKS,
        "mapping_file": display_path(mapping_path),
        "nonempty_mapping_file": display_path(nonempty_path),
        "tasks": manifests,
    }
    manifest_path = output_root / "export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(manifests)} annotated raw files to {output_root}")
    print(f"Wrote mapping file: {mapping_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
