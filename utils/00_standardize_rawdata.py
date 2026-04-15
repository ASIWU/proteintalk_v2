#!/usr/bin/env python3
"""Standardize ProteinTalk raw data into reproducible task-level artifacts.

This script follows `data/Data_Process.md` and produces:
1. standardized task-level info CSV files
2. task-level expression matrices / protein orders
3. global meta json files
4. file-level and task-level audit manifests

Notes
-----
- `ptv1` is processed into its own isolated output space and meta index.
- `extra_*` validation files in this checkout do not contain perturbation proteome
  matrices. For those tasks this script emits empty expression structures and
  records the missing-expression reason in the audit manifest.
- For large matrices, materializing a duplicated `sample_id -> vector` pickle is
  prohibitively expensive. A deterministic `sample_id -> row_index` map is always
  emitted. A full expression dict is only materialized for smaller tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "rawdata"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "standardized"

STANDARD_INFO_COLUMNS = [
    "sample_id",
    "machineID_new",
    "Cell_plate",
    "Cell",
    "cell_type",
    "pert_id1",
    "pert_id2",
    "batch",
    "pert_time",
    "pert_dose1",
    "pert_dose2",
    "PRISM1st_label_total",
    "PRISM2nd_label_total",
    "instrument",
    "cell_pertid_time",
    "drugname",
    "smiles",
    "target_protein_list",
    "control",
    "synergy",
]

REAL_EXPRESSION_TASKS = {
    "ptv3_main_singledrug",
    "ptv3_main_doubledrug",
    "ptv3_extra_baseline",
    "ptv1_aivc",
}

FULL_DICT_VALUE_LIMIT = 40_000_000

PTV3_TASK_ORDER = [
    "ptv3_main_singledrug",
    "ptv3_main_doubledrug",
    "ptv3_extra_baseline",
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


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_free_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_lookup_text(value: object) -> str:
    text = normalize_free_text(value).upper()
    text = text.replace("-", "").replace("_", "")
    return text


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_free_text(value).lower())


def safe_read_csv(
    path: Path,
    *,
    encodings: Iterable[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    if encodings is None:
        encodings = ("utf-8", "utf-8-sig", "latin1", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is None:
        last_error = RuntimeError(f"failed to read {path}")
    raise last_error


def clean_nullable_string(series: pd.Series) -> pd.Series:
    series = series.astype("string")
    series = series.str.strip()
    return series.fillna("")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def json_list_string(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def default_standard_frame(sample_ids: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"sample_id": sample_ids.astype(str)})
    for column in STANDARD_INFO_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def choose_first_non_empty(*values: object) -> str:
    for value in values:
        text = normalize_free_text(value)
        if text:
            return text
    return ""


def parse_uniprot_token(token: str) -> bool:
    token = token.strip()
    return bool(
        re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", token)
        or re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{3}[0-9]", token)
        or re.fullmatch(r"[A-NR-Z][0-9](?:[A-Z0-9]{3}[0-9]){2}", token)
        or re.fullmatch(r"A0A[A-Z0-9]{7}", token)
    )


def parse_main_uniprot(column_name: str) -> str | None:
    if "_" not in column_name:
        return None
    token = column_name.split("_", 1)[0]
    return token if token else None


def parse_ptv1_uniprot(column_name: str) -> str | None:
    for token in column_name.split("."):
        if parse_uniprot_token(token):
            return token
    return None


def parse_target_list(value: object) -> list[str]:
    raw = normalize_free_text(value)
    if not raw or raw.upper() == "NA":
        return []
    tokens = [token.strip() for token in re.split(r"[;,|]+", raw) if token.strip()]
    return [token for token in tokens if parse_uniprot_token(token)]


def parse_smiles_to_existing_maps(single_info: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    smiles_map_sets: dict[str, set[str]] = defaultdict(set)
    for smiles_col in ("smiles", "Smiles_no_chiral", "Smiles_with_chiral"):
        if smiles_col not in single_info.columns:
            continue
        subset = single_info[["pert_id", smiles_col]].dropna().astype(str)
        for pert_id, smiles in subset.itertuples(index=False):
            smiles = smiles.strip()
            if smiles:
                smiles_map_sets[smiles].add(pert_id.strip())

    name_map_sets: dict[str, set[str]] = defaultdict(set)
    for _, row in single_info[["pert_id", "drugname", "synonyms"]].dropna(subset=["pert_id"]).iterrows():
        pert_id = normalize_free_text(row["pert_id"])
        if not pert_id:
            continue
        for raw_name in (row.get("drugname"), row.get("synonyms")):
            if pd.isna(raw_name):
                continue
            for token in re.split(r"[;|,]+", str(raw_name)):
                normalized = normalize_name(token)
                if normalized:
                    name_map_sets[normalized].add(pert_id)

    ambiguous_smiles = {key: sorted(values) for key, values in smiles_map_sets.items() if len(values) > 1}
    return (
        {key: next(iter(values)) for key, values in smiles_map_sets.items() if len(values) == 1},
        {key: next(iter(values)) for key, values in name_map_sets.items() if len(values) == 1},
        ambiguous_smiles,
    )


def namespaced_id(namespace: str, raw_value: object) -> str:
    raw = normalize_free_text(raw_value)
    slug = normalize_name(raw)
    if slug:
        return f"{namespace}::{slug}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{namespace}::{digest}"


def make_generated_sample_ids(task_name: str, size: int) -> list[str]:
    return [f"{task_name}__{idx:06d}" for idx in range(size)]


def ensure_standard_column_order(df: pd.DataFrame) -> pd.DataFrame:
    leading = [column for column in STANDARD_INFO_COLUMNS if column in df.columns]
    trailing = [column for column in df.columns if column not in leading]
    return df[leading + trailing]


def jsonize_target_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda values: json_list_string(values if isinstance(values, list) else []))
    return df


def validate_unique_sample_ids(df: pd.DataFrame, task_name: str) -> None:
    duplicated = df["sample_id"].duplicated(keep=False)
    if duplicated.any():
        sample_ids = df.loc[duplicated, "sample_id"].astype(str).tolist()[:10]
        raise ValueError(f"{task_name} has duplicated sample_id values: {sample_ids}")


def normalize_control_column(series: pd.Series, sample_ids: set[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw = clean_nullable_string(series)
    normalized = []
    status = []
    is_control = []
    for value in raw.tolist():
        if not value:
            normalized.append("")
            status.append("missing")
            is_control.append(False)
            continue
        if value in sample_ids:
            normalized.append(value)
            status.append("ok")
            is_control.append(False)
            continue
        if value.lower() == "control":
            normalized.append("")
            status.append("literal_control_without_sample_id")
            is_control.append(True)
            continue
        normalized.append("")
        status.append("unresolved_non_sample_id")
        is_control.append(False)
    return (
        pd.Series(normalized, dtype="string").fillna(""),
        pd.Series(status, dtype="string").fillna(""),
        pd.Series(is_control, dtype="boolean"),
    )


def extract_ptv3_single_controls(single_df: pd.DataFrame) -> pd.DataFrame:
    mask = single_df["control"] == single_df["sample_id"]
    controls = single_df.loc[mask].copy()
    controls["control_source_task"] = "ptv3_main_singledrug"
    controls["control_pool_kind"] = "main_single_self_control"
    return controls


def build_control_pool(single_df: pd.DataFrame, extra_baseline_df: pd.DataFrame) -> pd.DataFrame:
    single_controls = extract_ptv3_single_controls(single_df)
    baseline_controls = extra_baseline_df.copy()
    baseline_controls["control"] = baseline_controls["sample_id"]
    baseline_controls["control_source_task"] = "ptv3_extra_baseline"
    baseline_controls["control_pool_kind"] = "extra_baseline"
    pool_columns = [
        "sample_id",
        "machineID_new",
        "Cell",
        "cell_type",
        "batch",
        "Cell_plate",
        "control_source_task",
        "control_pool_kind",
    ]
    return pd.concat(
        [single_controls[pool_columns], baseline_controls[pool_columns]],
        ignore_index=True,
    )


def match_controls(control_pool: pd.DataFrame, query_df: pd.DataFrame) -> pd.DataFrame:
    control = control_pool.copy()
    query = query_df.copy().reset_index(drop=False).rename(columns={"index": "__row_index"})
    for frame in (control, query):
        for column in ("machineID_new", "Cell", "cell_type", "batch", "Cell_plate"):
            frame[f"__norm_{column}"] = frame[column].map(normalize_lookup_text)

    merged = query.merge(
        control,
        on="__norm_Cell",
        how="left",
        suffixes=("", "_ctrl"),
    )
    if merged.empty:
        result = query_df.copy()
        result["control"] = ""
        result["control_match_level"] = "no_cell_match"
        result["control_match_source_task"] = ""
        result["control_match_pool_kind"] = ""
        result["control_match_score"] = 0
        return result

    merged["match_machine"] = merged["__norm_machineID_new"] == merged["__norm_machineID_new_ctrl"]
    merged["match_type"] = merged["__norm_cell_type"] == merged["__norm_cell_type_ctrl"]
    merged["match_batch"] = merged["__norm_batch"] == merged["__norm_batch_ctrl"]
    merged["match_plate"] = merged["__norm_Cell_plate"] == merged["__norm_Cell_plate_ctrl"]
    merged["control_match_score"] = (
        merged["match_machine"].astype(int) * 8
        + merged["match_type"].astype(int) * 4
        + merged["match_batch"].astype(int) * 2
        + merged["match_plate"].astype(int)
    )

    def label_match_level(row: pd.Series) -> str:
        if not normalize_free_text(row.get("sample_id_ctrl")):
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
    best = merged.drop_duplicates(subset="__row_index", keep="first")
    mapping = best.set_index("__row_index")

    result = query_df.copy()
    result["control"] = result.index.map(mapping["sample_id_ctrl"].fillna(""))
    result["control_match_level"] = result.index.map(mapping["control_match_level"].fillna("no_cell_match"))
    result["control_match_source_task"] = result.index.map(mapping["control_source_task"].fillna(""))
    result["control_match_pool_kind"] = result.index.map(mapping["control_pool_kind"].fillna(""))
    result["control_match_score"] = result.index.map(mapping["control_match_score"].fillna(0)).astype(int)
    result["control_match_machine"] = result.index.map(mapping["match_machine"].fillna(False)).astype(bool)
    result["control_match_type"] = result.index.map(mapping["match_type"].fillna(False)).astype(bool)
    result["control_match_batch"] = result.index.map(mapping["match_batch"].fillna(False)).astype(bool)
    result["control_match_plate"] = result.index.map(mapping["match_plate"].fillna(False)).astype(bool)
    return result


def build_expression_descriptor(
    *,
    task_name: str,
    sample_ids: list[str],
    protein_order: list[str],
    output_dir: Path,
    matrix_available: bool,
    matrix_reason: str = "",
    full_dict_materialized: bool = False,
) -> dict[str, object]:
    return {
        "task_name": task_name,
        "matrix_available": matrix_available,
        "missing_reason": matrix_reason,
        "sample_count": len(sample_ids),
        "protein_count": len(protein_order),
        "matrix_path": str(output_dir / "expression_matrix.npy"),
        "protein_order_path": str(output_dir / "protein_order.json"),
        "sample_ids_path": str(output_dir / "sample_ids.json"),
        "sample_id_to_row_index_path": str(output_dir / "sample_id_to_row_index.json"),
        "expression_dict_path": str(output_dir / "expression_dict.pkl") if full_dict_materialized else "",
        "expression_dict_materialized": full_dict_materialized,
    }


def write_empty_expression_outputs(task_name: str, info_df: pd.DataFrame, output_dir: Path, reason: str) -> dict[str, object]:
    sample_ids = info_df["sample_id"].astype(str).tolist()
    matrix = np.empty((len(sample_ids), 0), dtype=np.float32)
    np.save(output_dir / "expression_matrix.npy", matrix)
    dump_json(output_dir / "protein_order.json", [])
    dump_json(output_dir / "sample_ids.json", sample_ids)
    dump_json(
        output_dir / "sample_id_to_row_index.json",
        {sample_id: idx for idx, sample_id in enumerate(sample_ids)},
    )
    with (output_dir / "expression_dict.pkl").open("wb") as handle:
        pickle.dump({sample_id: np.empty((0,), dtype=np.float32) for sample_id in sample_ids}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return build_expression_descriptor(
        task_name=task_name,
        sample_ids=sample_ids,
        protein_order=[],
        output_dir=output_dir,
        matrix_available=False,
        matrix_reason=reason,
        full_dict_materialized=True,
    )


def write_expression_outputs(
    *,
    task_name: str,
    expr_path: Path,
    sample_id_col: str,
    protein_columns: list[str],
    protein_order: list[str],
    info_df: pd.DataFrame,
    output_dir: Path,
    encodings: Iterable[str] | None = None,
    chunksize: int = 256,
) -> dict[str, object]:
    sample_ids = info_df["sample_id"].astype(str).tolist()
    row_index_map = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    matrix_path = output_dir / "expression_matrix.npy"
    temp_matrix_path = Path(tempfile.gettempdir()) / f"{task_name}__expression_matrix.npy"
    if temp_matrix_path.exists():
        temp_matrix_path.unlink()
    matrix = np.lib.format.open_memmap(
        temp_matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(sample_ids), len(protein_columns)),
    )
    matrix[:] = np.nan

    seen_sample_ids: set[str] = set()
    usecols = [sample_id_col] + protein_columns
    if encodings is None:
        encodings = ("utf-8", "utf-8-sig", "latin1", "gb18030")
    reader = None
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            reader = pd.read_csv(
                expr_path,
                encoding=encoding,
                usecols=usecols,
                chunksize=chunksize,
                low_memory=False,
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if reader is None:
        raise last_error if last_error is not None else RuntimeError(f"failed to read {expr_path}")

    for chunk in reader:
        chunk = chunk.copy()
        chunk[sample_id_col] = chunk[sample_id_col].astype(str)
        chunk = chunk[chunk[sample_id_col].isin(row_index_map)]
        if chunk.empty:
            continue
        rows = [row_index_map[sample_id] for sample_id in chunk[sample_id_col]]
        matrix[np.asarray(rows, dtype=np.int64)] = chunk[protein_columns].to_numpy(dtype=np.float32, copy=True)
        seen_sample_ids.update(chunk[sample_id_col].tolist())

    del matrix
    shutil.move(str(temp_matrix_path), str(matrix_path))

    missing_sample_ids = [sample_id for sample_id in sample_ids if sample_id not in seen_sample_ids]
    dump_json(output_dir / "protein_order.json", protein_order)
    dump_json(output_dir / "sample_ids.json", sample_ids)
    dump_json(output_dir / "sample_id_to_row_index.json", row_index_map)

    full_dict_materialized = len(sample_ids) * max(len(protein_order), 1) <= FULL_DICT_VALUE_LIMIT
    if full_dict_materialized:
        loaded = np.load(matrix_path)
        expr_dict = {sample_id: np.asarray(loaded[idx], dtype=np.float32) for idx, sample_id in enumerate(sample_ids)}
        with (output_dir / "expression_dict.pkl").open("wb") as handle:
            pickle.dump(expr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    descriptor = build_expression_descriptor(
        task_name=task_name,
        sample_ids=sample_ids,
        protein_order=protein_order,
        output_dir=output_dir,
        matrix_available=True,
        full_dict_materialized=full_dict_materialized,
    )
    descriptor["missing_expression_rows"] = missing_sample_ids
    return descriptor


def merged_smiles_for_double(smiles1: str, smiles2: str) -> str:
    if smiles1 and smiles2:
        return f"{smiles1} || {smiles2}"
    return choose_first_non_empty(smiles1, smiles2)


def merged_targets_for_double(targets1: list[str], targets2: list[str]) -> list[str]:
    merged = []
    for token in targets1 + targets2:
        if token not in merged:
            merged.append(token)
    return merged


@dataclass
class TaskResult:
    task_name: str
    dataset_group: str
    info_path: str
    expression: dict[str, object]
    sample_count: int
    protein_count: int
    pert_ids: list[str]
    protein_order: list[str]
    pert_smiles_map: dict[str, str]
    pert_target_map: dict[str, list[str]]
    pert_target_text_map: dict[str, str]
    audit: dict[str, object] = field(default_factory=dict)


def standardize_main_singledrug(task_dir: Path) -> TaskResult:
    info_path = RAW_ROOT / "singledrug" / "20260403_ptv3_v2_bind_bio_sampleID_machineID_details.csv"
    expr_path = RAW_ROOT / "singledrug" / "20250113_ptv3_unique_mat_28602samp_10982prot_finall_v2.csv"

    info_raw = safe_read_csv(info_path, low_memory=False)
    expr_ids = pd.read_csv(expr_path, usecols=["samp_ID"], low_memory=False)["samp_ID"].astype(str)
    sample_ids = set(expr_ids.tolist())

    info_raw["sample_id"] = info_raw["sample_id"].astype(str)
    validate_unique_sample_ids(info_raw[["sample_id"]].copy(), "ptv3_main_singledrug_raw")

    control_series, control_status, _ = normalize_control_column(info_raw["control"], sample_ids)

    standard = default_standard_frame(info_raw["sample_id"])
    standard["machineID_new"] = clean_nullable_string(info_raw["machineID_new"])
    standard["Cell_plate"] = clean_nullable_string(info_raw["Cell_plate"])
    standard["Cell"] = clean_nullable_string(info_raw["Cell"])
    standard["cell_type"] = clean_nullable_string(info_raw["cell_type"])
    standard["pert_id1"] = clean_nullable_string(info_raw["pert_id"].astype("string"))
    standard["pert_id2"] = ""
    standard["batch"] = clean_nullable_string(info_raw["batch"])
    standard["pert_time"] = clean_numeric(info_raw["pert_time"])
    standard["pert_dose1"] = clean_numeric(info_raw["pert_dose"])
    standard["pert_dose2"] = np.nan
    standard["PRISM1st_label_total"] = clean_nullable_string(info_raw["PRISM1st_label_total"])
    standard["PRISM2nd_label_total"] = ""
    standard["instrument"] = clean_nullable_string(info_raw["instrument"])
    standard["cell_pertid_time"] = clean_nullable_string(info_raw["cell_pertid_time"])
    standard["drugname"] = clean_nullable_string(info_raw["drugname"])
    standard["smiles"] = info_raw.apply(
        lambda row: choose_first_non_empty(row.get("Smiles_with_chiral"), row.get("smiles"), row.get("Smiles_no_chiral")),
        axis=1,
    )
    target_lists = info_raw["targetv2"].map(parse_target_list)
    standard["target_protein_list"] = target_lists
    standard["control"] = control_series
    standard["synergy"] = np.nan

    standard["control_raw"] = clean_nullable_string(info_raw["control"])
    standard["control_status"] = control_status
    standard["targetv2_raw"] = clean_nullable_string(info_raw["targetv2"])
    standard["targetMatchv2_raw"] = clean_nullable_string(info_raw["targetMatchv2"])
    standard["source_file_info"] = str(info_path.relative_to(REPO_ROOT))
    standard["source_file_expression"] = str(expr_path.relative_to(REPO_ROOT))
    standard["expression_available"] = True

    standard = jsonize_target_columns(standard, ("target_protein_list",))
    standard = ensure_standard_column_order(standard)
    validate_unique_sample_ids(standard, "ptv3_main_singledrug")

    info_out = task_dir / "info.csv"
    standard.to_csv(info_out, index=False)

    header = pd.read_csv(expr_path, nrows=0).columns.tolist()
    protein_columns = [column for column in header if column != "samp_ID"]
    protein_order = [parse_main_uniprot(column) for column in protein_columns]
    expression = write_expression_outputs(
        task_name="ptv3_main_singledrug",
        expr_path=expr_path,
        sample_id_col="samp_ID",
        protein_columns=protein_columns,
        protein_order=[protein for protein in protein_order if protein is not None],
        info_df=standard,
        output_dir=task_dir,
        encodings=("utf-8",),
    )

    pert_smiles_map: dict[str, str] = {}
    pert_target_map: dict[str, list[str]] = {}
    pert_target_text_map: dict[str, str] = {}
    for row in info_raw.itertuples(index=False):
        pert_id = normalize_free_text(getattr(row, "pert_id"))
        if not pert_id:
            continue
        pert_smiles_map.setdefault(
            pert_id,
            choose_first_non_empty(getattr(row, "Smiles_with_chiral"), getattr(row, "smiles"), getattr(row, "Smiles_no_chiral")),
        )
        targets = parse_target_list(getattr(row, "targetv2"))
        if pert_id not in pert_target_map or (not pert_target_map[pert_id] and targets):
            pert_target_map[pert_id] = targets
        raw_target = normalize_free_text(getattr(row, "targetv2"))
        if raw_target:
            pert_target_text_map.setdefault(pert_id, raw_target)

    audit = {
        "raw_files": [
            str(info_path.relative_to(REPO_ROOT)),
            str(expr_path.relative_to(REPO_ROOT)),
        ],
        "category": "ptv3_singledrug",
        "table_kinds": {
            str(info_path.relative_to(REPO_ROOT)): "sample_info_table",
            str(expr_path.relative_to(REPO_ROOT)): "expression_table",
        },
        "column_mapping": {
            "sample_id": "sample_id",
            "machineID_new": "machineID_new",
            "Cell_plate": "Cell_plate",
            "Cell": "Cell",
            "cell_type": "cell_type",
            "pert_id1": "pert_id",
            "pert_id2": "filled_empty_for_single_drug",
            "pert_time": "pert_time",
            "pert_dose1": "pert_dose",
            "instrument": "instrument",
            "cell_pertid_time": "cell_pertid_time",
            "drugname": "drugname",
            "smiles": "Smiles_with_chiral > smiles > Smiles_no_chiral",
            "target_protein_list": "targetv2",
            "control": "control (only valid sample_id values kept)",
        },
        "special_rules": [
            "latin1 fallback is required for the sample info csv",
            "control values that are not valid sample_id values are blanked in the standardized control column and preserved in control_raw",
        ],
        "issues": [
            {
                "kind": "unresolved_control_reference",
                "count": int((standard["control_status"] == "unresolved_non_sample_id").sum()),
            }
        ],
    }

    return TaskResult(
        task_name="ptv3_main_singledrug",
        dataset_group="ptv3",
        info_path=str(info_out),
        expression=expression,
        sample_count=len(standard),
        protein_count=len(load_json(task_dir / "protein_order.json")),
        pert_ids=sorted({pert_id for pert_id in standard["pert_id1"].astype(str).tolist() if pert_id}),
        protein_order=load_json(task_dir / "protein_order.json"),
        pert_smiles_map=pert_smiles_map,
        pert_target_map=pert_target_map,
        pert_target_text_map=pert_target_text_map,
        audit=audit,
    )


def standardize_main_doubledrug(task_dir: Path, main_single_maps: dict[str, dict[str, object]]) -> TaskResult:
    info_path = RAW_ROOT / "doubledrug" / "20260414ptv3_J_3549sampinfo_check_prism1_label_add_prism2_label_add_machineID_detail.csv"
    expr_path = RAW_ROOT / "doubledrug" / "20250211_ptv3_J_3549samp_9205prot_finall_edit.csv"

    info_raw = pd.read_csv(info_path, low_memory=False)
    expr_ids = pd.read_csv(expr_path, usecols=["samp_id"], low_memory=False)["samp_id"].astype(str)
    sample_ids = set(expr_ids.tolist())

    control_series, control_status, _ = normalize_control_column(info_raw["control"], sample_ids)

    smiles_map = main_single_maps["pert_smiles_map"]
    target_map = main_single_maps["pert_target_map"]

    standard = default_standard_frame(info_raw["sample_id"].astype(str))
    standard["machineID_new"] = clean_nullable_string(info_raw["machineID_new"])
    standard["Cell_plate"] = clean_nullable_string(info_raw["Cell_plate"])
    standard["Cell"] = clean_nullable_string(info_raw["Cell"])
    standard["cell_type"] = clean_nullable_string(info_raw["cell_type"])
    standard["pert_id1"] = clean_nullable_string(info_raw["pert_id1"].astype("string"))
    standard["pert_id2"] = clean_nullable_string(info_raw["pert_id2"].astype("string"))
    standard["batch"] = clean_nullable_string(info_raw["batch"])
    standard["pert_time"] = clean_numeric(info_raw["pert_time"])
    standard["pert_dose1"] = clean_numeric(info_raw["pert_dose1"])
    standard["pert_dose2"] = clean_numeric(info_raw["pert_dose2"])
    standard["PRISM1st_label_total"] = clean_nullable_string(info_raw["PRISM1st_label_total"])
    standard["PRISM2nd_label_total"] = clean_nullable_string(info_raw["PRISM2nd_label_total"])
    standard["instrument"] = clean_nullable_string(info_raw["machine_ID_detail"])
    standard["cell_pertid_time"] = ""
    standard["drugname"] = clean_nullable_string(info_raw["pert_name"])
    smiles1 = info_raw["pert_id1"].map(lambda value: smiles_map.get(normalize_free_text(value), ""))
    smiles2 = info_raw["pert_id2"].map(lambda value: smiles_map.get(normalize_free_text(value), ""))
    target1 = info_raw["pert_id1"].map(lambda value: target_map.get(normalize_free_text(value), []))
    target2 = info_raw["pert_id2"].map(lambda value: target_map.get(normalize_free_text(value), []))
    standard["smiles"] = [merged_smiles_for_double(a, b) for a, b in zip(smiles1, smiles2)]
    standard["target_protein_list"] = [merged_targets_for_double(a, b) for a, b in zip(target1, target2)]
    standard["control"] = control_series
    standard["synergy"] = clean_numeric(info_raw["synergy"])

    standard["smiles1"] = smiles1
    standard["smiles2"] = smiles2
    standard["target_protein_list1"] = target1
    standard["target_protein_list2"] = target2
    standard["control_raw"] = clean_nullable_string(info_raw["control"])
    standard["control_status"] = control_status
    standard["source_file_info"] = str(info_path.relative_to(REPO_ROOT))
    standard["source_file_expression"] = str(expr_path.relative_to(REPO_ROOT))
    standard["expression_available"] = True

    standard = jsonize_target_columns(
        standard,
        ("target_protein_list", "target_protein_list1", "target_protein_list2"),
    )
    standard = ensure_standard_column_order(standard)
    validate_unique_sample_ids(standard, "ptv3_main_doubledrug")

    info_out = task_dir / "info.csv"
    standard.to_csv(info_out, index=False)

    header = pd.read_csv(expr_path, nrows=0).columns.tolist()
    protein_columns = [column for column in header if column != "samp_id"]
    protein_order = [parse_main_uniprot(column) for column in protein_columns]
    expression = write_expression_outputs(
        task_name="ptv3_main_doubledrug",
        expr_path=expr_path,
        sample_id_col="samp_id",
        protein_columns=protein_columns,
        protein_order=[protein for protein in protein_order if protein is not None],
        info_df=standard,
        output_dir=task_dir,
        encodings=("utf-8",),
    )

    pert_smiles_map: dict[str, str] = {}
    pert_target_map: dict[str, list[str]] = {}
    for side in ("pert_id1", "pert_id2"):
        for pert_id in info_raw[side].dropna().astype(str).tolist():
            pert_id = pert_id.strip()
            if not pert_id:
                continue
            if pert_id in smiles_map:
                pert_smiles_map.setdefault(pert_id, smiles_map[pert_id])
            if pert_id in target_map:
                pert_target_map.setdefault(pert_id, target_map[pert_id])

    audit = {
        "raw_files": [
            str(info_path.relative_to(REPO_ROOT)),
            str(expr_path.relative_to(REPO_ROOT)),
        ],
        "category": "ptv3_doubledrug",
        "table_kinds": {
            str(info_path.relative_to(REPO_ROOT)): "sample_info_table",
            str(expr_path.relative_to(REPO_ROOT)): "expression_table",
        },
        "column_mapping": {
            "sample_id": "sample_id",
            "pert_id1": "pert_id1",
            "pert_id2": "pert_id2",
            "pert_dose1": "pert_dose1",
            "pert_dose2": "pert_dose2",
            "PRISM2nd_label_total": "PRISM2nd_label_total",
            "instrument": "machine_ID_detail",
            "drugname": "pert_name",
            "smiles": "resolved from ptv3_main_singledrug pert_id -> smiles map",
            "target_protein_list": "union of side-specific target maps",
        },
        "special_rules": [
            "double-drug smiles and targets are backfilled from the main single-drug pert_id registry",
            "cell_pertid_time is unavailable in the raw double-drug table and is left blank",
        ],
        "issues": [
            {
                "kind": "unresolved_control_reference",
                "count": int((standard["control_status"] == "unresolved_non_sample_id").sum()),
            }
        ],
    }

    return TaskResult(
        task_name="ptv3_main_doubledrug",
        dataset_group="ptv3",
        info_path=str(info_out),
        expression=expression,
        sample_count=len(standard),
        protein_count=len(load_json(task_dir / "protein_order.json")),
        pert_ids=sorted(
            {
                pert_id
                for column in ("pert_id1", "pert_id2")
                for pert_id in standard[column].astype(str).tolist()
                if pert_id
            }
        ),
        protein_order=load_json(task_dir / "protein_order.json"),
        pert_smiles_map=pert_smiles_map,
        pert_target_map=pert_target_map,
        pert_target_text_map={},
        audit=audit,
    )


def standardize_extra_baseline(task_dir: Path) -> TaskResult:
    info_path = RAW_ROOT / "extra_baseline" / "260414ptv3_unseenCell_baselineProt_info.csv"
    expr_path = RAW_ROOT / "extra_baseline" / "260102ptv3_unseenCell_baselineProt.csv"

    expr_ids = pd.read_csv(expr_path, usecols=["sample_id"], low_memory=False)["sample_id"].astype(str)
    info_raw = pd.read_csv(info_path, low_memory=False)
    info_raw["sample_id"] = info_raw["sample_id"].astype(str)
    merged = pd.DataFrame({"sample_id": expr_ids})
    merged = merged.merge(info_raw, on="sample_id", how="left", indicator=True)

    standard = default_standard_frame(merged["sample_id"])
    standard["machineID_new"] = clean_nullable_string(merged["machineID_new"])
    standard["Cell_plate"] = clean_nullable_string(merged["Cell_plate"])
    standard["Cell"] = clean_nullable_string(merged["Cell"])
    standard["cell_type"] = clean_nullable_string(merged["cell_type"])
    standard["pert_id1"] = "control"
    standard["pert_id2"] = ""
    standard["batch"] = clean_nullable_string(merged["batch"])
    standard["pert_time"] = clean_numeric(merged["pert_time"]).fillna(0)
    standard["pert_dose1"] = clean_numeric(merged["pert_dose"]).fillna(0)
    standard["pert_dose2"] = np.nan
    standard["PRISM1st_label_total"] = clean_nullable_string(merged["PRISM1st_label_total"])
    standard["PRISM2nd_label_total"] = ""
    standard["instrument"] = clean_nullable_string(merged["machine_ID_detail"])
    standard["cell_pertid_time"] = ""
    standard["drugname"] = "control"
    standard["smiles"] = ""
    standard["target_protein_list"] = [[] for _ in range(len(standard))]
    standard["control"] = standard["sample_id"]
    standard["synergy"] = np.nan
    standard["raw_join_status"] = clean_nullable_string(merged["_merge"])
    standard["raw_record_issue"] = np.where(
        merged["_merge"].eq("left_only"),
        "info_missing_in_raw_file",
        "",
    )
    standard["source_file_info"] = str(info_path.relative_to(REPO_ROOT))
    standard["source_file_expression"] = str(expr_path.relative_to(REPO_ROOT))
    standard["expression_available"] = True

    standard = jsonize_target_columns(standard, ("target_protein_list",))
    standard = ensure_standard_column_order(standard)
    validate_unique_sample_ids(standard, "ptv3_extra_baseline")

    info_out = task_dir / "info.csv"
    standard.to_csv(info_out, index=False)

    header = pd.read_csv(expr_path, nrows=0).columns.tolist()
    protein_columns = [column for column in header if column != "sample_id"]
    protein_order = protein_columns
    expression = write_expression_outputs(
        task_name="ptv3_extra_baseline",
        expr_path=expr_path,
        sample_id_col="sample_id",
        protein_columns=protein_columns,
        protein_order=protein_order,
        info_df=standard,
        output_dir=task_dir,
        encodings=("utf-8",),
    )

    audit = {
        "raw_files": [
            str(info_path.relative_to(REPO_ROOT)),
            str(expr_path.relative_to(REPO_ROOT)),
        ],
        "category": "ptv3_extra_baseline",
        "table_kinds": {
            str(info_path.relative_to(REPO_ROOT)): "sample_info_table",
            str(expr_path.relative_to(REPO_ROOT)): "expression_table",
        },
        "column_mapping": {
            "pert_id1": "filled_control",
            "control": "sample_id",
            "instrument": "machine_ID_detail",
        },
        "special_rules": [
            "all extra baseline rows are treated as control candidates",
            "three expression rows without matching info rows are retained with placeholder metadata",
        ],
        "issues": [
            {
                "kind": "info_missing_in_raw_file",
                "count": int((standard["raw_record_issue"] == "info_missing_in_raw_file").sum()),
            }
        ],
    }

    return TaskResult(
        task_name="ptv3_extra_baseline",
        dataset_group="ptv3",
        info_path=str(info_out),
        expression=expression,
        sample_count=len(standard),
        protein_count=len(protein_order),
        pert_ids=["control"],
        protein_order=protein_order,
        pert_smiles_map={"control": ""},
        pert_target_map={"control": []},
        pert_target_text_map={},
        audit=audit,
    )


def resolve_single_extra_pert_id(
    row: pd.Series,
    smiles_to_pert: dict[str, str],
    name_to_pert: dict[str, str],
) -> tuple[str, str]:
    for source_column in ("smiles", "Smiles_with_chiral", "Smiles_no_chiral"):
        value = normalize_free_text(row.get(source_column))
        if value and value in smiles_to_pert:
            return smiles_to_pert[value], f"existing_by_{source_column}"
    name = normalize_name(row.get("drug_name"))
    if name and name in name_to_pert:
        return name_to_pert[name], "existing_by_drug_name"
    drug_id = normalize_free_text(row.get("drug_ID"))
    if drug_id:
        return drug_id, "raw_drug_ID"
    return namespaced_id("extra_single", row.get("drug_name")), "generated_namespace_id"


def standardize_extra_single_task(
    *,
    task_name: str,
    file_name: str,
    task_dir: Path,
    control_pool: pd.DataFrame,
    smiles_to_pert: dict[str, str],
    name_to_pert: dict[str, str],
) -> TaskResult:
    info_path = RAW_ROOT / "extra_singledrug" / file_name
    raw = pd.read_csv(info_path, low_memory=False)
    standard = default_standard_frame(pd.Series(make_generated_sample_ids(task_name, len(raw))))
    standard["machineID_new"] = clean_nullable_string(raw["machineID_new"])
    standard["Cell_plate"] = clean_nullable_string(raw["Cell_plate"])
    standard["Cell"] = clean_nullable_string(raw["Cell"])
    standard["cell_type"] = clean_nullable_string(raw["cell_type"])
    resolved = raw.apply(lambda row: resolve_single_extra_pert_id(row, smiles_to_pert, name_to_pert), axis=1)
    standard["pert_id1"] = [item[0] for item in resolved]
    standard["pert_id2"] = ""
    standard["batch"] = clean_nullable_string(raw["batch"])
    standard["pert_time"] = clean_numeric(raw["pert_time"])
    standard["pert_dose1"] = clean_numeric(raw["pert_dose"])
    standard["pert_dose2"] = np.nan
    standard["PRISM1st_label_total"] = clean_nullable_string(raw["PRISM1st_label_total"])
    standard["PRISM2nd_label_total"] = clean_nullable_string(raw["PRISM2nd_label_total"])
    standard["instrument"] = clean_nullable_string(raw["machineID_new"])
    standard["cell_pertid_time"] = ""
    standard["drugname"] = clean_nullable_string(raw["drug_name"])
    standard["smiles"] = raw.apply(
        lambda row: choose_first_non_empty(row.get("Smiles_with_chiral"), row.get("smiles"), row.get("Smiles_no_chiral")),
        axis=1,
    )
    standard["target_protein_list"] = [[] for _ in range(len(raw))]
    standard["control"] = ""
    standard["synergy"] = np.nan

    standard["pert_id_resolution"] = [item[1] for item in resolved]
    standard["drug_ID_raw"] = clean_nullable_string(raw["drug_ID"])
    standard["target_raw"] = clean_nullable_string(raw["target"])
    standard["moa_raw"] = clean_nullable_string(raw["moa"])
    standard["source_file_info"] = str(info_path.relative_to(REPO_ROOT))
    standard["source_row_index"] = raw.index
    standard["expression_available"] = False

    standard = match_controls(control_pool, standard)
    standard = jsonize_target_columns(standard, ("target_protein_list",))
    standard = ensure_standard_column_order(standard)
    validate_unique_sample_ids(standard, task_name)

    info_out = task_dir / "info.csv"
    standard.to_csv(info_out, index=False)
    expression = write_empty_expression_outputs(
        task_name=task_name,
        info_df=standard,
        output_dir=task_dir,
        reason="raw_checkout_contains_metadata_only_for_this_extra_single_task",
    )

    pert_smiles_map: dict[str, str] = {}
    pert_target_text_map: dict[str, str] = {}
    for row in standard.itertuples(index=False):
        pert_id = normalize_free_text(getattr(row, "pert_id1"))
        if not pert_id:
            continue
        pert_smiles_map.setdefault(pert_id, normalize_free_text(getattr(row, "smiles")))
        raw_target = normalize_free_text(getattr(row, "target_raw"))
        if raw_target:
            pert_target_text_map.setdefault(pert_id, raw_target)

    audit = {
        "raw_files": [str(info_path.relative_to(REPO_ROOT))],
        "category": "ptv3_extra_singledrug",
        "table_kinds": {str(info_path.relative_to(REPO_ROOT)): "sample_info_table"},
        "column_mapping": {
            "sample_id": "generated_from_task_name_and_row_index",
            "pert_id1": "existing pert_id by smiles/name > drug_ID > generated namespace id",
            "PRISM2nd_label_total": "PRISM2nd_label_total",
            "drugname": "drug_name",
            "smiles": "Smiles_with_chiral > smiles > Smiles_no_chiral",
            "control": "matched from ptv3 control pool",
        },
        "special_rules": [
            "this task has no raw perturbation proteome matrix in the checkout, so an empty expression structure is emitted",
            "control matching is audited with level / score / source task columns",
        ],
        "issues": [
            {
                "kind": "missing_expression_matrix",
                "count": len(standard),
            },
            {
                "kind": "unmatched_control",
                "count": int(standard["control"].eq("").sum()),
            },
        ],
    }

    return TaskResult(
        task_name=task_name,
        dataset_group="ptv3",
        info_path=str(info_out),
        expression=expression,
        sample_count=len(standard),
        protein_count=0,
        pert_ids=sorted({pert_id for pert_id in standard["pert_id1"].astype(str).tolist() if pert_id}),
        protein_order=[],
        pert_smiles_map=pert_smiles_map,
        pert_target_map={pert_id: [] for pert_id in pert_smiles_map},
        pert_target_text_map=pert_target_text_map,
        audit=audit,
    )


def resolve_double_pert_id(
    *,
    side_prefix: str,
    row: pd.Series,
    smiles_to_pert: dict[str, str],
    name_to_pert: dict[str, str],
    explicit_id_columns: tuple[str, ...],
    name_columns: tuple[str, ...],
    namespace: str,
) -> tuple[str, str]:
    for column in (f"{side_prefix}", f"{side_prefix.capitalize()}_with_chiral", f"{side_prefix.capitalize()}_no_chiral"):
        value = normalize_free_text(row.get(column))
        if value and value in smiles_to_pert:
            return smiles_to_pert[value], f"existing_by_{column}"
    for column in name_columns:
        normalized = normalize_name(row.get(column))
        if normalized and normalized in name_to_pert:
            return name_to_pert[normalized], f"existing_by_{column}"
    for column in explicit_id_columns:
        explicit = normalize_free_text(row.get(column))
        if explicit:
            return explicit, f"raw_{column}"
    name_fallback = choose_first_non_empty(*(row.get(column) for column in name_columns))
    return namespaced_id(namespace, name_fallback), "generated_namespace_id"


def standardize_extra_double_task(
    *,
    task_name: str,
    file_name: str,
    task_dir: Path,
    control_pool: pd.DataFrame,
    smiles_to_pert: dict[str, str],
    name_to_pert: dict[str, str],
    file_kind: str,
) -> TaskResult:
    info_path = RAW_ROOT / "extra_doubeldrug" / file_name
    raw = pd.read_csv(info_path, low_memory=False)

    if file_kind == "guomics":
        resolved_1 = [(normalize_free_text(value), "raw_pert_id1") for value in raw["pert_id1"]]
        resolved_2 = [(normalize_free_text(value), "raw_pert_id2") for value in raw["pert_id2"]]
        name_columns_1 = ("Anchor_name",)
        name_columns_2 = ("Library_name",)
    elif file_kind == "nc":
        resolved_1 = raw.apply(
            lambda row: resolve_double_pert_id(
                side_prefix="smiles1",
                row=row,
                smiles_to_pert=smiles_to_pert,
                name_to_pert=name_to_pert,
                explicit_id_columns=(),
                name_columns=("anchor_name",),
                namespace="extra_nc_anchor",
            ),
            axis=1,
        )
        resolved_2 = raw.apply(
            lambda row: resolve_double_pert_id(
                side_prefix="smiles2",
                row=row,
                smiles_to_pert=smiles_to_pert,
                name_to_pert=name_to_pert,
                explicit_id_columns=(),
                name_columns=("library_name",),
                namespace="extra_nc_library",
            ),
            axis=1,
        )
        name_columns_1 = ("anchor_name",)
        name_columns_2 = ("library_name",)
    else:
        resolved_1 = raw.apply(
            lambda row: resolve_double_pert_id(
                side_prefix="smiles1",
                row=row,
                smiles_to_pert=smiles_to_pert,
                name_to_pert=name_to_pert,
                explicit_id_columns=("anchor_ID",),
                name_columns=("Anchor.Name",),
                namespace="extra_nature_anchor",
            ),
            axis=1,
        )
        resolved_2 = raw.apply(
            lambda row: resolve_double_pert_id(
                side_prefix="smiles2",
                row=row,
                smiles_to_pert=smiles_to_pert,
                name_to_pert=name_to_pert,
                explicit_id_columns=("lib_ID",),
                name_columns=("Library.Name",),
                namespace="extra_nature_library",
            ),
            axis=1,
        )
        name_columns_1 = ("Anchor.Name",)
        name_columns_2 = ("Library.Name",)

    standard = default_standard_frame(pd.Series(make_generated_sample_ids(task_name, len(raw))))
    standard["machineID_new"] = clean_nullable_string(raw["machineID_new"])
    standard["Cell_plate"] = clean_nullable_string(raw["Cell_plate"])
    standard["Cell"] = clean_nullable_string(raw["Cell"])
    standard["cell_type"] = clean_nullable_string(raw["cell_type"])
    standard["pert_id1"] = [item[0] for item in resolved_1]
    standard["pert_id2"] = [item[0] for item in resolved_2]
    standard["batch"] = clean_nullable_string(raw["batch"])
    standard["pert_time"] = clean_numeric(raw["pert_time"])
    standard["pert_dose1"] = clean_numeric(raw["pert_dose"])
    standard["pert_dose2"] = clean_numeric(raw["pert_dose"])
    standard["PRISM1st_label_total"] = "non-responsive"
    standard["PRISM2nd_label_total"] = ""
    standard["instrument"] = clean_nullable_string(raw["machineID_new"])
    standard["cell_pertid_time"] = ""
    drugname_1 = raw[list(name_columns_1)].apply(lambda row: choose_first_non_empty(*row.tolist()), axis=1)
    drugname_2 = raw[list(name_columns_2)].apply(lambda row: choose_first_non_empty(*row.tolist()), axis=1)
    standard["drugname"] = [f"{a} || {b}" if a and b else choose_first_non_empty(a, b) for a, b in zip(drugname_1, drugname_2)]
    smiles1 = raw.apply(
        lambda row: choose_first_non_empty(row.get("Smiles1_with_chiral"), row.get("smiles1"), row.get("Smiles1_no_chiral")),
        axis=1,
    )
    smiles2 = raw.apply(
        lambda row: choose_first_non_empty(row.get("Smiles2_with_chiral"), row.get("smiles2"), row.get("Smiles2_no_chiral")),
        axis=1,
    )
    standard["smiles"] = [merged_smiles_for_double(a, b) for a, b in zip(smiles1, smiles2)]
    standard["target_protein_list"] = [[] for _ in range(len(raw))]
    standard["control"] = ""
    standard["synergy"] = clean_nullable_string(raw["synergy"])

    standard["smiles1"] = smiles1
    standard["smiles2"] = smiles2
    standard["pert_id1_resolution"] = [item[1] for item in resolved_1]
    standard["pert_id2_resolution"] = [item[1] for item in resolved_2]
    if file_kind == "nc":
        standard["target1_raw"] = clean_nullable_string(raw["anchor_Primary_Target"])
        standard["target2_raw"] = clean_nullable_string(raw["library_Primary_Target"])
    elif file_kind == "nature":
        standard["target1_raw"] = clean_nullable_string(raw["Anchor.Target"])
        standard["target2_raw"] = clean_nullable_string(raw["library.Target"])
        standard["anchor_ID_raw"] = clean_nullable_string(raw["anchor_ID"])
        standard["lib_ID_raw"] = clean_nullable_string(raw["lib_ID"])
    else:
        standard["target1_raw"] = ""
        standard["target2_raw"] = ""
    standard["source_file_info"] = str(info_path.relative_to(REPO_ROOT))
    standard["source_row_index"] = raw.index
    standard["expression_available"] = False

    standard = match_controls(control_pool, standard)
    standard = jsonize_target_columns(standard, ("target_protein_list",))
    standard = ensure_standard_column_order(standard)
    validate_unique_sample_ids(standard, task_name)

    info_out = task_dir / "info.csv"
    standard.to_csv(info_out, index=False)
    expression = write_empty_expression_outputs(
        task_name=task_name,
        info_df=standard,
        output_dir=task_dir,
        reason="raw_checkout_contains_metadata_only_for_this_extra_double_task",
    )

    pert_smiles_map: dict[str, str] = {}
    pert_target_text_map: dict[str, str] = {}
    for row in standard.itertuples(index=False):
        pert1 = normalize_free_text(getattr(row, "pert_id1"))
        pert2 = normalize_free_text(getattr(row, "pert_id2"))
        smiles_1 = normalize_free_text(getattr(row, "smiles1"))
        smiles_2 = normalize_free_text(getattr(row, "smiles2"))
        if pert1:
            pert_smiles_map.setdefault(pert1, smiles_1)
            target1 = normalize_free_text(getattr(row, "target1_raw", ""))
            if target1:
                pert_target_text_map.setdefault(pert1, target1)
        if pert2:
            pert_smiles_map.setdefault(pert2, smiles_2)
            target2 = normalize_free_text(getattr(row, "target2_raw", ""))
            if target2:
                pert_target_text_map.setdefault(pert2, target2)

    audit = {
        "raw_files": [str(info_path.relative_to(REPO_ROOT))],
        "category": "ptv3_extra_doubledrug",
        "table_kinds": {str(info_path.relative_to(REPO_ROOT)): "sample_info_table"},
        "column_mapping": {
            "sample_id": "generated_from_task_name_and_row_index",
            "pert_id1": "existing pert_id by smiles/name > explicit file ID > generated namespace id",
            "pert_id2": "existing pert_id by smiles/name > explicit file ID > generated namespace id",
            "drugname": f"{name_columns_1[0]} || {name_columns_2[0]}",
            "smiles": "Smiles*_with_chiral > smiles* > Smiles*_no_chiral",
            "control": "matched from ptv3 control pool",
        },
        "special_rules": [
            "this task has no raw perturbation proteome matrix in the checkout, so an empty expression structure is emitted",
            "synergy is preserved as the raw category label from the source csv",
        ],
        "issues": [
            {
                "kind": "missing_expression_matrix",
                "count": len(standard),
            },
            {
                "kind": "unmatched_control",
                "count": int(standard["control"].eq("").sum()),
            },
        ],
    }

    return TaskResult(
        task_name=task_name,
        dataset_group="ptv3",
        info_path=str(info_out),
        expression=expression,
        sample_count=len(standard),
        protein_count=0,
        pert_ids=sorted(
            {
                pert_id
                for column in ("pert_id1", "pert_id2")
                for pert_id in standard[column].astype(str).tolist()
                if pert_id
            }
        ),
        protein_order=[],
        pert_smiles_map=pert_smiles_map,
        pert_target_map={pert_id: [] for pert_id in pert_smiles_map},
        pert_target_text_map=pert_target_text_map,
        audit=audit,
    )


def parse_ptv1_split_file(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) == 1:
                cell, pert_id = parts[0].split("_", 1)
                pairs.add((cell, pert_id))
                continue
            first_cell, first_pert = parts[0].split("_", 1)
            pairs.add((first_cell, first_pert))
            for pert_id in parts[1:]:
                pairs.add((first_cell, pert_id))
    return pairs


def build_ptv1_control_column(df: pd.DataFrame) -> pd.Series:
    result = pd.Series([""] * len(df), dtype="string")
    mask_is_control = df["pert_time"].eq(0)
    result.loc[mask_is_control] = df.loc[mask_is_control, "sample_id"].astype(str)
    control_map = (
        df.loc[mask_is_control, ["BioRep", "Cell_plate", "sample_id"]]
        .drop_duplicates(subset=["BioRep", "Cell_plate"])
        .set_index(["BioRep", "Cell_plate"])["sample_id"]
        .to_dict()
    )
    indexer = list(zip(df["BioRep"].tolist(), df["Cell_plate"].tolist()))
    mapped = [control_map.get(key, "") for key in indexer]
    result.loc[~mask_is_control] = pd.Series(mapped, index=df.index).loc[~mask_is_control]
    return result


def standardize_ptv1(task_dir: Path) -> TaskResult:
    mixed_path = RAW_ROOT / "ptv1" / "aivc.csv"
    info_path = RAW_ROOT / "ptv1" / "aivc_info.csv"
    drug_meta_path = RAW_ROOT / "ptv1" / "ptv1.csv"
    e115_map_path = RAW_ROOT / "ptv1" / "ptds4_84drug_E115ID.csv"
    prediction_path = RAW_ROOT / "ptv1" / "test12091214_sample_predictions_E115id.csv"
    split_dir = RAW_ROOT / "ptv1" / "experiment_type_list"

    info_raw = pd.read_csv(info_path, low_memory=False)
    mixed_header = pd.read_csv(mixed_path, nrows=0).columns.tolist()
    info_start_idx = mixed_header.index("Library_dose")
    protein_columns = mixed_header[1:info_start_idx]
    protein_order = []
    unresolved_protein_columns = []
    resolved_protein_columns = []
    for column in protein_columns:
        protein = parse_ptv1_uniprot(column)
        if protein is None:
            unresolved_protein_columns.append(column)
            continue
        resolved_protein_columns.append(column)
        protein_order.append(protein)

    info_raw["sample_id"] = info_raw["Sample_ID"].astype(str)
    standard = default_standard_frame(info_raw["sample_id"])
    standard["machineID_new"] = clean_nullable_string(info_raw["machine"])
    standard["Cell_plate"] = clean_nullable_string(info_raw["protein_plate"])
    standard["Cell"] = info_raw.apply(
        lambda row: choose_first_non_empty(row.get("Cell.Line.name"), row.get("protein_plate")),
        axis=1,
    )
    standard["cell_type"] = ""
    standard["pert_id1"] = clean_nullable_string(info_raw["pert_id"])
    standard["pert_id2"] = clean_nullable_string(info_raw["Anchor_id"])
    standard["batch"] = "no"
    standard["pert_time"] = clean_numeric(info_raw["pert_time"])
    standard["pert_dose1"] = 0.0
    standard["pert_dose2"] = 0.0
    standard["PRISM1st_label_total"] = clean_nullable_string(info_raw["NY_label"])
    standard["PRISM2nd_label_total"] = ""
    standard["instrument"] = clean_nullable_string(info_raw["machine"])
    standard["cell_pertid_time"] = ""
    standard["drugname"] = info_raw.apply(
        lambda row: choose_first_non_empty(row.get("drugNameAB"), row.get("pert_iname")),
        axis=1,
    )
    standard["smiles"] = ""
    standard["target_protein_list"] = [[] for _ in range(len(info_raw))]
    standard["synergy"] = clean_nullable_string(info_raw["Synergy"])
    standard["BioRep"] = clean_nullable_string(info_raw["BioRep"])
    standard["smiles_raw"] = ""

    drug_meta = pd.read_csv(drug_meta_path, low_memory=False)
    drug_meta["Pert_ID"] = drug_meta["Pert_ID"].astype(str)
    ptv1_smiles_map: dict[str, str] = {}
    ptv1_target_map: dict[str, list[str]] = {}
    for _, row in drug_meta.iterrows():
        pert_id = normalize_free_text(row["Pert_ID"])
        if not pert_id:
            continue
        ptv1_smiles_map[pert_id] = normalize_free_text(row["SMILES"])
        targets = [
            token
            for token in (
                normalize_free_text(row["Targeted_protein 1 (Uniprot_ID)"]),
                normalize_free_text(row["Targeted_protein 2 (Uniprot_ID)"]),
            )
            if token and token.lower() != "nan"
        ]
        ptv1_target_map[pert_id] = targets
    standard["smiles"] = standard["pert_id1"].map(lambda value: ptv1_smiles_map.get(value, ""))
    standard["target_protein_list"] = standard["pert_id1"].map(lambda value: ptv1_target_map.get(value, []))
    standard["control"] = build_ptv1_control_column(
        pd.DataFrame(
            {
                "sample_id": standard["sample_id"],
                "pert_time": standard["pert_time"],
                "BioRep": standard["BioRep"],
                "Cell_plate": standard["Cell_plate"],
            }
        )
    )
    standard["smiles_raw"] = standard["smiles"]

    train_pairs = parse_ptv1_split_file(split_dir / "train_experiment_type_list.txt")
    val_pairs = parse_ptv1_split_file(split_dir / "val_experiment_type_list.txt")
    test_pairs = parse_ptv1_split_file(split_dir / "test_experiment_type_list.txt")
    split_values = []
    for row in standard.itertuples(index=False):
        key = (normalize_free_text(getattr(row, "Cell_plate")), normalize_free_text(getattr(row, "pert_id1")))
        if key in test_pairs:
            split_values.append("test")
        elif key in val_pairs:
            split_values.append("val")
        elif key in train_pairs:
            split_values.append("train")
        else:
            split_values.append("no")
    standard["data_split"] = split_values
    standard["source_file_info"] = str(info_path.relative_to(REPO_ROOT))
    standard["source_file_expression"] = str(mixed_path.relative_to(REPO_ROOT))
    standard["source_file_drug_meta"] = str(drug_meta_path.relative_to(REPO_ROOT))
    standard["expression_available"] = True

    standard = jsonize_target_columns(standard, ("target_protein_list",))
    standard = ensure_standard_column_order(standard)
    validate_unique_sample_ids(standard, "ptv1_aivc")

    info_out = task_dir / "info.csv"
    standard.to_csv(info_out, index=False)

    expression = write_expression_outputs(
        task_name="ptv1_aivc",
        expr_path=mixed_path,
        sample_id_col="Sample_ID",
        protein_columns=resolved_protein_columns,
        protein_order=protein_order,
        info_df=standard,
        output_dir=task_dir,
        encodings=("utf-8",),
    )

    pert_smiles_map = {pert_id: smiles for pert_id, smiles in ptv1_smiles_map.items() if smiles}
    pert_target_map = {pert_id: targets for pert_id, targets in ptv1_target_map.items()}
    prediction_rows = len(pd.read_csv(prediction_path, low_memory=False))
    e115_rows = len(pd.read_csv(e115_map_path, low_memory=False))

    audit = {
        "raw_files": [
            str(mixed_path.relative_to(REPO_ROOT)),
            str(info_path.relative_to(REPO_ROOT)),
            str(drug_meta_path.relative_to(REPO_ROOT)),
            str(e115_map_path.relative_to(REPO_ROOT)),
            str(prediction_path.relative_to(REPO_ROOT)),
            str((split_dir / "train_experiment_type_list.txt").relative_to(REPO_ROOT)),
            str((split_dir / "val_experiment_type_list.txt").relative_to(REPO_ROOT)),
            str((split_dir / "test_experiment_type_list.txt").relative_to(REPO_ROOT)),
        ],
        "category": "ptv1",
        "table_kinds": {
            str(mixed_path.relative_to(REPO_ROOT)): "mixed_expression_and_info_table",
            str(info_path.relative_to(REPO_ROOT)): "sample_info_table",
            str(drug_meta_path.relative_to(REPO_ROOT)): "drug_metadata_table",
            str(e115_map_path.relative_to(REPO_ROOT)): "mapping_table",
            str(prediction_path.relative_to(REPO_ROOT)): "prediction_reference_table",
            str((split_dir / "train_experiment_type_list.txt").relative_to(REPO_ROOT)): "split_reference",
            str((split_dir / "val_experiment_type_list.txt").relative_to(REPO_ROOT)): "split_reference",
            str((split_dir / "test_experiment_type_list.txt").relative_to(REPO_ROOT)): "split_reference",
        },
        "column_mapping": {
            "sample_id": "Sample_ID",
            "machineID_new": "machine",
            "Cell_plate": "protein_plate",
            "Cell": "Cell.Line.name with protein_plate fallback",
            "pert_id1": "pert_id",
            "pert_id2": "Anchor_id",
            "PRISM1st_label_total": "NY_label",
            "control": "derived from pert_time == 0 within (BioRep, Cell_plate)",
            "data_split": "derived from experiment_type_list using (protein_plate, pert_id)",
        },
        "special_rules": [
            "ptv1 is isolated into its own standardized output root and meta index",
            "11 protein columns without resolvable UniProt accession are excluded from the standardized expression matrix",
        ],
        "issues": [
            {"kind": "unresolved_protein_columns", "count": len(unresolved_protein_columns)},
            {"kind": "prediction_reference_rows", "count": prediction_rows},
            {"kind": "e115_mapping_rows", "count": e115_rows},
        ],
        "unresolved_protein_columns": unresolved_protein_columns,
    }

    return TaskResult(
        task_name="ptv1_aivc",
        dataset_group="ptv1",
        info_path=str(info_out),
        expression=expression,
        sample_count=len(standard),
        protein_count=len(protein_order),
        pert_ids=sorted({pert_id for pert_id in standard["pert_id1"].astype(str).tolist() if pert_id}),
        protein_order=protein_order,
        pert_smiles_map=pert_smiles_map,
        pert_target_map=pert_target_map,
        pert_target_text_map={},
        audit=audit,
    )


def build_global_meta(dataset_group: str, task_results: list[TaskResult], output_root: Path) -> None:
    protein_index: dict[str, int] = {}
    pert_index: dict[str, int] = {}
    pert_smiles: dict[str, str] = {}
    pert_targets: dict[str, list[str]] = {}
    pert_target_text: dict[str, str] = {}
    pert_conflicts: list[dict[str, object]] = []

    for task in task_results:
        for protein in task.protein_order:
            if protein and protein not in protein_index:
                protein_index[protein] = len(protein_index)
        for pert_id in task.pert_ids:
            if pert_id and pert_id not in pert_index:
                pert_index[pert_id] = len(pert_index)
        for pert_id, smiles in task.pert_smiles_map.items():
            if not pert_id:
                continue
            existing = pert_smiles.get(pert_id, "")
            if existing and smiles and existing != smiles:
                pert_conflicts.append(
                    {
                        "pert_id": pert_id,
                        "existing_smiles": existing,
                        "new_smiles": smiles,
                        "task_name": task.task_name,
                    }
                )
            elif smiles:
                pert_smiles.setdefault(pert_id, smiles)
        for pert_id, targets in task.pert_target_map.items():
            if pert_id not in pert_targets or (not pert_targets[pert_id] and targets):
                pert_targets[pert_id] = targets
        for pert_id, target_text in task.pert_target_text_map.items():
            if target_text:
                pert_target_text.setdefault(pert_id, target_text)

    payload = {
        "dataset_group": dataset_group,
        "generated_at": iso_now(),
        "protein_index": protein_index,
        "pert_index": pert_index,
        "pertid_to_smiles": pert_smiles,
        "pertid_to_target_protein_list": pert_targets,
        "pertid_to_target_text": pert_target_text,
        "task_names": [task.task_name for task in task_results],
        "pert_mapping_conflicts": pert_conflicts,
    }
    dump_json(output_root / "global_meta.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize ProteinTalk raw data")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for standardized outputs",
    )
    args = parser.parse_args()

    output_root = ensure_dir(Path(args.output_root))
    ptv3_root = ensure_dir(output_root / "ptv3")
    ptv1_root = ensure_dir(output_root / "ptv1")
    ptv3_tasks_root = ensure_dir(ptv3_root / "tasks")
    ptv1_tasks_root = ensure_dir(ptv1_root / "tasks")

    task_results: list[TaskResult] = []
    file_audit: dict[str, object] = {
        "generated_at": iso_now(),
        "source_document": str((REPO_ROOT / "data" / "Data_Process.md").relative_to(REPO_ROOT)),
        "tasks": {},
    }

    single_task_dir = ensure_dir(ptv3_tasks_root / "ptv3_main_singledrug")
    single_result = standardize_main_singledrug(single_task_dir)
    task_results.append(single_result)

    main_single_info = pd.read_csv(single_result.info_path, low_memory=False)
    main_single_maps = {
        "pert_smiles_map": single_result.pert_smiles_map,
        "pert_target_map": single_result.pert_target_map,
    }
    smiles_to_pert, name_to_pert, ambiguous_smiles = parse_smiles_to_existing_maps(
        safe_read_csv(RAW_ROOT / "singledrug" / "20260403_ptv3_v2_bind_bio_sampleID_machineID_details.csv", low_memory=False)
    )

    double_task_dir = ensure_dir(ptv3_tasks_root / "ptv3_main_doubledrug")
    double_result = standardize_main_doubledrug(double_task_dir, main_single_maps)
    task_results.append(double_result)

    extra_baseline_task_dir = ensure_dir(ptv3_tasks_root / "ptv3_extra_baseline")
    extra_baseline_result = standardize_extra_baseline(extra_baseline_task_dir)
    task_results.append(extra_baseline_result)

    extra_baseline_info = pd.read_csv(extra_baseline_result.info_path, low_memory=False)
    control_pool = build_control_pool(main_single_info, extra_baseline_info)

    extra_single_specs = [
        ("ptv3_extra_singledrug_mat1_480_faims", "20260413ptv3_PRISM1st_validation_phenotype_mat1_480_faims_add_PRISM2nd_label_dup.csv"),
        ("ptv3_extra_singledrug_mat1_qe", "20260413ptv3_PRISM1st_validation_phenotype_mat1_qe_add_PRISM2nd_label_dup.csv"),
        ("ptv3_extra_singledrug_mat2_480_faims", "20260413ptv3_PRISM1st_validation_phenotype_mat2_480_faims_add_PRISM2nd_label_dup.csv"),
        ("ptv3_extra_singledrug_mat2_qe", "20260413ptv3_PRISM1st_validation_phenotype_mat2_qe_add_PRISM2nd_label_dup.csv"),
        ("ptv3_extra_singledrug_mat3_qe", "20260413ptv3_PRISM1st_validation_phenotype_mat3_add_PRISM2nd_label_dup.csv"),
        ("ptv3_extra_singledrug_mat4_qe", "20260413ptv3_PRISM1st_validation_phenotype_mat4_add_PRISM2nd_label_dup.csv"),
    ]
    for task_name, file_name in extra_single_specs:
        task_dir = ensure_dir(ptv3_tasks_root / task_name)
        task_result = standardize_extra_single_task(
            task_name=task_name,
            file_name=file_name,
            task_dir=task_dir,
            control_pool=control_pool,
            smiles_to_pert=smiles_to_pert,
            name_to_pert=name_to_pert,
        )
        task_results.append(task_result)

    extra_double_specs = [
        ("ptv3_extra_doubledrug_guomics", "20260410ptv3_Guomics_drug_combo_vali_unique.csv", "guomics"),
        ("ptv3_extra_doubledrug_nc", "20260411NC_combo_info_unique.csv", "nc"),
        ("ptv3_extra_doubledrug_nature", "20260411nature_drugComb_info_unique.csv", "nature"),
    ]
    for task_name, file_name, file_kind in extra_double_specs:
        task_dir = ensure_dir(ptv3_tasks_root / task_name)
        task_result = standardize_extra_double_task(
            task_name=task_name,
            file_name=file_name,
            task_dir=task_dir,
            control_pool=control_pool,
            smiles_to_pert=smiles_to_pert,
            name_to_pert=name_to_pert,
            file_kind=file_kind,
        )
        task_results.append(task_result)

    ptv1_task_dir = ensure_dir(ptv1_tasks_root / "ptv1_aivc")
    ptv1_result = standardize_ptv1(ptv1_task_dir)

    ptv3_results = [result for result in task_results if result.dataset_group == "ptv3"]
    build_global_meta("ptv3", ptv3_results, ptv3_root)
    build_global_meta("ptv1", [ptv1_result], ptv1_root)

    for result in ptv3_results + [ptv1_result]:
        file_audit["tasks"][result.task_name] = {
            "task_name": result.task_name,
            "dataset_group": result.dataset_group,
            "info_path": result.info_path,
            "expression": result.expression,
            "sample_count": result.sample_count,
            "protein_count": result.protein_count,
            "audit": result.audit,
        }
    file_audit["smiles_resolution_ambiguity_count"] = len(ambiguous_smiles)
    file_audit["smiles_resolution_ambiguous_examples"] = dict(list(ambiguous_smiles.items())[:20])
    dump_json(output_root / "file_audit.json", file_audit)


if __name__ == "__main__":
    main()
