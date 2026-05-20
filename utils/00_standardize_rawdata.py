#!/usr/bin/env python3
"""Standardize ProteinTalk raw data into reproducible task-level artifacts.

This script follows `docs/Data_Process.md` and produces:
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


def normalize_identifier_fragment(value: object) -> str:
    text = normalize_name(value)
    if text:
        return text
    raw = normalize_free_text(value)
    if raw:
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return ""


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


def copy_pert_id1_to_blank_pert_id2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pert1 = clean_nullable_string(df["pert_id1"])
    pert2 = clean_nullable_string(df["pert_id2"])
    df["pert_id2"] = pert2.where(pert2.ne(""), pert1)
    return df


def choose_first_non_empty(*values: object) -> str:
    for value in values:
        text = normalize_free_text(value)
        if text:
            return text
    return ""


def clean_placeholder_string(
    series: pd.Series,
    *,
    placeholders: set[str] | None = None,
) -> pd.Series:
    if placeholders is None:
        placeholders = {"na", "nan", "no", "none", "null"}
    values = []
    for value in series.tolist():
        text = normalize_free_text(value)
        values.append("" if text.lower() in placeholders else text)
    return pd.Series(values, index=series.index, dtype="string").fillna("")


def parse_uniprot_token(token: str) -> bool:
    token = token.strip()
    return bool(
        re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", token)
        or re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{3}[0-9]", token)
        or re.fullmatch(r"[A-NR-Z][0-9](?:[A-Z0-9]{3}[0-9]){2}", token)
        or re.fullmatch(r"A0A[A-Z0-9]{7}", token)
    )


def parse_ptv3_single_matrix_uniprot(column_name: str) -> str | None:
    if "_" not in column_name:
        return None
    token = column_name.split("_", 1)[0]
    return token if parse_uniprot_token(token) else None


def parse_ptv3_direct_uniprot(column_name: str) -> str | None:
    return column_name if parse_uniprot_token(column_name) else None


def parse_ptv1_uniprot(column_name: str) -> str | None:
    for token in column_name.split("."):
        if parse_uniprot_token(token):
            return token
    return None


def resolve_expression_columns(
    *,
    task_name: str,
    expr_path: Path,
    protein_columns: list[str],
) -> tuple[list[str], list[str], list[str], str]:
    file_name = expr_path.name
    if file_name == "20250113_ptv3_unique_mat_28602samp_10982prot_finall_v2.csv":
        parser = parse_ptv3_single_matrix_uniprot
        allow_unresolved = False
        rule_text = "protein columns use `UniProtID_GeneSymbol`; the token before the first `_` must be a valid UniProt accession"
    elif file_name in {
        "20260422ptv3_J_3496samp_9202prot_final_edit.csv",
        "20260417ptv3_J_3509samp_9112prot_finall_edit.csv",
    }:
        parser = parse_ptv3_direct_uniprot
        allow_unresolved = False
        rule_text = "protein columns are direct UniProt accession values"
    elif file_name == "20250211_ptv3_J_3549samp_9205prot_finall_edit.csv":
        parser = parse_ptv3_single_matrix_uniprot
        allow_unresolved = False
        rule_text = "legacy double-drug protein columns use `UniProtID_GeneSymbol`; the token before the first `_` must be a valid UniProt accession"
    elif file_name == "260102ptv3_unseenCell_baselineProt.csv":
        parser = parse_ptv3_direct_uniprot
        allow_unresolved = False
        rule_text = "protein columns are direct UniProt accession values"
    elif file_name == "aivc.csv":
        parser = parse_ptv1_uniprot
        allow_unresolved = True
        rule_text = "protein columns are dot-delimited descriptors; the first token that matches a UniProt accession is used"
    else:
        raise ValueError(f"{task_name}: no explicit protein parsing rule is defined for {expr_path}")

    resolved_columns: list[str] = []
    protein_order: list[str] = []
    unresolved_columns: list[str] = []
    for column in protein_columns:
        protein = parser(column)
        if protein is None:
            unresolved_columns.append(column)
            continue
        resolved_columns.append(column)
        protein_order.append(protein)

    duplicated_proteins = sorted({protein for protein in protein_order if protein_order.count(protein) > 1})
    if duplicated_proteins:
        raise ValueError(f"{task_name}: duplicated UniProt IDs after parsing: {duplicated_proteins[:10]}")
    if unresolved_columns and not allow_unresolved:
        raise ValueError(
            f"{task_name}: unresolved protein columns under explicit parsing rule: {unresolved_columns[:10]}"
        )
    return resolved_columns, protein_order, unresolved_columns, rule_text


def parse_target_list(value: object) -> list[str]:
    raw = normalize_free_text(value)
    if not raw or raw.upper() == "NA":
        return []
    tokens = [token.strip() for token in re.split(r"[;,|]+", raw) if token.strip()]
    return [token for token in tokens if parse_uniprot_token(token)]


EXTRA_TARGET_GENE_ALIAS_MAP = {
    "bclxl": ["BCL2L1"],
    "bclw": ["BCL2L2"],
    "dnapk": ["PRKDC"],
    "erk5": ["MAPK7"],
    "ir": ["INSR"],
    "mek1": ["MAP2K1"],
    "mek2": ["MAP2K2"],
    "mtorc1": ["MTOR"],
    "mtorc2": ["MTOR"],
}


def append_unique(items: list[str], values: Iterable[str]) -> list[str]:
    for value in values:
        if value and value not in items:
            items.append(value)
    return items


def build_extra_target_maps(mapping_path: Path) -> dict[str, dict[str, list[str]]]:
    mapping_df = pd.read_csv(mapping_path, low_memory=False)
    gene_to_uniprots: dict[str, list[str]] = defaultdict(list)
    drug_to_uniprots: dict[str, list[str]] = defaultdict(list)

    for row in mapping_df.itertuples(index=False):
        gene_key = normalize_name(getattr(row, "gene"))
        uniprot_id = normalize_free_text(getattr(row, "UniprotID_final"))
        if not gene_key or not parse_uniprot_token(uniprot_id):
            continue
        append_unique(gene_to_uniprots[gene_key], [uniprot_id])
        for raw_drug in re.split(r"[;|]+", normalize_free_text(getattr(row, "prism1st_drug"))):
            drug_key = normalize_name(raw_drug)
            if drug_key:
                append_unique(drug_to_uniprots[drug_key], [uniprot_id])

    return {
        "gene_to_uniprots": dict(gene_to_uniprots),
        "drug_to_uniprots": dict(drug_to_uniprots),
    }


def target_text_tokens(raw_target: object) -> list[str]:
    raw = normalize_free_text(raw_target)
    if not raw:
        return []
    return [token.strip() for token in re.split(r"[;,|]+", raw) if token.strip()]


def expand_target_gene_candidates(token: str) -> list[str]:
    compact = re.sub(r"\s+", "", normalize_free_text(token))
    if not compact:
        return []

    normalized = normalize_name(compact)
    aliases = EXTRA_TARGET_GENE_ALIAS_MAP.get(normalized)
    if aliases:
        return aliases

    numeric_pair = re.fullmatch(r"([A-Za-z-]+)(\d+)/(\d+)", compact)
    if numeric_pair:
        prefix, first, second = numeric_pair.groups()
        return [f"{prefix}{first}", f"{prefix}{second}"]

    return [compact]


def target_drug_name_keys(drug_name: object) -> list[str]:
    raw = normalize_free_text(drug_name)
    if not raw:
        return []

    candidates: list[str] = []

    def add(text: str) -> None:
        normalized = normalize_name(text)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(raw)
    add(re.sub(r"^(?:S-\(\+\)-|R-\(-\)-|S-|R-|rac-)", "", raw, flags=re.IGNORECASE))
    add(re.sub(r"\([^)]*\)", "", raw))
    return candidates


def resolve_extra_target_protein_list(
    *,
    raw_target: object,
    drug_names: Iterable[object],
    target_maps: dict[str, dict[str, list[str]]],
) -> list[str]:
    resolved: list[str] = []
    gene_to_uniprots = target_maps["gene_to_uniprots"]
    drug_to_uniprots = target_maps["drug_to_uniprots"]

    for token in target_text_tokens(raw_target):
        for candidate in expand_target_gene_candidates(token):
            append_unique(resolved, gene_to_uniprots.get(normalize_name(candidate), []))

    for drug_name in drug_names:
        for drug_key in target_drug_name_keys(drug_name):
            append_unique(resolved, drug_to_uniprots.get(drug_key, []))

    return resolved


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


def make_generated_sample_ids(task_name: str, size: int) -> list[str]:
    return [f"{task_name}__{idx:06d}" for idx in range(size)]


def ensure_standard_column_order(df: pd.DataFrame) -> pd.DataFrame:
    leading = [column for column in STANDARD_INFO_COLUMNS if column in df.columns]
    trailing = [column for column in df.columns if column not in leading]
    return df[leading + trailing]


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"none of the candidate paths exist: {joined}")


def resolve_header_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    columns_set = set(columns)
    for candidate in candidates:
        if candidate in columns_set:
            return candidate
    joined = ", ".join(candidates)
    raise KeyError(f"none of the expected columns were found: {joined}")


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


def build_unified_external_pert_id(
    *,
    explicit_id: object = "",
    smiles: object = "",
    name: object = "",
) -> tuple[str, str]:
    explicit = normalize_identifier_fragment(explicit_id)
    if explicit:
        return f"extid::{explicit}", "unified_by_raw_id"
    smiles_text = normalize_free_text(smiles)
    if smiles_text:
        digest = hashlib.sha1(smiles_text.encode("utf-8")).hexdigest()[:12]
        return f"extsmiles::{digest}", "unified_by_smiles"
    name_slug = normalize_identifier_fragment(name)
    if name_slug:
        return f"extname::{name_slug}", "unified_by_name"
    digest = hashlib.sha1(
        "|".join([normalize_free_text(explicit_id), smiles_text, normalize_free_text(name)]).encode("utf-8")
    ).hexdigest()[:12]
    return f"extunk::{digest}", "unified_by_fallback_hash"


def normalize_control_column(
    control_series: pd.Series,
    *,
    row_sample_ids: pd.Series,
    valid_sample_ids: set[str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw = clean_nullable_string(control_series)
    sample_ids = clean_nullable_string(row_sample_ids)
    normalized = []
    status = []
    is_control = []
    for value, sample_id in zip(raw.tolist(), sample_ids.tolist()):
        if not value:
            normalized.append("")
            status.append("missing")
            is_control.append(False)
            continue
        if value.lower() == "control":
            normalized.append(sample_id)
            status.append("self_control_literal")
            is_control.append(True)
            continue
        if value == sample_id:
            normalized.append(sample_id)
            status.append("self_control_sample_id")
            is_control.append(True)
            continue
        if value in valid_sample_ids:
            normalized.append(value)
            status.append("ok_reference")
            is_control.append(False)
            continue
        normalized.append("")
        status.append("unresolved_non_sample_id")
        is_control.append(False)
    return (
        pd.Series(normalized, dtype="string").fillna(""),
        pd.Series(status, dtype="string").fillna(""),
        pd.Series(is_control, dtype="boolean"),
    )


def extract_standardized_controls(df: pd.DataFrame, *, task_name: str, pool_kind: str) -> pd.DataFrame:
    mask = clean_nullable_string(df["control"]).eq(clean_nullable_string(df["sample_id"]))
    controls = df.loc[mask].copy()
    controls["control_source_task"] = task_name
    controls["control_pool_kind"] = pool_kind
    return controls


def build_control_pool(single_df: pd.DataFrame, double_df: pd.DataFrame, extra_baseline_df: pd.DataFrame) -> pd.DataFrame:
    single_controls = extract_standardized_controls(
        single_df,
        task_name="ptv3_main_singledrug",
        pool_kind="main_single_self_control",
    )
    double_controls = extract_standardized_controls(
        double_df,
        task_name="ptv3_main_doubledrug",
        pool_kind="main_double_self_control",
    )
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
        [single_controls[pool_columns], double_controls[pool_columns], baseline_controls[pool_columns]],
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
        duplicate_rows = chunk.loc[chunk[sample_id_col].duplicated(keep=False), sample_id_col].astype(str).unique().tolist()
        if duplicate_rows:
            raise ValueError(f"{task_name}: duplicated sample_id values inside expression chunk: {duplicate_rows[:10]}")
        overlapping = sorted(set(chunk[sample_id_col].tolist()) & seen_sample_ids)
        if overlapping:
            raise ValueError(f"{task_name}: duplicated sample_id values across expression chunks: {overlapping[:10]}")
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


def build_canonical_smiles_map(task_results: list[TaskResult]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for task in task_results:
        for pert_id, smiles in task.pert_smiles_map.items():
            normalized_pert_id = normalize_free_text(pert_id)
            normalized_smiles = normalize_free_text(smiles)
            if normalized_pert_id and normalized_smiles and normalized_pert_id not in canonical:
                canonical[normalized_pert_id] = normalized_smiles
    return canonical


def canonical_smiles_series(
    df: pd.DataFrame,
    *,
    pert_id_column: str,
    canonical_smiles: dict[str, str],
    fallback_column: str | None = None,
) -> pd.Series:
    resolved = pd.Series(
        [canonical_smiles.get(normalize_free_text(value), "") for value in df[pert_id_column]],
        index=df.index,
        dtype="string",
    ).fillna("")
    if fallback_column and fallback_column in df.columns:
        fallback = clean_nullable_string(df[fallback_column])
        resolved = resolved.where(resolved.ne(""), fallback)
    return resolved.fillna("")


def rewrite_task_info_smiles(task: TaskResult, canonical_smiles: dict[str, str]) -> None:
    info_path = Path(task.info_path)
    df = pd.read_csv(info_path, low_memory=False)

    smiles1 = canonical_smiles_series(
        df,
        pert_id_column="pert_id1",
        canonical_smiles=canonical_smiles,
        fallback_column="smiles1" if "smiles1" in df.columns else "smiles",
    )
    if "smiles1" in df.columns:
        df["smiles1"] = smiles1

    if "pert_id2" in df.columns:
        smiles2 = canonical_smiles_series(
            df,
            pert_id_column="pert_id2",
            canonical_smiles=canonical_smiles,
            fallback_column="smiles2" if "smiles2" in df.columns else None,
        )
    else:
        smiles2 = pd.Series([""] * len(df), index=df.index, dtype="string")
    if "smiles2" in df.columns:
        df["smiles2"] = smiles2

    combined_smiles = [merged_smiles_for_double(a, b) for a, b in zip(smiles1.tolist(), smiles2.tolist())]
    if "smiles" in df.columns:
        existing_smiles = clean_nullable_string(df["smiles"])
        df["smiles"] = [new if new else old for new, old in zip(combined_smiles, existing_smiles.tolist())]

    df.to_csv(info_path, index=False)
    task.pert_smiles_map = {
        normalize_free_text(pert_id): canonical_smiles.get(normalize_free_text(pert_id), normalize_free_text(smiles))
        for pert_id, smiles in task.pert_smiles_map.items()
        if normalize_free_text(pert_id)
    }


def apply_canonical_smiles(task_results: list[TaskResult]) -> None:
    grouped: dict[str, list[TaskResult]] = defaultdict(list)
    for task in task_results:
        grouped[task.dataset_group].append(task)
    for group_results in grouped.values():
        canonical_smiles = build_canonical_smiles_map(group_results)
        for task in group_results:
            rewrite_task_info_smiles(task, canonical_smiles)


def standardize_main_singledrug(task_dir: Path) -> TaskResult:
    info_path = RAW_ROOT / "singledrug" / "20260403_ptv3_v2_bind_bio_sampleID_machineID_details.csv"
    expr_path = RAW_ROOT / "singledrug" / "20250113_ptv3_unique_mat_28602samp_10982prot_finall_v2.csv"

    info_raw = safe_read_csv(info_path, low_memory=False)
    expr_ids = pd.read_csv(expr_path, usecols=["samp_ID"], low_memory=False)["samp_ID"].astype(str)
    sample_ids = set(expr_ids.tolist())

    info_raw["sample_id"] = info_raw["sample_id"].astype(str)
    validate_unique_sample_ids(info_raw[["sample_id"]].copy(), "ptv3_main_singledrug_raw")

    control_series, control_status, _ = normalize_control_column(
        info_raw["control"],
        row_sample_ids=info_raw["sample_id"],
        valid_sample_ids=sample_ids,
    )

    standard = default_standard_frame(info_raw["sample_id"])
    standard["machineID_new"] = clean_nullable_string(info_raw["machineID_new"])
    standard["Cell_plate"] = clean_nullable_string(info_raw["Cell_plate"])
    standard["Cell"] = clean_nullable_string(info_raw["Cell"])
    standard["cell_type"] = clean_nullable_string(info_raw["cell_type"])
    standard["pert_id1"] = clean_nullable_string(info_raw["pert_id"].astype("string"))
    standard["pert_id2"] = standard["pert_id1"]
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
    resolved_protein_columns, protein_order, unresolved_protein_columns, protein_rule = resolve_expression_columns(
        task_name="ptv3_main_singledrug",
        expr_path=expr_path,
        protein_columns=protein_columns,
    )
    expression = write_expression_outputs(
        task_name="ptv3_main_singledrug",
        expr_path=expr_path,
        sample_id_col="samp_ID",
        protein_columns=resolved_protein_columns,
        protein_order=protein_order,
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
            "pert_id2": "copied from pert_id1 for single-drug two-slot model input",
            "pert_time": "pert_time",
            "pert_dose1": "pert_dose",
            "instrument": "instrument",
            "cell_pertid_time": "cell_pertid_time",
            "drugname": "drugname",
            "smiles": "Smiles_with_chiral > smiles > Smiles_no_chiral",
            "target_protein_list": "targetv2",
            "control": "raw `control` is normalized so `control` or `sample_id` both mark self-control rows; valid foreign sample_id references are kept",
        },
        "protein_name_rule": protein_rule,
        "special_rules": [
            "sample info csv is read with multi-encoding fallback for robustness",
            "control values equal to `control` or the row sample_id are normalized into self-control rows",
            "control values that are not valid sample_id references are blanked in the standardized control column and preserved in control_raw",
        ],
        "issues": [
            {
                "kind": "unresolved_control_reference",
                "count": int((standard["control_status"] == "unresolved_non_sample_id").sum()),
            },
            {
                "kind": "unresolved_protein_columns",
                "count": len(unresolved_protein_columns),
            },
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
    info_path = resolve_existing_path(
        RAW_ROOT / "doubledrug" / "20260422ptv3_J_3496_sampinfo_final.csv",
        RAW_ROOT / "doubledrug" / "20260417ptv3_J_3509sampinfo.csv",
        RAW_ROOT / "doubledrug" / "20260414ptv3_J_3549sampinfo_check_prism1_label_add_prism2_label_add_machineID_detail.csv",
    )
    expr_path = resolve_existing_path(
        RAW_ROOT / "doubledrug" / "20260422ptv3_J_3496samp_9202prot_final_edit.csv",
        RAW_ROOT / "doubledrug" / "20260417ptv3_J_3509samp_9112prot_finall_edit.csv",
        RAW_ROOT / "doubledrug" / "20250211_ptv3_J_3549samp_9205prot_finall_edit.csv",
    )

    info_raw = pd.read_csv(info_path, low_memory=False)
    expr_header = pd.read_csv(expr_path, nrows=0).columns.tolist()
    expr_sample_id_col = resolve_header_column(expr_header, ("sample_id", "samp_id"))
    expr_ids = pd.read_csv(expr_path, usecols=[expr_sample_id_col], low_memory=False)[expr_sample_id_col].astype(str)
    sample_ids = set(expr_ids.tolist())

    control_series, control_status, _ = normalize_control_column(
        info_raw["control"],
        row_sample_ids=info_raw["sample_id"].astype(str),
        valid_sample_ids=sample_ids,
    )
    double_self_control_mask = (
        clean_nullable_string(info_raw["control"]).eq("")
        & clean_nullable_string(info_raw["sample_id"].astype("string")).isin(sample_ids)
        & clean_nullable_string(info_raw["pert_id1"]).str.lower().eq("control")
        & clean_nullable_string(info_raw["pert_id2"]).str.lower().eq("control")
    )
    control_series = control_series.copy()
    control_status = control_status.copy()
    control_series.loc[double_self_control_mask] = clean_nullable_string(
        info_raw.loc[double_self_control_mask, "sample_id"].astype("string")
    )
    control_status.loc[double_self_control_mask] = "self_control_double_control_row"

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
    raw_smiles1 = info_raw.apply(
        lambda row: choose_first_non_empty(row.get("Smiles1_with_chiral"), row.get("Smiles1_no_chiral")),
        axis=1,
    )
    raw_smiles2 = info_raw.apply(
        lambda row: choose_first_non_empty(row.get("Smiles2_with_chiral"), row.get("Smiles2_no_chiral")),
        axis=1,
    )
    smiles1 = pd.Series(
        [
            choose_first_non_empty(smiles_map.get(normalize_free_text(pert_id), ""), raw_smiles)
            for pert_id, raw_smiles in zip(info_raw["pert_id1"], raw_smiles1)
        ],
        index=info_raw.index,
        dtype="string",
    )
    smiles2 = pd.Series(
        [
            choose_first_non_empty(smiles_map.get(normalize_free_text(pert_id), ""), raw_smiles)
            for pert_id, raw_smiles in zip(info_raw["pert_id2"], raw_smiles2)
        ],
        index=info_raw.index,
        dtype="string",
    )
    target1 = info_raw["pert_id1"].map(lambda value: target_map.get(normalize_free_text(value), []))
    target2 = info_raw["pert_id2"].map(lambda value: target_map.get(normalize_free_text(value), []))
    standard["smiles"] = [merged_smiles_for_double(a, b) for a, b in zip(smiles1, smiles2)]
    standard["target_protein_list"] = [merged_targets_for_double(a, b) for a, b in zip(target1, target2)]
    standard["control"] = control_series
    standard["synergy"] = clean_nullable_string(info_raw["synergy"])

    standard["smiles1"] = smiles1
    standard["smiles2"] = smiles2
    standard["smiles1_raw"] = raw_smiles1
    standard["smiles2_raw"] = raw_smiles2
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

    protein_columns = [column for column in expr_header if column != expr_sample_id_col]
    resolved_protein_columns, protein_order, unresolved_protein_columns, protein_rule = resolve_expression_columns(
        task_name="ptv3_main_doubledrug",
        expr_path=expr_path,
        protein_columns=protein_columns,
    )
    expression = write_expression_outputs(
        task_name="ptv3_main_doubledrug",
        expr_path=expr_path,
        sample_id_col=expr_sample_id_col,
        protein_columns=resolved_protein_columns,
        protein_order=protein_order,
        info_df=standard,
        output_dir=task_dir,
        encodings=("utf-8",),
    )

    pert_smiles_map: dict[str, str] = {}
    pert_target_map: dict[str, list[str]] = {}
    for row_index, row in info_raw.iterrows():
        for side, raw_smiles in (
            ("pert_id1", raw_smiles1.loc[row_index]),
            ("pert_id2", raw_smiles2.loc[row_index]),
        ):
            pert_id = normalize_free_text(row.get(side))
            if not pert_id:
                continue
            resolved_smiles = choose_first_non_empty(smiles_map.get(pert_id, ""), raw_smiles)
            if resolved_smiles:
                pert_smiles_map.setdefault(pert_id, resolved_smiles)
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
            "smiles": "resolved from ptv3_main_singledrug pert_id -> smiles map with raw Smiles*_with_chiral / Smiles*_no_chiral fallback",
            "target_protein_list": "union of side-specific target maps",
        },
        "protein_name_rule": protein_rule,
        "special_rules": [
            "double-drug smiles and targets are backfilled from the main single-drug pert_id registry; side-specific raw smiles are retained as fallback for new double-drug raw schemas",
            "rows whose sample_id is a real expression row and whose pert_id1 / pert_id2 are both `control` are normalized as self-control rows even when raw control is blank",
            "cell_pertid_time is unavailable in the raw double-drug table and is left blank",
            "control values equal to `control` or the row sample_id are normalized into self-control rows",
        ],
        "issues": [
            {
                "kind": "unresolved_control_reference",
                "count": int((standard["control_status"] == "unresolved_non_sample_id").sum()),
            },
            {
                "kind": "unresolved_protein_columns",
                "count": len(unresolved_protein_columns),
            },
            {
                "kind": "double_self_control_rows_from_blank_control",
                "count": int(double_self_control_mask.sum()),
            },
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
    dropped_missing_info_sample_ids = merged.loc[merged["_merge"].eq("left_only"), "sample_id"].astype(str).tolist()
    merged = merged.loc[merged["_merge"].eq("both")].copy()

    standard = default_standard_frame(merged["sample_id"])
    standard["machineID_new"] = clean_nullable_string(merged["machineID_new"])
    standard["Cell_plate"] = clean_nullable_string(merged["Cell_plate"])
    standard["Cell"] = clean_nullable_string(merged["Cell"])
    standard["cell_type"] = clean_nullable_string(merged["cell_type"])
    standard["pert_id1"] = "control"
    standard["pert_id2"] = "control"
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
    resolved_protein_columns, protein_order, unresolved_protein_columns, protein_rule = resolve_expression_columns(
        task_name="ptv3_extra_baseline",
        expr_path=expr_path,
        protein_columns=protein_columns,
    )
    expression = write_expression_outputs(
        task_name="ptv3_extra_baseline",
        expr_path=expr_path,
        sample_id_col="sample_id",
        protein_columns=resolved_protein_columns,
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
        "protein_name_rule": protein_rule,
        "special_rules": [
            "all extra baseline rows are treated as control candidates",
            "expression rows without matching info rows are dropped before standardization",
        ],
        "issues": [
            {
                "kind": "info_missing_in_raw_file_dropped",
                "count": len(dropped_missing_info_sample_ids),
                "sample_ids": dropped_missing_info_sample_ids,
            },
            {
                "kind": "unresolved_protein_columns",
                "count": len(unresolved_protein_columns),
            },
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
    return build_unified_external_pert_id(
        explicit_id=row.get("drug_ID"),
        smiles=choose_first_non_empty(row.get("Smiles_with_chiral"), row.get("smiles"), row.get("Smiles_no_chiral")),
        name=row.get("drug_name"),
    )


def standardize_extra_single_task(
    *,
    task_name: str,
    file_name: str,
    task_dir: Path,
    control_pool: pd.DataFrame,
    smiles_to_pert: dict[str, str],
    name_to_pert: dict[str, str],
    target_maps: dict[str, dict[str, list[str]]],
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
    standard["pert_id2"] = standard["pert_id1"]
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
    target_lists = raw.apply(
        lambda row: resolve_extra_target_protein_list(
            raw_target=row.get("target"),
            drug_names=(row.get("drug_name"),),
            target_maps=target_maps,
        ),
        axis=1,
    )
    standard["target_protein_list"] = target_lists
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
    pert_target_map: dict[str, list[str]] = {}
    pert_target_text_map: dict[str, str] = {}
    for row in standard.itertuples(index=False):
        pert_id = normalize_free_text(getattr(row, "pert_id1"))
        if not pert_id:
            continue
        pert_smiles_map.setdefault(pert_id, normalize_free_text(getattr(row, "smiles")))
        targets = json.loads(getattr(row, "target_protein_list"))
        if pert_id not in pert_target_map:
            pert_target_map[pert_id] = []
        append_unique(pert_target_map[pert_id], targets)
        raw_target = normalize_free_text(getattr(row, "target_raw"))
        if raw_target:
            pert_target_text_map.setdefault(pert_id, raw_target)

    audit = {
        "raw_files": [str(info_path.relative_to(REPO_ROOT))],
        "category": "ptv3_extra_singledrug",
        "table_kinds": {str(info_path.relative_to(REPO_ROOT)): "sample_info_table"},
        "column_mapping": {
            "sample_id": "generated_from_task_name_and_row_index",
            "pert_id1": "existing pert_id by smiles/name > unified external id derived from raw drug_ID / smiles / drug_name",
            "pert_id2": "copied from pert_id1 for single-drug two-slot model input",
            "PRISM2nd_label_total": "PRISM2nd_label_total",
            "drugname": "drug_name",
            "smiles": "Smiles_with_chiral > smiles > Smiles_no_chiral",
            "control": "matched from ptv3 control pool",
        },
        "special_rules": [
            "this task has no raw perturbation proteome matrix in the checkout, so an empty expression structure is emitted",
            "control matching is audited with level / score / source task columns",
            "unmapped perturbations use deterministic unified ids with prefixes `extid::`, `extsmiles::`, `extname::`, or `extunk::`",
            "target_protein_list is resolved from the extra target mapping file using raw target gene text plus PRISM drug-name lookup",
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
            {
                "kind": "unresolved_target_rows",
                "count": int(
                    (
                        (
                            standard["target_raw"].fillna("").ne("")
                            | standard["drugname"].fillna("").ne("")
                        )
                        & standard["target_protein_list"].eq("[]")
                    ).sum()
                ),
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
        pert_target_map=pert_target_map,
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
            return build_unified_external_pert_id(
                explicit_id=explicit,
                smiles=choose_first_non_empty(
                    row.get(f"{side_prefix.capitalize()}_with_chiral"),
                    row.get(side_prefix),
                    row.get(f"{side_prefix.capitalize()}_no_chiral"),
                ),
                name=choose_first_non_empty(*(row.get(column) for column in name_columns)),
            )
    return build_unified_external_pert_id(
        explicit_id="",
        smiles=choose_first_non_empty(
            row.get(f"{side_prefix.capitalize()}_with_chiral"),
            row.get(side_prefix),
            row.get(f"{side_prefix.capitalize()}_no_chiral"),
        ),
        name=choose_first_non_empty(*(row.get(column) for column in name_columns)),
    )


def standardize_extra_double_task(
    *,
    task_name: str,
    file_name: str,
    task_dir: Path,
    control_pool: pd.DataFrame,
    smiles_to_pert: dict[str, str],
    name_to_pert: dict[str, str],
    target_maps: dict[str, dict[str, list[str]]],
    file_kind: str,
) -> TaskResult:
    if file_kind == "guomics":
        info_path = resolve_existing_path(
            RAW_ROOT / "extra_doubeldrug" / "260423ptv3_Guomics_drug_combo_unique_with_smlies.csv",
            RAW_ROOT / "extra_doubeldrug" / "260417ptv3_Guomics_drug_combo_unique_with_smlies.csv",
            RAW_ROOT / "extra_doubeldrug" / "20260410ptv3_Guomics_drug_combo_vali_unique.csv",
        )
    elif file_kind == "nc":
        info_path = resolve_existing_path(
            RAW_ROOT / "extra_doubeldrug" / "260424nc_drugComb_info_unique_with_smiles.csv",
            RAW_ROOT / "extra_doubeldrug" / file_name,
            RAW_ROOT / "extra_doubeldrug" / "20260411NC_combo_info_unique.csv",
        )
    elif file_kind == "nature":
        info_path = resolve_existing_path(
            RAW_ROOT / "extra_doubeldrug" / "260424nature_drugComb_info_unique_with_smiles.csv",
            RAW_ROOT / "extra_doubeldrug" / file_name,
            RAW_ROOT / "extra_doubeldrug" / "20260411nature_drugComb_info_unique.csv",
        )
    else:
        info_path = RAW_ROOT / "extra_doubeldrug" / file_name
    raw = pd.read_csv(info_path, low_memory=False)

    if file_kind == "guomics":
        resolved_1 = raw.apply(
            lambda row: resolve_double_pert_id(
                side_prefix="smiles1",
                row=row,
                smiles_to_pert=smiles_to_pert,
                name_to_pert=name_to_pert,
                explicit_id_columns=("pert_id1",),
                name_columns=("Anchor_name",),
            ),
            axis=1,
        )
        resolved_2 = raw.apply(
            lambda row: resolve_double_pert_id(
                side_prefix="smiles2",
                row=row,
                smiles_to_pert=smiles_to_pert,
                name_to_pert=name_to_pert,
                explicit_id_columns=("pert_id2",),
                name_columns=("Library_name",),
            ),
            axis=1,
        )
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
    target1_lists = [
        resolve_extra_target_protein_list(
            raw_target=choose_first_non_empty(
                row.get("anchor_Primary_Target"),
                row.get("Anchor.Target"),
            ),
            drug_names=(choose_first_non_empty(*(row.get(column) for column in name_columns_1)),),
            target_maps=target_maps,
        )
        for _, row in raw.iterrows()
    ]
    target2_lists = [
        resolve_extra_target_protein_list(
            raw_target=choose_first_non_empty(
                row.get("library_Primary_Target"),
                row.get("library.Target"),
            ),
            drug_names=(choose_first_non_empty(*(row.get(column) for column in name_columns_2)),),
            target_maps=target_maps,
        )
        for _, row in raw.iterrows()
    ]
    standard["smiles"] = [merged_smiles_for_double(a, b) for a, b in zip(smiles1, smiles2)]
    standard["target_protein_list"] = [merged_targets_for_double(a, b) for a, b in zip(target1_lists, target2_lists)]
    standard["control"] = ""
    standard["synergy"] = clean_nullable_string(raw["synergy"])

    standard["smiles1"] = smiles1
    standard["smiles2"] = smiles2
    for raw_smiles_column in (
        "smiles1",
        "smiles2",
        "Smiles1_no_chiral",
        "Smiles1_with_chiral",
        "Smiles2_no_chiral",
        "Smiles2_with_chiral",
    ):
        if raw_smiles_column in raw.columns:
            standard[f"{raw_smiles_column}_raw"] = clean_nullable_string(raw[raw_smiles_column])
    standard["target_protein_list1"] = target1_lists
    standard["target_protein_list2"] = target2_lists
    standard["pert_id1_resolution"] = [item[1] for item in resolved_1]
    standard["pert_id2_resolution"] = [item[1] for item in resolved_2]
    if file_kind == "nc":
        standard["target1_raw"] = clean_nullable_string(raw["anchor_Primary_Target"])
        standard["target2_raw"] = clean_nullable_string(raw["library_Primary_Target"])
        for raw_column in ("anchor_lib", "group", "group1", "Cell2"):
            if raw_column in raw.columns:
                standard[f"{raw_column}_raw"] = clean_nullable_string(raw[raw_column])
    elif file_kind == "nature":
        standard["target1_raw"] = clean_nullable_string(raw["Anchor.Target"])
        standard["target2_raw"] = clean_nullable_string(raw["library.Target"])
        standard["anchor_ID_raw"] = clean_nullable_string(raw["anchor_ID"])
        standard["lib_ID_raw"] = clean_nullable_string(raw["lib_ID"])
        for raw_column in ("Tissue", "Cancer.Type", "Anchor.Pathway", "Library.Pathway", "Synergy?"):
            if raw_column in raw.columns:
                standard[f"{raw_column}_raw"] = clean_nullable_string(raw[raw_column])
    else:
        standard["target1_raw"] = ""
        standard["target2_raw"] = ""
        if "Library_Primary.Pathway" in raw.columns:
            standard["library_pathway_raw"] = clean_nullable_string(raw["Library_Primary.Pathway"])
    standard["source_file_info"] = str(info_path.relative_to(REPO_ROOT))
    standard["source_row_index"] = raw.index
    standard["expression_available"] = False

    standard = match_controls(control_pool, standard)
    standard = jsonize_target_columns(standard, ("target_protein_list", "target_protein_list1", "target_protein_list2"))
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
    pert_target_map: dict[str, list[str]] = {}
    pert_target_text_map: dict[str, str] = {}
    for row in standard.itertuples(index=False):
        pert1 = normalize_free_text(getattr(row, "pert_id1"))
        pert2 = normalize_free_text(getattr(row, "pert_id2"))
        smiles_1 = normalize_free_text(getattr(row, "smiles1"))
        smiles_2 = normalize_free_text(getattr(row, "smiles2"))
        if pert1:
            pert_smiles_map.setdefault(pert1, smiles_1)
            pert_target_map.setdefault(pert1, [])
            append_unique(pert_target_map[pert1], json.loads(getattr(row, "target_protein_list1")))
            target1 = normalize_free_text(getattr(row, "target1_raw", ""))
            if target1:
                pert_target_text_map.setdefault(pert1, target1)
        if pert2:
            pert_smiles_map.setdefault(pert2, smiles_2)
            pert_target_map.setdefault(pert2, [])
            append_unique(pert_target_map[pert2], json.loads(getattr(row, "target_protein_list2")))
            target2 = normalize_free_text(getattr(row, "target2_raw", ""))
            if target2:
                pert_target_text_map.setdefault(pert2, target2)

    audit = {
        "raw_files": [str(info_path.relative_to(REPO_ROOT))],
        "category": "ptv3_extra_doubledrug",
        "table_kinds": {str(info_path.relative_to(REPO_ROOT)): "sample_info_table"},
        "column_mapping": {
            "sample_id": "generated_from_task_name_and_row_index",
            "pert_id1": "existing pert_id by smiles/name > unified external id derived from explicit id / smiles / name",
            "pert_id2": "existing pert_id by smiles/name > unified external id derived from explicit id / smiles / name",
            "drugname": f"{name_columns_1[0]} || {name_columns_2[0]}",
            "smiles": "Smiles*_with_chiral > smiles* > Smiles*_no_chiral",
            "control": "matched from ptv3 control pool",
        },
        "special_rules": [
            "this task has no raw perturbation proteome matrix in the checkout, so an empty expression structure is emitted",
            "synergy is preserved as the raw category label from the source csv",
            "unmapped perturbations use deterministic unified ids with prefixes `extid::`, `extsmiles::`, `extname::`, or `extunk::`",
            "target_protein_list is resolved from the extra target mapping file using raw target gene text plus PRISM drug-name lookup",
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
            {
                "kind": "unresolved_target_rows",
                "count": int(
                    (
                        (
                            standard["target1_raw"].fillna("").ne("")
                            | standard["target2_raw"].fillna("").ne("")
                            | standard["drugname"].fillna("").ne("")
                        )
                        & standard["target_protein_list"].eq("[]")
                    ).sum()
                ),
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
        pert_target_map=pert_target_map,
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


def build_ptv1_control_lookup(
    df: pd.DataFrame,
    *,
    group_columns: list[str],
) -> tuple[dict[tuple[str, ...], str], dict[tuple[str, ...], int]]:
    controls = df.loc[df["pert_time"].eq(0), ["sample_id", *group_columns]].copy()
    controls["sample_id"] = clean_nullable_string(controls["sample_id"])
    for column in group_columns:
        controls[column] = clean_nullable_string(controls[column])
    controls = controls.loc[controls["sample_id"].ne("")]
    controls.sort_values(by=[*group_columns, "sample_id"], kind="mergesort", inplace=True)

    representative: dict[tuple[str, ...], str] = {}
    counts: dict[tuple[str, ...], int] = {}
    for key, group in controls.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        representative[key] = group["sample_id"].iloc[0]
        counts[key] = len(group)
    return representative, counts


def build_ptv1_control_assignment(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    result["control"] = pd.Series([""] * len(df), index=df.index, dtype="string")
    result["control_candidate_count"] = 0
    result["control_resolution"] = "unmatched_biorep_cell_plate"

    lookup, counts = build_ptv1_control_lookup(df, group_columns=["BioRep", "Cell_plate"])
    mask_is_control = df["pert_time"].eq(0)
    result.loc[mask_is_control, "control"] = df.loc[mask_is_control, "sample_id"].astype(str)
    result.loc[mask_is_control, "control_resolution"] = "self_control_by_pert_time_zero"

    keys = [
        (
            normalize_free_text(bio_rep),
            normalize_free_text(cell_plate),
        )
        for bio_rep, cell_plate in zip(df["BioRep"].tolist(), df["Cell_plate"].tolist())
    ]
    result["control_candidate_count"] = [counts.get(key, 0) for key in keys]

    mapped_control = [lookup.get(key, "") for key in keys]
    non_control_mask = ~mask_is_control
    result.loc[non_control_mask, "control"] = pd.Series(mapped_control, index=df.index).loc[non_control_mask]
    matched_non_control = non_control_mask & result["control"].astype("string").fillna("").ne("")
    result.loc[matched_non_control, "control_resolution"] = "matched_by_biorep_cell_plate"
    result["control_candidate_count"] = result["control_candidate_count"].astype(int)
    return result


def select_ptv1_extra_prediction_rows(raw: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["cell", "E115_id"]
    selected = raw.copy()
    selected["sample_name"] = clean_nullable_string(selected["sample_name"])
    selected["cell"] = clean_nullable_string(selected["cell"])
    selected["drug_cid"] = clean_nullable_string(selected["drug_cid"])
    selected["E115_id"] = clean_nullable_string(selected["E115_id"])
    selected["ground_truth"] = clean_nullable_string(selected["ground_truth"])
    selected["model"] = clean_nullable_string(selected["model"])
    selected["model_category"] = clean_nullable_string(selected["model_category"])
    selected["__preferred_model"] = selected["model"].eq("ppODE_swa1")

    aggregates = (
        selected.groupby(key_columns, dropna=False)
        .agg(
            prediction_row_count=("model", "size"),
            has_ppode_swa1=("__preferred_model", "any"),
            unique_ground_truth_count=("ground_truth", lambda values: len({value for value in values if value})),
            unique_ground_truth_values=(
                "ground_truth",
                lambda values: json_list_string(sorted({value for value in values if value})),
            ),
        )
        .reset_index()
    )

    selected.sort_values(
        by=[*key_columns, "__preferred_model", "sample_name", "drug_cid", "model_category", "model"],
        ascending=[True, True, False, True, True, True, True],
        kind="mergesort",
        inplace=True,
    )
    selected = selected.drop_duplicates(subset=key_columns, keep="first").copy()
    selected = selected.merge(aggregates, on=key_columns, how="left")
    selected["ground_truth_conflict"] = selected["unique_ground_truth_count"].fillna(0).astype(int).gt(1)
    return selected.reset_index(drop=True)


def match_ptv1_extra_controls(main_ptv1_info: pd.DataFrame, query_df: pd.DataFrame) -> pd.DataFrame:
    controls = main_ptv1_info.loc[
        clean_nullable_string(main_ptv1_info["control"]).eq(clean_nullable_string(main_ptv1_info["sample_id"])),
        ["sample_id", "Cell_plate"],
    ].copy()
    controls["sample_id"] = clean_nullable_string(controls["sample_id"])
    controls["Cell_plate"] = clean_nullable_string(controls["Cell_plate"])
    controls["__norm_cell_plate"] = controls["Cell_plate"].map(normalize_lookup_text)
    controls = controls.loc[controls["__norm_cell_plate"].ne("")].copy()
    controls.sort_values(by=["__norm_cell_plate", "sample_id"], kind="mergesort", inplace=True)

    representative = controls.drop_duplicates(subset="__norm_cell_plate", keep="first")
    matched_sample_map = representative.set_index("__norm_cell_plate")["sample_id"].to_dict()
    candidate_counts = controls.groupby("__norm_cell_plate").size().to_dict()

    result = query_df.copy()
    query_norm = result["Cell"].map(normalize_lookup_text)
    result["control"] = query_norm.map(lambda value: matched_sample_map.get(value, ""))
    matched_mask = result["control"].astype("string").fillna("").ne("")
    result["control_match_level"] = np.where(matched_mask, "cell_plate", "no_cell_plate_match")
    result["control_match_source_task"] = np.where(matched_mask, "ptv1_aivc", "")
    result["control_match_pool_kind"] = np.where(matched_mask, "ptv1_main_cell_plate_control", "")
    result["control_match_score"] = np.where(matched_mask, 1, 0).astype(int)
    result["control_candidate_count"] = [int(candidate_counts.get(value, 0)) for value in query_norm.tolist()]
    return result


def standardize_ptv1(task_dir: Path) -> TaskResult:
    mixed_path = RAW_ROOT / "ptv1" / "aivc.csv"
    info_path = RAW_ROOT / "ptv1" / "aivc_info.csv"
    drug_meta_path = RAW_ROOT / "ptv1" / "ptv1.csv"
    split_dir = RAW_ROOT / "ptv1" / "experiment_type_list"

    info_raw = pd.read_csv(info_path, low_memory=False)
    mixed_header = pd.read_csv(mixed_path, nrows=0).columns.tolist()
    info_start_idx = mixed_header.index("Library_dose")
    protein_columns = mixed_header[1:info_start_idx]
    resolved_protein_columns, protein_order, unresolved_protein_columns, protein_rule = resolve_expression_columns(
        task_name="ptv1_aivc",
        expr_path=mixed_path,
        protein_columns=protein_columns,
    )

    info_raw["sample_id"] = info_raw["Sample_ID"].astype(str)
    info_raw["pert_id_clean"] = clean_placeholder_string(info_raw["pert_id"])
    info_raw["Anchor_id_clean"] = clean_placeholder_string(info_raw["Anchor_id"])
    standard = default_standard_frame(info_raw["sample_id"])
    standard["machineID_new"] = clean_nullable_string(info_raw["machine"])
    standard["Cell_plate"] = clean_nullable_string(info_raw["protein_plate"])
    standard["Cell"] = info_raw.apply(
        lambda row: choose_first_non_empty(row.get("Cell.Line.name"), row.get("protein_plate")),
        axis=1,
    )
    standard["cell_type"] = ""
    standard["pert_id1"] = info_raw["pert_id_clean"]
    standard["pert_id2"] = info_raw["Anchor_id_clean"]
    standard = copy_pert_id1_to_blank_pert_id2(standard)
    standard["batch"] = "no"
    standard["pert_time"] = clean_numeric(info_raw["pert_time"])
    library_dose = clean_numeric(info_raw["Library_dose"])
    anchor_dose = clean_numeric(info_raw["Anchor_dose"])
    control_mask = standard["pert_time"].eq(0)
    standard["pert_dose1"] = library_dose.where(~control_mask, library_dose.fillna(0))
    standard["pert_dose2"] = anchor_dose.where(~control_mask, anchor_dose.fillna(0))
    standard["PRISM1st_label_total"] = clean_nullable_string(info_raw["NY_label"])
    standard["PRISM2nd_label_total"] = ""
    standard["instrument"] = clean_nullable_string(info_raw["machine"])
    standard["cell_pertid_time"] = ""
    standard["drugname"] = [
        choose_first_non_empty(drug_name_ab, pert_name, anchor_name)
        for drug_name_ab, pert_name, anchor_name in zip(
            clean_placeholder_string(info_raw["drugNameAB"]).tolist(),
            clean_nullable_string(info_raw["pert_iname"]).tolist(),
            clean_nullable_string(info_raw["Anchor_iname"]).tolist(),
        )
    ]
    standard["smiles"] = ""
    standard["target_protein_list"] = [[] for _ in range(len(info_raw))]
    standard["synergy"] = clean_nullable_string(info_raw["Synergy"])
    standard["BioRep"] = clean_nullable_string(info_raw["BioRep"])

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

    smiles1 = standard["pert_id1"].map(lambda value: ptv1_smiles_map.get(value, ""))
    smiles2 = standard["pert_id2"].map(lambda value: ptv1_smiles_map.get(value, ""))
    target1 = standard["pert_id1"].map(lambda value: ptv1_target_map.get(value, []))
    target2 = standard["pert_id2"].map(lambda value: ptv1_target_map.get(value, []))
    standard["smiles"] = [merged_smiles_for_double(a, b) for a, b in zip(smiles1.tolist(), smiles2.tolist())]
    standard["target_protein_list"] = [
        merged_targets_for_double(a, b) for a, b in zip(target1.tolist(), target2.tolist())
    ]
    control_assignment = build_ptv1_control_assignment(
        pd.DataFrame(
            {
                "sample_id": standard["sample_id"],
                "pert_time": standard["pert_time"],
                "BioRep": standard["BioRep"],
                "Cell_plate": standard["Cell_plate"],
            }
        )
    )
    standard["control"] = control_assignment["control"]
    standard["control_candidate_count"] = control_assignment["control_candidate_count"]
    standard["control_resolution"] = control_assignment["control_resolution"]
    standard["smiles1"] = smiles1
    standard["smiles2"] = smiles2
    standard["target_protein_list1"] = target1
    standard["target_protein_list2"] = target2

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

    standard = jsonize_target_columns(
        standard,
        ("target_protein_list", "target_protein_list1", "target_protein_list2"),
    )
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
    _, control_candidate_counts = build_ptv1_control_lookup(
        pd.DataFrame(
            {
                "sample_id": standard["sample_id"],
                "pert_time": standard["pert_time"],
                "BioRep": standard["BioRep"],
                "Cell_plate": standard["Cell_plate"],
            }
        ),
        group_columns=["BioRep", "Cell_plate"],
    )
    ambiguous_control_groups = sum(1 for count in control_candidate_counts.values() if count > 1)

    audit = {
        "raw_files": [
            str(mixed_path.relative_to(REPO_ROOT)),
            str(info_path.relative_to(REPO_ROOT)),
            str(drug_meta_path.relative_to(REPO_ROOT)),
            str((split_dir / "train_experiment_type_list.txt").relative_to(REPO_ROOT)),
            str((split_dir / "val_experiment_type_list.txt").relative_to(REPO_ROOT)),
            str((split_dir / "test_experiment_type_list.txt").relative_to(REPO_ROOT)),
        ],
        "category": "ptv1",
        "table_kinds": {
            str(mixed_path.relative_to(REPO_ROOT)): "mixed_expression_and_info_table",
            str(info_path.relative_to(REPO_ROOT)): "sample_info_table",
            str(drug_meta_path.relative_to(REPO_ROOT)): "drug_metadata_table",
            str((split_dir / "train_experiment_type_list.txt").relative_to(REPO_ROOT)): "split_reference",
            str((split_dir / "val_experiment_type_list.txt").relative_to(REPO_ROOT)): "split_reference",
            str((split_dir / "test_experiment_type_list.txt").relative_to(REPO_ROOT)): "split_reference",
        },
        "column_mapping": {
            "sample_id": "Sample_ID",
            "machineID_new": "machine",
            "Cell_plate": "protein_plate",
            "Cell": "Cell.Line.name with protein_plate fallback",
            "pert_id1": "pert_id with placeholder values blanked",
            "pert_id2": "Anchor_id with placeholder values blanked; blank single-drug slot copied from pert_id1",
            "pert_dose1": "Library_dose",
            "pert_dose2": "Anchor_dose",
            "PRISM1st_label_total": "NY_label",
            "control": "rows with pert_time == 0 are self-controls; perturbed rows match a deterministic representative control sample_id within the same (BioRep, protein_plate) group",
            "data_split": "derived from experiment_type_list using (protein_plate, pert_id)",
        },
        "protein_name_rule": protein_rule,
        "special_rules": [
            "ptv1 is isolated into its own standardized output root and meta index",
            "ptv1 aivc contains both single-drug and anchor-drug rows; standardized smiles and targets are merged across pert_id1 / pert_id2 when both are present",
            "protein parsing uses the first UniProt token embedded in each dot-delimited protein descriptor",
            "unresolved non-UniProt protein columns are excluded from the standardized expression matrix",
        ],
        "issues": [
            {"kind": "unresolved_protein_columns", "count": len(unresolved_protein_columns)},
            {"kind": "ambiguous_control_groups", "count": ambiguous_control_groups},
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
        pert_ids=sorted(
            {
                pert_id
                for column in ("pert_id1", "pert_id2")
                for pert_id in standard[column].astype(str).tolist()
                if pert_id
            }
        ),
        protein_order=protein_order,
        pert_smiles_map=pert_smiles_map,
        pert_target_map=pert_target_map,
        pert_target_text_map={},
        audit=audit,
    )


def standardize_ptv1_extra_singledrug(
    task_dir: Path,
    *,
    main_ptv1_info: pd.DataFrame,
    ptv3_meta: dict[str, object],
) -> TaskResult:
    prediction_path = RAW_ROOT / "ptv1_extra_singledrug" / "test12091214_sample_predictions_E115id.csv"
    e115_map_path = RAW_ROOT / "ptv1_extra_singledrug" / "ptds4_84drug_E115ID.csv"

    raw_prediction = pd.read_csv(prediction_path, low_memory=False)
    selected = select_ptv1_extra_prediction_rows(raw_prediction)
    selected["sample_id"] = selected["sample_name"].astype(str)

    e115_map = pd.read_csv(e115_map_path, low_memory=False)
    e115_map["drug_ID"] = clean_placeholder_string(e115_map["drug_ID"])
    e115_map["cmpdname"] = clean_nullable_string(e115_map["cmpdname"])
    e115_map["cmpdname_in_E115"] = clean_nullable_string(e115_map["cmpdname_in_E115"])
    e115_drug_name_map = {
        drug_id: choose_first_non_empty(cmpdname_in_e115, cmpdname)
        for drug_id, cmpdname, cmpdname_in_e115 in e115_map[["drug_ID", "cmpdname", "cmpdname_in_E115"]].itertuples(index=False)
        if drug_id
    }

    ptv3_smiles_map = {
        normalize_free_text(pert_id): normalize_free_text(smiles)
        for pert_id, smiles in dict(ptv3_meta.get("pertid_to_smiles", {})).items()
    }
    ptv3_target_map = {
        normalize_free_text(pert_id): [
            normalize_free_text(item) for item in items if normalize_free_text(item)
        ]
        for pert_id, items in dict(ptv3_meta.get("pertid_to_target_protein_list", {})).items()
        if normalize_free_text(pert_id)
    }

    standard = default_standard_frame(selected["sample_id"])
    standard["machineID_new"] = ""
    standard["Cell_plate"] = clean_nullable_string(selected["cell"])
    standard["Cell"] = clean_nullable_string(selected["cell"])
    standard["cell_type"] = ""
    standard["pert_id1"] = clean_placeholder_string(selected["E115_id"])
    standard["pert_id2"] = standard["pert_id1"]
    standard["batch"] = "no"
    standard["pert_time"] = np.nan
    standard["pert_dose1"] = np.nan
    standard["pert_dose2"] = np.nan
    standard["PRISM1st_label_total"] = ""
    standard["PRISM2nd_label_total"] = clean_nullable_string(selected["ground_truth"])
    standard["instrument"] = ""
    standard["cell_pertid_time"] = ""
    standard["drugname"] = standard["pert_id1"].map(lambda value: e115_drug_name_map.get(value, ""))
    standard["smiles"] = standard["pert_id1"].map(lambda value: ptv3_smiles_map.get(value, ""))
    standard["target_protein_list"] = standard["pert_id1"].map(lambda value: ptv3_target_map.get(value, []))
    standard["control"] = ""
    standard["synergy"] = np.nan
    standard["data_split"] = "test"

    standard["raw_sample_name"] = clean_nullable_string(selected["sample_name"])
    standard["raw_drug_cid"] = clean_nullable_string(selected["drug_cid"])
    standard["selected_model"] = clean_nullable_string(selected["model"])
    standard["selected_model_category"] = clean_nullable_string(selected["model_category"])
    standard["selected_prediction_score"] = clean_nullable_string(selected["prediction_score"])
    standard["selected_prediction_binary"] = clean_nullable_string(selected["prediction_binary"])
    standard["prediction_row_count"] = selected["prediction_row_count"].fillna(0).astype(int)
    standard["ground_truth_conflict"] = selected["ground_truth_conflict"].astype(bool)
    standard["ground_truth_values"] = selected["unique_ground_truth_values"].fillna("[]")
    standard["source_file_prediction"] = str(prediction_path.relative_to(REPO_ROOT))
    standard["source_file_e115_map"] = str(e115_map_path.relative_to(REPO_ROOT))
    standard["expression_available"] = False

    standard = match_ptv1_extra_controls(main_ptv1_info, standard)
    standard = jsonize_target_columns(standard, ("target_protein_list",))
    standard = ensure_standard_column_order(standard)
    validate_unique_sample_ids(standard, "ptv1_extra_singledrug")

    info_out = task_dir / "info.csv"
    standard.to_csv(info_out, index=False)
    expression = write_empty_expression_outputs(
        task_name="ptv1_extra_singledrug",
        info_df=standard,
        output_dir=task_dir,
        reason="ptv1_extra_singledrug provides labels only; control proteomes are appended from ptv1_aivc during stage 2",
    )

    pert_ids = sorted({pert_id for pert_id in standard["pert_id1"].astype(str).tolist() if pert_id})
    pert_smiles_map = {pert_id: ptv3_smiles_map.get(pert_id, "") for pert_id in pert_ids}
    pert_target_map = {pert_id: ptv3_target_map.get(pert_id, []) for pert_id in pert_ids}

    audit = {
        "raw_files": [
            str(prediction_path.relative_to(REPO_ROOT)),
            str(e115_map_path.relative_to(REPO_ROOT)),
        ],
        "category": "ptv1_extra_singledrug",
        "table_kinds": {
            str(prediction_path.relative_to(REPO_ROOT)): "model_expanded_prediction_table",
            str(e115_map_path.relative_to(REPO_ROOT)): "drug_name_mapping_table",
        },
        "column_mapping": {
            "sample_id": "sample_name after one-row-per-(cell, E115_id) selection",
            "Cell_plate": "cell",
            "Cell": "cell",
            "pert_id1": "E115_id",
            "pert_id2": "copied from pert_id1 for single-drug two-slot model input",
            "PRISM2nd_label_total": "ground_truth",
            "drugname": "cmpdname_in_E115 with cmpdname fallback",
            "smiles": "ptv3 global_meta pertid_to_smiles",
            "target_protein_list": "ptv3 global_meta pertid_to_target_protein_list",
            "control": "matched to ptv1_aivc control sample_ids by exact cell -> Cell_plate",
        },
        "special_rules": [
            "raw predictions are model-expanded; the workflow keeps one row per unique (cell, E115_id) pair and uses the `ppODE_swa1` row when it is present",
            "raw ground_truth disagreement across models is retained only as audit context in `ground_truth_conflict` / `ground_truth_values`; the standardized label comes from the selected row",
            "this task has no perturbation proteome matrix in the checkout, so stage 1 emits an empty expression structure and stage 2 appends matched control rows from ptv1_aivc",
            "ptv1 extra smiles and target lists are resolved from ptv3 global_meta, as required by the ptv1 workflow note",
        ],
        "issues": [
            {"kind": "missing_expression_matrix", "count": len(standard)},
            {"kind": "ground_truth_conflict_rows", "count": int(standard["ground_truth_conflict"].sum())},
            {"kind": "missing_ppode_swa1_row", "count": int((~selected["has_ppode_swa1"].astype(bool)).sum())},
            {"kind": "unmatched_control", "count": int(standard["control"].eq("").sum())},
            {"kind": "missing_ptv3_smiles", "count": int(standard["smiles"].eq("").sum())},
            {"kind": "missing_ptv3_targets", "count": int(standard["target_protein_list"].eq("[]").sum())},
        ],
    }

    return TaskResult(
        task_name="ptv1_extra_singledrug",
        dataset_group="ptv1",
        info_path=str(info_out),
        expression=expression,
        sample_count=len(standard),
        protein_count=0,
        pert_ids=pert_ids,
        protein_order=[],
        pert_smiles_map=pert_smiles_map,
        pert_target_map=pert_target_map,
        pert_target_text_map={},
        audit=audit,
    )


def build_global_meta_payload(dataset_group: str, task_results: list[TaskResult]) -> dict[str, object]:
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

    return {
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


def build_global_meta(dataset_group: str, task_results: list[TaskResult], output_root: Path) -> None:
    payload = build_global_meta_payload(dataset_group, task_results)
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
        "source_document": str((REPO_ROOT / "docs" / "Data_Process.md").relative_to(REPO_ROOT)),
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
    extra_target_maps = build_extra_target_maps(
        RAW_ROOT / "extra_singledrug" / "20260318_prism1st_target_gene_uniprotID_map.csv"
    )

    double_task_dir = ensure_dir(ptv3_tasks_root / "ptv3_main_doubledrug")
    double_result = standardize_main_doubledrug(double_task_dir, main_single_maps)
    task_results.append(double_result)
    main_double_info = pd.read_csv(double_result.info_path, low_memory=False)

    extra_baseline_task_dir = ensure_dir(ptv3_tasks_root / "ptv3_extra_baseline")
    extra_baseline_result = standardize_extra_baseline(extra_baseline_task_dir)
    task_results.append(extra_baseline_result)

    extra_baseline_info = pd.read_csv(extra_baseline_result.info_path, low_memory=False)
    control_pool = build_control_pool(main_single_info, main_double_info, extra_baseline_info)

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
            target_maps=extra_target_maps,
        )
        task_results.append(task_result)

    extra_double_specs = [
        ("ptv3_extra_doubledrug_guomics", "260423ptv3_Guomics_drug_combo_unique_with_smlies.csv", "guomics"),
        ("ptv3_extra_doubledrug_nc", "260424nc_drugComb_info_unique_with_smiles.csv", "nc"),
        ("ptv3_extra_doubledrug_nature", "260424nature_drugComb_info_unique_with_smiles.csv", "nature"),
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
            target_maps=extra_target_maps,
            file_kind=file_kind,
        )
        task_results.append(task_result)

    ptv3_results = [result for result in task_results if result.dataset_group == "ptv3"]
    ptv3_meta_payload = build_global_meta_payload("ptv3", ptv3_results)

    ptv1_task_dir = ensure_dir(ptv1_tasks_root / "ptv1_aivc")
    ptv1_result = standardize_ptv1(ptv1_task_dir)
    ptv1_main_info = pd.read_csv(ptv1_result.info_path, low_memory=False)

    ptv1_extra_task_dir = ensure_dir(ptv1_tasks_root / "ptv1_extra_singledrug")
    ptv1_extra_result = standardize_ptv1_extra_singledrug(
        ptv1_extra_task_dir,
        main_ptv1_info=ptv1_main_info,
        ptv3_meta=ptv3_meta_payload,
    )

    all_results = task_results + [ptv1_result, ptv1_extra_result]
    apply_canonical_smiles(all_results)

    build_global_meta("ptv3", ptv3_results, ptv3_root)
    build_global_meta("ptv1", [ptv1_result, ptv1_extra_result], ptv1_root)

    for result in all_results:
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
