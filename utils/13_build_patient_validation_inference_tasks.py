#!/usr/bin/env python3
"""Build inference-only tasks for the 260605 patient validation datasets.

The patient run keeps checkpoint-sensitive categorical covariates compatible
with the trained PTV3 models, but extends the generative feature space where the
pipeline supports it: new proteins are appended to ``protein_index`` and new
patient Cell labels receive independent ``cell_llm_index`` values.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "rawdata" / "ptv2drug_patientVali260605"
SOURCE_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready_patientVali260605v3"
DATASET_GROUP = "ptv3"
RUN_NAME = "patientVali260605v3"

BATCH_COVARIATE_COLUMNS = {
    "machineID_new": "machineID_new_index",
    "Cell_plate": "Cell_plate_index",
    "Cell": "Cell_index",
    "cell_type": "cell_type_index",
    "batch": "batch_index",
    "pert_time": "pert_time_index",
    "pert_dose1": "pert_dose1_index",
    "pert_dose2": "pert_dose2_index",
}
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

UNIPROT_PATTERN = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z0-9]{3}[0-9]){2}|"
    r"A0A[A-Z0-9]{7})$"
)


@dataclass(frozen=True)
class PatientDataset:
    key: str
    display_name: str
    info_path: Path
    matrix_path: Path
    sample_id_col: str
    combo_col: str
    default_cell_type: str
    expression_scale: str = "raw_abundance"


DATASETS = [
    PatientDataset(
        key="p1_ovarian",
        display_name="P1_guomics_ovarianCancer",
        info_path=RAW_ROOT / "P1_guomics_ovarianCancer" / "260106ZZOVA_proteome_na_ptv3_info.csv",
        matrix_path=RAW_ROOT / "P1_guomics_ovarianCancer" / "ZZOVA_proteome_na_ptv3_mat_20260102-2049.csv",
        sample_id_col="sample_id",
        combo_col="pert_id",
        default_cell_type="OVARY",
    ),
    PatientDataset(
        key="p2_lung2020",
        display_name="P2_lungCancer_2020cell",
        info_path=RAW_ROOT / "P2_lungCancer_2020cell" / "260106lungCancer_cell2020_info.csv",
        matrix_path=RAW_ROOT / "P2_lungCancer_2020cell" / "matrixTable S4B_20260102.csv",
        sample_id_col="sample_id",
        combo_col="pert_id",
        default_cell_type="Lung",
    ),
    PatientDataset(
        key="p3_lung2024",
        display_name="P3_lungCancer_2024cell",
        info_path=RAW_ROOT / "P3_lungCancer_2024cell" / "260106lungCancer_cell2024_info.csv",
        matrix_path=RAW_ROOT / "P3_lungCancer_2024cell" / "260102lungCancer_cell2024_Table S1E.csv",
        sample_id_col="sample_id",
        combo_col="pert_id",
        default_cell_type="Lung",
        expression_scale="log2_relative",
    ),
    PatientDataset(
        key="p4_colon",
        display_name="P4_colon_ref2_cellDis",
        info_path=RAW_ROOT / "P4_colon_ref2_cellDis" / "260106colon_ref2_cellDis_Adjuvant.chemotherapy_info.csv",
        matrix_path=RAW_ROOT / "P4_colon_ref2_cellDis" / "colon_ref2_cellDis_Adjuvant.chemotherapy_mat_20251231-2232.csv",
        sample_id_col="sample_id",
        combo_col="pert_id",
        default_cell_type="Colon",
    ),
    PatientDataset(
        key="p5_breast",
        display_name="P5_guomics_breastCancer",
        info_path=RAW_ROOT / "P5_guomics_breastCancer" / "breast_cancer_guomics_ptv2_484sampInfo260601.csv",
        matrix_path=RAW_ROOT / "P5_guomics_breastCancer" / "260601breast_cancer_guomics_unique_matrix_final.csv",
        sample_id_col="Patient_ID",
        combo_col="Drug_combination",
        default_cell_type="BREAST",
    ),
]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_free_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


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


def infer_cell_type(spec: PatientDataset, row: pd.Series) -> str:
    raw_value = normalize_free_text(row.get("cell_type"))
    if raw_value:
        return raw_value
    return spec.default_cell_type or "no"


def parse_uniprot_column(column_name: str) -> str | None:
    text = str(column_name).strip()
    if UNIPROT_PATTERN.fullmatch(text):
        return text
    for token in text.split("."):
        if UNIPROT_PATTERN.fullmatch(token):
            return token
    return None


def split_combo(value: object) -> list[str]:
    tokens: list[str] = []
    for token in normalize_free_text(value).replace(";", "+").split("+"):
        token = token.strip()
        if not token or token.lower() in {"nan", "none", "null"}:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def collect_patient_feature_space() -> dict[str, Any]:
    proteins: list[str] = []
    unresolved_protein_columns: list[str] = []
    drugs: list[str] = []
    cells: list[str] = []
    cell_types: list[str] = []
    machine_values: list[str] = []
    batch_values: list[str] = []
    for spec in DATASETS:
        info = pd.read_csv(spec.info_path, low_memory=False)
        for value in info[spec.sample_id_col].tolist():
            text = normalize_free_text(value)
            if text and text not in cells:
                cells.append(text)
        for value in info[spec.combo_col].tolist():
            for token in split_combo(value):
                if token not in drugs:
                    drugs.append(token)
        for _, row in info.iterrows():
            cell_type = infer_cell_type(spec, row)
            if cell_type and cell_type not in cell_types:
                cell_types.append(cell_type)
            machine = normalize_free_text(row.get("machineID_new"))
            if machine and machine not in machine_values:
                machine_values.append(machine)
        if spec.key not in batch_values:
            batch_values.append(spec.key)

        header = pd.read_csv(spec.matrix_path, nrows=0, low_memory=False)
        for column in header.columns:
            if column == "sample_id":
                continue
            uniprot = parse_uniprot_column(column)
            if uniprot is None:
                unresolved_protein_columns.append(str(column))
                continue
            if uniprot not in proteins:
                proteins.append(uniprot)
    return {
        "proteins": proteins,
        "unresolved_protein_columns": unresolved_protein_columns,
        "drugs": drugs,
        "cells": cells,
        "cell_types": cell_types,
        "machine_values": machine_values,
        "batch_values": batch_values,
    }


def extend_global_meta_for_patient_run(source_meta: dict[str, Any], feature_space: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = json.loads(json.dumps(source_meta))
    protein_index = {str(key): int(value) for key, value in meta["protein_index"].items()}
    protein_index_to_id = list(meta.get("protein_index_to_id", []))
    if len(protein_index_to_id) < len(protein_index):
        protein_index_to_id = [None] * len(protein_index)
        for protein_id, index in protein_index.items():
            protein_index_to_id[int(index)] = protein_id

    added_proteins: list[str] = []
    for protein_id in feature_space["proteins"]:
        if protein_id in protein_index:
            continue
        protein_index[protein_id] = len(protein_index_to_id)
        protein_index_to_id.append(protein_id)
        added_proteins.append(protein_id)

    meta["protein_index"] = protein_index
    meta["protein_index_to_id"] = protein_index_to_id
    meta["generated_at"] = iso_now()
    meta["source_root"] = str(RAW_ROOT)
    meta["source_training_ready_root"] = str(SOURCE_TRAINING_READY_ROOT)
    meta["patient_validation_extension"] = {
        "run_name": RUN_NAME,
        "source_meta_dataset_group": source_meta.get("dataset_group"),
        "appended_protein_count": len(added_proteins),
        "appended_protein_examples": added_proteins[:50],
        "protein_index_policy": "preserve existing PTV3 indices exactly; append patient-only proteins after the existing axis",
        "categorical_covariate_policy": "do not extend value_to_index for checkpoint-sensitive covariates; batch/machine/unknown Cell categories may map to no",
    }

    value_to_index = meta.get("value_to_index", {})
    missing_drugs = sorted(set(feature_space["drugs"]) - set(meta.get("pert_index", {})))
    missing_cell_types = sorted(
        canonicalize_text_value(value) for value in feature_space["cell_types"] if canonicalize_text_value(value) not in value_to_index.get("cell_type", {})
    )
    missing_cells = sorted(set(feature_space["cells"]) - set(value_to_index.get("Cell", {})))
    missing_machine = sorted(
        canonicalize_text_value(value)
        for value in feature_space["machine_values"]
        if canonicalize_text_value(value) not in value_to_index.get("machineID_new", {})
    )
    missing_batches = sorted(
        canonicalize_text_value(value)
        for value in feature_space["batch_values"]
        if canonicalize_text_value(value) not in value_to_index.get("batch", {})
    )
    audit = {
        "raw_unique_proteins": len(set(feature_space["proteins"])),
        "appended_protein_count": len(added_proteins),
        "appended_protein_examples": added_proteins[:50],
        "unresolved_protein_column_count": len(set(feature_space["unresolved_protein_columns"])),
        "unresolved_protein_column_examples": sorted(set(feature_space["unresolved_protein_columns"]))[:20],
        "raw_unique_drugs": len(set(feature_space["drugs"])),
        "missing_drug_count": len(missing_drugs),
        "missing_drug_examples": missing_drugs[:50],
        "raw_unique_cells": len(set(feature_space["cells"])),
        "new_cell_label_count_for_llm": len(missing_cells),
        "new_cell_label_examples": missing_cells[:50],
        "raw_cell_types": sorted(set(feature_space["cell_types"])),
        "missing_cell_type_count": len(missing_cell_types),
        "missing_cell_type_examples": missing_cell_types[:50],
        "machine_values_mapped_to_no_count": len(set(missing_machine)),
        "machine_values_mapped_to_no_examples": sorted(set(missing_machine))[:50],
        "batch_values_mapped_to_no_count": len(set(missing_batches)),
        "batch_values_mapped_to_no_examples": sorted(set(missing_batches))[:50],
    }
    return meta, audit


def safe_id_fragment(value: object) -> str:
    text = normalize_free_text(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "missing"


def combine_target_indices(tokens: list[str], meta: dict[str, Any]) -> str:
    combined: list[int] = []
    for token in tokens:
        for protein_idx in meta.get("pertid_to_target_protein_list", {}).get(token, []):
            try:
                value = int(protein_idx)
            except (TypeError, ValueError):
                continue
            if value not in combined:
                combined.append(value)
    return json.dumps(combined, ensure_ascii=False)


def add_index_columns(df: pd.DataFrame, meta: dict[str, Any], cell_llm_index: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    value_to_index = meta["value_to_index"]
    for field in DISCRETE_FIELDS:
        mapping_key = "pert_dose" if field in {"pert_dose1", "pert_dose2"} else field
        mapping = value_to_index[mapping_key]
        normalized = [canonicalize_discrete_value(field, value) for value in df[field].tolist()]
        df[f"{field}_norm"] = normalized
        df[f"{field}_index"] = [mapping.get(value, mapping["no"]) for value in normalized]

    pert_index = meta["pert_index"]
    no_pert = pert_index["no"]
    df["pert_index1"] = [pert_index.get(normalize_free_text(value) or "no", no_pert) for value in df["pert_id1"]]
    df["pert_index2"] = [pert_index.get(normalize_free_text(value) or "no", no_pert) for value in df["pert_id2"]]
    cell_llm_values = [normalize_free_text(value) or "no" for value in df["Cell"].tolist()]
    df["cell_llm_label"] = cell_llm_values
    df["cell_llm_index"] = [int(cell_llm_index.get(value, 0)) for value in cell_llm_values]
    df["cell_type_llm_index"] = df["cell_type_index"]
    return df


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


def write_split_files(split_dir: Path, df: pd.DataFrame, test_indices: list[int]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    row_to_set: dict[int, int] = {}
    set_info: dict[int, dict[str, list[int]]] = {}
    control_lookup = {
        str(row.sample_id): int(row.feature_row_index)
        for row in df.loc[df["is_control"]].itertuples(index=False)
    }
    grouped: dict[str, list[int]] = {}
    for row_idx in test_indices:
        control_id = str(df.loc[row_idx, "control"])
        grouped.setdefault(control_id, []).append(int(row_idx))
    for set_idx, control_id in enumerate(sorted(grouped)):
        control_idx = control_lookup[control_id]
        perturb_rows = sorted(grouped[control_id])
        set_info[set_idx] = {"control": [control_idx], "perturb": perturb_rows}
        row_to_set[control_idx] = set_idx
        for row_idx in perturb_rows:
            row_to_set[row_idx] = set_idx

    payloads = {
        "row_to_set_index.pkl": row_to_set,
        "set_info.pkl": set_info,
        "test_set_info_test_only.pkl": set_info,
        "train_set_info_test_only.pkl": {},
        "valid_set_info_test_only.pkl": {},
        "val_set_info_test_only.pkl": {},
        "test_indices_test_only.pkl": test_indices,
        "train_indices_test_only.pkl": [],
        "valid_indices_test_only.pkl": [],
        "val_indices_test_only.pkl": [],
    }
    for filename, payload in payloads.items():
        with (split_dir / filename).open("wb") as handle:
            pickle.dump(payload, handle)
    dump_json(
        split_dir / "split_manifest.json",
        {
            "generated_at": iso_now(),
            "strategy": "test_only",
            "test_anchor_count": len(test_indices),
            "set_count": len(set_info),
            "note": "Inference-only split generated from patient baseline controls.",
        },
    )


def build_cell_llm_index_from_tasks(
    task_payloads: dict[str, dict[str, Any]],
    *,
    all_cell_labels: list[str],
) -> dict[str, int]:
    mapping: dict[str, int] = {"no": 0}
    for payload in task_payloads.values():
        for row in payload["rows"]:
            label = normalize_free_text(row.get("Cell")) or "no"
            if label not in mapping:
                mapping[label] = len(mapping)
    for label in all_cell_labels:
        label = normalize_free_text(label) or "no"
        if label not in mapping:
            mapping[label] = len(mapping)
    return mapping


def transform_expression_values(spec: PatientDataset, values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    finite_before = np.isfinite(values)
    finite_value_count = int(finite_before.sum())
    negative_count = int(np.count_nonzero(values[finite_before] < 0.0)) if finite_value_count else 0
    raw_min = float(np.nanmin(values)) if finite_value_count else None
    raw_max = float(np.nanmax(values)) if finite_value_count else None

    if spec.expression_scale == "log2_relative":
        # P3 is already log2-scale relative expression, not raw abundance.
        # Convert back to a positive ratio-like scale before applying the
        # training pipeline's log1p transform.
        transformed = np.full(values.shape, np.nan, dtype=np.float32)
        transformed[finite_before] = np.log1p(np.exp2(values[finite_before])).astype(np.float32, copy=False)
        finite_after = np.isfinite(transformed)
        audit = {
            "expression_scale_input": spec.expression_scale,
            "expression_transform": "log2_exp2_then_log1p",
            "standard_expression_scale": False,
            "scale_caveat": (
                "Input appears log2-centered/relative rather than absolute raw abundance; "
                "exp2 produces ratio-like values, not calibrated raw intensities."
            ),
            "finite_value_count_before_transform": finite_value_count,
            "negative_value_count_before_transform": negative_count,
            "negative_values_mapped_to_nan": 0,
            "negative_values_converted_by_exp2": negative_count,
            "raw_min_before_transform": raw_min,
            "raw_max_before_transform": raw_max,
            "finite_negative_count_after_transform": int(np.count_nonzero(transformed[finite_after] < 0.0)) if finite_after.any() else 0,
            "min_after_transform": float(np.nanmin(transformed)) if finite_after.any() else None,
            "max_after_transform": float(np.nanmax(transformed)) if finite_after.any() else None,
        }
        return transformed, audit

    if spec.expression_scale != "raw_abundance":
        raise ValueError(f"unsupported expression_scale for {spec.key}: {spec.expression_scale!r}")

    transformed = values.astype(np.float32, copy=True)
    mapped_negative_count = 0
    if negative_count:
        transformed[finite_before & (transformed < 0.0)] = np.nan
        mapped_negative_count = negative_count
    finite_nonnegative = np.isfinite(transformed)
    transformed[finite_nonnegative] = np.log1p(transformed[finite_nonnegative])
    finite_after = np.isfinite(transformed)
    audit = {
        "expression_scale_input": spec.expression_scale,
        "expression_transform": "raw_log1p",
        "standard_expression_scale": True,
        "finite_value_count_before_transform": finite_value_count,
        "negative_value_count_before_transform": negative_count,
        "negative_values_mapped_to_nan": mapped_negative_count,
        "negative_values_converted_by_exp2": 0,
        "raw_min_before_transform": raw_min,
        "raw_max_before_transform": raw_max,
        "finite_negative_count_after_transform": int(np.count_nonzero(transformed[finite_after] < 0.0)) if finite_after.any() else 0,
        "min_after_transform": float(np.nanmin(transformed)) if finite_after.any() else None,
        "max_after_transform": float(np.nanmax(transformed)) if finite_after.any() else None,
    }
    return transformed, audit


def load_expression_matrix(
    spec: PatientDataset,
    meta: dict[str, Any],
) -> tuple[pd.Series, np.ndarray, list[int], list[str], dict[str, Any]]:
    header = pd.read_csv(spec.matrix_path, nrows=0, low_memory=False)
    if "sample_id" not in header.columns:
        raise ValueError(f"{spec.matrix_path}: missing sample_id column")

    selected_columns = ["sample_id"]
    ordered_indices: list[int] = []
    ordered_uniprot: list[str] = []
    missing_from_meta: list[str] = []
    unresolved_columns: list[str] = []
    seen_indices: set[int] = set()
    for column in header.columns:
        if column == "sample_id":
            continue
        uniprot = parse_uniprot_column(column)
        if uniprot is None:
            unresolved_columns.append(str(column))
            continue
        protein_idx = meta["protein_index"].get(uniprot)
        if protein_idx is None:
            missing_from_meta.append(uniprot)
            continue
        protein_idx = int(protein_idx)
        if protein_idx in seen_indices:
            continue
        seen_indices.add(protein_idx)
        selected_columns.append(column)
        ordered_indices.append(protein_idx)
        ordered_uniprot.append(uniprot)

    matrix_df = pd.read_csv(spec.matrix_path, usecols=selected_columns, low_memory=False)
    sample_ids = matrix_df["sample_id"].map(normalize_free_text)
    values = matrix_df.drop(columns=["sample_id"]).to_numpy(dtype=np.float32, copy=True)
    values, transform_audit = transform_expression_values(spec, values)

    audit = {
        "matrix_path": str(spec.matrix_path),
        "raw_rows": int(len(matrix_df)),
        "raw_protein_columns": int(len(header.columns) - 1),
        "kept_protein_columns": int(len(ordered_indices)),
        "unresolved_protein_columns": int(len(unresolved_columns)),
        "unresolved_examples": unresolved_columns[:20],
        "protein_columns_missing_from_ptv3_meta": int(len(set(missing_from_meta))),
        "missing_from_ptv3_meta_examples": sorted(set(missing_from_meta))[:20],
    }
    audit.update(transform_audit)
    return sample_ids, values, ordered_indices, ordered_uniprot, audit


def build_rows_for_dataset(
    spec: PatientDataset,
    meta: dict[str, Any],
    matrix_sample_ids: pd.Series,
    matrix_values: np.ndarray,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    info = pd.read_csv(spec.info_path, low_memory=False)
    info_sample_ids = info[spec.sample_id_col].map(normalize_free_text)
    matrix_lookup = {sample_id: idx for idx, sample_id in matrix_sample_ids.items()}

    task_rows: dict[str, list[dict[str, Any]]] = {"single": [], "double": []}
    task_vectors: dict[str, list[np.ndarray]] = {"single": [], "double": []}
    skipped: list[dict[str, Any]] = []
    controls_added: dict[str, set[str]] = {"single": set(), "double": set()}
    counts = {"single": 0, "double": 0, "unsupported_multi_drug": 0, "missing_matrix": 0}
    expression_transform = "log2_exp2_then_log1p" if spec.expression_scale == "log2_relative" else "raw_log1p"
    standard_compatible = spec.expression_scale == "raw_abundance"

    for raw_idx, row in info.iterrows():
        patient_id = normalize_free_text(info_sample_ids.iloc[raw_idx])
        tokens = split_combo(row.get(spec.combo_col))
        if patient_id not in matrix_lookup:
            counts["missing_matrix"] += 1
            skipped.append(
                {
                    "row_index": int(raw_idx),
                    "patient_sample_id": patient_id,
                    "raw_combo": normalize_free_text(row.get(spec.combo_col)),
                    "reason": "missing_baseline_matrix_row",
                }
            )
            continue
        if not tokens:
            skipped.append(
                {
                    "row_index": int(raw_idx),
                    "patient_sample_id": patient_id,
                    "raw_combo": normalize_free_text(row.get(spec.combo_col)),
                    "reason": "empty_drug_combo",
                }
            )
            continue
        if len(tokens) > 2:
            counts["unsupported_multi_drug"] += 1
            skipped.append(
                {
                    "row_index": int(raw_idx),
                    "patient_sample_id": patient_id,
                    "raw_combo": normalize_free_text(row.get(spec.combo_col)),
                    "resolved_tokens": tokens,
                    "reason": "current_model_has_two_drug_slots",
                }
            )
            continue

        kind = "single" if len(tokens) == 1 else "double"
        counts[kind] += 1
        control_id = f"{spec.key}::{patient_id}::control"
        if control_id not in controls_added[kind]:
            control_row = {
                "sample_id": control_id,
                "patient_sample_id": patient_id,
                "source_patient_dataset": spec.display_name,
                "source_patient_dataset_key": spec.key,
                "source_info_row_index": int(raw_idx),
                "raw_combo": "",
                "combo_tokens": "[]",
                "patient_validation_kind": kind,
                "standard_compatible": bool(standard_compatible),
                "expression_transform": expression_transform,
                "machineID_new": normalize_free_text(row.get("machineID_new")) or "no",
                "Cell_plate": "no",
                "Cell": patient_id,
                "cell_type": infer_cell_type(spec, row),
                "pert_id1": "no",
                "pert_id2": "no",
                "batch": spec.key,
                "pert_time": "no",
                "pert_dose1": "no",
                "pert_dose2": "no",
                "PRISM1st_label_total": "",
                "PRISM2nd_label_total": "",
                "synergy": "",
                "instrument": normalize_free_text(row.get("machineID_new")),
                "cell_pertid_time": "",
                "drugname": "",
                "smiles": "",
                "target_protein_list": "[]",
                "control": control_id,
                "is_control": True,
                "source_task": f"ptv3_{RUN_NAME}_{spec.key}_{kind}",
                "source_row_role": "patient_baseline_control",
                "feature_membership": "primary",
                "task_context": f"ptv3_{RUN_NAME}_{spec.key}_{kind}",
            }
            for column in info.columns:
                control_row[f"clinical__{column}"] = row.get(column)
            task_rows[kind].append(control_row)
            task_vectors[kind].append(matrix_values[matrix_lookup[patient_id]])
            controls_added[kind].add(control_id)

        pert_id1 = tokens[0]
        pert_id2 = tokens[0] if len(tokens) == 1 else tokens[1]
        pert_sample_id = (
            f"{spec.key}::{patient_id}::{kind}::"
            f"{safe_id_fragment('+'.join(tokens))}::row{int(raw_idx)}"
        )
        pert_row = {
            "sample_id": pert_sample_id,
            "patient_sample_id": patient_id,
            "source_patient_dataset": spec.display_name,
            "source_patient_dataset_key": spec.key,
            "source_info_row_index": int(raw_idx),
            "raw_combo": normalize_free_text(row.get(spec.combo_col)),
            "combo_tokens": json.dumps(tokens, ensure_ascii=False),
            "patient_validation_kind": kind,
            "standard_compatible": bool(standard_compatible),
            "expression_transform": expression_transform,
            "machineID_new": normalize_free_text(row.get("machineID_new")) or "no",
            "Cell_plate": "no",
            "Cell": patient_id,
            "cell_type": infer_cell_type(spec, row),
            "pert_id1": pert_id1,
            "pert_id2": pert_id2,
            "batch": spec.key,
            "pert_time": "no",
            "pert_dose1": "no",
            "pert_dose2": "no",
            "PRISM1st_label_total": "",
            "PRISM2nd_label_total": "",
            "synergy": "",
            "instrument": normalize_free_text(row.get("machineID_new")),
            "cell_pertid_time": "",
            "drugname": normalize_free_text(
                row.get("ptv3_pert_drug", row.get("Drug_ptv3", row.get("chemotherapy_drug_ptv3", "")))
            ),
            "smiles": ";".join(
                meta.get("pertid_to_smiles", {}).get(token, "")
                for token in tokens
                if meta.get("pertid_to_smiles", {}).get(token, "")
            ),
            "target_protein_list": combine_target_indices(tokens, meta),
            "control": control_id,
            "is_control": False,
            "source_task": f"ptv3_{RUN_NAME}_{spec.key}_{kind}",
            "source_row_role": "self",
            "feature_membership": "primary",
            "task_context": f"ptv3_{RUN_NAME}_{spec.key}_{kind}",
        }
        for column in info.columns:
            pert_row[f"clinical__{column}"] = row.get(column)
        task_rows[kind].append(pert_row)
        task_vectors[kind].append(np.full(matrix_values.shape[1], np.nan, dtype=np.float32))

    audit = {
        "display_name": spec.display_name,
        "info_path": str(spec.info_path),
        "sample_id_col": spec.sample_id_col,
        "combo_col": spec.combo_col,
        "raw_info_rows": int(len(info)),
        "counts": counts,
        "standard_compatible": bool(standard_compatible),
    }
    return task_rows, task_vectors, skipped, audit


def write_task(
    *,
    task_name: str,
    rows: list[dict[str, Any]],
    vectors: list[np.ndarray],
    ordered_indices: list[int],
    ordered_uniprot: list[str],
    meta: dict[str, Any],
    cell_llm_index: dict[str, int],
    output_root: Path,
) -> dict[str, Any]:
    task_dir = output_root / DATASET_GROUP / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = add_index_columns(df, meta, cell_llm_index)
    df["feature_row_index"] = np.arange(len(df), dtype=np.int32)
    df["processed_row_index"] = df["feature_row_index"]
    df["expression_row_index"] = df["feature_row_index"]
    matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)

    csv_path, native_path = save_feature_dataframe(df, task_dir)
    np.save(task_dir / "feature_expression_matrix.npy", matrix)
    dump_json(task_dir / "feature_ordered_protein_index.json", ordered_indices)
    dump_json(task_dir / "feature_ordered_protein_uniprot.json", ordered_uniprot)
    dump_json(task_dir / "feature_sample_ids.json", df["sample_id"].astype(str).tolist())
    df.to_csv(task_dir / "processed.csv", index=False)
    np.save(task_dir / "processed_expression_matrix.npy", matrix)
    dump_json(task_dir / "processed_ordered_protein_index.json", ordered_indices)
    dump_json(task_dir / "processed_ordered_protein_uniprot.json", ordered_uniprot)
    dump_json(task_dir / "processed_sample_ids.json", df["sample_id"].astype(str).tolist())
    dump_json(
        task_dir / "feature_loading_manifest.json",
        {
            "generated_at": iso_now(),
            "row_key_column": "sample_id",
            "expression_row_index_column": "expression_row_index",
            "expression_matrix_path": str(task_dir / "feature_expression_matrix.npy"),
            "ordered_protein_index_path": str(task_dir / "feature_ordered_protein_index.json"),
            "ordered_protein_uniprot_path": str(task_dir / "feature_ordered_protein_uniprot.json"),
            "sample_ids_path": str(task_dir / "feature_sample_ids.json"),
            "feature_table_csv_path": csv_path,
            "feature_table_native_path": native_path,
            "loading_contract": "Inference-only patient baseline controls paired with drug query rows.",
        },
    )

    test_indices = df.index[(~df["is_control"].astype(bool)) & df["source_row_role"].eq("self")].astype(int).tolist()
    write_split_files(output_root / DATASET_GROUP / "splits" / task_name, df, test_indices)
    return {
        "task_name": task_name,
        "task_dir": str(task_dir),
        "rows": int(len(df)),
        "control_rows": int(df["is_control"].astype(bool).sum()),
        "test_anchor_rows": int(len(test_indices)),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "feature_table_csv_path": csv_path,
        "feature_table_native_path": native_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ready-root", default=str(TRAINING_READY_ROOT))
    parser.add_argument("--source-training-ready-root", default=str(SOURCE_TRAINING_READY_ROOT))
    parser.add_argument("--run-name", default=RUN_NAME)
    return parser.parse_args()


def main() -> None:
    global RUN_NAME
    args = parse_args()
    RUN_NAME = str(args.run_name)
    output_root = Path(args.training_ready_root)
    source_root = Path(args.source_training_ready_root)
    source_meta_path = source_root / DATASET_GROUP / "global_meta.json"
    source_meta = load_json(source_meta_path)
    feature_space = collect_patient_feature_space()
    meta, feature_audit = extend_global_meta_for_patient_run(source_meta, feature_space)
    meta_path = output_root / DATASET_GROUP / "global_meta.json"

    summary: dict[str, Any] = {
        "generated_at": iso_now(),
        "raw_root": str(RAW_ROOT),
        "training_ready_root": str(output_root),
        "source_training_ready_root": str(source_root),
        "source_meta_path": str(source_meta_path),
        "meta_path": str(meta_path),
        "tasks": {},
        "datasets": {},
        "skipped_rows": [],
        "feature_audit": feature_audit,
        "notes": [
            "PTV3 protein_index is extended by appending patient-only proteins while preserving all existing PTV3 indices.",
            "Cell_index remains checkpoint-compatible; patient Cell LLM features use the separate cell_llm_index column.",
            "P5_guomics_breastCancer missing cell_type is assigned to BREAST, which already exists in PTV3 cell_type metadata.",
            "Raw-abundance patient matrices use log1p on finite nonnegative values.",
            "P3_lungCancer_2024cell is log2-relative scale and uses exp2 followed by log1p; its negative values are not mapped to NaN.",
            "Batch/machine values may map to the existing `no` index because this pipeline has no batch/machine generalization feature.",
            "Rows with more than two unique drug tokens are skipped because exp_07/08 are two-slot models.",
        ],
    }

    task_payloads: dict[str, dict[str, Any]] = {}
    for spec in DATASETS:
        matrix_sample_ids, matrix_values, ordered_indices, ordered_uniprot, matrix_audit = load_expression_matrix(spec, meta)
        task_rows, task_vectors, skipped, dataset_audit = build_rows_for_dataset(
            spec,
            meta,
            matrix_sample_ids,
            matrix_values,
        )
        dataset_audit["matrix_audit"] = matrix_audit
        summary["datasets"][spec.key] = dataset_audit
        summary["skipped_rows"].extend({"dataset": spec.key, **row} for row in skipped)
        for kind in ("single", "double"):
            rows = task_rows[kind]
            if not rows:
                continue
            task_name = f"ptv3_{RUN_NAME}_{spec.key}_{kind}"
            task_payloads[task_name] = {
                "rows": rows,
                "vectors": task_vectors[kind],
                "ordered_indices": ordered_indices,
                "ordered_uniprot": ordered_uniprot,
            }

    cell_llm_index = build_cell_llm_index_from_tasks(task_payloads, all_cell_labels=feature_space["cells"])
    meta["task_names"] = list(task_payloads)
    meta["patient_validation_extension"]["task_names"] = list(task_payloads)
    meta["patient_validation_extension"]["cell_llm_index_size"] = len(cell_llm_index)
    meta["patient_validation_extension"]["cell_llm_index_path"] = str(
        output_root / DATASET_GROUP / "derived" / f"{RUN_NAME}_cell_llm_index.json"
    )
    dump_json(meta_path, meta)
    dump_json(
        output_root / DATASET_GROUP / "derived" / f"{RUN_NAME}_cell_llm_index.json",
        cell_llm_index,
    )

    for task_name, payload in task_payloads.items():
        summary["tasks"][task_name] = write_task(
            task_name=task_name,
            rows=payload["rows"],
            vectors=payload["vectors"],
            ordered_indices=payload["ordered_indices"],
            ordered_uniprot=payload["ordered_uniprot"],
            meta=meta,
            cell_llm_index=cell_llm_index,
            output_root=output_root,
        )

    summary["cell_llm_index"] = {
        "path": str(output_root / DATASET_GROUP / "derived" / f"{RUN_NAME}_cell_llm_index.json"),
        "size": len(cell_llm_index),
        "reserved_no_index": 0,
    }

    summary_path = output_root / DATASET_GROUP / f"{RUN_NAME}_build_summary.json"
    dump_json(summary_path, summary)
    print(f"[patient-vali] wrote {len(summary['tasks'])} tasks")
    print(f"[patient-vali] skipped rows: {len(summary['skipped_rows'])}")
    print(f"[patient-vali] summary: {summary_path}")


if __name__ == "__main__":
    main()
