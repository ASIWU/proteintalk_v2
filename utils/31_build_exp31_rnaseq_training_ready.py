#!/usr/bin/env python3
"""Build exp31 RNA-seq PDX fine-tuning tasks.

This builder is intentionally copy-on-write.  It reads the maintained PTV3
training-ready root and writes a separate exp31 root so the original artifacts
remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GROUP = "ptv3"
RUN_NAME = "exp31_rnaseq"
SPLIT_STRATEGY = "brca_ft_valid_nonbrca_test"
RAW_RNA = REPO_ROOT / "data/rawdata/rna_seq/260617_2015_BFnm3954_MOESM10_ESM_sub.csv"
RAW_INFO = REPO_ROOT / "data/rawdata/rna_seq/260618pdx_pct_sample_info_with_smiles_check_comboAB.csv"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/training_ready"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/training_ready_exp31_rnaseq"
DEFAULT_EXP09_MANIFEST = (
    REPO_ROOT
    / "checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/run_manifest.json"
)

LABEL_TASKS = [
    (
        "sensitive_early",
        "sensitive_label_early_CRPR_vs_SDPD",
        "early sensitive CR/PR versus SD/PD",
    ),
    (
        "sensitive_late",
        "sensitive_label_late_CRPR_vs_SDPD",
        "late sensitive CR/PR versus SD/PD",
    ),
    (
        "disease_control_early",
        "disease_control_label_early_CRPRSD_vs_PD",
        "early disease-control CR/PR/SD versus PD",
    ),
    (
        "disease_control_late",
        "disease_control_label_late_CRPRSD_vs_PD",
        "late disease-control CR/PR/SD versus PD",
    ),
]

CANCER_DESCRIPTIONS = {
    "BRCA": "breast cancer",
    "CRC": "colorectal cancer",
    "PDAC": "pancreatic ductal adenocarcinoma",
    "NSCLC": "non-small-cell lung cancer",
    "CM": "cutaneous melanoma",
}

BATCH_FIELDS = [
    "machineID_new",
    "Cell_plate",
    "Cell",
    "cell_type",
    "batch",
    "pert_time",
    "pert_dose1",
    "pert_dose2",
]


@dataclass(frozen=True)
class ResolvedDrug:
    pert_id: str
    source: str
    component: str
    smiles: str
    candidates: list[str]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def safe_text(value: object) -> str:
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


def ordered_ids(index_mapping: dict[str, int]) -> list[str]:
    return [item for item, _ in sorted(index_mapping.items(), key=lambda pair: int(pair[1]))]


def canonical_smiles_or_raw(smiles: str) -> str:
    text = safe_text(smiles)
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_pert_id(smiles: str) -> str:
    basis = canonical_smiles_or_raw(smiles)
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"exp31smiles::{digest}"


def component_key(value: object) -> str:
    return safe_text(value).lower()


def split_components(value: object) -> list[str]:
    text = safe_text(value)
    if not text:
        return []
    parts = []
    for token in text.replace("+", ";").split(";"):
        token = token.strip()
        if token and token.lower() not in {"nan", "none", "null"} and token not in parts:
            parts.append(token)
    return parts


def parse_component_candidates(value: object) -> dict[str, list[str]]:
    text = safe_text(value)
    result: dict[str, list[str]] = {}
    if not text:
        return result
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        name, candidate = item.split("=", 1)
        key = component_key(name)
        candidate = safe_text(candidate)
        if not key or not candidate:
            continue
        result.setdefault(key, [])
        if candidate not in result[key]:
            result[key].append(candidate)
    return result


def choose_existing_candidate(candidates: list[str], pert_index: dict[str, int]) -> str | None:
    existing = [item for item in candidates if item in pert_index]
    if not existing:
        return None
    l9200 = sorted(item for item in existing if item.upper().startswith("L9200_"))
    if l9200:
        return l9200[0]
    return sorted(existing)[0]


def resolve_drug_slot(
    *,
    component: str,
    smiles: str,
    candidates_by_component: dict[str, list[str]],
    pert_index: dict[str, int],
) -> ResolvedDrug:
    key = component_key(component)
    candidates = candidates_by_component.get(key, [])
    chosen = choose_existing_candidate(candidates, pert_index)
    if chosen is not None:
        return ResolvedDrug(chosen, "existing_component_id", component, smiles, candidates)
    pert_id = smiles_pert_id(smiles)
    return ResolvedDrug(pert_id, "new_smiles_id", component, smiles, candidates)


def target_list_for_drugs(drugs: list[str], meta: dict[str, Any]) -> str:
    values: list[int] = []
    target_map = meta.get("pertid_to_target_protein_list", {})
    for pert_id in drugs:
        for item in target_map.get(pert_id, []):
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if idx not in values:
                values.append(idx)
    return json.dumps(values, ensure_ascii=False)


def index_value(meta: dict[str, Any], field: str, value: str = "no") -> int:
    mapping_key = "pert_dose" if field in {"pert_dose1", "pert_dose2"} else field
    mapping = meta["value_to_index"][mapping_key]
    return int(mapping.get(value, mapping["no"]))


def read_exp09_axis(manifest_path: Path) -> tuple[list[int], list[str]]:
    manifest = load_json(manifest_path)
    axis_path = Path(str(manifest["ordered_protein_index_path"]))
    ordered_index = [int(item) for item in load_json(axis_path)]
    task_dir = Path(str(manifest["task_dir"]))
    uniprot_path = task_dir / "feature_ordered_protein_uniprot.json"
    ordered_uniprot = [str(item) for item in load_json(uniprot_path)] if uniprot_path.exists() else []
    if ordered_uniprot and len(ordered_uniprot) != len(ordered_index):
        raise ValueError(f"axis index/uniprot length mismatch: {axis_path} vs {uniprot_path}")
    return ordered_index, ordered_uniprot


def load_rna_expression(
    *,
    rna_path: Path,
    sample_ids: list[str],
    ordered_protein_index: list[int],
    meta: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    rna = pd.read_csv(rna_path, low_memory=False)
    if "uniprot_ID" not in rna.columns:
        raise ValueError(f"{rna_path} missing uniprot_ID column")
    missing_samples = sorted(set(sample_ids) - set(map(str, rna.columns)))
    if missing_samples:
        raise ValueError(f"RNA matrix missing sample columns: {missing_samples[:20]}")

    protein_index = {str(key): int(value) for key, value in meta["protein_index"].items()}
    uniprot_values = rna["uniprot_ID"].map(safe_text)
    protein_to_rna_row: dict[int, int] = {}
    empty_uniprot = 0
    for row_idx, uniprot in enumerate(uniprot_values.tolist()):
        if not uniprot:
            empty_uniprot += 1
            continue
        protein_idx = protein_index.get(uniprot)
        if protein_idx is None:
            continue
        protein_to_rna_row.setdefault(int(protein_idx), int(row_idx))

    values = rna[sample_ids].to_numpy(dtype=np.float32, copy=True)
    aligned = np.full((len(sample_ids), len(ordered_protein_index)), np.nan, dtype=np.float32)
    covered_cols = 0
    for col_idx, protein_idx in enumerate(ordered_protein_index):
        rna_row = protein_to_rna_row.get(int(protein_idx))
        if rna_row is None:
            continue
        aligned[:, col_idx] = values[rna_row, :]
        covered_cols += 1

    finite_before = np.isfinite(aligned)
    negative = finite_before & (aligned < 0.0)
    aligned[negative] = np.nan
    finite_nonnegative = np.isfinite(aligned)
    aligned[finite_nonnegative] = np.log1p(aligned[finite_nonnegative])
    audit = {
        "rna_path": str(rna_path),
        "raw_shape": [int(rna.shape[0]), int(rna.shape[1])],
        "sample_count": int(len(sample_ids)),
        "axis_size": int(len(ordered_protein_index)),
        "axis_covered_by_rna": int(covered_cols),
        "axis_missing_from_rna": int(len(ordered_protein_index) - covered_cols),
        "rna_nonempty_uniprot_rows": int(len(rna) - empty_uniprot),
        "rna_empty_uniprot_rows": int(empty_uniprot),
        "finite_value_count_before_transform": int(finite_before.sum()),
        "negative_value_count_before_transform": int(negative.sum()),
        "finite_value_count_after_transform": int(np.isfinite(aligned).sum()),
        "expression_transform": "finite non-negative RNA values log1p; missing/negative values as NaN",
    }
    return aligned, audit


def save_dataframe(df: pd.DataFrame, task_dir: Path) -> tuple[str, str]:
    csv_path = task_dir / "feature_table.csv"
    df.to_csv(csv_path, index=False)
    try:
        parquet_path = task_dir / "feature_table.parquet"
        df.to_parquet(parquet_path, index=False)
        return str(csv_path), str(parquet_path)
    except Exception:
        pickle_path = task_dir / "feature_table.pkl"
        df.to_pickle(pickle_path)
        return str(csv_path), str(pickle_path)


def find_brca_split(info: pd.DataFrame, label_col: str, *, seed_start: int, seed_end: int) -> dict[str, Any]:
    brca = info.loc[info["cancer_type"].eq("BRCA")].copy()
    samples = sorted(brca["sample_id"].astype(str).unique().tolist())
    if len(samples) < 3:
        raise ValueError("BRCA split requires at least 3 samples")
    valid_size = max(1, int(round(len(samples) * 0.2)))
    for seed in range(seed_start, seed_end + 1):
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(samples, dtype=object)
        rng.shuffle(shuffled)
        valid_samples = set(map(str, shuffled[:valid_size]))
        train_samples = set(samples) - valid_samples
        train_labels = brca.loc[brca["sample_id"].astype(str).isin(train_samples), label_col].astype(int)
        valid_labels = brca.loc[brca["sample_id"].astype(str).isin(valid_samples), label_col].astype(int)
        if train_labels.nunique() == 2 and valid_labels.nunique() == 2:
            return {
                "seed": int(seed),
                "train_samples": sorted(train_samples),
                "valid_samples": sorted(valid_samples),
                "valid_size": int(valid_size),
                "train_positive": int(train_labels.sum()),
                "train_negative": int(len(train_labels) - train_labels.sum()),
                "valid_positive": int(valid_labels.sum()),
                "valid_negative": int(len(valid_labels) - valid_labels.sum()),
            }
    raise ValueError(f"could not find BRCA split with both classes for {label_col}")


def write_split_files(
    *,
    split_dir: Path,
    df: pd.DataFrame,
    split_samples: dict[str, set[str]],
    split_summary: dict[str, Any],
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    row_to_set: dict[int, int] = {}
    all_set_info: dict[int, dict[str, list[int]]] = {}
    split_payloads: dict[str, dict[int, dict[str, list[int]]]] = {}
    split_indices: dict[str, list[int]] = {}
    next_set_idx = 0

    control_lookup = {
        str(row.patient_sample_id): int(row.feature_row_index)
        for row in df.loc[df["is_control"].astype(bool)].itertuples(index=False)
    }
    perturb_df = df.loc[~df["is_control"].astype(bool)].copy()
    for split_name, samples in split_samples.items():
        indices = (
            perturb_df.index[perturb_df["patient_sample_id"].astype(str).isin(samples)]
            .astype(int)
            .tolist()
        )
        split_indices[split_name] = sorted(indices)
        set_info: dict[int, dict[str, list[int]]] = {}
        grouped = perturb_df.loc[indices].groupby("patient_sample_id", sort=True)
        for sample_id, group in grouped:
            control_row = control_lookup[str(sample_id)]
            perturb_rows = sorted(int(idx) for idx in group.index.tolist())
            set_info[next_set_idx] = {"control": [control_row], "perturb": perturb_rows}
            all_set_info[next_set_idx] = set_info[next_set_idx]
            row_to_set[control_row] = next_set_idx
            for row_idx in perturb_rows:
                row_to_set[row_idx] = next_set_idx
            next_set_idx += 1
        split_payloads[split_name] = set_info

    dump_pickle(split_dir / "row_to_set_index.pkl", row_to_set)
    dump_pickle(split_dir / "set_info.pkl", all_set_info)
    for split_name in ("train", "valid", "test"):
        dump_pickle(split_dir / f"{split_name}_indices_{SPLIT_STRATEGY}.pkl", split_indices[split_name])
        dump_pickle(split_dir / f"{split_name}_set_info_{SPLIT_STRATEGY}.pkl", split_payloads[split_name])
    dump_pickle(split_dir / f"val_indices_{SPLIT_STRATEGY}.pkl", split_indices["valid"])
    dump_pickle(split_dir / f"val_set_info_{SPLIT_STRATEGY}.pkl", split_payloads["valid"])

    manifest = {
        "generated_at": iso_now(),
        "strategy": SPLIT_STRATEGY,
        "policy": "BRCA samples train/valid for fine-tuning; non-BRCA samples held out for zero-shot test",
        "split_summary": split_summary,
        "anchor_counts": {key: int(len(value)) for key, value in split_indices.items()},
        "set_counts": {key: int(len(value)) for key, value in split_payloads.items()},
    }
    dump_json(split_dir / "split_manifest.json", manifest)
    return manifest


def build_base_rows(
    *,
    info: pd.DataFrame,
    sample_ids: list[str],
    sample_expression: np.ndarray,
    sample_to_expr_row: dict[str, int],
    meta: dict[str, Any],
    drug_resolutions: dict[int, list[ResolvedDrug]],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    control_seen: set[str] = set()
    cell_llm_index = {"no": 0}

    no_pert = "no"
    for sample_id in sample_ids:
        if sample_id not in cell_llm_index:
            cell_llm_index[sample_id] = len(cell_llm_index)

    def add_common_indices(row: dict[str, Any]) -> None:
        for field in BATCH_FIELDS:
            row[f"{field}_index"] = index_value(meta, field, "no")
        row["pert_index1"] = int(meta["pert_index"].get(str(row["pert_id1"]), meta["pert_index"]["no"]))
        row["pert_index2"] = int(meta["pert_index"].get(str(row["pert_id2"]), meta["pert_index"]["no"]))
        row["cell_type_llm_index"] = int(index_value(meta, "cell_type", "no"))

    for raw_idx, raw in info.iterrows():
        sample_id = str(raw["sample_id"])
        cancer_type = str(raw["cancer_type"])
        if sample_id not in control_seen:
            control_id = f"{RUN_NAME}::{sample_id}::control"
            control_row = {
                "sample_id": control_id,
                "patient_sample_id": sample_id,
                "source_info_row_index": -1,
                "raw_treatment": "",
                "treatment_type": "control",
                "cancer_type": cancer_type,
                "cancer_type_description": CANCER_DESCRIPTIONS.get(cancer_type, cancer_type),
                "machineID_new": "no",
                "Cell_plate": "no",
                "Cell": sample_id,
                "cell_type": cancer_type,
                "batch": RUN_NAME,
                "pert_time": "no",
                "pert_dose1": "no",
                "pert_dose2": "no",
                "pert_id1": no_pert,
                "pert_id2": no_pert,
                "drugname": "",
                "smiles": "",
                "target_protein_list": "[]",
                "control": control_id,
                "is_control": True,
                "source_row_role": "exp31_baseline_control",
                "feature_membership": "primary",
                "PRISM1st_label_total": "",
                "PRISM2nd_label_total": "",
                "synergy": "",
                "unified_label_mask": 1.0,
                "exp31_label_value": "",
                "exp31_label_source": "",
                "cell_llm_label": (
                    f"PDX-2015nm baseline RNA sample {sample_id}; "
                    f"cancer type {cancer_type} {CANCER_DESCRIPTIONS.get(cancer_type, cancer_type)}"
                ),
                "cell_llm_index": int(cell_llm_index[sample_id]),
            }
            add_common_indices(control_row)
            rows.append(control_row)
            vectors.append(sample_expression[sample_to_expr_row[sample_id]])
            control_seen.add(sample_id)

        drugs = drug_resolutions[int(raw_idx)]
        kind = "double" if len(drugs) == 2 else "single"
        pert_id1 = drugs[0].pert_id
        pert_id2 = drugs[1].pert_id if len(drugs) == 2 else drugs[0].pert_id
        smiles_values = [drug.smiles for drug in drugs if drug.smiles]
        pert_row = {
            "sample_id": f"{RUN_NAME}::{sample_id}::{kind}::row{int(raw_idx)}",
            "patient_sample_id": sample_id,
            "source_info_row_index": int(raw_idx),
            "raw_treatment": safe_text(raw.get("treatment")),
            "treatment_type": kind,
            "cancer_type": cancer_type,
            "cancer_type_description": CANCER_DESCRIPTIONS.get(cancer_type, cancer_type),
            "machineID_new": "no",
            "Cell_plate": "no",
            "Cell": sample_id,
            "cell_type": cancer_type,
            "batch": RUN_NAME,
            "pert_time": "no",
            "pert_dose1": "no",
            "pert_dose2": "no",
            "pert_id1": pert_id1,
            "pert_id2": pert_id2,
            "drugname": safe_text(raw.get("treatment")),
            "smiles": " || ".join(smiles_values),
            "target_protein_list": target_list_for_drugs([pert_id1, pert_id2], meta),
            "control": f"{RUN_NAME}::{sample_id}::control",
            "is_control": False,
            "source_row_role": "exp31_treatment_query",
            "feature_membership": "primary",
            "PRISM1st_label_total": "",
            "PRISM2nd_label_total": "",
            "synergy": "",
            "unified_label_mask": 1.0,
            "exp31_label_value": "",
            "exp31_label_source": "",
            "cell_llm_label": (
                f"PDX-2015nm baseline RNA sample {sample_id}; "
                f"cancer type {cancer_type} {CANCER_DESCRIPTIONS.get(cancer_type, cancer_type)}"
            ),
            "cell_llm_index": int(cell_llm_index[sample_id]),
            "exp31_drug_resolution": json.dumps(
                [
                    {
                        "pert_id": drug.pert_id,
                        "source": drug.source,
                        "component": drug.component,
                        "candidates": drug.candidates,
                    }
                    for drug in drugs
                ],
                ensure_ascii=False,
            ),
        }
        for col in info.columns:
            pert_row[f"clinical__{col}"] = raw.get(col)
        add_common_indices(pert_row)
        rows.append(pert_row)
        vectors.append(np.full(sample_expression.shape[1], np.nan, dtype=np.float32))

    for row_idx, row in enumerate(rows):
        row["feature_row_index"] = int(row_idx)
        row["processed_row_index"] = int(row_idx)
        row["expression_row_index"] = int(row_idx)
        row["source_task"] = RUN_NAME
        row["task_context"] = RUN_NAME
    return rows, vectors


def write_task(
    *,
    task_name: str,
    label_key: str,
    label_description: str,
    base_rows: list[dict[str, Any]],
    base_vectors: list[np.ndarray],
    info: pd.DataFrame,
    output_root: Path,
    ordered_protein_index: list[int],
    ordered_uniprot: list[str],
    split_summary: dict[str, Any],
) -> dict[str, Any]:
    task_dir = output_root / DATASET_GROUP / "tasks" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in base_rows]
    label_by_raw_idx = {
        int(idx): int(value)
        for idx, value in info[label_key].astype(int).items()
    }
    for row in rows:
        if bool(row["is_control"]):
            continue
        value = label_by_raw_idx[int(row["source_info_row_index"])]
        row["synergy"] = int(value)
        row["PRISM1st_label_total"] = int(value)
        row["exp31_label_value"] = int(value)
        row["exp31_label_source"] = label_key
        row["unified_label_mask"] = 0.0

    df = pd.DataFrame(rows).reset_index(drop=True)
    matrix = np.stack(base_vectors, axis=0).astype(np.float32, copy=False)
    csv_path, native_path = save_dataframe(df, task_dir)
    df.to_csv(task_dir / "processed.csv", index=False)
    np.save(task_dir / "feature_expression_matrix.npy", matrix)
    np.save(task_dir / "processed_expression_matrix.npy", matrix)
    dump_json(task_dir / "feature_ordered_protein_index.json", ordered_protein_index)
    dump_json(task_dir / "processed_ordered_protein_index.json", ordered_protein_index)
    if ordered_uniprot:
        dump_json(task_dir / "feature_ordered_protein_uniprot.json", ordered_uniprot)
        dump_json(task_dir / "processed_ordered_protein_uniprot.json", ordered_uniprot)
    sample_ids = df["sample_id"].astype(str).tolist()
    dump_json(task_dir / "feature_sample_ids.json", sample_ids)
    dump_json(task_dir / "processed_sample_ids.json", sample_ids)
    dump_json(
        task_dir / "feature_loading_manifest.json",
        {
            "generated_at": iso_now(),
            "task_name": task_name,
            "label_key": label_key,
            "label_description": label_description,
            "row_key_column": "sample_id",
            "expression_row_index_column": "expression_row_index",
            "expression_matrix_path": str(task_dir / "feature_expression_matrix.npy"),
            "ordered_protein_index_path": str(task_dir / "feature_ordered_protein_index.json"),
            "ordered_protein_uniprot_path": str(task_dir / "feature_ordered_protein_uniprot.json"),
            "sample_ids_path": str(task_dir / "feature_sample_ids.json"),
            "feature_table_csv_path": csv_path,
            "feature_table_native_path": native_path,
            "loading_contract": "exp31 BRCA fine-tune and non-BRCA zero-shot PDX RNA task",
            "unified_label_note": "synergy column carries the current exp31 clinical binary label for unified-head compatibility; it is not a drug synergy label.",
        },
    )

    train_samples = set(split_summary["train_samples"])
    valid_samples = set(split_summary["valid_samples"])
    test_samples = set(
        info.loc[~info["cancer_type"].eq("BRCA"), "sample_id"].astype(str).unique().tolist()
    )
    split_manifest = write_split_files(
        split_dir=output_root / DATASET_GROUP / "splits" / task_name,
        df=df,
        split_samples={"train": train_samples, "valid": valid_samples, "test": test_samples},
        split_summary=split_summary,
    )
    perturb = df.loc[~df["is_control"].astype(bool)]
    return {
        "task_name": task_name,
        "label_key": label_key,
        "label_description": label_description,
        "task_dir": str(task_dir),
        "rows": int(len(df)),
        "control_rows": int(df["is_control"].astype(bool).sum()),
        "perturb_rows": int(len(perturb)),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "feature_table_csv_path": csv_path,
        "feature_table_native_path": native_path,
        "split_manifest": split_manifest,
        "label_counts": {
            "train": label_counts_for_samples(info, label_key, split_summary["train_samples"]),
            "valid": label_counts_for_samples(info, label_key, split_summary["valid_samples"]),
            "test": label_counts_for_samples(info, label_key, sorted(test_samples)),
        },
    }


def label_counts_for_samples(info: pd.DataFrame, label_key: str, samples: list[str]) -> dict[str, int]:
    values = info.loc[info["sample_id"].astype(str).isin(set(samples)), label_key].astype(int)
    pos = int(values.sum())
    return {"count": int(len(values)), "positive": pos, "negative": int(len(values) - pos)}


def extend_meta_and_artifacts(
    *,
    source_root: Path,
    output_root: Path,
    source_meta: dict[str, Any],
    new_pert_smiles: dict[str, str],
    task_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    derived_out = output_root / DATASET_GROUP / "derived"
    derived_out.mkdir(parents=True, exist_ok=True)
    derived_src = source_root / DATASET_GROUP / "derived"
    meta = json.loads(json.dumps(source_meta))

    pert_index = {str(key): int(value) for key, value in meta["pert_index"].items()}
    pert_index_to_id = list(meta.get("pert_index_to_id", []))
    if len(pert_index_to_id) < len(pert_index):
        pert_index_to_id = [None] * len(pert_index)
        for pert_id, idx in pert_index.items():
            pert_index_to_id[int(idx)] = pert_id

    added_pert_ids: list[str] = []
    for pert_id in sorted(new_pert_smiles):
        if pert_id in pert_index:
            continue
        pert_index[pert_id] = len(pert_index_to_id)
        pert_index_to_id.append(pert_id)
        added_pert_ids.append(pert_id)
    meta["pert_index"] = pert_index
    meta["pert_index_to_id"] = pert_index_to_id
    meta.setdefault("pertid_to_smiles", {})
    meta.setdefault("pertid_to_target_protein_list", {})
    for pert_id, smiles in new_pert_smiles.items():
        meta["pertid_to_smiles"][pert_id] = smiles
        meta["pertid_to_target_protein_list"][pert_id] = []
    meta["task_names"] = task_names
    meta["generated_at"] = iso_now()
    meta["exp31_rnaseq_extension"] = {
        "source_training_ready_root": str(source_root.resolve()),
        "added_pert_count": int(len(added_pert_ids)),
        "added_pert_ids": added_pert_ids,
        "protein_axis_policy": "preserve exp09 checkpoint protein axis; do not extend source protein_index",
        "pdi_policy_for_added_drugs": "append all-zero PDI rows for new SMILES-only drugs",
        "ddi_policy_for_added_drugs": "append Morgan/Tanimoto DDI rows and columns",
    }
    dump_json(output_root / DATASET_GROUP / "global_meta.json", meta)

    # Copy unchanged artifacts first.
    copied: dict[str, str] = {}
    for name in [
        "ppi_matrix.npy",
        "ppi_matrix.meta.json",
        "protein_embedding_esm.pkl",
        "protein_embedding_esm.meta.json",
        "cell_type_llm_embedding_qwen3_4096_v2.npz",
        "cell_type_llm_embedding_qwen3_4096_v2.json",
    ]:
        src = derived_src / name
        if src.exists():
            dst = derived_out / name
            shutil.copy2(src, dst)
            copied[name] = str(dst)

    drug_summary = write_extended_drug_embedding(
        source_path=derived_src / "drug_embedding_morgan_2048.pkl",
        output_path=derived_out / "drug_embedding_morgan_2048.pkl",
        meta=meta,
        source_meta=source_meta,
    )
    ddi_summary = write_extended_ddi(
        source_path=derived_src / "ddi_matrix.npy",
        output_path=derived_out / "ddi_matrix.npy",
        meta=meta,
        source_meta=source_meta,
    )
    pdi_summary = write_extended_pdi(
        source_path=derived_src / "pdi_matrix.npy",
        output_path=derived_out / "pdi_matrix.npy",
        meta=meta,
        source_meta=source_meta,
    )
    return meta, {
        "copied_artifacts": copied,
        "drug_embedding": drug_summary,
        "ddi_matrix": ddi_summary,
        "pdi_matrix": pdi_summary,
    }


def morgan_fingerprint(smiles: str, *, radius: int = 2, n_bits: int = 2048) -> tuple[Any, np.ndarray, str | None]:
    mol = Chem.MolFromSmiles(safe_text(smiles))
    fallback = None
    if mol is None:
        mol = Chem.MolFromSmiles("")
        fallback = "invalid_or_missing_smiles_empty_fingerprint"
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    vector = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, vector)
    return fp, vector, fallback


def write_extended_drug_embedding(
    *,
    source_path: Path,
    output_path: Path,
    meta: dict[str, Any],
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    payload = load_pickle(source_path)
    source_matrix = np.asarray(payload["embedding_matrix"], dtype=np.float32)
    output_matrix = np.zeros((len(meta["pert_index"]), source_matrix.shape[1]), dtype=np.float32)
    source_index = {str(key): int(value) for key, value in source_meta["pert_index"].items()}
    fallback_items = dict(payload.get("smiles_fallback_items", {}))
    copied_rows = 0
    generated_rows = 0
    for pert_id, target_idx in meta["pert_index"].items():
        target_idx = int(target_idx)
        if pert_id in source_index:
            output_matrix[target_idx] = source_matrix[source_index[pert_id]]
            copied_rows += 1
            continue
        _, vector, fallback = morgan_fingerprint(meta["pertid_to_smiles"].get(pert_id, ""))
        output_matrix[target_idx] = vector
        generated_rows += 1
        if fallback is not None:
            fallback_items[pert_id] = fallback
    out_payload = dict(payload)
    out_payload.update(
        {
            "item_to_index": meta["pert_index"],
            "index_to_item": ordered_ids(meta["pert_index"]),
            "embedding_matrix": output_matrix,
            "smiles_fallback_items": fallback_items,
            "exp31_extension": {
                "source_embedding": str(source_path),
                "copied_rows": copied_rows,
                "generated_rows": generated_rows,
            },
        }
    )
    dump_pickle(output_path, out_payload)
    meta_payload = {
        "generated_at": iso_now(),
        "kind": "drug_embedding",
        "source_embedding": str(source_path),
        "shape": list(output_matrix.shape),
        "copied_rows": copied_rows,
        "generated_rows": generated_rows,
        "fallback_item_count": len(fallback_items),
    }
    dump_json(output_path.with_suffix(".meta.json"), meta_payload)
    return meta_payload


def write_extended_ddi(
    *,
    source_path: Path,
    output_path: Path,
    meta: dict[str, Any],
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    source = np.load(source_path)
    n_total = len(meta["pert_index"])
    output = np.zeros((n_total, n_total), dtype=np.float32)
    n_source = int(source.shape[0])
    output[:n_source, :n_source] = np.asarray(source, dtype=np.float32)

    pert_order = ordered_ids(meta["pert_index"])
    source_index = {str(key): int(value) for key, value in source_meta["pert_index"].items()}
    fingerprints: list[Any | None] = [None] * n_total
    fallback_items: dict[str, str] = {}
    for pert_id in pert_order:
        idx = int(meta["pert_index"][pert_id])
        smiles = meta.get("pertid_to_smiles", {}).get(pert_id, "")
        fp, _, fallback = morgan_fingerprint(smiles)
        fingerprints[idx] = fp
        if fallback is not None:
            fallback_items[pert_id] = fallback

    new_indices = [int(meta["pert_index"][pert_id]) for pert_id in pert_order if pert_id not in source_index]
    for row_idx in new_indices:
        fp = fingerprints[row_idx]
        if fp is None:
            continue
        output[row_idx, row_idx] = 1.0
        sims = DataStructs.BulkTanimotoSimilarity(fp, [fingerprints[col] for col in range(n_total)])
        for col_idx, sim in enumerate(sims):
            output[row_idx, col_idx] = float(sim)
            output[col_idx, row_idx] = float(sim)
        output[row_idx, row_idx] = 1.0

    np.save(output_path, output)
    payload = {
        "generated_at": iso_now(),
        "kind": "ddi_matrix",
        "source_matrix": str(source_path),
        "shape": list(output.shape),
        "source_shape": list(source.shape),
        "new_row_count": int(len(new_indices)),
        "row_axis": "pert_index",
        "col_axis": "pert_index",
        "fallback_items": fallback_items,
        "policy": "source block copied unchanged; new rows/columns computed by Morgan/Tanimoto",
    }
    dump_json(output_path.with_suffix(".meta.json"), payload)
    return payload


def write_extended_pdi(
    *,
    source_path: Path,
    output_path: Path,
    meta: dict[str, Any],
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    source = np.load(source_path)
    n_total = len(meta["pert_index"])
    output = np.zeros((n_total, source.shape[1]), dtype=np.float32)
    output[: source.shape[0], :] = np.asarray(source, dtype=np.float32)
    new_count = n_total - int(source.shape[0])
    np.save(output_path, output)
    payload = {
        "generated_at": iso_now(),
        "kind": "pdi_matrix",
        "source_matrix": str(source_path),
        "shape": list(output.shape),
        "source_shape": list(source.shape),
        "new_row_count": int(new_count),
        "new_row_policy": "all-zero PDI rows for exp31 SMILES-only drugs",
        "row_axis": "pert_index",
        "col_axis": "protein_index",
        "source_pert_count": int(len(source_meta["pert_index"])),
    }
    dump_json(output_path.with_suffix(".meta.json"), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-path", default=str(RAW_RNA))
    parser.add_argument("--info-path", default=str(RAW_INFO))
    parser.add_argument("--source-training-ready-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-manifest", default=str(DEFAULT_EXP09_MANIFEST))
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-end", type=int, default=1042)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_training_ready_root)
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"output root is not empty; use --force to rebuild: {output_root}")
    if args.force and output_root.exists():
        shutil.rmtree(output_root)

    source_meta_path = source_root / DATASET_GROUP / "global_meta.json"
    source_meta = load_json(source_meta_path)
    ordered_protein_index, ordered_uniprot = read_exp09_axis(Path(args.checkpoint_manifest))

    info_raw = pd.read_csv(args.info_path, low_memory=False)
    required_cols = {"sample_id", "cancer_type", "smiles_a", "smiles_b", "treatment_components", "ptv_component_guomics_ids"}
    missing = sorted(required_cols - set(info_raw.columns))
    if missing:
        raise ValueError(f"sample info missing required columns: {missing}")
    for _, label_key, _ in LABEL_TASKS:
        if label_key not in info_raw.columns:
            raise ValueError(f"sample info missing label column: {label_key}")

    missing_cancer = info_raw["cancer_type"].isna() | info_raw["cancer_type"].map(safe_text).eq("")
    info = info_raw.loc[~missing_cancer].copy().reset_index(drop=True)
    for _, label_key, _ in LABEL_TASKS:
        values = set(info[label_key].dropna().astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"{label_key} has non-binary values: {sorted(values)}")

    pert_index = {str(key): int(value) for key, value in source_meta["pert_index"].items()}
    drug_resolutions: dict[int, list[ResolvedDrug]] = {}
    new_pert_smiles: dict[str, str] = {}
    for raw_idx, raw in info.iterrows():
        smiles_slots = [safe_text(raw.get("smiles_a"))]
        if safe_text(raw.get("smiles_b")):
            smiles_slots.append(safe_text(raw.get("smiles_b")))
        components = split_components(raw.get("treatment_components"))
        if len(components) != len(smiles_slots):
            components = split_components(raw.get("treatment"))
        if len(components) != len(smiles_slots):
            components = [f"slot{i + 1}" for i in range(len(smiles_slots))]
        candidates = parse_component_candidates(raw.get("ptv_component_guomics_ids"))
        resolved = [
            resolve_drug_slot(
                component=component,
                smiles=smiles,
                candidates_by_component=candidates,
                pert_index=pert_index,
            )
            for component, smiles in zip(components, smiles_slots, strict=True)
        ]
        for drug in resolved:
            if drug.source == "new_smiles_id":
                new_pert_smiles.setdefault(drug.pert_id, drug.smiles)
        drug_resolutions[int(raw_idx)] = resolved

    task_names = [f"ptv3_exp31_rnaseq_{suffix}" for suffix, _, _ in LABEL_TASKS]
    meta, artifact_summary = extend_meta_and_artifacts(
        source_root=source_root,
        output_root=output_root,
        source_meta=source_meta,
        new_pert_smiles=new_pert_smiles,
        task_names=task_names,
    )

    sample_ids = sorted(info["sample_id"].astype(str).unique().tolist())
    sample_expression, rna_audit = load_rna_expression(
        rna_path=Path(args.rna_path),
        sample_ids=sample_ids,
        ordered_protein_index=ordered_protein_index,
        meta=meta,
    )
    sample_to_expr_row = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    base_rows, base_vectors = build_base_rows(
        info=info,
        sample_ids=sample_ids,
        sample_expression=sample_expression,
        sample_to_expr_row=sample_to_expr_row,
        meta=meta,
        drug_resolutions=drug_resolutions,
    )

    tasks: dict[str, Any] = {}
    split_summaries: dict[str, Any] = {}
    for suffix, label_key, label_description in LABEL_TASKS:
        task_name = f"ptv3_exp31_rnaseq_{suffix}"
        split_summary = find_brca_split(
            info,
            label_key,
            seed_start=int(args.seed_start),
            seed_end=int(args.seed_end),
        )
        split_summaries[task_name] = split_summary
        tasks[task_name] = write_task(
            task_name=task_name,
            label_key=label_key,
            label_description=label_description,
            base_rows=base_rows,
            base_vectors=base_vectors,
            info=info,
            output_root=output_root,
            ordered_protein_index=ordered_protein_index,
            ordered_uniprot=ordered_uniprot,
            split_summary=split_summary,
        )

    resolution_counts: dict[str, int] = {}
    for resolved in drug_resolutions.values():
        for drug in resolved:
            resolution_counts[drug.source] = resolution_counts.get(drug.source, 0) + 1

    summary = {
        "generated_at": iso_now(),
        "run_name": RUN_NAME,
        "raw_info_path": str(Path(args.info_path).resolve()),
        "raw_rna_path": str(Path(args.rna_path).resolve()),
        "source_training_ready_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "checkpoint_manifest": str(Path(args.checkpoint_manifest).resolve()),
        "source_meta_path": str(source_meta_path.resolve()),
        "meta_path": str((output_root / DATASET_GROUP / "global_meta.json").resolve()),
        "raw_rows": int(len(info_raw)),
        "dropped_missing_cancer_type_rows": int(missing_cancer.sum()),
        "kept_rows": int(len(info)),
        "unique_samples": int(info["sample_id"].nunique()),
        "cancer_type_counts": info["cancer_type"].value_counts(dropna=False).to_dict(),
        "treatment_type_counts": {
            "single": int(info["smiles_b"].isna().sum()),
            "double": int(info["smiles_b"].notna().sum()),
        },
        "new_pert_count": int(len(new_pert_smiles)),
        "new_pert_smiles": new_pert_smiles,
        "drug_resolution_counts": resolution_counts,
        "rna_audit": rna_audit,
        "artifact_summary": artifact_summary,
        "tasks": tasks,
        "split_summaries": split_summaries,
        "notes": [
            "Original data/training_ready/ptv3 artifacts are read-only inputs.",
            "New exp31 SMILES drugs receive Morgan and DDI features, but PDI rows are intentionally zero.",
            "The synergy column in task tables carries the current exp31 binary clinical label for unified-head compatibility.",
            "Cell_index and cell_type_index remain checkpoint-compatible and map PDX-specific categories to no; cell_llm_index carries PDX sample/cancer text semantics.",
        ],
    }
    summary_path = output_root / DATASET_GROUP / "exp31_rnaseq_build_summary.json"
    dump_json(summary_path, summary)
    print(f"[exp31] wrote root: {output_root}")
    print(f"[exp31] wrote tasks: {', '.join(tasks)}")
    print(f"[exp31] new pert ids: {len(new_pert_smiles)}")
    print(f"[exp31] summary: {summary_path}")


if __name__ == "__main__":
    main()
