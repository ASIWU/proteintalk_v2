#!/usr/bin/env python3
"""Build compact exp33 double-drug virtual-screen inference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.npy_io import safe_np_load


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GROUP = "ptv3"
EXPERIMENT_NAME = "exp33_vc_doubledrug_epoch2"
SPLIT_STRATEGY = "test_only"
EXPECTED_AXIS_SIZE = 11_092
EXPECTED_DRUG_COUNT = 227
EXPECTED_CELL_COUNT = 28
EXPECTED_RAW_ROWS = 2_526_720
EXPECTED_UNIQUE_ROWS = 1_124_928
EXPECTED_CHECKPOINT_SHA256 = "7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076"

DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/training_ready"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/training_ready_exp33_vc_doubledrug"
DEFAULT_RAW_ROOT = REPO_ROOT / "data/rawdata/vc_doubledrug"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt"
)
STANDARDIZED_MAIN_SINGLE = (
    REPO_ROOT / "data/standardized/ptv3/tasks/ptv3_main_singledrug/info.csv"
)
GRAPH_FEATURE_PATH = (
    REPO_ROOT / "graph_cache/ptv01_08_posweight_combo/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy"
)

QUERY_COLUMNS = [
    "cell",
    "drug_A_name",
    "drug_A_smiles",
    "drug_B_name",
    "drug_B_smiles",
    "drug_A_concentration_uM",
    "assumed_combo_IC50_B_uM",
]
KEY_COLUMNS = [
    "tissue",
    "source_cell",
    "pert_id1",
    "pert_id2",
    "pert_dose1_index",
    "pert_dose2_index",
    "pert_time",
]
UNSEEN_CELLS = {
    "COLO205",
    "COLO320",
    "DLD1",
    "HCC2935",
    "HCT8",
    "NCIH1666",
    "NCIH1755",
    "NCIH526",
    "NCIH716",
    "SHP77",
}
TISSUE_SPECS = {
    "colon": {
        "task_name": "ptv3_exp33_vc_colon_double",
        "query_glob": "PTV2_virtual_screen_colon_IC50_robustness_7col_260722.csv",
        "control_glob": "*_colon_*_control.csv",
        "control_unique_glob": "*_colon_*_control_unique.csv",
        "raw_rows": 403_200,
        "unique_rows": 218_736,
        "cells": 6,
        "expression_rows": 7,
        "cell_type": "COLON",
    },
    "lung": {
        "task_name": "ptv3_exp33_vc_lung_double",
        "query_glob": "PTV2_virtual_screen_lung_IC50_robustness_7col_260722.csv",
        "control_glob": "*_lung_*_control.csv",
        "control_unique_glob": "*_lung_*_control_unique.csv",
        "raw_rows": 806_400,
        "unique_rows": 462_210,
        "cells": 15,
        "expression_rows": 16,
        "cell_type": "LUNG",
    },
    "pancreas": {
        "task_name": "ptv3_exp33_vc_pancreas_double",
        "query_glob": "PTV2_virtual_screen_pancreas_IC50_robustness_7col_260722.csv",
        "control_glob": "*_pancreas_*_control.csv",
        "control_unique_glob": "*_pancreas_*_control_unique.csv",
        "raw_rows": 1_317_120,
        "unique_rows": 443_982,
        "cells": 7,
        "expression_rows": 8,
        "cell_type": "PANCREAS",
    },
}


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def dump_pickle(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def canonical_text(value: object) -> str:
    text = clean_text(value)
    if not text:
        return "no"
    text = re.sub(r"[^A-Z0-9]+", "_", text.upper())
    return re.sub(r"_+", "_", text).strip("_") or "no"


def compact_number(value: float) -> str:
    return f"{float(value):g}"


def canonical_isomeric_smiles(value: object) -> str:
    text = clean_text(value)
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {text!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def find_exactly_one(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one file for {root / pattern}, found {matches}")
    return matches[0]


def value_index(meta: dict[str, Any], field: str, value: str) -> int:
    mapping_key = "pert_dose" if field in {"pert_dose1", "pert_dose2"} else field
    mapping = meta["value_to_index"][mapping_key]
    if value not in mapping:
        raise KeyError(f"missing categorical mapping {mapping_key}={value!r}")
    return int(float(mapping[value]))


def feature_table_path(task_dir: Path) -> Path:
    for name in ("feature_table.parquet", "feature_table.pkl", "feature_table.csv"):
        path = task_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing feature table under {task_dir}")


def read_feature_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".pkl":
        frame = pd.read_pickle(path)
        return frame if columns is None else frame[columns]
    return pd.read_csv(path, usecols=columns, low_memory=False)


def checkpoint_axis(checkpoint_manifest: dict[str, Any]) -> tuple[list[int], list[str], Path, Path]:
    index_path = Path(str(checkpoint_manifest["ordered_protein_index_path"]))
    task_dir = Path(str(checkpoint_manifest["task_dir"]))
    uniprot_path = task_dir / "feature_ordered_protein_uniprot.json"
    indices = [int(item) for item in load_json(index_path)]
    uniprot = [str(item) for item in load_json(uniprot_path)]
    if len(indices) != EXPECTED_AXIS_SIZE or len(uniprot) != EXPECTED_AXIS_SIZE:
        raise ValueError(
            f"checkpoint axis must contain {EXPECTED_AXIS_SIZE} entries, "
            f"got {len(indices)} indices and {len(uniprot)} UniProt IDs"
        )
    if len(indices) != len(set(indices)) or len(uniprot) != len(set(uniprot)):
        raise ValueError("checkpoint protein axis contains duplicates")
    return indices, uniprot, index_path, uniprot_path


def checkpoint_training_support(checkpoint_manifest: dict[str, Any]) -> dict[str, Any]:
    task_dir = Path(str(checkpoint_manifest["task_dir"]))
    split_dir = Path(str(checkpoint_manifest["split_summary"]["split_dir"]))
    strategy = str(checkpoint_manifest["split_strategy"])
    columns = [
        "Cell",
        "Cell_norm",
        "machineID_new_index",
        "Cell_plate_index",
        "Cell_index",
        "cell_type_index",
        "batch_index",
        "pert_time_index",
        "pert_dose1_index",
        "pert_dose2_index",
    ]
    table = read_feature_table(feature_table_path(task_dir), columns=columns)
    train_indices = [int(item) for item in load_pickle(split_dir / f"train_indices_{strategy}.pkl")]
    train = table.iloc[train_indices]
    support: dict[str, Any] = {
        "checkpoint_task": str(checkpoint_manifest["task_name"]),
        "split_strategy": strategy,
        "train_rows": int(len(train)),
        "seen_cell_names": sorted(train["Cell_norm"].astype(str).unique().tolist()),
    }
    for column in columns[2:]:
        support[column] = sorted(
            pd.to_numeric(train[column], errors="raise").astype(np.int64).unique().tolist()
        )
    return support


def collect_query_drug_records(raw_paths: dict[str, Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for tissue, path in raw_paths.items():
        for chunk in pd.read_csv(
            path,
            usecols=["drug_A_name", "drug_A_smiles", "drug_B_name", "drug_B_smiles"],
            chunksize=100_000,
            low_memory=False,
        ):
            side_a = chunk[["drug_A_name", "drug_A_smiles"]].rename(
                columns={"drug_A_name": "query_drug_name", "drug_A_smiles": "query_smiles"}
            )
            side_b = chunk[["drug_B_name", "drug_B_smiles"]].rename(
                columns={"drug_B_name": "query_drug_name", "drug_B_smiles": "query_smiles"}
            )
            side_a["observed_tissue"] = tissue
            side_b["observed_tissue"] = tissue
            parts.extend([side_a.drop_duplicates(), side_b.drop_duplicates()])
    records = pd.concat(parts, ignore_index=True)
    records["query_drug_name"] = records["query_drug_name"].map(clean_text)
    records["query_smiles"] = records["query_smiles"].map(clean_text)
    grouped = (
        records.groupby(["query_drug_name", "query_smiles"], sort=False)["observed_tissue"]
        .agg(lambda values: json.dumps(sorted(set(values)), separators=(",", ":")))
        .reset_index()
    )
    if len(grouped) != EXPECTED_DRUG_COUNT:
        raise ValueError(f"expected {EXPECTED_DRUG_COUNT} query drug records, found {len(grouped)}")
    if grouped["query_drug_name"].eq("").any() or grouped["query_smiles"].eq("").any():
        raise ValueError("query drug records contain empty names or SMILES")
    return grouped


def drug_id_tier(pert_id: str) -> str:
    if re.fullmatch(r"\d+", pert_id):
        return "numeric"
    if pert_id.startswith("L9200_"):
        return "L9200"
    if pert_id.startswith(("extid::", "extsmiles::")):
        return "external"
    return "other"


def build_drug_mapping(
    *,
    query_records: pd.DataFrame,
    meta: dict[str, Any],
    derived_root: Path,
    graph_feature_path: Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    registry: dict[str, list[str]] = defaultdict(list)
    for pert_id, smiles in meta["pertid_to_smiles"].items():
        if clean_text(smiles):
            registry[canonical_isomeric_smiles(smiles)].append(str(pert_id))

    pert_index = {str(key): int(value) for key, value in meta["pert_index"].items()}
    audit_rows: list[dict[str, Any]] = []
    canonical_to_selected: dict[str, str] = {}
    for record in query_records.itertuples(index=False):
        canonical = canonical_isomeric_smiles(record.query_smiles)
        candidates = sorted(registry.get(canonical, []), key=lambda item: pert_index[item])
        if not candidates:
            raise ValueError(
                f"no model drug ID matches {record.query_drug_name!r} by canonical isomeric SMILES"
            )
        tier_candidates = {
            tier: [item for item in candidates if drug_id_tier(item) == tier]
            for tier in ("numeric", "L9200", "external", "other")
        }
        selected_tier = next((tier for tier in tier_candidates if tier_candidates[tier]), "")
        selected_candidates = tier_candidates[selected_tier]
        exception = ""
        if record.query_drug_name == "(-)-Menthol":
            if "L9200_2195" not in candidates:
                raise ValueError("confirmed (-)-Menthol ID L9200_2195 is absent from structure matches")
            selected_id = "L9200_2195"
            selected_tier = "L9200"
            exception = "confirmed native-double alias"
        else:
            if len(selected_candidates) != 1:
                raise ValueError(
                    f"ambiguous highest-priority drug IDs for {record.query_drug_name!r}: "
                    f"tier={selected_tier} candidates={selected_candidates}"
                )
            selected_id = selected_candidates[0]
        prior = canonical_to_selected.setdefault(canonical, selected_id)
        if prior != selected_id:
            raise ValueError(f"one query structure resolved inconsistently: {prior} vs {selected_id}")
        audit_rows.append(
            {
                "query_drug_name": record.query_drug_name,
                "query_smiles": record.query_smiles,
                "canonical_isomeric_smiles": canonical,
                "observed_tissues": record.observed_tissue,
                "candidate_count": len(candidates),
                "candidate_ids": json.dumps(candidates, ensure_ascii=False, separators=(",", ":")),
                "numeric_candidate_ids": json.dumps(
                    tier_candidates["numeric"], ensure_ascii=False, separators=(",", ":")
                ),
                "L9200_candidate_ids": json.dumps(
                    tier_candidates["L9200"], ensure_ascii=False, separators=(",", ":")
                ),
                "external_candidate_ids": json.dumps(
                    tier_candidates["external"], ensure_ascii=False, separators=(",", ":")
                ),
                "selected_priority_tier": selected_tier,
                "selected_pert_id": selected_id,
                "selected_pert_index": pert_index[selected_id],
                "selection_exception": exception,
                "matched_by_name_fallback": False,
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values(
        ["selected_pert_index", "query_drug_name"], kind="stable"
    )
    if len(audit) != EXPECTED_DRUG_COUNT or audit["selected_pert_id"].nunique() != EXPECTED_DRUG_COUNT:
        raise ValueError("query drug mapping is not a one-to-one set of 227 selected model IDs")
    tier_counts = audit["selected_priority_tier"].value_counts().to_dict()
    if tier_counts != {"L9200": 176, "numeric": 51}:
        raise ValueError(f"unexpected selected drug-ID tier counts: {tier_counts}")

    with (derived_root / "drug_embedding_morgan_2048.pkl").open("rb") as handle:
        morgan_payload = pickle.load(handle)
    morgan = np.asarray(morgan_payload["embedding_matrix"], dtype=np.float32)
    graph = safe_np_load(graph_feature_path, mmap_mode="r")
    pdi = safe_np_load(derived_root / "pdi_matrix.npy", mmap_mode="r")
    ddi = safe_np_load(derived_root / "ddi_matrix.npy", mmap_mode="r")
    selected_indices = audit["selected_pert_index"].to_numpy(dtype=np.int64)
    for label, row_count in {
        "Morgan": morgan.shape[0],
        "graph": graph.shape[0],
        "PDI": pdi.shape[0],
        "DDI": ddi.shape[0],
    }.items():
        if selected_indices.min() < 0 or selected_indices.max() >= row_count:
            raise ValueError(f"selected drug index exceeds {label} rows")
    if ddi.ndim != 2 or ddi.shape[1] != ddi.shape[0]:
        raise ValueError(f"DDI matrix must be square, got {ddi.shape}")
    if not np.isfinite(morgan[selected_indices]).all():
        raise ValueError("selected Morgan rows contain non-finite values")
    if not np.isfinite(np.asarray(graph[selected_indices], dtype=np.float32)).all():
        raise ValueError("selected graph rows contain non-finite values")
    if not np.isfinite(np.asarray(pdi[selected_indices], dtype=np.float32)).all():
        raise ValueError("selected PDI rows contain non-finite values")
    selected_ddi = np.asarray(ddi[np.ix_(selected_indices, selected_indices)], dtype=np.float32)
    if not np.isfinite(selected_ddi).all():
        raise ValueError("selected DDI submatrix contains non-finite values")
    feature_audit = {
        "query_drugs": len(audit),
        "selected_id_tier_counts": {str(key): int(value) for key, value in tier_counts.items()},
        "external_selected_ids": int(audit["selected_pert_id"].str.startswith(("extid::", "extsmiles::")).sum()),
        "morgan_shape": list(map(int, morgan.shape)),
        "graph_shape": list(map(int, graph.shape)),
        "pdi_shape": list(map(int, pdi.shape)),
        "ddi_shape": list(map(int, ddi.shape)),
        "all_selected_feature_indices_legal": True,
        "all_selected_feature_values_finite": True,
    }
    return audit.reset_index(drop=True), canonical_to_selected, feature_audit


def load_control_metadata(
    *,
    raw_root: Path,
    raw_control_paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, str]]]]:
    raw_parts: list[pd.DataFrame] = []
    all_sample_ids: list[str] = []
    for tissue, path in raw_control_paths.items():
        frame = pd.read_csv(path, usecols=["sample_id", "cell"], low_memory=False)
        frame["tissue"] = tissue
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["source_cell"] = frame["cell"].map(clean_text)
        raw_parts.append(frame[["tissue", "source_cell", "sample_id"]])
        all_sample_ids.extend(frame["sample_id"].tolist())
    if len(all_sample_ids) != 111 or len(set(all_sample_ids)) != 111:
        raise ValueError("raw control sample IDs must contain exactly 111 unique records")

    header = pd.read_csv(STANDARDIZED_MAIN_SINGLE, nrows=0)
    required = [
        "sample_id",
        "machineID_new",
        "Cell",
        "cell_type",
        "Cell_plate",
        "batch",
        "pert_time",
        "control",
    ]
    missing = sorted(set(required) - set(header.columns))
    if missing:
        raise ValueError(f"standardized main-single metadata lacks columns: {missing}")
    standardized = pd.read_csv(
        STANDARDIZED_MAIN_SINGLE,
        usecols=required,
        low_memory=False,
    )
    standardized["sample_id"] = standardized["sample_id"].astype(str)
    standardized = standardized.loc[standardized["sample_id"].isin(all_sample_ids)].copy()
    if len(standardized) != 111 or standardized["sample_id"].nunique() != 111:
        raise ValueError("not all 111 raw control sample IDs map uniquely to standardized metadata")
    merged = pd.concat(raw_parts, ignore_index=True).merge(
        standardized,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged[["machineID_new", "Cell", "cell_type"]].isna().any().any():
        raise ValueError("control metadata join produced missing model covariates")

    cell_rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for (tissue, source_cell), group in merged.groupby(["tissue", "source_cell"], sort=False):
        unique_values: dict[str, str] = {}
        for field in ("machineID_new", "Cell", "cell_type"):
            values = sorted({clean_text(item) for item in group[field]})
            if len(values) != 1:
                raise ValueError(
                    f"{tissue}/{source_cell}: control samples disagree on {field}: {values}"
                )
            unique_values[field] = values[0]
        record = {
            "tissue": tissue,
            "source_cell": source_cell,
            "Cell": canonical_text(unique_values["Cell"]),
            "cell_type": canonical_text(unique_values["cell_type"]),
            "machineID_new": canonical_text(unique_values["machineID_new"]),
            "raw_control_sample_count": int(len(group)),
            "raw_control_6h_count": int(
                pd.to_numeric(group["pert_time"], errors="raise").eq(6).sum()
            ),
            "raw_control_24h_count": int(
                pd.to_numeric(group["pert_time"], errors="raise").eq(24).sum()
            ),
            "raw_control_sample_ids": json.dumps(
                group["sample_id"].astype(str).tolist(), separators=(",", ":")
            ),
            "raw_control_Cell_plate_count": int(group["Cell_plate"].nunique()),
            "raw_control_batch_count": int(group["batch"].nunique()),
        }
        cell_rows.append(record)
        metadata[tissue][source_cell] = {
            key: str(value)
            for key, value in record.items()
            if key in {"Cell", "cell_type", "machineID_new"}
        }
    cells = pd.DataFrame(cell_rows)
    if len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError(f"expected {EXPECTED_CELL_COUNT} baseline cells, found {len(cells)}")
    return cells, metadata


def load_baseline_matrix(
    *,
    tissue: str,
    unique_control_path: Path,
    source_cells: list[str],
    axis_uniprot: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    columns = pd.read_csv(unique_control_path, nrows=0).columns.astype(str).tolist()
    if not columns or columns[0] != "Cell" or len(columns) != len(set(columns)):
        raise ValueError(f"{unique_control_path}: invalid or duplicate baseline columns")
    available = set(columns[1:])
    selected_proteins = [protein for protein in axis_uniprot if protein in available]
    baseline = pd.read_csv(
        unique_control_path,
        usecols=["Cell", *selected_proteins],
        low_memory=False,
    )
    baseline["Cell"] = baseline["Cell"].map(clean_text)
    if baseline["Cell"].duplicated().any() or set(baseline["Cell"]) != set(source_cells):
        raise ValueError(f"{tissue}: unique baseline cells differ from query cells")
    baseline = baseline.set_index("Cell").loc[source_cells]
    aligned = np.full((len(source_cells) + 1, len(axis_uniprot)), np.nan, dtype=np.float32)
    axis_position = {protein: idx for idx, protein in enumerate(axis_uniprot)}
    selected_positions = [axis_position[protein] for protein in selected_proteins]
    values = baseline[selected_proteins].to_numpy(dtype=np.float32, copy=True)
    finite = np.isfinite(values)
    if np.isinf(values).any() or np.any(values[finite] < 0):
        raise ValueError(f"{tissue}: baseline contains infinite or negative finite values")
    values[finite] = np.log1p(values[finite])
    aligned[np.ix_(np.arange(len(source_cells)), selected_positions)] = values
    if not np.isnan(aligned[-1]).all():
        raise AssertionError("query expression sentinel must be entirely NaN")
    finite_by_cell = np.isfinite(aligned[:-1]).sum(axis=1)
    return aligned, {
        "tissue": tissue,
        "path": str(unique_control_path.resolve()),
        "sha256": sha256_file(unique_control_path),
        "cells": len(source_cells),
        "input_proteins": len(columns) - 1,
        "axis_overlap": len(selected_proteins),
        "axis_missing": len(axis_uniprot) - len(selected_proteins),
        "expression_rows": int(aligned.shape[0]),
        "query_nan_sentinel_row": int(aligned.shape[0] - 1),
        "finite_values_min_per_cell": int(finite_by_cell.min()),
        "finite_values_max_per_cell": int(finite_by_cell.max()),
        "transform": "finite nonnegative values log1p; absent/missing values remain NaN",
    }


def combine_targets(
    pert_id1: str,
    pert_id2: str,
    target_map: dict[str, list[int]],
) -> str:
    combined: list[int] = []
    for pert_id in (pert_id1, pert_id2):
        for value in target_map.get(pert_id, []):
            index = int(value)
            if index not in combined:
                combined.append(index)
    return json.dumps(combined, separators=(",", ":"))


def clipped_dose_bucket(values: pd.Series, *, column: str) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric < 0):
        raise ValueError(f"{column} contains non-finite or negative values")
    clipped = np.clip(numeric, 0.0, 10.0)
    buckets = np.ceil(clipped).astype(np.int64)
    return clipped, buckets


def build_unique_keys(
    *,
    tissue: str,
    query_path: Path,
    canonical_to_selected: dict[str, str],
    raw_smiles_to_canonical: dict[str, str],
    expected_rows: int,
    expected_unique_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    raw_rows = 0
    source_cells: list[str] = []
    dose1_min = math.inf
    dose1_max = -math.inf
    dose2_min = math.inf
    dose2_max = -math.inf
    dose2_clipped_rows = 0
    observed_a_buckets: set[int] = set()
    observed_b_buckets: set[int] = set()
    for chunk in pd.read_csv(query_path, chunksize=100_000, low_memory=False):
        if list(chunk.columns) != QUERY_COLUMNS:
            raise ValueError(f"{query_path}: query columns differ from the fixed seven-column schema")
        raw_rows += len(chunk)
        source_cells.extend(
            item for item in chunk["cell"].map(clean_text).tolist() if item not in source_cells
        )
        canonical_a = chunk["drug_A_smiles"].map(clean_text).map(raw_smiles_to_canonical)
        canonical_b = chunk["drug_B_smiles"].map(clean_text).map(raw_smiles_to_canonical)
        if canonical_a.isna().any() or canonical_b.isna().any():
            raise ValueError(f"{tissue}: a query SMILES is absent from the 227-drug audit")
        pert_id1 = canonical_a.map(canonical_to_selected)
        pert_id2 = canonical_b.map(canonical_to_selected)
        if pert_id1.isna().any() or pert_id2.isna().any():
            raise ValueError(f"{tissue}: a query structure lacks a selected model ID")
        clipped1, bucket1 = clipped_dose_bucket(
            chunk["drug_A_concentration_uM"], column="drug_A_concentration_uM"
        )
        clipped2, bucket2 = clipped_dose_bucket(
            chunk["assumed_combo_IC50_B_uM"], column="assumed_combo_IC50_B_uM"
        )
        raw1 = pd.to_numeric(chunk["drug_A_concentration_uM"], errors="raise").to_numpy(
            dtype=np.float64
        )
        raw2 = pd.to_numeric(chunk["assumed_combo_IC50_B_uM"], errors="raise").to_numpy(
            dtype=np.float64
        )
        dose1_min = min(dose1_min, float(raw1.min()))
        dose1_max = max(dose1_max, float(raw1.max()))
        dose2_min = min(dose2_min, float(raw2.min()))
        dose2_max = max(dose2_max, float(raw2.max()))
        dose2_clipped_rows += int(np.count_nonzero(raw2 > 10.0))
        observed_a_buckets.update(map(int, np.unique(bucket1)))
        observed_b_buckets.update(map(int, np.unique(bucket2)))
        compact = pd.DataFrame(
            {
                "tissue": tissue,
                "source_cell": chunk["cell"].map(clean_text),
                "pert_id1": pert_id1,
                "pert_id2": pert_id2,
                "pert_dose1_index": bucket1,
                "pert_dose2_index": bucket2,
                "pert_time": 24,
                "raw_dose1_representative_uM": raw1,
                "raw_dose2_representative_uM": raw2,
                "pert_dose1_clipped_representative_uM": clipped1,
                "pert_dose2_clipped_representative_uM": clipped2,
            }
        )
        parts.append(compact.drop_duplicates(subset=KEY_COLUMNS, keep="first"))
    if raw_rows != expected_rows:
        raise ValueError(f"{tissue}: expected {expected_rows} raw rows, found {raw_rows}")
    unique = pd.concat(parts, ignore_index=True).drop_duplicates(subset=KEY_COLUMNS, keep="first")
    unique = unique.reset_index(drop=True)
    if len(unique) != expected_unique_rows:
        raise ValueError(
            f"{tissue}: expected {expected_unique_rows} unique model keys, found {len(unique)}"
        )
    unique["model_key_rank"] = np.arange(len(unique), dtype=np.int64)
    unique["model_key_id"] = [
        f"exp33::{tissue}::{rank:07d}" for rank in unique["model_key_rank"].tolist()
    ]
    return unique, {
        "query_path": str(query_path.resolve()),
        "query_sha256": sha256_file(query_path),
        "raw_rows": raw_rows,
        "unique_model_keys": len(unique),
        "source_cells": source_cells,
        "dose1_raw_min": dose1_min,
        "dose1_raw_max": dose1_max,
        "dose2_raw_min": dose2_min,
        "dose2_raw_max": dose2_max,
        "dose2_rows_clipped_above_10": dose2_clipped_rows,
        "dose1_buckets": sorted(observed_a_buckets),
        "dose2_buckets": sorted(observed_b_buckets),
    }


def make_task_table(
    *,
    tissue: str,
    unique_keys: pd.DataFrame,
    source_cells: list[str],
    cell_metadata: dict[str, dict[str, str]],
    meta: dict[str, Any],
    training_support: dict[str, Any],
    drug_name_by_id: dict[str, str],
) -> tuple[pd.DataFrame, dict[int, dict[str, list[int]]], dict[int, int]]:
    no_pert_index = int(meta["pert_index"]["no"])
    no_indices = {
        field: value_index(meta, field, "no")
        for field in ("Cell_plate", "Cell", "batch", "pert_time", "pert_dose1", "pert_dose2")
    }
    time_index = value_index(meta, "pert_time", "24")
    pert_index = {str(key): int(value) for key, value in meta["pert_index"].items()}
    target_map = {
        str(key): [int(item) for item in values]
        for key, values in meta["pertid_to_target_protein_list"].items()
    }
    smiles_map = {str(key): clean_text(value) for key, value in meta["pertid_to_smiles"].items()}
    seen_cell_indices = set(map(int, training_support["Cell_index"]))

    cell_records: dict[str, dict[str, Any]] = {}
    controls: list[dict[str, Any]] = []
    for expression_row_index, source_cell in enumerate(source_cells):
        recovered = cell_metadata[source_cell]
        true_cell_index = value_index(meta, "Cell", recovered["Cell"])
        cell_seen = true_cell_index in seen_cell_indices
        if (recovered["Cell"] in UNSEEN_CELLS) == cell_seen:
            raise ValueError(
                f"{tissue}/{source_cell}: checkpoint seen/unseen result contradicts fixed audit "
                f"for Cell={recovered['Cell']}"
            )
        record = {
            "machineID_new": recovered["machineID_new"],
            "machineID_new_norm": recovered["machineID_new"],
            "machineID_new_index": value_index(
                meta, "machineID_new", recovered["machineID_new"]
            ),
            "Cell_plate": "no",
            "Cell_plate_norm": "no",
            "Cell_plate_index": no_indices["Cell_plate"],
            "Cell": recovered["Cell"],
            "Cell_norm": recovered["Cell"],
            "Cell_index": true_cell_index if cell_seen else no_indices["Cell"],
            "cell_llm_index": true_cell_index,
            "cell_type": recovered["cell_type"],
            "cell_type_norm": recovered["cell_type"],
            "cell_type_index": value_index(meta, "cell_type", recovered["cell_type"]),
            "cell_type_llm_index": value_index(meta, "cell_type", recovered["cell_type"]),
            "batch": "no",
            "batch_norm": "no",
            "batch_index": no_indices["batch"],
            "cell_seen_in_checkpoint_train": cell_seen,
            "true_Cell_index": true_cell_index,
        }
        cell_records[source_cell] = record
        control_id = f"exp33::{tissue}::{recovered['Cell']}::control"
        controls.append(
            {
                **record,
                "tissue": tissue,
                "source_cell": source_cell,
                "sample_id": control_id,
                "control": control_id,
                "is_control": True,
                "source_row_role": "exp33_mixed_time_unique_baseline_control",
                "source_task": TISSUE_SPECS[tissue]["task_name"],
                "task_context": EXPERIMENT_NAME,
                "feature_membership": "primary",
                "training_label_scope": "inference_only_unlabeled",
                "pert_id1": "no",
                "pert_id2": "no",
                "pert_index1": no_pert_index,
                "pert_index2": no_pert_index,
                "pert_time": "no",
                "pert_time_norm": "no",
                "pert_time_index": no_indices["pert_time"],
                "pert_dose1": "no",
                "pert_dose2": "no",
                "pert_dose1_norm": "no",
                "pert_dose2_norm": "no",
                "pert_dose1_index": no_indices["pert_dose1"],
                "pert_dose2_index": no_indices["pert_dose2"],
                "raw_dose1_representative_uM": np.nan,
                "raw_dose2_representative_uM": np.nan,
                "drugname": "",
                "drugname1": "",
                "drugname2": "",
                "smiles": "",
                "smiles1": "",
                "smiles2": "",
                "target_protein_list": "[]",
                "PRISM1st_label_total": "",
                "PRISM2nd_label_total": "",
                "synergy": "",
                "unified_label_mask": 1.0,
                "model_key_id": "",
                "model_key_rank": -1,
                "expression_row_index": expression_row_index,
            }
        )

    query = unique_keys.copy()
    for column in next(iter(cell_records.values())):
        query[column] = query["source_cell"].map(
            {cell: record[column] for cell, record in cell_records.items()}
        )
    if query.isna().any().get("Cell", False):
        raise ValueError(f"{tissue}: a query cell lacks recovered control metadata")
    query["sample_id"] = query["model_key_id"]
    query["control"] = query["source_cell"].map(
        {
            cell: f"exp33::{tissue}::{record['Cell']}::control"
            for cell, record in cell_records.items()
        }
    )
    query["is_control"] = False
    query["source_row_role"] = "exp33_unique_double_drug_query"
    query["source_task"] = TISSUE_SPECS[tissue]["task_name"]
    query["task_context"] = EXPERIMENT_NAME
    query["feature_membership"] = "primary"
    query["training_label_scope"] = "inference_only_unlabeled"
    query["pert_index1"] = query["pert_id1"].map(pert_index).astype(np.int64)
    query["pert_index2"] = query["pert_id2"].map(pert_index).astype(np.int64)
    query["pert_time_norm"] = "24"
    query["pert_time_index"] = time_index
    query["pert_dose1"] = query["pert_dose1_clipped_representative_uM"]
    query["pert_dose2"] = query["pert_dose2_clipped_representative_uM"]
    query["pert_dose1_norm"] = query["pert_dose1"].map(compact_number)
    query["pert_dose2_norm"] = query["pert_dose2"].map(compact_number)
    query["drugname1"] = query["pert_id1"].map(drug_name_by_id)
    query["drugname2"] = query["pert_id2"].map(drug_name_by_id)
    query["drugname"] = query["drugname1"] + " + " + query["drugname2"]
    query["smiles1"] = query["pert_id1"].map(smiles_map)
    query["smiles2"] = query["pert_id2"].map(smiles_map)
    query["smiles"] = query["smiles1"] + "." + query["smiles2"]
    target_cache: dict[tuple[str, str], str] = {}
    query["target_protein_list"] = [
        target_cache.setdefault(
            (pert_id1, pert_id2),
            combine_targets(pert_id1, pert_id2, target_map),
        )
        for pert_id1, pert_id2 in zip(query["pert_id1"], query["pert_id2"], strict=True)
    ]
    query["PRISM1st_label_total"] = ""
    query["PRISM2nd_label_total"] = ""
    query["synergy"] = ""
    query["unified_label_mask"] = 1.0
    query["expression_row_index"] = len(source_cells)
    query = query.drop(
        columns=[
            "pert_dose1_clipped_representative_uM",
            "pert_dose2_clipped_representative_uM",
        ]
    )

    table = pd.concat([pd.DataFrame(controls), query], ignore_index=True, sort=False)
    for column in ("pert_time", "pert_dose1", "pert_dose2"):
        table[column] = table[column].map(clean_text)
    table["feature_row_index"] = np.arange(len(table), dtype=np.int64)
    table["processed_row_index"] = table["feature_row_index"]
    if table["sample_id"].duplicated().any():
        raise ValueError(f"{tissue}: feature-table sample IDs are not unique")

    set_info: dict[int, dict[str, list[int]]] = {}
    row_to_set: dict[int, int] = {}
    for set_idx, source_cell in enumerate(source_cells):
        control_idx = set_idx
        perturb_indices = table.index[
            (~table["is_control"].astype(bool)) & table["source_cell"].eq(source_cell)
        ].astype(int).tolist()
        set_info[set_idx] = {"control": [control_idx], "perturb": perturb_indices}
        row_to_set[control_idx] = set_idx
        for row_idx in perturb_indices:
            row_to_set[row_idx] = set_idx
    if len(row_to_set) != len(table):
        raise ValueError(f"{tissue}: row_to_set does not cover the complete feature table")
    return table, set_info, row_to_set


def hardlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace existing path: {target}")
    os.link(source, target)


def write_split_files(
    *,
    split_dir: Path,
    table: pd.DataFrame,
    set_info: dict[int, dict[str, list[int]]],
    row_to_set: dict[int, int],
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=False)
    query_indices = table.index[~table["is_control"].astype(bool)].astype(int).tolist()
    empty_indices: list[int] = []
    empty_sets: dict[int, dict[str, list[int]]] = {}
    dump_pickle(split_dir / "row_to_set_index.pkl", row_to_set)
    dump_pickle(split_dir / "set_info.pkl", set_info)
    dump_pickle(
        split_dir / "set_to_grouping.pkl",
        {set_idx: f"exp33_{set_idx + 1}" for set_idx in set_info},
    )
    for split_name, indices, payload in (
        ("train", empty_indices, empty_sets),
        ("valid", empty_indices, empty_sets),
        ("test", query_indices, set_info),
    ):
        dump_pickle(split_dir / f"{split_name}_indices_{SPLIT_STRATEGY}.pkl", indices)
        dump_pickle(split_dir / f"{split_name}_set_info_{SPLIT_STRATEGY}.pkl", payload)
    dump_pickle(split_dir / f"val_indices_{SPLIT_STRATEGY}.pkl", empty_indices)
    dump_pickle(split_dir / f"val_set_info_{SPLIT_STRATEGY}.pkl", empty_sets)
    manifest = {
        "generated_at": iso_now(),
        "strategy": SPLIT_STRATEGY,
        "policy": "inference-only; every deduplicated double-drug model key is one test anchor",
        "anchor_counts": {"train": 0, "valid": 0, "test": len(query_indices)},
        "set_counts": {"train": 0, "valid": 0, "test": len(set_info)},
    }
    dump_json(split_dir / "split_manifest.json", manifest)
    return manifest


def write_task(
    *,
    build_group_root: Path,
    final_group_root: Path,
    tissue: str,
    table: pd.DataFrame,
    expression: np.ndarray,
    set_info: dict[int, dict[str, list[int]]],
    row_to_set: dict[int, int],
    ordered_indices: list[int],
    ordered_uniprot: list[str],
    expression_audit: dict[str, Any],
) -> dict[str, Any]:
    task_name = str(TISSUE_SPECS[tissue]["task_name"])
    task_dir = build_group_root / "tasks" / task_name
    final_task_dir = final_group_root / "tasks" / task_name
    split_dir = build_group_root / "splits" / task_name
    task_dir.mkdir(parents=True, exist_ok=False)

    feature_parquet = task_dir / "feature_table.parquet"
    feature_csv = task_dir / "feature_table.csv"
    table.to_parquet(feature_parquet, index=False)
    table.to_csv(feature_csv, index=False)
    hardlink(feature_csv, task_dir / "processed.csv")
    np.save(task_dir / "feature_expression_matrix.npy", expression.astype(np.float32, copy=False))
    hardlink(
        task_dir / "feature_expression_matrix.npy",
        task_dir / "processed_expression_matrix.npy",
    )
    dump_json(task_dir / "feature_ordered_protein_index.json", ordered_indices)
    hardlink(
        task_dir / "feature_ordered_protein_index.json",
        task_dir / "processed_ordered_protein_index.json",
    )
    dump_json(task_dir / "feature_ordered_protein_uniprot.json", ordered_uniprot)
    hardlink(
        task_dir / "feature_ordered_protein_uniprot.json",
        task_dir / "processed_ordered_protein_uniprot.json",
    )
    sample_ids = table["sample_id"].astype(str).tolist()
    dump_json(task_dir / "feature_sample_ids.json", sample_ids)
    hardlink(task_dir / "feature_sample_ids.json", task_dir / "processed_sample_ids.json")
    split_manifest = write_split_files(
        split_dir=split_dir,
        table=table,
        set_info=set_info,
        row_to_set=row_to_set,
    )
    loading_manifest = {
        "generated_at": iso_now(),
        "experiment_name": EXPERIMENT_NAME,
        "task_name": task_name,
        "row_key_column": "sample_id",
        "expression_row_index_column": "expression_row_index",
        "expression_matrix_path": str(
            (final_task_dir / "feature_expression_matrix.npy").resolve(strict=False)
        ),
        "ordered_protein_index_path": str(
            (final_task_dir / "feature_ordered_protein_index.json").resolve(strict=False)
        ),
        "ordered_protein_uniprot_path": str(
            (final_task_dir / "feature_ordered_protein_uniprot.json").resolve(strict=False)
        ),
        "sample_ids_path": str(
            (final_task_dir / "feature_sample_ids.json").resolve(strict=False)
        ),
        "feature_table_csv": str((final_task_dir / "feature_table.csv").resolve(strict=False)),
        "feature_table_native": str(
            (final_task_dir / "feature_table.parquet").resolve(strict=False)
        ),
        "loading_contract": (
            "Feature rows use validated expression_row_index indirection into a compact matrix "
            "containing one baseline per cell plus one all-NaN query sentinel."
        ),
        "expression_policy": expression_audit,
        "label_policy": "unlabeled inference; no AUROC/AUPRC",
    }
    dump_json(task_dir / "feature_loading_manifest.json", loading_manifest)
    return {
        "task_name": task_name,
        "tissue": tissue,
        "task_dir": str(final_task_dir.resolve(strict=False)),
        "split_dir": str(
            (final_group_root / "splits" / task_name).resolve(strict=False)
        ),
        "rows": int(len(table)),
        "control_rows": int(table["is_control"].astype(bool).sum()),
        "query_rows": int((~table["is_control"].astype(bool)).sum()),
        "set_count": int(len(set_info)),
        "matrix_shape": list(map(int, expression.shape)),
        "split_manifest": split_manifest,
        "expression_audit": expression_audit,
    }


def symlink_readonly_alias(source: Path, target: Path) -> dict[str, Any]:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace existing alias: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source.resolve()), str(target), target_is_directory=source.is_dir())
    return {
        "alias": str(target.absolute()),
        "source": str(source.resolve()),
        "kind": "read_only_symlink",
        "is_directory": source.is_dir(),
    }


def validate_artifacts(output_root: Path) -> dict[str, Any]:
    from dataset.training_ready_fast_dataset import (
        FastProteinTalkDataset,
        FastTrainingReadyArtifacts,
    )

    group_root = output_root / DATASET_GROUP
    meta_path = group_root / "global_meta.json"
    meta = load_json(meta_path)
    with (group_root / "derived/drug_embedding_morgan_2048.pkl").open("rb") as handle:
        drug_embedding = np.asarray(pickle.load(handle)["embedding_matrix"], dtype=np.float32)
    validation: dict[str, Any] = {"validated_at": iso_now(), "tasks": {}}
    for tissue, spec in TISSUE_SPECS.items():
        task_name = str(spec["task_name"])
        task_dir = group_root / "tasks" / task_name
        split_dir = group_root / "splits" / task_name
        artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
        if len(artifacts.df) != int(spec["unique_rows"]) + int(spec["cells"]):
            raise ValueError(f"{task_name}: unexpected feature row count")
        if tuple(artifacts.expression_matrix.shape) != (
            int(spec["expression_rows"]),
            EXPECTED_AXIS_SIZE,
        ):
            raise ValueError(f"{task_name}: unexpected compact expression shape")
        test_indices = [
            int(item)
            for item in load_pickle(split_dir / f"test_indices_{SPLIT_STRATEGY}.pkl")
        ]
        row_to_set = {
            int(key): int(value)
            for key, value in load_pickle(split_dir / "row_to_set_index.pkl").items()
        }
        set_info = load_pickle(split_dir / f"test_set_info_{SPLIT_STRATEGY}.pkl")
        dataset = FastProteinTalkDataset(
            artifacts=artifacts,
            indices=[test_indices[0], test_indices[-1]],
            row_to_set_index=row_to_set,
            set_info=set_info,
            mode="eval",
            drug_embedding_matrix=drug_embedding,
            batch_cov_list=[
                "machineID_new",
                "Cell_plate",
                "Cell",
                "cell_type",
                "batch",
                "pert_time",
                "pert_dose1",
                "pert_dose2",
            ],
            target_protein_max_length=32,
            effective_key1="PRISM1st_label_total",
            effective_key2="synergy",
        )
        for item in (dataset[0], dataset[1]):
            if not np.isfinite(item["control_expression"]).any():
                raise ValueError(f"{task_name}: sampled control expression has no finite value")
            if not np.isnan(item["perturb_expression"]).all():
                raise ValueError(f"{task_name}: sampled query expression is not the NaN sentinel")
        validation["tasks"][task_name] = {
            "feature_rows": len(artifacts.df),
            "expression_rows": int(artifacts.expression_matrix.shape[0]),
            "test_rows": len(test_indices),
            "expression_row_index_min": int(artifacts.expression_row_indices.min()),
            "expression_row_index_max": int(artifacts.expression_row_indices.max()),
            "sampled_control_and_sentinel_reads_valid": True,
        }
    validation["all_checks_passed"] = True
    return validation


def build(args: argparse.Namespace) -> None:
    source_root = Path(args.source_training_ready_root)
    output_root = Path(args.output_root)
    raw_root = Path(args.raw_root)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_manifest_path = checkpoint_path.parent / "run_manifest.json"
    source_group_root = source_root / DATASET_GROUP
    meta_path = source_group_root / "global_meta.json"
    derived_root = source_group_root / "derived"
    control_root = raw_root / "ptv2_cell_combo_virtualScreen260722_control_260724"

    required = [
        checkpoint_path,
        checkpoint_manifest_path,
        meta_path,
        STANDARDIZED_MAIN_SINGLE,
        GRAPH_FEATURE_PATH,
        derived_root / "drug_embedding_morgan_2048.pkl",
        derived_root / "pdi_matrix.npy",
        derived_root / "ddi_matrix.npy",
        derived_root / "cell_llm_embedding_qwen3_4096.npz",
        derived_root / "cell_type_llm_embedding_qwen3_4096_v2.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing exp33 inputs: {missing}")
    if checkpoint_path.name != "epoch=2.ckpt":
        raise ValueError(f"exp33 requires epoch=2.ckpt, got {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {checkpoint_hash} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing exp33 root: {output_root}")
    staging_root = output_root.with_name(f".{output_root.name}.building-{os.getpid()}")
    if staging_root.exists():
        raise FileExistsError(f"staging path already exists: {staging_root}")

    query_paths: dict[str, Path] = {}
    raw_control_paths: dict[str, Path] = {}
    unique_control_paths: dict[str, Path] = {}
    for tissue, spec in TISSUE_SPECS.items():
        query_paths[tissue] = find_exactly_one(raw_root, str(spec["query_glob"]))
        raw_control_paths[tissue] = find_exactly_one(control_root, str(spec["control_glob"]))
        unique_control_paths[tissue] = find_exactly_one(
            control_root, str(spec["control_unique_glob"])
        )
        if raw_control_paths[tissue].name.endswith("_unique.csv"):
            raise ValueError(f"{tissue}: raw control path accidentally selected unique file")
    raw_paths = {
        **{f"{tissue}_query": path for tissue, path in query_paths.items()},
        **{f"{tissue}_control": path for tissue, path in raw_control_paths.items()},
        **{f"{tissue}_control_unique": path for tissue, path in unique_control_paths.items()},
    }
    raw_hashes_before = {key: sha256_file(path) for key, path in raw_paths.items()}

    try:
        build_group_root = staging_root / DATASET_GROUP
        final_group_root = output_root / DATASET_GROUP
        build_group_root.mkdir(parents=True, exist_ok=False)
        meta = load_json(meta_path)
        checkpoint_manifest = load_json(checkpoint_manifest_path)
        if checkpoint_manifest.get("run_status") != "fit_completed":
            raise ValueError("checkpoint manifest is not fit_completed")
        if checkpoint_manifest.get("task_head") != "unified":
            raise ValueError("checkpoint task_head must be unified")
        ordered_indices, ordered_uniprot, axis_index_path, axis_uniprot_path = checkpoint_axis(
            checkpoint_manifest
        )
        protein_index = {str(key): int(value) for key, value in meta["protein_index"].items()}
        for index, uniprot in zip(ordered_indices, ordered_uniprot, strict=True):
            if protein_index.get(uniprot) != index:
                raise ValueError(f"checkpoint/global protein index mismatch for {uniprot}")
        training_support = checkpoint_training_support(checkpoint_manifest)

        query_records = collect_query_drug_records(query_paths)
        drug_audit, canonical_to_selected, drug_feature_audit = build_drug_mapping(
            query_records=query_records,
            meta=meta,
            derived_root=derived_root,
            graph_feature_path=GRAPH_FEATURE_PATH,
        )
        raw_smiles_to_canonical = dict(
            zip(drug_audit["query_smiles"], drug_audit["canonical_isomeric_smiles"], strict=True)
        )
        drug_name_by_id = dict(
            zip(drug_audit["selected_pert_id"], drug_audit["query_drug_name"], strict=True)
        )
        cells, recovered_metadata = load_control_metadata(
            raw_root=raw_root,
            raw_control_paths=raw_control_paths,
        )

        with np.load(derived_root / "cell_llm_embedding_qwen3_4096.npz") as payload:
            cell_llm_rows = int(payload["embedding_matrix"].shape[0])
            cell_llm_names = [str(item) for item in payload["cell_names"].tolist()]
        with np.load(derived_root / "cell_type_llm_embedding_qwen3_4096_v2.npz") as payload:
            cell_type_llm_rows = int(payload["embedding_matrix"].shape[0])
            cell_type_llm_names = [str(item) for item in payload["cell_type_names"].tolist()]

        cell_audit_rows: list[dict[str, Any]] = []
        seen_cell_indices = set(map(int, training_support["Cell_index"]))
        for record in cells.itertuples(index=False):
            true_cell_index = value_index(meta, "Cell", record.Cell)
            model_cell_index = true_cell_index if true_cell_index in seen_cell_indices else 0
            machine_index = value_index(meta, "machineID_new", record.machineID_new)
            cell_type_index = value_index(meta, "cell_type", record.cell_type)
            if true_cell_index >= cell_llm_rows or cell_llm_names[true_cell_index] != record.Cell:
                raise ValueError(f"Cell LLM row mismatch for {record.Cell}")
            if (
                cell_type_index >= cell_type_llm_rows
                or cell_type_llm_names[cell_type_index] != record.cell_type
            ):
                raise ValueError(f"cell-type LLM row mismatch for {record.cell_type}")
            cell_audit_rows.append(
                {
                    **record._asdict(),
                    "true_Cell_index": true_cell_index,
                    "model_Cell_index": model_cell_index,
                    "cell_llm_index": true_cell_index,
                    "cell_seen_in_checkpoint_train": true_cell_index in seen_cell_indices,
                    "machineID_new_index": machine_index,
                    "machine_seen_in_checkpoint_train": machine_index
                    in set(map(int, training_support["machineID_new_index"])),
                    "Cell_plate": "no",
                    "Cell_plate_index": value_index(meta, "Cell_plate", "no"),
                    "Cell_plate_seen_in_checkpoint_train": False,
                    "cell_type_index": cell_type_index,
                    "cell_type_llm_index": cell_type_index,
                    "cell_type_seen_in_checkpoint_train": cell_type_index
                    in set(map(int, training_support["cell_type_index"])),
                    "batch": "no",
                    "batch_index": value_index(meta, "batch", "no"),
                    "batch_seen_in_checkpoint_train": False,
                    "pert_time": 24,
                    "pert_time_index": value_index(meta, "pert_time", "24"),
                    "pert_time_seen_in_checkpoint_train": value_index(meta, "pert_time", "24")
                    in set(map(int, training_support["pert_time_index"])),
                }
            )
        covariate_audit = pd.DataFrame(cell_audit_rows)
        seen_names = set(covariate_audit.loc[
            covariate_audit["cell_seen_in_checkpoint_train"], "Cell"
        ])
        unseen_names = set(covariate_audit.loc[
            ~covariate_audit["cell_seen_in_checkpoint_train"], "Cell"
        ])
        if len(seen_names) != 18 or unseen_names != UNSEEN_CELLS:
            raise ValueError(
                f"unexpected Cell seen/unseen partition: seen={sorted(seen_names)} "
                f"unseen={sorted(unseen_names)}"
            )
        if not covariate_audit["machine_seen_in_checkpoint_train"].all():
            raise ValueError("a selected machine is unseen in checkpoint training")
        if not covariate_audit["cell_type_seen_in_checkpoint_train"].all():
            raise ValueError("a selected cell type is unseen in checkpoint training")
        if not covariate_audit["pert_time_seen_in_checkpoint_train"].all():
            raise ValueError("24h is unseen in checkpoint training")

        aliases = {
            "global_meta": symlink_readonly_alias(
                meta_path, build_group_root / "global_meta.json"
            ),
            "derived": symlink_readonly_alias(
                derived_root, build_group_root / "derived"
            ),
        }
        aliases["global_meta"]["alias"] = str(
            (final_group_root / "global_meta.json").resolve(strict=False)
        )
        aliases["derived"]["alias"] = str(
            (final_group_root / "derived").resolve(strict=False)
        )
        drug_audit_path = build_group_root / "exp33_vc_doubledrug_drug_id_mapping_audit.csv"
        covariate_audit_path = build_group_root / "exp33_vc_doubledrug_covariate_audit.csv"
        drug_audit.to_csv(drug_audit_path, index=False)
        covariate_audit.to_csv(covariate_audit_path, index=False)

        task_summaries: dict[str, Any] = {}
        query_audits: dict[str, Any] = {}
        all_dose1_buckets: set[int] = set()
        all_dose2_buckets: set[int] = set()
        for tissue, spec in TISSUE_SPECS.items():
            unique_keys, query_audit = build_unique_keys(
                tissue=tissue,
                query_path=query_paths[tissue],
                canonical_to_selected=canonical_to_selected,
                raw_smiles_to_canonical=raw_smiles_to_canonical,
                expected_rows=int(spec["raw_rows"]),
                expected_unique_rows=int(spec["unique_rows"]),
            )
            source_cells = list(query_audit["source_cells"])
            expected_source_cells = cells.loc[
                cells["tissue"].eq(tissue), "source_cell"
            ].tolist()
            if set(source_cells) != set(expected_source_cells):
                raise ValueError(f"{tissue}: query/baseline cell sets differ")
            if len(source_cells) != int(spec["cells"]):
                raise ValueError(f"{tissue}: unexpected query cell count")
            expression, expression_audit = load_baseline_matrix(
                tissue=tissue,
                unique_control_path=unique_control_paths[tissue],
                source_cells=source_cells,
                axis_uniprot=ordered_uniprot,
            )
            table, set_info, row_to_set = make_task_table(
                tissue=tissue,
                unique_keys=unique_keys,
                source_cells=source_cells,
                cell_metadata=recovered_metadata[tissue],
                meta=meta,
                training_support=training_support,
                drug_name_by_id=drug_name_by_id,
            )
            task_summary = write_task(
                build_group_root=build_group_root,
                final_group_root=final_group_root,
                tissue=tissue,
                table=table,
                expression=expression,
                set_info=set_info,
                row_to_set=row_to_set,
                ordered_indices=ordered_indices,
                ordered_uniprot=ordered_uniprot,
                expression_audit=expression_audit,
            )
            task_summaries[str(spec["task_name"])] = task_summary
            query_audits[tissue] = query_audit
            all_dose1_buckets.update(map(int, query_audit["dose1_buckets"]))
            all_dose2_buckets.update(map(int, query_audit["dose2_buckets"]))
            del unique_keys, table, expression, set_info, row_to_set

        if not all_dose1_buckets.issubset(
            set(map(int, training_support["pert_dose1_index"]))
        ):
            raise ValueError("an exp33 A-dose bucket is unseen in checkpoint slot 1")
        if not all_dose2_buckets.issubset(
            set(map(int, training_support["pert_dose2_index"]))
        ):
            raise ValueError("an exp33 B-dose bucket is unseen in checkpoint slot 2")
        if all_dose2_buckets != set(range(1, 11)):
            raise ValueError(f"unexpected B-dose buckets: {sorted(all_dose2_buckets)}")

        raw_hashes_after = {key: sha256_file(path) for key, path in raw_paths.items()}
        if raw_hashes_after != raw_hashes_before:
            raise RuntimeError("one or more raw exp33 inputs changed during construction")
        raw_input_audit = {
            key: {
                "path": str(raw_paths[key].resolve()),
                "sha256_before": raw_hashes_before[key],
                "sha256_after": raw_hashes_after[key],
                "unchanged": True,
            }
            for key in raw_paths
        }
        summary = {
            "generated_at": iso_now(),
            "experiment_name": EXPERIMENT_NAME,
            "dataset_group": DATASET_GROUP,
            "source_training_ready_root": str(source_root.resolve()),
            "output_root": str(output_root.resolve()),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            "source_global_meta_path": str(meta_path.resolve()),
            "source_global_meta_sha256": sha256_file(meta_path),
            "protein_axis": {
                "count": len(ordered_indices),
                "ordered_index_path": str(axis_index_path.resolve()),
                "ordered_index_sha256": sha256_file(axis_index_path),
                "ordered_uniprot_path": str(axis_uniprot_path.resolve()),
                "ordered_uniprot_sha256": sha256_file(axis_uniprot_path),
            },
            "raw_inputs": raw_input_audit,
            "readonly_reused_artifacts": aliases,
            "drug_mapping": {
                **drug_feature_audit,
                "audit_path": str(
                    (
                        final_group_root
                        / "exp33_vc_doubledrug_drug_id_mapping_audit.csv"
                    ).resolve(strict=False)
                ),
                "audit_sha256": sha256_file(drug_audit_path),
                "selection_rule": (
                    "canonical isomeric SMILES only; numeric ID > L9200 > external; "
                    "fail on same-tier ambiguity except confirmed (-)-Menthol=L9200_2195"
                ),
            },
            "covariates": {
                "audit_path": str(
                    (
                        final_group_root / "exp33_vc_doubledrug_covariate_audit.csv"
                    ).resolve(strict=False)
                ),
                "audit_sha256": sha256_file(covariate_audit_path),
                "cells": len(covariate_audit),
                "seen_cells": sorted(seen_names),
                "unseen_cells": sorted(unseen_names),
                "Cell_plate_policy": "no; OOD because index 0 is absent from checkpoint train rows",
                "batch_policy": "no; OOD because index 0 is absent from checkpoint train rows",
                "cell_llm_policy": (
                    "seen Cell uses true Cell_index and true cell_llm_index; unseen Cell uses "
                    "Cell_index=0 but retains true nonzero cell_llm_index"
                ),
                "cell_type_policy": "true seen cell_type_index and cell_type_llm_index",
                "machine_policy": "per-cell recovered QE or 480_FAIMS; both train-seen",
                "time_policy": "24h; train-seen",
                "dose1_buckets": sorted(all_dose1_buckets),
                "dose2_buckets": sorted(all_dose2_buckets),
                "all_dose_buckets_train_seen": True,
            },
            "checkpoint_training_support": training_support,
            "query_audits": query_audits,
            "tasks": task_summaries,
            "expected_counts": {
                "query_drugs": EXPECTED_DRUG_COUNT,
                "cells": EXPECTED_CELL_COUNT,
                "raw_rows": EXPECTED_RAW_ROWS,
                "unique_model_keys": EXPECTED_UNIQUE_ROWS,
            },
            "fixed_settings": {
                "inference_only": True,
                "pert_time": 24,
                "slot1": "drug A / drug_A_concentration_uM",
                "slot2": "drug B / assumed_combo_IC50_B_uM",
                "dose_rule": "clip to [0,10], then ceil to categorical index",
                "baseline": "same-tissue *_control_unique.csv mixed 6h/24h arithmetic mean",
                "score": "pred_unified_combo_prob sourced only from pred_task_prob",
                "rank_outputs": False,
            },
            "acceptance_checks": {
                "checkpoint_sha256_exact": True,
                "raw_hashes_unchanged": True,
                "query_drugs_227": len(drug_audit) == EXPECTED_DRUG_COUNT,
                "numeric_ids_51": int((drug_audit["selected_priority_tier"] == "numeric").sum())
                == 51,
                "L9200_ids_176": int((drug_audit["selected_priority_tier"] == "L9200").sum())
                == 176,
                "external_ids_0": int(
                    drug_audit["selected_priority_tier"].eq("external").sum()
                )
                == 0,
                "baseline_cells_28": len(covariate_audit) == EXPECTED_CELL_COUNT,
                "seen_cells_18": len(seen_names) == 18,
                "unseen_cells_10": unseen_names == UNSEEN_CELLS,
                "unique_model_keys_1124928": sum(
                    int(item["query_rows"]) for item in task_summaries.values()
                )
                == EXPECTED_UNIQUE_ROWS,
                "compact_expression_rows_7_16_8": [
                    task_summaries[str(TISSUE_SPECS[tissue]["task_name"])][
                        "matrix_shape"
                    ][0]
                    for tissue in TISSUE_SPECS
                ]
                == [7, 16, 8],
            },
            "notes": [
                "No training or fine-tuning is performed.",
                "Query perturb_expression rows all reference one all-NaN sentinel per tissue.",
                "Cell_plate=no and batch=no are intentionally explicit OOD categories, not inferred plate/batch values.",
                "Labels are absent; AUROC/AUPRC must not be reported.",
            ],
        }
        summary_path = build_group_root / "exp33_vc_doubledrug_build_summary.json"
        dump_json(summary_path, summary)
        os.replace(staging_root, output_root)
        validation = validate_artifacts(output_root)
        validation_path = (
            output_root / DATASET_GROUP / "exp33_vc_doubledrug_preflight.json"
        )
        dump_json(validation_path, validation)
        print(f"[exp33] built {output_root}")
        for task_name, task_summary in task_summaries.items():
            print(
                f"[exp33] {task_name}: rows={task_summary['rows']} "
                f"queries={task_summary['query_rows']} matrix={task_summary['matrix_shape']}"
            )
        print(f"[exp33] preflight: {validation_path}")
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-training-ready-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate an already-built exp33 root without rebuilding it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        output_root = Path(args.output_root)
        validation = validate_artifacts(output_root)
        path = output_root / DATASET_GROUP / "exp33_vc_doubledrug_preflight.json"
        dump_json(path, validation)
        print(f"[exp33] preflight passed: {path}")
        return
    build(args)


if __name__ == "__main__":
    main()
