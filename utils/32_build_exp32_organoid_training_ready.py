#!/usr/bin/env python3
"""Build exp32 organoid single-drug inference-only training-ready tasks.

The builder is deliberately copy-on-write.  It never changes the maintained
``data/training_ready/ptv3`` artifacts: the new root contains task/split files
and read-only symlinks to the original global metadata and derived features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GROUP = "ptv3"
EXPERIMENT_NAME = "exp32_organoid_exp09_single_sensitivity"
SPLIT_STRATEGY = "test_only"
EXPECTED_SAMPLE_COUNT = 13
EXPECTED_DRUG_COUNT = 3217
EXPECTED_AXIS_SIZE = 11092
EXPECTED_QUERY_ROWS = EXPECTED_SAMPLE_COUNT * EXPECTED_DRUG_COUNT
EXPECTED_TASK_ROWS = EXPECTED_SAMPLE_COUNT + EXPECTED_QUERY_ROWS

DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/training_ready"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/training_ready_exp32_organoid"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=5.ckpt"
)
DEFAULT_B_INPUT = REPO_ROOT / "data/rawdata/rna_seq/260709_B260625QC_merged_replicates_add_CellType.csv"
DEFAULT_CAC_INPUT = REPO_ROOT / "data/rawdata/rna_seq/260709_CAC260627QC_merged_replicates_add_CellType.csv"
DEFAULT_SAMPLE_INFO = REPO_ROOT / "data/rawdata/rna_seq/260709_samp_inf.csv"

SOURCE_TASKS = [
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

DEVICE_CONFIGS = [
    {
        "device": "B",
        "device_slug": "qe",
        "task_name": "ptv3_exp32_organoid_qe_single",
        "input_path": DEFAULT_B_INPUT,
        "sample_prefix": "B260625",
        "instrument_raw": "QE_HF",
        "machineID_new": "QE",
        "batch": "exp32_organoid_qe",
    },
    {
        "device": "CAC",
        "device_slug": "480_faims",
        "task_name": "ptv3_exp32_organoid_480_faims_single",
        "input_path": DEFAULT_CAC_INPUT,
        "sample_prefix": "CAC260627",
        "instrument_raw": "480_FAIMS",
        "machineID_new": "480_FAIMS",
        "batch": "exp32_organoid_480_faims",
    },
]

TISSUE_TO_CELL_TYPE = {
    "LUNG": "LUNG",
    "PANCREAS": "PANCREAS",
    "COLON": "COLON",
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


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_np_load(path: str | Path, *, mmap_mode: str | None = None) -> np.ndarray:
    path = Path(path)
    if mmap_mode is None:
        return np.load(path)
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except OSError as exc:
        print(f"[exp32] mmap unavailable for {path}: {exc}; loading normally")
        return np.load(path)


def safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


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


def read_drug_columns(path: Path) -> pd.DataFrame:
    columns = ["sample_id", "control", "is_control", "pert_id1", "pert_id2"]
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)[columns]
    return pd.read_csv(path, usecols=lambda name: name in set(columns), low_memory=False)


def control_mask(table: pd.DataFrame) -> pd.Series:
    if "is_control" in table.columns:
        values = table["is_control"]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(bool)
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return table["sample_id"].astype(str).eq(table["control"].astype(str))


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def resolve_graph_feature_path(checkpoint_manifest: dict[str, Any]) -> Path:
    graph_meta = checkpoint_manifest.get("graph_feature_meta") or {}
    candidate = graph_meta.get("feature_path") if isinstance(graph_meta, dict) else None
    if candidate:
        path = Path(str(candidate))
        if path.exists():
            return path
    fallback = REPO_ROOT / "graph_cache/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy"
    if not fallback.exists():
        raise FileNotFoundError("checkpoint graph feature cache is unavailable")
    return fallback


def build_drug_scope(
    *,
    source_group_root: Path,
    meta: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pert_index = {str(key): int(value) for key, value in meta["pert_index"].items()}
    source_tasks: dict[str, set[str]] = defaultdict(set)
    slot_counts: dict[tuple[str, str], int] = defaultdict(int)
    source_summaries: list[dict[str, Any]] = []

    for task_name in SOURCE_TASKS:
        table_path = feature_table_path(source_group_root / "tasks" / task_name)
        table = read_drug_columns(table_path)
        query = table.loc[~control_mask(table)]
        task_drugs: set[str] = set()
        for slot in ("pert_id1", "pert_id2"):
            values = query[slot].map(safe_text)
            for pert_id, count in values.value_counts().items():
                if not pert_id or pert_id == "no":
                    continue
                task_drugs.add(pert_id)
                source_tasks[pert_id].add(task_name)
                slot_counts[(pert_id, task_name)] += int(count)
        source_summaries.append(
            {
                "task_name": task_name,
                "feature_table_path": str(table_path.resolve()),
                "feature_table_sha256": sha256_file(table_path),
                "rows": int(len(table)),
                "query_rows": int(len(query)),
                "unique_drugs": int(len(task_drugs)),
            }
        )

    drug_ids = sorted(source_tasks, key=lambda pert_id: pert_index.get(pert_id, 10**12))
    if len(drug_ids) != EXPECTED_DRUG_COUNT:
        raise ValueError(f"expected {EXPECTED_DRUG_COUNT} source drugs, found {len(drug_ids)}")
    missing_index = sorted(set(drug_ids) - set(pert_index))
    if missing_index:
        raise ValueError(f"drug scope contains ids absent from pert_index: {missing_index[:20]}")

    derived_root = source_group_root / "derived"
    embedding_path = derived_root / "drug_embedding_morgan_2048.pkl"
    embedding_payload = load_pickle(embedding_path)
    embedding = np.asarray(embedding_payload["embedding_matrix"], dtype=np.float32)
    payload_index = {str(key): int(value) for key, value in embedding_payload.get("item_to_index", {}).items()}
    if payload_index and payload_index != pert_index:
        raise ValueError("Morgan item_to_index differs from global pert_index")
    selected_indices = np.asarray([pert_index[item] for item in drug_ids], dtype=np.int64)
    selected_embedding = embedding[selected_indices]
    if not np.isfinite(selected_embedding).all():
        raise ValueError("selected Morgan rows contain non-finite values")
    nonzero_embedding = np.any(np.abs(selected_embedding) > 0, axis=1)
    if not nonzero_embedding.all():
        raise ValueError("selected drug scope contains all-zero Morgan rows")

    graph_feature_path = resolve_graph_feature_path(checkpoint_manifest)
    graph_features = safe_np_load(graph_feature_path, mmap_mode="r")
    selected_graph = np.asarray(graph_features[selected_indices], dtype=np.float32)
    if not np.isfinite(selected_graph).all():
        raise ValueError("selected graph-feature rows contain non-finite values")
    nonzero_graph = np.any(np.abs(selected_graph) > 0, axis=1)
    if not nonzero_graph.all():
        raise ValueError("selected drug scope contains all-zero graph-feature rows")

    smiles_map = meta.get("pertid_to_smiles", {})
    target_map = meta.get("pertid_to_target_protein_list", {})
    scope_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    missing_smiles: list[str] = []
    for rank, pert_id in enumerate(drug_ids):
        smiles = safe_text(smiles_map.get(pert_id))
        if not smiles:
            missing_smiles.append(pert_id)
        targets: list[int] = []
        for value in target_map.get(pert_id, []):
            try:
                target = int(value)
            except (TypeError, ValueError):
                continue
            if target not in targets:
                targets.append(target)
        tasks = sorted(source_tasks[pert_id], key=SOURCE_TASKS.index)
        scope_rows.append(
            {
                "scope_rank": int(rank),
                "pert_id": pert_id,
                "pert_index": int(pert_index[pert_id]),
                "smiles": smiles,
                "target_protein_list": json.dumps(targets, separators=(",", ":")),
                "target_protein_count": int(len(targets)),
                "source_tasks": json.dumps(tasks, ensure_ascii=False),
                "source_task_count": int(len(tasks)),
                "morgan_row_finite": True,
                "morgan_row_nonzero": True,
                "graph_row_finite": True,
                "graph_row_nonzero": True,
            }
        )
        for task_name in tasks:
            source_rows.append(
                {
                    "pert_id": pert_id,
                    "pert_index": int(pert_index[pert_id]),
                    "task_name": task_name,
                    "slot_occurrences": int(slot_counts[(pert_id, task_name)]),
                }
            )
    if missing_smiles:
        raise ValueError(f"drug scope contains missing SMILES: {missing_smiles[:20]}")

    scope_df = pd.DataFrame(scope_rows)
    source_df = pd.DataFrame(source_rows)
    audit = {
        "generated_at": iso_now(),
        "policy": "union of pert_id1 and pert_id2 from non-control rows in explicit exp01-exp09 main+extra tasks",
        "ordering": "ascending global pert_index",
        "expected_drug_count": EXPECTED_DRUG_COUNT,
        "drug_count": int(len(scope_df)),
        "drug_ids": drug_ids,
        "source_tasks": source_summaries,
        "source_task_count": int(len(SOURCE_TASKS)),
        "pert_index_count": int(len(pert_index)),
        "morgan_embedding_path": str(embedding_path.resolve()),
        "morgan_embedding_sha256": sha256_file(embedding_path),
        "morgan_embedding_shape": [int(value) for value in embedding.shape],
        "graph_feature_path": str(graph_feature_path.resolve()),
        "graph_feature_sha256": sha256_file(graph_feature_path),
        "graph_feature_shape": [int(value) for value in graph_features.shape],
        "all_drugs_in_pert_index": True,
        "all_morgan_rows_finite_nonzero": True,
        "all_graph_rows_finite_nonzero": True,
        "all_smiles_nonempty": True,
        "drugs_with_nonempty_targets": int((scope_df["target_protein_count"] > 0).sum()),
    }
    return scope_df, source_df, audit


def checkpoint_axis(checkpoint_manifest: dict[str, Any]) -> tuple[list[int], list[str], Path, Path]:
    index_path = Path(str(checkpoint_manifest["ordered_protein_index_path"]))
    task_dir = Path(str(checkpoint_manifest["task_dir"]))
    uniprot_path = task_dir / "feature_ordered_protein_uniprot.json"
    indices = [int(item) for item in load_json(index_path)]
    uniprot = [str(item) for item in load_json(uniprot_path)]
    if len(indices) != EXPECTED_AXIS_SIZE or len(uniprot) != EXPECTED_AXIS_SIZE:
        raise ValueError(
            f"exp09 axis must have {EXPECTED_AXIS_SIZE} entries; got {len(indices)} and {len(uniprot)}"
        )
    if len(set(indices)) != len(indices) or len(set(uniprot)) != len(uniprot):
        raise ValueError("exp09 protein axis is not unique")
    return indices, uniprot, index_path, uniprot_path


def load_sample_info(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    info = pd.read_csv(path, low_memory=False)
    required = {"samp_ID", "pat_ID", "cell_type", "cell_type.1"}
    missing = sorted(required - set(info.columns))
    if missing:
        raise ValueError(f"sample-info file is missing columns: {missing}")
    samp_ids = pd.to_numeric(info["samp_ID"], errors="raise").astype(int)
    if sorted(samp_ids.tolist()) != list(range(1, EXPECTED_SAMPLE_COUNT + 1)):
        raise ValueError("sample-info samp_ID must be exactly 1..13")
    if samp_ids.duplicated().any() or info["pat_ID"].map(safe_text).eq("").any():
        raise ValueError("sample-info has duplicate samp_ID or missing pat_ID")
    mapping: dict[int, dict[str, Any]] = {}
    for row_idx, row in info.iterrows():
        samp_id = int(samp_ids.iloc[row_idx])
        mapping[samp_id] = {
            "samp_ID": samp_id,
            "pat_ID": safe_text(row["pat_ID"]),
            "cell_type_zh": safe_text(row["cell_type"]),
            "tissue": safe_text(row["cell_type.1"]),
            "sample_info_audit_value": safe_text(row.get("Unnamed: 4")),
        }
    audit = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(len(info)),
        "columns": list(map(str, info.columns)),
        "unknown_final_column_policy": "audit only; excluded from model features",
    }
    return mapping, audit


def sample_suffix(sample_id: str, expected_prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(expected_prefix)}_PTV2_(\d+)", sample_id)
    if match is None:
        raise ValueError(f"unexpected sample id for prefix {expected_prefix}: {sample_id}")
    suffix = int(match.group(1))
    if suffix < 1 or suffix > EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"sample suffix outside 1..13: {sample_id}")
    return suffix


def load_device_expression(
    *,
    config: dict[str, Any],
    raw_path: Path,
    axis_uniprot: list[str],
    sample_info: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    raw = pd.read_csv(raw_path, low_memory=False)
    if list(raw.columns[:2]) != ["prot_gene", "cell_type"]:
        raise ValueError(f"{raw_path} must start with prot_gene, cell_type")
    if len(raw) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"{raw_path} must contain exactly 13 samples")
    sample_ids = raw["prot_gene"].map(safe_text)
    if sample_ids.duplicated().any() or sample_ids.eq("").any():
        raise ValueError(f"{raw_path} contains duplicate or empty sample ids")

    suffixes = [sample_suffix(item, str(config["sample_prefix"])) for item in sample_ids]
    if sorted(suffixes) != list(range(1, EXPECTED_SAMPLE_COUNT + 1)):
        raise ValueError(f"{raw_path} sample suffixes must be exactly 1..13")

    protein_headers = list(map(str, raw.columns[2:]))
    protein_ids = [item.split("_", 1)[0].strip() for item in protein_headers]
    if any(not item for item in protein_ids) or len(set(protein_ids)) != len(protein_ids):
        raise ValueError(f"{raw_path} has empty or duplicate parsed UniProt ids")
    protein_to_col = {protein: idx for idx, protein in enumerate(protein_ids)}
    raw_values = raw.iloc[:, 2:].to_numpy(dtype=np.float32, copy=True)
    if np.isinf(raw_values).any():
        raise ValueError(f"{raw_path} contains infinite expression values")
    finite = np.isfinite(raw_values)
    if np.any(raw_values[finite] < 0):
        raise ValueError(f"{raw_path} contains negative finite expression values")

    aligned = np.full((EXPECTED_SAMPLE_COUNT, len(axis_uniprot)), np.nan, dtype=np.float32)
    overlap = 0
    for axis_col, uniprot in enumerate(axis_uniprot):
        raw_col = protein_to_col.get(uniprot)
        if raw_col is None:
            continue
        values = raw_values[:, raw_col]
        finite_values = np.isfinite(values)
        transformed = np.full(values.shape, np.nan, dtype=np.float32)
        transformed[finite_values] = np.log1p(values[finite_values]).astype(np.float32, copy=False)
        aligned[:, axis_col] = transformed
        overlap += 1

    sample_records: list[dict[str, Any]] = []
    order = np.argsort(np.asarray(suffixes))
    aligned = aligned[order]
    for out_row, raw_row_idx in enumerate(order.tolist()):
        raw_sample_id = sample_ids.iloc[raw_row_idx]
        suffix = int(suffixes[raw_row_idx])
        tissue = safe_text(raw.iloc[raw_row_idx]["cell_type"])
        tissue_key = tissue.upper()
        if tissue_key not in TISSUE_TO_CELL_TYPE:
            raise ValueError(f"unsupported tissue {tissue!r} for {raw_sample_id}")
        metadata = sample_info[suffix]
        if tissue.casefold() != str(metadata["tissue"]).casefold():
            raise ValueError(
                f"tissue mismatch for {raw_sample_id}: matrix={tissue!r}, sample_info={metadata['tissue']!r}"
            )
        sample_records.append(
            {
                **metadata,
                "organoid_sample_id": raw_sample_id,
                "sample_pair_id": f"PTV2_{suffix}",
                "tissue": tissue,
                "cell_type": TISSUE_TO_CELL_TYPE[tissue_key],
                "expression_row": int(out_row),
                "finite_axis_proteins": int(np.isfinite(aligned[out_row]).sum()),
                "missing_axis_proteins": int(np.isnan(aligned[out_row]).sum()),
            }
        )

    audit = {
        "input_path": str(raw_path.resolve()),
        "input_sha256": sha256_file(raw_path),
        "raw_shape": [int(value) for value in raw.shape],
        "sample_count": int(len(sample_records)),
        "input_protein_count": int(len(protein_ids)),
        "axis_size": int(len(axis_uniprot)),
        "axis_overlap": int(overlap),
        "axis_missing": int(len(axis_uniprot) - overlap),
        "finite_values_after_transform": int(np.isfinite(aligned).sum()),
        "nan_values_after_transform": int(np.isnan(aligned).sum()),
        "expression_transform": "finite non-negative values log1p; missing proteins/values remain NaN",
        "sample_ids": [record["organoid_sample_id"] for record in sample_records],
        "sample_pair_ids": [record["sample_pair_id"] for record in sample_records],
        "tissue_counts": pd.Series([record["tissue"] for record in sample_records]).value_counts().to_dict(),
    }
    return sample_records, aligned, audit


def write_npy_streaming(
    path: Path,
    *,
    shape: tuple[int, int],
    control_vectors: dict[int, np.ndarray],
    chunk_rows: int = 64,
) -> dict[str, Any]:
    """Write a C-order float32 .npy without mmap or a full in-memory matrix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(np.float32)
    with path.open("wb") as handle:
        header = {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        }
        np.lib.format.write_array_header_2_0(handle, header)
        header_bytes = int(handle.tell())
        for start in range(0, shape[0], chunk_rows):
            end = min(shape[0], start + chunk_rows)
            block = np.full((end - start, shape[1]), np.nan, dtype=dtype)
            for row_idx, vector in control_vectors.items():
                if start <= row_idx < end:
                    if vector.shape != (shape[1],):
                        raise ValueError(f"control vector {row_idx} has shape {vector.shape}")
                    block[row_idx - start] = vector
            handle.write(block.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    expected_size = header_bytes + shape[0] * shape[1] * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise IOError(f"streamed npy size mismatch for {path}: {actual_size} != {expected_size}")
    return {
        "path": str(path.resolve()),
        "shape": [int(shape[0]), int(shape[1])],
        "dtype": str(dtype),
        "header_bytes": header_bytes,
        "data_bytes": int(shape[0] * shape[1] * dtype.itemsize),
        "file_size": int(actual_size),
        "write_mode": "sequential_npy_v2_header_no_mmap",
        "chunk_rows": int(chunk_rows),
    }


def hardlink_or_copy(source: Path, target: Path) -> dict[str, Any]:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
        mode = "hardlink"
    except OSError as exc:
        shutil.copy2(source, target)
        mode = f"copy_fallback:{type(exc).__name__}:{exc}"
    same_inode = bool(source.stat().st_dev == target.stat().st_dev and source.stat().st_ino == target.stat().st_ino)
    if source.stat().st_size != target.stat().st_size:
        raise IOError(f"linked/copied file size mismatch: {source} vs {target}")
    if not same_inode and sha256_file(source) != sha256_file(target):
        raise IOError(f"copied file checksum mismatch: {source} vs {target}")
    return {
        "source": str(source.resolve()),
        "target": str(target.resolve()),
        "mode": mode,
        "same_inode": same_inode,
        "size": int(source.stat().st_size),
    }


def save_feature_table(table: pd.DataFrame, task_dir: Path) -> dict[str, str]:
    feature_csv = task_dir / "feature_table.csv"
    processed_csv = task_dir / "processed.csv"
    table.to_csv(feature_csv, index=False)
    table.to_csv(processed_csv, index=False)
    try:
        native = task_dir / "feature_table.parquet"
        table.to_parquet(native, index=False)
    except Exception:
        native = task_dir / "feature_table.pkl"
        table.to_pickle(native)
    return {
        "feature_table_csv": str(feature_csv.resolve()),
        "feature_table_native": str(native.resolve()),
        "processed_csv": str(processed_csv.resolve()),
    }


def write_split_files(
    *,
    split_dir: Path,
    table: pd.DataFrame,
    set_info: dict[int, dict[str, list[int]]],
    row_to_set: dict[int, int],
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    query_indices = table.index[~table["is_control"].astype(bool)].astype(int).tolist()
    empty_indices: list[int] = []
    empty_sets: dict[int, dict[str, list[int]]] = {}
    dump_pickle(split_dir / "row_to_set_index.pkl", row_to_set)
    dump_pickle(split_dir / "set_info.pkl", set_info)
    dump_pickle(split_dir / "set_to_grouping.pkl", {idx: f"PTV2_{idx + 1}" for idx in set_info})
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
        "policy": "inference-only: all 3,217 single-drug query rows per organoid sample are test anchors",
        "anchor_counts": {"train": 0, "valid": 0, "test": int(len(query_indices))},
        "set_counts": {"train": 0, "valid": 0, "test": int(len(set_info))},
        "rows_per_test_set": {
            str(key): {
                "control": int(len(value["control"])),
                "perturb": int(len(value["perturb"])),
            }
            for key, value in set_info.items()
        },
    }
    dump_json(split_dir / "split_manifest.json", manifest)
    return manifest


def make_task_rows(
    *,
    config: dict[str, Any],
    sample_records: list[dict[str, Any]],
    aligned_expression: np.ndarray,
    scope_df: pd.DataFrame,
    meta: dict[str, Any],
    raw_hash: str,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, dict[str, list[int]]], dict[int, int]]:
    rows: list[dict[str, Any]] = []
    control_vectors: dict[int, np.ndarray] = {}
    set_info: dict[int, dict[str, list[int]]] = {}
    row_to_set: dict[int, int] = {}
    no_pert_index = int(meta["pert_index"]["no"])
    no_index = {
        field: value_index(meta, field, "no")
        for field in ("Cell_plate", "Cell", "batch", "pert_time", "pert_dose1", "pert_dose2")
    }
    machine_index = value_index(meta, "machineID_new", str(config["machineID_new"]))
    time_index = value_index(meta, "pert_time", "24")
    dose_index = value_index(meta, "pert_dose1", "10")

    def common_row(sample: dict[str, Any]) -> dict[str, Any]:
        cell_type_index = value_index(meta, "cell_type", str(sample["cell_type"]))
        return {
            "device": config["device"],
            "device_slug": config["device_slug"],
            "instrument": config["instrument_raw"],
            "instrument_raw": config["instrument_raw"],
            "machineID_new": config["machineID_new"],
            "machineID_new_index": machine_index,
            "Cell_plate": "no",
            "Cell_plate_index": no_index["Cell_plate"],
            "Cell": sample["organoid_sample_id"],
            "Cell_index": no_index["Cell"],
            "cell_llm_index": no_index["Cell"],
            "cell_type": sample["cell_type"],
            "cell_type_index": cell_type_index,
            "cell_type_llm_index": cell_type_index,
            "tissue": sample["tissue"],
            "cell_type_zh": sample["cell_type_zh"],
            "batch": config["batch"],
            "batch_index": no_index["batch"],
            "organoid_sample_id": sample["organoid_sample_id"],
            "patient_sample_id": sample["organoid_sample_id"],
            "sample_pair_id": sample["sample_pair_id"],
            "samp_ID": sample["samp_ID"],
            "pat_ID": sample["pat_ID"],
            "sample_info_audit_value": sample["sample_info_audit_value"],
            "source_input_file": str(Path(config["input_path"]).resolve()),
            "source_input_sha256": raw_hash,
            "source_task": config["task_name"],
            "task_context": EXPERIMENT_NAME,
            "feature_membership": "primary",
            "training_label_scope": "inference_only_unlabeled",
            "PRISM1st_label_total": "",
            "PRISM2nd_label_total": "",
            "synergy": "",
            "unified_label_mask": 1.0,
        }

    for set_idx, sample in enumerate(sample_records):
        control_idx = len(rows)
        control_id = f"exp32::{config['device_slug']}::{sample['sample_pair_id']}::control"
        control_row = {
            **common_row(sample),
            "sample_id": control_id,
            "control": control_id,
            "is_control": True,
            "source_row_role": "exp32_organoid_baseline_control",
            "prediction_type": "control",
            "pert_id1": "no",
            "pert_id2": "no",
            "pert_index1": no_pert_index,
            "pert_index2": no_pert_index,
            "pert_time": "no",
            "pert_time_index": no_index["pert_time"],
            "pert_dose1": "no",
            "pert_dose2": "no",
            "pert_dose1_index": no_index["pert_dose1"],
            "pert_dose2_index": no_index["pert_dose2"],
            "drugname": "",
            "smiles": "",
            "target_protein_list": "[]",
            "query_drug_scope_rank": -1,
            "cell_pertid_time": f"{sample['organoid_sample_id']}_control",
        }
        rows.append(control_row)
        control_vectors[control_idx] = np.asarray(
            aligned_expression[int(sample["expression_row"])], dtype=np.float32
        )
        perturb_rows: list[int] = []
        for drug in scope_df.itertuples(index=False):
            row_idx = len(rows)
            pert_id = str(drug.pert_id)
            query_id = f"exp32::{config['device_slug']}::{sample['sample_pair_id']}::drug::{int(drug.scope_rank)}"
            query_row = {
                **common_row(sample),
                "sample_id": query_id,
                "control": control_id,
                "is_control": False,
                "source_row_role": "exp32_organoid_single_drug_query",
                "prediction_type": "sensitivity",
                "pert_id1": pert_id,
                "pert_id2": pert_id,
                "pert_index1": int(drug.pert_index),
                "pert_index2": int(drug.pert_index),
                "pert_time": 24,
                "pert_time_index": time_index,
                "pert_dose1": 10,
                "pert_dose2": 10,
                "pert_dose1_index": dose_index,
                "pert_dose2_index": dose_index,
                "drugname": pert_id,
                "smiles": str(drug.smiles),
                "target_protein_list": str(drug.target_protein_list),
                "query_drug_scope_rank": int(drug.scope_rank),
                "cell_pertid_time": f"{sample['organoid_sample_id']}_{pert_id}_24",
            }
            rows.append(query_row)
            perturb_rows.append(row_idx)
            row_to_set[row_idx] = set_idx
        row_to_set[control_idx] = set_idx
        set_info[set_idx] = {"control": [control_idx], "perturb": perturb_rows}

    for row_idx, row in enumerate(rows):
        row["feature_row_index"] = int(row_idx)
        row["processed_row_index"] = int(row_idx)
        row["expression_row_index"] = int(row_idx)
    table = pd.DataFrame(rows).reset_index(drop=True)
    if len(table) != EXPECTED_TASK_ROWS or int(table["is_control"].sum()) != EXPECTED_SAMPLE_COUNT:
        raise AssertionError("constructed task row counts do not match exp32 contract")
    return table, control_vectors, set_info, row_to_set


def write_task(
    *,
    output_group_root: Path,
    config: dict[str, Any],
    sample_records: list[dict[str, Any]],
    aligned_expression: np.ndarray,
    expression_audit: dict[str, Any],
    scope_df: pd.DataFrame,
    meta: dict[str, Any],
    ordered_indices: list[int],
    ordered_uniprot: list[str],
    raw_hash: str,
) -> dict[str, Any]:
    task_name = str(config["task_name"])
    task_dir = output_group_root / "tasks" / task_name
    split_dir = output_group_root / "splits" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    table, control_vectors, set_info, row_to_set = make_task_rows(
        config=config,
        sample_records=sample_records,
        aligned_expression=aligned_expression,
        scope_df=scope_df,
        meta=meta,
        raw_hash=raw_hash,
    )
    table_paths = save_feature_table(table, task_dir)
    feature_matrix = task_dir / "feature_expression_matrix.npy"
    matrix_summary = write_npy_streaming(
        feature_matrix,
        shape=(EXPECTED_TASK_ROWS, EXPECTED_AXIS_SIZE),
        control_vectors=control_vectors,
    )
    matrix_link = hardlink_or_copy(feature_matrix, task_dir / "processed_expression_matrix.npy")

    dump_json(task_dir / "feature_ordered_protein_index.json", ordered_indices)
    dump_json(task_dir / "processed_ordered_protein_index.json", ordered_indices)
    dump_json(task_dir / "feature_ordered_protein_uniprot.json", ordered_uniprot)
    dump_json(task_dir / "processed_ordered_protein_uniprot.json", ordered_uniprot)
    sample_ids = table["sample_id"].astype(str).tolist()
    dump_json(task_dir / "feature_sample_ids.json", sample_ids)
    dump_json(task_dir / "processed_sample_ids.json", sample_ids)
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
        "expression_matrix_path": str(feature_matrix.resolve()),
        "ordered_protein_index_path": str((task_dir / "feature_ordered_protein_index.json").resolve()),
        "ordered_protein_uniprot_path": str((task_dir / "feature_ordered_protein_uniprot.json").resolve()),
        "sample_ids_path": str((task_dir / "feature_sample_ids.json").resolve()),
        **table_paths,
        "loading_contract": "exp32 organoid baseline-proteome inference-only single-drug sensitivity",
        "expression_policy": "13 real log1p baseline controls; all 41,821 perturbation query rows are NaN",
        "label_policy": "unlabeled inference; PRISM and synergy columns empty",
        "matrix_summary": matrix_summary,
        "processed_matrix_reuse": matrix_link,
    }
    dump_json(task_dir / "feature_loading_manifest.json", loading_manifest)
    query = table.loc[~table["is_control"].astype(bool)]
    return {
        "task_name": task_name,
        "device": config["device"],
        "machineID_new": config["machineID_new"],
        "task_dir": str(task_dir.resolve()),
        "split_dir": str(split_dir.resolve()),
        "rows": int(len(table)),
        "control_rows": int(table["is_control"].sum()),
        "query_rows": int(len(query)),
        "set_count": int(len(set_info)),
        "matrix_shape": [EXPECTED_TASK_ROWS, EXPECTED_AXIS_SIZE],
        "matrix_dtype": "float32",
        "matrix_summary": matrix_summary,
        "processed_matrix_reuse": matrix_link,
        "feature_table_paths": table_paths,
        "split_manifest": split_manifest,
        "expression_audit": expression_audit,
        "query_contract": {
            "same_drug_two_slots": bool((query["pert_id1"] == query["pert_id2"]).all()),
            "time_24": bool(pd.to_numeric(query["pert_time"]).eq(24).all()),
            "dose1_10": bool(pd.to_numeric(query["pert_dose1"]).eq(10).all()),
            "dose2_10": bool(pd.to_numeric(query["pert_dose2"]).eq(10).all()),
            "prediction_type_sensitivity": bool(query["prediction_type"].eq("sensitivity").all()),
            "labels_empty": True,
        },
    }


def symlink_readonly_alias(source: Path, target: Path) -> dict[str, Any]:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace existing alias: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source.resolve()), str(target), target_is_directory=source.is_dir())
    if target.resolve() != source.resolve():
        raise IOError(f"symlink did not resolve to source: {target} -> {source}")
    return {
        "alias": str(target.absolute()),
        "source": str(source.resolve()),
        "kind": "read_only_symlink",
        "is_directory": bool(source.is_dir()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-training-ready-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--b-input", default=str(DEFAULT_B_INPUT))
    parser.add_argument("--cac-input", default=str(DEFAULT_CAC_INPUT))
    parser.add_argument("--sample-info", default=str(DEFAULT_SAMPLE_INFO))
    parser.add_argument("--force", action="store_true", help="Delete and rebuild a non-empty exp32 output root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_training_ready_root)
    output_root = Path(args.output_root)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_manifest_path = checkpoint_path.parent / "run_manifest.json"
    raw_paths = {
        "B": Path(args.b_input),
        "CAC": Path(args.cac_input),
        "sample_info": Path(args.sample_info),
    }
    required_paths = [source_root / DATASET_GROUP / "global_meta.json", checkpoint_path, checkpoint_manifest_path]
    required_paths.extend(raw_paths.values())
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required exp32 inputs: {missing}")
    if checkpoint_path.name != "epoch=5.ckpt":
        raise ValueError(f"exp32 requires epoch=5.ckpt, got {checkpoint_path}")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise FileExistsError(f"output root is not empty; use --force to rebuild: {output_root}")
        shutil.rmtree(output_root)
    elif args.force and output_root.exists():
        shutil.rmtree(output_root)

    raw_hashes_before = {key: sha256_file(path) for key, path in raw_paths.items()}
    source_group_root = source_root / DATASET_GROUP
    output_group_root = output_root / DATASET_GROUP
    output_group_root.mkdir(parents=True, exist_ok=True)
    meta_path = source_group_root / "global_meta.json"
    meta = load_json(meta_path)
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    if checkpoint_manifest.get("run_status") != "fit_completed":
        raise ValueError("exp09 checkpoint manifest is not fit_completed")
    if checkpoint_manifest.get("task_head") != "unified":
        raise ValueError("exp09 checkpoint manifest task_head is not unified")

    ordered_indices, ordered_uniprot, axis_index_path, axis_uniprot_path = checkpoint_axis(checkpoint_manifest)
    protein_index = {str(key): int(value) for key, value in meta["protein_index"].items()}
    for idx, uniprot in zip(ordered_indices, ordered_uniprot, strict=True):
        if protein_index.get(uniprot) != idx:
            raise ValueError(f"axis/global protein mapping mismatch for {uniprot}: {idx}")

    sample_info, sample_info_audit = load_sample_info(raw_paths["sample_info"])
    scope_df, source_df, scope_audit = build_drug_scope(
        source_group_root=source_group_root,
        meta=meta,
        checkpoint_manifest=checkpoint_manifest,
    )
    scope_csv = output_group_root / "exp32_organoid_drug_scope.csv"
    scope_source_csv = output_group_root / "exp32_organoid_drug_scope_sources.csv"
    scope_json = output_group_root / "exp32_organoid_drug_scope.json"
    scope_df.to_csv(scope_csv, index=False)
    source_df.to_csv(scope_source_csv, index=False)
    scope_audit.update(
        {
            "scope_csv": str(scope_csv.resolve()),
            "scope_source_csv": str(scope_source_csv.resolve()),
        }
    )
    dump_json(scope_json, scope_audit)

    aliases = {
        "global_meta": symlink_readonly_alias(meta_path, output_group_root / "global_meta.json"),
        "derived": symlink_readonly_alias(source_group_root / "derived", output_group_root / "derived"),
    }

    configs = [dict(item) for item in DEVICE_CONFIGS]
    configs[0]["input_path"] = raw_paths["B"]
    configs[1]["input_path"] = raw_paths["CAC"]
    tasks: dict[str, Any] = {}
    device_audits: dict[str, Any] = {}
    for config in configs:
        raw_path = Path(config["input_path"])
        sample_records, aligned_expression, expression_audit = load_device_expression(
            config=config,
            raw_path=raw_path,
            axis_uniprot=ordered_uniprot,
            sample_info=sample_info,
        )
        device_audits[str(config["device"])] = expression_audit
        task_summary = write_task(
            output_group_root=output_group_root,
            config=config,
            sample_records=sample_records,
            aligned_expression=aligned_expression,
            expression_audit=expression_audit,
            scope_df=scope_df,
            meta=meta,
            ordered_indices=ordered_indices,
            ordered_uniprot=ordered_uniprot,
            raw_hash=raw_hashes_before[str(config["device"])],
        )
        tasks[str(config["task_name"])] = task_summary

    raw_hashes_after = {key: sha256_file(path) for key, path in raw_paths.items()}
    if raw_hashes_after != raw_hashes_before:
        raise RuntimeError("one or more raw exp32 inputs changed during the build")
    raw_input_audit = {
        key: {
            "path": str(raw_paths[key].resolve()),
            "sha256_before": raw_hashes_before[key],
            "sha256_after": raw_hashes_after[key],
            "unchanged": raw_hashes_before[key] == raw_hashes_after[key],
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
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "source_global_meta_path": str(meta_path.resolve()),
        "source_global_meta_sha256": sha256_file(meta_path),
        "raw_inputs": raw_input_audit,
        "sample_info_audit": sample_info_audit,
        "protein_axis": {
            "count": int(len(ordered_indices)),
            "ordered_index_path": str(axis_index_path.resolve()),
            "ordered_index_sha256": sha256_file(axis_index_path),
            "ordered_uniprot_path": str(axis_uniprot_path.resolve()),
            "ordered_uniprot_sha256": sha256_file(axis_uniprot_path),
        },
        "drug_scope": {
            **scope_audit,
            "scope_json": str(scope_json.resolve()),
            "scope_csv_sha256": sha256_file(scope_csv),
            "scope_source_csv_sha256": sha256_file(scope_source_csv),
        },
        "readonly_reused_artifacts": aliases,
        "device_expression_audits": device_audits,
        "tasks": tasks,
        "expected_counts": {
            "samples_per_device": EXPECTED_SAMPLE_COUNT,
            "drug_count": EXPECTED_DRUG_COUNT,
            "query_rows_per_task": EXPECTED_QUERY_ROWS,
            "rows_per_task": EXPECTED_TASK_ROWS,
            "combined_prediction_rows": EXPECTED_QUERY_ROWS * 2,
        },
        "fixed_settings": {
            "prediction_type": "sensitivity",
            "single_drug_same_two_slots": True,
            "pert_time": 24,
            "pert_dose1": 10,
            "pert_dose2": 10,
            "B_machine_mapping": "QE_HF -> QE",
            "CAC_machine_mapping": "480_FAIMS",
            "unseen_Cell_Cell_plate_batch_index": 0,
        },
        "acceptance_checks": {
            "raw_hashes_unchanged": True,
            "checkpoint_is_epoch5": True,
            "checkpoint_manifest_fit_completed": True,
            "protein_axis_size_11092": len(ordered_indices) == EXPECTED_AXIS_SIZE,
            "drug_scope_size_3217": len(scope_df) == EXPECTED_DRUG_COUNT,
            "two_tasks_built": len(tasks) == 2,
            "expected_combined_predictions": EXPECTED_QUERY_ROWS * 2 == 83642,
        },
        "notes": [
            "This is inference-only; no model training or fine-tuning is performed.",
            "All query labels are empty, so AUROC/AUPRC are undefined and must not be reported.",
            "Original global metadata and derived artifacts are reused through read-only symlinks.",
            "The final unnamed sample-info column is retained only as an audit field.",
        ],
    }
    summary_path = output_group_root / "exp32_organoid_build_summary.json"
    dump_json(summary_path, summary)
    print(f"[exp32] wrote training-ready root: {output_root}")
    print(f"[exp32] wrote drug scope: {scope_csv} ({len(scope_df)} drugs)")
    for task_name, task_summary in tasks.items():
        print(
            f"[exp32] {task_name}: rows={task_summary['rows']} "
            f"queries={task_summary['query_rows']} shape={task_summary['matrix_shape']}"
        )
    print(f"[exp32] summary: {summary_path}")


if __name__ == "__main__":
    main()
