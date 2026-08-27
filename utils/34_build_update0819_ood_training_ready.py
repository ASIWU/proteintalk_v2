#!/usr/bin/env python3
"""Build inference-only Exp34 artifacts for the update-0819 OOD drugs.

The source PTV3 artifacts are read-only.  Two new drug IDs and feature rows are
appended in a separate training-ready root.  The two inference tasks share all
numeric features and differ only in ``target_protein_list``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.graph_feature_utils import (  # noqa: E402
    _csr_from_npy,
    _dense_row_normalized_context,
    _normalize_csr_rows,
    _random_projection,
    _sparse_stats,
    _standardize,
)
from utils.npy_io import safe_np_load  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_GROUP = "ptv3"
EXPERIMENT_NAME = "exp34_update0819_ood_epoch2"
SPLIT_STRATEGY = "test_only"
EXPECTED_INPUT_ROWS = 14
EXPECTED_CELL_COUNT = 7
EXPECTED_DRUG_COUNT = 2
EXPECTED_AXIS_SIZE = 11_092
EXPECTED_SOURCE_PERT_COUNT = 6_131
EXPECTED_CHECKPOINT_SHA256 = "7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076"
SIMILARITY_THRESHOLD = 0.5
GRAPH_FEATURE_DIM = 128
GRAPH_FEATURE_SEED = 17

DEFAULT_INPUT = REPO_ROOT / "data/rawdata/update_0819/260513ptv_drug_cell_predict.csv"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/training_ready"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/training_ready_exp34_update0819_ood"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt"
)
DEFAULT_SOURCE_GRAPH = (
    REPO_ROOT
    / "graph_cache/ptv01_08_posweight_combo/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy"
)

TASKS = {
    "target_none": "ptv3_exp34_update0819_ood_target_none",
    "target_mechanism": "ptv3_exp34_update0819_ood_target_mechanism",
}
MECHANISM_TARGETS = {
    "daraxonrasib": ["P01111", "P01112", "P01116"],  # NRAS, HRAS, KRAS
    "zoldonrasib": ["P01116"],  # KRAS
}


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def canonical_isomeric_smiles(value: object) -> str:
    text = clean_text(value)
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {text!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def exp34_pert_id(smiles: str) -> str:
    digest = hashlib.sha1(canonical_isomeric_smiles(smiles).encode("utf-8")).hexdigest()[:12]
    return f"exp34smiles::{digest}"


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


def value_index(meta: dict[str, Any], field: str, value: str) -> int:
    mapping_key = "pert_dose" if field in {"pert_dose1", "pert_dose2"} else field
    mapping = meta["value_to_index"][mapping_key]
    if value not in mapping:
        raise KeyError(f"missing categorical mapping {mapping_key}={value!r}")
    return int(float(mapping[value]))


def ordered_ids(mapping: dict[str, int]) -> list[str]:
    result: list[str | None] = [None] * len(mapping)
    for item, index in mapping.items():
        index = int(index)
        if index < 0 or index >= len(result) or result[index] is not None:
            raise ValueError("index mapping is not a dense one-to-one axis")
        result[index] = str(item)
    if any(item is None for item in result):
        raise ValueError("index mapping contains gaps")
    return [str(item) for item in result]


def symlink_alias(source: Path, target: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())
    return {"source": str(source), "target": str(target), "kind": "symlink"}


def hardlink_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def morgan_fingerprint(smiles: str, *, n_bits: int = 2048) -> tuple[Any, np.ndarray]:
    molecule = Chem.MolFromSmiles(clean_text(smiles))
    if molecule is None:
        raise ValueError(f"cannot fingerprint invalid SMILES: {smiles!r}")
    fingerprint = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits).GetFingerprint(molecule)
    vector = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, vector)
    return fingerprint, vector


def read_and_validate_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    expected_columns = {"Cell_name", "cell_in_ptvdrug", "drug_name", "drug_name2", "smiles"}
    missing = sorted(expected_columns - set(frame.columns))
    if missing:
        raise ValueError(f"input is missing columns: {missing}")
    if len(frame) != EXPECTED_INPUT_ROWS:
        raise ValueError(f"expected {EXPECTED_INPUT_ROWS} input rows, found {len(frame)}")
    if frame[list(expected_columns)].isna().any().any():
        raise ValueError("input contains missing required values")
    if frame.duplicated().any():
        raise ValueError("input contains duplicate rows")
    if frame["cell_in_ptvdrug"].nunique() != EXPECTED_CELL_COUNT:
        raise ValueError("input must contain exactly seven cell aliases")
    if frame["drug_name"].nunique() != EXPECTED_DRUG_COUNT:
        raise ValueError("input must contain exactly two drugs")
    if len(frame) != frame["cell_in_ptvdrug"].nunique() * frame["drug_name"].nunique():
        raise ValueError("input is not the complete cell-by-drug Cartesian product")
    frame = frame.copy()
    frame["input_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["Cell_norm"] = frame["cell_in_ptvdrug"].map(canonical_text)
    frame["drug_name_norm"] = frame["drug_name"].map(lambda value: clean_text(value).lower())
    for drug_name, group in frame.groupby("drug_name_norm", sort=False):
        canonical = group["smiles"].map(canonical_isomeric_smiles).unique().tolist()
        if len(canonical) != 1:
            raise ValueError(f"{drug_name}: multiple structures in input")
    return frame


def checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 mismatch")
    manifest = load_json(checkpoint.parent / "run_manifest.json")
    if manifest.get("run_status") != "fit_completed":
        raise ValueError("checkpoint manifest is not fit_completed")
    if manifest.get("task_name") != "ptv3_main_doubledrug":
        raise ValueError("checkpoint task is not ptv3_main_doubledrug")
    if manifest.get("split_strategy") != "all_train_subset_test":
        raise ValueError("checkpoint split strategy mismatch")
    if manifest.get("task_head") != "unified" or manifest.get("model_type") != "fast_delta":
        raise ValueError("checkpoint model contract mismatch")
    return manifest


def select_fixed_controls(
    *, manifest: dict[str, Any], cells: list[str], meta: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray, list[int], list[str]]:
    task_dir = Path(str(manifest["task_dir"]))
    table = pd.read_parquet(task_dir / "feature_table.parquet").reset_index(drop=True)
    table["feature_row_index"] = np.arange(len(table), dtype=np.int64)
    expression = safe_np_load(task_dir / "feature_expression_matrix.npy", mmap_mode="r")
    ordered_indices = [int(item) for item in load_json(task_dir / "feature_ordered_protein_index.json")]
    ordered_uniprot = [str(item) for item in load_json(task_dir / "feature_ordered_protein_uniprot.json")]
    if len(ordered_indices) != EXPECTED_AXIS_SIZE or expression.shape[1] != EXPECTED_AXIS_SIZE:
        raise ValueError("checkpoint protein axis is not 11,092")

    cell_mapping = meta["value_to_index"]["Cell"]
    selected: list[pd.Series] = []
    for cell in cells:
        if cell not in cell_mapping:
            raise KeyError(f"Cell alias is absent from global mapping: {cell}")
        candidates = table.loc[
            table["is_control"].astype(bool) & table["Cell_norm"].astype(str).eq(cell)
        ].sort_values("feature_row_index")
        if len(candidates) != 4:
            raise ValueError(f"{cell}: expected four controls, found {len(candidates)}")
        row = candidates.iloc[0].copy()
        if int(row["Cell_index"]) != int(float(cell_mapping[cell])):
            raise ValueError(f"{cell}: control Cell index mismatch")
        selected.append(row)
    controls = pd.DataFrame(selected).reset_index(drop=True)
    return controls, expression, ordered_indices, ordered_uniprot


def audit_drug_similarity(
    *, query_drugs: pd.DataFrame, meta: dict[str, Any], threshold: float
) -> pd.DataFrame:
    known: list[tuple[str, Any]] = []
    for pert_id, smiles in meta["pertid_to_smiles"].items():
        if not clean_text(smiles):
            continue
        try:
            fingerprint, _ = morgan_fingerprint(str(smiles))
        except ValueError:
            continue
        known.append((str(pert_id), fingerprint))
    target_map = {
        str(key): sorted({int(item) for item in values})
        for key, values in meta.get("pertid_to_target_protein_list", {}).items()
    }

    rows: list[dict[str, Any]] = []
    for drug in query_drugs.itertuples(index=False):
        fingerprint, _ = morgan_fingerprint(str(drug.smiles))
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, [item[1] for item in known]
        )
        maximum = float(max(similarities)) if similarities else 0.0
        top_ids = sorted(
            known[index][0]
            for index, similarity in enumerate(similarities)
            if abs(float(similarity) - maximum) <= 1e-12
        )
        threshold_ids = sorted(
            known[index][0]
            for index, similarity in enumerate(similarities)
            if float(similarity) >= threshold
        )
        transferred: list[int] = []
        transfer_source_ids: list[str] = []
        if threshold_ids:
            top_at_threshold = [item for item in top_ids if item in threshold_ids]
            nonempty = [target_map.get(item, []) for item in top_at_threshold if target_map.get(item)]
            unique_target_sets = {tuple(items) for items in nonempty}
            if len(unique_target_sets) > 1:
                raise ValueError(
                    f"{drug.drug_name}: tied top similar drugs disagree on targets: {top_at_threshold}"
                )
            if unique_target_sets:
                transferred = list(next(iter(unique_target_sets)))
                transfer_source_ids = [
                    item for item in top_at_threshold if target_map.get(item) == transferred
                ]
        rows.append(
            {
                "drug_name": str(drug.drug_name),
                "drug_name2": str(drug.drug_name2),
                "pert_id": str(drug.pert_id),
                "query_smiles": str(drug.smiles),
                "canonical_isomeric_smiles": canonical_isomeric_smiles(drug.smiles),
                "maximum_morgan_tanimoto": maximum,
                "top_candidate_ids": json.dumps(top_ids, separators=(",", ":")),
                "threshold": float(threshold),
                "threshold_candidate_count": len(threshold_ids),
                "threshold_candidate_ids": json.dumps(threshold_ids, separators=(",", ":")),
                "transferred_target_indices": json.dumps(transferred, separators=(",", ":")),
                "target_transfer_source_ids": json.dumps(
                    transfer_source_ids, separators=(",", ":")
                ),
                "only_target_list_transferred": True,
            }
        )
    return pd.DataFrame(rows)


def extend_meta(
    *, source_meta: dict[str, Any], query_drugs: pd.DataFrame, similarity_audit: pd.DataFrame
) -> dict[str, Any]:
    meta = json.loads(json.dumps(source_meta))
    pert_index = {str(key): int(value) for key, value in meta["pert_index"].items()}
    if len(pert_index) != EXPECTED_SOURCE_PERT_COUNT:
        raise ValueError("unexpected source perturbation count")
    index_to_id = ordered_ids(pert_index)
    transfer_by_id = {
        str(row.pert_id): json.loads(str(row.transferred_target_indices))
        for row in similarity_audit.itertuples(index=False)
    }
    for drug in query_drugs.itertuples(index=False):
        pert_id = str(drug.pert_id)
        if pert_id in pert_index:
            raise ValueError(f"new OOD ID already exists: {pert_id}")
        pert_index[pert_id] = len(index_to_id)
        index_to_id.append(pert_id)
        meta["pertid_to_smiles"][pert_id] = str(drug.smiles)
        meta.setdefault("pertid_to_target_protein_list", {})[pert_id] = transfer_by_id[pert_id]
        meta.setdefault("pertid_to_target_uniprot_list", {})[pert_id] = []
        meta.setdefault("pertid_to_target_protein_list", {})[pert_id] = transfer_by_id[pert_id]
        meta.setdefault("pertid_to_missing_target_uniprot", {})[pert_id] = []
    meta["pert_index"] = pert_index
    meta["pert_index_to_id"] = index_to_id
    meta["task_names"] = list(TASKS.values())
    meta["generated_at"] = iso_now()
    meta["exp34_ood_extension"] = {
        "experiment_name": EXPERIMENT_NAME,
        "source_pert_count": EXPECTED_SOURCE_PERT_COUNT,
        "extended_pert_count": len(pert_index),
        "added_pert_ids": query_drugs["pert_id"].astype(str).tolist(),
        "all_added_drugs_are_ood": True,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "similar_drug_policy": "new ID and numeric features; transfer target_protein_list only",
        "pdi_policy": "source block unchanged; new rows all zero in both target scenarios",
        "target_scenarios": {
            "target_none": "similarity transfer only; empty for the current two drugs",
            "target_mechanism": "task-table-only RAS direct-target override; PDI remains zero",
        },
    }
    return meta


def write_extended_drug_embedding(
    *, source_path: Path, output_path: Path, source_meta: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any]:
    payload = load_pickle(source_path)
    source = np.asarray(payload["embedding_matrix"], dtype=np.float32)
    output = np.zeros((len(meta["pert_index"]), source.shape[1]), dtype=np.float32)
    output[: source.shape[0]] = source
    new_ids = ordered_ids(meta["pert_index"])[source.shape[0] :]
    for pert_id in new_ids:
        _, vector = morgan_fingerprint(meta["pertid_to_smiles"][pert_id], n_bits=source.shape[1])
        output[int(meta["pert_index"][pert_id])] = vector
    result = dict(payload)
    result.update(
        {
            "item_to_index": meta["pert_index"],
            "index_to_item": ordered_ids(meta["pert_index"]),
            "embedding_matrix": output,
            "exp34_ood_extension": {
                "source_path": str(source_path.resolve()),
                "source_rows_copied_exactly": int(source.shape[0]),
                "new_rows_generated": len(new_ids),
            },
        }
    )
    dump_pickle(output_path, result)
    summary = {
        "kind": "drug_embedding",
        "shape": list(map(int, output.shape)),
        "source_shape": list(map(int, source.shape)),
        "source_block_exact": bool(np.array_equal(output[: source.shape[0]], source)),
        "new_rows_nonzero": bool(np.all(np.any(output[source.shape[0] :] != 0, axis=1))),
    }
    dump_json(output_path.with_suffix(".meta.json"), summary)
    return summary


def write_extended_ddi(
    *, source_path: Path, output_path: Path, source_meta: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any]:
    source = np.asarray(safe_np_load(source_path), dtype=np.float32)
    total = len(meta["pert_index"])
    output = np.zeros((total, total), dtype=np.float32)
    output[: source.shape[0], : source.shape[1]] = source
    fingerprints: list[Any] = []
    for pert_id in ordered_ids(meta["pert_index"]):
        smiles = clean_text(meta["pertid_to_smiles"].get(pert_id, ""))
        if smiles:
            fingerprint, _ = morgan_fingerprint(smiles)
        else:
            fingerprint = AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(
                Chem.MolFromSmiles("")
            )
        fingerprints.append(fingerprint)
    for row_index in range(source.shape[0], total):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[row_index], fingerprints
        )
        output[row_index, :] = np.asarray(similarities, dtype=np.float32)
        output[:, row_index] = np.asarray(similarities, dtype=np.float32)
        output[row_index, row_index] = 1.0
    np.save(output_path, output)
    summary = {
        "kind": "ddi_matrix",
        "shape": list(map(int, output.shape)),
        "source_shape": list(map(int, source.shape)),
        "source_block_exact": bool(
            np.array_equal(output[: source.shape[0], : source.shape[1]], source)
        ),
        "new_rows_finite": bool(np.isfinite(output[source.shape[0] :]).all()),
        "policy": "source block exact; new rows and columns use radius-2/2048 Morgan Tanimoto",
    }
    dump_json(output_path.with_suffix(".meta.json"), summary)
    return summary


def write_extended_pdi(*, source_path: Path, output_path: Path, total_rows: int) -> dict[str, Any]:
    source = np.asarray(safe_np_load(source_path), dtype=np.float32)
    output = np.zeros((total_rows, source.shape[1]), dtype=np.float32)
    output[: source.shape[0]] = source
    np.save(output_path, output)
    summary = {
        "kind": "pdi_matrix",
        "shape": list(map(int, output.shape)),
        "source_shape": list(map(int, source.shape)),
        "source_block_exact": bool(np.array_equal(output[: source.shape[0]], source)),
        "new_rows_all_zero": bool(np.count_nonzero(output[source.shape[0] :]) == 0),
        "policy": "source block exact; OOD rows all zero for both target scenarios",
    }
    dump_json(output_path.with_suffix(".meta.json"), summary)
    return summary


def base_graph_raw_and_stats(
    *,
    ppi_path: Path,
    pdi_path: Path,
    ddi_path: Path,
    protein_embedding: np.ndarray,
    drug_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]], dict[str, Any]]:
    protein_projection = _random_projection(
        protein_embedding.shape[1], GRAPH_FEATURE_DIM, seed=GRAPH_FEATURE_SEED
    )
    drug_projection = _random_projection(
        drug_embedding.shape[1], GRAPH_FEATURE_DIM, seed=GRAPH_FEATURE_SEED + 1
    )
    protein_projected = np.nan_to_num(protein_embedding @ protein_projection).astype(np.float32)
    drug_projected = np.nan_to_num(drug_embedding @ drug_projection).astype(np.float32)
    pdi = _csr_from_npy(pdi_path, name="PDI")
    ppi = _csr_from_npy(ppi_path, name="PPI")
    pdi_norm = _normalize_csr_rows(pdi)
    ppi_norm = _normalize_csr_rows(ppi)
    ppi_neighbor = np.asarray(ppi_norm @ protein_projected, dtype=np.float32)
    pdi_direct = np.asarray(pdi_norm @ protein_projected, dtype=np.float32)
    pdi_ppi = np.asarray(pdi_norm @ ppi_neighbor, dtype=np.float32)
    ddi_context, ddi_stats = _dense_row_normalized_context(ddi_path, drug_projected)
    protein_node_projection = _random_projection(
        pdi.shape[1], GRAPH_FEATURE_DIM, seed=GRAPH_FEATURE_SEED + 2
    )
    drug_node_projection = _random_projection(
        drug_projected.shape[0], GRAPH_FEATURE_DIM, seed=GRAPH_FEATURE_SEED + 3
    )
    ppi_neighbor_struct = np.asarray(ppi_norm @ protein_node_projection, dtype=np.float32)
    pdi_struct = np.asarray(pdi_norm @ protein_node_projection, dtype=np.float32)
    pdi_ppi_struct = np.asarray(pdi_norm @ ppi_neighbor_struct, dtype=np.float32)
    ddi_struct = _dense_row_normalized_context(ddi_path, drug_node_projection)[0]
    blocks = [
        ("pdi_direct", pdi_direct),
        ("pdi_ppi", pdi_ppi),
        ("ddi_context", ddi_context),
        ("pdi_struct", pdi_struct),
        ("pdi_ppi_struct", pdi_ppi_struct),
        ("ddi_struct", ddi_struct),
        ("pdi_stats", _sparse_stats(pdi)),
        ("ddi_stats", ddi_stats),
    ]
    raw = np.concatenate([block for _, block in blocks], axis=1).astype(np.float32)
    standardized, mean, std = _standardize(raw)
    slices: dict[str, list[int]] = {}
    cursor = 0
    for name, block in blocks:
        slices[name] = [cursor, cursor + int(block.shape[1])]
        cursor += int(block.shape[1])
    shared = {
        "protein_projected": protein_projected,
        "ppi_neighbor": ppi_neighbor,
        "protein_node_projection": protein_node_projection,
        "ppi_neighbor_struct": ppi_neighbor_struct,
    }
    return standardized, mean, std, slices, shared


def new_graph_raw_rows(
    *,
    ddi_path: Path,
    drug_embedding: np.ndarray,
    new_row_start: int,
) -> np.ndarray:
    drug_projection = _random_projection(
        drug_embedding.shape[1], GRAPH_FEATURE_DIM, seed=GRAPH_FEATURE_SEED + 1
    )
    drug_projected = np.nan_to_num(drug_embedding @ drug_projection).astype(np.float32)
    ddi = np.asarray(safe_np_load(ddi_path, mmap_mode="r"), dtype=np.float32)
    block = np.asarray(ddi[new_row_start:], dtype=np.float32)
    row_sum = block.sum(axis=1).astype(np.float32)
    denominator = np.where(row_sum > 0, row_sum, 1.0).astype(np.float32)
    normalized = block / denominator[:, None]
    ddi_context = np.asarray(normalized @ drug_projected, dtype=np.float32)
    ddi_stats = np.stack(
        [
            np.log1p(row_sum),
            np.log1p((block > 0).sum(axis=1).astype(np.float32)),
            block.max(axis=1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    drug_node_projection = _random_projection(
        drug_embedding.shape[0], GRAPH_FEATURE_DIM, seed=GRAPH_FEATURE_SEED + 3
    )
    ddi_struct = np.asarray(normalized @ drug_node_projection, dtype=np.float32)
    n_new = drug_embedding.shape[0] - new_row_start
    zeros128 = np.zeros((n_new, GRAPH_FEATURE_DIM), dtype=np.float32)
    zeros3 = np.zeros((n_new, 3), dtype=np.float32)
    return np.concatenate(
        [
            zeros128,
            zeros128,
            ddi_context,
            zeros128,
            zeros128,
            ddi_struct,
            zeros3,
            ddi_stats,
        ],
        axis=1,
    ).astype(np.float32)


def write_extended_graph_features(
    *,
    source_graph_path: Path,
    output_path: Path,
    ppi_path: Path,
    source_pdi_path: Path,
    source_ddi_path: Path,
    extended_pdi_path: Path,
    extended_ddi_path: Path,
    protein_embedding_path: Path,
    source_drug_embedding_path: Path,
    extended_drug_embedding_path: Path,
    final_output_root: Path,
    staging_root: Path,
) -> dict[str, Any]:
    protein_embedding = np.asarray(
        load_pickle(protein_embedding_path)["embedding_matrix"], dtype=np.float32
    )
    source_drug_embedding = np.asarray(
        load_pickle(source_drug_embedding_path)["embedding_matrix"], dtype=np.float32
    )
    extended_drug_embedding = np.asarray(
        load_pickle(extended_drug_embedding_path)["embedding_matrix"], dtype=np.float32
    )
    rebuilt, mean, std, slices, _ = base_graph_raw_and_stats(
        ppi_path=ppi_path,
        pdi_path=source_pdi_path,
        ddi_path=source_ddi_path,
        protein_embedding=protein_embedding,
        drug_embedding=source_drug_embedding,
    )
    source_graph = np.asarray(safe_np_load(source_graph_path, mmap_mode="r"), dtype=np.float32)
    if rebuilt.shape != source_graph.shape:
        raise ValueError("rebuilt source graph shape mismatch")
    max_abs = float(np.max(np.abs(rebuilt - source_graph)))
    if max_abs > 1e-5:
        raise ValueError(f"source graph regeneration mismatch: max_abs={max_abs}")
    raw_new = new_graph_raw_rows(
        ddi_path=extended_ddi_path,
        drug_embedding=extended_drug_embedding,
        new_row_start=source_graph.shape[0],
    )
    standardized_new = np.nan_to_num((raw_new - mean) / std).astype(np.float32)
    output = np.concatenate([source_graph, standardized_new], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output)
    stats_path = output_path.with_suffix(".standardization.npz")
    np.savez(stats_path, mean=mean, std=std)

    final_group = final_output_root / DATASET_GROUP
    actual_group = staging_root / DATASET_GROUP

    def signature(expected_resolved_path: Path, actual_path: Path) -> dict[str, Any]:
        stat = actual_path.stat()
        return {
            "path": str(expected_resolved_path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    sources = {
        "ppi_matrix": signature(
            (actual_group / "derived/ppi_matrix.npy").resolve(),
            actual_group / "derived/ppi_matrix.npy",
        ),
        "pdi_matrix": signature(
            final_group / "derived/pdi_matrix.npy", extended_pdi_path
        ),
        "ddi_matrix": signature(
            final_group / "derived/ddi_matrix.npy", extended_ddi_path
        ),
    }
    final_feature_path = final_group / "graph_cache" / output_path.name
    summary = {
        "dataset_group": DATASET_GROUP,
        "graph_feature_dim": GRAPH_FEATURE_DIM,
        "seed": GRAPH_FEATURE_SEED,
        "include_structural_rp": True,
        "include_multihop": False,
        "sources": sources,
        "protein_embedding_shape": list(map(int, protein_embedding.shape)),
        "drug_embedding_shape": list(map(int, extended_drug_embedding.shape)),
        "feature_path": str(final_feature_path),
        "feature_shape": list(map(int, output.shape)),
        "feature_slices": slices,
        "standardization_mean_shape": list(map(int, mean.shape)),
        "standardization_std_shape": list(map(int, std.shape)),
        "source_graph_path": str(source_graph_path.resolve()),
        "source_graph_rows_copied_exactly": int(source_graph.shape[0]),
        "source_graph_regeneration_max_abs": max_abs,
        "new_graph_rows": int(len(standardized_new)),
        "new_graph_rows_finite": bool(np.isfinite(standardized_new).all()),
        "standardization_policy": "new rows use source 6131-row mean/std; source rows copied exactly",
        "graph_feature_description": (
            "concat(normalized PDI pooled protein projection, normalized PDI pooled "
            "PPI-neighbor protein projection, normalized DDI pooled drug projection, "
            "structural random projections, PDI row stats, DDI row stats)"
        ),
    }
    dump_json(output_path.with_suffix(".meta.json"), summary)
    return summary


def copy_control_row(source: pd.Series, *, task_name: str, row_index: int) -> dict[str, Any]:
    row = source.to_dict()
    cell = str(source["Cell_norm"])
    control_id = f"exp34::{task_name}::{cell}::control"
    row.update(
        {
            "sample_id": control_id,
            "control": control_id,
            "is_control": True,
            "source_row_role": "exp34_fixed_first_checkpoint_control",
            "source_checkpoint_sample_id": str(source["sample_id"]),
            "source_checkpoint_feature_row_index": int(source["feature_row_index"]),
            "feature_membership": "primary",
            "training_label_scope": "inference_only_unlabeled",
            "task_context": EXPERIMENT_NAME,
            "cell_llm_index": int(source["Cell_index"]),
            "cell_type_llm_index": int(source["cell_type_index"]),
            "pert_id1": "no",
            "pert_id2": "no",
            "pert_time": "no",
            "pert_time_norm": "no",
            "pert_dose1": "no",
            "pert_dose2": "no",
            "pert_dose1_norm": "no",
            "pert_dose2_norm": "no",
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
            "target_scenario": "control",
            "feature_row_index": row_index,
            "processed_row_index": row_index,
            "expression_row_index": row_index,
        }
    )
    return row


def build_task(
    *,
    scenario: str,
    task_name: str,
    input_frame: pd.DataFrame,
    query_drugs: pd.DataFrame,
    similarity_audit: pd.DataFrame,
    controls: pd.DataFrame,
    source_expression: np.ndarray,
    meta: dict[str, Any],
    ordered_indices: list[int],
    ordered_uniprot: list[str],
    task_dir: Path,
    split_dir: Path,
) -> dict[str, Any]:
    no_pert = int(meta["pert_index"]["no"])
    no_indices = {
        field: value_index(meta, field, "no")
        for field in ("pert_time", "pert_dose1", "pert_dose2")
    }
    time_index = value_index(meta, "pert_time", "24")
    dose_index = value_index(meta, "pert_dose1", "10")
    mechanism_indices = {
        drug: sorted(int(meta["protein_index"][uniprot]) for uniprot in uniprot_ids)
        for drug, uniprot_ids in MECHANISM_TARGETS.items()
    }
    transfer_targets = {
        str(row.pert_id): json.loads(str(row.transferred_target_indices))
        for row in similarity_audit.itertuples(index=False)
    }
    drug_lookup = query_drugs.set_index("drug_name_norm").to_dict("index")
    control_lookup = {str(row.Cell_norm): row for row in controls.itertuples(index=False)}

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    control_row_by_cell: dict[str, int] = {}
    set_index_by_cell: dict[str, int] = {}
    set_info: dict[int, dict[str, list[int]]] = {}
    row_to_set: dict[int, int] = {}
    for set_index, control in enumerate(controls.itertuples(index=False)):
        source = pd.Series(control._asdict())
        row_index = len(rows)
        row = copy_control_row(source, task_name=task_name, row_index=row_index)
        row["pert_index1"] = no_pert
        row["pert_index2"] = no_pert
        row["pert_time_index"] = no_indices["pert_time"]
        row["pert_dose1_index"] = no_indices["pert_dose1"]
        row["pert_dose2_index"] = no_indices["pert_dose2"]
        rows.append(row)
        vectors.append(
            np.asarray(source_expression[int(source["expression_row_index"])], dtype=np.float32)
        )
        cell = str(source["Cell_norm"])
        control_row_by_cell[cell] = row_index
        set_index_by_cell[cell] = set_index
        row_to_set[row_index] = set_index
        set_info[set_index] = {"control": [row_index], "perturb": []}

    for record in input_frame.sort_values("input_row_index").itertuples(index=False):
        cell = str(record.Cell_norm)
        control = control_lookup[cell]
        drug = drug_lookup[str(record.drug_name_norm)]
        pert_id = str(drug["pert_id"])
        targets = (
            mechanism_indices[str(record.drug_name_norm)]
            if scenario == "target_mechanism"
            else transfer_targets[pert_id]
        )
        row_index = len(rows)
        sample_id = f"exp34::{scenario}::{int(record.input_row_index):02d}"
        row = dict(control._asdict())
        row.update(
            {
                "sample_id": sample_id,
                "control": rows[control_row_by_cell[cell]]["sample_id"],
                "is_control": False,
                "source_row_role": "exp34_update0819_ood_single_drug_query",
                "source_checkpoint_sample_id": str(control.sample_id),
                "source_checkpoint_feature_row_index": int(control.feature_row_index),
                "feature_membership": "primary",
                "training_label_scope": "inference_only_unlabeled",
                "task_context": EXPERIMENT_NAME,
                "cell_llm_index": int(control.Cell_index),
                "cell_type_llm_index": int(control.cell_type_index),
                "input_row_index": int(record.input_row_index),
                "Cell_name_input": str(record.Cell_name),
                "cell_in_ptvdrug_input": str(record.cell_in_ptvdrug),
                "drug_name": str(record.drug_name),
                "drug_name2": str(record.drug_name2),
                "drugname": str(record.drug_name),
                "drugname1": str(record.drug_name),
                "drugname2": str(record.drug_name),
                "pert_id1": pert_id,
                "pert_id2": pert_id,
                "pert_index1": int(meta["pert_index"][pert_id]),
                "pert_index2": int(meta["pert_index"][pert_id]),
                "pert_time": "24",
                "pert_time_norm": "24",
                "pert_time_index": time_index,
                "pert_dose1": "10",
                "pert_dose2": "10",
                "pert_dose1_norm": "10",
                "pert_dose2_norm": "10",
                "pert_dose1_index": dose_index,
                "pert_dose2_index": dose_index,
                "smiles": str(record.smiles),
                "smiles1": str(record.smiles),
                "smiles2": str(record.smiles),
                "target_protein_list": json.dumps(targets, separators=(",", ":")),
                "target_scenario": scenario,
                "ood_drug": True,
                "maximum_morgan_tanimoto": float(drug["maximum_morgan_tanimoto"]),
                "historical_target_transfer": bool(transfer_targets[pert_id]),
                "PRISM1st_label_total": "",
                "PRISM2nd_label_total": "",
                "synergy": "",
                "unified_label_mask": 1.0,
                "feature_row_index": row_index,
                "processed_row_index": row_index,
                "expression_row_index": row_index,
            }
        )
        rows.append(row)
        vectors.append(np.full(EXPECTED_AXIS_SIZE, np.nan, dtype=np.float32))
        set_index = set_index_by_cell[cell]
        row_to_set[row_index] = set_index
        set_info[set_index]["perturb"].append(row_index)

    table = pd.DataFrame(rows).reset_index(drop=True)
    matrix = np.stack(vectors).astype(np.float32)
    if len(table) != EXPECTED_CELL_COUNT + EXPECTED_INPUT_ROWS or matrix.shape != (
        EXPECTED_CELL_COUNT + EXPECTED_INPUT_ROWS,
        EXPECTED_AXIS_SIZE,
    ):
        raise AssertionError("task row or matrix shape mismatch")
    perturb = table.loc[~table["is_control"].astype(bool)]
    if not (perturb["pert_index1"] == perturb["pert_index2"]).all():
        raise AssertionError("single-drug two-slot contract failed")
    if scenario == "target_none" and perturb["target_protein_list"].ne("[]").any():
        raise AssertionError("target_none unexpectedly contains targets")

    task_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(task_dir / "feature_table.parquet", index=False)
    table.to_csv(task_dir / "feature_table.csv", index=False)
    table.to_csv(task_dir / "processed.csv", index=False)
    np.save(task_dir / "feature_expression_matrix.npy", matrix)
    matrix_alias = hardlink_or_copy(
        task_dir / "feature_expression_matrix.npy", task_dir / "processed_expression_matrix.npy"
    )
    for prefix in ("feature", "processed"):
        dump_json(task_dir / f"{prefix}_ordered_protein_index.json", ordered_indices)
        dump_json(task_dir / f"{prefix}_ordered_protein_uniprot.json", ordered_uniprot)
        dump_json(task_dir / f"{prefix}_sample_ids.json", table["sample_id"].astype(str).tolist())

    query_indices = perturb.index.astype(int).tolist()
    empty_indices: list[int] = []
    empty_sets: dict[int, dict[str, list[int]]] = {}
    dump_pickle(split_dir / "row_to_set_index.pkl", row_to_set)
    dump_pickle(split_dir / "set_info.pkl", set_info)
    dump_pickle(
        split_dir / "set_to_grouping.pkl",
        {set_index: cell for cell, set_index in set_index_by_cell.items()},
    )
    for split_name, indices, sets in (
        ("train", empty_indices, empty_sets),
        ("valid", empty_indices, empty_sets),
        ("test", query_indices, set_info),
    ):
        dump_pickle(split_dir / f"{split_name}_indices_{SPLIT_STRATEGY}.pkl", indices)
        dump_pickle(split_dir / f"{split_name}_set_info_{SPLIT_STRATEGY}.pkl", sets)
    dump_pickle(split_dir / f"val_indices_{SPLIT_STRATEGY}.pkl", empty_indices)
    dump_pickle(split_dir / f"val_set_info_{SPLIT_STRATEGY}.pkl", empty_sets)
    split_manifest = {
        "generated_at": iso_now(),
        "strategy": SPLIT_STRATEGY,
        "policy": "inference-only; fixed first checkpoint control per Cell",
        "control_selection": "minimum checkpoint feature_row_index",
        "anchor_counts": {"train": 0, "valid": 0, "test": EXPECTED_INPUT_ROWS},
        "set_counts": {"train": 0, "valid": 0, "test": EXPECTED_CELL_COUNT},
    }
    dump_json(split_dir / "split_manifest.json", split_manifest)
    return {
        "task_name": task_name,
        "scenario": scenario,
        "rows": int(len(table)),
        "control_rows": int(table["is_control"].astype(bool).sum()),
        "query_rows": int((~table["is_control"].astype(bool)).sum()),
        "matrix_shape": list(map(int, matrix.shape)),
        "matrix_alias": matrix_alias,
        "target_counts": perturb["target_protein_list"].value_counts().to_dict(),
        "split_manifest": split_manifest,
    }


def preflight_existing(output_root: Path, checkpoint: Path) -> dict[str, Any]:
    group = output_root / DATASET_GROUP
    summary_path = group / "exp34_update0819_ood_build_summary.json"
    required = [
        group / "global_meta.json",
        group / "derived/drug_embedding_morgan_2048.pkl",
        group / "derived/pdi_matrix.npy",
        group / "derived/ddi_matrix.npy",
        group / "graph_cache/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy",
        summary_path,
    ]
    for task_name in TASKS.values():
        required.extend(
            [
                group / "tasks" / task_name / "feature_table.parquet",
                group / "tasks" / task_name / "feature_expression_matrix.npy",
                group / "splits" / task_name / "test_indices_test_only.pkl",
                group / "splits" / task_name / "set_info.pkl",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing exp34 artifacts: {missing}")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 mismatch")
    meta = load_json(group / "global_meta.json")
    if len(meta["pert_index"]) != EXPECTED_SOURCE_PERT_COUNT + EXPECTED_DRUG_COUNT:
        raise ValueError("extended perturbation axis size mismatch")
    drug = np.asarray(load_pickle(group / "derived/drug_embedding_morgan_2048.pkl")["embedding_matrix"])
    pdi = safe_np_load(group / "derived/pdi_matrix.npy", mmap_mode="r")
    ddi = safe_np_load(group / "derived/ddi_matrix.npy", mmap_mode="r")
    graph = safe_np_load(
        group / "graph_cache/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy",
        mmap_mode="r",
    )
    expected_rows = EXPECTED_SOURCE_PERT_COUNT + EXPECTED_DRUG_COUNT
    if drug.shape != (expected_rows, 2048):
        raise ValueError(f"drug embedding shape mismatch: {drug.shape}")
    if pdi.shape != (expected_rows, len(meta["protein_index"])):
        raise ValueError(f"PDI shape mismatch: {pdi.shape}")
    if ddi.shape != (expected_rows, expected_rows):
        raise ValueError(f"DDI shape mismatch: {ddi.shape}")
    if graph.shape != (expected_rows, 774):
        raise ValueError(f"graph shape mismatch: {graph.shape}")
    if np.count_nonzero(np.asarray(pdi[-EXPECTED_DRUG_COUNT:])) != 0:
        raise ValueError("new PDI rows are not zero")
    if not np.isfinite(np.asarray(graph[-EXPECTED_DRUG_COUNT:])).all():
        raise ValueError("new graph rows are non-finite")
    result = {
        "status": "ok",
        "output_root": str(output_root.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "pert_count": expected_rows,
        "task_count": len(TASKS),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build(args: argparse.Namespace) -> None:
    input_path = Path(args.input_csv)
    source_root = Path(args.source_training_ready_root)
    output_root = Path(args.output_root)
    checkpoint = Path(args.checkpoint)
    source_graph_path = Path(args.source_graph)
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"output root is not empty; use --force: {output_root}")
    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    staging_root = output_root.with_name(f".{output_root.name}.building-{os.getpid()}")
    if staging_root.exists():
        raise FileExistsError(staging_root)

    frame = read_and_validate_input(input_path)
    source_group = source_root / DATASET_GROUP
    source_meta = load_json(source_group / "global_meta.json")
    manifest = checkpoint_manifest(checkpoint)
    cells = frame[["input_row_index", "Cell_norm"]].drop_duplicates("Cell_norm").sort_values(
        "input_row_index"
    )["Cell_norm"].astype(str).tolist()
    controls, source_expression, ordered_indices, ordered_uniprot = select_fixed_controls(
        manifest=manifest, cells=cells, meta=source_meta
    )
    query_drugs = (
        frame[["drug_name", "drug_name2", "drug_name_norm", "smiles"]]
        .drop_duplicates("drug_name_norm")
        .reset_index(drop=True)
    )
    query_drugs["pert_id"] = query_drugs["smiles"].map(exp34_pert_id)
    if query_drugs["pert_id"].nunique() != EXPECTED_DRUG_COUNT:
        raise ValueError("query drug structures do not resolve to two unique OOD IDs")
    similarity_audit = audit_drug_similarity(
        query_drugs=query_drugs, meta=source_meta, threshold=float(args.similarity_threshold)
    )
    query_drugs = query_drugs.merge(
        similarity_audit[["pert_id", "maximum_morgan_tanimoto"]], on="pert_id", validate="one_to_one"
    )
    if (similarity_audit["threshold_candidate_count"] != 0).any():
        print("[exp34] warning: a query drug crossed the historical similarity threshold")
    meta = extend_meta(
        source_meta=source_meta, query_drugs=query_drugs, similarity_audit=similarity_audit
    )

    try:
        group = staging_root / DATASET_GROUP
        derived = group / "derived"
        graph_cache = group / "graph_cache"
        derived.mkdir(parents=True, exist_ok=False)
        graph_cache.mkdir(parents=True, exist_ok=False)
        dump_json(group / "global_meta.json", meta)
        similarity_audit.to_csv(group / "exp34_update0819_drug_similarity_audit.csv", index=False)
        query_drugs.to_csv(group / "exp34_update0819_drug_registry.csv", index=False)
        controls.to_csv(group / "exp34_update0819_fixed_control_audit.csv", index=False)

        aliases: dict[str, Any] = {}
        for name in (
            "ppi_matrix.npy",
            "ppi_matrix.meta.json",
            "protein_embedding_esm.pkl",
            "protein_embedding_esm.meta.json",
            "cell_llm_embedding_qwen3_4096.npz",
            "cell_llm_embedding_qwen3_4096.json",
            "cell_type_llm_embedding_qwen3_4096_v2.npz",
            "cell_type_llm_embedding_qwen3_4096_v2.json",
        ):
            source = source_group / "derived" / name
            if source.exists():
                aliases[name] = symlink_alias(source, derived / name)

        drug_summary = write_extended_drug_embedding(
            source_path=source_group / "derived/drug_embedding_morgan_2048.pkl",
            output_path=derived / "drug_embedding_morgan_2048.pkl",
            source_meta=source_meta,
            meta=meta,
        )
        ddi_summary = write_extended_ddi(
            source_path=source_group / "derived/ddi_matrix.npy",
            output_path=derived / "ddi_matrix.npy",
            source_meta=source_meta,
            meta=meta,
        )
        pdi_summary = write_extended_pdi(
            source_path=source_group / "derived/pdi_matrix.npy",
            output_path=derived / "pdi_matrix.npy",
            total_rows=len(meta["pert_index"]),
        )
        graph_path = graph_cache / "ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy"
        graph_summary = write_extended_graph_features(
            source_graph_path=source_graph_path,
            output_path=graph_path,
            ppi_path=derived / "ppi_matrix.npy",
            source_pdi_path=source_group / "derived/pdi_matrix.npy",
            source_ddi_path=source_group / "derived/ddi_matrix.npy",
            extended_pdi_path=derived / "pdi_matrix.npy",
            extended_ddi_path=derived / "ddi_matrix.npy",
            protein_embedding_path=derived / "protein_embedding_esm.pkl",
            source_drug_embedding_path=source_group / "derived/drug_embedding_morgan_2048.pkl",
            extended_drug_embedding_path=derived / "drug_embedding_morgan_2048.pkl",
            final_output_root=output_root,
            staging_root=staging_root,
        )

        task_summaries: dict[str, Any] = {}
        for scenario, task_name in TASKS.items():
            task_summaries[scenario] = build_task(
                scenario=scenario,
                task_name=task_name,
                input_frame=frame,
                query_drugs=query_drugs,
                similarity_audit=similarity_audit,
                controls=controls,
                source_expression=source_expression,
                meta=meta,
                ordered_indices=ordered_indices,
                ordered_uniprot=ordered_uniprot,
                task_dir=group / "tasks" / task_name,
                split_dir=group / "splits" / task_name,
            )

        summary = {
            "generated_at": iso_now(),
            "experiment_name": EXPERIMENT_NAME,
            "input_csv": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "source_training_ready_root": str(source_root.resolve()),
            "output_root": str(output_root.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "source_graph": str(source_graph_path.resolve()),
            "fixed_settings": {
                "all_query_drugs_ood": True,
                "pert_time": 24,
                "pert_dose1": 10,
                "pert_dose2": 10,
                "single_drug_same_two_slots": True,
                "control_policy": "minimum checkpoint feature_row_index per Cell",
                "similarity_threshold": float(args.similarity_threshold),
                "similar_drug_transfer_scope": "target_protein_list only",
                "new_pdi_rows": "all zero",
                "mechanism_targets": MECHANISM_TARGETS,
                "PPIA_policy": "recorded as tri-complex cofactor; excluded from target list",
            },
            "counts": {
                "input_rows": len(frame),
                "cells": frame["Cell_norm"].nunique(),
                "drugs": len(query_drugs),
                "source_perturbations": len(source_meta["pert_index"]),
                "extended_perturbations": len(meta["pert_index"]),
            },
            "readonly_aliases": aliases,
            "drug_embedding": drug_summary,
            "pdi_matrix": pdi_summary,
            "ddi_matrix": ddi_summary,
            "graph_features": graph_summary,
            "tasks": task_summaries,
            "acceptance_checks": {
                "all_cells_known": True,
                "all_cells_checkpoint_train_seen": True,
                "source_numeric_blocks_unchanged": True,
                "current_drugs_have_no_similarity_candidate_at_threshold": bool(
                    (similarity_audit["threshold_candidate_count"] == 0).all()
                ),
            },
        }
        dump_json(group / "exp34_update0819_ood_build_summary.json", summary)
        os.replace(staging_root, output_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    preflight_existing(output_root, checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--source-training-ready-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-graph", default=str(DEFAULT_SOURCE_GRAPH))
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.similarity_threshold <= 1.0:
        raise ValueError("--similarity-threshold must be in [0,1]")
    if args.preflight_only:
        preflight_existing(Path(args.output_root), Path(args.checkpoint))
        return
    build(args)


if __name__ == "__main__":
    main()
