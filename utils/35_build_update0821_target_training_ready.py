#!/usr/bin/env python3
"""Build the Exp35 update-0821 CSV-driven manual-target inference artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER_PATH = REPO_ROOT / "utils/34_build_update0819_ood_training_ready.py"
SPEC = importlib.util.spec_from_file_location("exp34_ood_builder_base", BASE_BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load base OOD builder: {BASE_BUILDER_PATH}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

EXPERIMENT_NAME = "exp35_update0821_target_epoch2"
TASK_NAME = "ptv3_exp35_update0821_manual_target"
SCENARIO = "manual_target"
BUILDER_SCENARIO = "target_mechanism"
ARTIFACT_STEM = "exp35_update0821_target"
SUMMARY_NAME = f"{ARTIFACT_STEM}_build_summary.json"
DEFAULT_INPUT = (
    REPO_ROOT / "data/rawdata/update_0821/260820ptv_drug_cell_predict_target.csv"
)
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data/training_ready"
DEFAULT_OUTPUT_ROOT = Path("/tmp/proteintalk_exp35_update0821_target_runtime")
DEFAULT_CHECKPOINT = base.DEFAULT_CHECKPOINT
DEFAULT_SOURCE_GRAPH = base.DEFAULT_SOURCE_GRAPH
TARGET_COLUMN = "target"
TARGET_LIMIT = 32
UNIPROT_ACCESSION = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])"
)


def parse_target_cell(value: object) -> list[str]:
    text = base.clean_text(value)
    if not text:
        raise ValueError("target contains an empty value")
    items = [item.strip().upper() for item in text.split(";") if item.strip()]
    if not items:
        raise ValueError("target contains no UniProt accessions")
    invalid = sorted({item for item in items if UNIPROT_ACCESSION.fullmatch(item) is None})
    if invalid:
        raise ValueError(f"target contains invalid UniProt accessions: {invalid}")
    unique = sorted(set(items))
    if len(unique) > TARGET_LIMIT:
        raise ValueError(f"target contains {len(unique)} proteins; maximum is {TARGET_LIMIT}")
    return unique


def read_and_validate_input(path: Path) -> pd.DataFrame:
    frame = base.read_and_validate_input(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"input is missing required column: {TARGET_COLUMN}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("target contains missing values")
    key = ["cell_in_ptvdrug", "drug_name"]
    if frame.duplicated(key).any():
        duplicates = frame.loc[frame.duplicated(key, keep=False), key].to_dict("records")
        raise ValueError(f"input contains duplicate cell-drug keys: {duplicates}")
    frame = frame.copy()
    frame["manual_target_uniprot"] = frame[TARGET_COLUMN].map(parse_target_cell).map(
        lambda values: json.dumps(values, separators=(",", ":"))
    )
    for drug_name, group in frame.groupby("drug_name_norm", sort=False):
        target_sets = group["manual_target_uniprot"].unique().tolist()
        if len(target_sets) != 1:
            raise ValueError(f"{drug_name}: conflicting target sets across cells")
    return frame


def target_contract(
    frame: pd.DataFrame, meta: dict[str, Any]
) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    protein_index = {str(key): int(value) for key, value in meta["protein_index"].items()}
    uniprot_by_drug: dict[str, list[str]] = {}
    index_by_drug: dict[str, list[int]] = {}
    for drug_name, group in frame.groupby("drug_name_norm", sort=False):
        values = json.loads(str(group["manual_target_uniprot"].iloc[0]))
        missing = sorted(item for item in values if item not in protein_index)
        if missing:
            raise ValueError(f"{drug_name}: targets absent from protein axis: {missing}")
        ordered = sorted(values, key=lambda item: protein_index[item])
        uniprot_by_drug[str(drug_name)] = ordered
        index_by_drug[str(drug_name)] = [protein_index[item] for item in ordered]
    return uniprot_by_drug, index_by_drug


def rewrite_task_identity(task_dir: Path) -> None:
    table_path = task_dir / "feature_table.parquet"
    table = pd.read_parquet(table_path)
    old_ids = table["sample_id"].astype(str).tolist()
    new_ids: list[str] = []
    for is_control, sample_id, input_index in zip(
        table["is_control"].astype(bool),
        old_ids,
        table["input_row_index"],
        strict=True,
    ):
        if is_control:
            new_ids.append(sample_id.replace("exp34::", "exp35::", 1))
        else:
            new_ids.append(f"exp35::{SCENARIO}::{int(input_index):02d}")
    id_map = dict(zip(old_ids, new_ids, strict=True))
    table["sample_id"] = new_ids
    table["control"] = table["control"].astype(str).map(id_map)
    query = ~table["is_control"].astype(bool)
    table.loc[query, "target_scenario"] = SCENARIO
    table.loc[query, "source_row_role"] = "exp35_update0821_manual_target_query"
    table.loc[~query, "source_row_role"] = "exp35_fixed_first_checkpoint_control"
    table.to_parquet(table_path, index=False)
    table.to_csv(task_dir / "feature_table.csv", index=False)
    table.to_csv(task_dir / "processed.csv", index=False)
    for prefix in ("feature", "processed"):
        base.dump_json(
            task_dir / f"{prefix}_sample_ids.json", table["sample_id"].astype(str).tolist()
        )


def preflight_existing(output_root: Path, checkpoint: Path) -> dict[str, Any]:
    group = output_root / base.DATASET_GROUP
    summary_path = group / SUMMARY_NAME
    task_dir = group / "tasks" / TASK_NAME
    split_dir = group / "splits" / TASK_NAME
    required = [
        group / "global_meta.json",
        group / "derived/drug_embedding_morgan_2048.pkl",
        group / "derived/pdi_matrix.npy",
        group / "derived/ddi_matrix.npy",
        group / "graph_cache/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy",
        summary_path,
        task_dir / "feature_table.parquet",
        task_dir / "feature_expression_matrix.npy",
        split_dir / "test_indices_test_only.pkl",
        split_dir / "set_info.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Exp35 artifacts: {missing}")
    if base.sha256_file(checkpoint) != base.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 mismatch")
    meta = base.load_json(group / "global_meta.json")
    expected_rows = base.EXPECTED_SOURCE_PERT_COUNT + base.EXPECTED_DRUG_COUNT
    if len(meta["pert_index"]) != expected_rows:
        raise ValueError("extended perturbation axis size mismatch")
    drug = np.asarray(
        base.load_pickle(group / "derived/drug_embedding_morgan_2048.pkl")["embedding_matrix"]
    )
    pdi = base.safe_np_load(group / "derived/pdi_matrix.npy", mmap_mode="r")
    ddi = base.safe_np_load(group / "derived/ddi_matrix.npy", mmap_mode="r")
    graph = base.safe_np_load(
        group / "graph_cache/ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy", mmap_mode="r"
    )
    if drug.shape != (expected_rows, 2048):
        raise ValueError(f"drug embedding shape mismatch: {drug.shape}")
    if pdi.shape != (expected_rows, len(meta["protein_index"])):
        raise ValueError(f"PDI shape mismatch: {pdi.shape}")
    if ddi.shape != (expected_rows, expected_rows):
        raise ValueError(f"DDI shape mismatch: {ddi.shape}")
    if graph.shape != (expected_rows, 774):
        raise ValueError(f"graph shape mismatch: {graph.shape}")
    if np.count_nonzero(np.asarray(pdi[-base.EXPECTED_DRUG_COUNT :])) != 0:
        raise ValueError("new PDI rows are not zero")
    table = pd.read_parquet(task_dir / "feature_table.parquet")
    query = table.loc[~table["is_control"].astype(bool)]
    if len(query) != base.EXPECTED_INPUT_ROWS:
        raise ValueError("manual-target task query count mismatch")
    if not query["target_scenario"].astype(str).eq(SCENARIO).all():
        raise ValueError("manual-target scenario mismatch")
    expected_targets = {"[1374,1375,1376]": 7, "[1376]": 7}
    if query["target_protein_list"].value_counts().to_dict() != expected_targets:
        raise ValueError("manual target index counts mismatch")
    summary = base.load_json(summary_path)
    summary_input = Path(str(summary.get("input_csv", "")))
    if not summary_input.is_file():
        raise ValueError(f"build summary input CSV is unavailable: {summary_input}")
    if summary.get("input_sha256") != base.sha256_file(summary_input):
        raise ValueError("build summary input SHA mismatch")
    result = {
        "status": "ok",
        "output_root": str(output_root.resolve()),
        "checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA256,
        "pert_count": expected_rows,
        "task_count": 1,
        "query_count": len(query),
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
    source_group = source_root / base.DATASET_GROUP
    source_meta = base.load_json(source_group / "global_meta.json")
    target_uniprot, target_indices = target_contract(frame, source_meta)
    manifest = base.checkpoint_manifest(checkpoint)
    cells = frame[["input_row_index", "Cell_norm"]].drop_duplicates("Cell_norm").sort_values(
        "input_row_index"
    )["Cell_norm"].astype(str).tolist()
    controls, source_expression, ordered_indices, ordered_uniprot = base.select_fixed_controls(
        manifest=manifest, cells=cells, meta=source_meta
    )
    query_drugs = (
        frame[["drug_name", "drug_name2", "drug_name_norm", "smiles"]]
        .drop_duplicates("drug_name_norm")
        .reset_index(drop=True)
    )
    query_drugs["pert_id"] = query_drugs["smiles"].map(base.exp34_pert_id)
    if query_drugs["pert_id"].nunique() != base.EXPECTED_DRUG_COUNT:
        raise ValueError("query structures do not resolve to two unique OOD IDs")
    similarity_audit = base.audit_drug_similarity(
        query_drugs=query_drugs, meta=source_meta, threshold=float(args.similarity_threshold)
    )
    query_drugs = query_drugs.merge(
        similarity_audit[["pert_id", "maximum_morgan_tanimoto"]],
        on="pert_id",
        validate="one_to_one",
    )

    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.MECHANISM_TARGETS = target_uniprot
    base.TASKS = {SCENARIO: TASK_NAME}
    meta = base.extend_meta(
        source_meta=source_meta, query_drugs=query_drugs, similarity_audit=similarity_audit
    )
    drug_to_pert = query_drugs.set_index("drug_name_norm")["pert_id"].astype(str).to_dict()
    for drug_name, pert_id in drug_to_pert.items():
        meta["pertid_to_target_protein_list"][pert_id] = target_indices[drug_name]
        meta["pertid_to_target_uniprot_list"][pert_id] = target_uniprot[drug_name]
        meta["pertid_to_missing_target_uniprot"][pert_id] = []
    meta.pop("exp34_ood_extension", None)
    meta["exp35_update0821_manual_target_extension"] = {
        "experiment_name": EXPERIMENT_NAME,
        "source_pert_count": base.EXPECTED_SOURCE_PERT_COUNT,
        "extended_pert_count": len(meta["pert_index"]),
        "added_pert_ids": query_drugs["pert_id"].astype(str).tolist(),
        "manual_target_uniprot_by_drug": target_uniprot,
        "manual_target_index_by_drug": target_indices,
        "target_source": str(input_path.resolve()),
        "pdi_policy": "new OOD rows remain zero; manual targets enter through target tokens",
        "task_name": TASK_NAME,
    }

    try:
        group = staging_root / base.DATASET_GROUP
        derived = group / "derived"
        graph_cache = group / "graph_cache"
        derived.mkdir(parents=True, exist_ok=False)
        graph_cache.mkdir(parents=True, exist_ok=False)
        base.dump_json(group / "global_meta.json", meta)
        similarity_audit.to_csv(group / f"{ARTIFACT_STEM}_drug_similarity_audit.csv", index=False)
        query_drugs.to_csv(group / f"{ARTIFACT_STEM}_drug_registry.csv", index=False)
        controls.to_csv(group / f"{ARTIFACT_STEM}_fixed_control_audit.csv", index=False)

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
                aliases[name] = base.symlink_alias(source, derived / name)

        drug_summary = base.write_extended_drug_embedding(
            source_path=source_group / "derived/drug_embedding_morgan_2048.pkl",
            output_path=derived / "drug_embedding_morgan_2048.pkl",
            source_meta=source_meta,
            meta=meta,
        )
        ddi_summary = base.write_extended_ddi(
            source_path=source_group / "derived/ddi_matrix.npy",
            output_path=derived / "ddi_matrix.npy",
            source_meta=source_meta,
            meta=meta,
        )
        pdi_summary = base.write_extended_pdi(
            source_path=source_group / "derived/pdi_matrix.npy",
            output_path=derived / "pdi_matrix.npy",
            total_rows=len(meta["pert_index"]),
        )
        graph_path = graph_cache / "ptv3_ppi_pdi_ddi_dim128_seed17_structrp.npy"
        graph_summary = base.write_extended_graph_features(
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
        task_dir = group / "tasks" / TASK_NAME
        task_summary = base.build_task(
            scenario=BUILDER_SCENARIO,
            task_name=TASK_NAME,
            input_frame=frame,
            query_drugs=query_drugs,
            similarity_audit=similarity_audit,
            controls=controls,
            source_expression=source_expression,
            meta=meta,
            ordered_indices=ordered_indices,
            ordered_uniprot=ordered_uniprot,
            task_dir=task_dir,
            split_dir=group / "splits" / TASK_NAME,
        )
        rewrite_task_identity(task_dir)
        task_summary["scenario"] = SCENARIO
        summary = {
            "generated_at": base.iso_now(),
            "experiment_name": EXPERIMENT_NAME,
            "input_csv": str(input_path.resolve()),
            "input_sha256": base.sha256_file(input_path),
            "target_column": TARGET_COLUMN,
            "target_uniprot_by_drug": target_uniprot,
            "target_index_by_drug": target_indices,
            "source_training_ready_root": str(source_root.resolve()),
            "output_root": str(output_root.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA256,
            "source_graph": str(source_graph_path.resolve()),
            "fixed_settings": {
                "pert_time": 24,
                "pert_dose1": 10,
                "pert_dose2": 10,
                "single_drug_same_two_slots": True,
                "control_policy": "minimum checkpoint feature_row_index per Cell",
                "manual_target_only": True,
                "new_pdi_rows": "all zero",
                "target_protein_max_length": TARGET_LIMIT,
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
            "tasks": {SCENARIO: task_summary},
            "acceptance_checks": {
                "all_cells_known": True,
                "all_cells_checkpoint_train_seen": True,
                "source_numeric_blocks_unchanged": True,
                "manual_targets_present_in_protein_axis": True,
                "manual_targets_written_to_meta_and_task": True,
            },
        }
        base.dump_json(group / SUMMARY_NAME, summary)
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
    parser.add_argument("--similarity-threshold", type=float, default=base.SIMILARITY_THRESHOLD)
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
