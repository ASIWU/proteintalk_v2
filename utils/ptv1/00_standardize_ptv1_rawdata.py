#!/usr/bin/env python3
"""Build isolated PTV1 stage-1 standardized artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _shared import REPO_ROOT, dump_json, iso_now, load_json, load_repo_module


std = load_repo_module("utils/00_standardize_rawdata.py", "ptv_stage1_shared")

RAW_ROOT = REPO_ROOT / "data" / "rawdata"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "standardized"
DEFAULT_PTV3_META_CANDIDATES = (
    REPO_ROOT / "data" / "training_ready" / "ptv3" / "global_meta.json",
    REPO_ROOT / "data" / "standardized" / "ptv3" / "global_meta.json",
)
PTV1_CELL_TYPE = "BREAST"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_id(value: object) -> str:
    text = std.normalize_free_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def choose_ptv3_meta(explicit_path: Path | None) -> Path:
    candidates = (explicit_path,) if explicit_path is not None else DEFAULT_PTV3_META_CANDIDATES
    for path in candidates:
        if path is not None and path.exists():
            return path
    checked = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"could not find PTV3 metadata for PTV1 extra lookup; checked {checked}")


def unique_text_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = std.normalize_free_text(value)
        if not text or text in result:
            continue
        result.append(text)
    return result


def build_ptv3_lookup_maps(meta_path: Path) -> tuple[dict[str, str], dict[str, list[str]], str]:
    meta = load_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"PTV3 meta must be a JSON object: {meta_path}")

    smiles_map = {
        normalize_id(pert_id): std.normalize_free_text(smiles)
        for pert_id, smiles in dict(meta.get("pertid_to_smiles", {})).items()
        if normalize_id(pert_id)
    }

    raw_target_map = dict(meta.get("pertid_to_target_uniprot_list", {}))
    target_source = "pertid_to_target_uniprot_list"
    if not raw_target_map:
        raw_target_map = dict(meta.get("pertid_to_target_protein_list", {}))
        target_source = "pertid_to_target_protein_list"

    target_map: dict[str, list[str]] = {}
    for pert_id, targets in raw_target_map.items():
        normalized = normalize_id(pert_id)
        values = unique_text_list(targets)
        if values and all(not value.isdigit() for value in values):
            target_map[normalized] = values
        else:
            target_map.setdefault(normalized, [])
    return smiles_map, target_map, target_source


def lookup(mapping: dict[str, Any], pert_id: str, default: Any) -> Any:
    candidates = [pert_id, pert_id.lstrip("#")]
    if not pert_id.startswith("#"):
        candidates.append(f"#{pert_id}")
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return default


def standardize_ptv1_extra_singledrug(
    task_dir: Path,
    *,
    main_ptv1_info: pd.DataFrame,
    ptv3_meta_path: Path,
) -> Any:
    raw_path = RAW_ROOT / "ptv1_extra_singledrug" / "kept_samples_drugfilter_edit.csv"
    raw = pd.read_csv(raw_path, low_memory=False)
    required = {"experiment_type", "cellline", "cid", "drug_id", "new_pheno"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{raw_path} is missing required columns: {missing}")

    raw = raw.copy()
    raw["sample_id"] = std.clean_nullable_string(raw["experiment_type"])
    raw["cellline_clean"] = std.clean_nullable_string(raw["cellline"])
    raw["drug_id_clean"] = raw["drug_id"].map(normalize_id)
    raw["new_pheno_clean"] = std.clean_nullable_string(raw["new_pheno"])
    if raw["sample_id"].eq("").any():
        raise ValueError("ptv1_extra_singledrug contains blank experiment_type values")

    smiles_map, target_map, target_source = build_ptv3_lookup_maps(ptv3_meta_path)

    standard = std.default_standard_frame(raw["sample_id"])
    standard["machineID_new"] = ""
    standard["Cell_plate"] = raw["cellline_clean"]
    standard["Cell"] = raw["cellline_clean"]
    standard["cell_type"] = PTV1_CELL_TYPE
    standard["pert_id1"] = raw["drug_id_clean"]
    standard["pert_id2"] = standard["pert_id1"]
    standard["batch"] = "no"
    standard["pert_time"] = np.nan
    standard["pert_dose1"] = np.nan
    standard["pert_dose2"] = np.nan
    standard["PRISM1st_label_total"] = ""
    standard["PRISM2nd_label_total"] = raw["new_pheno_clean"]
    standard["instrument"] = ""
    standard["cell_pertid_time"] = ""
    standard["drugname"] = ""
    standard["smiles"] = standard["pert_id1"].map(lambda value: lookup(smiles_map, value, ""))
    standard["target_protein_list"] = standard["pert_id1"].map(lambda value: lookup(target_map, value, []))
    standard["control"] = ""
    standard["synergy"] = np.nan
    standard["data_split"] = "test"
    standard["raw_experiment_type"] = raw["sample_id"]
    standard["raw_cellline"] = raw["cellline_clean"]
    standard["raw_cid"] = raw["cid"].map(normalize_id)
    standard["raw_drug_id"] = raw["drug_id_clean"]
    standard["source_file_prediction"] = str(raw_path.relative_to(REPO_ROOT))
    standard["source_ptv3_meta"] = str(ptv3_meta_path)
    standard["expression_available"] = False

    standard = std.match_ptv1_extra_controls(main_ptv1_info, standard)
    standard = std.jsonize_target_columns(standard, ("target_protein_list",))
    standard = std.ensure_standard_column_order(standard)
    std.validate_unique_sample_ids(standard, "ptv1_extra_singledrug")

    info_out = task_dir / "info.csv"
    standard.to_csv(info_out, index=False)
    expression = std.write_empty_expression_outputs(
        task_name="ptv1_extra_singledrug",
        info_df=standard,
        output_dir=task_dir,
        reason="PTV1 extra single-drug file contains labels only; matched PTV1 AIVC controls are appended during stage 2.",
    )

    pert_ids = sorted({pert_id for pert_id in standard["pert_id1"].astype(str).tolist() if pert_id})
    pert_smiles_map = {pert_id: lookup(smiles_map, pert_id, "") for pert_id in pert_ids}
    pert_target_map = {pert_id: lookup(target_map, pert_id, []) for pert_id in pert_ids}

    audit = {
        "raw_files": [str(raw_path.relative_to(REPO_ROOT)), str(ptv3_meta_path)],
        "category": "ptv1_extra_singledrug",
        "table_kinds": {
            str(raw_path.relative_to(REPO_ROOT)): "kept_ptv1_extra_single_drug_label_table",
            str(ptv3_meta_path): "ptv3_pert_smiles_and_target_lookup",
        },
        "column_mapping": {
            "sample_id": "experiment_type",
            "Cell_plate": "cellline",
            "Cell": "cellline",
            "cell_type": f"constant {PTV1_CELL_TYPE}; PTV1 breast cancer lineage label",
            "pert_id1": "drug_id, used directly as the PTV3 pert id",
            "pert_id2": "copied from pert_id1",
            "PRISM2nd_label_total": "new_pheno",
            "smiles": "PTV3 pertid_to_smiles",
            "target_protein_list": f"PTV3 {target_source}, stored as UniProt IDs before PTV1 index encoding",
            "control": "matched to PTV1 AIVC controls by exact normalized cellline -> Cell_plate",
        },
        "issues": [
            {"kind": "missing_expression_matrix", "count": int(len(standard))},
            {"kind": "unmatched_control", "count": int(standard["control"].astype("string").fillna("").eq("").sum())},
            {"kind": "missing_ptv3_smiles", "count": int(standard["smiles"].astype("string").fillna("").eq("").sum())},
            {"kind": "empty_target_list", "count": int(standard["target_protein_list"].astype(str).eq("[]").sum())},
        ],
        "special_rules": [
            "PTV1 extra single-drug uses the current kept_samples_drugfilter_edit.csv source file.",
            "drug_id is not remapped through the old E115 table; it is used directly as a PTV3 pert id.",
            "The task has no perturbation proteome matrix; PTV1 AIVC baseline controls are joined in stage 2.",
            "cell_type is fixed to BREAST because PTV1 contains breast cancer cell-line assays without a reliable row-level subtype field.",
        ],
    }

    return std.TaskResult(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ptv3-meta", type=Path, default=None, help="PTV3 meta JSON used for PTV1 extra drug lookup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    ptv1_root = ensure_dir(output_root / "ptv1")
    tasks_root = ensure_dir(ptv1_root / "tasks")
    ptv3_meta_path = choose_ptv3_meta(args.ptv3_meta)

    main_task_dir = ensure_dir(tasks_root / "ptv1_aivc")
    main_result = std.standardize_ptv1(main_task_dir)
    main_info = pd.read_csv(main_result.info_path, low_memory=False)

    extra_task_dir = ensure_dir(tasks_root / "ptv1_extra_singledrug")
    extra_result = standardize_ptv1_extra_singledrug(
        extra_task_dir,
        main_ptv1_info=main_info,
        ptv3_meta_path=ptv3_meta_path,
    )

    results = [main_result, extra_result]
    std.apply_canonical_smiles(results)
    std.build_global_meta("ptv1", results, ptv1_root)

    task_payloads = {
        result.task_name: {
            "task_name": result.task_name,
            "dataset_group": result.dataset_group,
            "info_path": result.info_path,
            "expression": result.expression,
            "sample_count": result.sample_count,
            "protein_count": result.protein_count,
            "audit": result.audit,
        }
        for result in results
    }
    audit = {
        "generated_at": iso_now(),
        "source_document": str((REPO_ROOT / "docs" / "Data_Process_ptv1.md").relative_to(REPO_ROOT)),
        "dataset_groups": {
            "ptv1": {
                "global_meta_path": str(ptv1_root / "global_meta.json"),
                "task_names": [result.task_name for result in results],
            }
        },
        "tasks": task_payloads,
    }
    dump_json(ptv1_root / "file_audit.json", audit)

    root_audit_path = output_root / "file_audit.json"
    if root_audit_path.exists():
        root_audit = load_json(root_audit_path)
        if not isinstance(root_audit, dict):
            root_audit = {}
    else:
        root_audit = {}
    root_audit["generated_at"] = iso_now()
    root_audit.setdefault("source_document", str((REPO_ROOT / "docs" / "Data_Process_ptv1.md").relative_to(REPO_ROOT)))
    root_audit.setdefault("dataset_groups", {})
    root_audit.setdefault("tasks", {})
    root_audit["dataset_groups"]["ptv1"] = audit["dataset_groups"]["ptv1"]
    root_audit["tasks"].update(task_payloads)
    dump_json(root_audit_path, root_audit)

    print(f"[done] wrote PTV1 standardized artifacts under {ptv1_root}")


if __name__ == "__main__":
    main()
