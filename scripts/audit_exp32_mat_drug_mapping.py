#!/usr/bin/env python3
"""Audit Exp32 SMILES origins and mat1-4 extra-single drug-ID mappings."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARDIZED_ROOT = REPO_ROOT / "data/standardized/ptv3/tasks"
MAIN_RAW_PATH = (
    REPO_ROOT
    / "data/rawdata/update_0623"
    / "260513ptv3_EGH_28602sampinfo_with_smiles_check_prism1_label_add_prism2_label_add_machine_details.csv"
)
DEFAULT_SCOPE = REPO_ROOT / "data/training_ready_exp32_organoid/ptv3/exp32_organoid_drug_scope.csv"
DEFAULT_META = REPO_ROOT / "data/training_ready/ptv3/global_meta.json"
DEFAULT_OUTPUT_PREFIX = (
    REPO_ROOT
    / "outputs/2026-07/2026-07-22"
    / "20260722_exp32_mat1_4_drug_mapping_audit"
)

TASK_SPECS = [
    ("ptv3_main_singledrug", "main_single"),
    ("ptv3_main_doubledrug", "main_double"),
    ("ptv3_extra_singledrug_mat1_480_faims", "extra_single"),
    ("ptv3_extra_singledrug_mat1_qe", "extra_single"),
    ("ptv3_extra_singledrug_mat2_480_faims", "extra_single"),
    ("ptv3_extra_singledrug_mat2_qe", "extra_single"),
    ("ptv3_extra_singledrug_mat3_qe", "extra_single"),
    ("ptv3_extra_singledrug_mat4_qe", "extra_single"),
    ("ptv3_extra_doubledrug_guomics", "extra_double_guomics"),
    ("ptv3_extra_doubledrug_nc", "extra_double_nc"),
    ("ptv3_extra_doubledrug_nature", "extra_double_nature"),
]

MAT_TASKS = [task for task, kind in TASK_SPECS if kind == "extra_single"]


def clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).lower())


def normalize_identifier_fragment(value: object) -> str:
    text = normalize_name(value)
    if text:
        return text
    raw = clean(value)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12] if raw else ""


def json_values(values: Any) -> str:
    result: list[str] = []
    for value in values:
        text = clean(value)
        if text and text not in result:
            result.append(text)
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


STANDARD_NEEDED_COLUMNS = {
    "sample_id",
    "source_file_info",
    "source_row_index",
    "pert_id1",
    "pert_id2",
    "pert_id_resolution",
    "pert_id1_resolution",
    "pert_id2_resolution",
}

RAW_NEEDED_COLUMNS = {
    "sample_id",
    "pert_id",
    "pert_id1",
    "pert_id2",
    "drug_ID",
    "anchor_ID",
    "lib_ID",
    "drugname",
    "synonyms",
    "pert_name",
    "drug_name",
    "Anchor_name",
    "Library_name",
    "anchor_name",
    "library_name",
    "Anchor.Name",
    "Library.Name",
    "smiles",
    "smiles1",
    "smiles2",
    "Smiles_no_chiral",
    "Smiles_with_chiral",
    "Smiles1_no_chiral",
    "Smiles1_with_chiral",
    "Smiles2_no_chiral",
    "Smiles2_with_chiral",
}


def read_aligned_task(task_name: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    standard_path = STANDARDIZED_ROOT / task_name / "info.csv"
    standard = pd.read_csv(
        standard_path,
        usecols=lambda column: column in STANDARD_NEEDED_COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    sources = sorted({clean(value) for value in standard["source_file_info"] if clean(value)})
    if len(sources) != 1:
        raise ValueError(f"{task_name}: expected exactly one source_file_info, found {sources}")
    raw_path = REPO_ROOT / sources[0]
    raw = pd.read_csv(
        raw_path,
        usecols=lambda column: column in RAW_NEEDED_COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    if len(standard) != len(raw):
        raise ValueError(f"{task_name}: standardized/raw row mismatch {len(standard)} != {len(raw)}")
    if "source_row_index" in standard:
        indices = pd.to_numeric(standard["source_row_index"], errors="raise").astype(int).tolist()
        if indices != list(range(len(raw))):
            raise ValueError(f"{task_name}: source_row_index differs from raw row order")
    elif "sample_id" in raw:
        if standard["sample_id"].map(clean).tolist() != raw["sample_id"].map(clean).tolist():
            raise ValueError(f"{task_name}: standardized/raw sample_id order differs")
    return standard, raw, sources[0]


def first_nonempty(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series("", index=frame.index, dtype="string"), pd.Series("", index=frame.index, dtype="string")
    values = frame[available].apply(lambda column: column.map(clean))
    selected = values.replace("", pd.NA).bfill(axis=1).iloc[:, 0].fillna("").astype("string")
    selected_column = pd.Series("", index=frame.index, dtype="string")
    for column in available:
        mask = selected_column.eq("") & values[column].ne("")
        selected_column.loc[mask] = column
    return selected, selected_column


def origin_candidates(
    task_name: str,
    kind: str,
    standard: pd.DataFrame,
    raw: pd.DataFrame,
    raw_source: str,
) -> pd.DataFrame:
    base = pd.DataFrame(index=raw.index)
    base["source_task"] = task_name
    base["source_kind"] = kind
    base["raw_source_file"] = raw_source
    raw_row_indices = raw.index.to_numpy(dtype=np.int64)
    base["raw_row_index_0based"] = raw_row_indices

    if kind in {"main_single", "extra_single"}:
        if kind == "main_single":
            raw_id_column, name_column = "pert_id", "drugname"
            resolution = pd.Series("raw_main_pert_id", index=raw.index)
        else:
            raw_id_column, name_column = "drug_ID", "drug_name"
            resolution = standard["pert_id_resolution"].map(clean)
        selected, selected_column = first_nonempty(raw, ["Smiles_with_chiral", "smiles", "Smiles_no_chiral"])
        base["sequence"] = raw_row_indices
        base["drug_id"] = standard["pert_id1"].map(clean)
        base["raw_drug_id"] = raw[raw_id_column].map(clean)
        base["raw_drug_name"] = raw[name_column].map(clean)
        base["raw_legacy_smiles"] = raw.get("smiles", pd.Series("", index=raw.index)).map(clean)
        base["raw_Smiles_no_chiral"] = raw.get("Smiles_no_chiral", pd.Series("", index=raw.index)).map(clean)
        base["raw_Smiles_with_chiral"] = raw.get("Smiles_with_chiral", pd.Series("", index=raw.index)).map(clean)
        base["selected_raw_column"] = selected_column
        base["selected_smiles"] = selected
        base["id_resolution"] = resolution
        return base

    frames: list[pd.DataFrame] = []
    for side in (1, 2):
        item = base.copy()
        item["sequence"] = raw_row_indices * 2 + (side - 1)
        item["drug_id"] = standard[f"pert_id{side}"].map(clean)
        if kind == "main_double":
            id_column = f"pert_id{side}"
            name_column = "pert_name"
            resolution = pd.Series("raw_main_pert_id", index=raw.index)
            legacy_column = ""
        elif kind == "extra_double_guomics":
            id_column = f"pert_id{side}"
            name_column = "Anchor_name" if side == 1 else "Library_name"
            resolution = standard[f"pert_id{side}_resolution"].map(clean)
            legacy_column = f"smiles{side}"
        elif kind == "extra_double_nc":
            id_column = ""
            name_column = "anchor_name" if side == 1 else "library_name"
            resolution = standard[f"pert_id{side}_resolution"].map(clean)
            legacy_column = f"smiles{side}"
        elif kind == "extra_double_nature":
            id_column = "anchor_ID" if side == 1 else "lib_ID"
            name_column = "Anchor.Name" if side == 1 else "Library.Name"
            resolution = standard[f"pert_id{side}_resolution"].map(clean)
            legacy_column = f"smiles{side}"
        else:
            raise AssertionError(kind)
        with_column = f"Smiles{side}_with_chiral"
        no_column = f"Smiles{side}_no_chiral"
        columns = [with_column, legacy_column, no_column] if legacy_column else [with_column, no_column]
        selected, selected_column = first_nonempty(raw, columns)
        item["raw_drug_id"] = raw[id_column].map(clean) if id_column else ""
        item["raw_drug_name"] = raw[name_column].map(clean)
        item["raw_legacy_smiles"] = raw[legacy_column].map(clean) if legacy_column else ""
        item["raw_Smiles_no_chiral"] = raw[no_column].map(clean)
        item["raw_Smiles_with_chiral"] = raw[with_column].map(clean)
        item["selected_raw_column"] = selected_column
        item["selected_smiles"] = selected
        item["id_resolution"] = resolution
        frames.append(item)
    return pd.concat(frames, ignore_index=True).sort_values("sequence", kind="stable")


def build_exp32_origin(scope: pd.DataFrame, meta: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scope_ids = set(scope["pert_id"].map(clean))
    canonical_origin: dict[str, dict[str, Any]] = {}
    source_inventory: list[dict[str, Any]] = []
    for task_name, kind in TASK_SPECS:
        print(f"[origin] {task_name}", flush=True)
        standard, raw, raw_source = read_aligned_task(task_name)
        source_inventory.append(
            {
                "source_task": task_name,
                "source_kind": kind,
                "raw_source_file": raw_source,
                "raw_rows": int(len(raw)),
                "raw_sha256": sha256_file(REPO_ROOT / raw_source),
            }
        )
        id_columns = [column for column in ("pert_id1", "pert_id2") if column in standard]
        in_scope = pd.Series(False, index=standard.index)
        for column in id_columns:
            in_scope |= standard[column].map(clean).isin(scope_ids)
        standard = standard.loc[in_scope].copy()
        raw = raw.loc[in_scope].copy()
        candidates = origin_candidates(task_name, kind, standard, raw, raw_source)
        candidates = candidates.loc[candidates["drug_id"].isin(scope_ids)].copy()
        # Stage-1 task maps are first-row wins, including an empty first value.
        if kind == "main_double":
            task_first = candidates.loc[candidates["selected_smiles"].map(clean).ne("")].drop_duplicates(
                "drug_id", keep="first"
            )
        else:
            task_first = candidates.drop_duplicates("drug_id", keep="first")
        for row in task_first.to_dict("records"):
            drug_id = clean(row["drug_id"])
            selected = clean(row["selected_smiles"])
            if drug_id and selected and drug_id not in canonical_origin:
                canonical_origin[drug_id] = row
        del standard, raw, candidates, task_first
        gc.collect()

    missing = sorted(scope_ids - set(canonical_origin))
    if missing:
        raise ValueError(f"missing canonical raw origins for Exp32 IDs: {missing[:20]}")

    meta_smiles = {str(key): clean(value) for key, value in meta["pertid_to_smiles"].items()}
    output_rows: list[dict[str, Any]] = []
    for scope_row in scope.to_dict("records"):
        drug_id = clean(scope_row["pert_id"])
        origin = canonical_origin[drug_id]
        model_smiles = meta_smiles.get(drug_id, "")
        if model_smiles != clean(origin["selected_smiles"]):
            raise ValueError(f"{drug_id}: reconstructed raw origin differs from global_meta SMILES")
        output_rows.append(
            {
                "scope_rank": int(scope_row["scope_rank"]),
                "drug_id": drug_id,
                "pert_index": int(scope_row["pert_index"]),
                "exp32_model_smiles": model_smiles,
                "canonical_origin_task": origin["source_task"],
                "canonical_origin_kind": origin["source_kind"],
                "raw_source_file": origin["raw_source_file"],
                "raw_row_index_0based": int(origin["raw_row_index_0based"]),
                "raw_csv_line_number_1based": int(origin["raw_row_index_0based"]) + 2,
                "raw_drug_id": origin["raw_drug_id"],
                "raw_drug_name": origin["raw_drug_name"],
                "raw_legacy_smiles": origin["raw_legacy_smiles"],
                "raw_Smiles_no_chiral": origin["raw_Smiles_no_chiral"],
                "raw_Smiles_with_chiral": origin["raw_Smiles_with_chiral"],
                "selected_raw_column": origin["selected_raw_column"],
                "id_resolution_at_origin": origin["id_resolution"],
                "source_tasks_in_exp32_scope": clean(scope_row["source_tasks"]),
                "model_smiles_exact_origin_match": True,
            }
        )
    return pd.DataFrame(output_rows), source_inventory


def build_main_lookup(raw: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]], set[str]]:
    smiles_sets: dict[str, set[str]] = defaultdict(set)
    name_sets: dict[str, set[str]] = defaultdict(set)
    main_details: dict[str, dict[str, Any]] = {}
    for row_index, row in raw.iterrows():
        pert_id = clean(row.get("pert_id"))
        if not pert_id:
            continue
        for column in ("smiles", "Smiles_no_chiral", "Smiles_with_chiral"):
            value = clean(row.get(column))
            if value:
                smiles_sets[value].add(pert_id)
        for raw_name in (row.get("drugname"), row.get("synonyms")):
            for token in re.split(r"[;|,]+", clean(raw_name)):
                normalized = normalize_name(token)
                if normalized:
                    name_sets[normalized].add(pert_id)
        detail = main_details.setdefault(
            pert_id,
            {
                "main_raw_first_row_index_0based": int(row_index),
                "main_raw_drug_names": [],
                "main_raw_synonyms": [],
                "main_raw_legacy_smiles": [],
                "main_raw_Smiles_no_chiral": [],
                "main_raw_Smiles_with_chiral": [],
            },
        )
        for key, column in (
            ("main_raw_drug_names", "drugname"),
            ("main_raw_synonyms", "synonyms"),
            ("main_raw_legacy_smiles", "smiles"),
            ("main_raw_Smiles_no_chiral", "Smiles_no_chiral"),
            ("main_raw_Smiles_with_chiral", "Smiles_with_chiral"),
        ):
            value = clean(row.get(column))
            if value and value not in detail[key]:
                detail[key].append(value)
    smiles_map = {key: next(iter(values)) for key, values in smiles_sets.items() if len(values) == 1}
    name_map = {key: next(iter(values)) for key, values in name_sets.items() if len(values) == 1}
    return smiles_map, name_map, main_details, set(main_details)


def replay_single_mapping(row: pd.Series, smiles_map: dict[str, str], name_map: dict[str, str]) -> tuple[str, str]:
    for column in ("raw_legacy_smiles", "raw_Smiles_with_chiral", "raw_Smiles_no_chiral"):
        value = clean(row[column])
        if value and value in smiles_map:
            source_name = {
                "raw_legacy_smiles": "smiles",
                "raw_Smiles_with_chiral": "Smiles_with_chiral",
                "raw_Smiles_no_chiral": "Smiles_no_chiral",
            }[column]
            return smiles_map[value], f"existing_by_{source_name}"
    name = normalize_name(row["raw_drug_name"])
    if name and name in name_map:
        return name_map[name], "existing_by_drug_name"
    explicit = normalize_identifier_fragment(row["raw_drug_id"])
    if explicit:
        return f"extid::{explicit}", "unified_by_raw_id"
    selected = clean(row["raw_model_candidate_smiles"])
    if selected:
        return f"extsmiles::{hashlib.sha1(selected.encode('utf-8')).hexdigest()[:12]}", "unified_by_smiles"
    name_slug = normalize_identifier_fragment(row["raw_drug_name"])
    if name_slug:
        return f"extname::{name_slug}", "unified_by_name"
    return "extunk::da39a3ee5e6b", "unified_by_fallback_hash"


def main_name_match_evidence(raw_name: str, detail: dict[str, Any]) -> tuple[str, list[str]]:
    """Classify a normalized name match as a main primary-name or synonym hit."""
    normalized = normalize_name(raw_name)
    if not normalized:
        return "", []
    matched_primary: list[str] = []
    matched_synonyms: list[str] = []
    for value in detail.get("main_raw_drug_names", []):
        for token in re.split(r"[;|,]+", clean(value)):
            token = clean(token)
            if normalize_name(token) == normalized and token not in matched_primary:
                matched_primary.append(token)
    for value in detail.get("main_raw_synonyms", []):
        for token in re.split(r"[;|,]+", clean(value)):
            token = clean(token)
            if normalize_name(token) == normalized and token not in matched_synonyms:
                matched_synonyms.append(token)
    if matched_primary and matched_synonyms:
        source = "primary_name_and_synonym"
    elif matched_primary:
        source = "primary_name"
    elif matched_synonyms:
        source = "synonym"
    else:
        source = ""
    return source, matched_primary + matched_synonyms


def canonical_smiles(smiles: str, *, isomeric: bool) -> str:
    mol = Chem.MolFromSmiles(clean(smiles)) if clean(smiles) else None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric) if mol is not None else ""


def build_mat_unique_records() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task_name in MAT_TASKS:
        print(f"[mat] {task_name}", flush=True)
        standard, raw, raw_source = read_aligned_task(task_name)
        match = re.search(r"_mat([1-4])(?:_|$)", task_name)
        if match is None:
            raise ValueError(f"cannot parse mat number from {task_name}")
        device = "480_FAIMS" if "480_faims" in task_name else "QE"
        selected, _ = first_nonempty(raw, ["Smiles_with_chiral", "smiles", "Smiles_no_chiral"])
        frame = pd.DataFrame(
            {
                "mat": f"mat{match.group(1)}",
                "source_task": task_name,
                "device": device,
                "raw_source_file": raw_source,
                "raw_row_index_0based": np.arange(len(raw), dtype=np.int64),
                "raw_drug_id": raw["drug_ID"].map(clean),
                "raw_drug_name": raw["drug_name"].map(clean),
                "raw_legacy_smiles": raw["smiles"].map(clean),
                "raw_Smiles_no_chiral": raw["Smiles_no_chiral"].map(clean),
                "raw_Smiles_with_chiral": raw["Smiles_with_chiral"].map(clean),
                "raw_model_candidate_smiles": selected,
                "resolved_drug_id": standard["pert_id1"].map(clean),
                "resolution_method": standard["pert_id_resolution"].map(clean),
            }
        )
        frame["raw_occurrence_count"] = 1
        frames.append(
            frame.groupby(
                [
                    "mat",
                    "source_task",
                    "device",
                    "raw_source_file",
                    "raw_drug_id",
                    "raw_drug_name",
                    "raw_legacy_smiles",
                    "raw_Smiles_no_chiral",
                    "raw_Smiles_with_chiral",
                    "raw_model_candidate_smiles",
                    "resolved_drug_id",
                    "resolution_method",
                ],
                dropna=False,
                as_index=False,
            ).agg(
                raw_occurrence_count=("raw_occurrence_count", "sum"),
                first_raw_row_index_0based=("raw_row_index_0based", "min"),
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_mat_audit(meta: dict[str, Any]) -> pd.DataFrame:
    unique_records = build_mat_unique_records()
    print(f"[mat-audit] unique task/raw records={len(unique_records)}", flush=True)
    gc.collect()
    main_raw = pd.read_csv(
        MAIN_RAW_PATH,
        usecols=["pert_id", "drugname", "synonyms", "smiles", "Smiles_no_chiral", "Smiles_with_chiral"],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    smiles_map, name_map, main_details, main_ids = build_main_lookup(main_raw)
    meta_smiles = {str(key): clean(value) for key, value in meta["pertid_to_smiles"].items()}
    fingerprint_generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    uncharger = rdMolStandardize.Uncharger()
    tautomer = rdMolStandardize.TautomerEnumerator()
    tautomer.SetMaxTransforms(100)
    tautomer.SetMaxTautomers(100)
    chemical_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    rows: list[dict[str, Any]] = []
    group_columns = ["mat", "raw_drug_id"]
    grouped_records = unique_records.groupby(group_columns, sort=True, dropna=False)
    for group_index, ((mat, raw_drug_id), group) in enumerate(grouped_records, start=1):
        if group_index % 500 == 0:
            print(f"[mat-audit] processed groups={group_index}", flush=True)
        value_columns = [
            "raw_drug_name",
            "raw_legacy_smiles",
            "raw_Smiles_no_chiral",
            "raw_Smiles_with_chiral",
            "raw_model_candidate_smiles",
            "resolved_drug_id",
            "resolution_method",
        ]
        values = {column: sorted({clean(value) for value in group[column] if clean(value)}) for column in value_columns}
        conflict_columns = [column for column in value_columns if len(values[column]) != 1]
        primary = {column: values[column][0] if len(values[column]) == 1 else "" for column in value_columns}
        replay_id, replay_method = replay_single_mapping(pd.Series({"raw_drug_id": raw_drug_id, **primary}), smiles_map, name_map)
        resolved_id = primary["resolved_drug_id"]
        method = primary["resolution_method"]
        model_smiles = meta_smiles.get(resolved_id, "")
        raw_smiles = primary["raw_model_candidate_smiles"]
        chemical_applicable = method.startswith("existing_by_")
        chemical = {
            "raw_parseable": False,
            "model_parseable": False,
            "isomeric_equal": False,
            "connectivity_equal": False,
            "parent_equal": False,
            "tautomer_parent_equal": False,
            "morgan_equal": False,
            "morgan_tanimoto": None,
        }
        if chemical_applicable:
            chemical_key = (raw_smiles, model_smiles, method)
            if chemical_key not in chemical_cache:
                raw_mol = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
                model_mol = Chem.MolFromSmiles(model_smiles) if model_smiles else None
                raw_iso = canonical_smiles(raw_smiles, isomeric=True)
                model_iso = canonical_smiles(model_smiles, isomeric=True)
                raw_connectivity = canonical_smiles(raw_smiles, isomeric=False)
                model_connectivity = canonical_smiles(model_smiles, isomeric=False)
                chemical.update(
                    {
                        "raw_parseable": bool(raw_mol is not None),
                        "model_parseable": bool(model_mol is not None),
                        "isomeric_equal": bool(raw_iso and raw_iso == model_iso),
                        "connectivity_equal": bool(raw_connectivity and raw_connectivity == model_connectivity),
                    }
                )
                if raw_mol is not None and model_mol is not None:
                    raw_fp = fingerprint_generator.GetFingerprint(raw_mol)
                    model_fp = fingerprint_generator.GetFingerprint(model_mol)
                    chemical["morgan_equal"] = raw_fp.ToBitString() == model_fp.ToBitString()
                    chemical["morgan_tanimoto"] = float(DataStructs.TanimotoSimilarity(raw_fp, model_fp))
                    if method == "existing_by_drug_name" and not chemical["connectivity_equal"]:
                        try:
                            raw_parent = uncharger.uncharge(rdMolStandardize.FragmentParent(raw_mol))
                            model_parent = uncharger.uncharge(rdMolStandardize.FragmentParent(model_mol))
                            raw_parent_text = Chem.MolToSmiles(raw_parent, canonical=True, isomericSmiles=False)
                            model_parent_text = Chem.MolToSmiles(model_parent, canonical=True, isomericSmiles=False)
                            chemical["parent_equal"] = bool(
                                raw_parent_text and raw_parent_text == model_parent_text
                            )
                            raw_tautomer = tautomer.Canonicalize(raw_parent)
                            model_tautomer = tautomer.Canonicalize(model_parent)
                            raw_tautomer_text = Chem.MolToSmiles(
                                raw_tautomer, canonical=True, isomericSmiles=False
                            )
                            model_tautomer_text = Chem.MolToSmiles(
                                model_tautomer, canonical=True, isomericSmiles=False
                            )
                            chemical["tautomer_parent_equal"] = bool(
                                raw_tautomer_text and raw_tautomer_text == model_tautomer_text
                            )
                        except Exception:
                            chemical["parent_equal"] = False
                            chemical["tautomer_parent_equal"] = False
                chemical_cache[chemical_key] = dict(chemical)
            chemical = chemical_cache[chemical_key]

        isomeric_equal = bool(chemical["isomeric_equal"])
        connectivity_equal = bool(chemical["connectivity_equal"])
        parent_equal = bool(chemical["parent_equal"])
        tautomer_parent_equal = bool(chemical["tautomer_parent_equal"])
        morgan_equal = bool(chemical["morgan_equal"])
        morgan_tanimoto = chemical["morgan_tanimoto"]

        exact_raw_fields = [
            column
            for column in ("raw_legacy_smiles", "raw_Smiles_no_chiral", "raw_Smiles_with_chiral")
            if model_smiles and model_smiles == primary[column]
        ]
        severity = "pass"
        status = ""
        if conflict_columns or replay_id != resolved_id or replay_method != method:
            severity = "error"
            status = "mapping_replay_or_within_mat_conflict"
        elif method.startswith("unified_by_"):
            severity = "info"
            status = "external_id_not_mapped_to_main"
        elif method == "existing_by_drug_name":
            if connectivity_equal:
                status = "name_mapping_exact_connectivity"
            elif parent_equal:
                severity = "review"
                status = "name_mapping_salt_or_charge_variant"
            elif tautomer_parent_equal:
                severity = "review"
                status = "name_mapping_tautomer_variant"
            elif morgan_tanimoto is not None and morgan_tanimoto < 0.5:
                severity = "high_risk"
                status = "name_mapping_major_structure_mismatch"
            else:
                severity = "review"
                status = "name_mapping_structure_mismatch"
        elif connectivity_equal and isomeric_equal:
            status = "smiles_mapping_isomeric_structure_match"
        elif connectivity_equal:
            severity = "review"
            status = "smiles_mapping_connectivity_only_stereo_diff"
        else:
            severity = "high_risk"
            status = "smiles_mapping_connectivity_mismatch"

        detail = main_details.get(resolved_id, {})
        name_match_source, matched_main_name_tokens = main_name_match_evidence(
            primary["raw_drug_name"], detail
        )
        rows.append(
            {
                "mat": mat,
                "raw_drug_id": clean(raw_drug_id),
                "raw_drug_name": primary["raw_drug_name"],
                "raw_legacy_smiles": primary["raw_legacy_smiles"],
                "raw_Smiles_no_chiral": primary["raw_Smiles_no_chiral"],
                "raw_Smiles_with_chiral": primary["raw_Smiles_with_chiral"],
                "raw_model_candidate_smiles": raw_smiles,
                "resolved_drug_id": resolved_id,
                "resolution_method": method,
                "name_match_source": name_match_source if method == "existing_by_drug_name" else "",
                "matched_main_name_tokens": (
                    json.dumps(matched_main_name_tokens, ensure_ascii=False, separators=(",", ":"))
                    if method == "existing_by_drug_name"
                    else "[]"
                ),
                "resolved_to_main_existing_id": bool(resolved_id in main_ids),
                "replayed_drug_id": replay_id,
                "replayed_resolution_method": replay_method,
                "mapping_replay_exact": bool(replay_id == resolved_id and replay_method == method),
                "within_mat_conflict_columns": json.dumps(conflict_columns, separators=(",", ":")),
                "source_tasks": json_values(group["source_task"]),
                "devices": json_values(group["device"]),
                "raw_source_files": json_values(group["raw_source_file"]),
                "raw_occurrence_count": int(group["raw_occurrence_count"].sum()),
                "example_raw_row_index_0based": int(group["first_raw_row_index_0based"].min()),
                "model_canonical_smiles": model_smiles,
                "model_smiles_exact_raw_fields": json.dumps(exact_raw_fields, separators=(",", ":")),
                "chemical_comparison_applicable": chemical_applicable,
                "raw_smiles_parseable": bool(chemical["raw_parseable"]),
                "model_smiles_parseable": bool(chemical["model_parseable"]),
                "isomeric_structure_equal": isomeric_equal,
                "connectivity_equal_ignoring_stereo": connectivity_equal,
                "fragment_charge_parent_equal": parent_equal,
                "tautomer_parent_equal": tautomer_parent_equal,
                "morgan_radius2_2048_equal": morgan_equal,
                "morgan_radius2_2048_tanimoto": morgan_tanimoto,
                "audit_severity": severity,
                "audit_status": status,
                "main_raw_first_row_index_0based": detail.get("main_raw_first_row_index_0based", ""),
                "main_raw_drug_names": json.dumps(detail.get("main_raw_drug_names", []), ensure_ascii=False, separators=(",", ":")),
                "main_raw_synonyms": json.dumps(detail.get("main_raw_synonyms", []), ensure_ascii=False, separators=(",", ":")),
                "main_raw_legacy_smiles_all": json.dumps(detail.get("main_raw_legacy_smiles", []), ensure_ascii=False, separators=(",", ":")),
                "main_raw_Smiles_no_chiral_all": json.dumps(detail.get("main_raw_Smiles_no_chiral", []), ensure_ascii=False, separators=(",", ":")),
                "main_raw_Smiles_with_chiral_all": json.dumps(detail.get("main_raw_Smiles_with_chiral", []), ensure_ascii=False, separators=(",", ":")),
            }
        )

    print(f"[mat-audit] materializing rows={len(rows)}", flush=True)
    audit = pd.DataFrame(rows)
    cross = audit.groupby("raw_drug_id").agg(
        cross_mat_resolved_id_count=("resolved_drug_id", "nunique"),
        cross_mat_resolution_method_count=("resolution_method", "nunique"),
        cross_mat_raw_name_count=("raw_drug_name", "nunique"),
        cross_mat_raw_model_smiles_count=("raw_model_candidate_smiles", "nunique"),
        mats=("mat", lambda values: json_values(values)),
    )
    audit = audit.merge(cross, on="raw_drug_id", how="left", validate="many_to_one")
    audit["cross_mat_mapping_consistent"] = (
        audit["cross_mat_resolved_id_count"].eq(1)
        & audit["cross_mat_resolution_method_count"].eq(1)
        & audit["cross_mat_raw_name_count"].eq(1)
        & audit["cross_mat_raw_model_smiles_count"].eq(1)
    )
    inconsistent = ~audit["cross_mat_mapping_consistent"]
    audit.loc[inconsistent, "audit_severity"] = "error"
    audit.loc[inconsistent, "audit_status"] = "cross_mat_raw_id_mapping_conflict"
    return audit.sort_values(["mat", "raw_drug_id"], kind="stable").reset_index(drop=True)


def value_counts_nested(frame: pd.DataFrame, group: str, value: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for key, subset in frame.groupby(group, sort=True):
        result[str(key)] = {str(name): int(count) for name, count in subset[value].value_counts().items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--global-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    scope = pd.read_csv(args.scope, dtype=str, keep_default_na=False, low_memory=False)
    if len(scope) != 3217 or scope["pert_id"].nunique() != 3217:
        raise ValueError("Exp32 scope must contain exactly 3,217 unique pert_id values")
    meta = load_json(args.global_meta)

    origin_path = Path(f"{args.output_prefix}_exp32_3217_smiles_origin.csv")
    mat_path = Path(f"{args.output_prefix}_mat1_4_unique_drug_mapping.csv")
    manual_review_path = Path(f"{args.output_prefix}_manual_review_unique.csv")
    high_risk_path = Path(f"{args.output_prefix}_high_risk_unique.csv")
    summary_path = Path(f"{args.output_prefix}_summary.json")
    origin_path.parent.mkdir(parents=True, exist_ok=True)

    origin, source_inventory = build_exp32_origin(scope, meta)
    origin.to_csv(origin_path, index=False, encoding="utf-8-sig")
    print(f"[write] {origin_path}", flush=True)
    origin_count = int(len(origin))
    origin_unique_count = int(origin["drug_id"].nunique())
    origin_all_exact = bool(origin["model_smiles_exact_origin_match"].all())
    origin_task_counts = {str(k): int(v) for k, v in origin["canonical_origin_task"].value_counts().items()}
    origin_column_counts = {str(k): int(v) for k, v in origin["selected_raw_column"].value_counts().items()}
    del origin
    gc.collect()

    mat_audit = build_mat_audit(meta)
    mat_audit.to_csv(mat_path, index=False, encoding="utf-8-sig")

    unique_raw = mat_audit.sort_values(["raw_drug_id", "mat"], kind="stable").drop_duplicates("raw_drug_id")
    manual_review = unique_raw.loc[unique_raw["audit_severity"].isin(["review", "high_risk"])].sort_values(
        ["audit_severity", "morgan_radius2_2048_tanimoto", "raw_drug_id"],
        ascending=[True, True, True],
        kind="stable",
    )
    high_risk = unique_raw.loc[unique_raw["audit_severity"].eq("high_risk")].sort_values(
        ["morgan_radius2_2048_tanimoto", "raw_drug_id"], kind="stable"
    )
    mapped_to_main = unique_raw.loc[unique_raw["resolved_to_main_existing_id"]]
    raw_ids_per_main = mapped_to_main.groupby("resolved_drug_id")["raw_drug_id"].nunique()
    manual_review.to_csv(manual_review_path, index=False, encoding="utf-8-sig")
    high_risk.to_csv(high_risk_path, index=False, encoding="utf-8-sig")
    summary = {
        "scope": {
            "exp32_drug_count": origin_count,
            "unique_drug_ids": origin_unique_count,
            "all_model_smiles_exact_raw_origin": origin_all_exact,
            "origin_task_counts": origin_task_counts,
            "origin_raw_column_counts": origin_column_counts,
            "source_inventory": source_inventory,
        },
        "mat1_4": {
            "rows_by_mat": {str(k): int(v) for k, v in mat_audit["mat"].value_counts().sort_index().items()},
            "unique_raw_drug_ids_union": int(mat_audit["raw_drug_id"].nunique()),
            "unique_resolved_drug_ids_union": int(mat_audit["resolved_drug_id"].nunique()),
            "unique_raw_drug_ids_mapped_to_main": int(len(mapped_to_main)),
            "unique_main_drug_ids_reached": int(mapped_to_main["resolved_drug_id"].nunique()),
            "main_ids_with_multiple_raw_ids": int(raw_ids_per_main.gt(1).sum()),
            "maximum_raw_ids_per_main_id": int(raw_ids_per_main.max()),
            "unique_raw_mapping_method_counts": {
                str(k): int(v) for k, v in unique_raw["resolution_method"].value_counts().items()
            },
            "method_counts_by_mat": value_counts_nested(mat_audit, "mat", "resolution_method"),
            "severity_counts_by_mat": value_counts_nested(mat_audit, "mat", "audit_severity"),
            "status_counts_by_mat": value_counts_nested(mat_audit, "mat", "audit_status"),
            "unique_raw_severity_counts": {
                str(k): int(v) for k, v in unique_raw["audit_severity"].value_counts().items()
            },
            "unique_raw_status_counts": {
                str(k): int(v) for k, v in unique_raw["audit_status"].value_counts().items()
            },
            "name_match_source_counts": {
                str(k): int(v)
                for k, v in unique_raw.loc[
                    unique_raw["resolution_method"].eq("existing_by_drug_name"), "name_match_source"
                ].value_counts().items()
            },
            "mapping_replay_mismatch_count": int((~mat_audit["mapping_replay_exact"]).sum()),
            "cross_mat_inconsistent_row_count": int((~mat_audit["cross_mat_mapping_consistent"]).sum()),
            "within_mat_conflict_row_count": int(mat_audit["within_mat_conflict_columns"].ne("[]").sum()),
        },
        "artifacts": {
            "exp32_smiles_origin_csv": str(origin_path),
            "mat1_4_mapping_csv": str(mat_path),
            "manual_review_unique_csv": str(manual_review_path),
            "high_risk_unique_csv": str(high_risk_path),
        },
    }
    summary["artifacts"]["summary_json"] = str(summary_path)
    dump_json(summary_path, summary)

    print(f"exp32_origin={origin_path} rows={origin_count}")
    print(f"mat_mapping={mat_path} rows={len(mat_audit)} unique_raw_ids={mat_audit['raw_drug_id'].nunique()}")
    print(f"manual_review={manual_review_path} rows={len(manual_review)}")
    print(f"high_risk={high_risk_path} rows={len(high_risk)}")
    print(f"summary={summary_path}")
    print("origin_tasks=" + json.dumps(summary["scope"]["origin_task_counts"], sort_keys=True))
    print("origin_columns=" + json.dumps(summary["scope"]["origin_raw_column_counts"], sort_keys=True))
    print("methods=" + json.dumps(summary["mat1_4"]["unique_raw_mapping_method_counts"], sort_keys=True))
    print("severity_by_mat=" + json.dumps(summary["mat1_4"]["severity_counts_by_mat"], sort_keys=True))


if __name__ == "__main__":
    main()
