#!/usr/bin/env python3
"""Export PTV3 single-drug non-ablation data as gene-axis AnnData files.

The source ProteinTalk artifacts use UniProt accessions as the measured
feature axis. This exporter maps those accessions to HGNC gene symbols, drops
features without a reliable one-to-one gene mapping, collapses duplicate gene
columns by mean, and writes AnnData files with split metadata for exp01,
exp02, exp03, and exp07.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import requests


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready" / "ptv3"
TASKS_ROOT = TRAINING_READY_ROOT / "tasks"
SPLITS_ROOT = TRAINING_READY_ROOT / "splits"
FASTA_PATH = TRAINING_READY_ROOT / "derived" / "idmapping_2026_04_27.fasta"

HGNC_COMPLETE_SET_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
UNIPROT_REST_URL_TEMPLATE = "https://rest.uniprot.org/uniprotkb/{accession}.json"

MAIN_TASK = "ptv3_main_singledrug"
EXTRA_SINGLE_TASKS = [
    "ptv3_extra_singledrug_mat1_480_faims",
    "ptv3_extra_singledrug_mat1_qe",
    "ptv3_extra_singledrug_mat2_480_faims",
    "ptv3_extra_singledrug_mat2_qe",
    "ptv3_extra_singledrug_mat3_qe",
    "ptv3_extra_singledrug_mat4_qe",
]
ALL_TASKS = [MAIN_TASK, *EXTRA_SINGLE_TASKS]

MAIN_SPLIT_SPECS = [
    *[
        (f"exp01_pert_stratified_fold{fold}", f"pert_stratified_5fold_fold{fold}")
        for fold in range(5)
    ],
    *[(f"exp02_cell_type_fold{fold}", f"cell_type_5fold_fold{fold}") for fold in range(5)],
    *[(f"exp03_cell_fold{fold}", f"cell_5fold_fold{fold}") for fold in range(5)],
    ("exp07_all_single_for_extra", "all_train_subset_test"),
]
EXTRA_SPLIT_SPECS = [("extra_single_test_only", "test_only")]


@dataclass(frozen=True)
class GeneMapping:
    uniprot_id: str
    hgnc_symbol: str
    hgnc_id: str
    status: str
    source: str
    source_symbol: str = ""
    entry_type: str = ""
    primary_accession: str = ""
    redirected_to: str = ""
    inactive_reason: str = ""
    demerged_to: str = ""
    note: str = ""

    @property
    def mapped(self) -> bool:
        return self.status == "mapped" and bool(self.hgnc_symbol)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def split_pipe(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def parse_fasta_gene_names(path: Path) -> dict[str, str]:
    accession_to_gene: dict[str, str] = {}
    header_re = re.compile(r"^>\w+\|([^|]+)\|")
    gene_re = re.compile(r"\bGN=([^\s]+)")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            accession_match = header_re.match(line)
            gene_match = gene_re.search(line)
            if accession_match and gene_match:
                accession_to_gene[accession_match.group(1)] = gene_match.group(1)
    return accession_to_gene


def download_hgnc_complete_set(cache_path: Path, *, refresh: bool = False) -> Path:
    if cache_path.exists() and not refresh:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(HGNC_COMPLETE_SET_URL, timeout=120)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return cache_path


def load_hgnc_maps(tsv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    current_by_symbol: dict[str, dict[str, str]] = {}
    aliases: dict[str, list[dict[str, str]]] = defaultdict(list)
    previous: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_uniprot: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in df.to_dict("records"):
        record = {
            "hgnc_id": str(row.get("hgnc_id", "")).strip(),
            "symbol": str(row.get("symbol", "")).strip(),
            "name": str(row.get("name", "")).strip(),
        }
        symbol = record["symbol"]
        if symbol:
            current_by_symbol[symbol] = record
        for token in split_pipe(row.get("alias_symbol", "")):
            aliases[token].append(record)
        for token in split_pipe(row.get("prev_symbol", "")):
            previous[token].append(record)
        for token in split_pipe(row.get("uniprot_ids", "")):
            by_uniprot[token].append(record)

    return {
        "current_by_symbol": current_by_symbol,
        "aliases": aliases,
        "previous": previous,
        "by_uniprot": by_uniprot,
    }


def unique_record(records: list[dict[str, str]]) -> dict[str, str] | None:
    if len(records) != 1:
        return None
    return records[0]


def resolve_hgnc_symbol(symbol: str, hgnc_maps: dict[str, Any]) -> tuple[str, str, str]:
    symbol = str(symbol).strip()
    if not symbol:
        return "", "", "empty"
    current = hgnc_maps["current_by_symbol"]
    if symbol in current:
        record = current[symbol]
        return record["symbol"], record["hgnc_id"], "hgnc_current_symbol"

    previous = unique_record(hgnc_maps["previous"].get(symbol, []))
    if previous:
        return previous["symbol"], previous["hgnc_id"], "hgnc_previous_symbol"

    alias = unique_record(hgnc_maps["aliases"].get(symbol, []))
    if alias:
        return alias["symbol"], alias["hgnc_id"], "hgnc_alias_symbol"

    return symbol, "", "uniprot_gene_name_no_hgnc_id"


def rest_gene_mapping(uniprot_id: str, hgnc_maps: dict[str, Any]) -> GeneMapping:
    response = requests.get(UNIPROT_REST_URL_TEMPLATE.format(accession=uniprot_id), timeout=30)
    response.raise_for_status()
    data = response.json()

    entry_type = str(data.get("entryType") or "")
    primary_accession = str(data.get("primaryAccession") or "")
    redirected_to = primary_accession if primary_accession and primary_accession != uniprot_id else ""

    inactive = data.get("inactiveReason") or {}
    inactive_reason = str(inactive.get("inactiveReasonType") or "")
    demerged_to = "|".join(str(item) for item in inactive.get("mergeDemergeTo", []) or [])

    hgnc_symbol = ""
    hgnc_id = ""
    for ref in data.get("uniProtKBCrossReferences", []) or []:
        if ref.get("database") != "HGNC":
            continue
        hgnc_id = str(ref.get("id") or "")
        for prop in ref.get("properties", []) or []:
            if prop.get("key") == "GeneName":
                hgnc_symbol = str(prop.get("value") or "")
                break
        if hgnc_symbol:
            break

    if not hgnc_symbol:
        genes = data.get("genes") or []
        if genes:
            hgnc_symbol = str((genes[0].get("geneName") or {}).get("value") or "")

    if hgnc_symbol:
        approved_symbol, approved_hgnc_id, source = resolve_hgnc_symbol(hgnc_symbol, hgnc_maps)
        if hgnc_id and not approved_hgnc_id:
            approved_hgnc_id = hgnc_id
        return GeneMapping(
            uniprot_id=uniprot_id,
            hgnc_symbol=approved_symbol,
            hgnc_id=approved_hgnc_id,
            status="mapped",
            source=f"uniprot_rest_{source}",
            source_symbol=hgnc_symbol,
            entry_type=entry_type,
            primary_accession=primary_accession,
            redirected_to=redirected_to,
            inactive_reason=inactive_reason,
            demerged_to=demerged_to,
        )

    if inactive_reason == "DEMERGED" and demerged_to:
        return GeneMapping(
            uniprot_id=uniprot_id,
            hgnc_symbol="",
            hgnc_id="",
            status="unmapped_ambiguous_inactive_demerged",
            source="uniprot_rest_inactive",
            entry_type=entry_type,
            primary_accession=primary_accession,
            inactive_reason=inactive_reason,
            demerged_to=demerged_to,
            note="inactive UniProt accession demerged to multiple accessions; no one-to-one HGNC gene mapping",
        )

    return GeneMapping(
        uniprot_id=uniprot_id,
        hgnc_symbol="",
        hgnc_id="",
        status="unmapped_no_hgnc_gene",
        source="uniprot_rest",
        entry_type=entry_type,
        primary_accession=primary_accession,
        redirected_to=redirected_to,
        inactive_reason=inactive_reason,
        demerged_to=demerged_to,
    )


def load_rest_cache(path: Path) -> dict[str, GeneMapping]:
    if not path.exists():
        return {}
    payload = load_json(path)
    return {key: GeneMapping(**value) for key, value in payload.items()}


def save_rest_cache(path: Path, cache: dict[str, GeneMapping]) -> None:
    dump_json(path, {key: value.__dict__ for key, value in sorted(cache.items())})


def build_gene_mapping(
    uniprot_ids: list[str],
    *,
    output_root: Path,
    refresh_hgnc: bool,
    refresh_uniprot_rest: bool,
) -> dict[str, GeneMapping]:
    mapping_root = output_root / "mapping"
    hgnc_path = download_hgnc_complete_set(mapping_root / "hgnc_complete_set.txt", refresh=refresh_hgnc)
    hgnc_maps = load_hgnc_maps(hgnc_path)
    fasta_gene_names = parse_fasta_gene_names(FASTA_PATH)
    rest_cache_path = mapping_root / "uniprot_rest_gene_mapping_cache.json"
    rest_cache = {} if refresh_uniprot_rest else load_rest_cache(rest_cache_path)

    result: dict[str, GeneMapping] = {}
    for uniprot_id in sorted(set(uniprot_ids)):
        hgnc_records = hgnc_maps["by_uniprot"].get(uniprot_id, [])
        if len(hgnc_records) == 1:
            record = hgnc_records[0]
            result[uniprot_id] = GeneMapping(
                uniprot_id=uniprot_id,
                hgnc_symbol=record["symbol"],
                hgnc_id=record["hgnc_id"],
                status="mapped",
                source="hgnc_uniprot_ids",
                source_symbol=record["symbol"],
            )
            continue
        if len(hgnc_records) > 1:
            result[uniprot_id] = GeneMapping(
                uniprot_id=uniprot_id,
                hgnc_symbol="",
                hgnc_id="",
                status="unmapped_ambiguous_hgnc_uniprot",
                source="hgnc_uniprot_ids",
                note="UniProt accession maps to multiple HGNC records: "
                + "|".join(record["symbol"] for record in hgnc_records),
            )
            continue

        fasta_symbol = fasta_gene_names.get(uniprot_id, "")
        if fasta_symbol:
            approved_symbol, hgnc_id, source = resolve_hgnc_symbol(fasta_symbol, hgnc_maps)
            result[uniprot_id] = GeneMapping(
                uniprot_id=uniprot_id,
                hgnc_symbol=approved_symbol,
                hgnc_id=hgnc_id,
                status="mapped",
                source=f"fasta_gn_{source}",
                source_symbol=fasta_symbol,
            )
            continue

        if uniprot_id not in rest_cache:
            rest_cache[uniprot_id] = rest_gene_mapping(uniprot_id, hgnc_maps)
        result[uniprot_id] = rest_cache[uniprot_id]

    save_rest_cache(rest_cache_path, rest_cache)
    write_mapping_audit(mapping_root / "uniprot_to_hgnc_gene_mapping.csv", result)
    return result


def write_mapping_audit(path: Path, mapping: dict[str, GeneMapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "uniprot_id",
        "hgnc_symbol",
        "hgnc_id",
        "status",
        "source",
        "source_symbol",
        "entry_type",
        "primary_accession",
        "redirected_to",
        "inactive_reason",
        "demerged_to",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for uniprot_id in sorted(mapping):
            writer.writerow(mapping[uniprot_id].__dict__)


def task_protein_ids(task_name: str) -> list[str]:
    return [str(item) for item in load_json(TASKS_ROOT / task_name / "feature_ordered_protein_uniprot.json")]


def mapped_gene_order(uniprot_ids: list[str], gene_mapping: dict[str, GeneMapping]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for uniprot_id in uniprot_ids:
        record = gene_mapping[uniprot_id]
        if not record.mapped:
            continue
        gene = record.hgnc_symbol
        if gene not in seen:
            order.append(gene)
            seen.add(gene)
    return order


def build_var_table(
    gene_order: list[str],
    uniprot_ids: list[str],
    gene_mapping: dict[str, GeneMapping],
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for col_idx, uniprot_id in enumerate(uniprot_ids):
        record = gene_mapping[uniprot_id]
        if record.mapped and record.hgnc_symbol in gene_order:
            groups[record.hgnc_symbol].append(col_idx)

    hgnc_ids: dict[str, str] = {}
    source_uniprots: dict[str, list[str]] = {}
    source_statuses: dict[str, list[str]] = {}
    for uniprot_id in uniprot_ids:
        record = gene_mapping[uniprot_id]
        if not record.mapped or record.hgnc_symbol not in groups:
            continue
        source_uniprots.setdefault(record.hgnc_symbol, []).append(uniprot_id)
        source_statuses.setdefault(record.hgnc_symbol, []).append(record.source)
        if record.hgnc_symbol not in hgnc_ids and record.hgnc_id:
            hgnc_ids[record.hgnc_symbol] = record.hgnc_id

    var = pd.DataFrame(index=pd.Index(gene_order, name="hgnc_symbol"))
    var["hgnc_symbol"] = gene_order
    var["hgnc_id"] = [hgnc_ids.get(gene, "") for gene in gene_order]
    var["source_uniprot_ids"] = ["|".join(source_uniprots.get(gene, [])) for gene in gene_order]
    var["source_uniprot_count"] = [len(source_uniprots.get(gene, [])) for gene in gene_order]
    var["mapping_sources"] = ["|".join(sorted(set(source_statuses.get(gene, [])))) for gene in gene_order]
    var["collapsed_duplicate_uniprot"] = var["source_uniprot_count"] > 1
    return var, groups


def collapse_to_gene_matrix(
    matrix: np.ndarray,
    gene_order: list[str],
    groups: dict[str, list[int]],
) -> np.ndarray:
    out = np.empty((matrix.shape[0], len(gene_order)), dtype=np.float32)
    for out_col, gene in enumerate(gene_order):
        source_cols = groups[gene]
        if len(source_cols) == 1:
            out[:, out_col] = matrix[:, source_cols[0]]
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                out[:, out_col] = np.nanmean(matrix[:, source_cols], axis=1).astype(np.float32, copy=False)
    return out


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def first_smiles(value: Any) -> str:
    text = clean_string(value)
    if "||" in text:
        return text.split("||", 1)[0].strip()
    return text


def bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].fillna(False).astype(bool)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def build_obs(task_name: str) -> pd.DataFrame:
    df = pd.read_parquet(TASKS_ROOT / task_name / "feature_table.parquet").reset_index(drop=True)
    sample_id = df["sample_id"].map(clean_string)
    control_bool = bool_series(df, "is_control")
    drug_name = df["drugname"].map(clean_string) if "drugname" in df.columns else pd.Series("", index=df.index)
    pert_id = df["pert_id1"].map(clean_string) if "pert_id1" in df.columns else pd.Series("", index=df.index)
    drug_token = drug_name.mask(drug_name.eq(""), pert_id)
    drug_token = drug_token.mask(control_bool, "")
    sample_id_set = set(sample_id)
    control_sample = df["control"].map(clean_string) if "control" in df.columns else pd.Series("", index=df.index)
    matched_control = control_sample.where((~control_bool) & control_sample.isin(sample_id_set), "")

    obs = pd.DataFrame(index=pd.Index(sample_id, name="sample_id"))
    obs["sample_id"] = sample_id.to_numpy()
    obs["dataset"] = task_name
    obs["source_task"] = df.get("source_task", pd.Series(task_name, index=df.index)).map(clean_string).to_numpy()
    obs["source_row_role"] = df.get("source_row_role", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["feature_membership"] = df.get("feature_membership", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["control"] = control_bool.to_numpy()
    obs["is_control"] = control_bool.to_numpy()
    obs["control_sample_id"] = control_sample.to_numpy()
    obs["matched_control_obs_name"] = matched_control.to_numpy()
    obs["gene_pt"] = ""
    obs["drug_pt"] = drug_token.to_numpy()
    obs["env_pt"] = ""
    obs["CRISPR"] = ""
    obs["perturbation"] = np.where(control_bool.to_numpy(), "control", drug_token.to_numpy())
    obs["perturbation_type"] = np.where(control_bool.to_numpy(), "control", "drug")
    obs["drug_id"] = pert_id.to_numpy()
    obs["drug_name"] = drug_name.to_numpy()
    obs["drug_smiles"] = df.get("smiles", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["drug_smiles_primary"] = df.get("smiles", pd.Series("", index=df.index)).map(first_smiles).to_numpy()
    obs["cell_line"] = df.get("Cell", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["cell_cluster"] = obs["cell_line"].to_numpy()
    obs["cell"] = df.get("cell_type", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["cell_type"] = obs["cell"].to_numpy()
    obs["Cell_plate"] = df.get("Cell_plate", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["machineID_new"] = df.get("machineID_new", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["instrument"] = df.get("instrument", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["batch"] = df.get("batch", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["pert_time"] = numeric_series(df, "pert_time").to_numpy()
    obs["pert_dose1"] = numeric_series(df, "pert_dose1").to_numpy()
    obs["pert_dose2"] = numeric_series(df, "pert_dose2").to_numpy()
    obs["PRISM1st_label_total"] = df.get("PRISM1st_label_total", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["PRISM2nd_label_total"] = df.get("PRISM2nd_label_total", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["synergy"] = df.get("synergy", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["control_status"] = df.get("control_status", pd.Series("", index=df.index)).map(clean_string).to_numpy()
    obs["expression_available"] = bool_series(df, "expression_available").to_numpy()
    obs["target_protein_uniprot_list"] = df.get(
        "target_protein_uniprot_list", pd.Series("", index=df.index)
    ).map(clean_string).to_numpy()
    obs["target_protein_count"] = numeric_series(df, "target_protein_count").fillna(0).astype(int).to_numpy()
    obs["feature_row_index"] = np.arange(len(obs), dtype=np.int64)
    return obs


def load_split_indices(split_dir: Path, split_name: str, strategy: str) -> list[int]:
    candidates = [split_dir / f"{split_name}_indices_{strategy}.pkl"]
    if split_name == "valid":
        candidates.append(split_dir / f"val_indices_{strategy}.pkl")
    for candidate in candidates:
        if candidate.exists():
            return [int(item) for item in load_pickle(candidate)]
    raise FileNotFoundError(f"missing {split_name} indices for {strategy} under {split_dir}")


def add_split_columns(obs: pd.DataFrame, task_name: str, split_specs: list[tuple[str, str]]) -> dict[str, Any]:
    split_dir = SPLITS_ROOT / task_name
    controls = obs["control"].to_numpy(dtype=bool)
    split_uns: dict[str, Any] = {}
    for public_name, source_strategy in split_specs:
        split_label = np.full(len(obs), "unused", dtype=object)
        split_label[controls] = "control"

        memberships: dict[str, list[int]] = {}
        membership_masks: dict[str, np.ndarray] = {}
        for split_name in ("train", "valid", "test"):
            indices = load_split_indices(split_dir, split_name, source_strategy)
            memberships[split_name] = indices
            mask = np.zeros(len(obs), dtype=bool)
            mask[indices] = True
            membership_masks[split_name] = mask
            obs[f"is_{split_name}_{public_name}"] = mask

        for idx in range(len(obs)):
            if controls[idx]:
                continue
            labels = [name for name in ("train", "valid", "test") if membership_masks[name][idx]]
            if labels:
                split_label[idx] = "+".join(labels)

        exact_col = f"split_{public_name}"
        obs[exact_col] = split_label
        vcbench_label = split_label.copy()
        vcbench_label[vcbench_label == "valid"] = "val"
        vcbench_label[vcbench_label == "control"] = "train"
        obs[f"split_vcbench_{public_name}"] = vcbench_label

        overlaps = {
            "train_valid": int(np.logical_and(membership_masks["train"], membership_masks["valid"]).sum()),
            "train_test": int(np.logical_and(membership_masks["train"], membership_masks["test"]).sum()),
            "valid_test": int(np.logical_and(membership_masks["valid"], membership_masks["test"]).sum()),
        }
        split_uns[public_name] = {
            "source_strategy": source_strategy,
            "obs_columns": {
                "split": exact_col,
                "vcbench_split": f"split_vcbench_{public_name}",
                "train_membership": f"is_train_{public_name}",
                "valid_membership": f"is_valid_{public_name}",
                "test_membership": f"is_test_{public_name}",
            },
            "counts": {name: len(indices) for name, indices in memberships.items()},
            "overlaps": overlaps,
            "train_obs_names": obs.index[membership_masks["train"]].astype(str).tolist(),
            "valid_obs_names": obs.index[membership_masks["valid"]].astype(str).tolist(),
            "test_obs_names": obs.index[membership_masks["test"]].astype(str).tolist(),
            "control_obs_names": obs.index[controls].astype(str).tolist(),
            "note": "train/valid/test memberships are exact source split indices; controls are a shared control pool",
        }
    if split_specs:
        first_public_name = split_specs[0][0]
        obs["split"] = obs[f"split_vcbench_{first_public_name}"]
        obs["default_split_strategy"] = first_public_name
    return split_uns


def write_anndata(
    *,
    task_name: str,
    output_path: Path,
    gene_order: list[str],
    gene_mapping: dict[str, GeneMapping],
    split_specs: list[tuple[str, str]],
    axis_name: str,
    compression: str,
) -> dict[str, Any]:
    uniprot_ids = task_protein_ids(task_name)
    var, groups = build_var_table(gene_order, uniprot_ids, gene_mapping)
    source_matrix = np.load(TASKS_ROOT / task_name / "feature_expression_matrix.npy").astype(np.float32, copy=False)
    gene_matrix = collapse_to_gene_matrix(source_matrix, gene_order, groups)
    del source_matrix

    obs = build_obs(task_name)
    splits_uns = add_split_columns(obs, task_name, split_specs)

    mapped_uniprots = [uid for uid in uniprot_ids if gene_mapping[uid].mapped]
    dropped_uniprots = [uid for uid in uniprot_ids if not gene_mapping[uid].mapped]
    duplicate_gene_counts = Counter(gene_mapping[uid].hgnc_symbol for uid in mapped_uniprots)
    duplicate_gene_counts = Counter({gene: count for gene, count in duplicate_gene_counts.items() if count > 1})

    adata = ad.AnnData(X=gene_matrix, obs=obs, var=var)
    adata.uns["ptv3_transcriptome_export"] = {
        "task_name": task_name,
        "axis_name": axis_name,
        "source_task_dir": str(TASKS_ROOT / task_name),
        "source_expression_matrix": str(TASKS_ROOT / task_name / "feature_expression_matrix.npy"),
        "source_feature_axis": "UniProt accession",
        "export_feature_axis": "HGNC gene symbol",
        "values_note": "The numeric values are the original proteomics expression values relabeled onto a gene-symbol axis; they are not RNA-seq counts.",
        "uniprot_feature_count": len(uniprot_ids),
        "mapped_uniprot_count": len(mapped_uniprots),
        "dropped_uniprot_count": len(dropped_uniprots),
        "dropped_uniprot_ids": dropped_uniprots,
        "gene_count": len(gene_order),
        "collapsed_duplicate_gene_count": len(duplicate_gene_counts),
        "collapsed_duplicate_gene_examples": dict(list(sorted(duplicate_gene_counts.items()))[:20]),
        "split_strategies": list(splits_uns),
    }
    adata.uns["splits"] = splits_uns
    adata.uns["control_contract"] = {
        "control_column": "control",
        "control_true_means": "control sample row",
        "perturbation_to_control_column": "matched_control_obs_name",
        "original_control_sample_id_column": "control_sample_id",
    }
    adata.uns["drug_contract"] = {
        "drug_token_column": "drug_pt",
        "drug_id_column": "drug_id",
        "drug_name_column": "drug_name",
        "smiles_column": "drug_smiles",
        "primary_smiles_column": "drug_smiles_primary",
    }
    adata.strings_to_categoricals()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression=compression)

    summary = {
        "task_name": task_name,
        "output_path": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "axis_name": axis_name,
        "uniprot_feature_count": len(uniprot_ids),
        "mapped_uniprot_count": len(mapped_uniprots),
        "dropped_uniprot_count": len(dropped_uniprots),
        "collapsed_duplicate_gene_count": len(duplicate_gene_counts),
        "control_count": int(np.asarray(adata.obs["control"]).astype(bool).sum()),
        "perturbed_count": int((~np.asarray(adata.obs["control"]).astype(bool)).sum()),
        "splits": {
            name: {
                "source_strategy": spec["source_strategy"],
                "counts": spec["counts"],
                "overlaps": spec["overlaps"],
            }
            for name, spec in splits_uns.items()
        },
    }
    del adata
    del gene_matrix
    return summary


def write_readme(output_root: Path, summary: dict[str, Any]) -> None:
    native_rows = "\n".join(
        f"| `{item['task_name']}` | `{Path(item['output_path']).relative_to(output_root)}` | "
        f"{item['shape'][0]} x {item['shape'][1]} | {item['control_count']} |"
        for item in summary["native_outputs"]
    )
    common_rows = "\n".join(
        f"| `{item['task_name']}` | `{Path(item['output_path']).relative_to(output_root)}` | "
        f"{item['shape'][0]} x {item['shape'][1]} | {item['control_count']} |"
        for item in summary["common_outputs"]
    )
    split_rows = []
    for name, spec in summary["native_outputs"][0]["splits"].items():
        counts = spec["counts"]
        overlaps = spec["overlaps"]
        split_rows.append(
            f"| `{name}` | `{spec['source_strategy']}` | {counts['train']} | {counts['valid']} | "
            f"{counts['test']} | {overlaps['train_test']} |"
        )
    split_table = "\n".join(split_rows)
    readme = f"""# PTV3 Single-Drug Transcriptome AnnData Export

This directory contains AnnData exports for the single-drug, non-ablation PTV3 experiments:

- exp01: `pert_stratified_5fold_fold0..4`
- exp02: `cell_type_5fold_fold0..4`
- exp03: `cell_5fold_fold0..4`
- exp07: `all_train_subset_test` training split plus six extra single-drug test-only datasets

Excluded by design: exp04 `w/o MSE`, exp05 `w/o proteome/graph`, exp06 double drug, and exp08 extra double drug.

## Files

Primary common-gene-axis files are under `common_gene_axis/`. These files all use the same `adata.var_names`
ordered by the main single-drug task, restricted to genes present in every exported task. Use these when training
on the converted main single-drug data and inferring on converted extra single-drug data.

| task | file | shape | control rows |
| --- | --- | ---: | ---: |
{common_rows}

Native files are under `native_gene_axis/`. These keep each task's own mapped gene axis after UniProt-to-HGNC
conversion and duplicate-gene collapse.

| task | file | shape | control rows |
| --- | --- | ---: | ---: |
{native_rows}

Mapping and audit files:

- `mapping/uniprot_to_hgnc_gene_mapping.csv`: one row per UniProt accession in the included tasks.
- `mapping/hgnc_complete_set.txt`: HGNC reference table downloaded during export.
- `summary.json`: machine-readable export summary.

## How To Read

```python
import anndata as ad

adata = ad.read_h5ad("common_gene_axis/ptv3_main_singledrug.common_hgnc.h5ad")
print(adata.X.shape)
print(adata.obs.head())
print(adata.var.head())
```

`adata.X` contains the original proteomics expression values on an HGNC gene-symbol axis. This is a gene-axis
conversion of proteomics measurements, not RNA-seq counts.

Missing expression values remain as `NaN`, matching the source training-ready matrices. The extra single-drug
perturbation rows are inference-only rows and do not have observed post-perturbation expression in the source
artifacts; their `adata.X` rows are therefore `NaN`. Their matched control rows are present and finite where the
source baseline/control proteomics was available.

## Split Metadata

The exact split definitions are in `adata.uns["splits"]`. For example:

```python
split = adata.uns["splits"]["exp01_pert_stratified_fold0"]
train = adata[split["train_obs_names"]]
valid = adata[split["valid_obs_names"]]
test = adata[split["test_obs_names"]]
test_expression = test.X
```

Main single-drug split counts:

| public split | source split strategy | train | valid | test | train-test overlap |
| --- | --- | ---: | ---: | ---: | ---: |
{split_table}

For convenience, each split also has row-level columns:

- `split_<public_split>`: exact human-readable label. Values may be `train+test` for `all_train_subset_test`.
- `is_train_<public_split>`, `is_valid_<public_split>`, `is_test_<public_split>`: exact boolean memberships.
- `split_vcbench_<public_split>`: convenience label using `val` instead of `valid` and putting controls in train.

The default `adata.obs["split"]` is a compatibility column for the first split in each file. Use
`adata.uns["splits"]` for exact multi-split work.

## Controls

Controls are identified by:

- `adata.obs["control"] == True`
- `adata.obs["perturbation"] == "control"`
- empty perturbation columns: `gene_pt == ""`, `drug_pt == ""`, `env_pt == ""`, `CRISPR == ""`

For each perturbed row, `matched_control_obs_name` gives the control sample row to use when it is present in the
same AnnData file. The original control sample id from the ProteinTalk metadata is stored in `control_sample_id`.

## Drug Perturbation And SMILES

Drug perturbation fields are:

- `drug_pt`: drug token, using `drug_name` when available and falling back to `drug_id`.
- `drug_id`: original `pert_id1`.
- `drug_name`: original drug name.
- `drug_smiles`: original SMILES string.
- `drug_smiles_primary`: first SMILES before the `||` separator.

Controls have empty `drug_pt`, `drug_id`, `drug_name`, and SMILES fields.

## Gene Mapping

UniProt accessions were mapped to HGNC gene symbols using:

1. HGNC complete set `uniprot_ids`.
2. Local UniProt FASTA `GN=` fields, resolved through HGNC current/previous/alias symbols when possible.
3. UniProt REST API for accessions missing from the local FASTA.

Features without a reliable one-to-one HGNC gene symbol were dropped. Multiple UniProt columns mapping to the same
HGNC symbol were collapsed by row-wise mean; the original UniProt accessions are recorded in
`adata.var["source_uniprot_ids"]`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "transcriptome_anndata" / "ptv3_single_nonablation",
    )
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", None])
    parser.add_argument("--refresh-hgnc", action="store_true")
    parser.add_argument("--refresh-uniprot-rest", action="store_true")
    parser.add_argument("--native-only", action="store_true", help="Only write native gene-axis h5ad files")
    parser.add_argument("--common-only", action="store_true", help="Only write common gene-axis h5ad files")
    args = parser.parse_args()

    if args.native_only and args.common_only:
        raise SystemExit("--native-only and --common-only are mutually exclusive")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    task_to_uniprot_ids = {task: task_protein_ids(task) for task in ALL_TASKS}
    all_uniprot_ids = sorted({uid for ids in task_to_uniprot_ids.values() for uid in ids})
    gene_mapping = build_gene_mapping(
        all_uniprot_ids,
        output_root=output_root,
        refresh_hgnc=args.refresh_hgnc,
        refresh_uniprot_rest=args.refresh_uniprot_rest,
    )

    task_to_native_gene_order = {
        task: mapped_gene_order(task_to_uniprot_ids[task], gene_mapping)
        for task in ALL_TASKS
    }
    common_gene_set = set(task_to_native_gene_order[MAIN_TASK])
    for task in EXTRA_SINGLE_TASKS:
        common_gene_set &= set(task_to_native_gene_order[task])
    common_gene_order = [gene for gene in task_to_native_gene_order[MAIN_TASK] if gene in common_gene_set]

    summary: dict[str, Any] = {
        "output_root": str(output_root),
        "source_training_ready_root": str(TRAINING_READY_ROOT),
        "included_tasks": ALL_TASKS,
        "included_experiments": ["exp01", "exp02", "exp03", "exp07"],
        "excluded_experiments": ["exp04_w_o_mse", "exp05_w_o_proteome_graph", "exp06_double_drug", "exp08_extra_double_drug"],
        "all_uniprot_count": len(all_uniprot_ids),
        "mapped_uniprot_count": sum(1 for uid in all_uniprot_ids if gene_mapping[uid].mapped),
        "unmapped_uniprot_ids": [uid for uid in all_uniprot_ids if not gene_mapping[uid].mapped],
        "native_gene_counts": {task: len(genes) for task, genes in task_to_native_gene_order.items()},
        "common_gene_count": len(common_gene_order),
        "native_outputs": [],
        "common_outputs": [],
    }

    if not args.common_only:
        for task in ALL_TASKS:
            split_specs = MAIN_SPLIT_SPECS if task == MAIN_TASK else EXTRA_SPLIT_SPECS
            output_path = output_root / "native_gene_axis" / f"{task}.hgnc.h5ad"
            summary["native_outputs"].append(
                write_anndata(
                    task_name=task,
                    output_path=output_path,
                    gene_order=task_to_native_gene_order[task],
                    gene_mapping=gene_mapping,
                    split_specs=split_specs,
                    axis_name="native_hgnc_gene_axis",
                    compression=args.compression,
                )
            )

    if not args.native_only:
        for task in ALL_TASKS:
            split_specs = MAIN_SPLIT_SPECS if task == MAIN_TASK else EXTRA_SPLIT_SPECS
            output_path = output_root / "common_gene_axis" / f"{task}.common_hgnc.h5ad"
            summary["common_outputs"].append(
                write_anndata(
                    task_name=task,
                    output_path=output_path,
                    gene_order=common_gene_order,
                    gene_mapping=gene_mapping,
                    split_specs=split_specs,
                    axis_name="common_hgnc_gene_axis",
                    compression=args.compression,
                )
            )

    dump_json(output_root / "summary.json", summary)
    write_readme(output_root, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
