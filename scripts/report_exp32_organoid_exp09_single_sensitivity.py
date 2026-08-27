#!/usr/bin/env python3
"""Preflight, validate, and report exp32 organoid sensitivity inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_PREFIX = "20260710_exp32_organoid_exp09_single_sensitivity"
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data/training_ready_exp32_organoid"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=5.ckpt"
)
DEFAULT_OUTPUT_BASE = REPO_ROOT / "outputs"
DEFAULT_RESULT_MARKDOWN = REPO_ROOT / "docs/2026-07-10_exp32_organoid_exp09_single_sensitivity_results.md"

EXPECTED_DRUG_COUNT = 3217
EXPECTED_SAMPLE_COUNT = 13
EXPECTED_QUERY_ROWS = EXPECTED_DRUG_COUNT * EXPECTED_SAMPLE_COUNT
EXPECTED_CONTROL_ROWS = EXPECTED_SAMPLE_COUNT
EXPECTED_TOTAL_ROWS = EXPECTED_QUERY_ROWS + EXPECTED_CONTROL_ROWS
EXPECTED_PROTEIN_COUNT = 11092
EXPECTED_BATCH_COVARIATES = [
    "machineID_new",
    "Cell_plate",
    "Cell",
    "cell_type",
    "batch",
    "pert_time",
    "pert_dose1",
    "pert_dose2",
]
EXPECTED_DOSE_COVARIATES = ["pert_dose1", "pert_dose2"]
ALLOWED_CHECKPOINT_MISMATCH_KEYS = {"meta_path", "task_dir", "ordered_protein_index_path"}
PATH_CONFIG_KEYS = {
    "meta_path",
    "protein_embedding_path",
    "drug_embedding_path",
    "ppi_matrix_path",
    "pdi_matrix_path",
    "ddi_matrix_path",
    "cell_llm_embedding_path",
    "cell_type_llm_embedding_path",
    "random_control_expression_path",
}

RAW_INPUT_PATHS = (
    REPO_ROOT / "data/rawdata/rna_seq/260709_B260625QC_merged_replicates_add_CellType.csv",
    REPO_ROOT / "data/rawdata/rna_seq/260709_CAC260627QC_merged_replicates_add_CellType.csv",
    REPO_ROOT / "data/rawdata/rna_seq/260709_samp_inf.csv",
)

HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
SAMPLE_SUFFIX_RE = re.compile(r"PTV2_(\d+)$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    device: str
    source_prefix: str
    machine: str
    machine_index: int
    batch: str


TASK_SPECS = (
    TaskSpec(
        task_name="ptv3_exp32_organoid_qe_single",
        device="B",
        source_prefix="B260625_PTV2_",
        machine="QE",
        machine_index=2,
        batch="exp32_organoid_qe",
    ),
    TaskSpec(
        task_name="ptv3_exp32_organoid_480_faims_single",
        device="CAC",
        source_prefix="CAC260627_PTV2_",
        machine="480_FAIMS",
        machine_index=1,
        batch="exp32_organoid_480_faims",
    ),
)


@dataclass
class DrugScope:
    ids: list[str]
    frame: pd.DataFrame
    payload: Any


@dataclass
class TaskAudit:
    spec: TaskSpec
    task_dir: Path
    split_dir: Path
    table_path: Path
    table: pd.DataFrame
    query_indices: list[int]
    control_indices: list[int]
    sample_column: str
    matrix_audit: dict[str, Any]


@dataclass
class PreflightContext:
    group_root: Path
    build_summary_path: Path
    build_summary: dict[str, Any]
    scope: DrugScope
    meta_path: Path
    meta: dict[str, Any]
    checkpoint_manifest_path: Path
    checkpoint_path: Path
    checkpoint_manifest: dict[str, Any]
    checkpoint_axis: list[int]
    tasks: dict[str, TaskAudit]
    hash_audit: list[dict[str, Any]]


class ValidationError(RuntimeError):
    """Raised when an exp32 acceptance condition is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hkt_now() -> str:
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).replace(microsecond=0).isoformat()


def output_bucket_for_prefix(prefix: str) -> Path:
    token = str(prefix)[:8]
    try:
        output_date = datetime.strptime(token, "%Y%m%d").date().isoformat()
    except ValueError:
        output_date = datetime.now(ZoneInfo("Asia/Hong_Kong")).date().isoformat()
    return DEFAULT_OUTPUT_BASE / output_date[:7] / output_date


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except (OSError, ValueError):
        return str(path)


def resolve_recorded_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def same_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return resolve_recorded_path(left).resolve() == resolve_recorded_path(right).resolve()
    except (OSError, ValueError):
        return str(left) == str(right)


def load_json(path: Path) -> Any:
    require(path.exists(), f"missing JSON artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: Path) -> Any:
    require(path.exists(), f"missing pickle artifact: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "<na>"} else text


def clean_series(series: pd.Series) -> pd.Series:
    return series.map(clean_text).astype("string")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    text = clean_series(series).str.lower()
    return numeric.eq(1) | text.isin({"true", "yes", "y", "control"})


def numeric_series(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    require(parsed.notna().all(), f"{name} contains missing/non-numeric values")
    return parsed


def require_empty_labels(table: pd.DataFrame, indices: list[int], columns: Iterable[str], label: str) -> None:
    for column in columns:
        if column not in table.columns:
            continue
        values = clean_series(table.iloc[indices][column])
        require(values.eq("").all(), f"{label}: {column} must be empty on every query row")


def find_table(task_dir: Path, stem: str = "feature_table") -> tuple[pd.DataFrame, Path]:
    candidates = [task_dir / f"{stem}.parquet", task_dir / f"{stem}.pkl", task_dir / f"{stem}.csv"]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path).reset_index(drop=True), path
        if path.suffix == ".pkl":
            return pd.read_pickle(path).reset_index(drop=True), path
        return pd.read_csv(path, low_memory=False).reset_index(drop=True), path
    raise ValidationError(f"missing {stem}.parquet/pkl/csv under {task_dir}")


def load_predictions(output_dir: Path) -> tuple[pd.DataFrame, Path]:
    for path in (output_dir / "predictions.parquet", output_dir / "predictions.csv"):
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path), path
            return pd.read_csv(path, low_memory=False), path
    raise ValidationError(f"missing predictions.parquet/csv under {output_dir}")


def group_root(training_ready_root: Path) -> Path:
    return training_ready_root if training_ready_root.name == "ptv3" else training_ready_root / "ptv3"


def sample_number(value: Any) -> int:
    match = SAMPLE_SUFFIX_RE.search(clean_text(value))
    if not match:
        raise ValidationError(f"cannot parse PTV2 numeric suffix from sample id: {value!r}")
    number = int(match.group(1))
    require(1 <= number <= EXPECTED_SAMPLE_COUNT, f"sample suffix outside 1..13: {value!r}")
    return number


def expected_tissue(number: int) -> str:
    if number in {1, 13}:
        return "PANCREAS"
    if number == 5:
        return "COLON"
    return "LUNG"


def extract_scope_records(payload: Any) -> list[dict[str, Any]]:
    candidate = payload
    if isinstance(payload, dict):
        for key in ("drugs", "drug_ids", "pert_ids", "drug_scope", "rows"):
            if key in payload:
                candidate = payload[key]
                break
    if isinstance(candidate, dict):
        candidate = [{"pert_id": key, **(value if isinstance(value, dict) else {})} for key, value in candidate.items()]
    require(isinstance(candidate, list), "drug-scope JSON does not contain a list of drugs")
    records: list[dict[str, Any]] = []
    for item in candidate:
        if isinstance(item, str):
            records.append({"pert_id": item})
        elif isinstance(item, dict):
            pert_id = next((item.get(key) for key in ("pert_id", "drug_id", "id") if item.get(key) is not None), None)
            require(pert_id is not None, f"drug-scope record lacks pert_id/drug_id/id: {item}")
            records.append({**item, "pert_id": clean_text(pert_id)})
        else:
            raise ValidationError(f"unexpected drug-scope record: {item!r}")
    return records


def load_drug_scope(json_path: Path, csv_path: Path) -> DrugScope:
    payload = load_json(json_path)
    records = extract_scope_records(payload)
    json_ids = [clean_text(row.get("pert_id")) for row in records]
    require(all(json_ids), "drug-scope JSON contains an empty pert_id")
    require(len(json_ids) == EXPECTED_DRUG_COUNT, f"drug-scope JSON has {len(json_ids)} drugs, expected 3217")
    require(len(set(json_ids)) == EXPECTED_DRUG_COUNT, "drug-scope JSON contains duplicate drug IDs")

    require(csv_path.exists(), f"missing drug-scope CSV: {csv_path}")
    frame = pd.read_csv(csv_path, low_memory=False)
    id_column = next((name for name in ("pert_id", "drug_id", "id") if name in frame.columns), None)
    require(id_column is not None, f"drug-scope CSV lacks pert_id/drug_id/id: {csv_path}")
    csv_ids = clean_series(frame[id_column]).tolist()
    require(len(csv_ids) == EXPECTED_DRUG_COUNT, f"drug-scope CSV has {len(csv_ids)} drugs, expected 3217")
    require(len(set(csv_ids)) == EXPECTED_DRUG_COUNT, "drug-scope CSV contains duplicate drug IDs")
    require(set(csv_ids) == set(json_ids), "drug-scope JSON and CSV drug sets differ")
    frame = frame.copy()
    frame["pert_id"] = csv_ids
    order_lookup = {pert_id: index for index, pert_id in enumerate(json_ids)}
    frame["scope_order"] = [order_lookup[item] for item in csv_ids]
    if isinstance(payload, dict) and payload.get("drug_count") is not None:
        require(int(payload["drug_count"]) == EXPECTED_DRUG_COUNT, "drug-scope JSON drug_count is not 3217")
    require("no" not in {item.lower() for item in json_ids}, "drug scope must not contain control pert_id 'no'")
    return DrugScope(ids=json_ids, frame=frame, payload=payload)


def collect_hash_records(payload: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    path_field_names = {
        "path",
        "file",
        "file_path",
        "source",
        "source_path",
        "input_path",
        "artifact_path",
    }

    def walk(value: Any, label: str, path_hint: str | None = None) -> None:
        if isinstance(value, dict):
            local_path = path_hint
            for key in path_field_names:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    local_path = candidate
                    break
            for key, candidate in value.items():
                key_lower = str(key).lower()
                if "sha256" in key_lower and isinstance(candidate, str) and HEX_SHA256.fullmatch(candidate.strip()):
                    base = key_lower
                    for suffix in ("_sha256_before", "_sha256_after", "_sha256"):
                        if base.endswith(suffix):
                            base = base[: -len(suffix)]
                            break
                    if base.startswith("sha256"):
                        base = ""
                    path_candidates = (
                        f"{base}_path" if base else "path",
                        base,
                        f"{base}_file" if base else "file",
                        "path",
                        "file_path",
                        "input_path",
                        "source_path",
                        "source_input_file",
                    )
                    paired_path = next(
                        (
                            value.get(path_key)
                            for path_key in path_candidates
                            if isinstance(value.get(path_key), str) and clean_text(value.get(path_key))
                        ),
                        local_path,
                    )
                    if paired_path:
                        records.append({"path": str(paired_path), "sha256": candidate.lower(), "label": f"{label}.{key}"})
                if isinstance(key, str) and ("/" in key or key.endswith((".csv", ".json", ".npy", ".pkl", ".npz", ".ckpt"))):
                    if isinstance(candidate, str) and HEX_SHA256.fullmatch(candidate.strip()):
                        records.append({"path": key, "sha256": candidate.lower(), "label": label})
                    elif isinstance(candidate, dict):
                        walk(candidate, f"{label}.{key}", key)
                    else:
                        walk(candidate, f"{label}.{key}", None)
                else:
                    walk(candidate, f"{label}.{key}", local_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{label}[{index}]", path_hint)

    walk(payload, "build_summary")
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for record in records:
        unique[(record["path"], record["sha256"], record["label"])] = record
    return list(unique.values())


def validate_build_summary_contract(
    summary: dict[str, Any],
    *,
    root: Path,
    checkpoint_path: Path,
    scope_json_path: Path,
    scope_csv_path: Path,
) -> None:
    require(summary.get("experiment_name") == "exp32_organoid_exp09_single_sensitivity", "build summary experiment_name mismatch")
    require(summary.get("dataset_group") == "ptv3", "build summary dataset_group mismatch")
    require(same_path(summary.get("output_root"), root.parent), "build summary output_root mismatch")
    recorded_checkpoint = resolve_recorded_path(summary.get("checkpoint_path"))
    require(recorded_checkpoint.exists(), f"build summary checkpoint is missing: {recorded_checkpoint}")
    require(
        recorded_checkpoint.resolve().parent == checkpoint_path.resolve().parent,
        "selected checkpoint is not from the exp09 run used to build exp32",
    )
    require(int((summary.get("protein_axis") or {}).get("count", -1)) == EXPECTED_PROTEIN_COUNT, "build summary protein-axis count mismatch")

    drug_summary = summary.get("drug_scope") or {}
    require(int(drug_summary.get("drug_count", -1)) == EXPECTED_DRUG_COUNT, "build summary drug count mismatch")
    require(int(drug_summary.get("expected_drug_count", -1)) == EXPECTED_DRUG_COUNT, "build summary expected drug count mismatch")
    require(len(drug_summary.get("drug_ids") or []) == EXPECTED_DRUG_COUNT, "build summary drug_ids length mismatch")
    require(same_path(drug_summary.get("scope_json"), scope_json_path), "build summary scope_json path mismatch")
    require(same_path(drug_summary.get("scope_csv"), scope_csv_path), "build summary scope_csv path mismatch")
    for key in (
        "all_drugs_in_pert_index",
        "all_morgan_rows_finite_nonzero",
        "all_graph_rows_finite_nonzero",
        "all_smiles_nonempty",
    ):
        require(drug_summary.get(key) is True, f"build summary drug-scope check is not true: {key}")

    fixed = summary.get("fixed_settings") or {}
    require(fixed.get("prediction_type") == "sensitivity", "build summary prediction type mismatch")
    require(fixed.get("single_drug_same_two_slots") is True, "build summary single-drug slot policy mismatch")
    for key, expected in (("pert_time", 24), ("pert_dose1", 10), ("pert_dose2", 10), ("unseen_Cell_Cell_plate_batch_index", 0)):
        require(int(fixed.get(key, -1)) == expected, f"build summary fixed setting {key} mismatch")

    acceptance = summary.get("acceptance_checks") or {}
    require(acceptance and all(value is True for value in acceptance.values()), "one or more build-summary acceptance checks are not true")
    raw_inputs = summary.get("raw_inputs") or {}
    require(len(raw_inputs) == 3, "build summary must contain the two matrices and sample-info raw input")
    for label, record in raw_inputs.items():
        require(record.get("unchanged") is True, f"build summary raw input {label} is not marked unchanged")
        require(record.get("sha256_before") == record.get("sha256_after"), f"build summary raw input {label} before/after hashes differ")

    task_summaries = summary.get("tasks") or {}
    require(set(task_summaries) == {spec.task_name for spec in TASK_SPECS}, "build summary task set mismatch")
    for spec in TASK_SPECS:
        task = task_summaries[spec.task_name]
        require(task.get("task_name") == spec.task_name, f"build summary task_name mismatch for {spec.task_name}")
        require(task.get("device") == spec.device, f"build summary device mismatch for {spec.task_name}")
        require(task.get("machineID_new") == spec.machine, f"build summary machine mismatch for {spec.task_name}")
        require(int(task.get("rows", -1)) == EXPECTED_TOTAL_ROWS, f"build summary row count mismatch for {spec.task_name}")
        require(int(task.get("control_rows", -1)) == EXPECTED_CONTROL_ROWS, f"build summary control count mismatch for {spec.task_name}")
        require(int(task.get("query_rows", -1)) == EXPECTED_QUERY_ROWS, f"build summary query count mismatch for {spec.task_name}")
        require(int(task.get("set_count", -1)) == EXPECTED_SAMPLE_COUNT, f"build summary set count mismatch for {spec.task_name}")
        require(task.get("matrix_shape") == [EXPECTED_TOTAL_ROWS, EXPECTED_PROTEIN_COUNT], f"build summary matrix shape mismatch for {spec.task_name}")
        require(task.get("matrix_dtype") == "float32", f"build summary matrix dtype mismatch for {spec.task_name}")
        query_contract = task.get("query_contract") or {}
        require(query_contract and all(value is True for value in query_contract.values()), f"build summary query contract failed for {spec.task_name}")

    reused = summary.get("readonly_reused_artifacts") or {}
    for key in ("global_meta", "derived"):
        record = reused.get(key) or {}
        alias = resolve_recorded_path(record.get("alias"))
        source = resolve_recorded_path(record.get("source"))
        require(record.get("kind") == "read_only_symlink", f"build summary {key} is not a read-only symlink")
        require(alias.is_symlink(), f"exp32 {key} alias is not a symlink: {alias}")
        require(alias.resolve() == source.resolve(), f"exp32 {key} alias resolves to the wrong source")


def validate_hashes(build_summary: dict[str, Any], required_raw_paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = collect_hash_records(build_summary)
    require(records, "build summary does not contain any SHA-256 records")
    audit: list[dict[str, Any]] = []
    cache: dict[Path, str] = {}
    by_resolved_path: dict[Path, list[dict[str, str]]] = {}
    for record in records:
        path = resolve_recorded_path(record["path"])
        require(path.exists(), f"hash-recorded artifact is missing: {path}")
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = sha256_file(path)
        actual = cache[resolved]
        require(actual == record["sha256"], f"SHA-256 mismatch for {path}: recorded={record['sha256']} actual={actual}")
        by_resolved_path.setdefault(resolved, []).append(record)
        audit.append({"path": str(path), "sha256": actual, "record_label": record["label"]})

    for raw_path in required_raw_paths:
        require(raw_path.exists(), f"required raw input is missing: {raw_path}")
        matches = by_resolved_path.get(raw_path.resolve(), [])
        require(matches, f"build summary has no SHA-256 record for raw input: {raw_path}")
        before = {row["sha256"] for row in matches if "before" in row["label"].lower()}
        after = {row["sha256"] for row in matches if "after" in row["label"].lower()}
        if before or after:
            require(before and after, f"raw input hash audit lacks before/after pair: {raw_path}")
            require(before == after, f"raw input changed between before/after hashes: {raw_path}")
    return audit


def validate_checkpoint_contract(checkpoint_path: Path) -> tuple[Path, dict[str, Any], list[int]]:
    require(checkpoint_path.exists(), f"missing checkpoint: {checkpoint_path}")
    require(checkpoint_path.suffix == ".ckpt", f"exp32 checkpoint must be a .ckpt file: {checkpoint_path}")
    manifest_path = checkpoint_path.resolve().parent / "run_manifest.json"
    manifest = load_json(manifest_path)
    require(isinstance(manifest, dict), f"checkpoint manifest is not a JSON object: {manifest_path}")
    expected = {
        "run_status": "fit_completed",
        "dataset_group": "ptv3",
        "task_name": "ptv3_main_doubledrug",
        "split_strategy": "all_train_subset_test",
        "model_type": "fast_delta",
        "task_head": "unified",
        "task_label_key": "synergy",
        "task_mask_key": "unified_label_mask",
        "task_label_policy": "unified_synergy_first_else_response",
        "graph_feature_mode": "real",
        "graph_feature_dim": 128,
        "graph_feature_seed": 17,
        "graph_structural_rp": True,
        "graph_multihop": False,
        "graph_drug_concat": True,
        "graph_pair_add_scale": 0.5,
        "pair_fusion_mode": "dual",
        "pair_type_features": True,
        "protein_concat_mode": "pcep",
        "protein_concat_dim": 64,
        "protein_concat_topk": 512,
        "use_ddi": True,
        "use_target": True,
        "target_protein_max_length": 32,
        "cell_llm_mode": "frozen",
        "cell_type_llm_mode": "frozen",
        "hidden_dim": 512,
        "expression_latent_dim": 768,
        "covariate_embedding_dim": 96,
        "dropout": 0.15,
        "control_layers": 2,
        "fusion_layers": 3,
        "target_layers": 2,
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"checkpoint manifest {key}={manifest.get(key)!r}, expected {value!r}")
    require(manifest.get("batch_cov_list") == EXPECTED_BATCH_COVARIATES, "checkpoint batch_cov_list differs from exp09 contract")
    require(manifest.get("use_dose_covariate") is True, "checkpoint dose covariates are not enabled")
    require(manifest.get("dose_covariate_fields") == EXPECTED_DOSE_COVARIATES, "checkpoint dose fields differ")
    require(manifest.get("target_expression_mode") == "off", "checkpoint target_expression_mode must remain off")
    axis_path = resolve_recorded_path(manifest.get("ordered_protein_index_path"))
    axis = [int(item) for item in load_json(axis_path)]
    require(len(axis) == EXPECTED_PROTEIN_COUNT, f"checkpoint protein axis has {len(axis)} rows, expected 11092")
    require(len(set(axis)) == len(axis), "checkpoint protein axis contains duplicates")
    return manifest_path, manifest, axis


def validate_global_meta(meta_path: Path, checkpoint_manifest: dict[str, Any], scope: DrugScope) -> dict[str, Any]:
    meta = load_json(meta_path)
    require(isinstance(meta, dict), f"global meta is not a JSON object: {meta_path}")
    source_meta_path = resolve_recorded_path(checkpoint_manifest.get("meta_path"))
    source_meta = load_json(source_meta_path)
    for key in ("pert_index", "pert_index_to_id", "protein_index", "protein_index_to_id", "value_to_index", "special_values"):
        require(meta.get(key) == source_meta.get(key), f"exp32 global_meta modifies checkpoint index/category key: {key}")
    pert_index = {str(key): int(value) for key, value in meta.get("pert_index", {}).items()}
    missing = sorted(set(scope.ids) - set(pert_index))
    require(not missing, f"drug scope contains IDs absent from pert_index: {missing[:10]}")
    smiles_map = {str(key): clean_text(value) for key, value in meta.get("pertid_to_smiles", {}).items()}
    missing_smiles = [item for item in scope.ids if not smiles_map.get(item)]
    require(not missing_smiles, f"drug scope contains IDs without SMILES: {missing_smiles[:10]}")

    values = meta.get("value_to_index", {})
    categorical_expected = {
        ("machineID_new", "QE"): 2,
        ("machineID_new", "480_FAIMS"): 1,
        ("cell_type", "LUNG"): 8,
        ("cell_type", "PANCREAS"): 11,
        ("cell_type", "COLON"): 4,
        ("pert_time", "24"): 2,
    }
    for (field, token), expected in categorical_expected.items():
        require(int(values.get(field, {}).get(token, -1)) == expected, f"global_meta {field}[{token}] index mismatch")
    require(int(values.get("pert_dose", {}).get("10", -1)) == 10, "global_meta dose 10 index mismatch")
    for field in ("Cell", "Cell_plate", "batch"):
        require(int(values.get(field, {}).get("no", -1)) == 0, f"global_meta {field} no index must be 0")
    return meta


def read_npy_header(handle: Any, path: Path) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    version = np.lib.format.read_magic(handle)
    shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)  # type: ignore[attr-defined]
    require(not fortran_order, f"Fortran-order expression matrix is unsupported: {path}")
    return tuple(int(item) for item in shape), bool(fortran_order), np.dtype(dtype), int(handle.tell())


def read_exact_array(handle: Any, *, count: int, dtype: np.dtype[Any], path: Path) -> np.ndarray:
    size = int(count * dtype.itemsize)
    payload = handle.read(size)
    require(len(payload) == size, f"unexpected EOF while streaming {path}")
    return np.frombuffer(payload, dtype=dtype, count=count)


def validate_matrix_pair(task_dir: Path, table: pd.DataFrame, control_indices: list[int], query_indices: list[int], chunk_rows: int) -> dict[str, Any]:
    feature_path = task_dir / "feature_expression_matrix.npy"
    processed_path = task_dir / "processed_expression_matrix.npy"
    require(feature_path.exists(), f"missing expression matrix: {feature_path}")
    require(processed_path.exists(), f"missing processed expression matrix: {processed_path}")
    expected_shape = (EXPECTED_TOTAL_ROWS, EXPECTED_PROTEIN_COUNT)
    same_inode = False
    try:
        same_inode = os.path.samefile(feature_path, processed_path)
    except OSError:
        same_inode = False

    control_set = set(control_indices)
    query_set = set(query_indices)
    finite_counts: list[int] = []
    with feature_path.open("rb") as feature_handle:
        feature_shape, _, feature_dtype, _ = read_npy_header(feature_handle, feature_path)
        require(feature_shape == expected_shape, f"{feature_path} shape={feature_shape}, expected {expected_shape}")
        require(feature_dtype == np.dtype(np.float32), f"{feature_path} dtype={feature_dtype}, expected float32")
        processed_handle = None
        try:
            if not same_inode:
                processed_handle = processed_path.open("rb")
                processed_shape, _, processed_dtype, _ = read_npy_header(processed_handle, processed_path)
                require(processed_shape == expected_shape, f"{processed_path} shape={processed_shape}, expected {expected_shape}")
                require(processed_dtype == feature_dtype, f"{processed_path} dtype differs from feature matrix")
            else:
                with processed_path.open("rb") as handle:
                    processed_shape, _, processed_dtype, _ = read_npy_header(handle, processed_path)
                require(processed_shape == expected_shape, f"{processed_path} shape={processed_shape}, expected {expected_shape}")
                require(processed_dtype == feature_dtype, f"{processed_path} dtype differs from feature matrix")

            for start in range(0, EXPECTED_TOTAL_ROWS, chunk_rows):
                stop = min(EXPECTED_TOTAL_ROWS, start + chunk_rows)
                row_count = stop - start
                values = read_exact_array(
                    feature_handle,
                    count=row_count * EXPECTED_PROTEIN_COUNT,
                    dtype=feature_dtype,
                    path=feature_path,
                ).reshape(row_count, EXPECTED_PROTEIN_COUNT)
                if processed_handle is not None:
                    processed_values = read_exact_array(
                        processed_handle,
                        count=row_count * EXPECTED_PROTEIN_COUNT,
                        dtype=feature_dtype,
                        path=processed_path,
                    ).reshape(row_count, EXPECTED_PROTEIN_COUNT)
                    require(
                        np.array_equal(values, processed_values, equal_nan=True),
                        f"feature/processed expression matrices differ in rows {start}:{stop} for {task_dir.name}",
                    )
                for offset, row_index in enumerate(range(start, stop)):
                    row = values[offset]
                    if row_index in query_set:
                        require(np.isnan(row).all(), f"query expression contains a non-NaN value at row {row_index} in {task_dir.name}")
                    elif row_index in control_set:
                        finite = np.isfinite(row)
                        count = int(finite.sum())
                        require(count > 1000, f"control row {row_index} has too few finite proteins: {count}")
                        require(not np.any(row[finite] < 0.0), f"control row {row_index} contains negative log1p expression")
                        finite_counts.append(count)
                    else:
                        raise ValidationError(f"expression row {row_index} is neither control nor query")
        finally:
            if processed_handle is not None:
                processed_handle.close()
    require(len(finite_counts) == EXPECTED_CONTROL_ROWS, f"{task_dir.name}: expression scan did not see 13 controls")
    return {
        "feature_path": str(feature_path),
        "processed_path": str(processed_path),
        "shape": list(expected_shape),
        "dtype": str(np.dtype(np.float32)),
        "same_inode": same_inode,
        "control_finite_min": min(finite_counts),
        "control_finite_max": max(finite_counts),
    }


def normalize_set_info(payload: Any) -> dict[int, dict[str, list[int]]]:
    require(isinstance(payload, dict), "set_info payload is not a dictionary")
    result: dict[int, dict[str, list[int]]] = {}
    for key, value in payload.items():
        require(isinstance(value, dict), f"set_info[{key}] is not a dictionary")
        result[int(key)] = {
            "control": [int(item) for item in value.get("control", [])],
            "perturb": [int(item) for item in value.get("perturb", [])],
        }
    return result


def validate_splits(split_dir: Path, table: pd.DataFrame, control_indices: list[int], query_indices: list[int]) -> None:
    test_indices = [int(item) for item in load_pickle(split_dir / "test_indices_test_only.pkl")]
    require(test_indices == query_indices, f"{split_dir}: test_only indices are not exactly all query rows in feature order")
    for split in ("train", "valid", "val"):
        path = split_dir / f"{split}_indices_test_only.pkl"
        require([int(item) for item in load_pickle(path)] == [], f"{path} must be empty")

    set_info = normalize_set_info(load_pickle(split_dir / "set_info.pkl"))
    require(len(set_info) == EXPECTED_SAMPLE_COUNT, f"{split_dir}: set count={len(set_info)}, expected 13")
    seen_controls: list[int] = []
    seen_queries: list[int] = []
    for set_index, payload in sorted(set_info.items()):
        require(len(payload["control"]) == 1, f"set {set_index} must have exactly one control")
        require(len(payload["perturb"]) == EXPECTED_DRUG_COUNT, f"set {set_index} must have exactly 3217 queries")
        control_index = payload["control"][0]
        require(control_index in control_indices, f"set {set_index} control index is not a control row")
        require(set(payload["perturb"]).issubset(query_indices), f"set {set_index} contains a non-query perturb row")
        control_id = clean_text(table.iloc[control_index]["sample_id"])
        linked_controls = set(clean_series(table.iloc[payload["perturb"]]["control"]).tolist())
        require(linked_controls == {control_id}, f"set {set_index} query control links do not match its control row")
        seen_controls.append(control_index)
        seen_queries.extend(payload["perturb"])
    require(sorted(seen_controls) == sorted(control_indices), "set_info does not cover every control exactly once")
    require(sorted(seen_queries) == sorted(query_indices), "set_info does not cover every query exactly once")

    row_to_set = {int(key): int(value) for key, value in load_pickle(split_dir / "row_to_set_index.pkl").items()}
    require(set(row_to_set) == set(range(EXPECTED_TOTAL_ROWS)), "row_to_set_index does not cover all feature rows")
    test_set_info = normalize_set_info(load_pickle(split_dir / "test_set_info_test_only.pkl"))
    require(test_set_info == set_info, "test_set_info_test_only differs from set_info")
    for split in ("train", "valid", "val"):
        require(load_pickle(split_dir / f"{split}_set_info_test_only.pkl") == {}, f"{split} set_info must be empty")

    manifest = load_json(split_dir / "split_manifest.json")
    require(manifest.get("strategy") == "test_only", f"{split_dir}: split manifest strategy is not test_only")
    manifest_test_count = manifest.get("test_anchor_count")
    if manifest_test_count is None:
        manifest_test_count = (manifest.get("anchor_counts") or {}).get("test")
    manifest_set_count = manifest.get("set_count")
    if manifest_set_count is None:
        manifest_set_count = (manifest.get("set_counts") or {}).get("test")
    require(int(manifest_test_count or -1) == EXPECTED_QUERY_ROWS, f"{split_dir}: test anchor count mismatch")
    require(int(manifest_set_count or -1) == EXPECTED_SAMPLE_COUNT, f"{split_dir}: split set count mismatch")


def validate_task(
    spec: TaskSpec,
    root: Path,
    checkpoint_axis: list[int],
    meta: dict[str, Any],
    scope: DrugScope,
    chunk_rows: int,
) -> TaskAudit:
    task_dir = root / "tasks" / spec.task_name
    split_dir = root / "splits" / spec.task_name
    require(task_dir.is_dir(), f"missing exp32 task directory: {task_dir}")
    require(split_dir.is_dir(), f"missing exp32 split directory: {split_dir}")
    table, table_path = find_table(task_dir)
    require(len(table) == EXPECTED_TOTAL_ROWS, f"{spec.task_name}: feature rows={len(table)}, expected 41834")
    required_columns = {
        "sample_id",
        "control",
        "is_control",
        "machineID_new",
        "machineID_new_index",
        "Cell_plate",
        "Cell_plate_index",
        "Cell",
        "Cell_index",
        "cell_type",
        "cell_type_index",
        "batch",
        "batch_index",
        "pert_id1",
        "pert_id2",
        "pert_index1",
        "pert_index2",
        "pert_time",
        "pert_time_index",
        "pert_dose1",
        "pert_dose1_index",
        "pert_dose2",
        "pert_dose2_index",
        "PRISM1st_label_total",
        "PRISM2nd_label_total",
        "synergy",
        "feature_row_index",
        "processed_row_index",
        "expression_row_index",
        "pat_ID",
    }
    missing_columns = sorted(required_columns - set(table.columns))
    require(not missing_columns, f"{spec.task_name}: missing feature columns {missing_columns}")
    is_control = bool_series(table["is_control"])
    control_indices = np.flatnonzero(is_control.to_numpy()).astype(int).tolist()
    query_indices = np.flatnonzero((~is_control).to_numpy()).astype(int).tolist()
    require(len(control_indices) == EXPECTED_CONTROL_ROWS, f"{spec.task_name}: control rows={len(control_indices)}, expected 13")
    require(len(query_indices) == EXPECTED_QUERY_ROWS, f"{spec.task_name}: query rows={len(query_indices)}, expected 41821")
    expected_positions = np.arange(len(table), dtype=np.int64)
    for column in ("feature_row_index", "processed_row_index", "expression_row_index"):
        actual = numeric_series(table[column], f"{spec.task_name}.{column}").astype(np.int64).to_numpy()
        require(np.array_equal(actual, expected_positions), f"{spec.task_name}: {column} is not row-position aligned")

    sample_column = next(
        (column for column in ("organoid_sample_id", "source_sample_id", "Cell") if column in table.columns),
        "Cell",
    )
    query = table.iloc[query_indices].copy()
    controls = table.iloc[control_indices].copy()
    sample_values = clean_series(query[sample_column])
    numbers = sample_values.map(sample_number).astype(int)
    expected_samples = {f"{spec.source_prefix}{number}" for number in range(1, EXPECTED_SAMPLE_COUNT + 1)}
    require(set(sample_values.tolist()) == expected_samples, f"{spec.task_name}: sample ID set differs from expected {spec.source_prefix}1..13")
    require(set(clean_series(controls[sample_column]).tolist()) == expected_samples, f"{spec.task_name}: control sample IDs differ from query samples")
    counts = query.assign(_sample=sample_values).groupby("_sample", sort=False).size()
    require(counts.eq(EXPECTED_DRUG_COUNT).all(), f"{spec.task_name}: every sample must have exactly 3217 queries")

    require(clean_series(query["machineID_new"]).eq(spec.machine).all(), f"{spec.task_name}: machineID_new must be {spec.machine}")
    require(numeric_series(query["machineID_new_index"], f"{spec.task_name}.machine index").eq(spec.machine_index).all(), f"{spec.task_name}: machine index mismatch")
    require(clean_series(query["Cell_plate"]).str.lower().eq("no").all(), f"{spec.task_name}: Cell_plate must be no")
    require(numeric_series(query["Cell_plate_index"], f"{spec.task_name}.plate index").eq(0).all(), f"{spec.task_name}: Cell_plate_index must be 0")
    require(
        clean_series(query["Cell"]).tolist() == sample_values.tolist(),
        f"{spec.task_name}: Cell must preserve the organoid sample ID",
    )
    require(numeric_series(query["Cell_index"], f"{spec.task_name}.Cell index").eq(0).all(), f"{spec.task_name}: Cell_index must be 0")
    for column in ("cell_llm_index", "Cell_llm_index"):
        if column in query.columns:
            require(numeric_series(query[column], f"{spec.task_name}.{column}").eq(0).all(), f"{spec.task_name}: {column} must be 0")
    require(clean_series(query["batch"]).eq(spec.batch).all(), f"{spec.task_name}: batch must be {spec.batch}")
    require(numeric_series(query["batch_index"], f"{spec.task_name}.batch index").eq(0).all(), f"{spec.task_name}: batch_index must be 0")

    tissues = clean_series(query["cell_type"]).str.upper()
    expected_tissues = numbers.map(expected_tissue).astype("string")
    require(tissues.reset_index(drop=True).equals(expected_tissues.reset_index(drop=True)), f"{spec.task_name}: cell_type does not match the PTV2 tissue map")
    tissue_indices = tissues.map({"LUNG": 8, "PANCREAS": 11, "COLON": 4}).astype(int)
    actual_tissue_indices = numeric_series(query["cell_type_index"], f"{spec.task_name}.cell_type index").astype(int)
    require(np.array_equal(tissue_indices.to_numpy(), actual_tissue_indices.to_numpy()), f"{spec.task_name}: cell_type_index mismatch")
    if "cell_type_llm_index" in query.columns:
        actual = numeric_series(query["cell_type_llm_index"], f"{spec.task_name}.cell_type LLM index").astype(int)
        require(np.array_equal(actual.to_numpy(), tissue_indices.to_numpy()), f"{spec.task_name}: cell_type_llm_index mismatch")

    pert1 = clean_series(query["pert_id1"])
    pert2 = clean_series(query["pert_id2"])
    require(pert1.equals(pert2), f"{spec.task_name}: single-drug rows must duplicate the drug in both slots")
    scope_set = set(scope.ids)
    for sample_id, group in query.assign(_sample=sample_values, _drug=pert1).groupby("_sample", sort=False):
        drugs = group["_drug"].tolist()
        require(len(drugs) == EXPECTED_DRUG_COUNT and len(set(drugs)) == EXPECTED_DRUG_COUNT, f"{spec.task_name}/{sample_id}: duplicate or missing drugs")
        require(set(drugs) == scope_set, f"{spec.task_name}/{sample_id}: drug set differs from exp32 scope")
    pert_index = {str(key): int(value) for key, value in meta["pert_index"].items()}
    expected_indices = pert1.map(pert_index)
    require(expected_indices.notna().all(), f"{spec.task_name}: query contains drug absent from pert_index")
    for column in ("pert_index1", "pert_index2"):
        actual = numeric_series(query[column], f"{spec.task_name}.{column}").astype(int)
        require(np.array_equal(actual.to_numpy(), expected_indices.astype(int).to_numpy()), f"{spec.task_name}: {column} does not match global pert_index")

    require(pd.to_numeric(query["pert_time"], errors="coerce").eq(24).all(), f"{spec.task_name}: pert_time must be 24")
    require(numeric_series(query["pert_time_index"], f"{spec.task_name}.time index").eq(2).all(), f"{spec.task_name}: pert_time_index must be 2")
    for slot in (1, 2):
        require(pd.to_numeric(query[f"pert_dose{slot}"], errors="coerce").eq(10).all(), f"{spec.task_name}: pert_dose{slot} must be 10")
        require(numeric_series(query[f"pert_dose{slot}_index"], f"{spec.task_name}.dose{slot} index").eq(10).all(), f"{spec.task_name}: pert_dose{slot}_index must be 10")
    if "prediction_type" in query.columns:
        require(clean_series(query["prediction_type"]).str.lower().eq("sensitivity").all(), f"{spec.task_name}: prediction_type must be sensitivity")
    require_empty_labels(
        table,
        query_indices,
        ("PRISM1st_label_total", "PRISM2nd_label_total", "synergy", "task_label", "response_label", "synergy_label"),
        spec.task_name,
    )

    pat_ids = clean_series(query["pat_ID"])
    require(pat_ids.ne("").all(), f"{spec.task_name}: pat_ID is incomplete")
    pat_by_sample = pd.DataFrame({"sample": sample_values, "pat": pat_ids}).drop_duplicates()
    require(len(pat_by_sample) == EXPECTED_SAMPLE_COUNT, f"{spec.task_name}: sample-to-pat_ID mapping is not one-to-one")
    require(pat_by_sample["pat"].nunique() == EXPECTED_SAMPLE_COUNT, f"{spec.task_name}: pat_ID values are not unique across 13 samples")

    axis = [int(item) for item in load_json(task_dir / "feature_ordered_protein_index.json")]
    require(axis == checkpoint_axis, f"{spec.task_name}: ordered protein axis differs from checkpoint axis")
    processed_axis = [int(item) for item in load_json(task_dir / "processed_ordered_protein_index.json")]
    require(processed_axis == axis, f"{spec.task_name}: processed and feature protein axes differ")
    sample_ids = [clean_text(item) for item in load_json(task_dir / "feature_sample_ids.json")]
    require(sample_ids == clean_series(table["sample_id"]).tolist(), f"{spec.task_name}: feature_sample_ids order differs from feature table")
    processed_sample_ids = [clean_text(item) for item in load_json(task_dir / "processed_sample_ids.json")]
    require(processed_sample_ids == sample_ids, f"{spec.task_name}: processed_sample_ids differ from feature_sample_ids")
    loading_manifest = load_json(task_dir / "feature_loading_manifest.json")
    require("inference" in str(loading_manifest.get("loading_contract", "")).lower(), f"{spec.task_name}: loading manifest does not declare inference-only contract")

    validate_splits(split_dir, table, control_indices, query_indices)
    matrix_audit = validate_matrix_pair(task_dir, table, control_indices, query_indices, chunk_rows)
    return TaskAudit(
        spec=spec,
        task_dir=task_dir,
        split_dir=split_dir,
        table_path=table_path,
        table=table,
        query_indices=query_indices,
        control_indices=control_indices,
        sample_column=sample_column,
        matrix_audit=matrix_audit,
    )


def preflight(args: argparse.Namespace) -> PreflightContext:
    root = group_root(args.training_ready_root)
    build_summary_path = args.build_summary or root / "exp32_organoid_build_summary.json"
    scope_json_path = args.drug_scope_json or root / "exp32_organoid_drug_scope.json"
    scope_csv_path = args.drug_scope_csv or root / "exp32_organoid_drug_scope.csv"
    build_summary = load_json(build_summary_path)
    require(isinstance(build_summary, dict), f"build summary is not a JSON object: {build_summary_path}")
    validate_build_summary_contract(
        build_summary,
        root=root,
        checkpoint_path=args.checkpoint_path,
        scope_json_path=scope_json_path,
        scope_csv_path=scope_csv_path,
    )
    serialized_summary = json.dumps(build_summary, ensure_ascii=False)
    for spec in TASK_SPECS:
        require(spec.task_name in serialized_summary, f"build summary does not mention task {spec.task_name}")
    for raw_path in args.raw_input_paths:
        require(raw_path.name in serialized_summary, f"build summary does not mention raw input {raw_path.name}")
    hash_audit = validate_hashes(build_summary, args.raw_input_paths)

    scope = load_drug_scope(scope_json_path, scope_csv_path)
    checkpoint_manifest_path, checkpoint_manifest, checkpoint_axis = validate_checkpoint_contract(args.checkpoint_path)
    meta_path = root / "global_meta.json"
    meta = validate_global_meta(meta_path, checkpoint_manifest, scope)
    tasks = {
        spec.task_name: validate_task(spec, root, checkpoint_axis, meta, scope, args.matrix_chunk_rows)
        for spec in TASK_SPECS
    }

    left = tasks[TASK_SPECS[0].task_name]
    right = tasks[TASK_SPECS[1].task_name]
    left_query = left.table.iloc[left.query_indices]
    right_query = right.table.iloc[right.query_indices]
    left_map = pd.DataFrame(
        {
            "sample_pair_id": clean_series(left_query[left.sample_column]).map(lambda value: f"PTV2_{sample_number(value)}"),
            "pat_ID": clean_series(left_query["pat_ID"]),
            "tissue": clean_series(left_query["cell_type"]).str.upper(),
        }
    ).drop_duplicates()
    right_map = pd.DataFrame(
        {
            "sample_pair_id": clean_series(right_query[right.sample_column]).map(lambda value: f"PTV2_{sample_number(value)}"),
            "pat_ID": clean_series(right_query["pat_ID"]),
            "tissue": clean_series(right_query["cell_type"]).str.upper(),
        }
    ).drop_duplicates()
    paired = left_map.merge(right_map, on="sample_pair_id", suffixes=("_b", "_cac"), validate="one_to_one")
    require(len(paired) == EXPECTED_SAMPLE_COUNT, "B/CAC sample metadata cannot form 13 one-to-one pairs")
    require(paired["pat_ID_b"].eq(paired["pat_ID_cac"]).all(), "B/CAC pat_ID mapping differs")
    require(paired["tissue_b"].eq(paired["tissue_cac"]).all(), "B/CAC tissue mapping differs")

    return PreflightContext(
        group_root=root,
        build_summary_path=build_summary_path,
        build_summary=build_summary,
        scope=scope,
        meta_path=meta_path,
        meta=meta,
        checkpoint_manifest_path=checkpoint_manifest_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_axis=checkpoint_axis,
        tasks=tasks,
        hash_audit=hash_audit,
    )


def config_values_match(key: str, current: Any, checkpoint: Any) -> bool:
    if key in PATH_CONFIG_KEYS:
        return same_path(current, checkpoint)
    if isinstance(current, (list, tuple)) or isinstance(checkpoint, (list, tuple)):
        try:
            return list(current) == list(checkpoint)
        except TypeError:
            return False
    if isinstance(current, float) or isinstance(checkpoint, float):
        try:
            return math.isclose(float(current), float(checkpoint), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return current == checkpoint


def mismatch_keys(records: Any) -> list[str]:
    require(isinstance(records, list), "checkpoint_config_mismatches must be a list")
    keys: list[str] = []
    for record in records:
        if isinstance(record, dict):
            key = clean_text(record.get("key"))
        else:
            key = clean_text(record).split(":", 1)[0]
        require(key, f"checkpoint mismatch record lacks a key: {record!r}")
        keys.append(key)
    return keys


def validate_inference_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    audit: TaskAudit,
    context: PreflightContext,
    *,
    smoke: bool,
    expected_rows: int,
) -> dict[str, Any]:
    require(clean_text(manifest.get("completed_at")), f"inference manifest lacks completed_at: {manifest_path}")
    require(manifest.get("dataset_group") == "ptv3", f"{manifest_path}: dataset_group mismatch")
    require(manifest.get("task_name") == audit.spec.task_name, f"{manifest_path}: task_name mismatch")
    require(manifest.get("model_type") == "fast_delta", f"{manifest_path}: model_type mismatch")
    require(
        same_path(manifest.get("checkpoint_path"), context.checkpoint_path),
        f"{manifest_path}: checkpoint is not the selected exp09 checkpoint ({context.checkpoint_path})",
    )
    require(same_path(manifest.get("task_dir"), audit.task_dir), f"{manifest_path}: task_dir mismatch")
    require(same_path(manifest.get("meta_path"), context.meta_path), f"{manifest_path}: meta_path mismatch")
    require(manifest.get("split_strategy") == "test_only", f"{manifest_path}: split_strategy must be test_only")
    require(manifest.get("split_name") == "test", f"{manifest_path}: split_name must be test")
    require(manifest.get("task_head") == "unified", f"{manifest_path}: task_head must be unified")
    require(manifest.get("task_label_key") == "synergy", f"{manifest_path}: task_label_key mismatch")
    require(manifest.get("task_mask_key") == "unified_label_mask", f"{manifest_path}: task_mask_key mismatch")
    require(manifest.get("task_label_policy") == "unified_synergy_first_else_response", f"{manifest_path}: task label policy mismatch")
    require(manifest.get("batch_cov_list") == EXPECTED_BATCH_COVARIATES, f"{manifest_path}: batch_cov_list mismatch")
    require(manifest.get("use_dose_covariate") is True, f"{manifest_path}: dose covariates must be enabled")
    require(manifest.get("dose_covariate_fields") == EXPECTED_DOSE_COVARIATES, f"{manifest_path}: dose fields mismatch")
    require(int(manifest.get("protein_axis_size", -1)) == EXPECTED_PROTEIN_COUNT, f"{manifest_path}: task protein axis size mismatch")
    require(int(manifest.get("checkpoint_protein_axis_size", -1)) == EXPECTED_PROTEIN_COUNT, f"{manifest_path}: checkpoint protein axis size mismatch")
    require(manifest.get("protein_axis_matches_checkpoint") is True, f"{manifest_path}: protein axis size flag is false")
    require(int(manifest.get("n_predictions", -1)) == expected_rows, f"{manifest_path}: n_predictions={manifest.get('n_predictions')}, expected {expected_rows}")
    require(manifest.get("save_expression_pred") is False, f"{manifest_path}: expression predictions must not be saved")
    require(manifest.get("allow_partial_checkpoint_load") is False, f"{manifest_path}: partial checkpoint loading is forbidden")
    require(manifest.get("allow_missing_checkpoint_manifest") is False, f"{manifest_path}: missing checkpoint manifest override is forbidden")
    require(manifest.get("allow_incomplete_checkpoint_manifest") is False, f"{manifest_path}: incomplete checkpoint override is forbidden")
    require(manifest.get("limit_batches") == (1 if smoke else None), f"{manifest_path}: unexpected limit_batches for {'smoke' if smoke else 'full'} run")
    require(manifest.get("graph_feature_mode") == "real", f"{manifest_path}: graph feature mode is not real")
    require(manifest.get("cell_llm_mode") == "frozen", f"{manifest_path}: Cell LLM is not frozen")
    require(manifest.get("cell_type_llm_mode") == "frozen", f"{manifest_path}: cell-type LLM is not frozen")
    require(manifest.get("cell_llm_summary", {}).get("index_column") == "Cell_index", f"{manifest_path}: Cell LLM index column mismatch")
    require(manifest.get("cell_type_llm_summary", {}).get("index_column") == "cell_type_index", f"{manifest_path}: cell-type LLM index column mismatch")

    model_config = manifest.get("inference_model_config")
    validation = manifest.get("checkpoint_config_validation")
    mismatches = manifest.get("checkpoint_config_mismatches")
    require(isinstance(model_config, dict) and model_config, f"{manifest_path}: missing inference_model_config")
    require(isinstance(validation, dict) and validation, f"{manifest_path}: missing checkpoint_config_validation")
    require(validation.get("inference_model_config") == model_config, f"{manifest_path}: nested/top-level inference model configs differ")
    require(validation.get("manifest_present") is True, f"{manifest_path}: checkpoint validation says manifest absent")
    require(validation.get("checkpoint_run_status") == "fit_completed", f"{manifest_path}: checkpoint validation status mismatch")
    require(manifest.get("checkpoint_architecture_matches") is True, f"{manifest_path}: checkpoint_architecture_matches must be true")
    require(validation.get("architecture_matches") is True, f"{manifest_path}: nested architecture_matches must be true")
    comparisons = validation.get("comparisons")
    require(isinstance(comparisons, list), f"{manifest_path}: checkpoint comparisons must be a list")
    require(int(validation.get("comparison_count", -1)) == len(comparisons), f"{manifest_path}: comparison_count mismatch")
    require(len(comparisons) == len(model_config), f"{manifest_path}: not every inference model config field was compared")
    require(all(isinstance(row, dict) and row.get("match") is True for row in comparisons if row.get("category") == "architecture"), f"{manifest_path}: architecture comparison contains a mismatch")
    normalized_checkpoint = validation.get("normalized_checkpoint_model_config")
    require(isinstance(normalized_checkpoint, dict), f"{manifest_path}: normalized checkpoint config missing")
    for key, current_value in model_config.items():
        if key in ALLOWED_CHECKPOINT_MISMATCH_KEYS:
            continue
        require(key in normalized_checkpoint, f"{manifest_path}: normalized checkpoint config lacks {key}")
        require(config_values_match(key, current_value, normalized_checkpoint[key]), f"{manifest_path}: model config {key} differs from checkpoint")

    keys = mismatch_keys(mismatches)
    nested_keys = mismatch_keys(validation.get("mismatches"))
    require(keys == nested_keys, f"{manifest_path}: top-level and nested mismatch records differ")
    require(int(validation.get("mismatch_count", -1)) == len(keys), f"{manifest_path}: mismatch_count mismatch")
    require(set(keys).issubset(ALLOWED_CHECKPOINT_MISMATCH_KEYS), f"{manifest_path}: forbidden checkpoint mismatch keys {sorted(set(keys) - ALLOWED_CHECKPOINT_MISMATCH_KEYS)}")
    for record in mismatches:
        if isinstance(record, dict):
            require(record.get("category") == "path", f"{manifest_path}: allowed mismatch is not categorized as path: {record}")
    require(manifest.get("checkpoint_config_matches") is (len(keys) == 0), f"{manifest_path}: checkpoint_config_matches inconsistent with mismatches")
    require(validation.get("matches") is (len(keys) == 0), f"{manifest_path}: nested matches inconsistent with mismatches")
    require(manifest.get("checkpoint_artifact_paths_match") is (len(keys) == 0), f"{manifest_path}: artifact path match flag inconsistent")
    require(validation.get("artifact_paths_match") is (len(keys) == 0), f"{manifest_path}: nested artifact path match flag inconsistent")
    require(manifest.get("allow_checkpoint_config_mismatch") is True, f"{manifest_path}: exp32 must explicitly allow only recorded path mismatches")
    return {"mismatch_keys": keys, "comparison_count": len(comparisons)}


def compare_prediction_feature_column(pred: pd.Series, feature: pd.Series, column: str, label: str) -> None:
    if column in {"feature_row_index", "pert_index1", "pert_index2", "pert_time", "pert_dose1", "pert_dose2"}:
        left = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(feature, errors="coerce").to_numpy(dtype=np.float64)
        require(np.array_equal(left, right, equal_nan=True), f"{label}: prediction {column} differs from feature table")
    else:
        require(clean_series(pred).equals(clean_series(feature)), f"{label}: prediction {column} differs from feature table")


def validate_predictions(
    audit: TaskAudit,
    context: PreflightContext,
    output_dir: Path,
    *,
    smoke: bool,
    expected_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = output_dir / "run_manifest.json"
    manifest = load_json(manifest_path)
    require(isinstance(manifest, dict), f"inference manifest is not an object: {manifest_path}")
    manifest_audit = validate_inference_manifest(
        manifest,
        manifest_path,
        audit,
        context,
        smoke=smoke,
        expected_rows=expected_rows,
    )
    predictions, prediction_path = load_predictions(output_dir)
    require(len(predictions) == expected_rows, f"{audit.spec.task_name}: prediction rows={len(predictions)}, expected {expected_rows}")
    required = {
        "feature_row_index",
        "sample_id",
        "control",
        "Cell",
        "cell_type",
        "pert_id1",
        "pert_id2",
        "pert_index1",
        "pert_index2",
        "pred_task_prob",
        "pred_response_prob",
        "pred_synergy_prob",
        "task_label",
        "response_label",
        "synergy_label",
        "pert_time",
        "pert_dose1",
        "pert_dose2",
        "batch",
    }
    missing = sorted(required - set(predictions.columns))
    require(not missing, f"{audit.spec.task_name}: predictions missing columns {missing}")
    row_indices = pd.to_numeric(predictions["feature_row_index"], errors="coerce")
    require(row_indices.notna().all(), f"{audit.spec.task_name}: invalid feature_row_index in predictions")
    row_indices_list = row_indices.astype(int).tolist()
    expected_indices = audit.query_indices[:expected_rows] if smoke else audit.query_indices
    require(row_indices_list == expected_indices, f"{audit.spec.task_name}: prediction rows are not the expected test split order")
    require(len(set(row_indices_list)) == expected_rows, f"{audit.spec.task_name}: duplicate prediction feature rows")

    probability = pd.to_numeric(predictions["pred_task_prob"], errors="coerce").to_numpy(dtype=np.float64)
    require(np.isfinite(probability).all(), f"{audit.spec.task_name}: pred_task_prob contains non-finite values")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), f"{audit.spec.task_name}: pred_task_prob outside [0,1]")
    for duplicate in ("pred_response_prob", "pred_synergy_prob"):
        values = pd.to_numeric(predictions[duplicate], errors="coerce").to_numpy(dtype=np.float64)
        require(np.array_equal(values, probability), f"{audit.spec.task_name}: unified {duplicate} does not duplicate pred_task_prob")
    require_empty_labels(predictions, list(range(len(predictions))), ("task_label", "response_label", "synergy_label"), audit.spec.task_name)

    feature_rows = audit.table.iloc[row_indices_list].reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    for column in (
        "sample_id",
        "control",
        "Cell",
        "cell_type",
        "pert_id1",
        "pert_id2",
        "pert_index1",
        "pert_index2",
        "pert_time",
        "pert_dose1",
        "pert_dose2",
        "batch",
    ):
        if column in predictions.columns and column in feature_rows.columns:
            compare_prediction_feature_column(predictions[column], feature_rows[column], column, audit.spec.task_name)

    enriched = feature_rows.copy()
    enriched["inference_sample_id"] = clean_series(predictions["sample_id"])
    enriched["pred_task_prob"] = probability
    return enriched, {
        "manifest_path": str(manifest_path),
        "prediction_path": str(prediction_path),
        "rows": len(predictions),
        **manifest_audit,
    }


def build_combined_frame(enriched_by_task: dict[str, pd.DataFrame], context: PreflightContext) -> pd.DataFrame:
    scope_smiles = {str(key): clean_text(value) for key, value in context.meta.get("pertid_to_smiles", {}).items()}
    scope_order = {pert_id: index for index, pert_id in enumerate(context.scope.ids)}
    frames: list[pd.DataFrame] = []
    for spec in TASK_SPECS:
        audit = context.tasks[spec.task_name]
        table = enriched_by_task[spec.task_name]
        sample_ids = clean_series(table[audit.sample_column])
        numbers = sample_ids.map(sample_number).astype(int)
        cell_type = clean_series(table["cell_type"]).str.upper()
        tissue_source = clean_series(table["tissue"]) if "tissue" in table.columns else cell_type.str.title()
        drug_ids = clean_series(table["pert_id1"])
        output = pd.DataFrame(
            {
                "device": spec.device,
                "task_name": spec.task_name,
                "feature_row_index": pd.to_numeric(table["feature_row_index"], errors="raise").astype(int),
                "inference_sample_id": clean_series(table["inference_sample_id"]),
                "sample_id": sample_ids,
                "sample_pair_id": numbers.map(lambda number: f"PTV2_{number}"),
                "samp_ID": table["samp_ID"].tolist() if "samp_ID" in table.columns else numbers.tolist(),
                "pat_ID": clean_series(table["pat_ID"]),
                "tissue": tissue_source,
                "cell_type": cell_type,
                "machineID_new": clean_series(table["machineID_new"]),
                "drug_id": drug_ids,
                "smiles": drug_ids.map(scope_smiles),
                "pert_time": pd.to_numeric(table["pert_time"], errors="raise"),
                "pert_dose1": pd.to_numeric(table["pert_dose1"], errors="raise"),
                "pert_dose2": pd.to_numeric(table["pert_dose2"], errors="raise"),
                "prediction_type": "sensitivity",
                "pred_sensitivity_prob": pd.to_numeric(table["pred_task_prob"], errors="raise"),
                "_sample_number": numbers,
                "_scope_order": drug_ids.map(scope_order),
            }
        )
        require(output["smiles"].map(clean_text).ne("").all(), f"{spec.task_name}: final output has missing SMILES")
        require(output["_scope_order"].notna().all(), f"{spec.task_name}: final output contains a drug outside scope")
        frames.append(output)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["device", "_sample_number", "_scope_order"], kind="mergesort").reset_index(drop=True)
    require(len(combined) == EXPECTED_QUERY_ROWS * 2, f"combined prediction rows={len(combined)}, expected 83642")
    require(not combined.duplicated(["device", "sample_pair_id", "drug_id"]).any(), "combined predictions contain duplicate device/sample/drug keys")
    return combined.drop(columns=["_sample_number", "_scope_order"])


def probability_distribution(values: pd.Series) -> dict[str, Any]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "probability distribution received empty/non-finite values")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def correlation(left: pd.Series, right: pd.Series) -> tuple[float, float, float]:
    x = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
    require(len(x) == len(y) and len(x) > 1, "correlation arrays are not aligned")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "correlation arrays contain non-finite values")
    require(float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0, "correlation is undefined for a constant prediction vector")
    pearson = float(np.corrcoef(x, y)[0, 1])
    rank_x = pd.Series(x).rank(method="average")
    rank_y = pd.Series(y).rank(method="average")
    spearman = float(rank_x.corr(rank_y, method="pearson"))
    mae = float(np.mean(np.abs(x - y)))
    require(math.isfinite(pearson) and math.isfinite(spearman), "device correlation is non-finite")
    return pearson, spearman, mae


def build_analysis(combined: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    sample_distributions: list[dict[str, Any]] = []
    for (device, sample_pair_id), group in combined.groupby(["device", "sample_pair_id"], sort=True):
        row = group.iloc[0]
        sample_distributions.append(
            {
                "device": device,
                "sample_pair_id": sample_pair_id,
                "sample_id": row["sample_id"],
                "pat_ID": row["pat_ID"],
                "tissue": row["tissue"],
                **probability_distribution(group["pred_sensitivity_prob"]),
            }
        )
    device_distributions = [
        {"device": device, **probability_distribution(group["pred_sensitivity_prob"])}
        for device, group in combined.groupby("device", sort=True)
    ]

    sorted_predictions = combined.sort_values(
        ["device", "sample_pair_id", "pred_sensitivity_prob", "drug_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top20 = sorted_predictions.groupby(["device", "sample_pair_id"], sort=False).head(20).copy()
    top20["rank"] = top20.groupby(["device", "sample_pair_id"], sort=False).cumcount() + 1
    top20 = top20[
        [
            "device",
            "sample_pair_id",
            "sample_id",
            "pat_ID",
            "tissue",
            "rank",
            "drug_id",
            "smiles",
            "pert_time",
            "pert_dose1",
            "pert_dose2",
            "pred_sensitivity_prob",
        ]
    ]
    require(len(top20) == 2 * EXPECTED_SAMPLE_COUNT * 20, f"top-20 table rows={len(top20)}, expected 520")

    b = combined.loc[combined["device"].eq("B")].copy()
    cac = combined.loc[combined["device"].eq("CAC")].copy()
    paired = b.merge(
        cac,
        on=["sample_pair_id", "drug_id"],
        how="inner",
        suffixes=("_b", "_cac"),
        validate="one_to_one",
    )
    require(len(paired) == EXPECTED_QUERY_ROWS, f"B/CAC paired rows={len(paired)}, expected 41821")
    require(paired["pat_ID_b"].eq(paired["pat_ID_cac"]).all(), "paired predictions have inconsistent pat_ID")
    require(paired["cell_type_b"].eq(paired["cell_type_cac"]).all(), "paired predictions have inconsistent tissue")
    paired["delta_cac_minus_b"] = paired["pred_sensitivity_prob_cac"] - paired["pred_sensitivity_prob_b"]
    paired["abs_delta"] = paired["delta_cac_minus_b"].abs()

    agreement: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    for sample_pair_id, group in paired.groupby("sample_pair_id", sort=True):
        require(len(group) == EXPECTED_DRUG_COUNT, f"{sample_pair_id}: paired drug rows != 3217")
        pearson, spearman, mae = correlation(group["pred_sensitivity_prob_b"], group["pred_sensitivity_prob_cac"])
        agreement.append(
            {
                "sample_pair_id": sample_pair_id,
                "pat_ID": group.iloc[0]["pat_ID_b"],
                "tissue": group.iloc[0]["tissue_b"],
                "count": len(group),
                "pearson": pearson,
                "spearman": spearman,
                "mae": mae,
                "mean_delta_cac_minus_b": float(group["delta_cac_minus_b"].mean()),
            }
        )
        for k in (50, 100):
            top_b = set(group.nlargest(k, "pred_sensitivity_prob_b", keep="all").sort_values(["pred_sensitivity_prob_b", "drug_id"], ascending=[False, True]).head(k)["drug_id"])
            top_cac = set(group.nlargest(k, "pred_sensitivity_prob_cac", keep="all").sort_values(["pred_sensitivity_prob_cac", "drug_id"], ascending=[False, True]).head(k)["drug_id"])
            overlap = len(top_b & top_cac)
            union = len(top_b | top_cac)
            overlaps.append(
                {
                    "sample_pair_id": sample_pair_id,
                    "k": k,
                    "overlap_count": overlap,
                    "overlap_fraction": float(overlap / k),
                    "jaccard": float(overlap / union),
                }
            )
    overall_pearson, overall_spearman, overall_mae = correlation(
        paired["pred_sensitivity_prob_b"], paired["pred_sensitivity_prob_cac"]
    )
    shift = {
        "definition": "CAC minus B on the exact 41,821 paired sample-drug rows",
        "paired_count": len(paired),
        "mean_delta": float(paired["delta_cac_minus_b"].mean()),
        "median_delta": float(paired["delta_cac_minus_b"].median()),
        "mean_absolute_delta": float(paired["abs_delta"].mean()),
        "p90_absolute_delta": float(paired["abs_delta"].quantile(0.90)),
        "p95_absolute_delta": float(paired["abs_delta"].quantile(0.95)),
        "overall_pearson": overall_pearson,
        "overall_spearman": overall_spearman,
        "overall_mae": overall_mae,
    }
    differential = (
        paired.groupby("drug_id", sort=False)
        .agg(
            mean_b=("pred_sensitivity_prob_b", "mean"),
            mean_cac=("pred_sensitivity_prob_cac", "mean"),
            mean_delta_cac_minus_b=("delta_cac_minus_b", "mean"),
            mean_absolute_delta=("abs_delta", "mean"),
            max_absolute_delta=("abs_delta", "max"),
        )
        .reset_index()
    )
    differential["smiles"] = differential["drug_id"].map(
        paired.drop_duplicates("drug_id").set_index("drug_id")["smiles_b"].to_dict()
    )
    differential = differential.sort_values(
        ["mean_absolute_delta", "max_absolute_delta", "drug_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    largest_differences = differential.head(20).to_dict("records")
    return (
        {
            "device_distributions": device_distributions,
            "sample_distributions": sample_distributions,
            "device_agreement_by_sample": agreement,
            "top_overlap_by_sample": overlaps,
            "overall_device_shift": shift,
            "largest_device_difference_drugs": largest_differences,
        },
        top20,
    )


def fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.{digits}f}" if math.isfinite(number) else ""


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    device_rows = [
        [
            row["device"],
            str(row["count"]),
            fmt(row["mean"]),
            fmt(row["median"]),
            fmt(row["p90"]),
            fmt(row["p95"]),
            fmt(row["min"]),
            fmt(row["max"]),
        ]
        for row in analysis["device_distributions"]
    ]
    sample_rows = [
        [
            row["device"],
            row["sample_pair_id"],
            row["pat_ID"],
            row["tissue"],
            fmt(row["mean"]),
            fmt(row["median"]),
            fmt(row["p90"]),
            fmt(row["p95"]),
        ]
        for row in analysis["sample_distributions"]
    ]
    agreement_rows = [
        [
            row["sample_pair_id"],
            row["pat_ID"],
            row["tissue"],
            fmt(row["pearson"]),
            fmt(row["spearman"]),
            fmt(row["mae"]),
            fmt(row["mean_delta_cac_minus_b"]),
        ]
        for row in analysis["device_agreement_by_sample"]
    ]
    overlap_lookup: dict[str, dict[int, dict[str, Any]]] = {}
    for row in analysis["top_overlap_by_sample"]:
        overlap_lookup.setdefault(row["sample_pair_id"], {})[int(row["k"])] = row
    overlap_rows = []
    for sample_pair_id, by_k in sorted(overlap_lookup.items(), key=lambda item: sample_number(item[0])):
        k50, k100 = by_k[50], by_k[100]
        overlap_rows.append(
            [
                sample_pair_id,
                str(k50["overlap_count"]),
                fmt(k50["jaccard"]),
                str(k100["overlap_count"]),
                fmt(k100["jaccard"]),
            ]
        )
    difference_rows = [
        [
            str(index + 1),
            row["drug_id"],
            fmt(row["mean_b"]),
            fmt(row["mean_cac"]),
            fmt(row["mean_delta_cac_minus_b"]),
            fmt(row["mean_absolute_delta"]),
            fmt(row["max_absolute_delta"]),
        ]
        for index, row in enumerate(analysis["largest_device_difference_drugs"])
    ]
    shift = analysis["overall_device_shift"]
    lines = [
        "# Exp32 Organoid Exp09 Single-drug Sensitivity Results",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Experiment prefix: `{summary['exp_prefix']}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        "- Query: 3,217 single drugs per sample, represented by identical drug slots.",
        "- Exposure: 24 hours; both dose slots are 10.",
        "- Score: `pred_sensitivity_prob`, taken only from the unified head's `pred_task_prob`.",
        "- This experiment has no ground-truth response labels, so no label-based performance metrics are computed.",
        "",
        "## Validation",
        "",
        "- Both tasks completed with exactly 41,821 query predictions; combined rows: 83,642.",
        "- All probabilities are finite and within `[0, 1]`.",
        "- The two devices form exactly 41,821 one-to-one sample-drug pairs.",
        "- Protein axis, drug scope, feature indices, covariates, hashes, and exp09 architecture contract passed reporter validation.",
        "",
        "## Artifacts",
        "",
        f"- Combined CSV: `{summary['outputs']['predictions_csv']}`",
        f"- Combined Parquet: `{summary['outputs']['predictions_parquet']}`",
        f"- Per-sample top-20 CSV: `{summary['outputs']['top20_csv']}`",
        f"- Summary JSON: `{summary['outputs']['summary_json']}`",
        "",
        "## Device Probability Distributions",
        "",
        markdown_table(["device", "n", "mean", "median", "P90", "P95", "min", "max"], device_rows),
        "",
        "## Sample Probability Distributions",
        "",
        markdown_table(["device", "sample", "pat_ID", "tissue", "mean", "median", "P90", "P95"], sample_rows),
        "",
        "## B versus CAC Agreement",
        "",
        f"Overall paired Pearson: `{fmt(shift['overall_pearson'])}`; Spearman: `{fmt(shift['overall_spearman'])}`; MAE: `{fmt(shift['overall_mae'])}`.",
        "",
        markdown_table(["sample", "pat_ID", "tissue", "Pearson", "Spearman", "MAE", "mean CAC-B"], agreement_rows),
        "",
        "## Top-drug Agreement",
        "",
        markdown_table(["sample", "top-50 overlap", "top-50 Jaccard", "top-100 overlap", "top-100 Jaccard"], overlap_rows),
        "",
        "## Overall Device Shift",
        "",
        f"The signed difference is defined as CAC minus B over `{shift['paired_count']}` exact pairs.",
        "",
        f"- Mean signed difference: `{fmt(shift['mean_delta'])}`",
        f"- Median signed difference: `{fmt(shift['median_delta'])}`",
        f"- Mean absolute difference: `{fmt(shift['mean_absolute_delta'])}`",
        f"- P90/P95 absolute difference: `{fmt(shift['p90_absolute_delta'])}` / `{fmt(shift['p95_absolute_delta'])}`",
        "",
        "## Drugs With Largest Device Differences",
        "",
        markdown_table(["rank", "drug", "mean B", "mean CAC", "mean CAC-B", "mean abs diff", "max abs diff"], difference_rows),
        "",
        "## Interpretation Notes",
        "",
        "- Higher scores indicate higher predicted sensitivity under the standardized exposure condition.",
        "- B and CAC are retained as separate instrument measurements; predictions are not averaged.",
        "- Applying 24 hours and dose 10 to all 3,217 drugs is an intentional standardized extrapolation, including drugs without that exact historical condition.",
        "- The top-20 CSV contains 20 ranked drugs for each of 13 samples on each device (520 rows).",
        "",
    ]
    return "\n".join(lines)


def default_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = args.exp_prefix
    return {
        "predictions_csv": args.predictions_csv or args.output_root / f"{base}_predictions.csv",
        "predictions_parquet": args.predictions_parquet or args.output_root / f"{base}_predictions.parquet",
        "summary_json": args.summary_json or args.output_root / f"{base}_summary.json",
        "top20_csv": args.top20_csv or args.output_root / f"{base}_top20_by_sample.csv",
        "markdown": args.output_markdown or DEFAULT_RESULT_MARKDOWN,
    }


def run_results(args: argparse.Namespace, context: PreflightContext, *, smoke: bool) -> None:
    expected_rows = args.smoke_expected_rows if smoke else EXPECTED_QUERY_ROWS
    enriched_by_task: dict[str, pd.DataFrame] = {}
    task_results: list[dict[str, Any]] = []
    for spec in TASK_SPECS:
        audit = context.tasks[spec.task_name]
        output_dir = args.output_root / args.exp_prefix / spec.task_name
        enriched, result = validate_predictions(
            audit,
            context,
            output_dir,
            smoke=smoke,
            expected_rows=expected_rows,
        )
        enriched_by_task[spec.task_name] = enriched
        task_results.append({"task_name": spec.task_name, "device": spec.device, "output_dir": str(output_dir), **result})
    if smoke:
        print(f"[smoke] ok: {len(TASK_SPECS)} tasks x {expected_rows} predictions")
        return

    combined = build_combined_frame(enriched_by_task, context)
    analysis, top20 = build_analysis(combined)
    outputs = default_output_paths(args)
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outputs["predictions_csv"], index=False)
    combined.to_parquet(outputs["predictions_parquet"], index=False)
    top20.to_csv(outputs["top20_csv"], index=False)
    summary = {
        "generated_at": iso_now(),
        "generated_at_hkt": hkt_now(),
        "status": "complete",
        "exp_prefix": args.exp_prefix,
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_manifest_path": str(context.checkpoint_manifest_path),
        "training_ready_root": str(args.training_ready_root),
        "build_summary_path": str(context.build_summary_path),
        "drug_scope_count": len(context.scope.ids),
        "protein_axis_count": len(context.checkpoint_axis),
        "prediction_semantics": {
            "prediction_type": "sensitivity",
            "source_column": "pred_task_prob",
            "output_column": "pred_sensitivity_prob",
            "single_drug_slot_policy": "pert_id1 equals pert_id2",
            "pert_time": 24,
            "pert_dose1": 10,
            "pert_dose2": 10,
            "has_ground_truth_labels": False,
        },
        "task_results": task_results,
        "data_preflight": {
            "hash_audit": context.hash_audit,
            "tasks": [
                {
                    "task_name": audit.spec.task_name,
                    "device": audit.spec.device,
                    "feature_table_path": str(audit.table_path),
                    "feature_rows": len(audit.table),
                    "control_rows": len(audit.control_indices),
                    "query_rows": len(audit.query_indices),
                    "matrix": audit.matrix_audit,
                }
                for audit in context.tasks.values()
            ],
        },
        "validation": {
            "rows_per_task": EXPECTED_QUERY_ROWS,
            "combined_rows": len(combined),
            "paired_rows": EXPECTED_QUERY_ROWS,
            "raw_and_source_hash_records_validated": len(context.hash_audit),
            "probabilities_finite_and_bounded": True,
            "exact_protein_axis_match": True,
            "architecture_matches_checkpoint": True,
        },
        "analysis": analysis,
        "outputs": {key: repo_rel(path) for key, path in outputs.items()},
    }
    dump_json(outputs["summary_json"], summary)
    outputs["markdown"].write_text(render_markdown(summary), encoding="utf-8")
    print(f"[report] wrote {outputs['predictions_csv']}")
    print(f"[report] wrote {outputs['predictions_parquet']}")
    print(f"[report] wrote {outputs['top20_csv']}")
    print(f"[report] wrote {outputs['summary_json']}")
    print(f"[report] wrote {outputs['markdown']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-prefix", default=DEFAULT_EXP_PREFIX)
    parser.add_argument("--training-ready-root", type=Path, default=DEFAULT_TRAINING_READY_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--build-summary", type=Path)
    parser.add_argument("--drug-scope-json", type=Path)
    parser.add_argument("--drug-scope-csv", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--predictions-parquet", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--top20-csv", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-expected-rows", type=int, default=256)
    parser.add_argument("--matrix-chunk-rows", type=int, default=256)
    parser.add_argument("--raw-input-paths", nargs="*", type=Path, default=list(RAW_INPUT_PATHS))
    args = parser.parse_args()
    require(not (args.preflight_only and args.smoke), "--preflight-only and --smoke are mutually exclusive")
    require(args.smoke_expected_rows > 0, "--smoke-expected-rows must be positive")
    require(args.matrix_chunk_rows > 0, "--matrix-chunk-rows must be positive")
    args.training_ready_root = args.training_ready_root.expanduser()
    args.checkpoint_path = args.checkpoint_path.expanduser()
    args.output_root = (args.output_root or output_bucket_for_prefix(args.exp_prefix)).expanduser()
    args.raw_input_paths = [path.expanduser() for path in args.raw_input_paths]
    return args


def main() -> int:
    args = parse_args()
    try:
        context = preflight(args)
        print(
            "[preflight] ok: "
            f"{len(context.tasks)} tasks, {len(context.scope.ids)} drugs, "
            f"{len(context.checkpoint_axis)} proteins, {len(context.hash_audit)} hash records"
        )
        if args.preflight_only:
            return 0
        run_results(args, context, smoke=args.smoke)
        return 0
    except (ValidationError, FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f"[error] {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
