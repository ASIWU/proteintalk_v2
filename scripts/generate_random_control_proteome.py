#!/usr/bin/env python3
"""Generate saved random control-expression artifacts for a training-ready task."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.training_ready_fast_dataset import load_feature_table
from utils.npy_io import safe_np_load


DEFAULT_POLICY = "per_protein_normal_clip"
POLICY_DESCRIPTIONS = {
    "per_protein_normal_clip": (
        "sample each protein independently from control-row per-protein mean/std, "
        "then clip negatives to zero; this matches the original seed42 random control artifact"
    ),
    "global_normal_clip": (
        "sample every value from the same global control-expression normal distribution, "
        "then clip negatives to zero"
    ),
    "global_value_bootstrap": (
        "sample every value independently from the pooled finite values observed in real control rows"
    ),
    "fixed_gene_permutation": (
        "generate per-protein normal samples, then apply one fixed random gene-column permutation to all rows"
    ),
    "per_row_gene_permutation": (
        "generate per-protein normal samples, then independently permute gene columns in every row"
    ),
    "cross_cell_real_control": (
        "replace each row with a real control-expression row sampled from a different Cell when possible"
    ),
    "zero_control": "write an all-zero diagnostic matrix; excluded from winner selection",
}
POLICIES = tuple(POLICY_DESCRIPTIONS)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_bool_series(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(dtype=float) != 0.0
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"}).to_numpy(dtype=bool)


def output_path_for(task_dir: Path, seed: int, policy: str) -> Path:
    if policy == DEFAULT_POLICY:
        return task_dir / f"random_control_expression_seed{seed}.npy"
    return task_dir / f"random_control_expression_{policy}_seed{seed}.npy"


def meta_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".meta.json")


def finite_summary(values: np.ndarray, *, chunk_size: int = 1024) -> dict[str, float | int | None]:
    arr = np.asarray(values)
    if arr.size == 0:
        return {"finite_count": 0, "mean": None, "std": None, "min": None, "max": None}

    finite_count = 0
    total = 0.0
    total_sq = 0.0
    min_value = math.inf
    max_value = -math.inf

    if arr.ndim == 0:
        iterator = [arr.reshape(1)]
    elif arr.ndim == 1:
        iterator = (arr[start : start + chunk_size] for start in range(0, arr.shape[0], chunk_size))
    else:
        iterator = (arr[start : start + chunk_size] for start in range(0, arr.shape[0], chunk_size))

    for chunk in iterator:
        finite = np.asarray(chunk)[np.isfinite(chunk)]
        if finite.size == 0:
            continue
        finite64 = finite.astype(np.float64, copy=False)
        finite_count += int(finite64.size)
        total += float(finite64.sum())
        total_sq += float(np.square(finite64).sum())
        min_value = min(min_value, float(finite64.min()))
        max_value = max(max_value, float(finite64.max()))

    if finite_count == 0:
        return {"finite_count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = total / finite_count
    variance = max(0.0, total_sq / finite_count - mean * mean)
    return {
        "finite_count": finite_count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min_value,
        "max": max_value,
    }


def control_statistics(control_matrix: np.ndarray) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        protein_mean = np.nanmean(control_matrix, axis=0).astype(np.float32, copy=False)
        protein_std = np.nanstd(control_matrix, axis=0).astype(np.float32, copy=False)
        global_mean = float(np.nanmean(control_matrix))
        global_std = float(np.nanstd(control_matrix))

    if not np.isfinite(global_mean):
        global_mean = 0.0
    if not np.isfinite(global_std) or global_std <= 0.0:
        global_std = 1.0

    mean_fallback = ~np.isfinite(protein_mean)
    std_fallback = (~np.isfinite(protein_std)) | (protein_std <= 0.0)
    protein_mean = protein_mean.copy()
    protein_std = protein_std.copy()
    protein_mean[mean_fallback] = np.float32(global_mean)
    protein_std[std_fallback] = np.float32(global_std)
    return {
        "protein_mean": protein_mean,
        "protein_std": protein_std,
        "global_mean": global_mean,
        "global_std": global_std,
        "mean_fallback": mean_fallback,
        "std_fallback": std_fallback,
    }


def sample_per_protein_normal(
    *,
    rng: np.random.Generator,
    row_count: int,
    protein_mean: np.ndarray,
    protein_std: np.ndarray,
) -> tuple[np.ndarray, int]:
    sampled = rng.normal(loc=protein_mean, scale=protein_std, size=(row_count, protein_mean.shape[0]))
    sampled = sampled.astype(np.float32, copy=False)
    negatives = sampled < 0.0
    clipped_negative_count = int(negatives.sum())
    sampled[negatives] = 0.0
    return sampled, clipped_negative_count


def cell_values(feature_table: pd.DataFrame) -> tuple[np.ndarray, str | None]:
    for column in ("Cell", "Cell_norm", "Cell_index", "cell_type", "cell_type_index"):
        if column in feature_table.columns:
            values = feature_table[column].fillna("__missing__").astype(str).to_numpy()
            return values, column
    return np.full(len(feature_table), "__missing__", dtype=object), None


def cross_cell_choice_index(
    *,
    feature_table: pd.DataFrame,
    control_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    cells, cell_column = cell_values(feature_table)
    all_controls = np.flatnonzero(control_mask)
    if all_controls.size == 0:
        raise ValueError("cross_cell_real_control requires at least one control row")

    eligible_by_cell: dict[str, np.ndarray] = {}
    fallback_same_cell_count = 0
    for raw_cell in np.unique(cells):
        cell = str(raw_cell)
        different = all_controls[cells[all_controls] != cell]
        if different.size == 0:
            eligible_by_cell[cell] = all_controls
            fallback_same_cell_count += int((cells == cell).sum())
        else:
            eligible_by_cell[cell] = different

    chosen = np.empty(len(feature_table), dtype=np.int64)
    for row_idx, raw_cell in enumerate(cells):
        eligible = eligible_by_cell.get(str(raw_cell), all_controls)
        chosen[row_idx] = int(rng.choice(eligible))

    meta = {
        "cross_cell_control_cell_column": cell_column,
        "cross_cell_unique_cell_count": int(len(eligible_by_cell)),
        "cross_cell_fallback_same_cell_row_count": int(fallback_same_cell_count),
    }
    return chosen, meta


def generate_matrix(
    *,
    policy: str,
    rng: np.random.Generator,
    expression_matrix: np.ndarray,
    control_matrix: np.ndarray,
    feature_table: pd.DataFrame,
    control_mask: np.ndarray,
    stats: dict[str, Any],
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    row_count, protein_count = int(expression_matrix.shape[0]), int(expression_matrix.shape[1])
    random_matrix = np.empty((row_count, protein_count), dtype=np.float32)
    meta: dict[str, Any] = {"clipped_negative_count": 0}

    if policy == "zero_control":
        random_matrix.fill(0.0)
        return random_matrix, meta

    if policy == "global_value_bootstrap":
        finite_values = np.asarray(control_matrix[np.isfinite(control_matrix)], dtype=np.float32)
        if finite_values.size == 0:
            raise ValueError("global_value_bootstrap requires finite control-expression values")
        meta["global_bootstrap_value_count"] = int(finite_values.size)
        for start in range(0, row_count, chunk_size):
            end = min(row_count, start + chunk_size)
            sample_index = rng.integers(0, finite_values.size, size=(end - start, protein_count), dtype=np.int64)
            random_matrix[start:end] = finite_values[sample_index]
        return random_matrix, meta

    if policy == "cross_cell_real_control":
        chosen_rows, choice_meta = cross_cell_choice_index(
            feature_table=feature_table,
            control_mask=control_mask,
            rng=rng,
        )
        meta.update(choice_meta)
        for start in range(0, row_count, chunk_size):
            end = min(row_count, start + chunk_size)
            random_matrix[start:end] = np.asarray(expression_matrix[chosen_rows[start:end]], dtype=np.float32)
        return random_matrix, meta

    fixed_permutation: np.ndarray | None = None
    if policy == "fixed_gene_permutation":
        fixed_permutation = rng.permutation(protein_count)
        meta["fixed_gene_permutation_head"] = fixed_permutation[:20].astype(int).tolist()

    for start in range(0, row_count, chunk_size):
        end = min(row_count, start + chunk_size)
        n_rows = end - start
        if policy == "global_normal_clip":
            sampled = rng.normal(
                loc=stats["global_mean"],
                scale=stats["global_std"],
                size=(n_rows, protein_count),
            )
            sampled = sampled.astype(np.float32, copy=False)
            negatives = sampled < 0.0
            meta["clipped_negative_count"] += int(negatives.sum())
            sampled[negatives] = 0.0
        elif policy in {"per_protein_normal_clip", "fixed_gene_permutation", "per_row_gene_permutation"}:
            sampled, clipped = sample_per_protein_normal(
                rng=rng,
                row_count=n_rows,
                protein_mean=stats["protein_mean"],
                protein_std=stats["protein_std"],
            )
            meta["clipped_negative_count"] += clipped
        else:
            raise ValueError(f"unsupported random control-expression policy: {policy}")

        if fixed_permutation is not None:
            sampled = sampled[:, fixed_permutation]
        elif policy == "per_row_gene_permutation":
            for local_row in range(n_rows):
                sampled[local_row] = sampled[local_row, rng.permutation(protein_count)]
        random_matrix[start:end] = sampled

    return random_matrix, meta


def generate_random_control(args: argparse.Namespace) -> dict[str, Any]:
    task_dir = Path(args.task_dir)
    if not task_dir.exists():
        raise FileNotFoundError(f"missing task directory: {task_dir}")

    policy = str(args.policy).strip().lower()
    if policy not in POLICY_DESCRIPTIONS:
        raise ValueError(f"unsupported --policy={args.policy!r}; choose one of: {', '.join(POLICIES)}")

    expression_path = task_dir / "feature_expression_matrix.npy"
    if not expression_path.exists():
        raise FileNotFoundError(f"missing expression matrix: {expression_path}")

    output_path = Path(args.output_path) if args.output_path else output_path_for(task_dir, args.seed, policy)
    meta_path = meta_path_for(output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite to replace it")

    feature_table = load_feature_table(task_dir)
    if "is_control" not in feature_table.columns:
        raise KeyError(f"{task_dir}: feature_table requires an is_control column")

    expression_matrix = safe_np_load(expression_path, mmap_mode="r")
    if expression_matrix.ndim != 2:
        raise ValueError(f"feature_expression_matrix.npy must be 2D; got shape {expression_matrix.shape}")
    if len(feature_table) != expression_matrix.shape[0]:
        raise ValueError(
            f"feature_table rows {len(feature_table)} != expression rows {expression_matrix.shape[0]}"
        )

    control_mask = parse_bool_series(feature_table["is_control"])
    control_row_count = int(control_mask.sum())
    if control_row_count <= 0:
        raise ValueError(f"{task_dir}: no control rows found in feature_table.is_control")

    control_matrix = np.asarray(expression_matrix[control_mask], dtype=np.float32)
    stats = control_statistics(control_matrix)

    rng = np.random.default_rng(int(args.seed))
    row_count, protein_count = int(expression_matrix.shape[0]), int(expression_matrix.shape[1])
    chunk_size = max(1, int(args.chunk_size))
    random_matrix, policy_meta = generate_matrix(
        policy=policy,
        rng=rng,
        expression_matrix=expression_matrix,
        control_matrix=control_matrix,
        feature_table=feature_table,
        control_mask=control_mask,
        stats=stats,
        chunk_size=chunk_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, random_matrix, allow_pickle=False)

    mean_fallback = np.asarray(stats["mean_fallback"], dtype=bool)
    std_fallback = np.asarray(stats["std_fallback"], dtype=bool)
    meta = {
        "generated_at": iso_now(),
        "generator": "scripts/generate_random_control_proteome.py",
        "task_dir": str(task_dir.resolve()),
        "feature_table_rows": int(len(feature_table)),
        "expression_matrix_path": str(expression_path.resolve()),
        "output_path": str(output_path.resolve()),
        "policy": policy,
        "policy_description": POLICY_DESCRIPTIONS[policy],
        "seed": int(args.seed),
        "shape": [row_count, protein_count],
        "dtype": "float32",
        "control_row_count": control_row_count,
        "control_row_policy": "feature_table.is_control == true",
        "per_protein_stats_policy": (
            "sample each protein independently from Normal(control-row nanmean, control-row nanstd); "
            "fallback to global control std when per-protein std is missing or <=0; "
            "fallback to global control mean when per-protein mean is missing; clip negatives to 0"
            if policy in {"per_protein_normal_clip", "fixed_gene_permutation", "per_row_gene_permutation"}
            else None
        ),
        "global_control_mean": stats["global_mean"],
        "global_control_std": stats["global_std"],
        "mean_fallback_protein_count": int(mean_fallback.sum()),
        "std_fallback_protein_count": int(std_fallback.sum()),
        "fallback_protein_count": int((mean_fallback | std_fallback).sum()),
        "control_stats": finite_summary(control_matrix, chunk_size=chunk_size),
        "random_stats": finite_summary(random_matrix, chunk_size=chunk_size),
        "has_negative_values": bool(np.any(random_matrix < 0.0)),
    }
    meta.update(policy_meta)

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(meta), handle, ensure_ascii=False, indent=2)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        required=True,
        help="Training-ready task directory containing feature_table and feature_expression_matrix.npy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=POLICIES, default=DEFAULT_POLICY)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta = generate_random_control(args)
    print(
        "[random-control] wrote "
        f"{meta['output_path']} shape={tuple(meta['shape'])} "
        f"policy={meta['policy']} seed={meta['seed']} controls={meta['control_row_count']} "
        f"clipped={meta.get('clipped_negative_count', 0)}"
    )


if __name__ == "__main__":
    main()
