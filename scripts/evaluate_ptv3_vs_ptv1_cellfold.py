#!/usr/bin/env python3
"""Compare label-aligned PTV3 predictions with PTV1-flow cell_5fold results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_PTV1_RUN_ROOT = Path(
    "baseline/ptv2_benchmark_260514/retrain_runs/"
    "flow_v2_cell_celltype_5fold_retrain_readme_20260617_1720/cell_5fold"
)
DEFAULT_PTV1_DATA_ROOT = Path("baseline/ptv2_benchmark_260514/data/1_4EGHv2biorep_bind_unified/cell_5fold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="PTV3 experiment prefix before _exp03_single_cell_5fold...")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--ptv1-run-root", type=Path, default=DEFAULT_PTV1_RUN_ROOT)
    parser.add_argument("--ptv1-data-root", type=Path, default=DEFAULT_PTV1_DATA_ROOT)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def parse_experiment_type(value: str) -> tuple[str, str, str]:
    cell, rest = str(value).split("_", 1)
    parts = [part.lstrip("#") for part in rest.strip().split()]
    if len(parts) < 2:
        raise ValueError(f"cannot parse experiment_type: {value!r}")
    return str(cell), str(parts[0]), str(parts[1])


def load_ptv1_valid_key_order(data_root: Path, fold: int) -> pd.DataFrame:
    prefix = data_root / f"fold{fold}" / "test" / f"cell_5fold_fold{fold}_"
    loo = pd.read_csv(str(prefix) + "loo_label.csv", header=None)
    pheno = pd.read_csv(str(prefix) + "pheno.csv", header=None)
    labels = loo[0].astype(str).to_numpy()
    times = loo[1].to_numpy()
    first_index_by_experiment: dict[str, int] = {}
    times_by_experiment: dict[str, set[int]] = {}
    for row_idx, (experiment_type, time_value) in enumerate(zip(labels, times)):
        first_index_by_experiment.setdefault(experiment_type, row_idx)
        try:
            time_int = int(time_value)
        except (TypeError, ValueError):
            continue
        times_by_experiment.setdefault(experiment_type, set()).add(time_int)

    records: list[dict[str, Any]] = []
    for experiment_type in sorted(times_by_experiment):
        if "#" not in experiment_type or not {6, 24}.issubset(times_by_experiment[experiment_type]):
            continue
        first_idx = first_index_by_experiment[experiment_type]
        label = pd.to_numeric(pd.Series([pheno.iloc[first_idx, 0]]), errors="coerce").iloc[0]
        if pd.isna(label) or float(label) not in {0.0, 1.0}:
            continue
        records.append(
            {
                "_key": parse_experiment_type(experiment_type),
                "_label": float(label),
                "experiment_type": experiment_type,
            }
        )
    return pd.DataFrame(records)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def prediction_score_column(df: pd.DataFrame) -> str:
    for column in ("pred_task_prob", "predicted_probabilities", "response_prob", "probability"):
        if column in df.columns:
            return column
    raise KeyError(f"no prediction score column found in columns={list(df.columns)}")


def label_column(df: pd.DataFrame) -> str:
    for column in ("ptv1_cell_5fold_label", "ground_truth", "response_label"):
        if column in df.columns:
            return column
    raise KeyError(f"no label column found in columns={list(df.columns)}")


def encode_label_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    lowered = series.astype(str).str.strip().str.lower()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[lowered.isin({"sensitive", "responsive", "1", "true", "yes", "y"})] = 1.0
    result.loc[lowered.isin({"non-responsive", "nonresponsive", "0", "false", "no", "n"})] = 0.0
    return result


def metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    positives = int(np.sum(y_true == 1.0))
    negatives = int(np.sum(y_true == 0.0))
    result: dict[str, float | int] = {
        "n": int(len(y_true)),
        "positive": positives,
        "negative": negatives,
        "auprc_baseline": float(positives / len(y_true)) if len(y_true) else float("nan"),
    }
    if positives > 0 and negatives > 0:
        result["average_precision"] = float(average_precision_score(y_true, y_score))
        result["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        result["average_precision"] = float("nan")
        result["auroc"] = float("nan")
    return result


def load_ptv1_predictions(run_root: Path, data_root: Path, fold: int) -> pd.DataFrame:
    candidates = sorted((run_root / f"fold{fold}" / "eval").glob("*/predictions.csv"))
    if not candidates:
        raise FileNotFoundError(f"missing PTV1 predictions under {run_root / f'fold{fold}' / 'eval'}")
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    df = pd.read_csv(path)
    score_col = prediction_score_column(df)
    key_order = load_ptv1_valid_key_order(data_root, fold)
    if len(df) != len(key_order):
        raise ValueError(
            f"PTV1 prediction rows ({len(df)}) do not match reconstructed valid key count ({len(key_order)})"
        )
    result = key_order.copy()
    result["_score"] = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    csv_labels = encode_label_series(df[label_column(df)])
    mismatches = int((csv_labels.to_numpy(float) != result["_label"].to_numpy(float)).sum())
    if mismatches:
        print(f"[warn] {path} has {mismatches} labels mismatching reconstructed PTV1 split order")
    return result[["_key", "_label", "_score", "experiment_type"]].copy()


def load_ptv3_predictions(args: argparse.Namespace, fold: int) -> pd.DataFrame:
    prediction_root = args.prediction_root or Path("outputs") / f"{args.prefix}_cell_drug_fold_predictions"
    prediction_dir = prediction_root / "exp03" / f"fold{fold}"
    parquet_path = prediction_dir / "predictions.parquet"
    csv_path = prediction_dir / "predictions.csv"
    if parquet_path.exists():
        predictions = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        predictions = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"missing PTV3 predictions under {prediction_dir}")

    manifest_path = (
        args.checkpoint_root
        / f"{args.prefix}_exp03_single_cell_5fold_single_cell_fold{fold}"
        / "run_manifest.json"
    )
    manifest = load_json(manifest_path)
    task_dir = Path(str(manifest["task_dir"]))
    feature_table = pd.read_parquet(task_dir / "feature_table.parquet") if (task_dir / "feature_table.parquet").exists() else pd.read_csv(task_dir / "feature_table.csv", low_memory=False)
    if "feature_row_index" not in predictions.columns:
        raise KeyError(f"PTV3 predictions missing feature_row_index: {prediction_dir}")
    row_indices = pd.to_numeric(predictions["feature_row_index"], errors="raise").astype(int).to_numpy()
    metadata_cols = [
        column
        for column in (
            "Cell",
            "pert_id1",
            "pert_id2",
            "ptv1_cell_5fold_label",
            "ptv1_cell_5fold_experiment_type",
        )
        if column in feature_table.columns
    ]
    metadata = feature_table.iloc[row_indices][metadata_cols].reset_index(drop=True)
    merged = pd.concat([predictions.reset_index(drop=True), metadata.add_prefix("meta_")], axis=1)
    score_col = prediction_score_column(merged)
    merged["_score"] = pd.to_numeric(merged[score_col], errors="coerce")
    merged["_label"] = encode_label_series(merged["meta_ptv1_cell_5fold_label"])
    merged["_key"] = list(
        zip(
            merged["meta_Cell"].astype(str),
            merged["meta_pert_id1"].astype(str),
            merged["meta_pert_id2"].astype(str),
        )
    )
    return merged[["_key", "_label", "_score"]].copy()


def group_by_key(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.loc[df["_label"].isin([0.0, 1.0]) & np.isfinite(df["_score"])].copy()
    grouped = (
        valid.groupby("_key", sort=True)
        .agg(_label=("_label", "first"), _score=("_score", "mean"), rows=("_score", "size"))
        .reset_index()
    )
    label_conflicts = valid.groupby("_key")["_label"].nunique(dropna=True)
    conflicts = int((label_conflicts > 1).sum())
    if conflicts:
        raise ValueError(f"{conflicts} grouped keys have conflicting labels")
    return grouped


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for fold in args.folds:
        ptv1 = group_by_key(load_ptv1_predictions(args.ptv1_run_root, args.ptv1_data_root, fold))
        ptv3 = group_by_key(load_ptv3_predictions(args, fold))
        ptv1_metrics = metrics(ptv1["_label"].to_numpy(float), ptv1["_score"].to_numpy(float))
        ptv3_metrics = metrics(ptv3["_label"].to_numpy(float), ptv3["_score"].to_numpy(float))
        common = ptv1.merge(ptv3, on="_key", suffixes=("_ptv1", "_ptv3"))
        common_ptv1_metrics = metrics(common["_label_ptv1"].to_numpy(float), common["_score_ptv1"].to_numpy(float))
        common_ptv3_metrics = metrics(common["_label_ptv1"].to_numpy(float), common["_score_ptv3"].to_numpy(float))
        records.append(
            {
                "fold": fold,
                "ptv1_n": ptv1_metrics["n"],
                "ptv1_positive": ptv1_metrics["positive"],
                "ptv1_average_precision": ptv1_metrics["average_precision"],
                "ptv1_auroc": ptv1_metrics["auroc"],
                "ptv3_n": ptv3_metrics["n"],
                "ptv3_positive": ptv3_metrics["positive"],
                "ptv3_average_precision": ptv3_metrics["average_precision"],
                "ptv3_auroc": ptv3_metrics["auroc"],
                "common_n": int(len(common)),
                "common_label_mismatch": int((common["_label_ptv1"] != common["_label_ptv3"]).sum()),
                "common_ptv1_average_precision": common_ptv1_metrics["average_precision"],
                "common_ptv1_auroc": common_ptv1_metrics["auroc"],
                "common_ptv3_average_precision_on_ptv1_labels": common_ptv3_metrics["average_precision"],
                "common_ptv3_auroc_on_ptv1_labels": common_ptv3_metrics["auroc"],
            }
        )

    if len(records) > 1:
        mean_record: dict[str, Any] = {"fold": "mean"}
        for key in records[0]:
            if key == "fold":
                continue
            values = [float(record[key]) for record in records if record[key] is not None]
            mean_record[key] = float(np.mean(values)) if values else float("nan")
        records.append(mean_record)

    columns = list(records[0]) if records else []
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2, allow_nan=True)

    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
