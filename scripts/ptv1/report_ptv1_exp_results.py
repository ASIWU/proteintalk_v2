#!/usr/bin/env python3
"""Report PTV1 exp_11, exp_12, and exp_13 metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_COLUMNS = ("auroc", "auprc", "nauprc", "count", "pos", "neg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Experiment prefix or leading prefix fragment.")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--precision", type=int, default=6)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metrics_from_test_result(result: dict[str, Any]) -> dict[str, Any]:
    count = finite_float(result.get("test/task_count"))
    baseline = finite_float(result.get("test/task_auprc_baseline"))
    pos = round(count * baseline) if count is not None and baseline is not None else None
    neg = round(count - pos) if count is not None and pos is not None else None
    return {
        "auroc": finite_float(result.get("test/task_auroc")),
        "auprc": finite_float(result.get("test/task_auprc")),
        "nauprc": finite_float(result.get("test/task_nauprc")),
        "count": count,
        "pos": pos,
        "neg": neg,
    }


def record(exp: str, task: str, split: str, metrics: dict[str, Any], source: Path) -> dict[str, Any]:
    return {"exp": exp, "task": task, "split": split, **metrics, "source": str(source)}


def iter_manifests(root: Path, prefix: str) -> list[Path]:
    return sorted(root.glob(f"{prefix}*/run_manifest.json"))


def collect_training_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for path in iter_manifests(args.checkpoint_root, args.prefix):
        manifest = load_json(path)
        if manifest.get("dataset_group") != "ptv1" or manifest.get("task_name") != "ptv1_aivc":
            continue
        results = manifest.get("test_results") or []
        if not results or not isinstance(results[0], dict):
            continue
        split = str(manifest.get("split_strategy") or "")
        metrics = metrics_from_test_result(results[0])
        if split == "fixed_experiment_type":
            records.append(record("exp11", "ptv1_aivc", split, metrics, path))
        elif split.startswith("pert_id_5fold_fold"):
            item = record("exp12", "ptv1_aivc", split, metrics, path)
            records.append(item)
            fold_records.append(item)
    if len({item["split"] for item in fold_records}) == 5:
        aggregate = {
            key: mean([float(item[key]) for item in fold_records if item.get(key) is not None])
            for key in ("auroc", "auprc", "nauprc")
        }
        aggregate["count"] = sum(float(item["count"] or 0) for item in fold_records)
        aggregate["pos"] = sum(int(item["pos"] or 0) for item in fold_records)
        aggregate["neg"] = sum(int(item["neg"] or 0) for item in fold_records)
        records.append(record("exp12", "ptv1_aivc", "mean5", aggregate, args.checkpoint_root / f"{args.prefix}*"))
    return records


def encode_label(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is not None and math.isfinite(number):
        rounded = round(number)
        if abs(number - rounded) < 1e-8 and rounded in {0, 1}:
            return int(rounded)
    text = str(value).strip().lower()
    if text in {"1", "true", "y", "yes", "responsive", "sensitive"}:
        return 1
    if text in {"0", "false", "n", "no", "non-responsive", "nonresponsive"}:
        return 0
    return None


def prediction_metrics(df: pd.DataFrame) -> dict[str, Any]:
    labels = np.asarray([encode_label(value) for value in df["task_label"].tolist()], dtype=object)
    probs = pd.to_numeric(df["pred_task_prob"], errors="coerce").to_numpy(dtype=float)
    keep = np.asarray([value is not None for value in labels], dtype=bool) & np.isfinite(probs)
    y = np.asarray([int(value) for value in labels[keep]], dtype=int)
    p = probs[keep]
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    out = {"auroc": None, "auprc": None, "nauprc": None, "count": int(len(y)), "pos": pos, "neg": neg}
    if len(y) and pos and neg:
        auprc = float(average_precision_score(y, p))
        baseline = pos / len(y)
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = auprc
        out["nauprc"] = auprc / baseline if baseline > 0 else None
    return out


def read_predictions(path: Path) -> pd.DataFrame:
    parquet_path = path / "predictions.parquet"
    csv_path = path / "predictions.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"missing predictions under {path}")


def collect_exp13_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(args.output_root.glob(f"{args.prefix}*/ptv1_extra_singledrug/metrics.json")):
        task_dir = metrics_path.parent
        try:
            predictions = read_predictions(task_dir)
        except FileNotFoundError:
            metrics = load_json(metrics_path).get("task", {})
            if not isinstance(metrics, dict):
                continue
            count = finite_float(metrics.get("count"))
            baseline = finite_float(metrics.get("auprc_baseline"))
            pos = round(count * baseline) if count is not None and baseline is not None else None
            neg = round(count - pos) if count is not None and pos is not None else None
            records.append(
                record(
                    "exp13",
                    "ptv1_extra_singledrug",
                    "overall",
                    {
                        "auroc": finite_float(metrics.get("auroc")),
                        "auprc": finite_float(metrics.get("auprc")),
                        "nauprc": finite_float(metrics.get("nauprc")),
                        "count": count,
                        "pos": pos,
                        "neg": neg,
                    },
                    metrics_path,
                )
            )
            continue
        records.append(record("exp13", "ptv1_extra_singledrug", "overall", prediction_metrics(predictions), task_dir))
        if "Cell" in predictions.columns:
            for cell, group in predictions.groupby("Cell", sort=True):
                records.append(record("exp13", "ptv1_extra_singledrug", f"cell:{cell}", prediction_metrics(group), task_dir))
    return records


def format_value(value: Any, precision: int) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def print_markdown(records: list[dict[str, Any]], precision: int) -> None:
    print("| exp | task | split | AUROC | AUPRC | n-AUPRC | count | pos | neg |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(
            "| {exp} | {task} | {split} | {auroc} | {auprc} | {nauprc} | {count} | {pos} | {neg} |".format(
                exp=row["exp"],
                task=row["task"],
                split=row["split"],
                auroc=format_value(row.get("auroc"), precision),
                auprc=format_value(row.get("auprc"), precision),
                nauprc=format_value(row.get("nauprc"), precision),
                count=str(int(row["count"])) if row.get("count") is not None else "",
                pos=str(int(row["pos"])) if row.get("pos") is not None else "",
                neg=str(int(row["neg"])) if row.get("neg") is not None else "",
            )
        )


def print_csv(records: list[dict[str, Any]]) -> None:
    columns = ("exp", "task", "split", *METRIC_COLUMNS, "source")
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    for row in records:
        writer.writerow({key: row.get(key, "") for key in columns})


def main() -> int:
    args = parse_args()
    records = collect_training_records(args) + collect_exp13_records(args)
    if not records:
        print("[error] no PTV1 experiment records found", file=sys.stderr)
        return 1
    if args.format == "csv":
        print_csv(records)
    else:
        print_markdown(records, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
