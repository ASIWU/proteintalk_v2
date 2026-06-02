#!/usr/bin/env python3
"""Report PTV3 exp_01 through exp_08 metrics with nAUPRC by default."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any


EXP_FOLD_SPECS = (
    ("exp01", "single_pert_stratified_5fold_single_pert_stratified_fold"),
    ("exp02", "single_cell_type_5fold_single_cell_type_fold"),
    ("exp03", "single_cell_5fold_single_cell_fold"),
    ("exp04", "single_no_mse_5fold_single_no_mse_fold"),
    ("exp05", "single_no_graph_5fold_single_no_graph_fold"),
    ("exp06", "double_pert_pair_5fold_double_pert_pair_fold"),
)
EXP07_SUFFIX = "exp07_extra_single_all_train_infer_all_single_for_extra"
EXP08_SUFFIX = "exp08_extra_double_all_train_infer_all_single_double_for_extra"
METRIC_COLUMNS = ("auprc", "auprc_baseline", "nauprc", "auroc", "acc", "count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Base experiment prefix, without _expNN suffix.")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--exp08-root", type=Path, default=None, help="Override exp08 output directory.")
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


def metric_record(
    *,
    exp: str,
    task: str,
    split: str,
    metrics: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    return {
        "exp": exp,
        "task": task,
        "split": split,
        "auprc": finite_float(metrics.get("auprc")),
        "auprc_baseline": finite_float(metrics.get("auprc_baseline")),
        "nauprc": finite_float(metrics.get("nauprc")),
        "auroc": finite_float(metrics.get("auroc")),
        "acc": finite_float(metrics.get("acc")),
        "count": finite_float(metrics.get("count")),
        "source": str(source),
    }


def fold_metrics_from_manifest(manifest: dict[str, Any]) -> dict[str, Any] | None:
    results = manifest.get("test_results") or []
    if not results or not isinstance(results[0], dict):
        return None
    result = results[0]
    return {
        "auprc": result.get("test/task_auprc"),
        "auprc_baseline": result.get("test/task_auprc_baseline"),
        "nauprc": result.get("test/task_nauprc"),
        "auroc": result.get("test/task_auroc"),
        "acc": result.get("test/task_acc"),
        "count": result.get("test/task_count"),
    }


def summarize_folds(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for exp, suffix in EXP_FOLD_SPECS:
        fold_metrics: list[dict[str, Any]] = []
        for fold in range(5):
            manifest_path = args.checkpoint_root / f"{args.prefix}_{exp}_{suffix}{fold}" / "run_manifest.json"
            if not manifest_path.exists():
                continue
            metrics = fold_metrics_from_manifest(load_json(manifest_path))
            if metrics is None:
                continue
            fold_metrics.append(metrics)
            records.append(
                metric_record(
                    exp=exp,
                    task=exp,
                    split=f"fold{fold}",
                    metrics=metrics,
                    source=manifest_path,
                )
            )
        if len(fold_metrics) == 5:
            aggregate = {
                key: (mean(values) if (values := [v for item in fold_metrics if (v := finite_float(item.get(key))) is not None]) else None)
                for key in METRIC_COLUMNS
            }
            aggregate["count"] = sum(finite_float(item.get("count")) or 0.0 for item in fold_metrics)
            records.append(
                metric_record(
                    exp=exp,
                    task=exp,
                    split="mean5",
                    metrics=aggregate,
                    source=args.checkpoint_root / f"{args.prefix}_{exp}_*",
                )
            )
    return records


def collect_extra_records(root: Path, exp: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics_payload = load_json(metrics_path)
        task_metrics = metrics_payload.get("task")
        if not isinstance(task_metrics, dict):
            continue
        records.append(
            metric_record(
                exp=exp,
                task=metrics_path.parent.name,
                split="extra",
                metrics=task_metrics,
                source=metrics_path,
            )
        )
    return records


def collect_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = summarize_folds(args)
    exp07_root = args.output_root / f"{args.prefix}_{EXP07_SUFFIX}"
    exp08_root = args.exp08_root or (args.output_root / f"{args.prefix}_{EXP08_SUFFIX}")
    records.extend(collect_extra_records(exp07_root, "exp07"))
    records.extend(collect_extra_records(exp08_root, "exp08"))
    return records


def format_value(value: Any, precision: int) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def print_markdown(records: list[dict[str, Any]], precision: int) -> None:
    print("| exp | task | split | AUPRC | baseline | nAUPRC | AUROC | ACC | count |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(
            "| {exp} | {task} | {split} | {auprc} | {baseline} | {nauprc} | {auroc} | {acc} | {count} |".format(
                exp=row["exp"],
                task=row["task"],
                split=row["split"],
                auprc=format_value(row.get("auprc"), precision),
                baseline=format_value(row.get("auprc_baseline"), precision),
                nauprc=format_value(row.get("nauprc"), precision),
                auroc=format_value(row.get("auroc"), precision),
                acc=format_value(row.get("acc"), precision),
                count=str(int(row["count"])) if row.get("count") is not None else "",
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
    records = collect_records(args)
    if not records:
        print("[error] no exp metrics found", file=sys.stderr)
        return 1
    if args.format == "csv":
        print_csv(records)
    else:
        print_markdown(records, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
