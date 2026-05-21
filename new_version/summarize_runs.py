#!/usr/bin/env python3
"""Summarize new_version training manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRIC_KEYS = [
    "test/task_auprc",
    "test/task_auroc",
    "test/task_acc",
    "test/total_loss",
    "test/loss1",
    "test/loss2",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_method(experiment_name: str, prefix: str) -> str:
    suffix = experiment_name[len(prefix) :].lstrip("_")
    if "_fold" not in suffix:
        return suffix or "unknown"
    return suffix.rsplit("_fold", 1)[0]


def infer_fold(experiment_name: str) -> str:
    if "_fold" not in experiment_name:
        return ""
    return experiment_name.rsplit("_fold", 1)[-1]


def metric_value(manifest: dict[str, Any], key: str) -> float | None:
    results = manifest.get("test_results") or []
    if not results or not isinstance(results[0], dict):
        return None
    value = results[0].get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def elapsed_seconds(manifest: dict[str, Any]) -> float | None:
    start = manifest.get("generated_at")
    end = manifest.get("fit_completed_at")
    if not start or not end:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize new_version run manifests")
    parser.add_argument("--checkpoint-dir", default="new_version/checkpoints")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(checkpoint_dir.glob(f"{args.prefix}*/run_manifest.json")):
        manifest = load_json(manifest_path)
        experiment_name = manifest.get("experiment_name", manifest_path.parent.name)
        row: dict[str, Any] = {
            "method": infer_method(experiment_name, args.prefix),
            "fold": infer_fold(experiment_name),
            "experiment": experiment_name,
            "status": manifest.get("run_status", ""),
            "test_status": manifest.get("test_status", ""),
            "best_model_score": manifest.get("best_model_score", ""),
            "positive_weight": manifest.get("positive_weight", ""),
            "fit_elapsed_sec": elapsed_seconds(manifest),
            "max_epochs": manifest.get("args", {}).get("max_epochs", ""),
            "batch_size": manifest.get("args", {}).get("batch_size", ""),
        }
        for key in METRIC_KEYS:
            row[key] = metric_value(manifest, key)
        rows.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    lines = []
    header = [
        "level",
        "method",
        "fold",
        "experiment",
        "status",
        "test_status",
        "best_model_score",
        "positive_weight",
        "fit_elapsed_sec",
        "max_epochs",
        "batch_size",
        *METRIC_KEYS,
    ]
    lines.append("\t".join(header))
    for row in rows:
        lines.append(
            "\t".join(
                [
                    "fold",
                    str(row["method"]),
                    str(row["fold"]),
                    str(row["experiment"]),
                    str(row["status"]),
                    str(row["test_status"]),
                    str(row["best_model_score"]),
                    str(row["positive_weight"]),
                    fmt(row["fit_elapsed_sec"]),
                    str(row["max_epochs"]),
                    str(row["batch_size"]),
                    *[fmt(row[key]) for key in METRIC_KEYS],
                ]
            )
        )
    for method, method_rows in sorted(grouped.items()):
        completed = [
            row
            for row in method_rows
            if row.get("status") == "fit_completed" and row.get("test_status") == "test_completed"
        ]
        for stat_name in ["mean", "std"]:
            values = []
            for key in METRIC_KEYS:
                metric_values = [row[key] for row in completed if row[key] is not None]
                if not metric_values:
                    values.append("")
                elif stat_name == "mean":
                    values.append(fmt(mean(metric_values)))
                else:
                    values.append(fmt(pstdev(metric_values)))
            lines.append(
                "\t".join(
                    [
                        stat_name,
                        method,
                        "",
                        "",
                        f"n={len(completed)}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        *values,
                    ]
                )
            )

    text = "\n".join(lines) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
