#!/usr/bin/env python3
"""Summarize model-size sweep manifests for unseen-drug and unseen-cell runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_from_manifest(manifest: dict[str, Any], key: str) -> float | None:
    results = manifest.get("test_results") or []
    if not results:
        return None
    value = results[0].get(key)
    if value is None:
        return None
    return float(value)


def infer_task(split_strategy: str) -> str:
    if split_strategy.startswith("pert_stratified_5fold_fold"):
        return "unseen_drug"
    if split_strategy.startswith("cell_5fold_fold"):
        return "unseen_cell"
    return split_strategy


def infer_fold(split_strategy: str) -> int | None:
    match = re.search(r"fold(\d+)$", split_strategy)
    if not match:
        return None
    return int(match.group(1))


def checkpoint_epoch(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"epoch=(\d+)", path)
    if not match:
        return None
    return int(match.group(1))


def load_runtime_map(log_dir: Path) -> dict[str, int]:
    runtime_by_artifact: dict[str, int] = {}
    for path in sorted(log_dir.glob("*runtime_summary.tsv")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    artifact = row.get("artifact") or ""
                    duration = row.get("duration_sec") or ""
                    if artifact and duration.isdigit():
                        runtime_by_artifact[str(Path(artifact))] = int(duration)
        except OSError:
            continue
    return runtime_by_artifact


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{int(value)}"


def summarize(values: list[float]) -> dict[str, float | None]:
    finite = [v for v in values if v is not None and not math.isnan(v)]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(finite),
        "std": pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-prefix",
        action="append",
        required=True,
        help="Experiment prefix used by run_model_size_sweep_2gpu.sh. Repeat to merge multiple sweeps.",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    runtime_by_artifact = load_runtime_map(log_dir)

    records: list[dict[str, Any]] = []
    prefixes = list(args.exp_prefix)
    for prefix in prefixes:
        manifest_paths = sorted(checkpoint_dir.glob(f"{prefix}*/run_manifest.json"))
        for manifest_path in manifest_paths:
            manifest = load_json(manifest_path)
            split_strategy = str(manifest.get("split_strategy") or "")
            task = infer_task(split_strategy)
            if task not in {"unseen_drug", "unseen_cell"}:
                continue
            record = {
                "experiment_name": manifest.get("experiment_name"),
                "manifest_path": str(manifest_path),
                "task": task,
                "fold": infer_fold(split_strategy),
                "split_strategy": split_strategy,
                "run_status": manifest.get("run_status"),
                "test_status": manifest.get("test_status"),
                "hidden_dim": manifest.get("hidden_dim"),
                "expression_latent_dim": manifest.get("expression_latent_dim"),
                "covariate_embedding_dim": manifest.get("covariate_embedding_dim"),
                "model_parameter_count": manifest.get("model_parameter_count"),
                "trainable_parameter_count": manifest.get("trainable_parameter_count"),
                "best_valid_score": manifest.get("best_model_score"),
                "best_epoch": checkpoint_epoch(manifest.get("best_model_path")),
                "test_auroc": metric_from_manifest(manifest, "test/auroc"),
                "test_auprc": metric_from_manifest(manifest, "test/auprc"),
                "test_acc": metric_from_manifest(manifest, "test/acc"),
                "test_count": metric_from_manifest(manifest, "test/task_count"),
                "mse_weight": manifest.get("mse_weight"),
                "learning_rate": manifest.get("learning_rate"),
                "dropout": manifest.get("dropout"),
                "weight_decay": manifest.get("weight_decay"),
                "covariate_unk_for_unseen": manifest.get("covariate_unk_for_unseen"),
                "covariate_unk_dropout": manifest.get("covariate_unk_dropout"),
                "batch_cov_list": manifest.get("batch_cov_list"),
                "duration_sec": runtime_by_artifact.get(str(manifest_path.parent)),
            }
            record["size_key"] = (
                f"h{record['hidden_dim']}_e{record['expression_latent_dim']}_c{record['covariate_embedding_dim']}"
            )
            records.append(record)

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["task"],
                record["size_key"],
                str(record.get("learning_rate")),
                str(record.get("mse_weight")),
                f"{record.get('covariate_unk_for_unseen')}:{record.get('covariate_unk_dropout')}",
            )
        ].append(record)

    aggregates: list[dict[str, Any]] = []
    for (task, size_key, _learning_rate, _mse_weight, _covariate_unk), rows in grouped.items():
        rows = sorted(rows, key=lambda row: (-1 if row["fold"] is None else row["fold"]))
        aggregates.append(
            {
                "task": task,
                "size_key": size_key,
                "hidden_dim": rows[0]["hidden_dim"],
                "expression_latent_dim": rows[0]["expression_latent_dim"],
                "covariate_embedding_dim": rows[0]["covariate_embedding_dim"],
                "fold_count": len(rows),
                "folds": [row["fold"] for row in rows],
                "params": rows[0]["model_parameter_count"],
                "trainable_params": rows[0]["trainable_parameter_count"],
                "mse_weight": rows[0]["mse_weight"],
                "learning_rate": rows[0]["learning_rate"],
                "dropout": rows[0]["dropout"],
                "weight_decay": rows[0]["weight_decay"],
                "covariate_unk_for_unseen": rows[0]["covariate_unk_for_unseen"],
                "covariate_unk_dropout": rows[0]["covariate_unk_dropout"],
                "auroc": summarize([row["test_auroc"] for row in rows if row["test_auroc"] is not None]),
                "auprc": summarize([row["test_auprc"] for row in rows if row["test_auprc"] is not None]),
                "acc": summarize([row["test_acc"] for row in rows if row["test_acc"] is not None]),
                "duration_sec": summarize(
                    [float(row["duration_sec"]) for row in rows if row["duration_sec"] is not None]
                ),
                "fold_rows": rows,
            }
        )

    aggregates.sort(key=lambda row: (row["task"], int(row["hidden_dim"] or 0), float(row["learning_rate"] or 0.0)))

    lines = [
        f"# Model Size Sweep Report: `{', '.join(prefixes)}`",
        "",
        "## Aggregate",
        "",
        "| task | hidden | expr_latent | cov_dim | params | folds | AUROC mean | AUPRC mean | AUPRC std | avg sec/fold | LR | MSE | cov UNK |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in aggregates:
        lines.append(
            "| {task} | {hidden} | {expr} | {cov} | {params} | {folds} | {auroc} | {auprc} | {auprc_std} | {sec} | {lr} | {mse} | {unk} |".format(
                task=row["task"],
                hidden=fmt_int(row["hidden_dim"]),
                expr=fmt_int(row["expression_latent_dim"]),
                cov=fmt_int(row["covariate_embedding_dim"]),
                params=fmt_int(row["params"]),
                folds=row["fold_count"],
                auroc=fmt(row["auroc"]["mean"]),
                auprc=fmt(row["auprc"]["mean"]),
                auprc_std=fmt(row["auprc"]["std"]),
                sec=fmt(row["duration_sec"]["mean"], 1),
                lr=fmt(float(row["learning_rate"])) if row["learning_rate"] is not None else "-",
                mse=fmt(float(row["mse_weight"])) if row["mse_weight"] is not None else "-",
                unk=f"{row['covariate_unk_for_unseen']} / {row['covariate_unk_dropout']}",
            )
        )

    lines.extend(
        [
            "",
            "## Fold Detail",
            "",
            "| task | hidden | fold | AUROC | AUPRC | ACC | best valid | best epoch | sec | manifest |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in aggregates:
        for fold_row in row["fold_rows"]:
            lines.append(
                "| {task} | {hidden} | {fold} | {auroc} | {auprc} | {acc} | {best} | {epoch} | {sec} | `{manifest}` |".format(
                    task=fold_row["task"],
                    hidden=fmt_int(fold_row["hidden_dim"]),
                    fold=fmt_int(fold_row["fold"]),
                    auroc=fmt(fold_row["test_auroc"]),
                    auprc=fmt(fold_row["test_auprc"]),
                    acc=fmt(fold_row["test_acc"]),
                    best=fmt(float(fold_row["best_valid_score"]))
                    if fold_row["best_valid_score"] is not None
                    else "-",
                    epoch=fmt_int(fold_row["best_epoch"]),
                    sec=fmt(fold_row["duration_sec"], 1) if fold_row["duration_sec"] is not None else "-",
                    manifest=fold_row["manifest_path"],
                )
            )

    report = "\n".join(lines) + "\n"
    print(report)

    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report, encoding="utf-8")
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"records": records, "aggregates": aggregates}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
