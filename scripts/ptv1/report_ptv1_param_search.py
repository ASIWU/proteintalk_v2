#!/usr/bin/env python3
"""Summarize PTV1 exp_12 parameter searches."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


EPOCH_RE = re.compile(r"epoch=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--baseline-prefix", default="20260603_2037_ptv1")
    parser.add_argument("--baseline-config", default="formal_baseline")
    parser.add_argument("--format", choices=("markdown", "tsv"), default="markdown")
    parser.add_argument("--precision", type=int, default=6)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def parse_epoch(path: str | None) -> int | None:
    if not path:
        return None
    match = EPOCH_RE.search(str(path))
    if match is None:
        return None
    return int(match.group(1))


def test_metrics(manifest: dict[str, Any]) -> dict[str, Any] | None:
    results = manifest.get("test_results") or []
    if not results or not isinstance(results[0], dict):
        return None
    row = results[0]
    count = finite(row.get("test/task_count"))
    baseline = finite(row.get("test/task_auprc_baseline"))
    pos = round(count * baseline) if count is not None and baseline is not None else None
    neg = round(count - pos) if count is not None and pos is not None else None
    return {
        "auroc": finite(row.get("test/task_auroc")),
        "auprc": finite(row.get("test/task_auprc")),
        "nauprc": finite(row.get("test/task_nauprc")),
        "count": count,
        "pos": pos,
        "neg": neg,
    }


def manifest_record(path: Path, stage: str, config: str) -> dict[str, Any] | None:
    manifest = load_json(path)
    if manifest.get("dataset_group") != "ptv1":
        return None
    if manifest.get("task_name") != "ptv1_aivc":
        return None
    split = str(manifest.get("split_strategy") or "")
    if not split.startswith("pert_id_5fold_fold"):
        return None
    metrics = test_metrics(manifest)
    if metrics is None:
        return None
    fold = split.rsplit("fold", 1)[-1]
    checkpoint = manifest.get("best_model_path") or manifest.get("test_checkpoint_path")
    return {
        "stage": stage,
        "config": config,
        "fold": fold,
        "split": split,
        "selected_epoch": parse_epoch(checkpoint),
        "source": str(path),
        "learning_rate": manifest.get("learning_rate"),
        "batch_size": manifest.get("batch_size"),
        "dropout": manifest.get("dropout"),
        "weight_decay": manifest.get("weight_decay"),
        "mse_weight": manifest.get("mse_weight"),
        "mse_target_mode": manifest.get("mse_target_mode"),
        "mse_inactive_label_weight": manifest.get("mse_inactive_label_weight"),
        "ranking_loss_weight": manifest.get("ranking_loss_weight"),
        "positive_weight": manifest.get("positive_weight"),
        "focal_loss": manifest.get("focal_loss"),
        "graph_feature_mode": manifest.get("graph_feature_mode"),
        "graph_structural_rp": manifest.get("graph_structural_rp"),
        "graph_drug_concat": manifest.get("graph_drug_concat"),
        "graph_logit_scale": manifest.get("graph_logit_scale"),
        "protein_concat_mode": manifest.get("protein_concat_mode"),
        "protein_concat_topk": manifest.get("protein_concat_topk"),
        "protein_concat_score_mode": manifest.get("protein_concat_score_mode"),
        "protein_concat_expr_scale": manifest.get("protein_concat_expr_scale"),
        "batch_cov_list": manifest.get("batch_cov_list"),
        "use_dose_covariate": manifest.get("use_dose_covariate"),
        "covariate_unk_for_unseen": manifest.get("covariate_unk_for_unseen"),
        "covariate_unk_fields": manifest.get("covariate_unk_fields"),
        "covariate_unk_dropout": manifest.get("covariate_unk_dropout"),
        "control_expression_dropout": manifest.get("control_expression_dropout"),
        "target_expression_mode": manifest.get("target_expression_mode"),
        "active_label_sampling_weight": manifest.get("active_label_sampling_weight"),
        "positive_label_sampling_weight": manifest.get("positive_label_sampling_weight"),
        **metrics,
    }


def iter_search_manifests(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pattern = re.compile(rf"^{re.escape(args.base_prefix)}_(stage[0-9]+)_(.+)_ptv1_unseen_drug_fold[0-9]+$")
    for manifest_path in sorted(args.checkpoint_root.glob(f"{args.base_prefix}_stage*_ptv1_unseen_drug_fold*/run_manifest.json")):
        match = pattern.match(manifest_path.parent.name)
        if match is None:
            continue
        record = manifest_record(manifest_path, match.group(1), match.group(2))
        if record is not None:
            records.append(record)
    if args.baseline_prefix:
        for manifest_path in sorted(args.checkpoint_root.glob(f"{args.baseline_prefix}_ptv1_unseen_drug_fold*/run_manifest.json")):
            record = manifest_record(manifest_path, "baseline", args.baseline_config)
            if record is not None:
                records.append(record)
    return records


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["stage"], row["config"])].append(row)
    out: list[dict[str, Any]] = []
    for (stage, config), rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["fold"]))
        item: dict[str, Any] = {
            "stage": stage,
            "config": config,
            "folds": ",".join(row["fold"] for row in rows),
            "n_folds": len(rows),
            "count": sum(finite(row.get("count")) or 0.0 for row in rows),
            "pos": sum(int(row.get("pos") or 0) for row in rows),
            "neg": sum(int(row.get("neg") or 0) for row in rows),
            "epochs": ",".join("" if row.get("selected_epoch") is None else str(row["selected_epoch"]) for row in rows),
            "fold_auprc": ",".join(fmt(row.get("auprc"), 4) for row in rows),
        }
        for key in (
            "learning_rate",
            "batch_size",
            "dropout",
            "weight_decay",
            "mse_weight",
            "mse_target_mode",
            "mse_inactive_label_weight",
            "ranking_loss_weight",
            "positive_weight",
            "focal_loss",
            "graph_feature_mode",
            "graph_structural_rp",
            "graph_drug_concat",
            "graph_logit_scale",
            "protein_concat_mode",
            "protein_concat_topk",
            "protein_concat_score_mode",
            "protein_concat_expr_scale",
            "batch_cov_list",
            "use_dose_covariate",
            "covariate_unk_for_unseen",
            "covariate_unk_fields",
            "covariate_unk_dropout",
            "control_expression_dropout",
            "target_expression_mode",
            "active_label_sampling_weight",
            "positive_label_sampling_weight",
        ):
            values = [row.get(key) for row in rows if row.get(key) is not None]
            item[key] = values[0] if values else None
        for key in ("auprc", "nauprc", "auroc"):
            values = [value for row in rows if (value := finite(row.get(key))) is not None]
            item[f"mean_{key}"] = mean(values) if values else None
            item[f"std_{key}"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        out.append(item)
    return sorted(out, key=lambda row: (finite(row.get("mean_auprc")) is None, -(finite(row.get("mean_auprc")) or -1.0), row["stage"], row["config"]))


def fmt(value: Any, precision: int) -> str:
    number = finite(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def cell(value: Any, precision: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if finite(value) is not None:
        return fmt(value, precision)
    return str(value)


def print_markdown(rows: list[dict[str, Any]], precision: int) -> None:
    print("| rank | stage | config | folds | mean AUPRC | std AUPRC | mean n-AUPRC | mean AUROC | count | pos | neg | epochs | fold AUPRCs | lr | batch | dropout | MSE | target | graph | graph logit | PCEP | PCEP score | batch covs | ctrl drop | target expr | inactive-MSE | ranking | pos_weight | focal |")
    print("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|---:|---|---|---|---:|---|---:|---:|---|---|")
    for rank, row in enumerate(rows, start=1):
        graph = str(row.get("graph_feature_mode") or "")
        if row.get("graph_feature_mode") == "real":
            graph = f"real/srp={cell(row.get('graph_structural_rp'), precision)}/drug={cell(row.get('graph_drug_concat'), precision)}"
        pcep = str(row.get("protein_concat_mode") or "")
        if row.get("protein_concat_mode") not in (None, "", "off"):
            pcep = f"{row.get('protein_concat_mode')}/top{cell(row.get('protein_concat_topk'), 0)}"
        print(
            "| {rank} | {stage} | {config} | {folds} | {mean_auprc} | {std_auprc} | {mean_nauprc} | {mean_auroc} | {count} | {pos} | {neg} | {epochs} | {fold_auprc} | {lr} | {batch} | {dropout} | {mse} | {target} | {graph} | {graph_logit} | {pcep} | {pcep_score} | {batch_covs} | {ctrl_drop} | {target_expr} | {inactive} | {ranking} | {pos_weight} | {focal} |".format(
                rank=rank,
                stage=row["stage"],
                config=row["config"],
                folds=row["folds"],
                mean_auprc=fmt(row.get("mean_auprc"), precision),
                std_auprc=fmt(row.get("std_auprc"), precision),
                mean_nauprc=fmt(row.get("mean_nauprc"), precision),
                mean_auroc=fmt(row.get("mean_auroc"), precision),
                count=str(int(row["count"])) if finite(row.get("count")) is not None else "",
                pos=str(int(row["pos"])) if row.get("pos") is not None else "",
                neg=str(int(row["neg"])) if row.get("neg") is not None else "",
                epochs=row.get("epochs") or "",
                fold_auprc=row.get("fold_auprc") or "",
                lr=fmt(row.get("learning_rate"), 6),
                batch=str(row.get("batch_size") or ""),
                dropout=fmt(row.get("dropout"), 4),
                mse=fmt(row.get("mse_weight"), 4),
                target=str(row.get("mse_target_mode") or ""),
                graph=graph,
                graph_logit=cell(row.get("graph_logit_scale"), precision),
                pcep=pcep,
                pcep_score=str(row.get("protein_concat_score_mode") or ""),
                batch_covs=cell(row.get("batch_cov_list"), precision),
                ctrl_drop=cell(row.get("control_expression_dropout"), precision),
                target_expr=str(row.get("target_expression_mode") or ""),
                inactive=fmt(row.get("mse_inactive_label_weight"), 4),
                ranking=fmt(row.get("ranking_loss_weight"), 4),
                pos_weight=str(row.get("positive_weight") or ""),
                focal=str(row.get("focal_loss") or ""),
            )
        )


def print_tsv(rows: list[dict[str, Any]], precision: int) -> None:
    columns = (
        "stage",
        "config",
        "folds",
        "n_folds",
        "mean_auprc",
        "std_auprc",
        "mean_nauprc",
        "mean_auroc",
        "count",
        "pos",
        "neg",
        "epochs",
        "fold_auprc",
        "learning_rate",
        "batch_size",
        "dropout",
        "mse_weight",
        "mse_target_mode",
        "mse_inactive_label_weight",
        "ranking_loss_weight",
        "positive_weight",
        "focal_loss",
        "graph_feature_mode",
        "graph_structural_rp",
        "graph_drug_concat",
        "graph_logit_scale",
        "protein_concat_mode",
        "protein_concat_topk",
        "protein_concat_score_mode",
        "protein_concat_expr_scale",
        "batch_cov_list",
        "use_dose_covariate",
        "covariate_unk_for_unseen",
        "covariate_unk_fields",
        "covariate_unk_dropout",
        "control_expression_dropout",
        "target_expression_mode",
        "active_label_sampling_weight",
        "positive_label_sampling_weight",
    )
    print("\t".join(columns))
    for row in rows:
        print(
            "\t".join(
                fmt(row.get(col), precision) if col.startswith(("mean_", "std_")) else cell(row.get(col), precision)
                for col in columns
            )
        )


def main() -> int:
    args = parse_args()
    records = iter_search_manifests(args)
    if not records:
        raise SystemExit(f"no PTV1 parameter search manifests found for {args.base_prefix!r}")
    rows = aggregate(records)
    if args.format == "markdown":
        print_markdown(rows, args.precision)
    else:
        print_tsv(rows, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
