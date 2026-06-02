#!/usr/bin/env python3
"""Summarize staged dose-covariate parameter searches."""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scripts" / "report_ptv3_exp_results.py"


def load_report_module():
    spec = importlib.util.spec_from_file_location("ptv3_report", REPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "all"),
        default="all",
    )
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


def collect_prefixes(args: argparse.Namespace) -> list[tuple[str, str]]:
    pattern = re.compile(rf"^{re.escape(args.base_prefix)}_(stage[1-6])_(.+)_exp0[1-8]_")
    found: set[tuple[str, str]] = set()
    for root in (args.checkpoint_root, args.output_root):
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            stage, config = match.group(1), match.group(2)
            if args.stage != "all" and stage != args.stage:
                continue
            found.add((stage, config))
    return sorted(found)


def collect_records(report: Any, args: argparse.Namespace, stage: str, config: str) -> list[dict[str, Any]]:
    ns = argparse.Namespace(
        prefix=f"{args.base_prefix}_{stage}_{config}",
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
        exp08_root=None,
    )
    return report.collect_records(ns)


def index_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["exp"], row["task"], row["split"]): row for row in records}


def aggregate_fold_records(records: list[dict[str, Any]], exp: str) -> dict[str, Any] | None:
    fold_rows = [
        row
        for row in records
        if row.get("exp") == exp and row.get("task") == exp and str(row.get("split", "")).startswith("fold")
    ]
    if not fold_rows:
        return None
    aggregate: dict[str, Any] = {"exp": exp, "task": exp, "split": f"mean{len(fold_rows)}"}
    for key in ("auprc", "auprc_baseline", "nauprc", "auroc", "acc"):
        values = [value for row in fold_rows if (value := finite(row.get(key))) is not None]
        aggregate[key] = sum(values) / len(values) if values else None
    aggregate["count"] = sum(finite(row.get("count")) or 0.0 for row in fold_rows)
    return aggregate


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    return finite(row.get(key))


def fmt(value: Any, precision: int) -> str:
    number = finite(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def stage1_row(stage: str, config: str, record_list: list[dict[str, Any]]) -> dict[str, Any]:
    exp01 = aggregate_fold_records(record_list, "exp01")
    exp04 = aggregate_fold_records(record_list, "exp04")
    exp05 = aggregate_fold_records(record_list, "exp05")
    auprc = metric(exp01, "auprc")
    gap_mse = None if auprc is None or metric(exp04, "auprc") is None else auprc - metric(exp04, "auprc")
    gap_graph = None if auprc is None or metric(exp05, "auprc") is None else auprc - metric(exp05, "auprc")
    min_gap = None if gap_mse is None or gap_graph is None else min(gap_mse, gap_graph)
    score = None
    if auprc is not None and min_gap is not None:
        score = auprc + 0.5 * min(min_gap, 0.05) - 0.5 * max(0.05 - min_gap, 0.0)
    return {
        "stage": stage,
        "config": config,
        "task": "exp01_gap",
        "auprc": auprc,
        "nauprc": metric(exp01, "nauprc"),
        "auroc": metric(exp01, "auroc"),
        "gap_mse_auprc": gap_mse,
        "gap_graph_auprc": gap_graph,
        "score": score,
        "count": metric(exp01, "count"),
    }


def single_exp_row(
    stage: str,
    config: str,
    task: str,
    exp: str,
    record_list: list[dict[str, Any]],
) -> dict[str, Any]:
    row = aggregate_fold_records(record_list, exp)
    return {
        "stage": stage,
        "config": config,
        "task": task,
        "auprc": metric(row, "auprc"),
        "nauprc": metric(row, "nauprc"),
        "auroc": metric(row, "auroc"),
        "gap_mse_auprc": None,
        "gap_graph_auprc": None,
        "score": metric(row, "nauprc"),
        "count": metric(row, "count"),
    }


def extra_rows(
    stage: str,
    config: str,
    records: dict[tuple[str, str, str], dict[str, Any]],
    exp: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matched = [row for key, row in records.items() if key[0] == exp and key[2] == "extra"]
    for row in sorted(matched, key=lambda item: item["task"]):
        rows.append(
            {
                "stage": stage,
                "config": config,
                "task": row["task"],
                "auprc": metric(row, "auprc"),
                "nauprc": metric(row, "nauprc"),
                "auroc": metric(row, "auroc"),
                "gap_mse_auprc": None,
                "gap_graph_auprc": None,
                "score": metric(row, "nauprc"),
                "count": metric(row, "count"),
            }
        )
    if matched:
        values = [metric(row, "nauprc") for row in matched]
        auprcs = [metric(row, "auprc") for row in matched]
        aurocs = [metric(row, "auroc") for row in matched]
        rows.append(
            {
                "stage": stage,
                "config": config,
                "task": f"{exp}_mean",
                "auprc": sum(v for v in auprcs if v is not None) / len([v for v in auprcs if v is not None]),
                "nauprc": sum(v for v in values if v is not None) / len([v for v in values if v is not None]),
                "auroc": sum(v for v in aurocs if v is not None) / len([v for v in aurocs if v is not None]),
                "gap_mse_auprc": None,
                "gap_graph_auprc": None,
                "score": sum(v for v in values if v is not None) / len([v for v in values if v is not None]),
                "count": sum(metric(row, "count") or 0.0 for row in matched),
            }
        )
    return rows


def summarize(args: argparse.Namespace) -> list[dict[str, Any]]:
    report = load_report_module()
    rows: list[dict[str, Any]] = []
    for stage, config in collect_prefixes(args):
        record_list = collect_records(report, args, stage, config)
        records = index_records(record_list)
        if stage == "stage1":
            rows.append(stage1_row(stage, config, record_list))
        elif stage == "stage2":
            rows.append(single_exp_row(stage, config, "exp03_unseen_cell", "exp03", record_list))
        elif stage == "stage3":
            rows.append(single_exp_row(stage, config, "exp02_unseen_cell_type", "exp02", record_list))
        elif stage == "stage4":
            rows.append(single_exp_row(stage, config, "exp06_double_unseen_drug", "exp06", record_list))
        elif stage == "stage5":
            rows.extend(extra_rows(stage, config, records, "exp07"))
        elif stage == "stage6":
            rows.extend(extra_rows(stage, config, records, "exp08"))
    return rows


def print_rows(rows: list[dict[str, Any]], precision: int, output_format: str) -> None:
    columns = (
        "stage",
        "config",
        "task",
        "auprc",
        "nauprc",
        "auroc",
        "gap_mse_auprc",
        "gap_graph_auprc",
        "score",
        "count",
    )
    if output_format == "tsv":
        print("\t".join(columns))
        for row in rows:
            print("\t".join(fmt(row.get(col), precision) if col not in {"stage", "config", "task"} else str(row.get(col, "")) for col in columns))
        return
    print("| stage | config | task | AUPRC | nAUPRC | AUROC | gap vs no-MSE | gap vs no-graph | score | count |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {stage} | {config} | {task} | {auprc} | {nauprc} | {auroc} | {gap_mse} | {gap_graph} | {score} | {count} |".format(
                stage=row.get("stage", ""),
                config=row.get("config", ""),
                task=row.get("task", ""),
                auprc=fmt(row.get("auprc"), precision),
                nauprc=fmt(row.get("nauprc"), precision),
                auroc=fmt(row.get("auroc"), precision),
                gap_mse=fmt(row.get("gap_mse_auprc"), precision),
                gap_graph=fmt(row.get("gap_graph_auprc"), precision),
                score=fmt(row.get("score"), precision),
                count=str(int(row["count"])) if finite(row.get("count")) is not None else "",
            )
        )


def main() -> int:
    args = parse_args()
    rows = summarize(args)
    rows.sort(key=lambda row: (row.get("stage", ""), row.get("task", ""), -(finite(row.get("score")) or -1e9)))
    print_rows(rows, args.precision, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
