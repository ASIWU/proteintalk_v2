#!/usr/bin/env python3
"""Report exp09 unified-head valid and oracle extra-data metrics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import report_cell_drug_time_eval as base


METHODS = base.METHODS
EXTRA_DOUBLE_TEST_LABEL_GROUPS = base.EXTRA_DOUBLE_TEST_LABEL_GROUPS
EPOCH_RE = re.compile(r"epoch=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-root", type=Path, required=True)
    parser.add_argument("--oracle-root-glob", required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("markdown", "csv", "none"), default="markdown")
    parser.add_argument("--precision", type=int, default=6)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    return base.finite_float(value)


def score_tuple(row: dict[str, Any]) -> tuple[float, float, float]:
    auprc = finite_float(row.get("auprc"))
    auroc = finite_float(row.get("auroc"))
    nauprc = finite_float(row.get("nauprc"))
    return (
        auprc if auprc is not None else -math.inf,
        auroc if auroc is not None else -math.inf,
        nauprc if nauprc is not None else -math.inf,
    )


def checkpoint_epoch(manifest: dict[str, Any]) -> int | None:
    text = str(manifest.get("checkpoint_path") or "")
    match = EPOCH_RE.search(text)
    return int(match.group(1)) if match else None


def exp_spec(task_name: str) -> tuple[str, str, bool]:
    if "extra_doubledrug" in task_name:
        return "exp08", "extra double", True
    return "exp07", "extra single", False


def annotate(
    record: dict[str, Any],
    *,
    view: str,
    epoch: int | None,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    result = dict(record)
    result["view"] = view
    result["epoch"] = "" if epoch is None else int(epoch)
    result["checkpoint_path"] = checkpoint_path or ""
    return result


def aggregate_method_records(
    records: list[dict[str, Any]],
    *,
    split: str,
    task: str,
    view: str,
) -> dict[str, Any]:
    aggregate = base.aggregate_records(records, split=split, task=task)
    epochs = {row.get("epoch") for row in records if row.get("epoch") != ""}
    checkpoints = {str(row.get("checkpoint_path") or "") for row in records if row.get("checkpoint_path")}
    aggregate["view"] = view
    aggregate["epoch"] = epochs.pop() if len(epochs) == 1 else ""
    aggregate["checkpoint_path"] = checkpoints.pop() if len(checkpoints) == 1 else "multiple"
    return aggregate


def collect_root_records(root: Path, *, view: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_exp_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        task_dir = manifest_path.parent
        manifest = base.load_json(manifest_path)
        task_name = str(manifest.get("task_name") or task_dir.name)
        exp, exp_label, double_drug = exp_spec(task_name)
        predictions, source = base.load_predictions(task_dir)
        predictions = base.enrich_with_feature_metadata(predictions, manifest)
        epoch = checkpoint_epoch(manifest)
        checkpoint_path = str(manifest.get("checkpoint_path") or "")
        test_label = predictions.get("test_label")
        if double_drug and test_label is not None:
            labels = test_label.astype("string").fillna("").str.strip()
            grouped_frames = [
                (group, predictions if group == "combined" else predictions.loc[labels.eq(group)].copy())
                for group in EXTRA_DOUBLE_TEST_LABEL_GROUPS
            ]
        else:
            grouped_frames = [("extra", predictions)]
        for split_name, split_predictions in grouped_frames:
            if split_predictions.empty:
                continue
            for method in METHODS:
                record = base.evaluate_prediction_frame(
                    split_predictions,
                    exp=exp,
                    task=task_name,
                    split=split_name,
                    method=method,
                    double_drug=double_drug,
                    source=source,
                )
                record = annotate(record, view=view, epoch=epoch, checkpoint_path=checkpoint_path)
                records.append(record)
                if split_name in {"extra", "combined"}:
                    by_exp_method.setdefault((exp, method), []).append(record)
    for (exp, method), method_records in sorted(by_exp_method.items()):
        if not method_records:
            continue
        task = "extra single" if exp == "exp07" else "extra double"
        mean_record = aggregate_method_records(method_records, split="mean_extra", task=task, view=view)
        mean_record["exp"] = exp
        mean_record["method"] = method
        records.append(mean_record)
    return records


def select_oracle(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected_records: list[dict[str, Any]] = []
    selection: dict[str, dict[str, Any]] = {}
    for exp in ("exp07", "exp08"):
        candidates = [
            row
            for row in records
            if row.get("exp") == exp and row.get("split") == "mean_extra" and row.get("method") == "original"
        ]
        if not candidates:
            continue
        best = max(candidates, key=score_tuple)
        best_epoch = best.get("epoch")
        selection[exp] = {
            "epoch": best_epoch,
            "auroc": best.get("auroc"),
            "auprc": best.get("auprc"),
            "auprc_baseline": best.get("auprc_baseline"),
            "nauprc": best.get("nauprc"),
            "valid_count": best.get("valid_count"),
            "checkpoint_path": best.get("checkpoint_path"),
        }
        for row in records:
            if row.get("exp") == exp and row.get("epoch") == best_epoch:
                selected = dict(row)
                selected["view"] = "oracle"
                selected_records.append(selected)
    return selected_records, selection


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "view",
        "epoch",
        "exp",
        "task",
        "split",
        "method",
        "auroc",
        "auprc",
        "auprc_baseline",
        "nauprc",
        "valid_count",
        "positive_count",
        "negative_count",
        "cell_drug_label_conflicts",
        "missing_time_groups",
        "checkpoint_path",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, records: list[dict[str, Any]], selection: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"oracle_selection": selection, "rows": records}, handle, ensure_ascii=False, indent=2, allow_nan=True)


def format_value(value: Any, precision: int) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def print_table(title: str, rows: list[dict[str, Any]], precision: int) -> None:
    if not rows:
        return
    print(f"\n## {title}")
    print("| view | epoch | exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |")
    print("|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {view} | {epoch} | {exp} | {task} | {split} | {method} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | {pos} | {neg} | {conflicts} | {missing_time} |".format(
                view=row.get("view", ""),
                epoch=row.get("epoch", ""),
                exp=row.get("exp", ""),
                task=row.get("task", ""),
                split=row.get("split", ""),
                method=row.get("method", ""),
                auroc=format_value(row.get("auroc"), precision),
                auprc=format_value(row.get("auprc"), precision),
                baseline=format_value(row.get("auprc_baseline"), precision),
                nauprc=format_value(row.get("nauprc"), precision),
                count=int(row.get("valid_count") or 0),
                pos=int(row.get("positive_count") or 0),
                neg=int(row.get("negative_count") or 0),
                conflicts=int(row.get("cell_drug_label_conflicts") or 0),
                missing_time=int(row.get("missing_time_groups") or 0),
            )
        )


def print_markdown(records: list[dict[str, Any]], selection: dict[str, dict[str, Any]], precision: int) -> None:
    if selection:
        print("\n## Oracle Selection")
        print("| exp | epoch | AUROC | AUPRC | baseline | n-AUPRC | count | checkpoint |")
        print("|---|---:|---:|---:|---:|---:|---:|---|")
        for exp, row in sorted(selection.items()):
            print(
                "| {exp} | {epoch} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | `{checkpoint}` |".format(
                    exp=exp,
                    epoch=row.get("epoch", ""),
                    auroc=format_value(row.get("auroc"), precision),
                    auprc=format_value(row.get("auprc"), precision),
                    baseline=format_value(row.get("auprc_baseline"), precision),
                    nauprc=format_value(row.get("nauprc"), precision),
                    count=int(row.get("valid_count") or 0),
                    checkpoint=row.get("checkpoint_path", ""),
                )
            )

    mean_rows = [row for row in records if row.get("split") == "mean_extra"]
    subset_rows = [row for row in records if row.get("split") != "mean_extra"]
    print_table("Mean Summary", sorted(mean_rows, key=lambda row: (row.get("view", ""), row.get("exp", ""), row.get("method", ""))), precision)
    print_table("Subset Detail", sorted(subset_rows, key=lambda row: (row.get("view", ""), row.get("exp", ""), row.get("task", ""), row.get("split", ""), row.get("method", ""))), precision)


def main() -> int:
    args = parse_args()
    valid_records = collect_root_records(args.valid_root, view="valid")
    oracle_candidate_records: list[dict[str, Any]] = []
    for root_name in sorted(glob.glob(args.oracle_root_glob)):
        root = Path(root_name)
        if root.is_dir():
            oracle_candidate_records.extend(collect_root_records(root, view="oracle_candidate"))
    oracle_records, selection = select_oracle(oracle_candidate_records)
    records = valid_records + oracle_records
    if not records:
        raise SystemExit("no exp09 valid/oracle records collected")
    if args.csv_out:
        write_csv(args.csv_out, records)
    if args.json_out:
        write_json(args.json_out, records, selection)
    if args.format == "markdown":
        print_markdown(records, selection, args.precision)
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
