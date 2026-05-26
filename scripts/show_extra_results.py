#!/usr/bin/env python3
"""Summarize extra single/double inference metrics from output folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/20260513_extra_double_all_train_infer_all_single_double_for_extra")
METRIC_KEYS = ("auroc", "auprc", "auprc_baseline", "nauprc", "acc")
COUNT_KEYS = ("valid_count", "positive_count", "negative_count", "count")
DEFAULT_COLUMNS = (
    ("task_name", "task"),
    ("head", "head"),
    ("auroc", "auroc"),
    ("auprc", "auprc"),
    ("auprc_baseline", "base"),
    ("nauprc", "nauprc"),
    ("acc", "acc"),
    ("valid_count", "valid"),
    ("positive_count", "pos"),
    ("negative_count", "neg"),
    ("n_predictions", "preds"),
)
AGGREGATE_COLUMNS = (
    ("head", "head"),
    ("datasets", "datasets"),
    ("mean_auroc", "mean_auroc"),
    ("mean_auprc", "mean_auprc"),
    ("mean_auprc_baseline", "mean_base"),
    ("mean_nauprc", "mean_nauprc"),
    ("mean_acc", "mean_acc"),
    ("valid_count", "valid_total"),
    ("positive_count", "pos_total"),
    ("negative_count", "neg_total"),
)
SORT_KEYS = (
    "task_name",
    "head",
    "auroc",
    "auprc",
    "auprc_baseline",
    "nauprc",
    "acc",
    "valid_count",
    "positive_count",
    "negative_count",
    "n_predictions",
)
NUMERIC_KEYS = {
    "auroc",
    "auprc",
    "auprc_baseline",
    "nauprc",
    "acc",
    "valid_count",
    "positive_count",
    "negative_count",
    "n_predictions",
    "datasets",
    "mean_auroc",
    "mean_auprc",
    "mean_auprc_baseline",
    "mean_nauprc",
    "mean_acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print a compact table of metrics.json files under extra inference "
            "output directories. By default it summarizes the current extra "
            "single-drug run."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_OUTPUT_DIR],
        help=(
            "Output directories or metrics.json files to summarize. Directories "
            "are searched recursively. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--all-heads",
        action="store_true",
        help=(
            "Show every non-empty metric head in each metrics.json. Without this, "
            "the manifest task_head is used, which is response for extra single "
            "and synergy for extra double."
        ),
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Also show heads with valid_count=0.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "csv"),
        default="table",
        help="Output format for stdout.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional CSV path for the detailed rows.",
    )
    parser.add_argument(
        "--sort-by",
        choices=SORT_KEYS,
        default="task_name",
        help="Column used to sort detailed rows.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort detailed rows in descending order.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for metric columns in text output.",
    )
    parser.add_argument(
        "--include-paths",
        action="store_true",
        help="Append metrics, prediction, and checkpoint paths to each detailed row.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Do not print the average-by-head summary for table/markdown output.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def find_metric_files(paths: list[Path]) -> list[Path]:
    metric_files: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"missing path: {path}")
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(path.rglob("metrics.json"))
        for candidate in candidates:
            if candidate.name != "metrics.json":
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                metric_files.append(candidate)
                seen.add(resolved)
    return sorted(metric_files)


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


def int_count(value: Any) -> int | None:
    number = finite_float(value)
    if number is None:
        return None
    return int(number)


def has_metric_shape(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in (*METRIC_KEYS, *COUNT_KEYS))


def choose_heads(
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    all_heads: bool,
) -> list[str]:
    candidates = [key for key, value in metrics.items() if has_metric_shape(value)]
    if all_heads:
        return candidates

    preferred = manifest.get("task_head")
    if isinstance(preferred, str) and preferred in candidates:
        return [preferred]
    if "task" in candidates:
        return ["task"]
    return candidates[:1]


def short_path(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if len(path.parts) >= 2:
        return str(Path(*path.parts[-2:]))
    return str(path)


def build_record(
    metrics_path: Path,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    head: str,
) -> dict[str, Any]:
    values = metrics.get(head, {})
    if not isinstance(values, dict):
        values = {}

    record: dict[str, Any] = {
        "task_name": str(manifest.get("task_name") or metrics_path.parent.name),
        "head": head,
        "split": "/".join(
            part
            for part in (
                str(manifest.get("split_strategy") or ""),
                str(manifest.get("split_name") or ""),
            )
            if part
        ),
        "n_predictions": int_count(manifest.get("n_predictions")),
        "checkpoint": short_path(manifest.get("checkpoint_path")),
        "prediction_path": str(manifest.get("prediction_path") or ""),
        "metrics_path": str(metrics_path),
    }
    for key in METRIC_KEYS:
        record[key] = finite_float(values.get(key))
    record["valid_count"] = int_count(values.get("valid_count"))
    if record["valid_count"] is None:
        record["valid_count"] = int_count(values.get("count"))
    record["positive_count"] = int_count(values.get("positive_count"))
    record["negative_count"] = int_count(values.get("negative_count"))
    return record


def collect_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Path]]:
    metric_files = find_metric_files(args.paths)
    records: list[dict[str, Any]] = []
    for metrics_path in metric_files:
        metrics = load_json(metrics_path)
        manifest_path = metrics_path.with_name("run_manifest.json")
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        for head in choose_heads(metrics, manifest, args.all_heads):
            record = build_record(metrics_path, metrics, manifest, head)
            if not args.show_empty and (record.get("valid_count") or 0) <= 0:
                continue
            records.append(record)
    return records, metric_files


def sort_records(records: list[dict[str, Any]], sort_by: str, descending: bool) -> None:
    def sort_value(record: dict[str, Any]) -> tuple[int, Any]:
        value = record.get(sort_by)
        if value is None:
            return (1, "")
        return (0, value)

    records.sort(key=sort_value, reverse=descending)


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_head: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if (record.get("valid_count") or 0) <= 0:
            continue
        by_head.setdefault(str(record["head"]), []).append(record)

    aggregates: list[dict[str, Any]] = []
    for head in sorted(by_head):
        head_records = by_head[head]
        aggregate: dict[str, Any] = {
            "head": head,
            "datasets": len(head_records),
            "valid_count": sum(record.get("valid_count") or 0 for record in head_records),
            "positive_count": sum(record.get("positive_count") or 0 for record in head_records),
            "negative_count": sum(record.get("negative_count") or 0 for record in head_records),
        }
        for metric_key in METRIC_KEYS:
            values = [record[metric_key] for record in head_records if record.get(metric_key) is not None]
            aggregate[f"mean_{metric_key}"] = mean(values) if values else None
        aggregates.append(aggregate)
    return aggregates


def display_value(value: Any, key: str, precision: int) -> str:
    if value is None:
        return ""
    if key in METRIC_KEYS or key.startswith("mean_"):
        if isinstance(value, (float, int)):
            return f"{float(value):.{precision}f}"
    if key in NUMERIC_KEYS:
        if isinstance(value, (float, int)):
            return str(int(value))
    return str(value)


def render_table(rows: list[dict[str, Any]], columns: tuple[tuple[str, str], ...], precision: int) -> str:
    rendered_rows = [
        [display_value(row.get(key), key, precision) for key, _header in columns]
        for row in rows
    ]
    headers = [header for _key, header in columns]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered_rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rendered_rows:
        cells: list[str] = []
        for index, cell in enumerate(row):
            key = columns[index][0]
            if key in NUMERIC_KEYS:
                cells.append(cell.rjust(widths[index]))
            else:
                cells.append(cell.ljust(widths[index]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def render_markdown(rows: list[dict[str, Any]], columns: tuple[tuple[str, str], ...], precision: int) -> str:
    headers = [header for _key, header in columns]
    align = ["---:" if key in NUMERIC_KEYS else "---" for key, _header in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        values = [display_value(row.get(key), key, precision) for key, _header in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], columns: tuple[tuple[str, str], ...], output_path: Path | None) -> None:
    handle = output_path.open("w", newline="", encoding="utf-8") if output_path else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _header in columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key, _header in columns})
    finally:
        if output_path:
            handle.close()


def detailed_columns(include_paths: bool) -> tuple[tuple[str, str], ...]:
    if not include_paths:
        return DEFAULT_COLUMNS
    return DEFAULT_COLUMNS + (
        ("split", "split"),
        ("checkpoint", "checkpoint"),
        ("prediction_path", "prediction_path"),
        ("metrics_path", "metrics_path"),
    )


def print_text_report(args: argparse.Namespace, records: list[dict[str, Any]], metric_files: list[Path]) -> None:
    columns = detailed_columns(args.include_paths)
    if args.format == "table":
        print(f"Metric files found: {len(metric_files)}")
        print(f"Rows shown: {len(records)}")
        print()
        print(render_table(records, columns, args.precision))
        if not args.no_aggregate:
            aggregates = aggregate_records(records)
            if aggregates:
                print()
                print("Average by head:")
                print(render_table(aggregates, AGGREGATE_COLUMNS, args.precision))
    elif args.format == "markdown":
        print(render_markdown(records, columns, args.precision))
        if not args.no_aggregate:
            aggregates = aggregate_records(records)
            if aggregates:
                print()
                print("Average by head:")
                print(render_markdown(aggregates, AGGREGATE_COLUMNS, args.precision))


def main() -> int:
    args = parse_args()
    try:
        records, metric_files = collect_records(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not metric_files:
        print("[error] no metrics.json files found", file=sys.stderr)
        return 1
    if not records:
        print("[error] no metric rows to show; try --show-empty or --all-heads", file=sys.stderr)
        return 1

    sort_records(records, args.sort_by, args.descending)
    columns = detailed_columns(args.include_paths)
    if args.csv_out:
        write_csv(records, columns, args.csv_out)
    if args.format == "csv":
        write_csv(records, columns, None)
    else:
        print_text_report(args, records, metric_files)
        if args.csv_out:
            print()
            print(f"Wrote CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
