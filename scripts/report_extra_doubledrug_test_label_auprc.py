#!/usr/bin/env python3
"""Report updated extra double-drug metrics by test_label from existing predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "rawdata" / "update_0526" / "extra_doubledrug"
TEST_LABEL_GROUPS = ("unseenCell_seenDrugCombo", "unseenCell_unseenDrugCombo")
POSITIVE_LABELS = {"syn", "synergy", "synergistic", "y", "yes", "1", "true"}
NEGATIVE_LABELS = {"non-syn", "nonsyn", "non-synergy", "non_synergy", "n", "no", "0", "false"}


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    raw_file: str


TASK_SPECS = (
    TaskSpec("ptv3_extra_doubledrug_guomics", "260525ptv3_Guomics_drug_combo_unique_with_smlies_test_label.csv"),
    TaskSpec("ptv3_extra_doubledrug_nc", "260525nc_drugComb_info_unique_with_smiles_test_label.csv"),
    TaskSpec("ptv3_extra_doubledrug_nature", "260525nature_drugComb_info_unique_with_smiles_test_label.csv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join existing extra double-drug predictions to update_0526 test/test_label "
            "metadata and recompute AUROC/AUPRC plus nAUPRC=AUPRC/(positive_count/valid_count) "
            "for each evaluation group."
        )
    )
    parser.add_argument("output_root", type=Path, help="Inference output root containing per-task directories")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--prob-column", default="pred_task_prob")
    parser.add_argument("--label-column", default="task_label")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow predictions with fewer rows than the raw CSV, using feature_row_index to align rows.",
    )
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("table", "markdown", "csv"), default="table")
    parser.add_argument("--precision", type=int, default=6)
    return parser.parse_args()


def load_predictions(task_dir: Path) -> tuple[pd.DataFrame, Path]:
    parquet_path = task_dir / "predictions.parquet"
    csv_path = task_dir / "predictions.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), parquet_path
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False), csv_path
    raise FileNotFoundError(f"missing predictions parquet/csv under {task_dir}")


def encode_binary_label(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number == 1.0:
            return 1.0
        if number == 0.0:
            return 0.0
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None
    if text in POSITIVE_LABELS:
        return 1.0
    if text in NEGATIVE_LABELS:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number == 1.0:
        return 1.0
    if number == 0.0:
        return 0.0
    return None


def align_raw_metadata(predictions: pd.DataFrame, raw: pd.DataFrame, *, allow_partial: bool) -> pd.DataFrame:
    if len(predictions) != len(raw) and not allow_partial:
        raise ValueError(
            f"prediction/raw row count mismatch: predictions={len(predictions)} raw={len(raw)}. "
            "Use --allow-partial for smoke or filtered-split predictions."
        )
    if "feature_row_index" in predictions.columns:
        row_indices = pd.to_numeric(predictions["feature_row_index"], errors="raise").astype(int).to_numpy()
        if np.any(row_indices < 0) or np.any(row_indices >= len(raw)):
            raise ValueError("feature_row_index contains values outside the raw CSV row range")
        return raw.iloc[row_indices].reset_index(drop=True)
    if len(predictions) != len(raw):
        raise ValueError("partial predictions require feature_row_index for row alignment")
    return raw.reset_index(drop=True)


def eval_test_mask(metadata: pd.DataFrame) -> pd.Series:
    test_raw = metadata["test"]
    test_numeric = pd.to_numeric(test_raw, errors="coerce")
    test_text = test_raw.astype("string").fillna("").str.strip().str.lower()
    test_eval = test_numeric.eq(1) | test_text.isin({"true", "yes", "y"})
    labels = metadata["test_label"].astype("string").fillna("").str.strip()
    return test_eval & labels.ne("") & labels.str.lower().ne("delete")


def auprc_record(
    *,
    task_name: str,
    group: str,
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    group_mask: np.ndarray,
    label_column: str,
    prob_column: str,
    prediction_path: Path,
) -> dict[str, Any]:
    labels = np.asarray([encode_binary_label(value) for value in predictions[label_column].tolist()], dtype=object)
    probs = pd.to_numeric(predictions[prob_column], errors="coerce").to_numpy(dtype=np.float64)
    known = np.asarray([value is not None for value in labels], dtype=bool)
    y_true = np.asarray([0.0 if value is None else float(value) for value in labels], dtype=np.float64)
    valid = group_mask & known & np.isfinite(probs)
    y_true = y_true[valid]
    y_prob = probs[valid]
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    valid_count = int(len(y_true))
    baseline = (positive_count / valid_count) if valid_count else float("nan")
    auroc = float("nan")
    auprc = float("nan")
    acc = float("nan")
    if valid_count:
        acc = float(((y_prob >= 0.5).astype(np.float64) == y_true).mean())
    if positive_count > 0 and negative_count > 0:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    nauprc = (auprc / baseline) if baseline and math.isfinite(baseline) and math.isfinite(auprc) else float("nan")
    return {
        "task_name": task_name,
        "group": group,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_baseline": baseline,
        "nauprc": nauprc,
        "acc": acc,
        "valid_count": valid_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "raw_test_count": int(group_mask.sum()),
        "prediction_path": str(prediction_path),
        "test_label_values": ",".join(sorted(metadata.loc[group_mask, "test_label"].astype(str).unique())),
    }


def summarize_task(
    spec: TaskSpec,
    *,
    output_root: Path,
    raw_root: Path,
    label_column: str,
    prob_column: str,
    allow_partial: bool,
) -> list[dict[str, Any]]:
    predictions, prediction_path = load_predictions(output_root / spec.task_name)
    for column in (label_column, prob_column):
        if column not in predictions.columns:
            raise KeyError(f"{prediction_path}: missing required column {column!r}")
    raw_path = raw_root / spec.raw_file
    raw = pd.read_csv(raw_path, low_memory=False)
    for column in ("test", "test_label"):
        if column not in raw.columns:
            raise KeyError(f"{raw_path}: missing required column {column!r}")
    metadata = align_raw_metadata(predictions, raw, allow_partial=allow_partial)
    eval_mask = eval_test_mask(metadata).to_numpy(dtype=bool)
    labels = metadata["test_label"].astype("string").fillna("").str.strip()
    records = []
    for group in TEST_LABEL_GROUPS:
        group_mask = eval_mask & labels.eq(group).to_numpy(dtype=bool)
        records.append(
            auprc_record(
                task_name=spec.task_name,
                group=group,
                predictions=predictions,
                metadata=metadata,
                group_mask=group_mask,
                label_column=label_column,
                prob_column=prob_column,
                prediction_path=prediction_path,
            )
        )
    records.append(
        auprc_record(
            task_name=spec.task_name,
            group="combined",
            predictions=predictions,
            metadata=metadata,
            group_mask=eval_mask,
            label_column=label_column,
            prob_column=prob_column,
            prediction_path=prediction_path,
        )
    )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_name",
        "group",
        "auroc",
        "auprc",
        "auprc_baseline",
        "nauprc",
        "acc",
        "valid_count",
        "positive_count",
        "negative_count",
        "raw_test_count",
        "prediction_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, ensure_ascii=False, indent=2, allow_nan=True)


def format_float(value: Any, precision: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return "nan"
    return f"{number:.{precision}f}"


def print_rows(rows: list[dict[str, Any]], *, output_format: str, precision: int) -> None:
    columns = [
        "task_name",
        "group",
        "auroc",
        "auprc",
        "auprc_baseline",
        "nauprc",
        "acc",
        "valid_count",
        "positive_count",
        "negative_count",
    ]
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in columns}
            out["auroc"] = format_float(out["auroc"], precision)
            out["auprc"] = format_float(out["auprc"], precision)
            out["auprc_baseline"] = format_float(out["auprc_baseline"], precision)
            out["nauprc"] = format_float(out["nauprc"], precision)
            out["acc"] = format_float(out["acc"], precision)
            writer.writerow(out)
        return
    if output_format == "markdown":
        print("| task | group | AUROC | AUPRC | baseline | nAUPRC | ACC | valid | pos | neg |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            print(
                "| {task_name} | {group} | {auroc} | {auprc} | {auprc_baseline} | {nauprc} | {acc} | "
                "{valid_count} | {positive_count} | {negative_count} |".format(
                    **{
                        **row,
                        "auroc": format_float(row["auroc"], precision),
                        "auprc": format_float(row["auprc"], precision),
                        "auprc_baseline": format_float(row["auprc_baseline"], precision),
                        "nauprc": format_float(row["nauprc"], precision),
                        "acc": format_float(row["acc"], precision),
                    }
                )
            )
        return
    widths = {key: len(key) for key in columns}
    formatted = []
    for row in rows:
        item = {key: str(row.get(key, "")) for key in columns}
        item["auroc"] = format_float(row["auroc"], precision)
        item["auprc"] = format_float(row["auprc"], precision)
        item["auprc_baseline"] = format_float(row["auprc_baseline"], precision)
        item["nauprc"] = format_float(row["nauprc"], precision)
        item["acc"] = format_float(row["acc"], precision)
        formatted.append(item)
        for key, value in item.items():
            widths[key] = max(widths[key], len(value))
    print("  ".join(key.ljust(widths[key]) for key in columns))
    print("  ".join("-" * widths[key] for key in columns))
    for row in formatted:
        print("  ".join(row[key].ljust(widths[key]) for key in columns))


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    rows: list[dict[str, Any]] = []
    for spec in TASK_SPECS:
        rows.extend(
            summarize_task(
                spec,
                output_root=output_root,
                raw_root=args.raw_root,
                label_column=args.label_column,
                prob_column=args.prob_column,
                allow_partial=args.allow_partial,
            )
        )
    if args.csv_out:
        write_csv(args.csv_out, rows)
    if args.json_out:
        write_json(args.json_out, rows)
    print_rows(rows, output_format=args.format, precision=args.precision)


if __name__ == "__main__":
    main()
