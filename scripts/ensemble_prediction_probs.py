#!/usr/bin/env python3
"""Average ProteinTalk prediction parquet probabilities and report metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


POSITIVE_VALUES = {"sensitive", "responsive", "y", "yes", "1", "true"}
NEGATIVE_VALUES = {"non-responsive", "nonresponsive", "n", "no", "0", "false"}


def encode_binary_labels(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    mask = []
    for value in values:
        text = str(value).strip().lower()
        if text in POSITIVE_VALUES:
            labels.append(1.0)
            mask.append(False)
        elif text in NEGATIVE_VALUES:
            labels.append(0.0)
            mask.append(False)
        else:
            labels.append(0.0)
            mask.append(True)
    return np.asarray(labels, dtype=np.float32), np.asarray(mask, dtype=bool)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    keep = ~mask
    y_true = labels[keep]
    y_prob = probs[keep]
    if y_true.size == 0:
        return {"auroc": math.nan, "auprc": math.nan, "auprc_baseline": math.nan, "nauprc": math.nan, "count": 0.0}
    baseline = float(np.mean(y_true == 1.0))
    if np.unique(y_true).size < 2:
        return {"auroc": math.nan, "auprc": math.nan, "auprc_baseline": baseline, "nauprc": math.nan, "count": float(y_true.size)}
    auprc = float(average_precision_score(y_true, y_prob))
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": auprc,
        "auprc_baseline": baseline,
        "nauprc": auprc / baseline if baseline > 0 else math.nan,
        "count": float(y_true.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, help="Input predictions.parquet path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prob-column", default="pred_task_prob")
    parser.add_argument("--id-column", default="feature_row_index")
    parser.add_argument("--label-column", default="task_label")
    args = parser.parse_args()

    paths = [Path(item) for item in args.prediction]
    frames = [pd.read_parquet(path) for path in paths]
    if len(frames) < 2:
        raise ValueError("at least two --prediction files are required")
    base = frames[0].copy()
    if args.id_column not in base.columns:
        raise KeyError(f"missing id column in first prediction: {args.id_column}")
    prob_columns = []
    for index, (path, frame) in enumerate(zip(paths, frames, strict=True)):
        if args.id_column not in frame.columns or args.prob_column not in frame.columns:
            raise KeyError(f"{path} must contain {args.id_column!r} and {args.prob_column!r}")
        column = f"{args.prob_column}_member{index}"
        member = frame[[args.id_column, args.prob_column]].rename(columns={args.prob_column: column})
        base = base.merge(member, on=args.id_column, how="left", validate="one_to_one")
        if base[column].isna().any():
            raise ValueError(f"{path} does not cover every row from the first prediction")
        prob_columns.append(column)
    probs = base[prob_columns].astype(float).to_numpy()
    base[args.prob_column] = probs.mean(axis=1)

    labels, mask = encode_binary_labels(base[args.label_column])
    metrics = {
        "task": binary_metrics(labels, base[args.prob_column].astype(float).to_numpy(), mask),
        "members": [str(path) for path in paths],
        "prob_column": args.prob_column,
        "id_column": args.id_column,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base.drop(columns=prob_columns).to_parquet(output_dir / "predictions.parquet", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(json.dumps(metrics["task"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
