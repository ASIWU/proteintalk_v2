#!/usr/bin/env python3
"""Summarize fold-0 covariate diagnostics and ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.training_ready_fast_dataset import (
    BATCH_COVARIATE_COLUMNS,
    FastTrainingReadyArtifacts,
    encode_response_label,
    load_indices,
)


DEFAULT_COVARIATES = ["machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time"]
DEFAULT_TASKS = {
    "unseen_drug_fold0": ("ptv3_main_singledrug", "pert_stratified_5fold_fold0"),
    "unseen_cell_fold0": ("ptv3_main_singledrug", "cell_5fold_fold0"),
}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)


def parse_profiles(values: list[str]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for value in values:
        name, sep, payload = value.partition(":")
        if not name or not sep:
            raise ValueError(f"profile must be NAME:COV1,COV2 or NAME:__none__, got {value!r}")
        covariates = [] if payload == "__none__" else [item for item in payload.split(",") if item]
        profiles[name] = {"covariates": covariates}
    return profiles


def label_array(df: pd.DataFrame, rows: list[int]) -> tuple[np.ndarray, np.ndarray]:
    labels: list[float] = []
    masks: list[float] = []
    for value in df.iloc[rows]["PRISM1st_label_total"]:
        label, mask = encode_response_label(value)
        labels.append(float(label))
        masks.append(float(mask))
    return np.asarray(labels, dtype=np.float32), np.asarray(masks, dtype=np.float32)


def split_label_summary(df: pd.DataFrame, rows: list[int]) -> dict[str, Any]:
    labels, masks = label_array(df, rows)
    active = masks < 0.5
    positives = (labels[active] >= 0.5).sum()
    count = int(active.sum())
    return {
        "rows": int(len(rows)),
        "active_count": count,
        "positive_count": int(positives),
        "positive_rate": float(positives / count) if count else None,
    }


def covariate_split_summary(df: pd.DataFrame, rows: list[int], train_values: set[int], field: str) -> dict[str, Any]:
    source_col = BATCH_COVARIATE_COLUMNS.get(field, f"{field}_index")
    values = pd.to_numeric(df.iloc[rows][source_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    unique_values = set(int(item) for item in values)
    unseen_mask = np.asarray([int(item) not in train_values for item in values], dtype=bool)
    labels, masks = label_array(df, rows)
    active = masks < 0.5
    seen_active = active & ~unseen_mask
    unseen_active = active & unseen_mask

    def rate(mask: np.ndarray) -> float | None:
        denom = int(mask.sum())
        if denom == 0:
            return None
        return float((labels[mask] >= 0.5).sum() / denom)

    return {
        "rows": int(len(rows)),
        "unique": int(len(unique_values)),
        "unseen_unique": int(len(unique_values - train_values)),
        "unseen_rows": int(unseen_mask.sum()),
        "unseen_row_rate": float(unseen_mask.mean()) if len(values) else None,
        "seen_active_positive_rate": rate(seen_active),
        "unseen_active_positive_rate": rate(unseen_active),
    }


def top_categories(df: pd.DataFrame, rows: list[int], field: str, limit: int = 8) -> list[dict[str, Any]]:
    source_col = BATCH_COVARIATE_COLUMNS.get(field, f"{field}_index")
    raw_col = field if field in df.columns else source_col
    values = df.iloc[rows][raw_col].astype(str).tolist()
    counts = Counter(values)
    return [{"value": value, "rows": int(count)} for value, count in counts.most_common(limit)]


def build_split_diagnostics(
    *,
    artifacts: FastTrainingReadyArtifacts,
    split_root: Path,
    covariates: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    df = artifacts.df
    for task_label, (_, strategy) in DEFAULT_TASKS.items():
        train_rows = load_indices(split_root, "train", strategy)
        valid_rows = load_indices(split_root, "valid", strategy)
        test_rows = load_indices(split_root, "test", strategy)
        task_payload: dict[str, Any] = {
            "split_strategy": strategy,
            "labels": {
                "train": split_label_summary(df, train_rows),
                "valid": split_label_summary(df, valid_rows),
                "test": split_label_summary(df, test_rows),
            },
            "covariates": {},
        }
        for field in covariates:
            source_col = BATCH_COVARIATE_COLUMNS.get(field, f"{field}_index")
            if source_col not in df.columns:
                continue
            train_values = set(
                int(item)
                for item in pd.to_numeric(df.iloc[train_rows][source_col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
            task_payload["covariates"][field] = {
                "train": covariate_split_summary(df, train_rows, train_values, field),
                "valid": covariate_split_summary(df, valid_rows, train_values, field),
                "test": covariate_split_summary(df, test_rows, train_values, field),
                "test_top_categories": top_categories(df, test_rows, field),
            }
        result[task_label] = task_payload
    return result


def checkpoint_epoch(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"epoch=(\d+)", path)
    return int(match.group(1)) if match else None


def parse_runtime_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    runtimes: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("kind") != "train":
                continue
            exp = str(row.get("experiment") or "")
            try:
                duration = int(float(row.get("duration_sec") or 0))
            except ValueError:
                duration = None
            runtimes[exp] = {
                "status": row.get("status"),
                "duration_sec": duration,
                "start_utc": row.get("start_utc"),
                "end_utc": row.get("end_utc"),
            }
    return runtimes


def find_manifest(checkpoint_dir: Path, exp_prefix: str, task_label: str, profile: str) -> Path | None:
    suffix_by_task = {
        "unseen_drug_fold0": "single_pert_stratified_fold0",
        "unseen_cell_fold0": "single_cell_fold0",
    }
    suffix = suffix_by_task.get(task_label)
    if suffix is None:
        return None
    path = checkpoint_dir / f"{exp_prefix}_{task_label}_{profile}_{suffix}" / "run_manifest.json"
    return path if path.exists() else None


def collect_run_results(
    *,
    checkpoint_dir: Path,
    runtime_summary: Path,
    exp_prefix: str,
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    runtimes = parse_runtime_summary(runtime_summary)
    rows: list[dict[str, Any]] = []
    for task_label in DEFAULT_TASKS:
        for profile, profile_payload in profiles.items():
            manifest_path = find_manifest(checkpoint_dir, exp_prefix, task_label, profile)
            row: dict[str, Any] = {
                "task": task_label,
                "profile": profile,
                "covariates": profile_payload["covariates"],
                "manifest": str(manifest_path) if manifest_path else None,
                "status": "missing",
            }
            if manifest_path is None:
                rows.append(row)
                continue
            manifest = load_json(manifest_path)
            test_results = manifest.get("test_results") or []
            metrics = test_results[0] if test_results else {}
            exp_name = str(manifest.get("experiment_name") or manifest_path.parent.name)
            row.update(
                {
                    "status": manifest.get("run_status"),
                    "test_status": manifest.get("test_status"),
                    "experiment_name": exp_name,
                    "split_strategy": manifest.get("split_strategy"),
                    "batch_cov_list": manifest.get("batch_cov_list"),
                    "covariate_unk_for_unseen": bool(manifest.get("covariate_unk_for_unseen", False)),
                    "covariate_unk_fields": manifest.get("covariate_unk_fields") or [],
                    "covariate_unk_dropout": manifest.get("covariate_unk_dropout"),
                    "best_valid_auprc": manifest.get("best_model_score"),
                    "best_epoch": checkpoint_epoch(manifest.get("best_model_path")),
                    "test_auroc": metrics.get("test/auroc"),
                    "test_auprc": metrics.get("test/auprc"),
                    "test_acc": metrics.get("test/acc"),
                    "test_count": metrics.get("test/task_count"),
                    "duration_sec": runtimes.get(exp_name, {}).get("duration_sec"),
                    "runtime_status": runtimes.get(exp_name, {}).get("status"),
                }
            )
            rows.append(row)

    baselines = {
        row["task"]: row
        for row in rows
        if row["profile"] == "full" and row.get("test_auprc") is not None
    }
    for row in rows:
        base = baselines.get(row["task"])
        if not base:
            continue
        for metric in ["test_auroc", "test_auprc", "best_valid_auprc"]:
            if row.get(metric) is not None and base.get(metric) is not None:
                row[f"delta_{metric}"] = float(row[metric] - base[metric])
    return rows


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Covariate Analysis Fold0",
        "",
        f"Generated at: {payload['generated_at']}",
        "",
        "## Run Results",
        "",
    ]
    results = payload["run_results"]
    for task in DEFAULT_TASKS:
        task_rows = [row for row in results if row["task"] == task]
        task_rows.sort(key=lambda row: (row.get("test_auprc") is None, -(row.get("test_auprc") or -1.0)))
        lines.append(f"### {task}")
        table_rows = []
        for row in task_rows:
            table_rows.append(
                [
                    row["profile"],
                    ",".join(row.get("batch_cov_list") or row.get("covariates") or []) or "none",
                    "yes" if row.get("covariate_unk_for_unseen") else "no",
                    format_float(row.get("test_auroc")),
                    format_float(row.get("test_auprc")),
                    format_float(row.get("delta_test_auprc")),
                    format_float(row.get("best_valid_auprc")),
                    str(row.get("best_epoch") if row.get("best_epoch") is not None else "-"),
                    str(row.get("duration_sec") if row.get("duration_sec") is not None else "-"),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "profile",
                    "covariates",
                    "unk",
                    "test_auroc",
                    "test_auprc",
                    "delta_auprc_vs_full",
                    "best_valid_auprc",
                    "best_epoch",
                    "sec",
                ],
                table_rows,
            )
        )
        lines.append("")

    lines.extend(["## Split Diagnostics", ""])
    diagnostics = payload["split_diagnostics"]
    for task, task_payload in diagnostics.items():
        lines.append(f"### {task}")
        label_rows = []
        for split in ["train", "valid", "test"]:
            info = task_payload["labels"][split]
            label_rows.append(
                [
                    split,
                    str(info["rows"]),
                    str(info["active_count"]),
                    str(info["positive_count"]),
                    format_float(info["positive_rate"]),
                ]
            )
        lines.append(markdown_table(["split", "rows", "active", "positive", "pos_rate"], label_rows))
        lines.append("")
        cov_rows = []
        for field, field_payload in task_payload["covariates"].items():
            test_info = field_payload["test"]
            valid_info = field_payload["valid"]
            cov_rows.append(
                [
                    field,
                    str(field_payload["train"]["unique"]),
                    str(valid_info["unseen_unique"]),
                    format_float(valid_info["unseen_row_rate"]),
                    str(test_info["unseen_unique"]),
                    format_float(test_info["unseen_row_rate"]),
                    format_float(test_info["seen_active_positive_rate"]),
                    format_float(test_info["unseen_active_positive_rate"]),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "field",
                    "train_unique",
                    "valid_unseen_unique",
                    "valid_unseen_row_rate",
                    "test_unseen_unique",
                    "test_unseen_row_rate",
                    "test_seen_pos_rate",
                    "test_unseen_pos_rate",
                ],
                cov_rows,
            )
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-prefix", required=True)
    parser.add_argument("--checkpoint-dir", default=str(REPO_ROOT / "checkpoints"))
    parser.add_argument("--runtime-summary", default=None)
    parser.add_argument("--training-ready-root", default=str(REPO_ROOT / "data" / "training_ready"))
    parser.add_argument("--dataset-group", default="ptv3")
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    profiles = parse_profiles(args.profiles)
    training_ready_root = Path(args.training_ready_root)
    task_name = "ptv3_main_singledrug"
    task_dir = training_ready_root / args.dataset_group / "tasks" / task_name
    meta_path = training_ready_root / args.dataset_group / "global_meta.json"
    split_root = training_ready_root / args.dataset_group / "splits" / task_name
    artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
    runtime_summary = Path(args.runtime_summary) if args.runtime_summary else REPO_ROOT / "logs" / f"{args.exp_prefix}_runtime_summary.tsv"

    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "exp_prefix": args.exp_prefix,
        "profiles": profiles,
        "runtime_summary": str(runtime_summary),
        "split_diagnostics": build_split_diagnostics(
            artifacts=artifacts,
            split_root=split_root,
            covariates=DEFAULT_COVARIATES,
        ),
        "run_results": collect_run_results(
            checkpoint_dir=Path(args.checkpoint_dir),
            runtime_summary=runtime_summary,
            exp_prefix=args.exp_prefix,
            profiles=profiles,
        ),
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    dump_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"[covariate-analysis] wrote {output_json}")
    print(f"[covariate-analysis] wrote {output_md}")


if __name__ == "__main__":
    main()
