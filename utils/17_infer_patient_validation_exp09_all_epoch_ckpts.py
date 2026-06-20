#!/usr/bin/env python3
"""Infer patientVali260605v3 with every saved exp09 unified-head checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "utils" / "16_infer_patient_validation_all_epoch_ckpts.py"
DEFAULT_RUN_NAME = "patientVali260605v3"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "20260615_patientVali260605v3_exp09_all_epoch_ckpts"
DEFAULT_READABLE_DIR = REPO_ROOT / "outputs" / "0615v3_exp09_all_epoch_ckpts"
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready_patientVali260605v3"
DEFAULT_EXP09_DIR = (
    REPO_ROOT / "checkpoints" / "20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra"
)
DEFAULT_BUILD_SUMMARY = (
    REPO_ROOT
    / "data"
    / "training_ready_patientVali260605v3"
    / "ptv3"
    / "patientVali260605v3_build_summary.json"
)
DEFAULT_FEATURE_SUMMARY = (
    REPO_ROOT
    / "data"
    / "training_ready_patientVali260605v3"
    / "ptv3"
    / "patientVali260605v3_feature_build_summary.json"
)
DEFAULT_PYTHON_BIN = "/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python"

CKPT_RE = re.compile(r"epoch=(\d+)(?:-step=(\d+))?\.ckpt$")


def load_base_module():
    spec = importlib.util.spec_from_file_location("patient_all_epoch_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--training-ready-root", type=Path, default=DEFAULT_TRAINING_READY_ROOT)
    parser.add_argument("--exp09-dir", type=Path, default=DEFAULT_EXP09_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readable-output-dir", type=Path, default=DEFAULT_READABLE_DIR)
    parser.add_argument("--build-summary", type=Path, default=DEFAULT_BUILD_SUMMARY)
    parser.add_argument("--feature-summary", type=Path, default=DEFAULT_FEATURE_SUMMARY)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="256")
    parser.add_argument("--force", action="store_true", help="Rerun inference even when predictions already exist.")
    parser.add_argument("--skip-infer", action="store_true", help="Only rebuild combined/readable outputs from existing predictions.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-checkpoints", type=int, help="Debug helper: only process the first N epoch checkpoints.")
    return parser.parse_args()


def epoch_ckpts(directory: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    for path in directory.glob("epoch=*.ckpt"):
        match = CKPT_RE.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return [path for _, path in sorted(checkpoints)]


def checkpoint_meta(path: Path) -> dict[str, Any]:
    match = CKPT_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported checkpoint filename: {path}")
    epoch = int(match.group(1))
    step_text = match.group(2)
    step = int(step_text) if step_text is not None else -1
    return {
        "checkpoint_family": "exp09",
        "checkpoint_epoch": epoch,
        "checkpoint_step": step,
        "checkpoint_label": f"exp09_epoch{epoch:03d}",
        "checkpoint_path": base.repo_relative(path),
    }


def normalize_exp09_prediction_frame(frame: pd.DataFrame, build_summary: dict[str, Any]) -> pd.DataFrame:
    readable = base.normalize_prediction_frame(frame, build_summary)
    readable["model_source"] = readable["checkpoint_label"].astype(str) + "_unified_task"
    readable["score_name"] = "pred_task_prob"
    readable["prediction_score"] = pd.to_numeric(readable["pred_task_prob"], errors="coerce")
    readable["rank_in_task_checkpoint"] = (
        readable.groupby(["checkpoint_label", "task_name"])["prediction_score"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    readable["rank_in_dataset_checkpoint"] = (
        readable.groupby(["checkpoint_label", "dataset"])["prediction_score"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    return readable.sort_values(
        ["checkpoint_epoch", "dataset", "task_name", "rank_in_task_checkpoint", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def write_readme(
    path: Path,
    *,
    generated_at: str,
    readable: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    task_checkpoint_summary: pd.DataFrame,
    dataset_checkpoint_summary: pd.DataFrame,
    manifest: dict[str, Any],
    feature_summary: dict[str, Any],
    top_n: int,
) -> None:
    all_drugs = pd.concat([readable["drug_1"], readable["drug_2"]]).replace("", pd.NA).dropna()
    overall_rows = [
        {"metric": "Prediction rows across all checkpoints", "value": base.format_int(len(readable))},
        {"metric": "Single-drug prediction rows", "value": base.format_int(readable["treatment_type"].eq("single").sum())},
        {"metric": "Double-drug prediction rows", "value": base.format_int(readable["treatment_type"].eq("double").sum())},
        {"metric": "exp09 checkpoints", "value": base.format_int(readable["checkpoint_label"].nunique())},
        {"metric": "Tasks", "value": base.format_int(readable["task_name"].nunique())},
        {"metric": "Datasets", "value": base.format_int(readable["dataset"].nunique())},
        {"metric": "Unique cells", "value": base.format_int(readable["cell_id"].nunique())},
        {"metric": "Unique drugs", "value": base.format_int(all_drugs.nunique())},
    ]
    checkpoint_rows = [
        {
            "label": row["checkpoint_label"],
            "rows": base.format_int(row["prediction_rows"]),
            "tasks": base.format_int(row["tasks"]),
            "single": base.format_int(row["single_predictions"]),
            "double": base.format_int(row["double_predictions"]),
            "mean": base.format_float(row["score_mean"]),
            "min": base.format_float(row["score_min"]),
            "max": base.format_float(row["score_max"]),
        }
        for _, row in checkpoint_summary.sort_values(["checkpoint_epoch"]).iterrows()
    ]
    task_rows = [
        {
            "label": row["checkpoint_label"],
            "task": row["task_name"],
            "rows": base.format_int(row["prediction_rows"]),
            "score": row["score_name"],
            "mean": base.format_float(row["score_mean"]),
            "min": base.format_float(row["score_min"]),
            "max": base.format_float(row["score_max"]),
        }
        for _, row in task_checkpoint_summary.sort_values(["checkpoint_epoch", "task_name"]).iterrows()
    ]
    dataset_rows = [
        {
            "label": row["checkpoint_label"],
            "dataset": row["dataset"],
            "rows": base.format_int(row["prediction_rows"]),
            "single": base.format_int(row["single_predictions"]),
            "double": base.format_int(row["double_predictions"]),
            "mean": base.format_float(row["score_mean"]),
            "transform": row.get("expression_transform", ""),
        }
        for _, row in dataset_checkpoint_summary.sort_values(["checkpoint_epoch", "dataset"]).iterrows()
    ]
    top_rows = []
    for _, row in readable.sort_values("prediction_score", ascending=False).head(top_n).iterrows():
        top_rows.append(
            {
                "rank": base.format_int(len(top_rows) + 1),
                "ckpt": row["checkpoint_label"],
                "dataset": row["dataset"],
                "type": row["treatment_type"],
                "cell": row["cell_id"],
                "drug_pair": row["drug_pair"],
                "score": base.format_float(row["prediction_score"]),
            }
        )

    lines = [
        "# patientVali260605v3 Exp09 All-Epoch Checkpoint Inference",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "This directory contains readable CSV exports for patientVali260605v3 inference using every saved exp09 unified-head epoch checkpoint.",
        "No feature rebuild was performed; the existing v3 training-ready data and derived features were reused.",
        "",
        "## Files",
        "",
        "- `combined_predictions_readable.csv`: all row-level predictions with checkpoint metadata.",
        "- `per_task_checkpoint_csv/*.csv`: one readable CSV per checkpoint/task pair.",
        "- `checkpoint_summary.csv`: one row per checkpoint.",
        "- `task_checkpoint_summary.csv`: one row per checkpoint/task pair.",
        "- `dataset_checkpoint_summary.csv`: one row per checkpoint/dataset pair.",
        "- `manifest.json`: machine-readable input/output record.",
        "",
        "## Source Checkpoints",
        "",
        f"- exp09: `{manifest['source_checkpoints']['exp09_dir']}`",
        "",
        "## Score Columns",
        "",
        "- exp09 unified-head checkpoints use `pred_task_prob` as `prediction_score` for both single-drug and double-drug tasks.",
        "- `pred_response_prob` and `pred_synergy_prob` are retained in the row-level CSV for compatibility with older exports.",
        "",
        "## Overall Counts",
        "",
        base.markdown_table(overall_rows, [("metric", "Metric"), ("value", "Value")]),
        "",
        "## Checkpoint Summary",
        "",
        base.markdown_table(
            checkpoint_rows,
            [
                ("label", "Checkpoint"),
                ("rows", "Rows"),
                ("tasks", "Tasks"),
                ("single", "Single"),
                ("double", "Double"),
                ("mean", "Score mean"),
                ("min", "Min"),
                ("max", "Max"),
            ],
        ),
        "",
        "## Task by Checkpoint",
        "",
        base.markdown_table(
            task_rows,
            [
                ("label", "Checkpoint"),
                ("task", "Task"),
                ("rows", "Rows"),
                ("score", "Score"),
                ("mean", "Mean"),
                ("min", "Min"),
                ("max", "Max"),
            ],
        ),
        "",
        "## Dataset by Checkpoint",
        "",
        base.markdown_table(
            dataset_rows,
            [
                ("label", "Checkpoint"),
                ("dataset", "Dataset"),
                ("rows", "Rows"),
                ("single", "Single"),
                ("double", "Double"),
                ("mean", "Score mean"),
                ("transform", "Expression transform"),
            ],
        ),
        "",
        f"## Top {top_n} Predictions",
        "",
        base.markdown_table(
            top_rows,
            [
                ("rank", "Rank"),
                ("ckpt", "Checkpoint"),
                ("dataset", "Dataset"),
                ("type", "Type"),
                ("cell", "Cell"),
                ("drug_pair", "Drug pair"),
                ("score", "Score"),
            ],
        ),
        "",
        "## Feature Artifacts",
        "",
        *base.collect_feature_lines(feature_summary),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.training_ready_root = base.resolve_path(args.training_ready_root)
    args.exp09_dir = base.resolve_path(args.exp09_dir)
    args.output_root = base.resolve_path(args.output_root)
    args.readable_output_dir = base.resolve_path(args.readable_output_dir)
    args.build_summary = base.resolve_path(args.build_summary)
    args.feature_summary = base.resolve_path(args.feature_summary)

    build_summary = base.read_json(args.build_summary)
    feature_summary = base.read_json(args.feature_summary, required=False)
    derived_root = args.training_ready_root / "ptv3" / "derived"
    task_prefix = f"ptv3_{args.run_name}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.readable_output_dir.mkdir(parents=True, exist_ok=True)

    common_args = base.build_common_infer_args(args, derived_root)
    exp09_extra_args = [
        "--graph-pair-add-scale",
        "0.5",
        "--pair-fusion-mode",
        "dual",
        "--pair-type-features",
        "--use-ddi",
    ]
    task_specs: list[tuple[str, str]] = [
        *((suffix, "single") for suffix in base.SINGLE_TASKS),
        *((suffix, "double") for suffix in base.DOUBLE_TASKS),
    ]
    checkpoints = epoch_ckpts(args.exp09_dir)
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]
    if not checkpoints:
        raise RuntimeError(f"No exp09 epoch checkpoints found under {args.exp09_dir}")

    inference_records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    generated_at = base.iso_now()

    for checkpoint in checkpoints:
        meta = checkpoint_meta(checkpoint)
        for task_suffix, treatment_type in task_specs:
            task_name = f"{task_prefix}_{task_suffix}"
            output_dir = args.output_root / meta["checkpoint_label"] / task_name
            if args.skip_infer:
                prediction_path = output_dir / "predictions.parquet"
                if not prediction_path.exists():
                    prediction_path = output_dir / "predictions.csv"
                    if not prediction_path.exists():
                        raise FileNotFoundError(output_dir / "predictions.parquet")
            else:
                print(f"[infer] {meta['checkpoint_label']} {task_name}")
                prediction_path = base.run_infer(
                    args=args,
                    task_name=task_name,
                    task_head="unified",
                    checkpoint=checkpoint,
                    output_dir=output_dir,
                    common_args=common_args,
                    extra_args=exp09_extra_args,
                )
            frame = base.read_prediction(prediction_path)
            for key, value in meta.items():
                frame[key] = value
            frame["task_name"] = task_name
            frame["treatment_type"] = treatment_type
            frame["source_prediction_file"] = base.repo_relative(prediction_path)
            score = pd.to_numeric(frame["pred_task_prob"], errors="coerce")
            inference_records.append(
                {
                    **meta,
                    "task_name": task_name,
                    "task_head": "unified",
                    "treatment_type": treatment_type,
                    "prediction_file": base.repo_relative(prediction_path),
                    "prediction_rows": int(len(frame)),
                    "score_column": "pred_task_prob",
                    "score_min": float(score.min()),
                    "score_mean": float(score.mean()),
                    "score_max": float(score.max()),
                }
            )
            frames.append(frame)

    combined_raw = pd.concat(frames, ignore_index=True)
    combined_raw_parquet = args.output_root / "combined_predictions.parquet"
    combined_raw_csv = args.output_root / "combined_predictions.csv"
    combined_raw.to_parquet(combined_raw_parquet, index=False)
    combined_raw.to_csv(combined_raw_csv, index=False, float_format="%.8g")

    readable = normalize_exp09_prediction_frame(combined_raw, build_summary)
    checkpoint_summary = base.make_checkpoint_summary(readable)
    task_checkpoint_summary = base.make_task_checkpoint_summary(readable)
    dataset_checkpoint_summary = base.make_dataset_checkpoint_summary(readable, build_summary)

    readable_csv = args.readable_output_dir / "combined_predictions_readable.csv"
    checkpoint_summary_csv = args.readable_output_dir / "checkpoint_summary.csv"
    task_checkpoint_summary_csv = args.readable_output_dir / "task_checkpoint_summary.csv"
    dataset_checkpoint_summary_csv = args.readable_output_dir / "dataset_checkpoint_summary.csv"
    readable.to_csv(readable_csv, index=False, float_format="%.8g")
    checkpoint_summary.to_csv(checkpoint_summary_csv, index=False, float_format="%.8g")
    task_checkpoint_summary.to_csv(task_checkpoint_summary_csv, index=False, float_format="%.8g")
    dataset_checkpoint_summary.to_csv(dataset_checkpoint_summary_csv, index=False, float_format="%.8g")
    per_task_files = base.write_per_task_checkpoint_csvs(readable, args.readable_output_dir)

    manifest = {
        "generated_at": generated_at,
        "run_name": args.run_name,
        "training_ready_root": base.repo_relative(args.training_ready_root),
        "source_checkpoints": {
            "exp09_dir": base.repo_relative(args.exp09_dir),
            "exp09_checkpoints": [base.repo_relative(path) for path in checkpoints],
        },
        "raw_output_root": base.repo_relative(args.output_root),
        "readable_output_dir": base.repo_relative(args.readable_output_dir),
        "combined_raw_parquet": base.repo_relative(combined_raw_parquet),
        "combined_raw_csv": base.repo_relative(combined_raw_csv),
        "combined_readable_csv": base.repo_relative(readable_csv),
        "checkpoint_summary_csv": base.repo_relative(checkpoint_summary_csv),
        "task_checkpoint_summary_csv": base.repo_relative(task_checkpoint_summary_csv),
        "dataset_checkpoint_summary_csv": base.repo_relative(dataset_checkpoint_summary_csv),
        "per_task_checkpoint_csv": per_task_files,
        "inference_records": inference_records,
        "total_prediction_rows": int(len(readable)),
        "single_prediction_rows": int(readable["treatment_type"].eq("single").sum()),
        "double_prediction_rows": int(readable["treatment_type"].eq("double").sum()),
        "score_column": "pred_task_prob",
    }
    base.write_json(args.output_root / "inference_summary.json", manifest)
    base.write_json(args.readable_output_dir / "manifest.json", manifest)
    write_readme(
        args.readable_output_dir / "README.md",
        generated_at=generated_at,
        readable=readable,
        checkpoint_summary=checkpoint_summary,
        task_checkpoint_summary=task_checkpoint_summary,
        dataset_checkpoint_summary=dataset_checkpoint_summary,
        manifest=manifest,
        feature_summary=feature_summary,
        top_n=args.top_n,
    )
    print(f"[done] readable outputs: {base.repo_relative(args.readable_output_dir)}")
    print(f"[done] rows={len(readable)} checkpoints={readable['checkpoint_label'].nunique()}")


if __name__ == "__main__":
    main()
