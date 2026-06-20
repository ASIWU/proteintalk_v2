#!/usr/bin/env python3
"""Infer patientVali260605v3 with every saved exp07/exp08 epoch checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "patientVali260605v3"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "20260612_patientVali260605v3_all_epoch_ckpts"
DEFAULT_READABLE_DIR = REPO_ROOT / "outputs" / "0612v3_all_epoch_ckpts"
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready_patientVali260605v3"
DEFAULT_EXP07_DIR = (
    REPO_ROOT
    / "checkpoints"
    / "20260612_ptv01_08_posweight_combo_selected_v1_allckpt_exp07_extra_single_all_train_infer_all_single_for_extra"
)
DEFAULT_EXP08_DIR = (
    REPO_ROOT
    / "checkpoints"
    / "20260612_ptv01_08_posweight_combo_selected_v1_allckpt_exp08_extra_double_all_train_infer_all_single_double_for_extra"
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

SINGLE_TASKS = (
    "p2_lung2020_single",
    "p3_lung2024_single",
    "p5_breast_single",
)
DOUBLE_TASKS = (
    "p1_ovarian_double",
    "p2_lung2020_double",
    "p3_lung2024_double",
    "p4_colon_double",
    "p5_breast_double",
)

CKPT_RE = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt$")

READABLE_COLUMNS = [
    "checkpoint_family",
    "checkpoint_label",
    "checkpoint_epoch",
    "checkpoint_step",
    "checkpoint_path",
    "dataset",
    "dataset_display_name",
    "task_name",
    "treatment_type",
    "model_source",
    "score_name",
    "prediction_score",
    "rank_in_task_checkpoint",
    "rank_in_dataset_checkpoint",
    "cell_id",
    "cell_type",
    "drug_1",
    "drug_2",
    "drug_pair",
    "sample_id",
    "control_sample_id",
    "pred_response_prob",
    "pred_synergy_prob",
    "pred_task_prob",
    "feature_row_index",
    "pert_index1",
    "pert_index2",
    "pert_time",
    "pert_dose1",
    "pert_dose2",
    "batch",
    "source_prediction_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--training-ready-root", type=Path, default=DEFAULT_TRAINING_READY_ROOT)
    parser.add_argument("--exp07-dir", type=Path, default=DEFAULT_EXP07_DIR)
    parser.add_argument("--exp08-dir", type=Path, default=DEFAULT_EXP08_DIR)
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
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def epoch_ckpts(directory: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    for path in directory.glob("epoch=*-step=*.ckpt"):
        match = CKPT_RE.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return [path for _, path in sorted(checkpoints)]


def checkpoint_meta(path: Path, family: str) -> dict[str, Any]:
    match = CKPT_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported checkpoint filename: {path}")
    epoch = int(match.group(1))
    step = int(match.group(2))
    return {
        "checkpoint_family": family,
        "checkpoint_epoch": epoch,
        "checkpoint_step": step,
        "checkpoint_label": f"{family}_epoch{epoch:03d}",
        "checkpoint_path": repo_relative(path),
    }


def task_type_from_name(task_name: str) -> str:
    if task_name.endswith("_single"):
        return "single"
    if task_name.endswith("_double"):
        return "double"
    return "unknown"


def dataset_from_task(task_name: str, datasets: dict[str, Any]) -> str:
    for dataset in datasets:
        if f"_{dataset}_" in task_name:
            return dataset
    return ""


def build_common_infer_args(args: argparse.Namespace, derived_root: Path) -> list[str]:
    return [
        "--dataset-group",
        "ptv3",
        "--training-ready-root",
        str(args.training_ready_root),
        "--model-type",
        "fast_delta",
        "--split-strategy",
        "test_only",
        "--split-name",
        "test",
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--protein-embedding-path",
        str(derived_root / "protein_embedding_esm.pkl"),
        "--drug-embedding-path",
        str(derived_root / "drug_embedding_morgan_2048.pkl"),
        "--ppi-matrix-path",
        str(derived_root / "ppi_matrix.npy"),
        "--pdi-matrix-path",
        str(derived_root / "pdi_matrix.npy"),
        "--ddi-matrix-path",
        str(derived_root / "ddi_matrix.npy"),
        "--graph-structural-rp",
        "--graph-drug-concat",
        "--cell-llm-mode",
        "frozen",
        "--cell-llm-embedding-path",
        str(derived_root / f"cell_llm_embedding_{args.run_name}_qwen3_4096.npz"),
        "--cell-llm-index-column",
        "cell_llm_index",
        "--cell-type-llm-mode",
        "frozen",
        "--cell-type-llm-embedding-path",
        str(derived_root / "cell_type_llm_embedding_qwen3_4096_v2.npz"),
        "--cell-type-llm-index-column",
        "cell_type_llm_index",
        "--batch-cov-list",
        "machineID_new",
        "Cell_plate",
        "Cell",
        "cell_type",
        "batch",
        "pert_time",
        "pert_dose1",
        "pert_dose2",
        "--allow-checkpoint-config-mismatch",
    ]


def run_infer(
    *,
    args: argparse.Namespace,
    task_name: str,
    task_head: str,
    checkpoint: Path,
    output_dir: Path,
    common_args: list[str],
    extra_args: list[str],
) -> Path:
    prediction_path = output_dir / "predictions.parquet"
    fallback_csv = output_dir / "predictions.csv"
    if not args.force and (prediction_path.exists() or fallback_csv.exists()):
        return prediction_path if prediction_path.exists() else fallback_csv

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "infer.log"
    cmd = [
        args.python_bin,
        "-u",
        "infer.py",
        *common_args,
        "--task-name",
        task_name,
        "--task-head",
        task_head,
        "--checkpoint-path",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-60:])
        raise RuntimeError(f"Inference failed for {task_name} {checkpoint.name}; log tail:\n{tail}")
    if prediction_path.exists():
        return prediction_path
    if fallback_csv.exists():
        return fallback_csv
    raise FileNotFoundError(f"Inference completed but no predictions were written in {output_dir}")


def read_prediction(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_prediction_frame(frame: pd.DataFrame, build_summary: dict[str, Any]) -> pd.DataFrame:
    df = frame.copy()
    dataset_meta = build_summary.get("datasets", {})
    for column in [
        "checkpoint_family",
        "checkpoint_label",
        "checkpoint_epoch",
        "checkpoint_step",
        "checkpoint_path",
        "task_name",
        "feature_row_index",
        "sample_id",
        "control",
        "Cell",
        "cell_type",
        "pert_id1",
        "pert_id2",
        "pert_index1",
        "pert_index2",
        "pred_task_prob",
        "pred_response_prob",
        "pred_synergy_prob",
        "pert_time",
        "pert_dose1",
        "pert_dose2",
        "batch",
        "source_prediction_file",
    ]:
        if column not in df.columns:
            df[column] = ""

    readable = pd.DataFrame(index=df.index)
    readable["checkpoint_family"] = df["checkpoint_family"].astype(str)
    readable["checkpoint_label"] = df["checkpoint_label"].astype(str)
    readable["checkpoint_epoch"] = pd.to_numeric(df["checkpoint_epoch"], errors="coerce").astype("Int64")
    readable["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce").astype("Int64")
    readable["checkpoint_path"] = df["checkpoint_path"].astype(str)
    readable["task_name"] = df["task_name"].astype(str)
    readable["dataset"] = df["batch"].fillna("").astype(str)
    missing_dataset = readable["dataset"].eq("") | readable["dataset"].eq("nan")
    if missing_dataset.any():
        readable.loc[missing_dataset, "dataset"] = readable.loc[missing_dataset, "task_name"].map(
            lambda task: dataset_from_task(task, dataset_meta)
        )
    readable["dataset_display_name"] = readable["dataset"].map(
        lambda key: dataset_meta.get(str(key), {}).get("display_name", str(key))
    )
    readable["treatment_type"] = readable["task_name"].map(task_type_from_name)
    readable["model_source"] = readable.apply(
        lambda row: f"{row['checkpoint_label']}_{'response' if row['treatment_type'] == 'single' else 'synergy'}",
        axis=1,
    )
    readable["score_name"] = readable["treatment_type"].map(
        {"single": "pred_response_prob", "double": "pred_synergy_prob"}
    ).fillna("pred_task_prob")
    for column in ["pred_task_prob", "pred_response_prob", "pred_synergy_prob"]:
        readable[column] = pd.to_numeric(df[column], errors="coerce")
    readable["prediction_score"] = readable["pred_response_prob"].where(
        readable["score_name"].eq("pred_response_prob"),
        readable["pred_synergy_prob"],
    )
    readable["cell_id"] = df["Cell"].fillna("").astype(str)
    readable["cell_type"] = df["cell_type"].fillna("").astype(str)
    readable["drug_1"] = df["pert_id1"].fillna("").astype(str)
    readable["drug_2"] = df["pert_id2"].fillna("").astype(str)
    readable.loc[readable["treatment_type"].eq("single"), "drug_2"] = ""
    readable["drug_pair"] = readable["drug_1"]
    is_double = readable["treatment_type"].eq("double")
    readable.loc[is_double, "drug_pair"] = (
        readable.loc[is_double, "drug_1"] + " + " + readable.loc[is_double, "drug_2"]
    )
    readable["sample_id"] = df["sample_id"].fillna("").astype(str)
    readable["control_sample_id"] = df["control"].fillna("").astype(str)
    readable["feature_row_index"] = pd.to_numeric(df["feature_row_index"], errors="coerce").astype("Int64")
    readable["pert_index1"] = pd.to_numeric(df["pert_index1"], errors="coerce").astype("Int64")
    readable["pert_index2"] = pd.to_numeric(df["pert_index2"], errors="coerce").astype("Int64")
    readable["pert_time"] = df["pert_time"].fillna("").astype(str)
    readable["pert_dose1"] = df["pert_dose1"].fillna("").astype(str)
    readable["pert_dose2"] = df["pert_dose2"].fillna("").astype(str)
    readable["batch"] = df["batch"].fillna("").astype(str)
    readable["source_prediction_file"] = df["source_prediction_file"].fillna("").astype(str)
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
    readable = readable[READABLE_COLUMNS]
    readable["_family_order"] = readable["checkpoint_family"].map({"exp07": 0, "exp08": 1}).fillna(2)
    readable = readable.sort_values(
        [
            "_family_order",
            "checkpoint_epoch",
            "dataset",
            "task_name",
            "rank_in_task_checkpoint",
            "sample_id",
        ],
        kind="mergesort",
    ).drop(columns=["_family_order"])
    return readable.reset_index(drop=True)


def make_task_checkpoint_summary(readable: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["checkpoint_family", "checkpoint_label", "checkpoint_epoch", "checkpoint_step", "task_name"]
    for keys, group in readable.groupby(group_cols, sort=True):
        family, label, epoch, step, task_name = keys
        first = group.iloc[0]
        rows.append(
            {
                "checkpoint_family": family,
                "checkpoint_label": label,
                "checkpoint_epoch": int(epoch),
                "checkpoint_step": int(step),
                "task_name": task_name,
                "dataset": first["dataset"],
                "dataset_display_name": first["dataset_display_name"],
                "treatment_type": first["treatment_type"],
                "score_name": first["score_name"],
                "prediction_rows": int(len(group)),
                "unique_cells": int(group["cell_id"].nunique()),
                "unique_drug_pairs": int(group["drug_pair"].nunique()),
                "score_min": float(group["prediction_score"].min()),
                "score_mean": float(group["prediction_score"].mean()),
                "score_max": float(group["prediction_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def make_checkpoint_summary(readable: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["checkpoint_family", "checkpoint_label", "checkpoint_epoch", "checkpoint_step"]
    for keys, group in readable.groupby(group_cols, sort=True):
        family, label, epoch, step = keys
        rows.append(
            {
                "checkpoint_family": family,
                "checkpoint_label": label,
                "checkpoint_epoch": int(epoch),
                "checkpoint_step": int(step),
                "prediction_rows": int(len(group)),
                "tasks": int(group["task_name"].nunique()),
                "datasets": int(group["dataset"].nunique()),
                "single_predictions": int(group["treatment_type"].eq("single").sum()),
                "double_predictions": int(group["treatment_type"].eq("double").sum()),
                "unique_cells": int(group["cell_id"].nunique()),
                "score_min": float(group["prediction_score"].min()),
                "score_mean": float(group["prediction_score"].mean()),
                "score_max": float(group["prediction_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def make_dataset_checkpoint_summary(readable: pd.DataFrame, build_summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_meta = build_summary.get("datasets", {})
    group_cols = ["checkpoint_family", "checkpoint_label", "checkpoint_epoch", "dataset"]
    for keys, group in readable.groupby(group_cols, sort=True):
        family, label, epoch, dataset = keys
        meta = dataset_meta.get(str(dataset), {})
        counts = meta.get("counts", {})
        matrix_audit = meta.get("matrix_audit", {})
        rows.append(
            {
                "checkpoint_family": family,
                "checkpoint_label": label,
                "checkpoint_epoch": int(epoch),
                "dataset": dataset,
                "dataset_display_name": meta.get("display_name", dataset),
                "prediction_rows": int(len(group)),
                "single_predictions": int(group["treatment_type"].eq("single").sum()),
                "double_predictions": int(group["treatment_type"].eq("double").sum()),
                "unique_cells": int(group["cell_id"].nunique()),
                "unique_drugs": int(
                    pd.concat([group["drug_1"], group["drug_2"]]).replace("", pd.NA).dropna().nunique()
                ),
                "unsupported_multi_drug_rows": counts.get("unsupported_multi_drug", 0),
                "missing_matrix_rows": counts.get("missing_matrix", 0),
                "expression_transform": matrix_audit.get("expression_transform"),
                "score_min": float(group["prediction_score"].min()),
                "score_mean": float(group["prediction_score"].mean()),
                "score_max": float(group["prediction_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def format_float(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def format_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if value is None or pd.isna(value):
                value = ""
            values.append(str(value).replace("\n", " "))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def infer_array_shape(path: Path) -> tuple[int, ...] | None:
    if not path.exists():
        return None
    if path.suffix == ".npy":
        return tuple(np.load(path, mmap_mode="r").shape)
    if path.suffix == ".npz":
        with np.load(path) as data:
            for key in ("embedding_matrix", "embeddings", "arr_0"):
                if key in data.files:
                    return tuple(data[key].shape)
            if data.files:
                return tuple(data[data.files[0]].shape)
    return None


def collect_feature_lines(feature_summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for name, item in feature_summary.get("artifacts", {}).items():
        shape = item.get("shape")
        output = item.get("output", "")
        if not shape and output:
            shape = infer_array_shape(Path(output))
        shape_text = " x ".join(str(dim) for dim in shape) if shape else ""
        line = f"- `{name}`: {shape_text}"
        if output:
            line += f" ({repo_relative(Path(output))})"
        lines.append(line)
    return lines


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
        {"metric": "Prediction rows across all checkpoints", "value": format_int(len(readable))},
        {"metric": "Single-drug prediction rows", "value": format_int(readable["treatment_type"].eq("single").sum())},
        {"metric": "Double-drug prediction rows", "value": format_int(readable["treatment_type"].eq("double").sum())},
        {"metric": "Checkpoints", "value": format_int(readable["checkpoint_label"].nunique())},
        {"metric": "exp07 checkpoints", "value": format_int(readable.loc[readable["checkpoint_family"].eq("exp07"), "checkpoint_label"].nunique())},
        {"metric": "exp08 checkpoints", "value": format_int(readable.loc[readable["checkpoint_family"].eq("exp08"), "checkpoint_label"].nunique())},
        {"metric": "Tasks", "value": format_int(readable["task_name"].nunique())},
        {"metric": "Datasets", "value": format_int(readable["dataset"].nunique())},
        {"metric": "Unique cells", "value": format_int(readable["cell_id"].nunique())},
        {"metric": "Unique drugs", "value": format_int(all_drugs.nunique())},
    ]
    checkpoint_rows = []
    for _, row in checkpoint_summary.sort_values(["checkpoint_family", "checkpoint_epoch"]).iterrows():
        checkpoint_rows.append(
            {
                "label": row["checkpoint_label"],
                "rows": format_int(row["prediction_rows"]),
                "tasks": format_int(row["tasks"]),
                "single": format_int(row["single_predictions"]),
                "double": format_int(row["double_predictions"]),
                "mean": format_float(row["score_mean"]),
                "min": format_float(row["score_min"]),
                "max": format_float(row["score_max"]),
            }
        )
    task_rows = []
    for _, row in task_checkpoint_summary.sort_values(["checkpoint_family", "checkpoint_epoch", "task_name"]).iterrows():
        task_rows.append(
            {
                "label": row["checkpoint_label"],
                "task": row["task_name"],
                "rows": format_int(row["prediction_rows"]),
                "score": row["score_name"],
                "mean": format_float(row["score_mean"]),
                "min": format_float(row["score_min"]),
                "max": format_float(row["score_max"]),
            }
        )
    dataset_rows = []
    for _, row in dataset_checkpoint_summary.sort_values(["checkpoint_family", "checkpoint_epoch", "dataset"]).iterrows():
        dataset_rows.append(
            {
                "label": row["checkpoint_label"],
                "dataset": row["dataset"],
                "rows": format_int(row["prediction_rows"]),
                "single": format_int(row["single_predictions"]),
                "double": format_int(row["double_predictions"]),
                "mean": format_float(row["score_mean"]),
                "transform": row.get("expression_transform", ""),
            }
        )
    top_rows = []
    for _, row in readable.sort_values("prediction_score", ascending=False).head(top_n).iterrows():
        top_rows.append(
            {
                "rank": format_int(len(top_rows) + 1),
                "ckpt": row["checkpoint_label"],
                "dataset": row["dataset"],
                "type": row["treatment_type"],
                "cell": row["cell_id"],
                "drug_pair": row["drug_pair"],
                "score": format_float(row["prediction_score"]),
            }
        )
    lines = [
        "# patientVali260605v3 All-Epoch Checkpoint Inference",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "This directory contains readable CSV exports for patientVali260605v3 inference using every saved exp07/exp08 epoch checkpoint.",
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
        f"- exp07: `{manifest['source_checkpoints']['exp07_dir']}`",
        f"- exp08: `{manifest['source_checkpoints']['exp08_dir']}`",
        "",
        "## Score Columns",
        "",
        "- exp07 single-drug checkpoints use `pred_response_prob` as `prediction_score`.",
        "- exp08 double-drug checkpoints use `pred_synergy_prob` as `prediction_score`.",
        "",
        "## Overall Counts",
        "",
        markdown_table(overall_rows, [("metric", "Metric"), ("value", "Value")]),
        "",
        "## Checkpoint Summary",
        "",
        markdown_table(
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
        markdown_table(
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
        markdown_table(
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
        markdown_table(
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
        *collect_feature_lines(feature_summary),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_task_checkpoint_csvs(readable: pd.DataFrame, output_dir: Path) -> list[str]:
    per_task_dir = output_dir / "per_task_checkpoint_csv"
    per_task_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for (checkpoint_label, task_name), group in readable.groupby(["checkpoint_label", "task_name"], sort=True):
        path = per_task_dir / f"{checkpoint_label}__{task_name}.csv"
        group.to_csv(path, index=False, float_format="%.8g")
        generated.append(repo_relative(path))
    return generated


def main() -> None:
    args = parse_args()
    args.training_ready_root = resolve_path(args.training_ready_root)
    args.exp07_dir = resolve_path(args.exp07_dir)
    args.exp08_dir = resolve_path(args.exp08_dir)
    args.output_root = resolve_path(args.output_root)
    args.readable_output_dir = resolve_path(args.readable_output_dir)
    args.build_summary = resolve_path(args.build_summary)
    args.feature_summary = resolve_path(args.feature_summary)

    build_summary = read_json(args.build_summary)
    feature_summary = read_json(args.feature_summary, required=False)
    derived_root = args.training_ready_root / "ptv3" / "derived"
    task_prefix = f"ptv3_{args.run_name}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.readable_output_dir.mkdir(parents=True, exist_ok=True)

    common_args = build_common_infer_args(args, derived_root)
    inference_records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    generated_at = iso_now()

    work_items: list[tuple[str, Path, str, tuple[str, ...], str, list[str]]] = [
        ("exp07", ckpt, "response", SINGLE_TASKS, "single", [])
        for ckpt in epoch_ckpts(args.exp07_dir)
    ] + [
        (
            "exp08",
            ckpt,
            "synergy",
            DOUBLE_TASKS,
            "double",
            ["--graph-pair-add-scale", "0.5", "--pair-fusion-mode", "dual", "--pair-type-features", "--use-ddi"],
        )
        for ckpt in epoch_ckpts(args.exp08_dir)
    ]
    if not work_items:
        raise RuntimeError("No epoch checkpoints found.")

    for family, checkpoint, task_head, task_suffixes, treatment_type, extra_args in work_items:
        meta = checkpoint_meta(checkpoint, family)
        for task_suffix in task_suffixes:
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
                prediction_path = run_infer(
                    args=args,
                    task_name=task_name,
                    task_head=task_head,
                    checkpoint=checkpoint,
                    output_dir=output_dir,
                    common_args=common_args,
                    extra_args=extra_args,
                )
            frame = read_prediction(prediction_path)
            for key, value in meta.items():
                frame[key] = value
            frame["task_name"] = task_name
            frame["treatment_type"] = treatment_type
            frame["source_prediction_file"] = repo_relative(prediction_path)
            score_column = "pred_response_prob" if treatment_type == "single" else "pred_synergy_prob"
            score = pd.to_numeric(frame[score_column], errors="coerce")
            inference_records.append(
                {
                    **meta,
                    "task_name": task_name,
                    "task_head": task_head,
                    "treatment_type": treatment_type,
                    "prediction_file": repo_relative(prediction_path),
                    "prediction_rows": int(len(frame)),
                    "score_column": score_column,
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

    readable = normalize_prediction_frame(combined_raw, build_summary)
    checkpoint_summary = make_checkpoint_summary(readable)
    task_checkpoint_summary = make_task_checkpoint_summary(readable)
    dataset_checkpoint_summary = make_dataset_checkpoint_summary(readable, build_summary)

    readable_csv = args.readable_output_dir / "combined_predictions_readable.csv"
    checkpoint_summary_csv = args.readable_output_dir / "checkpoint_summary.csv"
    task_checkpoint_summary_csv = args.readable_output_dir / "task_checkpoint_summary.csv"
    dataset_checkpoint_summary_csv = args.readable_output_dir / "dataset_checkpoint_summary.csv"
    readable.to_csv(readable_csv, index=False, float_format="%.8g")
    checkpoint_summary.to_csv(checkpoint_summary_csv, index=False, float_format="%.8g")
    task_checkpoint_summary.to_csv(task_checkpoint_summary_csv, index=False, float_format="%.8g")
    dataset_checkpoint_summary.to_csv(dataset_checkpoint_summary_csv, index=False, float_format="%.8g")
    per_task_files = write_per_task_checkpoint_csvs(readable, args.readable_output_dir)

    manifest = {
        "generated_at": generated_at,
        "run_name": args.run_name,
        "training_ready_root": repo_relative(args.training_ready_root),
        "source_checkpoints": {
            "exp07_dir": repo_relative(args.exp07_dir),
            "exp08_dir": repo_relative(args.exp08_dir),
            "exp07_checkpoints": [repo_relative(path) for path in epoch_ckpts(args.exp07_dir)],
            "exp08_checkpoints": [repo_relative(path) for path in epoch_ckpts(args.exp08_dir)],
        },
        "raw_output_root": repo_relative(args.output_root),
        "readable_output_dir": repo_relative(args.readable_output_dir),
        "combined_raw_parquet": repo_relative(combined_raw_parquet),
        "combined_raw_csv": repo_relative(combined_raw_csv),
        "combined_readable_csv": repo_relative(readable_csv),
        "checkpoint_summary_csv": repo_relative(checkpoint_summary_csv),
        "task_checkpoint_summary_csv": repo_relative(task_checkpoint_summary_csv),
        "dataset_checkpoint_summary_csv": repo_relative(dataset_checkpoint_summary_csv),
        "per_task_checkpoint_csv": per_task_files,
        "inference_records": inference_records,
        "total_prediction_rows": int(len(readable)),
        "single_prediction_rows": int(readable["treatment_type"].eq("single").sum()),
        "double_prediction_rows": int(readable["treatment_type"].eq("double").sum()),
    }
    write_json(args.output_root / "inference_summary.json", manifest)
    write_json(args.readable_output_dir / "manifest.json", manifest)
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
    print(f"[done] readable outputs: {repo_relative(args.readable_output_dir)}")
    print(f"[done] rows={len(readable)} checkpoints={readable['checkpoint_label'].nunique()}")


if __name__ == "__main__":
    main()
