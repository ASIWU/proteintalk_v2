#!/usr/bin/env python3
"""Prepare readable CSV and README artifacts for patient validation inference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "patientVali260605v3"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "0612v3"
DEFAULT_COMBINED_PARQUET = REPO_ROOT / "outputs" / "20260612_patientVali260605v3_combined_predictions.parquet"
DEFAULT_COMBINED_CSV = REPO_ROOT / "outputs" / "20260612_patientVali260605v3_combined_predictions.csv"
DEFAULT_INFERENCE_SUMMARY = REPO_ROOT / "outputs" / "20260612_patientVali260605v3_inference_summary.json"
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


READABLE_COLUMNS = [
    "dataset",
    "dataset_display_name",
    "task_name",
    "treatment_type",
    "model_source",
    "score_name",
    "prediction_score",
    "rank_in_task",
    "rank_in_dataset",
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
    "inference_group",
    "inference_head",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--combined-parquet", type=Path, default=DEFAULT_COMBINED_PARQUET)
    parser.add_argument("--combined-csv", type=Path, default=DEFAULT_COMBINED_CSV)
    parser.add_argument("--inference-summary", type=Path, default=DEFAULT_INFERENCE_SUMMARY)
    parser.add_argument("--build-summary", type=Path, default=DEFAULT_BUILD_SUMMARY)
    parser.add_argument("--feature-summary", type=Path, default=DEFAULT_FEATURE_SUMMARY)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_combined_predictions(parquet_path: Path, csv_path: Path) -> tuple[pd.DataFrame, Path]:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), parquet_path
    if csv_path.exists():
        return pd.read_csv(csv_path, keep_default_na=False), csv_path
    raise FileNotFoundError(f"Missing combined prediction file: {parquet_path} or {csv_path}")


def task_type_from_name(task_name: Any) -> str:
    text = str(task_name)
    if text.endswith("_single"):
        return "single"
    if text.endswith("_double"):
        return "double"
    return "unknown"


def dataset_from_task(task_name: str, datasets: dict[str, Any]) -> str:
    for dataset in datasets:
        if f"_{dataset}_" in task_name:
            return dataset
    return ""


def normalize_prediction_frame(
    frame: pd.DataFrame,
    *,
    inference_summary: dict[str, Any],
    build_summary: dict[str, Any],
) -> pd.DataFrame:
    df = frame.copy()
    task_meta = {str(item.get("task_name")): item for item in inference_summary.get("tasks", [])}
    dataset_meta = build_summary.get("datasets", {})

    for column in [
        "inference_group",
        "task_name",
        "inference_head",
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
    ]:
        if column not in df.columns:
            df[column] = ""

    readable = pd.DataFrame(index=df.index)
    readable["task_name"] = df["task_name"].astype(str)
    readable["dataset"] = df["batch"].astype(str)
    missing_dataset = readable["dataset"].eq("") | readable["dataset"].eq("nan")
    if missing_dataset.any():
        readable.loc[missing_dataset, "dataset"] = readable.loc[missing_dataset, "task_name"].map(
            lambda task: dataset_from_task(task, dataset_meta)
        )
    readable["dataset_display_name"] = readable["dataset"].map(
        lambda key: dataset_meta.get(str(key), {}).get("display_name", str(key))
    )
    readable["treatment_type"] = readable["task_name"].map(task_type_from_name)
    readable["model_source"] = readable["treatment_type"].map(
        {
            "single": "exp07_single_response",
            "double": "exp08_double_synergy",
        }
    ).fillna("unknown")
    readable["score_name"] = readable["task_name"].map(
        lambda task: task_meta.get(str(task), {}).get(
            "score_column",
            "pred_response_prob" if task_type_from_name(task) == "single" else "pred_synergy_prob",
        )
    )

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
    readable["inference_group"] = df["inference_group"].fillna("").astype(str)
    readable["inference_head"] = df["inference_head"].fillna("").astype(str)

    readable["rank_in_task"] = (
        readable.groupby("task_name")["prediction_score"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    readable["rank_in_dataset"] = (
        readable.groupby("dataset")["prediction_score"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )

    readable = readable[READABLE_COLUMNS]
    treatment_order = {"single": 0, "double": 1, "unknown": 2}
    readable["_treatment_order"] = readable["treatment_type"].map(treatment_order).fillna(2)
    readable = readable.sort_values(
        ["dataset", "_treatment_order", "task_name", "rank_in_task", "sample_id"],
        kind="mergesort",
    ).drop(columns=["_treatment_order"])
    return readable.reset_index(drop=True)


def make_task_summary(readable: pd.DataFrame, inference_summary: dict[str, Any]) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    task_meta = {str(item.get("task_name")): item for item in inference_summary.get("tasks", [])}
    for task_name, group in readable.groupby("task_name", sort=True):
        first = group.iloc[0]
        meta = task_meta.get(str(task_name), {})
        summary_rows.append(
            {
                "task_name": task_name,
                "dataset": first["dataset"],
                "dataset_display_name": first["dataset_display_name"],
                "treatment_type": first["treatment_type"],
                "model_source": first["model_source"],
                "score_name": first["score_name"],
                "prediction_rows": int(len(group)),
                "unique_cells": int(group["cell_id"].nunique()),
                "unique_drug_pairs": int(group["drug_pair"].nunique()),
                "score_min": float(group["prediction_score"].min()),
                "score_mean": float(group["prediction_score"].mean()),
                "score_max": float(group["prediction_score"].max()),
                "source_score_min": meta.get("score_min"),
                "source_score_mean": meta.get("score_mean"),
                "source_score_max": meta.get("score_max"),
            }
        )
    return pd.DataFrame(summary_rows)


def make_dataset_summary(readable: pd.DataFrame, build_summary: dict[str, Any]) -> pd.DataFrame:
    dataset_meta = build_summary.get("datasets", {})
    rows: list[dict[str, Any]] = []
    for dataset, meta in dataset_meta.items():
        group = readable[readable["dataset"].eq(dataset)]
        counts = meta.get("counts", {})
        matrix_audit = meta.get("matrix_audit", {})
        rows.append(
            {
                "dataset": dataset,
                "dataset_display_name": meta.get("display_name", dataset),
                "raw_info_rows": meta.get("raw_info_rows"),
                "prediction_rows": int(len(group)),
                "single_predictions": int(group["treatment_type"].eq("single").sum()),
                "double_predictions": int(group["treatment_type"].eq("double").sum()),
                "unique_cells": int(group["cell_id"].nunique()) if not group.empty else 0,
                "unique_drugs": int(
                    pd.concat([group["drug_1"], group["drug_2"]])
                    .replace("", pd.NA)
                    .dropna()
                    .nunique()
                )
                if not group.empty
                else 0,
                "unsupported_multi_drug_rows": counts.get("unsupported_multi_drug", 0),
                "missing_matrix_rows": counts.get("missing_matrix", 0),
                "standard_compatible": meta.get("standard_compatible"),
                "expression_scale_input": matrix_audit.get("expression_scale_input"),
                "expression_transform": matrix_audit.get("expression_transform"),
                "kept_protein_columns": matrix_audit.get("kept_protein_columns"),
                "score_min": float(group["prediction_score"].min()) if not group.empty else None,
                "score_mean": float(group["prediction_score"].mean()) if not group.empty else None,
                "score_max": float(group["prediction_score"].max()) if not group.empty else None,
            }
        )
    extra_datasets = sorted(set(readable["dataset"]) - set(dataset_meta))
    for dataset in extra_datasets:
        group = readable[readable["dataset"].eq(dataset)]
        rows.append(
            {
                "dataset": dataset,
                "dataset_display_name": dataset,
                "raw_info_rows": None,
                "prediction_rows": int(len(group)),
                "single_predictions": int(group["treatment_type"].eq("single").sum()),
                "double_predictions": int(group["treatment_type"].eq("double").sum()),
                "unique_cells": int(group["cell_id"].nunique()),
                "unique_drugs": int(
                    pd.concat([group["drug_1"], group["drug_2"]])
                    .replace("", pd.NA)
                    .dropna()
                    .nunique()
                ),
                "unsupported_multi_drug_rows": None,
                "missing_matrix_rows": None,
                "standard_compatible": None,
                "expression_scale_input": None,
                "expression_transform": None,
                "kept_protein_columns": None,
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
            preferred_keys = ("embedding_matrix", "embeddings", "arr_0")
            for key in preferred_keys:
                if key in data.files:
                    return tuple(data[key].shape)
            if data.files:
                return tuple(data[data.files[0]].shape)
    return None


def collect_feature_lines(feature_summary: dict[str, Any]) -> list[str]:
    artifacts = feature_summary.get("artifacts", {})
    lines = []
    for name, item in artifacts.items():
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
    run_name: str,
    generated_at: str,
    source_combined: Path,
    inference_summary_path: Path,
    build_summary_path: Path,
    feature_summary_path: Path,
    readable: pd.DataFrame,
    task_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    feature_summary: dict[str, Any],
    top_n: int,
) -> None:
    all_drugs = pd.concat([readable["drug_1"], readable["drug_2"]]).replace("", pd.NA).dropna()
    total_unsupported = dataset_summary["unsupported_multi_drug_rows"].fillna(0).sum()
    total_missing_matrix = dataset_summary["missing_matrix_rows"].fillna(0).sum()

    overall_rows = [
        {"metric": "Prediction rows", "value": format_int(len(readable))},
        {"metric": "Datasets", "value": format_int(readable["dataset"].nunique())},
        {"metric": "Tasks", "value": format_int(readable["task_name"].nunique())},
        {"metric": "Single-drug rows", "value": format_int(readable["treatment_type"].eq("single").sum())},
        {"metric": "Double-drug rows", "value": format_int(readable["treatment_type"].eq("double").sum())},
        {"metric": "Unique cells", "value": format_int(readable["cell_id"].nunique())},
        {"metric": "Unique drugs", "value": format_int(all_drugs.nunique())},
        {"metric": "Unsupported multi-drug rows skipped", "value": format_int(total_unsupported)},
        {"metric": "Missing-matrix rows skipped", "value": format_int(total_missing_matrix)},
    ]

    dataset_rows = []
    for _, row in dataset_summary.iterrows():
        dataset_rows.append(
            {
                "dataset": row["dataset"],
                "display": row["dataset_display_name"],
                "pred_rows": format_int(row["prediction_rows"]),
                "single": format_int(row["single_predictions"]),
                "double": format_int(row["double_predictions"]),
                "cells": format_int(row["unique_cells"]),
                "drugs": format_int(row["unique_drugs"]),
                "skipped_multi": format_int(row["unsupported_multi_drug_rows"]),
                "transform": row.get("expression_transform", ""),
                "score_mean": format_float(row.get("score_mean")),
            }
        )

    task_rows = []
    for _, row in task_summary.sort_values(["dataset", "treatment_type", "task_name"]).iterrows():
        task_rows.append(
            {
                "task": row["task_name"],
                "type": row["treatment_type"],
                "rows": format_int(row["prediction_rows"]),
                "score": row["score_name"],
                "min": format_float(row["score_min"]),
                "mean": format_float(row["score_mean"]),
                "max": format_float(row["score_max"]),
            }
        )

    top_rows = []
    top_predictions = readable.sort_values("prediction_score", ascending=False).head(top_n)
    for _, row in top_predictions.iterrows():
        top_rows.append(
            {
                "rank": format_int(len(top_rows) + 1),
                "dataset": row["dataset"],
                "type": row["treatment_type"],
                "cell": row["cell_id"],
                "drug_pair": row["drug_pair"],
                "score": format_float(row["prediction_score"]),
            }
        )

    lines = [
        f"# {run_name} 0612v3 Outputs",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "This directory contains readable CSV exports and a short numeric summary for the June 12 v3 patient-validation inference run.",
        "",
        "## Files",
        "",
        "- `combined_predictions_readable.csv`: row-level prediction table with readable score, dataset, cell, and drug-pair columns.",
        "- `per_task_csv/*.csv`: one readable CSV per inference task, converted from each task's parquet prediction file.",
        "- `task_summary.csv`: one row per inference task.",
        "- `dataset_summary.csv`: one row per patient-validation dataset.",
        "- `manifest.json`: machine-readable record of inputs and generated files.",
        "",
        "## Source Inputs",
        "",
        f"- Combined predictions: `{repo_relative(source_combined)}`",
        f"- Inference summary: `{repo_relative(inference_summary_path)}`",
        f"- Data-build summary: `{repo_relative(build_summary_path)}`",
        f"- Feature summary: `{repo_relative(feature_summary_path)}`",
        "",
        "## Score Columns",
        "",
        "- Single-drug rows use `pred_response_prob` as `prediction_score` from exp_07.",
        "- Double-drug rows use `pred_synergy_prob` as `prediction_score` from exp_08.",
        "- Ground-truth labels are not available for this patient-validation inference run.",
        "",
        "## Overall Counts",
        "",
        markdown_table(overall_rows, [("metric", "Metric"), ("value", "Value")]),
        "",
        "## Dataset Summary",
        "",
        markdown_table(
            dataset_rows,
            [
                ("dataset", "Dataset"),
                ("display", "Display name"),
                ("pred_rows", "Pred rows"),
                ("single", "Single"),
                ("double", "Double"),
                ("cells", "Cells"),
                ("drugs", "Drugs"),
                ("skipped_multi", "Skipped multi"),
                ("transform", "Expression transform"),
                ("score_mean", "Score mean"),
            ],
        ),
        "",
        "## Task Summary",
        "",
        markdown_table(
            task_rows,
            [
                ("task", "Task"),
                ("type", "Type"),
                ("rows", "Rows"),
                ("score", "Score"),
                ("min", "Min"),
                ("mean", "Mean"),
                ("max", "Max"),
            ],
        ),
        "",
        f"## Top {top_n} Predictions",
        "",
        markdown_table(
            top_rows,
            [
                ("rank", "Rank"),
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


def write_per_task_csvs(
    *,
    inference_summary: dict[str, Any],
    build_summary: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    generated: list[str] = []
    per_task_dir = output_dir / "per_task_csv"
    per_task_dir.mkdir(parents=True, exist_ok=True)

    task_meta = {str(item.get("task_name")): item for item in inference_summary.get("tasks", [])}
    for prediction_file in inference_summary.get("prediction_files", []):
        prediction_path = Path(prediction_file)
        if not prediction_path.is_absolute():
            prediction_path = REPO_ROOT / prediction_path
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        task_name = prediction_path.parent.name
        group_name = prediction_path.parents[1].name
        frame = pd.read_parquet(prediction_path)
        frame.insert(0, "task_name", task_name)
        frame.insert(0, "inference_group", group_name)
        treatment_type = task_type_from_name(task_name)
        frame.insert(
            2,
            "inference_head",
            "single_response_exp07" if treatment_type == "single" else "double_synergy_exp08",
        )
        if task_name in task_meta:
            frame["score_column"] = task_meta[task_name].get("score_column", "")

        readable = normalize_prediction_frame(
            frame,
            inference_summary=inference_summary,
            build_summary=build_summary,
        )
        task_output = per_task_dir / f"{task_name}.csv"
        readable.to_csv(task_output, index=False, float_format="%.8g")
        generated.append(repo_relative(task_output))
    return generated


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_summary = read_json(args.inference_summary)
    build_summary = read_json(args.build_summary)
    feature_summary = read_json(args.feature_summary, required=False)

    combined, source_combined = load_combined_predictions(args.combined_parquet, args.combined_csv)
    readable = normalize_prediction_frame(
        combined,
        inference_summary=inference_summary,
        build_summary=build_summary,
    )
    task_summary = make_task_summary(readable, inference_summary)
    dataset_summary = make_dataset_summary(readable, build_summary)

    combined_output = output_dir / "combined_predictions_readable.csv"
    task_summary_output = output_dir / "task_summary.csv"
    dataset_summary_output = output_dir / "dataset_summary.csv"
    readme_output = output_dir / "README.md"
    manifest_output = output_dir / "manifest.json"

    readable.to_csv(combined_output, index=False, float_format="%.8g")
    task_summary.to_csv(task_summary_output, index=False, float_format="%.8g")
    dataset_summary.to_csv(dataset_summary_output, index=False, float_format="%.8g")
    per_task_outputs = write_per_task_csvs(
        inference_summary=inference_summary,
        build_summary=build_summary,
        output_dir=output_dir,
    )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    write_readme(
        readme_output,
        run_name=args.run_name,
        generated_at=generated_at,
        source_combined=source_combined,
        inference_summary_path=args.inference_summary,
        build_summary_path=args.build_summary,
        feature_summary_path=args.feature_summary,
        readable=readable,
        task_summary=task_summary,
        dataset_summary=dataset_summary,
        feature_summary=feature_summary,
        top_n=args.top_n,
    )

    manifest = {
        "generated_at": generated_at,
        "run_name": args.run_name,
        "source_inputs": {
            "combined_predictions": repo_relative(source_combined),
            "inference_summary": repo_relative(args.inference_summary),
            "build_summary": repo_relative(args.build_summary),
            "feature_summary": repo_relative(args.feature_summary),
        },
        "outputs": {
            "combined_predictions_readable_csv": repo_relative(combined_output),
            "task_summary_csv": repo_relative(task_summary_output),
            "dataset_summary_csv": repo_relative(dataset_summary_output),
            "readme": repo_relative(readme_output),
            "per_task_csv": per_task_outputs,
        },
        "row_counts": {
            "combined_predictions": int(len(readable)),
            "tasks": int(readable["task_name"].nunique()),
            "datasets": int(readable["dataset"].nunique()),
            "single_predictions": int(readable["treatment_type"].eq("single").sum()),
            "double_predictions": int(readable["treatment_type"].eq("double").sum()),
        },
    }
    write_json(manifest_output, manifest)

    print(f"Wrote {repo_relative(combined_output)}")
    print(f"Wrote {repo_relative(task_summary_output)}")
    print(f"Wrote {repo_relative(dataset_summary_output)}")
    print(f"Wrote {repo_relative(readme_output)}")
    print(f"Wrote {len(per_task_outputs)} per-task CSV files under {repo_relative(output_dir / 'per_task_csv')}")


if __name__ == "__main__":
    main()
