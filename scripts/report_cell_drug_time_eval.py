#!/usr/bin/env python3
"""Report original and cell-drug-dose time-collapsed metrics for PTV3 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
METHODS = ("original", "cell-drug-dose-bylasttime", "cell-drug-dose-avgtime")
METRIC_COLUMNS = ("auroc", "auprc", "auprc_baseline", "nauprc")
POSITIVE_LABELS = {"sensitive", "syn", "synergy", "synergistic", "y", "yes", "1", "true"}
NEGATIVE_LABELS = {
    "non-responsive",
    "nonresponsive",
    "non-sensitive",
    "non-syn",
    "nonsyn",
    "non-synergy",
    "non_synergy",
    "n",
    "no",
    "0",
    "false",
}


@dataclass(frozen=True)
class FoldExperiment:
    exp: str
    label: str
    checkpoint_suffix: str
    double_drug: bool = False


@dataclass(frozen=True)
class ExtraExperiment:
    exp: str
    label: str
    output_suffix: str
    double_drug: bool


FOLD_EXPERIMENTS = (
    FoldExperiment("exp01", "single unseen drug", "exp01_single_pert_stratified_5fold_single_pert_stratified_fold"),
    FoldExperiment("exp02", "single unseen cell type", "exp02_single_cell_type_5fold_single_cell_type_fold"),
    FoldExperiment("exp03", "single unseen cell", "exp03_single_cell_5fold_single_cell_fold"),
    FoldExperiment("exp04", "single w/o MSE", "exp04_single_no_mse_5fold_single_no_mse_fold"),
    FoldExperiment("exp05", "single w/o graph", "exp05_single_no_graph_5fold_single_no_graph_fold"),
    FoldExperiment("exp06", "double unseen drug pair", "exp06_double_pert_pair_5fold_double_pert_pair_fold", True),
)
EXTRA_EXPERIMENTS = (
    ExtraExperiment("exp07", "extra single", "exp07_extra_single_all_train_infer_all_single_for_extra", False),
    ExtraExperiment("exp08", "extra double", "exp08_extra_double_all_train_infer_all_single_double_for_extra", True),
)
EXTRA_DOUBLE_TEST_LABEL_GROUPS = (
    "unseenCell_seenDrugCombo",
    "unseenCell_unseenDrugCombo",
    "combined",
)


INFER_VALUE_ARGS = {
    "hidden_dim": "--hidden-dim",
    "expression_latent_dim": "--expression-latent-dim",
    "covariate_embedding_dim": "--covariate-embedding-dim",
    "dropout": "--dropout",
    "control_layers": "--control-layers",
    "fusion_layers": "--fusion-layers",
    "target_layers": "--target-layers",
    "target_protein_max_length": "--target-protein-max-length",
    "graph_feature_mode": "--graph-feature-mode",
    "graph_feature_dim": "--graph-feature-dim",
    "graph_feature_seed": "--graph-feature-seed",
    "graph_cache_dir": "--graph-cache-dir",
    "graph_layers": "--graph-layers",
    "graph_init_scale": "--graph-init-scale",
    "graph_pair_add_scale": "--graph-pair-add-scale",
    "graph_logit_scale": "--graph-logit-scale",
    "graph_jump_fusion": "--graph-jump-fusion",
    "graph_jump_gate": "--graph-jump-gate",
    "graph_jump_temperature": "--graph-jump-temperature",
    "pair_fusion_mode": "--pair-fusion-mode",
    "cell_pair_film_scale": "--cell-pair-film-scale",
    "target_expression_mode": "--target-expression-mode",
    "target_expression_dim": "--target-expression-dim",
    "target_expression_topk": "--target-expression-topk",
    "target_expression_ppi_topk": "--target-expression-ppi-topk",
    "target_expression_ppi_alpha": "--target-expression-ppi-alpha",
    "target_expression_ppi_norm": "--target-expression-ppi-norm",
    "target_expression_degree_penalty": "--target-expression-degree-penalty",
    "target_expression_init_scale": "--target-expression-init-scale",
    "target_expression_seed": "--target-expression-seed",
    "target_expression_fusion_mode": "--target-expression-fusion-mode",
    "target_expression_cell_gate_mode": "--target-expression-cell-gate-mode",
    "target_expression_cell_gate_scale": "--target-expression-cell-gate-scale",
    "target_expression_cell_gate_temperature": "--target-expression-cell-gate-temperature",
    "target_expression_chunk_size": "--target-expression-chunk-size",
    "target_expression_cache_dir": "--target-expression-cache-dir",
    "protein_concat_mode": "--protein-concat-mode",
    "protein_concat_dim": "--protein-concat-dim",
    "protein_concat_topk": "--protein-concat-topk",
    "protein_concat_init_scale": "--protein-concat-init-scale",
    "protein_concat_seed": "--protein-concat-seed",
    "protein_concat_score_mode": "--protein-concat-score-mode",
    "protein_concat_expr_scale": "--protein-concat-expr-scale",
    "control_logit_scale": "--control-logit-scale",
    "pair_logit_scale": "--pair-logit-scale",
    "target_logit_scale": "--target-logit-scale",
    "covariate_logit_scale": "--covariate-logit-scale",
    "response_base_logit_scale": "--response-base-logit-scale",
    "response_delta_mode": "--response-delta-mode",
    "response_delta_dim": "--response-delta-dim",
    "response_delta_seed": "--response-delta-seed",
    "delta_logit_scale": "--delta-logit-scale",
    "control_expression_dropout": "--control-expression-dropout",
    "init_delta_scale": "--init-delta-scale",
    "cell_type_llm_mode": "--cell-type-llm-mode",
    "cell_type_llm_embedding_path": "--cell-type-llm-embedding-path",
}
INFER_BOOL_ARGS = {
    "graph_structural_rp": "--graph-structural-rp",
    "graph_multihop": "--graph-multihop",
    "graph_drug_concat": "--graph-drug-concat",
    "pair_type_features": "--pair-type-features",
    "response_delta_detach": "--response-delta-detach",
    "delta_logit_learnable": "--delta-logit-learnable",
    "use_ddi": "--use-ddi",
    "zero_init_delta_head": "--zero-init-delta-head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Base experiment prefix, without _expNN suffix.")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--training-ready-root", type=Path, default=DEFAULT_TRAINING_READY_ROOT)
    parser.add_argument("--fold-prediction-root", type=Path, default=None)
    parser.add_argument("--materialize-fold-predictions", action="store_true")
    parser.add_argument("--force-infer", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--infer-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("markdown", "csv", "none"), default="markdown")
    parser.add_argument("--precision", type=int, default=6)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def encode_binary_label(value: Any) -> float | None:
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


def load_predictions(prediction_dir: Path) -> tuple[pd.DataFrame, Path]:
    parquet_path = prediction_dir / "predictions.parquet"
    csv_path = prediction_dir / "predictions.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), parquet_path
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False), csv_path
    raise FileNotFoundError(f"missing predictions.parquet/csv under {prediction_dir}")


def feature_table_path(task_dir: Path) -> Path:
    parquet_path = task_dir / "feature_table.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = task_dir / "feature_table.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"missing feature_table.parquet/csv under {task_dir}")


def load_feature_rows(task_dir: Path, row_indices: np.ndarray) -> pd.DataFrame:
    table_path = feature_table_path(task_dir)
    if table_path.suffix == ".parquet":
        table = pd.read_parquet(table_path)
    else:
        table = pd.read_csv(table_path, low_memory=False)
    if np.any(row_indices < 0) or np.any(row_indices >= len(table)):
        raise ValueError(f"feature_row_index outside table range for {table_path}")
    return table.iloc[row_indices].reset_index(drop=True)


def enrich_with_feature_metadata(predictions: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    if "feature_row_index" not in predictions.columns:
        return predictions.copy()
    task_dir_value = manifest.get("task_dir")
    if not task_dir_value:
        return predictions.copy()
    row_indices = pd.to_numeric(predictions["feature_row_index"], errors="raise").astype(int).to_numpy()
    feature_rows = load_feature_rows(Path(task_dir_value), row_indices)
    result = predictions.copy()
    metadata_columns = (
        "Cell",
        "cell_type",
        "pert_id1",
        "pert_id2",
        "pert_time",
        "pert_time_norm",
        "pert_dose1",
        "pert_dose2",
        "pert_dose1_norm",
        "pert_dose2_norm",
        "batch",
        "test",
        "test_label",
    )
    for column in metadata_columns:
        if column not in feature_rows.columns:
            continue
        values = feature_rows[column].reset_index(drop=True)
        if column not in result.columns:
            result[column] = values
        else:
            existing = result[column].reset_index(drop=True)
            result[column] = existing.where(existing.notna(), values)
    return result


def metric_values(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    keep = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[keep]
    y_prob = y_prob[keep]
    valid_count = int(y_true.size)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    baseline = (positive_count / valid_count) if valid_count else float("nan")
    auroc = float("nan")
    auprc = float("nan")
    if positive_count > 0 and negative_count > 0:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    nauprc = auprc / baseline if baseline and math.isfinite(baseline) and math.isfinite(auprc) else float("nan")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "auprc_baseline": baseline,
        "nauprc": nauprc,
        "valid_count": valid_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def choose_label_column(frame: pd.DataFrame, double_drug: bool) -> str:
    candidates = ["task_label"]
    candidates.append("synergy_label" if double_drug else "response_label")
    candidates.append("synergy" if double_drug else "PRISM1st_label_total")
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(f"none of the label columns exist: {candidates}")


def compact_numeric_token(value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number == 0:
        number = 0.0
    return f"{number:.12g}"


def canonical_token(value: Any, *, missing: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return missing
    number_token = compact_numeric_token(value)
    if number_token is not None:
        return number_token
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return missing
    return text


def dose_token_series(frame: pd.DataFrame, field: str) -> pd.Series:
    norm_column = f"{field}_norm"
    raw_values = frame.get(field, pd.Series([np.nan] * len(frame)))
    raw_tokens = raw_values.map(lambda value: canonical_token(value, missing="no"))
    if norm_column not in frame.columns:
        return raw_tokens.astype("string").fillna("no")

    norm_tokens = frame[norm_column].map(lambda value: canonical_token(value, missing=""))
    norm_tokens = norm_tokens.astype("string").fillna("")
    raw_tokens = raw_tokens.astype("string").fillna("no")
    return norm_tokens.where(norm_tokens.ne(""), raw_tokens).astype("string").fillna("no")


def add_eval_columns(frame: pd.DataFrame, *, double_drug: bool) -> pd.DataFrame:
    if "pred_task_prob" not in frame.columns:
        raise KeyError("prediction table missing pred_task_prob")
    label_column = choose_label_column(frame, double_drug)
    result = frame.copy()
    result["_prob"] = pd.to_numeric(result["pred_task_prob"], errors="coerce")
    result["_label"] = [encode_binary_label(value) for value in result[label_column].tolist()]
    result["_time"] = pd.to_numeric(result.get("pert_time", pd.Series([np.nan] * len(result))), errors="coerce")
    result["_cell"] = result.get("Cell", pd.Series([""] * len(result))).map(canonical_token).astype("string").fillna("")
    result["_drug1"] = result.get("pert_id1", pd.Series([""] * len(result))).map(canonical_token).astype("string").fillna("")
    result["_drug2"] = result.get("pert_id2", pd.Series([""] * len(result))).map(canonical_token).astype("string").fillna("")
    result["_dose1"] = dose_token_series(result, "pert_dose1")
    result["_dose2"] = dose_token_series(result, "pert_dose2")
    if double_drug:
        drug_a: list[str] = []
        dose_a: list[str] = []
        drug_b: list[str] = []
        dose_b: list[str] = []
        for first_drug, first_dose, second_drug, second_dose in zip(
            result["_drug1"].astype(str).tolist(),
            result["_dose1"].astype(str).tolist(),
            result["_drug2"].astype(str).tolist(),
            result["_dose2"].astype(str).tolist(),
            strict=True,
        ):
            first_pair = (first_drug, first_dose)
            second_pair = (second_drug, second_dose)
            left, right = sorted((first_pair, second_pair))
            drug_a.append(left[0])
            dose_a.append(left[1])
            drug_b.append(right[0])
            dose_b.append(right[1])
        result["_drug_a"] = drug_a
        result["_dose_a"] = dose_a
        result["_drug_b"] = drug_b
        result["_dose_b"] = dose_b
    else:
        result["_drug_a"] = result["_drug1"].astype(str)
        result["_dose_a"] = result["_dose1"].astype(str)
        result["_drug_b"] = ""
        result["_dose_b"] = ""
    return result


def original_frame(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["_label"].notna() & np.isfinite(frame["_prob"].astype(float))
    return frame.loc[valid, ["_label", "_prob"]].copy()


def collapsed_frame(frame: pd.DataFrame, *, method: str) -> tuple[pd.DataFrame, int, int]:
    if method not in {"cell-drug-dose-bylasttime", "cell-drug-dose-avgtime"}:
        raise ValueError(f"unexpected collapsed method: {method}")
    group_cols = ["_cell", "_drug_a", "_dose_a", "_drug_b", "_dose_b"]
    records: list[dict[str, float]] = []
    conflict_count = 0
    missing_time_count = 0
    for _, group in frame.groupby(group_cols, dropna=False, sort=False):
        finite_prob = group[np.isfinite(group["_prob"].astype(float)) & group["_label"].notna()].copy()
        if finite_prob.empty:
            continue
        labels = sorted({float(value) for value in finite_prob["_label"].dropna().tolist()})
        if len(labels) != 1:
            conflict_count += 1
            continue
        selected = finite_prob
        if method == "cell-drug-dose-bylasttime":
            finite_time = selected[np.isfinite(selected["_time"].astype(float))]
            if finite_time.empty:
                missing_time_count += 1
            else:
                max_time = float(finite_time["_time"].max())
                selected = selected[selected["_time"].eq(max_time)]
        records.append({"_label": labels[0], "_prob": float(selected["_prob"].mean())})
    return pd.DataFrame.from_records(records), conflict_count, missing_time_count


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    exp: str,
    task: str,
    split: str,
    method: str,
    double_drug: bool,
    source: Path,
) -> dict[str, Any]:
    prepared = add_eval_columns(frame, double_drug=double_drug)
    conflict_count = 0
    missing_time_count = 0
    if method == "original":
        eval_frame = original_frame(prepared)
    else:
        eval_frame, conflict_count, missing_time_count = collapsed_frame(prepared, method=method)
    metrics = metric_values(eval_frame["_label"].to_numpy(), eval_frame["_prob"].to_numpy())
    return {
        "exp": exp,
        "task": task,
        "split": split,
        "method": method,
        **metrics,
        "cell_drug_label_conflicts": conflict_count,
        "missing_time_groups": missing_time_count,
        "source": str(source),
    }


def aggregate_records(records: list[dict[str, Any]], *, split: str, task: str) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty records")
    aggregate: dict[str, Any] = {
        "exp": records[0]["exp"],
        "task": task,
        "split": split,
        "method": records[0]["method"],
        "valid_count": sum(int(row.get("valid_count") or 0) for row in records),
        "positive_count": sum(int(row.get("positive_count") or 0) for row in records),
        "negative_count": sum(int(row.get("negative_count") or 0) for row in records),
        "cell_drug_label_conflicts": sum(int(row.get("cell_drug_label_conflicts") or 0) for row in records),
        "missing_time_groups": sum(int(row.get("missing_time_groups") or 0) for row in records),
        "source": "mean_of_records",
    }
    for column in METRIC_COLUMNS:
        values = [value for row in records if (value := finite_float(row.get(column))) is not None]
        aggregate[column] = mean(values) if values else float("nan")
    return aggregate


def build_infer_command(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    checkpoint_path: str,
    output_dir: Path,
) -> list[str]:
    command = [
        args.python_bin,
        "-u",
        str(REPO_ROOT / "infer.py"),
        "--training-ready-root",
        str(args.training_ready_root),
        "--dataset-group",
        str(manifest["dataset_group"]),
        "--model-type",
        str(manifest["model_type"]),
        "--task-name",
        str(manifest["task_name"]),
        "--split-strategy",
        str(manifest["split_strategy"]),
        "--split-name",
        "test",
        "--task-head",
        str(manifest["task_head"]),
        "--checkpoint-path",
        checkpoint_path,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(args.infer_batch_size),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]
    for key, flag in INFER_VALUE_ARGS.items():
        value = manifest.get(key)
        if value is not None and value != "":
            command.extend([flag, str(value)])
    for key, flag in INFER_BOOL_ARGS.items():
        if bool(manifest.get(key)):
            command.append(flag)
    if bool(manifest.get("use_dose_covariate")):
        command.append("--use-dose-covariate")
        fields = manifest.get("dose_covariate_fields") or []
        if fields:
            command.append("--dose-covariate-fields")
            command.extend(str(field) for field in fields)
    batch_cov_list = manifest.get("batch_cov_list") or []
    command.append("--batch-cov-list")
    command.extend(str(field) for field in batch_cov_list)
    if manifest.get("residual_expression") is False:
        command.append("--absolute-expression-head")
    return command


def ensure_fold_predictions(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    prediction_dir: Path,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    manifest = load_json(manifest_path)
    prediction_path = prediction_dir / "predictions.parquet"
    prediction_csv = prediction_dir / "predictions.csv"
    if args.materialize_fold_predictions and (args.force_infer or not (prediction_path.exists() or prediction_csv.exists())):
        checkpoint_path = manifest.get("test_checkpoint_path") or manifest.get("best_model_path") or str(manifest_path.parent / "last.ckpt")
        command = build_infer_command(
            args=args,
            manifest=manifest,
            checkpoint_path=str(checkpoint_path),
            output_dir=prediction_dir,
        )
        env = os.environ.copy()
        env.setdefault("PTV_PROGRESS_BAR", "0")
        print(f"[infer] {manifest_path.parent.name} -> {prediction_dir}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    predictions, source = load_predictions(prediction_dir)
    return predictions, source, manifest


def collect_fold_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    prediction_root = args.fold_prediction_root or (args.output_root / f"{args.prefix}_cell_drug_fold_predictions")
    for spec in FOLD_EXPERIMENTS:
        by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
        for fold in range(5):
            manifest_path = args.checkpoint_root / f"{args.prefix}_{spec.checkpoint_suffix}{fold}" / "run_manifest.json"
            if not manifest_path.exists():
                continue
            prediction_dir = prediction_root / spec.exp / f"fold{fold}"
            predictions, source, manifest = ensure_fold_predictions(
                args=args,
                manifest_path=manifest_path,
                prediction_dir=prediction_dir,
            )
            predictions = enrich_with_feature_metadata(predictions, manifest)
            for method in METHODS:
                record = evaluate_prediction_frame(
                    predictions,
                    exp=spec.exp,
                    task=spec.label,
                    split=f"fold{fold}",
                    method=method,
                    double_drug=spec.double_drug,
                    source=source,
                )
                records.append(record)
                by_method[method].append(record)
        for method, method_records in by_method.items():
            if len(method_records) == 5:
                records.append(aggregate_records(method_records, split="mean5", task=spec.label))
    return records


def collect_extra_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in EXTRA_EXPERIMENTS:
        root = args.output_root / f"{args.prefix}_{spec.output_suffix}"
        if not root.exists():
            continue
        by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
        for metrics_path in sorted(root.glob("*/run_manifest.json")):
            task_dir = metrics_path.parent
            predictions, source = load_predictions(task_dir)
            manifest = load_json(metrics_path)
            predictions = enrich_with_feature_metadata(predictions, manifest)
            test_label = predictions.get("test_label")
            if spec.exp == "exp08" and test_label is not None:
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
                    record = evaluate_prediction_frame(
                        split_predictions,
                        exp=spec.exp,
                        task=task_dir.name,
                        split=split_name,
                        method=method,
                        double_drug=spec.double_drug,
                        source=source,
                    )
                    records.append(record)
                    if split_name in {"extra", "combined"}:
                        by_method[method].append(record)
        for method, method_records in by_method.items():
            if method_records:
                records.append(aggregate_records(method_records, split="mean_extra", task=spec.label))
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"rows": records}, handle, ensure_ascii=False, indent=2, allow_nan=True)


def format_value(value: Any, precision: int) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def print_markdown(records: list[dict[str, Any]], precision: int) -> None:
    fold_summary = [row for row in records if row["split"] == "mean5"]
    extra_subset = [
        row
        for row in records
        if row["split"] in {"extra", "combined", "unseenCell_seenDrugCombo", "unseenCell_unseenDrugCombo"}
    ]
    extra_mean = [row for row in records if row["split"] == "mean_extra"]
    fold_detail = [row for row in records if row["split"] not in {"mean5", "mean_extra", "extra"}]
    for title, rows in (
        ("Fold Summary", fold_summary),
        ("Extra Subset Summary", extra_subset),
        ("Extra Mean", extra_mean),
        ("Fold Detail", fold_detail),
    ):
        if not rows:
            continue
        print(f"\n## {title}")
        print("| exp | task | split | method | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | conflicts | missing_time |")
        print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            print(
                "| {exp} | {task} | {split} | {method} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | {pos} | {neg} | {conflicts} | {missing_time} |".format(
                    exp=row["exp"],
                    task=row["task"],
                    split=row["split"],
                    method=row["method"],
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


def main() -> int:
    args = parse_args()
    records = collect_fold_records(args)
    records.extend(collect_extra_records(args))
    if not records:
        print("[error] no records collected", file=sys.stderr)
        return 1
    if args.csv_out:
        write_csv(args.csv_out, records)
    if args.json_out:
        write_json(args.json_out, records)
    if args.format == "markdown":
        print_markdown(records, args.precision)
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
