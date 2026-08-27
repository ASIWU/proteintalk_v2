#!/usr/bin/env python3
"""Preflight and summarize exp31 RNA-seq PDX fine-tuning benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


TASKS = [
    {
        "suffix": "sensitive_early",
        "task_name": "ptv3_exp31_rnaseq_sensitive_early",
        "label_key": "sensitive_label_early_CRPR_vs_SDPD",
        "label_description": "early sensitive CR/PR versus SD/PD",
    },
    {
        "suffix": "sensitive_late",
        "task_name": "ptv3_exp31_rnaseq_sensitive_late",
        "label_key": "sensitive_label_late_CRPR_vs_SDPD",
        "label_description": "late sensitive CR/PR versus SD/PD",
    },
    {
        "suffix": "disease_control_early",
        "task_name": "ptv3_exp31_rnaseq_disease_control_early",
        "label_key": "disease_control_label_early_CRPRSD_vs_PD",
        "label_description": "early disease-control CR/PR/SD versus PD",
    },
    {
        "suffix": "disease_control_late",
        "task_name": "ptv3_exp31_rnaseq_disease_control_late",
        "label_key": "disease_control_label_late_CRPRSD_vs_PD",
        "label_description": "late disease-control CR/PR/SD versus PD",
    },
]

SPLIT_STRATEGY = "brca_ft_valid_nonbrca_test"
EXPECTED_SOURCE_ROOT = "data/training_ready"
DEFAULT_EXP_PREFIX = "20260709_exp31_rnaseq"
DEFAULT_TRAINING_READY_ROOT = Path("data/training_ready_exp31_rnaseq")
DEFAULT_CKPT_ROOT = Path("checkpoints")
DEFAULT_OUTPUT_BASE = Path("outputs")
DEFAULT_INIT_CHECKPOINT = Path(
    "checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/last.ckpt"
)
DEFAULT_MARKDOWN = Path("docs/2026-07-09_exp31_rnaseq_pdx_ft_benchmark_results.md")
DEFAULT_CSV = Path("outputs/2026-07/2026-07-09/20260709_exp31_rnaseq_pdx_ft_benchmark_summary.csv")
DEFAULT_JSON = Path("outputs/2026-07/2026-07-09/20260709_exp31_rnaseq_pdx_ft_benchmark_summary.json")


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def output_bucket_for_prefix(prefix: str) -> Path:
    token = str(prefix)[:8]
    try:
        output_date = datetime.strptime(token, "%Y%m%d").date().isoformat()
    except ValueError:
        output_date = datetime.now().date().isoformat()
    return DEFAULT_OUTPUT_BASE / output_date[:7] / output_date


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


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


def fmt(value: Any, precision: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def int_or_blank(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    return str(number)


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def same_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        left_path = Path(str(left)).expanduser()
        right_path = Path(str(right)).expanduser()
        if not left_path.is_absolute():
            left_path = Path.cwd() / left_path
        if not right_path.is_absolute():
            right_path = Path.cwd() / right_path
        return left_path.resolve() == right_path.resolve()
    except (OSError, ValueError):
        return str(left) == str(right)


def read_table(path: Path) -> pd.DataFrame:
    parquet = path.with_suffix(".parquet")
    pkl = path.with_suffix(".pkl")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    if pkl.exists():
        return pd.read_pickle(pkl)
    raise FileNotFoundError(path)


def read_predictions(output_dir: Path) -> pd.DataFrame:
    parquet = output_dir / "predictions.parquet"
    csv_path = output_dir / "predictions.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(parquet)


def binary_metrics(y_true: Any, y_prob: Any) -> dict[str, float | int | None]:
    labels = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=np.float64)
    probs = pd.to_numeric(pd.Series(y_prob), errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(labels) & np.isfinite(probs)
    labels = labels[keep]
    probs = probs[keep]
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    count = int(labels.size)
    result: dict[str, float | int | None] = {
        "count": count,
        "positive": pos,
        "negative": neg,
        "auroc": None,
        "auprc": None,
        "auprc_baseline": None,
        "nauprc": None,
    }
    if count == 0:
        return result
    baseline = float(pos / count)
    result["auprc_baseline"] = baseline
    if pos > 0 and neg > 0:
        auprc = float(average_precision_score(labels, probs))
        result["auprc"] = auprc
        result["auroc"] = float(roc_auc_score(labels, probs))
        result["nauprc"] = auprc / baseline if baseline > 0 else None
    return result


def metric_from_metrics_json(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    task = metrics.get("task")
    if not isinstance(task, dict):
        return {}
    return {
        "auroc": finite_float(task.get("auroc")),
        "auprc": finite_float(task.get("auprc")),
        "auprc_baseline": finite_float(task.get("auprc_baseline")),
        "nauprc": finite_float(task.get("nauprc")),
        "count": int(task.get("count") or task.get("valid_count") or 0),
        "positive": int(task.get("positive_count") or 0),
        "negative": int(task.get("negative_count") or 0),
    }


def enrich_predictions(predictions: pd.DataFrame, task_dir: Path) -> pd.DataFrame:
    if "feature_row_index" not in predictions.columns:
        return predictions.copy()
    feature_table = read_table(task_dir / "feature_table.csv")
    row_indices = pd.to_numeric(predictions["feature_row_index"], errors="raise").astype(int).to_numpy()
    if row_indices.size and (row_indices.min() < 0 or row_indices.max() >= len(feature_table)):
        raise ValueError(f"feature_row_index outside feature table range: {task_dir}")
    metadata_cols = [
        "patient_sample_id",
        "source_info_row_index",
        "treatment_type",
        "cancer_type",
        "cancer_type_description",
        "exp31_label_source",
        "exp31_label_value",
        "clinical__treatment",
        "clinical__smiles_a",
        "clinical__smiles_b",
        "clinical__ptv_match_status",
    ]
    metadata_cols = [col for col in metadata_cols if col in feature_table.columns]
    metadata = feature_table.iloc[row_indices][metadata_cols].reset_index(drop=True)
    result = predictions.reset_index(drop=True).copy()
    for col in metadata_cols:
        if col not in result.columns:
            result[col] = metadata[col].to_numpy()
    return result


def group_metrics(enriched: pd.DataFrame, *, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    groups: list[tuple[str, str | None, pd.DataFrame]] = [("overall", "all", enriched)]
    if "cancer_type" in enriched.columns:
        for value, group in enriched.groupby("cancer_type", dropna=False, sort=True):
            groups.append(("cancer_type", str(value), group))
    if "treatment_type" in enriched.columns:
        for value, group in enriched.groupby("treatment_type", dropna=False, sort=True):
            groups.append(("treatment_type", str(value), group))
    for group_type, group_value, group in groups:
        metrics = binary_metrics(group["task_label"], group["pred_task_prob"])
        records.append(
            {
                "source": source,
                "group_type": group_type,
                "group_value": group_value,
                **metrics,
            }
        )
    return records


def validate_split(task_dir: Path, split_dir: Path, label_key: str) -> dict[str, Any]:
    table = read_table(task_dir / "feature_table.csv")
    manifest = load_json(split_dir / "split_manifest.json")
    result = {
        "manifest": manifest,
        "split_counts": {},
        "class_counts": {},
        "errors": [],
    }
    for split_name in ("train", "valid", "test"):
        indices_path = split_dir / f"{split_name}_indices_{SPLIT_STRATEGY}.pkl"
        if not indices_path.exists():
            result["errors"].append(f"missing {indices_path}")
            continue
        indices = [int(item) for item in load_pickle(indices_path)]
        rows = table.iloc[indices]
        labels = pd.to_numeric(rows["exp31_label_value"], errors="coerce")
        pos = int(labels.eq(1).sum())
        neg = int(labels.eq(0).sum())
        result["split_counts"][split_name] = int(len(indices))
        result["class_counts"][split_name] = {"positive": pos, "negative": neg}
        if len(indices) == 0:
            result["errors"].append(f"{split_name} split is empty")
        if pos == 0 or neg == 0:
            result["errors"].append(f"{split_name} split lacks both classes for {label_key}")
    return result


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not args.init_checkpoint.exists():
        errors.append(f"missing init checkpoint: {args.init_checkpoint}")
    root = args.training_ready_root / "ptv3"
    summary_path = root / "exp31_rnaseq_build_summary.json"
    meta_path = root / "global_meta.json"
    if not summary_path.exists():
        errors.append(f"missing build summary: {summary_path}")
        return {"generated_at": iso_now()}, errors
    if not meta_path.exists():
        errors.append(f"missing global meta: {meta_path}")
        return {"generated_at": iso_now()}, errors

    summary = load_json(summary_path)
    meta = load_json(meta_path)
    extension = meta.get("exp31_rnaseq_extension", {})
    if EXPECTED_SOURCE_ROOT not in str(extension.get("source_training_ready_root", "")):
        errors.append("global_meta does not record source_training_ready_root=data/training_ready")

    derived = root / "derived"
    expected_artifacts = [
        "drug_embedding_morgan_2048.pkl",
        "ddi_matrix.npy",
        "pdi_matrix.npy",
        "ppi_matrix.npy",
        "protein_embedding_esm.pkl",
        "cell_llm_embedding_exp31_rnaseq_qwen3_4096.npz",
        "cell_type_llm_embedding_qwen3_4096_v2.npz",
    ]
    for name in expected_artifacts:
        if not (derived / name).exists():
            errors.append(f"missing derived artifact: {derived / name}")

    added_ids = [str(item) for item in extension.get("added_pert_ids", [])]
    pdi_path = derived / "pdi_matrix.npy"
    if pdi_path.exists() and added_ids:
        pdi = np.load(pdi_path)
        pert_index = {str(k): int(v) for k, v in meta["pert_index"].items()}
        for pert_id in added_ids:
            idx = pert_index.get(pert_id)
            if idx is None:
                errors.append(f"added pert_id missing from pert_index: {pert_id}")
                continue
            row = np.asarray(pdi[idx])
            if np.nanmax(np.abs(row)) > 0.0:
                errors.append(f"new SMILES-only PDI row is not zero: {pert_id}")

    task_preflight: dict[str, Any] = {}
    for spec in TASKS:
        task_dir = root / "tasks" / spec["task_name"]
        split_dir = root / "splits" / spec["task_name"]
        if not task_dir.exists():
            errors.append(f"missing task dir: {task_dir}")
            continue
        if not split_dir.exists():
            errors.append(f"missing split dir: {split_dir}")
            continue
        loading_manifest_path = task_dir / "feature_loading_manifest.json"
        if loading_manifest_path.exists():
            loading = load_json(loading_manifest_path)
            if loading.get("label_key") != spec["label_key"]:
                errors.append(f"{spec['task_name']} label_key mismatch")
            note = str(loading.get("unified_label_note", ""))
            if "not a drug synergy label" not in note:
                errors.append(f"{spec['task_name']} missing unified label compatibility note")
        split_check = validate_split(task_dir, split_dir, spec["label_key"])
        errors.extend(f"{spec['task_name']}: {message}" for message in split_check["errors"])
        task_preflight[spec["suffix"]] = split_check

    return {
        "generated_at": iso_now(),
        "init_checkpoint_path": str(args.init_checkpoint),
        "build_summary_path": str(summary_path),
        "global_meta_path": str(meta_path),
        "build_summary": summary,
        "added_pert_ids": added_ids,
        "task_preflight": task_preflight,
    }, errors


def collect_task_result(args: argparse.Namespace, spec: dict[str, str], source: str) -> dict[str, Any]:
    suffix = spec["suffix"]
    task_name = spec["task_name"]
    if source == "ft":
        output_exp = f"{args.exp_prefix}_{suffix}_ft_infer"
        checkpoint_exp = f"{args.exp_prefix}_{suffix}_ft"
    elif source == "zeroshot":
        output_exp = f"{args.exp_prefix}_{suffix}_exp09_zeroshot"
        checkpoint_exp = None
    else:
        raise ValueError(source)

    task_dir = args.training_ready_root / "ptv3/tasks" / task_name
    output_dir = args.output_root / output_exp / task_name
    metrics_path = output_dir / "metrics.json"
    infer_manifest_path = output_dir / "run_manifest.json"
    train_manifest_path = args.checkpoint_root / checkpoint_exp / "run_manifest.json" if checkpoint_exp else None

    record: dict[str, Any] = {
        "suffix": suffix,
        "task_name": task_name,
        "label_key": spec["label_key"],
        "label_description": spec["label_description"],
        "source": source,
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "infer_manifest_path": str(infer_manifest_path),
        "train_manifest_path": str(train_manifest_path) if train_manifest_path else None,
        "status": "missing",
        "valid_auprc": None,
        "test_metrics": {},
        "group_metrics": [],
        "errors": [],
    }
    if train_manifest_path is not None:
        if train_manifest_path.exists():
            manifest = load_json(train_manifest_path)
            train_args = manifest.get("args")
            init_checkpoint_path = train_args.get("checkpoint_path") if isinstance(train_args, dict) else None
            record["init_checkpoint_path"] = init_checkpoint_path
            record["train_run_status"] = manifest.get("run_status")
            record["valid_auprc"] = finite_float(manifest.get("best_model_score"))
            record["best_model_path"] = manifest.get("best_model_path")
            if manifest.get("run_status") != "fit_completed":
                record["errors"].append(f"train run_status={manifest.get('run_status')!r}")
            if manifest.get("task_head") != "unified":
                record["errors"].append(f"train task_head={manifest.get('task_head')!r}")
            if manifest.get("have_mse_loss") is not False:
                record["errors"].append("fine-tune expected have_mse_loss=false")
            if not same_path(init_checkpoint_path, args.init_checkpoint):
                record["errors"].append(
                    f"train args.checkpoint_path={init_checkpoint_path!r}, expected {str(args.init_checkpoint)!r}"
                )
        else:
            record["errors"].append(f"missing train manifest: {train_manifest_path}")

    if not metrics_path.exists():
        record["errors"].append(f"missing metrics: {metrics_path}")
        return record
    if not infer_manifest_path.exists():
        record["errors"].append(f"missing inference manifest: {infer_manifest_path}")
        return record

    metrics = load_json(metrics_path)
    infer_manifest = load_json(infer_manifest_path)
    record["infer_checkpoint_path"] = infer_manifest.get("checkpoint_path")
    record["infer_n_predictions"] = infer_manifest.get("n_predictions")
    record["infer_split_strategy"] = infer_manifest.get("split_strategy")
    record["protein_axis_matches_checkpoint"] = infer_manifest.get("protein_axis_matches_checkpoint")
    if infer_manifest.get("split_strategy") != SPLIT_STRATEGY:
        record["errors"].append(f"infer split_strategy={infer_manifest.get('split_strategy')!r}")
    if infer_manifest.get("task_head") != "unified":
        record["errors"].append(f"infer task_head={infer_manifest.get('task_head')!r}")
    if source == "zeroshot" and not same_path(infer_manifest.get("checkpoint_path"), args.init_checkpoint):
        record["errors"].append(
            f"zero-shot checkpoint_path={infer_manifest.get('checkpoint_path')!r}, expected {str(args.init_checkpoint)!r}"
        )

    predictions = read_predictions(output_dir)
    enriched = enrich_predictions(predictions, task_dir)
    group_rows = group_metrics(enriched, source=source)
    native_task_metrics = metric_from_metrics_json(metrics)
    overall_metrics = {
        key: value
        for key, value in group_rows[0].items()
        if key not in {"source", "group_type", "group_value"}
    }
    record["metrics_json_task"] = native_task_metrics
    record["test_metrics"] = overall_metrics
    record["group_metrics"] = group_rows
    record["status"] = "ok" if not record["errors"] else "invalid"
    return record


def collect_results(args: argparse.Namespace) -> dict[str, Any]:
    preflight_summary, preflight_errors = preflight(args)
    records = []
    for spec in TASKS:
        records.append(collect_task_result(args, spec, "ft"))
        records.append(collect_task_result(args, spec, "zeroshot"))

    ft_records = [row for row in records if row["source"] == "ft"]
    selectable = [row for row in ft_records if row.get("status") == "ok" and finite_float(row.get("valid_auprc")) is not None]
    selectable.sort(
        key=lambda row: (
            finite_float(row.get("valid_auprc")) or float("-inf"),
            finite_float(row.get("test_metrics", {}).get("auprc")) or float("-inf"),
            row["suffix"],
        ),
        reverse=True,
    )
    selected = selectable[0] if selectable else None
    errors = list(preflight_errors)
    for row in records:
        errors.extend(str(item) for item in row.get("errors", []))
    return {
        "generated_at": iso_now(),
        "exp_prefix": args.exp_prefix,
        "init_checkpoint_path": str(args.init_checkpoint),
        "split_strategy": SPLIT_STRATEGY,
        "training_ready_root": str(args.training_ready_root),
        "checkpoint_root": str(args.checkpoint_root),
        "output_root": str(args.output_root),
        "preflight": preflight_summary,
        "records": records,
        "selected_by_brca_valid_auprc": selected,
        "errors": errors,
    }


def flatten_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in summary["records"]:
        base = {
            "suffix": record["suffix"],
            "task_name": record["task_name"],
            "label_key": record["label_key"],
            "source": record["source"],
            "status": record["status"],
            "valid_auprc": record.get("valid_auprc"),
            "output_dir": record["output_dir"],
        }
        for group in record.get("group_metrics", []):
            rows.append(
                {
                    **base,
                    "group_type": group.get("group_type"),
                    "group_value": group.get("group_value"),
                    "count": group.get("count"),
                    "positive": group.get("positive"),
                    "negative": group.get("negative"),
                    "auroc": group.get("auroc"),
                    "auprc": group.get("auprc"),
                    "auprc_baseline": group.get("auprc_baseline"),
                    "nauprc": group.get("nauprc"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "suffix",
        "task_name",
        "label_key",
        "source",
        "status",
        "valid_auprc",
        "group_type",
        "group_value",
        "count",
        "positive",
        "negative",
        "auroc",
        "auprc",
        "auprc_baseline",
        "nauprc",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    selected = summary.get("selected_by_brca_valid_auprc")
    lines = [
        "# Exp31 RNA-seq PDX Fine-tune Benchmark",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Experiment prefix: `{summary['exp_prefix']}`",
        f"- Split: `{SPLIT_STRATEGY}` (BRCA train/valid, non-BRCA test)",
        f"- Init checkpoint: `{summary['init_checkpoint_path']}`.",
        "- Label carrier note: exp31 writes each clinical label into `synergy` only for unified-head compatibility; it is not a biological synergy label.",
        "",
    ]
    errors = summary.get("errors", [])
    if errors:
        lines.extend(["## Status", "", f"- Errors/warnings: `{len(errors)}`", ""])
        for item in errors[:30]:
            lines.append(f"- {item}")
        if len(errors) > 30:
            lines.append(f"- ... {len(errors) - 30} more")
        lines.append("")
    else:
        lines.extend(["## Status", "", "- All expected preflight and result files passed reporter checks.", ""])

    if selected:
        lines.extend(
            [
                "## Selected Setting",
                "",
                f"- Selected label: `{selected['suffix']}`",
                f"- BRCA valid AUPRC: `{fmt(selected.get('valid_auprc'))}`",
                f"- Non-BRCA test AUPRC: `{fmt(selected.get('test_metrics', {}).get('auprc'))}`",
                f"- Non-BRCA test AUROC: `{fmt(selected.get('test_metrics', {}).get('auroc'))}`",
                "",
            ]
        )

    overview_rows: list[list[str]] = []
    for record in summary["records"]:
        metrics = record.get("test_metrics", {})
        overview_rows.append(
            [
                record["suffix"],
                record["source"],
                record["status"],
                fmt(record.get("valid_auprc")),
                int_or_blank(metrics.get("count")),
                int_or_blank(metrics.get("positive")),
                fmt(metrics.get("auroc")),
                fmt(metrics.get("auprc")),
                fmt(metrics.get("auprc_baseline")),
                fmt(metrics.get("nauprc")),
            ]
        )
    lines.extend(
        [
            "## Overall Metrics",
            "",
            markdown_table(
                ["label", "source", "status", "valid AUPRC", "n", "pos", "AUROC", "AUPRC", "base", "nAUPRC"],
                overview_rows,
            ),
            "",
        ]
    )

    group_rows: list[list[str]] = []
    for record in summary["records"]:
        for group in record.get("group_metrics", []):
            if group.get("group_type") == "overall":
                continue
            group_rows.append(
                [
                    record["suffix"],
                    record["source"],
                    str(group.get("group_type")),
                    str(group.get("group_value")),
                    int_or_blank(group.get("count")),
                    int_or_blank(group.get("positive")),
                    fmt(group.get("auroc")),
                    fmt(group.get("auprc")),
                    fmt(group.get("auprc_baseline")),
                    fmt(group.get("nauprc")),
                ]
            )
    if group_rows:
        lines.extend(
            [
                "## Stratified Metrics",
                "",
                markdown_table(
                    ["label", "source", "group", "value", "n", "pos", "AUROC", "AUPRC", "base", "nAUPRC"],
                    group_rows,
                ),
                "",
            ]
        )

    preflight_summary = summary.get("preflight", {})
    build_summary = preflight_summary.get("build_summary", {})
    if build_summary:
        lines.extend(
            [
                "## Data Build",
                "",
                f"- Build summary: `{repo_rel(Path(preflight_summary.get('build_summary_path', '')) )}`",
                f"- Added SMILES-only drugs: `{len(preflight_summary.get('added_pert_ids', []))}`",
                f"- Dropped missing cancer_type rows: `{build_summary.get('dropped_missing_cancer_type_rows')}`",
                f"- RNA axis coverage: `{build_summary.get('rna_audit', {}).get('axis_covered_by_rna')}` / `{build_summary.get('rna_audit', {}).get('axis_size')}` proteins.",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-prefix", default=DEFAULT_EXP_PREFIX)
    parser.add_argument("--training-ready-root", type=Path, default=DEFAULT_TRAINING_READY_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = output_bucket_for_prefix(args.exp_prefix)

    if args.preflight_only:
        payload, errors = preflight(args)
        payload["errors"] = errors
        dump_json(args.output_json, payload)
        print(f"[preflight] wrote {args.output_json}")
        if errors:
            raise SystemExit("\n".join(errors))
        print("[preflight] ok")
        return

    summary = collect_results(args)
    flat_rows = flatten_rows(summary)
    dump_json(args.output_json, summary)
    write_csv(args.output_csv, flat_rows)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(f"[report] wrote {args.output_markdown}")
    print(f"[report] wrote {args.output_csv}")
    print(f"[report] wrote {args.output_json}")
    if summary["errors"]:
        raise SystemExit(f"report completed with {len(summary['errors'])} errors")


if __name__ == "__main__":
    main()
