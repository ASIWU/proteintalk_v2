#!/usr/bin/env python3
"""Compare selected Cell+cell_type LLM PTV3 suite against the tuned Cell LLM baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = "20260608_cell_celltype_llm_clip10_selected_v1"
DEFAULT_BASELINE = "20260604_cell_llm_clip10_tuned_selected_v1"

EXPECTED_FOLDS = {
    "exp01": ("exp01_single_pert_stratified_5fold_single_pert_stratified_fold", "real"),
    "exp02": ("exp02_single_cell_type_5fold_single_cell_type_fold", "real"),
    "exp03": ("exp03_single_cell_5fold_single_cell_fold", "real"),
    "exp04": ("exp04_single_no_mse_5fold_single_no_mse_fold", "real"),
    "exp05": ("exp05_single_no_graph_5fold_single_no_graph_fold", "zero"),
    "exp06": ("exp06_double_pert_pair_5fold_double_pert_pair_fold", "real"),
}
EXPECTED_EXTRA = {
    "exp07": ("exp07_extra_single_all_train_infer_all_single_for_extra", "real"),
    "exp08": ("exp08_extra_double_all_train_infer_all_single_double_for_extra", "real"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def audit_manifests(prefix: str, checkpoint_root: Path) -> list[str]:
    errors: list[str] = []

    def check_manifest(exp: str, manifest_path: Path, expected_graph: str) -> None:
        if not manifest_path.exists():
            errors.append(f"missing manifest: {manifest_path}")
            return
        manifest = load_json(manifest_path)
        if manifest.get("run_status") != "fit_completed":
            errors.append(f"{manifest_path}: run_status={manifest.get('run_status')!r}")
        if manifest.get("cell_llm_mode") != "frozen":
            errors.append(f"{manifest_path}: cell_llm_mode={manifest.get('cell_llm_mode')!r}")
        if manifest.get("cell_type_llm_mode") != "frozen":
            errors.append(f"{manifest_path}: cell_type_llm_mode={manifest.get('cell_type_llm_mode')!r}")
        cell_summary = manifest.get("cell_llm_summary") or {}
        cell_type_summary = manifest.get("cell_type_llm_summary") or {}
        if int(cell_summary.get("embedding_rows", -1)) != 74:
            errors.append(f"{manifest_path}: bad cell_llm_summary.embedding_rows")
        if int(cell_type_summary.get("embedding_rows", -1)) != 14:
            errors.append(f"{manifest_path}: bad cell_type_llm_summary.embedding_rows")
        if cell_summary.get("index_column") != "Cell_index":
            errors.append(f"{manifest_path}: bad cell_llm index_column")
        if cell_type_summary.get("index_column") != "cell_type_index":
            errors.append(f"{manifest_path}: bad cell_type_llm index_column")
        if manifest.get("graph_feature_mode") != expected_graph:
            errors.append(f"{manifest_path}: graph_feature_mode={manifest.get('graph_feature_mode')!r}")
        if exp in {"exp07", "exp08"}:
            policy = manifest.get("reference_epoch_policy") or {}
            if not policy:
                errors.append(f"{manifest_path}: missing reference_epoch_policy")

    for exp, (suffix, expected_graph) in EXPECTED_FOLDS.items():
        for fold in range(5):
            manifest_path = checkpoint_root / f"{prefix}_{suffix}{fold}" / "run_manifest.json"
            check_manifest(exp, manifest_path, expected_graph)
    for exp, (suffix, expected_graph) in EXPECTED_EXTRA.items():
        manifest_path = checkpoint_root / f"{prefix}_{suffix}" / "run_manifest.json"
        check_manifest(exp, manifest_path, expected_graph)
    return errors


def build_comparison(prefix: str, baseline_prefix: str, output_root: Path) -> pd.DataFrame:
    new_path = output_root / f"{prefix}_cell_drug_dose_time_eval.csv"
    baseline_path = output_root / f"{baseline_prefix}_cell_drug_dose_time_eval.csv"
    if not new_path.exists():
        raise FileNotFoundError(f"missing new report CSV: {new_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"missing baseline report CSV: {baseline_path}")
    new_df = pd.read_csv(new_path)
    baseline_df = pd.read_csv(baseline_path)
    key_cols = ["exp", "task", "split", "method"]
    metric_cols = ["auroc", "auprc", "auprc_baseline", "nauprc", "valid_count", "positive_count", "negative_count"]
    merged = new_df.merge(
        baseline_df,
        on=key_cols,
        how="inner",
        suffixes=("_new", "_baseline"),
    )
    rows = []
    for _, row in merged.iterrows():
        item = {key: row[key] for key in key_cols}
        for metric in metric_cols:
            new_value = finite(row.get(f"{metric}_new"))
            baseline_value = finite(row.get(f"{metric}_baseline"))
            item[f"{metric}_new"] = new_value
            item[f"{metric}_baseline"] = baseline_value
            if new_value is not None and baseline_value is not None:
                item[f"{metric}_delta"] = new_value - baseline_value
            else:
                item[f"{metric}_delta"] = None
        rows.append(item)
    return pd.DataFrame(rows)


def write_markdown(path: Path, comparison: pd.DataFrame, audit_errors: list[str], prefix: str, baseline_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean = comparison[(comparison["split"].isin(["mean5", "mean_extra"])) & (comparison["method"] == "original")]
    lines = [
        f"# {prefix} Cell + cell_type LLM Selected Suite",
        "",
        f"Baseline: `{baseline_prefix}`.",
        f"Audit errors: `{len(audit_errors)}`.",
        "",
        "## Mean Original Metrics",
        "",
        "| exp | task | split | AUROC new | AUPRC new | n-AUPRC new | AUPRC delta | AUROC delta | count |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in mean.sort_values(["exp", "split"]).iterrows():
        lines.append(
            "| {exp} | {task} | {split} | {auroc:.6f} | {auprc:.6f} | {nauprc:.6f} | {dauprc:+.6f} | {dauroc:+.6f} | {count:.0f} |".format(
                exp=row["exp"],
                task=row["task"],
                split=row["split"],
                auroc=row["auroc_new"],
                auprc=row["auprc_new"],
                nauprc=row["nauprc_new"],
                dauprc=row["auprc_delta"],
                dauroc=row["auroc_delta"],
                count=row["valid_count_new"],
            )
        )
    if audit_errors:
        lines.extend(["", "## Audit Errors", ""])
        lines.extend(f"- {error}" for error in audit_errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--baseline-prefix", default=DEFAULT_BASELINE)
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--tsv-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = build_comparison(args.prefix, args.baseline_prefix, args.output_root)
    audit_errors = audit_manifests(args.prefix, args.checkpoint_root)
    tsv_out = args.tsv_out or args.output_root / f"{args.prefix}_cell_celltype_llm_comparison.tsv"
    markdown_out = args.markdown_out or REPO_ROOT / "logs" / f"{args.prefix}_cell_celltype_llm_comparison.md"
    tsv_out.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(tsv_out, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    write_markdown(markdown_out, comparison, audit_errors, args.prefix, args.baseline_prefix)
    print(f"[comparison] rows={len(comparison)} audit_errors={len(audit_errors)}")
    print(f"[comparison] tsv={tsv_out}")
    print(f"[comparison] markdown={markdown_out}")
    if audit_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
