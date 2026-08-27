#!/usr/bin/env python3
"""Summarize exp_04_v2 global mean/std random-expression 5-fold results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PREFIX = "20260709_exp04_v2_global_meanstd_random_expr"
DEFAULT_CURRENT_BASELINE_PREFIX = "20260708_exp04_v2"
DEFAULT_MAXDROP_PREFIX = "20260708_exp04_v2_maxdrop_random_expr_clean"
DEFAULT_TASK_NAME = "ptv3_main_singledrug"
DEFAULT_TASK_HEAD = "response"
DEFAULT_COVARIATES = ("machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time")
RUN_SUFFIX = "exp04_v2_random_no_mse_fold"
METRICS = {
    "auprc": "test/task_auprc",
    "auprc_baseline": "test/task_auprc_baseline",
    "nauprc": "test/task_nauprc",
    "auroc": "test/task_auroc",
    "acc": "test/task_acc",
    "count": "test/task_count",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def fmt(value: Any, precision: int = 6) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def config_value(manifest: dict[str, Any], key: str) -> Any:
    if key in manifest:
        return manifest[key]
    args = manifest.get("args")
    if isinstance(args, dict):
        return args.get(key)
    return None


def normalize_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def same_path(actual: Any, expected: Path) -> bool:
    return normalize_path(actual) == str(expected.expanduser().resolve())


def first_test_result(manifest: dict[str, Any]) -> dict[str, Any]:
    results = manifest.get("test_results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0]
    return {}


def extract_metrics(manifest: dict[str, Any]) -> dict[str, float | None]:
    result = first_test_result(manifest)
    return {name: finite_float(result.get(source_key)) for name, source_key in METRICS.items()}


def metric_value(row: dict[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return finite_float(metrics.get(metric))


def parse_folds(value: str) -> list[int]:
    folds = [int(item) for item in value.split()]
    if not folds:
        raise ValueError("--folds must contain at least one fold")
    return folds


def artifact_header(path: Path) -> dict[str, Any]:
    import numpy as np
    from numpy.lib import format as np_format

    with path.open("rb") as handle:
        version = np_format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np_format.read_array_header_1_0(handle)
        else:
            shape, fortran_order, dtype = np_format.read_array_header_2_0(handle)
    return {
        "path": str(path),
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": bool(fortran_order),
    }


def validate_manifest(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    fold: int,
    expected_artifact: Path,
    require_random_saved: bool,
) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{manifest_path}: {message}")

    check(manifest.get("run_status") == "fit_completed", f"run_status={manifest.get('run_status')!r}")
    check(manifest.get("test_status") == "test_completed", f"test_status={manifest.get('test_status')!r}")
    check(manifest.get("task_name") == DEFAULT_TASK_NAME, f"task_name={manifest.get('task_name')!r}")
    check(manifest.get("split_strategy") == f"pert_stratified_5fold_fold{fold}", "unexpected split_strategy")
    check(manifest.get("task_head") == DEFAULT_TASK_HEAD, f"task_head={manifest.get('task_head')!r}")
    check(config_value(manifest, "have_mse_loss") is False, "expected have_mse_loss=false")
    if require_random_saved:
        check(config_value(manifest, "control_expression_mode") == "random_saved", "expected random_saved")
        check(same_path(config_value(manifest, "random_control_expression_path"), expected_artifact), "unexpected artifact path")
    check(config_value(manifest, "graph_feature_mode") == "real", "expected graph_feature_mode=real")
    check(int(config_value(manifest, "target_protein_max_length") or 0) == 32, "expected target max length 32")
    check(config_value(manifest, "protein_concat_mode") == "pcep", "expected protein_concat_mode=pcep")
    check(tuple(config_value(manifest, "batch_cov_list") or ()) == DEFAULT_COVARIATES, "unexpected batch_cov_list")
    check(bool(first_test_result(manifest)), "missing test_results[0]")
    return errors


def collect_fold_records(
    *,
    prefix: str,
    label: str,
    checkpoint_root: Path,
    expected_artifact: Path,
    folds: list[int],
    require_random_saved: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for fold in folds:
        exp_name = f"{prefix}_{RUN_SUFFIX}{fold}"
        manifest_path = checkpoint_root / exp_name / "run_manifest.json"
        base = {
            "set": label,
            "fold": fold,
            "experiment_name": exp_name,
            "manifest_path": str(manifest_path),
            "expected_artifact_path": str(expected_artifact),
        }
        if not manifest_path.exists():
            errors.append(f"missing manifest: {manifest_path}")
            records.append({**base, "status": "missing"})
            continue
        manifest = load_json(manifest_path)
        validation_errors = validate_manifest(
            manifest=manifest,
            manifest_path=manifest_path,
            fold=fold,
            expected_artifact=expected_artifact,
            require_random_saved=require_random_saved,
        )
        errors.extend(validation_errors)
        records.append(
            {
                **base,
                "status": "ok" if not validation_errors else "invalid",
                "run_status": manifest.get("run_status"),
                "test_status": manifest.get("test_status"),
                "random_control_expression_path": config_value(manifest, "random_control_expression_path"),
                "metrics": extract_metrics(manifest),
            }
        )
    return records, errors


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [row for row in records if row.get("status") == "ok"]
    summary: dict[str, Any] = {"fold_count": len(ok_records)}
    for metric in METRICS:
        finite = [
            float(value)
            for value in (metric_value(row, metric) for row in ok_records)
            if value is not None
        ]
        summary[f"mean_{metric}"] = statistics.fmean(finite) if finite else None
        summary[f"std_{metric}"] = statistics.pstdev(finite) if len(finite) > 1 else 0.0 if finite else None
        summary[f"values_{metric}"] = finite
    return summary


def find_policy_record(payload: Any, policy: str) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if payload.get("policy") == policy:
            return payload
        for value in payload.values():
            found = find_policy_record(value, policy)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_policy_record(value, policy)
            if found is not None:
                return found
    return None


def find_reference_auprc(screen_summary: dict[str, Any]) -> float | None:
    selection = screen_summary.get("selection")
    if isinstance(selection, dict):
        value = finite_float(selection.get("reference_auprc"))
        if value is not None:
            return value
    record = find_policy_record(screen_summary, "real_control_full_nomse")
    if record:
        return finite_float(record.get("auprc"))
    return None


def build_comparison(
    *,
    global_summary: dict[str, Any],
    current_summary: dict[str, Any],
    maxdrop_summary: dict[str, Any] | None,
    screen_summary: dict[str, Any],
) -> dict[str, Any]:
    global_mean = finite_float(global_summary.get("mean_auprc"))
    current_mean = finite_float(current_summary.get("mean_auprc"))
    maxdrop_mean = finite_float(maxdrop_summary.get("mean_auprc")) if maxdrop_summary else None
    screen_global = find_policy_record(screen_summary, "global_normal_clip")
    screen_real = find_reference_auprc(screen_summary)
    screen_global_auprc = None
    screen_global_drop = None
    if screen_global:
        screen_global_auprc = finite_float(screen_global.get("auprc"))
        if screen_global_auprc is None and isinstance(screen_global.get("metrics"), dict):
            screen_global_auprc = finite_float(screen_global["metrics"].get("auprc"))
        screen_global_drop = finite_float(screen_global.get("drop_vs_real_auprc"))
        if screen_global_drop is None and screen_real is not None and screen_global_auprc is not None:
            screen_global_drop = screen_real - screen_global_auprc
    return {
        "global_mean_auprc": global_mean,
        "current_seed42_mean_auprc": current_mean,
        "maxdrop_mean_auprc": maxdrop_mean,
        "global_minus_current_seed42_mean_auprc": (
            global_mean - current_mean if global_mean is not None and current_mean is not None else None
        ),
        "global_minus_maxdrop_mean_auprc": (
            global_mean - maxdrop_mean if global_mean is not None and maxdrop_mean is not None else None
        ),
        "screen_real_fold0_auprc": screen_real,
        "screen_global_fold0_auprc": screen_global_auprc,
        "screen_global_drop_vs_real_auprc": screen_global_drop,
    }


def markdown_summary_table(summaries: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "| set | folds | mean AUPRC | std AUPRC | mean nAUPRC | mean AUROC | mean ACC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in summaries:
        lines.append(
            "| {label} | {folds} | {auprc} | {std_auprc} | {nauprc} | {auroc} | {acc} |".format(
                label=label,
                folds=summary.get("fold_count", 0),
                auprc=fmt(summary.get("mean_auprc")),
                std_auprc=fmt(summary.get("std_auprc")),
                nauprc=fmt(summary.get("mean_nauprc")),
                auroc=fmt(summary.get("mean_auroc")),
                acc=fmt(summary.get("mean_acc")),
            )
        )
    return lines


def markdown_fold_table(
    *,
    global_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    maxdrop_records: list[dict[str, Any]] | None,
) -> list[str]:
    current_by_fold = {row.get("fold"): row for row in current_records}
    maxdrop_by_fold = {row.get("fold"): row for row in maxdrop_records or []}
    lines = [
        "| fold | global AUPRC | current seed42 AUPRC | max-drop AUPRC | global-current | global-maxdrop | global nAUPRC | global AUROC | global ACC | count |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_records:
        fold = row.get("fold")
        current = current_by_fold.get(fold)
        maxdrop = maxdrop_by_fold.get(fold)
        global_auprc = metric_value(row, "auprc")
        current_auprc = metric_value(current, "auprc")
        maxdrop_auprc = metric_value(maxdrop, "auprc")
        global_current = (
            global_auprc - current_auprc if global_auprc is not None and current_auprc is not None else None
        )
        global_maxdrop = (
            global_auprc - maxdrop_auprc if global_auprc is not None and maxdrop_auprc is not None else None
        )
        lines.append(
            "| {fold} | {global_auprc} | {current_auprc} | {maxdrop_auprc} | {global_current} | {global_maxdrop} | {nauprc} | {auroc} | {acc} | {count} |".format(
                fold=fold,
                global_auprc=fmt(global_auprc),
                current_auprc=fmt(current_auprc),
                maxdrop_auprc=fmt(maxdrop_auprc),
                global_current=fmt(global_current),
                global_maxdrop=fmt(global_maxdrop),
                nauprc=fmt(metric_value(row, "nauprc")),
                auroc=fmt(metric_value(row, "auroc")),
                acc=fmt(metric_value(row, "acc")),
                count=fmt(metric_value(row, "count"), precision=0),
            )
        )
    return lines


def write_markdown(
    *,
    args: argparse.Namespace,
    artifact_meta: dict[str, Any],
    artifact_info: dict[str, Any] | None,
    global_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    maxdrop_records: list[dict[str, Any]] | None,
    global_summary: dict[str, Any],
    current_summary: dict[str, Any],
    maxdrop_summary: dict[str, Any] | None,
    comparison: dict[str, Any],
    errors: list[str],
) -> None:
    summaries = [
        ("global_meanstd_random", global_summary),
        ("current_seed42_random", current_summary),
    ]
    if maxdrop_summary is not None:
        summaries.append(("maxdrop_random", maxdrop_summary))

    lines = [
        "# exp_04_v2 Global Mean/Std Random Expression Results",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Global mean/std prefix: `{args.prefix}`",
        f"- Current seed42 baseline prefix: `{args.current_baseline_prefix}`",
        f"- Max-drop comparison prefix: `{args.maxdrop_prefix}`",
        f"- Task: `{DEFAULT_TASK_NAME}` / `{DEFAULT_TASK_HEAD}`",
        f"- Split: `pert_stratified_5fold_fold0..4`",
        f"- Artifact: `{args.global_artifact_path}`",
        "",
        "## Validation",
    ]
    if errors:
        lines.append(f"- Validation errors: `{len(errors)}`")
        lines.extend(f"- {error}" for error in errors[:60])
        if len(errors) > 60:
            lines.append(f"- ... {len(errors) - 60} more")
    else:
        lines.append("- All expected manifests are present and match the global mean/std random-expression setting.")

    lines.extend(
        [
            "",
            "## Artifact",
            f"- Policy: `{artifact_meta.get('policy')}`",
            f"- Global control mean: `{fmt(artifact_meta.get('global_control_mean'))}`",
            f"- Global control std: `{fmt(artifact_meta.get('global_control_std'))}`",
            f"- Shape: `{artifact_info.get('shape') if artifact_info else ''}`",
            f"- Dtype: `{artifact_info.get('dtype') if artifact_info else ''}`",
            "",
            "## 5-Fold Summary",
        ]
    )
    lines.extend(markdown_summary_table(summaries))
    lines.extend(
        [
            "",
            "## Comparison",
            f"- Global mean/std mean AUPRC: `{fmt(comparison.get('global_mean_auprc'))}`",
            f"- Current seed42 random mean AUPRC: `{fmt(comparison.get('current_seed42_mean_auprc'))}`",
            f"- Global minus current seed42 mean AUPRC: `{fmt(comparison.get('global_minus_current_seed42_mean_auprc'))}`",
            f"- Max-drop random mean AUPRC: `{fmt(comparison.get('maxdrop_mean_auprc'))}`",
            f"- Global minus max-drop mean AUPRC: `{fmt(comparison.get('global_minus_maxdrop_mean_auprc'))}`",
            f"- Fold0 screen real-control AUPRC: `{fmt(comparison.get('screen_real_fold0_auprc'))}`",
            f"- Fold0 screen global mean/std AUPRC: `{fmt(comparison.get('screen_global_fold0_auprc'))}`",
            f"- Fold0 screen global drop vs real: `{fmt(comparison.get('screen_global_drop_vs_real_auprc'))}`",
            "",
            "## Fold Metrics",
        ]
    )
    lines.extend(
        markdown_fold_table(
            global_records=global_records,
            current_records=current_records,
            maxdrop_records=maxdrop_records,
        )
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--current-baseline-prefix", default=DEFAULT_CURRENT_BASELINE_PREFIX)
    parser.add_argument("--maxdrop-prefix", default=DEFAULT_MAXDROP_PREFIX)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--folds", default="0 1 2 3 4")
    parser.add_argument(
        "--global-artifact-path",
        type=Path,
        default=Path(
            "data/training_ready/ptv3/tasks/ptv3_main_singledrug/"
            "random_control_expression_global_normal_clip_seed42.npy"
        ),
    )
    parser.add_argument(
        "--current-baseline-artifact-path",
        type=Path,
        default=Path("data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_seed42.npy"),
    )
    parser.add_argument(
        "--maxdrop-artifact-path",
        type=Path,
        default=Path(
            "data/training_ready/ptv3/tasks/ptv3_main_singledrug/"
            "random_control_expression_per_row_gene_permutation_seed42.npy"
        ),
    )
    parser.add_argument(
        "--screen-json",
        type=Path,
        default=Path("outputs/2026-07/2026-07-08/20260708_exp04_v2_random_expression_screen_summary.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/2026-07-09_exp04_v2_global_meanstd_random_expression_results.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/2026-07/2026-07-09/20260709_exp04_v2_global_meanstd_random_expression_summary.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-maxdrop-comparison", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folds = parse_folds(args.folds)
    errors: list[str] = []

    artifact_meta_path = args.global_artifact_path.with_suffix(".meta.json")
    artifact_meta: dict[str, Any] = {}
    artifact_info: dict[str, Any] | None = None
    if not args.global_artifact_path.exists():
        errors.append(f"missing global artifact: {args.global_artifact_path}")
    else:
        artifact_info = artifact_header(args.global_artifact_path)
    if not artifact_meta_path.exists():
        errors.append(f"missing global artifact meta: {artifact_meta_path}")
    else:
        artifact_meta = load_json(artifact_meta_path)
        if artifact_meta.get("policy") != "global_normal_clip":
            errors.append(f"unexpected artifact policy: {artifact_meta.get('policy')!r}")

    if not args.current_baseline_artifact_path.exists():
        errors.append(f"missing current seed42 artifact: {args.current_baseline_artifact_path}")
    if not args.skip_maxdrop_comparison and not args.maxdrop_artifact_path.exists():
        errors.append(f"missing max-drop artifact: {args.maxdrop_artifact_path}")

    global_records, global_errors = collect_fold_records(
        prefix=args.prefix,
        label="global_meanstd_random",
        checkpoint_root=args.checkpoint_root,
        expected_artifact=args.global_artifact_path,
        folds=folds,
    )
    current_records, current_errors = collect_fold_records(
        prefix=args.current_baseline_prefix,
        label="current_seed42_random",
        checkpoint_root=args.checkpoint_root,
        expected_artifact=args.current_baseline_artifact_path,
        folds=folds,
    )
    errors.extend(global_errors)
    errors.extend(current_errors)

    maxdrop_records: list[dict[str, Any]] | None = None
    maxdrop_summary: dict[str, Any] | None = None
    if not args.skip_maxdrop_comparison:
        maxdrop_records, maxdrop_errors = collect_fold_records(
            prefix=args.maxdrop_prefix,
            label="maxdrop_random",
            checkpoint_root=args.checkpoint_root,
            expected_artifact=args.maxdrop_artifact_path,
            folds=folds,
        )
        errors.extend(maxdrop_errors)
        maxdrop_summary = aggregate(maxdrop_records)

    screen_summary = load_json(args.screen_json) if args.screen_json.exists() else {}
    global_summary = aggregate(global_records)
    current_summary = aggregate(current_records)
    comparison = build_comparison(
        global_summary=global_summary,
        current_summary=current_summary,
        maxdrop_summary=maxdrop_summary,
        screen_summary=screen_summary,
    )

    payload = {
        "generated_at": iso_now(),
        "prefix": args.prefix,
        "current_baseline_prefix": args.current_baseline_prefix,
        "maxdrop_prefix": None if args.skip_maxdrop_comparison else args.maxdrop_prefix,
        "global_artifact_path": str(args.global_artifact_path),
        "artifact_meta": artifact_meta,
        "artifact_info": artifact_info,
        "global_records": global_records,
        "current_baseline_records": current_records,
        "maxdrop_records": maxdrop_records,
        "global_summary": global_summary,
        "current_baseline_summary": current_summary,
        "maxdrop_summary": maxdrop_summary,
        "comparison": comparison,
        "validation_errors": errors,
    }
    dump_json(args.json_output, payload)
    write_markdown(
        args=args,
        artifact_meta=artifact_meta,
        artifact_info=artifact_info,
        global_records=global_records,
        current_records=current_records,
        maxdrop_records=maxdrop_records,
        global_summary=global_summary,
        current_summary=current_summary,
        maxdrop_summary=maxdrop_summary,
        comparison=comparison,
        errors=errors,
    )
    print(f"[report] wrote {args.markdown_output}")
    print(f"[report] wrote {args.json_output}")
    if errors and not args.allow_incomplete:
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
