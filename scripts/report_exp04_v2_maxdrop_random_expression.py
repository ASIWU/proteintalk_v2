#!/usr/bin/env python3
"""Summarize final exp_04_v2 5-fold max-drop random-expression results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PREFIX = "20260708_exp04_v2_maxdrop_random_expr"
DEFAULT_CURRENT_BASELINE_PREFIX = "20260708_exp04_v2"
DEFAULT_TASK_NAME = "ptv3_main_singledrug"
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


def load_screen_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def winner_from_screen(screen: dict[str, Any]) -> dict[str, Any]:
    selection = screen.get("selection")
    if not isinstance(selection, dict):
        return {}
    winner = selection.get("winner")
    return winner if isinstance(winner, dict) else {}


def validate_manifest(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    fold: int,
    expected_artifact: Path,
) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{manifest_path}: {message}")

    check(manifest.get("run_status") == "fit_completed", f"run_status={manifest.get('run_status')!r}")
    check(manifest.get("test_status") == "test_completed", f"test_status={manifest.get('test_status')!r}")
    check(manifest.get("task_name") == DEFAULT_TASK_NAME, f"task_name={manifest.get('task_name')!r}")
    check(manifest.get("split_strategy") == f"pert_stratified_5fold_fold{fold}", "unexpected split_strategy")
    check(manifest.get("task_head") == "response", f"task_head={manifest.get('task_head')!r}")
    check(config_value(manifest, "have_mse_loss") is False, "expected have_mse_loss=false")
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


def metric_value(row: dict[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return finite_float(metrics.get(metric))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [row for row in records if row.get("status") == "ok"]
    summary: dict[str, Any] = {"fold_count": len(ok_records)}
    for metric in METRICS:
        values = [metric_value(row, metric) for row in ok_records]
        finite = [float(value) for value in values if value is not None]
        summary[f"mean_{metric}"] = statistics.fmean(finite) if finite else None
        summary[f"std_{metric}"] = statistics.pstdev(finite) if len(finite) > 1 else 0.0 if finite else None
        summary[f"values_{metric}"] = finite
    return summary


def compute_comparison(
    *,
    final_summary: dict[str, Any],
    current_summary: dict[str, Any],
    screen_summary: dict[str, Any],
    final_records: list[dict[str, Any]],
) -> dict[str, Any]:
    selection = screen_summary.get("selection") if isinstance(screen_summary.get("selection"), dict) else {}
    winner = selection.get("winner") if isinstance(selection.get("winner"), dict) else {}
    final_mean = finite_float(final_summary.get("mean_auprc"))
    current_mean = finite_float(current_summary.get("mean_auprc"))
    final_fold0 = next((row for row in final_records if row.get("fold") == 0 and row.get("status") == "ok"), None)
    final_fold0_auprc = metric_value(final_fold0, "auprc")
    real_reference_auprc = finite_float(selection.get("reference_auprc"))
    return {
        "winner_policy": winner.get("policy"),
        "winner_artifact_path": winner.get("artifact_path"),
        "screen_reference_fold0_auprc": real_reference_auprc,
        "screen_winner_fold0_auprc": finite_float(winner.get("auprc")),
        "screen_winner_drop_vs_real_auprc": finite_float(winner.get("drop_vs_real_auprc")),
        "final_mean_auprc": final_mean,
        "current_seed42_mean_auprc": current_mean,
        "final_auprc_drop_vs_current_seed42_mean": (
            current_mean - final_mean if current_mean is not None and final_mean is not None else None
        ),
        "final_fold0_auprc": final_fold0_auprc,
        "final_fold0_drop_vs_screen_real_reference": (
            real_reference_auprc - final_fold0_auprc
            if real_reference_auprc is not None and final_fold0_auprc is not None
            else None
        ),
    }


def markdown_summary_table(final_summary: dict[str, Any], current_summary: dict[str, Any]) -> list[str]:
    lines = [
        "| set | folds | mean AUPRC | std AUPRC | mean nAUPRC | mean AUROC | mean ACC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("maxdrop_random", final_summary), ("current_seed42_random", current_summary)):
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


def markdown_fold_table(final_records: list[dict[str, Any]], current_records: list[dict[str, Any]]) -> list[str]:
    current_by_fold = {row.get("fold"): row for row in current_records}
    lines = [
        "| fold | final AUPRC | current seed42 AUPRC | final-current | final nAUPRC | final AUROC | final count |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_records:
        fold = row.get("fold")
        current = current_by_fold.get(fold)
        final_auprc = metric_value(row, "auprc")
        current_auprc = metric_value(current, "auprc")
        delta = final_auprc - current_auprc if final_auprc is not None and current_auprc is not None else None
        lines.append(
            "| {fold} | {final_auprc} | {current_auprc} | {delta} | {nauprc} | {auroc} | {count} |".format(
                fold=fold,
                final_auprc=fmt(final_auprc),
                current_auprc=fmt(current_auprc),
                delta=fmt(delta),
                nauprc=fmt(metric_value(row, "nauprc")),
                auroc=fmt(metric_value(row, "auroc")),
                count=fmt(metric_value(row, "count"), precision=0),
            )
        )
    return lines


def write_markdown(
    *,
    args: argparse.Namespace,
    winner_artifact: Path,
    final_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    final_summary: dict[str, Any],
    current_summary: dict[str, Any],
    comparison: dict[str, Any],
    errors: list[str],
) -> None:
    lines = [
        "# exp_04_v2 Max-Drop Random Expression Results",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Final prefix: `{args.prefix}`",
        f"- Current seed42 prefix: `{args.current_baseline_prefix}`",
        f"- Task: `{DEFAULT_TASK_NAME}` / `response`",
        f"- Winner artifact: `{winner_artifact}`",
        "",
        "## Validation",
    ]
    if errors:
        lines.append(f"- Validation errors: `{len(errors)}`")
        lines.extend(f"- {error}" for error in errors[:50])
        if len(errors) > 50:
            lines.append(f"- ... {len(errors) - 50} more")
    else:
        lines.append("- Final and current seed42 manifests are present and match expected settings.")

    lines.extend(
        [
            "",
            "## Screen Selection",
            f"- Winner policy: `{comparison.get('winner_policy')}`",
            f"- Screen fold0 real-control AUPRC: `{fmt(comparison.get('screen_reference_fold0_auprc'))}`",
            f"- Screen fold0 winner AUPRC: `{fmt(comparison.get('screen_winner_fold0_auprc'))}`",
            f"- Screen fold0 drop vs real: `{fmt(comparison.get('screen_winner_drop_vs_real_auprc'))}`",
            "",
            "## 5-Fold Summary",
        ]
    )
    lines.extend(markdown_summary_table(final_summary, current_summary))
    lines.extend(
        [
            "",
            "## Comparison",
            f"- Final mean AUPRC: `{fmt(comparison.get('final_mean_auprc'))}`",
            f"- Current seed42 random mean AUPRC: `{fmt(comparison.get('current_seed42_mean_auprc'))}`",
            f"- Mean AUPRC drop vs current seed42 random: `{fmt(comparison.get('final_auprc_drop_vs_current_seed42_mean'))}`",
            f"- Final fold0 drop vs screen real-control reference: `{fmt(comparison.get('final_fold0_drop_vs_screen_real_reference'))}`",
            "",
            "## Fold Metrics",
        ]
    )
    lines.extend(markdown_fold_table(final_records, current_records))
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_folds(value: str) -> list[int]:
    result = []
    for item in value.split():
        result.append(int(item))
    if not result:
        raise ValueError("--folds must contain at least one fold")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--current-baseline-prefix", default=DEFAULT_CURRENT_BASELINE_PREFIX)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--training-ready-root", type=Path, default=Path("data/training_ready"))
    parser.add_argument(
        "--screen-json",
        type=Path,
        default=Path("outputs/2026-07/2026-07-08/20260708_exp04_v2_random_expression_screen_summary.json"),
    )
    parser.add_argument("--winner-artifact-path", type=Path, default=None)
    parser.add_argument(
        "--current-baseline-artifact-path",
        type=Path,
        default=Path("data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_seed42.npy"),
    )
    parser.add_argument("--folds", default="0 1 2 3 4")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/2026-07-08_exp04_v2_maxdrop_random_expression_results.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/2026-07/2026-07-08/20260708_exp04_v2_maxdrop_random_expression_summary.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folds = parse_folds(args.folds)
    screen_summary = load_screen_summary(args.screen_json)
    screen_winner = winner_from_screen(screen_summary)
    winner_path = args.winner_artifact_path
    if winner_path is None:
        artifact = screen_winner.get("artifact_path")
        if artifact:
            winner_path = Path(str(artifact))
    if winner_path is None:
        raise SystemExit("winner artifact path is required; run screen reporter first or pass --winner-artifact-path")

    errors: list[str] = []
    if not winner_path.exists():
        errors.append(f"missing winner artifact: {winner_path}")
    if not args.current_baseline_artifact_path.exists():
        errors.append(f"missing current baseline artifact: {args.current_baseline_artifact_path}")

    final_records, final_errors = collect_fold_records(
        prefix=args.prefix,
        label="maxdrop_random",
        checkpoint_root=args.checkpoint_root,
        expected_artifact=winner_path,
        folds=folds,
    )
    current_records, current_errors = collect_fold_records(
        prefix=args.current_baseline_prefix,
        label="current_seed42_random",
        checkpoint_root=args.checkpoint_root,
        expected_artifact=args.current_baseline_artifact_path,
        folds=folds,
    )
    errors.extend(final_errors)
    errors.extend(current_errors)

    final_summary = aggregate(final_records)
    current_summary = aggregate(current_records)
    comparison = compute_comparison(
        final_summary=final_summary,
        current_summary=current_summary,
        screen_summary=screen_summary,
        final_records=final_records,
    )
    payload = {
        "generated_at": iso_now(),
        "prefix": args.prefix,
        "current_baseline_prefix": args.current_baseline_prefix,
        "winner_artifact_path": str(winner_path),
        "screen_summary_path": str(args.screen_json),
        "final_records": final_records,
        "current_baseline_records": current_records,
        "final_summary": final_summary,
        "current_baseline_summary": current_summary,
        "comparison": comparison,
        "validation_errors": errors,
    }
    dump_json(args.json_output, payload)
    write_markdown(
        args=args,
        winner_artifact=winner_path,
        final_records=final_records,
        current_records=current_records,
        final_summary=final_summary,
        current_summary=current_summary,
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
