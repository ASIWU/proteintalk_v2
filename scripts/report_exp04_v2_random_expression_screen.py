#!/usr/bin/env python3
"""Summarize exp_04_v2 fold0 random-expression screen manifests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PREFIX = "20260708_exp04_v2_random_expr_screen"
DEFAULT_TASK_NAME = "ptv3_main_singledrug"
DEFAULT_SPLIT_STRATEGY = "pert_stratified_5fold_fold0"
DEFAULT_COVARIATES = ("machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time")
REFERENCE_POLICY = "real_control_full_nomse"
ZERO_DIAGNOSTIC_POLICY = "zero_control"
RANDOM_POLICIES = (
    "per_protein_normal_clip",
    "global_normal_clip",
    "global_value_bootstrap",
    "fixed_gene_permutation",
    "per_row_gene_permutation",
    "cross_cell_real_control",
)
POLICIES = (REFERENCE_POLICY, *RANDOM_POLICIES, ZERO_DIAGNOSTIC_POLICY)
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


def task_dir(args: argparse.Namespace) -> Path:
    if args.task_dir is not None:
        return args.task_dir
    return args.training_ready_root / "ptv3/tasks" / DEFAULT_TASK_NAME


def artifact_path_for(task_directory: Path, policy: str) -> Path | None:
    if policy == REFERENCE_POLICY:
        return None
    return task_directory / f"random_control_expression_{policy}_seed42.npy"


def artifact_meta(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return load_json(meta_path)


def role_for(policy: str) -> str:
    if policy == REFERENCE_POLICY:
        return "real_reference"
    if policy == ZERO_DIAGNOSTIC_POLICY:
        return "zero_diagnostic"
    return "random_candidate"


def validate_manifest(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    policy: str,
    expected_artifact: Path | None,
    args: argparse.Namespace,
) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{manifest_path}: {message}")

    check(manifest.get("run_status") == "fit_completed", f"run_status={manifest.get('run_status')!r}")
    check(manifest.get("test_status") == "test_completed", f"test_status={manifest.get('test_status')!r}")
    check(manifest.get("task_name") == DEFAULT_TASK_NAME, f"task_name={manifest.get('task_name')!r}")
    check(manifest.get("split_strategy") == DEFAULT_SPLIT_STRATEGY, f"split_strategy={manifest.get('split_strategy')!r}")
    check(manifest.get("task_head") == "response", f"task_head={manifest.get('task_head')!r}")
    check(config_value(manifest, "have_mse_loss") is False, "expected have_mse_loss=false")
    check(config_value(manifest, "graph_feature_mode") == "real", "expected graph_feature_mode=real")
    check(int(config_value(manifest, "target_protein_max_length") or 0) == 32, "expected target max length 32")
    check(config_value(manifest, "protein_concat_mode") == "pcep", "expected protein_concat_mode=pcep")
    check(tuple(config_value(manifest, "batch_cov_list") or ()) == DEFAULT_COVARIATES, "unexpected batch_cov_list")

    if policy == REFERENCE_POLICY:
        check(config_value(manifest, "control_expression_mode") == "real", "expected real control expression")
        check(config_value(manifest, "random_control_expression_path") in {None, ""}, "unexpected random path")
    else:
        check(config_value(manifest, "control_expression_mode") == "random_saved", "expected random_saved")
        if expected_artifact is None:
            check(False, "missing expected random artifact path")
        else:
            check(same_path(config_value(manifest, "random_control_expression_path"), expected_artifact), "unexpected artifact path")

    expected_drug = args.training_ready_root / "ptv3/derived/drug_embedding_morgan_2048.pkl"
    check(same_path(config_value(manifest, "drug_embedding_path"), expected_drug), "unexpected drug_embedding_path")
    check(bool(first_test_result(manifest)), "missing test_results[0]")
    return errors


def collect_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    directory = task_dir(args)
    for policy in POLICIES:
        exp_name = f"{args.prefix}_f0_{policy}"
        manifest_path = args.checkpoint_root / exp_name / "run_manifest.json"
        expected_artifact = artifact_path_for(directory, policy)
        artifact_payload = artifact_meta(expected_artifact)
        base_record = {
            "policy": policy,
            "role": role_for(policy),
            "experiment_name": exp_name,
            "manifest_path": str(manifest_path),
            "artifact_path": str(expected_artifact) if expected_artifact is not None else None,
            "artifact_meta": artifact_payload,
        }
        if expected_artifact is not None and not expected_artifact.exists():
            errors.append(f"missing artifact: {expected_artifact}")
        if not manifest_path.exists():
            errors.append(f"missing manifest: {manifest_path}")
            records.append({**base_record, "status": "missing"})
            continue

        manifest = load_json(manifest_path)
        validation_errors = validate_manifest(
            manifest=manifest,
            manifest_path=manifest_path,
            policy=policy,
            expected_artifact=expected_artifact,
            args=args,
        )
        errors.extend(validation_errors)
        records.append(
            {
                **base_record,
                "status": "ok" if not validation_errors else "invalid",
                "run_status": manifest.get("run_status"),
                "test_status": manifest.get("test_status"),
                "have_mse_loss": config_value(manifest, "have_mse_loss"),
                "control_expression_mode": config_value(manifest, "control_expression_mode"),
                "random_control_expression_path": config_value(manifest, "random_control_expression_path"),
                "protein_concat_mode": config_value(manifest, "protein_concat_mode"),
                "graph_feature_mode": config_value(manifest, "graph_feature_mode"),
                "target_protein_max_length": config_value(manifest, "target_protein_max_length"),
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


def compute_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {row["policy"]: row for row in records if row.get("status") == "ok"}
    reference = indexed.get(REFERENCE_POLICY)
    reference_auprc = metric_value(reference, "auprc")
    candidates = []
    for policy in RANDOM_POLICIES:
        row = indexed.get(policy)
        candidate = {
            "policy": policy,
            "artifact_path": row.get("artifact_path") if row else None,
            "auprc": metric_value(row, "auprc"),
            "nauprc": metric_value(row, "nauprc"),
            "auroc": metric_value(row, "auroc"),
            "drop_vs_real_auprc": None,
        }
        if reference_auprc is not None and candidate["auprc"] is not None:
            candidate["drop_vs_real_auprc"] = reference_auprc - candidate["auprc"]
        candidates.append(candidate)

    eligible = [row for row in candidates if row.get("auprc") is not None]
    winner = None
    if eligible:
        winner = min(
            eligible,
            key=lambda row: (
                float(row["auprc"]),
                float(row["auroc"]) if row.get("auroc") is not None else math.inf,
                row["policy"],
            ),
        )
    drop = None if winner is None else finite_float(winner.get("drop_vs_real_auprc"))
    return {
        "reference_policy": REFERENCE_POLICY,
        "reference_auprc": reference_auprc,
        "candidate_policies": candidates,
        "winner": winner,
        "winner_drop_vs_real_auprc": drop,
        "no_obvious_collapse": bool(drop is None or drop < 0.02),
        "zero_control_excluded_from_selection": True,
    }


def markdown_metrics(records: list[dict[str, Any]], selection: dict[str, Any]) -> list[str]:
    reference_auprc = finite_float(selection.get("reference_auprc"))
    lines = [
        "| role | policy | status | AUPRC | drop vs real | nAUPRC | AUROC | ACC | count | artifact |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in records:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        auprc = finite_float(metrics.get("auprc"))
        drop = reference_auprc - auprc if reference_auprc is not None and auprc is not None else None
        artifact = row.get("artifact_path") or ""
        lines.append(
            "| {role} | {policy} | {status} | {auprc} | {drop} | {nauprc} | {auroc} | {acc} | {count} | `{artifact}` |".format(
                role=row.get("role", ""),
                policy=row.get("policy", ""),
                status=row.get("status", ""),
                auprc=fmt(metrics.get("auprc")),
                drop=fmt(drop),
                nauprc=fmt(metrics.get("nauprc")),
                auroc=fmt(metrics.get("auroc")),
                acc=fmt(metrics.get("acc")),
                count=fmt(metrics.get("count"), precision=0),
                artifact=artifact,
            )
        )
    return lines


def write_markdown(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    selection: dict[str, Any],
    errors: list[str],
) -> None:
    winner = selection.get("winner") if isinstance(selection.get("winner"), dict) else None
    lines = [
        "# exp_04_v2 Random Expression Fold0 Screen",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Prefix: `{args.prefix}`",
        f"- Task: `{DEFAULT_TASK_NAME}` / `response`",
        f"- Split: `{DEFAULT_SPLIT_STRATEGY}`",
        f"- Checkpoint root: `{args.checkpoint_root}`",
        "",
        "## Validation",
    ]
    if errors:
        lines.append(f"- Validation errors: `{len(errors)}`")
        lines.extend(f"- {error}" for error in errors[:50])
        if len(errors) > 50:
            lines.append(f"- ... {len(errors) - 50} more")
    else:
        lines.append("- All expected manifests are present and match the screen settings.")

    lines.extend(["", "## Decision"])
    if winner is None:
        lines.append("- Winner: unavailable because no random candidate has complete metrics.")
    else:
        lines.append(f"- Winner: `{winner['policy']}`")
        lines.append(f"- Winner artifact: `{winner.get('artifact_path')}`")
        lines.append(f"- AUPRC drop vs real-control no-MSE fold0: `{fmt(winner.get('drop_vs_real_auprc'))}`")
        if selection.get("no_obvious_collapse"):
            lines.append("- Interpretation: no random policy produced an obvious collapse by the 0.02 AUPRC-drop rule.")
        else:
            lines.append("- Interpretation: the selected policy produced the largest observed fold0 degradation.")
        lines.append("- `zero_control` is diagnostic only and was excluded from winner selection.")

    lines.extend(["", "## Metrics"])
    lines.extend(markdown_metrics(records, selection))
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--training-ready-root", type=Path, default=Path("data/training_ready"))
    parser.add_argument("--task-dir", type=Path, default=None)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/2026-07-08_exp04_v2_random_expression_maxdrop_screen.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/2026-07/2026-07-08/20260708_exp04_v2_random_expression_screen_summary.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, errors = collect_records(args)
    selection = compute_selection(records)
    payload = {
        "generated_at": iso_now(),
        "prefix": args.prefix,
        "task_name": DEFAULT_TASK_NAME,
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "records": records,
        "selection": selection,
        "validation_errors": errors,
    }
    dump_json(args.json_output, payload)
    write_markdown(args, records, selection, errors)
    print(f"[report] wrote {args.markdown_output}")
    print(f"[report] wrote {args.json_output}")
    if errors and not args.allow_incomplete:
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
