#!/usr/bin/env python3
"""Summarize exp01/exp03 fold0 feature-attribution manifests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PREFIX = "20260708_feature_attr"
DEFAULT_TASK_NAME = "ptv3_main_singledrug"
DEFAULT_COVARIATES = ("machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time")
METRICS = {
    "auprc": "test/task_auprc",
    "auprc_baseline": "test/task_auprc_baseline",
    "nauprc": "test/task_nauprc",
    "auroc": "test/task_auroc",
    "acc": "test/task_acc",
    "count": "test/task_count",
}
PANELS = {
    "exp01": {
        "label": "exp01 fold0 unseen-drug",
        "split_strategy": "pert_stratified_5fold_fold0",
    },
    "exp03": {
        "label": "exp03 fold0 unseen-cell",
        "split_strategy": "cell_5fold_fold0",
    },
}
LEAVE_ONE_OUT = (
    ("graph", "graph_zero"),
    ("morgan", "morgan_zero"),
    ("target", "target_zero"),
    ("covariate", "cov_none"),
    ("control_pcep_total", "no_control_pcep"),
    ("pcep", "pcep_off"),
)
ONLY_FEATURES = (
    ("morgan", "morgan_only"),
    ("graph", "graph_only"),
    ("target", "target_only"),
    ("covariate", "cov_only"),
    ("control_pcep", "control_only"),
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    have_mse_loss: bool = False
    graph_feature_mode: str = "real"
    target_protein_max_length: int = 32
    protein_concat_mode: str = "pcep"
    batch_cov_list: tuple[str, ...] = DEFAULT_COVARIATES
    control_expression_mode: str = "real"
    control_artifact: str | None = None
    drug_artifact: str = "real"


VARIANTS = (
    VariantSpec("full_mse", have_mse_loss=True),
    VariantSpec("full_nomse"),
    VariantSpec("random_control", control_expression_mode="random_saved", control_artifact="random"),
    VariantSpec(
        "no_control_pcep",
        protein_concat_mode="off",
        control_expression_mode="random_saved",
        control_artifact="zero",
    ),
    VariantSpec("pcep_off", protein_concat_mode="off"),
    VariantSpec("cov_none", batch_cov_list=()),
    VariantSpec("graph_zero", graph_feature_mode="zero"),
    VariantSpec("target_zero", target_protein_max_length=0),
    VariantSpec("morgan_zero", drug_artifact="zero"),
    VariantSpec(
        "morgan_only",
        graph_feature_mode="zero",
        target_protein_max_length=0,
        protein_concat_mode="off",
        batch_cov_list=(),
        control_expression_mode="random_saved",
        control_artifact="zero",
    ),
    VariantSpec(
        "graph_only",
        target_protein_max_length=0,
        protein_concat_mode="off",
        batch_cov_list=(),
        control_expression_mode="random_saved",
        control_artifact="zero",
        drug_artifact="zero",
    ),
    VariantSpec(
        "target_only",
        graph_feature_mode="zero",
        protein_concat_mode="off",
        batch_cov_list=(),
        control_expression_mode="random_saved",
        control_artifact="zero",
        drug_artifact="zero",
    ),
    VariantSpec(
        "cov_only",
        graph_feature_mode="zero",
        target_protein_max_length=0,
        protein_concat_mode="off",
        control_expression_mode="random_saved",
        control_artifact="zero",
        drug_artifact="zero",
    ),
    VariantSpec(
        "control_only",
        graph_feature_mode="zero",
        target_protein_max_length=0,
        batch_cov_list=(),
        drug_artifact="zero",
    ),
)
VARIANT_BY_NAME = {spec.name: spec for spec in VARIANTS}


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


def expected_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.training_ready_root
    return {
        "real_drug": root / "ptv3/derived/drug_embedding_morgan_2048.pkl",
        "zero_drug": args.zero_drug_embedding_path,
        "random_control": args.random_control_expression_path,
        "zero_control": args.zero_control_expression_path,
    }


def validate_manifest(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    panel: str,
    spec: VariantSpec,
    paths: dict[str, Path],
) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{manifest_path}: {message}")

    check(manifest.get("run_status") == "fit_completed", f"run_status={manifest.get('run_status')!r}")
    check(manifest.get("test_status") == "test_completed", f"test_status={manifest.get('test_status')!r}")
    check(manifest.get("task_name") == DEFAULT_TASK_NAME, f"task_name={manifest.get('task_name')!r}")
    check(manifest.get("split_strategy") == PANELS[panel]["split_strategy"], f"split_strategy={manifest.get('split_strategy')!r}")
    check(manifest.get("task_head") == "response", f"task_head={manifest.get('task_head')!r}")
    check(config_value(manifest, "have_mse_loss") is spec.have_mse_loss, "unexpected have_mse_loss")
    check(config_value(manifest, "graph_feature_mode") == spec.graph_feature_mode, "unexpected graph_feature_mode")
    check(
        int(config_value(manifest, "target_protein_max_length") or 0) == spec.target_protein_max_length,
        "unexpected target_protein_max_length",
    )
    check(config_value(manifest, "protein_concat_mode") == spec.protein_concat_mode, "unexpected protein_concat_mode")
    check(tuple(config_value(manifest, "batch_cov_list") or ()) == spec.batch_cov_list, "unexpected batch_cov_list")
    check(
        config_value(manifest, "control_expression_mode") == spec.control_expression_mode,
        "unexpected control_expression_mode",
    )

    control_path = config_value(manifest, "random_control_expression_path")
    if spec.control_artifact == "random":
        check(same_path(control_path, paths["random_control"]), "unexpected random control artifact")
    elif spec.control_artifact == "zero":
        check(same_path(control_path, paths["zero_control"]), "unexpected zero control artifact")
    else:
        check(control_path in {None, ""}, f"unexpected control artifact path {control_path!r}")

    drug_path = config_value(manifest, "drug_embedding_path")
    expected_drug = paths["zero_drug"] if spec.drug_artifact == "zero" else paths["real_drug"]
    check(same_path(drug_path, expected_drug), "unexpected drug_embedding_path")
    check(bool(first_test_result(manifest)), "missing test_results[0]")
    return errors


def collect_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    paths = expected_paths(args)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for panel in PANELS:
        for spec in VARIANTS:
            exp_name = f"{args.prefix}_{panel}_f0_{spec.name}"
            manifest_path = args.checkpoint_root / exp_name / "run_manifest.json"
            if not manifest_path.exists():
                errors.append(f"missing manifest: {manifest_path}")
                records.append(
                    {
                        "panel": panel,
                        "variant": spec.name,
                        "experiment_name": exp_name,
                        "manifest_path": str(manifest_path),
                        "status": "missing",
                    }
                )
                continue
            manifest = load_json(manifest_path)
            validation_errors = validate_manifest(
                manifest=manifest,
                manifest_path=manifest_path,
                panel=panel,
                spec=spec,
                paths=paths,
            )
            errors.extend(validation_errors)
            records.append(
                {
                    "panel": panel,
                    "panel_label": PANELS[panel]["label"],
                    "variant": spec.name,
                    "experiment_name": exp_name,
                    "manifest_path": str(manifest_path),
                    "status": "ok" if not validation_errors else "invalid",
                    "run_status": manifest.get("run_status"),
                    "test_status": manifest.get("test_status"),
                    "have_mse_loss": config_value(manifest, "have_mse_loss"),
                    "graph_feature_mode": config_value(manifest, "graph_feature_mode"),
                    "target_protein_max_length": config_value(manifest, "target_protein_max_length"),
                    "protein_concat_mode": config_value(manifest, "protein_concat_mode"),
                    "batch_cov_list": config_value(manifest, "batch_cov_list"),
                    "control_expression_mode": config_value(manifest, "control_expression_mode"),
                    "random_control_expression_path": config_value(manifest, "random_control_expression_path"),
                    "drug_embedding_path": config_value(manifest, "drug_embedding_path"),
                    "metrics": extract_metrics(manifest),
                }
            )
    return records, errors


def by_panel_variant(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["panel"], row["variant"]): row for row in records if row.get("status") == "ok"}


def metric_value(row: dict[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return finite_float(metrics.get(metric))


def compute_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = by_panel_variant(records)
    analysis: dict[str, Any] = {}
    for panel in PANELS:
        full = indexed.get((panel, "full_nomse"))
        panel_analysis: dict[str, Any] = {"leave_one_out": [], "only_feature": [], "summary": {}}
        for feature, variant in LEAVE_ONE_OUT:
            ablated = indexed.get((panel, variant))
            drop_row = {"feature": feature, "variant": variant}
            for metric in ("auprc", "nauprc", "auroc"):
                baseline = metric_value(full, metric)
                ablated_value = metric_value(ablated, metric)
                drop_row[f"full_nomse_{metric}"] = baseline
                drop_row[f"{variant}_{metric}"] = ablated_value
                drop_row[f"{metric}_drop"] = (
                    baseline - ablated_value if baseline is not None and ablated_value is not None else None
                )
            panel_analysis["leave_one_out"].append(drop_row)

        for feature, variant in ONLY_FEATURES:
            row = indexed.get((panel, variant))
            panel_analysis["only_feature"].append(
                {
                    "feature": feature,
                    "variant": variant,
                    "auprc": metric_value(row, "auprc"),
                    "nauprc": metric_value(row, "nauprc"),
                    "auroc": metric_value(row, "auroc"),
                    "count": metric_value(row, "count"),
                }
            )

        loo_candidates = [
            row
            for row in panel_analysis["leave_one_out"]
            if finite_float(row.get("auprc_drop")) is not None
        ]
        only_candidates = [
            row
            for row in panel_analysis["only_feature"]
            if finite_float(row.get("auprc")) is not None
        ]
        if loo_candidates:
            panel_analysis["summary"]["largest_auprc_drop"] = max(
                loo_candidates,
                key=lambda row: float(row["auprc_drop"]),
            )
        if only_candidates:
            panel_analysis["summary"]["strongest_only_feature"] = max(
                only_candidates,
                key=lambda row: float(row["auprc"]),
            )
        analysis[panel] = panel_analysis
    return analysis


def markdown_metrics(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| panel | variant | status | MSE | AUPRC | baseline | nAUPRC | AUROC | ACC | count |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        lines.append(
            "| {panel} | {variant} | {status} | {mse} | {auprc} | {baseline} | {nauprc} | {auroc} | {acc} | {count} |".format(
                panel=row["panel"],
                variant=row["variant"],
                status=row.get("status", ""),
                mse=str(row.get("have_mse_loss", "")),
                auprc=fmt(metrics.get("auprc")),
                baseline=fmt(metrics.get("auprc_baseline")),
                nauprc=fmt(metrics.get("nauprc")),
                auroc=fmt(metrics.get("auroc")),
                acc=fmt(metrics.get("acc")),
                count=fmt(metrics.get("count"), precision=0),
            )
        )
    return lines


def markdown_analysis(analysis: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for panel, panel_analysis in analysis.items():
        lines.append(f"## {PANELS[panel]['label']}")
        summary = panel_analysis.get("summary", {})
        largest = summary.get("largest_auprc_drop")
        strongest = summary.get("strongest_only_feature")
        if largest:
            lines.append(
                "- Largest full-model AUPRC drop: "
                f"`{largest['feature']}` via `{largest['variant']}` ({fmt(largest.get('auprc_drop'))})."
            )
        if strongest:
            lines.append(
                "- Strongest only-feature AUPRC: "
                f"`{strongest['feature']}` via `{strongest['variant']}` ({fmt(strongest.get('auprc'))})."
            )
        lines.extend(
            [
                "",
                "| feature | ablation variant | full AUPRC | ablated AUPRC | AUPRC drop | nAUPRC drop | AUROC drop |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in panel_analysis["leave_one_out"]:
            variant = row["variant"]
            lines.append(
                "| {feature} | {variant} | {full} | {ablated} | {auprc_drop} | {nauprc_drop} | {auroc_drop} |".format(
                    feature=row["feature"],
                    variant=variant,
                    full=fmt(row.get("full_nomse_auprc")),
                    ablated=fmt(row.get(f"{variant}_auprc")),
                    auprc_drop=fmt(row.get("auprc_drop")),
                    nauprc_drop=fmt(row.get("nauprc_drop")),
                    auroc_drop=fmt(row.get("auroc_drop")),
                )
            )
        lines.extend(
            [
                "",
                "| feature | only-feature variant | AUPRC | nAUPRC | AUROC | count |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in panel_analysis["only_feature"]:
            lines.append(
                "| {feature} | {variant} | {auprc} | {nauprc} | {auroc} | {count} |".format(
                    feature=row["feature"],
                    variant=row["variant"],
                    auprc=fmt(row.get("auprc")),
                    nauprc=fmt(row.get("nauprc")),
                    auroc=fmt(row.get("auroc")),
                    count=fmt(row.get("count"), precision=0),
                )
            )
        lines.append("")
    return lines


def write_markdown(args: argparse.Namespace, records: list[dict[str, Any]], analysis: dict[str, Any], errors: list[str]) -> None:
    lines = [
        "# exp01/exp03 Fold0 Feature Attribution Results",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Prefix: `{args.prefix}`",
        f"- Task: `{DEFAULT_TASK_NAME}` / `response`",
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
        lines.append("- All expected manifests are present and match the variant matrix.")
    lines.extend(["", "## Metrics"])
    lines.extend(markdown_metrics(records))
    lines.extend(["", "## Attribution"])
    lines.extend(markdown_analysis(analysis))
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--training-ready-root", type=Path, default=Path("data/training_ready"))
    parser.add_argument(
        "--random-control-expression-path",
        type=Path,
        default=Path("data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_seed42.npy"),
    )
    parser.add_argument(
        "--zero-control-expression-path",
        type=Path,
        default=Path("data/training_ready/ptv3/tasks/ptv3_main_singledrug/zero_control_expression.npy"),
    )
    parser.add_argument(
        "--zero-drug-embedding-path",
        type=Path,
        default=Path("data/training_ready/ptv3/derived/drug_embedding_morgan_2048_zero.pkl"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/2026-07-08_feature_attribution_exp01_exp03_fold0_results.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/2026-07/2026-07-08/20260708_feature_attr_fold0_summary.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, errors = collect_records(args)
    analysis = compute_analysis(records)
    payload = {
        "generated_at": iso_now(),
        "prefix": args.prefix,
        "records": records,
        "analysis": analysis,
        "validation_errors": errors,
    }
    if errors and not args.allow_incomplete:
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        return 1
    dump_json(args.json_output, payload)
    write_markdown(args, records, analysis, errors)
    print(f"[report] wrote {args.markdown_output}")
    print(f"[report] wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
