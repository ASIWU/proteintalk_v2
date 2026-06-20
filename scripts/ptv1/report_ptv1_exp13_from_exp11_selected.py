#!/usr/bin/env python3
"""Report the official PTV1 exp_13 rerun pinned to the exp_11 best config."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


OFFICIAL_CANDIDATE = "mse050_drop010"
DEFAULT_SOURCE_EXP11 = "20260608_ptv1_cell_llm_tune_v1_mse050_drop010_ptv1_random_split"
DEFAULT_OUTPUT_PREFIX = "20260608_ptv1_cell_llm_exp13_from_exp11_graphon_v1_mse050_drop010"
EXP13_TASK_DIR = "ptv1_extra_singledrug"
EPOCH_RE = re.compile(r"epoch=(\d+)")

EXPECTED_SOURCE_CONFIG: dict[str, Any] = {
    "learning_rate": 2e-4,
    "batch_size": 256,
    "dropout": 0.10,
    "weight_decay": 1e-4,
    "mse_weight": 0.50,
    "mse_target_mode": "all",
    "graph_feature_mode": "real",
    "graph_structural_rp": True,
    "graph_drug_concat": True,
    "graph_logit_scale": 2.0,
    "cell_llm_mode": "frozen",
}

KEY_PARAM_COMPARE = tuple(EXPECTED_SOURCE_CONFIG)
BOOL_KEYS = {"graph_structural_rp", "graph_drug_concat"}
INT_KEYS = {"batch_size", "max_epochs"}
FLOAT_KEYS = {"learning_rate", "dropout", "weight_decay", "mse_weight", "graph_logit_scale"}


@dataclass
class Audit:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-exp11-name", default=DEFAULT_SOURCE_EXP11)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--expected-extra-rows", type=int, default=218)
    parser.add_argument("--format", choices=("markdown", "tsv", "json"), default="markdown")
    parser.add_argument("--precision", type=int, default=6)
    return parser.parse_args()


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
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt(value: Any, precision: int = 6) -> str:
    number = finite(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def manifest_value(manifest: dict[str, Any], key: str) -> Any:
    if key in manifest:
        return manifest[key]
    args = manifest.get("args")
    if isinstance(args, dict):
        return args.get(key)
    return None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return None


def normalize_config(key: str, value: Any) -> Any:
    if key in BOOL_KEYS:
        return as_bool(value)
    if key in INT_KEYS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if key in FLOAT_KEYS:
        number = finite(value)
        return number
    if value is None:
        return None
    return str(value)


def same_config(key: str, actual: Any, expected: Any) -> bool:
    actual_norm = normalize_config(key, actual)
    expected_norm = normalize_config(key, expected)
    if isinstance(expected_norm, float):
        return actual_norm is not None and math.isclose(float(actual_norm), expected_norm, rel_tol=1e-9, abs_tol=1e-9)
    return actual_norm == expected_norm


def path_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def canonical_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve(strict=False))


def same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    return canonical_path(left) == canonical_path(right)


def parse_epoch_from_checkpoint(path: Any) -> int | None:
    if not path:
        return None
    match = EPOCH_RE.search(str(path))
    return int(match.group(1)) if match else None


def source_checkpoint(manifest: dict[str, Any]) -> str | None:
    checkpoint = manifest.get("best_model_path") or manifest.get("test_checkpoint_path")
    return str(checkpoint) if checkpoint else None


def source_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    results = manifest.get("test_results") or []
    if not results or not isinstance(results[0], dict):
        return {"auroc": None, "auprc": None, "nauprc": None, "rows": None}
    row = results[0]
    return {
        "auroc": finite(row.get("test/task_auroc")),
        "auprc": finite(row.get("test/task_auprc")),
        "nauprc": finite(row.get("test/task_nauprc")),
        "rows": finite(row.get("test/task_count")),
    }


def extra_metrics(metrics_path: Path) -> dict[str, Any]:
    payload = load_json(metrics_path)
    task = payload.get("task")
    if not isinstance(task, dict):
        raise ValueError(f"missing task metrics: {metrics_path}")
    return {
        "auroc": finite(task.get("auroc")),
        "auprc": finite(task.get("auprc")),
        "nauprc": finite(task.get("nauprc")),
        "rows": finite(task.get("count")),
    }


def parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as first_error:
        try:
            import pandas as pd  # type: ignore[import-not-found]

            return int(len(pd.read_parquet(path)))
        except Exception as second_error:
            raise RuntimeError(
                f"could not read prediction rows from {path}: {first_error}; {second_error}"
            ) from second_error


def require_file(path: Path, audit: Audit, label: str) -> bool:
    exists = path.exists()
    audit.check(exists, f"missing {label}: {path}")
    return exists


def load_extra_record(args: argparse.Namespace, mode: str, audit: Audit) -> dict[str, Any]:
    suffix = {
        "direct": "_extra_direct_from_exp11",
        "all_train": "_all_ptv1_for_extra_from_exp11",
    }[mode]
    exp_name = f"{args.output_prefix}{suffix}"
    task_dir = args.output_root / exp_name / EXP13_TASK_DIR
    manifest_path = task_dir / "run_manifest.json"
    metrics_path = task_dir / "metrics.json"
    predictions_path = task_dir / "predictions.parquet"
    record: dict[str, Any] = {
        "setting": mode,
        "exp_name": exp_name,
        "task_dir": str(task_dir),
        "run_manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }
    if not (
        require_file(manifest_path, audit, f"{mode} inference manifest")
        and require_file(metrics_path, audit, f"{mode} metrics")
        and require_file(predictions_path, audit, f"{mode} predictions")
    ):
        return record

    manifest = load_json(manifest_path)
    metrics = extra_metrics(metrics_path)
    parquet_rows = parquet_row_count(predictions_path)
    record.update(
        {
            "manifest": manifest,
            "checkpoint_path": manifest.get("checkpoint_path"),
            "checkpoint_run_manifest_path": manifest.get("checkpoint_run_manifest_path"),
            "n_predictions": finite(manifest.get("n_predictions")),
            "prediction_rows": parquet_rows,
            **metrics,
        }
    )
    return record


def validate_graph_on(manifest: dict[str, Any], audit: Audit, label: str) -> None:
    graph_mode = manifest_value(manifest, "graph_feature_mode")
    audit.check(graph_mode == "real", f"{label} must use graph_feature_mode=real, found {graph_mode!r}")
    audit.check(graph_mode != "off", f"{label} has forbidden GRAPH_FEATURE_MODE=off")


def validate_source(args: argparse.Namespace, source_manifest: dict[str, Any], source_manifest_path: Path, audit: Audit) -> dict[str, Any]:
    audit.check(args.source_exp11_name.endswith(f"{OFFICIAL_CANDIDATE}_ptv1_random_split"), "source exp_11 name is not the official mse050_drop010 run")
    audit.check("mse025_graph_off" not in args.source_exp11_name, "source exp_11 name contains forbidden graph-off candidate")
    audit.check(source_manifest.get("dataset_group") == "ptv1", "source exp_11 dataset_group must be ptv1")
    audit.check(source_manifest.get("task_name") == "ptv1_aivc", "source exp_11 task_name must be ptv1_aivc")
    audit.check(source_manifest.get("split_strategy") == "fixed_experiment_type", "source exp_11 split_strategy must be fixed_experiment_type")
    audit.check(source_manifest.get("run_status") == "fit_completed", "source exp_11 run_status must be fit_completed")
    validate_graph_on(source_manifest, audit, "source exp_11")

    for key, expected in EXPECTED_SOURCE_CONFIG.items():
        actual = manifest_value(source_manifest, key)
        audit.check(same_config(key, actual, expected), f"source exp_11 {key} expected {expected!r}, found {actual!r}")

    checkpoint = source_checkpoint(source_manifest)
    audit.check(bool(checkpoint), "source exp_11 missing best_model_path/test_checkpoint_path")
    if checkpoint:
        audit.check(Path(checkpoint).exists(), f"source exp_11 checkpoint does not exist: {checkpoint}")
    selected_epoch = parse_epoch_from_checkpoint(checkpoint)
    audit.check(selected_epoch is not None, f"could not parse source selected epoch from checkpoint: {checkpoint}")

    record = {
        "setting": "exp_11_source",
        "exp_name": args.source_exp11_name,
        "run_manifest_path": str(source_manifest_path),
        "checkpoint_path": checkpoint,
        "selected_epoch": selected_epoch,
        **source_metrics(source_manifest),
    }
    return record


def validate_extra_record(record: dict[str, Any], expected_rows: int, audit: Audit) -> None:
    mode = record["setting"]
    manifest = record.get("manifest")
    if not isinstance(manifest, dict):
        return
    audit.check("mse025_graph_off" not in record["exp_name"], f"{mode} exp_name contains forbidden graph-off candidate")
    audit.check(manifest.get("dataset_group") == "ptv1", f"{mode} inference dataset_group must be ptv1")
    audit.check(manifest.get("task_name") == EXP13_TASK_DIR, f"{mode} inference task_name must be {EXP13_TASK_DIR}")
    audit.check(manifest.get("split_strategy") == "test_only", f"{mode} inference split_strategy must be test_only")
    validate_graph_on(manifest, audit, f"{mode} inference")
    audit.check(int(record.get("n_predictions") or -1) == expected_rows, f"{mode} n_predictions must be {expected_rows}, found {record.get('n_predictions')!r}")
    audit.check(int(record.get("rows") or -1) == expected_rows, f"{mode} metrics count must be {expected_rows}, found {record.get('rows')!r}")
    audit.check(int(record.get("prediction_rows") or -1) == expected_rows, f"{mode} predictions rows must be {expected_rows}, found {record.get('prediction_rows')!r}")


def validate_direct(direct: dict[str, Any], source_manifest_path: Path, source_record: dict[str, Any], audit: Audit) -> None:
    validate_extra_record(direct, int(direct.get("expected_rows", 218)), audit)
    manifest = direct.get("manifest")
    if not isinstance(manifest, dict):
        return
    audit.check(
        same_path(manifest.get("checkpoint_path"), source_record.get("checkpoint_path")),
        "direct inference checkpoint_path must equal the source exp_11 best checkpoint",
    )
    audit.check(
        same_path(manifest.get("checkpoint_run_manifest_path"), source_manifest_path),
        "direct inference checkpoint_run_manifest_path must reference the source exp_11 manifest",
    )


def validate_all_train(
    args: argparse.Namespace,
    all_train: dict[str, Any],
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    source_record: dict[str, Any],
    audit: Audit,
) -> dict[str, Any]:
    validate_extra_record(all_train, args.expected_extra_rows, audit)
    exp_name = f"{args.output_prefix}_all_ptv1_for_extra_from_exp11"
    train_manifest_path = args.checkpoint_root / exp_name / "run_manifest.json"
    train_record: dict[str, Any] = {
        "exp_name": exp_name,
        "run_manifest_path": str(train_manifest_path),
        "selected_checkpoint_path": str((args.checkpoint_root / exp_name / "last.ckpt").resolve(strict=False)),
    }
    if not require_file(train_manifest_path, audit, "all_train checkpoint manifest"):
        return train_record

    train_manifest = load_json(train_manifest_path)
    train_record["manifest"] = train_manifest
    train_record["max_epochs"] = normalize_config("max_epochs", manifest_value(train_manifest, "max_epochs"))
    train_record["checkpoint_path"] = str((args.checkpoint_root / exp_name / "last.ckpt").resolve(strict=False))
    audit.check(train_manifest.get("dataset_group") == "ptv1", "all_train training dataset_group must be ptv1")
    audit.check(train_manifest.get("task_name") == "ptv1_aivc", "all_train training task_name must be ptv1_aivc")
    audit.check(train_manifest.get("split_strategy") == "all_train_subset_test", "all_train training split_strategy must be all_train_subset_test")
    audit.check(train_manifest.get("run_status") == "fit_completed", "all_train training run_status must be fit_completed")
    validate_graph_on(train_manifest, audit, "all_train training")

    for key in KEY_PARAM_COMPARE:
        source_value = manifest_value(source_manifest, key)
        train_value = manifest_value(train_manifest, key)
        audit.check(
            same_config(key, train_value, source_value),
            f"all_train {key} must match source exp_11: source={source_value!r}, all_train={train_value!r}",
        )

    selected_epoch = source_record.get("selected_epoch")
    expected_max_epochs = int(selected_epoch) + 1 if selected_epoch is not None else None
    policy = train_manifest.get("exp13_from_exp11_policy")
    audit.check(isinstance(policy, dict), "all_train manifest missing exp13_from_exp11_policy")
    if isinstance(policy, dict):
        audit.check(policy.get("enabled") is True, "all_train exp13_from_exp11_policy.enabled must be true")
        audit.check(policy.get("selected_epoch") == selected_epoch, f"all_train policy selected_epoch must be {selected_epoch}")
        audit.check(policy.get("applied_max_epochs") == expected_max_epochs, f"all_train policy applied_max_epochs must be {expected_max_epochs}")
        audit.check(
            same_path(policy.get("reference_manifest_path"), source_manifest_path),
            "all_train policy reference_manifest_path must reference source exp_11 manifest",
        )
        audit.check(
            same_path(policy.get("reference_checkpoint_path"), source_record.get("checkpoint_path")),
            "all_train policy reference_checkpoint_path must reference source exp_11 checkpoint",
        )
        audit.check(
            same_path(policy.get("selected_checkpoint_path"), args.checkpoint_root / exp_name / "last.ckpt"),
            "all_train policy selected_checkpoint_path must be last.ckpt",
        )
    audit.check(train_record.get("max_epochs") == expected_max_epochs, f"all_train max_epochs must be {expected_max_epochs}")
    audit.check((args.checkpoint_root / exp_name / "last.ckpt").exists(), "all_train last.ckpt is missing")

    infer_manifest = all_train.get("manifest")
    if isinstance(infer_manifest, dict):
        audit.check(
            same_path(infer_manifest.get("checkpoint_run_manifest_path"), train_manifest_path),
            "all_train inference must reference the all_train checkpoint manifest",
        )
        audit.check(
            same_path(infer_manifest.get("checkpoint_path"), args.checkpoint_root / exp_name / "last.ckpt"),
            "all_train inference checkpoint_path must be last.ckpt",
        )

    return train_record


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], Audit]:
    audit = Audit()
    source_manifest_path = args.checkpoint_root / args.source_exp11_name / "run_manifest.json"
    require_file(source_manifest_path, audit, "source exp_11 manifest")
    if not source_manifest_path.exists():
        return {"args": vars(args), "rows": []}, audit

    source_manifest = load_json(source_manifest_path)
    source_record = validate_source(args, source_manifest, source_manifest_path, audit)

    direct = load_extra_record(args, "direct", audit)
    direct["expected_rows"] = args.expected_extra_rows
    all_train = load_extra_record(args, "all_train", audit)
    all_train["expected_rows"] = args.expected_extra_rows

    validate_direct(direct, source_manifest_path, source_record, audit)
    all_train_checkpoint = validate_all_train(args, all_train, source_manifest, source_manifest_path, source_record, audit)

    generated_at = datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()
    rows = [
        source_record,
        {
            "setting": "exp_13_direct_from_exp11",
            "exp_name": direct.get("exp_name"),
            "auroc": direct.get("auroc"),
            "auprc": direct.get("auprc"),
            "nauprc": direct.get("nauprc"),
            "rows": direct.get("rows"),
            "n_predictions": direct.get("n_predictions"),
            "prediction_rows": direct.get("prediction_rows"),
            "checkpoint_path": direct.get("checkpoint_path"),
            "checkpoint_run_manifest_path": direct.get("checkpoint_run_manifest_path"),
            "run_manifest_path": direct.get("run_manifest_path"),
        },
        {
            "setting": "exp_13_all_train_from_exp11",
            "exp_name": all_train.get("exp_name"),
            "auroc": all_train.get("auroc"),
            "auprc": all_train.get("auprc"),
            "nauprc": all_train.get("nauprc"),
            "rows": all_train.get("rows"),
            "n_predictions": all_train.get("n_predictions"),
            "prediction_rows": all_train.get("prediction_rows"),
            "selected_epoch": source_record.get("selected_epoch"),
            "applied_max_epochs": (source_record.get("selected_epoch") + 1) if source_record.get("selected_epoch") is not None else None,
            "checkpoint_path": all_train.get("checkpoint_path"),
            "checkpoint_run_manifest_path": all_train.get("checkpoint_run_manifest_path"),
            "run_manifest_path": all_train.get("run_manifest_path"),
            "training_manifest_path": all_train_checkpoint.get("run_manifest_path"),
        },
    ]
    return {
        "generated_at_hkt": generated_at,
        "candidate": OFFICIAL_CANDIDATE,
        "source_exp11_name": args.source_exp11_name,
        "output_prefix": args.output_prefix,
        "expected_extra_rows": args.expected_extra_rows,
        "source_manifest_path": str(source_manifest_path),
        "source_checkpoint": source_record.get("checkpoint_path"),
        "selected_epoch": source_record.get("selected_epoch"),
        "applied_max_epochs": (source_record.get("selected_epoch") + 1) if source_record.get("selected_epoch") is not None else None,
        "all_train_checkpoint_manifest_path": all_train_checkpoint.get("run_manifest_path"),
        "rows": rows,
        "audit": {"ok": audit.ok, "errors": audit.errors, "warnings": audit.warnings},
    }, audit


def markdown_table(rows: list[dict[str, Any]], precision: int) -> None:
    print("| setting | AUPRC | AUROC | nAUPRC | rows | source checkpoint | selected epoch | applied max epochs |")
    print("|---|---:|---:|---:|---:|---|---:|---:|")
    for row in rows:
        print(
            "| {setting} | {auprc} | {auroc} | {nauprc} | {rows} | `{checkpoint}` | {epoch} | {max_epochs} |".format(
                setting=row.get("setting", ""),
                auprc=fmt(row.get("auprc"), precision),
                auroc=fmt(row.get("auroc"), precision),
                nauprc=fmt(row.get("nauprc"), precision),
                rows="" if row.get("rows") is None else str(int(row.get("rows"))),
                checkpoint=path_text(row.get("checkpoint_path")),
                epoch="" if row.get("selected_epoch") is None else str(row.get("selected_epoch")),
                max_epochs="" if row.get("applied_max_epochs") is None else str(row.get("applied_max_epochs")),
            )
        )


def print_markdown(report: dict[str, Any], args: argparse.Namespace) -> None:
    print("# PTV1 exp_13 From exp_11 Graph-On Official Rerun")
    print()
    print(f"Generated: `{report['generated_at_hkt']}`")
    print()
    print("This is an exp_13-only official rerun. It is not a cross-candidate leaderboard.")
    print(f"The run is pinned to the exp_11 best candidate `{OFFICIAL_CANDIDATE}` with graph features enabled.")
    print()
    print("## Source")
    print()
    print(f"- source exp_11: `{report['source_exp11_name']}`")
    print(f"- source checkpoint: `{report['source_checkpoint']}`")
    print(f"- selected epoch: `{report['selected_epoch']}`")
    print(f"- all_train max epochs: `{report['applied_max_epochs']}`")
    print(f"- output prefix: `{report['output_prefix']}`")
    print()
    print("## Results")
    print()
    markdown_table(report["rows"], args.precision)
    print()
    print("## Audit")
    print()
    print(f"- status: `{'PASS' if report['audit']['ok'] else 'FAIL'}`")
    print(f"- expected extra rows: `{report['expected_extra_rows']}`")
    print("- graph policy: `GRAPH_FEATURE_MODE=real`, `GRAPH_STRUCTURAL_RP=1`, `GRAPH_DRUG_CONCAT=1`")
    print("- direct policy: inference checkpoint must be the source exp_11 best checkpoint")
    print("- all_train policy: train all PTV1 with exp_11 parameters for selected epoch plus one, then infer from `last.ckpt`")
    print("- graph-off candidates are excluded from official selection and are not result rows in this report")
    if report["audit"]["warnings"]:
        print()
        print("Warnings:")
        for item in report["audit"]["warnings"]:
            print(f"- {item}")
    if report["audit"]["errors"]:
        print()
        print("Errors:")
        for item in report["audit"]["errors"]:
            print(f"- {item}")
    print()
    print("## Artifacts")
    print()
    print(f"- source manifest: `{report['source_manifest_path']}`")
    print(f"- all_train checkpoint manifest: `{report['all_train_checkpoint_manifest_path']}`")
    for row in report["rows"][1:]:
        print(f"- {row['setting']} inference manifest: `{row.get('run_manifest_path')}`")


def print_tsv(report: dict[str, Any], args: argparse.Namespace) -> None:
    columns = (
        "setting",
        "exp_name",
        "auprc",
        "auroc",
        "nauprc",
        "rows",
        "n_predictions",
        "prediction_rows",
        "selected_epoch",
        "applied_max_epochs",
        "checkpoint_path",
        "checkpoint_run_manifest_path",
        "run_manifest_path",
        "training_manifest_path",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in report["rows"]:
        writer.writerow({column: row.get(column, "") for column in columns})


def main() -> int:
    args = parse_args()
    report, audit = build_report(args)
    if args.format == "markdown":
        print_markdown(report, args)
    elif args.format == "tsv":
        print_tsv(report, args)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if audit.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
