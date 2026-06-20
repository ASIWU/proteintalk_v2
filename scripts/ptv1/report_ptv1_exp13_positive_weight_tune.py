#!/usr/bin/env python3
"""Summarize PTV1 exp13 positive-weight tuning results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_GRID = (
    ("posw0p5", "0.5"),
    ("posw1", "1"),
    ("posw10", "10"),
    ("posw20", "20"),
    ("posw50", "50"),
    ("posw100", "100"),
    ("posw200", "200"),
    ("posw500", "500"),
    ("posw_negpos", "neg/pos"),
)
EXP11_SUFFIX = "_ptv1_random_split"
VALID_SUFFIX = "_extra_valid_from_exp11"
ORACLE_RE_TEMPLATE = r"^{base}_(?P<candidate>.+)_extra_oracle_epoch(?P<epoch>[0-9]+)_from_exp11$"
AUDIT_RE_TEMPLATE = r"^{base}_(?P<candidate>.+)_exp11_test_oracle_epoch(?P<epoch>[0-9]+)_from_exp11$"
EPOCH_RE = re.compile(r"epoch=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--metadata-tsv", type=Path, default=None)
    parser.add_argument("--expected-exp11-test-rows", type=int, default=799)
    parser.add_argument("--expected-extra-rows", type=int, default=218)
    parser.add_argument("--expected-candidates", type=int, default=9)
    parser.add_argument("--format", choices=("markdown", "tsv", "json"), default="markdown")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--strict", action="store_true")
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


def fmt(value: Any, precision: int) -> str:
    number = finite(value)
    return "" if number is None else f"{number:.{precision}f}"


def cell(value: Any, precision: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    number = finite(value)
    if number is not None:
        return fmt(number, precision)
    return str(value)


def parse_epoch(path: Any) -> int | None:
    if not path:
        return None
    match = EPOCH_RE.search(str(path))
    return int(match.group(1)) if match else None


def config_value(manifest: dict[str, Any], key: str) -> Any:
    if key in manifest:
        return manifest[key]
    args = manifest.get("args")
    return args.get(key) if isinstance(args, dict) else None


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def metrics_from_train_manifest(manifest: dict[str, Any]) -> dict[str, Any] | None:
    results = manifest.get("test_results") or []
    if not results or not isinstance(results[0], dict):
        return None
    row = results[0]
    count = finite(row.get("test/task_count"))
    baseline = finite(row.get("test/task_auprc_baseline"))
    pos = round(count * baseline) if count is not None and baseline is not None else None
    neg = round(count - pos) if count is not None and pos is not None else None
    return {
        "auroc": finite(row.get("test/task_auroc")),
        "auprc": finite(row.get("test/task_auprc")),
        "nauprc": finite(row.get("test/task_nauprc")),
        "count": count,
        "pos": pos,
        "neg": neg,
    }


def metrics_from_infer(metrics_path: Path) -> dict[str, Any] | None:
    metrics = load_json(metrics_path).get("task", {})
    if not isinstance(metrics, dict):
        return None
    count = finite(metrics.get("count"))
    baseline = finite(metrics.get("auprc_baseline"))
    pos = round(count * baseline) if count is not None and baseline is not None else None
    neg = round(count - pos) if count is not None and pos is not None else None
    return {
        "auroc": finite(metrics.get("auroc")),
        "auprc": finite(metrics.get("auprc")),
        "nauprc": finite(metrics.get("nauprc")),
        "count": count,
        "pos": pos,
        "neg": neg,
    }


def candidate_from_name(name: str, base_prefix: str, suffix: str) -> str | None:
    head = f"{base_prefix}_"
    if not name.startswith(head) or not name.endswith(suffix):
        return None
    candidate = name[len(head) : -len(suffix)]
    return candidate or None


def read_metadata(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    path = args.metadata_tsv or Path("logs") / f"{args.base_prefix}_positive_weight_candidates.tsv"
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                candidate = row.get("candidate")
                if not candidate:
                    continue
                rows[candidate] = {
                    "candidate": candidate,
                    "requested_positive_weight": row.get("requested_positive_weight"),
                    "resolved_positive_weight": finite(row.get("resolved_positive_weight")),
                    "neg_count": finite(row.get("neg_count")),
                    "pos_count": finite(row.get("pos_count")),
                    "neg_pos_ratio": finite(row.get("neg_pos_ratio")),
                    "metadata_source": str(path),
                }
    if rows:
        return rows
    return {
        candidate: {
            "candidate": candidate,
            "requested_positive_weight": requested,
            "resolved_positive_weight": None,
            "neg_count": None,
            "pos_count": None,
            "neg_pos_ratio": None,
            "metadata_source": None,
        }
        for candidate, requested in DEFAULT_GRID
    }


def policy_ok(manifest: dict[str, Any]) -> bool:
    cell_type_path = str(config_value(manifest, "cell_type_llm_embedding_path") or "")
    return (
        str(config_value(manifest, "dataset_group") or "") == "ptv1"
        and str(config_value(manifest, "graph_feature_mode") or "") == "real"
        and truthy(config_value(manifest, "graph_structural_rp"))
        and truthy(config_value(manifest, "graph_drug_concat"))
        and str(config_value(manifest, "cell_llm_mode") or "") == "frozen"
        and str(config_value(manifest, "cell_type_llm_mode") or "") == "frozen"
        and str(config_value(manifest, "mse_target_mode") or "") == "pdi"
        and str(config_value(manifest, "mse_weight") or "") in {"0.5", "0.50"}
        and cell_type_path.endswith("cell_type_llm_embedding_qwen3_4096_v3.npz")
    )


def collect_exp11(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(args.checkpoint_root.glob(f"{args.base_prefix}_*{EXP11_SUFFIX}/run_manifest.json")):
        candidate = candidate_from_name(manifest_path.parent.name, args.base_prefix, EXP11_SUFFIX)
        if candidate is None:
            continue
        manifest = load_json(manifest_path)
        if manifest.get("dataset_group") != "ptv1" or manifest.get("task_name") != "ptv1_aivc":
            continue
        checkpoint = manifest.get("best_model_path") or manifest.get("test_checkpoint_path")
        metrics = metrics_from_train_manifest(manifest)
        record = {
            "candidate": candidate,
            "source": str(manifest_path),
            "run_status": manifest.get("run_status"),
            "split_strategy": manifest.get("split_strategy"),
            "valid_auprc": finite(manifest.get("best_model_score")),
            "valid_monitor": manifest.get("monitor"),
            "selected_epoch": parse_epoch(checkpoint),
            "best_checkpoint": checkpoint,
            "policy_ok": policy_ok(manifest),
            "learning_rate": config_value(manifest, "learning_rate"),
            "dropout": config_value(manifest, "dropout"),
            "mse_weight": config_value(manifest, "mse_weight"),
            "mse_target_mode": config_value(manifest, "mse_target_mode"),
            "positive_weight": config_value(manifest, "positive_weight"),
            "graph_feature_mode": config_value(manifest, "graph_feature_mode"),
            "graph_structural_rp": config_value(manifest, "graph_structural_rp"),
            "graph_drug_concat": config_value(manifest, "graph_drug_concat"),
            "graph_logit_scale": config_value(manifest, "graph_logit_scale"),
            "cell_llm_mode": config_value(manifest, "cell_llm_mode"),
            "cell_llm_embedding_path": config_value(manifest, "cell_llm_embedding_path"),
            "cell_type_llm_mode": config_value(manifest, "cell_type_llm_mode"),
            "cell_type_llm_embedding_path": config_value(manifest, "cell_type_llm_embedding_path"),
            "max_epochs": config_value(manifest, "max_epochs"),
            "best_ckpt_metric": config_value(manifest, "best_ckpt_metric"),
        }
        if metrics:
            record.update({f"test_{key}": value for key, value in metrics.items()})
        records[candidate] = record
    return records


def extra_record(metrics_path: Path, mode: str, epoch: int | None, expected_rows: int) -> dict[str, Any] | None:
    metrics = metrics_from_infer(metrics_path)
    if metrics is None:
        return None
    manifest_path = metrics_path.parent / "run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    n_predictions = finite(manifest.get("n_predictions"))
    return {
        "mode": mode,
        "epoch": epoch,
        "checkpoint": manifest.get("checkpoint_path"),
        "source": str(metrics_path),
        "run_manifest": str(manifest_path) if manifest_path.exists() else None,
        "n_predictions": n_predictions,
        "rows_ok": int(n_predictions) == expected_rows if n_predictions is not None else False,
        "task_name": manifest.get("task_name"),
        "split_strategy": manifest.get("split_strategy"),
        "cell_llm_mode": manifest.get("cell_llm_mode"),
        "cell_type_llm_mode": manifest.get("cell_type_llm_mode"),
        "cell_type_llm_embedding_path": manifest.get("cell_type_llm_embedding_path"),
        "graph_feature_mode": manifest.get("graph_feature_mode"),
        **metrics,
    }


def collect_exp13(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    oracle_re = re.compile(ORACLE_RE_TEMPLATE.format(base=re.escape(args.base_prefix)))
    for metrics_path in sorted(args.output_root.glob(f"{args.base_prefix}_*/ptv1_extra_singledrug/metrics.json")):
        exp_name = metrics_path.parent.parent.name
        candidate = candidate_from_name(exp_name, args.base_prefix, VALID_SUFFIX)
        if candidate is not None:
            row = extra_record(metrics_path, "valid", None, args.expected_extra_rows)
            if row is not None:
                records.setdefault(candidate, {})["valid"] = row
            continue
        match = oracle_re.match(exp_name)
        if match:
            candidate = match.group("candidate")
            epoch = int(match.group("epoch"))
            row = extra_record(metrics_path, "oracle", epoch, args.expected_extra_rows)
            if row is not None:
                current = records.setdefault(candidate, {}).get("oracle")
                if current is None or (finite(row.get("auprc")) or -math.inf) > (finite(current.get("auprc")) or -math.inf):
                    records[candidate]["oracle"] = row
    return records


def collect_oracle_exp11_audit(args: argparse.Namespace) -> dict[str, dict[int, dict[str, Any]]]:
    records: dict[str, dict[int, dict[str, Any]]] = {}
    audit_re = re.compile(AUDIT_RE_TEMPLATE.format(base=re.escape(args.base_prefix)))
    for metrics_path in sorted(args.output_root.glob(f"{args.base_prefix}_*/ptv1_aivc/metrics.json")):
        exp_name = metrics_path.parent.parent.name
        match = audit_re.match(exp_name)
        if not match:
            continue
        candidate = match.group("candidate")
        epoch = int(match.group("epoch"))
        row = extra_record(metrics_path, "oracle_exp11_test", epoch, args.expected_exp11_test_rows)
        if row is not None:
            records.setdefault(candidate, {})[epoch] = row
    return records


def choose_best(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    valid = [row for row in rows if finite(row.get(key)) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: (finite(row.get(key)) or -math.inf, str(row.get("candidate") or "")))


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    metadata = read_metadata(args)
    exp11 = collect_exp11(args)
    exp13 = collect_exp13(args)
    audits = collect_oracle_exp11_audit(args)
    candidate_order = list(metadata)
    for candidate in sorted(set(exp11) | set(exp13) | set(audits)):
        if candidate not in metadata:
            metadata[candidate] = {"candidate": candidate}
            candidate_order.append(candidate)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for candidate in candidate_order:
        meta = metadata.get(candidate, {})
        row: dict[str, Any] = {
            "candidate": candidate,
            "requested_positive_weight": meta.get("requested_positive_weight"),
            "resolved_positive_weight": meta.get("resolved_positive_weight"),
            "neg_count": meta.get("neg_count"),
            "pos_count": meta.get("pos_count"),
            "neg_pos_ratio": meta.get("neg_pos_ratio"),
        }
        exp11_row = exp11.get(candidate)
        modes = exp13.get(candidate, {})
        valid_row = modes.get("valid")
        oracle_row = modes.get("oracle")
        audit_row = None
        if oracle_row and oracle_row.get("epoch") is not None:
            audit_row = audits.get(candidate, {}).get(int(oracle_row["epoch"]))

        if exp11_row:
            row.update(
                {
                    "policy_ok": exp11_row.get("policy_ok"),
                    "exp11_run_status": exp11_row.get("run_status"),
                    "exp11_valid_auprc": exp11_row.get("valid_auprc"),
                    "exp11_valid_monitor": exp11_row.get("valid_monitor"),
                    "exp11_selected_epoch": exp11_row.get("selected_epoch"),
                    "exp11_selected_checkpoint": exp11_row.get("best_checkpoint"),
                    "exp11_test_auprc": exp11_row.get("test_auprc"),
                    "exp11_test_auroc": exp11_row.get("test_auroc"),
                    "exp11_test_nauprc": exp11_row.get("test_nauprc"),
                    "exp11_test_count": exp11_row.get("test_count"),
                    "exp11_test_pos": exp11_row.get("test_pos"),
                    "exp11_test_neg": exp11_row.get("test_neg"),
                    "learning_rate": exp11_row.get("learning_rate"),
                    "dropout": exp11_row.get("dropout"),
                    "mse_weight": exp11_row.get("mse_weight"),
                    "mse_target_mode": exp11_row.get("mse_target_mode"),
                    "positive_weight_manifest": exp11_row.get("positive_weight"),
                    "graph_feature_mode": exp11_row.get("graph_feature_mode"),
                    "graph_structural_rp": exp11_row.get("graph_structural_rp"),
                    "graph_drug_concat": exp11_row.get("graph_drug_concat"),
                    "graph_logit_scale": exp11_row.get("graph_logit_scale"),
                    "cell_llm_mode": exp11_row.get("cell_llm_mode"),
                    "cell_llm_embedding_path": exp11_row.get("cell_llm_embedding_path"),
                    "cell_type_llm_mode": exp11_row.get("cell_type_llm_mode"),
                    "cell_type_llm_embedding_path": exp11_row.get("cell_type_llm_embedding_path"),
                    "max_epochs": exp11_row.get("max_epochs"),
                    "best_ckpt_metric": exp11_row.get("best_ckpt_metric"),
                    "exp11_source": exp11_row.get("source"),
                }
            )
            if row["resolved_positive_weight"] is None:
                row["resolved_positive_weight"] = exp11_row.get("positive_weight")
        if valid_row:
            row.update(
                {
                    "exp13_valid_auprc": valid_row.get("auprc"),
                    "exp13_valid_auroc": valid_row.get("auroc"),
                    "exp13_valid_nauprc": valid_row.get("nauprc"),
                    "exp13_valid_count": valid_row.get("count"),
                    "exp13_valid_rows_ok": valid_row.get("rows_ok"),
                    "exp13_valid_checkpoint": valid_row.get("checkpoint"),
                    "exp13_valid_source": valid_row.get("source"),
                }
            )
        if oracle_row:
            row.update(
                {
                    "exp13_oracle_auprc": oracle_row.get("auprc"),
                    "exp13_oracle_auroc": oracle_row.get("auroc"),
                    "exp13_oracle_nauprc": oracle_row.get("nauprc"),
                    "exp13_oracle_count": oracle_row.get("count"),
                    "exp13_oracle_rows_ok": oracle_row.get("rows_ok"),
                    "exp13_oracle_epoch": oracle_row.get("epoch"),
                    "exp13_oracle_checkpoint": oracle_row.get("checkpoint"),
                    "exp13_oracle_source": oracle_row.get("source"),
                }
            )
        if audit_row:
            row.update(
                {
                    "oracle_exp11_test_auprc": audit_row.get("auprc"),
                    "oracle_exp11_test_auroc": audit_row.get("auroc"),
                    "oracle_exp11_test_nauprc": audit_row.get("nauprc"),
                    "oracle_exp11_test_count": audit_row.get("count"),
                    "oracle_exp11_test_rows_ok": audit_row.get("rows_ok"),
                    "oracle_exp11_test_checkpoint": audit_row.get("checkpoint"),
                    "oracle_exp11_test_source": audit_row.get("source"),
                }
            )

        check_row(row, args, errors)
        rows.append(row)

    if len(rows) != args.expected_candidates:
        errors.append(f"expected {args.expected_candidates} candidates, found {len(rows)}")
    selections = {
        "best_valid": choose_best(rows, "exp13_valid_auprc"),
        "best_oracle": choose_best(rows, "exp13_oracle_auprc"),
    }
    return rows, selections, errors


def close_int(value: Any, expected: int) -> bool:
    number = finite(value)
    return number is not None and int(number) == expected


def check_row(row: dict[str, Any], args: argparse.Namespace, errors: list[str]) -> None:
    candidate = row["candidate"]
    required = (
        "exp11_valid_auprc",
        "exp11_selected_checkpoint",
        "exp11_test_auprc",
        "exp13_valid_auprc",
        "exp13_oracle_auprc",
        "exp13_oracle_epoch",
        "exp13_oracle_checkpoint",
        "oracle_exp11_test_auprc",
    )
    for key in required:
        if row.get(key) in (None, ""):
            errors.append(f"{candidate}: missing {key}")
    if row.get("exp11_run_status") != "fit_completed":
        errors.append(f"{candidate}: exp11 is not fit_completed")
    if row.get("policy_ok") is not True:
        errors.append(f"{candidate}: fixed-config policy audit failed")
    if not close_int(row.get("exp11_test_count"), args.expected_exp11_test_rows):
        errors.append(f"{candidate}: exp11 selected test rows != {args.expected_exp11_test_rows}")
    if row.get("exp13_valid_rows_ok") is not True:
        errors.append(f"{candidate}: exp13 valid rows != {args.expected_extra_rows}")
    if row.get("exp13_oracle_rows_ok") is not True:
        errors.append(f"{candidate}: exp13 oracle rows != {args.expected_extra_rows}")
    if row.get("oracle_exp11_test_rows_ok") is not True:
        errors.append(f"{candidate}: oracle checkpoint exp11 test rows != {args.expected_exp11_test_rows}")
    checkpoint = row.get("exp13_oracle_checkpoint")
    if checkpoint and not Path(str(checkpoint)).exists():
        errors.append(f"{candidate}: oracle checkpoint path does not exist: {checkpoint}")
    if str(row.get("graph_feature_mode") or "") == "off":
        errors.append(f"{candidate}: graph-off artifact is not allowed")
    if str(row.get("cell_type_llm_embedding_path") or "").endswith("cell_type_llm_embedding_qwen3_4096_v3.npz") is False:
        errors.append(f"{candidate}: cell_type LLM embedding is not v3")


def selection_text(label: str, row: dict[str, Any] | None, precision: int) -> str:
    if row is None:
        return f"- {label}: unavailable"
    if label == "best valid":
        return (
            f"- {label}: `{row['candidate']}` requested `{row.get('requested_positive_weight')}` "
            f"resolved `{fmt(row.get('resolved_positive_weight'), precision)}`; "
            f"exp13 valid AUPRC/AUROC/nAUPRC "
            f"{fmt(row.get('exp13_valid_auprc'), precision)} / "
            f"{fmt(row.get('exp13_valid_auroc'), precision)} / "
            f"{fmt(row.get('exp13_valid_nauprc'), precision)}"
        )
    return (
        f"- {label}: `{row['candidate']}` requested `{row.get('requested_positive_weight')}` "
        f"resolved `{fmt(row.get('resolved_positive_weight'), precision)}`; "
        f"exp13 oracle AUPRC/AUROC/nAUPRC "
        f"{fmt(row.get('exp13_oracle_auprc'), precision)} / "
        f"{fmt(row.get('exp13_oracle_auroc'), precision)} / "
        f"{fmt(row.get('exp13_oracle_nauprc'), precision)}; "
        f"epoch `{row.get('exp13_oracle_epoch')}`"
    )


def basename(value: Any) -> str:
    return "" if not value else Path(str(value)).name


def print_markdown(
    rows: list[dict[str, Any]],
    selections: dict[str, Any],
    errors: list[str],
    args: argparse.Namespace,
) -> None:
    print("# PTV1 exp13 Positive-Weight Tuning Report")
    print()
    print(f"- base prefix: `{args.base_prefix}`")
    print("- scope: exp11 train/valid/test plus exp13 test only; no exp12 and no all_train")
    print("- fixed baseline config: `MSE_WEIGHT=0.50`, `MSE_TARGET_MODE=pdi`, `LR=2e-4`, `DROPOUT=0.15`")
    print("- fixed architecture: graph `real` + structural RP + drug concat, Cell LLM frozen v2, cell-type LLM frozen v3")
    print("- exp13 valid: exp11 validation-selected checkpoint inferred on exp13 test")
    print("- exp13 oracle: diagnostic best exp13 test AUPRC over saved exp11 epoch checkpoints; not official validation selection")
    print(f"- audit status: `{'ok' if not errors else 'incomplete'}`")
    print()
    print("## Final selections")
    print()
    print(selection_text("best valid", selections.get("best_valid"), args.precision))
    print(selection_text("best oracle", selections.get("best_oracle"), args.precision))
    print()
    print("## Candidate summary")
    print()
    print("| candidate | requested | resolved | exp11 valid AUPRC | exp11 epoch | exp11 test AUPRC | exp11 AUROC | exp11 nAUPRC | exp11 rows | exp13 valid AUPRC | valid AUROC | valid nAUPRC | valid rows | oracle epoch | oracle ckpt | exp13 oracle AUPRC | oracle AUROC | oracle nAUPRC | oracle rows | oracle ckpt exp11 AUPRC | oracle ckpt exp11 AUROC | oracle ckpt exp11 nAUPRC | oracle exp11 rows |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {candidate} | {requested} | {resolved} | {valid_auprc} | {epoch} | {exp11_test_auprc} | {exp11_test_auroc} | {exp11_test_nauprc} | {exp11_rows} | {exp13_valid_auprc} | {exp13_valid_auroc} | {exp13_valid_nauprc} | {exp13_valid_rows} | {oracle_epoch} | {oracle_ckpt} | {exp13_oracle_auprc} | {exp13_oracle_auroc} | {exp13_oracle_nauprc} | {oracle_rows} | {audit_auprc} | {audit_auroc} | {audit_nauprc} | {audit_rows} |".format(
                candidate=row["candidate"],
                requested=row.get("requested_positive_weight") or "",
                resolved=fmt(row.get("resolved_positive_weight"), args.precision),
                valid_auprc=fmt(row.get("exp11_valid_auprc"), args.precision),
                epoch="" if row.get("exp11_selected_epoch") is None else str(row.get("exp11_selected_epoch")),
                exp11_test_auprc=fmt(row.get("exp11_test_auprc"), args.precision),
                exp11_test_auroc=fmt(row.get("exp11_test_auroc"), args.precision),
                exp11_test_nauprc=fmt(row.get("exp11_test_nauprc"), args.precision),
                exp11_rows="" if finite(row.get("exp11_test_count")) is None else str(int(row["exp11_test_count"])),
                exp13_valid_auprc=fmt(row.get("exp13_valid_auprc"), args.precision),
                exp13_valid_auroc=fmt(row.get("exp13_valid_auroc"), args.precision),
                exp13_valid_nauprc=fmt(row.get("exp13_valid_nauprc"), args.precision),
                exp13_valid_rows="" if finite(row.get("exp13_valid_count")) is None else str(int(row["exp13_valid_count"])),
                oracle_epoch="" if row.get("exp13_oracle_epoch") is None else str(row.get("exp13_oracle_epoch")),
                oracle_ckpt=basename(row.get("exp13_oracle_checkpoint")),
                exp13_oracle_auprc=fmt(row.get("exp13_oracle_auprc"), args.precision),
                exp13_oracle_auroc=fmt(row.get("exp13_oracle_auroc"), args.precision),
                exp13_oracle_nauprc=fmt(row.get("exp13_oracle_nauprc"), args.precision),
                oracle_rows="" if finite(row.get("exp13_oracle_count")) is None else str(int(row["exp13_oracle_count"])),
                audit_auprc=fmt(row.get("oracle_exp11_test_auprc"), args.precision),
                audit_auroc=fmt(row.get("oracle_exp11_test_auroc"), args.precision),
                audit_nauprc=fmt(row.get("oracle_exp11_test_nauprc"), args.precision),
                audit_rows="" if finite(row.get("oracle_exp11_test_count")) is None else str(int(row["oracle_exp11_test_count"])),
            )
        )
    print()
    print("## Audit")
    print()
    if errors:
        for error in errors:
            print(f"- {error}")
    else:
        print("- all expected candidates completed")
        print(f"- exp11 selected-test rows are `{args.expected_exp11_test_rows}` for every candidate")
        print(f"- exp13 valid/oracle rows are `{args.expected_extra_rows}` for every candidate")
        print("- all official rows pass graph/Cell LLM/cell-type LLM/MSE target policy checks")


def print_tsv(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    columns = (
        "candidate",
        "requested_positive_weight",
        "resolved_positive_weight",
        "neg_count",
        "pos_count",
        "neg_pos_ratio",
        "policy_ok",
        "exp11_valid_auprc",
        "exp11_valid_monitor",
        "exp11_selected_epoch",
        "exp11_selected_checkpoint",
        "exp11_test_auprc",
        "exp11_test_auroc",
        "exp11_test_nauprc",
        "exp11_test_count",
        "exp11_test_pos",
        "exp11_test_neg",
        "exp13_valid_auprc",
        "exp13_valid_auroc",
        "exp13_valid_nauprc",
        "exp13_valid_count",
        "exp13_valid_rows_ok",
        "exp13_valid_checkpoint",
        "exp13_oracle_epoch",
        "exp13_oracle_checkpoint",
        "exp13_oracle_auprc",
        "exp13_oracle_auroc",
        "exp13_oracle_nauprc",
        "exp13_oracle_count",
        "exp13_oracle_rows_ok",
        "oracle_exp11_test_auprc",
        "oracle_exp11_test_auroc",
        "oracle_exp11_test_nauprc",
        "oracle_exp11_test_count",
        "oracle_exp11_test_rows_ok",
        "oracle_exp11_test_checkpoint",
        "learning_rate",
        "dropout",
        "mse_weight",
        "mse_target_mode",
        "positive_weight_manifest",
        "graph_feature_mode",
        "graph_structural_rp",
        "graph_drug_concat",
        "graph_logit_scale",
        "cell_llm_mode",
        "cell_llm_embedding_path",
        "cell_type_llm_mode",
        "cell_type_llm_embedding_path",
        "max_epochs",
        "best_ckpt_metric",
        "exp11_source",
        "exp13_valid_source",
        "exp13_oracle_source",
        "oracle_exp11_test_source",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: cell(row.get(key), args.precision) for key in columns})


def main() -> int:
    args = parse_args()
    rows, selections, errors = build_rows(args)
    if args.format == "markdown":
        print_markdown(rows, selections, errors, args)
    elif args.format == "tsv":
        print_tsv(rows, args)
    else:
        print(json.dumps({"rows": rows, "selections": selections, "errors": errors}, ensure_ascii=False, indent=2))
    if args.strict and errors:
        print(f"[error] strict audit failed with {len(errors)} issue(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
