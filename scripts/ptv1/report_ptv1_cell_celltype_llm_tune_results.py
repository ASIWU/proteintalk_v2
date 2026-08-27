#!/usr/bin/env python3
"""Summarize PTV1 graph + Cell LLM + cell_type LLM tuning results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


EXP11_SUFFIX = "_ptv1_random_split"
EXP12_RE = re.compile(r"^(?P<prefix>.+)_ptv1_unseen_drug_fold(?P<fold>[0-4])$")
VALID_SUFFIX = "_extra_valid_from_exp11"
ORACLE_RE_TEMPLATE = r"^{base}_(?P<candidate>.+)_extra_oracle_epoch(?P<epoch>[0-9]+)_from_exp11$"
EPOCH_RE = re.compile(r"epoch=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prefix", required=True)
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


def fmt(value: Any, precision: int) -> str:
    number = finite(value)
    return "" if number is None else f"{number:.{precision}f}"


def cell(value: Any, precision: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    number = finite(value)
    if number is not None:
        return fmt(number, precision)
    return str(value)


def candidate_from_name(name: str, base_prefix: str, suffix: str) -> str | None:
    head = f"{base_prefix}_"
    if not name.startswith(head) or not name.endswith(suffix):
        return None
    candidate = name[len(head) : -len(suffix)]
    return candidate or None


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


def policy_ok(manifest: dict[str, Any]) -> bool:
    return (
        str(config_value(manifest, "graph_feature_mode") or "") != "off"
        and str(config_value(manifest, "cell_llm_mode") or "") == "frozen"
        and str(config_value(manifest, "cell_type_llm_mode") or "") == "frozen"
        and truthy(config_value(manifest, "graph_structural_rp"))
        and truthy(config_value(manifest, "graph_drug_concat"))
    )


def config_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "learning_rate",
        "batch_size",
        "dropout",
        "weight_decay",
        "mse_weight",
        "mse_target_mode",
        "positive_weight",
        "positive_label_sampling_weight",
        "focal_loss",
        "label_smoothing",
        "graph_feature_mode",
        "graph_structural_rp",
        "graph_drug_concat",
        "graph_logit_scale",
        "protein_concat_mode",
        "protein_concat_score_mode",
        "target_expression_mode",
        "control_expression_dropout",
        "batch_cov_list",
        "use_dose_covariate",
        "cell_llm_mode",
        "cell_llm_embedding_path",
        "cell_type_llm_mode",
        "cell_type_llm_embedding_path",
        "accelerator",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
    )
    out = {key: config_value(manifest, key) for key in keys}
    out["policy_ok"] = policy_ok(manifest)
    return out


def config_snapshot_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "learning_rate",
        "batch_size",
        "dropout",
        "weight_decay",
        "mse_weight",
        "mse_target_mode",
        "positive_weight",
        "positive_label_sampling_weight",
        "focal_loss",
        "label_smoothing",
        "graph_feature_mode",
        "graph_structural_rp",
        "graph_drug_concat",
        "graph_logit_scale",
        "protein_concat_mode",
        "protein_concat_score_mode",
        "target_expression_mode",
        "control_expression_dropout",
        "batch_cov_list",
        "use_dose_covariate",
        "cell_llm_mode",
        "cell_llm_embedding_path",
        "cell_type_llm_mode",
        "cell_type_llm_embedding_path",
        "accelerator",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "policy_ok",
    )
    out: dict[str, Any] = {}
    for key in keys:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        out[key] = values[0] if values else None
    return out


def test_metrics(manifest: dict[str, Any]) -> dict[str, Any] | None:
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


def training_record(path: Path, candidate: str, exp: str, fold: str | None = None) -> dict[str, Any] | None:
    manifest = load_json(path)
    if manifest.get("dataset_group") != "ptv1" or manifest.get("task_name") != "ptv1_aivc":
        return None
    if manifest.get("run_status") != "fit_completed":
        return None
    metrics = test_metrics(manifest)
    if metrics is None:
        return None
    checkpoint = manifest.get("best_model_path") or manifest.get("test_checkpoint_path")
    return {
        "candidate": candidate,
        "exp": exp,
        "fold": fold,
        "split": manifest.get("split_strategy"),
        "selected_epoch": parse_epoch(checkpoint),
        "best_checkpoint": checkpoint,
        "source": str(path),
        **config_snapshot(manifest),
        **metrics,
    }


def collect_exp11(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(args.checkpoint_root.glob(f"{args.base_prefix}_*{EXP11_SUFFIX}/run_manifest.json")):
        candidate = candidate_from_name(path.parent.name, args.base_prefix, EXP11_SUFFIX)
        if candidate is None:
            continue
        record = training_record(path, candidate, "exp11")
        if record is not None and record.get("split") == "fixed_experiment_type":
            records[candidate] = record
    return records


def collect_exp12(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    per_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.checkpoint_root.glob(f"{args.base_prefix}_*_ptv1_unseen_drug_fold[0-4]/run_manifest.json")):
        match = EXP12_RE.match(path.parent.name)
        if match is None:
            continue
        candidate_prefix = match.group("prefix")
        head = f"{args.base_prefix}_"
        if not candidate_prefix.startswith(head):
            continue
        candidate = candidate_prefix[len(head) :]
        fold = match.group("fold")
        record = training_record(path, candidate, "exp12", fold)
        if record is not None and str(record.get("split") or "").startswith("pert_id_5fold_fold"):
            per_fold[candidate].append(record)

    aggregates: dict[str, dict[str, Any]] = {}
    for candidate, rows in per_fold.items():
        rows = sorted(rows, key=lambda row: int(row["fold"]))
        item: dict[str, Any] = {
            "candidate": candidate,
            "exp": "exp12",
            "folds": ",".join(str(row["fold"]) for row in rows),
            "n_folds": len(rows),
            "count": sum(finite(row.get("count")) or 0.0 for row in rows),
            "pos": sum(int(row.get("pos") or 0) for row in rows),
            "neg": sum(int(row.get("neg") or 0) for row in rows),
            "epochs": ",".join("" if row.get("selected_epoch") is None else str(row["selected_epoch"]) for row in rows),
            "fold_auprc": ",".join(fmt(row.get("auprc"), 4) for row in rows),
            "sources": [row["source"] for row in rows],
        }
        for key in ("auprc", "nauprc", "auroc"):
            values = [value for row in rows if (value := finite(row.get(key))) is not None]
            item[f"mean_{key}"] = mean(values) if values else None
            item[f"std_{key}"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        item.update(config_snapshot_source(rows))
        aggregates[candidate] = item
    return aggregates


def metrics_from_extra(metrics_path: Path) -> dict[str, Any] | None:
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


def extra_record(args: argparse.Namespace, candidate: str, mode: str, metrics_path: Path, epoch: int | None) -> dict[str, Any] | None:
    metrics = metrics_from_extra(metrics_path)
    if metrics is None:
        return None
    run_manifest_path = metrics_path.parent / "run_manifest.json"
    run_manifest = load_json(run_manifest_path) if run_manifest_path.exists() else {}
    n_predictions = finite(run_manifest.get("n_predictions"))
    row = {
        "candidate": candidate,
        "exp": "exp13",
        "mode": mode,
        "epoch": epoch,
        "n_predictions": n_predictions,
        "expected_extra_rows": args.expected_extra_rows,
        "rows_ok": int(n_predictions) == args.expected_extra_rows if n_predictions is not None else False,
        "checkpoint_path": run_manifest.get("checkpoint_path"),
        "checkpoint_run_manifest_path": run_manifest.get("checkpoint_run_manifest_path"),
        "source": str(metrics_path),
        "run_manifest": str(run_manifest_path) if run_manifest_path.exists() else None,
        **config_snapshot(run_manifest),
        **metrics,
    }
    return row


def collect_exp13(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = defaultdict(dict)
    oracle_re = re.compile(ORACLE_RE_TEMPLATE.format(base=re.escape(args.base_prefix)))
    for metrics_path in sorted(args.output_root.glob(f"{args.base_prefix}_*/ptv1_extra_singledrug/metrics.json")):
        exp_name = metrics_path.parent.parent.name
        candidate = candidate_from_name(exp_name, args.base_prefix, VALID_SUFFIX)
        if candidate is not None:
            row = extra_record(args, candidate, "valid", metrics_path, None)
            if row is not None:
                records[candidate]["valid"] = row
            continue
        match = oracle_re.match(exp_name)
        if match:
            candidate = match.group("candidate")
            epoch = int(match.group("epoch"))
            row = extra_record(args, candidate, "oracle", metrics_path, epoch)
            if row is not None:
                current = records[candidate].get("oracle")
                if current is None or (finite(row.get("auprc")) or -math.inf) > (finite(current.get("auprc")) or -math.inf):
                    records[candidate]["oracle"] = row
    return records


def choose_best(items: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    valid = [item for item in items if finite(item.get(metric)) is not None]
    if not valid:
        return None
    return max(valid, key=lambda item: (finite(item.get(metric)) or -math.inf, str(item.get("candidate") or "")))


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exp11 = collect_exp11(args)
    exp12 = collect_exp12(args)
    exp13 = collect_exp13(args)
    candidates = sorted(set(exp11) | set(exp12) | set(exp13))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row: dict[str, Any] = {"candidate": candidate}
        exp11_row = exp11.get(candidate)
        exp12_row = exp12.get(candidate)
        exp13_modes = exp13.get(candidate, {})
        source = exp11_row or exp12_row or next(iter(exp13_modes.values()), {})
        row.update(config_snapshot_source([source]))
        if exp11_row:
            row.update({
                "exp11_auroc": exp11_row.get("auroc"),
                "exp11_auprc": exp11_row.get("auprc"),
                "exp11_nauprc": exp11_row.get("nauprc"),
                "exp11_count": exp11_row.get("count"),
                "exp11_epoch": exp11_row.get("selected_epoch"),
                "exp11_checkpoint": exp11_row.get("best_checkpoint"),
                "exp11_source": exp11_row.get("source"),
            })
        if exp12_row:
            row.update({
                "exp12_folds": exp12_row.get("folds"),
                "exp12_n_folds": exp12_row.get("n_folds"),
                "exp12_mean_auroc": exp12_row.get("mean_auroc"),
                "exp12_mean_auprc": exp12_row.get("mean_auprc"),
                "exp12_mean_nauprc": exp12_row.get("mean_nauprc"),
                "exp12_std_auprc": exp12_row.get("std_auprc"),
                "exp12_epochs": exp12_row.get("epochs"),
                "exp12_fold_auprc": exp12_row.get("fold_auprc"),
                "exp12_complete": exp12_row.get("n_folds") == 5,
            })
        else:
            row["exp12_complete"] = False
        for mode in ("valid", "oracle"):
            mode_row = exp13_modes.get(mode)
            if mode_row:
                row[f"exp13_{mode}_auroc"] = mode_row.get("auroc")
                row[f"exp13_{mode}_auprc"] = mode_row.get("auprc")
                row[f"exp13_{mode}_nauprc"] = mode_row.get("nauprc")
                row[f"exp13_{mode}_count"] = mode_row.get("count")
                row[f"exp13_{mode}_rows_ok"] = mode_row.get("rows_ok")
                row[f"exp13_{mode}_epoch"] = mode_row.get("epoch")
                row[f"exp13_{mode}_checkpoint"] = mode_row.get("checkpoint_path")
                row[f"exp13_{mode}_source"] = mode_row.get("source")
        rows.append(row)

    selections = {
        "exp11_best": choose_best([row for row in rows if row.get("exp11_auprc") is not None], "exp11_auprc"),
        "exp12_best": choose_best([row for row in rows if row.get("exp12_complete")], "exp12_mean_auprc"),
        "exp13_valid_best": choose_best([row for row in rows if row.get("exp13_valid_auprc") is not None], "exp13_valid_auprc"),
        "exp13_oracle_best": choose_best([row for row in rows if row.get("exp13_oracle_auprc") is not None], "exp13_oracle_auprc"),
    }
    rows.sort(key=lambda row: (finite(row.get("exp11_auprc")) is None, -(finite(row.get("exp11_auprc")) or -1.0), row["candidate"]))
    return rows, selections


def selection_text(label: str, row: dict[str, Any] | None, precision: int) -> str:
    if row is None:
        return f"- {label}: unavailable"
    if label == "exp_11 best":
        return f"- {label}: `{row['candidate']}` AUPRC {fmt(row.get('exp11_auprc'), precision)}, AUROC {fmt(row.get('exp11_auroc'), precision)}, epoch `{row.get('exp11_epoch')}`"
    if label == "exp_12 best":
        return f"- {label}: `{row['candidate']}` mean5 AUPRC {fmt(row.get('exp12_mean_auprc'), precision)}, AUROC {fmt(row.get('exp12_mean_auroc'), precision)}"
    if label == "exp_13 valid best":
        return f"- {label}: `{row['candidate']}` AUPRC {fmt(row.get('exp13_valid_auprc'), precision)}, AUROC {fmt(row.get('exp13_valid_auroc'), precision)}"
    return f"- {label}: `{row['candidate']}` AUPRC {fmt(row.get('exp13_oracle_auprc'), precision)}, AUROC {fmt(row.get('exp13_oracle_auroc'), precision)}, epoch `{row.get('exp13_oracle_epoch')}`"


def print_markdown(rows: list[dict[str, Any]], selections: dict[str, Any], args: argparse.Namespace) -> None:
    print("# PTV1 Graph + Cell/Cell-Type LLM Tuning Report")
    print()
    print(f"- base prefix: `{args.base_prefix}`")
    print(f"- required policy: `GRAPH_FEATURE_MODE!=off`, `CELL_LLM_MODE=frozen`, `CELL_TYPE_LLM_MODE=frozen`")
    print(f"- candidates found: `{len(rows)}`")
    print(f"- exp13 valid: exp11 validation-selected checkpoint inferred on `ptv1_extra_singledrug`")
    print(f"- exp13 oracle: best exp13 AUPRC across saved exp11 epoch checkpoints; diagnostic only")
    print()
    print("## Final selections")
    print()
    print(selection_text("exp_11 best", selections.get("exp11_best"), args.precision))
    print(selection_text("exp_12 best", selections.get("exp12_best"), args.precision))
    print(selection_text("exp_13 valid best", selections.get("exp13_valid_best"), args.precision))
    print(selection_text("exp_13 oracle best", selections.get("exp13_oracle_best"), args.precision))
    print()
    print("## Candidate summary")
    print()
    print("| rank | candidate | policy | exp11 AUPRC | exp11 AUROC | exp11 epoch | exp12 folds | exp12 mean AUPRC | exp12 mean AUROC | exp13 valid AUPRC | valid AUROC | valid n | exp13 oracle AUPRC | oracle AUROC | oracle epoch | oracle n | lr | dropout | MSE | target | graph | cell LLM | cell-type LLM |")
    print("|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for rank, row in enumerate(rows, start=1):
        graph = str(row.get("graph_feature_mode") or "")
        if row.get("graph_feature_mode") == "real":
            graph = f"real/srp={cell(row.get('graph_structural_rp'), 0)}/drug={cell(row.get('graph_drug_concat'), 0)}/logit={fmt(row.get('graph_logit_scale'), 2)}"
        print(
            "| {rank} | {candidate} | {policy} | {exp11_auprc} | {exp11_auroc} | {exp11_epoch} | {exp12_folds} | {exp12_mean_auprc} | {exp12_mean_auroc} | {valid_auprc} | {valid_auroc} | {valid_n} | {oracle_auprc} | {oracle_auroc} | {oracle_epoch} | {oracle_n} | {lr} | {dropout} | {mse} | {target} | {graph} | {cell_llm} | {cell_type_llm} |".format(
                rank=rank,
                candidate=row["candidate"],
                policy="yes" if row.get("policy_ok") else "no",
                exp11_auprc=fmt(row.get("exp11_auprc"), args.precision),
                exp11_auroc=fmt(row.get("exp11_auroc"), args.precision),
                exp11_epoch="" if row.get("exp11_epoch") is None else str(row.get("exp11_epoch")),
                exp12_folds=row.get("exp12_folds") or "",
                exp12_mean_auprc=fmt(row.get("exp12_mean_auprc"), args.precision),
                exp12_mean_auroc=fmt(row.get("exp12_mean_auroc"), args.precision),
                valid_auprc=fmt(row.get("exp13_valid_auprc"), args.precision),
                valid_auroc=fmt(row.get("exp13_valid_auroc"), args.precision),
                valid_n=str(int(row["exp13_valid_count"])) if finite(row.get("exp13_valid_count")) is not None else "",
                oracle_auprc=fmt(row.get("exp13_oracle_auprc"), args.precision),
                oracle_auroc=fmt(row.get("exp13_oracle_auroc"), args.precision),
                oracle_epoch="" if row.get("exp13_oracle_epoch") is None else str(row.get("exp13_oracle_epoch")),
                oracle_n=str(int(row["exp13_oracle_count"])) if finite(row.get("exp13_oracle_count")) is not None else "",
                lr=fmt(row.get("learning_rate"), 6),
                dropout=fmt(row.get("dropout"), 4),
                mse=fmt(row.get("mse_weight"), 4),
                target=str(row.get("mse_target_mode") or ""),
                graph=graph,
                cell_llm=str(row.get("cell_llm_mode") or ""),
                cell_type_llm=str(row.get("cell_type_llm_mode") or ""),
            )
        )


def print_tsv(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    columns = (
        "candidate",
        "policy_ok",
        "exp11_auprc",
        "exp11_auroc",
        "exp11_nauprc",
        "exp11_epoch",
        "exp11_checkpoint",
        "exp12_n_folds",
        "exp12_folds",
        "exp12_mean_auprc",
        "exp12_std_auprc",
        "exp12_mean_nauprc",
        "exp12_mean_auroc",
        "exp13_valid_auprc",
        "exp13_valid_auroc",
        "exp13_valid_nauprc",
        "exp13_valid_count",
        "exp13_valid_rows_ok",
        "exp13_valid_checkpoint",
        "exp13_oracle_auprc",
        "exp13_oracle_auroc",
        "exp13_oracle_nauprc",
        "exp13_oracle_count",
        "exp13_oracle_rows_ok",
        "exp13_oracle_epoch",
        "exp13_oracle_checkpoint",
        "learning_rate",
        "batch_size",
        "dropout",
        "weight_decay",
        "mse_weight",
        "mse_target_mode",
        "positive_weight",
        "positive_label_sampling_weight",
        "focal_loss",
        "label_smoothing",
        "graph_feature_mode",
        "graph_structural_rp",
        "graph_drug_concat",
        "graph_logit_scale",
        "protein_concat_mode",
        "protein_concat_score_mode",
        "target_expression_mode",
        "control_expression_dropout",
        "batch_cov_list",
        "cell_llm_mode",
        "cell_llm_embedding_path",
        "cell_type_llm_mode",
        "cell_type_llm_embedding_path",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: cell(row.get(key), args.precision) for key in columns})


def main() -> int:
    args = parse_args()
    rows, selections = build_rows(args)
    if not rows:
        print(f"[error] no records found for base prefix {args.base_prefix!r}", file=sys.stderr)
        return 1
    if args.format == "markdown":
        print_markdown(rows, selections, args)
    elif args.format == "tsv":
        print_tsv(rows, args)
    else:
        print(json.dumps({"rows": rows, "selections": selections}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
