#!/usr/bin/env python3
"""Summarize PTV1 frozen Cell LLM fine-tune search results."""

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
EXP13_SUFFIXES = {
    "direct": "_extra_direct_from_exp11",
    "all_train": "_all_ptv1_for_extra_from_exp11",
}
EPOCH_RE = re.compile(r"epoch=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--exp11-auprc-threshold", type=float, default=0.883452)
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
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def cell(value: Any, precision: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if finite(value) is not None:
        return fmt(value, precision)
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


def config_value(manifest: dict[str, Any], key: str) -> Any:
    if key in manifest:
        return manifest[key]
    args = manifest.get("args")
    if isinstance(args, dict):
        return args.get(key)
    return None


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
        "accelerator",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
    )
    return {key: config_value(manifest, key) for key in keys}


def training_record(path: Path, candidate: str, exp: str, fold: str | None = None) -> dict[str, Any] | None:
    manifest = load_json(path)
    if manifest.get("dataset_group") != "ptv1":
        return None
    if manifest.get("task_name") != "ptv1_aivc":
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
        for key, value in config_snapshot_source(rows).items():
            item[key] = value
        aggregates[candidate] = item
    return aggregates


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
        "accelerator",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
    )
    out: dict[str, Any] = {}
    for key in keys:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        out[key] = values[0] if values else None
    return out


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


def collect_exp13(args: argparse.Namespace) -> dict[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for metrics_path in sorted(args.output_root.glob(f"{args.base_prefix}_*/ptv1_extra_singledrug/metrics.json")):
        exp_name = metrics_path.parent.parent.name
        mode = None
        candidate = None
        for mode_name, suffix in EXP13_SUFFIXES.items():
            candidate = candidate_from_name(exp_name, args.base_prefix, suffix)
            if candidate is not None:
                mode = mode_name
                break
        if candidate is None or mode is None:
            continue
        metrics = metrics_from_extra(metrics_path)
        if metrics is None:
            continue
        run_manifest_path = metrics_path.parent / "run_manifest.json"
        n_predictions = None
        checkpoint_manifest = None
        if run_manifest_path.exists():
            run_manifest = load_json(run_manifest_path)
            n_predictions = finite(run_manifest.get("n_predictions"))
            checkpoint_manifest = run_manifest.get("checkpoint_run_manifest_path")
        records[candidate][mode] = {
            "candidate": candidate,
            "exp": "exp13",
            "mode": mode,
            "n_predictions": n_predictions,
            "expected_extra_rows": args.expected_extra_rows,
            "rows_ok": int(n_predictions) == args.expected_extra_rows if n_predictions is not None else False,
            "source": str(metrics_path),
            "run_manifest": str(run_manifest_path) if run_manifest_path.exists() else None,
            "checkpoint_run_manifest_path": checkpoint_manifest,
            **metrics,
        }
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
        for key, value in config_snapshot_source([source]).items():
            row[key] = value
        if exp11_row:
            row.update(
                {
                    "exp11_auroc": exp11_row.get("auroc"),
                    "exp11_auprc": exp11_row.get("auprc"),
                    "exp11_nauprc": exp11_row.get("nauprc"),
                    "exp11_count": exp11_row.get("count"),
                    "exp11_epoch": exp11_row.get("selected_epoch"),
                    "exp11_source": exp11_row.get("source"),
                    "exp11_ok": (finite(exp11_row.get("auprc")) or -math.inf) >= args.exp11_auprc_threshold,
                }
            )
        else:
            row["exp11_ok"] = False
        if exp12_row:
            row.update(
                {
                    "exp12_folds": exp12_row.get("folds"),
                    "exp12_n_folds": exp12_row.get("n_folds"),
                    "exp12_mean_auroc": exp12_row.get("mean_auroc"),
                    "exp12_mean_auprc": exp12_row.get("mean_auprc"),
                    "exp12_mean_nauprc": exp12_row.get("mean_nauprc"),
                    "exp12_std_auprc": exp12_row.get("std_auprc"),
                    "exp12_epochs": exp12_row.get("epochs"),
                    "exp12_fold_auprc": exp12_row.get("fold_auprc"),
                    "exp12_complete": exp12_row.get("n_folds") == 5,
                }
            )
        else:
            row["exp12_complete"] = False
        mode_records = []
        for mode in ("direct", "all_train"):
            mode_row = exp13_modes.get(mode)
            if mode_row:
                mode_records.append(mode_row)
                row[f"exp13_{mode}_auroc"] = mode_row.get("auroc")
                row[f"exp13_{mode}_auprc"] = mode_row.get("auprc")
                row[f"exp13_{mode}_nauprc"] = mode_row.get("nauprc")
                row[f"exp13_{mode}_count"] = mode_row.get("count")
                row[f"exp13_{mode}_rows_ok"] = mode_row.get("rows_ok")
                row[f"exp13_{mode}_source"] = mode_row.get("source")
        best_mode = choose_best(mode_records, "auprc")
        if best_mode:
            row["exp13_best_mode"] = best_mode.get("mode")
            row["exp13_best_auprc"] = best_mode.get("auprc")
            row["exp13_best_nauprc"] = best_mode.get("nauprc")
            row["exp13_best_count"] = best_mode.get("count")
            row["exp13_selectable"] = bool(row.get("exp11_ok"))
        else:
            row["exp13_selectable"] = False
        rows.append(row)

    selections = {
        "exp11_best": choose_best([row for row in rows if row.get("exp11_auprc") is not None], "exp11_auprc"),
        "exp12_best": choose_best(
            [row for row in rows if row.get("exp12_complete") and row.get("exp12_mean_auprc") is not None],
            "exp12_mean_auprc",
        ),
        "exp13_constrained_best": choose_best(
            [row for row in rows if row.get("exp11_ok") and row.get("exp13_best_auprc") is not None],
            "exp13_best_auprc",
        ),
    }
    rows.sort(
        key=lambda row: (
            finite(row.get("exp13_best_auprc")) is None,
            -(finite(row.get("exp13_best_auprc")) or -1.0),
            finite(row.get("exp11_auprc")) is None,
            -(finite(row.get("exp11_auprc")) or -1.0),
            row["candidate"],
        )
    )
    return rows, selections


def selection_text(label: str, row: dict[str, Any] | None, precision: int) -> str:
    if row is None:
        return f"- {label}: unavailable"
    if label.startswith("exp_11"):
        return f"- {label}: `{row['candidate']}` AUPRC {fmt(row.get('exp11_auprc'), precision)}"
    if label.startswith("exp_12"):
        return f"- {label}: `{row['candidate']}` mean5 AUPRC {fmt(row.get('exp12_mean_auprc'), precision)}"
    mode = row.get("exp13_best_mode") or ""
    return f"- {label}: `{row['candidate']}` mode `{mode}` extra AUPRC {fmt(row.get('exp13_best_auprc'), precision)}"


def print_markdown(rows: list[dict[str, Any]], selections: dict[str, Any], args: argparse.Namespace) -> None:
    print(f"# PTV1 Frozen Cell LLM Fine-tune Report")
    print()
    print(f"- base prefix: `{args.base_prefix}`")
    print(f"- exp_11 AUPRC threshold for exp_13 selection: `{args.exp11_auprc_threshold:.6f}`")
    print(f"- candidates found: `{len(rows)}`")
    print()
    print("## Final selections")
    print()
    print(selection_text("exp_11 best", selections.get("exp11_best"), args.precision))
    print(selection_text("exp_12 best", selections.get("exp12_best"), args.precision))
    print(selection_text("exp_13 constrained best", selections.get("exp13_constrained_best"), args.precision))
    print()
    print("## Candidate summary")
    print()
    print("| rank | candidate | exp11 AUPRC | exp11 ok | exp12 folds | exp12 mean AUPRC | exp13 direct AUPRC | direct n | exp13 all_train AUPRC | all_train n | exp13 best | lr | dropout | MSE | pos_weight | pos_sample | focal | smooth | graph | graph_logit | PCEP | target | target_expr | ctrl_drop |")
    print("|---:|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|---:|---|---:|---|---|---|---:|")
    for rank, row in enumerate(rows, start=1):
        graph = str(row.get("graph_feature_mode") or "")
        if row.get("graph_feature_mode") == "real":
            graph = f"real/srp={cell(row.get('graph_structural_rp'), 0)}/drug={cell(row.get('graph_drug_concat'), 0)}"
        pcep = str(row.get("protein_concat_mode") or "")
        if row.get("protein_concat_mode") not in (None, "", "off"):
            pcep = f"{row.get('protein_concat_mode')}/{row.get('protein_concat_score_mode') or ''}"
        best = ""
        if row.get("exp13_best_mode"):
            best = f"{row.get('exp13_best_mode')}:{fmt(row.get('exp13_best_auprc'), args.precision)}"
        print(
            "| {rank} | {candidate} | {exp11_auprc} | {exp11_ok} | {exp12_folds} | {exp12_mean_auprc} | {direct_auprc} | {direct_n} | {all_auprc} | {all_n} | {best} | {lr} | {dropout} | {mse} | {pos_weight} | {pos_sample} | {focal} | {smooth} | {graph} | {graph_logit} | {pcep} | {target} | {target_expr} | {ctrl_drop} |".format(
                rank=rank,
                candidate=row["candidate"],
                exp11_auprc=fmt(row.get("exp11_auprc"), args.precision),
                exp11_ok="yes" if row.get("exp11_ok") else "no",
                exp12_folds=row.get("exp12_folds") or "",
                exp12_mean_auprc=fmt(row.get("exp12_mean_auprc"), args.precision),
                direct_auprc=fmt(row.get("exp13_direct_auprc"), args.precision),
                direct_n=str(int(row["exp13_direct_count"])) if finite(row.get("exp13_direct_count")) is not None else "",
                all_auprc=fmt(row.get("exp13_all_train_auprc"), args.precision),
                all_n=str(int(row["exp13_all_train_count"])) if finite(row.get("exp13_all_train_count")) is not None else "",
                best=best,
                lr=fmt(row.get("learning_rate"), 6),
                dropout=fmt(row.get("dropout"), 4),
                mse=fmt(row.get("mse_weight"), 4),
                pos_weight=cell(row.get("positive_weight"), args.precision),
                pos_sample=fmt(row.get("positive_label_sampling_weight"), 2),
                focal=cell(row.get("focal_loss"), args.precision),
                smooth=fmt(row.get("label_smoothing"), 3),
                graph=graph,
                graph_logit=fmt(row.get("graph_logit_scale"), 3),
                pcep=pcep,
                target=str(row.get("mse_target_mode") or ""),
                target_expr=str(row.get("target_expression_mode") or ""),
                ctrl_drop=fmt(row.get("control_expression_dropout"), 3),
            )
        )


def print_tsv(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    columns = (
        "candidate",
        "exp11_auprc",
        "exp11_ok",
        "exp11_epoch",
        "exp12_n_folds",
        "exp12_folds",
        "exp12_mean_auprc",
        "exp12_std_auprc",
        "exp12_mean_nauprc",
        "exp12_mean_auroc",
        "exp13_direct_auprc",
        "exp13_direct_nauprc",
        "exp13_direct_count",
        "exp13_direct_rows_ok",
        "exp13_all_train_auprc",
        "exp13_all_train_nauprc",
        "exp13_all_train_count",
        "exp13_all_train_rows_ok",
        "exp13_best_mode",
        "exp13_best_auprc",
        "exp13_selectable",
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
        "accelerator",
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
        print(f"[error] no fine-tune records found for base prefix {args.base_prefix!r}", file=sys.stderr)
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
