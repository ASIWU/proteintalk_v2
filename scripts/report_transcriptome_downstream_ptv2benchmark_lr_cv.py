#!/usr/bin/env python3
"""Summarize PTV2-benchmark-aligned transcriptome MLP LR CV results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = "20260629_transcriptome_mlp_ptv2benchmark_lr_cv_ep150_v1"
BRANCHES = ("cpa", "biolord")
EXPS = ("exp01", "exp02", "exp03")
FOLDS = (0, 1, 2, 3, 4)
LR_GRID = (
    ("lr5e5", 5e-5),
    ("lr1e4", 1e-4),
    ("lr2e4", 2e-4),
    ("lr5e4", 5e-4),
    ("lr1e3", 1e-3),
)


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_finite(values: list[Any]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def sum_finite(values: list[Any]) -> int:
    return int(sum(number for value in values if (number := finite(value)) is not None))


def fmt(value: Any, precision: int = 6) -> str:
    number = finite(value)
    if number is None:
        return ""
    return f"{number:.{precision}f}"


def int_text(value: Any) -> str:
    number = finite(value)
    if number is None:
        return ""
    return str(int(number))


def metric_payload(metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "loss": finite(metrics.get("loss")),
        "auroc": finite(metrics.get("auroc")),
        "auprc": finite(metrics.get("auprc")),
        "auprc_baseline": finite(metrics.get("auprc_baseline")),
        "nauprc": finite(metrics.get("nauprc")),
        "acc": finite(metrics.get("acc")),
        "count": finite(metrics.get("count")),
        "positive_count": finite(metrics.get("positive_count")),
        "negative_count": finite(metrics.get("negative_count")),
    }


def config_fields(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config") or {}
    model = config.get("model") or {}
    training = config.get("training") or {}
    return {
        "hidden_dim": model.get("hidden_dim"),
        "drug_hidden_dim": model.get("drug_hidden_dim"),
        "fusion_hidden_dim": model.get("fusion_hidden_dim"),
        "dropout": model.get("dropout"),
        "activation": model.get("activation"),
        "optimizer": training.get("optimizer"),
        "lr": training.get("lr"),
        "weight_decay": training.get("weight_decay"),
        "batch_size": training.get("batch_size"),
        "fixed_epochs": training.get("fixed_epochs") or training.get("max_epochs"),
        "merge_val_into_train": training.get("merge_val_into_train"),
        "no_validation": training.get("no_validation"),
        "selection_mode": training.get("selection_mode"),
    }


def artifact_issues(run_dir: Path, payload: dict[str, Any] | None) -> list[str]:
    issues: list[str] = []
    final_checkpoint = run_dir / "final_checkpoint.pt"
    if payload:
        final_checkpoint = Path(payload.get("final_checkpoint") or final_checkpoint)
    if not final_checkpoint.is_file():
        issues.append(f"missing final checkpoint: {final_checkpoint}")
    for name in ("predictions_train.csv", "predictions_test.csv"):
        if not (run_dir / name).is_file():
            issues.append(f"missing prediction file: {run_dir / name}")
    return issues


def run_record(
    metrics_path: Path,
    *,
    branch: str,
    exp: str,
    lr_name: str,
    lr: float,
    fold: int,
) -> dict[str, Any]:
    base = {
        "stage": "cv",
        "branch": branch,
        "exp": exp,
        "lr_name": lr_name,
        "lr": lr,
        "fold": fold,
        "source": str(metrics_path),
    }
    if not metrics_path.exists():
        return {**base, "run_status": "missing", "artifact_issues": [f"missing metrics: {metrics_path}"]}
    payload = load_json(metrics_path)
    test_metrics = metric_payload((payload.get("splits") or {}).get("test"))
    train_metrics = metric_payload((payload.get("splits") or {}).get("train"))
    best = payload.get("best") or {}
    record = {
        **base,
        "run_status": payload.get("run_status"),
        "final_epoch": best.get("epoch"),
        "monitor": best.get("monitor"),
        "train_loss": train_metrics.get("loss"),
        "train_auroc": train_metrics.get("auroc"),
        "train_auprc": train_metrics.get("auprc"),
        "test_loss": test_metrics.get("loss"),
        "test_auroc": test_metrics.get("auroc"),
        "test_auprc": test_metrics.get("auprc"),
        "test_auprc_baseline": test_metrics.get("auprc_baseline"),
        "test_nauprc": test_metrics.get("nauprc"),
        "test_acc": test_metrics.get("acc"),
        "test_count": test_metrics.get("count"),
        "test_pos": test_metrics.get("positive_count"),
        "test_neg": test_metrics.get("negative_count"),
        "artifact_issues": artifact_issues(metrics_path.parent, payload),
    }
    record.update(config_fields(payload))
    return record


def collect_cv_records(prefix: str, output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cv_root = output_root / prefix / "cv"
    for branch in BRANCHES:
        for exp in EXPS:
            for lr_name, lr in LR_GRID:
                for fold in FOLDS:
                    metrics_path = cv_root / branch / exp / lr_name / f"fold{fold}" / "metrics.json"
                    records.append(run_record(metrics_path, branch=branch, exp=exp, lr_name=lr_name, lr=lr, fold=fold))
    return records


def summarize_lr(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["branch"]), str(record["exp"]), str(record["lr_name"]))].append(record)
    rows: list[dict[str, Any]] = []
    for (branch, exp, lr_name), items in sorted(grouped.items()):
        ok_items = [item for item in items if item.get("run_status") == "ok"]
        first = ok_items[0] if ok_items else items[0]
        row = {
            "branch": branch,
            "exp": exp,
            "lr_name": lr_name,
            "lr": first.get("lr"),
            "ok_folds": len(ok_items),
            "expected_folds": len(FOLDS),
            "mean_test_auroc": mean_finite([item.get("test_auroc") for item in ok_items]),
            "mean_test_auprc": mean_finite([item.get("test_auprc") for item in ok_items]),
            "mean_test_auprc_baseline": mean_finite([item.get("test_auprc_baseline") for item in ok_items]),
            "mean_test_nauprc": mean_finite([item.get("test_nauprc") for item in ok_items]),
            "count": sum_finite([item.get("test_count") for item in ok_items]),
            "pos": sum_finite([item.get("test_pos") for item in ok_items]),
            "neg": sum_finite([item.get("test_neg") for item in ok_items]),
            "status": "ok" if len(ok_items) == len(FOLDS) else "incomplete",
        }
        for key in (
            "hidden_dim",
            "drug_hidden_dim",
            "fusion_hidden_dim",
            "dropout",
            "activation",
            "optimizer",
            "weight_decay",
            "batch_size",
            "fixed_epochs",
            "merge_val_into_train",
            "no_validation",
            "selection_mode",
        ):
            row[key] = first.get(key)
        rows.append(row)

    for branch in BRANCHES:
        for exp in EXPS:
            group = [row for row in rows if row.get("branch") == branch and row.get("exp") == exp]
            group.sort(key=summary_rank_key, reverse=True)
            for rank, row in enumerate(group, start=1):
                row["rank_by_mean_test_auprc"] = rank
    return rows


def summary_rank_key(row: dict[str, Any]) -> tuple[int, float, float, float, str]:
    complete = 1 if row.get("status") == "ok" else 0
    auprc = finite(row.get("mean_test_auprc"))
    nauprc = finite(row.get("mean_test_nauprc"))
    auroc = finite(row.get("mean_test_auroc"))
    return (
        complete,
        auprc if auprc is not None else -math.inf,
        nauprc if nauprc is not None else -math.inf,
        auroc if auroc is not None else -math.inf,
        str(row.get("lr_name")),
    )


def select_best_lr(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for branch in BRANCHES:
        for exp in EXPS:
            candidates = [row for row in summary if row.get("branch") == branch and row.get("exp") == exp]
            if not candidates:
                selected.append({"branch": branch, "exp": exp, "status": "missing"})
                continue
            best = max(candidates, key=summary_rank_key).copy()
            best["selection_metric"] = "mean5_test_auprc"
            best["selection_note"] = "test-selected"
            selected.append(best)
    return selected


def best_fold_records(records: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_lr = {
        (str(row.get("branch")), str(row.get("exp"))): str(row.get("lr_name"))
        for row in selected
        if row.get("lr_name")
    }
    for record in records:
        key = (str(record.get("branch")), str(record.get("exp")))
        if record.get("lr_name") == selected_lr.get(key):
            rows.append(record.copy())
    return sorted(rows, key=lambda item: (str(item.get("branch")), str(item.get("exp")), int(item.get("fold", -1))))


def anomalies(records: list[dict[str, Any]], summary: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for row in summary:
        if row.get("status") != "ok":
            issues.append(
                "incomplete LR: {branch} {exp} {lr_name} ok_folds={ok}/{expected}".format(
                    branch=row.get("branch"),
                    exp=row.get("exp"),
                    lr_name=row.get("lr_name"),
                    ok=row.get("ok_folds"),
                    expected=row.get("expected_folds"),
                )
            )
    for row in selected:
        if row.get("status") != "ok":
            issues.append(f"selected LR incomplete/missing: {row.get('branch')} {row.get('exp')} {row.get('lr_name')}")
    for record in records:
        label = "{branch} {exp} {lr_name} fold{fold}".format(**record)
        if record.get("run_status") != "ok":
            issues.append(f"non-OK fold: {label} status={record.get('run_status')} source={record.get('source')}")
        for item in record.get("artifact_issues") or []:
            issues.append(f"{label}: {item}")
        if finite(record.get("test_pos")) == 0 or finite(record.get("test_neg")) == 0:
            issues.append(f"one-class test labels: {label} pos={record.get('test_pos')} neg={record.get('test_neg')}")
        if record.get("run_status") == "ok":
            for metric_name in ("test_auroc", "test_auprc", "test_auprc_baseline", "test_nauprc"):
                if finite(record.get(metric_name)) is None:
                    issues.append(f"NaN/Inf metric: {label} {metric_name}={record.get(metric_name)}")
    return issues


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def selected_lr_payload(prefix: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, dict[str, Any]] = {branch: {} for branch in BRANCHES}
    for row in selected:
        branch = str(row.get("branch"))
        exp = str(row.get("exp"))
        payload.setdefault(branch, {})[exp] = {
            "lr_name": row.get("lr_name"),
            "lr": row.get("lr"),
            "status": row.get("status"),
            "selection_metric": row.get("selection_metric"),
            "selection_note": row.get("selection_note"),
            "mean_test_auprc": row.get("mean_test_auprc"),
            "ok_folds": row.get("ok_folds"),
            "expected_folds": row.get("expected_folds"),
        }
    return {"prefix": prefix, "generated_at": now_text(), "selected": payload}


def write_markdown(
    path: Path,
    *,
    prefix: str,
    selected: list[dict[str, Any]],
    all_lr_summary: list[dict[str, Any]],
    best_folds: list[dict[str, Any]],
    issues: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed_folds = sum(int(row.get("ok_folds") or 0) for row in all_lr_summary)
    expected_folds = sum(int(row.get("expected_folds") or 0) for row in all_lr_summary)
    is_complete = completed_folds == expected_folds and not issues
    status_text = "COMPLETE" if is_complete else "INTERIM / INCOMPLETE"
    title_suffix = "" if is_complete else " (Interim Incomplete Snapshot)"
    lines = [
        f"# 2026-06-29 Transcriptome MLP PTV2-Benchmark-Aligned LR CV Results{title_suffix}",
        "",
        f"Generated: `{now_text()}`.",
        f"Prefix: `{prefix}`.",
        "",
        "## Completion Status",
        "",
        f"- Status: `{status_text}`.",
        f"- Completed CV folds: `{completed_folds}/{expected_folds}`.",
        f"- Anomaly count: `{len(issues)}`.",
    ]
    if not is_complete:
        lines.extend(
            [
                "- This file is a progress snapshot while the tmux CV run is still active.",
                "- Do not treat the best-LR or final mean5 tables as final until all `150/150` folds are complete and anomaly count is `0`.",
                "- Re-run the reporter after CV completion to overwrite this file with the final report.",
            ]
        )
    lines.extend(
        [
        "",
        "## Setup",
        "",
        "- Input transcriptome: CPA/BioLord prediction h5ad `.X`, 9843 common-HGNC generated transcriptome features.",
        "- Drug input: Morgan fingerprint, 2048 bits, keyed by `drug_id` with SMILES fallback.",
        "- Label: `PRISM1st_label_total`, `sensitive=1`, `non-responsive=0`.",
        "- Split policy: each LR runs full 5-fold CV; each fold trains on `train+val` and evaluates `test`.",
        "- Scaler policy: expression mean/std is fit on merged `train+val` only, then applied to test.",
        "- LR selection: best LR for each branch+exp is selected by mean5 test AUPRC. This is a test-selected best result, not an independent holdout estimate.",
        "",
        "## PTV2 Benchmark Alignment",
        "",
        "- Aligned parameters: `hidden_dim=64`, `drug_hidden_dim=32`, `fusion_hidden_dim=32`, `dropout=0.0`, `activation=relu`, `batch_size=64`, `optimizer=AdamW`, `weight_decay=0.01`, `fixed_epochs=150`.",
        "- LR grid: `5e-5`, `1e-4`, `2e-4`, `5e-4`, `1e-3`.",
        "- Not included: PTV1/benchmark ODE dynamics, proteomics reconstruction MSE, SWAG, validation scheduler, early stopping, control/covariate/graph/dose/time inputs.",
        "",
        "## Best LR Selection",
        "",
        "Selection is final only when completion status is `COMPLETE`.",
        "",
        "| branch | exp | best LR | folds | mean5 AUROC | mean5 AUPRC | baseline | n-AUPRC | count | pos | neg | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected:
        lines.append(
            "| {branch} | {exp} | {lr} | {folds}/{expected} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | {pos} | {neg} | {status} |".format(
                branch=row.get("branch", ""),
                exp=row.get("exp", ""),
                lr=row.get("lr", ""),
                folds=row.get("ok_folds", ""),
                expected=row.get("expected_folds", ""),
                auroc=fmt(row.get("mean_test_auroc")),
                auprc=fmt(row.get("mean_test_auprc")),
                baseline=fmt(row.get("mean_test_auprc_baseline")),
                nauprc=fmt(row.get("mean_test_nauprc")),
                count=int_text(row.get("count")),
                pos=int_text(row.get("pos")),
                neg=int_text(row.get("neg")),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Final Mean5 Test",
            "",
            "| branch | exp | LR | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        lines.append(
            "| {branch} | {exp} | {lr} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | {pos} | {neg} |".format(
                branch=row.get("branch", ""),
                exp=row.get("exp", ""),
                lr=row.get("lr", ""),
                auroc=fmt(row.get("mean_test_auroc")),
                auprc=fmt(row.get("mean_test_auprc")),
                baseline=fmt(row.get("mean_test_auprc_baseline")),
                nauprc=fmt(row.get("mean_test_nauprc")),
                count=int_text(row.get("count")),
                pos=int_text(row.get("pos")),
                neg=int_text(row.get("neg")),
            )
        )
    lines.extend(
        [
            "",
            "## Best LR Fold Detail",
            "",
            "| branch | exp | fold | LR | epoch | test AUROC | test AUPRC | baseline | n-AUPRC | count | pos | neg |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_folds:
        lines.append(
            "| {branch} | {exp} | {fold} | {lr} | {epoch} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | {pos} | {neg} |".format(
                branch=row.get("branch", ""),
                exp=row.get("exp", ""),
                fold=row.get("fold", ""),
                lr=row.get("lr", ""),
                epoch=row.get("final_epoch", ""),
                auroc=fmt(row.get("test_auroc")),
                auprc=fmt(row.get("test_auprc")),
                baseline=fmt(row.get("test_auprc_baseline")),
                nauprc=fmt(row.get("test_nauprc")),
                count=int_text(row.get("test_count")),
                pos=int_text(row.get("test_pos")),
                neg=int_text(row.get("test_neg")),
            )
        )
    lines.extend(
        [
            "",
            "## All LR Mean5 Test Summary",
            "",
            "| branch | exp | rank | LR | folds | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg | status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    ordered_summary = sorted(
        all_lr_summary,
        key=lambda row: (str(row.get("branch")), str(row.get("exp")), int(row.get("rank_by_mean_test_auprc") or 999)),
    )
    for row in ordered_summary:
        lines.append(
            "| {branch} | {exp} | {rank} | {lr} | {folds}/{expected} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | {pos} | {neg} | {status} |".format(
                branch=row.get("branch", ""),
                exp=row.get("exp", ""),
                rank=row.get("rank_by_mean_test_auprc", ""),
                lr=row.get("lr", ""),
                folds=row.get("ok_folds", ""),
                expected=row.get("expected_folds", ""),
                auroc=fmt(row.get("mean_test_auroc")),
                auprc=fmt(row.get("mean_test_auprc")),
                baseline=fmt(row.get("mean_test_auprc_baseline")),
                nauprc=fmt(row.get("mean_test_nauprc")),
                count=int_text(row.get("count")),
                pos=int_text(row.get("pos")),
                neg=int_text(row.get("neg")),
                status=row.get("status", ""),
            )
        )
    lines.extend(["", "## Exception Summary", ""])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No missing files, non-OK folds, one-class test folds, or NaN/Inf metric values detected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(prefix: str, output_root: Path) -> dict[str, Any]:
    records = collect_cv_records(prefix, output_root)
    all_lr_summary = summarize_lr(records)
    selected = select_best_lr(all_lr_summary)
    best_folds = best_fold_records(records, selected)
    issue_list = anomalies(records, all_lr_summary, selected)
    return {
        "prefix": prefix,
        "generated_at": now_text(),
        "records": records,
        "all_lr_summary": all_lr_summary,
        "selected": selected,
        "best_fold_records": best_folds,
        "anomalies": issue_list,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=REPO_ROOT / "docs/2026-06-29_transcriptome_mlp_ptv2benchmark_lr_cv_results.md",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--all-lr-summary-csv-out", type=Path)
    parser.add_argument("--best-lr-summary-csv-out", type=Path)
    parser.add_argument("--best-fold-csv-out", type=Path)
    parser.add_argument("--all-fold-csv-out", type=Path)
    parser.add_argument("--selected-lr-out", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if report anomalies are present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.prefix, args.output_root)
    json_out = args.json_out or args.output_root / f"{args.prefix}_ptv2benchmark_lr_cv_report.json"
    all_lr_summary_csv = args.all_lr_summary_csv_out or args.output_root / f"{args.prefix}_ptv2benchmark_lr_cv_all_lr_summary.csv"
    best_lr_summary_csv = args.best_lr_summary_csv_out or args.output_root / f"{args.prefix}_ptv2benchmark_lr_cv_best_lr_summary.csv"
    best_fold_csv = args.best_fold_csv_out or args.output_root / f"{args.prefix}_ptv2benchmark_lr_cv_best_folds.csv"
    all_fold_csv = args.all_fold_csv_out or args.output_root / f"{args.prefix}_ptv2benchmark_lr_cv_all_folds.csv"
    selected_lr_out = args.selected_lr_out or args.output_root / f"{args.prefix}_ptv2benchmark_lr_cv_selected_lrs.json"

    write_csv(all_lr_summary_csv, report["all_lr_summary"])
    write_csv(best_lr_summary_csv, report["selected"])
    write_csv(best_fold_csv, report["best_fold_records"])
    write_csv(all_fold_csv, report["records"])
    write_json(json_out, report)
    write_json(selected_lr_out, selected_lr_payload(args.prefix, report["selected"]))
    write_markdown(
        args.markdown_out,
        prefix=args.prefix,
        selected=report["selected"],
        all_lr_summary=report["all_lr_summary"],
        best_folds=report["best_fold_records"],
        issues=report["anomalies"],
    )
    print(f"[report] markdown={args.markdown_out}")
    print(f"[report] json={json_out}")
    print(f"[report] selected_lrs={selected_lr_out}")
    print(f"[report] anomalies={len(report['anomalies'])}")
    if args.strict and report["anomalies"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
