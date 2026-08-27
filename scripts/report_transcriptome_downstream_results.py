#!/usr/bin/env python3
"""Summarize transcriptome downstream MLP tuning and final test results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = "20260626_transcriptome_mlp_official_fixed_residual_v1"
BRANCHES = ("cpa", "biolord")
EXPS = ("exp01", "exp02", "exp03")
TUNE_FOLDS = (0, 2, 4)
FINAL_FOLDS = (0, 1, 2, 3, 4)
METRIC_COLUMNS = ("auroc", "auprc", "auprc_baseline", "nauprc")


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_finite(values: list[Any]) -> float | None:
    finite_values = [number for value in values if (number := finite(value)) is not None]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


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
        "lr": training.get("lr"),
        "weight_decay": training.get("weight_decay"),
        "batch_size": training.get("batch_size"),
        "max_epochs": training.get("max_epochs"),
        "patience": training.get("patience"),
    }


def run_record(path: Path, *, branch: str, exp: str, fold: int, config_name: str | None, stage: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "stage": stage,
            "branch": branch,
            "exp": exp,
            "fold": fold,
            "config_name": config_name,
            "run_status": "missing",
            "source": str(path),
        }
    payload = load_json(path)
    val_metrics = metric_payload((payload.get("splits") or {}).get("val"))
    test_metrics = metric_payload((payload.get("splits") or {}).get("test"))
    best = payload.get("best") or {}
    record = {
        "stage": stage,
        "branch": branch,
        "exp": exp,
        "fold": fold,
        "config_name": config_name or payload.get("config_name"),
        "run_status": payload.get("run_status"),
        "best_epoch": best.get("epoch"),
        "val_loss": val_metrics.get("loss"),
        "val_auroc": val_metrics.get("auroc"),
        "val_auprc": val_metrics.get("auprc"),
        "val_nauprc": val_metrics.get("nauprc"),
        "test_loss": test_metrics.get("loss"),
        "test_auroc": test_metrics.get("auroc"),
        "test_auprc": test_metrics.get("auprc"),
        "test_auprc_baseline": test_metrics.get("auprc_baseline"),
        "test_nauprc": test_metrics.get("nauprc"),
        "test_count": test_metrics.get("count"),
        "test_pos": test_metrics.get("positive_count"),
        "test_neg": test_metrics.get("negative_count"),
        "source": str(path),
    }
    record.update(config_fields(payload))
    return record


def collect_tune_records(prefix: str, output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tune_root = output_root / prefix / "tune"
    for branch in BRANCHES:
        for exp in EXPS:
            exp_root = tune_root / branch / exp
            if not exp_root.exists():
                continue
            for config_dir in sorted(path for path in exp_root.iterdir() if path.is_dir()):
                for fold in TUNE_FOLDS:
                    metrics_path = config_dir / f"fold{fold}" / "metrics.json"
                    records.append(
                        run_record(
                            metrics_path,
                            branch=branch,
                            exp=exp,
                            fold=fold,
                            config_name=config_dir.name,
                            stage="tune",
                        )
                    )
    return records


def summarize_tune(records: list[dict[str, Any]], *, expected_folds: int = len(TUNE_FOLDS)) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["branch"]), str(record["exp"]), str(record["config_name"]))].append(record)
    rows: list[dict[str, Any]] = []
    for (branch, exp, config_name), items in sorted(grouped.items()):
        ok_items = [item for item in items if item.get("run_status") == "ok"]
        first_ok = ok_items[0] if ok_items else items[0]
        row = {
            "branch": branch,
            "exp": exp,
            "config_name": config_name,
            "folds": ",".join(str(item["fold"]) for item in sorted(items, key=lambda item: int(item["fold"]))),
            "ok_folds": len(ok_items),
            "expected_folds": expected_folds,
            "mean_val_auprc": mean_finite([item.get("val_auprc") for item in ok_items]),
            "mean_val_nauprc": mean_finite([item.get("val_nauprc") for item in ok_items]),
            "mean_val_auroc": mean_finite([item.get("val_auroc") for item in ok_items]),
            "mean_val_loss": mean_finite([item.get("val_loss") for item in ok_items]),
            "status": "ok" if len(ok_items) == expected_folds else "incomplete",
            "sources": [item.get("source") for item in items],
        }
        for key in (
            "hidden_dim",
            "drug_hidden_dim",
            "fusion_hidden_dim",
            "dropout",
            "activation",
            "lr",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
        ):
            row[key] = first_ok.get(key)
        rows.append(row)
    return rows


def winner_key(row: dict[str, Any]) -> tuple[int, float, float, float, float, str]:
    complete = 1 if row.get("status") == "ok" else 0
    val_auprc = finite(row.get("mean_val_auprc"))
    val_nauprc = finite(row.get("mean_val_nauprc"))
    val_auroc = finite(row.get("mean_val_auroc"))
    val_loss = finite(row.get("mean_val_loss"))
    return (
        complete,
        val_auprc if val_auprc is not None else -math.inf,
        val_nauprc if val_nauprc is not None else -math.inf,
        val_auroc if val_auroc is not None else -math.inf,
        -(val_loss if val_loss is not None else math.inf),
        str(row.get("config_name")),
    )


def select_winners(tune_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for branch in BRANCHES:
        for exp in EXPS:
            candidates = [row for row in tune_summary if row.get("branch") == branch and row.get("exp") == exp]
            if not candidates:
                winners.append({"branch": branch, "exp": exp, "status": "missing"})
                continue
            winner = max(candidates, key=winner_key).copy()
            winner["status"] = "ok" if winner.get("status") == "ok" else "incomplete"
            winners.append(winner)
    return winners


def selected_config_payload(prefix: str, winners: list[dict[str, Any]]) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {branch: {} for branch in BRANCHES}
    for winner in winners:
        branch = str(winner.get("branch"))
        exp = str(winner.get("exp"))
        selected.setdefault(branch, {})[exp] = {
            "config_name": winner.get("config_name"),
            "status": winner.get("status"),
            "selection_metrics": {
                "mean_val_auprc": winner.get("mean_val_auprc"),
                "mean_val_nauprc": winner.get("mean_val_nauprc"),
                "mean_val_auroc": winner.get("mean_val_auroc"),
                "mean_val_loss": winner.get("mean_val_loss"),
                "ok_folds": winner.get("ok_folds"),
                "expected_folds": winner.get("expected_folds"),
            },
            "params": {
                "hidden_dim": winner.get("hidden_dim"),
                "drug_hidden_dim": winner.get("drug_hidden_dim"),
                "fusion_hidden_dim": winner.get("fusion_hidden_dim"),
                "dropout": winner.get("dropout"),
                "activation": winner.get("activation"),
                "lr": winner.get("lr"),
                "weight_decay": winner.get("weight_decay"),
                "batch_size": winner.get("batch_size"),
            },
        }
    return {"prefix": prefix, "generated_at": now_text(), "selected": selected}


def collect_final_records(prefix: str, output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    final_root = output_root / prefix / "final"
    for branch in BRANCHES:
        for exp in EXPS:
            for fold in FINAL_FOLDS:
                metrics_path = final_root / branch / exp / f"fold{fold}" / "metrics.json"
                records.append(
                    run_record(metrics_path, branch=branch, exp=exp, fold=fold, config_name=None, stage="final")
                )
    return records


def summarize_final(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["branch"]), str(record["exp"]))].append(record)
    rows: list[dict[str, Any]] = []
    for (branch, exp), items in sorted(grouped.items()):
        ok_items = [item for item in items if item.get("run_status") == "ok"]
        rows.append(
            {
                "branch": branch,
                "exp": exp,
                "split": "mean5",
                "ok_folds": len(ok_items),
                "expected_folds": len(FINAL_FOLDS),
                "auroc": mean_finite([item.get("test_auroc") for item in ok_items]),
                "auprc": mean_finite([item.get("test_auprc") for item in ok_items]),
                "auprc_baseline": mean_finite([item.get("test_auprc_baseline") for item in ok_items]),
                "nauprc": mean_finite([item.get("test_nauprc") for item in ok_items]),
                "count": sum_finite([item.get("test_count") for item in ok_items]),
                "pos": sum_finite([item.get("test_pos") for item in ok_items]),
                "neg": sum_finite([item.get("test_neg") for item in ok_items]),
                "status": "ok" if len(ok_items) == len(FINAL_FOLDS) else "incomplete",
            }
        )
    return rows


def anomalies(tune_summary: list[dict[str, Any]], winners: list[dict[str, Any]], final_records: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for winner in winners:
        if winner.get("status") != "ok":
            issues.append(f"tune winner incomplete/missing: {winner.get('branch')} {winner.get('exp')}")
    for row in tune_summary:
        if row.get("status") != "ok":
            issues.append(
                "tune incomplete: {branch} {exp} {config_name} ok_folds={ok}/{expected}".format(
                    branch=row.get("branch"),
                    exp=row.get("exp"),
                    config_name=row.get("config_name"),
                    ok=row.get("ok_folds"),
                    expected=row.get("expected_folds"),
                )
            )
    for record in final_records:
        if record.get("run_status") != "ok":
            issues.append(
                "final not ok: {branch} {exp} fold{fold} status={status} source={source}".format(
                    branch=record.get("branch"),
                    exp=record.get("exp"),
                    fold=record.get("fold"),
                    status=record.get("run_status"),
                    source=record.get("source"),
                )
            )
        if finite(record.get("test_pos")) == 0 or finite(record.get("test_neg")) == 0:
            issues.append(
                "final one-class test labels: {branch} {exp} fold{fold} pos={pos} neg={neg}".format(
                    branch=record.get("branch"),
                    exp=record.get("exp"),
                    fold=record.get("fold"),
                    pos=record.get("test_pos"),
                    neg=record.get("test_neg"),
                )
            )
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


def write_markdown(
    path: Path,
    *,
    prefix: str,
    tune_summary: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    final_summary: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    issues: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 2026-06-26 Transcriptome MLP exp01-exp03 Results",
        "",
        f"Generated: `{now_text()}`.",
        f"Prefix: `{prefix}`.",
        "",
        "## Setup",
        "",
        "- Input transcriptome: CPA/BioLord prediction h5ad `.X`, 9843 common-HGNC features.",
        "- Drug input: Morgan fingerprint, 2048 bits, keyed by `drug_id` with SMILES fallback.",
        "- Label: `PRISM1st_label_total`, `sensitive=1`, `non-responsive=0`.",
        "- Model: independent transcriptome and drug towers, concat fusion, one BCE logit.",
        "- Validation selection: mean validation AUPRC across screen folds `0,2,4`; tie-breakers are n-AUPRC, AUROC, then validation loss.",
        "",
        "## Tuning Winners",
        "",
        "| branch | exp | winner | folds | val AUROC | val AUPRC | val n-AUPRC | val loss | hidden | drug hidden | dropout | lr | weight decay |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in winners:
        lines.append(
            "| {branch} | {exp} | {config} | {folds}/{expected} | {auroc} | {auprc} | {nauprc} | {loss} | {hidden} | {drug_hidden} | {dropout} | {lr} | {wd} |".format(
                branch=row.get("branch", ""),
                exp=row.get("exp", ""),
                config=row.get("config_name", ""),
                folds=row.get("ok_folds", ""),
                expected=row.get("expected_folds", ""),
                auroc=fmt(row.get("mean_val_auroc")),
                auprc=fmt(row.get("mean_val_auprc")),
                nauprc=fmt(row.get("mean_val_nauprc")),
                loss=fmt(row.get("mean_val_loss")),
                hidden=row.get("hidden_dim", ""),
                drug_hidden=row.get("drug_hidden_dim", ""),
                dropout=row.get("dropout", ""),
                lr=row.get("lr", ""),
                wd=row.get("weight_decay", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Final Mean5 Test",
            "",
            "| branch | exp | folds | AUROC | AUPRC | baseline | n-AUPRC | count | pos | neg |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_summary:
        lines.append(
            "| {branch} | {exp} | {folds}/{expected} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | {pos} | {neg} |".format(
                branch=row.get("branch", ""),
                exp=row.get("exp", ""),
                folds=row.get("ok_folds", ""),
                expected=row.get("expected_folds", ""),
                auroc=fmt(row.get("auroc")),
                auprc=fmt(row.get("auprc")),
                baseline=fmt(row.get("auprc_baseline")),
                nauprc=fmt(row.get("nauprc")),
                count=int_text(row.get("count")),
                pos=int_text(row.get("pos")),
                neg=int_text(row.get("neg")),
            )
        )
    lines.extend(
        [
            "",
            "## Final Fold Detail",
            "",
            "| branch | exp | fold | config | best epoch | val AUPRC | test AUROC | test AUPRC | baseline | n-AUPRC | count | pos | neg |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(final_records, key=lambda item: (str(item.get("branch")), str(item.get("exp")), int(item.get("fold", -1)))):
        lines.append(
            "| {branch} | {exp} | {fold} | {config} | {epoch} | {val_auprc} | {auroc} | {auprc} | {baseline} | {nauprc} | {count} | {pos} | {neg} |".format(
                branch=row.get("branch", ""),
                exp=row.get("exp", ""),
                fold=row.get("fold", ""),
                config=row.get("config_name") or "",
                epoch=row.get("best_epoch") or "",
                val_auprc=fmt(row.get("val_auprc")),
                auroc=fmt(row.get("test_auroc")),
                auprc=fmt(row.get("test_auprc")),
                baseline=fmt(row.get("test_auprc_baseline")),
                nauprc=fmt(row.get("test_nauprc")),
                count=int_text(row.get("test_count")),
                pos=int_text(row.get("test_pos")),
                neg=int_text(row.get("test_neg")),
            )
        )
    lines.extend(["", "## Exception Summary", ""])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No missing, non-OK, or one-class final folds detected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(prefix: str, output_root: Path) -> dict[str, Any]:
    tune_records = collect_tune_records(prefix, output_root)
    tune_summary = summarize_tune(tune_records)
    winners = select_winners(tune_summary)
    final_records = collect_final_records(prefix, output_root)
    final_summary = summarize_final(final_records)
    issue_list = anomalies(tune_summary, winners, final_records)
    return {
        "prefix": prefix,
        "generated_at": now_text(),
        "tune_records": tune_records,
        "tune_summary": tune_summary,
        "winners": winners,
        "final_records": final_records,
        "final_summary": final_summary,
        "anomalies": issue_list,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--markdown-out", type=Path, default=REPO_ROOT / "docs/2026-06-26_transcriptome_mlp_exp01_03_results.md")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--tune-csv-out", type=Path)
    parser.add_argument("--final-summary-csv-out", type=Path)
    parser.add_argument("--final-fold-csv-out", type=Path)
    parser.add_argument("--selected-configs-out", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if report anomalies are present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.prefix, args.output_root)
    json_out = args.json_out or args.output_root / f"{args.prefix}_transcriptome_mlp_report.json"
    tune_csv = args.tune_csv_out or args.output_root / f"{args.prefix}_transcriptome_mlp_tune_summary.csv"
    final_summary_csv = args.final_summary_csv_out or args.output_root / f"{args.prefix}_transcriptome_mlp_final_summary.csv"
    final_fold_csv = args.final_fold_csv_out or args.output_root / f"{args.prefix}_transcriptome_mlp_final_folds.csv"
    selected_configs_out = args.selected_configs_out or args.output_root / f"{args.prefix}_transcriptome_mlp_selected_configs.json"

    write_csv(tune_csv, report["tune_summary"])
    write_csv(final_summary_csv, report["final_summary"])
    write_csv(final_fold_csv, report["final_records"])
    write_json(json_out, report)
    write_json(selected_configs_out, selected_config_payload(args.prefix, report["winners"]))
    write_markdown(
        args.markdown_out,
        prefix=args.prefix,
        tune_summary=report["tune_summary"],
        winners=report["winners"],
        final_summary=report["final_summary"],
        final_records=report["final_records"],
        issues=report["anomalies"],
    )
    print(f"[report] markdown={args.markdown_out}")
    print(f"[report] json={json_out}")
    print(f"[report] selected_configs={selected_configs_out}")
    print(f"[report] anomalies={len(report['anomalies'])}")
    if args.strict and report["anomalies"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
