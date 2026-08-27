#!/usr/bin/env python3
"""Summarize corrected Cell LLM clip10 parameter-search stages."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scripts" / "report_ptv3_exp_results.py"
STAGES = ("stage1", "stage2", "stage3", "stage4")
PROMOTION_PRIMARY_MIN = 3
PROMOTION_MAX = 5
DEFAULT_PROMOTION_MARGIN = 0.002


def load_report_module() -> Any:
    spec = importlib.util.spec_from_file_location("ptv3_report", REPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--format", choices=("markdown", "tsv", "json"), default="markdown")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--min-folds", type=int, default=1)
    parser.add_argument("--promotion-margin", type=float, default=DEFAULT_PROMOTION_MARGIN)
    parser.add_argument("--emit-promoted-configs", action="store_true")
    parser.add_argument("--emit-selected-config", action="store_true")
    return parser.parse_args()


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


def collect_prefixes(args: argparse.Namespace) -> list[tuple[str, str]]:
    pattern = re.compile(rf"^{re.escape(args.base_prefix)}_(stage[1-4])_(.+)_exp0[1-6]_")
    found: set[tuple[str, str]] = set()
    for root in (args.checkpoint_root, args.output_root):
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            stage, config = match.group(1), match.group(2)
            if args.stage != "all" and stage != args.stage:
                continue
            found.add((stage, config))
    return sorted(found)


def collect_records(report: Any, args: argparse.Namespace, stage: str, config: str) -> list[dict[str, Any]]:
    ns = argparse.Namespace(
        prefix=f"{args.base_prefix}_{stage}_{config}",
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
        exp08_root=None,
    )
    return report.collect_records(ns)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def manifest_errors(source: Any, expected_graph_mode: str | None) -> list[str]:
    path = Path(str(source))
    if not path.exists() or path.name != "run_manifest.json":
        return [f"missing manifest {path}"]
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    if "cell_type_llm" in text:
        errors.append("old cell_type_llm key")
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid manifest JSON: {exc}"]
    if manifest.get("run_status") != "fit_completed":
        errors.append(f"run_status={manifest.get('run_status')!r}")
    if manifest.get("test_status") != "test_completed":
        errors.append(f"test_status={manifest.get('test_status')!r}")
    if manifest.get("cell_llm_mode") != "frozen":
        errors.append(f"cell_llm_mode={manifest.get('cell_llm_mode')!r}")
    summary = manifest.get("cell_llm_summary")
    if not isinstance(summary, dict) or summary.get("embedding_rows") != 74:
        errors.append("cell_llm_summary.embedding_rows!=74")
    if expected_graph_mode is not None and manifest.get("graph_feature_mode") != expected_graph_mode:
        errors.append(f"graph_feature_mode={manifest.get('graph_feature_mode')!r}")
    return errors


def aggregate_fold_records(
    records: list[dict[str, Any]],
    exp: str,
    *,
    expected_graph_mode: str | None,
) -> dict[str, Any] | None:
    fold_rows = [
        row
        for row in records
        if row.get("exp") == exp and row.get("task") == exp and str(row.get("split", "")).startswith("fold")
    ]
    if not fold_rows:
        return None

    aggregate: dict[str, Any] = {
        "exp": exp,
        "task": exp,
        "fold_count": len(fold_rows),
        "folds": ",".join(sorted(str(row.get("split", "")) for row in fold_rows)),
    }
    for key in ("auprc", "auprc_baseline", "nauprc", "auroc", "acc"):
        values = [value for row in fold_rows if (value := finite(row.get(key))) is not None]
        aggregate[key] = sum(values) / len(values) if values else None
    aggregate["count"] = sum(finite(row.get("count")) or 0.0 for row in fold_rows)

    errors: list[str] = []
    for row in fold_rows:
        source_errors = manifest_errors(row.get("source"), expected_graph_mode)
        errors.extend(f"{row.get('split')}: {error}" for error in source_errors)
    aggregate["validation_errors"] = errors
    return aggregate


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    return finite(row.get(key))


def fold_count(row: dict[str, Any] | None) -> int:
    if row is None:
        return 0
    try:
        return int(row.get("fold_count") or 0)
    except (TypeError, ValueError):
        return 0


def validation_errors(*rows: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    for row in rows:
        if not row:
            continue
        errors.extend(str(item) for item in row.get("validation_errors", []))
    return errors


def stage1_row(stage: str, config: str, records: list[dict[str, Any]], min_folds: int) -> dict[str, Any]:
    exp01 = aggregate_fold_records(records, "exp01", expected_graph_mode="real")
    exp04 = aggregate_fold_records(records, "exp04", expected_graph_mode="real")
    exp05 = aggregate_fold_records(records, "exp05", expected_graph_mode="zero")
    exp01_auprc = metric(exp01, "auprc")
    exp04_auprc = metric(exp04, "auprc")
    exp05_auprc = metric(exp05, "auprc")
    gap_no_mse = None if exp01_auprc is None or exp04_auprc is None else exp01_auprc - exp04_auprc
    gap_no_graph = None if exp01_auprc is None or exp05_auprc is None else exp01_auprc - exp05_auprc
    score = None
    if exp01_auprc is not None and gap_no_mse is not None and gap_no_graph is not None:
        score = exp01_auprc + 0.5 * gap_no_mse + 0.5 * gap_no_graph
    errors = validation_errors(exp01, exp04, exp05)
    complete = fold_count(exp01) >= min_folds and fold_count(exp04) >= min_folds and fold_count(exp05) >= min_folds
    return {
        "stage": stage,
        "config": config,
        "task": "exp01_exp04_exp05",
        "folds": min(fold_count(exp01), fold_count(exp04), fold_count(exp05)),
        "complete": complete,
        "valid_cell_llm": not errors,
        "auprc": exp01_auprc,
        "nauprc": metric(exp01, "nauprc"),
        "auroc": metric(exp01, "auroc"),
        "exp04_auprc": exp04_auprc,
        "exp05_auprc": exp05_auprc,
        "gap_no_mse_auprc": gap_no_mse,
        "gap_no_graph_auprc": gap_no_graph,
        "score": score,
        "rank_metric": score,
        "count": metric(exp01, "count"),
        "validation_errors": errors,
    }


def single_exp_row(
    stage: str,
    config: str,
    task: str,
    exp: str,
    records: list[dict[str, Any]],
    min_folds: int,
) -> dict[str, Any]:
    row = aggregate_fold_records(records, exp, expected_graph_mode="real")
    errors = validation_errors(row)
    auprc = metric(row, "auprc")
    return {
        "stage": stage,
        "config": config,
        "task": task,
        "folds": fold_count(row),
        "complete": fold_count(row) >= min_folds,
        "valid_cell_llm": not errors,
        "auprc": auprc,
        "nauprc": metric(row, "nauprc"),
        "auroc": metric(row, "auroc"),
        "exp04_auprc": None,
        "exp05_auprc": None,
        "gap_no_mse_auprc": None,
        "gap_no_graph_auprc": None,
        "score": auprc,
        "rank_metric": auprc,
        "count": metric(row, "count"),
        "validation_errors": errors,
    }


def summarize(args: argparse.Namespace) -> list[dict[str, Any]]:
    report = load_report_module()
    rows: list[dict[str, Any]] = []
    for stage, config in collect_prefixes(args):
        records = collect_records(report, args, stage, config)
        if stage == "stage1":
            rows.append(stage1_row(stage, config, records, args.min_folds))
        elif stage == "stage2":
            rows.append(single_exp_row(stage, config, "exp03_unseen_cell", "exp03", records, args.min_folds))
        elif stage == "stage3":
            rows.append(single_exp_row(stage, config, "exp02_unseen_cell_type", "exp02", records, args.min_folds))
        elif stage == "stage4":
            rows.append(single_exp_row(stage, config, "exp06_double_unseen_drug", "exp06", records, args.min_folds))
    return rows


def rank_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    complete = 1.0 if row.get("complete") else 0.0
    valid = 1.0 if row.get("valid_cell_llm") else 0.0
    if row.get("stage") == "stage1":
        return (
            complete,
            valid,
            finite(row.get("score")) if finite(row.get("score")) is not None else -math.inf,
            finite(row.get("auprc")) if finite(row.get("auprc")) is not None else -math.inf,
            finite(row.get("gap_no_graph_auprc")) if finite(row.get("gap_no_graph_auprc")) is not None else -math.inf,
            finite(row.get("gap_no_mse_auprc")) if finite(row.get("gap_no_mse_auprc")) is not None else -math.inf,
        )
    return (
        complete,
        valid,
        finite(row.get("auprc")) if finite(row.get("auprc")) is not None else -math.inf,
        finite(row.get("nauprc")) if finite(row.get("nauprc")) is not None else -math.inf,
        finite(row.get("auroc")) if finite(row.get("auroc")) is not None else -math.inf,
    )


def ranked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for stage in STAGES:
        stage_rows = [row for row in rows if row.get("stage") == stage]
        stage_rows.sort(key=rank_sort_key, reverse=True)
        for index, row in enumerate(stage_rows, 1):
            item = dict(row)
            item["rank"] = index
            ranked.append(item)
    return ranked


def promotable_rows(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in ranked_rows(rows)
        if row.get("stage") == stage
        and row.get("complete")
        and row.get("valid_cell_llm")
        and finite(row.get("rank_metric")) is not None
    ]
    candidates.sort(key=rank_sort_key, reverse=True)
    return candidates


def promotion_configs(rows: list[dict[str, Any]], stage: str, margin: float) -> list[str]:
    ranked = promotable_rows(rows, stage)
    if not ranked:
        return []
    promoted = ranked[: min(PROMOTION_PRIMARY_MIN, len(ranked))]
    if len(ranked) > PROMOTION_PRIMARY_MIN:
        threshold = finite(ranked[PROMOTION_PRIMARY_MIN - 1].get("rank_metric"))
        if threshold is not None:
            for row in ranked[PROMOTION_PRIMARY_MIN:PROMOTION_MAX]:
                value = finite(row.get("rank_metric"))
                if value is None or threshold - value >= margin:
                    break
                promoted.append(row)
    return [str(row["config"]) for row in promoted]


def selected_config(rows: list[dict[str, Any]], stage: str) -> str | None:
    ranked = promotable_rows(rows, stage)
    if not ranked:
        return None
    return str(ranked[0]["config"])


def print_markdown(rows: list[dict[str, Any]], precision: int) -> None:
    print(
        "| stage | rank | config | task | folds | complete | valid Cell LLM | "
        "AUPRC | n-AUPRC | AUROC | exp04 AUPRC | exp05 AUPRC | gap no-MSE | gap no-graph | score | errors |"
    )
    print("|---|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in ranked_rows(rows):
        errors = "; ".join(row.get("validation_errors", []))
        print(
            "| {stage} | {rank} | {config} | {task} | {folds} | {complete} | {valid} | "
            "{auprc} | {nauprc} | {auroc} | {exp04} | {exp05} | {gap_mse} | {gap_graph} | {score} | {errors} |".format(
                stage=row.get("stage", ""),
                rank=row.get("rank", ""),
                config=row.get("config", ""),
                task=row.get("task", ""),
                folds=row.get("folds", 0),
                complete="yes" if row.get("complete") else "no",
                valid="yes" if row.get("valid_cell_llm") else "no",
                auprc=fmt(row.get("auprc"), precision),
                nauprc=fmt(row.get("nauprc"), precision),
                auroc=fmt(row.get("auroc"), precision),
                exp04=fmt(row.get("exp04_auprc"), precision),
                exp05=fmt(row.get("exp05_auprc"), precision),
                gap_mse=fmt(row.get("gap_no_mse_auprc"), precision),
                gap_graph=fmt(row.get("gap_no_graph_auprc"), precision),
                score=fmt(row.get("score"), precision),
                errors=errors,
            )
        )


def print_tsv(rows: list[dict[str, Any]], precision: int) -> None:
    columns = (
        "stage",
        "rank",
        "config",
        "task",
        "folds",
        "complete",
        "valid_cell_llm",
        "auprc",
        "nauprc",
        "auroc",
        "exp04_auprc",
        "exp05_auprc",
        "gap_no_mse_auprc",
        "gap_no_graph_auprc",
        "score",
        "validation_errors",
    )
    print("\t".join(columns))
    for row in ranked_rows(rows):
        values = []
        for column in columns:
            if column == "validation_errors":
                values.append("; ".join(row.get(column, [])))
            elif column in {"auprc", "nauprc", "auroc", "exp04_auprc", "exp05_auprc", "gap_no_mse_auprc", "gap_no_graph_auprc", "score"}:
                values.append(fmt(row.get(column), precision))
            else:
                values.append(str(row.get(column, "")))
        print("\t".join(values))


def main() -> int:
    args = parse_args()
    if args.min_folds < 1:
        raise SystemExit("--min-folds must be >= 1")
    rows = summarize(args)
    if args.emit_promoted_configs or args.emit_selected_config:
        if args.stage == "all":
            raise SystemExit("--emit-promoted-configs/--emit-selected-config requires --stage stageN")
        if args.emit_selected_config:
            config = selected_config(rows, args.stage)
            if config is None:
                raise SystemExit(f"no complete valid candidate found for {args.stage}")
            print(config)
            return 0
        configs = promotion_configs(rows, args.stage, args.promotion_margin)
        if not configs:
            raise SystemExit(f"no complete valid candidates found for {args.stage}")
        print(" ".join(configs))
        return 0

    if args.format == "json":
        print(json.dumps(ranked_rows(rows), ensure_ascii=False, indent=2))
    elif args.format == "tsv":
        print_tsv(rows, args.precision)
    else:
        print_markdown(rows, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
