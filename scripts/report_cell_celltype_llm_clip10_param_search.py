#!/usr/bin/env python3
"""Summarize Cell + cell_type LLM clip10 parameter-search stages."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_REPORT_PATH = REPO_ROOT / "scripts" / "report_cell_llm_clip10_param_search.py"


def load_base_report() -> Any:
    spec = importlib.util.spec_from_file_location("cell_llm_tuning_report", BASE_REPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_REPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_REPORT = load_base_report()


def patched_base_report() -> Any:
    BASE_REPORT.manifest_errors = manifest_errors
    return BASE_REPORT


def manifest_errors(source: Any, expected_graph_mode: str | None) -> list[str]:
    path = Path(str(source))
    if not path.exists() or path.name != "run_manifest.json":
        return [f"missing manifest {path}"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid manifest JSON: {exc}"]

    errors: list[str] = []
    if manifest.get("run_status") != "fit_completed":
        errors.append(f"run_status={manifest.get('run_status')!r}")
    if manifest.get("test_status") != "test_completed":
        errors.append(f"test_status={manifest.get('test_status')!r}")

    if manifest.get("cell_llm_mode") != "frozen":
        errors.append(f"cell_llm_mode={manifest.get('cell_llm_mode')!r}")
    cell_summary = manifest.get("cell_llm_summary")
    if not isinstance(cell_summary, dict):
        errors.append("missing cell_llm_summary")
    else:
        if cell_summary.get("embedding_rows") != 74:
            errors.append("cell_llm_summary.embedding_rows!=74")
        if cell_summary.get("feature_dim") != 4096:
            errors.append("cell_llm_summary.feature_dim!=4096")
        if cell_summary.get("index_column") != "Cell_index":
            errors.append(f"cell_llm_summary.index_column={cell_summary.get('index_column')!r}")

    if manifest.get("cell_type_llm_mode") != "frozen":
        errors.append(f"cell_type_llm_mode={manifest.get('cell_type_llm_mode')!r}")
    cell_type_summary = manifest.get("cell_type_llm_summary")
    if not isinstance(cell_type_summary, dict):
        errors.append("missing cell_type_llm_summary")
    else:
        if cell_type_summary.get("embedding_rows") != 14:
            errors.append("cell_type_llm_summary.embedding_rows!=14")
        if cell_type_summary.get("feature_dim") != 4096:
            errors.append("cell_type_llm_summary.feature_dim!=4096")
        if cell_type_summary.get("index_column") != "cell_type_index":
            errors.append(f"cell_type_llm_summary.index_column={cell_type_summary.get('index_column')!r}")

    if expected_graph_mode is not None and manifest.get("graph_feature_mode") != expected_graph_mode:
        errors.append(f"graph_feature_mode={manifest.get('graph_feature_mode')!r}")
    return errors


def summarize(args: Any) -> list[dict[str, Any]]:
    return patched_base_report().summarize(args)


def rank_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    return patched_base_report().rank_sort_key(row)


def ranked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return patched_base_report().ranked_rows(rows)


def promotable_rows(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return patched_base_report().promotable_rows(rows, stage)


def main() -> int:
    return int(patched_base_report().main())


if __name__ == "__main__":
    raise SystemExit(main())
