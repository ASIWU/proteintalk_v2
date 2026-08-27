#!/usr/bin/env python3
"""Shared helpers for the dedicated PTV1 pipeline entrypoints."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_repo_module(relative_path: str, module_name: str) -> ModuleType:
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=True)


def merge_root_audit(path: Path, *, dataset_group: str, dataset_payload: dict, task_payloads: dict) -> None:
    if path.exists():
        try:
            audit = load_json(path)
        except Exception:
            audit = {}
        if not isinstance(audit, dict):
            audit = {}
    else:
        audit = {}

    audit["generated_at"] = iso_now()
    audit.setdefault("dataset_groups", {})
    audit.setdefault("tasks", {})
    audit["dataset_groups"][dataset_group] = dataset_payload
    for task_name, payload in task_payloads.items():
        audit["tasks"][task_name] = payload
    dump_json(path, audit)
