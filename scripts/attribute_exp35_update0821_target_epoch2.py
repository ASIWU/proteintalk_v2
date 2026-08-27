#!/usr/bin/env python3
"""Run the Exp34-validated attribution engine for the single Exp35 target task."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = REPO_ROOT / "scripts/attribute_exp34_update0819_ood_epoch2.py"
SPEC = importlib.util.spec_from_file_location("exp35_attribution_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load attribution engine: {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

PREFIX = "20260821_exp35_update0821_target_epoch2"
TASK_NAME = "ptv3_exp35_update0821_manual_target"
base.TASKS = {"manual_target": TASK_NAME}
base.DEFAULT_RUN_ROOT = REPO_ROOT / "outputs/2026-08/2026-08-21" / PREFIX
base.DEFAULT_RUNTIME_ROOT = Path("/tmp/proteintalk_exp35_update0821_target_runtime")
base.DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/2026-08/2026-08-21" / f"{PREFIX}_protein_attribution"
)
base.DEFAULT_REQUIRED_TMUX_TARGET = "gpu1:0"
base.ATTRIBUTION_REPORT_TITLE = "Exp35 Update-0821 Manual-Target Protein Attribution"
os.environ.setdefault("EXP34_TMUX_TARGET", os.environ.get("EXP35_TMUX_TARGET", "gpu1:0"))


if __name__ == "__main__":
    base.run(base.parse_args())
