#!/usr/bin/env python3
"""Print compact status for the 2026-06-04 Cell LLM clip10 goal."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_SCREEN_CONFIGS = {
    "stage1": (
        "base",
        "mse050",
        "mse075",
        "mse100",
        "mse050_drop010",
        "mse050_drop020",
        "mse075_drop010",
        "mse100_drop010",
        "mse050_lr1e4",
        "mse050_lr3e4",
        "mse050_warmdecay",
        "mse050_pre1",
        "mse050_pre2",
        "mse050_target_pdi",
        "mse050_target_ppi",
    ),
    "stage2": (
        "base",
        "covdrop010",
        "drop010",
        "drop020",
        "bs128",
        "lr1e4",
        "lr3e4",
        "ctrl_drop010",
        "covunk_cell",
        "covdrop010_drop010",
        "covdrop010_drop020",
        "covdrop010_bs128",
        "covdrop010_lr1e4",
        "covdrop010_lr3e4",
    ),
    "stage3": (
        "base",
        "covdrop010",
        "drop010",
        "drop020",
        "bs128",
        "lr1e4",
        "lr3e4",
        "ctrl_drop010",
        "covunk_celltype",
        "covdrop010_drop010",
        "covdrop010_drop020",
        "covdrop010_bs128",
        "covdrop010_lr1e4",
        "covdrop010_lr3e4",
    ),
    "stage4": (
        "base",
        "drop010",
        "drop020",
        "bs128",
        "lr1e4",
        "lr3e4",
        "dbl_mse010",
        "dbl_mse050",
        "rank005",
        "drop020_lr1e4",
        "drop020_lr3e4",
        "drop020_bs128",
        "drop020_mseinactive010",
        "drop020_mseinactive050",
    ),
}


EXPECTED_MANIFESTS_PER_CONFIG = {"stage1": 9, "stage2": 3, "stage3": 3, "stage4": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--screen-prefix", default="20260604_cell_llm_clip10_tune_v1")
    parser.add_argument("--full-prefix", default="20260604_cell_llm_clip10_tune_v1_full")
    parser.add_argument("--selected-prefix", default="20260604_cell_llm_clip10_tuned_selected_v1")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def expected_graph_mode(name: str) -> str:
    if "_exp05_" in name or "_single_no_graph_" in name:
        return "zero"
    return "real"


def count_screen(args: argparse.Namespace) -> None:
    pattern = re.compile(
        rf"^{re.escape(args.screen_prefix)}_(stage[1-4])_(.+)_(exp0[1-6]_.+)_fold[0-4]$"
    )
    by_stage: Counter[str] = Counter()
    by_stage_config: defaultdict[tuple[str, str], int] = defaultdict(int)
    incomplete: list[str] = []
    bad: list[str] = []
    for manifest_path in args.checkpoint_root.glob(f"{args.screen_prefix}_stage*/run_manifest.json"):
        match = pattern.match(manifest_path.parent.name)
        if not match:
            continue
        stage, config, _exp = match.groups()
        by_stage[stage] += 1
        by_stage_config[(stage, config)] += 1
        text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(text)
        if "cell_type_llm" in text:
            bad.append(f"{manifest_path}: old cell_type_llm")
        if manifest.get("run_status") != "fit_completed" or manifest.get("test_status") != "test_completed":
            incomplete.append(
                f"{manifest_path}: run_status={manifest.get('run_status')!r}; "
                f"test_status={manifest.get('test_status')!r}"
            )
        if manifest.get("cell_llm_mode") != "frozen":
            bad.append(f"{manifest_path}: cell_llm_mode={manifest.get('cell_llm_mode')!r}")
        summary = manifest.get("cell_llm_summary")
        if not isinstance(summary, dict) or summary.get("embedding_rows") != 74:
            bad.append(f"{manifest_path}: bad cell_llm_summary")
        expected_graph = expected_graph_mode(manifest_path.parent.name)
        if manifest.get("graph_feature_mode") != expected_graph:
            bad.append(
                f"{manifest_path}: graph_feature_mode={manifest.get('graph_feature_mode')!r}, "
                f"expected {expected_graph!r}"
            )

    print("screen_manifests")
    for stage in ("stage1", "stage2", "stage3", "stage4"):
        expected = len(EXPECTED_SCREEN_CONFIGS[stage]) * EXPECTED_MANIFESTS_PER_CONFIG[stage]
        print(f"  {stage}: {by_stage[stage]}/{expected}")
    print("screen_missing_or_partial_configs")
    for stage in ("stage1", "stage2", "stage3", "stage4"):
        expected_count = EXPECTED_MANIFESTS_PER_CONFIG[stage]
        for config in EXPECTED_SCREEN_CONFIGS[stage]:
            count = by_stage_config[(stage, config)]
            if count != expected_count:
                print(f"  {stage}/{config}: {count}/{expected_count}")
    print("screen_configs")
    for (stage, config), count in sorted(by_stage_config.items()):
        print(f"  {stage}/{config}: {count}")
    print(f"screen_incomplete_manifests: {len(incomplete)}")
    for item in incomplete[:20]:
        print(f"  {item}")
    print(f"screen_validation_errors: {len(bad)}")
    for item in bad[:20]:
        print(f"  {item}")


def count_prefix(args: argparse.Namespace, prefix: str, label: str) -> None:
    manifests = sorted(args.checkpoint_root.glob(f"{prefix}*/run_manifest.json"))
    incomplete = []
    bad = []
    for manifest_path in manifests:
        text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(text)
        if "cell_type_llm" in text:
            bad.append(f"{manifest_path}: old cell_type_llm")
        if manifest.get("run_status") != "fit_completed":
            incomplete.append(f"{manifest_path}: run_status={manifest.get('run_status')!r}")
        if manifest.get("cell_llm_mode") != "frozen":
            bad.append(f"{manifest_path}: cell_llm_mode={manifest.get('cell_llm_mode')!r}")
        expected_graph = expected_graph_mode(manifest_path.parent.name)
        if manifest.get("graph_feature_mode") != expected_graph:
            bad.append(
                f"{manifest_path}: graph_feature_mode={manifest.get('graph_feature_mode')!r}, "
                f"expected {expected_graph!r}"
            )
    print(f"{label}_manifests: {len(manifests)}")
    print(f"{label}_incomplete_manifests: {len(incomplete)}")
    for item in incomplete[:20]:
        print(f"  {item}")
    print(f"{label}_validation_errors: {len(bad)}")
    for item in bad[:20]:
        print(f"  {item}")


def main() -> int:
    args = parse_args()
    count_screen(args)
    count_prefix(args, args.full_prefix, "full")
    count_prefix(args, args.selected_prefix, "selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
