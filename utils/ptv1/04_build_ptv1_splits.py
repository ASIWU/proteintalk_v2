#!/usr/bin/env python3
"""Build PTV1-only split artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _shared import REPO_ROOT, dump_json, iso_now, load_json, load_repo_module


splitter = load_repo_module("utils/09_build_data_splits.py", "ptv_split_shared")

DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ready-root", type=Path, default=DEFAULT_TRAINING_READY_ROOT)
    parser.add_argument("--task", default=None, choices=[None, "ptv1_aivc", "ptv1_extra_singledrug"])
    parser.add_argument("--output-subdir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--subset-test-ratio", type=float, default=0.2)
    parser.add_argument("--stratified-top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = splitter.select_tasks(args.training_ready_root, "ptv1", args.task)
    manifests: dict[str, Any] = {}
    for group, task_name in selected:
        task_dir = args.training_ready_root / group / "tasks" / task_name
        output_dir = args.training_ready_root / group / args.output_subdir / task_name
        print(f"[split] {group}/{task_name} -> {output_dir}")
        manifest = splitter.build_task_splits(
            dataset_group=group,
            task_name=task_name,
            task_dir=task_dir,
            output_dir=output_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            n_folds=args.n_folds,
            subset_test_ratio=args.subset_test_ratio,
            stratified_top_n=args.stratified_top_n,
        )
        manifests[f"{group}/{task_name}"] = {
            "output_dir": manifest["output_dir"],
            "valid_anchor_count": manifest["pairing"]["valid_anchor_count"],
            "strategies": [item["strategy"] for item in manifest["splits"]],
        }

    output_path = args.training_ready_root / "split_build_manifest.json"
    if output_path.exists():
        payload = load_json(output_path)
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {}
    payload["generated_at"] = iso_now()
    payload["training_ready_root"] = str(args.training_ready_root)
    payload.setdefault("tasks", {})
    payload["tasks"].update(manifests)
    dump_json(output_path, payload)
    dump_json(
        args.training_ready_root / "ptv1" / "split_build_manifest.json",
        {"generated_at": iso_now(), "training_ready_root": str(args.training_ready_root), "tasks": manifests},
    )
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
