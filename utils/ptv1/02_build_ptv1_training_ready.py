#!/usr/bin/env python3
"""Build isolated PTV1 stage-2 training-ready artifacts."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from _shared import REPO_ROOT, dump_json, iso_now, load_json, load_repo_module


builder = load_repo_module("utils/02_build_training_ready_data.py", "ptv_stage2_shared")

DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "standardized"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "training_ready"
PTV1_TASKS = ("ptv1_aivc", "ptv1_extra_singledrug")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if hasattr(builder, "build_task_specs"):
        all_task_specs = builder.build_task_specs(include_prism2_main_tasks=False)
    else:
        all_task_specs = builder.TASK_SPECS
    task_specs = [spec for spec in all_task_specs if spec.dataset_group == "ptv1"]
    meta, task_manifests = builder.build_dataset_group(
        dataset_group="ptv1",
        input_root=args.input_root,
        output_root=args.output_root,
        task_specs=task_specs,
    )
    gc.collect()

    dataset_payload = {
        "global_meta_path": str(args.output_root / "ptv1" / "global_meta.json"),
        "task_names": list(PTV1_TASKS),
        "protein_index_size": len(meta["protein_index"]),
        "pert_index_size": len(meta["pert_index"]),
    }
    task_payloads = {task_name: task_manifests[task_name] for task_name in PTV1_TASKS}
    audit = {
        "generated_at": iso_now(),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "implementation_notes": [
            "PTV1 stage-2 is built without invoking the mixed PTV3/PTV1 builder entrypoint.",
            "PTV1 keeps an independent protein_index and pert_index under data/training_ready/ptv1.",
            "PTV1 extra single-drug rows append matched PTV1 AIVC controls before feature matrix alignment.",
        ],
        "dataset_groups": {"ptv1": dataset_payload},
        "tasks": task_payloads,
    }
    dump_json(args.output_root / "ptv1" / "file_audit.json", audit)

    root_audit_path = args.output_root / "file_audit.json"
    if root_audit_path.exists():
        root_audit = load_json(root_audit_path)
        if not isinstance(root_audit, dict):
            root_audit = {}
    else:
        root_audit = {}
    root_audit["generated_at"] = iso_now()
    root_audit.setdefault("input_root", str(args.input_root))
    root_audit.setdefault("output_root", str(args.output_root))
    root_audit.setdefault("dataset_groups", {})
    root_audit.setdefault("tasks", {})
    root_audit["dataset_groups"]["ptv1"] = dataset_payload
    root_audit["tasks"].update(task_payloads)
    dump_json(root_audit_path, root_audit)

    print(f"[done] wrote PTV1 training-ready artifacts under {args.output_root / 'ptv1'}")


if __name__ == "__main__":
    main()
