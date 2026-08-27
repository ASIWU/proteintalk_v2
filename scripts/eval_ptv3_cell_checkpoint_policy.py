#!/usr/bin/env python3
"""Evaluate PTV3 exp03 cell-fold checkpoints under an alternate checkpoint policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_SUFFIX = "exp03_single_cell_5fold"
FOLD_SUFFIX = "single_cell_fold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", action="append", required=True, help="Base prefix without _exp03 suffix.")
    parser.add_argument("--folds", default="2", help="Space-separated fold ids.")
    parser.add_argument("--checkpoint-kind", choices=("best", "last"), default="last")
    parser.add_argument("--eval-suffix", default=None, help="Suffix appended to source prefix for temp/report prefix.")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp/ptv3_cell_checkpoint_policy_eval"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--infer-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force-infer", action="store_true")
    parser.add_argument("--summary-csv", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def checkpoint_path(source_dir: Path, manifest: dict[str, Any], kind: str) -> Path:
    if kind == "best":
        path = manifest.get("best_model_path") or str(source_dir / "last.ckpt")
    else:
        path = str(source_dir / "last.ckpt")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing {kind} checkpoint: {resolved}")
    return resolved


def materialize_temp_manifest(
    *,
    source_prefix: str,
    eval_prefix: str,
    fold: str,
    checkpoint_root: Path,
    temp_root: Path,
    checkpoint_kind: str,
) -> None:
    source_dir = checkpoint_root / f"{source_prefix}_{EXP_SUFFIX}_{FOLD_SUFFIX}{fold}"
    source_manifest = source_dir / "run_manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(f"missing source manifest: {source_manifest}")
    manifest = load_json(source_manifest)
    selected_checkpoint = checkpoint_path(source_dir, manifest, checkpoint_kind)
    target_dir = temp_root / f"{eval_prefix}_{EXP_SUFFIX}_{FOLD_SUFFIX}{fold}"
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest["test_checkpoint_path"] = str(selected_checkpoint.resolve())
    manifest["checkpoint_policy_eval"] = {
        "source_prefix": source_prefix,
        "source_manifest": str(source_manifest.resolve()),
        "checkpoint_kind": checkpoint_kind,
        "selected_checkpoint_path": str(selected_checkpoint.resolve()),
    }
    with (target_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def run_report(args: argparse.Namespace, eval_prefix: str) -> Path:
    csv_out = args.output_root / f"{eval_prefix}_cell_drug_time_eval.csv"
    json_out = args.output_root / f"{eval_prefix}_cell_drug_time_eval.json"
    md_out = REPO_ROOT / "logs" / f"{eval_prefix}_cell_drug_time_eval.md"
    command = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "report_cell_drug_time_eval.py"),
        "--prefix",
        eval_prefix,
        "--checkpoint-root",
        str(args.temp_root),
        "--output-root",
        str(args.output_root),
        "--materialize-fold-predictions",
        "--device",
        args.device,
        "--infer-batch-size",
        str(args.infer_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--csv-out",
        str(csv_out),
        "--json-out",
        str(json_out),
        "--format",
        "markdown",
    ]
    if args.force_infer:
        command.append("--force-infer")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    with md_out.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy(), stdout=handle, check=True)
    return csv_out


def collect_rows(source_prefix: str, eval_prefix: str, csv_path: Path, checkpoint_kind: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(csv_path)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "source_prefix": source_prefix,
                "eval_prefix": eval_prefix,
                "checkpoint_kind": checkpoint_kind,
                "split": row.get("split"),
                "method": row.get("method"),
                "auroc": row.get("auroc"),
                "auprc": row.get("auprc"),
                "valid_count": row.get("valid_count"),
                "positive_count": row.get("positive_count"),
                "csv": str(csv_path),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    args.temp_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds = args.folds.split()
    eval_suffix = args.eval_suffix or f"{args.checkpoint_kind}ckpt"
    all_rows: list[dict[str, Any]] = []

    for source_prefix in args.source_prefix:
        eval_prefix = f"{source_prefix}_{eval_suffix}"
        for fold in folds:
            materialize_temp_manifest(
                source_prefix=source_prefix,
                eval_prefix=eval_prefix,
                fold=fold,
                checkpoint_root=args.checkpoint_root,
                temp_root=args.temp_root,
                checkpoint_kind=args.checkpoint_kind,
            )
        csv_path = run_report(args, eval_prefix)
        all_rows.extend(collect_rows(source_prefix, eval_prefix, csv_path, args.checkpoint_kind))

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()) if all_rows else [])
            if all_rows:
                writer.writeheader()
                writer.writerows(all_rows)

    for row in all_rows:
        print(
            "{source_prefix}\t{checkpoint_kind}\t{split}\t{method}\tAUPRC={auprc:.6f}\tAUROC={auroc:.6f}\tn={valid_count}\tpos={positive_count}".format(
                **row
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
