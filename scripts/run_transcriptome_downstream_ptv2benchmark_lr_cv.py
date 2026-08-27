#!/usr/bin/env python3
"""Run PTV2-benchmark-aligned LR CV for transcriptome downstream MLP."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import report_transcriptome_downstream_ptv2benchmark_lr_cv as report


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = report.DEFAULT_PREFIX
TRAIN_SCRIPT = REPO_ROOT / "scripts/train_transcriptome_downstream_mlp.py"
REPORT_SCRIPT = REPO_ROOT / "scripts/report_transcriptome_downstream_ptv2benchmark_lr_cv.py"
PTV2_CONFIG = {
    "hidden_dim": 64,
    "drug_hidden_dim": 32,
    "fusion_hidden_dim": 32,
    "dropout": 0.0,
    "activation": "relu",
    "batch_size": 64,
    "weight_decay": 0.01,
    "fixed_epochs": 150,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def run_is_complete(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    final_checkpoint_path = run_dir / "final_checkpoint.pt"
    train_predictions_path = run_dir / "predictions_train.csv"
    test_predictions_path = run_dir / "predictions_test.csv"
    if not (
        metrics_path.exists()
        and final_checkpoint_path.exists()
        and train_predictions_path.exists()
        and test_predictions_path.exists()
    ):
        return False
    try:
        metrics = load_json(metrics_path)
    except Exception:
        return False
    return metrics.get("run_status") == "ok"


def subprocess_run(cmd: list[str], *, dry_run: bool) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def config_seed(base_seed: int, *, branch: str, exp: str, lr_name: str, fold: int) -> int:
    branch_offset = report.BRANCHES.index(branch) * 10000
    exp_offset = report.EXPS.index(exp) * 1000
    lr_offset = [name for name, _ in report.LR_GRID].index(lr_name) * 100
    return base_seed + branch_offset + exp_offset + lr_offset + fold


def train_command(
    args: argparse.Namespace,
    *,
    branch: str,
    exp: str,
    lr_name: str,
    lr: float,
    fold: int,
    output_dir: Path,
) -> list[str]:
    cmd = [
        args.python_bin,
        str(TRAIN_SCRIPT),
        "--branch",
        branch,
        "--exp",
        exp,
        "--fold",
        str(fold),
        "--stage",
        "cv",
        "--config-name",
        lr_name,
        "--output-dir",
        str(output_dir),
        "--hidden-dim",
        str(args.hidden_dim),
        "--drug-hidden-dim",
        str(args.drug_hidden_dim),
        "--fusion-hidden-dim",
        str(args.fusion_hidden_dim),
        "--dropout",
        str(args.dropout),
        "--activation",
        args.activation,
        "--lr",
        str(lr),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--fixed-epochs",
        str(args.fixed_epochs),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--seed",
        str(config_seed(args.seed, branch=branch, exp=exp, lr_name=lr_name, fold=fold)),
        "--merge-val-into-train",
        "--no-validation",
        "--log-every",
        str(args.log_every),
    ]
    if args.limit_rows is not None:
        cmd.extend(["--limit-rows", str(args.limit_rows)])
    return cmd


def run_cv(args: argparse.Namespace) -> None:
    requested_lrs = dict(report.LR_GRID)
    for branch in args.branches:
        for exp in args.exps:
            for lr_name in args.lrs:
                lr = requested_lrs[lr_name]
                for fold in args.folds:
                    output_dir = args.output_root / args.prefix / "cv" / branch / exp / lr_name / f"fold{fold}"
                    if not args.force and run_is_complete(output_dir):
                        print(f"[skip] complete cv {branch} {exp} {lr_name} fold{fold}", flush=True)
                        continue
                    cmd = train_command(
                        args,
                        branch=branch,
                        exp=exp,
                        lr_name=lr_name,
                        lr=lr,
                        fold=fold,
                        output_dir=output_dir,
                    )
                    subprocess_run(cmd, dry_run=args.dry_run)


def run_report(args: argparse.Namespace, *, strict: bool) -> None:
    cmd = [
        args.python_bin,
        str(REPORT_SCRIPT),
        "--prefix",
        args.prefix,
        "--output-root",
        str(args.output_root),
    ]
    if strict:
        cmd.append("--strict")
    subprocess_run(cmd, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    lr_names = [name for name, _ in report.LR_GRID]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--branches", nargs="+", choices=report.BRANCHES, default=list(report.BRANCHES))
    parser.add_argument("--exps", nargs="+", choices=report.EXPS, default=list(report.EXPS))
    parser.add_argument("--folds", nargs="+", type=int, choices=report.FOLDS, default=list(report.FOLDS))
    parser.add_argument("--lrs", nargs="+", choices=lr_names, default=lr_names)
    parser.add_argument("--stage", choices=("all", "cv", "report"), default="all")
    parser.add_argument("--hidden-dim", type=int, default=PTV2_CONFIG["hidden_dim"])
    parser.add_argument("--drug-hidden-dim", type=int, default=PTV2_CONFIG["drug_hidden_dim"])
    parser.add_argument("--fusion-hidden-dim", type=int, default=PTV2_CONFIG["fusion_hidden_dim"])
    parser.add_argument("--dropout", type=float, default=PTV2_CONFIG["dropout"])
    parser.add_argument("--activation", choices=("silu", "relu"), default=PTV2_CONFIG["activation"])
    parser.add_argument("--batch-size", type=int, default=PTV2_CONFIG["batch_size"])
    parser.add_argument("--weight-decay", type=float, default=PTV2_CONFIG["weight_decay"])
    parser.add_argument("--fixed-epochs", type=int, default=PTV2_CONFIG["fixed_epochs"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=29000)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"all", "cv"}:
        run_cv(args)
        if args.stage == "cv":
            return 0
    run_report(args, strict=args.stage == "all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
