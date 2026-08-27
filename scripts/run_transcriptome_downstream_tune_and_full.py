#!/usr/bin/env python3
"""Run transcriptome downstream MLP tuning followed by selected full folds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import report_transcriptome_downstream_results as report


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = "20260626_transcriptome_mlp_official_fixed_residual_v1"
TRAIN_SCRIPT = REPO_ROOT / "scripts/train_transcriptome_downstream_mlp.py"
REPORT_SCRIPT = REPO_ROOT / "scripts/report_transcriptome_downstream_results.py"
TUNE_CONFIGS = [
    {
        "config_name": "base",
        "hidden_dim": 512,
        "drug_hidden_dim": 128,
        "fusion_hidden_dim": 256,
        "dropout": 0.20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    {
        "config_name": "regularized",
        "hidden_dim": 256,
        "drug_hidden_dim": 128,
        "fusion_hidden_dim": 256,
        "dropout": 0.30,
        "lr": 1e-3,
        "weight_decay": 3e-4,
    },
    {
        "config_name": "wide",
        "hidden_dim": 1024,
        "drug_hidden_dim": 256,
        "fusion_hidden_dim": 512,
        "dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
    },
    {
        "config_name": "lowdrop_lowlr",
        "hidden_dim": 512,
        "drug_hidden_dim": 128,
        "fusion_hidden_dim": 256,
        "dropout": 0.10,
        "lr": 3e-4,
        "weight_decay": 1e-4,
    },
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_is_complete(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    test_predictions_path = run_dir / "predictions_test.csv"
    if not metrics_path.exists() or not checkpoint_path.exists() or not test_predictions_path.exists():
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


def train_command(
    args: argparse.Namespace,
    *,
    branch: str,
    exp: str,
    fold: int,
    stage: str,
    config: dict[str, Any],
    output_dir: Path,
    max_epochs: int,
    patience: int,
    seed: int,
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
        stage,
        "--config-name",
        str(config["config_name"]),
        "--output-dir",
        str(output_dir),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--drug-hidden-dim",
        str(config["drug_hidden_dim"]),
        "--fusion-hidden-dim",
        str(config["fusion_hidden_dim"]),
        "--dropout",
        str(config["dropout"]),
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--batch-size",
        str(args.batch_size),
        "--max-epochs",
        str(max_epochs),
        "--patience",
        str(patience),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--seed",
        str(seed),
    ]
    if args.limit_rows is not None:
        cmd.extend(["--limit-rows", str(args.limit_rows)])
    return cmd


def run_train_if_needed(
    args: argparse.Namespace,
    *,
    branch: str,
    exp: str,
    fold: int,
    stage: str,
    config: dict[str, Any],
    output_dir: Path,
    max_epochs: int,
    patience: int,
    seed: int,
) -> None:
    if not args.force and run_is_complete(output_dir):
        print(f"[skip] complete {stage} {branch} {exp} {config['config_name']} fold{fold}", flush=True)
        return
    cmd = train_command(
        args,
        branch=branch,
        exp=exp,
        fold=fold,
        stage=stage,
        config=config,
        output_dir=output_dir,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
    )
    subprocess_run(cmd, dry_run=args.dry_run)


def config_seed(base_seed: int, *, branch: str, exp: str, fold: int, config_index: int) -> int:
    branch_offset = report.BRANCHES.index(branch) * 1000
    exp_offset = report.EXPS.index(exp) * 100
    return base_seed + branch_offset + exp_offset + fold * 10 + config_index


def run_tune(args: argparse.Namespace) -> None:
    for branch in args.branches:
        for exp in args.exps:
            for config_index, config in enumerate(TUNE_CONFIGS):
                for fold in args.tune_folds:
                    output_dir = args.output_root / args.prefix / "tune" / branch / exp / config["config_name"] / f"fold{fold}"
                    run_train_if_needed(
                        args,
                        branch=branch,
                        exp=exp,
                        fold=fold,
                        stage="tune",
                        config=config,
                        output_dir=output_dir,
                        max_epochs=args.tune_max_epochs,
                        patience=args.tune_patience,
                        seed=config_seed(args.seed, branch=branch, exp=exp, fold=fold, config_index=config_index),
                    )


def build_selected_configs(args: argparse.Namespace) -> dict[str, Any]:
    tune_report = report.build_report(args.prefix, args.output_root)
    selected = report.selected_config_payload(args.prefix, tune_report["winners"])
    selected_path = args.output_root / f"{args.prefix}_transcriptome_mlp_selected_configs.json"
    write_json(selected_path, selected)
    print(f"[select] selected_configs={selected_path}", flush=True)
    for branch in args.branches:
        for exp in args.exps:
            item = selected.get("selected", {}).get(branch, {}).get(exp)
            if not item or item.get("status") != "ok":
                raise RuntimeError(f"missing complete selected tuning config for {branch} {exp}: {item}")
    return selected


def config_from_selected(selected: dict[str, Any], branch: str, exp: str) -> dict[str, Any]:
    item = selected["selected"][branch][exp]
    params = item["params"]
    return {
        "config_name": item["config_name"],
        "hidden_dim": int(params["hidden_dim"]),
        "drug_hidden_dim": int(params["drug_hidden_dim"]),
        "fusion_hidden_dim": int(params["fusion_hidden_dim"]),
        "dropout": float(params["dropout"]),
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
    }


def load_or_build_selected_configs(args: argparse.Namespace) -> dict[str, Any]:
    selected_path = args.output_root / f"{args.prefix}_transcriptome_mlp_selected_configs.json"
    if selected_path.exists() and not args.reselect:
        return load_json(selected_path)
    return build_selected_configs(args)


def run_final(args: argparse.Namespace, selected: dict[str, Any]) -> None:
    for branch in args.branches:
        for exp in args.exps:
            config = config_from_selected(selected, branch, exp)
            config_index = next(
                (idx for idx, item in enumerate(TUNE_CONFIGS) if item["config_name"] == config["config_name"]),
                0,
            )
            for fold in args.final_folds:
                output_dir = args.output_root / args.prefix / "final" / branch / exp / f"fold{fold}"
                run_train_if_needed(
                    args,
                    branch=branch,
                    exp=exp,
                    fold=fold,
                    stage="final",
                    config=config,
                    output_dir=output_dir,
                    max_epochs=args.final_max_epochs,
                    patience=args.final_patience,
                    seed=config_seed(args.seed + 5000, branch=branch, exp=exp, fold=fold, config_index=config_index),
                )


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


def parse_ints(values: list[str]) -> list[int]:
    return [int(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--branches", nargs="+", choices=report.BRANCHES, default=list(report.BRANCHES))
    parser.add_argument("--exps", nargs="+", choices=report.EXPS, default=list(report.EXPS))
    parser.add_argument("--tune-folds", nargs="+", type=int, default=list(report.TUNE_FOLDS))
    parser.add_argument("--final-folds", nargs="+", type=int, default=list(report.FINAL_FOLDS))
    parser.add_argument("--stage", choices=("all", "tune", "final", "report"), default="all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tune-max-epochs", type=int, default=50)
    parser.add_argument("--tune-patience", type=int, default=8)
    parser.add_argument("--final-max-epochs", type=int, default=80)
    parser.add_argument("--final-patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"all", "tune"}:
        run_tune(args)
        build_selected_configs(args)
        if args.stage == "tune":
            return 0
    selected = load_or_build_selected_configs(args)
    if args.stage in {"all", "final"}:
        run_final(args, selected)
        run_report(args, strict=True)
    elif args.stage == "report":
        run_report(args, strict=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
