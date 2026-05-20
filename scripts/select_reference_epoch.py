#!/usr/bin/env python3
"""Select a fixed training epoch from reference fold run manifests."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EPOCH_RE = re.compile(r"epoch=(\d+)")
GLOB_META = set("*?[]")
DEFAULT_HOMOGENEITY_FIELDS = (
    "dataset_group",
    "task_name",
    "task_head",
    "model_type",
    "monitor",
    "monitor_mode",
    "best_ckpt_metric",
    "effective_key1",
    "effective_key2",
    "task_label_key",
    "task_mask_key",
    "batch_cov_list",
    "fusion_mode",
    "perturb_fusion_mode",
    "num_heads",
    "num_layers",
    "hidden_dim",
    "dropout",
    "cls_type",
    "graph_dropout",
    "use_target",
    "target_protein_fusion_model",
    "gate_weight",
    "optimizer_name",
    "learning_rate",
    "mse_weight",
    "have_mse_loss",
    "bce_weight",
    "positive_weight",
    "focal_loss",
    "pdi_mode",
    "pdi_input_orientation",
    "max_epochs",
)


@dataclass(frozen=True)
class ReferenceRun:
    manifest_path: Path
    experiment_name: str
    dataset_group: str
    task_name: str
    task_head: str
    model_type: str
    split_strategy: str
    checkpoint_path: str
    epoch: int
    best_model_score: float | None
    monitor: str | None
    monitor_mode: str | None
    manifest: dict

    def summary(self) -> dict:
        return {
            "manifest_path": str(self.manifest_path),
            "experiment_name": self.experiment_name,
            "dataset_group": self.dataset_group,
            "task_name": self.task_name,
            "task_head": self.task_head,
            "model_type": self.model_type,
            "split_strategy": self.split_strategy,
            "checkpoint_path": self.checkpoint_path,
            "epoch": self.epoch,
            "best_model_score": self.best_model_score,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "best_ckpt_metric": self.manifest.get("best_ckpt_metric"),
        }


def has_glob_meta(text: str) -> bool:
    return any(char in text for char in GLOB_META)


def expand_input(text: str) -> list[Path]:
    if has_glob_meta(text):
        return [Path(path) for path in sorted(glob.glob(text))]

    path = Path(text)
    if path.exists():
        return [path]

    # A run prefix such as checkpoints/20260510_single_pert_stratified_5fold
    # is convenient on the shell, so treat missing non-globs as prefixes too.
    return [Path(path) for path in sorted(glob.glob(f"{text}*"))]


def iter_manifest_paths(inputs: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw_path in inputs:
        for path in expand_input(raw_path):
            if path.is_file():
                candidates = [path]
            elif path.is_dir():
                direct_manifest = path / "run_manifest.json"
                if direct_manifest.exists():
                    candidates = [direct_manifest]
                else:
                    candidates = sorted(path.rglob("run_manifest.json"))
            else:
                candidates = []

            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield candidate


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return payload


def parse_epoch(checkpoint_path: str | None) -> int | None:
    if not checkpoint_path:
        return None
    match = EPOCH_RE.search(str(checkpoint_path))
    if match is None:
        return None
    return int(match.group(1))


def checkpoint_exists(checkpoint_path: str) -> bool:
    path = Path(checkpoint_path)
    if path.exists():
        return True
    if not path.is_absolute() and (Path.cwd() / path).exists():
        return True
    return False


def json_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def validate_reference_set(runs: list[ReferenceRun], args: argparse.Namespace) -> None:
    errors: list[str] = []
    split_strategies = [run.split_strategy for run in runs]
    if not args.allow_duplicate_split_strategies:
        duplicates = sorted({value for value in split_strategies if split_strategies.count(value) > 1})
        if duplicates:
            errors.append("duplicate split_strategy values: " + ", ".join(duplicates))

    if args.split_strategy_regex:
        pattern = re.compile(args.split_strategy_regex)
        mismatches = [
            f"{run.manifest_path}: split_strategy={run.split_strategy!r}"
            for run in runs
            if pattern.fullmatch(run.split_strategy) is None
        ]
        if mismatches:
            errors.append(
                "reference split_strategy values do not match "
                f"{args.split_strategy_regex!r}:\n  " + "\n  ".join(mismatches)
            )

    if not args.allow_mixed_reference_config:
        for field in DEFAULT_HOMOGENEITY_FIELDS:
            values: dict[str, list[str]] = {}
            for run in runs:
                values.setdefault(json_key(run.manifest.get(field)), []).append(str(run.manifest_path))
            if len(values) > 1:
                rendered = []
                for value, paths in sorted(values.items()):
                    rendered.append(f"{field}={value}: {', '.join(paths)}")
                errors.append("mixed reference config for " + field + ":\n  " + "\n  ".join(rendered))

    if errors:
        raise SystemExit("invalid reference fold set:\n- " + "\n- ".join(errors))


def load_reference_runs(args: argparse.Namespace) -> list[ReferenceRun]:
    runs: list[ReferenceRun] = []
    skipped: list[str] = []
    manifest_paths = list(iter_manifest_paths(args.paths))
    if not manifest_paths:
        raise SystemExit(f"no run_manifest.json files matched: {', '.join(args.paths)}")

    for manifest_path in manifest_paths:
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:
            skipped.append(f"{manifest_path}: failed to read manifest ({exc})")
            continue

        task_name = str(manifest.get("task_name") or "")
        if args.task_name and task_name != args.task_name:
            skipped.append(f"{manifest_path}: task_name={task_name!r} does not match {args.task_name!r}")
            continue

        task_head = str(manifest.get("task_head") or "")
        if args.expect_task_head and task_head != args.expect_task_head:
            skipped.append(f"{manifest_path}: task_head={task_head!r} does not match {args.expect_task_head!r}")
            continue

        model_type = str(manifest.get("model_type") or "")
        if args.expect_model_type and model_type != args.expect_model_type:
            skipped.append(f"{manifest_path}: model_type={model_type!r} does not match {args.expect_model_type!r}")
            continue

        dataset_group = str(manifest.get("dataset_group") or "")
        if args.expect_dataset_group and dataset_group != args.expect_dataset_group:
            skipped.append(
                f"{manifest_path}: dataset_group={dataset_group!r} does not match {args.expect_dataset_group!r}"
            )
            continue

        run_status = manifest.get("run_status")
        if run_status != "fit_completed" and not args.allow_incomplete:
            skipped.append(f"{manifest_path}: run_status={run_status!r} is not fit_completed")
            continue

        test_status = manifest.get("test_status")
        if args.require_test_completed and test_status != "test_completed":
            skipped.append(f"{manifest_path}: test_status={test_status!r} is not test_completed")
            continue

        checkpoint_path = manifest.get(args.checkpoint_field)
        epoch = parse_epoch(checkpoint_path)
        if epoch is None:
            skipped.append(
                f"{manifest_path}: {args.checkpoint_field}={checkpoint_path!r} does not contain epoch=<N>"
            )
            continue
        if not args.allow_missing_checkpoint and not checkpoint_exists(str(checkpoint_path)):
            skipped.append(f"{manifest_path}: checkpoint does not exist: {checkpoint_path}")
            continue

        best_model_score = manifest.get("best_model_score")
        if best_model_score is None and not args.allow_missing_score:
            skipped.append(f"{manifest_path}: best_model_score is missing/null")
            continue
        runs.append(
            ReferenceRun(
                manifest_path=manifest_path,
                experiment_name=str(manifest.get("experiment_name") or manifest_path.parent.name),
                dataset_group=dataset_group,
                task_name=task_name,
                task_head=task_head,
                model_type=model_type,
                split_strategy=str(manifest.get("split_strategy") or ""),
                checkpoint_path=str(checkpoint_path),
                epoch=epoch,
                best_model_score=float(best_model_score) if isinstance(best_model_score, (int, float)) else None,
                monitor=manifest.get("monitor"),
                monitor_mode=manifest.get("monitor_mode"),
                manifest=manifest,
            )
        )

    if len(runs) < args.min_count:
        message = (
            f"only {len(runs)} usable reference runs found; need at least {args.min_count}. "
            f"Inputs: {', '.join(args.paths)}"
        )
        if skipped:
            message += "\nSkipped:\n  " + "\n  ".join(skipped)
        raise SystemExit(message)

    for item in skipped:
        print(f"[reference-epoch] skipped {item}", file=sys.stderr)
    validate_reference_set(runs, args)
    return runs


def round_epoch(value: float, rounding: str) -> int:
    if rounding == "floor":
        return int(math.floor(value))
    if rounding == "ceil":
        return int(math.ceil(value))
    if rounding == "nearest":
        return int(math.floor(value + 0.5))
    raise ValueError(f"unsupported rounding: {rounding}")


def aggregate_epoch(epochs: list[int], method: str, rounding: str) -> tuple[int, float]:
    if method == "median":
        value = float(statistics.median(epochs))
    elif method == "mean":
        value = float(statistics.mean(epochs))
    elif method == "min":
        value = float(min(epochs))
    elif method == "max":
        value = float(max(epochs))
    else:
        raise ValueError(f"unsupported aggregation method: {method}")
    return round_epoch(value, rounding), value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read reference fold run_manifest.json files, extract each best_model_path epoch, "
            "and print one fixed epoch for all-train extra-data evaluation."
        )
    )
    parser.add_argument("paths", nargs="+", help="Reference run dirs, manifest paths, glob patterns, or run prefixes.")
    parser.add_argument("--task-name", default=None, help="Only use manifests for this task_name.")
    parser.add_argument("--expect-task-head", default=None, help="Require this manifest task_head.")
    parser.add_argument("--expect-model-type", default=None, help="Require this manifest model_type.")
    parser.add_argument("--expect-dataset-group", default=None, help="Require this manifest dataset_group.")
    parser.add_argument(
        "--split-strategy-regex",
        default=None,
        help="Require each split_strategy to fully match this regular expression.",
    )
    parser.add_argument(
        "--checkpoint-field",
        default="best_model_path",
        help="Manifest checkpoint field to parse. Defaults to best_model_path.",
    )
    parser.add_argument(
        "--method",
        choices=["median", "mean", "min", "max"],
        default="median",
        help="How to aggregate fold best epochs. Median is the default.",
    )
    parser.add_argument(
        "--rounding",
        choices=["nearest", "floor", "ceil"],
        default="nearest",
        help="Rounding used when the aggregated epoch is fractional.",
    )
    parser.add_argument("--min-count", type=int, default=1, help="Minimum usable reference manifests required.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow manifests whose run_status is not fit_completed.",
    )
    parser.add_argument(
        "--require-test-completed",
        action="store_true",
        help="Require reference folds to have test_status=test_completed.",
    )
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Allow reference manifests whose parsed checkpoint path no longer exists.",
    )
    parser.add_argument(
        "--allow-missing-score",
        action="store_true",
        help="Allow reference manifests with null best_model_score.",
    )
    parser.add_argument(
        "--allow-mixed-reference-config",
        action="store_true",
        help="Allow reference folds to differ in model, metric, loss, or optimizer configuration.",
    )
    parser.add_argument(
        "--allow-duplicate-split-strategies",
        action="store_true",
        help="Allow multiple reference manifests with the same split_strategy.",
    )
    parser.add_argument("--summary-json", default=None, help="Optional path to write selected epoch details as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_count < 1:
        raise SystemExit("--min-count must be >= 1")

    runs = load_reference_runs(args)
    epochs = [run.epoch for run in runs]
    selected_epoch, raw_value = aggregate_epoch(epochs, args.method, args.rounding)
    details = ", ".join(
        f"{run.experiment_name}:epoch={run.epoch}"
        for run in sorted(runs, key=lambda item: item.experiment_name)
    )
    monitors = sorted({f"{run.monitor or 'none'}:{run.monitor_mode or 'none'}" for run in runs})
    print(
        "[reference-epoch] "
        f"count={len(runs)} method={args.method} raw={raw_value:.3f} "
        f"selected_epoch={selected_epoch} monitors={','.join(monitors)}",
        file=sys.stderr,
    )
    print(f"[reference-epoch] folds={details}", file=sys.stderr)
    if args.summary_json:
        payload = {
            "selected_epoch": selected_epoch,
            "raw_epoch_value": raw_value,
            "method": args.method,
            "rounding": args.rounding,
            "count": len(runs),
            "monitors": monitors,
            "runs": [run.summary() for run in sorted(runs, key=lambda item: item.experiment_name)],
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(selected_epoch)


if __name__ == "__main__":
    main()
