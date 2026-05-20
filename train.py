#!/usr/bin/env python3
"""Train ProteinTalk models from the new `data/training_ready` format."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

try:
    from pytorch_lightning.loggers import WandbLogger
except Exception:  # pragma: no cover - only needed when wandb is unavailable.
    WandbLogger = None

from dataset.training_ready_dataset import (
    ProteinTalkDataset,
    TrainingReadyArtifacts,
    load_embedding_matrix,
    load_indices,
    load_json,
    load_row_to_set,
    load_set_info,
)
from model.training_ready_lightning import ProteinTalkLightning, UnfreezeCallback
from model.training_ready_models import GRAPH_MODEL_NAMES, ModelArtifacts, SELECTED_MODEL_NAMES, build_model


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def json_safe(value: object) -> object:
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return json_safe(detached.item())
        return json_safe(detached.tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def infer_label_key(task_name: str) -> str:
    if "extra_singledrug" in task_name or task_name == "ptv1_extra_singledrug":
        return "PRISM2nd_label_total"
    return "PRISM1st_label_total"


def infer_task_head(task_name: str) -> str:
    return "synergy" if "doubledrug" in task_name else "response"


def resolve_task_loss_config(
    *,
    task_name: str,
    task_head: str,
    effective_key1: str,
    effective_key2: str,
    task_label_key: str | None = None,
    task_mask_key: str | None = None,
) -> dict[str, str]:
    resolved_head = infer_task_head(task_name) if task_head == "auto" else task_head
    if resolved_head not in {"response", "synergy"}:
        raise ValueError("task_head must be one of: auto, response, synergy")
    resolved_label = task_label_key or (effective_key2 if resolved_head == "synergy" else effective_key1)
    resolved_mask = task_mask_key or ("synergy_label_mask" if resolved_head == "synergy" else "sensitive_label_mask")
    return {
        "task_head": resolved_head,
        "task_label_key": resolved_label,
        "task_mask_key": resolved_mask,
    }


def category_sizes(meta: dict[str, Any], batch_cov_list: list[str]) -> list[int]:
    sizes: list[int] = []
    for field in batch_cov_list:
        mapping_key = "pert_dose" if field in {"pert_dose1", "pert_dose2"} else field
        mapping = meta["value_to_index"][mapping_key]
        values = [int(float(value)) for value in mapping.values()]
        sizes.append(max(values) + 1 if values else 1)
    return sizes


def default_derived_paths(training_ready_root: Path, dataset_group: str) -> dict[str, Path]:
    derived = training_ready_root / dataset_group / "derived"
    return {
        "protein_embedding": derived / "protein_embedding_esm.pkl",
        "drug_embedding": derived / "drug_embedding_morgan_2048.pkl",
        "pdi_matrix": derived / "pdi_matrix.npy",
    }


def load_pdi_matrix(path: Path, *, model_type: str, pdi_mode: str) -> np.ndarray | None:
    if model_type not in GRAPH_MODEL_NAMES:
        return None
    matrix = np.load(path).astype(np.float32, copy=False)
    if pdi_mode == "real":
        return matrix
    if pdi_mode == "zero":
        return np.zeros_like(matrix, dtype=np.float32)
    raise ValueError(f"unsupported pdi_mode={pdi_mode!r}")


def parse_limit_batches(value: str) -> int | float:
    """Preserve Lightning's `1` means one batch, while `1.0` means all batches."""

    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return float(text)


BEST_CKPT_METRIC_ALIASES = {
    "total_loss": "valid_total_loss",
    "valid_total_loss": "valid_total_loss",
    "loss1": "valid_loss1",
    "valid_loss1": "valid_loss1",
    "loss2": "valid_loss2",
    "valid_loss2": "valid_loss2",
    "auprc": "valid_auprc",
    "valid_auprc": "valid_auprc",
}

BEST_CKPT_METRIC_CONFIG = {
    "valid_total_loss": ("val/total_loss", "min"),
    "valid_loss1": ("val/loss1", "min"),
    "valid_loss2": ("val/loss2", "min"),
    "valid_auprc": ("val/task_auprc", "max"),
}


def default_monitor_mode(monitor: str | None) -> str:
    if monitor is None:
        return "min"
    metric_name = monitor.lower()
    if any(token in metric_name for token in ("auprc", "auroc", "acc", "pcc", "r2")):
        return "max"
    return "min"


def resolve_checkpoint_selection(args: argparse.Namespace) -> dict[str, str | None]:
    normalized_metric = BEST_CKPT_METRIC_ALIASES[args.best_ckpt_metric]
    monitor_override = args.monitor.strip() if args.monitor is not None else None
    if monitor_override is not None:
        if monitor_override.lower() in {"", "none", "null"}:
            return {
                "best_ckpt_metric": normalized_metric,
                "monitor": None,
                "monitor_mode": args.monitor_mode or "min",
                "monitor_source": "monitor_override_none",
            }
        return {
            "best_ckpt_metric": normalized_metric,
            "monitor": monitor_override,
            "monitor_mode": args.monitor_mode or default_monitor_mode(monitor_override),
            "monitor_source": "monitor_override",
        }
    monitor, default_mode = BEST_CKPT_METRIC_CONFIG[normalized_metric]
    return {
        "best_ckpt_metric": normalized_metric,
        "monitor": monitor,
        "monitor_mode": args.monitor_mode or default_mode,
        "monitor_source": "best_ckpt_metric",
    }


def resolve_scheduler_monitor(checkpoint_selection: dict[str, str | None]) -> tuple[str, str]:
    monitor = checkpoint_selection["monitor"]
    mode = checkpoint_selection["monitor_mode"]
    if monitor is not None:
        return monitor, mode or default_monitor_mode(monitor)
    best_metric = checkpoint_selection["best_ckpt_metric"] or "valid_auprc"
    default_monitor, default_mode = BEST_CKPT_METRIC_CONFIG[best_metric]
    return default_monitor, default_mode


class MonitorMetricGuard(pl.Callback):
    def __init__(self, monitor: str | None) -> None:
        super().__init__()
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or not self.monitor:
            return
        if self.monitor not in trainer.callback_metrics:
            available = ", ".join(sorted(trainer.callback_metrics))
            raise RuntimeError(
                f"checkpoint monitor metric {self.monitor!r} was not logged. "
                f"Available metrics: {available}"
            )
        value = trainer.callback_metrics[self.monitor]
        value_tensor = value.detach() if isinstance(value, torch.Tensor) else torch.tensor(float(value))
        if not torch.isfinite(value_tensor).all():
            raise RuntimeError(
                f"checkpoint monitor metric {self.monitor!r} is non-finite: {value_tensor.detach().cpu().tolist()}. "
                "For AUPRC/AUROC this usually means the evaluated validation batches contain only one class. "
                "Use full validation data or choose BEST_CKPT_METRIC=total_loss/loss1/loss2 for this run."
            )


def looks_like_multi_device(devices: object, accelerator: object) -> bool:
    accelerator_text = str(accelerator).lower()
    if accelerator_text in {"cpu", "mps"}:
        return False
    if isinstance(devices, int):
        return devices > 1
    text = str(devices).strip().lower()
    if text in {"auto", "-1"}:
        return torch.cuda.device_count() > 1
    if text.isdigit():
        return int(text) > 1
    if "," in text:
        return len([part for part in text.split(",") if part.strip()]) > 1
    return False


def resolve_strategy(args) -> str:
    if args.strategy != "auto":
        return args.strategy
    if args.model_type in GRAPH_MODEL_NAMES and looks_like_multi_device(args.devices, args.accelerator):
        return "ddp_find_unused_parameters_true"
    return "auto"


def build_data_loaders(args, artifacts: TrainingReadyArtifacts, drug_embedding: np.ndarray, drug_mode: str):
    split_dir = Path(args.split_dir) if args.split_dir else Path(args.training_ready_root) / args.dataset_group / "splits" / args.task_name
    train_indices = load_indices(split_dir, "train", args.split_strategy)
    valid_indices = load_indices(split_dir, "valid", args.split_strategy)
    test_indices = load_indices(split_dir, "test", args.split_strategy)
    if not train_indices:
        raise ValueError(f"split {args.split_strategy!r} has no train indices; choose a training split, not test_only")
    if not valid_indices:
        raise ValueError(
            f"split {args.split_strategy!r} has no valid indices; formal training requires a non-empty "
            "validation split to avoid selecting checkpoints on the test split"
        )
    row_to_set = load_row_to_set(split_dir)
    train_set_info = load_set_info(split_dir, "train", args.split_strategy)
    valid_set_info = load_set_info(split_dir, "valid", args.split_strategy)
    test_set_info = load_set_info(split_dir, "test", args.split_strategy)
    valid_source = "valid"

    dataset_kwargs = {
        "artifacts": artifacts,
        "batch_cov_list": args.batch_cov_list,
        "drug_mode": drug_mode,
        "drug_embedding_matrix": drug_embedding if drug_mode == "embedding" else None,
        "target_protein_max_length": args.target_protein_max_length,
        "effective_key1": args.effective_key1,
        "effective_key2": args.effective_key2,
    }
    train_dataset = ProteinTalkDataset(
        indices=train_indices,
        row_to_set_index=row_to_set,
        set_info=train_set_info,
        mode="train",
        epoch_len=args.epoch_len,
        **dataset_kwargs,
    )
    valid_dataset = ProteinTalkDataset(
        indices=valid_indices,
        row_to_set_index=row_to_set,
        set_info=valid_set_info,
        mode="eval",
        **dataset_kwargs,
    )
    test_dataset = ProteinTalkDataset(
        indices=test_indices,
        row_to_set_index=row_to_set,
        set_info=test_set_info,
        mode="eval",
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    df = artifacts.df
    def split_counts(indices: list[int]) -> dict[str, Any]:
        subset = df.iloc[indices]
        result: dict[str, Any] = {"count": len(indices)}
        if "feature_membership" in subset.columns:
            result["feature_membership"] = subset["feature_membership"].astype(str).value_counts(dropna=False).to_dict()
        if "source_task" in subset.columns:
            result["source_task"] = subset["source_task"].astype(str).value_counts(dropna=False).to_dict()
        label_key = getattr(args, "task_label_key", None)
        if label_key and label_key in subset.columns:
            non_empty = subset[label_key].astype("string").fillna("").str.strip().ne("")
            result["active_label_key"] = label_key
            result["active_label_nonempty_count"] = int(non_empty.sum())
            result["active_label_empty_count"] = int((~non_empty).sum())
        return result

    split_audit = {
        "train": split_counts(train_indices),
        "valid": split_counts(valid_indices),
        "test": split_counts(test_indices),
        "train_valid_overlap": len(set(train_indices) & set(valid_indices)),
        "train_test_overlap": len(set(train_indices) & set(test_indices)),
        "valid_test_overlap": len(set(valid_indices) & set(test_indices)),
    }

    return train_loader, valid_loader, test_loader, {
        "split_dir": str(split_dir),
        "train_count": len(train_indices),
        "valid_count": len(valid_indices),
        "valid_source": valid_source,
        "test_count": len(test_indices),
        "audit": split_audit,
    }


def load_model_state(lightning_model: ProteinTalkLightning, checkpoint_path: str | None, *, strict: bool = True) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = lightning_model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)} examples={missing[:5]}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)} examples={unexpected[:5]}")


def build_logger(args: argparse.Namespace, experiment_name: str):
    backend = "wandb" if args.log_to_wandb else args.logger_backend
    if backend == "none":
        return False

    loggers = []
    if backend in {"tensorboard", "both"}:
        loggers.append(TensorBoardLogger(save_dir=args.log_dir, name=experiment_name, version=None))
    if backend in {"wandb", "both"}:
        if WandbLogger is None:
            raise ImportError(
                "WandbLogger is unavailable. Install wandb in the flow_v2 environment or use --logger-backend tensorboard."
            )
        wandb_kwargs: dict[str, Any] = {
            "project": args.wandb_project,
            "name": experiment_name,
            "save_dir": args.log_dir,
            "log_model": args.wandb_log_model,
        }
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            wandb_kwargs["group"] = args.wandb_group
        if args.wandb_tags:
            wandb_kwargs["tags"] = args.wandb_tags
        if args.wandb_mode:
            wandb_kwargs["mode"] = args.wandb_mode
        loggers.append(WandbLogger(**wandb_kwargs))

    return loggers[0] if len(loggers) == 1 else loggers


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProteinTalk from data/training_ready")
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", choices=["ptv1", "ptv3"], default="ptv3")
    parser.add_argument("--task-name", default="ptv3_main_doubledrug")
    parser.add_argument("--split-strategy", default="pert_id_5fold_fold0")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--model-type", choices=sorted(SELECTED_MODEL_NAMES), default="attention_v10_hetero_cls_ee")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=None, help="Weight for active task label loss")
    parser.add_argument("--bce-weight1", type=float, default=1.0)
    parser.add_argument("--bce-weight2", type=float, default=1.0)
    parser.add_argument("--no-mse-loss", action="store_false", dest="have_mse_loss")
    parser.add_argument("--optimizer-name", "--optimizer_name", default="adamw")
    parser.add_argument("--positive-weight", "--positive_weight", type=float, default=None)
    parser.add_argument("--positive-weight1", type=float, default=None)
    parser.add_argument("--positive-weight2", type=float, default=None)
    parser.add_argument("--focal-loss", "--focal_loss", action="store_true", dest="focal_loss")
    parser.add_argument("--scheduler-name", "--scheduler_name", choices=["cosine", "step", "plateau", "cosine_warmup"], default=None)
    parser.add_argument("--fusion-mode", choices=["concat", "add"], default="concat")
    parser.add_argument("--perturb-fusion-mode", choices=["add", "concat", "mlp"], default="add")
    parser.add_argument("--target-protein-max-length", type=int, default=10)
    parser.add_argument("--cls-type", default="all_1")
    parser.add_argument("--graph-dropout", action="store_true")
    parser.add_argument(
        "--use-target",
        dest="use_target",
        action="store_true",
        default=True,
        help="Include target protein tokens in the PDI hetero graph model (default)",
    )
    parser.add_argument(
        "--no-use-target",
        dest="use_target",
        action="store_false",
        help="Disable target protein tokens in the PDI hetero graph model",
    )
    parser.add_argument("--target-protein-fusion-model", choices=["concat", "gate"], default="concat")
    parser.add_argument("--gate-weight", type=float, default=1.0)
    parser.add_argument("--pdi-input-orientation", choices=["drug_by_protein", "protein_by_drug"], default="drug_by_protein")
    parser.add_argument(
        "--pdi-mode",
        choices=["real", "zero"],
        default="real",
        help="Use the real PDI matrix or an all-zero matrix for controlled no-PDI ablation.",
    )
    parser.add_argument(
        "--emb-dataset-path",
        default="/mnt/shared-storage-user/beam/wuhao/H100/proteintalk/baseline/Geneformer/data/prot2gene_new/embed/geneformer_emb.npy",
    )
    parser.add_argument("--gene-emb-dim", type=int, default=768)
    parser.add_argument("--effective-key1", default=None)
    parser.add_argument("--effective-key2", default="synergy")
    parser.add_argument("--task-head", choices=["auto", "response", "synergy"], default="auto")
    parser.add_argument("--task-label-key", default=None)
    parser.add_argument("--task-mask-key", default=None)
    parser.add_argument(
        "--batch-cov-list",
        nargs="*",
        default=["machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time"],
    )
    parser.add_argument("--protein-embedding-path", default=None)
    parser.add_argument("--drug-embedding-path", default=None)
    parser.add_argument("--pdi-matrix-path", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epoch-len", type=int, default=None)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--save-every-n-epochs", type=int, default=1)
    parser.add_argument("--save-every-n-train-steps", type=int, default=None)
    parser.add_argument("--save-top-k", type=int, default=-1)
    parser.add_argument("--save-last-ckpt", action="store_true", default=True)
    parser.add_argument("--no-save-last-ckpt", action="store_false", dest="save_last_ckpt")
    parser.add_argument("--checkpoint-filename", default=None)
    parser.add_argument(
        "--best-ckpt-metric",
        choices=sorted(BEST_CKPT_METRIC_ALIASES),
        default="valid_auprc",
        help=(
            "Named validation metric for best-checkpoint selection. "
            "Aliases: total_loss, loss1, loss2, auprc."
        ),
    )
    parser.add_argument(
        "--monitor",
        default=None,
        help="Raw Lightning metric override for ModelCheckpoint; use only when --best-ckpt-metric is insufficient.",
    )
    parser.add_argument("--monitor-mode", choices=["min", "max"], default=None)
    parser.add_argument("--logger-backend", choices=["tensorboard", "wandb", "both", "none"], default="tensorboard")
    parser.add_argument("--log-to-wandb", "--log_to_wandb", action="store_true", dest="log_to_wandb")
    parser.add_argument("--wandb-project", default="aivc_proteintalk")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb-log-model", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument(
        "--allow-nonfinite-monitor",
        action="store_true",
        help="Allow training to continue when the checkpoint monitor is NaN/Inf.",
    )
    parser.add_argument("--unfreeze-at-epoch", type=int, default=None)
    parser.add_argument("--unfreeze-layer-name", default="embedding_proj")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-batches", type=int, default=0, help="Run N train batches through the model and exit")
    parser.add_argument("--limit-train-batches", type=parse_limit_batches, default=1.0)
    parser.add_argument("--limit-val-batches", type=parse_limit_batches, default=1.0)
    parser.add_argument("--limit-test-batches", type=parse_limit_batches, default=1.0)
    parser.add_argument("--skip-test", action="store_true", help="Skip trainer.test after fitting")
    parser.add_argument("--checkpoint-path", default=None, help="Optional checkpoint to resume/load before training")
    parser.add_argument(
        "--allow-partial-checkpoint-load",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys when initializing weights",
    )
    args = parser.parse_args()
    if args.split_strategy == "all_train_subset_test" and not args.skip_test:
        raise ValueError(
            "`all_train_subset_test` uses validation/test subsets drawn from the training anchors; "
            "run it with --skip-test and evaluate final claims with infer.py on the external extra-data tasks"
        )
    checkpoint_selection = resolve_checkpoint_selection(args)
    scheduler_monitor, scheduler_monitor_mode = resolve_scheduler_monitor(checkpoint_selection)

    training_ready_root = Path(args.training_ready_root)
    task_dir = training_ready_root / args.dataset_group / "tasks" / args.task_name
    meta_path = training_ready_root / args.dataset_group / "global_meta.json"
    defaults = default_derived_paths(training_ready_root, args.dataset_group)
    protein_embedding_path = Path(args.protein_embedding_path) if args.protein_embedding_path else defaults["protein_embedding"]
    drug_embedding_path = Path(args.drug_embedding_path) if args.drug_embedding_path else defaults["drug_embedding"]
    pdi_matrix_path = Path(args.pdi_matrix_path) if args.pdi_matrix_path else defaults["pdi_matrix"]
    args.effective_key1 = args.effective_key1 or infer_label_key(args.task_name)

    pl.seed_everything(args.seed, workers=True)
    artifacts = TrainingReadyArtifacts(task_dir, meta_path)
    protein_embedding = load_embedding_matrix(protein_embedding_path)
    drug_embedding = load_embedding_matrix(drug_embedding_path)
    pdi_matrix = load_pdi_matrix(pdi_matrix_path, model_type=args.model_type, pdi_mode=args.pdi_mode)
    drug_mode = "index" if args.model_type in GRAPH_MODEL_NAMES else "embedding"
    task_loss_config = resolve_task_loss_config(
        task_name=args.task_name,
        task_head=args.task_head,
        effective_key1=args.effective_key1,
        effective_key2=args.effective_key2,
        task_label_key=args.task_label_key,
        task_mask_key=args.task_mask_key,
    )
    args.task_head = task_loss_config["task_head"]
    args.task_label_key = task_loss_config["task_label_key"]
    args.task_mask_key = task_loss_config["task_mask_key"]
    active_bce_weight = args.bce_weight
    if active_bce_weight is None:
        active_bce_weight = args.bce_weight2 if task_loss_config["task_head"] == "synergy" else args.bce_weight1
    active_positive_weight = args.positive_weight
    if active_positive_weight is None:
        active_positive_weight = (
            args.positive_weight2 if task_loss_config["task_head"] == "synergy" else args.positive_weight1
        )

    train_loader, valid_loader, test_loader, split_summary = build_data_loaders(args, artifacts, drug_embedding, drug_mode)
    model = build_model(
        args.model_type,
        artifacts=ModelArtifacts(
            protein_embedding=protein_embedding,
            drug_embedding=drug_embedding,
            ordered_protein_index=artifacts.ordered_protein_index,
            pdi_matrix=pdi_matrix,
        ),
        topk_genes=artifacts.expression_matrix.shape[1],
        batch_cov_list=args.batch_cov_list,
        batch_cov_category_sizes=category_sizes(artifacts.meta, args.batch_cov_list),
        hidden_dim=args.hidden_dim,
        perturb_fusion_mode=args.perturb_fusion_mode,
        target_protein_max_length=args.target_protein_max_length,
        dropout=args.dropout,
        fusion_mode=args.fusion_mode,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        cls_type=args.cls_type,
        graph_dropout=args.graph_dropout,
        use_target=args.use_target,
        target_protein_fusion_model=args.target_protein_fusion_model,
        gate_weight=args.gate_weight,
        reverse_pdi=args.pdi_input_orientation == "drug_by_protein",
        emb_dataset_path=args.emb_dataset_path,
        gene_emb_dim=args.gene_emb_dim,
    )
    lightning_model = ProteinTalkLightning(
        model,
        mse_weight=args.mse_weight,
        bce_weight=active_bce_weight,
        bce_weight1=args.bce_weight1,
        bce_weight2=args.bce_weight2,
        learning_rate=args.learning_rate,
        effective_key1=args.effective_key1,
        effective_key2=args.effective_key2,
        task_label_key=task_loss_config["task_label_key"],
        task_mask_key=task_loss_config["task_mask_key"],
        task_head=task_loss_config["task_head"],
        optimizer_name=args.optimizer_name,
        positive_weight=active_positive_weight,
        positive_weight1=args.positive_weight1,
        positive_weight2=args.positive_weight2,
        scheduler_name=args.scheduler_name,
        scheduler_monitor=scheduler_monitor,
        scheduler_monitor_mode=scheduler_monitor_mode,
        focal_loss=args.focal_loss,
        have_mse_loss=args.have_mse_loss,
    )
    load_model_state(lightning_model, args.checkpoint_path, strict=not args.allow_partial_checkpoint_load)

    experiment_name = args.experiment_name or f"{args.task_name}_{args.model_type}_{args.split_strategy}"
    manifest = {
        "generated_at": iso_now(),
        "run_status": "fit_started",
        "experiment_name": experiment_name,
        "dataset_group": args.dataset_group,
        "task_name": args.task_name,
        "split_strategy": args.split_strategy,
        "model_type": args.model_type,
        "task_dir": str(task_dir.resolve()),
        "meta_path": str(meta_path.resolve()),
        "protein_embedding_path": str(protein_embedding_path.resolve()),
        "drug_embedding_path": str(drug_embedding_path.resolve()),
        "pdi_matrix_path": str(pdi_matrix_path.resolve()) if pdi_matrix is not None else None,
        "pdi_mode": args.pdi_mode if args.model_type in GRAPH_MODEL_NAMES else None,
        "ordered_protein_index_path": str((task_dir / "feature_ordered_protein_index.json").resolve()),
        "pdi_input_orientation": args.pdi_input_orientation,
        "effective_key1": args.effective_key1,
        "effective_key2": args.effective_key2,
        "task_head": task_loss_config["task_head"],
        "task_label_key": task_loss_config["task_label_key"],
        "task_mask_key": task_loss_config["task_mask_key"],
        "batch_cov_list": args.batch_cov_list,
        "fusion_mode": args.fusion_mode,
        "perturb_fusion_mode": args.perturb_fusion_mode,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "cls_type": args.cls_type,
        "graph_dropout": args.graph_dropout,
        "use_target": args.use_target,
        "target_protein_fusion_model": args.target_protein_fusion_model,
        "gate_weight": args.gate_weight,
        "optimizer_name": args.optimizer_name,
        "mse_weight": args.mse_weight,
        "have_mse_loss": args.have_mse_loss,
        "bce_weight": active_bce_weight,
        "bce_weight1": args.bce_weight1,
        "bce_weight2": args.bce_weight2,
        "positive_weight": active_positive_weight,
        "positive_weight1": args.positive_weight1,
        "positive_weight2": args.positive_weight2,
        "focal_loss": args.focal_loss,
        "scheduler_name": args.scheduler_name,
        "unfreeze_at_epoch": args.unfreeze_at_epoch,
        "unfreeze_layer_name": args.unfreeze_layer_name,
        "gradient_clip_val": args.gradient_clip_val,
        "strategy": resolve_strategy(args),
        "allow_partial_checkpoint_load": args.allow_partial_checkpoint_load,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "target_protein_max_length": args.target_protein_max_length,
        "gene_emb_dim": args.gene_emb_dim,
        "emb_dataset_path": args.emb_dataset_path if args.model_type == "baseline_emb_v3" else None,
        "split_summary": split_summary,
        "test_policy": "skip_test" if args.skip_test else "after_fit_best_validation_checkpoint",
        "best_ckpt_metric": checkpoint_selection["best_ckpt_metric"],
        "monitor": checkpoint_selection["monitor"] or "none",
        "monitor_mode": checkpoint_selection["monitor_mode"],
        "monitor_source": checkpoint_selection["monitor_source"],
        "scheduler_monitor": scheduler_monitor,
        "scheduler_monitor_mode": scheduler_monitor_mode,
        "allow_nonfinite_monitor": args.allow_nonfinite_monitor,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "save_every_n_epochs": args.save_every_n_epochs,
        "save_every_n_train_steps": args.save_every_n_train_steps,
        "save_top_k": args.save_top_k,
        "save_last_ckpt": args.save_last_ckpt,
        "checkpoint_filename": args.checkpoint_filename,
        "logger_backend": "wandb" if args.log_to_wandb else args.logger_backend,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_group": args.wandb_group,
        "wandb_tags": args.wandb_tags,
        "wandb_mode": args.wandb_mode,
        "wandb_log_model": args.wandb_log_model,
        "log_every_n_steps": args.log_every_n_steps,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "limit_test_batches": args.limit_test_batches,
    }

    if args.dry_run_batches:
        lightning_model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                out = lightning_model(batch)
                print(
                    f"dry_run batch={batch_idx} expression={tuple(out[0].shape)} "
                    f"response_logits={tuple(out[1].shape)} synergy_logits={tuple(out[2].shape)}"
                )
                if batch_idx + 1 >= args.dry_run_batches:
                    break
        return

    run_dir = Path(args.checkpoint_dir) / experiment_name
    dump_json(run_dir / "run_manifest.json", manifest)

    monitor = checkpoint_selection["monitor"]
    monitor_mode = checkpoint_selection["monitor_mode"] or "min"
    if args.save_every_n_train_steps is not None and args.save_every_n_train_steps <= 0:
        raise ValueError("--save-every-n-train-steps must be a positive integer")
    if args.save_every_n_train_steps is not None and monitor and monitor.startswith("val/"):
        raise ValueError(
            "Step-based checkpointing cannot monitor a validation metric during train steps. "
            "Use --monitor none to save periodic step checkpoints, or monitor a train/*_step metric."
        )
    checkpoint_filename = args.checkpoint_filename
    if checkpoint_filename is None:
        checkpoint_filename = "{epoch}-{step}" if args.save_every_n_train_steps is not None else "{epoch}"
    checkpoint_kwargs: dict[str, Any] = {
        "dirpath": str(run_dir),
        "filename": checkpoint_filename,
        "monitor": monitor,
        "mode": monitor_mode,
        "save_top_k": args.save_top_k,
        "save_last": args.save_last_ckpt,
        "save_on_train_epoch_end": False,
    }
    if args.save_every_n_train_steps is not None:
        checkpoint_kwargs["every_n_train_steps"] = args.save_every_n_train_steps
    else:
        checkpoint_kwargs["every_n_epochs"] = args.save_every_n_epochs
    checkpoint_callback = ModelCheckpoint(
        **checkpoint_kwargs,
    )
    logger = build_logger(args, experiment_name)
    callbacks = [
        checkpoint_callback,
    ]
    if monitor and not args.allow_nonfinite_monitor:
        callbacks.append(MonitorMetricGuard(monitor))
    if logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    if args.unfreeze_at_epoch is not None:
        callbacks.append(
            UnfreezeCallback(unfreeze_at_epoch=args.unfreeze_at_epoch, layer_name=args.unfreeze_layer_name)
        )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=resolve_strategy(args),
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip_val,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )
    trainer.fit(lightning_model, train_loader, valid_loader)
    manifest["run_status"] = "fit_completed"
    manifest["fit_completed_at"] = iso_now()
    manifest["best_model_path"] = checkpoint_callback.best_model_path
    manifest["best_model_score"] = (
        float(checkpoint_callback.best_model_score.detach().cpu())
        if checkpoint_callback.best_model_score is not None
        else None
    )
    if not args.skip_test:
        test_ckpt_path = checkpoint_callback.best_model_path or None
        manifest["test_checkpoint_path"] = test_ckpt_path
        manifest["test_started_at"] = iso_now()
        manifest["test_status"] = "running"
        dump_json(run_dir / "run_manifest.json", manifest)
        test_results = json_safe(trainer.test(lightning_model, test_loader, ckpt_path=test_ckpt_path))
        test_metric_names = sorted(
            {
                metric_name
                for result in test_results
                if isinstance(result, dict)
                for metric_name in result
            }
        )
        manifest["test_results"] = test_results
        manifest["test_result_detail"] = {
            "source": "pytorch_lightning.Trainer.test return value",
            "checkpoint_path": test_ckpt_path,
            "task_name": args.task_name,
            "split_strategy": args.split_strategy,
            "split_name": "test",
            "test_count": split_summary["test_count"],
            "limit_test_batches": args.limit_test_batches,
            "metric_names": test_metric_names,
        }
        manifest["test_status"] = "test_completed"
        manifest["test_completed_at"] = iso_now()
    else:
        manifest["test_checkpoint_path"] = None
        manifest["test_status"] = "skipped"
        manifest["test_results"] = []
        manifest["test_result_detail"] = {
            "source": "pytorch_lightning.Trainer.test return value",
            "reason": "--skip-test was set",
            "metric_names": [],
        }
    dump_json(run_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
