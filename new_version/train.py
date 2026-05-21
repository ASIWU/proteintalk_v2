#!/usr/bin/env python3
"""Train the fast ProteinTalk model from existing training-ready artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from fast_delta_model import FastDeltaDrugResponseModel
from graph_feature_utils import build_or_load_graph_features
from fast_lightning import FastProteinTalkLightning
from training_ready_fast_dataset import (
    FastProteinTalkDataset,
    FastTrainingReadyArtifacts,
    category_sizes,
    compute_positive_weight,
    dump_json,
    load_embedding_matrix,
    load_indices,
    load_json,
    load_row_to_set,
    load_set_info,
)


DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
DEFAULT_CHECKPOINT_DIR = THIS_DIR / "checkpoints"
DEFAULT_LOG_DIR = THIS_DIR / "logs"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def parse_limit_batches(value: str) -> int | float:
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return float(text)


def looks_like_multi_device(devices: object, accelerator: object) -> bool:
    if str(accelerator).lower() in {"cpu", "mps"}:
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


def resolve_strategy(args: argparse.Namespace) -> str:
    if args.strategy != "auto":
        return args.strategy
    if looks_like_multi_device(args.devices, args.accelerator):
        return "ddp"
    return "auto"


def default_derived_paths(training_ready_root: Path, dataset_group: str) -> dict[str, Path]:
    derived = training_ready_root / dataset_group / "derived"
    return {
        "protein_embedding": derived / "protein_embedding_esm.pkl",
        "drug_embedding": derived / "drug_embedding_morgan_2048.pkl",
        "ppi_matrix": derived / "ppi_matrix.npy",
        "pdi_matrix": derived / "pdi_matrix.npy",
        "ddi_matrix": derived / "ddi_matrix.npy",
    }


def graph_feature_blocks_from_meta(meta: dict[str, Any] | None) -> list[dict[str, int | str]] | None:
    if not meta:
        return None
    slices = meta.get("feature_slices")
    if not isinstance(slices, dict):
        return None
    blocks = []
    for name, span in slices.items():
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        blocks.append({"name": str(name), "start": int(span[0]), "end": int(span[1])})
    blocks.sort(key=lambda item: int(item["start"]))
    return blocks


def resolve_positive_weight(args: argparse.Namespace, artifacts: FastTrainingReadyArtifacts, train_indices: list[int]) -> float | None:
    text = str(args.positive_weight).strip().lower()
    if text in {"", "none", "null", "0"}:
        return None
    if text == "auto":
        return compute_positive_weight(
            df=artifacts.df,
            indices=train_indices,
            label_key=args.task_label_key,
            task_head=args.task_head,
            max_weight=args.max_positive_weight,
        )
    return float(text)


def split_counts(artifacts: FastTrainingReadyArtifacts, indices: list[int], label_key: str) -> dict[str, Any]:
    subset = artifacts.df.iloc[indices]
    result: dict[str, Any] = {"count": len(indices)}
    if "pert_id1" in subset.columns:
        result["pert_id1_unique"] = int(subset["pert_id1"].nunique(dropna=True))
    if "Cell" in subset.columns:
        result["cell_unique"] = int(subset["Cell"].nunique(dropna=True))
    if label_key in subset.columns:
        result["label_counts"] = subset[label_key].astype("string").fillna("<NA>").value_counts(dropna=False).to_dict()
    return result


def build_dataloaders(
    args: argparse.Namespace,
    artifacts: FastTrainingReadyArtifacts,
    drug_embedding: np.ndarray,
    ddi_matrix: np.ndarray | None,
    graph_feature_matrix: np.ndarray | None,
):
    split_dir = Path(args.split_dir) if args.split_dir else Path(args.training_ready_root) / args.dataset_group / "splits" / args.task_name
    train_indices = load_indices(split_dir, "train", args.split_strategy)
    valid_indices = load_indices(split_dir, "valid", args.split_strategy)
    test_indices = load_indices(split_dir, "test", args.split_strategy)
    if not train_indices:
        raise ValueError(f"split {args.split_strategy!r} has no train indices")
    if not valid_indices:
        raise ValueError(f"split {args.split_strategy!r} has no valid indices")
    row_to_set = load_row_to_set(split_dir)
    train_set_info = load_set_info(split_dir, "train", args.split_strategy)
    valid_set_info = load_set_info(split_dir, "valid", args.split_strategy)
    test_set_info = load_set_info(split_dir, "test", args.split_strategy)
    dataset_kwargs = {
        "artifacts": artifacts,
        "drug_embedding_matrix": drug_embedding,
        "batch_cov_list": args.batch_cov_list,
        "target_protein_max_length": args.target_protein_max_length,
        "effective_key1": args.effective_key1,
        "effective_key2": args.effective_key2,
        "ddi_matrix": ddi_matrix,
        "graph_feature_matrix": graph_feature_matrix,
        "graph_feature_enabled": args.graph_feature_mode == "real",
    }
    train_dataset = FastProteinTalkDataset(
        indices=train_indices,
        row_to_set_index=row_to_set,
        set_info=train_set_info,
        mode="train",
        epoch_len=args.epoch_len,
        **dataset_kwargs,
    )
    valid_dataset = FastProteinTalkDataset(
        indices=valid_indices,
        row_to_set_index=row_to_set,
        set_info=valid_set_info,
        mode="eval",
        **dataset_kwargs,
    )
    test_dataset = FastProteinTalkDataset(
        indices=test_indices,
        row_to_set_index=row_to_set,
        set_info=test_set_info,
        mode="eval",
        **dataset_kwargs,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.drop_last,
        **loader_kwargs,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    split_summary = {
        "split_dir": str(split_dir.resolve()),
        "train": split_counts(artifacts, train_indices, args.task_label_key),
        "valid": split_counts(artifacts, valid_indices, args.task_label_key),
        "test": split_counts(artifacts, test_indices, args.task_label_key),
        "train_valid_overlap": len(set(train_indices) & set(valid_indices)),
        "train_test_overlap": len(set(train_indices) & set(test_indices)),
        "valid_test_overlap": len(set(valid_indices) & set(test_indices)),
    }
    return train_loader, valid_loader, test_loader, split_summary, train_indices


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train new_version FastDelta ProteinTalk model")
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", choices=["ptv1", "ptv3"], default="ptv3")
    parser.add_argument("--task-name", default="ptv3_main_singledrug")
    parser.add_argument("--split-strategy", default="pert_stratified_5fold_fold0")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--expression-latent-dim", type=int, default=512)
    parser.add_argument("--covariate-embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--control-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=3)
    parser.add_argument("--target-layers", type=int, default=2)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", default="none", help="float, auto, none, or 0")
    parser.add_argument("--max-positive-weight", type=float, default=20.0)
    parser.add_argument("--no-mse-loss", action="store_false", dest="have_mse_loss")
    parser.add_argument("--mse-gene-subsample", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--optimizer-name", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--scheduler-name", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--task-head", choices=["auto", "response", "synergy"], default="auto")
    parser.add_argument("--effective-key1", default=None)
    parser.add_argument("--effective-key2", default="synergy")
    parser.add_argument("--task-label-key", default=None)
    parser.add_argument(
        "--batch-cov-list",
        nargs="*",
        default=["machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time"],
    )
    parser.add_argument("--target-protein-max-length", type=int, default=32)
    parser.add_argument("--protein-embedding-path", default=None)
    parser.add_argument("--drug-embedding-path", default=None)
    parser.add_argument("--ppi-matrix-path", default=None)
    parser.add_argument("--pdi-matrix-path", default=None)
    parser.add_argument("--ddi-matrix-path", default=None)
    parser.add_argument("--graph-feature-mode", choices=["real", "zero", "off"], default="real")
    parser.add_argument("--graph-feature-dim", type=int, default=128)
    parser.add_argument("--graph-feature-seed", type=int, default=17)
    parser.add_argument("--graph-structural-rp", action="store_true")
    parser.add_argument("--graph-multihop", action="store_true")
    parser.add_argument("--graph-cache-dir", default=str(THIS_DIR / "graph_cache"))
    parser.add_argument("--force-graph-cache-rebuild", action="store_true")
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-init-scale", type=float, default=0.1)
    parser.add_argument("--graph-drug-concat", action="store_true")
    parser.add_argument("--graph-pair-add-scale", type=float, default=0.0)
    parser.add_argument("--graph-logit-scale", type=float, default=0.0)
    parser.add_argument("--graph-jump-fusion", choices=["concat", "selective"], default="concat")
    parser.add_argument("--graph-jump-gate", choices=["softmax", "sparsemax"], default="softmax")
    parser.add_argument("--graph-jump-temperature", type=float, default=1.0)
    parser.add_argument("--protein-concat-mode", choices=["off", "pcep"], default="off")
    parser.add_argument("--protein-concat-dim", type=int, default=64)
    parser.add_argument("--protein-concat-topk", type=int, default=512)
    parser.add_argument("--protein-concat-init-scale", type=float, default=0.1)
    parser.add_argument("--protein-concat-seed", type=int, default=23)
    parser.add_argument("--use-ddi", action="store_true")
    parser.add_argument("--absolute-expression-head", action="store_false", dest="residual_expression")
    parser.add_argument("--init-delta-scale", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--save-top-k", type=int, default=1)
    parser.add_argument("--save-last-ckpt", action="store_true", default=True)
    parser.add_argument("--no-save-last-ckpt", action="store_false", dest="save_last_ckpt")
    parser.add_argument("--monitor", default="val/task_auprc")
    parser.add_argument("--monitor-mode", choices=["min", "max"], default="max")
    parser.add_argument("--logger-backend", choices=["tensorboard", "none"], default="tensorboard")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epoch-len", type=int, default=None)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-batches", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=parse_limit_batches, default=1.0)
    parser.add_argument("--limit-val-batches", type=parse_limit_batches, default=1.0)
    parser.add_argument("--limit-test-batches", type=parse_limit_batches, default=1.0)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if args.scheduler_name == "none":
        args.scheduler_name = None
    args.task_head = infer_task_head(args.task_name) if args.task_head == "auto" else args.task_head
    args.effective_key1 = args.effective_key1 or infer_label_key(args.task_name)
    args.task_label_key = args.task_label_key or (args.effective_key2 if args.task_head == "synergy" else args.effective_key1)

    pl.seed_everything(args.seed, workers=True)
    training_ready_root = Path(args.training_ready_root)
    task_dir = training_ready_root / args.dataset_group / "tasks" / args.task_name
    meta_path = training_ready_root / args.dataset_group / "global_meta.json"
    defaults = default_derived_paths(training_ready_root, args.dataset_group)
    protein_embedding_path = Path(args.protein_embedding_path) if args.protein_embedding_path else defaults["protein_embedding"]
    drug_embedding_path = Path(args.drug_embedding_path) if args.drug_embedding_path else defaults["drug_embedding"]
    ppi_matrix_path = Path(args.ppi_matrix_path) if args.ppi_matrix_path else defaults["ppi_matrix"]
    pdi_matrix_path = Path(args.pdi_matrix_path) if args.pdi_matrix_path else defaults["pdi_matrix"]
    ddi_matrix_path = Path(args.ddi_matrix_path) if args.ddi_matrix_path else defaults["ddi_matrix"]

    artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
    protein_embedding = load_embedding_matrix(protein_embedding_path)
    drug_embedding = load_embedding_matrix(drug_embedding_path)
    graph_feature_matrix = None
    graph_feature_meta = None
    if args.graph_feature_mode in {"real", "zero"}:
        graph_feature_matrix, graph_feature_meta = build_or_load_graph_features(
            cache_dir=args.graph_cache_dir,
            dataset_group=args.dataset_group,
            ppi_matrix_path=ppi_matrix_path,
            pdi_matrix_path=pdi_matrix_path,
            ddi_matrix_path=ddi_matrix_path,
            protein_embedding=protein_embedding,
            drug_embedding=drug_embedding,
            graph_feature_dim=args.graph_feature_dim,
            seed=args.graph_feature_seed,
            include_structural_rp=args.graph_structural_rp,
            include_multihop=args.graph_multihop,
            force_rebuild=args.force_graph_cache_rebuild,
        )
    ddi_matrix = np.load(ddi_matrix_path, mmap_mode="r") if args.use_ddi else None
    train_loader, valid_loader, test_loader, split_summary, train_indices = build_dataloaders(
        args,
        artifacts,
        drug_embedding,
        ddi_matrix,
        graph_feature_matrix,
    )
    positive_weight = resolve_positive_weight(args, artifacts, train_indices)

    model = FastDeltaDrugResponseModel(
        n_genes=int(artifacts.expression_matrix.shape[1]),
        drug_embedding_dim=int(drug_embedding.shape[1]),
        protein_embedding=protein_embedding,
        ordered_protein_index=artifacts.ordered_protein_index,
        covariate_sizes=category_sizes(artifacts.meta, args.batch_cov_list),
        hidden_dim=args.hidden_dim,
        expression_latent_dim=args.expression_latent_dim,
        covariate_embedding_dim=args.covariate_embedding_dim,
        dropout=args.dropout,
        control_layers=args.control_layers,
        fusion_layers=args.fusion_layers,
        target_layers=args.target_layers,
        graph_feature_dim=0 if graph_feature_matrix is None else int(graph_feature_matrix.shape[1]),
        graph_layers=args.graph_layers,
        graph_init_scale=args.graph_init_scale,
        graph_drug_concat=args.graph_drug_concat,
        graph_pair_add_scale=args.graph_pair_add_scale,
        graph_logit_scale=args.graph_logit_scale,
        graph_feature_blocks=graph_feature_blocks_from_meta(graph_feature_meta),
        graph_jump_fusion=args.graph_jump_fusion,
        graph_jump_gate=args.graph_jump_gate,
        graph_jump_temperature=args.graph_jump_temperature,
        protein_concat_mode=args.protein_concat_mode,
        protein_concat_dim=args.protein_concat_dim,
        protein_concat_topk=args.protein_concat_topk,
        protein_concat_init_scale=args.protein_concat_init_scale,
        protein_concat_seed=args.protein_concat_seed,
        use_ddi=args.use_ddi,
        residual_expression=args.residual_expression,
        init_delta_scale=args.init_delta_scale,
    )
    if args.compile_model:
        model = torch.compile(model)
    lightning_model = FastProteinTalkLightning(
        model,
        task_head=args.task_head,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mse_weight=args.mse_weight,
        bce_weight=args.bce_weight,
        positive_weight=positive_weight,
        have_mse_loss=args.have_mse_loss,
        optimizer_name=args.optimizer_name,
        scheduler_name=args.scheduler_name,
        max_epochs=args.max_epochs,
        mse_gene_subsample=args.mse_gene_subsample,
        label_smoothing=args.label_smoothing,
    )

    experiment_name = args.experiment_name or f"{args.task_name}_fast_delta_{args.split_strategy}"
    run_dir = Path(args.checkpoint_dir) / experiment_name
    manifest = {
        "generated_at": iso_now(),
        "run_status": "initialized",
        "experiment_name": experiment_name,
        "implementation": "new_version.fast_delta",
        "dataset_group": args.dataset_group,
        "task_name": args.task_name,
        "split_strategy": args.split_strategy,
        "task_head": args.task_head,
        "effective_key1": args.effective_key1,
        "effective_key2": args.effective_key2,
        "task_label_key": args.task_label_key,
        "task_dir": str(task_dir.resolve()),
        "meta_path": str(meta_path.resolve()),
        "protein_embedding_path": str(protein_embedding_path.resolve()),
        "drug_embedding_path": str(drug_embedding_path.resolve()),
        "ppi_matrix_path": str(ppi_matrix_path.resolve()) if args.graph_feature_mode in {"real", "zero"} else None,
        "pdi_matrix_path": str(pdi_matrix_path.resolve()) if args.graph_feature_mode in {"real", "zero"} else None,
        "ddi_matrix_path": str(ddi_matrix_path.resolve()) if (args.use_ddi or args.graph_feature_mode in {"real", "zero"}) else None,
        "graph_feature_mode": args.graph_feature_mode,
        "graph_feature_meta": json_safe(graph_feature_meta),
        "split_summary": split_summary,
        "positive_weight": positive_weight,
        "args": json_safe(vars(args)),
        "model_parameter_count": int(sum(param.numel() for param in model.parameters())),
        "trainable_parameter_count": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
    }

    if args.dry_run_batches:
        lightning_model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.accelerator != "cpu" else "cpu")
        lightning_model.to(device)
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                out = lightning_model(batch)
                print(
                    f"dry_run batch={batch_idx} expression={tuple(out[0].shape)} "
                    f"response_logits={tuple(out[1].shape)} synergy_logits={tuple(out[2].shape)}"
                )
                if batch_idx + 1 >= args.dry_run_batches:
                    break
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest["run_status"] = "fit_started"
    dump_json(run_dir / "run_manifest.json", manifest)

    callbacks: list[Any] = []
    checkpointing_enabled = args.save_top_k != 0 or args.save_last_ckpt
    if checkpointing_enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(run_dir),
                filename="{epoch}-{step}",
                monitor=args.monitor if args.monitor.lower() not in {"none", "null", ""} else None,
                mode=args.monitor_mode,
                save_top_k=args.save_top_k,
                save_last=args.save_last_ckpt,
                save_on_train_epoch_end=False,
            )
        )
    logger: TensorBoardLogger | bool
    if args.logger_backend == "none":
        logger = False
    else:
        logger = TensorBoardLogger(save_dir=args.log_dir, name=experiment_name, version=None)
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=resolve_strategy(args),
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_checkpointing=checkpointing_enabled,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )
    trainer.fit(lightning_model, train_loader, valid_loader)
    manifest["run_status"] = "fit_completed"
    manifest["fit_completed_at"] = iso_now()
    checkpoint_callback = next((callback for callback in callbacks if isinstance(callback, ModelCheckpoint)), None)
    if checkpoint_callback is not None:
        manifest["best_model_path"] = checkpoint_callback.best_model_path
        manifest["best_model_score"] = (
            float(checkpoint_callback.best_model_score.detach().cpu())
            if checkpoint_callback.best_model_score is not None
            else None
        )
    else:
        manifest["best_model_path"] = None
        manifest["best_model_score"] = None
    if args.skip_test:
        manifest["test_status"] = "skipped"
        manifest["test_results"] = []
    else:
        test_ckpt_path = manifest["best_model_path"] or None
        manifest["test_status"] = "running"
        manifest["test_checkpoint_path"] = test_ckpt_path
        dump_json(run_dir / "run_manifest.json", manifest)
        manifest["test_results"] = json_safe(trainer.test(lightning_model, test_loader, ckpt_path=test_ckpt_path))
        manifest["test_status"] = "test_completed"
        manifest["test_completed_at"] = iso_now()
    dump_json(run_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
