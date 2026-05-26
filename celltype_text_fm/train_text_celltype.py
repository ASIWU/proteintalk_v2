#!/usr/bin/env python3
"""Train fast_delta with frozen cell-type biomedical text embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

import train as ptv_train
from celltype_text_fm.text_features import DEFAULT_MODEL_NAME, load_or_build_cell_type_text_features
from dataset.training_ready_fast_dataset import (
    FastProteinTalkDataset,
    FastTrainingReadyArtifacts,
    load_embedding_matrix,
    load_indices,
    load_row_to_set,
    load_set_info,
)
from model.fast_delta_model import FastDeltaDrugResponseModel
from model.fast_lightning import FastProteinTalkLightning
from model.graph_feature_utils import build_or_load_graph_features
from model.training_ready_models import FAST_DELTA_MODEL_NAME


def parse_limit_batches(value: str) -> int | float:
    return ptv_train.parse_limit_batches(value)


def maybe_drop_covariates(args: argparse.Namespace) -> None:
    batch_covariates = list(args.batch_cov_list)
    if args.drop_cell_covariate:
        batch_covariates = [item for item in batch_covariates if item != "Cell"]
    if args.drop_cell_type_covariate:
        batch_covariates = [item for item in batch_covariates if item != "cell_type"]
    args.batch_cov_list = batch_covariates


def build_text_data_loaders(
    args: argparse.Namespace,
    artifacts: FastTrainingReadyArtifacts,
    drug_embedding: np.ndarray,
    ddi_matrix: np.ndarray | None,
    graph_feature_matrix: np.ndarray | None,
    text_feature_matrix: np.ndarray | None,
):
    split_dir = (
        Path(args.split_dir)
        if args.split_dir
        else Path(args.training_ready_root) / args.dataset_group / "splits" / args.task_name
    )
    train_indices = load_indices(split_dir, "train", args.split_strategy)
    valid_indices = load_indices(split_dir, "valid", args.split_strategy)
    test_indices = load_indices(split_dir, "test", args.split_strategy)
    raw_train_indices = list(train_indices)
    train_indices, train_filter_summary = ptv_train.filter_train_indices_by_inactive_label_ratio(
        artifacts=artifacts,
        indices=train_indices,
        label_key=args.task_label_key,
        task_head=args.task_head,
        max_inactive_ratio=args.inactive_label_train_ratio,
        seed=args.seed,
    )
    if not train_indices:
        raise ValueError(f"split {args.split_strategy!r} has no train indices after filtering")
    if not valid_indices:
        raise ValueError(f"split {args.split_strategy!r} has no valid indices")
    row_to_set = load_row_to_set(split_dir)
    train_set_info = load_set_info(split_dir, "train", args.split_strategy)
    valid_set_info = load_set_info(split_dir, "valid", args.split_strategy)
    test_set_info = load_set_info(split_dir, "test", args.split_strategy)
    covariate_unknown_indices = ptv_train.fast_covariate_unknown_indices(args, artifacts)
    covariate_known_values = (
        ptv_train.fast_covariate_known_values(args, artifacts, train_indices) if covariate_unknown_indices else {}
    )
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
        "covariate_known_values": covariate_known_values,
        "covariate_unknown_indices": covariate_unknown_indices,
        "covariate_unk_dropout": args.covariate_unk_dropout,
        "prior_feature_matrix": text_feature_matrix,
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
    train_sampler = ptv_train.fast_active_label_sampler(
        artifacts=artifacts,
        indices=train_indices,
        label_key=args.task_label_key,
        task_head=args.task_head,
        active_weight=args.active_label_sampling_weight,
        positive_weight=args.positive_label_sampling_weight,
        num_samples=len(train_dataset),
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        drop_last=args.drop_last,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    split_summary = {
        "split_dir": str(split_dir.resolve()),
        "raw_train": ptv_train.fast_split_counts(artifacts, raw_train_indices, args.task_label_key),
        "train": ptv_train.fast_split_counts(artifacts, train_indices, args.task_label_key),
        "valid": ptv_train.fast_split_counts(artifacts, valid_indices, args.task_label_key),
        "test": ptv_train.fast_split_counts(artifacts, test_indices, args.task_label_key),
        "train_filter": train_filter_summary,
        "train_valid_overlap": len(set(train_indices) & set(valid_indices)),
        "train_test_overlap": len(set(train_indices) & set(test_indices)),
        "valid_test_overlap": len(set(valid_indices) & set(test_indices)),
        "covariate_unk_for_unseen": args.covariate_unk_for_unseen,
        "covariate_unk_fields": list(covariate_unknown_indices),
        "covariate_unk_dropout": args.covariate_unk_dropout,
    }
    return train_loader, valid_loader, test_loader, split_summary, train_indices


def run_training(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("high")
    maybe_drop_covariates(args)
    checkpoint_selection = ptv_train.resolve_checkpoint_selection(args)
    scheduler_monitor, scheduler_monitor_mode = ptv_train.resolve_scheduler_monitor(checkpoint_selection)
    if args.scheduler_name == "none":
        args.scheduler_name = None
    args.effective_key1 = args.effective_key1 or ptv_train.infer_label_key(args.task_name)
    task_loss_config = ptv_train.resolve_task_loss_config(
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

    training_ready_root = Path(args.training_ready_root)
    task_dir = training_ready_root / args.dataset_group / "tasks" / args.task_name
    meta_path = training_ready_root / args.dataset_group / "global_meta.json"
    defaults = ptv_train.default_derived_paths(training_ready_root, args.dataset_group)
    protein_embedding_path = Path(args.protein_embedding_path) if args.protein_embedding_path else defaults["protein_embedding"]
    drug_embedding_path = Path(args.drug_embedding_path) if args.drug_embedding_path else defaults["drug_embedding"]
    ppi_matrix_path = Path(args.ppi_matrix_path) if args.ppi_matrix_path else defaults["ppi_matrix"]
    pdi_matrix_path = Path(args.pdi_matrix_path) if args.pdi_matrix_path else defaults["pdi_matrix"]
    ddi_matrix_path = Path(args.ddi_matrix_path) if args.ddi_matrix_path else defaults["ddi_matrix"]

    pl.seed_everything(args.seed, workers=True)
    artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
    protein_embedding = load_embedding_matrix(protein_embedding_path)
    drug_embedding = load_embedding_matrix(drug_embedding_path)
    text_feature_matrix = None
    text_feature_meta: dict[str, Any] = {"mode": "off", "feature_dim": 0}
    if args.cell_type_text_mode != "off":
        text_feature_matrix, text_feature_meta = load_or_build_cell_type_text_features(
            df=artifacts.df,
            cache_path=args.cell_type_text_cache,
            model_name=args.cell_type_text_model,
            device=args.cell_type_text_device,
            batch_size=args.cell_type_text_batch_size,
            force_rebuild=args.force_cell_type_text_rebuild,
        )
        text_feature_meta = {
            **text_feature_meta,
            "mode": args.cell_type_text_mode,
            "feature_dim": int(text_feature_matrix.shape[1]),
            "cache_path": str(Path(args.cell_type_text_cache).resolve()),
        }
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
    train_loader, valid_loader, test_loader, split_summary, train_indices = build_text_data_loaders(
        args,
        artifacts,
        drug_embedding,
        ddi_matrix,
        graph_feature_matrix,
        text_feature_matrix,
    )
    active_positive_weight = ptv_train.resolve_fast_positive_weight(args, artifacts, train_indices)
    active_bce_weight = 1.0 if args.bce_weight is None else float(args.bce_weight)
    covariate_model_sizes = ptv_train.fast_covariate_model_sizes(args, artifacts)
    aux_covariate_indices = ptv_train.fast_aux_covariate_indices(args)
    aux_covariate_sizes = ptv_train.fast_aux_covariate_sizes(args, artifacts, covariate_model_sizes)
    aux_covariate_contrastive_indices = ptv_train.fast_aux_covariate_contrastive_indices(args)
    ranking_loss_group_index = ptv_train.fast_ranking_loss_group_index(args)
    mse_gene_weights, mse_gene_weight_summary = ptv_train.build_fast_mse_gene_weights(
        args=args,
        artifacts=artifacts,
        train_indices=train_indices,
        pdi_matrix_path=pdi_matrix_path,
    )
    text_feature_dim = 0 if text_feature_matrix is None else int(text_feature_matrix.shape[1])
    model = FastDeltaDrugResponseModel(
        n_genes=int(artifacts.expression_matrix.shape[1]),
        drug_embedding_dim=int(drug_embedding.shape[1]),
        protein_embedding=protein_embedding,
        ordered_protein_index=artifacts.ordered_protein_index,
        covariate_sizes=covariate_model_sizes,
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
        graph_feature_blocks=ptv_train.graph_feature_blocks_from_meta(graph_feature_meta),
        graph_jump_fusion=args.graph_jump_fusion,
        graph_jump_gate=args.graph_jump_gate,
        graph_jump_temperature=args.graph_jump_temperature,
        pair_fusion_mode=args.pair_fusion_mode,
        pair_type_features=args.pair_type_features,
        cell_pair_film_scale=args.cell_pair_film_scale,
        protein_concat_mode=args.protein_concat_mode,
        protein_concat_dim=args.protein_concat_dim,
        protein_concat_topk=args.protein_concat_topk,
        protein_concat_init_scale=args.protein_concat_init_scale,
        protein_concat_seed=args.protein_concat_seed,
        protein_concat_score_mode=args.protein_concat_score_mode,
        protein_concat_expr_scale=args.protein_concat_expr_scale,
        control_logit_scale=args.control_logit_scale,
        pair_logit_scale=args.pair_logit_scale,
        pair_logit_gate=args.pair_logit_gate,
        target_logit_scale=args.target_logit_scale,
        covariate_logit_scale=args.covariate_logit_scale,
        aux_covariate_sizes=aux_covariate_sizes,
        prior_feature_dim=text_feature_dim,
        prior_logit_scale=args.cell_type_text_logit_scale,
        prior_fixed_logit_scale=0.0,
        use_ddi=args.use_ddi,
        residual_expression=args.residual_expression,
        init_delta_scale=args.init_delta_scale,
    )
    lightning_model = FastProteinTalkLightning(
        model,
        task_head=args.task_head,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mse_weight=args.mse_weight,
        bce_weight=active_bce_weight,
        positive_weight=active_positive_weight,
        have_mse_loss=args.have_mse_loss,
        mse_inactive_label_weight=args.mse_inactive_label_weight,
        optimizer_name=args.optimizer_name,
        scheduler_name=args.scheduler_name,
        max_epochs=args.max_epochs,
        mse_gene_subsample=args.mse_gene_subsample,
        mse_gene_weights=mse_gene_weights,
        label_smoothing=args.label_smoothing,
        aux_covariate_loss_weight=args.aux_covariate_loss_weight,
        aux_covariate_indices=aux_covariate_indices,
        aux_covariate_label_smoothing=args.aux_covariate_loss_label_smoothing,
        aux_covariate_contrastive_weight=args.aux_covariate_contrastive_weight,
        aux_covariate_contrastive_indices=aux_covariate_contrastive_indices,
        aux_covariate_contrastive_temperature=args.aux_covariate_contrastive_temperature,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_loss_margin=args.ranking_loss_margin,
        ranking_loss_group_index=ranking_loss_group_index,
    )
    ptv_train.load_model_state(lightning_model, args.checkpoint_path, strict=not args.allow_partial_checkpoint_load)

    experiment_name = args.experiment_name or f"{args.task_name}_celltype_text_{args.split_strategy}"
    run_dir = Path(args.checkpoint_dir) / experiment_name
    manifest = {
        "generated_at": ptv_train.iso_now(),
        "run_status": "initialized",
        "experiment_name": experiment_name,
        "implementation": "celltype_text_fm_fast_delta",
        "dataset_group": args.dataset_group,
        "task_name": args.task_name,
        "split_strategy": args.split_strategy,
        "model_type": "celltype_text_fast_delta",
        "backbone_model_type": FAST_DELTA_MODEL_NAME,
        "task_dir": str(task_dir.resolve()),
        "meta_path": str(meta_path.resolve()),
        "protein_embedding_path": str(protein_embedding_path.resolve()),
        "drug_embedding_path": str(drug_embedding_path.resolve()),
        "ppi_matrix_path": str(ppi_matrix_path.resolve()) if args.graph_feature_mode in {"real", "zero"} else None,
        "pdi_matrix_path": str(pdi_matrix_path.resolve()) if args.graph_feature_mode in {"real", "zero"} else None,
        "ddi_matrix_path": str(ddi_matrix_path.resolve()) if (args.use_ddi or args.graph_feature_mode in {"real", "zero"}) else None,
        "cell_type_text_feature_meta": ptv_train.json_safe(text_feature_meta),
        "cell_type_text_mode": args.cell_type_text_mode,
        "cell_type_text_model": args.cell_type_text_model,
        "cell_type_text_logit_scale": args.cell_type_text_logit_scale,
        "drop_cell_covariate": args.drop_cell_covariate,
        "drop_cell_type_covariate": args.drop_cell_type_covariate,
        "batch_cov_list": list(args.batch_cov_list),
        "graph_feature_mode": args.graph_feature_mode,
        "graph_feature_dim": args.graph_feature_dim,
        "graph_structural_rp": args.graph_structural_rp,
        "graph_drug_concat": args.graph_drug_concat,
        "graph_logit_scale": args.graph_logit_scale,
        "protein_concat_mode": args.protein_concat_mode,
        "mse_weight": args.mse_weight,
        "have_mse_loss": args.have_mse_loss,
        "bce_weight": active_bce_weight,
        "positive_weight": active_positive_weight,
        "mse_gene_weight_summary": mse_gene_weight_summary,
        "split_summary": split_summary,
        "best_ckpt_metric": checkpoint_selection["best_ckpt_metric"],
        "monitor": checkpoint_selection["monitor"] or "none",
        "monitor_mode": checkpoint_selection["monitor_mode"],
        "scheduler_monitor": scheduler_monitor,
        "scheduler_monitor_mode": scheduler_monitor_mode,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "expression_latent_dim": args.expression_latent_dim,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
        "logger_backend": "wandb" if args.log_to_wandb else args.logger_backend,
        "args": ptv_train.json_safe(vars(args)),
        "model_parameter_count": int(sum(param.numel() for param in model.parameters())),
        "trainable_parameter_count": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
    }
    if args.dry_run_batches:
        lightning_model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() and str(args.accelerator).lower() != "cpu" else "cpu")
        lightning_model.to(device)
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                out = lightning_model(batch)
                print(
                    f"dry_run batch={batch_idx} expression={tuple(out[0].shape)} "
                    f"response_logits={tuple(out[1].shape)} synergy_logits={tuple(out[2].shape)} "
                    f"prior_features={tuple(batch.get('prior_features', torch.empty(0)).shape)}"
                )
                if batch_idx + 1 >= args.dry_run_batches:
                    break
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest["run_status"] = "fit_started"
    ptv_train.dump_json(run_dir / "run_manifest.json", manifest)

    monitor = checkpoint_selection["monitor"]
    monitor_mode = checkpoint_selection["monitor_mode"] or "max"
    checkpointing_enabled = args.save_top_k != 0 or args.save_last_ckpt
    callbacks: list[Any] = []
    checkpoint_callback = None
    if checkpointing_enabled:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(run_dir),
            filename=args.checkpoint_filename or "{epoch}-{step}",
            monitor=monitor if monitor and monitor.lower() not in {"none", "null", ""} else None,
            mode=monitor_mode,
            save_top_k=args.save_top_k,
            save_last=args.save_last_ckpt,
            save_on_train_epoch_end=False,
        )
        callbacks.append(checkpoint_callback)
    if monitor and not args.allow_nonfinite_monitor:
        callbacks.append(ptv_train.MonitorMetricGuard(monitor))
    logger = ptv_train.build_logger(args, experiment_name)
    if logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=ptv_train.resolve_strategy(args),
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_checkpointing=checkpointing_enabled,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        enable_progress_bar=os.environ.get("PTV_PROGRESS_BAR", "1") != "0",
    )
    trainer.fit(lightning_model, train_loader, valid_loader)
    manifest["run_status"] = "fit_completed"
    manifest["fit_completed_at"] = ptv_train.iso_now()
    if checkpoint_callback is not None:
        manifest["best_model_path"] = checkpoint_callback.best_model_path
        manifest["best_model_score"] = (
            float(checkpoint_callback.best_model_score.detach().cpu())
            if checkpoint_callback.best_model_score is not None
            else None
        )
    if not args.skip_test:
        test_ckpt_path = manifest.get("best_model_path") or None
        manifest["test_checkpoint_path"] = test_ckpt_path
        manifest["test_started_at"] = ptv_train.iso_now()
        manifest["test_status"] = "running"
        ptv_train.dump_json(run_dir / "run_manifest.json", manifest)
        manifest["test_results"] = ptv_train.json_safe(trainer.test(lightning_model, test_loader, ckpt_path=test_ckpt_path))
        manifest["test_status"] = "test_completed"
        manifest["test_completed_at"] = ptv_train.iso_now()
    else:
        manifest["test_checkpoint_path"] = None
        manifest["test_status"] = "skipped"
        manifest["test_results"] = []
    ptv_train.dump_json(run_dir / "run_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cell-type text embedding fast_delta experiment")
    parser.add_argument("--training-ready-root", default=str(ptv_train.DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", choices=["ptv3"], default="ptv3")
    parser.add_argument("--task-name", default="ptv3_main_singledrug")
    parser.add_argument("--split-strategy", default="cell_5fold_fold0")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--cell-type-text-mode", choices=["off", "sapbert"], default="sapbert")
    parser.add_argument("--cell-type-text-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cell-type-text-cache", default="celltype_text_fm/artifacts/cell_type_sapbert_features.npz")
    parser.add_argument("--cell-type-text-device", default="cpu")
    parser.add_argument("--cell-type-text-batch-size", type=int, default=16)
    parser.add_argument("--force-cell-type-text-rebuild", action="store_true")
    parser.add_argument("--cell-type-text-logit-scale", type=float, default=0.0)
    parser.add_argument("--drop-cell-covariate", action="store_true")
    parser.add_argument("--drop-cell-type-covariate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--expression-latent-dim", type=int, default=512)
    parser.add_argument("--covariate-embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--control-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=3)
    parser.add_argument("--target-layers", type=int, default=2)
    parser.add_argument("--mse-weight", type=float, default=0.075)
    parser.add_argument("--bce-weight", type=float, default=None)
    parser.add_argument("--positive-weight", default="none")
    parser.add_argument("--max-positive-weight", type=float, default=20.0)
    parser.add_argument("--no-mse-loss", action="store_false", dest="have_mse_loss")
    parser.add_argument("--mse-inactive-label-weight", type=float, default=1.0)
    parser.add_argument("--mse-gene-subsample", type=int, default=0)
    parser.add_argument("--mse-gene-weight-mode", choices=["off", "variance", "pdi", "variance_pdi"], default="off")
    parser.add_argument("--mse-gene-weight-topk", type=int, default=4096)
    parser.add_argument("--mse-gene-weight-scale", type=float, default=2.0)
    parser.add_argument("--active-label-sampling-weight", type=float, default=1.0)
    parser.add_argument("--positive-label-sampling-weight", type=float, default=1.0)
    parser.add_argument("--inactive-label-train-ratio", type=float, default=-1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer-name", default="adamw")
    parser.add_argument("--scheduler-name", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--target-protein-max-length", type=int, default=32)
    parser.add_argument("--effective-key1", default=None)
    parser.add_argument("--effective-key2", default="synergy")
    parser.add_argument("--task-head", choices=["auto", "response", "synergy"], default="response")
    parser.add_argument("--task-label-key", default=None)
    parser.add_argument("--task-mask-key", default=None)
    parser.add_argument(
        "--batch-cov-list",
        nargs="*",
        default=["machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time"],
    )
    parser.add_argument("--covariate-unk-for-unseen", action="store_true", default=True)
    parser.add_argument("--no-covariate-unk-for-unseen", action="store_false", dest="covariate_unk_for_unseen")
    parser.add_argument("--covariate-unk-fields", nargs="*", default=[])
    parser.add_argument("--covariate-unk-dropout", type=float, default=0.15)
    parser.add_argument("--protein-embedding-path", default=None)
    parser.add_argument("--drug-embedding-path", default=None)
    parser.add_argument("--ppi-matrix-path", default=None)
    parser.add_argument("--pdi-matrix-path", default=None)
    parser.add_argument("--ddi-matrix-path", default=None)
    parser.add_argument("--graph-feature-mode", choices=["real", "zero", "off"], default="real")
    parser.add_argument("--graph-feature-dim", type=int, default=128)
    parser.add_argument("--graph-feature-seed", type=int, default=17)
    parser.add_argument("--graph-structural-rp", action="store_true", default=True)
    parser.add_argument("--no-graph-structural-rp", action="store_false", dest="graph_structural_rp")
    parser.add_argument("--graph-multihop", action="store_true")
    parser.add_argument("--graph-cache-dir", default=str(REPO_ROOT / "graph_cache"))
    parser.add_argument("--force-graph-cache-rebuild", action="store_true")
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-init-scale", type=float, default=0.1)
    parser.add_argument("--graph-drug-concat", action="store_true", default=True)
    parser.add_argument("--no-graph-drug-concat", action="store_false", dest="graph_drug_concat")
    parser.add_argument("--graph-pair-add-scale", type=float, default=0.0)
    parser.add_argument("--graph-logit-scale", type=float, default=2.0)
    parser.add_argument("--graph-jump-fusion", choices=["concat", "selective"], default="concat")
    parser.add_argument("--graph-jump-gate", choices=["softmax", "sparsemax"], default="softmax")
    parser.add_argument("--graph-jump-temperature", type=float, default=1.0)
    parser.add_argument("--pair-fusion-mode", choices=["symmetric", "rich_symmetric", "ordered_concat", "dual"], default="symmetric")
    parser.add_argument("--pair-type-features", action="store_true")
    parser.add_argument("--cell-pair-film-scale", type=float, default=0.0)
    parser.add_argument("--protein-concat-mode", choices=["off", "pcep", "pcep_cell", "pcep_dual"], default="pcep")
    parser.add_argument("--protein-concat-dim", type=int, default=64)
    parser.add_argument("--protein-concat-topk", type=int, default=512)
    parser.add_argument("--protein-concat-init-scale", type=float, default=0.1)
    parser.add_argument("--protein-concat-seed", type=int, default=23)
    parser.add_argument("--protein-concat-score-mode", choices=["multiply", "additive", "magnitude"], default="multiply")
    parser.add_argument("--protein-concat-expr-scale", type=float, default=1.0)
    parser.add_argument("--control-logit-scale", type=float, default=0.0)
    parser.add_argument("--pair-logit-scale", type=float, default=0.0)
    parser.add_argument("--pair-logit-gate", action="store_true")
    parser.add_argument("--target-logit-scale", type=float, default=0.0)
    parser.add_argument("--covariate-logit-scale", type=float, default=0.0)
    parser.add_argument("--aux-covariate-loss-fields", nargs="*", default=[])
    parser.add_argument("--aux-covariate-loss-weight", type=float, default=0.0)
    parser.add_argument("--aux-covariate-loss-label-smoothing", type=float, default=0.0)
    parser.add_argument("--aux-covariate-contrastive-fields", nargs="*", default=[])
    parser.add_argument("--aux-covariate-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--aux-covariate-contrastive-temperature", type=float, default=0.2)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--ranking-loss-margin", type=float, default=0.0)
    parser.add_argument("--ranking-loss-group-field", default="Cell")
    parser.add_argument("--use-ddi", action="store_true")
    parser.add_argument("--absolute-expression-head", action="store_false", dest="residual_expression")
    parser.add_argument("--init-delta-scale", type=float, default=0.1)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epoch-len", type=int, default=None)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--save-top-k", type=int, default=1)
    parser.add_argument("--save-last-ckpt", action="store_true", default=True)
    parser.add_argument("--no-save-last-ckpt", action="store_false", dest="save_last_ckpt")
    parser.add_argument("--checkpoint-filename", default=None)
    parser.add_argument("--best-ckpt-metric", choices=sorted(ptv_train.BEST_CKPT_METRIC_ALIASES), default="valid_auprc")
    parser.add_argument("--monitor", default=None)
    parser.add_argument("--monitor-mode", choices=["min", "max"], default=None)
    parser.add_argument("--logger-backend", choices=["tensorboard", "wandb", "both", "none"], default="none")
    parser.add_argument("--log-to-wandb", action="store_true", dest="log_to_wandb")
    parser.add_argument("--wandb-project", default="aivc_proteintalk")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb-log-model", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--allow-nonfinite-monitor", action="store_true")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
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
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--allow-partial-checkpoint-load", action="store_true")
    parser.set_defaults(model_type=FAST_DELTA_MODEL_NAME)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

