#!/usr/bin/env python3
"""Run inference from the new `data/training_ready` format."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.training_ready_dataset import (
    ProteinTalkDataset,
    TrainingReadyArtifacts,
    build_pairing_from_table,
    default_anchor_indices,
    load_embedding_matrix,
    load_indices,
    load_json,
    load_row_to_set,
    load_set_info,
)
from model.training_ready_lightning import ProteinTalkLightning, binary_metrics, compute_validation_metrics
from model.training_ready_models import GRAPH_MODEL_NAMES, ModelArtifacts, SELECTED_MODEL_NAMES, build_model
from train import category_sizes, default_derived_paths, infer_label_key, load_pdi_matrix, resolve_task_loss_config


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    return obj


def write_dataframe(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def load_checkpoint(lightning_model: ProteinTalkLightning, checkpoint_path: str, *, strict: bool = True) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = lightning_model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)} examples={missing[:5]}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)} examples={unexpected[:5]}")


def infer_checkpoint_manifest_path(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).resolve().parent / "run_manifest.json"


def configs_match(expected: object, actual: object) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return bool(np.isclose(float(expected), float(actual), rtol=1e-6, atol=1e-8))
        except (TypeError, ValueError):
            return expected == actual
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        return list(expected) == list(actual)
    return expected == actual


def current_model_config(args) -> dict[str, object]:
    # Architecture settings plus the active prediction head belong here.
    # Evaluation label keys and masks may legitimately differ for external
    # validation, e.g. train on PRISM1st but test extra single-drug PRISM2nd.
    # The head itself must match; using a response-trained checkpoint for
    # synergy inference would silently evaluate an untrained head.
    return {
        "dataset_group": args.dataset_group,
        "model_type": args.model_type,
        "task_head": args.task_head,
        "meta_path": args.meta_path,
        "protein_embedding_path": args.protein_embedding_path_resolved,
        "drug_embedding_path": args.drug_embedding_path_resolved,
        "pdi_matrix_path": args.pdi_matrix_path_resolved if args.model_type in GRAPH_MODEL_NAMES else None,
        "pdi_mode": args.pdi_mode if args.model_type in GRAPH_MODEL_NAMES else None,
        "pdi_input_orientation": args.pdi_input_orientation,
        "fusion_mode": args.fusion_mode,
        "perturb_fusion_mode": args.perturb_fusion_mode,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "cls_type": args.cls_type,
        "graph_dropout": args.graph_dropout,
        "use_target": args.use_target,
        "target_protein_fusion_model": args.target_protein_fusion_model,
        "gate_weight": args.gate_weight,
        "batch_cov_list": args.batch_cov_list,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "target_protein_max_length": args.target_protein_max_length,
        "gene_emb_dim": args.gene_emb_dim,
        "emb_dataset_path": args.emb_dataset_path if args.model_type == "baseline_emb_v3" else None,
    }


def validate_checkpoint_config(
    args,
    *,
    allow_mismatch: bool = False,
    allow_missing_manifest: bool = False,
    allow_incomplete_manifest: bool = False,
) -> dict[str, object]:
    manifest_path = infer_checkpoint_manifest_path(args.checkpoint_path)
    if not manifest_path.exists():
        message = (
            f"missing checkpoint run_manifest.json next to checkpoint: {manifest_path}. "
            "External validation requires manifest-backed checkpoint config validation"
        )
        if allow_missing_manifest:
            print(f"[checkpoint] WARNING {message}")
            return {}
        raise FileNotFoundError(
            message + "\nUse --allow-missing-checkpoint-manifest only for intentional migration/debug runs."
        )
    manifest = load_json(manifest_path)
    run_status = manifest.get("run_status")
    if run_status != "fit_completed":
        message = (
            "checkpoint run_manifest.json is not marked fit_completed "
            f"(run_status={run_status!r}); refusing to infer from a possibly incomplete or stale training run"
        )
        if allow_incomplete_manifest:
            print(f"[checkpoint] WARNING {message}")
        else:
            raise ValueError(
                message + "\nUse --allow-incomplete-checkpoint-manifest only for intentional migration/debug runs."
            )
    current = current_model_config(args)
    mismatches: list[str] = []
    for key, expected in current.items():
        if key not in manifest:
            mismatches.append(f"{key}: missing from checkpoint manifest")
            continue
        actual = manifest[key]
        if not configs_match(expected, actual):
            mismatches.append(f"{key}: current={expected!r} checkpoint={actual!r}")
    checkpoint_axis_path = manifest.get("ordered_protein_index_path")
    current_axis_path = getattr(args, "ordered_protein_index_path", None)
    if args.model_type not in GRAPH_MODEL_NAMES:
        if not checkpoint_axis_path:
            mismatches.append("ordered_protein_index_path: missing from checkpoint manifest")
        elif current_axis_path:
            checkpoint_axis = load_json(checkpoint_axis_path)
            current_axis = load_json(current_axis_path)
            if list(checkpoint_axis) != list(current_axis):
                mismatches.append(
                    "ordered_protein_index: current task protein axis differs from checkpoint protein axis "
                    f"(current_path={current_axis_path!r}, checkpoint_path={checkpoint_axis_path!r})"
                )
    if not mismatches:
        return manifest
    message = "checkpoint config mismatch:\n  " + "\n  ".join(mismatches)
    if allow_mismatch:
        print(f"[checkpoint] WARNING {message}")
        return manifest
    raise ValueError(message + "\nUse --allow-checkpoint-config-mismatch only for intentional migration/debug runs.")


def resolve_inference_indices(args, artifacts: TrainingReadyArtifacts):
    if args.split_strategy:
        split_dir = Path(args.split_dir) if args.split_dir else Path(args.training_ready_root) / args.dataset_group / "splits" / args.task_name
        indices = load_indices(split_dir, args.split_name, args.split_strategy)
        row_to_set = load_row_to_set(split_dir)
        set_info = load_set_info(split_dir, args.split_name, args.split_strategy)
        return indices, row_to_set, set_info, str(split_dir)
    indices = default_anchor_indices(artifacts.df)
    row_to_set, set_info = build_pairing_from_table(artifacts.df, indices)
    return indices, row_to_set, set_info, None


def main() -> None:
    started_at = iso_now()
    parser = argparse.ArgumentParser(description="Inference for ProteinTalk training-ready tasks")
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", choices=["ptv1", "ptv3"], default="ptv3")
    parser.add_argument("--task-name", default="ptv3_extra_doubledrug_guomics")
    parser.add_argument("--split-strategy", default="test_only", help="Use an existing split strategy; set empty string to infer all anchors")
    parser.add_argument("--split-name", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-type", choices=sorted(SELECTED_MODEL_NAMES), default="attention_v10_hetero_cls_ee")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
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
        help="Use the real PDI matrix or an all-zero matrix for no-PDI ablation checkpoints.",
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-batches", type=int, default=None, help="Optional smoke-test cap on inference batches")
    parser.add_argument("--save-expression-pred", action="store_true")
    parser.add_argument(
        "--allow-partial-checkpoint-load",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys",
    )
    parser.add_argument(
        "--allow-checkpoint-config-mismatch",
        action="store_true",
        help="Allow mismatch between checkpoint run_manifest.json and inference model args",
    )
    parser.add_argument(
        "--allow-missing-checkpoint-manifest",
        action="store_true",
        help="Allow inference from a checkpoint directory without run_manifest.json for migration/debug runs",
    )
    parser.add_argument(
        "--allow-incomplete-checkpoint-manifest",
        action="store_true",
        help="Allow inference when checkpoint run_manifest.json is not marked fit_completed for migration/debug runs",
    )
    args = parser.parse_args()
    if args.split_strategy == "":
        args.split_strategy = None

    training_ready_root = Path(args.training_ready_root)
    task_dir = training_ready_root / args.dataset_group / "tasks" / args.task_name
    meta_path = training_ready_root / args.dataset_group / "global_meta.json"
    defaults = default_derived_paths(training_ready_root, args.dataset_group)
    protein_embedding_path = Path(args.protein_embedding_path) if args.protein_embedding_path else defaults["protein_embedding"]
    drug_embedding_path = Path(args.drug_embedding_path) if args.drug_embedding_path else defaults["drug_embedding"]
    pdi_matrix_path = Path(args.pdi_matrix_path) if args.pdi_matrix_path else defaults["pdi_matrix"]
    args.meta_path = str(meta_path.resolve())
    args.protein_embedding_path_resolved = str(protein_embedding_path.resolve())
    args.drug_embedding_path_resolved = str(drug_embedding_path.resolve())
    args.pdi_matrix_path_resolved = str(pdi_matrix_path.resolve())
    args.ordered_protein_index_path = str((task_dir / "feature_ordered_protein_index.json").resolve())
    args.effective_key1 = args.effective_key1 or infer_label_key(args.task_name)
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

    artifacts = TrainingReadyArtifacts(task_dir, meta_path)
    protein_embedding = load_embedding_matrix(protein_embedding_path)
    drug_embedding = load_embedding_matrix(drug_embedding_path)
    pdi_matrix = load_pdi_matrix(pdi_matrix_path, model_type=args.model_type, pdi_mode=args.pdi_mode)
    drug_mode = "index" if args.model_type in GRAPH_MODEL_NAMES else "embedding"
    indices, row_to_set, set_info, split_dir = resolve_inference_indices(args, artifacts)
    dataset = ProteinTalkDataset(
        artifacts,
        indices,
        row_to_set,
        set_info,
        mode="eval",
        batch_cov_list=args.batch_cov_list,
        drug_mode=drug_mode,
        drug_embedding_matrix=drug_embedding if drug_mode == "embedding" else None,
        target_protein_max_length=args.target_protein_max_length,
        effective_key1=args.effective_key1,
        effective_key2=args.effective_key2,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
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
        effective_key1=args.effective_key1,
        effective_key2=args.effective_key2,
        task_label_key=task_loss_config["task_label_key"],
        task_mask_key=task_loss_config["task_mask_key"],
        task_head=task_loss_config["task_head"],
    )
    checkpoint_manifest = validate_checkpoint_config(
        args,
        allow_mismatch=args.allow_checkpoint_config_mismatch,
        allow_missing_manifest=args.allow_missing_checkpoint_manifest,
        allow_incomplete_manifest=args.allow_incomplete_checkpoint_manifest,
    )
    load_checkpoint(lightning_model, args.checkpoint_path, strict=not args.allow_partial_checkpoint_load)
    device = torch.device(args.device)
    lightning_model.to(device).eval()
    checkpoint_axis_path = checkpoint_manifest.get("ordered_protein_index_path")
    checkpoint_axis_size = checkpoint_manifest.get("topk_genes")
    if checkpoint_axis_size is None and checkpoint_axis_path:
        checkpoint_axis_size = len(load_json(checkpoint_axis_path))

    prob1: list[np.ndarray] = []
    prob2: list[np.ndarray] = []
    true1_chunks: list[np.ndarray] = []
    true2_chunks: list[np.ndarray] = []
    mask1_chunks: list[np.ndarray] = []
    mask2_chunks: list[np.ndarray] = []
    row_index_chunks: list[np.ndarray] = []
    expression_chunks: list[np.ndarray] = []
    expression_true_chunks: list[np.ndarray] = []
    control_expression_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.limit_batches is not None and batch_idx >= args.limit_batches:
                break
            batch = move_to_device(batch, device)
            expression, logits1, logits2 = lightning_model(batch)
            prob1.append(torch.sigmoid(logits1.squeeze(-1)).detach().cpu().numpy())
            prob2.append(torch.sigmoid(logits2.squeeze(-1)).detach().cpu().numpy())
            true1_chunks.append(batch["perturb"][args.effective_key1].detach().cpu().numpy())
            true2_chunks.append(batch["perturb"][args.effective_key2].detach().cpu().numpy())
            mask1 = batch["perturb"].get("sensitive_label_mask")
            mask2 = batch["perturb"].get("synergy_label_mask")
            if mask1 is None:
                mask1 = torch.zeros_like(batch["perturb"][args.effective_key1])
            if mask2 is None:
                mask2 = torch.zeros_like(batch["perturb"][args.effective_key2])
            mask1_chunks.append(mask1.detach().cpu().numpy())
            mask2_chunks.append(mask2.detach().cpu().numpy())
            row_index_chunks.append(batch["perturb"]["row_index"].detach().cpu().numpy().astype(np.int64))
            if args.save_expression_pred:
                expression_chunks.append(expression.detach().cpu().numpy().astype(np.float32))
                expression_true_chunks.append(batch["perturb"]["expressions_hvg"].detach().cpu().numpy())
                control_expression_chunks.append(batch["control"]["expressions_hvg"].detach().cpu().numpy())

    pred_prob1 = np.concatenate(prob1) if prob1 else np.asarray([], dtype=np.float32)
    pred_prob2 = np.concatenate(prob2) if prob2 else np.asarray([], dtype=np.float32)
    true1 = np.concatenate(true1_chunks) if true1_chunks else np.asarray([], dtype=np.float32)
    true2 = np.concatenate(true2_chunks) if true2_chunks else np.asarray([], dtype=np.float32)
    mask1 = np.concatenate(mask1_chunks) if mask1_chunks else np.asarray([], dtype=np.float32)
    mask2 = np.concatenate(mask2_chunks) if mask2_chunks else np.asarray([], dtype=np.float32)
    active_prob = pred_prob2 if task_loss_config["task_head"] == "synergy" else pred_prob1
    active_true = true2 if task_loss_config["task_head"] == "synergy" else true1
    active_mask = mask2 if task_loss_config["task_head"] == "synergy" else mask1
    feature_row_indices = (
        np.concatenate(row_index_chunks).astype(np.int64, copy=False)
        if row_index_chunks
        else np.asarray([], dtype=np.int64)
    )
    rows = artifacts.df.iloc[feature_row_indices].reset_index(drop=False).rename(
        columns={"index": "feature_row_index_from_table"}
    )
    prediction_df = pd.DataFrame(
        {
            "feature_row_index": feature_row_indices,
            "sample_id": rows["sample_id"].astype(str),
            "control": rows["control"].astype(str),
            "Cell": rows.get("Cell", pd.Series([""] * len(rows))).astype(str),
            "cell_type": rows.get("cell_type", pd.Series([""] * len(rows))).astype(str),
            "pert_id1": rows.get("pert_id1", pd.Series([""] * len(rows))).astype(str),
            "pert_id2": rows.get("pert_id2", pd.Series([""] * len(rows))).astype(str),
            "pert_index1": pd.to_numeric(rows.get("pert_index1", pd.Series([np.nan] * len(rows))), errors="coerce"),
            "pert_index2": pd.to_numeric(rows.get("pert_index2", pd.Series([np.nan] * len(rows))), errors="coerce"),
            "pred_task_prob": active_prob,
            "pred_response_prob": pred_prob1,
            "pred_synergy_prob": pred_prob2,
            "task_label": rows.get(task_loss_config["task_label_key"], pd.Series([None] * len(rows))).tolist(),
            "response_label": rows.get(args.effective_key1, pd.Series([None] * len(rows))).tolist(),
            "synergy_label": rows.get(args.effective_key2, pd.Series([None] * len(rows))).tolist(),
        }
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "inference" / args.task_name / args.model_type
    prediction_path = write_dataframe(output_dir / "predictions.parquet", prediction_df)
    response_metrics = binary_metrics(
        pred_prob1,
        true1,
        mask1,
    )
    synergy_metrics = binary_metrics(
        pred_prob2,
        true2,
        mask2,
    )
    task_metrics = binary_metrics(active_prob, active_true, active_mask)
    task_metrics.update(
        {
            "task_head": task_loss_config["task_head"],
            "task_label_key": task_loss_config["task_label_key"],
            "task_mask_key": task_loss_config["task_mask_key"],
        }
    )
    metrics = {
        "task": task_metrics,
        "response": response_metrics,
        "synergy": synergy_metrics,
        "task1": response_metrics,
        "task2": synergy_metrics,
    }
    expression_pred = None
    if args.save_expression_pred:
        expression_pred = np.concatenate(expression_chunks, axis=0) if expression_chunks else np.asarray([], dtype=np.float32)
        expression_true = (
            np.concatenate(expression_true_chunks, axis=0)
            if expression_true_chunks
            else np.asarray([], dtype=np.float32)
        )
        control_expression = (
            np.concatenate(control_expression_chunks, axis=0)
            if control_expression_chunks
            else np.asarray([], dtype=np.float32)
        )
        if expression_pred.size and expression_true.size:
            metrics["legacy_validation_metrics"] = compute_validation_metrics(
                predictions=expression_pred,
                targets=expression_true,
                ny_pred1=pred_prob1,
                ny_true1=true1,
                mask1=mask1,
                ny_pred2=pred_prob2,
                ny_true2=true2,
                mask2=mask2,
                control_expression=control_expression,
            )
        np.save(output_dir / "expression_pred.npy", expression_pred)
    dump_json(output_dir / "metrics.json", metrics)
    completed_at = iso_now()
    dump_json(
        output_dir / "run_manifest.json",
        {
            "generated_at": completed_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "dataset_group": args.dataset_group,
            "task_name": args.task_name,
            "model_type": args.model_type,
            "checkpoint_path": args.checkpoint_path,
            "task_dir": str(task_dir),
            "meta_path": str(meta_path),
            "ordered_protein_index_path": args.ordered_protein_index_path,
            "protein_axis_size": int(artifacts.expression_matrix.shape[1]),
            "split_strategy": args.split_strategy,
            "split_name": args.split_name,
            "split_dir": split_dir,
            "checkpoint_run_manifest_path": str(infer_checkpoint_manifest_path(args.checkpoint_path)),
            "checkpoint_task_name": checkpoint_manifest.get("task_name"),
            "checkpoint_split_strategy": checkpoint_manifest.get("split_strategy"),
            "checkpoint_ordered_protein_index_path": checkpoint_axis_path,
            "checkpoint_protein_axis_size": checkpoint_axis_size,
            "protein_axis_matches_checkpoint": bool(checkpoint_axis_size == int(artifacts.expression_matrix.shape[1])),
            "task_head": task_loss_config["task_head"],
            "task_label_key": task_loss_config["task_label_key"],
            "task_mask_key": task_loss_config["task_mask_key"],
            "prediction_path": str(prediction_path),
            "n_predictions": int(len(prediction_df)),
            "save_expression_pred": bool(args.save_expression_pred),
            "allow_partial_checkpoint_load": bool(args.allow_partial_checkpoint_load),
            "allow_checkpoint_config_mismatch": bool(args.allow_checkpoint_config_mismatch),
            "allow_missing_checkpoint_manifest": bool(args.allow_missing_checkpoint_manifest),
            "allow_incomplete_checkpoint_manifest": bool(args.allow_incomplete_checkpoint_manifest),
            "limit_batches": args.limit_batches,
            "pdi_input_orientation": args.pdi_input_orientation,
            "pdi_mode": args.pdi_mode if args.model_type in GRAPH_MODEL_NAMES else None,
            "fusion_mode": args.fusion_mode,
            "perturb_fusion_mode": args.perturb_fusion_mode,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "cls_type": args.cls_type,
            "graph_dropout": args.graph_dropout,
            "use_target": args.use_target,
            "target_protein_fusion_model": args.target_protein_fusion_model,
            "gate_weight": args.gate_weight,
            "emb_dataset_path": args.emb_dataset_path if args.model_type == "baseline_emb_v3" else None,
        },
    )
    print(f"[infer] wrote {prediction_path} ({len(prediction_df)} rows)")


if __name__ == "__main__":
    main()
