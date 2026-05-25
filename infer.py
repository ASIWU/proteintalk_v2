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
from dataset.training_ready_fast_dataset import (
    BATCH_COVARIATE_COLUMNS as FAST_BATCH_COVARIATE_COLUMNS,
    FastProteinTalkDataset,
    FastTrainingReadyArtifacts,
    category_sizes as fast_category_sizes,
    load_embedding_matrix as load_fast_embedding_matrix,
    load_indices as load_fast_indices,
    load_row_to_set as load_fast_row_to_set,
    load_set_info as load_fast_set_info,
)
from model.fast_delta_model import FastDeltaDrugResponseModel
from model.fast_lightning import FastProteinTalkLightning, binary_metrics as fast_binary_metrics
from model.graph_feature_utils import build_or_load_graph_features
from model.training_ready_lightning import ProteinTalkLightning, binary_metrics, compute_validation_metrics
from model.training_ready_models import FAST_DELTA_MODEL_NAME, GRAPH_MODEL_NAMES, ModelArtifacts, SELECTED_MODEL_NAMES, build_model
from train import (
    build_fast_target_expression_weights,
    category_sizes,
    default_derived_paths,
    graph_feature_blocks_from_meta,
    infer_label_key,
    load_pdi_matrix,
    resolve_task_loss_config,
)


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


def resolve_fast_inference_indices(args, artifacts: FastTrainingReadyArtifacts):
    if args.split_strategy:
        split_dir = Path(args.split_dir) if args.split_dir else Path(args.training_ready_root) / args.dataset_group / "splits" / args.task_name
        indices = load_fast_indices(split_dir, args.split_name, args.split_strategy)
        row_to_set = load_fast_row_to_set(split_dir)
        set_info = load_fast_set_info(split_dir, args.split_name, args.split_strategy)
        return indices, row_to_set, set_info, str(split_dir)
    indices = default_anchor_indices(artifacts.df)
    row_to_set, set_info = build_pairing_from_table(artifacts.df, indices)
    return indices, row_to_set, set_info, None


LEGACY_FAST_MANIFEST_DEFAULTS = {
    "control_layers": 2,
    "fusion_layers": 3,
    "target_layers": 2,
    "graph_layers": 2,
    "graph_init_scale": 0.1,
    "graph_pair_add_scale": 0.0,
    "graph_jump_fusion": "concat",
    "graph_jump_gate": "softmax",
    "graph_jump_temperature": 1.0,
    "pair_fusion_mode": "symmetric",
    "pair_type_features": False,
    "cell_pair_film_scale": 0.0,
    "target_expression_mode": "off",
    "target_expression_dim": 64,
    "target_expression_topk": 256,
    "target_expression_ppi_topk": 32,
    "target_expression_ppi_alpha": 0.5,
    "target_expression_init_scale": 0.1,
    "target_expression_seed": 29,
    "target_expression_fusion_mode": "piece",
    "protein_concat_init_scale": 0.1,
    "protein_concat_seed": 23,
    "protein_concat_score_mode": "multiply",
    "protein_concat_expr_scale": 1.0,
    "control_logit_scale": 0.0,
    "pair_logit_scale": 0.0,
    "target_logit_scale": 0.0,
    "covariate_logit_scale": 0.0,
    "use_ddi": False,
    "residual_expression": True,
    "init_delta_scale": 0.1,
}


def current_fast_model_config(args) -> dict[str, object]:
    return {
        "dataset_group": args.dataset_group,
        "model_type": args.model_type,
        "task_head": args.task_head,
        "meta_path": args.meta_path,
        "protein_embedding_path": args.protein_embedding_path_resolved,
        "drug_embedding_path": args.drug_embedding_path_resolved,
        "ppi_matrix_path": args.ppi_matrix_path_resolved if args.graph_feature_mode in {"real", "zero"} else None,
        "pdi_matrix_path": args.pdi_matrix_path_resolved if args.graph_feature_mode in {"real", "zero"} else None,
        "ddi_matrix_path": args.ddi_matrix_path_resolved if args.graph_feature_mode in {"real", "zero"} else None,
        "graph_feature_mode": args.graph_feature_mode,
        "graph_feature_dim": args.graph_feature_dim,
        "graph_feature_seed": args.graph_feature_seed,
        "graph_structural_rp": args.graph_structural_rp,
        "graph_multihop": args.graph_multihop,
        "graph_drug_concat": args.graph_drug_concat,
        "graph_layers": args.graph_layers,
        "graph_init_scale": args.graph_init_scale,
        "graph_pair_add_scale": args.graph_pair_add_scale,
        "graph_logit_scale": args.graph_logit_scale,
        "graph_jump_fusion": args.graph_jump_fusion,
        "graph_jump_gate": args.graph_jump_gate,
        "graph_jump_temperature": args.graph_jump_temperature,
        "pair_fusion_mode": args.pair_fusion_mode,
        "pair_type_features": args.pair_type_features,
        "cell_pair_film_scale": args.cell_pair_film_scale,
        "target_expression_mode": args.target_expression_mode,
        "target_expression_dim": args.target_expression_dim,
        "target_expression_topk": args.target_expression_topk,
        "target_expression_ppi_topk": args.target_expression_ppi_topk,
        "target_expression_ppi_alpha": args.target_expression_ppi_alpha,
        "target_expression_init_scale": args.target_expression_init_scale,
        "target_expression_seed": args.target_expression_seed,
        "target_expression_fusion_mode": args.target_expression_fusion_mode,
        "protein_concat_mode": args.protein_concat_mode,
        "protein_concat_dim": args.protein_concat_dim,
        "protein_concat_topk": args.protein_concat_topk,
        "protein_concat_init_scale": args.protein_concat_init_scale,
        "protein_concat_seed": args.protein_concat_seed,
        "protein_concat_score_mode": args.protein_concat_score_mode,
        "protein_concat_expr_scale": args.protein_concat_expr_scale,
        "control_logit_scale": args.control_logit_scale,
        "pair_logit_scale": args.pair_logit_scale,
        "target_logit_scale": args.target_logit_scale,
        "covariate_logit_scale": args.covariate_logit_scale,
        "batch_cov_list": args.batch_cov_list,
        "hidden_dim": args.hidden_dim,
        "expression_latent_dim": args.expression_latent_dim,
        "covariate_embedding_dim": args.covariate_embedding_dim,
        "dropout": args.dropout,
        "control_layers": args.control_layers,
        "fusion_layers": args.fusion_layers,
        "target_layers": args.target_layers,
        "target_protein_max_length": args.target_protein_max_length,
        "use_ddi": args.use_ddi,
        "residual_expression": args.residual_expression,
        "init_delta_scale": args.init_delta_scale,
    }


def validate_fast_checkpoint_config(
    args,
    *,
    allow_mismatch: bool = False,
    allow_missing_manifest: bool = False,
    allow_incomplete_manifest: bool = False,
) -> dict[str, object]:
    manifest_path = infer_checkpoint_manifest_path(args.checkpoint_path)
    if not manifest_path.exists():
        if allow_missing_manifest:
            print(f"[checkpoint] WARNING missing checkpoint run_manifest.json: {manifest_path}")
            return {}
        raise FileNotFoundError(f"missing checkpoint run_manifest.json next to checkpoint: {manifest_path}")
    manifest = load_json(manifest_path)
    run_status = manifest.get("run_status")
    if run_status != "fit_completed":
        message = f"checkpoint run_manifest.json is not fit_completed: run_status={run_status!r}"
        if allow_incomplete_manifest:
            print(f"[checkpoint] WARNING {message}")
        else:
            raise ValueError(message)
    current = current_fast_model_config(args)
    mismatches: list[str] = []
    for key, expected in current.items():
        if key not in manifest:
            if key in LEGACY_FAST_MANIFEST_DEFAULTS and configs_match(expected, LEGACY_FAST_MANIFEST_DEFAULTS[key]):
                continue
            mismatches.append(f"{key}: missing from checkpoint manifest")
            continue
        actual = manifest[key]
        if not configs_match(expected, actual):
            mismatches.append(f"{key}: current={expected!r} checkpoint={actual!r}")
    if not mismatches:
        return manifest
    message = "checkpoint config mismatch:\n  " + "\n  ".join(mismatches)
    if allow_mismatch:
        print(f"[checkpoint] WARNING {message}")
        return manifest
    raise ValueError(message + "\nUse --allow-checkpoint-config-mismatch only for intentional migration/debug runs.")


def expression_alignment_index(source_axis: list[int], target_axis: list[int]) -> np.ndarray | None:
    if list(source_axis) == list(target_axis):
        return None
    source_lookup = {int(protein_idx): col_idx for col_idx, protein_idx in enumerate(source_axis)}
    return np.asarray([source_lookup.get(int(protein_idx), -1) for protein_idx in target_axis], dtype=np.int64)


def fast_checkpoint_covariate_unk_fields(checkpoint_manifest: dict[str, object], batch_cov_list: list[str]) -> set[str]:
    fields = checkpoint_manifest.get("covariate_unk_fields") or []
    if not isinstance(fields, list):
        return set()
    allowed = set(batch_cov_list)
    return {str(field) for field in fields if str(field) in allowed}


def fast_checkpoint_covariate_unknown_indices(
    checkpoint_manifest: dict[str, object],
    meta: dict[str, object],
    batch_cov_list: list[str],
) -> dict[str, int]:
    fields = fast_checkpoint_covariate_unk_fields(checkpoint_manifest, batch_cov_list)
    if not fields:
        return {}
    base_sizes = fast_category_sizes(meta, batch_cov_list)
    return {
        field: int(base_sizes[index])
        for index, field in enumerate(batch_cov_list)
        if field in fields
    }


def fast_checkpoint_covariate_model_sizes(
    checkpoint_manifest: dict[str, object],
    meta: dict[str, object],
    batch_cov_list: list[str],
) -> list[int]:
    fields = fast_checkpoint_covariate_unk_fields(checkpoint_manifest, batch_cov_list)
    base_sizes = fast_category_sizes(meta, batch_cov_list)
    return [
        int(size) + (1 if field in fields else 0)
        for field, size in zip(batch_cov_list, base_sizes, strict=True)
    ]


def fast_checkpoint_covariate_known_values(
    checkpoint_manifest: dict[str, object],
    batch_cov_list: list[str],
) -> dict[str, set[int]]:
    fields = fast_checkpoint_covariate_unk_fields(checkpoint_manifest, batch_cov_list)
    if not fields:
        return {}
    task_dir = checkpoint_manifest.get("task_dir")
    meta_path = checkpoint_manifest.get("meta_path")
    split_strategy = checkpoint_manifest.get("split_strategy")
    split_summary = checkpoint_manifest.get("split_summary") or {}
    split_dir = split_summary.get("split_dir") if isinstance(split_summary, dict) else None
    if not task_dir or not meta_path or not split_strategy or not split_dir:
        raise ValueError("covariate UNK inference requires checkpoint task_dir/meta_path/split_strategy/split_summary")
    checkpoint_artifacts = FastTrainingReadyArtifacts.load(Path(str(task_dir)), Path(str(meta_path)))
    train_indices = load_fast_indices(Path(str(split_dir)), "train", str(split_strategy))
    known: dict[str, set[int]] = {}
    for field in batch_cov_list:
        if field not in fields:
            continue
        source_col = FAST_BATCH_COVARIATE_COLUMNS.get(field, f"{field}_index")
        if source_col not in checkpoint_artifacts.df.columns:
            raise KeyError(f"batch covariate {field!r} requires missing column {source_col!r}")
        parsed = (
            pd.to_numeric(checkpoint_artifacts.df.iloc[train_indices][source_col], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
        known[field] = {int(value) for value in parsed}
    return known


def run_fast_inference(args) -> None:
    started_at = iso_now()
    training_ready_root = Path(args.training_ready_root)
    task_dir = training_ready_root / args.dataset_group / "tasks" / args.task_name
    meta_path = training_ready_root / args.dataset_group / "global_meta.json"
    defaults = default_derived_paths(training_ready_root, args.dataset_group)
    protein_embedding_path = Path(args.protein_embedding_path) if args.protein_embedding_path else defaults["protein_embedding"]
    drug_embedding_path = Path(args.drug_embedding_path) if args.drug_embedding_path else defaults["drug_embedding"]
    ppi_matrix_path = Path(args.ppi_matrix_path) if args.ppi_matrix_path else defaults["ppi_matrix"]
    pdi_matrix_path = Path(args.pdi_matrix_path) if args.pdi_matrix_path else defaults["pdi_matrix"]
    ddi_matrix_path = Path(args.ddi_matrix_path) if args.ddi_matrix_path else defaults["ddi_matrix"]
    args.meta_path = str(meta_path.resolve())
    args.protein_embedding_path_resolved = str(protein_embedding_path.resolve())
    args.drug_embedding_path_resolved = str(drug_embedding_path.resolve())
    args.ppi_matrix_path_resolved = str(ppi_matrix_path.resolve())
    args.pdi_matrix_path_resolved = str(pdi_matrix_path.resolve())
    args.ddi_matrix_path_resolved = str(ddi_matrix_path.resolve())
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

    artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
    protein_embedding = load_fast_embedding_matrix(protein_embedding_path)
    drug_embedding = load_fast_embedding_matrix(drug_embedding_path)
    checkpoint_manifest = validate_fast_checkpoint_config(
        args,
        allow_mismatch=args.allow_checkpoint_config_mismatch,
        allow_missing_manifest=args.allow_missing_checkpoint_manifest,
        allow_incomplete_manifest=args.allow_incomplete_checkpoint_manifest,
    )
    checkpoint_axis_path = checkpoint_manifest.get("ordered_protein_index_path")
    checkpoint_axis = (
        [int(item) for item in load_json(checkpoint_axis_path)]
        if checkpoint_axis_path
        else list(artifacts.ordered_protein_index)
    )
    expression_column_index = expression_alignment_index(artifacts.ordered_protein_index, checkpoint_axis)
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
    target_expression_weight_matrix, target_expression_summary = build_fast_target_expression_weights(
        args=args,
        artifacts=artifacts,
        pdi_matrix_path=pdi_matrix_path,
        ppi_matrix_path=ppi_matrix_path,
        ordered_protein_index=checkpoint_axis,
        cache_task_name=str(checkpoint_manifest.get("task_name") or args.task_name),
    )
    ddi_matrix = np.load(ddi_matrix_path, mmap_mode="r") if args.use_ddi else None
    indices, row_to_set, set_info, split_dir = resolve_fast_inference_indices(args, artifacts)
    covariate_unknown_indices = fast_checkpoint_covariate_unknown_indices(
        checkpoint_manifest,
        artifacts.meta,
        args.batch_cov_list,
    )
    covariate_known_values = fast_checkpoint_covariate_known_values(checkpoint_manifest, args.batch_cov_list)
    dataset = FastProteinTalkDataset(
        artifacts=artifacts,
        indices=indices,
        row_to_set_index=row_to_set,
        set_info=set_info,
        mode="eval",
        drug_embedding_matrix=drug_embedding,
        batch_cov_list=args.batch_cov_list,
        target_protein_max_length=args.target_protein_max_length,
        effective_key1=args.effective_key1,
        effective_key2=args.effective_key2,
        ddi_matrix=ddi_matrix,
        graph_feature_matrix=graph_feature_matrix,
        graph_feature_enabled=args.graph_feature_mode == "real",
        expression_column_index=expression_column_index,
        covariate_known_values=covariate_known_values,
        covariate_unknown_indices=covariate_unknown_indices,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = FastDeltaDrugResponseModel(
        n_genes=len(checkpoint_axis),
        drug_embedding_dim=int(drug_embedding.shape[1]),
        protein_embedding=protein_embedding,
        ordered_protein_index=checkpoint_axis,
        covariate_sizes=fast_checkpoint_covariate_model_sizes(checkpoint_manifest, artifacts.meta, args.batch_cov_list),
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
        pair_fusion_mode=args.pair_fusion_mode,
        pair_type_features=args.pair_type_features,
        cell_pair_film_scale=args.cell_pair_film_scale,
        target_expression_mode=args.target_expression_mode,
        target_expression_weight_matrix=target_expression_weight_matrix,
        target_expression_dim=args.target_expression_dim,
        target_expression_init_scale=args.target_expression_init_scale,
        target_expression_seed=args.target_expression_seed,
        target_expression_fusion_mode=args.target_expression_fusion_mode,
        protein_concat_mode=args.protein_concat_mode,
        protein_concat_dim=args.protein_concat_dim,
        protein_concat_topk=args.protein_concat_topk,
        protein_concat_init_scale=args.protein_concat_init_scale,
        protein_concat_seed=args.protein_concat_seed,
        protein_concat_score_mode=args.protein_concat_score_mode,
        protein_concat_expr_scale=args.protein_concat_expr_scale,
        control_logit_scale=args.control_logit_scale,
        pair_logit_scale=args.pair_logit_scale,
        target_logit_scale=args.target_logit_scale,
        covariate_logit_scale=args.covariate_logit_scale,
        use_ddi=args.use_ddi,
        residual_expression=args.residual_expression,
        init_delta_scale=args.init_delta_scale,
    )
    lightning_model = FastProteinTalkLightning(
        model,
        task_head=args.task_head,
        learning_rate=3e-4,
        positive_weight=None,
        have_mse_loss=True,
    )
    load_checkpoint(lightning_model, args.checkpoint_path, strict=not args.allow_partial_checkpoint_load)
    device = torch.device(args.device)
    lightning_model.to(device).eval()

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
            outputs = lightning_model(batch)
            expression, logits1, logits2 = outputs[:3]
            prob1.append(torch.sigmoid(logits1.squeeze(-1)).detach().cpu().numpy())
            prob2.append(torch.sigmoid(logits2.squeeze(-1)).detach().cpu().numpy())
            true1_chunks.append(batch["label1"].detach().cpu().numpy())
            true2_chunks.append(batch["label2"].detach().cpu().numpy())
            mask1_chunks.append(batch["mask1"].detach().cpu().numpy())
            mask2_chunks.append(batch["mask2"].detach().cpu().numpy())
            row_index_chunks.append(batch["row_index"].detach().cpu().numpy().astype(np.int64))
            if args.save_expression_pred:
                expression_chunks.append(expression.detach().cpu().numpy().astype(np.float32))
                expression_true_chunks.append(batch["perturb_expression"].detach().cpu().numpy())
                control_expression_chunks.append(batch["control_expression"].detach().cpu().numpy())

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
    response_metrics = fast_binary_metrics(true1, pred_prob1, mask1)
    synergy_metrics = fast_binary_metrics(true2, pred_prob2, mask2)
    task_metrics = fast_binary_metrics(active_true, active_prob, active_mask)
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
    checkpoint_axis_path = checkpoint_manifest.get("ordered_protein_index_path")
    checkpoint_axis_size = checkpoint_manifest.get("topk_genes")
    if checkpoint_axis_size is None and checkpoint_axis_path:
        checkpoint_axis_size = len(load_json(checkpoint_axis_path))
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
            "graph_feature_mode": args.graph_feature_mode,
            "graph_feature_meta": graph_feature_meta,
            "target_expression_summary": target_expression_summary,
            "covariate_unk_fields": list(covariate_unknown_indices),
        },
    )
    print(f"[infer] wrote {prediction_path} ({len(prediction_df)} rows)")


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
    parser.add_argument("--model-type", choices=sorted(SELECTED_MODEL_NAMES), default=FAST_DELTA_MODEL_NAME)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--expression-latent-dim", type=int, default=512)
    parser.add_argument("--covariate-embedding-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--control-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=3)
    parser.add_argument("--target-layers", type=int, default=2)
    parser.add_argument("--fusion-mode", choices=["concat", "add"], default="concat")
    parser.add_argument("--perturb-fusion-mode", choices=["add", "concat", "mlp"], default="add")
    parser.add_argument("--target-protein-max-length", type=int, default=32)
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
    parser.add_argument("--ppi-matrix-path", default=None)
    parser.add_argument("--pdi-matrix-path", default=None)
    parser.add_argument("--ddi-matrix-path", default=None)
    parser.add_argument("--graph-feature-mode", choices=["real", "zero", "off"], default="real")
    parser.add_argument("--graph-feature-dim", type=int, default=128)
    parser.add_argument("--graph-feature-seed", type=int, default=17)
    parser.add_argument("--graph-structural-rp", action="store_true")
    parser.add_argument("--graph-multihop", action="store_true")
    parser.add_argument("--graph-cache-dir", default=str(REPO_ROOT / "graph_cache"))
    parser.add_argument("--force-graph-cache-rebuild", action="store_true")
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-init-scale", type=float, default=0.1)
    parser.add_argument("--graph-drug-concat", action="store_true")
    parser.add_argument("--graph-pair-add-scale", type=float, default=0.0)
    parser.add_argument("--graph-logit-scale", type=float, default=2.0)
    parser.add_argument("--graph-jump-fusion", choices=["concat", "selective"], default="concat")
    parser.add_argument("--graph-jump-gate", choices=["softmax", "sparsemax"], default="softmax")
    parser.add_argument("--graph-jump-temperature", type=float, default=1.0)
    parser.add_argument(
        "--pair-fusion-mode",
        choices=["symmetric", "rich_symmetric", "ordered_concat", "dual"],
        default="symmetric",
    )
    parser.add_argument("--pair-type-features", action="store_true")
    parser.add_argument("--cell-pair-film-scale", type=float, default=0.0)
    parser.add_argument("--target-expression-mode", choices=["off", "pdi", "pdi_ppi"], default="off")
    parser.add_argument("--target-expression-dim", type=int, default=64)
    parser.add_argument("--target-expression-topk", type=int, default=256)
    parser.add_argument("--target-expression-ppi-topk", type=int, default=32)
    parser.add_argument("--target-expression-ppi-alpha", type=float, default=0.5)
    parser.add_argument("--target-expression-init-scale", type=float, default=0.1)
    parser.add_argument("--target-expression-seed", type=int, default=29)
    parser.add_argument(
        "--target-expression-fusion-mode",
        choices=["piece", "control_add", "pair_add"],
        default="piece",
    )
    parser.add_argument("--target-expression-chunk-size", type=int, default=64)
    parser.add_argument("--target-expression-cache-dir", default="")
    parser.add_argument("--force-target-expression-cache-rebuild", action="store_true")
    parser.add_argument("--protein-concat-mode", choices=["off", "pcep", "pcep_cell", "pcep_dual"], default="pcep")
    parser.add_argument("--protein-concat-dim", type=int, default=64)
    parser.add_argument("--protein-concat-topk", type=int, default=512)
    parser.add_argument("--protein-concat-init-scale", type=float, default=0.1)
    parser.add_argument("--protein-concat-seed", type=int, default=23)
    parser.add_argument("--protein-concat-score-mode", choices=["multiply", "additive", "magnitude"], default="multiply")
    parser.add_argument("--protein-concat-expr-scale", type=float, default=1.0)
    parser.add_argument("--control-logit-scale", type=float, default=0.0)
    parser.add_argument("--pair-logit-scale", type=float, default=0.0)
    parser.add_argument("--target-logit-scale", type=float, default=0.0)
    parser.add_argument("--covariate-logit-scale", type=float, default=0.0)
    parser.add_argument("--use-ddi", action="store_true")
    parser.add_argument("--absolute-expression-head", action="store_false", dest="residual_expression")
    parser.add_argument("--init-delta-scale", type=float, default=0.1)
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
    if args.model_type == FAST_DELTA_MODEL_NAME:
        run_fast_inference(args)
        return

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
