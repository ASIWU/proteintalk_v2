#!/usr/bin/env python3
"""Protein-level Gradient×Delta and Integrated Gradients for Exp34 OOD inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.training_ready_fast_dataset import (
    FastProteinTalkDataset,
    FastTrainingReadyArtifacts,
    load_embedding_matrix,
    load_indices,
    load_row_to_set,
    load_set_info,
)
from infer import (
    expression_alignment_index,
    fast_checkpoint_covariate_known_values,
    fast_checkpoint_covariate_model_sizes,
    fast_checkpoint_covariate_unknown_indices,
    load_checkpoint,
    move_to_device,
)
from model.fast_delta_model import FastDeltaDrugResponseModel
from model.fast_lightning import FastProteinTalkLightning
from train import graph_feature_blocks_from_meta
from utils.attribution_utils import (
    ascending_indices,
    attribution_stability,
    completeness_errors,
    descending_indices,
    gradient_x_delta,
    integrated_gradients,
    nanmedian_baseline,
    slice_batch,
    stable_abs_ranks,
)
from utils.npy_io import safe_np_load


DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2"
)
DEFAULT_RUNTIME_ROOT = Path("/tmp/proteintalk_exp34_update0819_ood_runtime")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2_protein_attribution"
)
DEFAULT_REQUIRED_TMUX_TARGET = "gpu1_deep:0"
ATTRIBUTION_REPORT_TITLE = "Exp34 Protein Attribution"
EXPECTED_CHECKPOINT_SHA256 = "7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076"
EXPECTED_AXIS_SIZE = 11_092
TASKS = {
    "target_none": "ptv3_exp34_update0819_ood_target_none",
    "target_mechanism": "ptv3_exp34_update0819_ood_target_mechanism",
}


@dataclass
class TaskContext:
    scenario: str
    task_name: str
    artifacts: FastTrainingReadyArtifacts
    dataset: FastProteinTalkDataset
    published_predictions: pd.DataFrame
    run_manifest: dict[str, Any]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_dir(path: Path, *, allow_existing: bool) -> None:
    if path.exists() and any(path.iterdir()) and not allow_existing:
        raise FileExistsError(f"attribution output is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def require_tmux_target(required: str) -> str:
    if not required:
        return ""
    actual = os.environ.get("EXP34_TMUX_TARGET", "").strip()
    if not actual:
        completed = subprocess.run(
            ["tmux", "display-message", "-p", "#S:#I"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            actual = completed.stdout.strip()
    require(actual == required, f"attribution must run in tmux {required}; current={actual or 'none'}")
    return actual


def load_row_embedding_features(
    *, artifacts: FastTrainingReadyArtifacts, path: Path, index_column: str
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["embedding_matrix"], dtype=np.float32)
    indices = (
        pd.to_numeric(artifacts.df[index_column], errors="coerce")
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )
    require(indices.min(initial=0) >= 0, f"{index_column}: negative index")
    require(indices.max(initial=0) < matrix.shape[0], f"{index_column}: index overflow")
    return np.asarray(matrix[indices], dtype=np.float32)


def training_control_median_baseline(
    checkpoint_manifest: dict[str, Any], checkpoint_axis: list[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    task_dir = Path(str(checkpoint_manifest["task_dir"]))
    meta_path = Path(str(checkpoint_manifest["meta_path"]))
    strategy = str(checkpoint_manifest["split_strategy"])
    split_dir = Path(str(checkpoint_manifest["split_summary"]["split_dir"]))
    artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
    train_sets = load_set_info(split_dir, "train", strategy)
    control_feature_rows = sorted(
        {
            int(row)
            for set_record in train_sets.values()
            for row in set_record.get("control", [])
        }
    )
    require(bool(control_feature_rows), "checkpoint train set references no controls")
    expression_rows = sorted(
        {int(artifacts.expression_row_indices[row]) for row in control_feature_rows}
    )
    matrix = np.asarray(artifacts.expression_matrix[expression_rows], dtype=np.float32)
    alignment = expression_alignment_index(artifacts.ordered_protein_index, checkpoint_axis)
    if alignment is not None:
        aligned = np.full((matrix.shape[0], len(checkpoint_axis)), np.nan, dtype=np.float32)
        valid = alignment >= 0
        aligned[:, valid] = matrix[:, alignment[valid]]
        matrix = aligned
    baseline, finite_counts, all_nan = nanmedian_baseline(matrix)
    require(baseline.shape == (len(checkpoint_axis),), "baseline axis mismatch")
    return baseline, {
        "policy": "per-protein nanmedian over checkpoint-train referenced control expression rows",
        "checkpoint_task": checkpoint_manifest["task_name"],
        "checkpoint_split_strategy": strategy,
        "control_feature_row_count": len(control_feature_rows),
        "unique_expression_row_count": len(expression_rows),
        "all_nan_protein_count": int(all_nan.sum()),
        "finite_count_min": int(finite_counts.min()),
        "finite_count_max": int(finite_counts.max()),
    }


def create_model(
    *,
    config: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    protein_embedding: np.ndarray,
    drug_embedding: np.ndarray,
    checkpoint_axis: list[int],
    graph_feature_meta: dict[str, Any],
    graph_feature_width: int,
    meta: dict[str, Any],
) -> FastProteinTalkLightning:
    model = FastDeltaDrugResponseModel(
        n_genes=len(checkpoint_axis),
        drug_embedding_dim=int(drug_embedding.shape[1]),
        protein_embedding=protein_embedding,
        ordered_protein_index=checkpoint_axis,
        covariate_sizes=fast_checkpoint_covariate_model_sizes(
            checkpoint_manifest, meta, list(config["batch_cov_list"])
        ),
        hidden_dim=int(config["hidden_dim"]),
        expression_latent_dim=int(config["expression_latent_dim"]),
        covariate_embedding_dim=int(config["covariate_embedding_dim"]),
        dropout=float(config["dropout"]),
        control_layers=int(config["control_layers"]),
        fusion_layers=int(config["fusion_layers"]),
        target_layers=int(config["target_layers"]),
        graph_feature_dim=int(graph_feature_width),
        graph_layers=int(config["graph_layers"]),
        graph_init_scale=float(config["graph_init_scale"]),
        graph_drug_concat=bool(config["graph_drug_concat"]),
        graph_pair_add_scale=float(config["graph_pair_add_scale"]),
        graph_logit_scale=float(config["graph_logit_scale"]),
        graph_feature_blocks=graph_feature_blocks_from_meta(graph_feature_meta),
        graph_jump_fusion=str(config["graph_jump_fusion"]),
        graph_jump_gate=str(config["graph_jump_gate"]),
        graph_jump_temperature=float(config["graph_jump_temperature"]),
        pair_fusion_mode=str(config["pair_fusion_mode"]),
        pair_type_features=bool(config["pair_type_features"]),
        cell_pair_film_scale=float(config["cell_pair_film_scale"]),
        target_expression_mode=str(config["target_expression_mode"]),
        target_expression_weight_matrix=None,
        target_expression_dim=int(config["target_expression_dim"]),
        target_expression_init_scale=float(config["target_expression_init_scale"]),
        target_expression_seed=int(config["target_expression_seed"]),
        target_expression_fusion_mode=str(config["target_expression_fusion_mode"]),
        target_expression_cell_gate_mode=str(config["target_expression_cell_gate_mode"]),
        target_expression_cell_gate_scale=float(config["target_expression_cell_gate_scale"]),
        target_expression_cell_gate_temperature=float(
            config["target_expression_cell_gate_temperature"]
        ),
        protein_concat_mode=str(config["protein_concat_mode"]),
        protein_concat_dim=int(config["protein_concat_dim"]),
        protein_concat_topk=int(config["protein_concat_topk"]),
        protein_concat_init_scale=float(config["protein_concat_init_scale"]),
        protein_concat_seed=int(config["protein_concat_seed"]),
        protein_concat_score_mode=str(config["protein_concat_score_mode"]),
        protein_concat_expr_scale=float(config["protein_concat_expr_scale"]),
        control_logit_scale=float(config["control_logit_scale"]),
        pair_logit_scale=float(config["pair_logit_scale"]),
        target_logit_scale=float(config["target_logit_scale"]),
        covariate_logit_scale=float(config["covariate_logit_scale"]),
        response_base_logit_scale=float(config["response_base_logit_scale"]),
        response_delta_mode=str(config["response_delta_mode"]),
        response_delta_dim=int(config["response_delta_dim"]),
        response_delta_seed=int(config["response_delta_seed"]),
        response_delta_detach=bool(config["response_delta_detach"]),
        delta_logit_scale=float(config["delta_logit_scale"]),
        delta_logit_learnable=bool(config["delta_logit_learnable"]),
        response_trajectory_mode=str(config["response_trajectory_mode"]),
        response_trajectory_dim=int(config["response_trajectory_dim"]),
        response_trajectory_seed=int(config["response_trajectory_seed"]),
        response_trajectory_detach=bool(config["response_trajectory_detach"]),
        trajectory_logit_scale=float(config["trajectory_logit_scale"]),
        trajectory_logit_learnable=bool(config["trajectory_logit_learnable"]),
        cell_type_feature_dim=int(config["cell_llm_feature_dim"]),
        cell_type_fusion_mode=str(config["cell_llm_fusion_mode"]),
        cell_type_condition_scale=float(config["cell_llm_condition_scale"]),
        cell_type_logit_scale=float(config["cell_llm_logit_scale"]),
        cell_type_dropout=float(config["cell_llm_dropout"]),
        cell_type_llm_feature_dim=int(config["cell_type_llm_feature_dim"]),
        cell_type_llm_fusion_mode=str(config["cell_type_llm_fusion_mode"]),
        cell_type_llm_condition_scale=float(config["cell_type_llm_condition_scale"]),
        cell_type_llm_logit_scale=float(config["cell_type_llm_logit_scale"]),
        cell_type_llm_dropout=float(config["cell_type_llm_dropout"]),
        control_drug_interaction_mode=str(config["control_drug_interaction_mode"]),
        control_drug_interaction_scale=float(config["control_drug_interaction_scale"]),
        control_drug_logit_scale=float(config["control_drug_logit_scale"]),
        observed_perturb_expression_mode=str(config["observed_perturb_expression_mode"]),
        observed_perturb_expression_scale=float(config["observed_perturb_expression_scale"]),
        observed_perturb_logit_scale=float(config["observed_perturb_logit_scale"]),
        prior_feature_dim=0,
        prior_logit_scale=float(config["cell_prior_logit_scale"]),
        prior_fixed_logit_scale=float(config["cell_prior_fixed_logit_scale"]),
        use_ddi=bool(config["use_ddi"]),
        residual_expression=bool(config["residual_expression"]),
        init_delta_scale=float(config["init_delta_scale"]),
        zero_init_delta_head=bool(config["zero_init_delta_head"]),
        control_expression_dropout=float(config["control_expression_dropout"]),
    )
    lightning = FastProteinTalkLightning(
        model,
        task_head="unified",
        learning_rate=3e-4,
        positive_weight=None,
        have_mse_loss=True,
    )
    return lightning


def build_task_context(
    *,
    scenario: str,
    task_name: str,
    run_root: Path,
    runtime_root: Path,
    checkpoint_manifest: dict[str, Any],
    drug_embedding: np.ndarray,
    ddi_matrix: np.ndarray,
    graph_feature_matrix: np.ndarray,
    checkpoint_axis: list[int],
    limit_samples: int | None,
) -> TaskContext:
    prediction_dir = run_root / task_name
    run_manifest = load_json(prediction_dir / "run_manifest.json")
    published = pd.read_parquet(prediction_dir / "predictions.parquet")
    task_dir = runtime_root / "ptv3/tasks" / task_name
    meta_path = runtime_root / "ptv3/global_meta.json"
    artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
    indices = load_indices(runtime_root / "ptv3/splits" / task_name, "test", "test_only")
    if limit_samples is not None:
        indices = indices[:limit_samples]
        published = published.iloc[:limit_samples].reset_index(drop=True)
    row_to_set = load_row_to_set(runtime_root / "ptv3/splits" / task_name)
    set_info = load_set_info(runtime_root / "ptv3/splits" / task_name, "test", "test_only")
    cell_features = load_row_embedding_features(
        artifacts=artifacts,
        path=Path(str(run_manifest["cell_llm_embedding_path"])),
        index_column="cell_llm_index",
    )
    cell_type_features = load_row_embedding_features(
        artifacts=artifacts,
        path=Path(str(run_manifest["cell_type_llm_embedding_path"])),
        index_column="cell_type_llm_index",
    )
    alignment = expression_alignment_index(artifacts.ordered_protein_index, checkpoint_axis)
    batch_covariates = list(run_manifest["batch_cov_list"])
    dataset = FastProteinTalkDataset(
        artifacts=artifacts,
        indices=indices,
        row_to_set_index=row_to_set,
        set_info=set_info,
        mode="eval",
        drug_embedding_matrix=drug_embedding,
        batch_cov_list=batch_covariates,
        target_protein_max_length=32,
        effective_key1="PRISM1st_label_total",
        effective_key2="synergy",
        ddi_matrix=ddi_matrix,
        graph_feature_matrix=graph_feature_matrix,
        graph_feature_enabled=True,
        expression_column_index=alignment,
        covariate_known_values=fast_checkpoint_covariate_known_values(
            checkpoint_manifest, batch_covariates
        ),
        covariate_unknown_indices=fast_checkpoint_covariate_unknown_indices(
            checkpoint_manifest, artifacts.meta, batch_covariates
        ),
        prior_feature_matrix=None,
        cell_type_feature_matrix=cell_features,
        cell_type_llm_feature_matrix=cell_type_features,
        control_expression_mode="real",
        random_control_expression_matrix=None,
    )
    return TaskContext(
        scenario=scenario,
        task_name=task_name,
        artifacts=artifacts,
        dataset=dataset,
        published_predictions=published,
        run_manifest=run_manifest,
    )


def run(args: argparse.Namespace) -> None:
    started_at = iso_now()
    resolved_tmux_target = require_tmux_target(str(args.required_tmux_target))
    run_root = Path(args.run_root)
    runtime_root = Path(args.training_ready_root)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, allow_existing=bool(args.allow_existing))
    require(runtime_root.exists(), f"missing runtime root: {runtime_root}")

    manifests = {
        scenario: load_json(run_root / task_name / "run_manifest.json")
        for scenario, task_name in TASKS.items()
    }
    checkpoint_path = Path(str(next(iter(manifests.values()))["checkpoint_path"]))
    require(sha256_file(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256, "checkpoint SHA mismatch")
    checkpoint_manifest = load_json(checkpoint_path.parent / "run_manifest.json")
    checkpoint_axis = [int(item) for item in load_json(checkpoint_manifest["ordered_protein_index_path"])]
    require(len(checkpoint_axis) == EXPECTED_AXIS_SIZE, "checkpoint axis mismatch")

    reference_manifest = next(iter(manifests.values()))
    config = dict(reference_manifest["inference_model_config"])
    for scenario, manifest in manifests.items():
        require(
            manifest["inference_model_config"] == reference_manifest["inference_model_config"],
            f"{scenario}: inference configs differ",
        )
        require(
            manifest["checkpoint_config_validation"]["architecture_matches"] is True,
            f"{scenario}: architecture mismatch",
        )
    protein_embedding = load_embedding_matrix(config["protein_embedding_path"])
    drug_embedding = load_embedding_matrix(config["drug_embedding_path"])
    ddi_matrix = np.asarray(safe_np_load(config["ddi_matrix_path"], mmap_mode="r"), dtype=np.float32)
    graph_meta = dict(reference_manifest["graph_feature_meta"])
    graph_feature_matrix = np.asarray(
        safe_np_load(graph_meta["feature_path"], mmap_mode="r"), dtype=np.float32
    )
    require(drug_embedding.shape[0] == ddi_matrix.shape[0] == graph_feature_matrix.shape[0], "drug axes differ")

    contexts = {
        scenario: build_task_context(
            scenario=scenario,
            task_name=task_name,
            run_root=run_root,
            runtime_root=runtime_root,
            checkpoint_manifest=checkpoint_manifest,
            drug_embedding=drug_embedding,
            ddi_matrix=ddi_matrix,
            graph_feature_matrix=graph_feature_matrix,
            checkpoint_axis=checkpoint_axis,
            limit_samples=args.limit_samples,
        )
        for scenario, task_name in TASKS.items()
    }
    reference_context = next(iter(contexts.values()))
    meta = reference_context.artifacts.meta
    lightning = create_model(
        config=config,
        checkpoint_manifest=checkpoint_manifest,
        protein_embedding=protein_embedding,
        drug_embedding=drug_embedding,
        checkpoint_axis=checkpoint_axis,
        graph_feature_meta=graph_meta,
        graph_feature_width=int(graph_feature_matrix.shape[1]),
        meta=meta,
    )
    load_checkpoint(lightning, str(checkpoint_path), strict=True)
    device = torch.device(args.device)
    lightning.to(device).eval()
    model = lightning.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    baseline_np, baseline_summary = training_control_median_baseline(
        checkpoint_manifest, checkpoint_axis
    )
    baseline_path = output_dir / "training_control_median_baseline.npy"
    np.save(baseline_path, baseline_np)
    baseline_summary.update(
        {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
            "axis_size": len(checkpoint_axis),
        }
    )
    dump_json(output_dir / "baseline_summary.json", baseline_summary)

    uniprot = [
        str(item)
        for item in load_json(
            runtime_root
            / "ptv3/tasks"
            / reference_context.task_name
            / "feature_ordered_protein_uniprot.json"
        )
    ]
    protein_indices = np.asarray(checkpoint_axis, dtype=np.int64)
    require(len(uniprot) == len(protein_indices) == EXPECTED_AXIS_SIZE, "output axis mismatch")
    dump_json(output_dir / "protein_axis_uniprot.json", uniprot)
    dump_json(output_dir / "protein_axis_index.json", checkpoint_axis)

    long_parts: list[pd.DataFrame] = []
    top_abs_parts: list[pd.DataFrame] = []
    top_positive_parts: list[pd.DataFrame] = []
    top_negative_parts: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    model_reproduction_rows: list[dict[str, Any]] = []

    for scenario, context in contexts.items():
        loader = DataLoader(
            context.dataset,
            batch_size=len(context.dataset),
            shuffle=False,
            num_workers=0,
        )
        batch = move_to_device(next(iter(loader)), device)
        inputs = torch.nan_to_num(
            batch["control_expression"].float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        baseline = torch.as_tensor(baseline_np, device=device, dtype=torch.float32).unsqueeze(0)
        baseline = baseline.expand_as(inputs)
        input_logits, raw_gradient, grad_delta = gradient_x_delta(
            model, batch, inputs, baseline
        )
        integrated, ig_input_logits, baseline_logits = integrated_gradients(
            model, batch, inputs, baseline, steps=int(args.ig_steps)
        )
        require(
            torch.allclose(input_logits, ig_input_logits, rtol=0.0, atol=1e-6),
            f"{scenario}: input logits changed across attribution methods",
        )
        logit_delta, absolute_error, relative_error = completeness_errors(
            integrated, ig_input_logits, baseline_logits
        )
        used_steps = np.full(len(inputs), int(args.ig_steps), dtype=np.int64)
        adaptive = (
            (absolute_error > float(args.adaptive_absolute_threshold))
            & (relative_error > float(args.adaptive_relative_threshold))
        ).detach().cpu().numpy()
        for sample_index in np.flatnonzero(adaptive):
            one_batch = slice_batch(batch, int(sample_index))
            refined, refined_input, refined_baseline = integrated_gradients(
                model,
                one_batch,
                inputs[sample_index : sample_index + 1],
                baseline[sample_index : sample_index + 1],
                steps=int(args.adaptive_ig_steps),
            )
            integrated[sample_index] = refined[0]
            ig_input_logits[sample_index] = refined_input[0]
            baseline_logits[sample_index] = refined_baseline[0]
            used_steps[sample_index] = int(args.adaptive_ig_steps)
        logit_delta, absolute_error, relative_error = completeness_errors(
            integrated, ig_input_logits, baseline_logits
        )

        feature_rows = batch["row_index"].detach().cpu().numpy().astype(np.int64)
        rows = context.artifacts.df.iloc[feature_rows].reset_index(drop=True)
        published = context.published_predictions.reset_index(drop=True)
        require(
            rows["sample_id"].astype(str).tolist()
            == published["sample_id"].astype(str).tolist(),
            f"{scenario}: published row alignment mismatch",
        )
        reproduced_probability = torch.sigmoid(input_logits).detach().cpu().numpy()
        published_probability = published["pred_task_prob"].to_numpy(dtype=np.float64)
        probability_error = np.abs(reproduced_probability - published_probability)
        require(
            float(probability_error.max(initial=0.0)) <= float(args.probability_tolerance),
            f"{scenario}: published probability reproduction failed: {probability_error.max()}",
        )

        values_input = inputs.detach().cpu().numpy().astype(np.float32)
        values_baseline = baseline.detach().cpu().numpy().astype(np.float32)
        values_gradient = raw_gradient.detach().cpu().numpy().astype(np.float32)
        values_grad_delta = grad_delta.detach().cpu().numpy().astype(np.float32)
        values_ig = integrated.detach().cpu().numpy().astype(np.float32)
        values_input_logits = input_logits.detach().cpu().numpy().astype(np.float64)
        values_baseline_logits = baseline_logits.detach().cpu().numpy().astype(np.float64)
        values_logit_delta = logit_delta.detach().cpu().numpy().astype(np.float64)
        values_abs_error = absolute_error.detach().cpu().numpy().astype(np.float64)
        values_rel_error = relative_error.detach().cpu().numpy().astype(np.float64)

        for sample_index, row in rows.iterrows():
            sample_id = str(row["sample_id"])
            ig_values = values_ig[sample_index]
            gxd_values = values_grad_delta[sample_index]
            ranks = stable_abs_ranks(ig_values, protein_indices)
            base_record = {
                "target_scenario": scenario,
                "task_name": context.task_name,
                "sample_id": sample_id,
                "input_row_index": int(row["input_row_index"]),
                "Cell": str(row["Cell"]),
                "Cell_name_input": str(row["Cell_name_input"]),
                "drug_name": str(row["drug_name"]),
                "drug_name2": str(row["drug_name2"]),
                "pert_id": str(row["pert_id1"]),
                "target_protein_list": str(row["target_protein_list"]),
                "pred_sensitivity_prob": float(published_probability[sample_index]),
                "decision_threshold": 0.5,
                "predicted_sensitive": int(published_probability[sample_index] >= 0.5),
                "predicted_response_label": (
                    "sensitive"
                    if published_probability[sample_index] >= 0.5
                    else "non-responsive"
                ),
            }
            long_frame = pd.DataFrame(
                {
                    **{key: value for key, value in base_record.items()},
                    "axis_column": np.arange(EXPECTED_AXIS_SIZE, dtype=np.int64),
                    "protein_index": protein_indices,
                    "uniprot": uniprot,
                    "control_expression": values_input[sample_index],
                    "baseline_expression": values_baseline[sample_index],
                    "expression_delta": values_input[sample_index]
                    - values_baseline[sample_index],
                    "raw_gradient": values_gradient[sample_index],
                    "gradient_x_delta": gxd_values,
                    "integrated_gradient": ig_values,
                    "abs_integrated_gradient": np.abs(ig_values),
                    "abs_ig_rank": ranks,
                    "direction": np.where(
                        ig_values > 0, "increase_sensitivity_logit", np.where(
                            ig_values < 0, "decrease_sensitivity_logit", "zero"
                        )
                    ),
                }
            )
            long_parts.append(long_frame)

            abs_order = np.lexsort((protein_indices, -np.abs(ig_values)))[: int(args.top_k)]
            positive_order = descending_indices(ig_values, protein_indices)
            positive_order = positive_order[ig_values[positive_order] > 0][: int(args.signed_top_k)]
            negative_order = ascending_indices(ig_values, protein_indices)
            negative_order = negative_order[ig_values[negative_order] < 0][: int(args.signed_top_k)]

            def top_frame(indices: np.ndarray, kind: str) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        **{key: value for key, value in base_record.items()},
                        "ranking_kind": kind,
                        "rank": np.arange(1, len(indices) + 1, dtype=np.int64),
                        "axis_column": indices.astype(np.int64),
                        "protein_index": protein_indices[indices],
                        "uniprot": np.asarray(uniprot, dtype=object)[indices],
                        "integrated_gradient": ig_values[indices],
                        "abs_integrated_gradient": np.abs(ig_values[indices]),
                        "gradient_x_delta": gxd_values[indices],
                        "control_expression": values_input[sample_index, indices],
                        "baseline_expression": values_baseline[sample_index, indices],
                    }
                )

            top_abs_parts.append(top_frame(abs_order, "absolute"))
            top_positive_parts.append(top_frame(positive_order, "positive"))
            top_negative_parts.append(top_frame(negative_order, "negative"))
            overlap, correlation = attribution_stability(gxd_values, ig_values)
            warning = bool(
                values_abs_error[sample_index] > float(args.adaptive_absolute_threshold)
                and values_rel_error[sample_index] > float(args.final_warning_relative_threshold)
            )
            diagnostic_rows.append(
                {
                    **base_record,
                    "published_probability": float(published_probability[sample_index]),
                    "reproduced_probability": float(reproduced_probability[sample_index]),
                    "probability_abs_error": float(probability_error[sample_index]),
                    "input_logit": float(values_input_logits[sample_index]),
                    "baseline_logit": float(values_baseline_logits[sample_index]),
                    "logit_delta": float(values_logit_delta[sample_index]),
                    "ig_sum": float(ig_values.sum(dtype=np.float64)),
                    "completeness_absolute_error": float(values_abs_error[sample_index]),
                    "completeness_relative_error": float(values_rel_error[sample_index]),
                    "ig_steps_used": int(used_steps[sample_index]),
                    "completeness_warning": warning,
                    "gradient_delta_ig_top100_overlap": overlap,
                    "gradient_delta_ig_abs_spearman": correlation,
                }
            )
            model_reproduction_rows.append(
                {
                    **base_record,
                    "published_probability": float(published_probability[sample_index]),
                    "reproduced_probability": float(reproduced_probability[sample_index]),
                    "absolute_error": float(probability_error[sample_index]),
                }
            )

    long_output = pd.concat(long_parts, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    reproduction = pd.DataFrame(model_reproduction_rows)
    expected_samples = sum(len(context.dataset) for context in contexts.values())
    require(
        len(long_output) == expected_samples * EXPECTED_AXIS_SIZE,
        "long attribution row count mismatch",
    )
    require(
        long_output.groupby("sample_id").size().eq(EXPECTED_AXIS_SIZE).all(),
        "a sample does not have exactly 11,092 protein rows",
    )
    numeric_columns = [
        "control_expression",
        "baseline_expression",
        "expression_delta",
        "raw_gradient",
        "gradient_x_delta",
        "integrated_gradient",
        "abs_integrated_gradient",
    ]
    require(
        np.isfinite(long_output[numeric_columns].to_numpy(dtype=np.float64)).all(),
        "non-finite attribution values",
    )

    paths = {
        "full_parquet": output_dir / "protein_attributions.parquet",
        "top_abs_csv": output_dir / "top200_absolute_ig_per_sample.csv",
        "top_positive_csv": output_dir / "top100_positive_ig_per_sample.csv",
        "top_negative_csv": output_dir / "top100_negative_ig_per_sample.csv",
        "diagnostics_csv": output_dir / "sample_diagnostics.csv",
        "reproduction_csv": output_dir / "published_probability_reproduction.csv",
        "summary_json": output_dir / "summary.json",
        "manifest_json": output_dir / "run_manifest.json",
        "results_md": output_dir / "results.md",
    }
    long_output.to_parquet(paths["full_parquet"], index=False)
    pd.concat(top_abs_parts, ignore_index=True).to_csv(paths["top_abs_csv"], index=False)
    pd.concat(top_positive_parts, ignore_index=True).to_csv(
        paths["top_positive_csv"], index=False
    )
    pd.concat(top_negative_parts, ignore_index=True).to_csv(
        paths["top_negative_csv"], index=False
    )
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    reproduction.to_csv(paths["reproduction_csv"], index=False)

    completed_at = iso_now()
    summary = {
        "started_at": started_at,
        "completed_at": completed_at,
        "method": {
            "coarse": "gradient_x_(control_expression-training_control_median)",
            "formal": "midpoint Integrated Gradients on response logit",
            "initial_steps": int(args.ig_steps),
            "adaptive_steps": int(args.adaptive_ig_steps),
        },
        "sample_count": expected_samples,
        "protein_count": EXPECTED_AXIS_SIZE,
        "attribution_row_count": int(len(long_output)),
        "target_scenarios": list(TASKS),
        "baseline": baseline_summary,
        "probability_reproduction_max_abs_error": float(
            reproduction["absolute_error"].max()
        ),
        "adaptive_sample_count": int((diagnostics["ig_steps_used"] > args.ig_steps).sum()),
        "completeness_warning_count": int(diagnostics["completeness_warning"].sum()),
        "completeness_absolute_error_max": float(
            diagnostics["completeness_absolute_error"].max()
        ),
        "completeness_relative_error_max": float(
            diagnostics["completeness_relative_error"].max()
        ),
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "interpretation": (
            "Signed attributions explain this checkpoint's response logit relative to the "
            "checkpoint-train control median; they are not causal biomarkers."
        ),
    }
    dump_json(paths["summary_json"], summary)
    run_manifest = {
        **summary,
        "run_root": str(run_root.resolve()),
        "runtime_training_ready_root": str(runtime_root.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "device": str(device),
        "tmux_target": resolved_tmux_target,
        "limit_samples": args.limit_samples,
        "arguments": vars(args),
    }
    dump_json(paths["manifest_json"], run_manifest)
    markdown = f"""# {ATTRIBUTION_REPORT_TITLE}

- Samples: `{expected_samples}`
- Proteins per sample: `{EXPECTED_AXIS_SIZE}`
- Formal method: midpoint Integrated Gradients on response logit
- Initial/adaptive steps: `{args.ig_steps}/{args.adaptive_ig_steps}`
- Baseline: checkpoint-train referenced control per-protein median
- Published-probability max reproduction error: `{summary['probability_reproduction_max_abs_error']:.3e}`
- Adaptive samples: `{summary['adaptive_sample_count']}`
- Completeness warnings: `{summary['completeness_warning_count']}`

Positive attribution increases the model sensitivity logit; negative attribution decreases it. Rankings are per sample only and are not causal biomarker evidence, especially because both drugs are chemical OOD.
"""
    paths["results_md"].write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--training-ready-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-tmux-target", default=DEFAULT_REQUIRED_TMUX_TARGET)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--adaptive-ig-steps", type=int, default=64)
    parser.add_argument("--adaptive-absolute-threshold", type=float, default=1e-3)
    parser.add_argument("--adaptive-relative-threshold", type=float, default=0.05)
    parser.add_argument("--final-warning-relative-threshold", type=float, default=0.10)
    parser.add_argument("--probability-tolerance", type=float, default=1e-7)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--signed-top-k", type=int, default=100)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    if args.ig_steps <= 0 or args.adaptive_ig_steps < args.ig_steps:
        parser.error("IG steps must be positive and adaptive steps must be >= initial steps")
    if args.limit_samples is not None and args.limit_samples <= 0:
        parser.error("--limit-samples must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
