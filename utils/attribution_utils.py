#!/usr/bin/env python3
"""Lightweight, model-agnostic attribution helpers."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.stats import spearmanr


def model_response_logits(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    control_expression: torch.Tensor,
) -> torch.Tensor:
    attributed_batch = dict(batch)
    attributed_batch["control_expression"] = control_expression
    return model(attributed_batch)[1].squeeze(-1)


def gradient_x_delta(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = inputs.detach().clone().requires_grad_(True)
    logits = model_response_logits(model, batch, x)
    gradient = torch.autograd.grad(logits.sum(), x, create_graph=False)[0]
    score = gradient * (inputs - baseline)
    return logits.detach(), gradient.detach(), score.detach()


def integrated_gradients(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    *,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Midpoint-rule Integrated Gradients for independent batched samples."""
    if steps <= 0:
        raise ValueError("Integrated Gradients steps must be positive")
    delta = inputs - baseline
    gradient_sum = torch.zeros_like(inputs)
    for step in range(steps):
        alpha = (float(step) + 0.5) / float(steps)
        point = (baseline + alpha * delta).detach().requires_grad_(True)
        logits = model_response_logits(model, batch, point)
        gradient = torch.autograd.grad(logits.sum(), point, create_graph=False)[0]
        gradient_sum.add_(gradient.detach())
    attribution = delta * (gradient_sum / float(steps))
    with torch.no_grad():
        input_logits = model_response_logits(model, batch, inputs)
        baseline_logits = model_response_logits(model, batch, baseline)
    return attribution.detach(), input_logits.detach(), baseline_logits.detach()


def completeness_errors(
    attribution: torch.Tensor,
    input_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attributed_delta = attribution.sum(dim=1)
    logit_delta = input_logits - baseline_logits
    absolute_error = torch.abs(attributed_delta - logit_delta)
    relative_error = absolute_error / torch.clamp(torch.abs(logit_delta), min=1e-6)
    return logit_delta, absolute_error, relative_error


def slice_batch(batch: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    batch_size = int(batch["control_expression"].shape[0])
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            result[key] = value[index : index + 1]
        else:
            result[key] = value
    return result


def stable_abs_ranks(values: np.ndarray, protein_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    order = np.lexsort((protein_indices, -np.abs(values)))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


def descending_indices(values: np.ndarray, protein_indices: np.ndarray) -> np.ndarray:
    return np.lexsort((np.asarray(protein_indices, dtype=np.int64), -np.asarray(values)))


def ascending_indices(values: np.ndarray, protein_indices: np.ndarray) -> np.ndarray:
    return np.lexsort((np.asarray(protein_indices, dtype=np.int64), np.asarray(values)))


def attribution_stability(
    gradient_delta: np.ndarray,
    integrated: np.ndarray,
    *,
    top_k: int = 100,
) -> tuple[float, float]:
    first = np.argsort(np.abs(gradient_delta))[::-1][:top_k]
    second = np.argsort(np.abs(integrated))[::-1][:top_k]
    overlap = len(set(map(int, first)) & set(map(int, second))) / float(top_k)
    correlation = spearmanr(np.abs(gradient_delta), np.abs(integrated)).statistic
    return float(overlap), float(correlation)


def nanmedian_baseline(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("baseline source matrix must be nonempty and 2D")
    finite_counts = np.isfinite(matrix).sum(axis=0).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        baseline = np.nanmedian(matrix, axis=0).astype(np.float32)
    all_nan = ~np.isfinite(baseline)
    baseline = np.nan_to_num(baseline, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return baseline, finite_counts, all_nan
