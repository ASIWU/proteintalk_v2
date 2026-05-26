#!/usr/bin/env python3
"""Lightning wrapper for training-ready double-drug ProteinTalk models."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def pearson_per_sample(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=1)
    a_scale = np.sum(a_centered**2, axis=1)
    b_scale = np.sum(b_centered**2, axis=1)
    corr = numerator / (np.sqrt(a_scale * b_scale) + eps)
    corr = corr.astype(np.float64)
    corr[(a_scale < eps) | (b_scale < eps)] = np.nan
    return corr


def r2_per_sample(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_bar = y_true.mean(axis=1, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
    ss_tot = np.sum((y_true - y_bar) ** 2, axis=1)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    r2 = r2.astype(np.float64)
    r2[ss_tot < eps] = np.nan
    return r2


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    mask = np.abs(y_true) > thr
    values = []
    for row_idx in range(y_true.shape[0]):
        row_mask = mask[row_idx]
        if not np.any(row_mask):
            values.append(np.nan)
            continue
        values.append(np.mean((sign_true[row_idx, row_mask] == sign_pred[row_idx, row_mask]).astype(np.float64)))
    return np.asarray(values, dtype=np.float64)


def broadcast_control_like(control_expression: np.ndarray, *, n_samples: int, n_genes: int) -> np.ndarray:
    control = np.asarray(control_expression, dtype=np.float64)
    if control.ndim == 1 and control.shape[0] == n_genes:
        return np.broadcast_to(control, (n_samples, n_genes)).copy()
    if control.ndim == 2 and control.shape == (1, n_genes):
        return np.broadcast_to(control, (n_samples, n_genes)).copy()
    if control.ndim == 2 and control.shape == (n_samples, 1):
        return np.broadcast_to(control, (n_samples, n_genes)).copy()
    if control.ndim == 2 and control.shape == (n_samples, n_genes):
        return control.copy()
    return np.broadcast_to(control, (n_samples, n_genes)).copy()


def select_topk_genes_by_abs(
    reference: np.ndarray,
    k: int,
    control_expression: np.ndarray | None = None,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    k = int(min(k, reference.shape[1]))
    if control_expression is not None:
        control = broadcast_control_like(control_expression, n_samples=reference.shape[0], n_genes=reference.shape[1])
        score = np.mean(np.abs(reference - control), axis=0)
    else:
        score = np.mean(np.abs(reference), axis=0)
    indices = np.argpartition(score, -k)[-k:]
    return indices[np.argsort(score[indices])[::-1]]


def roc_auc(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray | None = None) -> float:
    try:
        if mask is not None:
            keep = ~mask.astype(bool)
            if not np.any(keep):
                return float("nan")
            y_true = y_true[keep]
            y_score = y_score[keep]
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def auprc_score(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray | None = None) -> float:
    try:
        if mask is not None:
            keep = ~mask.astype(bool)
            if not np.any(keep):
                return float("nan")
            y_true = y_true[keep]
            y_score = y_score[keep]
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray | None = None, thr: float = 0.5) -> float:
    if mask is not None:
        keep = ~mask.astype(bool)
        if not np.any(keep):
            return float("nan")
        y_true = y_true[keep]
        y_prob = y_prob[keep]
    y_hat = (y_prob >= thr).astype(np.float64)
    return float((y_hat == y_true).mean()) if len(y_true) else float("nan")


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    if y is None:
        y = x
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    distances = x_norm + y_norm - 2.0 * (x @ y.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def mmd_rbf(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, subset_max: int = 512) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    x_count = min(x.shape[0], subset_max)
    y_count = min(y.shape[0], subset_max)
    x_idx = rng.choice(x.shape[0], size=x_count, replace=False) if x_count < x.shape[0] else np.arange(x.shape[0])
    y_idx = rng.choice(y.shape[0], size=y_count, replace=False) if y_count < y.shape[0] else np.arange(y.shape[0])
    x_sample = x[x_idx]
    y_sample = y[y_idx]
    combined = np.vstack([x_sample, y_sample])
    median_count = min(combined.shape[0], 1000)
    median_idx = rng.choice(combined.shape[0], size=median_count, replace=False) if median_count < combined.shape[0] else np.arange(combined.shape[0])
    distances = np.sqrt(pairwise_sq_dists(combined[median_idx]))
    upper = np.triu_indices_from(distances, k=1)
    positive = distances[upper][distances[upper] > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = np.exp(-gamma * pairwise_sq_dists(x_sample))
    k_yy = np.exp(-gamma * pairwise_sq_dists(y_sample))
    k_xy = np.exp(-gamma * pairwise_sq_dists(x_sample, y_sample))
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    m = k_xx.shape[0]
    n = k_yy.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    return float((k_xx.sum() / (m * (m - 1))) + (k_yy.sum() / (n * (n - 1))) - 2.0 * k_xy.mean())


def energy_distance_nd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, subset_max: int = 512) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    x_count = min(x.shape[0], subset_max)
    y_count = min(y.shape[0], subset_max)
    x_idx = rng.choice(x.shape[0], size=x_count, replace=False) if x_count < x.shape[0] else np.arange(x.shape[0])
    y_idx = rng.choice(y.shape[0], size=y_count, replace=False) if y_count < y.shape[0] else np.arange(y.shape[0])
    x_sample = x[x_idx]
    y_sample = y[y_idx]

    def offdiag_mean(distances: np.ndarray) -> float:
        upper = np.triu_indices_from(distances, k=1)
        return float(distances[upper].mean()) if upper[0].size else float("nan")

    d_xy = np.sqrt(pairwise_sq_dists(x_sample, y_sample))
    d_xx = np.sqrt(pairwise_sq_dists(x_sample))
    d_yy = np.sqrt(pairwise_sq_dists(y_sample))
    b = offdiag_mean(d_xx)
    c = offdiag_mean(d_yy)
    if not np.isfinite(b) or not np.isfinite(c):
        return float("nan")
    return float(2.0 * d_xy.mean() - b - c)


def compute_validation_metrics(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    ny_pred1: np.ndarray,
    ny_true1: np.ndarray,
    mask1: np.ndarray,
    ny_pred2: np.ndarray,
    ny_true2: np.ndarray,
    mask2: np.ndarray,
    control_expression: np.ndarray | None = None,
    top_k: int = 50,
    eps: float = 1e-8,
    direction_threshold: float = 1e-8,
    mmd_subset_max: int = 512,
    energy_subset_max: int = 512,
    random_state: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    valid_expression = np.isfinite(predictions) & np.isfinite(targets)
    predictions = np.where(valid_expression, predictions, 0.0)
    targets = np.where(valid_expression, targets, 0.0)
    if control_expression is not None:
        control_expression = np.asarray(control_expression, dtype=np.float64)
        control_expression = np.where(np.isfinite(control_expression), control_expression, 0.0)
    ny_pred1 = np.asarray(ny_pred1, dtype=np.float64).reshape(-1)
    ny_true1 = np.asarray(ny_true1, dtype=np.float64).reshape(-1)
    mask1 = np.asarray(mask1, dtype=np.float64).reshape(-1)
    ny_pred2 = np.asarray(ny_pred2, dtype=np.float64).reshape(-1)
    ny_true2 = np.asarray(ny_true2, dtype=np.float64).reshape(-1)
    mask2 = np.asarray(mask2, dtype=np.float64).reshape(-1)
    n_samples, n_genes = predictions.shape
    if targets.shape != (n_samples, n_genes):
        raise ValueError("predictions and targets must have the same shape [N, G]")

    mse_all = mse(predictions, targets)
    mae_all = mae(predictions, targets)
    pcc_all = safe_mean(pearson_per_sample(predictions, targets, eps=eps))
    r2_all = safe_mean(r2_per_sample(targets, predictions, eps=eps))
    direction_acc_all = safe_mean(direction_accuracy(targets, predictions, thr=direction_threshold))

    top_idx = select_topk_genes_by_abs(targets, top_k, control_expression=control_expression)
    pred_top = predictions[:, top_idx]
    true_top = targets[:, top_idx]
    mse_top = mse(pred_top, true_top)
    mae_top = mae(pred_top, true_top)
    pcc_top = safe_mean(pearson_per_sample(pred_top, true_top, eps=eps))
    r2_top = safe_mean(r2_per_sample(true_top, pred_top, eps=eps))
    direction_acc_top = safe_mean(direction_accuracy(true_top, pred_top, thr=direction_threshold))

    delta_pcc_all = float("nan")
    delta_r2_all = float("nan")
    delta_pcc_top50 = float("nan")
    delta_r2_top50 = float("nan")
    if control_expression is not None:
        control = broadcast_control_like(control_expression, n_samples=n_samples, n_genes=n_genes)
        delta_true = targets - control
        delta_pred = predictions - control
        delta_pcc_all = safe_mean(pearson_per_sample(delta_pred, delta_true, eps=eps))
        delta_r2_all = safe_mean(r2_per_sample(delta_true, delta_pred, eps=eps))
        delta_top_idx = select_topk_genes_by_abs(delta_true, top_k)
        delta_pred_top = delta_pred[:, delta_top_idx]
        delta_true_top = delta_true[:, delta_top_idx]
        delta_pcc_top50 = safe_mean(pearson_per_sample(delta_pred_top, delta_true_top, eps=eps))
        delta_r2_top50 = safe_mean(r2_per_sample(delta_true_top, delta_pred_top, eps=eps))

    return {
        "auroc": roc_auc(ny_true1, ny_pred1, mask=mask1),
        "auprc": auprc_score(ny_true1, ny_pred1, mask=mask1),
        "acc": accuracy(ny_true1, ny_pred1, mask=mask1),
        "auroc2": roc_auc(ny_true2, ny_pred2, mask=mask2),
        "auprc2": auprc_score(ny_true2, ny_pred2, mask=mask2),
        "acc2": accuracy(ny_true2, ny_pred2, mask=mask2),
        "mse_all": mse_all,
        "mae_all": mae_all,
        "pcc_all": pcc_all,
        "r2_all": r2_all,
        "direction_acc_all": direction_acc_all,
        "mse_top50": mse_top,
        "mae_top50": mae_top,
        "pcc_top50": pcc_top,
        "r2_top50": r2_top,
        "direction_acc_top50": direction_acc_top,
        "delta_pcc_all": delta_pcc_all,
        "delta_r2_all": delta_r2_all,
        "delta_pcc_top50": delta_pcc_top50,
        "delta_r2_top50": delta_r2_top50,
        "mmd": mmd_rbf(targets, predictions, rng=rng, subset_max=mmd_subset_max),
        "energy_distance": energy_distance_nd(targets, predictions, rng=rng, subset_max=energy_subset_max),
    }


class FocalLossWithAlpha(nn.Module):
    """Legacy focal loss used by the old trainer."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return alpha_t * (1.0 - pt) ** self.gamma * bce_loss


class UnfreezeCallback(pl.Callback):
    """Freeze a model layer at fit start, then unfreeze it at a chosen epoch."""

    def __init__(self, unfreeze_at_epoch: int = 10, layer_name: str = "embedding_proj") -> None:
        super().__init__()
        self.unfreeze_at_epoch = int(unfreeze_at_epoch)
        self.layer_name = layer_name

    def _get_layer(self, pl_module: "ProteinTalkLightning") -> nn.Module:
        layer = getattr(pl_module.model, self.layer_name)
        if layer is None:
            raise ValueError(f"model layer {self.layer_name!r} is None and cannot be frozen/unfrozen")
        return layer

    def on_fit_start(self, trainer, pl_module) -> None:
        layer = self._get_layer(pl_module)
        for param in layer.parameters():
            param.requires_grad = False
        rank_zero_info(f"Froze {self.layer_name} layer")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch != self.unfreeze_at_epoch:
            return
        layer = self._get_layer(pl_module)
        for param in layer.parameters():
            param.requires_grad = True
        rank_zero_info(f"Unfroze {self.layer_name} layer at epoch {trainer.current_epoch}")


class ProteinTalkLightning(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        mse_weight: float = 1.0,
        bce_weight: Optional[float] = None,
        bce_weight1: float = 1.0,
        bce_weight2: float = 1.0,
        learning_rate: float = 1e-4,
        effective_key1: str = "PRISM1st_label_total",
        effective_key2: str = "synergy",
        mask_key1: str = "sensitive_label_mask",
        mask_key2: str = "synergy_label_mask",
        task_label_key: Optional[str] = None,
        task_mask_key: Optional[str] = None,
        task_head: str = "response",
        optimizer_name: str = "adamw",
        positive_weight: Optional[float] = None,
        positive_weight1: Optional[float] = None,
        positive_weight2: Optional[float] = None,
        scheduler_name: Optional[str] = None,
        scheduler_monitor: str = "val/total_loss",
        scheduler_monitor_mode: str = "min",
        focal_loss: bool = False,
        have_mse_loss: bool = True,
        **optimizer_kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.mse_weight = mse_weight
        self.task_head = task_head.lower()
        if self.task_head not in {"response", "synergy"}:
            raise ValueError("task_head must be `response` or `synergy`")
        self.learning_rate = learning_rate
        self.effective_key1 = effective_key1
        self.effective_key2 = effective_key2
        self.mask_key1 = mask_key1
        self.mask_key2 = mask_key2
        self.task_label_key = task_label_key or (effective_key2 if self.task_head == "synergy" else effective_key1)
        self.task_mask_key = task_mask_key or (mask_key2 if self.task_head == "synergy" else mask_key1)
        self.bce_weight1 = bce_weight1
        self.bce_weight2 = bce_weight2
        self.bce_weight = float(bce_weight if bce_weight is not None else (bce_weight2 if self.task_head == "synergy" else bce_weight1))
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_monitor_mode = scheduler_monitor_mode
        self.have_mse_loss = have_mse_loss
        self.optimizer_kwargs = optimizer_kwargs
        self.focal_loss = focal_loss
        self.mse_loss_fn = nn.MSELoss(reduction="none")
        if positive_weight is not None:
            if self.task_head == "synergy" and positive_weight2 is None:
                positive_weight2 = positive_weight
            if self.task_head == "response" and positive_weight1 is None:
                positive_weight1 = positive_weight
        self.bce_loss_fn1 = self._make_bce_loss_fn(positive_weight1)
        self.bce_loss_fn2 = self._make_bce_loss_fn(positive_weight2)
        self.bce_loss_fn = self.bce_loss_fn2 if self.task_head == "synergy" else self.bce_loss_fn1
        self.validation_outputs: list[dict[str, torch.Tensor]] = []
        self.test_outputs: list[dict[str, torch.Tensor]] = []
        self.save_hyperparameters(ignore=["model"])

    def _make_bce_loss_fn(self, positive_weight: Optional[float]) -> nn.Module:
        if self.focal_loss:
            return FocalLossWithAlpha()
        if positive_weight is not None:
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(float(positive_weight)),
                reduction="none",
            )
        return nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, batch):
        return self.model(batch)

    def _select_task_logits(self, response_logits: torch.Tensor, synergy_logits: torch.Tensor) -> torch.Tensor:
        return synergy_logits if self.task_head == "synergy" else response_logits

    def _losses(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expression_pred, response_logits, synergy_logits = self(batch)
        expression_true = batch["perturb"]["expressions_hvg"].float()
        loss1 = self._compute_mse_loss(expression_pred, expression_true)
        task_logits = self._select_task_logits(response_logits, synergy_logits)
        loss2 = self._masked_bce(
            self.bce_loss_fn,
            task_logits.squeeze(-1),
            batch["perturb"][self.task_label_key].float(),
            batch["perturb"].get(self.task_mask_key),
        )
        total = self.bce_weight * loss2
        if self.have_mse_loss:
            total = total + self.mse_weight * loss1
        return total, loss1, loss2, expression_pred, response_logits, synergy_logits

    def _compute_mse_loss(self, expression_pred: torch.Tensor, expression_true: torch.Tensor) -> torch.Tensor:
        pred_flat = expression_pred.reshape(-1, expression_pred.shape[-1])
        true_flat = expression_true.reshape(-1, expression_true.shape[-1])
        mse_loss = expression_pred.new_tensor(0.0)
        if self.have_mse_loss:
            valid_mse = ~torch.isnan(true_flat)
            if valid_mse.any():
                mse_loss = self.mse_loss_fn(pred_flat[valid_mse], true_flat[valid_mse]).mean()
        return mse_loss

    def _per_sample_mse_loss(self, expression_pred: torch.Tensor, expression_true: torch.Tensor) -> torch.Tensor:
        if not self.have_mse_loss:
            return expression_pred.new_zeros(expression_pred.shape[0])
        valid = ~torch.isnan(expression_true)
        true_safe = torch.where(valid, expression_true, torch.zeros_like(expression_true))
        raw = self.mse_loss_fn(expression_pred, true_safe)
        numerator = (raw * valid.float()).sum(dim=1)
        denominator = valid.float().sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    def _per_sample_bce_loss(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_flat = logits.reshape(-1)
        label_flat = label.reshape(-1).float()
        if mask is not None:
            mask_flat = mask.reshape(-1)
            if mask_flat.shape[0] != label_flat.shape[0]:
                raise ValueError(
                    f"label mask shape {tuple(mask.shape)} does not match label shape {tuple(label.shape)}"
                )
            weights = 1.0 - mask_flat.float()
        else:
            weights = torch.ones_like(label_flat, dtype=torch.float)
        raw_loss = self.bce_loss_fn(logits_flat, label_flat)
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.squeeze()
        return raw_loss * weights, weights

    def _compute_masked_loss(
        self,
        expression_pred: torch.Tensor,
        expression_true: torch.Tensor,
        task_logits: torch.Tensor,
        task_label: torch.Tensor,
        task_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mse_loss = self._compute_mse_loss(expression_pred, expression_true)
        bce_loss = self._masked_bce(self.bce_loss_fn, task_logits, task_label, task_mask)
        return mse_loss, bce_loss

    def _masked_bce(
        self,
        loss_fn: nn.Module,
        logits: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        logits_flat = logits.reshape(-1)
        label_flat = label.reshape(-1).float()
        if mask is not None:
            mask_flat = mask.reshape(-1)
            if mask_flat.shape[0] != label_flat.shape[0]:
                raise ValueError(
                    f"label mask shape {tuple(mask.shape)} does not match label shape {tuple(label.shape)}"
                )
            weights = 1.0 - mask_flat.float()
        else:
            weights = torch.ones_like(label_flat, dtype=torch.float)
        raw_loss = loss_fn(logits_flat, label_flat)
        if raw_loss.ndim > 1:
            raw_loss = raw_loss.squeeze()
        return (raw_loss * weights).sum() / (weights.sum() + 1e-8)

    def training_step(self, batch, batch_idx):
        total, loss1, loss2, _, _, _ = self._losses(batch)
        self.log("train/loss1", loss1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss2", loss2, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mse_loss", loss1, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/bce_loss", loss2, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/total_loss", total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        total, loss1, loss2, expression_pred, logits1, logits2 = self._losses(batch)
        self.validation_outputs.append(
            self._collect_eval(batch, expression_pred, logits1, logits2, total, loss1, loss2)
        )
        return total

    def test_step(self, batch, batch_idx):
        total, loss1, loss2, expression_pred, logits1, logits2 = self._losses(batch)
        self.test_outputs.append(
            self._collect_eval(batch, expression_pred, logits1, logits2, total, loss1, loss2)
        )
        return total

    def _collect_eval(
        self,
        batch,
        expression_pred: torch.Tensor,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        total: torch.Tensor,
        loss1: torch.Tensor,
        loss2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask1 = batch["perturb"].get(self.mask_key1)
        mask2 = batch["perturb"].get(self.mask_key2)
        if mask1 is None:
            mask1 = torch.zeros_like(batch["perturb"][self.effective_key1])
        if mask2 is None:
            mask2 = torch.zeros_like(batch["perturb"][self.effective_key2])
        task_logits = self._select_task_logits(logits1, logits2).squeeze(-1)
        task_mask = batch["perturb"].get(self.task_mask_key)
        per_sample_loss1 = self._per_sample_mse_loss(
            expression_pred,
            batch["perturb"]["expressions_hvg"].float(),
        )
        per_sample_loss2_numer, per_sample_loss2_denom = self._per_sample_bce_loss(
            task_logits,
            batch["perturb"][self.task_label_key].float(),
            task_mask,
        )
        return {
            "row_index": batch["perturb"]["row_index"].detach().cpu(),
            "expression_pred": expression_pred.detach().cpu(),
            "expression_true": batch["perturb"]["expressions_hvg"].detach().cpu(),
            "control_expression": batch["control"]["expressions_hvg"].detach().cpu(),
            "prob1": torch.sigmoid(logits1.squeeze(-1)).detach().cpu(),
            "true1": batch["perturb"][self.effective_key1].detach().cpu(),
            "mask1": mask1.detach().cpu(),
            "prob2": torch.sigmoid(logits2.squeeze(-1)).detach().cpu(),
            "true2": batch["perturb"][self.effective_key2].detach().cpu(),
            "mask2": mask2.detach().cpu(),
            "total": total.detach().cpu(),
            "loss1": loss1.detach().cpu(),
            "loss2": loss2.detach().cpu(),
            "loss1_per_sample": per_sample_loss1.detach().cpu(),
            "loss2_per_sample_numer": per_sample_loss2_numer.detach().cpu(),
            "loss2_per_sample_denom": per_sample_loss2_denom.detach().cpu(),
        }

    def on_validation_epoch_end(self) -> None:
        self._finish_eval("val", self.validation_outputs)
        self.validation_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._finish_eval("test", self.test_outputs)
        self.test_outputs.clear()

    def _finish_eval(self, prefix: str, outputs: list[dict[str, torch.Tensor]]) -> None:
        if not outputs:
            return
        row_index = self._gather_eval_field(outputs, "row_index").numpy().reshape(-1)
        expression_pred = self._gather_eval_field(outputs, "expression_pred").numpy()
        expression_true = self._gather_eval_field(outputs, "expression_true").numpy()
        control_expression = self._gather_eval_field(outputs, "control_expression").numpy()
        prob1 = self._gather_eval_field(outputs, "prob1").numpy()
        true1 = self._gather_eval_field(outputs, "true1").numpy()
        mask1 = self._gather_eval_field(outputs, "mask1").numpy()
        prob2 = self._gather_eval_field(outputs, "prob2").numpy()
        true2 = self._gather_eval_field(outputs, "true2").numpy()
        mask2 = self._gather_eval_field(outputs, "mask2").numpy()
        loss1_per_sample = self._gather_eval_field(outputs, "loss1_per_sample").numpy().reshape(-1)
        loss2_per_sample_numer = self._gather_eval_field(outputs, "loss2_per_sample_numer").numpy().reshape(-1)
        loss2_per_sample_denom = self._gather_eval_field(outputs, "loss2_per_sample_denom").numpy().reshape(-1)
        if row_index.size:
            _, keep = np.unique(row_index, return_index=True)
            keep = np.sort(keep)
            expression_pred = expression_pred[keep]
            expression_true = expression_true[keep]
            control_expression = control_expression[keep]
            prob1 = prob1[keep]
            true1 = true1[keep]
            mask1 = mask1[keep]
            prob2 = prob2[keep]
            true2 = true2[keep]
            mask2 = mask2[keep]
            loss1_per_sample = loss1_per_sample[keep]
            loss2_per_sample_numer = loss2_per_sample_numer[keep]
            loss2_per_sample_denom = loss2_per_sample_denom[keep]
        loss1_value = float(np.mean(loss1_per_sample)) if loss1_per_sample.size and self.have_mse_loss else 0.0
        loss2_denom = float(np.sum(loss2_per_sample_denom))
        loss2_value = float(np.sum(loss2_per_sample_numer) / (loss2_denom + 1e-8)) if loss2_denom > 0 else 0.0
        total_loss_value = self.bce_weight * loss2_value
        if self.have_mse_loss:
            total_loss_value += self.mse_weight * loss1_value
        for log_key, metric_value in (
            ("total_loss", total_loss_value),
            ("loss1", loss1_value),
            ("loss2", loss2_value),
            ("mse_loss", loss1_value),
            ("bce_loss", loss2_value),
        ):
            self.log(
                f"{prefix}/{log_key}",
                self._metric_tensor(metric_value),
                on_epoch=True,
                prog_bar=(log_key in {"total_loss", "loss1", "loss2"}),
                sync_dist=False,
            )
        metrics = compute_validation_metrics(
            predictions=expression_pred,
            targets=expression_true,
            ny_pred1=prob1,
            ny_true1=true1,
            mask1=mask1,
            ny_pred2=prob2,
            ny_true2=true2,
            mask2=mask2,
            control_expression=control_expression,
        )
        expression_metric_names = {
            "mse_all",
            "mae_all",
            "pcc_all",
            "r2_all",
            "direction_acc_all",
            "mse_top50",
            "mae_top50",
            "pcc_top50",
            "r2_top50",
            "direction_acc_top50",
            "delta_pcc_all",
            "delta_r2_all",
            "delta_pcc_top50",
            "delta_r2_top50",
            "mmd",
            "energy_distance",
        }
        for metric_name in expression_metric_names:
            self.log(
                f"{prefix}/{metric_name}",
                self._metric_tensor(metrics[metric_name]),
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=True,
            )

        response_metrics = {
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "acc": metrics["acc"],
        }
        synergy_metrics = {
            "auroc": metrics["auroc2"],
            "auprc": metrics["auprc2"],
            "acc": metrics["acc2"],
        }
        active_metrics = synergy_metrics if self.task_head == "synergy" else response_metrics
        for metric_name, metric_value in active_metrics.items():
            value = self._metric_tensor(metric_value)
            self.log(
                f"{prefix}/{metric_name}",
                value,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=False,
            )
            self.log(
                f"{prefix}/task_{metric_name}",
                value,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                rank_zero_only=False,
            )
        for metric_name, metric_value in response_metrics.items():
            self.log(
                f"{prefix}/response_{metric_name}",
                self._metric_tensor(metric_value),
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                rank_zero_only=True,
            )
        for metric_name, metric_value in synergy_metrics.items():
            self.log(
                f"{prefix}/synergy_{metric_name}",
                self._metric_tensor(metric_value),
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
                rank_zero_only=True,
            )

    def _metric_tensor(self, value: float) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

    def _gather_eval_field(self, outputs: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        local = torch.cat([item[key] for item in outputs], dim=0).to(self.device)
        if torch.is_floating_point(local):
            local = local.float()
        if getattr(self.trainer, "world_size", 1) > 1:
            gathered = self.all_gather(local)
            local = gathered.reshape(-1, *local.shape[1:])
        return local.detach().cpu()

    def configure_optimizers(self):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
        elif optimizer_name == "adamw_fused" or optimizer_name.startswith("adamw_fused_"):
            special_layer = getattr(self.model, "embedding_proj", None)
            if special_layer is None:
                raise ValueError("adamw_fused requires model.embedding_proj; use the PDI hetero graph model")
            special_params = list(special_layer.parameters())
            special_param_ids = {id(param) for param in special_params}
            other_params = [param for param in self.parameters() if id(param) not in special_param_ids]
            if optimizer_name == "adamw_fused":
                special_lr = self.learning_rate / 10.0
            else:
                special_lr = self.learning_rate * float(optimizer_name.rsplit("_", 1)[-1])
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": self.learning_rate},
                    {"params": special_params, "lr": special_lr},
                ],
                lr=self.learning_rate,
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f"unsupported optimizer {self.optimizer_name!r}")

        if not self.scheduler_name:
            return optimizer
        scheduler_name = self.scheduler_name.lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.01)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_monitor_mode,
                patience=10,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": self.scheduler_monitor, "interval": "epoch"},
            }
        if scheduler_name == "cosine_warmup":
            warmup_epochs = 25
            total_epochs = 250
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        raise ValueError(f"unsupported scheduler {self.scheduler_name!r}")


def binary_metrics(prob: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    keep = (mask == 0) & np.isfinite(prob) & np.isfinite(true)
    prob = prob[keep]
    true = true[keep]
    result = {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "auprc_baseline": float("nan"),
        "nauprc": float("nan"),
        "acc": float("nan"),
        "valid_count": int(len(true)),
        "positive_count": int(np.sum(true == 1)) if len(true) else 0,
        "negative_count": int(np.sum(true == 0)) if len(true) else 0,
    }
    if len(true) == 0:
        return result
    baseline = float(np.mean(true == 1))
    result["auprc_baseline"] = baseline
    result["acc"] = accuracy(true, prob)
    if len(np.unique(true)) >= 2:
        auprc = float(average_precision_score(true, prob))
        result["auroc"] = float(roc_auc_score(true, prob))
        result["auprc"] = auprc
        result["nauprc"] = auprc / baseline if baseline > 0 else float("nan")
    return result
