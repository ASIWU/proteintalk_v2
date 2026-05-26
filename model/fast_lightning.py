#!/usr/bin/env python3
"""Lightning wrapper for the fast ProteinTalk model."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if mask is not None:
        keep = np.asarray(mask, dtype=np.float64).reshape(-1) < 0.5
        y_true = y_true[keep]
        y_prob = y_prob[keep]
    if y_true.size == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "auprc_baseline": float("nan"),
            "nauprc": float("nan"),
            "acc": float("nan"),
            "count": 0.0,
        }
    y_hat = (y_prob >= 0.5).astype(np.float64)
    acc = float((y_hat == y_true).mean())
    baseline = float(np.mean(y_true == 1))
    if np.unique(y_true).size < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "auprc_baseline": baseline,
            "nauprc": float("nan"),
            "acc": acc,
            "count": float(y_true.size),
        }
    auprc = float(average_precision_score(y_true, y_prob))
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": auprc,
        "auprc_baseline": baseline,
        "nauprc": auprc / baseline if baseline > 0 else float("nan"),
        "acc": acc,
        "count": float(y_true.size),
    }


class FastProteinTalkLightning(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        task_head: str,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        mse_weight: float = 0.25,
        bce_weight: float = 1.0,
        positive_weight: Optional[float] = None,
        have_mse_loss: bool = True,
        mse_inactive_label_weight: float = 1.0,
        optimizer_name: str = "adamw",
        scheduler_name: str | None = "cosine",
        max_epochs: int = 50,
        mse_gene_subsample: int = 0,
        mse_gene_weights: torch.Tensor | None = None,
        mse_target_weight_matrix: np.ndarray | torch.Tensor | None = None,
        mse_weight_schedule: str = "constant",
        mse_decay_start_epoch_frac: float = 0.4,
        mse_final_weight_multiplier: float = 1.0,
        label_smoothing: float = 0.0,
        aux_covariate_loss_weight: float = 0.0,
        aux_covariate_indices: list[int] | None = None,
        aux_covariate_label_smoothing: float = 0.0,
        aux_covariate_contrastive_weight: float = 0.0,
        aux_covariate_contrastive_indices: list[int] | None = None,
        aux_covariate_contrastive_temperature: float = 0.2,
        ranking_loss_weight: float = 0.0,
        ranking_loss_margin: float = 0.0,
        ranking_loss_group_index: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.task_head = task_head.lower()
        if self.task_head not in {"response", "synergy"}:
            raise ValueError("task_head must be response or synergy")
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.mse_weight = float(mse_weight)
        self.bce_weight = float(bce_weight)
        self.have_mse_loss = bool(have_mse_loss)
        self.mse_inactive_label_weight = float(mse_inactive_label_weight)
        if self.mse_inactive_label_weight < 0.0:
            raise ValueError("mse_inactive_label_weight must be non-negative")
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.max_epochs = int(max_epochs)
        self.mse_gene_subsample = int(mse_gene_subsample)
        if mse_gene_weights is None:
            mse_gene_weights = torch.empty(0, dtype=torch.float32)
        self.register_buffer("mse_gene_weights", mse_gene_weights.float(), persistent=False)
        self._init_mse_target_weights(mse_target_weight_matrix)
        self.mse_weight_schedule = str(mse_weight_schedule).lower()
        if self.mse_weight_schedule not in {"constant", "warmup_decay"}:
            raise ValueError("mse_weight_schedule must be constant or warmup_decay")
        self.mse_decay_start_epoch_frac = float(mse_decay_start_epoch_frac)
        if not 0.0 <= self.mse_decay_start_epoch_frac <= 1.0:
            raise ValueError("mse_decay_start_epoch_frac must be in [0, 1]")
        self.mse_final_weight_multiplier = float(mse_final_weight_multiplier)
        if self.mse_final_weight_multiplier < 0.0:
            raise ValueError("mse_final_weight_multiplier must be non-negative")
        self.label_smoothing = float(label_smoothing)
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.aux_covariate_loss_weight = float(aux_covariate_loss_weight)
        if self.aux_covariate_loss_weight < 0.0:
            raise ValueError("aux_covariate_loss_weight must be non-negative")
        self.aux_covariate_indices = list(aux_covariate_indices or [])
        self.aux_covariate_label_smoothing = float(aux_covariate_label_smoothing)
        if not 0.0 <= self.aux_covariate_label_smoothing < 1.0:
            raise ValueError("aux_covariate_label_smoothing must be in [0, 1)")
        self.aux_covariate_contrastive_weight = float(aux_covariate_contrastive_weight)
        if self.aux_covariate_contrastive_weight < 0.0:
            raise ValueError("aux_covariate_contrastive_weight must be non-negative")
        self.aux_covariate_contrastive_indices = list(aux_covariate_contrastive_indices or [])
        self.aux_covariate_contrastive_temperature = float(aux_covariate_contrastive_temperature)
        if self.aux_covariate_contrastive_temperature <= 0.0:
            raise ValueError("aux_covariate_contrastive_temperature must be positive")
        self.ranking_loss_weight = float(ranking_loss_weight)
        if self.ranking_loss_weight < 0.0:
            raise ValueError("ranking_loss_weight must be non-negative")
        self.ranking_loss_margin = float(ranking_loss_margin)
        self.ranking_loss_group_index = None if ranking_loss_group_index is None else int(ranking_loss_group_index)
        pos_weight_value = 1.0 if positive_weight is None else float(positive_weight)
        self.register_buffer("positive_weight", torch.tensor(pos_weight_value, dtype=torch.float32))
        self.validation_outputs: list[dict[str, torch.Tensor]] = []
        self.test_outputs: list[dict[str, torch.Tensor]] = []
        self.save_hyperparameters(ignore=["model", "mse_gene_weights", "mse_target_weight_matrix"])

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(batch)

    def _init_mse_target_weights(self, weight_matrix: np.ndarray | torch.Tensor | None) -> None:
        self.mse_target_enabled = weight_matrix is not None
        if weight_matrix is None:
            self.mse_target_drug_count = 0
            self.register_buffer("mse_target_indices", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("mse_target_values", torch.empty(0), persistent=False)
            self.register_buffer("mse_target_dense_weights", torch.empty(0), persistent=False)
            return

        matrix = np.asarray(weight_matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"mse_target_weight_matrix must be 2D; got {matrix.shape}")
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        matrix = np.clip(matrix, a_min=0.0, a_max=None)
        self.mse_target_drug_count = int(matrix.shape[0])
        positive_counts = np.count_nonzero(matrix > 0.0, axis=1)
        max_positive_count = int(positive_counts.max(initial=0))
        if max_positive_count == 0:
            self.mse_target_enabled = False
            self.register_buffer("mse_target_indices", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("mse_target_values", torch.empty(0), persistent=False)
            self.register_buffer("mse_target_dense_weights", torch.empty(0), persistent=False)
            return
        if 0 < max_positive_count < matrix.shape[1]:
            top_idx = np.argpartition(-matrix, kth=max_positive_count - 1, axis=1)[:, :max_positive_count]
            top_values = np.take_along_axis(matrix, top_idx, axis=1)
            top_values = np.where(top_values > 0.0, top_values, 0.0)
            self.register_buffer(
                "mse_target_indices",
                torch.tensor(top_idx.astype(np.int64, copy=False), dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "mse_target_values",
                torch.tensor(top_values.astype(np.float16, copy=False), dtype=torch.float16),
                persistent=False,
            )
            self.register_buffer("mse_target_dense_weights", torch.empty(0), persistent=False)
        else:
            self.register_buffer(
                "mse_target_dense_weights",
                torch.tensor(matrix.astype(np.float16, copy=False), dtype=torch.float16),
                persistent=False,
            )
            self.register_buffer("mse_target_indices", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("mse_target_values", torch.empty(0), persistent=False)

    def _active_label_and_mask(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self.task_head == "synergy":
            return batch["label2"].float(), batch["mask2"].float(), "synergy"
        return batch["label1"].float(), batch["mask1"].float(), "response"

    def _losses(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        outputs = self(batch)
        expression_pred, response_logits, synergy_logits = outputs[:3]
        aux_outputs = outputs[3] if len(outputs) > 3 else []
        aux_features = outputs[4] if len(outputs) > 4 else None
        expression_true = batch["perturb_expression"].float()
        label, mask, _ = self._active_label_and_mask(batch)
        loss1 = self._mse_loss(expression_pred, expression_true, mask, drug_indices=batch.get("drug_indices"))
        task_logits = synergy_logits.squeeze(-1) if self.task_head == "synergy" else response_logits.squeeze(-1)
        inactive_logits = response_logits if self.task_head == "synergy" else synergy_logits
        loss2 = self._masked_bce(task_logits, label, mask)
        aux_loss = self._aux_covariate_loss(batch, aux_outputs)
        contrastive_loss = self._aux_covariate_contrastive_loss(batch, aux_features)
        ranking_loss = self._ranking_loss(task_logits, label, mask, batch)
        total = self.bce_weight * loss2
        if self.have_mse_loss:
            total = total + self._effective_mse_weight() * loss1
        if self.aux_covariate_loss_weight > 0.0:
            total = total + self.aux_covariate_loss_weight * aux_loss
        if self.aux_covariate_contrastive_weight > 0.0:
            total = total + self.aux_covariate_contrastive_weight * contrastive_loss
        if self.ranking_loss_weight > 0.0:
            total = total + self.ranking_loss_weight * ranking_loss
        total = total + inactive_logits.sum() * 0.0 + expression_pred.sum() * 0.0
        return (
            total,
            loss1,
            loss2,
            aux_loss,
            contrastive_loss,
            ranking_loss,
            expression_pred,
            response_logits,
            synergy_logits,
        )

    def _aux_covariate_loss(self, batch: dict[str, torch.Tensor], aux_outputs: list[torch.Tensor]) -> torch.Tensor:
        if self.aux_covariate_loss_weight <= 0.0 or not self.aux_covariate_indices or not aux_outputs:
            return batch["control_expression"].new_tensor(0.0)
        covariates = batch.get("raw_covariates", batch["covariates"]).long()
        losses = []
        for logits, covariate_index in zip(aux_outputs, self.aux_covariate_indices, strict=True):
            target = covariates[:, int(covariate_index)].clamp(min=0, max=logits.shape[-1] - 1).to(logits.device)
            losses.append(
                F.cross_entropy(
                    logits,
                    target,
                    label_smoothing=self.aux_covariate_label_smoothing,
                )
            )
        if not losses:
            return batch["control_expression"].new_tensor(0.0)
        return torch.stack(losses).mean()

    def _aux_covariate_contrastive_loss(
        self,
        batch: dict[str, torch.Tensor],
        features: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self.aux_covariate_contrastive_weight <= 0.0
            or not self.aux_covariate_contrastive_indices
            or features is None
        ):
            return batch["control_expression"].new_tensor(0.0)
        covariates = batch.get("raw_covariates", batch["covariates"]).long().to(features.device)
        features = F.normalize(features.float(), dim=-1)
        losses = []
        for covariate_index in self.aux_covariate_contrastive_indices:
            if int(covariate_index) >= covariates.shape[1]:
                continue
            losses.append(self._supervised_contrastive_loss(features, covariates[:, int(covariate_index)]))
        if not losses:
            return batch["control_expression"].new_tensor(0.0)
        return torch.stack(losses).mean()

    def _supervised_contrastive_loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if features.shape[0] < 2:
            return features.new_tensor(0.0)
        logits = features @ features.t()
        logits = logits / self.aux_covariate_contrastive_temperature
        eye = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
        valid_pair = ~eye
        same = target.reshape(-1, 1).eq(target.reshape(1, -1))
        positive = same & valid_pair
        positive_count = positive.sum(dim=1)
        valid_anchor = positive_count > 0
        if not valid_anchor.any():
            return features.new_tensor(0.0)
        masked_logits = logits.masked_fill(~valid_pair, -torch.inf)
        log_denominator = torch.logsumexp(masked_logits, dim=1)
        log_prob = logits - log_denominator.unsqueeze(1)
        per_anchor = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive_count.clamp_min(1))
        return per_anchor[valid_anchor].mean()

    def _effective_mse_weight(self) -> float:
        if self.mse_weight_schedule == "constant" or self.max_epochs <= 1:
            return float(self.mse_weight)
        progress = float(self.current_epoch) / float(max(1, self.max_epochs - 1))
        if progress <= self.mse_decay_start_epoch_frac:
            multiplier = 1.0
        else:
            denom = max(1e-8, 1.0 - self.mse_decay_start_epoch_frac)
            ratio = min(1.0, max(0.0, (progress - self.mse_decay_start_epoch_frac) / denom))
            multiplier = 1.0 + ratio * (self.mse_final_weight_multiplier - 1.0)
        return float(self.mse_weight) * float(multiplier)

    def _mse_loss(
        self,
        expression_pred: torch.Tensor,
        expression_true: torch.Tensor,
        active_label_mask: torch.Tensor | None = None,
        drug_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.have_mse_loss:
            return expression_pred.new_tensor(0.0)
        pred = expression_pred
        true = expression_true
        if self.mse_target_enabled and drug_indices is not None and self.mse_gene_subsample <= 0:
            targeted_loss = self._mse_target_per_sample_loss(pred, true, drug_indices)
            if targeted_loss is not None:
                return self._finish_mse_per_sample_loss(targeted_loss, active_label_mask)
        if self.mse_gene_subsample > 0 and self.mse_gene_subsample < pred.shape[1] and self.training:
            index = torch.randperm(pred.shape[1], device=pred.device)[: self.mse_gene_subsample]
            per_sample_loss = self._full_mse_per_sample_loss(pred[:, index], true[:, index], gene_index=index)
        else:
            per_sample_loss = self._full_mse_per_sample_loss(pred, true)
        if per_sample_loss is None:
            return expression_pred.new_tensor(0.0)
        return self._finish_mse_per_sample_loss(per_sample_loss, active_label_mask)

    def _full_mse_per_sample_loss(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        *,
        gene_index: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        gene_weights = self._mse_gene_weights_for(pred.shape[1], pred.device, index=gene_index)
        valid = torch.isfinite(true)
        if not valid.any():
            return None
        true_safe = torch.where(valid, true, torch.zeros_like(true))
        raw = (pred - true_safe).pow(2)
        valid_weight = valid.float()
        if gene_weights is not None:
            valid_weight = valid_weight * gene_weights.reshape(1, -1)
        per_sample_valid = valid_weight.sum(dim=1).clamp_min(1.0)
        return (raw * valid_weight).sum(dim=1) / per_sample_valid

    def _mse_target_per_sample_loss(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        drug_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.mse_target_indices.numel() > 0:
            return self._sparse_mse_target_per_sample_loss(pred, true, drug_indices)
        if self.mse_target_dense_weights.numel() > 0:
            return self._dense_mse_target_per_sample_loss(pred, true, drug_indices)
        return None

    def _normalized_drug_indices(self, drug_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        indices = drug_indices.to(device=device, dtype=torch.long)
        if indices.ndim == 1:
            indices = torch.stack([indices, indices], dim=1)
        if indices.shape[1] == 1:
            indices = indices.expand(-1, 2)
        return indices[:, :2].clamp(min=0, max=max(0, self.mse_target_drug_count - 1))

    def _sparse_mse_target_per_sample_loss(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        drug_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        indices = self._normalized_drug_indices(drug_indices, pred.device)
        gene_indices = self.mse_target_indices[indices]
        weights = self.mse_target_values[indices].to(device=pred.device, dtype=pred.dtype)
        expanded_pred = pred.unsqueeze(1).expand(-1, 2, -1)
        expanded_true = true.unsqueeze(1).expand(-1, 2, -1)
        selected_pred = torch.gather(expanded_pred, dim=2, index=gene_indices)
        selected_true = torch.gather(expanded_true, dim=2, index=gene_indices)
        valid = torch.isfinite(selected_true) & (weights > 0.0)
        if not valid.any():
            return self._full_mse_per_sample_loss(pred, true)
        selected_true = torch.where(valid, selected_true, torch.zeros_like(selected_true))
        target_weight = weights * valid.to(weights.dtype)
        gene_weights = self._mse_gene_weights_for(pred.shape[1], pred.device)
        if gene_weights is not None:
            target_weight = target_weight * gene_weights[gene_indices].to(target_weight.dtype)
        raw = (selected_pred - selected_true).pow(2)
        denom = target_weight.sum(dim=2)
        slot_loss = (raw * target_weight).sum(dim=2) / denom.clamp_min(1e-8)
        valid_slot = denom > 0.0
        valid_count = valid_slot.sum(dim=1)
        per_sample = (slot_loss * valid_slot.to(slot_loss.dtype)).sum(dim=1) / valid_count.clamp_min(1).to(slot_loss.dtype)
        missing = valid_count == 0
        if missing.any():
            fallback = self._full_mse_per_sample_loss(pred, true)
            if fallback is not None:
                per_sample = torch.where(missing, fallback, per_sample)
        return per_sample

    def _dense_mse_target_per_sample_loss(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        drug_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        indices = self._normalized_drug_indices(drug_indices, pred.device)
        target_weight = self.mse_target_dense_weights[indices].to(device=pred.device, dtype=pred.dtype).mean(dim=1)
        valid = torch.isfinite(true) & (target_weight > 0.0)
        if not valid.any():
            return self._full_mse_per_sample_loss(pred, true)
        true_safe = torch.where(valid, true, torch.zeros_like(true))
        valid_weight = target_weight * valid.to(target_weight.dtype)
        gene_weights = self._mse_gene_weights_for(pred.shape[1], pred.device)
        if gene_weights is not None:
            valid_weight = valid_weight * gene_weights.reshape(1, -1)
        raw = (pred - true_safe).pow(2)
        denom = valid_weight.sum(dim=1)
        per_sample = (raw * valid_weight).sum(dim=1) / denom.clamp_min(1e-8)
        missing = denom <= 0.0
        if missing.any():
            fallback = self._full_mse_per_sample_loss(pred, true)
            if fallback is not None:
                per_sample = torch.where(missing, fallback, per_sample)
        return per_sample

    def _finish_mse_per_sample_loss(
        self,
        per_sample_loss: torch.Tensor,
        active_label_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mse_inactive_label_weight == 1.0 or active_label_mask is None:
            return per_sample_loss.mean()
        active = (active_label_mask.reshape(-1) < 0.5).to(per_sample_loss.dtype)
        sample_weight = torch.where(
            active > 0.5,
            torch.ones_like(per_sample_loss),
            per_sample_loss.new_full(per_sample_loss.shape, self.mse_inactive_label_weight),
        )
        return (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

    def _mse_gene_weights_for(
        self,
        gene_count: int,
        device: torch.device,
        *,
        index: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.mse_gene_weights.numel() == 0:
            return None
        weights = self.mse_gene_weights.to(device=device, dtype=torch.float32)
        if index is not None:
            return weights[index]
        if weights.numel() != gene_count:
            return None
        return weights

    def _masked_bce(self, logits: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = 1.0 - mask.float()
        label_flat = label.reshape(-1).float()
        if self.label_smoothing > 0.0:
            label_flat = label_flat * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        raw = F.binary_cross_entropy_with_logits(
            logits.reshape(-1),
            label_flat,
            reduction="none",
            pos_weight=self.positive_weight.to(logits.device),
        )
        return (raw * weights.reshape(-1)).sum() / weights.sum().clamp_min(1.0)

    def _ranking_loss(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.ranking_loss_weight <= 0.0:
            return logits.new_tensor(0.0)
        logits = logits.reshape(-1)
        label = label.reshape(-1).float().to(logits.device)
        active = (mask.reshape(-1).to(logits.device) < 0.5)
        if active.sum() < 2:
            return logits.new_tensor(0.0)
        if self.ranking_loss_group_index is None:
            groups = torch.zeros_like(label, dtype=torch.long)
        else:
            covariates = batch.get("raw_covariates", batch["covariates"]).long().to(logits.device)
            if self.ranking_loss_group_index >= covariates.shape[1]:
                groups = torch.zeros_like(label, dtype=torch.long)
            else:
                groups = covariates[:, self.ranking_loss_group_index]
        losses = []
        for group_value in torch.unique(groups[active]):
            keep = active & groups.eq(group_value)
            pos_logits = logits[keep & (label >= 0.5)]
            neg_logits = logits[keep & (label < 0.5)]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue
            diff = pos_logits.reshape(-1, 1) - neg_logits.reshape(1, -1)
            losses.append(F.softplus(self.ranking_loss_margin - diff).mean())
        if not losses:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total, loss1, loss2, aux_loss, contrastive_loss, ranking_loss, _, _, _ = self._losses(batch)
        self.log("train/total_loss", total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss1", loss1, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss2", loss2, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.have_mse_loss:
            self.log(
                "train/effective_mse_weight",
                self._metric_tensor(self._effective_mse_weight()),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        if self.aux_covariate_loss_weight > 0.0:
            self.log("train/aux_covariate_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.aux_covariate_contrastive_weight > 0.0:
            self.log("train/aux_covariate_contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.ranking_loss_weight > 0.0:
            self.log("train/ranking_loss", ranking_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return total

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total, loss1, loss2, aux_loss, contrastive_loss, ranking_loss, _, response_logits, synergy_logits = self._losses(batch)
        self._log_eval_losses("val", total, loss1, loss2, aux_loss, contrastive_loss, ranking_loss)
        self.validation_outputs.append(self._collect_eval(batch, response_logits, synergy_logits))
        return total

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total, loss1, loss2, aux_loss, contrastive_loss, ranking_loss, _, response_logits, synergy_logits = self._losses(batch)
        self._log_eval_losses("test", total, loss1, loss2, aux_loss, contrastive_loss, ranking_loss)
        self.test_outputs.append(self._collect_eval(batch, response_logits, synergy_logits))
        return total

    def _log_eval_losses(
        self,
        prefix: str,
        total: torch.Tensor,
        loss1: torch.Tensor,
        loss2: torch.Tensor,
        aux_loss: torch.Tensor,
        contrastive_loss: torch.Tensor,
        ranking_loss: torch.Tensor,
    ) -> None:
        self.log(f"{prefix}/total_loss", total, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/loss1", loss1, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}/loss2", loss2, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.aux_covariate_loss_weight > 0.0:
            self.log(f"{prefix}/aux_covariate_loss", aux_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.aux_covariate_contrastive_weight > 0.0:
            self.log(f"{prefix}/aux_covariate_contrastive_loss", contrastive_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.ranking_loss_weight > 0.0:
            self.log(f"{prefix}/ranking_loss", ranking_loss, on_epoch=True, prog_bar=False, sync_dist=True)

    def _collect_eval(
        self,
        batch: dict[str, torch.Tensor],
        response_logits: torch.Tensor,
        synergy_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "prob1": torch.sigmoid(response_logits.squeeze(-1)).float().detach(),
            "true1": batch["label1"].detach(),
            "mask1": batch["mask1"].detach(),
            "prob2": torch.sigmoid(synergy_logits.squeeze(-1)).float().detach(),
            "true2": batch["label2"].detach(),
            "mask2": batch["mask2"].detach(),
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
        prob1 = self._gather_eval_field(outputs, "prob1").numpy()
        true1 = self._gather_eval_field(outputs, "true1").numpy()
        mask1 = self._gather_eval_field(outputs, "mask1").numpy()
        prob2 = self._gather_eval_field(outputs, "prob2").numpy()
        true2 = self._gather_eval_field(outputs, "true2").numpy()
        mask2 = self._gather_eval_field(outputs, "mask2").numpy()
        response = binary_metrics(true1, prob1, mask1)
        synergy = binary_metrics(true2, prob2, mask2)
        active = synergy if self.task_head == "synergy" else response
        for metric_name, metric_value in active.items():
            self.log(
                f"{prefix}/task_{metric_name}",
                self._metric_tensor(metric_value),
                on_epoch=True,
                prog_bar=metric_name in {"auprc", "auroc", "acc"},
                sync_dist=True,
            )
            if metric_name != "count":
                self.log(
                    f"{prefix}/{metric_name}",
                    self._metric_tensor(metric_value),
                    on_epoch=True,
                    prog_bar=metric_name in {"auprc", "auroc", "acc"},
                    sync_dist=True,
                )
        for namespace, metrics in (("response", response), ("synergy", synergy)):
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"{prefix}/{namespace}_{metric_name}",
                    self._metric_tensor(metric_value),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

    def _gather_eval_field(self, outputs: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        local = torch.cat([item[key].reshape(-1) for item in outputs], dim=0).to(self.device)
        if torch.is_floating_point(local):
            local = local.float()
        if getattr(self.trainer, "world_size", 1) > 1:
            gathered = self.all_gather(local)
            local = gathered.reshape(-1)
        return local.detach().cpu()

    def _metric_tensor(self, value: float) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

    def configure_optimizers(self):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"unsupported optimizer: {self.optimizer_name!r}")
        if not self.scheduler_name:
            return optimizer
        scheduler_name = self.scheduler_name.lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.max_epochs),
                eta_min=self.learning_rate * 0.05,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/task_auprc", "interval": "epoch"},
            }
        raise ValueError(f"unsupported scheduler: {self.scheduler_name!r}")
