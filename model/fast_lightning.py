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
        return {"auroc": float("nan"), "auprc": float("nan"), "acc": float("nan"), "count": 0.0}
    y_hat = (y_prob >= 0.5).astype(np.float64)
    acc = float((y_hat == y_true).mean())
    if np.unique(y_true).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "acc": acc, "count": float(y_true.size)}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
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
        label_smoothing: float = 0.0,
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
        self.label_smoothing = float(label_smoothing)
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        pos_weight_value = 1.0 if positive_weight is None else float(positive_weight)
        self.register_buffer("positive_weight", torch.tensor(pos_weight_value, dtype=torch.float32))
        self.validation_outputs: list[dict[str, torch.Tensor]] = []
        self.test_outputs: list[dict[str, torch.Tensor]] = []
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(batch)

    def _active_label_and_mask(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self.task_head == "synergy":
            return batch["label2"].float(), batch["mask2"].float(), "synergy"
        return batch["label1"].float(), batch["mask1"].float(), "response"

    def _losses(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expression_pred, response_logits, synergy_logits = self(batch)
        expression_true = batch["perturb_expression"].float()
        label, mask, _ = self._active_label_and_mask(batch)
        loss1 = self._mse_loss(expression_pred, expression_true, mask)
        task_logits = synergy_logits.squeeze(-1) if self.task_head == "synergy" else response_logits.squeeze(-1)
        inactive_logits = response_logits if self.task_head == "synergy" else synergy_logits
        loss2 = self._masked_bce(task_logits, label, mask)
        total = self.bce_weight * loss2
        if self.have_mse_loss:
            total = total + self.mse_weight * loss1
        total = total + inactive_logits.sum() * 0.0 + expression_pred.sum() * 0.0
        return total, loss1, loss2, expression_pred, response_logits, synergy_logits

    def _mse_loss(
        self,
        expression_pred: torch.Tensor,
        expression_true: torch.Tensor,
        active_label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.have_mse_loss:
            return expression_pred.new_tensor(0.0)
        pred = expression_pred
        true = expression_true
        if self.mse_gene_subsample > 0 and self.mse_gene_subsample < pred.shape[1] and self.training:
            index = torch.randperm(pred.shape[1], device=pred.device)[: self.mse_gene_subsample]
            pred = pred[:, index]
            true = true[:, index]
        valid = torch.isfinite(true)
        if not valid.any():
            return expression_pred.new_tensor(0.0)
        true_safe = torch.where(valid, true, torch.zeros_like(true))
        raw = (pred - true_safe).pow(2)
        if self.mse_inactive_label_weight == 1.0 or active_label_mask is None:
            return (raw * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        per_sample_valid = valid.float().sum(dim=1).clamp_min(1.0)
        per_sample_loss = (raw * valid.float()).sum(dim=1) / per_sample_valid
        active = (active_label_mask.reshape(-1) < 0.5).to(per_sample_loss.dtype)
        sample_weight = torch.where(
            active > 0.5,
            torch.ones_like(per_sample_loss),
            per_sample_loss.new_full(per_sample_loss.shape, self.mse_inactive_label_weight),
        )
        return (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total, loss1, loss2, _, _, _ = self._losses(batch)
        self.log("train/total_loss", total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss1", loss1, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss2", loss2, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total, loss1, loss2, _, response_logits, synergy_logits = self._losses(batch)
        self._log_eval_losses("val", total, loss1, loss2)
        self.validation_outputs.append(self._collect_eval(batch, response_logits, synergy_logits))
        return total

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        total, loss1, loss2, _, response_logits, synergy_logits = self._losses(batch)
        self._log_eval_losses("test", total, loss1, loss2)
        self.test_outputs.append(self._collect_eval(batch, response_logits, synergy_logits))
        return total

    def _log_eval_losses(self, prefix: str, total: torch.Tensor, loss1: torch.Tensor, loss2: torch.Tensor) -> None:
        self.log(f"{prefix}/total_loss", total, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/loss1", loss1, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}/loss2", loss2, on_epoch=True, prog_bar=True, sync_dist=True)

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
