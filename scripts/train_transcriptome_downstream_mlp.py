#!/usr/bin/env python3
"""Train a downstream binary MLP on generated transcriptome predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scipy import sparse as scipy_sparse
except ImportError:  # pragma: no cover - anndata normally brings scipy.
    scipy_sparse = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION_ROOT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/maoxinjie/AIVC/"
    "ptv3_single_nonablation/output/official_fixed_residual_20260605"
)
DEFAULT_DRUG_FP_PATH = REPO_ROOT / "data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl"
SPLIT_PREFIXES = {
    "exp01": "split_vcbench_exp01_pert_stratified_fold",
    "exp02": "split_vcbench_exp02_cell_type_fold",
    "exp03": "split_vcbench_exp03_cell_fold",
}
LABEL_MAP = {"non-responsive": 0, "sensitive": 1}
OBS_EXPORT_COLUMNS = (
    "sample_id",
    "drug_id",
    "drug_name",
    "drug_smiles_primary",
    "cell",
    "cell_type",
    "pert_time",
    "pert_dose1",
    "pert_dose2",
    "PRISM1st_label_total",
)


@dataclass
class SplitData:
    part: str
    path: Path
    source_paths: tuple[Path, ...]
    expr: np.ndarray
    drug: np.ndarray
    y: np.ndarray
    obs: pd.DataFrame
    dropped_unlabelled: int
    fingerprint_stats: dict[str, int]


class TranscriptomeDrugMLP(nn.Module):
    def __init__(
        self,
        *,
        transcriptome_dim: int,
        drug_dim: int,
        hidden_dim: int,
        drug_hidden_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        act = activation_layer(activation)
        self.transcriptome_tower = nn.Sequential(
            nn.LayerNorm(transcriptome_dim),
            nn.Linear(transcriptome_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
        )
        self.drug_tower = nn.Sequential(
            nn.Linear(drug_dim, drug_hidden_dim),
            nn.LayerNorm(drug_hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + drug_hidden_dim, fusion_hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, transcriptome: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        transcriptome_features = self.transcriptome_tower(transcriptome)
        drug_features = self.drug_tower(drug)
        return self.fusion_head(torch.cat([transcriptome_features, drug_features], dim=-1)).squeeze(-1)


class MorganLookup:
    def __init__(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a dict payload")
        matrix = np.asarray(payload.get("embedding_matrix"), dtype=np.float32)
        item_to_index = payload.get("item_to_index")
        if matrix.ndim != 2 or not isinstance(item_to_index, dict):
            raise ValueError(f"{path} missing embedding_matrix/item_to_index")
        self.path = path
        self.matrix = matrix
        self.item_to_index = {str(key): int(value) for key, value in item_to_index.items()}
        self.radius = int(payload.get("radius", 2))
        self.n_bits = int(payload.get("n_bits", matrix.shape[1]))
        self._fallback_cache: dict[str, np.ndarray] = {}
        self._generator: Any | None = None

    def vector_for(self, drug_id: object, smiles: object) -> tuple[np.ndarray, bool]:
        key = normalize_id(drug_id)
        if key in self.item_to_index:
            return self.matrix[self.item_to_index[key]], True
        smiles_key = normalize_smiles(smiles)
        if smiles_key not in self._fallback_cache:
            self._fallback_cache[smiles_key] = self._fingerprint_from_smiles(smiles_key)
        return self._fallback_cache[smiles_key], False

    def _fingerprint_from_smiles(self, smiles: str) -> np.ndarray:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        if self._generator is None:
            self._generator = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles("")
        if mol is None:
            raise RuntimeError("RDKit failed to create fallback empty molecule")
        fingerprint = self._generator.GetFingerprint(mol)
        vector = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, vector)
        return vector


def activation_layer(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


def normalize_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    return str(value).strip()


def normalize_smiles(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def prediction_path(root: Path, branch: str, exp: str, fold: int, part: str) -> Path:
    split_prefix = SPLIT_PREFIXES[exp]
    split = f"{split_prefix}{fold}"
    return root / branch / split / f"{split}_{part}_predictions.h5ad"


def dense_float32(matrix: Any) -> np.ndarray:
    if scipy_sparse is not None and scipy_sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.array(matrix, dtype=np.float32, copy=True)


def encode_labels(series: pd.Series) -> pd.Series:
    def _encode(value: object) -> float:
        if value is None or pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        if text in LABEL_MAP:
            return float(LABEL_MAP[text])
        if text in {"1", "1.0", "true", "yes", "responsive"}:
            return 1.0
        if text in {"0", "0.0", "false", "no", "nonresponsive"}:
            return 0.0
        return np.nan

    return series.map(_encode)


def stratified_limit_indices(y: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    n_rows = int(y.shape[0])
    if limit is None or limit <= 0 or n_rows <= limit:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = rng.choice(n_rows, size=limit, replace=False)
    if len(np.unique(y[selected])) == 2 or len(np.unique(y)) < 2 or limit < 2:
        return np.sort(selected).astype(np.int64)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    forced = [int(rng.choice(pos)), int(rng.choice(neg))]
    remaining = np.setdiff1d(np.arange(n_rows, dtype=np.int64), np.asarray(forced, dtype=np.int64), assume_unique=False)
    extra = rng.choice(remaining, size=limit - len(forced), replace=False)
    return np.sort(np.concatenate([np.asarray(forced, dtype=np.int64), extra])).astype(np.int64)


def load_labelled_prediction(path: Path, *, label_col: str, limit_rows: int | None, seed: int, part: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"missing prediction h5ad: {path}")
    adata = ad.read_h5ad(path)
    if label_col not in adata.obs:
        raise KeyError(f"{path} missing label column {label_col!r}")
    expr = dense_float32(adata.X)
    obs = adata.obs.copy()
    obs.insert(0, "_obs_name", obs.index.astype(str))
    encoded = encode_labels(obs[label_col])
    keep = encoded.notna().to_numpy()
    dropped = int((~keep).sum())
    expr = expr[keep]
    y = encoded.loc[keep].astype(np.int64).to_numpy()
    obs = obs.loc[keep].reset_index(drop=True)
    selected = stratified_limit_indices(y, limit_rows, seed)
    expr = expr[selected]
    y = y[selected]
    obs = obs.iloc[selected].reset_index(drop=True)
    if expr.ndim != 2:
        raise ValueError(f"{path}: expected 2D .X, got shape {expr.shape}")
    if not np.isfinite(expr).all():
        raise ValueError(f"{path}: expression .X contains NaN/Inf after filtering ({part})")
    return expr, y.astype(np.float32), obs, dropped


def build_drug_matrix(obs: pd.DataFrame, lookup: MorganLookup) -> tuple[np.ndarray, dict[str, int]]:
    matrix = np.zeros((len(obs), lookup.n_bits), dtype=np.float32)
    stats = {"lookup_hits": 0, "smiles_fallbacks": 0, "empty_smiles_fallbacks": 0}
    for row_idx, row in obs.iterrows():
        smiles = row.get("drug_smiles_primary", row.get("drug_smiles", ""))
        vector, hit = lookup.vector_for(row.get("drug_id", ""), smiles)
        matrix[int(row_idx)] = vector
        if hit:
            stats["lookup_hits"] += 1
        else:
            stats["smiles_fallbacks"] += 1
            if not normalize_smiles(smiles):
                stats["empty_smiles_fallbacks"] += 1
    if not np.isfinite(matrix).all():
        raise ValueError("Morgan fingerprint matrix contains NaN/Inf")
    return matrix, stats


def standardize_expression(splits: dict[str, SplitData], eps: float) -> dict[str, Any]:
    train_expr = splits["train"].expr
    mean = train_expr.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_expr.std(axis=0, dtype=np.float64).astype(np.float32)
    zero_or_small = std < eps
    std[zero_or_small] = 1.0
    for split in splits.values():
        split.expr -= mean
        split.expr /= std
        if not np.isfinite(split.expr).all():
            raise ValueError(f"{split.part}: standardized expression contains NaN/Inf")
    return {
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "zero_std_features": int(zero_or_small.sum()),
        "eps": eps,
        "mean": mean,
        "std": std,
    }


def load_splits(args: argparse.Namespace) -> tuple[dict[str, SplitData], dict[str, Any], dict[str, np.ndarray]]:
    lookup = MorganLookup(args.drug_fp_path)
    splits: dict[str, SplitData] = {}
    for offset, part in enumerate(("train", "val", "test")):
        path = prediction_path(args.prediction_root, args.branch, args.exp, args.fold, part)
        expr, y, obs, dropped = load_labelled_prediction(
            path,
            label_col=args.label_col,
            limit_rows=args.limit_rows,
            seed=args.seed + offset,
            part=part,
        )
        drug, fp_stats = build_drug_matrix(obs, lookup)
        splits[part] = SplitData(
            part=part,
            path=path,
            source_paths=(path,),
            expr=expr,
            drug=drug,
            y=y,
            obs=obs,
            dropped_unlabelled=dropped,
            fingerprint_stats=fp_stats,
        )
    if args.merge_val_into_train:
        splits["train"] = merge_training_splits(splits["train"], splits["val"])
    if splits["train"].expr.shape[1] != args.expected_transcriptome_dim:
        raise ValueError(
            f"train .X dimension {splits['train'].expr.shape[1]} != expected {args.expected_transcriptome_dim}"
        )
    if splits["train"].drug.shape[1] != args.expected_drug_dim:
        raise ValueError(f"drug dimension {splits['train'].drug.shape[1]} != expected {args.expected_drug_dim}")
    train_unique = np.unique(splits["train"].y)
    if len(train_unique) < 2:
        raise ValueError(f"train labels must contain both classes, got {train_unique.tolist()}")
    scaler = standardize_expression(splits, args.standardize_eps)
    scaler_arrays = {"mean": scaler.pop("mean"), "std": scaler.pop("std")}
    summary = {
        "prediction_root": args.prediction_root,
        "drug_fp_path": args.drug_fp_path,
        "transcriptome_dim": int(splits["train"].expr.shape[1]),
        "drug_dim": int(splits["train"].drug.shape[1]),
        "scaler": scaler,
        "splits": {part: split_summary(split) for part, split in splits.items()},
    }
    return splits, summary, scaler_arrays


def merge_training_splits(train: SplitData, val: SplitData) -> SplitData:
    obs = pd.concat([train.obs, val.obs], axis=0, ignore_index=True, copy=False)
    fingerprint_stats = {
        key: int(train.fingerprint_stats.get(key, 0)) + int(val.fingerprint_stats.get(key, 0))
        for key in sorted(set(train.fingerprint_stats) | set(val.fingerprint_stats))
    }
    return SplitData(
        part="train",
        path=train.path,
        source_paths=(*train.source_paths, *val.source_paths),
        expr=np.concatenate([train.expr, val.expr], axis=0),
        drug=np.concatenate([train.drug, val.drug], axis=0),
        y=np.concatenate([train.y, val.y], axis=0),
        obs=obs,
        dropped_unlabelled=train.dropped_unlabelled + val.dropped_unlabelled,
        fingerprint_stats=fingerprint_stats,
    )


def split_summary(split: SplitData) -> dict[str, Any]:
    y_int = split.y.astype(np.int64)
    pos = int(y_int.sum())
    count = int(y_int.shape[0])
    summary = {
        "path": split.path,
        "rows": count,
        "positive": pos,
        "negative": count - pos,
        "label_baseline": (pos / count) if count else None,
        "dropped_unlabelled": split.dropped_unlabelled,
        "expr_shape": list(split.expr.shape),
        "drug_shape": list(split.drug.shape),
        "fingerprint": split.fingerprint_stats,
    }
    if split.source_paths and split.source_paths != (split.path,):
        summary["source_paths"] = list(split.source_paths)
    return summary


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(split: SplitData, *, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(split.expr)),
        torch.from_numpy(np.ascontiguousarray(split.drug)),
        torch.from_numpy(split.y.astype(np.float32, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, loss: float | None = None) -> dict[str, Any]:
    y_true = y_true.astype(np.float32)
    logits = logits.astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    count = int(y_true.size)
    pos = int(y_true.sum())
    neg = count - pos
    baseline = (pos / count) if count else math.nan
    result: dict[str, Any] = {
        "loss": loss,
        "auroc": math.nan,
        "auprc": math.nan,
        "auprc_baseline": baseline,
        "nauprc": math.nan,
        "acc": math.nan,
        "count": count,
        "positive_count": pos,
        "negative_count": neg,
    }
    if count:
        result["acc"] = float(((probs >= 0.5).astype(np.float32) == y_true).mean())
    if pos > 0 and neg > 0:
        auprc = float(average_precision_score(y_true, probs))
        result["auroc"] = float(roc_auc_score(y_true, probs))
        result["auprc"] = auprc
        result["nauprc"] = auprc / baseline if baseline > 0 else math.nan
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for expr, drug, y in loader:
        expr = expr.to(device, non_blocking=True)
        drug = drug.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(expr, drug)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_count = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    for expr, drug, y in loader:
        expr = expr.to(device, non_blocking=True)
        drug = drug.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(expr, drug)
        loss = criterion(logits, y)
        batch_count = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
    y_np = np.concatenate(all_y) if all_y else np.zeros((0,), dtype=np.float32)
    logits_np = np.concatenate(all_logits) if all_logits else np.zeros((0,), dtype=np.float32)
    metrics = compute_metrics(y_np, logits_np, total_loss / max(total_count, 1))
    return metrics, y_np, logits_np


def finite_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return number if math.isfinite(number) else -math.inf


def should_log_epoch(epoch: int, *, log_every: int, max_epochs: int) -> bool:
    return epoch == 1 or epoch == max_epochs or (log_every > 0 and epoch % log_every == 0)


def save_predictions(path: Path, split: SplitData, logits: np.ndarray) -> None:
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    columns = ["split", "row_index", "obs_name", *OBS_EXPORT_COLUMNS, "label", "logit", "probability"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row_idx, (obs_idx, row) in enumerate(split.obs.iterrows()):
            item: dict[str, Any] = {
                "split": split.part,
                "row_index": int(obs_idx),
                "obs_name": row.get("_obs_name", ""),
                "label": int(split.y[row_idx]),
                "logit": float(logits[row_idx]),
                "probability": float(probs[row_idx]),
            }
            for column in OBS_EXPORT_COLUMNS:
                item[column] = row.get(column, "")
            writer.writerow(item)


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits, data_summary, scaler_arrays = load_splits(args)
    run_config = {
        "run_status": "data_loaded" if args.check_data_only else "started",
        "started_at": now_text(),
        "stage": args.stage,
        "config_name": args.config_name,
        "branch": args.branch,
        "exp": args.exp,
        "fold": args.fold,
        "split_prefix": SPLIT_PREFIXES[args.exp],
        "label_col": args.label_col,
        "label_map": LABEL_MAP,
        "model": {
            "hidden_dim": args.hidden_dim,
            "drug_hidden_dim": args.drug_hidden_dim,
            "fusion_hidden_dim": args.fusion_hidden_dim,
            "dropout": args.dropout,
            "activation": args.activation,
        },
        "training": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "fixed_epochs": args.max_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "limit_rows": args.limit_rows,
            "merge_val_into_train": args.merge_val_into_train,
            "no_validation": args.no_validation,
            "selection_mode": "fixed_epoch_no_validation" if args.no_validation else "validation_auprc",
            "log_every": args.log_every,
        },
        "data": data_summary,
        "output_dir": args.output_dir,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(json_safe(run_config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.check_data_only:
        print(json.dumps(json_safe(run_config["data"]), indent=2, sort_keys=True))
        return run_config

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    pin_memory = device.type == "cuda"
    loaders = {
        part: make_loader(
            split,
            batch_size=args.batch_size,
            shuffle=(part == "train"),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        for part, split in splits.items()
    }
    model = TranscriptomeDrugMLP(
        transcriptome_dim=splits["train"].expr.shape[1],
        drug_dim=splits["train"].drug.shape[1],
        hidden_dim=args.hidden_dim,
        drug_hidden_dim=args.drug_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)
    train_y = splits["train"].y.astype(np.int64)
    pos = int(train_y.sum())
    neg = int(train_y.shape[0] - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_metrics_path = args.output_dir / "epoch_metrics.csv"
    epoch_rows: list[dict[str, Any]] = []

    if args.no_validation:
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(
                model,
                loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
            )
            row = {"epoch": epoch, "train_loss": train_loss}
            epoch_rows.append(row)
            pd.DataFrame(epoch_rows).to_csv(epoch_metrics_path, index=False)
            if should_log_epoch(epoch, log_every=args.log_every, max_epochs=args.max_epochs):
                print(
                    "[epoch] {branch} {exp} fold{fold} {cfg} epoch={epoch} train_loss={train_loss:.6f}".format(
                        branch=args.branch,
                        exp=args.exp,
                        fold=args.fold,
                        cfg=args.config_name,
                        epoch=epoch,
                        train_loss=train_loss,
                    ),
                    flush=True,
                )

        final_checkpoint_path = args.output_dir / "final_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler": scaler_arrays,
                "run_config": json_safe(run_config),
                "final_epoch": args.max_epochs,
                "training_mode": "fixed_epoch_no_validation",
            },
            final_checkpoint_path,
        )
        split_metrics: dict[str, Any] = {}
        prediction_paths: dict[str, str] = {}
        for part in ("train", "test"):
            metrics, _, logits = evaluate(model, loaders[part], criterion=criterion, device=device)
            split_metrics[part] = metrics
            pred_path = args.output_dir / f"predictions_{part}.csv"
            save_predictions(pred_path, splits[part], logits)
            prediction_paths[part] = str(pred_path)

        completed = time.time()
        payload = {
            "run_status": "ok",
            "started_at": run_config["started_at"],
            "completed_at": now_text(),
            "duration_seconds": completed - started,
            "stage": args.stage,
            "config_name": args.config_name,
            "branch": args.branch,
            "exp": args.exp,
            "fold": args.fold,
            "best": {
                "epoch": args.max_epochs,
                "monitor": "fixed_epoch_no_validation",
                "score": None,
                "val_metrics": None,
            },
            "config": run_config,
            "splits": split_metrics,
            "final_checkpoint": str(final_checkpoint_path),
            "prediction_paths": prediction_paths,
        }
        metrics_path = args.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"[done] metrics={metrics_path} final_epoch={args.max_epochs} test_auprc={split_metrics['test'].get('auprc')}",
            flush=True,
        )
        return payload

    best_checkpoint_path = args.output_dir / "best_checkpoint.pt"
    best_score = -math.inf
    best_epoch = 0
    best_val_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics, _, _ = evaluate(model, loaders["val"], criterion=criterion, device=device)
        score = finite_score(val_metrics.get("auprc"))
        improved = best_val_metrics is None or score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": scaler_arrays,
                    "run_config": json_safe(run_config),
                    "best_epoch": best_epoch,
                    "best_val_metrics": json_safe(best_val_metrics),
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss"),
            "val_auroc": val_metrics.get("auroc"),
            "val_auprc": val_metrics.get("auprc"),
            "val_auprc_baseline": val_metrics.get("auprc_baseline"),
            "val_nauprc": val_metrics.get("nauprc"),
            "val_acc": val_metrics.get("acc"),
            "best_epoch": best_epoch,
            "best_val_auprc": best_val_metrics.get("auprc") if best_val_metrics else math.nan,
        }
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(epoch_metrics_path, index=False)
        if should_log_epoch(epoch, log_every=args.log_every, max_epochs=args.max_epochs):
            print(
                "[epoch] {branch} {exp} fold{fold} {cfg} epoch={epoch} "
                "train_loss={train_loss:.6f} val_auprc={val_auprc:.6f} val_auroc={val_auroc:.6f} best={best:.6f}".format(
                    branch=args.branch,
                    exp=args.exp,
                    fold=args.fold,
                    cfg=args.config_name,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_auprc=float(val_metrics.get("auprc") or math.nan),
                    val_auroc=float(val_metrics.get("auroc") or math.nan),
                    best=best_score,
                ),
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            print(f"[early-stop] epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    checkpoint = load_torch_checkpoint(best_checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    split_metrics: dict[str, Any] = {}
    prediction_paths: dict[str, str] = {}
    for part in ("train", "val", "test"):
        metrics, _, logits = evaluate(model, loaders[part], criterion=criterion, device=device)
        split_metrics[part] = metrics
        pred_path = args.output_dir / f"predictions_{part}.csv"
        save_predictions(pred_path, splits[part], logits)
        prediction_paths[part] = str(pred_path)

    completed = time.time()
    payload = {
        "run_status": "ok",
        "started_at": run_config["started_at"],
        "completed_at": now_text(),
        "duration_seconds": completed - started,
        "stage": args.stage,
        "config_name": args.config_name,
        "branch": args.branch,
        "exp": args.exp,
        "fold": args.fold,
        "best": {
            "epoch": best_epoch,
            "monitor": "val_auprc",
            "score": best_score,
            "val_metrics": best_val_metrics,
        },
        "config": run_config,
        "splits": split_metrics,
        "best_checkpoint": str(best_checkpoint_path),
        "prediction_paths": prediction_paths,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] metrics={metrics_path} best_epoch={best_epoch} test_auprc={split_metrics['test'].get('auprc')}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("cpa", "biolord"), required=True)
    parser.add_argument("--exp", choices=tuple(SPLIT_PREFIXES), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("tune", "final", "cv", "smoke"), default="final")
    parser.add_argument("--config-name", default="manual")
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--drug-fp-path", type=Path, default=DEFAULT_DRUG_FP_PATH)
    parser.add_argument("--label-col", default="PRISM1st_label_total")
    parser.add_argument("--expected-transcriptome-dim", type=int, default=9843)
    parser.add_argument("--expected-drug-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--drug-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--activation", choices=("silu", "relu"), default="silu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--fixed-epochs", dest="max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--standardize-eps", type=float, default=1e-6)
    parser.add_argument("--merge-val-into-train", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()
    if args.max_epochs is None:
        args.max_epochs = 80
    if args.max_epochs < 1:
        parser.error("--max-epochs/--fixed-epochs must be >= 1")
    if args.merge_val_into_train and not args.no_validation:
        parser.error("--merge-val-into-train requires --no-validation to avoid validation leakage")
    return args


def main() -> int:
    args = parse_args()
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
