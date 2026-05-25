#!/usr/bin/env python3
"""Fast dataset utilities for ProteinTalk training-ready artifacts.

This module keeps the full protein expression axis but loads expression
matrices with numpy memmap so each worker reads only the rows needed for a
batch.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


BATCH_COVARIATE_COLUMNS = {
    "machineID_new": "machineID_new_index",
    "Cell_plate": "Cell_plate_index",
    "Cell": "Cell_index",
    "cell_type": "cell_type_index",
    "batch": "batch_index",
    "pert_time": "pert_time_index",
    "pert_dose1": "pert_dose1_index",
    "pert_dose2": "pert_dose2_index",
}


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def dump_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def load_feature_table(task_dir: str | Path) -> pd.DataFrame:
    task_dir = Path(task_dir)
    parquet_path = task_dir / "feature_table.parquet"
    pickle_path = task_dir / "feature_table.pkl"
    csv_path = task_dir / "feature_table.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path).reset_index(drop=True)
    if pickle_path.exists():
        return pd.read_pickle(pickle_path).reset_index(drop=True)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)
    raise FileNotFoundError(f"missing feature_table parquet/pickle/csv under {task_dir}")


def load_embedding_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32, copy=False)
    payload = load_pickle(path)
    if isinstance(payload, dict) and "embedding_matrix" in payload:
        return np.asarray(payload["embedding_matrix"], dtype=np.float32)
    if isinstance(payload, np.ndarray):
        return payload.astype(np.float32, copy=False)
    raise ValueError(f"{path} does not contain an embedding matrix")


def load_indices(split_dir: str | Path, split_name: str, strategy: str) -> list[int]:
    split_dir = Path(split_dir)
    candidates = [split_dir / f"{split_name}_indices_{strategy}.pkl"]
    if split_name == "valid":
        candidates.append(split_dir / f"val_indices_{strategy}.pkl")
    for path in candidates:
        if path.exists():
            return [int(item) for item in load_pickle(path)]
    raise FileNotFoundError(f"missing {split_name} indices for strategy {strategy} under {split_dir}")


def load_set_info(split_dir: str | Path, split_name: str, strategy: str | None = None) -> dict[int, dict[str, list[int]]]:
    split_dir = Path(split_dir)
    candidates: list[Path] = []
    if strategy is not None:
        candidates.append(split_dir / f"{split_name}_set_info_{strategy}.pkl")
        if split_name == "valid":
            candidates.append(split_dir / f"val_set_info_{strategy}.pkl")
    candidates.append(split_dir / "set_info.pkl")
    for path in candidates:
        if path.exists():
            raw = load_pickle(path)
            return {
                int(key): {
                    "control": [int(item) for item in value["control"]],
                    "perturb": [int(item) for item in value["perturb"]],
                }
                for key, value in raw.items()
            }
    raise FileNotFoundError(f"missing set_info for {split_name}/{strategy} under {split_dir}")


def load_row_to_set(split_dir: str | Path) -> dict[int, int]:
    raw = load_pickle(Path(split_dir) / "row_to_set_index.pkl")
    return {int(key): int(value) for key, value in raw.items()}


def parse_json_list(value: object) -> list[int]:
    if isinstance(value, list):
        parsed = value
    else:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
    result: list[int] = []
    for item in parsed:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def pad_list(values: Iterable[int], *, length: int, pad_value: int) -> np.ndarray:
    result = list(values)
    if len(result) >= length:
        result = result[:length]
    else:
        result.extend([pad_value] * (length - len(result)))
    return np.asarray(result, dtype=np.int64)


def encode_binary_label(value: object, *, positive_values: set[str], negative_values: set[str]) -> tuple[float, float]:
    """Return ``(label, mask)``, with mask semantics ``1=ignore``."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0, 1.0
    if isinstance(value, (bool, np.bool_)):
        return (1.0, 0.0) if bool(value) else (0.0, 0.0)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return 0.0, 1.0
        if numeric == 1.0:
            return 1.0, 0.0
        if numeric == 0.0:
            return 0.0, 0.0
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return 0.0, 1.0
    if text in positive_values:
        return 1.0, 0.0
    if text in negative_values:
        return 0.0, 0.0
    try:
        numeric = float(text)
    except ValueError:
        return 0.0, 1.0
    if not np.isfinite(numeric):
        return 0.0, 1.0
    if numeric == 1.0:
        return 1.0, 0.0
    if numeric == 0.0:
        return 0.0, 0.0
    return 0.0, 1.0


def encode_response_label(value: object) -> tuple[float, float]:
    return encode_binary_label(
        value,
        positive_values={"sensitive", "responsive", "y", "yes", "1", "true"},
        negative_values={"non-responsive", "nonresponsive", "n", "no", "0", "false"},
    )


def encode_synergy_label(value: object) -> tuple[float, float]:
    return encode_binary_label(
        value,
        positive_values={"syn", "synergy", "synergistic", "y", "yes", "1", "true"},
        negative_values={"non-syn", "nonsyn", "non-synergy", "non_synergy", "n", "no", "0", "false"},
    )


def category_sizes(meta: dict[str, Any], batch_cov_list: list[str]) -> list[int]:
    sizes: list[int] = []
    for field in batch_cov_list:
        mapping_key = "pert_dose" if field in {"pert_dose1", "pert_dose2"} else field
        mapping = meta["value_to_index"][mapping_key]
        values = []
        for item in mapping.values():
            try:
                values.append(int(float(item)))
            except (TypeError, ValueError):
                continue
        sizes.append(max(values) + 1 if values else 1)
    return sizes


@dataclass
class FastTrainingReadyArtifacts:
    task_dir: Path
    meta_path: Path
    df: pd.DataFrame
    expression_matrix: np.ndarray
    ordered_protein_index: list[int]
    sample_ids: list[str]
    meta: dict[str, Any]

    @classmethod
    def load(cls, task_dir: str | Path, meta_path: str | Path) -> "FastTrainingReadyArtifacts":
        task_dir = Path(task_dir)
        meta_path = Path(meta_path)
        expression_path = task_dir / "feature_expression_matrix.npy"
        if not expression_path.exists():
            raise FileNotFoundError(f"missing expression matrix: {expression_path}")
        df = load_feature_table(task_dir)
        expression_matrix = np.load(expression_path, mmap_mode="r")
        ordered_protein_index = [int(item) for item in load_json(task_dir / "feature_ordered_protein_index.json")]
        sample_ids = [str(item) for item in load_json(task_dir / "feature_sample_ids.json")]
        meta = load_json(meta_path)
        if len(df) != expression_matrix.shape[0]:
            raise ValueError(
                f"{task_dir}: feature_table rows {len(df)} != expression rows {expression_matrix.shape[0]}"
            )
        return cls(
            task_dir=task_dir,
            meta_path=meta_path,
            df=df,
            expression_matrix=expression_matrix,
            ordered_protein_index=ordered_protein_index,
            sample_ids=sample_ids,
            meta=meta,
        )


class FastProteinTalkDataset(Dataset):
    """Emit paired control/perturb rows for the fast model."""

    def __init__(
        self,
        *,
        artifacts: FastTrainingReadyArtifacts,
        indices: list[int],
        row_to_set_index: dict[int, int],
        set_info: dict[int, dict[str, list[int]]],
        mode: str,
        drug_embedding_matrix: np.ndarray,
        batch_cov_list: list[str],
        target_protein_max_length: int,
        effective_key1: str,
        effective_key2: str = "synergy",
        ddi_matrix: np.ndarray | None = None,
        graph_feature_matrix: np.ndarray | None = None,
        graph_feature_enabled: bool = False,
        expression_column_index: np.ndarray | None = None,
        epoch_len: int | None = None,
        covariate_known_values: dict[str, set[int]] | None = None,
        covariate_unknown_indices: dict[str, int] | None = None,
        covariate_unk_dropout: float = 0.0,
        prior_feature_matrix: np.ndarray | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.df = artifacts.df
        self.expression_matrix = artifacts.expression_matrix
        self.indices = [int(idx) for idx in indices]
        self.row_to_set_index = row_to_set_index
        self.set_info = set_info
        self.mode = mode
        self.drug_embedding_matrix = np.asarray(drug_embedding_matrix, dtype=np.float32)
        self.batch_cov_list = list(batch_cov_list)
        self.target_protein_max_length = int(target_protein_max_length)
        self.effective_key1 = effective_key1
        self.effective_key2 = effective_key2
        self.ddi_matrix = ddi_matrix
        self.graph_feature_matrix = None if graph_feature_matrix is None else np.asarray(graph_feature_matrix, dtype=np.float32)
        self.graph_feature_enabled = bool(graph_feature_enabled and self.graph_feature_matrix is not None)
        self.graph_feature_dim = 0 if self.graph_feature_matrix is None else int(self.graph_feature_matrix.shape[1])
        self.expression_column_index = (
            None if expression_column_index is None else np.asarray(expression_column_index, dtype=np.int64)
        )
        self.epoch_len = epoch_len
        self.covariate_known_values = covariate_known_values or {}
        self.covariate_unknown_indices = covariate_unknown_indices or {}
        self.covariate_unk_dropout = float(covariate_unk_dropout)
        self.prior_feature_matrix = (
            None if prior_feature_matrix is None else np.asarray(prior_feature_matrix, dtype=np.float32)
        )
        self.prior_feature_dim = 0 if self.prior_feature_matrix is None else int(self.prior_feature_matrix.shape[1])
        if self.prior_feature_matrix is not None and self.prior_feature_matrix.shape[0] != len(self.df):
            raise ValueError(
                "prior_feature_matrix row count must match feature table; "
                f"got {self.prior_feature_matrix.shape[0]} and {len(self.df)}"
            )
        if self.covariate_unk_dropout < 0.0 or self.covariate_unk_dropout >= 1.0:
            raise ValueError("covariate_unk_dropout must be in [0, 1)")
        self.dataset_len = len(self.indices)
        if self.dataset_len == 0:
            raise ValueError("FastProteinTalkDataset received zero indices")

        self.no_pert_index = int(artifacts.meta["special_values"]["pert_index"]["no"])
        self.target_pad_index = len(artifacts.meta["protein_index"])
        self._raw_covariates = self._build_covariate_matrix(apply_unknown_mapping=False)
        self._covariates = self._build_covariate_matrix(apply_unknown_mapping=True)
        self._pert_indices = self._build_perturbation_indices()
        self._target_indices, self._target_mask = self._build_target_matrix()
        self._label1, self._mask1, self._label2, self._mask2 = self._build_labels()
        self._ddi_values = self._build_ddi_values()

    def __len__(self) -> int:
        if self.mode == "train" and self.epoch_len is not None:
            return int(self.epoch_len)
        return self.dataset_len

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if self.mode == "train":
            anchor_idx = self.indices[idx % self.dataset_len]
            set_idx = self.row_to_set_index[anchor_idx]
            control_candidates = self.set_info[set_idx]["control"]
            control_row = int(np.random.choice(control_candidates))
            perturb_row = int(anchor_idx)
        else:
            anchor_idx = self.indices[idx]
            set_idx = self.row_to_set_index[anchor_idx]
            control_row = int(sorted(self.set_info[set_idx]["control"])[0])
            perturb_row = int(anchor_idx)
        return self._format_pair(control_row, perturb_row)

    def _format_pair(self, control_row: int, perturb_row: int) -> dict[str, np.ndarray]:
        pert_indices = self._pert_indices[perturb_row]
        output = {
            "control_expression": self._expression_row(control_row),
            "perturb_expression": self._expression_row(perturb_row),
            "drug_embeddings": np.asarray(self.drug_embedding_matrix[pert_indices], dtype=np.float32),
            "drug_indices": pert_indices.astype(np.int64, copy=False),
            "graph_features": self._graph_features_for(pert_indices),
            "graph_feature_mask": np.asarray(1.0 if self.graph_feature_enabled else 0.0, dtype=np.float32),
            "target_indices": self._target_indices[perturb_row],
            "target_mask": self._target_mask[perturb_row],
            "covariates": self._covariates_for(perturb_row),
            "raw_covariates": self._raw_covariates[perturb_row],
            "prior_features": self._prior_features_for(perturb_row),
            "ddi_value": np.asarray(self._ddi_values[perturb_row], dtype=np.float32),
            "label1": np.asarray(self._label1[perturb_row], dtype=np.float32),
            "mask1": np.asarray(self._mask1[perturb_row], dtype=np.float32),
            "label2": np.asarray(self._label2[perturb_row], dtype=np.float32),
            "mask2": np.asarray(self._mask2[perturb_row], dtype=np.float32),
            "row_index": np.asarray(perturb_row, dtype=np.int64),
        }
        return output

    def _expression_row(self, row_idx: int) -> np.ndarray:
        row = np.asarray(self.expression_matrix[row_idx], dtype=np.float32)
        if self.expression_column_index is None:
            return np.array(row, dtype=np.float32, copy=True)
        aligned = np.full(self.expression_column_index.shape[0], np.nan, dtype=np.float32)
        valid = self.expression_column_index >= 0
        aligned[valid] = row[self.expression_column_index[valid]]
        return aligned

    def _graph_features_for(self, pert_indices: np.ndarray) -> np.ndarray:
        if self.graph_feature_matrix is None:
            return np.zeros((2, 0), dtype=np.float32)
        if not self.graph_feature_enabled:
            return np.zeros((2, self.graph_feature_dim), dtype=np.float32)
        clipped = np.clip(pert_indices, 0, self.graph_feature_matrix.shape[0] - 1)
        return np.asarray(self.graph_feature_matrix[clipped], dtype=np.float32)

    def _build_covariate_matrix(self, *, apply_unknown_mapping: bool) -> np.ndarray:
        values = []
        for field in self.batch_cov_list:
            source_col = BATCH_COVARIATE_COLUMNS.get(field, f"{field}_index")
            if source_col not in self.df.columns:
                raise KeyError(f"batch covariate {field!r} requires missing column {source_col!r}")
            col = pd.to_numeric(self.df[source_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
            if apply_unknown_mapping and field in self.covariate_known_values and field in self.covariate_unknown_indices:
                known = np.fromiter(self.covariate_known_values[field], dtype=np.int64)
                unseen = ~np.isin(col, known)
                if unseen.any():
                    col = col.copy()
                    col[unseen] = int(self.covariate_unknown_indices[field])
            values.append(col)
        if not values:
            return np.zeros((len(self.df), 0), dtype=np.int64)
        return np.stack(values, axis=1).astype(np.int64, copy=False)

    def _covariates_for(self, row_idx: int) -> np.ndarray:
        covariates = np.asarray(self._covariates[row_idx], dtype=np.int64)
        if self.mode != "train" or self.covariate_unk_dropout <= 0.0 or not self.covariate_unknown_indices:
            return covariates
        covariates = covariates.copy()
        for col_idx, field in enumerate(self.batch_cov_list):
            unknown_index = self.covariate_unknown_indices.get(field)
            if unknown_index is not None and np.random.random() < self.covariate_unk_dropout:
                covariates[col_idx] = int(unknown_index)
        return covariates

    def _prior_features_for(self, row_idx: int) -> np.ndarray:
        if self.prior_feature_matrix is None:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(self.prior_feature_matrix[row_idx], dtype=np.float32)

    def _build_perturbation_indices(self) -> np.ndarray:
        result = np.zeros((len(self.df), 2), dtype=np.int64)
        for out_col, source_col in enumerate(["pert_index1", "pert_index2"]):
            if source_col not in self.df.columns:
                result[:, out_col] = self.no_pert_index
                continue
            values = (
                pd.to_numeric(self.df[source_col], errors="coerce")
                .fillna(self.no_pert_index)
                .astype(np.int64)
                .clip(lower=0, upper=self.drug_embedding_matrix.shape[0] - 1)
                .to_numpy()
            )
            result[:, out_col] = values
        return result

    def _build_target_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        target_indices = np.full(
            (len(self.df), self.target_protein_max_length),
            fill_value=self.target_pad_index,
            dtype=np.int64,
        )
        target_mask = np.zeros((len(self.df), self.target_protein_max_length), dtype=np.float32)
        if "target_protein_list" not in self.df.columns:
            return target_indices, target_mask
        max_valid_index = self.target_pad_index - 1
        for row_idx, value in enumerate(self.df["target_protein_list"].to_numpy()):
            parsed = [idx for idx in parse_json_list(value) if 0 <= idx <= max_valid_index]
            padded = pad_list(parsed, length=self.target_protein_max_length, pad_value=self.target_pad_index)
            target_indices[row_idx] = padded
            target_mask[row_idx] = (padded != self.target_pad_index).astype(np.float32)
        return target_indices, target_mask

    def _build_labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        labels1 = np.zeros(len(self.df), dtype=np.float32)
        masks1 = np.ones(len(self.df), dtype=np.float32)
        labels2 = np.zeros(len(self.df), dtype=np.float32)
        masks2 = np.ones(len(self.df), dtype=np.float32)
        series1 = self.df[self.effective_key1] if self.effective_key1 in self.df.columns else [None] * len(self.df)
        series2 = self.df[self.effective_key2] if self.effective_key2 in self.df.columns else [None] * len(self.df)
        for row_idx, value in enumerate(series1):
            labels1[row_idx], masks1[row_idx] = encode_response_label(value)
        for row_idx, value in enumerate(series2):
            labels2[row_idx], masks2[row_idx] = encode_synergy_label(value)
        return labels1, masks1, labels2, masks2

    def _build_ddi_values(self) -> np.ndarray:
        if self.ddi_matrix is None:
            return np.zeros(len(self.df), dtype=np.float32)
        p1 = self._pert_indices[:, 0]
        p2 = self._pert_indices[:, 1]
        return np.asarray(self.ddi_matrix[p1, p2], dtype=np.float32)


def compute_positive_weight(
    *,
    df: pd.DataFrame,
    indices: list[int],
    label_key: str,
    task_head: str,
    max_weight: float = 20.0,
) -> float | None:
    if label_key not in df.columns:
        return None
    encoder = encode_synergy_label if task_head == "synergy" else encode_response_label
    positives = 0
    negatives = 0
    for value in df.iloc[indices][label_key]:
        label, mask = encoder(value)
        if mask >= 0.5:
            continue
        if label >= 0.5:
            positives += 1
        else:
            negatives += 1
    if positives == 0 or negatives == 0:
        return None
    return float(min(max_weight, max(1.0, negatives / positives)))
