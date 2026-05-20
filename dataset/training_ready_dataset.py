#!/usr/bin/env python3
"""PyTorch dataset for `data/training_ready` ProteinTalk artifacts."""

from __future__ import annotations

import json
import pickle
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
    """Return `(label, mask)`, where mask follows legacy semantics: 1=ignore."""

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


class TrainingReadyArtifacts:
    """Lazy container for one task's feature table, expression matrix, and metadata."""

    def __init__(self, task_dir: str | Path, meta_path: str | Path) -> None:
        self.task_dir = Path(task_dir)
        self.meta_path = Path(meta_path)
        self.df = load_feature_table(self.task_dir)
        self.expression_matrix = np.load(self.task_dir / "feature_expression_matrix.npy").astype(np.float32, copy=False)
        self.ordered_protein_index = list(load_json(self.task_dir / "feature_ordered_protein_index.json"))
        self.sample_ids = list(load_json(self.task_dir / "feature_sample_ids.json"))
        self.meta = load_json(self.meta_path)
        if len(self.df) != self.expression_matrix.shape[0]:
            raise ValueError(
                f"{self.task_dir}: feature_table rows {len(self.df)} != expression rows {self.expression_matrix.shape[0]}"
            )


def load_indices(split_dir: str | Path, split_name: str, strategy: str) -> list[int]:
    split_dir = Path(split_dir)
    candidates = [
        split_dir / f"{split_name}_indices_{strategy}.pkl",
    ]
    if split_name == "valid":
        candidates.append(split_dir / f"val_indices_{strategy}.pkl")
    for path in candidates:
        if path.exists():
            return [int(item) for item in load_pickle(path)]
    raise FileNotFoundError(f"missing {split_name} indices for strategy {strategy} under {split_dir}")


def load_set_info(split_dir: str | Path, split_name: str, strategy: str | None = None) -> dict[int, dict[str, list[int]]]:
    split_dir = Path(split_dir)
    if strategy is not None:
        candidates = [split_dir / f"{split_name}_set_info_{strategy}.pkl"]
        if split_name == "valid":
            candidates.append(split_dir / f"val_set_info_{strategy}.pkl")
        for path in candidates:
            if path.exists():
                raw = load_pickle(path)
                return {int(k): {"control": list(v["control"]), "perturb": list(v["perturb"])} for k, v in raw.items()}
    path = split_dir / "set_info.pkl"
    raw = load_pickle(path)
    return {int(k): {"control": list(v["control"]), "perturb": list(v["perturb"])} for k, v in raw.items()}


def load_row_to_set(split_dir: str | Path) -> dict[int, int]:
    raw = load_pickle(Path(split_dir) / "row_to_set_index.pkl")
    return {int(k): int(v) for k, v in raw.items()}


def default_anchor_indices(df: pd.DataFrame) -> list[int]:
    is_control = df.get("is_control", pd.Series([False] * len(df))).fillna(False).astype(bool)
    if "source_row_role" in df.columns:
        source_self = df["source_row_role"].astype("string").fillna("").str.strip().eq("self")
    else:
        source_self = pd.Series([True] * len(df), index=df.index)
    if "feature_membership" in df.columns:
        primary = df["feature_membership"].astype("string").fillna("").str.strip().eq("primary")
    else:
        primary = pd.Series([True] * len(df), index=df.index)
    return [int(idx) for idx in df.index[(~is_control) & source_self & primary]]


def build_pairing_from_table(df: pd.DataFrame, anchor_indices: list[int]) -> tuple[dict[int, int], dict[int, dict[str, list[int]]]]:
    sample_ids = df["sample_id"].astype("string").fillna("").str.strip()
    controls = df["control"].astype("string").fillna("").str.strip()
    sample_to_index = {str(sample_id): int(idx) for idx, sample_id in sample_ids.items() if str(sample_id)}
    grouped: dict[str, list[int]] = {}
    for idx in anchor_indices:
        control_id = str(controls.iloc[idx])
        if control_id in sample_to_index:
            grouped.setdefault(control_id, []).append(int(idx))
    set_info: dict[int, dict[str, list[int]]] = {}
    row_to_set: dict[int, int] = {}
    for set_idx, control_id in enumerate(sorted(grouped)):
        control_idx = sample_to_index[control_id]
        perturb_indices = sorted(grouped[control_id])
        set_info[set_idx] = {"control": [control_idx], "perturb": perturb_indices}
        row_to_set[control_idx] = set_idx
        for row_idx in perturb_indices:
            row_to_set[row_idx] = set_idx
    valid_indices = {row for info in set_info.values() for row in info["perturb"]}
    missing = sorted(set(anchor_indices) - valid_indices)
    if missing:
        raise ValueError(f"{len(missing)} anchor rows do not resolve to a control row; examples={missing[:20]}")
    return row_to_set, set_info


class ProteinTalkDataset(Dataset):
    """Dataset that emits one paired `{control, perturb}` sample per item."""

    def __init__(
        self,
        artifacts: TrainingReadyArtifacts,
        indices: list[int],
        row_to_set_index: dict[int, int],
        set_info: dict[int, dict[str, list[int]]],
        *,
        mode: str,
        batch_cov_list: list[str],
        drug_mode: str,
        drug_embedding_matrix: np.ndarray | None = None,
        target_protein_max_length: int = 10,
        effective_key1: str = "PRISM1st_label_total",
        effective_key2: str = "synergy",
        epoch_len: int | None = None,
    ) -> None:
        if drug_mode not in {"embedding", "index"}:
            raise ValueError("drug_mode must be `embedding` or `index`")
        if drug_mode == "embedding" and drug_embedding_matrix is None:
            raise ValueError("drug_embedding_matrix is required when drug_mode='embedding'")
        self.artifacts = artifacts
        self.df = artifacts.df
        self.expression_matrix = artifacts.expression_matrix
        self.indices = [int(idx) for idx in indices]
        self.row_to_set_index = row_to_set_index
        self.set_info = set_info
        self.mode = mode
        self.batch_cov_list = list(batch_cov_list)
        self.drug_mode = drug_mode
        self.drug_embedding_matrix = drug_embedding_matrix
        self.target_protein_max_length = int(target_protein_max_length)
        self.effective_key1 = effective_key1
        self.effective_key2 = effective_key2
        self.epoch_len = epoch_len
        self.dataset_len = len(self.indices)
        self.no_pert_index = int(artifacts.meta["special_values"]["pert_index"]["no"])
        self.target_pad_index = len(artifacts.meta["protein_index"])
        if self.dataset_len == 0:
            raise ValueError("ProteinTalkDataset received zero indices")

    def __len__(self) -> int:
        return int(self.epoch_len) if self.epoch_len is not None and self.mode == "train" else self.dataset_len

    def __getitem__(self, idx: int) -> dict[str, dict[str, np.ndarray]]:
        if self.mode == "train":
            anchor_idx = self.indices[idx % self.dataset_len]
            set_idx = self.row_to_set_index[anchor_idx]
            info = self.set_info[set_idx]
            control_row = int(np.random.choice(info["control"]))
            perturb_row = int(anchor_idx)
        else:
            anchor_idx = self.indices[idx]
            set_idx = self.row_to_set_index[anchor_idx]
            info = self.set_info[set_idx]
            control_row = int(sorted(info["control"])[0])
            perturb_row = int(anchor_idx)
        return {
            "control": self._format_row(control_row),
            "perturb": self._format_row(perturb_row),
        }

    def _format_row(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        expression = self.expression_matrix[row].astype(np.float32, copy=False)
        output: dict[str, np.ndarray] = {
            "expressions_hvg": expression,
            "row_index": np.asarray(row, dtype=np.int64),
            "index": np.asarray(row, dtype=np.int64),
        }
        for field in self.batch_cov_list:
            source_col = BATCH_COVARIATE_COLUMNS.get(field, f"{field}_index")
            if source_col not in self.df.columns:
                raise KeyError(f"batch covariate {field!r} requires missing column {source_col!r}")
            value = pd.to_numeric(pd.Series([self.df.iloc[row][source_col]]), errors="coerce").fillna(0).iloc[0]
            output[field] = np.asarray(value, dtype=np.int64)

        pert_indices = self._perturbation_indices(row)
        if self.drug_mode == "index":
            output["pert_id"] = pert_indices.astype(np.int64)
        else:
            assert self.drug_embedding_matrix is not None
            output["pert_id"] = self.drug_embedding_matrix[pert_indices].astype(np.float32, copy=False)

        output["target_protein_list"] = pad_list(
            parse_json_list(self.df.iloc[row].get("target_protein_list", "[]")),
            length=self.target_protein_max_length,
            pad_value=self.target_pad_index,
        )
        label1, mask1 = encode_response_label(self.df.iloc[row].get(self.effective_key1))
        label2, mask2 = encode_synergy_label(self.df.iloc[row].get(self.effective_key2))
        output[self.effective_key1] = np.asarray(label1, dtype=np.float32)
        output["sensitive_label_mask"] = np.asarray(mask1, dtype=np.float32)
        output[self.effective_key2] = np.asarray(label2, dtype=np.float32)
        output["synergy_label_mask"] = np.asarray(mask2, dtype=np.float32)
        return output

    def _perturbation_indices(self, row: int) -> np.ndarray:
        row_data = self.df.iloc[int(row)]
        pert1 = pd.to_numeric(pd.Series([row_data["pert_index1"]]), errors="coerce").fillna(self.no_pert_index).iloc[0]
        pert2 = pd.to_numeric(pd.Series([row_data["pert_index2"]]), errors="coerce").fillna(self.no_pert_index).iloc[0]
        return np.asarray([pert1, pert2], dtype=np.int64)
