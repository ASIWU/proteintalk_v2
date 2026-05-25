#!/usr/bin/env python3
"""Build frozen biomedical text embeddings for cell-type labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


CELL_TYPE_PROMPTS = {
    "breast": [
        "breast cancer epithelial cell line",
        "mammary gland carcinoma cell line",
        "breast tumor cell type",
    ],
    "lung": [
        "lung cancer epithelial cell line",
        "pulmonary carcinoma cell line",
        "lung tumor cell type",
    ],
    "pancreas": [
        "pancreatic cancer epithelial cell line",
        "pancreatic ductal carcinoma cell line",
        "pancreas tumor cell type",
    ],
    "ovary": [
        "ovarian cancer epithelial cell line",
        "ovary carcinoma cell line",
        "ovarian tumor cell type",
    ],
    "skin": [
        "skin cancer cell line",
        "melanoma cell line",
        "cutaneous tumor cell type",
    ],
    "kidney": [
        "kidney cancer epithelial cell line",
        "renal carcinoma cell line",
        "kidney tumor cell type",
    ],
    "thyroid; medulla": [
        "medullary thyroid carcinoma cell line",
        "thyroid medulla cancer cell line",
        "thyroid tumor cell type",
    ],
    "colon": [
        "colon cancer epithelial cell line",
        "colorectal carcinoma cell line",
        "colon tumor cell type",
    ],
}


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def prompts_for_cell_type(name: str) -> list[str]:
    normalized = normalize_name(name)
    if normalized in CELL_TYPE_PROMPTS:
        return CELL_TYPE_PROMPTS[normalized]
    return [
        f"{normalized} cancer cell line",
        f"{normalized} tumor cell type",
        f"{normalized} cell type",
    ]


def cls_pool(outputs: Any) -> torch.Tensor:
    return outputs.last_hidden_state[:, 0, :]


def encode_texts(
    texts: list[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    encoded_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            pooled = cls_pool(model(**tokens))
            encoded_batches.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(encoded_batches, dim=0).numpy().astype(np.float32)


def cell_type_index_name_map(df: pd.DataFrame) -> dict[int, str]:
    if "cell_type_index" not in df.columns or "cell_type" not in df.columns:
        raise ValueError("feature_table must contain cell_type and cell_type_index columns")
    result: dict[int, str] = {}
    subset = df[["cell_type_index", "cell_type"]].dropna().drop_duplicates()
    for _, row in subset.iterrows():
        result[int(row["cell_type_index"])] = str(row["cell_type"])
    return dict(sorted(result.items()))


def build_cell_type_embeddings(
    *,
    df: pd.DataFrame,
    model_name: str,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    index_to_name = cell_type_index_name_map(df)
    all_prompts: list[str] = []
    prompt_spans: dict[int, tuple[int, int]] = {}
    for cell_type_index, name in index_to_name.items():
        start = len(all_prompts)
        prompts = prompts_for_cell_type(name)
        all_prompts.extend(prompts)
        prompt_spans[int(cell_type_index)] = (start, len(all_prompts))
    prompt_embeddings = encode_texts(all_prompts, model_name=model_name, device=device, batch_size=batch_size)
    dim = int(prompt_embeddings.shape[1])
    max_index = max(index_to_name) if index_to_name else 0
    type_embeddings = np.zeros((max_index + 1, dim), dtype=np.float32)
    prompt_record = {}
    for cell_type_index, name in index_to_name.items():
        start, end = prompt_spans[int(cell_type_index)]
        averaged = prompt_embeddings[start:end].mean(axis=0)
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm
        type_embeddings[int(cell_type_index)] = averaged.astype(np.float32, copy=False)
        prompt_record[str(cell_type_index)] = {
            "name": name,
            "prompts": all_prompts[start:end],
        }
    meta = {
        "model_name": model_name,
        "embedding_dim": dim,
        "cell_type_count": len(index_to_name),
        "cell_type_prompts": prompt_record,
    }
    return type_embeddings, meta


def row_feature_matrix(df: pd.DataFrame, type_embeddings: np.ndarray) -> np.ndarray:
    indices = pd.to_numeric(df["cell_type_index"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    if indices.max(initial=0) >= type_embeddings.shape[0]:
        raise ValueError("cell_type_index exceeds type embedding row count")
    return type_embeddings[indices].astype(np.float32, copy=False)


def load_or_build_cell_type_text_features(
    *,
    df: pd.DataFrame,
    cache_path: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 16,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        payload = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(payload["meta_json"].item()))
        if meta.get("model_name") == model_name and int(meta.get("row_count", -1)) == len(df):
            return payload["row_features"].astype(np.float32, copy=False), meta
    type_embeddings, meta = build_cell_type_embeddings(
        df=df,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    row_features = row_feature_matrix(df, type_embeddings)
    meta = {
        **meta,
        "row_count": int(len(df)),
        "feature_kind": "cell_type_text_embedding",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        row_features=row_features.astype(np.float32, copy=False),
        type_embeddings=type_embeddings.astype(np.float32, copy=False),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    return row_features, meta

