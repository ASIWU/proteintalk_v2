#!/usr/bin/env python3
"""Compressed PPI/PDI/DDI graph features for the fast model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def _source_signature(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _random_projection(in_dim: int, out_dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(max(1, in_dim))
    return rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(np.float32)


def _csr_from_npy(path: str | Path, *, name: str) -> sparse.csr_matrix:
    matrix = np.load(path, mmap_mode="r")
    dense = np.asarray(matrix, dtype=np.float32)
    csr = sparse.csr_matrix(dense)
    csr.eliminate_zeros()
    if csr.nnz == 0:
        raise ValueError(f"{name} matrix has no nonzero entries: {path}")
    return csr


def _normalize_csr_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    normalized = matrix.tocsr(copy=True).astype(np.float32)
    row_sum = np.asarray(normalized.sum(axis=1)).reshape(-1).astype(np.float32)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    np.divide(1.0, row_sum, out=inv, where=row_sum > 0)
    counts = np.diff(normalized.indptr)
    normalized.data *= np.repeat(inv, counts)
    return normalized


def _sparse_row_max(matrix: sparse.csr_matrix) -> np.ndarray:
    result = np.zeros(matrix.shape[0], dtype=np.float32)
    for row_idx in range(matrix.shape[0]):
        start = matrix.indptr[row_idx]
        end = matrix.indptr[row_idx + 1]
        if end > start:
            result[row_idx] = float(matrix.data[start:end].max())
    return result


def _sparse_stats(matrix: sparse.csr_matrix) -> np.ndarray:
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float32)
    row_nnz = np.diff(matrix.indptr).astype(np.float32)
    row_max = _sparse_row_max(matrix)
    return np.stack(
        [
            np.log1p(row_sum),
            np.log1p(row_nnz),
            row_max,
        ],
        axis=1,
    ).astype(np.float32)


def _dense_row_normalized_context(matrix_path: str | Path, embedding: np.ndarray, *, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.shape[1] != embedding.shape[0]:
        raise ValueError(f"DDI matrix shape {matrix.shape} is incompatible with embedding {embedding.shape}")
    context = np.zeros((matrix.shape[0], embedding.shape[1]), dtype=np.float32)
    stats = np.zeros((matrix.shape[0], 3), dtype=np.float32)
    for start in range(0, matrix.shape[0], chunk_size):
        end = min(start + chunk_size, matrix.shape[0])
        block = np.asarray(matrix[start:end], dtype=np.float32)
        row_sum = block.sum(axis=1).astype(np.float32)
        denom = np.where(row_sum > 0, row_sum, 1.0).astype(np.float32)
        normalized = block / denom[:, None]
        context[start:end] = normalized @ embedding
        stats[start:end, 0] = np.log1p(row_sum)
        stats[start:end, 1] = np.log1p((block > 0).sum(axis=1).astype(np.float32))
        stats[start:end, 2] = block.max(axis=1).astype(np.float32)
    return context, stats


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    standardized = (features - mean) / std
    return np.nan_to_num(standardized.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), mean, std


def build_or_load_graph_features(
    *,
    cache_dir: str | Path,
    dataset_group: str,
    ppi_matrix_path: str | Path,
    pdi_matrix_path: str | Path,
    ddi_matrix_path: str | Path,
    protein_embedding: np.ndarray,
    drug_embedding: np.ndarray,
    graph_feature_dim: int = 64,
    seed: int = 17,
    include_structural_rp: bool = False,
    include_multihop: bool = False,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a drug-indexed compressed graph feature matrix.

    Feature blocks:
    - direct PDI target context: normalized PDI row pooled over projected protein embeddings
    - PPI-propagated target context: normalized PDI row pooled over normalized PPI neighbors
    - DDI context: normalized DDI row pooled over projected Morgan drug embeddings
    - optional multi-hop heterogeneous contexts through PPI and DDI
    - optional structural random projections of normalized PDI, PDI-PPI, and DDI rows
    - compact source statistics from PDI and DDI rows
    """

    if graph_feature_dim <= 0:
        raise ValueError("graph_feature_dim must be positive")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    structural_tag = "_structrp" if include_structural_rp else ""
    multihop_tag = "_multihop" if include_multihop else ""
    feature_path = cache_dir / f"{dataset_group}_ppi_pdi_ddi_dim{graph_feature_dim}_seed{seed}{structural_tag}{multihop_tag}.npy"
    meta_path = feature_path.with_suffix(".meta.json")
    expected_sources = {
        "ppi_matrix": _source_signature(ppi_matrix_path),
        "pdi_matrix": _source_signature(pdi_matrix_path),
        "ddi_matrix": _source_signature(ddi_matrix_path),
    }
    expected = {
        "dataset_group": dataset_group,
        "graph_feature_dim": int(graph_feature_dim),
        "seed": int(seed),
        "include_structural_rp": bool(include_structural_rp),
        "include_multihop": bool(include_multihop),
        "sources": expected_sources,
        "protein_embedding_shape": list(map(int, protein_embedding.shape)),
        "drug_embedding_shape": list(map(int, drug_embedding.shape)),
    }
    existing_meta = _read_meta(meta_path)
    if (
        not force_rebuild
        and feature_path.exists()
        and existing_meta is not None
        and all(existing_meta.get(key) == value for key, value in expected.items())
    ):
        return np.load(feature_path, mmap_mode="r"), existing_meta

    protein_embedding = np.asarray(protein_embedding, dtype=np.float32)
    drug_embedding = np.asarray(drug_embedding, dtype=np.float32)
    protein_projection = _random_projection(protein_embedding.shape[1], graph_feature_dim, seed=seed)
    drug_projection = _random_projection(drug_embedding.shape[1], graph_feature_dim, seed=seed + 1)
    protein_projected = np.nan_to_num(protein_embedding @ protein_projection, nan=0.0, posinf=0.0, neginf=0.0)
    drug_projected = np.nan_to_num(drug_embedding @ drug_projection, nan=0.0, posinf=0.0, neginf=0.0)

    pdi = _csr_from_npy(pdi_matrix_path, name="PDI")
    ppi = _csr_from_npy(ppi_matrix_path, name="PPI")
    if pdi.shape[1] != protein_projected.shape[0]:
        raise ValueError(f"PDI matrix shape {pdi.shape} is incompatible with protein embeddings {protein_projected.shape}")
    if ppi.shape[0] != ppi.shape[1] or ppi.shape[0] != protein_projected.shape[0]:
        raise ValueError(f"PPI matrix shape {ppi.shape} is incompatible with protein embeddings {protein_projected.shape}")
    if pdi.shape[0] != drug_projected.shape[0]:
        raise ValueError(f"PDI matrix shape {pdi.shape} is incompatible with drug embeddings {drug_projected.shape}")

    pdi_norm = _normalize_csr_rows(pdi)
    ppi_norm = _normalize_csr_rows(ppi)
    ppi_neighbor_embedding = np.asarray(ppi_norm @ protein_projected, dtype=np.float32)
    pdi_direct = np.asarray(pdi_norm @ protein_projected, dtype=np.float32)
    pdi_ppi = np.asarray(pdi_norm @ ppi_neighbor_embedding, dtype=np.float32)
    ddi_context, ddi_stats = _dense_row_normalized_context(ddi_matrix_path, drug_projected)
    if ddi_context.shape[0] != drug_projected.shape[0]:
        raise ValueError(f"DDI matrix rows {ddi_context.shape[0]} != drug embedding rows {drug_projected.shape[0]}")

    pdi_stats = _sparse_stats(pdi)
    feature_blocks: list[tuple[str, np.ndarray]] = [
        ("pdi_direct", pdi_direct),
        ("pdi_ppi", pdi_ppi),
        ("ddi_context", ddi_context),
    ]
    if include_multihop:
        ppi2_neighbor_embedding = np.asarray(ppi_norm @ ppi_neighbor_embedding, dtype=np.float32)
        feature_blocks.extend(
            [
                ("pdi_ppi2", np.asarray(pdi_norm @ ppi2_neighbor_embedding, dtype=np.float32)),
                ("ddi2_context", _dense_row_normalized_context(ddi_matrix_path, ddi_context)[0]),
                ("ddi_pdi_context", _dense_row_normalized_context(ddi_matrix_path, pdi_direct)[0]),
                ("ddi_pdi_ppi_context", _dense_row_normalized_context(ddi_matrix_path, pdi_ppi)[0]),
            ]
        )
    if include_structural_rp:
        protein_node_projection = _random_projection(pdi.shape[1], graph_feature_dim, seed=seed + 2)
        drug_node_projection = _random_projection(drug_projected.shape[0], graph_feature_dim, seed=seed + 3)
        ppi_neighbor_struct = np.asarray(ppi_norm @ protein_node_projection, dtype=np.float32)
        pdi_struct = np.asarray(pdi_norm @ protein_node_projection, dtype=np.float32)
        pdi_ppi_struct = np.asarray(pdi_norm @ ppi_neighbor_struct, dtype=np.float32)
        ddi_struct = _dense_row_normalized_context(ddi_matrix_path, drug_node_projection)[0]
        feature_blocks.extend(
            [
                ("pdi_struct", pdi_struct),
                ("pdi_ppi_struct", pdi_ppi_struct),
                ("ddi_struct", ddi_struct),
            ]
        )
        if include_multihop:
            ppi2_neighbor_struct = np.asarray(ppi_norm @ ppi_neighbor_struct, dtype=np.float32)
            feature_blocks.extend(
                [
                    ("pdi_ppi2_struct", np.asarray(pdi_norm @ ppi2_neighbor_struct, dtype=np.float32)),
                    ("ddi2_struct", _dense_row_normalized_context(ddi_matrix_path, ddi_struct)[0]),
                    ("ddi_pdi_struct", _dense_row_normalized_context(ddi_matrix_path, pdi_struct)[0]),
                    ("ddi_pdi_ppi_struct", _dense_row_normalized_context(ddi_matrix_path, pdi_ppi_struct)[0]),
                ]
            )
    feature_blocks.extend([("pdi_stats", pdi_stats), ("ddi_stats", ddi_stats)])
    graph_features = np.concatenate([block for _, block in feature_blocks], axis=1)
    graph_features, mean, std = _standardize(graph_features)
    np.save(feature_path, graph_features.astype(np.float32))

    slices: dict[str, list[int]] = {}
    cursor = 0
    for name, block in feature_blocks:
        next_cursor = cursor + int(block.shape[1])
        slices[name] = [cursor, next_cursor]
        cursor = next_cursor
    meta = {
        **expected,
        "feature_path": str(feature_path.resolve()),
        "feature_shape": list(map(int, graph_features.shape)),
        "feature_slices": slices,
        "standardization_mean_shape": list(map(int, mean.shape)),
        "standardization_std_shape": list(map(int, std.shape)),
        "ppi_nnz": int(ppi.nnz),
        "pdi_nnz": int(pdi.nnz),
        "graph_feature_description": (
            "concat(normalized PDI pooled protein projection, normalized PDI pooled "
            "PPI-neighbor protein projection, normalized DDI pooled drug projection, "
            "optional multi-hop PPI/DDI meta-path contexts, optional structural random "
            "projections, PDI row stats, DDI row stats)"
        ),
    }
    _write_meta(meta_path, meta)
    return np.load(feature_path, mmap_mode="r"), meta
