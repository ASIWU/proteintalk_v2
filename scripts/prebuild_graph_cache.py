#!/usr/bin/env python3
"""Prebuild compressed PPI/PDI/DDI graph features before parallel training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.training_ready_fast_dataset import load_embedding_matrix
from model.graph_feature_utils import build_or_load_graph_features


DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def default_derived_paths(training_ready_root: Path, dataset_group: str) -> dict[str, Path]:
    derived = training_ready_root / dataset_group / "derived"
    return {
        "protein_embedding": derived / "protein_embedding_esm.pkl",
        "drug_embedding": derived / "drug_embedding_morgan_2048.pkl",
        "ppi_matrix": derived / "ppi_matrix.npy",
        "pdi_matrix": derived / "pdi_matrix.npy",
        "ddi_matrix": derived / "ddi_matrix.npy",
    }


def json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def require_existing(paths: dict[str, Path]) -> None:
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required graph-cache source files:\n  " + "\n  ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prebuild ProteinTalk compressed graph feature cache")
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", default="ptv3")
    parser.add_argument("--protein-embedding-path", default=None)
    parser.add_argument("--drug-embedding-path", default=None)
    parser.add_argument("--ppi-matrix-path", default=None)
    parser.add_argument("--pdi-matrix-path", default=None)
    parser.add_argument("--ddi-matrix-path", default=None)
    parser.add_argument("--graph-cache-dir", default=str(REPO_ROOT / "graph_cache"))
    parser.add_argument("--graph-feature-dim", type=int, default=128)
    parser.add_argument("--graph-feature-seed", type=int, default=17)
    parser.add_argument("--graph-structural-rp", action="store_true")
    parser.add_argument("--graph-multihop", action="store_true")
    parser.add_argument("--force-graph-cache-rebuild", action="store_true")
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_ready_root = Path(args.training_ready_root)
    defaults = default_derived_paths(training_ready_root, args.dataset_group)
    paths = {
        "protein_embedding": Path(args.protein_embedding_path) if args.protein_embedding_path else defaults["protein_embedding"],
        "drug_embedding": Path(args.drug_embedding_path) if args.drug_embedding_path else defaults["drug_embedding"],
        "ppi_matrix": Path(args.ppi_matrix_path) if args.ppi_matrix_path else defaults["ppi_matrix"],
        "pdi_matrix": Path(args.pdi_matrix_path) if args.pdi_matrix_path else defaults["pdi_matrix"],
        "ddi_matrix": Path(args.ddi_matrix_path) if args.ddi_matrix_path else defaults["ddi_matrix"],
    }
    require_existing(paths)

    started_at = iso_now()
    print(
        "[graph-cache] prebuilding "
        f"dataset_group={args.dataset_group} dim={args.graph_feature_dim} seed={args.graph_feature_seed} "
        f"structural_rp={bool(args.graph_structural_rp)} multihop={bool(args.graph_multihop)}"
    )
    protein_embedding = load_embedding_matrix(paths["protein_embedding"])
    drug_embedding = load_embedding_matrix(paths["drug_embedding"])
    features, meta = build_or_load_graph_features(
        cache_dir=args.graph_cache_dir,
        dataset_group=args.dataset_group,
        ppi_matrix_path=paths["ppi_matrix"],
        pdi_matrix_path=paths["pdi_matrix"],
        ddi_matrix_path=paths["ddi_matrix"],
        protein_embedding=protein_embedding,
        drug_embedding=drug_embedding,
        graph_feature_dim=args.graph_feature_dim,
        seed=args.graph_feature_seed,
        include_structural_rp=args.graph_structural_rp,
        include_multihop=args.graph_multihop,
        force_rebuild=args.force_graph_cache_rebuild,
    )
    sample = np.asarray(features[: min(int(features.shape[0]), 1024)], dtype=np.float32)
    if sample.size and not np.isfinite(sample).all():
        raise ValueError("graph feature cache contains non-finite values in the validation sample")

    completed_at = iso_now()
    summary: dict[str, Any] = {
        "started_at": started_at,
        "completed_at": completed_at,
        "dataset_group": args.dataset_group,
        "graph_cache_dir": str(Path(args.graph_cache_dir).resolve()),
        "graph_feature_shape": list(map(int, features.shape)),
        "graph_feature_dtype": str(features.dtype),
        "graph_feature_path": meta.get("feature_path"),
        "graph_feature_slices": meta.get("feature_slices"),
        "graph_structural_rp": bool(args.graph_structural_rp),
        "graph_multihop": bool(args.graph_multihop),
        "source_paths": {name: str(path.resolve()) for name, path in paths.items()},
    }
    if args.summary_json:
        dump_json(Path(args.summary_json), json_safe(summary))
    print(
        "[graph-cache] ready "
        f"path={summary['graph_feature_path']} shape={tuple(summary['graph_feature_shape'])} "
        f"summary={args.summary_json or 'none'}"
    )


if __name__ == "__main__":
    main()
