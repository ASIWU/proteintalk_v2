#!/usr/bin/env python3
"""Build PTV1-only drug and protein embedding artifacts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from _shared import REPO_ROOT, dump_json, iso_now, load_json, load_repo_module


emb = load_repo_module("utils/04_build_embeddings_from_global_meta.py", "ptv_embedding_shared")

DEFAULT_GLOBAL_META = REPO_ROOT / "data" / "training_ready" / "ptv1" / "global_meta.json"
DEFAULT_DERIVED_DIR = REPO_ROOT / "data" / "training_ready" / "ptv1" / "derived"
DEFAULT_FASTA = REPO_ROOT / "data" / "training_ready" / "ptv3" / "derived" / "idmapping_2026_04_27.fasta"
DEFAULT_ESM_MODEL = Path(
    "/mnt/shared-storage-user/beam/wuhao/hf_cache/models--facebook--esm2_t33_650M_UR50D/"
    "snapshots/08e4846e537177426273712802403f7ba8261b6c"
)


def write_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_embedding_meta(path: Path, payload: dict[str, Any], *, source_meta: Path) -> None:
    matrix = np.asarray(payload["embedding_matrix"])
    meta = {
        "generated_at": iso_now(),
        "source_global_meta": str(source_meta),
        "kind": payload.get("kind"),
        "embedding_name": payload.get("embedding_name"),
        "shape": list(matrix.shape),
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else None,
        "finite_value_count": int(np.isfinite(matrix).sum()),
        "nonzero_count": int(np.count_nonzero(matrix)),
        "fallback_item_count": int(
            len(payload.get("sequence_fallback_items", {})) + len(payload.get("smiles_fallback_items", {}))
        ),
        "sequence_fallback_items": payload.get("sequence_fallback_items", {}),
        "smiles_fallback_items": payload.get("smiles_fallback_items", {}),
        "model_name": payload.get("model_name"),
        "device_used": payload.get("device_used"),
    }
    dump_json(path.with_suffix(".meta.json"), meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-meta", type=Path, default=DEFAULT_GLOBAL_META)
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--model-name", type=Path, default=DEFAULT_ESM_MODEL)
    parser.add_argument("--protein-batch-size", type=int, default=4)
    parser.add_argument("--protein-max-length", type=int, default=1024)
    parser.add_argument("--drug-radius", type=int, default=2)
    parser.add_argument("--drug-n-bits", type=int, default=2048)
    parser.add_argument("--skip-protein", action="store_true")
    parser.add_argument("--skip-drug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_json(args.global_meta)
    if not isinstance(meta, dict) or meta.get("dataset_group") != "ptv1":
        raise ValueError(f"--global-meta must point to PTV1 global_meta.json: {args.global_meta}")
    args.derived_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_drug:
        payload = emb.build_drug_embedding_payload(meta=meta, radius=args.drug_radius, n_bits=args.drug_n_bits)
        output = args.derived_dir / "drug_embedding_morgan_2048.pkl"
        write_pickle(output, payload)
        write_embedding_meta(output, payload, source_meta=args.global_meta)
        print(f"[embedding] wrote {output}")

    if not args.skip_protein:
        sequence_lookup = emb.load_sequences_from_fasta(args.fasta)
        payload = emb.build_protein_embedding_payload(
            meta=meta,
            sequence_lookup=sequence_lookup,
            model_name=str(args.model_name),
            batch_size=args.protein_batch_size,
            max_length=args.protein_max_length,
        )
        output = args.derived_dir / "protein_embedding_esm.pkl"
        write_pickle(output, payload)
        write_embedding_meta(output, payload, source_meta=args.global_meta)
        print(f"[embedding] wrote {output}")


if __name__ == "__main__":
    main()
