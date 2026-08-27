#!/usr/bin/env python3
"""Build PTV1-only graph matrices aligned to PTV1 global metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _shared import REPO_ROOT, dump_json, iso_now, load_json, load_repo_module


graph = load_repo_module("utils/05_build_graph_matrices_from_global_meta.py", "ptv_graph_shared")

DEFAULT_GLOBAL_META = REPO_ROOT / "data" / "training_ready" / "ptv1" / "global_meta.json"
DEFAULT_DERIVED_DIR = REPO_ROOT / "data" / "training_ready" / "ptv1" / "derived"
DEFAULT_PPI_EDGE_PATH = Path("/root/beam_wuhao/H100/vcc_data/westlake/20250410_6508308PPI_protein_links_detailed_v12_both_prot1&2_.csv")


def augment_matrix_meta(matrix_path: Path, *, source_meta: Path) -> None:
    matrix = np.load(matrix_path)
    meta_path = matrix_path.with_suffix(".meta.json")
    if meta_path.exists():
        payload = load_json(meta_path)
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {}
    finite = np.isfinite(matrix)
    payload.update(
        {
            "generated_at": payload.get("generated_at", iso_now()),
            "source_global_meta": str(source_meta),
            "shape": list(matrix.shape),
            "finite_value_count": int(finite.sum()),
            "nonzero_count": int(np.count_nonzero(matrix)),
            "min": float(np.nanmin(matrix)) if matrix.size else None,
            "max": float(np.nanmax(matrix)) if matrix.size else None,
        }
    )
    dump_json(meta_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-meta", type=Path, default=DEFAULT_GLOBAL_META)
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    parser.add_argument("--ppi-edge-path", type=Path, default=DEFAULT_PPI_EDGE_PATH)
    parser.add_argument("--ppi-node-mapping-json", type=Path, default=None)
    parser.add_argument("--ppi-topk", type=int, default=0)
    parser.add_argument("--stitch-db-dir", type=Path, default=graph.DEFAULT_STITCH_DB_DIR)
    parser.add_argument("--pdi-links-path", type=Path, default=None)
    parser.add_argument("--pdi-pert-to-flat-json", type=Path, default=None)
    parser.add_argument("--pdi-protein-node-mapping-json", type=Path, default=None)
    parser.add_argument("--pdi-protein-mapping-db", type=Path, default=None)
    parser.add_argument("--pdi-chemical-inchikey-tsv", type=Path, default=None)
    parser.add_argument("--pdi-chunksize", type=int, default=500_000)
    parser.add_argument("--drug-radius", type=int, default=2)
    parser.add_argument("--drug-n-bits", type=int, default=2048)
    parser.add_argument("--skip-ppi", action="store_true")
    parser.add_argument("--skip-pdi", action="store_true")
    parser.add_argument("--skip-ddi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_json(args.global_meta)
    if not isinstance(meta, dict) or meta.get("dataset_group") != "ptv1":
        raise ValueError(f"--global-meta must point to PTV1 global_meta.json: {args.global_meta}")
    args.derived_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_ppi:
        output = args.derived_dir / "ppi_matrix.npy"
        graph.build_ppi_matrix(
            meta=meta,
            edge_path=args.ppi_edge_path,
            output_path=output,
            node_mapping_json=args.ppi_node_mapping_json,
            allow_online_mapping=False,
            topk=args.ppi_topk,
        )
        augment_matrix_meta(output, source_meta=args.global_meta)
        print(f"[graph] wrote {output}")

    if not args.skip_ddi:
        output = args.derived_dir / "ddi_matrix.npy"
        graph.build_ddi_matrix(meta=meta, output_path=output, radius=args.drug_radius, n_bits=args.drug_n_bits)
        augment_matrix_meta(output, source_meta=args.global_meta)
        print(f"[graph] wrote {output}")

    if not args.skip_pdi:
        output = args.derived_dir / "pdi_matrix.npy"
        graph.build_pdi_matrix(
            meta=meta,
            links_path=args.pdi_links_path,
            output_path=output,
            stitch_db_dir=args.stitch_db_dir,
            pert_to_flat_json=args.pdi_pert_to_flat_json,
            protein_node_mapping_json=args.pdi_protein_node_mapping_json,
            protein_mapping_db=args.pdi_protein_mapping_db,
            chemical_inchikey_tsv=args.pdi_chemical_inchikey_tsv,
            allow_online_protein_mapping=False,
            chunksize=args.pdi_chunksize,
        )
        augment_matrix_meta(output, source_meta=args.global_meta)
        print(f"[graph] wrote {output}")


if __name__ == "__main__":
    main()
