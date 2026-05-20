#!/usr/bin/env python3
"""Build PPI/DDI/PDI matrices aligned to training-ready global metadata."""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


DEFAULT_STITCH_DB_DIR = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/stitch_db")
STITCH_LINK_FILENAMES = (
    "protein_chemical.links.detailed.v5.0.parquet",
    "protein_chemical.links.detailed.v5.0.tsv",
)
STITCH_INCHIKEY_FILENAME = "chemicals.inchikeys.v5.0.tsv"
STITCH_UNIPROT_SQLITE_FILENAME = "uniprot_to_string.db"
PDI_SCORE_COLUMNS = ("combined_score", "experimental", "database", "prediction", "score")


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def ordered_ids(index_mapping: dict[str, int]) -> list[str]:
    return [item for item, _ in sorted(index_mapping.items(), key=lambda pair: pair[1])]


def mol_from_smiles_or_empty(pert_id: str, smiles: str) -> tuple[Any, str | None]:
    smiles = str(smiles or "").strip()
    if pert_id == "no":
        fallback_reason = "special_value_empty_smiles"
    elif not smiles:
        fallback_reason = "missing_smiles_empty_smiles"
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol, None
        fallback_reason = "invalid_smiles_empty_smiles"

    mol = Chem.MolFromSmiles("")
    if mol is None:
        raise RuntimeError("RDKit failed to create fallback molecule from empty SMILES.")
    return mol, fallback_reason


def normalize_score(value: object) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score > 1.0:
        score = score / 1000.0
    return float(min(max(score, 0.0), 1.0))


def load_table(path: Path, *, sep: str | None = None, chunksize: int | None = None):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if chunksize is not None:
            raise ValueError("chunked parquet loading is not supported by this script")
        return pd.read_parquet(path)
    if sep is None:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False, chunksize=chunksize)


def detect_edge_columns(columns: Iterable[str]) -> tuple[str, str]:
    columns = list(columns)
    for src, dst in (("prot1", "prot2"), ("protein1", "protein2"), ("source", "target")):
        if src in columns and dst in columns:
            return src, dst
    raise ValueError(f"Could not detect edge columns from {columns}")


def load_mapping_json(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = load_json(path)
    return {str(key): str(value) for key, value in payload.items()}


def resolve_existing_path(explicit_path: Path | None, candidates: Iterable[Path], *, description: str) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"{description} does not exist: {explicit_path}")
        return explicit_path

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {description}. Checked: {checked}")


def resolve_optional_path(explicit_path: Path | None, candidates: Iterable[Path], *, description: str) -> Path | None:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"{description} does not exist: {explicit_path}")
        return explicit_path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_delimited_sep(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tsv") or name.endswith(".tsv.gz") or name.endswith(".txt") or name.endswith(".txt.gz"):
        return "\t"
    return ","


def detect_pdi_score_column(columns: Iterable[str]) -> str:
    column_set = set(columns)
    for candidate in PDI_SCORE_COLUMNS:
        if candidate in column_set:
            return candidate
    raise ValueError(
        "PDI links table must contain one of "
        f"{', '.join(PDI_SCORE_COLUMNS)}. Available columns: {sorted(column_set)}"
    )


def open_pdi_link_chunks(path: Path, *, chunksize: int) -> tuple[Iterable[pd.DataFrame], str, str, int | None]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Streaming PDI parquet input requires `pyarrow` to be installed.") from exc

        parquet_file = pq.ParquetFile(path)
        columns = list(parquet_file.schema.names)
        expected = {"chemical", "protein"}
        if not expected.issubset(columns):
            raise ValueError("PDI links parquet must contain `chemical` and `protein` columns.")
        score_column = detect_pdi_score_column(columns)
        read_columns = ["chemical", "protein", score_column]

        def iter_row_groups() -> Iterable[pd.DataFrame]:
            for row_group_index in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_index, columns=read_columns)
                if table.num_rows:
                    yield table.to_pandas()

        return iter_row_groups(), score_column, "parquet", parquet_file.num_row_groups

    sep = infer_delimited_sep(path)
    header = pd.read_csv(path, sep=sep, nrows=0)
    expected = {"chemical", "protein"}
    if not expected.issubset(header.columns):
        raise ValueError("PDI links table must contain `chemical` and `protein` columns.")
    score_column = detect_pdi_score_column(header.columns)
    reader = pd.read_csv(
        path,
        sep=sep,
        low_memory=False,
        chunksize=chunksize,
        usecols=["chemical", "protein", score_column],
    )
    return reader, score_column, "delimited", None


def map_uniprot_to_string_online(uniprot_ids: list[str], *, batch_size: int = 100) -> dict[str, str]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Online STRING mapping requires `requests` to be installed.") from exc

    api_url = "https://string-db.org/api/tsv/get_string_ids"
    mapping: dict[str, str] = {}
    unique_ids = sorted(set(uniprot_ids))
    for start in tqdm(range(0, len(unique_ids), batch_size), desc="STRING ID batches"):
        batch = unique_ids[start : start + batch_size]
        params = {
            "identifiers": "\r".join(batch),
            "limit": 1,
            "echo_query": 1,
        }
        for attempt in range(3):
            response = requests.post(api_url, data=params, timeout=60)
            if response.status_code == 200:
                frame = pd.read_csv(io.StringIO(response.text), sep="\t")
                for _, row in frame.iterrows():
                    mapping[str(row["queryItem"])] = str(row["stringId"])
                break
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    return mapping


def load_uniprot_to_string_from_sqlite(
    uniprot_ids: list[str],
    db_path: Path,
    *,
    batch_size: int = 500,
) -> dict[str, str]:
    unique_ids = sorted(set(str(item) for item in uniprot_ids if item))
    mapping: dict[str, str] = {}
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        table_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(mapping)")
        }
        expected_columns = {"alias", "string_protein_id"}
        if not expected_columns.issubset(table_columns):
            raise ValueError(
                f"{db_path} must contain table `mapping(alias, string_protein_id)`. "
                f"Found columns: {sorted(table_columns)}"
            )

        for start in tqdm(range(0, len(unique_ids), batch_size), desc="SQLite UniProt mappings"):
            batch = unique_ids[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT alias, string_protein_id FROM mapping WHERE alias IN ({placeholders})"
            for alias, string_protein_id in connection.execute(query, batch):
                mapping.setdefault(str(alias), str(string_protein_id))
    finally:
        connection.close()
    return mapping


def build_reverse_mapping_lists(item_to_node: dict[str, str], item_to_index: dict[str, int]) -> dict[str, list[int]]:
    reverse_mapping: dict[str, list[int]] = defaultdict(list)
    for item, node_id in item_to_node.items():
        if not node_id or item not in item_to_index:
            continue
        reverse_mapping[str(node_id)].append(int(item_to_index[item]))
    for node_id in reverse_mapping:
        reverse_mapping[node_id] = sorted(set(reverse_mapping[node_id]))
    return dict(reverse_mapping)


def build_ppi_matrix(
    *,
    meta: dict[str, Any],
    edge_path: Path,
    output_path: Path,
    node_mapping_json: Path | None,
    allow_online_mapping: bool,
    topk: int,
) -> None:
    protein_order = ordered_ids(meta["protein_index"])
    matrix = np.zeros((len(protein_order), len(protein_order)), dtype=np.float32)
    node_mapping = load_mapping_json(node_mapping_json)

    real_proteins = [protein for protein in protein_order if protein not in {"control", "no"}]
    if not node_mapping and allow_online_mapping:
        node_mapping = map_uniprot_to_string_online(real_proteins)

    protein_to_node = {protein: node_mapping.get(protein, protein) for protein in real_proteins}
    node_to_protein_indices = build_reverse_mapping_lists(protein_to_node, meta["protein_index"])

    def write_ppi_output(
        *,
        warning: str,
        grouped_edge_count: int,
        score_summary: dict[str, Any] | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, matrix)
        dump_json(
            output_path.with_suffix(".meta.json"),
            {
                "kind": "ppi_matrix",
                "shape": list(matrix.shape),
                "row_axis": "protein_index",
                "col_axis": "protein_index",
                "protein_count": len(protein_order),
                "mapped_protein_count": sum(len(indices) for indices in node_to_protein_indices.values()),
                "topk": topk,
                "topk_policy": "disabled" if topk <= 0 else "keep strongest topk neighbors per protein row",
                "grouped_edge_count": grouped_edge_count,
                "warning": warning,
                "score_filter": {
                    "applied": False,
                    "note": "No PPI score threshold/filter is applied. All metadata-mapped non-self edges are written unless an explicit positive --topk is provided.",
                },
                "score_summary": score_summary or {},
                "self_loop_policy": "dropped",
                "edge_path": str(edge_path),
                "node_mapping_json": str(node_mapping_json) if node_mapping_json else "",
            },
        )

    frame = load_table(edge_path)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.concat(list(frame), ignore_index=True)
    src_col, dst_col = detect_edge_columns(frame.columns)

    channels = ["experimental", "database", "coexpression", "neighborhood", "fusion", "textmining"]
    if "cooccurrence" in frame.columns:
        channels.append("cooccurrence")
    if "cooccurence" in frame.columns:
        channels.append("cooccurence")
    alpha = {
        "experimental": 1.0,
        "database": 1.0,
        "coexpression": 0.8,
        "cooccurrence": 0.5,
        "cooccurence": 0.5,
        "fusion": 0.3,
        "neighborhood": 0.3,
        "textmining": 0.2,
    }

    keep_rows = frame[src_col].astype(str).isin(node_to_protein_indices) & frame[dst_col].astype(str).isin(node_to_protein_indices)
    frame = frame.loc[keep_rows].copy()
    if frame.empty:
        write_ppi_output(
            warning="no_ppi_edges_after_metadata_filter",
            grouped_edge_count=0,
        )
        return

    probabilities = []
    for channel in channels:
        if channel not in frame.columns:
            continue
        channel_scores = frame[channel].astype(np.float32).clip(lower=0) / 1000.0
        probabilities.append(alpha[channel] * channel_scores.to_numpy())
    if not probabilities:
        if "combined_score" not in frame.columns:
            raise ValueError("No supported PPI evidence channels were found in the edge file.")
        fused = frame["combined_score"].astype(np.float32).to_numpy() / 1000.0
    else:
        fused = 1.0 - np.prod(1.0 - np.vstack(probabilities), axis=0)
    if "combined_score" in frame.columns:
        fused = np.maximum(fused, frame["combined_score"].astype(np.float32).to_numpy() / 1000.0)
    fused = np.clip(fused, 0.0, 1.0)
    score_columns_used = [channel for channel in channels if channel in frame.columns]
    if "combined_score" in frame.columns:
        score_columns_used.append("combined_score")
    score_summary = {
        "source_scale": "STRING-like scores are read as 0-1000 values and divided by 1000 for matrix weights.",
        "score_columns_used": score_columns_used,
        "fused_score_min": float(np.min(fused)) if len(fused) else None,
        "fused_score_max": float(np.max(fused)) if len(fused) else None,
    }
    if "combined_score" in frame.columns:
        combined_scores = frame["combined_score"].astype(np.float32)
        score_summary["combined_score_min_raw"] = float(combined_scores.min())
        score_summary["combined_score_max_raw"] = float(combined_scores.max())
        score_summary["combined_score_min_scaled"] = float(combined_scores.min() / 1000.0)
        score_summary["combined_score_max_scaled"] = float(combined_scores.max() / 1000.0)

    grouped_weights: dict[tuple[int, int], float] = {}
    for src_node, dst_node, weight in tqdm(
        zip(frame[src_col].tolist(), frame[dst_col].tolist(), fused.tolist()),
        total=len(frame),
        desc="PPI edges",
    ):
        src_indices = node_to_protein_indices[str(src_node)]
        dst_indices = node_to_protein_indices[str(dst_node)]
        if str(src_node) == str(dst_node):
            continue
        for src_index in src_indices:
            for dst_index in dst_indices:
                if src_index == dst_index:
                    continue
                left, right = sorted((src_index, dst_index))
                key = (left, right)
                grouped_weights[key] = max(grouped_weights.get(key, 0.0), float(weight))

    per_row_neighbors: dict[int, list[tuple[int, float]]] = {}
    for (left, right), weight in grouped_weights.items():
        per_row_neighbors.setdefault(left, []).append((right, weight))
        per_row_neighbors.setdefault(right, []).append((left, weight))

    for row_index, neighbors in tqdm(per_row_neighbors.items(), desc="PPI top-k" if topk > 0 else "PPI matrix"):
        if topk > 0:
            neighbors = sorted(neighbors, key=lambda item: item[1], reverse=True)[:topk]
        for col_index, weight in neighbors:
            matrix[row_index, col_index] = max(matrix[row_index, col_index], weight)
            matrix[col_index, row_index] = max(matrix[col_index, row_index], weight)

    write_ppi_output(
        warning="",
        grouped_edge_count=len(grouped_weights),
        score_summary=score_summary,
    )


def build_ddi_matrix(
    *,
    meta: dict[str, Any],
    output_path: Path,
    radius: int,
    n_bits: int,
) -> None:
    pert_order = ordered_ids(meta["pert_index"])
    fingerprints: list[Any | None] = [None] * len(pert_order)
    smiles_fallback_items: dict[str, str] = {}
    fallback_indices: set[int] = set()
    fingerprint_generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    for pert_id in tqdm(pert_order, desc="DDI fingerprints"):
        pert_index = int(meta["pert_index"][pert_id])
        smiles = meta["pertid_to_smiles"].get(pert_id, "")
        mol, fallback_reason = mol_from_smiles_or_empty(pert_id, smiles)
        if fallback_reason is not None:
            smiles_fallback_items[pert_id] = fallback_reason
            fallback_indices.add(pert_index)
        fingerprints[pert_index] = fingerprint_generator.GetFingerprint(mol)

    matrix = np.zeros((len(pert_order), len(pert_order)), dtype=np.float32)
    for row_index in tqdm(range(len(pert_order)), desc="DDI rows"):
        fp = fingerprints[row_index]
        if fp is None:
            continue
        matrix[row_index, row_index] = 1.0
        valid_cols = [col for col in range(row_index + 1, len(pert_order)) if fingerprints[col] is not None]
        if not valid_cols:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, [fingerprints[col] for col in valid_cols])
        for col_index, similarity in zip(valid_cols, sims):
            if row_index in fallback_indices and col_index in fallback_indices:
                similarity = 0.0
            matrix[row_index, col_index] = float(similarity)
            matrix[col_index, row_index] = float(similarity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
    dump_json(
        output_path.with_suffix(".meta.json"),
        {
            "kind": "ddi_matrix",
            "shape": list(matrix.shape),
            "row_axis": "pert_index",
            "col_axis": "pert_index",
            "pert_count": len(pert_order),
            "fingerprint_count": sum(fingerprint is not None for fingerprint in fingerprints),
            "radius": radius,
            "n_bits": n_bits,
            "smiles_fallback_items": smiles_fallback_items,
            "fallback_similarity_policy": "fallback-vs-fallback off-diagonal similarity is forced to 0.0; diagonal remains 1.0",
        },
    )


def smiles_to_inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def build_pert_to_flat_chemical_id(meta: dict[str, Any], inchikey_tsv: Path) -> dict[str, str]:
    pert_to_inchikey = {}
    for pert_id in tqdm(ordered_ids(meta["pert_index"]), desc="InChIKey"):
        smiles = meta["pertid_to_smiles"].get(pert_id, "")
        if pert_id == "no" or not smiles:
            continue
        inchikey = smiles_to_inchikey(smiles)
        if inchikey:
            pert_to_inchikey[pert_id] = inchikey

    needed = set(pert_to_inchikey.values())
    if not needed:
        return {}
    inchikey_to_flat: dict[str, str] = {}
    header = pd.read_csv(inchikey_tsv, sep="\t", nrows=0)
    if "flat_chemical_id" not in header.columns or "inchikey" not in header.columns:
        raise ValueError("chemical InChIKey TSV must contain `flat_chemical_id` and `inchikey` columns")
    reader = pd.read_csv(
        inchikey_tsv,
        sep="\t",
        low_memory=False,
        chunksize=500_000,
        usecols=["flat_chemical_id", "inchikey"],
    )
    for chunk in tqdm(reader, desc="chemical InChIKey chunks"):
        matched = chunk.loc[chunk["inchikey"].isin(needed), ["inchikey", "flat_chemical_id"]]
        for inchikey, flat_id in matched.itertuples(index=False):
            inchikey_to_flat.setdefault(str(inchikey), str(flat_id))
        if len(inchikey_to_flat) == len(needed):
            break

    return {
        pert_id: inchikey_to_flat[inchikey]
        for pert_id, inchikey in pert_to_inchikey.items()
        if inchikey in inchikey_to_flat
    }


def build_pdi_matrix(
    *,
    meta: dict[str, Any],
    links_path: Path | None,
    output_path: Path,
    stitch_db_dir: Path,
    pert_to_flat_json: Path | None,
    protein_node_mapping_json: Path | None,
    protein_mapping_db: Path | None,
    chemical_inchikey_tsv: Path | None,
    allow_online_protein_mapping: bool,
    chunksize: int,
) -> None:
    pert_order = ordered_ids(meta["pert_index"])
    protein_order = ordered_ids(meta["protein_index"])
    matrix = np.zeros((len(pert_order), len(protein_order)), dtype=np.float32)
    stitch_db_dir = stitch_db_dir.expanduser()
    resolved_links_path = resolve_existing_path(
        links_path,
        (stitch_db_dir / filename for filename in STITCH_LINK_FILENAMES),
        description="PDI chemical-protein links table",
    )

    if pert_to_flat_json is not None:
        pert_to_flat = load_mapping_json(pert_to_flat_json)
        pert_mapping_source = str(pert_to_flat_json)
        resolved_inchikey_tsv = None
    else:
        resolved_inchikey_tsv = resolve_existing_path(
            chemical_inchikey_tsv,
            [stitch_db_dir / STITCH_INCHIKEY_FILENAME],
            description="STITCH chemical InChIKey TSV",
        )
        pert_to_flat = build_pert_to_flat_chemical_id(meta, resolved_inchikey_tsv)
        pert_mapping_source = str(resolved_inchikey_tsv)

    real_proteins = [protein for protein in protein_order if protein not in {"control", "no"}]
    protein_to_node = load_mapping_json(protein_node_mapping_json)
    protein_mapping_source = str(protein_node_mapping_json) if protein_node_mapping_json else ""
    resolved_protein_mapping_db = None
    if not protein_to_node:
        resolved_protein_mapping_db = resolve_optional_path(
            protein_mapping_db,
            [stitch_db_dir / STITCH_UNIPROT_SQLITE_FILENAME],
            description="UniProt-to-STRING SQLite mapping database",
        )
        if resolved_protein_mapping_db is not None:
            protein_to_node = load_uniprot_to_string_from_sqlite(real_proteins, resolved_protein_mapping_db)
            protein_mapping_source = str(resolved_protein_mapping_db)
    if not protein_to_node and allow_online_protein_mapping:
        protein_to_node = map_uniprot_to_string_online(real_proteins)
        protein_mapping_source = "STRING online API"
    if not protein_to_node:
        protein_to_node = {protein: protein for protein in real_proteins}
        protein_mapping_source = "identity_fallback"

    chemical_to_pert_indices = build_reverse_mapping_lists(pert_to_flat, meta["pert_index"])
    protein_node_to_indices = build_reverse_mapping_lists(protein_to_node, meta["protein_index"])
    chemical_nodes = set(chemical_to_pert_indices)
    protein_nodes = set(protein_node_to_indices)

    reader, score_column, links_format, total_chunks = open_pdi_link_chunks(resolved_links_path, chunksize=chunksize)

    matched_link_count = 0
    scanned_link_count = 0
    for chunk in tqdm(reader, desc="PDI chunks", total=total_chunks):
        scanned_link_count += int(len(chunk))
        filtered = chunk.loc[
            chunk["chemical"].astype(str).isin(chemical_nodes)
            & chunk["protein"].astype(str).isin(protein_nodes),
            ["chemical", "protein", score_column],
        ]
        matched_link_count += int(len(filtered))
        for chemical, protein, score in filtered.itertuples(index=False):
            weight = normalize_score(score)
            for pert_index in chemical_to_pert_indices[str(chemical)]:
                for protein_index in protein_node_to_indices[str(protein)]:
                    matrix[pert_index, protein_index] = max(matrix[pert_index, protein_index], weight)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
    dump_json(
        output_path.with_suffix(".meta.json"),
        {
            "kind": "pdi_matrix",
            "shape": list(matrix.shape),
            "row_axis": "pert_index",
            "col_axis": "protein_index",
            "pert_count": len(pert_order),
            "protein_count": len(protein_order),
            "mapped_pert_count": sum(len(indices) for indices in chemical_to_pert_indices.values()),
            "mapped_protein_count": sum(len(indices) for indices in protein_node_to_indices.values()),
            "unmapped_pert_count": len(pert_order) - sum(len(indices) for indices in chemical_to_pert_indices.values()),
            "unmapped_protein_count": len(protein_order) - sum(len(indices) for indices in protein_node_to_indices.values()),
            "nonzero_count": int(np.count_nonzero(matrix)),
            "score_column": score_column,
            "scanned_link_count": scanned_link_count,
            "matched_link_count": matched_link_count,
            "links_path": str(resolved_links_path),
            "links_format": links_format,
            "stitch_db_dir": str(stitch_db_dir),
            "pert_to_flat_json": str(pert_to_flat_json) if pert_to_flat_json else "",
            "pert_mapping_source": pert_mapping_source,
            "chemical_inchikey_tsv": str(resolved_inchikey_tsv) if resolved_inchikey_tsv else "",
            "protein_node_mapping_json": str(protein_node_mapping_json) if protein_node_mapping_json else "",
            "protein_mapping_db": str(resolved_protein_mapping_db) if resolved_protein_mapping_db else "",
            "protein_mapping_source": protein_mapping_source,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph matrices aligned to training-ready global metadata")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ppi_parser = subparsers.add_parser("ppi", help="Build a protein-protein interaction matrix")
    ppi_parser.add_argument("--global-meta", type=Path, required=True)
    ppi_parser.add_argument("--edge-path", type=Path, required=True)
    ppi_parser.add_argument("--output-npy", type=Path, required=True)
    ppi_parser.add_argument("--node-mapping-json", type=Path, default=None, help="Optional UniProt -> edge-node-id mapping JSON")
    ppi_parser.add_argument("--allow-online-mapping", action="store_true", help="Allow STRING API mapping when a local mapping JSON is not provided")
    ppi_parser.add_argument("--topk", type=int, default=0, help="Keep strongest K neighbors per protein row; 0 disables top-k pruning.")

    ddi_parser = subparsers.add_parser("ddi", help="Build a drug-drug similarity matrix")
    ddi_parser.add_argument("--global-meta", type=Path, required=True)
    ddi_parser.add_argument("--output-npy", type=Path, required=True)
    ddi_parser.add_argument("--radius", type=int, default=2)
    ddi_parser.add_argument("--n-bits", type=int, default=2048)

    pdi_parser = subparsers.add_parser("pdi", help="Build a perturbation-drug interaction matrix")
    pdi_parser.add_argument("--global-meta", type=Path, required=True)
    pdi_parser.add_argument("--output-npy", type=Path, required=True)
    pdi_parser.add_argument("--stitch-db-dir", type=Path, default=DEFAULT_STITCH_DB_DIR, help="Directory containing STITCH link, InChIKey, and mapping files")
    pdi_parser.add_argument("--links-path", type=Path, default=None, help="Optional STITCH chemical-protein links table; supports .parquet, .tsv, .csv")
    pdi_parser.add_argument("--links-tsv", type=Path, default=None, help="Backward-compatible alias for --links-path")
    pdi_parser.add_argument("--pert-to-flat-json", type=Path, default=None, help="Optional pert_id -> flat_chemical_id mapping JSON")
    pdi_parser.add_argument("--protein-node-mapping-json", type=Path, default=None, help="Optional UniProt -> protein-node-id mapping JSON")
    pdi_parser.add_argument("--protein-mapping-db", type=Path, default=None, help="Optional SQLite DB with mapping(alias, string_protein_id)")
    pdi_parser.add_argument("--chemical-inchikey-tsv", type=Path, default=None, help="Optional chemicals.inchikeys TSV; defaults to --stitch-db-dir")
    pdi_parser.add_argument("--allow-online-protein-mapping", action="store_true", help="Allow STRING API mapping when a local protein-node mapping JSON is not provided")
    pdi_parser.add_argument("--chunksize", type=int, default=500_000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_json(args.global_meta)
    if args.command == "ppi":
        if args.topk < 0:
            raise ValueError("--topk must be >= 0. Use 0 to disable top-k pruning.")
        build_ppi_matrix(
            meta=meta,
            edge_path=args.edge_path,
            output_path=args.output_npy,
            node_mapping_json=args.node_mapping_json,
            allow_online_mapping=args.allow_online_mapping,
            topk=args.topk,
        )
    elif args.command == "ddi":
        build_ddi_matrix(
            meta=meta,
            output_path=args.output_npy,
            radius=args.radius,
            n_bits=args.n_bits,
        )
    else:
        if args.links_path is not None and args.links_tsv is not None and args.links_path != args.links_tsv:
            raise ValueError("Use only one of --links-path or --links-tsv for PDI.")
        build_pdi_matrix(
            meta=meta,
            links_path=args.links_path or args.links_tsv,
            output_path=args.output_npy,
            stitch_db_dir=args.stitch_db_dir,
            pert_to_flat_json=args.pert_to_flat_json,
            protein_node_mapping_json=args.protein_node_mapping_json,
            protein_mapping_db=args.protein_mapping_db,
            chemical_inchikey_tsv=args.chemical_inchikey_tsv,
            allow_online_protein_mapping=args.allow_online_protein_mapping,
            chunksize=args.chunksize,
        )


if __name__ == "__main__":
    main()
