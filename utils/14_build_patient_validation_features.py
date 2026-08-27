#!/usr/bin/env python3
"""Build derived features for the patientVali260605v3 inference root."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready_patientVali260605v3"
DEFAULT_SOURCE_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
DATASET_GROUP = "ptv3"
RUN_NAME = "patientVali260605v3"
DEFAULT_FASTA = REPO_ROOT / "data" / "training_ready" / "ptv3" / "derived" / "idmapping_2026_04_27.fasta"
DEFAULT_ESM_MODEL = Path(
    "/mnt/shared-storage-user/beam/wuhao/hf_cache/models--facebook--esm2_t33_650M_UR50D/"
    "snapshots/08e4846e537177426273712802403f7ba8261b6c"
)
DEFAULT_PPI_EDGE_PATH = Path(
    "/root/beam_wuhao/H100/vcc_data/westlake/20250410_6508308PPI_protein_links_detailed_v12_both_prot1&2_.csv"
)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_repo_module(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


emb = load_repo_module("utils/04_build_embeddings_from_global_meta.py", "ptv_patient_embedding_shared")
graph = load_repo_module("utils/05_build_graph_matrices_from_global_meta.py", "ptv_patient_graph_shared")
cell_llm = load_repo_module("utils/11_build_cell_llm_embeddings.py", "ptv_patient_cell_llm_shared")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def ordered_ids(index_mapping: dict[str, int]) -> list[str]:
    return [item for item, _ in sorted(index_mapping.items(), key=lambda pair: int(pair[1]))]


def write_embedding_meta(path: Path, payload: dict[str, Any], *, source_meta: Path, extra: dict[str, Any] | None = None) -> None:
    matrix = np.asarray(payload["embedding_matrix"])
    fallback = {}
    fallback.update(payload.get("sequence_fallback_items", {}))
    fallback.update(payload.get("smiles_fallback_items", {}))
    meta = {
        "generated_at": iso_now(),
        "source_global_meta": str(source_meta),
        "kind": payload.get("kind"),
        "embedding_name": payload.get("embedding_name"),
        "shape": list(matrix.shape),
        "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else None,
        "finite_value_count": int(np.isfinite(matrix).sum()),
        "nonzero_count": int(np.count_nonzero(matrix)),
        "fallback_item_count": int(len(fallback)),
        "sequence_fallback_items": payload.get("sequence_fallback_items", {}),
        "smiles_fallback_items": payload.get("smiles_fallback_items", {}),
        "model_name": payload.get("model_name"),
        "device_used": payload.get("device_used"),
    }
    if extra:
        meta.update(extra)
    dump_json(path.with_suffix(".meta.json"), meta)


def augment_matrix_meta(matrix_path: Path, *, source_meta: Path, extra: dict[str, Any] | None = None) -> None:
    matrix = np.load(matrix_path, mmap_mode="r")
    finite = np.isfinite(matrix)
    meta_path = matrix_path.with_suffix(".meta.json")
    payload = load_json(meta_path) if meta_path.exists() else {}
    if not isinstance(payload, dict):
        payload = {}
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
    if extra:
        payload.update(extra)
    dump_json(meta_path, payload)


def build_incremental_protein_embedding(
    *,
    meta: dict[str, Any],
    source_embedding_path: Path,
    output_path: Path,
    fasta_path: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    source_meta_path: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    source_payload = load_pickle(source_embedding_path)
    source_matrix = np.asarray(source_payload["embedding_matrix"], dtype=np.float32)
    source_index = {str(key): int(value) for key, value in source_payload["item_to_index"].items()}
    protein_ids = ordered_ids(meta["protein_index"])
    embedding_dim = int(source_matrix.shape[1])
    output_matrix = np.zeros((len(protein_ids), embedding_dim), dtype=np.float32)

    copied_count = 0
    missing_items: list[tuple[int, str]] = []
    for protein_id in protein_ids:
        target_index = int(meta["protein_index"][protein_id])
        source_idx = source_index.get(protein_id)
        if source_idx is None:
            missing_items.append((target_index, protein_id))
            continue
        output_matrix[target_index] = source_matrix[source_idx]
        copied_count += 1

    sequence_lookup = emb.load_sequences_from_fasta(fasta_path)
    sequence_fallback_items = dict(source_payload.get("sequence_fallback_items", {}))
    generated_count = 0
    if missing_items:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        batch_items: list[tuple[int, str, str]] = []
        with torch.no_grad():
            for target_index, protein_id in missing_items:
                sequence = sequence_lookup.get(protein_id, "")
                if not sequence:
                    sequence_fallback_items[protein_id] = "missing_sequence_empty_sequence"
                batch_items.append((target_index, protein_id, sequence))
                if len(batch_items) < batch_size:
                    continue
                emb._flush_protein_batch(
                    batch_items=batch_items,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_length,
                    embedding_matrix=output_matrix,
                )
                generated_count += len(batch_items)
                batch_items = []
            if batch_items:
                emb._flush_protein_batch(
                    batch_items=batch_items,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_length,
                    embedding_matrix=output_matrix,
                )
                generated_count += len(batch_items)
        device_used = str(device)
    else:
        device_used = str(source_payload.get("device_used", "not_used_no_missing_proteins"))

    payload = {
        "kind": "protein_embedding",
        "embedding_name": "esm_mean_pooling",
        "model_name": str(model_name),
        "item_to_index": meta["protein_index"],
        "index_to_item": protein_ids,
        "embedding_matrix": output_matrix,
        "unresolved_items": {},
        "sequence_fallback_items": sequence_fallback_items,
        "device_used": device_used,
        "incremental_source_embedding": str(source_embedding_path),
        "copied_existing_rows": copied_count,
        "generated_appended_rows": generated_count,
    }
    write_pickle(output_path, payload)
    write_embedding_meta(
        output_path,
        payload,
        source_meta=source_meta_path,
        extra={
            "incremental_source_embedding": str(source_embedding_path),
            "copied_existing_rows": copied_count,
            "generated_appended_rows": generated_count,
            "missing_sequence_appended_rows": int(
                sum(1 for _, protein_id in missing_items if protein_id in sequence_fallback_items)
            ),
        },
    )
    return {
        "output": str(output_path),
        "shape": list(output_matrix.shape),
        "copied_existing_rows": copied_count,
        "generated_appended_rows": generated_count,
        "sequence_fallback_count": len(sequence_fallback_items),
    }


def read_feature_table(task_dir: Path) -> pd.DataFrame:
    for name, reader in (
        ("feature_table.parquet", pd.read_parquet),
        ("feature_table.pkl", pd.read_pickle),
        ("feature_table.csv", lambda path: pd.read_csv(path, low_memory=False)),
    ):
        path = task_dir / name
        if path.exists():
            return reader(path)
    raise FileNotFoundError(f"missing feature_table under {task_dir}")


def concise_patient_description(cell_label: str, rows: pd.DataFrame) -> str:
    first = rows.iloc[0]
    dataset = str(first.get("source_patient_dataset", "patient validation dataset"))
    cell_type = str(first.get("cell_type_norm", first.get("cell_type", "no")))
    patient_id = str(first.get("patient_sample_id", cell_label))
    instruments = sorted({str(value) for value in rows.get("machineID_new", pd.Series(dtype=object)).dropna().tolist() if str(value)})
    combos = []
    if "raw_combo" in rows:
        combos = sorted({str(value) for value in rows["raw_combo"].dropna().tolist() if str(value).strip()})[:8]
    clinical_bits: list[str] = []
    for column in rows.columns:
        if not column.startswith("clinical__"):
            continue
        value = first.get(column)
        if pd.isna(value) or str(value).strip() == "":
            continue
        name = column.removeprefix("clinical__")
        if name in {"sample_id", "Patient_ID", "pert_id", "Drug_combination"}:
            continue
        clinical_bits.append(f"{name}={value}")
        if len(clinical_bits) >= 12:
            break
    return (
        f"Patient-derived validation proteomics sample {patient_id} from {dataset}. "
        f"Cell or tissue context: {cell_type}. "
        f"Instrument or batch context: {', '.join(instruments) if instruments else 'not specified'}. "
        f"Observed treatment queries: {', '.join(combos) if combos else 'baseline/control row'}. "
        f"Available clinical metadata: {'; '.join(clinical_bits) if clinical_bits else 'not specified'}."
    )


def collect_cell_llm_records(training_ready_root: Path, cell_llm_index: dict[str, int]) -> list[dict[str, Any]]:
    task_root = training_ready_root / DATASET_GROUP / "tasks"
    frames = []
    for task_dir in sorted(task_root.iterdir()):
        if task_dir.is_dir():
            frames.append(read_feature_table(task_dir))
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    records: list[dict[str, Any]] = []
    for cell_label, index in sorted(cell_llm_index.items(), key=lambda item: int(item[1])):
        if int(index) == 0:
            records.append(
                {
                    "index": 0,
                    "canonical_name": "no",
                    "raw_values": {"no": 0},
                    "row_count": 0,
                    "description": "Missing or unspecified patient Cell label. Reserved zero-vector row.",
                    "prompt": None,
                }
            )
            continue
        rows = all_rows.loc[all_rows["cell_llm_index"].astype(int).eq(int(index))]
        records.append(
            {
                "index": int(index),
                "canonical_name": str(cell_label),
                "raw_values": {str(cell_label): int(len(rows))},
                "row_count": int(len(rows)),
                "description": concise_patient_description(str(cell_label), rows) if len(rows) else str(cell_label),
                "prompt": "deterministic_patient_metadata_template",
            }
        )
    return records


def build_patient_cell_llm_embedding(
    *,
    training_ready_root: Path,
    output_path: Path,
    env_file: Path,
    embedding_model: str,
    embedding_batch_size: int,
    timeout: float,
    retries: int,
    force: bool,
) -> dict[str, Any]:
    sidecar_path = output_path.with_suffix(".json")
    if output_path.exists() and sidecar_path.exists() and not force:
        print(f"[patient-cell-llm] existing artifact kept: {output_path}")
        payload = np.load(output_path, allow_pickle=False)
        return {"output": str(output_path), "shape": list(payload["embedding_matrix"].shape), "kept_existing": True}

    cell_llm_index = load_json(training_ready_root / DATASET_GROUP / "derived" / f"{RUN_NAME}_cell_llm_index.json")
    records = collect_cell_llm_records(training_ready_root, {str(k): int(v) for k, v in cell_llm_index.items()})
    embed_records = [record for record in records if int(record["index"]) != 0]
    texts = [str(record["description"]) for record in embed_records]

    cell_llm.load_env_file(env_file)
    proxy_summary = cell_llm.sanitize_proxy_env()
    api_key = cell_llm.os.environ.get("openai_api_key") or cell_llm.os.environ.get("OPENAI_API_KEY")
    base_url = cell_llm.os.environ.get("openai_base_url") or cell_llm.os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("missing openai_api_key/openai_base_url in .env or environment")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(timeout))
    embedded = cell_llm.embed_texts(
        client,
        model=embedding_model,
        texts=texts,
        batch_size=embedding_batch_size,
        retries=retries,
    )
    if embedded.size:
        norms = np.linalg.norm(embedded, axis=1, keepdims=True)
        embedded = embedded / np.clip(norms, a_min=1e-8, a_max=None)
    embedding_dim = int(embedded.shape[1]) if embedded.size else 0
    max_index = max(int(record["index"]) for record in records)
    embedding_matrix = np.zeros((max_index + 1, embedding_dim), dtype=np.float32)
    for offset, record in enumerate(embed_records):
        embedding_matrix[int(record["index"])] = embedded[offset]

    descriptions = {str(record["index"]): record for record in records}
    meta = {
        "kind": "cell_llm_embedding",
        "dataset_group": DATASET_GROUP,
        "description_model": "deterministic_patient_metadata_template",
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "cell_count": len(records),
        "nonzero_cell_count": len(embed_records),
        "input_field": "Cell",
        "index_column": "cell_llm_index",
        "normalized": True,
        "zero_index_reserved_for_no": True,
        "training_ready_root": str(training_ready_root.resolve()),
        "proxy_summary": proxy_summary,
        "run_name": RUN_NAME,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embedding_matrix=embedding_matrix.astype(np.float32, copy=False),
        cell_names=np.asarray([record["canonical_name"] for record in records]),
        cell_indices=np.asarray([int(record["index"]) for record in records], dtype=np.int64),
        descriptions_json=np.asarray(json.dumps(descriptions, ensure_ascii=False)),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    dump_json(sidecar_path, {"meta": meta, "cells": descriptions})
    return {"output": str(output_path), "shape": list(embedding_matrix.shape), "kept_existing": False}


def copy_existing_artifact(source_path: Path, output_path: Path, *, force: bool) -> dict[str, Any]:
    sidecar = source_path.with_suffix(".json")
    output_sidecar = output_path.with_suffix(".json")
    if output_path.exists() and output_sidecar.exists() and not force:
        return {"output": str(output_path), "source": str(source_path), "kept_existing": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    if sidecar.exists():
        shutil.copy2(sidecar, output_sidecar)
    return {"output": str(output_path), "source": str(source_path), "kept_existing": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ready-root", type=Path, default=DEFAULT_TRAINING_READY_ROOT)
    parser.add_argument("--source-training-ready-root", type=Path, default=DEFAULT_SOURCE_TRAINING_READY_ROOT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--model-name", type=Path, default=DEFAULT_ESM_MODEL)
    parser.add_argument("--protein-batch-size", type=int, default=4)
    parser.add_argument("--protein-max-length", type=int, default=1024)
    parser.add_argument("--ppi-edge-path", type=Path, default=DEFAULT_PPI_EDGE_PATH)
    parser.add_argument("--ppi-topk", type=int, default=0)
    parser.add_argument("--stitch-db-dir", type=Path, default=graph.DEFAULT_STITCH_DB_DIR)
    parser.add_argument("--pdi-links-path", type=Path, default=None)
    parser.add_argument("--pdi-chunksize", type=int, default=500_000)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--embedding-model", default=cell_llm.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-protein", action="store_true")
    parser.add_argument("--skip-drug", action="store_true")
    parser.add_argument("--skip-ppi", action="store_true")
    parser.add_argument("--skip-pdi", action="store_true")
    parser.add_argument("--skip-ddi", action="store_true")
    parser.add_argument("--skip-cell-llm", action="store_true")
    parser.add_argument("--skip-cell-type-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    global RUN_NAME
    args = parse_args()
    RUN_NAME = str(args.run_name)
    training_ready_root = args.training_ready_root
    source_root = args.source_training_ready_root
    meta_path = training_ready_root / DATASET_GROUP / "global_meta.json"
    source_meta_path = source_root / DATASET_GROUP / "global_meta.json"
    derived_dir = training_ready_root / DATASET_GROUP / "derived"
    source_derived = source_root / DATASET_GROUP / "derived"
    meta = load_json(meta_path)
    summary: dict[str, Any] = {
        "generated_at": iso_now(),
        "training_ready_root": str(training_ready_root),
        "source_training_ready_root": str(source_root),
        "global_meta": str(meta_path),
        "artifacts": {},
    }

    if not args.skip_drug:
        output = derived_dir / "drug_embedding_morgan_2048.pkl"
        payload = emb.build_drug_embedding_payload(meta=meta, radius=2, n_bits=2048)
        write_pickle(output, payload)
        write_embedding_meta(output, payload, source_meta=meta_path)
        summary["artifacts"]["drug_embedding"] = {"output": str(output), "shape": list(payload["embedding_matrix"].shape)}

    if not args.skip_protein:
        summary["artifacts"]["protein_embedding"] = build_incremental_protein_embedding(
            meta=meta,
            source_embedding_path=source_derived / "protein_embedding_esm.pkl",
            output_path=derived_dir / "protein_embedding_esm.pkl",
            fasta_path=args.fasta,
            model_name=str(args.model_name),
            batch_size=args.protein_batch_size,
            max_length=args.protein_max_length,
            source_meta_path=meta_path,
        )

    if not args.skip_ppi:
        output = derived_dir / "ppi_matrix.npy"
        graph.build_ppi_matrix(
            meta=meta,
            edge_path=args.ppi_edge_path,
            output_path=output,
            node_mapping_json=None,
            allow_online_mapping=False,
            topk=args.ppi_topk,
        )
        augment_matrix_meta(output, source_meta=meta_path)
        summary["artifacts"]["ppi_matrix"] = {"output": str(output), "shape": list(np.load(output, mmap_mode="r").shape)}

    if not args.skip_ddi:
        output = derived_dir / "ddi_matrix.npy"
        graph.build_ddi_matrix(meta=meta, output_path=output, radius=2, n_bits=2048)
        augment_matrix_meta(output, source_meta=meta_path)
        summary["artifacts"]["ddi_matrix"] = {"output": str(output), "shape": list(np.load(output, mmap_mode="r").shape)}

    if not args.skip_pdi:
        output = derived_dir / "pdi_matrix.npy"
        graph.build_pdi_matrix(
            meta=meta,
            links_path=args.pdi_links_path,
            output_path=output,
            stitch_db_dir=args.stitch_db_dir,
            pert_to_flat_json=None,
            protein_node_mapping_json=None,
            protein_mapping_db=None,
            chemical_inchikey_tsv=None,
            allow_online_protein_mapping=False,
            chunksize=args.pdi_chunksize,
        )
        augment_matrix_meta(output, source_meta=meta_path)
        summary["artifacts"]["pdi_matrix"] = {"output": str(output), "shape": list(np.load(output, mmap_mode="r").shape)}

    if not args.skip_cell_llm:
        summary["artifacts"]["cell_llm_embedding"] = build_patient_cell_llm_embedding(
            training_ready_root=training_ready_root,
            output_path=derived_dir / f"cell_llm_embedding_{RUN_NAME}_qwen3_4096.npz",
            env_file=args.env_file,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            timeout=args.timeout,
            retries=args.retries,
            force=args.force,
        )

    if not args.skip_cell_type_copy:
        summary["artifacts"]["cell_type_llm_embedding"] = copy_existing_artifact(
            source_derived / "cell_type_llm_embedding_qwen3_4096_v2.npz",
            derived_dir / "cell_type_llm_embedding_qwen3_4096_v2.npz",
            force=args.force,
        )

    summary_path = training_ready_root / DATASET_GROUP / f"{RUN_NAME}_feature_build_summary.json"
    dump_json(summary_path, summary)
    print(f"[patient-features] summary: {summary_path}")


if __name__ == "__main__":
    main()
