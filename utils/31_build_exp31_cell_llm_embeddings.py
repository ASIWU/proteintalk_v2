#!/usr/bin/env python3
"""Build exp31 PDX Cell LLM embeddings keyed by cell_llm_index."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data/training_ready_exp31_rnaseq"
DEFAULT_DESCRIPTION_MODEL = "gpt-5.4"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_SOURCE_CELL_TYPE_EMBEDDING = (
    DEFAULT_TRAINING_READY_ROOT / "ptv3" / "derived" / "cell_type_llm_embedding_qwen3_4096_v2.npz"
)
TISSUE_FALLBACK_MAP = {
    "BRCA": "BREAST",
    "CRC": "COLON",
    "PDAC": "PANCREAS",
    "NSCLC": "LUNG",
    "CM": "SKIN",
    "no": "no",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def sanitize_proxy_env() -> dict[str, Any]:
    all_proxy = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    socks_missing = importlib.util.find_spec("socksio") is None
    if not all_proxy or not socks_missing:
        return {"changed": False}
    scheme = urlparse(all_proxy).scheme.lower()
    has_http_proxy = bool(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"))
    if scheme.startswith("socks") and has_http_proxy:
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)
        return {"changed": True, "reason": "removed SOCKS ALL_PROXY because socksio is unavailable"}
    return {"changed": False}


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


def collect_records(training_ready_root: Path, dataset_group: str) -> list[dict[str, Any]]:
    meta = load_json(training_ready_root / dataset_group / "global_meta.json")
    task_root = training_ready_root / dataset_group / "tasks"
    records: dict[int, dict[str, Any]] = {
        0: {
            "index": 0,
            "canonical_name": "no",
            "label": "Missing or unspecified PDX sample label.",
            "sample_id": "no",
            "cancer_type": "no",
            "row_count": 0,
        }
    }
    for task_name in meta.get("task_names", []):
        task_dir = task_root / str(task_name)
        if not task_dir.exists():
            continue
        df = read_feature_table(task_dir)
        if "cell_llm_index" not in df.columns or "cell_llm_label" not in df.columns:
            raise KeyError(f"{task_dir} must contain cell_llm_index and cell_llm_label")
        for row in df[["cell_llm_index", "cell_llm_label", "patient_sample_id", "cancer_type"]].itertuples(index=False):
            idx = int(row.cell_llm_index)
            label = str(row.cell_llm_label)
            sample_id = str(row.patient_sample_id)
            cancer_type = str(row.cancer_type)
            existing = records.get(idx)
            if existing is not None and existing["label"] != label:
                raise ValueError(f"cell_llm_index {idx} has inconsistent labels")
            if existing is None:
                existing = {
                    "index": idx,
                    "canonical_name": sample_id,
                    "label": label,
                    "sample_id": sample_id,
                    "cancer_type": cancer_type,
                    "row_count": 0,
                }
                records[idx] = existing
            existing["row_count"] = int(existing["row_count"]) + 1
    return [records[idx] for idx in sorted(records)]


def description_prompt(record: dict[str, Any]) -> str:
    return (
        "Write one concise biomedical description for a patient-derived xenograft baseline RNA sample "
        "used in a cross-domain drug-response fine-tuning benchmark. Focus on the tumor lineage and "
        "the cancer-type context only. Do not mention drug response, sensitivity, labels, or outcomes. "
        "Use 2-3 sentences, no bullets.\n\n"
        "Dataset context: PDX-2015nm pan-cancer baseline RNA-seq cohort.\n"
        f"Sample label: {record['label']}\n"
        f"Sample id: {record['sample_id']}\n"
        f"Cancer type: {record['cancer_type']}"
    )


def chat_completion_text(client: OpenAI, *, model: str, prompt: str, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            for token_arg in ("max_completion_tokens", "max_tokens"):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You write accurate, compact biomedical tumor-context descriptions.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        **{token_arg: 220},
                    )
                    text = (response.choices[0].message.content or "").strip()
                    if text:
                        return text
                except TypeError:
                    continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"chat completion failed after {retries} attempts: {last_error}")


def embed_texts(client: OpenAI, *, model: str, texts: list[str], batch_size: int, retries: int) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                response = client.embeddings.create(model=model, input=batch)
                vectors.extend([item.embedding for item in response.data])
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(f"embedding failed after {retries} attempts: {last_error}")
    return np.asarray(vectors, dtype=np.float32)


def load_source_cell_type_embedding(path: Path) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    payload = np.load(path, allow_pickle=True)
    matrix = np.asarray(payload["embedding_matrix"], dtype=np.float32)
    if "cell_type_names" not in payload.files or "cell_type_indices" not in payload.files:
        raise KeyError(f"{path} must contain cell_type_names and cell_type_indices")
    names = [str(item) for item in payload["cell_type_names"].tolist()]
    indices = [int(item) for item in payload["cell_type_indices"].tolist()]
    name_to_row = {name: offset for offset, name in enumerate(names)}
    name_to_index = {name: idx for name, idx in zip(names, indices, strict=True)}
    source_meta: dict[str, Any] = {"name_to_index": name_to_index}
    if "meta_json" in payload.files:
        source_meta["meta"] = json.loads(str(payload["meta_json"].item()))
    if "descriptions_json" in payload.files:
        source_meta["descriptions"] = json.loads(str(payload["descriptions_json"].item()))
    return matrix, name_to_row, source_meta


def source_description(source_meta: dict[str, Any], source_name: str) -> str:
    source_index = source_meta.get("name_to_index", {}).get(source_name)
    descriptions = source_meta.get("descriptions", {})
    if isinstance(descriptions, dict) and source_index is not None:
        entry = descriptions.get(str(source_index))
        if isinstance(entry, dict):
            text = str(entry.get("description") or "").strip()
            if text:
                return text
    return f"Existing validated cell-type LLM embedding for {source_name}."


def build_offline_tissue_fallback(args: argparse.Namespace) -> None:
    """Create sample-indexed Cell embeddings from existing tissue LLM vectors.

    This mode avoids external API calls.  It preserves the exp31 Cell-channel
    contract by writing one row per `cell_llm_index`, but multiple PDX samples
    from the same cancer type intentionally share one existing tissue vector.
    """

    output_path = Path(args.output)
    sidecar_path = output_path.with_suffix(".json")
    if output_path.exists() and sidecar_path.exists() and not args.force:
        print(f"[exp31-cell-llm] existing artifact kept: {output_path}")
        return

    records = collect_records(Path(args.training_ready_root), args.dataset_group)
    max_index = max(int(record["index"]) for record in records)
    source_path = Path(args.source_cell_type_embedding)
    source_matrix, source_name_to_row, source_meta = load_source_cell_type_embedding(source_path)
    missing = sorted(set(TISSUE_FALLBACK_MAP.values()) - set(source_name_to_row))
    if missing:
        raise ValueError(f"source cell-type embedding missing fallback tissue rows: {missing}")

    matrix = np.zeros((max_index + 1, int(source_matrix.shape[1])), dtype=np.float32)
    descriptions: dict[str, dict[str, Any]] = {}
    for record in records:
        idx = int(record["index"])
        cancer_type = str(record.get("cancer_type", "no"))
        source_name = TISSUE_FALLBACK_MAP.get(cancer_type)
        if source_name is None:
            raise ValueError(f"no tissue fallback mapping for cancer_type={cancer_type!r}")
        if idx != 0:
            matrix[idx] = source_matrix[source_name_to_row[source_name]]
        description = source_description(source_meta, source_name)
        descriptions[str(idx)] = {
            **record,
            "description": description,
            "prompt": None,
            "source_cell_type_name": source_name,
            "source_cell_type_embedding_row": int(source_name_to_row[source_name]),
            "generation_mode": "offline_tissue_fallback_from_existing_cell_type_llm",
        }
        print(
            "[exp31-cell-llm] fallback "
            f"index={idx} cancer_type={cancer_type} source_cell_type={source_name}"
        )

    if args.normalize_embeddings and matrix.size:
        rows = np.arange(matrix.shape[0]) != 0
        norms = np.linalg.norm(matrix[rows], axis=1, keepdims=True)
        matrix[rows] = matrix[rows] / np.clip(norms, a_min=1e-8, a_max=None)

    meta = {
        "kind": "cell_llm_embedding",
        "dataset_group": args.dataset_group,
        "run_name": "exp31_rnaseq",
        "generated_at": iso_now(),
        "generation_mode": "offline_tissue_fallback_from_existing_cell_type_llm",
        "source_cell_type_embedding": str(source_path.resolve()),
        "tissue_fallback_map": TISSUE_FALLBACK_MAP,
        "embedding_dim": int(matrix.shape[1]),
        "cell_count": len(records),
        "nonzero_cell_count": max(0, len(records) - 1),
        "input_field": "cell_llm_label",
        "index_column": "cell_llm_index",
        "normalized": bool(args.normalize_embeddings),
        "zero_index_reserved_for_no": True,
        "training_ready_root": str(Path(args.training_ready_root).resolve()),
        "leakage_guard": "offline fallback uses cancer-type tissue descriptions only; no drug labels or outcomes",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embedding_matrix=matrix.astype(np.float32, copy=False),
        cell_names=np.asarray([record["canonical_name"] for record in records]),
        cell_indices=np.asarray([int(record["index"]) for record in records], dtype=np.int64),
        descriptions_json=np.asarray(json.dumps(descriptions, ensure_ascii=False)),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    dump_json(sidecar_path, {"meta": meta, "cells": descriptions})
    print(f"[exp31-cell-llm] wrote offline fallback {output_path} shape={matrix.shape}")
    print(f"[exp31-cell-llm] wrote {sidecar_path}")


def build_embeddings(args: argparse.Namespace) -> None:
    load_env_file(Path(args.env_file))
    proxy_summary = sanitize_proxy_env()
    api_key = os.environ.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("missing openai_api_key/openai_base_url in .env or environment")

    output_path = Path(args.output)
    sidecar_path = output_path.with_suffix(".json")
    if output_path.exists() and sidecar_path.exists() and not args.force:
        print(f"[exp31-cell-llm] existing artifact kept: {output_path}")
        return

    records = collect_records(Path(args.training_ready_root), args.dataset_group)
    max_index = max(int(record["index"]) for record in records)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(args.timeout))
    descriptions: dict[str, dict[str, Any]] = {}

    for record in records:
        idx = int(record["index"])
        if idx == 0:
            description = "Missing or unspecified PDX sample label. Reserved zero-vector row."
            prompt = None
        else:
            prompt = description_prompt(record)
            description = chat_completion_text(client, model=args.description_model, prompt=prompt, retries=args.retries)
        descriptions[str(idx)] = {**record, "description": description, "prompt": prompt}
        print(f"[exp31-cell-llm] described index={idx} label={record['label']}")

    embed_records = [record for record in records if int(record["index"]) != 0]
    embedded = embed_texts(
        client,
        model=args.embedding_model,
        texts=[descriptions[str(record["index"])]["description"] for record in embed_records],
        batch_size=args.embedding_batch_size,
        retries=args.retries,
    )
    if args.normalize_embeddings and embedded.size:
        norms = np.linalg.norm(embedded, axis=1, keepdims=True)
        embedded = embedded / np.clip(norms, a_min=1e-8, a_max=None)

    embedding_dim = int(embedded.shape[1]) if embedded.size else 0
    matrix = np.zeros((max_index + 1, embedding_dim), dtype=np.float32)
    for offset, record in enumerate(embed_records):
        matrix[int(record["index"])] = embedded[offset]

    meta = {
        "kind": "cell_llm_embedding",
        "dataset_group": args.dataset_group,
        "run_name": "exp31_rnaseq",
        "generated_at": iso_now(),
        "description_model": args.description_model,
        "embedding_model": args.embedding_model,
        "embedding_dim": embedding_dim,
        "cell_count": len(records),
        "nonzero_cell_count": len(embed_records),
        "input_field": "cell_llm_label",
        "index_column": "cell_llm_index",
        "normalized": bool(args.normalize_embeddings),
        "zero_index_reserved_for_no": True,
        "training_ready_root": str(Path(args.training_ready_root).resolve()),
        "proxy_summary": proxy_summary,
        "leakage_guard": "prompts exclude drug response, sensitivity labels, and outcomes",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embedding_matrix=matrix.astype(np.float32, copy=False),
        cell_names=np.asarray([record["canonical_name"] for record in records]),
        cell_indices=np.asarray([int(record["index"]) for record in records], dtype=np.int64),
        descriptions_json=np.asarray(json.dumps(descriptions, ensure_ascii=False)),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    dump_json(sidecar_path, {"meta": meta, "cells": descriptions})
    print(f"[exp31-cell-llm] wrote {output_path} shape={matrix.shape}")
    print(f"[exp31-cell-llm] wrote {sidecar_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", default="ptv3")
    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_TRAINING_READY_ROOT
            / "ptv3"
            / "derived"
            / "cell_llm_embedding_exp31_rnaseq_qwen3_4096.npz"
        ),
    )
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--description-model", default=DEFAULT_DESCRIPTION_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-normalize-embeddings", action="store_false", dest="normalize_embeddings")
    parser.add_argument(
        "--offline-tissue-fallback",
        action="store_true",
        help="Avoid API calls and build exp31 Cell embeddings from existing tissue-level cell_type LLM vectors.",
    )
    parser.add_argument("--source-cell-type-embedding", default=str(DEFAULT_SOURCE_CELL_TYPE_EMBEDDING))
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(normalize_embeddings=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline_tissue_fallback:
        build_offline_tissue_fallback(args)
    else:
        build_embeddings(args)


if __name__ == "__main__":
    main()
