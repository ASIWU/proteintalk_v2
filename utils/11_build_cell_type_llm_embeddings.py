#!/usr/bin/env python3
"""Build frozen LLM text embeddings for training-ready cell_type values."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_READY_ROOT = REPO_ROOT / "data" / "training_ready"
DEFAULT_DESCRIPTION_MODEL = "gpt-5.4"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


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
    """Prefer proxy_on2's HTTP(S) proxy if ALL_PROXY is SOCKS and socksio is absent."""

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


def canonicalize_text_value(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "no"
    text = str(value).strip()
    if not text:
        return "no"
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "no"


def read_feature_table(task_dir: Path) -> pd.DataFrame:
    parquet_path = task_dir / "feature_table.parquet"
    pickle_path = task_dir / "feature_table.pkl"
    csv_path = task_dir / "feature_table.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"missing feature_table under {task_dir}")


def collect_cell_type_records(training_ready_root: Path, dataset_group: str) -> list[dict[str, Any]]:
    meta_path = training_ready_root / dataset_group / "global_meta.json"
    meta = load_json(meta_path)
    mapping = meta["value_to_index"]["cell_type"]
    records: dict[int, dict[str, Any]] = {}
    for name, index in mapping.items():
        records[int(index)] = {
            "index": int(index),
            "canonical_name": str(name),
            "raw_values": {},
            "row_count": 0,
        }

    task_root = training_ready_root / dataset_group / "tasks"
    for task_name in meta.get("task_names", []):
        task_dir = task_root / str(task_name)
        if not task_dir.exists():
            continue
        df = read_feature_table(task_dir)
        if "cell_type" not in df.columns:
            continue
        for raw_value in df["cell_type"].tolist():
            canonical = canonicalize_text_value(raw_value)
            index = mapping.get(canonical, mapping.get("no", 0))
            record = records[int(index)]
            raw_text = "" if pd.isna(raw_value) else str(raw_value).strip()
            raw_key = raw_text or "no"
            record["raw_values"][raw_key] = int(record["raw_values"].get(raw_key, 0)) + 1
            record["row_count"] = int(record["row_count"]) + 1

    result = []
    for index in sorted(records):
        record = records[index]
        record["raw_values"] = dict(sorted(record["raw_values"].items(), key=lambda item: (-item[1], item[0])))
        result.append(record)
    return result


def display_name(canonical_name: str, raw_values: dict[str, int]) -> str:
    if raw_values:
        first = next(iter(raw_values))
        if first and first.lower() != "no":
            return first
    return canonical_name.lower().replace("_", " ")


def description_prompt(record: dict[str, Any]) -> str:
    name = display_name(str(record["canonical_name"]), record["raw_values"])
    variants = ", ".join(list(record["raw_values"])[:5]) or str(record["canonical_name"])
    return (
        "Write one concise biomedical description for a cancer cell type used in drug response "
        "proteomics experiments. Focus on tissue lineage, tumor context, common biological traits, "
        "and why the cell type may matter for perturbation response. Use 2-4 sentences, no bullets. "
        f"Cell type: {name}. Observed labels: {variants}."
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
                                "content": "You write accurate, compact biomedical cell-type descriptions.",
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
        print(f"[cell-type-llm] existing artifact kept: {output_path}")
        return

    records = collect_cell_type_records(Path(args.training_ready_root), args.dataset_group)
    max_index = max(int(record["index"]) for record in records)
    descriptions: dict[str, dict[str, Any]] = {}
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(args.timeout))

    for record in records:
        canonical = str(record["canonical_name"])
        if canonical == "no":
            description = "Missing or unspecified cell type. Reserved zero-vector row."
        else:
            description = chat_completion_text(
                client,
                model=args.description_model,
                prompt=description_prompt(record),
                retries=args.retries,
            )
        descriptions[str(record["index"])] = {
            **record,
            "description": description,
            "prompt": None if canonical == "no" else description_prompt(record),
        }
        print(f"[cell-type-llm] described index={record['index']} name={canonical}")

    embed_records = [record for record in records if str(record["canonical_name"]) != "no"]
    embed_text_values = [descriptions[str(record["index"])]["description"] for record in embed_records]
    embedded = embed_texts(
        client,
        model=args.embedding_model,
        texts=embed_text_values,
        batch_size=args.embedding_batch_size,
        retries=args.retries,
    )
    if args.normalize_embeddings and embedded.size:
        norms = np.linalg.norm(embedded, axis=1, keepdims=True)
        embedded = embedded / np.clip(norms, a_min=1e-8, a_max=None)

    embedding_dim = int(embedded.shape[1]) if embedded.size else 0
    embedding_matrix = np.zeros((max_index + 1, embedding_dim), dtype=np.float32)
    for offset, record in enumerate(embed_records):
        embedding_matrix[int(record["index"])] = embedded[offset]

    meta = {
        "kind": "cell_type_llm_embedding",
        "dataset_group": args.dataset_group,
        "description_model": args.description_model,
        "embedding_model": args.embedding_model,
        "embedding_dim": embedding_dim,
        "cell_type_count": len(records),
        "nonzero_cell_type_count": len(embed_records),
        "normalized": bool(args.normalize_embeddings),
        "zero_index_reserved_for_no": True,
        "training_ready_root": str(Path(args.training_ready_root).resolve()),
        "proxy_summary": proxy_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embedding_matrix=embedding_matrix.astype(np.float32, copy=False),
        cell_type_names=np.asarray([record["canonical_name"] for record in records]),
        cell_type_indices=np.asarray([int(record["index"]) for record in records], dtype=np.int64),
        descriptions_json=np.asarray(json.dumps(descriptions, ensure_ascii=False)),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    dump_json(sidecar_path, {"meta": meta, "cell_types": descriptions})
    print(f"[cell-type-llm] wrote {output_path} shape={embedding_matrix.shape}")
    print(f"[cell-type-llm] wrote {sidecar_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--dataset-group", default="ptv3")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_TRAINING_READY_ROOT / "ptv3" / "derived" / "cell_type_llm_embedding_qwen3_4096.npz"),
    )
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--description-model", default=DEFAULT_DESCRIPTION_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-normalize-embeddings", action="store_false", dest="normalize_embeddings")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(normalize_embeddings=True)
    return parser.parse_args()


def main() -> None:
    build_embeddings(parse_args())


if __name__ == "__main__":
    main()
