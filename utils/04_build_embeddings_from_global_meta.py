#!/usr/bin/env python3
"""Build drug/protein embeddings from training-ready global metadata.

This script is intentionally separate from the main data builder because protein
embeddings may require GPU access and substantial runtime.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META_ROOT = REPO_ROOT / "data" / "training_ready"
UNIPROT_ACCESSION_RE = re.compile(
    r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def build_drug_embedding_payload(
    *,
    meta: dict[str, Any],
    radius: int,
    n_bits: int,
) -> dict[str, Any]:
    pert_ids = ordered_ids(meta["pert_index"])
    embedding_matrix = np.zeros((len(pert_ids), n_bits), dtype=np.float32)
    smiles_fallback_items: dict[str, str] = {}
    fingerprint_generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for pert_id in tqdm(pert_ids, desc="Drug embeddings"):
        smiles = meta["pertid_to_smiles"].get(pert_id, "")
        mol, fallback_reason = mol_from_smiles_or_empty(pert_id, smiles)
        if fallback_reason is not None:
            smiles_fallback_items[pert_id] = fallback_reason
        fingerprint = fingerprint_generator.GetFingerprint(mol)
        vector = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, vector)
        embedding_matrix[meta["pert_index"][pert_id]] = vector

    return {
        "kind": "drug_embedding",
        "embedding_name": "morgan_fingerprint",
        "radius": radius,
        "n_bits": n_bits,
        "item_to_index": meta["pert_index"],
        "index_to_item": pert_ids,
        "embedding_matrix": embedding_matrix,
        "embedding_dim": n_bits,
        "unresolved_items": {},
        "smiles_fallback_items": smiles_fallback_items,
    }


def extract_header_aliases(header_text: str) -> list[str]:
    aliases: list[str] = []
    if not header_text:
        return aliases

    first_token = header_text.split()[0]
    if first_token:
        aliases.append(first_token)

    pipe_tokens = [token.strip() for token in header_text.split("|") if token.strip()]
    aliases.extend(pipe_tokens)
    if len(pipe_tokens) >= 2 and pipe_tokens[0].lower() in {"sp", "tr"}:
        aliases.append(pipe_tokens[1])

    aliases.extend(UNIPROT_ACCESSION_RE.findall(header_text))

    seen: set[str] = set()
    ordered: list[str] = []
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        ordered.append(alias)
    return ordered


def load_sequences_from_fasta(fasta_path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_aliases: list[str] = []
    current_chunks: list[str] = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_aliases:
                    sequence = "".join(current_chunks)
                    for alias in current_aliases:
                        sequences.setdefault(alias, sequence)
                current_aliases = extract_header_aliases(line[1:].strip())
                current_chunks = []
            else:
                current_chunks.append(line)
    if current_aliases:
        sequence = "".join(current_chunks)
        for alias in current_aliases:
            sequences.setdefault(alias, sequence)
    return sequences


def build_protein_embedding_payload(
    *,
    meta: dict[str, Any],
    sequence_lookup: dict[str, str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Protein embedding generation requires `torch` and `transformers`."
        ) from exc

    protein_ids = ordered_ids(meta["protein_index"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    embedding_dim = int(model.config.hidden_size)
    embedding_matrix = np.zeros((len(protein_ids), embedding_dim), dtype=np.float32)
    sequence_fallback_items: dict[str, str] = {}

    batch_items: list[tuple[int, str, str]] = []
    with torch.no_grad():
        for protein_id in tqdm(protein_ids, desc="Protein embeddings"):
            # Empty sequence fallback is intentional: every protein_index row must pass through ESM.
            if protein_id in {"control", "no"}:
                sequence = ""
                sequence_fallback_items[protein_id] = "special_value_empty_sequence"
            else:
                sequence = sequence_lookup.get(protein_id, "")
                if not sequence:
                    sequence_fallback_items[protein_id] = "missing_sequence_empty_sequence"
            batch_items.append((meta["protein_index"][protein_id], protein_id, sequence))
            if len(batch_items) < batch_size:
                continue
            _flush_protein_batch(
                batch_items=batch_items,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
                embedding_matrix=embedding_matrix,
            )
            batch_items = []
        if batch_items:
            _flush_protein_batch(
                batch_items=batch_items,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
                embedding_matrix=embedding_matrix,
            )

    return {
        "kind": "protein_embedding",
        "embedding_name": "esm_mean_pooling",
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "item_to_index": meta["protein_index"],
        "index_to_item": protein_ids,
        "embedding_matrix": embedding_matrix,
        "unresolved_items": {},
        "sequence_fallback_items": sequence_fallback_items,
        "device_used": str(device),
    }


def _flush_protein_batch(
    *,
    batch_items: list[tuple[int, str, str]],
    tokenizer: Any,
    model: Any,
    device: Any,
    max_length: int,
    embedding_matrix: np.ndarray,
) -> None:
    indices = [item[0] for item in batch_items]
    sequences = [item[2] for item in batch_items]
    encoded = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)
    hidden_state = outputs.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden_state.dtype)
    masked_hidden = hidden_state * attention_mask
    token_counts = attention_mask.sum(dim=1).clamp(min=1.0)
    pooled = (masked_hidden.sum(dim=1) / token_counts).detach().cpu().numpy().astype(np.float32)
    embedding_matrix[np.asarray(indices, dtype=np.int64)] = pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build drug/protein embeddings from training-ready global_meta.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    drug_parser = subparsers.add_parser("drug", help="Build pert-index ordered drug embeddings")
    drug_parser.add_argument("--global-meta", type=Path, required=True, help="Path to stage-2 global_meta.json")
    drug_parser.add_argument("--output-pkl", type=Path, required=True, help="Output pickle path")
    drug_parser.add_argument("--radius", type=int, default=2)
    drug_parser.add_argument("--n-bits", type=int, default=2048)

    protein_parser = subparsers.add_parser("protein", help="Build protein-index ordered protein embeddings")
    protein_parser.add_argument("--global-meta", type=Path, required=True, help="Path to stage-2 global_meta.json")
    protein_parser.add_argument("--output-pkl", type=Path, required=True, help="Output pickle path")
    protein_parser.add_argument("--fasta", type=Path, required=True, help="FASTA file containing sequences keyed by UniProt ID")
    protein_parser.add_argument("--model-name", type=str, required=True, help="Transformers model name or local model path")
    protein_parser.add_argument("--batch-size", type=int, default=4)
    protein_parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum tokenizer input length for protein sequences; this is not the embedding dimension.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_json(args.global_meta)
    if args.command == "drug":
        payload = build_drug_embedding_payload(
            meta=meta,
            radius=args.radius,
            n_bits=args.n_bits,
        )
    else:
        sequence_lookup = load_sequences_from_fasta(args.fasta)
        payload = build_protein_embedding_payload(
            meta=meta,
            sequence_lookup=sequence_lookup,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as handle:
        pickle.dump(payload, handle)


if __name__ == "__main__":
    main()
