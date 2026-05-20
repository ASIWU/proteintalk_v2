#!/usr/bin/env python3
"""Export UniProt accessions from a training-ready global_meta.json file.

The output is a plain text file with one UniProt accession per line. This is
intended for pasting/uploading to UniProt to retrieve a matching FASTA file for
protein embedding generation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)
DEFAULT_SPECIAL_VALUES = {"control", "no"}


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ordered_protein_ids(meta: dict[str, Any]) -> list[str]:
    index_to_id = meta.get("protein_index_to_id")
    if isinstance(index_to_id, list) and index_to_id:
        return [str(item).strip() for item in index_to_id]

    protein_index = meta.get("protein_index")
    if not isinstance(protein_index, dict):
        raise ValueError("global_meta.json must contain `protein_index` or `protein_index_to_id`.")

    try:
        return [
            str(item).strip()
            for item, _ in sorted(protein_index.items(), key=lambda pair: int(pair[1]))
        ]
    except Exception as exc:
        raise ValueError("`protein_index` values must be sortable integer indices.") from exc


def special_protein_values(meta: dict[str, Any]) -> set[str]:
    special_values = set(DEFAULT_SPECIAL_VALUES)
    payload = meta.get("special_values", {})
    if isinstance(payload, dict):
        protein_payload = payload.get("protein_index", {})
        if isinstance(protein_payload, dict):
            special_values.update(str(key).strip() for key in protein_payload)
    return special_values


def export_uniprot_ids(
    *,
    meta: dict[str, Any],
    include_special: bool,
    on_invalid: str,
) -> tuple[list[str], dict[str, Any]]:
    special_values = special_protein_values(meta)
    exported: list[str] = []
    seen: set[str] = set()
    skipped_special: list[str] = []
    invalid_ids: list[str] = []
    duplicate_ids: list[str] = []

    for protein_id in ordered_protein_ids(meta):
        if not protein_id:
            continue
        if protein_id in special_values and not include_special:
            skipped_special.append(protein_id)
            continue
        if not UNIPROT_ACCESSION_RE.fullmatch(protein_id):
            invalid_ids.append(protein_id)
            if on_invalid == "skip":
                continue
            if on_invalid == "error":
                continue
        if protein_id in seen:
            duplicate_ids.append(protein_id)
            continue
        seen.add(protein_id)
        exported.append(protein_id)

    if invalid_ids and on_invalid == "error":
        preview = ", ".join(invalid_ids[:10])
        raise ValueError(
            f"Found {len(invalid_ids)} non-UniProt protein IDs. "
            f"Use --on-invalid skip or --on-invalid include if this is expected. First values: {preview}"
        )

    audit = {
        "exported_count": len(exported),
        "skipped_special_count": len(skipped_special),
        "invalid_count": len(invalid_ids),
        "duplicate_count": len(duplicate_ids),
        "skipped_special": skipped_special,
        "invalid_ids": invalid_ids,
        "duplicate_ids": duplicate_ids,
    }
    return exported, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a UniProt accession txt list from training-ready global_meta.json."
    )
    parser.add_argument("--global-meta", type=Path, required=True, help="Path to data/training_ready/<dataset>/global_meta.json")
    parser.add_argument("--output-txt", type=Path, required=True, help="Output text file, one UniProt ID per line")
    parser.add_argument(
        "--include-special",
        action="store_true",
        help="Include special protein_index values such as `control` and `no`.",
    )
    parser.add_argument(
        "--on-invalid",
        choices=("error", "skip", "include"),
        default="error",
        help="How to handle protein IDs that are not valid UniProt accessions.",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="Optional JSON audit path with counts and skipped IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_json(args.global_meta)
    if not isinstance(meta, dict):
        raise ValueError("global_meta.json root must be a JSON object.")

    uniprot_ids, audit = export_uniprot_ids(
        meta=meta,
        include_special=args.include_special,
        on_invalid=args.on_invalid,
    )

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text("\n".join(uniprot_ids) + ("\n" if uniprot_ids else ""), encoding="utf-8")

    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"Wrote {audit['exported_count']} UniProt IDs to {args.output_txt} "
        f"(skipped_special={audit['skipped_special_count']}, invalid={audit['invalid_count']}, "
        f"duplicates={audit['duplicate_count']})."
    )


if __name__ == "__main__":
    main()
