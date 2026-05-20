# 2026-04-24 15:54 UniProt Export Tool Review

## Scope

- Added a utility to export UniProt accession lists from stage-2 `global_meta.json`.
- The intended output is a plain text file for UniProt FASTA retrieval before running protein embedding generation.

## Implementation

- New script: `utils/06_export_uniprot_ids_from_global_meta.py`.
- Reads `protein_index_to_id` when available, otherwise sorts `protein_index` by integer index.
- Excludes `control` and `no` by default.
- Validates each real protein ID as a UniProt accession by default.
- Writes one accession per line to `--output-txt`.
- Optionally writes an audit JSON with exported, skipped, invalid, and duplicate counts.

## Local Data Check

- `data/training_ready/ptv3/global_meta.json`: 11,343 real UniProt IDs, 0 invalid.
- `data/training_ready/ptv1/global_meta.json`: 5,576 real UniProt IDs, 0 invalid.

## Generated Files

- `data/training_ready/ptv3/derived/uniprot_ids.txt`
- `data/training_ready/ptv3/derived/uniprot_ids.audit.json`
- `data/training_ready/ptv1/derived/uniprot_ids.txt`
- `data/training_ready/ptv1/derived/uniprot_ids.audit.json`
