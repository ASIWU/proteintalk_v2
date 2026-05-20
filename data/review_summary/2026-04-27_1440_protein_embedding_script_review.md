# 2026-04-27 14:40 Protein Embedding Script Review

## Scope

- Reviewed `utils/04_build_embeddings_from_global_meta.py`, focusing on FASTA parsing and protein embedding row coverage.
- Checked the available PTV3 metadata, FASTA, and generated protein embedding pickle.

## Findings

- FASTA parsing supports standard UniProt headers such as `sp|A0A024RBG1|NUD4B_HUMAN ...`, multi-line sequences, and accession alias extraction.
- PTV3 `global_meta.json["protein_index"]` contains 11,345 entities: 11,343 real UniProt IDs plus `control` and `no`.
- The available PTV3 FASTA contains 11,343 records, but 7 real metadata proteins are absent from the parsed sequence lookup: `A0A0B4J2D5`, `P04745`, `P0DN76`, `P0DN79`, `Q6ZMK1`, `Q8IXS6`, and `Q9UPP5`.
- The script allocates an embedding matrix with one row per metadata protein, so the matrix row count matches the metadata count.
- The script does not generate model embeddings for missing sequences or special values. It leaves those rows as zero vectors and records them in `unresolved_items`.
- The generated PTV3 pickle currently has shape `(11345, 1280)`, 11,345 `index_to_item` entries, 11,345 `item_to_index` entries, and 9 zero rows: the 7 missing UniProt IDs plus `control` and `no`.
- Mean pooling currently uses the full attention mask, so Hugging Face ESM special tokens such as `<cls>` and `<eos>` are included in the pooled vectors.

## Conclusion

- The script preserves index-aligned matrix shape correctly.
- Under the requirement that every metadata entity must receive a generated embedding regardless of sequence mapping, the current behavior is incomplete because unresolved proteins receive zero placeholders rather than generated fallback embeddings.

## Verification

- `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile utils/04_build_embeddings_from_global_meta.py`
- Parsed `data/training_ready/ptv3/derived/idmapping_2026_04_27.fasta` with the script loader and compared coverage against `data/training_ready/ptv3/global_meta.json`.
- Inspected `data/training_ready/ptv3/derived/protein_embedding_esm.pkl` for shape, unresolved items, zero rows, and finite values.

## Follow-up Fix

- At 2026-04-27 14:47 HKT, `utils/04_build_embeddings_from_global_meta.py` was updated so missing FASTA mappings and `control` / `no` use empty sequence input instead of being skipped.
- `unresolved_items` is now empty when every row is sent through ESM, and `sequence_fallback_items` records rows that used the empty-sequence fallback.
