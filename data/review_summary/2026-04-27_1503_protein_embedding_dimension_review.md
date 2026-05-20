# 2026-04-27 15:03 Protein Embedding Dimension Review

## Scope

- Reviewed whether any repository code assumes protein embedding feature dimension is `1024`.
- Checked current PTV3 protein embedding pickle shape and command-line defaults.

## Findings

- The current PTV3 protein embedding matrix has shape `(11345, 1280)`.
- `11345` is the number of protein entities in `data/training_ready/ptv3/global_meta.json["protein_index"]`.
- `1280` is the ESM output feature dimension for the selected `facebook/esm2_t33_650M_UR50D` model.
- The `1024` value in current commands is `--max-length`, meaning tokenizer input sequence length limit, not embedding feature dimension.
- No runtime Python code was found that hard-codes protein embedding feature dimension as `1024`.

## Fixes Applied

- Added `embedding_dim` and `max_length` metadata to newly generated protein embedding payloads in `utils/04_build_embeddings_from_global_meta.py`.
- Clarified the `--max-length` CLI help text.
- Updated `utils/07_check_protein_embedding_count.py` to print matrix shape and feature dimension, and to validate payload `embedding_dim` when present.
- Clarified the README protein embedding command notes.

## Verification

- `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python -m py_compile utils/04_build_embeddings_from_global_meta.py utils/07_check_protein_embedding_count.py`
- `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/07_check_protein_embedding_count.py`
- `/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/04_build_embeddings_from_global_meta.py protein --help`
