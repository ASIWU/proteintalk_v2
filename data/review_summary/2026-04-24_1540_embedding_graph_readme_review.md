# 2026-04-24 15:40 Embedding / Graph README Review

## Scope

- Checked how to generate ligand/drug embeddings, protein embeddings, PPI, DDI, and PDI artifacts from the current codebase.
- Compared the available builder documentation with `README.md`.

## Findings

- The current embedding entrypoint is `utils/04_build_embeddings_from_global_meta.py`.
- The current graph entrypoint is `utils/05_build_graph_matrices_from_global_meta.py`.
- Both builder families are aligned to stage-2 `data/training_ready/<dataset>/global_meta.json`, not directly to stage-1 `data/standardized`.
- `README.md` documented stage-1 standardization only and did not include the stage-2 training-ready step or auxiliary embedding/graph commands.

## Update

- Added README coverage for:
  - stage-2 build and validation commands,
  - ligand/drug embedding,
  - protein embedding,
  - PPI matrix,
  - DDI matrix,
  - PDI matrix,
  - required external inputs and output file formats.
