# Review Summary

- Time: 2026-04-23 19:56
- Scope: locate the current scripts used to generate protein embedding, ligand/drug embedding, PPI, PDI, and DDI artifacts.

## Findings

- The current embedding entrypoint is `utils/04_build_embeddings_from_global_meta.py`.
- Use subcommand `drug` for ligand/drug embedding and subcommand `protein` for protein embedding.
- The current graph-matrix entrypoint is `utils/05_build_graph_matrices_from_global_meta.py`.
- Use subcommand `ppi` for protein-protein interaction matrices.
- Use subcommand `ddi` for drug-drug similarity matrices.
- Use subcommand `pdi` for perturbation/drug-protein interaction matrices.
- The repo guidance with example commands is documented in `docs/data_process_summary_02.md`.

## Notes

- These scripts build artifacts aligned to stage-2 `global_meta.json`.
- There is no separate dedicated top-level script for each artifact in the current codebase; the two unified entrypoints above are the intended builders.
