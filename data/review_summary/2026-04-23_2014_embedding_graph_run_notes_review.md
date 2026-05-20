# Review Summary

- Time: 2026-04-23 20:14
- Scope: extract practical run notes and failure conditions for the current embedding and graph builders.

## Findings

- All builders depend on stage-2 `global_meta.json`; outputs must stay aligned to the same metadata version.
- `utils/04_build_embeddings_from_global_meta.py` writes `.pkl` payloads and creates parent output directories automatically.
- Drug embedding depends on valid SMILES in `global_meta.json`; unresolved items are recorded in the pickle payload.
- Protein embedding requires `torch` and `transformers`, plus a FASTA keyed by UniProt-compatible IDs. It may use GPU automatically when available.
- `utils/05_build_graph_matrices_from_global_meta.py` writes `.npy` plus sibling `.meta.json`.
- PPI works best with a local UniProt-to-node mapping JSON; online mapping is optional but environment-dependent.
- PPI can fail if edge columns or supported score columns are missing, or if all edges are filtered out by metadata-space filtering / reference filtering.
- PDI requires either `--pert-to-flat-json` or `--chemical-inchikey-tsv`; the links table must contain `chemical` and `protein` plus one supported score column.
- DDI and drug embedding both derive from SMILES, so bad or missing SMILES will silently produce zero rows / unresolved entries rather than useful structure.

## Notes

- Recommended execution order after stage-2 data build: drug embedding, protein embedding, DDI, PPI, then PDI.
- Avoid mixing artifacts generated from different dataset roots such as `ptv1` and `ptv3`.
