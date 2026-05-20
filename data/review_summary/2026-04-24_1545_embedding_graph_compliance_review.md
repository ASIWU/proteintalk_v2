# 2026-04-24 15:45 Embedding / Graph Compliance Review

## Scope

- Compared the current embedding and graph builders against `docs/Data_Process.md` and `docs/Data_Process_2.md`.
- Used `docs/data_process_summary_01.md` and `docs/data_process_summary_02.md` only as background context.

## Compliance Check

- `docs/Data_Process.md` requires stable global metadata containing UniProt protein index, perturbation index, SMILES, and target protein lists. Stage-2 `data/training_ready/ptv3/global_meta.json` and `data/training_ready/ptv1/global_meta.json` contain `protein_index`, `protein_index_to_id`, `pert_index`, `pert_index_to_id`, `pertid_to_smiles`, and indexed target protein lists.
- Drug embedding is implemented in `utils/04_build_embeddings_from_global_meta.py drug`. It orders rows by `global_meta.json["pert_index"]`, writes a `.pkl` payload, and records unresolved perturbations.
- Protein embedding is implemented in `utils/04_build_embeddings_from_global_meta.py protein`. It orders rows by `global_meta.json["protein_index"]`, writes a `.pkl` payload, supports UniProt-compatible FASTA headers, and records missing sequences / special values.
- PPI is implemented in `utils/05_build_graph_matrices_from_global_meta.py ppi`. It builds a square matrix with axes ordered by `protein_index` and includes special protein rows/columns through the full matrix shape.
- DDI is implemented in `utils/05_build_graph_matrices_from_global_meta.py ddi`. It builds a square matrix with axes ordered by `pert_index` and includes the `"no"` perturbation row/column through the full matrix shape.
- PDI is implemented in `utils/05_build_graph_matrices_from_global_meta.py pdi`. It builds a rectangular `pert_index x protein_index` matrix and includes ID conversion paths through `pert-to-flat`, chemical InChIKey, and protein node mapping inputs.

## Fix Applied

- Tightened the PPI reference filter so fused score filtering is applied when `combined_score` is absent.
- Made textmining-only filtering tolerate missing non-text evidence columns.
- Allowed PPI inputs with only `combined_score` to build instead of failing before matrix construction.

## Remaining Conditions

- Protein embedding still requires a FASTA and a transformers model or local model directory.
- PPI and PDI still require external graph/link resources and, for best reproducibility, local mapping JSON/TSV files.
- The existing stage-2 validator does not validate generated embedding or graph artifacts after they are built.

## Verification

- `python -m py_compile utils/04_build_embeddings_from_global_meta.py utils/05_build_graph_matrices_from_global_meta.py`
- `python utils/04_build_embeddings_from_global_meta.py --help`
- `python utils/05_build_graph_matrices_from_global_meta.py --help`
- Synthetic combined-score-only PPI smoke test: verified matrix shape follows `protein_index` length and edge weights are written symmetrically by index.
