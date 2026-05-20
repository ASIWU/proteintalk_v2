# 2026-04-27 15:27 PPI Parameter Review

## Scope

- Reviewed current PPI implementation in `utils/05_build_graph_matrices_from_global_meta.py`.
- Compared it with the original scripts under `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/ppi_pdi/ppi/`.

## Findings

- PPI generation is not implemented in `utils/04_build_embeddings_from_global_meta.py`; it is implemented as the `ppi` subcommand in `utils/05_build_graph_matrices_from_global_meta.py`.
- `--node-mapping-json` is an optional compatibility layer for edge files whose node IDs are not UniProt IDs. It maps metadata UniProt accessions to the node IDs used in the edge table.
- The local Westlake raw CSV and processed parquet both include `prot1` / `prot2` columns containing UniProt IDs, so no `--node-mapping-json` is needed for those files.
- The current builder detects `prot1` / `prot2` before `protein1` / `protein2`. Therefore, if an edge file contains both UniProt columns and STRING node columns, a UniProt-to-STRING mapping should not be passed unless the code is also changed to use the STRING columns.
- `--topk` controls per-row sparsification. It keeps only the strongest K neighbors for each protein before final symmetric matrix writing. The default value `100` comes from the newer `process_ppi_string.py` logic, not from `get_ppi_string.py`.

## Recommendation

- For `/root/beam_wuhao/H100/vcc_data/westlake/processed_string/edges.parquet`, run PPI without `--node-mapping-json`.
- Keep `--topk 100` for consistency with `process_ppi_string.py`; tune it only after checking graph density / isolated-node counts from the generated matrix metadata.
