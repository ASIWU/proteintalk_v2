# 2026-04-27 15:35 PPI Original Scripts Comparison

## Scope

- Compared `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/ppi_pdi/ppi/process_ppi_string.py`.
- Compared `/root/beam_wuhao/H100/proteintalk/ProteinTalkv2/utils/ppi_pdi/ppi/get_ppi_string.py`.
- Checked how the current metadata-aligned PPI builder in `utils/05_build_graph_matrices_from_global_meta.py` relates to both scripts.

## Findings

- `process_ppi_string.py` is the more complete STRING preprocessing script. It reads the raw detailed STRING-like table, selects UniProt columns when present, computes fused evidence weights, filters low-confidence edges, removes self-loops, deduplicates undirected edges, applies top-k neighbor sparsification, and writes processed graph artifacts.
- `get_ppi_string.py` does not recompute evidence weights or filters. It expects a preprocessed parquet with `prot1`, `prot2`, and `w`, then aligns that edge table to a supplied UniProt list and writes a dense matrix.
- `process_ppi_string.py` is better as the source of PPI scoring/filtering logic.
- `get_ppi_string.py` is useful only if the processed edge parquet is already trusted and only matrix alignment is needed.
- The current `utils/05_build_graph_matrices_from_global_meta.py ppi` is intended to combine both needs: use the scoring/filtering behavior from `process_ppi_string.py`, but directly output a dense matrix aligned to `global_meta.json["protein_index"]`, including `control` and `no`.

## Recommendation

- For the current training-ready pipeline, use `utils/05_build_graph_matrices_from_global_meta.py ppi` with the raw detailed PPI CSV and `--topk 100`.
- Do not use `get_ppi_string.py` directly for PTV3/PTV1 because it depends on old metadata names and does not guarantee alignment to the current `global_meta.json` schema.
