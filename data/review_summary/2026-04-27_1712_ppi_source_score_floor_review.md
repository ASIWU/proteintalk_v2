# 2026-04-27 17:12 PPI Source Score Floor Review

## Scope

- Investigated why regenerated PPI distribution plots still show a nonzero value floor near `0.15` after removing PPI score filters.

## Findings

- The regenerated `ppi_matrix.meta.json` shows `score_filter.applied: false` and `topk: 0`.
- The current PPI matrix has `6504486` nonzero entries and nonzero minimum `0.1500000059`.
- The raw PPI input file has `6508308` rows and `combined_score` minimum `150`.
- After metadata protein filtering, the mapped raw PPI rows still have `combined_score` minimum `150`.
- Because the builder scales STRING-like scores by dividing by `1000`, a source minimum of `150` becomes a matrix minimum of `0.15`.

## Change Applied

- Added `score_summary` metadata to future PPI outputs, including raw and scaled `combined_score` min/max plus fused score min/max.
- This does not change matrix values; it makes the source score floor explicit in `ppi_matrix.meta.json`.

## Verification

- `python -m py_compile utils/05_build_graph_matrices_from_global_meta.py`
- Synthetic PPI smoke test confirmed an input `combined_score=50` is retained as matrix value `0.05`, proving there is no code-side `0.15` threshold.
