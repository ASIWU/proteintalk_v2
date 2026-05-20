# 2026-04-27 17:03 PPI Threshold Removal Review

## Scope

- Reviewed PPI score filtering in `utils/05_build_graph_matrices_from_global_meta.py`.
- Removed inherited confidence/reference filtering after user requested no unapproved PPI thresholds.

## Findings

- The previous PPI builder inherited `fused >= 0.30`, `combined_score >= 400`, and low-textmining-only filtering from older preprocessing logic.
- These filters were not explicitly requested for the current metadata-aligned pipeline.
- PDI's minimum nonzero value remains source-driven by STITCH `combined_score` minimum 150 and is not a code threshold.

## Changes Applied

- Deleted the PPI reference-filter function and its call site.
- Removed `reference_filter` metadata from new PPI outputs.
- New PPI outputs now write:
  - `score_filter.applied: false`
  - a note that no PPI score threshold/filter is applied before top-k pruning.
- PPI still drops self-loops and still filters to proteins present in `global_meta.json["protein_index"]`.
- `--topk` remains explicit, but its default is now `0`, meaning no top-k pruning. Positive `--topk N` must be passed deliberately to keep only the strongest N neighbors per protein.
- Updated `README.md` and `scripts/0427_1.sh` so the default PPI command no longer passes `--topk 100`.

## Verification

- `python -m py_compile utils/05_build_graph_matrices_from_global_meta.py`
- Synthetic low-score PPI smoke test with `combined_score=100` retained the edge and wrote value `0.1`.
