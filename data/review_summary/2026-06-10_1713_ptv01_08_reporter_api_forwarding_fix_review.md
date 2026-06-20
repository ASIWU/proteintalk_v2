# 2026-06-10 17:13 HKT PTV01-08 Reporter API Forwarding Fix Review

## Scope
- Investigated the runner failure after stage1-A report generation for `20260610_ptv01_08_posweight_combo_v1`.
- Reviewed `scripts/run_ptv01_08_posweight_combo_tune.sh`, `scripts/report_cell_celltype_llm_clip10_param_search.py`, and `scripts/report_cell_llm_clip10_param_search.py`.

## Finding
- `stage1_combo_refine_configs()` imports the Cell + cell-type tuning reporter and calls `ranked_rows()`, `summarize()`, and `rank_sort_key()`.
- The Cell + cell-type reporter wrapper only exposed `main()` and a replacement `manifest_errors()` function, so imported API users failed with `AttributeError: module 'r' has no attribute 'ranked_rows'`.
- The failure happens after training/reporting, not inside the fold jobs. Completed jobs and artifacts are valid.

## Fix
- Added a single cached base reporter module inside `scripts/report_cell_celltype_llm_clip10_param_search.py`.
- Added `patched_base_report()` to install the Cell + cell-type manifest validator before any CLI or imported API use.
- Forwarded `summarize()`, `rank_sort_key()`, `ranked_rows()`, and `promotable_rows()` to the patched base reporter.

## Verification
- `python -m py_compile scripts/report_cell_celltype_llm_clip10_param_search.py scripts/report_cell_llm_clip10_param_search.py` passed in `flow_v2`.
- `bash -n scripts/run_ptv01_08_posweight_combo_tune.sh` passed.
- Re-ran the failed import path against prefix `20260610_ptv01_08_posweight_combo_v1`; the wrapper exposes `ranked_rows`, `rank_sort_key`, and `promotable_rows`, and 39 complete/valid stage1 rows were ranked.

## Recovery
- The user can rerun the same one-click command with the same prefixes. The runner should skip completed fold jobs and continue into stage1-B and subsequent stages.
- No artifact deletion is required for this failure.
