# 2026-05-26 12:38 HKT extra_doubledrug test_label review

Scope: read-only review of the updated `data/rawdata/update_0526/extra_doubledrug`
CSV files and the current data-processing / inference flow. No business code was
changed.

Findings:
- The updated Guomics, NC, and Nature files have the same row counts and identical
  common-column values as the current files under `data/rawdata/extra_doubeldrug`.
  The only new columns are `test` and `test_label`.
- Current Stage-1 extra double-drug standardization still resolves the old
  `data/rawdata/extra_doubeldrug/*26042*.csv` files and does not preserve
  `test` / `test_label`.
- Current `test_only` split for extra tasks writes all primary non-control anchors
  to test, so `test=0` / `test_label=delete` rows would still be evaluated unless
  split generation or metric aggregation filters them.
- Current `infer.py` writes only a fixed subset of row metadata to
  `predictions.parquet` and computes one overall task metric. It does not yet emit
  per-`test_label` AUPRC.
- Because the updated raw files only add metadata columns, no GPU feature rebuild
  is needed for this update by itself. Existing prediction files can be post-joined
  by row order / generated sample id to compute the new grouped AUPRCs.

Risk notes:
- Guomics `unseenCell_seenDrugCombo` has only 23 evaluable rows and 1 positive
  synergy label, so its AUPRC will be unstable.
- Official reports should not reuse current `metrics.json` as-is because those
  overall metrics include rows now marked `delete`.
