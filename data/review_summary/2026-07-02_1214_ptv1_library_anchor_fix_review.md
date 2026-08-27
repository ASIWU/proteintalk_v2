# 2026-07-02 12:14 +0800 PTV1 Library/Anchor Fix Review

Reviewed and updated the PTV1 AIVC standardization/split path after finding that combo rows were encoded as `no + Anchor_id` and excluded from the fixed experiment-type split.

## Files Reviewed

- `utils/00_standardize_rawdata.py`
- `utils/09_build_data_splits.py`
- `utils/ptv1/01_validate_ptv1_standardized.py`
- `utils/ptv1/02_build_ptv1_training_ready.py`
- `utils/ptv1/03_validate_ptv1_training_ready.py`
- `utils/ptv1/04_build_ptv1_splits.py`
- `model/training_ready_models.py`
- `scripts/ptv1/run_ptv1_fine_tune_search.sh`

## Findings And Fixes

- PTV1 combo rows are now encoded as `Library_id + Anchor_id` in the two perturbation slots, with single-drug rows retaining duplicated single slots.
- Raw `experiment_type_list` entries are parsed as one canonical key per line. Double-drug entries now match fixed splits by canonical unordered pair instead of being split into two single-drug keys.
- PTV1 exp_12 grouping now uses a canonical perturbation pair key, preventing the same double-drug pair from crossing fold boundaries due to slot order.
- Validation now checks non-control slot coverage, combo `drugIdAB` agreement, combo perturbation indices not equal to special `no`, fixed-split combo coverage, and exp_12 pair disjointness.
- `model/training_ready_models.py` now lazy-loads PyG only for graph model execution; `fast_delta` import no longer pays the top-level `torch_geometric` import cost.
- GPU smoke revealed that tmux `gpu3` exposes the allocated H200 as local device `0`, so the formal run keeps the requested tmux session but uses `GPU_IDS=0`.

## Validation

- `python -m py_compile` passed for the touched standardization, split, validation, model, train, infer, and dataset entrypoints.
- PTV1 stage-1 and training-ready rebuild completed under `flow_v2`.
- `utils/ptv1/01_validate_ptv1_standardized.py` passed with `ptv1_aivc rows=15002`, `combo_rows=4570`, and `raw_split_unassigned_non_controls=16`.
- `utils/ptv1/03_validate_ptv1_training_ready.py` passed with `ptv1_aivc feature=15002x5576` and `ptv1_extra_singledrug feature=222x5576`.
- Smoke prefix `20260702_ptv1_library_anchor_fix_smoke` completed exp_11, exp_12 fold0, exp_13 direct, and exp_13 all-train with runtime status `0`.

