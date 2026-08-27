# PTV1 Library/Anchor Fix Rerun Status

Date: 2026-07-02 15:58 HKT

## Data And Code Status

- PTV1 AIVC double-drug rows now use true `Library_id + Anchor_id` perturbation slots.
- Fixed experiment-type splits and exp_12 folds use canonical single-drug or unordered double-drug keys.
- PTV1 standardized/training-ready/split/derived artifacts were rebuilt under `flow_v2`.
- Standardized and training-ready validation passed.
- `fast_delta` startup no longer imports PyG unless a graph model is actually instantiated.

## Smoke

- Prefix: `20260702_ptv1_library_anchor_fix_smoke`
- Worker: tmux `gpu3`
- CUDA mapping: the allocated H200 is visible as local device `0`, so smoke was run with `GPU_IDS=0`.
- Completed:
  - exp_11 `fixed_experiment_type`
  - exp_12 `pert_id_5fold_fold0`
  - exp_13 direct extra-single inference
  - exp_13 all-train extra-single inference
- Smoke summary:
  - `logs/20260702_ptv1_library_anchor_fix_smoke_baseline_mse050_runtime_summary.tsv`
  - `logs/20260702_ptv1_library_anchor_fix_smoke_fine_tune_results.md`
  - `outputs/2026-07/2026-07-02/20260702_ptv1_library_anchor_fix_smoke_fine_tune_results.tsv`

## Formal Run

- Prefix: `20260702_ptv1_library_anchor_fix_v1`
- Worker: tmux `gpu3`
- Command uses `GPU_IDS=0` inside the worker and graph cache `graph_cache/ptv1_library_anchor_fix_v1`.
- Candidate set: 32 configs from `scripts/ptv1/run_ptv1_fine_tune_search.sh`.
- Completed candidates with full status `0` rows: all 32 configs.
- Each completed candidate includes:
  - exp_11 `fixed_experiment_type`
  - exp_12 `pert_id_5fold_fold0` through `pert_id_5fold_fold4`
  - exp_13 direct extra-single inference
  - exp_13 all-train extra-single training and inference
- Current live status: formal suite completed.
- Aggregate runtime status at 2026-07-02 15:58 HKT: 288 rows, 288 status `0`, 0 nonzero.
- Runtime summary files are per candidate, for example `logs/20260702_ptv1_library_anchor_fix_v1_baseline_mse050_runtime_summary.tsv`.
- Final report:
  - `logs/20260702_ptv1_library_anchor_fix_v1_fine_tune_results.md`
  - `outputs/2026-07/2026-07-02/20260702_ptv1_library_anchor_fix_v1_fine_tune_results.tsv`

Final exp_11/12/13 result documentation: `docs/2026-07-02_ptv1_library_anchor_fix_exp11_12_13_results.md`.
