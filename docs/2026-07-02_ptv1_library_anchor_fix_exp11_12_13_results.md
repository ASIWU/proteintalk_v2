# PTV1 Library/Anchor Fix Exp 11/12/13 Results

Date: 2026-07-02 15:58 HKT

## Scope

- Data/code fix: PTV1 AIVC combo rows use true `Library_id + Anchor_id` two-slot perturbations.
- Formal prefix: `20260702_ptv1_library_anchor_fix_v1`
- Worker: tmux `gpu3`, local CUDA device `0` (`GPU_IDS=0`)
- Graph cache: `graph_cache/ptv1_library_anchor_fix_v1`
- Candidate count: 32

## Completion

- Runtime summary files: 32 candidate TSVs.
- Runtime rows: 288 total, 288 status `0`, 0 nonzero.
- Per candidate: exp_11 fixed split, exp_12 five folds, exp_13 direct inference, exp_13 all-train training plus inference.
- Exp_13 artifacts: 64 prediction parquet files and 64 metrics JSON files, matching 32 candidates x 2 exp_13 modes.

## Main Results

- exp_11 best: `focal_mse050`, AUPRC `0.876537`.
- exp_12 best: `mse025_pcep_off`, mean 5-fold AUPRC `0.547159`.
- exp_13 best by extra-single AUPRC: `mse000` all-train, AUPRC `0.640355`.
- exp_13 constrained best: unavailable because no candidate met the configured exp_11 AUPRC threshold `0.883452`.

## Report Artifacts

- Markdown report: `logs/20260702_ptv1_library_anchor_fix_v1_fine_tune_results.md`
- TSV report: `outputs/2026-07/2026-07-02/20260702_ptv1_library_anchor_fix_v1_fine_tune_results.tsv`
- Status note: `docs/2026-07-02_ptv1_library_anchor_fix_rerun_status.md`

## Notes

- `baseline_mse050` completed successfully after the Library/Anchor fix; its exp_11 AUPRC was `0.861633`, exp_12 mean AUPRC was `0.508686`, and exp_13 best mode was all-train with AUPRC `0.570846`.
- The top exp_13 rows were `mse000` all-train `0.640355`, `pos_auto_mse050` all-train `0.639750`, and `mse050_target_pdi` direct `0.632413`.
