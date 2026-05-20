# 2026-05-18 20:24 HKT 0518 Resume Launcher Review

## Scope

- Reviewed `scripts/0513_1.sh`, `scripts/exp_06_double_pert_pair_5fold.sh`, `scripts/exp_08_extra_double_all_train_infer.sh`, `scripts/ptv3_experiment_common.sh`, and `train.py`.
- Checked current artifacts for `20260513_double_pert_pair_5fold_double_pert_pair_fold0/1/2`.

## Findings

- `scripts/0513_1.sh` runs the double pert-pair 5-fold block before extra double and single no-pdi; stopping at fold2 means the correct continuation point is double folds `2 3 4`, then the remaining two original blocks.
- Fold0 and fold1 manifests are `fit_completed` with `test_completed`; fold2 manifest is still `fit_started`, and `logs/20260513_double_pert_pair_5fold_runtime_summary.tsv` records fold2 status `137`.
- `train.py --checkpoint-path` calls `load_model_state(...)` before `trainer.fit(...)`; it does not pass `ckpt_path` into `trainer.fit`, so it is not a full trainer-state resume mechanism.
- `ptv3_ensure_clean_path` refuses non-empty checkpoint/log paths unless `ALLOW_EXISTING_RUN=1`; blindly setting `ALLOW_EXISTING_RUN=1` would mix a rerun with stale failed-run files. Archiving incomplete fold2 artifacts before retraining keeps the restart clearer.

## Action

- Added `scripts/0518_1.sh`.
- The script skips completed candidate double folds, archives incomplete candidate artifacts under `_archived_failed_restarts/`, reruns pending folds from `2 3 4`, verifies all five reference folds are complete, then launches `exp_08_extra_double_all_train_infer` and `exp_05_single_no_pdi_5fold` with the original `0513_1.sh` defaults.
