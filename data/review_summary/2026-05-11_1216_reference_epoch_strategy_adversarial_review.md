# 2026-05-11 12:16 HKT Reference-Epoch Strategy Adversarial Review

## Scope

Reviewed the new reference-epoch policy used by:

- `scripts/select_reference_epoch.py`
- `scripts/ptv3_experiment_common.sh`
- `scripts/exp_07_extra_single_all_train_infer.sh`
- `scripts/exp_08_extra_double_all_train_infer.sh`

The target strategy is: choose a fixed epoch from independent 5-fold reference validation checkpoint selection, retrain all-data extra model to `selected_epoch + 1`, then evaluate extra data once with `last.ckpt`.

## Loopholes Found and Fixed

1. Reference paths could be too broad and silently mix experiments.
   - Fix: selector now rejects duplicate split strategies and mixed reference configs by default.
   - It checks model/loss/monitor/optimizer/key hyperparameter fields for homogeneity.

2. Reference paths could include the wrong task, head, model, dataset group, or split family.
   - Fix: selector now supports and scripts pass strict task/head/model/dataset/split-regex checks.
   - Script 07 defaults to `^pert_stratified_5fold_fold[0-9]+$`.
   - Script 08 defaults to `^pert_id_5fold_fold[0-9]+$`.

3. Reference manifests could point to missing or unusable checkpoints.
   - Fix: selector now requires parsed checkpoint files to exist by default and requires non-null `best_model_score`.

4. A partial reference run could be used accidentally.
   - Fix: scripts default to `REFERENCE_REQUIRE_TEST_COMPLETED=1`; selector requires `run_status=fit_completed`.

5. The default reference count was unsafe for a 5-fold policy.
   - Fix: `REFERENCE_EPOCH_MIN_COUNT` default changed from `1` to `5`.

6. `SAVE_LAST_CKPT=0` would make reference policy fail only after all-data training.
   - Fix: scripts now fail before training when reference policy is enabled and `SAVE_LAST_CKPT != 1`.

7. The all-data manifest did not contain enough reference provenance.
   - Fix: `reference_epoch_policy.reference_summary` now embeds fold-level reference details.

8. `SCHEDULER_NAME=plateau` could let the all-data validation split affect final weights through the learning-rate schedule.
   - Fix: scripts 07/08 now reject plateau scheduler when reference policy is enabled.

## Validation

- Strict selector accepted `checkpoints/20260510_single_pert_stratified_5fold` and selected median epoch `52`.
- Broad selector input `checkpoints/` failed as intended due duplicate `pert_stratified_5fold_fold0` and mixed reference config.
- `SAVE_LAST_CKPT=0` reference-policy script path failed before training as intended.
- `SCHEDULER_NAME=plateau` is now rejected in the reference-policy script path before training.
- `bash -n` passed for affected shell scripts.
- `py_compile` passed for Python entrypoints.
- `git diff --check` passed for changed files.
- 8-GPU bounded smoke passed:
  - `20260511_ref_policy_single_smoke3`
  - `20260511_ref_policy_double_smoke3`
  - `20260511_ref_policy_extra_single_smoke4`
  - `20260511_ref_policy_extra_double_smoke4`

## Residual Boundary

No known code-path loopholes remain in the reference binding, checkpoint selection, or extra inference execution path after these checks. This does not prove the chosen reference epoch is scientifically optimal; it proves the implemented strategy no longer uses extra test labels for checkpoint selection and fails fast on the identified invalid reference inputs.
