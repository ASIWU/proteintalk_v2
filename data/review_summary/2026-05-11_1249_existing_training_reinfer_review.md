# 2026-05-11 12:49 HKT - Existing-Training Extra Re-Inference Review

Reviewed the old extra single trained run reuse case.

Finding:

- Re-running `scripts/0509_3.sh` retrains the all-single model. That is unnecessary when the trained checkpoint directory already exists and only extra inference needs to be rerun with a chosen checkpoint.
- The existing old run `20260510_extra_single_all_train_infer_all_single_for_extra` has a `best_model_path` in its manifest pointing to `epoch=49.ckpt`.

Fix:

- Added `scripts/reinfer_extra_from_existing_training.sh`.
- The script resolves `best`, `last`, `reference`, or explicit checkpoints from an existing source run and runs only `infer.py` for extra single or extra double tasks.
- It writes to a new output experiment name by default to avoid overwriting previous outputs.

Validation:

- Shell syntax check passed.
- Bounded extra-single smoke selected `epoch=49.ckpt` from the old run and wrote 6 bounded outputs.
- Bounded extra-double smoke also passed against an existing double smoke checkpoint.
