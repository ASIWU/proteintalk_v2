# Process 3 / Process 4 Review Summary

Review time: 2026-04-27 20:56 HKT

Reviewed docs:

- `docs/Data_Process_3.md`
- `docs/Data_Process_4.md`

Implemented files reviewed during this pass:

- `utils/09_build_data_splits.py`
- `dataset/training_ready_dataset.py`
- `model/training_ready_models.py`
- `model/training_ready_lightning.py`
- `train.py`
- `infer.py`

Main findings:

1. Step 3 can now generate complete split artifacts for all current PTV1/PTV3 tasks under `data/training_ready`.
2. Step 4 can now construct training-ready datasets, build the requested model names, run a double-drug-compatible training loop, and run inference with structured outputs.
3. Graph model code intentionally consumes only PDI, matching the Step 4 instruction.
4. Current compact model implementation is interface-compatible but not guaranteed to be a line-by-line legacy architecture port. This is the main item requiring future user review.
5. Double-drug `pert_id` 5-fold split is implemented as ordered `pert_id1 + pert_id2` pair holdout. This is a documented assumption.
6. Extra single-drug tasks are implemented as `test_only` because Step 4 lists extra single-drug inference targets, although Step 3 only explicitly names extra_guomics/nc/nature.

Verification performed:

- `python -m py_compile` passed for all new Step 3/4 Python modules.
- `train.py --help` passed.
- `infer.py --help` passed.
- Non-graph double-drug dry run passed.
- Target protein embedding dry run passed.
- PDI graph dry run passed.
- One-batch training smoke test completed with finite losses and checkpoint output.
- Full `ptv3_extra_doubledrug_guomics` inference smoke test wrote 9001 predictions, matching the split manifest test anchor count.

Residual risks:

1. Exact legacy model architecture parity is not yet proven.
2. The desired double-drug split semantics may need revision if user expects unordered pair or individual-drug holdout.
3. Training currently runs test after fit; for quick experiments a later `--skip-test` option may be useful.
4. Inference metrics can be `NaN` for a task when that task has no valid labels in the target split.
