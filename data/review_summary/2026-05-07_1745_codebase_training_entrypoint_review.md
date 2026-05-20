# 2026-05-07 17:45 HKT Codebase Training Entrypoint Review

Update note, 2026-05-07 19:06 HKT: the high/medium findings in this review
were fixed in code. Numeric labels are now recognized, eval control selection
is deterministic, checkpoint loading/config checks are strict by default,
DDP strategy handling was added, valid fallback uses test `set_info`, baseline
embedding row mismatch now raises, and a focused smoke script was added at
`scripts/0507_training_stack_smoke.py`.

## Scope

Reviewed the current training-ready codebase, with emphasis on:

- `dataset/training_ready_dataset.py`
- dataloader construction in `train.py`
- `train.py`
- `infer.py`
- trainer/model interfaces in `model/training_ready_lightning.py` and
  `model/training_ready_models.py`

## Findings

### High: Numeric `0.0` / `1.0` labels are masked as missing

`dataset/training_ready_dataset.py:94` converts labels to lowercase strings and
only accepts `"0"` / `"1"`, not `"0.0"` / `"1.0"`.

Confirmed from current data:

- `ptv1_extra_singledrug` uses `PRISM2nd_label_total` as `float64`.
- Non-control label examples are `1.0` and `0.0`.
- `encode_response_label(1.0)` returns `(0.0, 1.0)`, meaning label value 0 and
  mask 1, so it is ignored.

Impact:

- `ptv1_extra_singledrug` task-1 labels are effectively all missing during
  inference metrics and any future training/evaluation path that uses that
  label.

Plan:

- Update `encode_binary_label` to normalize numeric values before string
  matching.
- Add a small label-encoding smoke test for `1`, `0`, `1.0`, `0.0`, `"1"`,
  `"0"`, `"Y"`, `"N"`, and current text labels.

### High: Evaluation and inference sampling is nondeterministic

`ProteinTalkDataset.__getitem__` uses `np.random.choice` in eval mode at
`dataset/training_ready_dataset.py:276` to choose a control row.

Impact:

- Validation, test, and inference can change across runs if a set has more than
  one control row.
- This affects checkpoint selection because validation loss depends on eval
  sampling.

Plan:

- Make eval mode deterministic by selecting a stable control row, likely the
  first sorted control index.
- Keep random control sampling only for train mode.
- Optionally add an explicit `eval_control_policy` later, but the default should
  be deterministic.

### High: Checkpoint loading can silently run with the wrong model/config

`train.py:156` loads checkpoint state with `strict=False` and does not report
missing/unexpected keys. `infer.py:69` prints missing/unexpected keys but still
continues by default.

Impact:

- After model consolidation, old checkpoints or checkpoints trained with
  different `--use-target`, gate, hidden size, layer count, or model type can
  partially load and then produce predictions with random/unloaded modules.

Plan:

- Add default strict config validation for inference:
  - read checkpoint hyperparameters and/or `run_manifest.json`;
  - compare model type, hidden dim, number of heads/layers, use-target, gate,
    fusion settings, ordered protein dimension, and expression dimension.
- Make `infer.py` fail on missing/unexpected checkpoint keys by default.
- Provide an explicit override such as `--allow-partial-checkpoint-load` for
  intentional migration/debug cases.
- Make `train.py --checkpoint-path` terminology clear: either weight
  initialization only or true Lightning resume.

### High: Multi-GPU training can fail with unused parameters

The consolidated graph model always defines modules that may be unused for a
given configuration:

- `target_proj` is unused when `--use-target` is false
  (`model/training_ready_models.py:344`, `model/training_ready_models.py:554`).
- `pert_proj` is unused in the no-target graph path unless
  `--perturb-fusion-mode mlp` is selected
  (`model/training_ready_models.py:527`).

`train.py` does not expose or set a DDP strategy
(`train.py:372`).

Impact:

- Multi-GPU DDP can error unless `find_unused_parameters=True` is used.
- Legacy `train_dd.py` used a DDP strategy that allowed unused parameters.

Plan:

- Add `--strategy` to `train.py`.
- Default to `ddp_find_unused_parameters_true` when using multi-device DDP, or
  document and set it explicitly for graph configs.
- Longer term: avoid constructing unused modules for no-target/no-mlp configs,
  but that is lower priority than making training robust.

### Medium: `valid_indices` fallback can pair test rows with empty valid set_info

In `train.build_data_loaders`, `valid_indices` falls back to `test_indices` when
empty (`train.py:90`), but `valid_set_info` is then loaded from the valid split
because `valid_indices` is no longer empty (`train.py:94`).

Impact:

- If a split has an empty valid list but non-empty test list, the validation
  dataset can use test indices with empty valid `set_info`, causing a KeyError
  or invalid pairing.

Plan:

- Preserve whether valid indices were originally present before fallback.
- If falling back to test indices, use `test_set_info`.
- Record the fallback in the run manifest.

### Medium: Baseline embedding row mismatch is silently clamped

`BaselineEmbV3.forward` clamps feature row indices to the embedding dataset
length at `model/training_ready_models.py:668`.

Impact:

- If `emb_dataset_path` has fewer rows than the current feature table, all
  out-of-range rows map to the last embedding row instead of failing.

Plan:

- Validate `emb_dataset.shape[0] > max(feature row index)` before training or
  inference.
- Remove silent clamp for row indices; raise a clear error on mismatch.

### Medium: Full expression metrics can be memory-heavy in inference

`infer.py` stores all expression predictions/labels/control expressions in
lists before writing metrics (`infer.py:208`, `infer.py:271`).

Impact:

- For large extra datasets and 11k proteins, `--save-expression-pred` can use a
  large amount of host memory.

Plan:

- Keep current behavior for small/medium runs.
- Add chunked `.npy` / memmap writing and streaming metric accumulation before
  running full-size production inference.

## Current Positives

- Dataset/dataloader contract matches the shared `{control, perturb}` shape.
- Split artifact loading is straightforward and aligned with Process 3 outputs.
- `train.py` now uses validation for checkpoint selection and tests from the
  best validation checkpoint.
- `infer.py` now computes the full legacy-style metric suite when expression
  prediction is requested.
- Active model choices are correctly consolidated to `attention_v10_hetero_cls_ee`
  and `baseline_emb_v3`.

## Next Modification Plan

1. Fix numeric label encoding in `dataset/training_ready_dataset.py`.
2. Make eval/test/inference control selection deterministic.
3. Harden checkpoint loading and config validation in `infer.py` and clarify
   `train.py --checkpoint-path` behavior.
4. Add `--strategy` / DDP unused-parameter handling to `train.py`.
5. Fix valid fallback set-info handling in `build_data_loaders`.
6. Add baseline embedding row-count validation and remove silent clamp.
7. Add a lightweight regression smoke script covering:
   - PTV1 extra single-drug numeric labels;
   - deterministic eval item retrieval;
   - train/infer CLI model config compatibility;
   - graph model no-target and target/gate forward.
