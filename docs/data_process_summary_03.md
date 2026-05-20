# Data Process 3 Implementation Summary

Updated: 2026-04-27 21:06 HKT

This file records the Step 3 split-generation implementation from `docs/Data_Process_3.md`. It exists separately from `docs/data_process_summary_04.md` because Step 3 and Step 4 have different purposes. I originally recorded Step 3 details inside the Step 4 summary because the user requested a new `data_process_summary_04.md` after Process 4; that was incomplete documentation organization, so this file now records Process 3 explicitly.

## Implemented Script

Implemented:

- `utils/09_build_data_splits.py`

Default command:

```bash
python utils/09_build_data_splits.py --dataset-group all
```

The script reads task data from:

```text
data/training_ready/<dataset_group>/tasks/<task_name>/
```

and writes split artifacts to:

```text
data/training_ready/<dataset_group>/splits/<task_name>/
```

It also writes the global manifest:

```text
data/training_ready/split_build_manifest.json
```

## Split Index Definition

All split indices are row indices in each task's `feature_table`.

The script only splits anchor rows satisfying:

```text
not is_control and source_row_role == self and feature_membership == primary
```

Control rows are not split as anchors. They are stored in `set_info.pkl` and used by the dataset during training/inference.

This matters because the current training dataset requires every perturbation row to resolve to a control row. Anchors without a matched control are skipped.

Current skipped anchor counts:

| Task | Candidate anchors | Valid anchors | Skipped missing-control anchors |
| --- | ---: | ---: | ---: |
| `ptv1/ptv1_aivc` | 14060 | 13137 | 923 |
| `ptv3/ptv3_main_singledrug` | 18144 | 17986 | 158 |

All other current tasks have `candidate_anchor_count == valid_anchor_count`.

If the workflow requirement means every non-control row must appear in split outputs regardless of control mapping, then this part is not fully satisfied and Process 2 or the dataset pairing rule needs a design decision. If the requirement means every trainable perturbation/control pair, the current behavior is consistent.

## Generated Artifacts

For each task:

- `set_info.pkl`
- `row_to_set_index.pkl`
- `set_to_grouping.pkl`
- `train_indices_<strategy>.pkl`
- `valid_indices_<strategy>.pkl`
- `test_indices_<strategy>.pkl`
- compatibility aliases `val_indices_<strategy>.pkl`
- `train_set_info_<strategy>.pkl`
- `valid_set_info_<strategy>.pkl`
- `test_set_info_<strategy>.pkl`
- compatibility aliases `val_set_info_<strategy>.pkl`
- `split_manifest.json`

The `split_manifest.json` records:

- strategy names
- split counts
- overlap counts
- pairing audit
- label coverage audit
- implementation notes for assumptions

## Implemented Split Policies

### PTV3 Main Single Drug

Task:

- `ptv3_main_singledrug`

Strategies:

- `random`
- `cell`
- `cell_type`
- `pert_stratified`
- `pert_id_5fold_fold0..4`
- `cell_5fold_fold0..4`
- `cell_type_5fold_fold0..4`
- `pert_stratified_5fold_fold0..4`
- `all_train_subset_test`

Implementation detail:

- `pert_id` split uses `pert_id1`.
- `all_train_subset_test` puts all anchors in train and also takes 20% as a test subset. Valid is a separate train subset and is disjoint from test.

### PTV3 Main Double Drug

Task:

- `ptv3_main_doubledrug`

Strategies:

- `pert_id_5fold_fold0..4`
- `all_train_subset_test`

Implementation detail:

- `docs/Data_Process_3.md` says double drug only needs pure `pert_id` 5-fold.
- The current feature table has `pert_id1` and `pert_id2`, so the script interprets pure `pert_id` as the ordered pair `pert_id1 + pert_id2`.
- This is an assumption. If the intended definition is first-drug holdout, second-drug holdout, unordered pair holdout, or any-drug holdout, the split implementation needs to change.

### PTV3 Extra Tasks

Tasks written as `test_only`:

- `ptv3_extra_singledrug_mat1_480_faims`
- `ptv3_extra_singledrug_mat1_qe`
- `ptv3_extra_singledrug_mat2_480_faims`
- `ptv3_extra_singledrug_mat2_qe`
- `ptv3_extra_singledrug_mat3_qe`
- `ptv3_extra_singledrug_mat4_qe`
- `ptv3_extra_doubledrug_guomics`
- `ptv3_extra_doubledrug_nc`
- `ptv3_extra_doubledrug_nature`

Implementation detail:

- Step 3 explicitly marks `extra_guomics`, `nc`, and `nature` as test-only.
- Extra single-drug tasks are also written as `test_only` because Step 4 says inference targets include `extra_singledrug`.
- This is an assumption and should be checked.

### PTV1 Main

Task:

- `ptv1_aivc`

Strategies:

- `fixed_experiment_type`
- `random`
- `pert_id_5fold_fold0..4`
- `all_train_subset_test`

Implementation detail:

- `fixed_experiment_type` now parses `data/rawdata/ptv1/experiment_type_list` directly during Step 3.
- Matching uses `(Cell_plate, pert_id1)`, the same key shape used by the stage-1 projection.
- `random` is also generated for PTV1 main data as a separate train/valid/test option.
- Rows not assigned by `experiment_type_list` stay out of the fixed split but can appear in `random`, `pert_id_5fold`, or `all_train_subset_test`.

### PTV1 Extra Single Drug

Task:

- `ptv1_extra_singledrug`

Strategy:

- `test_only`

## Current Split Counts

| Task | Valid anchor count | Strategies |
| --- | ---: | --- |
| `ptv1/ptv1_aivc` | 13137 | `fixed_experiment_type`, `random`, `pert_id_5fold_fold0..4`, `all_train_subset_test` |
| `ptv1/ptv1_extra_singledrug` | 182 | `test_only` |
| `ptv3/ptv3_main_singledrug` | 17986 | `random`, `cell`, `cell_type`, `pert_stratified`, 5-fold variants, `all_train_subset_test` |
| `ptv3/ptv3_main_doubledrug` | 19777 | `pert_id_5fold_fold0..4`, `all_train_subset_test` |
| `ptv3/ptv3_extra_singledrug_mat1_480_faims` | 15834 | `test_only` |
| `ptv3/ptv3_extra_singledrug_mat1_qe` | 15834 | `test_only` |
| `ptv3/ptv3_extra_singledrug_mat2_480_faims` | 12138 | `test_only` |
| `ptv3/ptv3_extra_singledrug_mat2_qe` | 12138 | `test_only` |
| `ptv3/ptv3_extra_singledrug_mat3_qe` | 17609 | `test_only` |
| `ptv3/ptv3_extra_singledrug_mat4_qe` | 11072 | `test_only` |
| `ptv3/ptv3_extra_doubledrug_guomics` | 9001 | `test_only` |
| `ptv3/ptv3_extra_doubledrug_nc` | 16394 | `test_only` |
| `ptv3/ptv3_extra_doubledrug_nature` | 23389 | `test_only` |

## Label Coverage

The split script writes label coverage to each `split_manifest.json`.

Current result:

- all checked valid anchors have non-empty labels for the checked label column.

Checked label columns:

- main single drug: `PRISM1st_label_total`
- extra single drug: `PRISM2nd_label_total`
- main/extra double drug: `synergy`
- PTV1 main: `PRISM1st_label_total`
- PTV1 extra single drug: `PRISM2nd_label_total`

Important caveat:

- The script audits label coverage but does not fail hard when some labels are missing.
- If an entire checked label column is absent or empty for all checked anchors, the script emits a `RuntimeWarning` and records `all_labels_missing: true` plus the warning message in `split_manifest.json`.
- Double-drug main native anchors have `synergy`; merged single-drug auxiliary train anchors have `PRISM1st_label_total` and no `synergy`. For `ptv3_main_doubledrug`, label coverage is checked on native `feature_membership="primary"` double-drug anchors only, because merged single-drug rows are train-only auxiliary rows.

## Compliance Review Against Data_Process_3

Satisfied:

- train/valid/test artifacts are generated.
- `random`, `pert_stratified`, `cell`, and `cell_type` are supported for PTV3 main single-drug data.
- valid set is included.
- 5-fold splits are generated for main single-drug and main double-drug data.
- `all_train_subset_test` is generated with test as a train subset.
- double-drug 5-fold does not include cell/cell_type folds; only `pert_id` family folds are generated.
- `extra_guomics`, `nc`, and `nature` are test-only.
- `ptv1_aivc` has fixed experiment-type, random, `pert_id` 5-fold, and `all_train_subset_test` splits.
- `ptv1_extra_singledrug` is test-only.

Partially satisfied or assumption-based:

- Label coverage is audited but not enforced as a hard error.
- PTV1 handling is in the same script rather than a separate script, although the logic is separate by dataset/task branch.
- Missing-control anchors are skipped. This is necessary for the current training dataset, but it means not every candidate non-control anchor reaches split files.
- PTV1 `fixed_experiment_type` only includes valid anchors matched by `experiment_type_list`; unmatched valid anchors are recorded in `split_manifest.json`.

## Verification Commands

Commands run:

```bash
python -m py_compile utils/09_build_data_splits.py
python utils/09_build_data_splits.py --help
python utils/09_build_data_splits.py --dataset-group ptv3 --task ptv3_main_singledrug
python utils/09_build_data_splits.py --dataset-group all
```

The generated `data/training_ready/split_build_manifest.json` confirms all current PTV1/PTV3 tasks were processed.

## User Decisions Recorded 2026-05-07

1. Missing-control anchors should remain skipped; Process 2 does not need to force every anchor onto a control.
2. Double-drug `pert_id` 5-fold should keep the ordered `pert_id1 + pert_id2` combination as the unseen unit.
3. All extra tasks, both single-drug and double-drug, should remain `test_only`.
4. Target UniProt IDs missing from `protein_index` should continue to be dropped.
5. PTV1 should keep both split methods: direct `experiment_type_list` fixed split and random split.
6. PTV1 model training is out of scope for now.

Still needing clarification:

1. Whether missing checked labels should hard-fail split generation or remain manifest-only audit entries.
