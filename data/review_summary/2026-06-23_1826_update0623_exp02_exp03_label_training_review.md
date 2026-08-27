# 2026-06-23 18:26 HKT update_0623 exp02/exp03 Label and Training Review

## Scope

- Reviewed why exp02 (`cell_type_5fold`) and exp03 (`cell_5fold`) can still look poor when compared against the legacy PTV1-flow benchmark after the `update_0623` metadata refresh.
- Checked whether `update_0623` labels propagated into standardized/training-ready PTV3 artifacts.
- Recomputed key-level label agreement between current PTV3 `PRISM1st_label_total` labels and legacy PTV1 `pheno.csv` labels for both `cell_type_5fold` and `cell_5fold`.
- Checked exp02/exp03 final run manifests under prefix `20260623_130336_update0623_nowandb`.

## Findings

- The refreshed metadata is being used. `utils/00_standardize_rawdata.py` points `ptv3_main_singledrug` to `data/rawdata/update_0623/260513ptv3_EGH_28602sampinfo_with_smiles_check_prism1_label_add_prism2_label_add_machine_details.csv`.
- Sample-level propagation is consistent:
  - raw update CSV vs standardized `info.csv`: 28,602/28,602 sample IDs joined; 0 mismatches for `PRISM1st_label_total`, `PRISM2nd_label_total`, `Cell`, `pert_time`, `pert_id -> pert_id1`, and `pert_dose -> pert_dose1`.
  - raw update CSV vs training-ready `feature_table.parquet`: 18,359/18,359 training-ready rows joined; 0 mismatches for the same fields.
- Current PTV3 main single-drug PRISM labels are internally consistent at key level:
  - 9,004 non-control `(Cell, pert_id1, pert_id2)` keys.
  - 0 conflicting PRISM1 labels across rows of the same key.
  - key label counts: 7,929 non-responsive keys and 1,075 sensitive keys.
- The legacy PTV1 benchmark label namespace is still not the same as current PTV3 PRISM labels:
  - For both `cell_type_5fold` and `cell_5fold`, the 5 test folds contain 3,673 PTV1 valid 0/1 keys.
  - All 3,673 keys exist in current raw PTV3 metadata, but 40 have no non-missing PRISM1 label and are filtered out of the training-ready main task.
  - On the remaining 3,633 comparable keys, 456 labels still disagree: mismatch rate `12.5516%`.
  - Confusion table on comparable keys:

| PTV1 label | PTV3 PRISM1 label 0 | PTV3 PRISM1 label 1 |
|---:|---:|---:|
| 0 | 2587 | 147 |
| 1 | 309 | 590 |

- This is almost the same issue documented on 2026-06-21. The previous report found 469 mismatches out of 3,673 comparable keys (`12.77%`). After `update_0623`, there are 456 mismatches among 3,633 comparable keys plus 40 PTV1-valid keys whose PRISM1 label is now missing.
- The current exp02/exp03 training commands still use `ptv3_main_singledrug`, not the PTV1-aligned derived task:
  - `scripts/run_ptv3_training_experiments.sh` calls exp02 as `ptv3_main_singledrug "cell_type_5fold_fold${fold}" response`.
  - The same script calls exp03 as `ptv3_main_singledrug "cell_5fold_fold${fold}" response`.
  - `scripts/exp_03_single_cell_5fold.sh` also defaults `TASK_NAME` to `ptv3_main_singledrug`.
- The dedicated PTV1-label aligned task exists separately as `data/training_ready/ptv3/tasks/ptv3_main_singledrug_ptv1_cell_5fold`, but it was not the task used by the `20260623_130336_update0623_nowandb` exp02/exp03 runs.

## Training Status

- exp02 and exp03 did not show a training failure in the final manifests:
  - All 10 exp02/exp03 fold manifests report `run_status=fit_completed` and `test_status=test_completed`.
  - exp02 row-level PRISM-label mean metrics: AUROC `0.934863`, AUPRC `0.791372`.
  - exp03 row-level PRISM-label mean metrics: AUROC `0.922944`, AUPRC `0.766116`.
- exp02 fold3/fold4 test sets are small after label filtering (`192` and `194` active response rows), so those folds should be interpreted with caution, but they are not low-scoring folds.
- The observed gap against PTV1-flow should therefore be attributed primarily to label/objective mismatch, not to a current training collapse.

## Conclusion

`update_0623` was correctly loaded and propagated, but it did not convert `ptv3_main_singledrug` into the legacy PTV1 `pheno.csv` label namespace. The current exp02/exp03 runs still train and evaluate against current PTV3 `PRISM1st_label_total`. If the target comparison is PTV1-flow, the correct path is to train/evaluate a PTV1-label aligned task such as `ptv3_main_singledrug_ptv1_cell_5fold` and to keep the PTV1 `-1` mask semantics.
