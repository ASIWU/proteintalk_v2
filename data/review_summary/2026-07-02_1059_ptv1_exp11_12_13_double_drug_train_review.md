# 2026-07-02 10:59 +0800 PTV1 exp11/12/13 Double-Drug Train Review

Reviewed whether PTV1 double-drug samples are involved when running the PTV1 exp_11/12/13 scripts.

## Files Inspected

- `scripts/ptv1/exp_11_ptv1_random_split.sh`
- `scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh`
- `scripts/ptv1/exp_13_ptv1_extra_single_all_train_infer.sh`
- `scripts/ptv1/exp_13_ptv1_extra_single_from_exp11.sh`
- `scripts/ptv1/ptv1_experiment_common.sh`
- `utils/00_standardize_rawdata.py`
- `utils/09_build_data_splits.py`
- `data/rawdata/ptv1/experiment_type_list/*.txt`
- `data/training_ready/ptv1/tasks/ptv1_aivc/feature_table.csv`
- `data/training_ready/ptv1/splits/ptv1_aivc/*_indices_*.pkl`

## Findings

- The raw PTV1 `experiment_type_list` files do contain double-drug entries:
  - train: 1071 lines, 288 double-drug lines.
  - valid: 306 lines, 93 double-drug lines.
  - test: 153 lines, 35 double-drug lines.
- Current PTV1 standardization represents raw combo rows with blank `pert_id1`, nonblank `pert_id2`, and combo `drugname`; all 4570 combo-like primary anchors have `data_split=no`.
- This happens because `standardize_ptv1()` assigns `data_split` by `(protein_plate, pert_id1)`, while raw combo rows have blank `pert_id`.
- `exp_11_ptv1_random_split.sh` actually trains `split_strategy=fixed_experiment_type`; its train/valid/test splits contain 0 combo-like double-drug anchors.
- `exp_12_ptv1_unseen_drug_5fold.sh` uses `pert_id_5fold_fold*`:
  - folds 0-3 train each include 3724 combo-like double-drug anchors, with 133 Y and 3591 N labels.
  - fold 4 train includes 0 combo-like double-drug anchors; fold 4 test includes 3724 combo-like double-drug anchors.
- `exp_13_ptv1_extra_single_all_train_infer.sh` and `exp_13_ptv1_extra_single_from_exp11.sh all_train` use `all_train_subset_test`; train includes 3724 combo-like double-drug anchors, with 133 Y and 3591 N labels.
- `exp_13_ptv1_extra_single_from_exp11.sh direct` performs inference from the exp_11 checkpoint and does not train an all-PTV1 model, so it does not add double-drug training rows beyond exp_11.
- For those combo-like rows, `pert_id1` is blank and maps to special perturbation index `no` (`pert_index1=147`), while `pert_id2` maps to the anchor drug only. The library drug from the raw combo row is not present in the two-slot drug input, so combo rows that enter training are not represented as the true two-drug pair.

## Conclusion

The raw fixed split list has double-drug entries, but current `fixed_experiment_type` training does not include the corresponding PTV1 combo rows. In contrast, exp_12 folds 0-3 and exp_13 all-train do include PTV1 combo-like double-drug anchors in training because those split strategies operate on valid anchors directly and do not filter by `data_split`. However, those combo rows are not correctly encoded as two-drug pairs in the current feature table; they are effectively encoded as `no + anchor_drug`.
