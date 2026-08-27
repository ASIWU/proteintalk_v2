# 2026-05-28 11:53 HKT Update 0526 Repro Review

## Scope
- Reviewed current `git diff` for code changes relative to the checked-out major version.
- Verified regenerated PTV3 data axes against `scripts/0427_1.sh` embedding and graph artifacts.
- Re-ran `exp_01` through `exp_08` on one GPU with prefix `20260528_repro_update0526_h512_lr2e4_v1`.
- Compared against previous reference prefix `20260526_full_h512_lr2e4_nowandb_v1`.

## Findings
- Data processing scripts `utils/00_standardize_rawdata.py`, `utils/02_build_training_ready_data.py`, and `utils/09_build_data_splits.py` have no working-tree diff.
- Current generated `global_meta.json` matches the 0427 protein/drug embedding item indexes exactly; PPI/PDI/DDI matrix shapes match those axes.
- Current dirty code diff is limited to MSE-gap experiment controls in training/inference/model scripts; the new controls are default-off or neutral for baseline runs.
- `exp_01` through `exp_06` primary metrics (`task_auprc`, `task_auroc`, `task_acc`, `task_count`) match the 20260526 reference run exactly.
- `exp_07` extra single metrics match the 20260526 reference run exactly.
- Scripted `exp_08` does not match the 20260526 reference output because current `test_only` splits for updated extra double tasks filter to `test == 1 and test_label != delete`.
- A no-code all-anchor inference rerun using the same `exp_08` checkpoint restores the 20260526 reference `exp_08` metrics exactly.

## Artifacts
- Runtime summary: `logs/20260528_repro_update0526_h512_lr2e4_v1_runtime_summary.tsv`
- Scripted filtered exp08 output: `outputs/2026-05/2026-05-28/20260528_repro_update0526_h512_lr2e4_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra`
- All-anchor exp08 output matching 20260526: `outputs/2026-05/2026-05-28/20260528_repro_update0526_h512_lr2e4_v1_exp08_extra_double_all_train_infer_all_single_double_for_extra_allanchors`
