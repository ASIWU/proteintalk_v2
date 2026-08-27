# exp_04_v2 Random Expression Fold0 Screen

- Generated: `2026-07-08T12:44:45+00:00`
- Prefix: `20260708_exp04_v2_random_expr_screen`
- Task: `ptv3_main_singledrug` / `response`
- Split: `pert_stratified_5fold_fold0`
- Checkpoint root: `checkpoints`

## Validation
- All expected manifests are present and match the screen settings.

## Decision
- Winner: `per_row_gene_permutation`
- Winner artifact: `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_per_row_gene_permutation_seed42.npy`
- AUPRC drop vs real-control no-MSE fold0: `0.047748`
- Interpretation: the selected policy produced the largest observed fold0 degradation.
- `zero_control` is diagnostic only and was excluded from winner selection.

## Metrics
| role | policy | status | AUPRC | drop vs real | nAUPRC | AUROC | ACC | count | artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| real_reference | real_control_full_nomse | ok | 0.702725 | 0.000000 | 5.152346 | 0.911140 | 0.913413 | 3534 | `` |
| random_candidate | per_protein_normal_clip | ok | 0.708677 | -0.005952 | 5.195983 | 0.927187 | 0.919921 | 3534 | `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_per_protein_normal_clip_seed42.npy` |
| random_candidate | global_normal_clip | ok | 0.659943 | 0.042782 | 4.838670 | 0.911880 | 0.902943 | 3534 | `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_global_normal_clip_seed42.npy` |
| random_candidate | global_value_bootstrap | ok | 0.659719 | 0.043006 | 4.837027 | 0.915190 | 0.908319 | 3534 | `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_global_value_bootstrap_seed42.npy` |
| random_candidate | fixed_gene_permutation | ok | 0.743994 | -0.041269 | 5.454926 | 0.924889 | 0.922750 | 3534 | `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_fixed_gene_permutation_seed42.npy` |
| random_candidate | per_row_gene_permutation | ok | 0.654977 | 0.047748 | 4.802262 | 0.912988 | 0.899830 | 3534 | `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_per_row_gene_permutation_seed42.npy` |
| random_candidate | cross_cell_real_control | ok | 0.687633 | 0.015092 | 5.041691 | 0.914120 | 0.911998 | 3534 | `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_cross_cell_real_control_seed42.npy` |
| zero_diagnostic | zero_control | ok | 0.716967 | -0.014242 | 5.256766 | 0.926465 | 0.919638 | 3534 | `data/training_ready/ptv3/tasks/ptv3_main_singledrug/random_control_expression_zero_control_seed42.npy` |
