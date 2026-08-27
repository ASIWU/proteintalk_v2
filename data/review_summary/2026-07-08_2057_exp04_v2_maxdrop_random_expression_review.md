# 2026-07-08 20:57 HKT exp_04_v2 Max-Drop Random Expression Review

Scope:
- Reviewed the random control-expression path, PCEP control-expression use, exp_04_v2 launcher, existing seed42 baseline manifests, and fold0 feature-attribution results.

Findings:
- The original seed42 per-protein random control expression did not degrade exp01 fold0; it reproduced AUPRC `0.708677`, slightly above real-control no-MSE fold0.
- The strongest degradation came from row-wise gene permutation, not from zero control. This indicates the harmful setting is inconsistent gene/protein alignment per row, while a stable fixed wrong mapping or all-zero control can still be learned around.
- Real but cross-cell control expression only modestly reduced fold0 AUPRC, so cell mismatch alone is not the strongest randomization.

Changes:
- Extended `scripts/generate_random_control_proteome.py` with multiple policy modes and policy metadata.
- Added `scripts/run_exp04_v2_random_expression_screen_fold0.sh`.
- Added screen and final reporters:
  - `scripts/report_exp04_v2_random_expression_screen.py`;
  - `scripts/report_exp04_v2_maxdrop_random_expression.py`.
- Updated `scripts/exp_04_v2_single_no_mse_random_baseline_5fold.sh` to default to `random_control_expression_per_row_gene_permutation_seed42.npy`.

Results:
- Fold0 screen winner: `per_row_gene_permutation`, AUPRC `0.654977`, drop `0.047748` versus real-control no-MSE fold0 AUPRC `0.702725`.
- Final clean 5-fold prefix: `20260708_exp04_v2_maxdrop_random_expr_clean`.
- Final mean AUPRC: `0.592230`; current seed42 random mean AUPRC: `0.673665`; mean AUPRC drop: `0.081436`.

Verification:
- Static checks passed for the modified Python and shell scripts.
- CPU artifact header/meta validation passed for all generated policies.
- Screen and final reporters completed with validation errors `0`.
- The formal GPU runs were launched through tmux `gpu2`; the final clean run used `RUN_PREFLIGHT=0`, `RUN_DATA_VALIDATION=0`, `RUN_INFERENCE=0`, and `PROGRESS_BAR=0`.
