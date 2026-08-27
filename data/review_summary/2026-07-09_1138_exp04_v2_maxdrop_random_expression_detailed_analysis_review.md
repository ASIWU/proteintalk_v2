# 2026-07-09 11:38 HKT exp_04_v2 Max-Drop Random Expression Detailed Analysis Review

Scope:
- Reviewed the generated screen report, final 5-fold report, screen/final JSON summaries, exp04 default script, and exp01/exp03 fold0 feature-attribution report.

Findings:
- The previous generated reports were accurate but too terse for interpretation.
- `per_row_gene_permutation` should be described as a max-drop or adversarial random-expression stress test, not as a neutral random baseline.
- The result demonstrates sensitivity to row-wise gene/protein misalignment in the control/PCEP path, but it does not prove full model dependence on control expression because graph/drug/target/covariate features remain active and predictive.
- Zero control and fixed gene permutation are important negative controls: zero control is easy to route around, and fixed permutation is stable enough for the model to adapt.

Changes:
- Added `docs/2026-07-09_exp04_v2_maxdrop_random_expression_detailed_analysis.md`.
- Updated the session history with the documentation-only analysis update.

Verification:
- The detailed report was cross-checked against `outputs/2026-07/2026-07-08/20260708_exp04_v2_random_expression_screen_summary.json`, `outputs/2026-07/2026-07-08/20260708_exp04_v2_maxdrop_random_expression_summary.json`, and the existing Markdown reports.
