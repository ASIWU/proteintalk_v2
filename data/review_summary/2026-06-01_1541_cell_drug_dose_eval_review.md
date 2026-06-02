# 2026-06-01 15:41 HKT Cell-Drug-Dose Evaluation Review

Reviewed the dose covariate path and the cell-drug collapsed evaluation logic before updating the reporter.

- Training-ready tables keep raw `pert_dose1`/`pert_dose2` float columns and also write `pert_dose1_norm`/`pert_dose2_norm` plus `pert_dose1_index`/`pert_dose2_index`.
- When `--use-dose-covariate` is enabled, `pert_dose1` and `pert_dose2` are appended to `--batch-cov-list`, and the dataset reads `pert_dose1_index`/`pert_dose2_index`.
- The fast model embeds dose as categorical covariates through `nn.Embedding`; it does not consume dose as continuous float features.
- The shared dose mapping uses normalized dose text mapped to `ceil(dose)` index values; missing values map to `no`.
- Previous exp06 cell-drug conflicts came from canonicalizing drug pairs without dose. The dose-aware grouping key now canonicalizes `(drug, dose)` tuples instead.
- Re-evaluating existing selected full-run predictions showed exp06 collapsed conflicts dropped from `10` to `0` under the cell-drug-dose key.
- Follow-up at 2026-06-01 16:03 HKT: exp07 and exp08 must be reported by extra subset, not only by `mean_extra`.
- Updated the reporter Markdown sections so exp07/exp08 subset rows are surfaced under `Extra Subset Summary`; `mean_extra` remains only as a convenience aggregate.
- Follow-up at 2026-06-01 16:25 HKT: exp08 also needs the historical `test_label` grouping: `unseenCell_seenDrugCombo`, `unseenCell_unseenDrugCombo`, and `combined`.
- Updated the cell-drug-dose reporter to emit those three exp08 groups per dataset while keeping `mean_extra` aggregated over `combined` rows only.
