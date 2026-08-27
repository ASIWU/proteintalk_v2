# Exp32 Main-Single Registry Versus Screen-Scope Review

- Review time: 2026-07-22 21:52 HKT (UTC+08:00)
- Scope: reconcile the 3,000+ main-single raw drug registry with the previously reported 2,210 Exp32 canonical-origin count and the 3,217-drug Exp32 screen scope.
- No raw, standardized, training-ready, feature, checkpoint, or prediction artifact was modified.

## Findings

- The update-0623 main-single raw table has 28,602 rows and 3,115 unique `pert_id` values. One is the `control` ID, leaving 3,114 actual drugs.
- The standardized main-single table preserves all 28,602 rows and all 3,115 IDs.
- The mat1-4 existing-drug lookup is built directly from this full raw main-single table. It therefore had access to all 3,115 main IDs; the mat mapping audit was not restricted to 2,210 IDs.
- The main-single training-ready feature table is label-filtered: non-control rows require non-empty `PRISM1st_label_total`. It has 17,935 query rows and only 2,119 unique query drug IDs, exactly matching the 2,119 main IDs with a non-empty PRISM1 label.
- Exp32 builds its screen scope from the union of non-control drugs in 11 training-ready feature tables, not from the raw/main canonical registry. This produces 3,217 IDs.
- Of those 3,217 IDs, 2,210 occur in the raw main-single registry:
  - 2,119 enter directly through the label-filtered main-single feature table;
  - 91 additional main-registry IDs enter through other extra/double feature tables.
- The prior `canonical_origin_task=ptv3_main_singledrug: 2,210` count means that 2,210 IDs in the already selected Exp32 scope obtain their first canonical SMILES from main-single raw data. It is not the total number of main-single drugs.
- A total of 905 raw main IDs are absent from the current Exp32 scope. One is `control`; the remaining 904 are real drugs with non-empty raw SMILES, entries in the global `pert_index`, stored global-meta SMILES, and nonzero Morgan features.
- The union of the existing Exp32 scope and all valid-SMILES main-single drugs is therefore 4,121 IDs.

## Interpretation

- There is no evidence that mat mapping accidentally used only 2,210 main drugs; it used the full raw main registry.
- There is a separate Exp32 library-coverage issue. If Exp32 was intended to screen only compounds represented in label-filtered training-ready task rows, 3,217 is consistent with the implemented contract.
- If Exp32 was intended to screen every valid main-single registry drug plus the current extra/double union, the current screen omitted 904 valid main drugs and the intended scope should be 4,121 rather than 3,217.
- The omission occurs in `utils/32_build_exp32_organoid_training_ready.py::build_drug_scope`, which reads training-ready feature tables after the stage-2 label filter in `utils/02_build_training_ready_data.py::apply_processed_filter`.
