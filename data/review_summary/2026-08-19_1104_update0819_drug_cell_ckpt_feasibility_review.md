# Update-0819 Drug/Cell Checkpoint Feasibility Review

- Review time: 2026-08-19 11:04 HKT (UTC+08:00)
- Input requested by the user: `data/raw_data/update_0819/260513ptv_drug_cell_predict.csv`
- Actual repository path: `data/rawdata/update_0819/260513ptv_drug_cell_predict.csv`
- Checkpoint: `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`
- Scope: determine whether the existing checkpoint and inference artifacts can predict the supplied cell/drug pairs, with explicit checks of cell and drug label membership.
- This review did not run inference and did not modify model code, checkpoint files, metadata, derived feature artifacts, or prediction outputs.

## Input Audit

- The CSV is readable with `utf-8-sig`, has 14 rows, no missing fields, no duplicate rows, and is the complete Cartesian product of 7 cell lines and 2 drugs.
- The two query drugs are:
  - `daraxonrasib` / `RMC-6236`;
  - `zoldonrasib` / `RMC-9805`.
- The CSV contains no perturbation time, dose, drug target, machine, plate, or batch fields.

## Cell Label Audit

The model-compatible cell identifier is `cell_in_ptvdrug`, not the display-name column `Cell_name`. After applying the repository's categorical normalization rule, all seven `cell_in_ptvdrug` values are present in `data/training_ready/ptv3/global_meta.json["value_to_index"]["Cell"]`, are seen in the checkpoint's `all_train_subset_test` training rows, and have control-expression rows in the checkpoint task.

| CSV `Cell_name` | Use this `cell_in_ptvdrug` label | Global Cell index | Seen in checkpoint train | Control rows in checkpoint task |
|---|---:|---:|---:|---:|
| AsPC-1 | ASPC1 | 4 | yes | 4 |
| BxPC-3 | BXPC3 | 8 | yes | 4 |
| Capan-2 | CAPAN2 | 11 | yes | 4 |
| Panc 10.05 | PANC1005 | 66 | yes | 4 |
| Panc 03.27 | PANC0327 | 63 | yes | 4 |
| Panc 05.04 | PANC0504 | 64 | yes | 4 |
| Hs 766T | HS766T | 31 | yes | 4 |

Normalizing the display names themselves produces labels such as `ASPC_1`, `BXPC_3`, and `HS_766T`, which are not in the Cell vocabulary. A builder must therefore explicitly use the supplied `cell_in_ptvdrug` aliases.

All seven checkpoint-task controls have `cell_type=Pancreas`. Each cell has four distinct control-expression rows, so a new inference task must define whether to use one fixed control, average controls, or predict over all four controls and aggregate.

## Drug Label and Structure Audit

- Both supplied SMILES strings parse successfully with the installed RDKit.
- Neither canonical isomeric SMILES matches any entry in `global_meta.json["pertid_to_smiles"]`.
- Neither drug name or alias was found in the repository's current data/metadata search.
- Neither query produces an exact radius-2, 2048-bit Morgan fingerprint match to any current PTV3 drug row.
- Maximum Tanimoto similarity to the current PTV3 Morgan registry is low:
  - daraxonrasib: `0.2260`;
  - zoldonrasib: `0.2047`.

The two drugs are therefore genuinely out of the current global drug vocabulary and are also strong chemical OOD cases for this checkpoint, rather than simple aliases of known model drugs.

## Runtime Contract and Feasibility

The checkpoint cannot be used **directly with the current artifacts** for this CSV:

- `fast_delta` does not read raw SMILES at inference time. It looks up `pert_index1` and `pert_index2` in the fixed Morgan, graph-feature, PDI, and DDI artifacts.
- The current PTV3 drug axis has 6,131 rows, including `no` at index 6,130. The two new drugs have no legal current indices or feature rows.
- The generic stage-2 mapping falls unknown perturbation IDs back to `no`, and the fast dataset clips out-of-range indices to the last available row. A naive run could therefore silently represent either new drug as `no`, producing invalid results.
- The checkpoint uses real graph features and DDI. Its drug feature contract includes the 2,048-dimensional Morgan row, the 774-dimensional generated graph row, target inputs, and a DDI scalar. The CSV supplies only SMILES and names.
- The checkpoint also conditions on perturbation time and both dose covariates. Those values are absent from the CSV.

The checkpoint is **technically reusable without retraining** after a dedicated OOV-drug inference package is built, because the model consumes fixed-width numerical drug features rather than a learned embedding table keyed to a fixed drug count. A valid package must preserve all existing indices and append the two new drugs, generate their Morgan rows, extend PDI/DDI and graph features with the same training configuration, define target-protein inputs, build the 14 single-drug query rows with `pert_index1 == pert_index2`, attach valid control-expression rows, and choose explicit time/dose/control policies.

Even after that engineering work, the output must be labeled as drug-OOD extrapolation. The low nearest-neighbor fingerprint similarities mean the checkpoint's accuracy for these drugs is not established by current validation results.

## Conclusion

- Cell-line side: ready; all 7 labels are globally known and checkpoint-train-seen.
- Drug side: not ready; both drugs are absent from the current label and numerical feature spaces.
- Current direct prediction: no.
- Conditional prediction with the same checkpoint: feasible after extending the inference-only drug artifacts and defining missing target, time, dose, and control policies, but scientifically high-risk because both drugs are strong OOD cases.
