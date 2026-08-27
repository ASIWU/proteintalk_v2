# Exp33 Double-drug Virtual-screen Feasibility Review

- Review time: 2026-07-27 11:10 HKT (UTC+08:00)
- Scope: proposed Exp33 double-drug virtual screen, the Exp31/Exp32 epoch-2 checkpoint identity, query/control coverage under `data/rawdata/vc_doubledrug`, baseline-proteome matching, drug-ID coverage, dose/time covariates, and current inference-artifact scalability.
- This review did not modify code, raw data, training-ready data, checkpoints, or prediction outputs.

## Checkpoint Identity

- The Exp31 epoch-2 rerun initialization and Exp32 epoch-2 inference both used:
  `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`.
- The checkpoint exists, is 389,369,531 bytes, and has SHA-256
  `7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076`.
- This is the common parent Exp09 checkpoint. It is not one of the four task-specific checkpoints produced by Exp31 fine-tuning; Exp32 did not fine-tune a new checkpoint.
- The parent Exp09 model supports two real perturbation slots and was trained through the unified single+double path, so its architecture can run double-drug inference.

## Query-data Coverage

The requested repository path is `data/rawdata/vc_doubledrug` (`rawdata`, not `raw_data`).

| tissue | query rows | cells | ordered drug pairs | unique A drug-dose records | unique B assumed doses |
|---|---:|---:|---:|---:|---:|
| colon | 403,200 | 6 | 4,800 | 50 | 441 |
| lung | 806,400 | 15 | 3,840 | 40 | 443 |
| pancreas | 1,317,120 | 7 | 3,840 | 140 | 441 |
| total | 2,526,720 | 28 | — | — | — |

- All seven required raw query columns are present.
- No query field is missing, all dose values are finite and non-negative, and every row is unique over cell, A drug/dose, B drug, and assumed B dose.
- The query contains 227 distinct drug-name/SMILES records. All 227 SMILES parse with RDKit and have a structural match in the current 3,217-drug Exp32/model scope.
- Drug-ID resolution is not yet unique:
  - 179/227 records resolve to one main-registry `pert_id` by exact raw SMILES;
  - 48/227 structures correspond to multiple existing model IDs/aliases;
  - no structure is completely unmatched.
- The 48 ambiguous records are common anchor drugs. Rows containing at least one ambiguous drug ID are:
  - colon: 359,352/403,200;
  - lung: 733,320/806,400;
  - pancreas: 1,197,756/1,317,120.
- A drug-ID selection policy or authoritative input ID mapping is therefore required before formal inference. Silent first-ID selection would change perturbation indices and can change PDI/graph features even when Morgan fingerprints are identical.

## Baseline-proteome Identification

The baseline directory exists:
`data/rawdata/vc_doubledrug/ptv2_cell_combo_virtualScreen260722_control_260724`.

- Each tissue has a raw `*_control.csv` file and a cell-aggregated `*_control_unique.csv` file.
- Exact query-cell coverage is complete:
  - colon: 6/6;
  - lung: 15/15;
  - pancreas: 7/7;
  - total: 28/28, with no missing or extra control cells.
- Raw control rows:
  - colon: 24 rows, four per cell;
  - lung: 59 rows, four per cell except `NCI-H647`, which has three;
  - pancreas: 28 rows, four per cell.
- The supplied `*_control_unique.csv` values equal the NaN-aware arithmetic mean of the corresponding raw control replicates. NaN patterns match exactly; the maximum relative numerical discrepancy is below `3.5e-15`.
- All 111 raw control sample IDs map exactly to the maintained standardized main-single metadata. This recovers one consistent `machineID_new`, one standardized `Cell`, and one `cell_type` per screened cell line.
- The unique baselines average controls across time:
  - 55 raw controls are 24-hour controls;
  - 56 are 6-hour controls;
  - every cell has two controls per time except `NCI-H647`, which has one at 24 hours and two at 6 hours.
- Protein-axis support:
  - colon: 9,714/11,092 Exp09-axis proteins present in the file;
  - lung: 10,333/11,092;
  - pancreas: 9,681/11,092.
- Per-cell finite values after axis selection range from 6,981 to 8,213. Values are non-negative and finite where present, so the maintained `log1p`-with-NaN-preservation rule is applicable.

## Proposed Baseline Rule Requiring Confirmation

1. Match query `cell` to `*_control_unique.csv::Cell` by exact string within the same tissue file.
2. Treat the supplied unique row as the baseline proteome (NaN-aware mean of all supplied raw replicates).
3. Recover the model `Cell`, `cell_type`, and `machineID_new` through the exact raw-control sample-ID joins to maintained standardized metadata.
4. Set `Cell_plate` and `batch` to `no`, because the unique baseline averages multiple plates and batches.
5. Align proteins by exact UniProt ID to the checkpoint's 11,092-protein axis, apply `log1p` to finite values, and leave absent/missing values as NaN.

This is deterministic and auditable, but it deliberately averages 6-hour and 24-hour baseline controls. A time-specific baseline rule is also possible from the raw control file and must be selected explicitly if the mixed-time mean is not intended.

## Dose, Time, and Output-semantics Gaps

- The query has no treatment-time column. The checkpoint vocabulary supports `6` and `24`; Exp32 used 24 hours, but Exp33 needs an explicit choice.
- Proposed slot semantics are:
  - `pert_id1` / `pert_dose1`: drug A / `drug_A_concentration_uM`;
  - `pert_id2` / `pert_dose2`: drug B / `assumed_combo_IC50_B_uM`.
- The maintained pipeline clips numeric dose values above 10 to 10 and uses `ceil(dose)` as the categorical index. Drug A never exceeds 10. Drug B exceeds 10 in 331,580 rows:
  - colon: 50,400;
  - lung: 103,800;
  - pancreas: 177,380.
- This dose-clipping/ceil behavior is checkpoint-compatible but requires confirmation because it discards distinctions above 10 uM.
- The checkpoint produces a unified-head probability. For a double-drug row this should be reported as an explicitly named unified/combo score unless the user confirms that it should be interpreted and labelled as synergy probability or combination sensitivity.

## Current-code Scalability

- A direct reuse of the Exp32 dense expression artifact design would require approximately
  `2,526,748 x 11,092 x 4` bytes, about 112 GB, for one float32 expression matrix including 28 controls.
- `FastTrainingReadyArtifacts.load` currently requires the expression-matrix row count to equal the feature-table row count, although inference uses real expression only from the matched control row when observed perturbation expression is off.
- Exp33 implementation should therefore use a compact expression-row indirection or an equivalent chunked inference design instead of materializing a 112-GB mostly-NaN matrix. This is an engineering change, not a missing-data problem.

## Information Required Before Formal Exp33 Execution

1. Confirm the exact checkpoint above, rather than an Exp31 task-specific fine-tuned checkpoint.
2. Provide authoritative `pert_id` values for the 48 ambiguous drugs, or approve a deterministic alias-selection rule.
3. Specify treatment time (`6` or `24` hours).
4. Confirm that the B slot dose is `assumed_combo_IC50_B_uM` and approve the existing `>10 -> 10`, then `ceil` dose rule.
5. Confirm the proposed mixed-time `*_control_unique.csv` baseline rule, or request time-specific raw-control averaging.
6. Define the desired score semantics/ranking name: unified combo score, synergy probability, or combination sensitivity.

## Documentation Note

- Repository instructions name `data/Data_Process_[1-4].md` and `data/Training_guideline.md`, but the current checkout stores these maintained guides under `docs/`.
- The relevant maintained contracts were found and reviewed in `docs/Data_Process_1.md`, `docs/Data_Process_2.md`, `docs/Data_Process_4.md`, and `docs/Training_guideline.md`.
