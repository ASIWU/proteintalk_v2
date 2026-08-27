# 2026-07-10 15:05 HKT Exp32 Organoid Exp09 Implementation Review

## Scope

- Reviewed the maintained workflow guidance in `docs/Data_Process_1.md` through `docs/Data_Process_4.md` and `docs/Training_guideline.md` (the repository `AGENTS.md` lists these under `data/`, but the maintained files are under `docs/`).
- Reviewed the exp09 runner, checkpoint manifest, inference configuration validation, fast dataset, graph cache, categorical mappings, Cell/Cell-type LLM loaders, split artifacts, and existing reporter patterns.
- Reviewed the two CellType-augmented organoid matrices and `260709_samp_inf.csv`.
- Implemented, executed, and acceptance-tested exp32 end to end. Raw inputs and the maintained `data/training_ready/ptv3` artifacts were not modified.

## Data Findings

- The B matrix has 13 samples and 7,416 protein columns; the CAC matrix has 13 samples and 8,282 protein columns.
- `prot_gene` contains sample IDs, not protein/gene names. Both devices cover exactly `PTV2_1` through `PTV2_13` once.
- Tissue assignments are complete and agree exactly with the suffix join to `260709_samp_inf.csv`: 10 Lung, 2 Pancreas, and 1 Colon.
- The fixed exp09 axis has 11,092 proteins. B overlaps 7,359 axis proteins and CAC overlaps 8,190.
- Expression inputs contain no negative or infinite values. Finite values are transformed by `log1p`; missing positions remain `NaN`.
- The explicit union of both drug slots from non-control rows in the 11 exp01-exp09 main+extra task tables is exactly 3,217 drugs. All 3,217 have valid pert indices, nonzero finite Morgan rows, nonzero finite graph rows, and nonempty SMILES; 2,174 have explicit target lists.

## Checkpoint And Feature Findings

- Selected checkpoint: `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=5.ckpt`, SHA-256 `a5734682c37d807b2b6d1bb5ea66ad908107f45e15605aa65c40f6bab9b6db97`.
- Checkpoint manifest is `fit_completed`, `fast_delta`, unified head, and exact exp09 11,092-protein axis.
- Required active settings are dual pair fusion, pair-type features, real graph, DDI, graph pair add scale 0.5, PCEP, target-protein tokens, frozen Cell LLM, frozen cell-type LLM, and the full eight-field covariate list including both dose slots.
- Existing mappings are sufficient: `QE=2`, `480_FAIMS=1`, `LUNG=8`, `PANCREAS=11`, `COLON=4`, `time 24=2`, `dose 10=10`; unseen Cell/plate/batch indices remain `0`.
- The new root can reuse the exact original meta and derived artifacts. Read-only symlinks ensure resolved artifact paths remain checkpoint-identical.
- Existing `infer.py` mismatch allowance was global and did not persist mismatch details. The implementation now records all 103 comparisons and exact mismatch records; the formal exp32 manifests show zero mismatches.

## Implemented Artifacts

- CPU builder: `utils/32_build_exp32_organoid_training_ready.py`.
- Training-ready root: `data/training_ready_exp32_organoid/ptv3`.
- Tasks: `ptv3_exp32_organoid_qe_single` and `ptv3_exp32_organoid_480_faims_single`.
- Drug-scope audit: `exp32_organoid_drug_scope.json`, `.csv`, and `_sources.csv` under the exp32 group root.
- GPU runner: `scripts/exp_32_organoid_exp09_single_sensitivity_infer.sh`.
- Reporter: `scripts/report_exp32_organoid_exp09_single_sensitivity.py`.
- Result report: `docs/2026-07-10_exp32_organoid_exp09_single_sensitivity_results.md`.

## Acceptance Evidence

- Each task has 41,834 rows: 13 controls and 41,821 queries.
- Each task has 13 sets with exactly one control and 3,217 queries; `test_only` contains precisely all query rows.
- Each expression matrix is float32 shape `(41834, 11092)`; every query vector is all `NaN`, every control has finite measured values, and processed/feature matrices share an inode.
- All query rows use identical drug slots, 24 hours, dose 10 in both slots, correct machine/tissue indices, and zero Cell/plate/batch indices.
- Raw B, CAC, and sample-info SHA-256 values match before and after building.
- Static checks, full preflight, two-task 256-row smoke inference, smoke reporter, two-task full inference, and final reporter all passed.
- Formal runtime summary records status 0 for both tasks and 41,821 predictions each.
- Final combined output contains 83,642 finite `[0,1]` probabilities and exactly 41,821 paired B/CAC sample-drug keys.
- Both inference manifests use epoch 5 and report 103/103 checkpoint configuration comparisons with zero mismatches.

## Descriptive Result Review

- Overall B/CAC Pearson: `0.992097`; Spearman: `0.993484`; MAE: `0.010684`.
- Mean signed shift `CAC - B`: `-0.002734`.
- Per-sample Pearson range: `0.985673` to `0.993963`.
- Top-50 overlap range: 40-49 drugs; top-100 overlap range: 90-98 drugs.
- Highest mean absolute device difference is for `L9200_430` (`0.201001`).

There are no ground-truth labels in exp32. AUROC and AUPRC are undefined and were not computed or reported.
