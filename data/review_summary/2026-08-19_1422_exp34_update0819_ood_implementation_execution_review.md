# Exp34 Update-0819 OOD Implementation and Execution Review

- Review time: 2026-08-19 14:22 HKT (UTC+08:00)
- Input: `data/rawdata/update_0819/260513ptv_drug_cell_predict.csv`
- Checkpoint: `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`
- Checkpoint SHA-256: `7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076`
- Execution target: outer tmux `gpu1_deep:0`, H200 worker `cuda:0`
- This experiment performed inference only; it did not train or fine-tune the checkpoint.

## Implemented Contract

- The 14 input rows form the complete product of seven checkpoint-train-seen
  pancreatic cell lines and two drugs.
- Each cell uses the checkpoint-task control with the minimum
  `feature_row_index`. Query covariates retain that control's machine, plate,
  batch, Cell, and cell-type indices.
- Both drugs receive stable new `exp34smiles::*` IDs and remain OOD even if a
  similar historical compound exists. Similarity can transfer only
  `target_protein_list`, with threshold `0.5` and fail-fast handling for tied
  candidates with conflicting target lists.
- Daraxonrasib and zoldonrasib have maximum historical Morgan Tanimoto values
  `0.2259887006` and `0.2046783626`; neither inherited historical targets.
- The extended drug axis has 6,133 rows. Original Morgan, DDI, PDI, and graph
  blocks are copied unchanged. New Morgan/DDI rows are generated from the query
  SMILES, new PDI rows are zero, and new graph rows use source graph mean/std.
- Source graph regeneration matched exactly (`max_abs=0.0`); both appended
  graph rows are finite.
- Two tasks share all numerical artifacts:
  - `target_none`: empty target lists;
  - `target_mechanism`: daraxonrasib `[NRAS, HRAS, KRAS]`, zoldonrasib `[KRAS]`.
  PPIA is recorded as a tri-complex cofactor but is not a model target, and PDI
  remains zero in both tasks.

## Storage and Runtime Handling

- The shared project filesystem reported only about 4.8 GB free and returned
  `Disk quota exceeded` while publishing a persistent expanded root. No user
  data was deleted. Failed staging output was automatically removed.
- The reproducible builder remains in the repository, while expanded matrices
  were built under `/tmp/proteintalk_exp34_update0819_ood_runtime` inside the
  same GPU worker that performed inference.
- Audits, build summary, axes, predictions, expression matrices, manifests,
  logs, and reports were copied or written to the dated repository output tree,
  so final interpretation does not depend on the lifetime of the `/tmp` root.

## Verification and Repairs

- New tests plus fast-dataset regression: `9 passed, 4 subtests passed`.
- Initial smoke was correctly blocked because an rlaunch worker cannot resolve
  the outer tmux socket. The runner now accepts the outer resolved target via
  `EXP34_TMUX_TARGET=gpu1_deep:0` while retaining the exact-target check.
- The first inference attempt exposed missing explicit `cell_llm_index` and
  `cell_type_llm_index` columns. The builder now writes the known Cell and
  cell-type indices explicitly; no predictions were produced by the failed
  attempt.
- The first report attempt treated source-control NaNs as an error. The report
  now applies the model's exact `nan_to_num` control contract before expression
  comparisons.
- Final smoke produced two rows per task and passed probability, expression,
  alignment, and target-branch checks.
- Formal inference produced 14 rows per task. Both runtime records have status
  `0` and duration 6 seconds.
- Strict reporting confirmed exactly four checkpoint mismatches, all expected
  artifact paths (`meta`, Morgan, PDI, DDI); architecture and protein axis match.

## Formal Results

- Combined predictions: 28 rows; target-branch comparisons: 14 rows.
- Expression predictions: `14 x 11092` per branch, all finite.
- `target_none` probability range: `0.0112188..0.0318734`, mean `0.0176361`.
- `target_mechanism` range: `0.0115026..0.0415120`, mean `0.0194207`.
- Mean mechanism-minus-none shift: `0.00178459`; maximum shift: `0.00963855`.
- Highest score in both branches is ASPC1 + zoldonrasib: `0.0318734` without
  targets and `0.0415120` with the KRAS target list.
- Mean by drug:
  - daraxonrasib: `0.016226` without targets, `0.016890` with targets;
  - zoldonrasib: `0.019046` without targets, `0.021952` with targets.

These are unlabeled chemical-OOD extrapolations. No AUROC/AUPRC is reported,
and the target-branch difference is a sensitivity analysis rather than an
uncertainty calibration.

## Primary Outputs

- `outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2_predictions.{csv,parquet}`
- `outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2_target_branch_comparison.{csv,parquet}`
- `outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2_summary.json`
- `outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2_results.md`
- Both task directories under `outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2/` contain `predictions.parquet`, `expression_pred.npy`, `metrics.json`, and `run_manifest.json`.
