# Exp33 VC Double-drug Epoch-2 Implementation and Execution Review

- Review time: 2026-07-27 13:40 HKT
- Scope: exp33 builder, compact-expression dataset compatibility, GPU runner,
  reporter, generated training-ready artifacts, smoke/formal manifests, and
  final prediction outputs.
- Mode: inference only; no training or fine-tuning.

## Reviewed implementation

- `dataset/training_ready_fast_dataset.py`
- `utils/33_build_exp33_vc_doubledrug_training_ready.py`
- `scripts/exp_33_vc_doubledrug_epoch2_infer.sh`
- `scripts/report_exp33_vc_doubledrug_epoch2.py`
- `tests/test_training_ready_fast_expression_row_index.py`
- `tests/test_report_exp33_model_key_normalization.py`

The implementation is isolated from the existing training-ready root. It reuses
the existing global metadata and derived arrays read-only and creates only the
three exp33 `test_only` tasks. The new expression-row indirection is optional,
so legacy feature tables continue to address expression rows by identity.
Malformed indirection values fail before dataset sampling.

## Data and model contract findings

- Checkpoint path:
  `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`.
- Checkpoint SHA-256:
  `7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076`.
- Raw query hashes before build, after build, and after reporting are identical.
- All 227 query drugs resolve uniquely by canonical isomeric SMILES: 51 numeric
  IDs and 176 `L9200_*` IDs, with no external IDs. `(-)-Menthol` resolves to the
  confirmed `L9200_2195` ID.
- All 28 baseline cells match. Colon, lung, and pancreas use 6, 15, and 7 real
  controls plus one all-NaN query sentinel, producing expression shapes
  `(7, 11092)`, `(16, 11092)`, and `(8, 11092)`.
- All 28 cells have real Cell LLM embeddings. Eighteen train-seen cells use the
  true statistical and LLM indices. The ten train-unseen cells use
  `model_Cell_index=0` and retain a true nonzero `cell_llm_index`.
- `COLON`, `LUNG`, `PANCREAS`, `QE`, `480_FAIMS`, 24h, A-dose buckets
  `{1,2,3,4,5,8,10}`, and B-dose buckets `{1,...,10}` are checkpoint-train-seen.
  `Cell_plate=no` and `batch=no` are explicitly recorded as training OOD.
- Drug, protein, graph, PPI/PDI/DDI, Cell LLM, and cell-type LLM indices passed
  the builder and dataset preflight checks.

## Execution review

The existing `gpu:0.0` tmux pane was inspected and confirmed idle before smoke
and formal launch. The pane hosts a remote H200 worker whose inner shell cannot
resolve the host tmux socket, so the runner's local guard was disabled only
after the outer pane was verified. The initial guard-only attempt stopped before
inference and is retained in the smoke log as provenance.

Smoke results:

- colon, lung, pancreas: 256 rows each;
- finite probabilities in `[0,1]`;
- checkpoint-config mismatch count `0`;
- separate `cell_llm_index` and `cell_type_llm_index` columns confirmed;
- no expression-prediction or ranking files. The generic inference
  `metrics.json` files contain only unlabeled `count=0`/`NaN` placeholders;
  no finite labeled metric is reported.

Formal runtime:

| tissue | status | seconds | unique rows |
|---|---:|---:|---:|
| colon | 0 | 32 | 218,736 |
| lung | 0 | 58 | 462,210 |
| pancreas | 0 | 53 | 443,982 |

The first CPU reporter attempt failed before publishing a full CSV/Parquet
because feature-table `pert_time` was a string while the raw reconstruction used
an integer. The fix normalizes all seven model-key fields before both uniqueness
checking and merging. Two focused regression tests pass. The three reproducible
artifacts emitted before that failed attempt were removed and regenerated at the
same fixed-prefix paths; GPU predictions and runtime logs were never removed or
overwritten.

## Final-output verification

- Unique prediction Parquet: 1,124,928 rows.
- Full prediction Parquet: 2,526,720 rows in 28 row groups.
- Full prediction CSV: 2,526,721 physical lines including one header.
- Tissue raw-row totals: colon 403,200; lung 806,400; pancreas 1,317,120.
- A separate streamed check verified:
  - all probabilities are finite and in `[0,1]`;
  - every full-row `model_key_id` exists in the unique table;
  - every repeated model key has exactly the unique-table score;
  - tissue blocks occur as colon, lung, pancreas;
  - each `source_row_index` is contiguous from zero in original file order.
- The only reported score is `pred_unified_combo_prob`, sourced from
  `pred_task_prob`. No AUROC/AUPRC or top-ranking output was generated.
- The final summary has all acceptance flags set to `true`.

Primary artifacts:

- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_unique_model_predictions.parquet`
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_predictions.parquet`
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_predictions.csv`
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_drug_id_mapping_audit.csv`
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_covariate_audit.csv`
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_summary.json`
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_results.md`
- formal and smoke runtime log/TSV files with the same prefixes.

## Limitations

This is an unlabeled virtual screen. The scores are model probabilities, not
measured combination response or synergy and not calibrated clinical
probabilities. Ten cell lines are statistically unseen by the checkpoint, and
plate/batch use explicit training-OOD `no` categories. The Cell LLM vectors,
true tissue, recovered machine, baseline proteome, dose, and time context reduce
information loss but do not remove those extrapolation limitations.
