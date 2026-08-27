# Exp35 Update-0821 Manual-target Implementation and Execution Review

- Review time: 2026-08-21 16:31 HKT (UTC+08:00)
- Input: `data/rawdata/update_0821/260820ptv_drug_cell_predict_target.csv`
- Input SHA-256: `d14a6375d8160039f1adf0a9cae1f32e31b9805447f89b5328b192b580a67909`
- Checkpoint: `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`
- Checkpoint SHA-256: `7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076`
- Execution target: tmux `gpu1:0`, H200 worker `cuda:0`
- Scope: inference and attribution only; no checkpoint training or fine-tuning.

## Verdict

The CSV-driven manual-target workflow is implemented and the requested formal
outputs are complete. The 14 query rows are valid, target parsing is explicit
and fail-fast, all target accessions occur on the checkpoint protein axis, and
the generated model inputs reproduce the previously validated Exp34 mechanism
condition.

All 14 predicted sensitivity probabilities are below 0.5 and are exported as
`non-responsive`. This label is the checkpoint's negative response class. It
must not be interpreted as validated clinical drug resistance because the
queries are unlabeled chemical-OOD extrapolations.

## Implemented Contract

- The builder consumes the `target` column rather than a hard-coded target map.
- Target cells are split on semicolons, trimmed, upper-cased, deduplicated, and
  checked against the UniProt accession pattern and the checkpoint protein
  axis. Empty targets, invalid accessions, more than 32 targets, conflicting
  per-drug target sets, and duplicate cell-drug keys fail.
- Daraxonrasib resolves to `P01111/P01112/P01116` and protein indices
  `[1374,1375,1376]`; zoldonrasib resolves to `P01116` and `[1376]`.
- The task uses the same seven checkpoint-train-seen cell controls, 24-hour
  exposure, 10-uM dose, OOD compound identities, Morgan/DDI extension, and
  fixed epoch-2 architecture as Exp34. New PDI rows remain zero.
- Only `ptv3_exp35_update0821_manual_target` is built and inferred; no
  target-none branch is created.
- The runner refuses existing outputs, verifies the checkpoint hash and build
  preflight, requires `gpu1:0`, and now verifies an NVIDIA driver plus PyTorch
  CUDA availability before runtime construction.

## Verification and Runtime Findings

- Seven `unittest` cases passed: parser normalization/deduplication, empty and
  invalid target rejection, real-input contract, protein-axis mapping,
  conflicting per-drug target rejection, and duplicate-key rejection.
- Shell syntax and Python byte-compilation checks passed.
- GPU smoke produced two predictions and a `2 x 11092` expression matrix.
  Smoke report export passed, and IG-4/8 reproduced both probabilities exactly
  with zero completeness warnings.
- The first attempted smoke ran in the outer tmux host rather than an H200
  worker. It failed at CUDA initialization and produced no valid inference
  directory. This led to the explicit CUDA preflight described above. The
  subsequent smoke and formal jobs ran inside an H200 worker in `gpu1:0`.
- Formal inference produced 14 predictions and a finite `14 x 11092`
  expression matrix.
- Regression against the August 19 mechanism branch:
  - probability maximum absolute error: `0.0`;
  - perturbed-expression maximum absolute error: `0.0`;
  - matched-control maximum absolute error: `9.5367431640625e-7`, within the
    declared `1e-6` float32/legacy-decimal-CSV round-trip tolerance.
- Formal IG-32/64 reproduced all probabilities exactly, emitted 155,288
  sample-protein records, performed zero adaptive reruns, and had zero
  completeness warnings. Maximum absolute completeness error was
  `0.0001820028`.

## Formal Result Summary

- Samples: 14 (seven cells x two drugs).
- Protein axis: 11,092.
- Sensitivity probability range: `0.0115025574..0.0415119715`.
- Mean sensitivity probability: `0.0194207398`.
- Fixed decision threshold: 0.5.
- Labels: 14 `non-responsive`, 0 `sensitive`.
- Matched-control export uses the exact model-input policy
  `nan_to_zero_as_model_input` and records each row's original NaN count.
- Protein importance is per sample only: Top-200 absolute IG, Top-100 positive
  IG, Top-100 negative IG, full 11,092-protein parquet, and diagnostics. No
  cross-sample aggregate ranking is produced.

## Primary Outputs

- `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_predictions.csv`
- `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_perturbed_proteome.csv`
- `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_matched_control_proteome.csv`
- `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_summary.json`
- `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_protein_attribution/top200_absolute_ig_per_sample.csv`
- `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_protein_attribution/top100_positive_ig_per_sample.csv`
- `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_protein_attribution/top100_negative_ig_per_sample.csv`
- `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_protein_attribution/protein_attributions.parquet`
- Raw inference artifacts and manifests are under
  `outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2/`.
