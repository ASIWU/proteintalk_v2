# Update-0821 Manual Target Input and Exp34 Re-inference Feasibility Review

- Review time: 2026-08-21 15:53 HKT (UTC+08:00)
- Requested path: `data/raw_data/update_0821.csv` (not present)
- Reviewed input: `data/rawdata/update_0821/260820ptv_drug_cell_predict_target.csv`
- Previous input: `data/rawdata/update_0819/260513ptv_drug_cell_predict.csv`
- Previous run: `20260819_exp34_update0819_ood_epoch2`
- Scope: data/schema/model-contract review only; no inference was launched.

## Verdict

The reviewed CSV is structurally valid and compatible with the previous Exp34
query cohort. Its five legacy columns are row-for-row identical to the August
19 input, and its manually supplied target sets are valid members of the
current PTV3 protein axis.

The file should not be row-concatenated with the August 19 file: that would
duplicate the same 14 cell-drug queries. It should instead replace the old
input or be joined to it by the unique `(cell_in_ptvdrug, drug_name)` key.

Re-inference is technically possible, but it would not create a new target
condition under the current data: the manual target sets are exactly the same
sets already used by the previous `target_mechanism` branch. Therefore, with
the same checkpoint and settings, the expected predictions are the existing
August 19 mechanism-target predictions.

## Data Checks

- UTF-8 BOM is present and is handled by the builder's `utf-8-sig` reader.
- Shape: 14 rows and 6 columns.
- Cohort: 7 distinct cell aliases x 2 drugs, forming a complete Cartesian
  product.
- Missing values: 0.
- Duplicate full rows: 0.
- Duplicate `(cell_in_ptvdrug, drug_name)` keys: 0.
- RDKit parsed every SMILES; each drug has exactly one canonical isomeric
  structure across all seven cell rows.
- `read_and_validate_input()` in
  `utils/34_build_update0819_ood_training_ready.py` accepted the file.
- The five legacy columns and row ordering are identical to the August 19
  input.
- Input SHA-256:
  `d14a6375d8160039f1adf0a9cae1f32e31b9805447f89b5328b192b580a67909`.

## Target Checks

After trimming the semicolon-separated values:

- daraxonrasib: `P01116; P01111; P01112`
  (KRAS, NRAS, HRAS; set-equivalent to the builder's NRAS, HRAS, KRAS order)
- zoldonrasib: `P01116` (KRAS)

All three accessions are valid six-character UniProt accessions and occur in
`data/training_ready/ptv3/global_meta.json`:

- `P01111 -> protein index 1374`
- `P01112 -> protein index 1375`
- `P01116 -> protein index 1376`

The previous Exp34 build summary confirms that its mechanism branch already
used `[1374,1375,1376]` for seven daraxonrasib rows and `[1376]` for seven
zoldonrasib rows. Thus target order in the raw CSV is immaterial after the
required set normalization and protein-index conversion.

## Current Integration Gap

The current pipeline does not actually consume the new `target` column:

1. `read_and_validate_input()` requires only `Cell_name`,
   `cell_in_ptvdrug`, `drug_name`, `drug_name2`, and `smiles`; extra columns
   are accepted but not interpreted.
2. `MECHANISM_TARGETS` is hard-coded in
   `utils/34_build_update0819_ood_training_ready.py`.
3. `scripts/exp_34_update0819_ood_epoch2_infer.sh` invokes the builder without
   an `--input-csv` argument, so its build path still defaults to the August 19
   file.

This happens to produce the intended targets today because the CSV and the
hard-coded sets match. It is nevertheless unsafe as a reusable manual-target
workflow: a future edit to the CSV could be silently ignored.

## Recommended Re-inference Contract

Before treating August 21 as a distinct experiment:

1. Parse the `target` column explicitly, trim delimiters/whitespace, remove
   duplicates, and validate every accession against `protein_index`.
2. Fail if one drug has conflicting target sets across cells, unless
   cell-specific targeting is intentionally supported and documented.
3. Convert UniProt accessions to sorted protein-index lists in the generated
   feature table; never pass raw strings to the model.
4. Pass the input path through the runner and record its path and SHA-256 in
   the build/run manifests.
5. Use a new experiment prefix and output directory. Do not overwrite or
   append to the completed August 19 output.
6. Optionally retain `target_none` as a sensitivity control, with the manual
   CSV-driven target condition as the formal branch.

Because the current targets are unchanged from the prior mechanism branch,
rerunning is useful only to prove the new CSV-driven ingestion/provenance
contract; it is not expected to change the biological prediction values.

## Verification Status

- Shell syntax check for the existing Exp34 runner: passed.
- Targeted pytest invocation could not run because the active `flow_v2`
  environment does not contain the `pytest` module.
- No source code, source CSV, prior output, or runtime artifact was modified by
  this review.
