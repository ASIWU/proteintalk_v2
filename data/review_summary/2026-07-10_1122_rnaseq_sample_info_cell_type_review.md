# RNA-seq sample metadata cell-type review

- Review time: 2026-07-10 11:22:56 +0800
- Reviewed file: `data/rawdata/rna_seq/260709_samp_inf.csv`
- Scope: Verify whether the newly added sample metadata contains cell/tumor type information.

## Findings

- The CSV contains 13 sample records with `samp_ID` values 1 through 13.
- Every record has non-null type annotations in two duplicate-named source columns:
  - the first `cell_type` column contains Chinese cancer types;
  - the second `cell_type` column contains English organ/site labels.
- The three represented classes are:
  - `Lung` / `肺癌`: 10 samples;
  - `Pancreas` / `胰腺癌`: 2 samples;
  - `Colon` / `结直肠癌`: 1 sample.
- The header is `samp_ID,pat_ID,cell_type,cell_type,`, so pandas normalizes the duplicate columns to `cell_type` and `cell_type.1`, and the final numeric field to `Unnamed: 4`.
- All five parsed fields are complete; `samp_ID` and `pat_ID` are unique.

## Integration note

The numeric `samp_ID` values align with the 1-13 suffix range used by the organoid proteomics sample names, but the CSV does not explicitly include those full proteomics column names. Joining by numeric suffix is therefore plausible but should be documented as a mapping rule before it is used in an inference task builder.

No raw data files were modified.
