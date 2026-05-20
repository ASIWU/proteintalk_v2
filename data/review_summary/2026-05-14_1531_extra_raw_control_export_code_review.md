# 2026-05-14 15:31 HKT Extra Raw Control Export Code Review

## Scope

- Reviewed `utils/10_export_extra_raw_with_controls.py`.
- Random-checked exported control mappings in `data/standardized/ptv3/extra_raw_with_controls/extra_sample_id_control_mapping.csv`.

## Code Review

- No blocking correctness findings.
- The script joins raw rows through stage-1 `source_row_index`, copies `control` only from stage-2 processed self rows, and records filtered rows with blank `control`, matching the requested export semantics.
- Minor cleanup opportunity: `clean_scalar` is currently unused and can be removed later without behavior change.

## Random Control Checks

- Random seed: `20260514`.
- Checked one non-empty control row from each of the 9 PTV3 extra single/double tasks.
- For each sampled non-empty row:
  - exported `sample_id` matched stage-1 `info.csv`;
  - exported `control` matched the task's stage-2 `processed.csv` self row;
  - the appended matched-control row existed in the same task `processed.csv`;
  - the source control row existed in its source task and satisfied `control == sample_id`;
  - recomputed control-pool ranking selected the same control sample id.
- Checked 6 random filtered rows with blank exported `control`; all were absent from stage-2 processed self rows as expected.
- Result: `FAILURES 0`.
