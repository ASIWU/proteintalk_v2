# 2026-05-14 15:47 HKT Extra Raw Filter Status Columns Review

## Scope

- Reviewed how filtered raw extra rows are marked after restricting exported controls to single-drug / extra-baseline sources.

## Finding

- `processed.csv` is still used to decide whether each raw row survived stage-2 filtering.
- The previous boolean `kept_after_filter` was correct but not explicit enough for external review.

## Update

- Added `stage2_filter_status`:
  - `kept_after_stage2_filter`
  - `filtered_out_before_stage2_processed`
- Added `control_export_status`:
  - `matched_allowed_control`
  - `filtered_out_blank_control`
  - `kept_but_no_allowed_control_match`

## Verification

- Re-exported all 9 PTV3 extra raw annotated CSVs.
- Current counts:
  - `filtered_out_before_stage2_processed / filtered_out_blank_control`: `265317`
  - `kept_after_stage2_filter / matched_allowed_control`: `133409`
  - `kept_but_no_allowed_control_match`: `0`
- Exported controls still only come from `ptv3_main_singledrug` or `ptv3_extra_baseline`.
