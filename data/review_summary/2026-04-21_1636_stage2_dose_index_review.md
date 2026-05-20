# 2026-04-21 16:36 HKT Review Summary

## Scope

Reviewed the stage-2 training-ready pipeline around `pert_dose` indexing, auxiliary builder documentation, and feature-loader usage.

Files checked:

- [utils/02_build_training_ready_data.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/02_build_training_ready_data.py:1)
- [utils/03_validate_training_ready_outputs.py](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/utils/03_validate_training_ready_outputs.py:1)
- [docs/Data_Process_2.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/Data_Process_2.md:1)
- [docs/data_process_summary_02.md](/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/docs/data_process_summary_02.md:1)

## Findings

1. The previous `pert_dose` implementation used sorted unique-value enumeration, so it did not satisfy the requested ceil-based binning rule.
2. The previous validator assumed dense integer ranges `0..len(mapping)-1`, which is incompatible with ceil-based dose bins and string-coded mapping values.
3. The summary doc previously described the auxiliary builders at a high level but did not provide runnable commands or a concrete dataloader access pattern.

## Actions Taken

1. Switched `pert_dose1` / `pert_dose2` onto a shared `ceil(dose)` string mapping with `"no" = string(max_numeric_index + 1)`.
2. Updated validation to check membership against the actual mapping values and to enforce the new `"no"` rule.
3. Added a builder README section and a minimal `Dataset.__getitem__` example to the stage-2 summary doc.

## Residual Risk

Existing files already written under `data/training_ready/` must be regenerated before downstream training consumes the new `pert_dose` rule.
