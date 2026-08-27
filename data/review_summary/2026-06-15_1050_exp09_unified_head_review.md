# 2026-06-15 10:50 HKT exp09 Unified-Head Review

- Reviewed exp07/exp08 launcher patterns, fast-delta task-head routing, inference metric export, and detailed result reporting utilities before implementing exp09.
- Implemented `--task-head unified` as a fast-delta-only mode that uses one response head for both single-drug sensitivity and double-drug synergy labels, with row-wise active-label selection.
- Added exp09 all-data training, valid reference-epoch evaluation, all-epoch oracle evaluation, and valid/oracle metric reporting scripts.
- Updated required experiment launchers so exp09 follows exp08.
- Verified syntax and import health with Python compile checks, shell `bash -n`, and a minimal unified-head Lightning smoke test.

