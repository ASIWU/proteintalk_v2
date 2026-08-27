# Repository Guidelines

## Environment
Use `conda activate flow_v2` to activate python env (/mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh).

## Docs
- the gudline workflow is ./data/Data_Process_[1-4].md and ./data/Training_guideline.md
- Every time a major update is made to the code, the modify history should be record in ./docs/2026-04-15_data_standardization_session_summary.md
- Every time review the codebase, the review summary should be recode in ./data/review_summary. (add date and time(including hours and minutes))

## Codebase reference
- the old version of this codebase is in /mnt/shared-storage-user/beam/wuhao/H100/proteintalk/ProteinTalkv2

## Smoke artifacts
- After a smoke test finishes (whether it passes or fails), record the needed validation evidence and delete that smoke run's temporary output directories, sidecar reports, attribution artifacts, and smoke-specific logs. Keep only formal-run artifacts unless the user explicitly asks to retain smoke files.
- Resolve and review the exact smoke paths before deletion, and never delete formal outputs as part of smoke cleanup.
