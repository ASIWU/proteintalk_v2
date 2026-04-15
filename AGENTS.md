# Repository Guidelines

## Environment
Use `conda activate flow_v2` to activate python env.

## Project Structure & Module Organization
This repository is primarily a data workspace, not an application package. Store source datasets under `data/rawdata/`, grouped by study or ingestion path such as `singledrug/`, `doubledrug/`, `ptv1/`, `extra_*`, and `xlsx/`. Keep shared metadata templates in `data/`; `data/info_template.json` is the reference shape for sample-level annotations. Put reusable utilities in `utils/`; the current script, `utils/0415_converxlsx2csv.sh`, recursively converts Excel files to CSV in place.

## Build, Test, and Development Commands
There is no project build step today. Use:

- `bash utils/0415_converxlsx2csv.sh` to convert all `.xlsx` files under `data/rawdata/`.
- `bash utils/0415_converxlsx2csv.sh data/rawdata/xlsx` to target one subtree.
- `bash utils/0415_converxlsx2csv.sh --help` to view script usage.

The converter depends on `xlsx2csv` being installed and writes `.csv` files next to the source spreadsheets.

## Coding Style & Naming Conventions
Follow the existing Bash style for utilities: `#!/usr/bin/env bash`, `set -euo pipefail`, quoted variables, and lowercase `snake_case` names. Keep JSON keys double-quoted and preserve the current 4-space indentation used in `data/info_template.json`. For dataset files, keep the established date-prefixed naming pattern, for example `20260413ptv3_...csv`, so ingestion order and provenance stay obvious.

## Testing Guidelines
This checkout does not contain an automated test suite yet. Validate data changes with targeted spot checks:

- confirm each generated `.csv` appears beside its `.xlsx` source;
- open a few rows to verify delimiters, headers, and encoding;
- ensure new files land in the correct study folder and match the metadata template where applicable.

If you add scripts, include a dry-run or `--help` path and document sample input/output in the script header.

## Commit & Pull Request Guidelines
Git history is not available in this mounted checkout, so no repository-specific convention could be verified. Use short, imperative commit subjects such as `data: add PRISM validation CSV` or `utils: tighten xlsx conversion`. In pull requests, summarize the dataset or script change, list affected paths, note any regeneration steps, and include before/after samples when a transform changes file structure or column layout.
