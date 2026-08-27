# Outputs Month/Day Organization Review

- Review time: 2026-07-21 11:32 HKT
- Scope: complete `outputs/` inventory, dated migration, reference rewriting, future runner defaults, and post-migration integrity.

## Result

- Final layout is `outputs/YYYY-MM/YYYY-MM-DD/<original-top-level-name>`.
- Root entries are limited to `2026-05/`, `2026-06/`, `2026-07/`, `_organization/`, and `README.md`.
- There are 31 date directories covering 2026-05-10 through 2026-07-20.
- All 1,978 pre-existing top-level artifacts/directories were moved. No output artifact was deleted, copied, merged, or renamed internally.

## Integrity Checks

- Pre-migration inventory: 17,436 files, 6,719 directories below `outputs/`, 0 symlinks, and 4,966,028,641 file bytes.
- Path-map validation: 1,978 old paths absent, 1,978 new paths present, and 1,978/1,978 top-level inode values unchanged.
- Post-migration month buckets contain exactly the original 17,436 files and no symlinks.
- Representative epoch-2 Exp31/Exp32 CSV, Parquet, PNG, PDF, JSON, and inference directories exist under `outputs/2026-07/2026-07-20/`.
- Migration used same-filesystem `Path.rename`; rollback logic was available for move failures. No deletion command was used.

## Date Assignment

| source | entries |
|---|---:|
| leading `YYYYMMDD` | 1,956 |
| leading `MMDD`, year fixed to 2026 | 5 |
| embedded `YYYYMMDD` | 12 |
| HKT modification-date fallback | 5 |

The five modification-date fallbacks are named debug artifacts and were routed to their actual 2026-05-25 or 2026-05-26 creation dates. All short/embedded-date assignments were inspected in the dry-run mapping.

## References And Future Writes

- Exact path rewrite changed 269 occurrences across 65 text files.
- Generic dated/wildcard rewrite changed 9 occurrences across 6 text files.
- Final dry-run reports zero remaining replacements. A separate search found zero explicit legacy dated/debug/smoke/audit/diagnostic output references outside `outputs/`.
- `scripts/ptv3_experiment_common.sh` now derives `OUTPUT_DIR=outputs/YYYY-MM/YYYY-MM-DD` from `EXP_PREFIX`. A probe using `EXP_PREFIX=20260721_output_layout_probe` resolved to `outputs/2026-07/2026-07-21`.
- `OUTPUT_DATE` and `OUTPUT_DIR` continue to override the automatic routing.
- Exp32's standalone runner applies the same routing before sourcing the common helper. Exp31/Exp32 reporters derive their default read root from the prefix date.
- Historical run-manifest path strings inside moved output artifacts were intentionally preserved as execution provenance; they can be resolved through the complete path-map audit.

## Audit Artifacts

- `outputs/README.md`
- `outputs/_organization/2026-07-21_output_path_map.json`
- `outputs/_organization/2026-07-21_reference_rewrite.json`
- `outputs/_organization/2026-07-21_generic_dated_reference_rewrite.json`

## Closing Verification

- Complete mapping validation passed: all 1,978 old paths are absent, all 1,978 new paths exist, and all 1,978 recorded top-level inode values match.
- Post-migration validation passed with 17,436 files, 4,966,028,641 file bytes, 31 date directories, and zero symlinks.
- Representative epoch-2 reads passed: Exp32 CSV and Parquet are both 83,642 rows by 18 columns; Exp31 CSV is 56 rows by 16 columns; PNG, PDF, JSON, and inference directories are readable.
- Python compilation, shell syntax, both reporter `--help` checks, automatic/override output-bucket probes, and the final rewrite dry-run all passed.
- The final legacy-reference search returned no matches outside historical output artifacts, and repository-wide `git diff --check` passed.
