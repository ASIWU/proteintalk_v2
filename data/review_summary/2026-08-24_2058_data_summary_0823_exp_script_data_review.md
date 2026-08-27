# Data Summary 0823 Exp-script Data Review

- Review time: 2026-08-24 21:04 HKT (UTC+08:00)
- Output: `data/data_summary_0823.txt`
- Scope: every experiment script whose filename starts with `exp` under
  `scripts/` or `scripts/ptv1/`; report/attribute/run helpers are outside the
  experiment-data scope.

## Review Method

- Read `data/data_summary_0413.txt` as the requested style baseline.
- Inspected Exp01-09, Exp21-28, Exp31-35 and PTV1 Exp11-13 task names, split
  strategies, label heads, training-ready roots, checkpoints, and inference
  targets.
- Cross-checked current raw provenance against
  `data/standardized/file_audit.json`, the PTV1 audit, training-ready feature
  tables, PRISM2 build summary, Exp31/32 build summaries, Exp33/34/35 builders,
  and Exp35 execution review.
- Checked every explicit `data/...` path in the new summary. The only regex
  false positive was prose text containing `data/label`; all actual documented
  paths exist.

## Current-version Decisions

- PTV3 main single metadata is only the update-0623 file. The 2025 expression
  CSV remains listed because update-0623 provides metadata only and the current
  pipeline still uses that expression matrix.
- PTV3 main double lists only the 20260422 metadata/expression pair; older
  20260414/17 variants are omitted.
- Extra single and extra double list only update-0527 query files; older
  February/April versions are omitted. Current extra baseline and target-map
  files remain listed because they are still active inputs.
- PTV1 extra single lists only `kept_samples_drugfilter_edit.csv`; old
  `test12091214...` and `ptds4...` sources are omitted from the current task.
- Exp34 lists only update-0819. Exp35 lists only update-0821 and explicitly does
  not present update-0819 as its input.
- Exp21-28 are documented as PRISM2-filtered tasks derived from the same current
  PTV3 raw inputs, rather than as a duplicate raw dataset.

## User-requested Rawdata-only Revision

- The final summary now lists files only when they exist under `data/rawdata/`.
- Removed the Exp04_v2 generated random-control NPY path; Exp04_v2 remains
  described as using the same raw single-drug files.
- Removed PTV3/PTV1 `global_meta`, embeddings, PPI/PDI/DDI, cell-LLM,
  training-ready, standardized, checkpoint, runtime, and output artifact paths.
- Removed the entire shared-derived-input file section.
- PTV1 extra now lists its current raw query file plus the raw PTV1 AIVC files
  used to source controls; processed drug lookup files are not listed.
- Exp32 records only its three organoid raw files; its derived drug scope is
  described without presenting a processed file.
- Exp34/35 retain their raw update CSVs but no longer identify processed
  matched-control artifacts as files.

## Coverage

The summary is organized by model task and covers:

- PTV3 main single, main double, extra single, extra double, and Exp09 unified;
- PRISM2 main single / PRISM2-aux double and their extra evaluations;
- PTV1 main and PTV1 extra single;
- Exp31 PDX RNA-seq fine-tuning, Exp32 organoid virtual screening, Exp33
  double-drug virtual screening, Exp34 update-0819 OOD inference, and Exp35
  update-0821 manual-target inference;
- no processed/derived artifact inventory; the final document is rawdata-only.

For each task the document records the linked Exp scripts, task purpose,
current raw inputs, label/split interpretation, and the relevant current data
scale. `exp21_26_rebuild_prism2_graphs.sh` is identified as a helper rather than
an independent task.
