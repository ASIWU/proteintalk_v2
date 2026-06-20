# 2026-06-08 21:17 HKT Cell + Cell-type LLM Selected Suite Review

## Scope

Reviewed and executed the PTV3 clip10 selected exp01-exp08 suite after adding frozen `cell_type` LLM embeddings alongside the existing frozen Cell LLM embeddings.

## Code Paths Reviewed

- `utils/11_build_cell_type_llm_embeddings.py`
- `dataset/training_ready_fast_dataset.py`
- `model/fast_delta_model.py`
- `train.py`
- `infer.py`
- `scripts/ptv3_experiment_common.sh`
- `scripts/run_cell_llm_clip10_param_search.sh`
- `scripts/run_cell_celltype_llm_clip10_selected_suite.sh`
- `scripts/report_cell_celltype_llm_selected_comparison.py`

## Findings

- The new cell-type LLM feature path is wired through dataset samples, train/infer CLI args, model construction, model forward pass, and run manifests.
- The selected runner explicitly validates both Cell and cell-type embedding artifacts before launch.
- The formal `_v2` cell-type artifact has shape `(14, 4096)`, zero row 0, finite nonzero rows, and metadata `kind=cell_type_llm_embedding`, `input_field=cell_type`, `index_column=cell_type_index`.
- API regeneration with `proxy_on` and `.env` credentials was attempted, but chat completion timed out; the formal `_v2` artifact reuses the existing validated Qwen3 cell-type embedding with enriched metadata.
- Completed selected suite prefix: `20260608_cell_celltype_llm_clip10_selected_v1`.
- Comparison audit completed with `audit_errors=0` across required selected-suite manifests.

## Result Summary

Mean original AUPRC deltas vs `20260604_cell_llm_clip10_tuned_selected_v1`:

- exp01: `-0.007663`
- exp02: `-0.008272`
- exp03: `-0.009600`
- exp04: `-0.005668`
- exp05: `+0.010999`
- exp06: `-0.032572`
- exp07: `-0.005514`
- exp08: `+0.009759`

Main artifacts:

- `docs/2026-06-08_cell_celltype_llm_clip10_selected_report.md`
- `outputs/20260608_cell_celltype_llm_clip10_selected_v1_cell_drug_dose_time_eval.csv`
- `outputs/20260608_cell_celltype_llm_clip10_selected_v1_cell_celltype_llm_comparison.tsv`
- `logs/20260608_cell_celltype_llm_clip10_selected_v1_cell_celltype_llm_comparison.md`

## Validation

- Shell syntax check passed for the updated selected runner and common runner scripts before launch.
- Python compile check passed for train/infer/model/dataset/report/builder files before launch.
- Full exp01-exp08 selected suite completed without run failure.
