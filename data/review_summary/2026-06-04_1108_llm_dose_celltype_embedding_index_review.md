# 2026-06-04 11:08 HKT LLM Dose Cell-Type Embedding Index Review

Reviewed the `20260602_llm_dose_graphallowed_selected_v1` experiment artifacts to determine whether frozen LLM embeddings were indexed by `cell_type` or `Cell`.

Evidence checked:

- `docs/2026-06-02_llm_dose_graphallowed_selected_results.md`
- `checkpoints/20260602_llm_dose_graphallowed_selected_v1*/run_manifest.json`
- `train.py`
- `scripts/ptv3_experiment_common.sh`
- `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096.json`

Findings:

- All 27 primary selected-suite run manifests contain `cell_type_llm_index_column: cell_type_index`.
- No primary selected-suite manifest contains `cell_type_llm_index_column: Cell_index`.
- The training path looks up frozen LLM features from `artifacts.df[cell_type_llm_index_column]`; for this suite that means `cell_type_index`.
- The embedding artifact is explicitly `kind: cell_type_llm_embedding`, with 14 cell-type rows and a zero row reserved for `no`.
- Conclusion: this experiment used `cell_type`/`cell_type_index` LLM embeddings, not `Cell`/`Cell_index` embeddings.
- Follow-up at 2026-06-04 11:10 HKT confirmed the description-generation prompt in `utils/11_build_cell_type_llm_embeddings.py`:
  - system: `You write accurate, compact biomedical cell-type descriptions.`
  - user template: `Write one concise biomedical description for a cancer cell type used in drug response proteomics experiments. Focus on tissue lineage, tumor context, common biological traits, and why the cell type may matter for perturbation response. Use 2-4 sentences, no bullets. Cell type: {name}. Observed labels: {variants}.`
  - the reserved `no` cell type has `prompt: null` and uses the zero-vector row.

No code or result-document changes were made.
