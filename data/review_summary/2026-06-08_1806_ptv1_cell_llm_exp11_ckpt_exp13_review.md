# 2026-06-08 18:06 HKT PTV1 Cell LLM and exp11 Checkpoint-on-exp13 Review

## Scope

- Answered whether the PTV1 dataset has a correct `cell-llm-embedding` artifact.
- Evaluated formal exp11 random-split checkpoints from prefix `20260608_ptv1_cell_llm_tune_v1` on the exp13 `ptv1_extra_singledrug` test-only dataset.
- Included both checkpoint files present in each of the 32 exp11 random-split run directories: the selected best epoch checkpoint and `last.ckpt`.

## Cell LLM Embedding Audit

- Ran `utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --validate-only` in `flow_v2`.
- Validation passed for `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz`.
- Artifact shape is `20 x 4096`.
- Metadata is `dataset_group=ptv1`, `kind=cell_llm_embedding`, `input_field=Cell`, `index_column=Cell_index`.
- PTV1 task coverage:
  - `ptv1_aivc`: 15002 rows, Cell indices 1 through 19.
  - `ptv1_extra_singledrug`: 222 rows, Cell indices 2, 7, 11, 18.
- Validator checks include row-0 zero vector, finite nonzero real-cell rows, Cell index alignment with `global_meta`, feature-table Cell columns, and absence of legacy `cell_type_llm` in the sidecar.

## exp11 Checkpoint Evaluation

- Reused existing direct exp13 outputs for the 32 selected best epoch exp11 checkpoints.
- Ran new direct exp13 inference for the 32 `last.ckpt` files into output directories ending with `_extra_direct_from_exp11_lastckpt`.
- Final audit covered 64 checkpoint evaluations with zero errors:
  - target dataset `ptv1_extra_singledrug`;
  - split strategy `test_only`;
  - `n_predictions=218` for every run;
  - inference manifest checkpoint path matches the checkpoint being evaluated;
  - `cell_llm_mode=frozen`.

## Best Result

- Best overall and best graph-on result:
  - candidate `focal_mse050`;
  - checkpoint `checkpoints/20260608_ptv1_cell_llm_tune_v1_focal_mse050_ptv1_random_split/last.ckpt`;
  - AUPRC `0.6301167927010799`;
  - AUROC `0.641635781671159`;
  - nAUPRC `1.2264773286503161`;
  - rows `218`.
- Full ranking written to `outputs/20260608_ptv1_exp11_all_ckpts_on_exp13_direct_metrics.tsv`.
