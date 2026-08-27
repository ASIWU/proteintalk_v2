# 2026-07-09 16:09 HKT exp31 RNA-seq PDX Fine-tune Tooling Review

- Reviewed the PTV3 training/inference entry points, exp09 unified runner settings, exp09 checkpoint manifest, current graph artifact shape requirements, and the RNA-seq PDX raw tables under `data/rawdata/rna_seq`.
- Added a copy-on-write exp31 training-ready builder that writes only to `data/training_ready_exp31_rnaseq/ptv3`; the maintained `data/training_ready/ptv3` source artifacts are read-only inputs.
- The builder creates four separate clinical-label tasks, uses BRCA sample-level train/valid splits, holds out all non-BRCA samples for test, stores the active clinical label in `synergy` for unified-head compatibility, and records that this is not a biological synergy label.
- New/unmatched drugs are retained by appending only Morgan/DDI drug-side artifacts and zero PDI rows in the exp31 copy; PPI and protein embeddings are copied unchanged.
- Added `scripts/exp_31_rnaseq_pdx_ft_benchmark.sh` for GPU-only fine-tuning/inference launch via `gpu2`; it initializes from the exp09 unified single+double checkpoint, trains all four labels separately with `--no-mse-loss`, and runs both fine-tuned and exp09 zero-shot non-BRCA inference.
- Added `scripts/report_exp31_rnaseq_pdx_ft_benchmark.py` for CPU preflight and final report generation, including overall, cancer-type, and treatment-type metrics and best-label selection by BRCA valid AUPRC.
- Attempted external API-based sample Cell LLM generation was blocked by sandbox review because it would export exp31 prompts to an unverified raw-IP OpenAI-compatible endpoint. Added and used an explicit offline tissue fallback that maps exp31 cancer types to the existing validated tissue-level LLM vectors: `BRCA->BREAST`, `CRC->COLON`, `PDAC->PANCREAS`, `NSCLC->LUNG`, `CM->SKIN`.
- CPU checks passed:
  - `python -m py_compile utils/31_build_exp31_rnaseq_training_ready.py utils/31_build_exp31_cell_llm_embeddings.py scripts/report_exp31_rnaseq_pdx_ft_benchmark.py train.py infer.py`
  - `bash -n scripts/exp_31_rnaseq_pdx_ft_benchmark.sh scripts/ptv3_experiment_common.sh`
  - `python scripts/report_exp31_rnaseq_pdx_ft_benchmark.py --preflight-only --output-json outputs/2026-07/2026-07-09/20260709_exp31_preflight.json`
- GPU run `20260709_exp31_rnaseq` was launched only through tmux `gpu2` and completed all expected stages with status `0` in `logs/20260709_exp31_rnaseq_runtime_summary.tsv`.
- Final reporter outputs:
  - `docs/2026-07-09_exp31_rnaseq_pdx_ft_benchmark_results.md`
  - `outputs/2026-07/2026-07-09/20260709_exp31_rnaseq_pdx_ft_benchmark_summary.csv`
  - `outputs/2026-07/2026-07-09/20260709_exp31_rnaseq_pdx_ft_benchmark_summary.json`
- Reporter validation errors: `0`.
- Best setting by BRCA valid AUPRC was `sensitive_early` with valid AUPRC `0.853725`; its non-BRCA test AUPRC/AUROC were `0.180543`/`0.617007` after fine-tuning versus `0.147293`/`0.629695` for exp09 zero-shot.
