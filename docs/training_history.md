# Training History

## 2026-06-15 11:00 HKT: exp09 Unified Single/Double Extra Evaluation

This entry records the `20260615_1101_exp09_selectedref_v1` unified-head experiment. The original interactive shell history was not persisted in the repo; the command below is the manifest-equivalent command reconstructed from `scripts/exp_09_unified_all_train_valid_oracle.sh`, the runtime summary, and `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/run_manifest.json`.

### Command

```bash
tmux send-keys -t gpu2 'cd /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2 && source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh && conda activate flow_v2 && EXP_PREFIX=20260615_1101_exp09_selectedref_v1 GPU_IDS=0 DEVICES=1 LOGGER_BACKEND=none LOG_TO_WANDB=0 WANDB_MODE=disabled MAX_EPOCHS=50 BATCH_SIZE=256 LEARNING_RATE=5e-5 DROPOUT=0.15 MSE_WEIGHT=0.5 MSE_TARGET_MODE=all MSE_INACTIVE_LABEL_WEIGHT=0.2 USE_DDI=1 PAIR_FUSION_MODE=dual PAIR_TYPE_FEATURES=1 GRAPH_PAIR_ADD_SCALE=0.5 CELL_LLM_MODE=frozen CELL_TYPE_LLM_MODE=frozen CELL_LLM_EMBEDDING_PATH=data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz CELL_TYPE_LLM_EMBEDDING_PATH=data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz REFERENCE_EXP01_5FOLD_CKPT_PATH=checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp01_single_pert_stratified_5fold REFERENCE_EXP06_5FOLD_CKPT_PATH=checkpoints/20260610_ptv01_08_posweight_combo_selected_v1_exp06_double_pert_pair_5fold SAVE_TOP_K=-1 SAVE_EVERY_N_EPOCHS=1 SAVE_LAST_CKPT=1 CHECKPOINT_FILENAME="{epoch}" MONITOR=none RUN_VALID=1 RUN_ORACLE=1 RUN_REPORT=1 bash scripts/exp_09_unified_all_train_valid_oracle.sh' C-m
```

### Date And Runtime

| field | value |
|---|---|
| Experiment prefix | `20260615_1101_exp09_selectedref_v1` |
| Train start | `2026-06-15 11:00:54 HKT` (`2026-06-15T03:00:54Z`) |
| Train end | `2026-06-15 11:03:53 HKT` (`2026-06-15T03:03:53Z`) |
| Full valid/oracle inference end | `2026-06-15 12:24:18 HKT` (`2026-06-15T04:24:18Z`) |
| Runtime summary | `logs/20260615_1101_exp09_selectedref_v1_runtime_summary.tsv` |
| Runtime status | 460 task rows, 0 nonzero statuses |

### Hyperparameters

| group | setting |
|---|---|
| Script | `scripts/exp_09_unified_all_train_valid_oracle.sh` |
| Train task | `ptv3_main_doubledrug` |
| Split strategy | `all_train_subset_test` |
| Model type | `fast_delta` |
| Task head | `unified` |
| Label policy | `unified_synergy_first_else_response` |
| Response label key | `PRISM1st_label_total` |
| Synergy label key | `synergy` |
| Optimizer | `adamw` |
| Learning rate | `5e-5` |
| Weight decay | `1e-4` |
| Scheduler | `cosine` |
| Batch size | `256` |
| Max epochs | `50` |
| Precision | `bf16-mixed` |
| Devices | `1` GPU |
| Seed | `42` |
| Hidden dim | `512` |
| Expression latent dim | `768` |
| Covariate embedding dim | `96` |
| Dropout | `0.15` |
| Control layers | `2` |
| Fusion layers | `3` |
| Target layers | `2` |
| MSE loss | enabled |
| MSE weight | `0.5` |
| MSE target mode | `all` |
| MSE inactive-label weight | `0.2` |
| BCE weight | `1.0` |
| Positive weights | disabled (`none`) |
| Label smoothing | `0.0` |
| Focal loss | disabled |
| Ranking loss | disabled |
| Batch covariates | `machineID_new`, `Cell_plate`, `Cell`, `cell_type`, `batch`, `pert_time`, `pert_dose1`, `pert_dose2` |
| Dose covariates | enabled: `pert_dose1`, `pert_dose2` |
| Graph feature mode | `real` |
| Graph feature dim | `128` |
| Graph structural random projection | enabled |
| Graph drug concat | enabled |
| Graph logit scale | `2.0` |
| Graph pair add scale | `0.5` |
| DDI | enabled |
| Pair fusion mode | `dual` |
| Pair type features | enabled |
| Target proteins | enabled, max length `32` |
| Protein concat mode | `pcep` |
| Protein concat dim/topk | `64` / `512` |
| Protein concat score mode | `multiply` |
| Cell LLM | `frozen`, `data/training_ready/ptv3/derived/cell_llm_embedding_qwen3_4096.npz` |
| Cell-type LLM | `frozen`, `data/training_ready/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz` |
| Save policy | `SAVE_TOP_K=-1`, `SAVE_EVERY_N_EPOCHS=1`, `SAVE_LAST_CKPT=1`, `CHECKPOINT_FILENAME={epoch}` |
| Monitor | `none` |
| Test policy | `--skip-test` during all-data training |

### Reference Epochs And Outputs

| target | reference source | selected epoch |
|---|---|---:|
| exp07 extra single valid | selected exp01 five folds | 5 |
| exp08 extra double valid | selected exp06 five folds | 2 |

Oracle scanned every saved checkpoint from `epoch=0.ckpt` through `epoch=49.ckpt`.

| output | path |
|---|---|
| Checkpoints | `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra` |
| Valid outputs | `outputs/2026-06/2026-06-15/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid` |
| Oracle outputs | `outputs/2026-06/2026-06-15/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_oracle_epoch*` |
| Result document | `docs/2026-06-15_exp09_unified_head_valid_oracle_results.md` |
| Result CSV | `outputs/2026-06/2026-06-15/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.csv` |
| Result JSON | `outputs/2026-06/2026-06-15/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra_valid_oracle_eval.json` |

Oracle best mean-extra original results:

| target | best epoch | AUROC | AUPRC | baseline | n-AUPRC | count |
|---|---:|---:|---:|---:|---:|---:|
| exp07 extra single | 2 | 0.780482 | 0.596656 | 0.209041 | 2.848712 | 92671 |
| exp08 extra double | 8 | 0.655564 | 0.101437 | 0.047592 | 2.423031 | 88970 |

## 2026-07-09 16:10 HKT: exp31 RNA-seq PDX BRCA Fine-tune Benchmark

This entry records the `20260709_exp31_rnaseq` benchmark. It fine-tunes the selected exp09 unified single+double checkpoint on BRCA PDX baseline RNA-seq data and evaluates zero-shot transfer on non-BRCA PDX tumor types.

### CPU Preparation Commands

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
python utils/31_build_exp31_rnaseq_training_ready.py --force
python utils/31_build_exp31_cell_llm_embeddings.py --offline-tissue-fallback --force
python scripts/report_exp31_rnaseq_pdx_ft_benchmark.py --preflight-only --output-json outputs/2026-07/2026-07-09/20260709_exp31_preflight.json
```

The OpenAI-compatible API path was not used for exp31 sample Cell embeddings because the sandbox reviewer blocked exporting exp31 prompts to the raw-IP `.env` endpoint. The generated Cell-channel artifact instead uses an explicit offline tissue fallback from the existing validated tissue-level LLM vectors:

`BRCA->BREAST`, `CRC->COLON`, `PDAC->PANCREAS`, `NSCLC->LUNG`, `CM->SKIN`.

### GPU Command

```bash
tmux send-keys -t gpu2 'cd /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2 && source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh && conda activate flow_v2 && EXP_PREFIX=20260709_exp31_rnaseq GPU_IDS=0 DEVICES=1 LOGGER_BACKEND=none LOG_TO_WANDB=0 WANDB_MODE=disabled RUN_PREFLIGHT=1 RUN_DATA_VALIDATION=0 PROGRESS_BAR=0 RUN_REPORT=0 bash scripts/exp_31_rnaseq_pdx_ft_benchmark.sh' C-m
```

### Date And Runtime

| field | value |
|---|---|
| Experiment prefix | `20260709_exp31_rnaseq` |
| GPU start | `2026-07-09 16:10:06 HKT` (`2026-07-09T08:10:06Z`) |
| GPU end | `2026-07-09 16:13:53 HKT` (`2026-07-09T08:13:53Z`) |
| Runtime summary | `logs/20260709_exp31_rnaseq_runtime_summary.tsv` |
| Runtime status | 12 task rows, 0 nonzero statuses |
| Final report | `docs/2026-07-09_exp31_rnaseq_pdx_ft_benchmark_results.md` |
| Final CSV | `outputs/2026-07/2026-07-09/20260709_exp31_rnaseq_pdx_ft_benchmark_summary.csv` |
| Final JSON | `outputs/2026-07/2026-07-09/20260709_exp31_rnaseq_pdx_ft_benchmark_summary.json` |

### Data And Split

| field | value |
|---|---|
| Raw RNA matrix | `data/rawdata/rna_seq/260617_2015_BFnm3954_MOESM10_ESM_sub.csv` |
| Raw sample info | `data/rawdata/rna_seq/260618pdx_pct_sample_info_with_smiles_check_comboAB.csv` |
| Training-ready root | `data/training_ready_exp31_rnaseq` |
| Dataset group | `ptv3` |
| Split strategy | `brca_ft_valid_nonbrca_test` |
| Train/valid policy | BRCA samples only |
| Test policy | non-BRCA samples only: `CRC`, `PDAC`, `NSCLC`, `CM` |
| Missing cancer type rows | 2 rows dropped |
| RNA protein axis | exp09 checkpoint axis, 11,092 proteins |
| RNA coverage | 10,159 / 11,092 proteins |
| New SMILES-only drugs | 5 appended in exp31 copy only |
| DDI policy | source DDI copied; new rows/cols from Morgan/Tanimoto |
| PDI policy | source PDI copied; new drug rows are all zero |
| Source data protection | original `data/training_ready` not modified |

### Hyperparameters

| group | setting |
|---|---|
| Script | `scripts/exp_31_rnaseq_pdx_ft_benchmark.sh` |
| Init checkpoint | `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/last.ckpt` |
| Tasks | `ptv3_exp31_rnaseq_sensitive_early`, `ptv3_exp31_rnaseq_sensitive_late`, `ptv3_exp31_rnaseq_disease_control_early`, `ptv3_exp31_rnaseq_disease_control_late` |
| Label columns | `sensitive_label_early_CRPR_vs_SDPD`, `sensitive_label_late_CRPR_vs_SDPD`, `disease_control_label_early_CRPRSD_vs_PD`, `disease_control_label_late_CRPRSD_vs_PD` |
| Label carrier | each exp31 clinical label is written to `synergy` for unified-head compatibility; not biological synergy |
| Model type | `fast_delta` |
| Task head | `unified` |
| MSE loss | disabled with `--no-mse-loss` |
| Optimizer | `adamw` |
| Learning rate | `1e-5` |
| Weight decay | `1e-4` |
| Scheduler | `cosine` |
| Batch size | `128` |
| Inference batch size | `256` |
| Max epochs | `30` |
| Precision | `bf16-mixed` |
| Devices | `1` GPU, `GPU_IDS=0` |
| Seed | `42` |
| Hidden dim | `512` |
| Expression latent dim | `768` |
| Covariate embedding dim | `96` |
| Dropout | `0.15` |
| Control layers | `2` |
| Fusion layers | `3` |
| Target layers | `2` |
| Best checkpoint metric | BRCA `valid_auprc` via `MONITOR=val/task_auprc`, `MONITOR_MODE=max` |
| Early stopping | patience `8`, min delta `0.0` |
| Save policy | `SAVE_TOP_K=1`, `SAVE_EVERY_N_EPOCHS=1`, `SAVE_LAST_CKPT=1` |
| Batch covariates | `machineID_new`, `Cell_plate`, `Cell`, `cell_type`, `batch`, `pert_time`, `pert_dose1`, `pert_dose2` |
| Dose covariates | enabled: `pert_dose1`, `pert_dose2` |
| Pair fusion mode | `dual` |
| Pair type features | enabled |
| DDI | enabled |
| Graph feature mode | `real` |
| Graph feature dim | `128` |
| Graph structural random projection | enabled |
| Graph drug concat | enabled |
| Graph pair add scale | `0.5` |
| Graph logit scale | `2.0` |
| Graph cache dir | `graph_cache/exp31_rnaseq` |
| Target proteins | enabled, max length `32` |
| Protein concat mode | `pcep` |
| Protein concat dim/topk | `64` / `512` |
| Cell LLM | `frozen`, `data/training_ready_exp31_rnaseq/ptv3/derived/cell_llm_embedding_exp31_rnaseq_qwen3_4096.npz` |
| Cell LLM index column | `cell_llm_index` |
| Cell-type LLM | `frozen`, `data/training_ready_exp31_rnaseq/ptv3/derived/cell_type_llm_embedding_qwen3_4096_v2.npz` |
| Logger | disabled: `LOGGER_BACKEND=none`, `LOG_TO_WANDB=0`, `WANDB_MODE=disabled` |

### Outputs And Results

Reporter validation errors: `0`.

| label | BRCA valid AUPRC | non-BRCA FT AUROC | non-BRCA FT AUPRC | exp09 zero-shot AUPRC |
|---|---:|---:|---:|---:|
| `sensitive_early` | 0.853725 | 0.617007 | 0.180543 | 0.147293 |
| `sensitive_late` | 0.836208 | 0.651117 | 0.141063 | 0.083950 |
| `disease_control_early` | 0.847971 | 0.660188 | 0.607784 | 0.472365 |
| `disease_control_late` | 0.649575 | 0.673322 | 0.254908 | 0.153886 |

Selection rule was BRCA valid AUPRC, so the selected setting is `sensitive_early`.

## 2026-07-10 15:05 HKT: exp32 Organoid Exp09 Single-drug Sensitivity Inference

This entry records the inference-only `exp32_organoid_exp09_single_sensitivity` run. No training, fine-tuning, label evaluation, AUROC, or AUPRC was performed.

### Commands

CPU data build and validation:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
python utils/32_build_exp32_organoid_training_ready.py
python -m py_compile utils/32_build_exp32_organoid_training_ready.py scripts/report_exp32_organoid_exp09_single_sensitivity.py infer.py
bash -n scripts/exp_32_organoid_exp09_single_sensitivity_infer.sh
python scripts/report_exp32_organoid_exp09_single_sensitivity.py --preflight-only
```

GPU smoke and full inference were launched only through tmux session `gpu2`:

```bash
tmux send-keys -t gpu2 'cd /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2 && source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh && conda activate flow_v2 && EXP_PREFIX=20260710_exp32_organoid_smoke GPU_IDS=0 INFER_BATCH_SIZE=256 INFER_LIMIT_BATCHES=1 LOGGER_BACKEND=none LOG_TO_WANDB=0 WANDB_MODE=disabled PROGRESS_BAR=0 bash scripts/exp_32_organoid_exp09_single_sensitivity_infer.sh' C-m
tmux send-keys -t gpu2 'cd /mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2 && source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh && conda activate flow_v2 && EXP_PREFIX=20260710_exp32_organoid_exp09_single_sensitivity GPU_IDS=0 INFER_BATCH_SIZE=256 LOGGER_BACKEND=none LOG_TO_WANDB=0 WANDB_MODE=disabled PROGRESS_BAR=0 bash scripts/exp_32_organoid_exp09_single_sensitivity_infer.sh' C-m
```

Final CPU report:

```bash
python scripts/report_exp32_organoid_exp09_single_sensitivity.py
```

### Data And Query Contract

| field | value |
|---|---|
| Training-ready root | `data/training_ready_exp32_organoid` |
| Tasks | `ptv3_exp32_organoid_qe_single`, `ptv3_exp32_organoid_480_faims_single` |
| Samples | 13 biological organoids on each of 2 devices |
| Drug scope | 3,217 unique drugs from exp01-exp09 main+extra non-control rows |
| Queries | 41,821 per device; 83,642 combined |
| Query slots | `pert_id1 == pert_id2` |
| Prediction meaning | single-drug sensitivity probability |
| Exposure | `pert_time=24`, `pert_dose1=pert_dose2=10` |
| B machine mapping | `QE_HF -> QE` |
| CAC machine mapping | `480_FAIMS` |
| Cell/plate/batch model indices | `0` (`no`); raw values retained for audit |
| Tissue categories | `LUNG`, `PANCREAS`, `COLON` with existing categorical and cell-type LLM indices |
| Protein axis | exact exp09 axis, 11,092 proteins |
| Expression transform | finite non-negative values `log1p`; missing values/proteins remain `NaN` |
| Matrix shape per task | `(41834, 11092)`, float32 |
| Matrix storage | streamed `.npy`; processed matrix hard-linked to feature matrix |
| Split | `test_only`; 13 sets, each 1 real baseline control + 3,217 queries |
| Labels | empty; inference-only |

The builder preserved the raw inputs exactly. SHA-256 values before and after the build matched:

- B matrix: `6ed2a164eeea820c994b62349e4c9ef2bba95fe35a7e83b7a0eddfc3efaf44e1`;
- CAC matrix: `388d5ba4f03532c460d3a948d9caf6356708c1b1b910c3af7e362b3851a875f0`;
- sample metadata: `a9ef79649c9b457e7e2dfe435c0ad7c5191387a001bd4de4ab10a79287bd927b`.

### Checkpoint And Model Contract

| field | value |
|---|---|
| Checkpoint | `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=5.ckpt` |
| Checkpoint SHA-256 | `a5734682c37d807b2b6d1bb5ea66ad908107f45e15605aa65c40f6bab9b6db97` |
| Model/head | `fast_delta` / `unified` |
| Pair fusion | `dual`; pair-type features enabled |
| DDI / graph | enabled / real graph, `graph_pair_add_scale=0.5` |
| Protein feature | PCEP enabled; target-protein tokens enabled |
| Cell LLM | frozen existing qwen3-4096 artifact, indexed by `Cell_index=0` |
| Cell-type LLM | frozen existing qwen3-4096 v2 artifact, indexed by tissue `cell_type_index` |
| Batch covariates | machine, plate, Cell, cell type, batch, time, and both dose slots |
| Inference batch size | 256 |
| Logger | disabled |
| Expression prediction | not saved |

`infer.py` recorded and compared all 103 active fast-model configuration fields. Both formal inference manifests report 103 comparisons, zero mismatches, architecture match `true`, and artifact-path match `true`.

### Runtime And Acceptance

| run/task | predictions | duration | status |
|---|---:|---:|---:|
| smoke QE | 256 | 22 s | 0 |
| smoke 480_FAIMS | 256 | 10 s | 0 |
| formal QE | 41,821 | 21 s | 0 |
| formal 480_FAIMS | 41,821 | 21 s | 0 |

Reporter validation passed with 31 SHA-256 records, exact ordered protein-axis equality, 41,821 rows per task, 83,642 finite probabilities in `[0,1]`, and exactly 41,821 one-to-one B/CAC sample-drug pairs.

### Outputs And Descriptive Results

- `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_predictions.csv`
- `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_predictions.parquet`
- `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_summary.json`
- `outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_top20_by_sample.csv`
- `docs/2026-07-10_exp32_organoid_exp09_single_sensitivity_results.md`

Descriptive cross-device results:

- overall B/CAC Pearson `0.992097`, Spearman `0.993484`, and MAE `0.010684`;
- mean signed shift `CAC - B = -0.002734`;
- B mean/median probability `0.072649 / 0.002208`;
- CAC mean/median probability `0.069915 / 0.000983`;
- per-sample Pearson range `0.985673` to `0.993963`;
- top-50 overlaps range from 40 to 49, and top-100 overlaps from 90 to 98.

These are unlabeled predictions. AUROC/AUPRC do not exist for this experiment and are not reported.

## 2026-07-20 21:05 HKT: Exp31/Exp32 Exp09 Epoch-2 Rerun

This entry records the complete epoch-2 rerun of the 2026-07-09 Exp31 four-label benchmark and the 2026-07-10 Exp32 organoid inference. The only experiment-level checkpoint change was the initialization/direct-inference checkpoint; data, splits, model configuration, and the original hyperparameters were retained.

### Checkpoint And Execution Isolation

| field | value |
|---|---|
| Checkpoint | `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt` |
| SHA-256 | `7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076` |
| Exp31 formal prefix | `20260720_exp31_rnaseq_epoch2` |
| Exp32 formal prefix | `20260720_exp32_organoid_exp09_epoch2_single_sensitivity` |
| GPU worker | `gpu2:0`, NVIDIA H200, `flow_v2` |
| GPU preflight policy | `RUN_PREFLIGHT=0 RUN_DATA_VALIDATION=0`; CPU checks completed locally |
| Existing-run policy | new prefixes only; `ALLOW_EXISTING_RUN` remained `0` |

CPU preflight validated the epoch-2 hash, all Exp31/Exp32 data and feature contracts, 3,217-drug scope, exact 11,092-protein axis, 31 Exp32 provenance hash records, reporter syntax, and runner syntax. Exp31 wrote `outputs/2026-07/2026-07-20/20260720_exp31_rnaseq_epoch2_preflight.json` with no errors.

GPU smoke used `20260720_exp31_rnaseq_epoch2_smoke` and `20260720_exp32_organoid_exp09_epoch2_smoke`. Exp31 ran all four labels with one train/valid/test/inference batch, producing 4 successful train stages and 8 successful 256-row inference stages. Exp32 produced two successful 256-row inference outputs; the CPU reporter `--smoke --smoke-expected-rows 256` passed.

Formal GPU launches were serialized on the same worker:

```bash
EXP31_INIT_CKPT=checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt EXP_PREFIX=20260720_exp31_rnaseq_epoch2 GPU_IDS=0 DEVICES=1 LOGGER_BACKEND=none LOG_TO_WANDB=0 WANDB_MODE=disabled RUN_PREFLIGHT=0 RUN_DATA_VALIDATION=0 RUN_INFERENCE=1 RUN_ZEROSHOT=1 RUN_REPORT=0 PROGRESS_BAR=0 bash scripts/exp_31_rnaseq_pdx_ft_benchmark.sh
EXP32_CHECKPOINT=/mnt/shared-storage-gpfs2/beam-gpfs02/wuhao/PTV/proteintalk_v2/checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt EXP_PREFIX=20260720_exp32_organoid_exp09_epoch2_single_sensitivity GPU_IDS=0 INFER_BATCH_SIZE=256 RUN_PREFLIGHT=0 RUN_DATA_VALIDATION=0 bash scripts/exp_32_organoid_exp09_single_sensitivity_infer.sh
```

### Exp31 Results

All 12 runtime rows completed with status `0` in 235 seconds total. Each of the eight non-BRCA inference outputs contains exactly 1,827 predictions. All four training manifests record `fit_completed`, an existing best checkpoint, and epoch 2 in `args.checkpoint_path`; all four zero-shot manifests also point exactly to epoch 2.

| label | BRCA valid AUPRC | FT non-BRCA AUROC | FT non-BRCA AUPRC | epoch-2 zero-shot AUPRC |
|---|---:|---:|---:|---:|
| `sensitive_early` | 0.656938 | 0.699766 | 0.232370 | 0.237176 |
| `sensitive_late` | 0.582358 | 0.725122 | 0.173471 | 0.160951 |
| `disease_control_early` | 0.850399 | 0.682257 | 0.624594 | 0.565795 |
| `disease_control_late` | 0.528473 | 0.661026 | 0.244798 | 0.219429 |

Selection by BRCA valid AUPRC chose `disease_control_early`.

Final Exp31 artifacts:

- `outputs/2026-07/2026-07-20/20260720_exp31_rnaseq_epoch2_pdx_ft_benchmark_summary.csv` (56 rows, unchanged 16-column schema);
- `outputs/2026-07/2026-07-20/20260720_exp31_rnaseq_epoch2_pdx_ft_benchmark_summary.json` (`errors=[]`);
- `docs/2026-07-20_exp31_rnaseq_epoch2_pdx_ft_benchmark_results.md` (unchanged report-section contract);
- `logs/20260720_exp31_rnaseq_epoch2_gpu.log` and `logs/20260720_exp31_rnaseq_epoch2_runtime_summary.tsv`.

### Exp32 Results

Both formal inference stages completed with status `0` in 25 seconds total. Each raw output contains 41,821 predictions and its manifest reports the exact epoch-2 checkpoint, complete architecture validation, and zero checkpoint-config mismatches. The final combined CSV and Parquet each contain 83,642 rows and 18 columns, spanning two devices, 13 samples, and 3,217 drugs; every probability is finite and within `[0,1]`. The top-20 table has 520 rows.

The overall B/CAC paired Pearson correlation is `0.995239`, Spearman is `0.998963`, MAE is `0.008309`, and mean signed `CAC - B` shift is `-0.007193`. These remain unlabeled descriptive predictions; no AUROC/AUPRC is computed for Exp32.

Final Exp32 artifacts:

- `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_predictions.csv` and `.parquet`;
- `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_top20_by_sample.csv`;
- `outputs/2026-07/2026-07-20/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_summary.json` (`status=complete`);
- `docs/2026-07-20_exp32_organoid_exp09_epoch2_single_sensitivity_results.md`;
- `logs/20260720_exp32_organoid_exp09_epoch2_single_sensitivity_gpu.log` and runtime TSV.

Final format comparison passed against the 2026-07-09/10 artifacts. Historical result mtimes and paths were unchanged, and all rerun outputs used new epoch-2 prefixes.

## 2026-07-27 13:40 HKT: exp33 Double-drug Virtual-screen Inference

This entry records inference-only screening of the colon, lung, and pancreas
`data/rawdata/vc_doubledrug` query files with the fixed exp09 epoch-2 unified-head
checkpoint. No training or fine-tuning was performed.

### Commands

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
python utils/33_build_exp33_vc_doubledrug_training_ready.py
python utils/33_build_exp33_vc_doubledrug_training_ready.py --preflight-only

# Sent to the already existing and verified-idle gpu:0.0 tmux pane.
EXP_PREFIX=20260727_exp33_vc_doubledrug_epoch2_smoke INFER_LIMIT_BATCHES=1 GPU_IDS=0 INFER_DEVICE=cuda:0 REQUIRE_GPU_TMUX=0 bash scripts/exp_33_vc_doubledrug_epoch2_infer.sh
python scripts/report_exp33_vc_doubledrug_epoch2.py --smoke --prefix 20260727_exp33_vc_doubledrug_epoch2_smoke --run-dir outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_smoke --output-dir outputs/2026-07/2026-07-27

# Sent to the same verified-idle gpu:0.0 tmux pane after smoke validation passed.
EXP_PREFIX=20260727_exp33_vc_doubledrug_epoch2 GPU_IDS=0 INFER_DEVICE=cuda:0 REQUIRE_GPU_TMUX=0 bash scripts/exp_33_vc_doubledrug_epoch2_infer.sh
python scripts/report_exp33_vc_doubledrug_epoch2.py --prefix 20260727_exp33_vc_doubledrug_epoch2 --run-dir outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2 --output-dir outputs/2026-07/2026-07-27
```

`REQUIRE_GPU_TMUX=0` was used only inside the remote GPU worker shell because that
inner shell cannot resolve the host tmux socket. The outer `gpu:0.0` pane was
inspected before both launches and was idle; no GPU command was launched outside
that pane.

### Data and covariate contract

- Checkpoint:
  `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`;
  SHA-256
  `7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076`.
- Structural canonical-isomeric-SMILES matching resolved all 227 query drugs:
  51 numeric IDs, 176 `L9200_*` IDs, and no external IDs. The confirmed
  `(-)-Menthol` exception resolves to `L9200_2195`.
- The 28 same-tissue raw-cell baselines come from the supplied mixed-6h/24h
  `*_control_unique.csv` files. They are aligned to 11,092 proteins and retain
  missing values as `NaN`.
- Of 28 cells, 18 are checkpoint-train-seen. The 10 unseen cells use
  `Cell_index=0` while retaining their true nonzero `cell_llm_index`.
  `cell_type`, machine, 24h, and all realized dose buckets are train-supported.
  `Cell_plate=no` and `batch=no` are explicit checkpoint-training OOD categories.
- The isolated training-ready root contains 1,124,928 unique model keys and
  compact expression matrices with 7, 16, and 8 rows for colon, lung, and
  pancreas respectively.

### Execution and results

The three one-batch GPU smoke jobs each produced 256 finite bounded probabilities,
with zero checkpoint-config mismatches and no expression-prediction files. The
formal GPU runtime rows all completed with status `0`:

| tissue | unique predictions | GPU seconds | probability min | probability max | probability mean |
|---|---:|---:|---:|---:|---:|
| colon | 218,736 | 32 | 0.000609 | 0.998385 | 0.413786 |
| lung | 462,210 | 58 | 0.000605 | 0.998509 | 0.336969 |
| pancreas | 443,982 | 53 | 0.000308 | 0.996284 | 0.083759 |
| total | 1,124,928 | 143 | — | — | — |

CPU reporting mapped the unique predictions back to all 2,526,720 raw rows in
the original tissue/file/row order. An independent streamed verification
confirmed exact score equality for every repeated `model_key_id`, finite
probabilities in `[0,1]`, and tissue row totals of 403,200, 806,400, and
1,317,120. Because the screen is unlabeled, no AUROC/AUPRC or top-ranking table
was produced.

Final artifacts:

- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_predictions.csv`;
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_predictions.parquet`;
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_unique_model_predictions.parquet`;
- drug-ID and covariate audit CSVs with the same prefix;
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_summary.json`;
- `outputs/2026-07/2026-07-27/20260727_exp33_vc_doubledrug_epoch2_results.md`;
- score-distribution PNG/PDF and its summary CSV with the same prefix;
- formal and smoke runtime logs/TSVs under `outputs/2026-07/2026-07-27`.

## 2026-08-19 14:22 HKT: Exp34 Update-0819 OOD Single-drug Inference

No training or fine-tuning was performed. The fixed exp09 epoch-2 checkpoint was
used for two target-only sensitivity branches over 14 cell-drug pairs at 24 h
and 10 uM.

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2

# CPU/test validation. The shared project volume was at quota, so expanded
# copy-on-write matrices were generated in /tmp on the execution worker.
python utils/34_build_update0819_ood_training_ready.py \
  --output-root /tmp/proteintalk_exp34_update0819_ood_runtime
python utils/34_build_update0819_ood_training_ready.py \
  --output-root /tmp/proteintalk_exp34_update0819_ood_runtime --preflight-only
PYTHONPATH=. pytest -q \
  tests/test_exp34_update0819_ood.py \
  tests/test_training_ready_fast_expression_row_index.py

# Both commands were sent to the existing gpu1_deep:0 tmux window. The inner
# rlaunch worker cannot access the outer tmux socket, so the resolved outer
# target is passed explicitly through EXP34_TMUX_TARGET.
EXP34_TMUX_TARGET=gpu1_deep:0 \
TRAINING_READY_ROOT=/tmp/proteintalk_exp34_update0819_ood_runtime \
EXP_PREFIX=20260819_exp34_update0819_ood_epoch2_smoke \
INFER_BATCH_SIZE=2 INFER_LIMIT_BATCHES=1 RUN_BUILD=0 \
bash scripts/exp_34_update0819_ood_epoch2_infer.sh

EXP34_TMUX_TARGET=gpu1_deep:0 \
TRAINING_READY_ROOT=/tmp/proteintalk_exp34_update0819_ood_runtime \
EXP_PREFIX=20260819_exp34_update0819_ood_epoch2 \
INFER_BATCH_SIZE=256 RUN_BUILD=0 \
bash scripts/exp_34_update0819_ood_epoch2_infer.sh

python scripts/report_exp34_update0819_ood_epoch2.py \
  --prefix 20260819_exp34_update0819_ood_epoch2 \
  --run-dir outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2 \
  --output-dir outputs/2026-08/2026-08-19 \
  --training-ready-root /tmp/proteintalk_exp34_update0819_ood_runtime
```

The formal target-none probability range is `0.011219..0.031873`; the
mechanism-target range is `0.011503..0.041512`. The mean mechanism-minus-none
probability shift is `0.001785`. These are unlabeled chemical-OOD predictions,
so AUROC/AUPRC and automatically emitted expression-reconstruction metrics are
not interpreted.

## 2026-08-19 14:44 HKT: Exp34 Per-sample Protein Attribution

The formal response logits were attributed to the 11,092 control-expression
proteins for all 28 target-scenario samples. The reference vector is the
checkpoint-train control median; non-protein inputs stay fixed at each query's
formal inference values.

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2

PYTHONPATH=. pytest -q \
  tests/test_exp34_protein_attribution.py \
  tests/test_exp34_update0819_ood.py \
  tests/test_training_ready_fast_expression_row_index.py

# Sent to gpu1_deep:0; runtime Exp34 artifacts remained in the same H200 worker.
EXP34_TMUX_TARGET=gpu1_deep:0 PYTHONUNBUFFERED=1 \
python scripts/attribute_exp34_update0819_ood_epoch2.py \
  --training-ready-root /tmp/proteintalk_exp34_update0819_ood_runtime \
  --output-dir outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2_protein_attribution_smoke \
  --device cuda:0 --limit-samples 2 --ig-steps 4 --adaptive-ig-steps 8

EXP34_TMUX_TARGET=gpu1_deep:0 PYTHONUNBUFFERED=1 \
python scripts/attribute_exp34_update0819_ood_epoch2.py \
  --training-ready-root /tmp/proteintalk_exp34_update0819_ood_runtime \
  --output-dir outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2_protein_attribution \
  --device cuda:0 --ig-steps 32 --adaptive-ig-steps 64
```

Formal attribution completed in about three seconds, reproduced all published
probabilities exactly, and generated 310,576 finite sample-protein records.
Every sample has a complete deterministic 1..11,092 absolute-IG rank. No
cross-cell or cross-drug consensus rank was produced, matching the requested
per-sample-only scope.

## 2026-08-21 16:31 HKT: Exp35 Update-0821 Manual-target Inference and Attribution

No training or fine-tuning was performed. Exp35 reused the fixed Exp09 epoch-2
checkpoint and the August 19 OOD feature contract, but read targets from
`data/rawdata/update_0821/260820ptv_drug_cell_predict_target.csv` and executed
only the manual-target task.

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2

python -m unittest tests.test_exp35_update0821_target -v
bash -n scripts/exp_34_update0819_ood_epoch2_infer.sh \
  scripts/exp_35_update0821_target_epoch2_infer.sh

# Commands below ran inside the H200 worker launched from tmux gpu1:0.
EXP35_TMUX_TARGET=gpu1:0 \
EXP_PREFIX=20260821_exp35_update0821_target_epoch2_gpu_smoke \
INFER_BATCH_SIZE=2 INFER_LIMIT_BATCHES=1 RUN_BUILD=1 \
bash scripts/exp_35_update0821_target_epoch2_infer.sh

python scripts/report_exp35_update0821_target_epoch2.py \
  --prefix 20260821_exp35_update0821_target_epoch2_gpu_smoke \
  --run-dir outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_gpu_smoke \
  --training-ready-root /tmp/proteintalk_exp35_update0821_target_runtime \
  --smoke

EXP35_TMUX_TARGET=gpu1:0 \
python scripts/attribute_exp35_update0821_target_epoch2.py \
  --run-root outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_gpu_smoke \
  --training-ready-root /tmp/proteintalk_exp35_update0821_target_runtime \
  --output-dir outputs/2026-08/2026-08-21/20260821_exp35_update0821_target_epoch2_gpu_smoke_protein_attribution \
  --device cuda:0 --limit-samples 2 --ig-steps 4 --adaptive-ig-steps 8

EXP35_TMUX_TARGET=gpu1:0 INFER_BATCH_SIZE=256 RUN_BUILD=0 \
bash scripts/exp_35_update0821_target_epoch2_infer.sh

python scripts/report_exp35_update0821_target_epoch2.py \
  --training-ready-root /tmp/proteintalk_exp35_update0821_target_runtime

EXP35_TMUX_TARGET=gpu1:0 \
python scripts/attribute_exp35_update0821_target_epoch2.py \
  --device cuda:0 --ig-steps 32 --adaptive-ig-steps 64
```

The first smoke command was accidentally issued from the outer `gpu1` host
shell before an H200 worker was active. Runtime construction passed, but CUDA
initialization failed and no valid prediction output was produced. The shared
runner now fails before construction unless both `nvidia-smi` and PyTorch CUDA
availability succeed. The smoke was then rerun in the same `gpu1` tmux session
inside an H200 worker and completed successfully.

Formal inference generated 14 probabilities and a finite `14 x 11092`
perturbed-expression matrix. All scores are below 0.5 and therefore map to the
checkpoint's `non-responsive` class. Regression against the August 19
`target_mechanism` branch gave zero maximum error for probability and predicted
expression; the model-input matched-control matrix differs from its legacy
decimal CSV by at most `9.536743e-7`.

Formal IG used the checkpoint-training median over 498 referenced control rows
as the baseline. It reproduced every published probability exactly, emitted
155,288 finite sample-protein rows, made no adaptive IG-64 reruns, and reported
zero completeness warnings. The maximum absolute completeness error was
`0.000182003`. Rankings remain per sample only.
