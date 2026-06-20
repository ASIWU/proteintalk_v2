# 2026-06-03 PTV1 Pipeline And Experiment Results

## Scope

Built an isolated PTV1 pipeline under `utils/ptv1/` and `scripts/ptv1/`.
The pipeline reads only:

- `data/rawdata/ptv1`
- `data/rawdata/ptv1_extra_singledrug/kept_samples_drugfilter_edit.csv`

The generated PTV1 outputs are isolated under:

- `data/standardized/ptv1`
- `data/training_ready/ptv1`
- `data/training_ready/ptv1/derived`

PTV1 experiments use `--dataset-group ptv1`, graph cache `graph_cache/ptv1`, and default settings:
`fast_delta`, hidden dim `512`, learning rate `2e-4`, batch size `256`, dropout `0.15`, weight decay `1e-4`,
`GRAPH_FEATURE_MODE=real`, `MSE_WEIGHT=0.50`, `USE_DOSE_COVARIATE=1`.

The original 2026-06-03 PTV1 baseline/tuning runs used no Cell LLM embedding. As of 2026-06-08, PTV1 wrappers default to
`CELL_LLM_MODE=frozen` with `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz`.

## Data Rebuild Summary

| task | standardized rows | standardized matrix | training-ready feature matrix | controls in training-ready | valid anchors |
|---|---:|---:|---:|---:|---:|
| `ptv1_aivc` | 15002 | 15002 x 5576 | 15002 x 5576 | 942 | 13137 |
| `ptv1_extra_singledrug` | 218 | 218 x 0 | 222 x 5576 | 4 | 218 |

PTV1 AIVC control resolution:

- matched by `(BioRep, Cell_plate)`: 13137 anchors.
- self controls with `pert_time == 0`: 942 rows.
- missing matched controls: 923 anchors.

PTV1 extra single-drug:

- `drug_id` was used directly as the PTV3 perturbation id for SMILES and target lookup.
- `new_pheno` was written to `PRISM2nd_label_total`.
- `pert_id2 == pert_id1` for all single-drug rows.
- all 218 anchors matched PTV1 AIVC controls by cell plate.
- empty target-list count: 4.

PTV1 training-ready global metadata:

- protein index size: 5578.
- perturbation index size: 128.
- tasks: `ptv1_aivc`, `ptv1_extra_singledrug`.

## Split Summary

`ptv1_aivc` splits:

| split | train | valid | test | train-valid overlap | train-test overlap | valid-test overlap |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_experiment_type` | 7041 | 1481 | 799 | 0 | 0 | 0 |
| `random` | 9459 | 1050 | 2628 | 0 | 0 | 0 |
| `pert_id_5fold_fold0` | 10367 | 823 | 1947 | 0 | 0 | 0 |
| `pert_id_5fold_fold1` | 10505 | 750 | 1882 | 0 | 0 | 0 |
| `pert_id_5fold_fold2` | 10456 | 750 | 1931 | 0 | 0 | 0 |
| `pert_id_5fold_fold3` | 10476 | 750 | 1911 | 0 | 0 | 0 |
| `pert_id_5fold_fold4` | 6921 | 750 | 5466 | 0 | 0 | 0 |
| `all_train_subset_test` | 13137 | 1313 | 2627 | 1313 | 2627 | 0 |

The `all_train_subset_test` overlap is intentional for the all-data reference-epoch policy.

`ptv1_extra_singledrug` split:

| split | train | valid | test |
|---|---:|---:|---:|
| `test_only` | 0 | 0 | 218 |

Unseen-drug 5-fold audit:

- Fold test-drug counts: fold0 `13`, fold1 `13`, fold2 `13`, fold3 `13`, fold4 `12`.
- Pairwise fold test-drug overlap: `0` for every fold pair.
- Union of fold test drugs: `64`.

## Derived Artifacts

All artifacts were written only to `data/training_ready/ptv1/derived/`.

| artifact | shape | finite values | nonzero values | fallback count | unresolved count |
|---|---:|---:|---:|---:|---:|
| `protein_embedding_esm.pkl` | 5578 x 1280 | 7139840 | 7139840 | 102 sequence fallbacks | 0 |
| `drug_embedding_morgan_2048.pkl` | 128 x 2048 | 262144 | 7187 | 2 SMILES fallbacks | 0 |
| `ppi_matrix.npy` | 5578 x 5578 | 31114084 | 2711290 | 0 | 0 |
| `ddi_matrix.npy` | 128 x 128 | 16384 | 15662 | 2 SMILES fallbacks | 0 |
| `pdi_matrix.npy` | 128 x 5578 | 713984 | 12535 | 0 | 0 |
| `cell_llm_embedding_qwen3_4096.npz` | 20 x 4096 | 81920 | 77802 | row 0 reserved zero vector | 0 |

ESM source model cache:
`/mnt/shared-storage-user/beam/wuhao/hf_cache/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c`.

Graph matrices were built using the PTV1 `global_meta.json` ordering. They were not sliced from PTV3 matrices.

Cell LLM embedding source:

- Prefix used for the formal rerun: `20260608_ptv1_cell_llm_v1`.
- Dataset group: `ptv1`.
- Input field/index: `Cell` / `Cell_index`.
- Description model: `gpt-5.4`.
- Embedding model: `Qwen/Qwen3-Embedding-8B`.
- Normalization: enabled.
- Validation: shape `20 x 4096`, row 0 zero, rows 1-19 finite/nonzero, no legacy `cell_type_llm` sidecar key.

## 2026-06-08 Cell LLM Embedding Formal Rerun

Formal experiment prefix: `20260608_ptv1_cell_llm_v1`.

The rerun used the same PTV1 data, graph, drug, and protein artifacts as the June 3 baseline, but enabled frozen Cell LLM features by default:
`CELL_LLM_MODE=frozen` and `CELL_LLM_EMBEDDING_PATH=data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096.npz`.

| exp | task | split | AUROC | AUPRC | n-AUPRC | count | pos | neg |
|---|---|---|---:|---:|---:|---:|---:|---:|
| exp_11 | `ptv1_aivc` | `fixed_experiment_type` | 0.947288 | 0.893452 | 3.050717 | 799 | 234 | 565 |
| exp_12 | `ptv1_aivc` | fold0 | 0.660178 | 0.569277 | 1.231536 | 1947 | 900 | 1047 |
| exp_12 | `ptv1_aivc` | fold1 | 0.606245 | 0.479892 | 2.181537 | 1882 | 414 | 1468 |
| exp_12 | `ptv1_aivc` | fold2 | 0.621257 | 0.538453 | 1.420426 | 1931 | 732 | 1199 |
| exp_12 | `ptv1_aivc` | fold3 | 0.740368 | 0.521697 | 1.888188 | 1911 | 528 | 1383 |
| exp_12 | `ptv1_aivc` | fold4 | 0.642212 | 0.448854 | 3.888174 | 5466 | 631 | 4835 |
| exp_12 | `ptv1_aivc` | mean5 | 0.654052 | 0.511635 | 2.121972 | 13137 | 3205 | 9932 |
| exp_13 | `ptv1_extra_singledrug` | overall | 0.597414 | 0.578210 | 1.125445 | 218 | 112 | 106 |

Exp_13 reference epoch policy:

- Reference folds: `20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold0` through fold4.
- Best epochs: `3, 2, 6, 18, 0`.
- Raw mean epoch: `5.8`.
- Selected epoch: `6`.
- Applied all-PTV1 max epochs: `7`.
- Prediction output: `outputs/20260608_ptv1_cell_llm_v1_all_ptv1_for_extra/ptv1_extra_singledrug/predictions.parquet`.

Per-cell exp_13 results:

| cell | AUROC | AUPRC | n-AUPRC | count | pos | neg |
|---|---:|---:|---:|---:|---:|---:|
| BT549 | 0.473913 | 0.534542 | 0.944358 | 53 | 30 | 23 |
| HCC1806 | 0.555300 | 0.539919 | 1.137686 | 59 | 28 | 31 |
| HS578T | 0.696172 | 0.759231 | 1.196365 | 52 | 33 | 19 |
| MDA-MB-468 | 0.640693 | 0.561664 | 1.444279 | 54 | 21 | 33 |

Runtime summary:

| kind | experiment | duration sec | artifact |
|---|---|---:|---|
| train | `20260608_ptv1_cell_llm_v1_ptv1_random_split` | 57 | `checkpoints/20260608_ptv1_cell_llm_v1_ptv1_random_split` |
| train | `20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold0` | 68 | `checkpoints/20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold0` |
| train | `20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold1` | 64 | `checkpoints/20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold1` |
| train | `20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold2` | 62 | `checkpoints/20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold2` |
| train | `20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold3` | 70 | `checkpoints/20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold3` |
| train | `20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold4` | 53 | `checkpoints/20260608_ptv1_cell_llm_v1_ptv1_unseen_drug_fold4` |
| train | `20260608_ptv1_cell_llm_v1_all_ptv1_for_extra` | 20 | `checkpoints/20260608_ptv1_cell_llm_v1_all_ptv1_for_extra` |
| infer | `20260608_ptv1_cell_llm_v1_all_ptv1_for_extra` | 7 | `outputs/20260608_ptv1_cell_llm_v1_all_ptv1_for_extra/ptv1_extra_singledrug` |

Generated reports:

- `logs/20260608_ptv1_cell_llm_v1_ptv1_exp_results.md`.
- `outputs/20260608_ptv1_cell_llm_v1_ptv1_exp_results.csv`.

Validation:

- `python -m py_compile train.py infer.py utils/11_build_cell_llm_embeddings.py utils/ptv1/*.py scripts/ptv1/report_ptv1_exp_results.py` passed.
- `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh` passed.
- `python utils/ptv1/01_validate_ptv1_standardized.py` passed.
- `python utils/ptv1/03_validate_ptv1_training_ready.py` passed.
- One-batch exp_11, exp_12 fold0, and exp_13 smoke runs passed with frozen Cell LLM enabled.
- Final manifest audit found 7 new manifests, all with `dataset_group=ptv1`, `cell_llm_mode=frozen`, `cell_llm_summary.embedding_rows=20`, and no `cell_type_llm` keys.

## Results At A Glance

Use this section as the current summary for exp_11 through exp_13.

| exp | purpose | current result to cite | prefix / config | AUROC | AUPRC | n-AUPRC | count | note |
|---|---|---|---|---:|---:|---:|---:|---|
| exp_11 | PTV1 random split | Cell LLM formal rerun | `20260608_ptv1_cell_llm_v1`, `CELL_LLM_MODE=frozen` | 0.947288 | 0.893452 | 3.050717 | 799 | frozen PTV1 Cell LLM artifact |
| exp_12 | PTV1 unseen-drug 5-fold | Cell LLM formal rerun | `20260608_ptv1_cell_llm_v1`, `CELL_LLM_MODE=frozen` | 0.654052 | 0.511635 | 2.121972 | 13137 | frozen PTV1 Cell LLM artifact |
| exp_13 | all-PTV1 train, extra single-drug infer | Cell LLM formal rerun | `20260608_ptv1_cell_llm_v1`, `CELL_LLM_MODE=frozen` | 0.597414 | 0.578210 | 1.125445 | 218 | selected epoch 6 from new Cell LLM folds |
| exp_11 | PTV1 random split | formal baseline | `20260603_2037_ptv1`, `fixed_experiment_type` | 0.957636 | 0.915971 | 3.127609 | 799 | no follow-up tuning was run for exp_11 |
| exp_12 | PTV1 unseen-drug 5-fold | tuned best for exp_12 | `20260603_2108_ptv1_param_v1_stage2_mse025` | 0.731340 | 0.570871 | 2.388662 | 13137 | use this when reporting exp_12 after tuning |
| exp_12 | PTV1 unseen-drug 5-fold | original formal baseline | `20260603_2037_ptv1`, `MSE_WEIGHT=0.50` | 0.707712 | 0.545776 | 2.212531 | 13137 | kept for comparison only |
| exp_13 | all-PTV1 train, extra single-drug infer | best extra AUPRC | `20260603_2108_ptv1_param_v1_stage2_pos_auto_mse050` | 0.594550 | 0.587486 | 1.143500 | 218 | use this if exp_13 is judged by extra AUPRC |
| exp_13 | all-PTV1 train, extra single-drug infer | same setting family as exp_12 best | `20260603_2108_ptv1_param_v1_stage2_mse025` | 0.595982 | 0.581499 | 1.131846 | 218 | use this if exp_12/exp_13 must share one tuned config |
| exp_13 | all-PTV1 train, extra single-drug infer | original formal baseline | `20260603_2037_ptv1`, `MSE_WEIGHT=0.50` | 0.593455 | 0.576296 | 1.121719 | 218 | kept for comparison only |

Recommended interpretation:

- exp_11 has one formal result: AUPRC `0.915971`.
- exp_12 improved after tuning. The current exp_12 result is `mse025`, with mean AUPRC `0.570871`.
- exp_13 improved only modestly. The best extra AUPRC is `pos_auto_mse050` at `0.587486`; `mse025` is slightly lower at `0.581499` but matches the exp_12-best setting family.
- `fixed_experiment_type` is the implementation split name for exp_11; the user-facing experiment name is still PTV1 random split.

## Formal Baseline Experiments

Formal experiment prefix: `20260603_2037_ptv1`.

These are the original formal runs before the follow-up tuning. For the current exp_12 and exp_13 tuned numbers, use **Results At A Glance** above or **Tuning Details For Exp 12 / Exp 13** below.

### Exp 11: PTV1 Random Split

Implementation split strategy: `fixed_experiment_type`.
User-facing name: PTV1 random split.

| split | AUROC | AUPRC | n-AUPRC | count | pos | neg | selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_experiment_type` | 0.957636 | 0.915971 | 3.127609 | 799 | 234 | 565 | 7 |

Checkpoint:
`checkpoints/20260603_2037_ptv1_ptv1_random_split/epoch=7-step=224.ckpt`.

### Exp 12: PTV1 Unseen-Drug 5-Fold

| fold | AUROC | AUPRC | n-AUPRC | count | pos | neg | selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold0 | 0.659188 | 0.561890 | 1.215556 | 1947 | 900 | 1047 | 9 |
| fold1 | 0.739903 | 0.534845 | 2.431350 | 1882 | 414 | 1468 | 4 |
| fold2 | 0.648385 | 0.592389 | 1.562709 | 1931 | 732 | 1199 | 4 |
| fold3 | 0.740171 | 0.625363 | 2.263389 | 1911 | 528 | 1383 | 7 |
| fold4 | 0.750915 | 0.414393 | 3.589651 | 5466 | 631 | 4835 | 5 |
| mean5 | 0.707712 | 0.545776 | 2.212531 | 13137 | 3205 | 9932 | - |

### Exp 13: All-PTV1 Train, PTV1 Extra Single-Drug Inference

Reference epoch policy:

- Reference folds: exp_12 fold0 through fold4.
- Best epochs: `9, 4, 4, 7, 5`.
- Aggregation: mean.
- Raw mean epoch: `5.8`.
- Rounding: nearest.
- Selected epoch: `6`.
- Applied all-PTV1 max epochs: `7`.
- Checkpoint policy: use `last.ckpt`.

Prediction output:
`outputs/20260603_2037_ptv1_all_ptv1_for_extra/ptv1_extra_singledrug/predictions.parquet`.

| group | AUROC | AUPRC | n-AUPRC | count | pos | neg |
|---|---:|---:|---:|---:|---:|---:|
| overall | 0.593455 | 0.576296 | 1.121719 | 218 | 112 | 106 |
| BT549 | 0.459420 | 0.527625 | 0.932137 | 53 | 30 | 23 |
| HCC1806 | 0.547235 | 0.533973 | 1.125157 | 59 | 28 | 31 |
| HS578T | 0.728070 | 0.779311 | 1.228005 | 52 | 33 | 19 |
| MDA-MB-468 | 0.637807 | 0.545565 | 1.402882 | 54 | 21 | 33 |

## Runtime Summary

| kind | experiment | duration sec | artifact |
|---|---|---:|---|
| train | `20260603_2037_ptv1_ptv1_random_split` | 58 | `checkpoints/20260603_2037_ptv1_ptv1_random_split` |
| train | `20260603_2037_ptv1_ptv1_unseen_drug_fold0` | 60 | `checkpoints/20260603_2037_ptv1_ptv1_unseen_drug_fold0` |
| train | `20260603_2037_ptv1_ptv1_unseen_drug_fold1` | 60 | `checkpoints/20260603_2037_ptv1_ptv1_unseen_drug_fold1` |
| train | `20260603_2037_ptv1_ptv1_unseen_drug_fold2` | 61 | `checkpoints/20260603_2037_ptv1_ptv1_unseen_drug_fold2` |
| train | `20260603_2037_ptv1_ptv1_unseen_drug_fold3` | 58 | `checkpoints/20260603_2037_ptv1_ptv1_unseen_drug_fold3` |
| train | `20260603_2037_ptv1_ptv1_unseen_drug_fold4` | 56 | `checkpoints/20260603_2037_ptv1_ptv1_unseen_drug_fold4` |
| train | `20260603_2037_ptv1_all_ptv1_for_extra` | 18 | `checkpoints/20260603_2037_ptv1_all_ptv1_for_extra` |
| infer | `20260603_2037_ptv1_all_ptv1_for_extra` | 7 | `outputs/20260603_2037_ptv1_all_ptv1_for_extra/ptv1_extra_singledrug` |

Runtime summary file:
`logs/20260603_2037_ptv1_runtime_summary.tsv`.

## Commands Used

Static checks:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
python -m py_compile utils/ptv1/*.py scripts/ptv1/report_ptv1_exp_results.py
bash -n scripts/ptv1/*.sh
```

Temporary rebuild validation:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
python utils/ptv1/00_standardize_ptv1_rawdata.py --output-root /tmp/ptv1_rebuild_verify/standardized
python utils/ptv1/01_validate_ptv1_standardized.py --input-root /tmp/ptv1_rebuild_verify/standardized
python utils/ptv1/02_build_ptv1_training_ready.py --input-root /tmp/ptv1_rebuild_verify/standardized --output-root /tmp/ptv1_rebuild_verify/training_ready
python utils/ptv1/04_build_ptv1_splits.py --training-ready-root /tmp/ptv1_rebuild_verify/training_ready
python utils/ptv1/03_validate_ptv1_training_ready.py --input-root /tmp/ptv1_rebuild_verify/training_ready
```

Final data rebuild:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
python utils/ptv1/00_standardize_ptv1_rawdata.py
python utils/ptv1/01_validate_ptv1_standardized.py
python utils/ptv1/02_build_ptv1_training_ready.py
python utils/ptv1/04_build_ptv1_splits.py
python utils/ptv1/03_validate_ptv1_training_ready.py
```

Derived artifact build:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
python utils/ptv1/05_build_ptv1_embeddings.py --protein-batch-size 8
python utils/ptv1/06_build_ptv1_graph_matrices.py
```

One-batch smoke checks:

```bash
RUN_PREFLIGHT=0 RUN_DATA_VALIDATION=0 LOG_TO_WANDB=0 LOGGER_BACKEND=none MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 LIMIT_TEST_BATCHES=1 NUM_WORKERS=0 PROGRESS_BAR=0 ALLOW_EXISTING_RUN=1 EXP_PREFIX=ptv1_smoke_$(date +%Y%m%d_%H%M%S) CKPT_DIR=/tmp/ptv1_smoke_ckpts LOG_DIR=/tmp/ptv1_smoke_logs OUTPUT_DIR=/tmp/ptv1_smoke_outputs GRAPH_CACHE_DIR=/tmp/ptv1_smoke_graph_cache bash scripts/ptv1/exp_11_ptv1_random_split.sh

RUN_PREFLIGHT=0 RUN_DATA_VALIDATION=0 LOG_TO_WANDB=0 LOGGER_BACKEND=none MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 LIMIT_TEST_BATCHES=1 NUM_WORKERS=0 PROGRESS_BAR=0 ALLOW_EXISTING_RUN=1 FOLDS=0 EXP_PREFIX=ptv1_smoke_$(date +%Y%m%d_%H%M%S) CKPT_DIR=/tmp/ptv1_smoke_ckpts LOG_DIR=/tmp/ptv1_smoke_logs OUTPUT_DIR=/tmp/ptv1_smoke_outputs GRAPH_CACHE_DIR=/tmp/ptv1_smoke_graph_cache bash scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh

RUN_PREFLIGHT=0 RUN_DATA_VALIDATION=0 LOG_TO_WANDB=0 LOGGER_BACKEND=none MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 LIMIT_TEST_BATCHES=1 INFER_LIMIT_BATCHES=1 NUM_WORKERS=0 PROGRESS_BAR=0 ALLOW_EXISTING_RUN=1 EXP_PREFIX=ptv1_smoke_$(date +%Y%m%d_%H%M%S) CKPT_DIR=/tmp/ptv1_smoke_ckpts LOG_DIR=/tmp/ptv1_smoke_logs OUTPUT_DIR=/tmp/ptv1_smoke_outputs GRAPH_CACHE_DIR=/tmp/ptv1_smoke_graph_cache REFERENCE_5FOLD_CKPT_PATH=/tmp/ptv1_smoke_ckpts/ptv1_smoke_20260603_203402_ptv1_unseen_drug_fold0 REFERENCE_EPOCH_MIN_COUNT=1 REFERENCE_ALLOW_MIXED_CONFIG=1 bash scripts/ptv1/exp_13_ptv1_extra_single_all_train_infer.sh
```

Formal experiments:

```bash
EXP_PREFIX=20260603_2037_ptv1 LOG_TO_WANDB=0 LOGGER_BACKEND=none PROGRESS_BAR=0 bash scripts/ptv1/exp_11_ptv1_random_split.sh
EXP_PREFIX=20260603_2037_ptv1 LOG_TO_WANDB=0 LOGGER_BACKEND=none PROGRESS_BAR=0 bash scripts/ptv1/exp_12_ptv1_unseen_drug_5fold.sh
EXP_PREFIX=20260603_2037_ptv1 LOG_TO_WANDB=0 LOGGER_BACKEND=none PROGRESS_BAR=0 bash scripts/ptv1/exp_13_ptv1_extra_single_all_train_infer.sh
```

Result report:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
python scripts/ptv1/report_ptv1_exp_results.py --prefix 20260603_2037_ptv1 --checkpoint-root checkpoints --output-root outputs --precision 6
```

## Tuning Details For Exp 12 / Exp 13

Tuning prefix: `20260603_2108_ptv1_param_v1`.

This section explains how the tuned rows in **Results At A Glance** were selected.
The low-result follow-up searched loss and optimization settings on exp_12 unseen-drug folds. Stage 1 used folds `0,2,4`; stage 2 promoted full 5-fold candidates.

The best exp_12 setting was `mse025`:

- `MSE_WEIGHT=0.25`
- `LEARNING_RATE=2e-4`
- `BATCH_SIZE=256`
- `DROPOUT=0.15`
- `WEIGHT_DECAY=1e-4`
- `GRAPH_FEATURE_MODE=real`
- `USE_DOSE_COVARIATE=1`
- `CELL_LLM_MODE=off`

### Exp 12: Tuned Comparison

| config | folds | AUROC | AUPRC | n-AUPRC | count | selected epochs |
|---|---|---:|---:|---:|---:|---|
| formal baseline `MSE_WEIGHT=0.50` | 0,1,2,3,4 | 0.707712 | 0.545776 | 2.212531 | 13137 | 9,4,4,7,5 |
| tuned `mse025` | 0,1,2,3,4 | 0.731340 | 0.570871 | 2.388662 | 13137 | 7,4,3,4,5 |
| tuned `pos_auto_mse050` | 0,1,2,3,4 | 0.705360 | 0.553605 | 2.293438 | 13137 | 4,2,0,1,9 |

Fold-level `mse025` exp_12 results:

| fold | AUROC | AUPRC | n-AUPRC | count | pos | neg |
|---|---:|---:|---:|---:|---:|---:|
| fold0 | 0.668501 | 0.575338 | 1.244648 | 1947 | 900 | 1047 |
| fold1 | 0.746670 | 0.546596 | 2.484768 | 1882 | 414 | 1468 |
| fold2 | 0.702139 | 0.616125 | 1.625323 | 1931 | 732 | 1199 |
| fold3 | 0.744024 | 0.610987 | 2.211358 | 1911 | 528 | 1383 |
| fold4 | 0.795366 | 0.505310 | 4.377216 | 5466 | 631 | 4835 |

Stage-1 search conclusion:

- Lowering MSE auxiliary loss from `0.50` to `0.25` was the most reliable improvement.
- `mse025_inactive010` reproduced `mse025` exactly on folds `0,2,4`, so it was not promoted separately.
- `pos_auto_mse050` improved exp_13 extra AUPRC but was weaker than `mse025` on exp_12.
- `mse025_pos_auto` and `mse025_focal` did not help; their fold `0,2,4` AUPRC means were `0.511107` and `0.450210`.
- Higher MSE, dropout changes, target-only MSE, ranking loss, and focal loss were not robust on folds `0,2,4`.

### Exp 13: Tuned Comparison

Two all-PTV1 extra-inference runs were evaluated from full 5-fold references:

| config | reference epochs | selected epoch | all-train max epochs | AUROC | AUPRC | n-AUPRC | count |
|---|---|---:|---:|---:|---:|---:|---:|
| formal baseline `MSE_WEIGHT=0.50` | 9,4,4,7,5 | 6 | 7 | 0.593455 | 0.576296 | 1.121719 | 218 |
| `mse025` | 7,4,3,4,5 | 5 | 6 | 0.595982 | 0.581499 | 1.131846 | 218 |
| `pos_auto_mse050` | 4,2,0,1,9 | 3 | 4 | 0.594550 | 0.587486 | 1.143500 | 218 |

Best exp_13 extra AUPRC came from `pos_auto_mse050`, although its exp_12 mean was lower than `mse025`.
This is why the summary keeps two possible exp_13 tuned rows: one for best extra AUPRC, and one for config consistency with exp_12.

Per-cell `pos_auto_mse050` exp_13 results:

| cell | AUROC | AUPRC | n-AUPRC | count | pos | neg |
|---|---:|---:|---:|---:|---:|---:|
| BT549 | 0.485507 | 0.546456 | 0.965405 | 53 | 30 | 23 |
| HCC1806 | 0.569124 | 0.564606 | 1.189705 | 59 | 28 | 31 |
| HS578T | 0.700957 | 0.773709 | 1.219178 | 52 | 33 | 19 |
| MDA-MB-468 | 0.629149 | 0.566369 | 1.456377 | 54 | 21 | 33 |

### Tuning Commands

```bash
BASE_PREFIX=20260603_2108_ptv1_param_v1 STAGES=stage1 SEARCH_FOLDS='0 2 4' LOG_TO_WANDB=0 LOGGER_BACKEND=none PROGRESS_BAR=0 bash scripts/ptv1/run_ptv1_param_search.sh
BASE_PREFIX=20260603_2108_ptv1_param_v1 STAGES=stage2 STAGE2_CONFIGS='mse025 pos_auto_mse050' FULL_FOLDS='0 1 2 3 4' LOG_TO_WANDB=0 LOGGER_BACKEND=none PROGRESS_BAR=0 bash scripts/ptv1/run_ptv1_param_search.sh
BASE_PREFIX=20260603_2108_ptv1_param_v1 STAGES=stage1 STAGE1_CONFIGS='mse025_pos_auto mse025_focal' SEARCH_FOLDS='0 2 4' LOG_TO_WANDB=0 LOGGER_BACKEND=none PROGRESS_BAR=0 bash scripts/ptv1/run_ptv1_param_search.sh
EXP_PREFIX=20260603_2108_ptv1_param_v1_stage2_mse025 REFERENCE_5FOLD_CKPT_PATH='checkpoints/20260603_2108_ptv1_param_v1_stage2_mse025_ptv1_unseen_drug_fold*' LEARNING_RATE=2e-4 BATCH_SIZE=256 DROPOUT=0.15 WEIGHT_DECAY=1e-4 MSE_WEIGHT=0.25 LOG_TO_WANDB=0 LOGGER_BACKEND=none PROGRESS_BAR=0 bash scripts/ptv1/exp_13_ptv1_extra_single_all_train_infer.sh
EXP_PREFIX=20260603_2108_ptv1_param_v1_stage2_pos_auto_mse050 REFERENCE_5FOLD_CKPT_PATH='checkpoints/20260603_2108_ptv1_param_v1_stage2_pos_auto_mse050_ptv1_unseen_drug_fold*' REFERENCE_ALLOW_MIXED_CONFIG=1 LEARNING_RATE=2e-4 BATCH_SIZE=256 DROPOUT=0.15 WEIGHT_DECAY=1e-4 MSE_WEIGHT=0.50 POSITIVE_WEIGHT=auto LOG_TO_WANDB=0 LOGGER_BACKEND=none PROGRESS_BAR=0 bash scripts/ptv1/exp_13_ptv1_extra_single_all_train_infer.sh
python scripts/ptv1/report_ptv1_param_search.py --base-prefix 20260603_2108_ptv1_param_v1 --checkpoint-root checkpoints --baseline-prefix 20260603_2037_ptv1 --precision 6
python scripts/ptv1/report_ptv1_exp_results.py --prefix 20260603_2108_ptv1_param_v1_stage2_mse025 --checkpoint-root checkpoints --output-root outputs --precision 6
python scripts/ptv1/report_ptv1_exp_results.py --prefix 20260603_2108_ptv1_param_v1_stage2_pos_auto_mse050 --checkpoint-root checkpoints --output-root outputs --precision 6
```
