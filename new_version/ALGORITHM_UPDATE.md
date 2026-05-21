# new_version algorithm iteration report

Date: 2026-05-20

## Scope

All implementation changes are under `new_version/`. Existing source files,
data artifacts, and original experiment scripts were not edited.

Read before implementation and experiment design:

- `docs/Data_Process_1.md`
- `docs/Data_Process_2.md`
- `docs/Data_Process_3.md`
- `docs/Data_Process_4.md`
- `docs/Training_gudline.md`
- `scripts/README_ptv3_experiments.md`

## Problem

The current graph Transformer path uses the full protein expression vector with
large graph-derived drug/protein features in the hot training path. For
`ptv3_main_singledrug`, the training-ready expression matrix is
`(18568, 10982)`, while the derived interaction matrices are also large:

- PDI: `(6113, 11345)`
- PPI: `(11345, 11345)`
- DDI: `(6113, 6113)`

The target task is unseen single-drug 5-fold performance, with a concrete goal
of improving efficiency enough to run meaningful iteration on the current
two-GPU server before requiring the standard 8-GPU experiment.

## New algorithm

The new model is `FastDeltaDrugResponseModel`.

Main changes:

- Keep the full protein expression input/output axis. No training-ready data
  truncation is introduced.
- Replace protein-token Transformer attention with a low-rank full-vector
  encoder/decoder. The main expression cost becomes `O(G * hidden)` instead of
  all-protein token attention.
- Predict perturbation proteome as `control_expression + delta`, initialized
  with a small learnable delta scale.
- Use raw Morgan drug embeddings for drug identity, avoiding a pure trainable
  drug ID table and improving cold-start behavior for unseen drugs.
- Use `target_protein_list` to aggregate fixed ESM protein embeddings as a
  compact target-protein signal.
- Explicitly use PPI + PDI + DDI through compressed graph features. The full
  matrices are read once and converted into a small drug-indexed cache, so they
  participate in the model without restoring heavy graph message passing.
- Load expression arrays with `np.load(..., mmap_mode="r")`, reducing per-process
  memory copying under DDP.
- Validation/test collect classification outputs by default, avoiding full
  expression prediction accumulation unless needed.

The graph feature cache is built as:

- `PDI @ projected_protein_embedding`: direct drug-target context.
- `PDI @ normalized_PPI @ projected_protein_embedding`: PPI-propagated target
  neighborhood context.
- `DDI @ projected_drug_embedding`: DDI drug-neighborhood context.
- Optional structural random projections of normalized PDI, PDI-PPI, and DDI
  rows.
- PDI/DDI row statistics: row sum, nonzero count, max value.

For `graph_feature_dim=128`, the baseline3 per-drug feature has shape
`(6113, 390)`. The boosted graph setting uses structural random projections and
has shape `(6113, 774)`. Both caches are stored under
`new_version/graph_cache/`.

The default recommended single-drug config is now:

```bash
--mse-weight 0.25 \
--positive-weight none \
--label-smoothing 0.0 \
--target-protein-max-length 32 \
--graph-feature-mode real \
--graph-feature-dim 128 \
--graph-structural-rp \
--graph-drug-concat \
--graph-logit-scale 2.0
```

This is encoded as the default in `new_version/run_single_unseen_5fold.sh`.

## New files

- `new_version/training_ready_fast_dataset.py`: memmap dataset and split loading.
- `new_version/graph_feature_utils.py`: compressed PPI/PDI/DDI feature builder.
- `new_version/fast_delta_model.py`: low-rank delta model definition.
- `new_version/fast_lightning.py`: Lightning training/loss/metrics wrapper.
- `new_version/train.py`: new training entrypoint.
- `new_version/run_single_unseen_5fold.sh`: default two-GPU unseen single-drug
  5-fold launcher.
- `new_version/run_single_unseen_sweep.sh`: method sweep launcher.
- `new_version/summarize_runs.py`: manifest-to-TSV experiment summarizer.
- `new_version/.gitignore`: ignores new_version experiment outputs/caches.

## Verification

Environment:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
```

Static check:

```bash
python -m py_compile new_version/training_ready_fast_dataset.py \
  new_version/graph_feature_utils.py \
  new_version/fast_delta_model.py \
  new_version/fast_lightning.py \
  new_version/train.py \
  new_version/summarize_runs.py
```

Two-GPU smoke test passed with DDP, bf16 mixed precision, limited train/valid/test
batches.

## Full two-GPU experiment matrix

All runs below used:

- GPUs: `CUDA_VISIBLE_DEVICES=0,1`
- devices: `2`
- precision: `bf16-mixed`
- batch size: `128`
- max epochs: `50`
- checkpoint selection: best validation `val/task_auprc`
- evaluation: test split from the selected best checkpoint

Completed full runs:

- `pert_stratified_5fold_fold*`: 12 methods x 5 folds = 60 complete train/valid/test runs.
- `pert_id_5fold_fold*`: 2 methods x 5 folds = 10 complete train/valid/test runs.
- graph-feature iterations on `pert_stratified_5fold_fold*`: 14 methods x 5 folds
  = 70 complete train/valid/test runs.
- Total new-model full runs: 140 runs, 7000 training epochs.

Raw summary files:

- `new_version/runtime_logs/20260520_fullsweep_v1/summary.tsv`
- `new_version/runtime_logs/20260520_fullsweep_v2/summary.tsv`
- `new_version/runtime_logs/20260520_pertid_bestcheck/summary.tsv`
- `new_version/runtime_logs/20260520_graphsweep_v1/summary.tsv`
- `new_version/runtime_logs/20260520_graphsweep_v2/summary.tsv`
- `new_version/runtime_logs/20260520_graphboost_v1/summary.tsv`
- `new_version/runtime_logs/20260520_graphboost_v2/summary.tsv`
- `new_version/runtime_logs/20260520_graphboost_v3/summary.tsv`

## Results: pert_stratified_5fold

Mean test metrics over five folds:

| method | AUPRC | AUROC | ACC | fit seconds/fold |
| --- | ---: | ---: | ---: | ---: |
| default: mse0.25 + positive auto | 0.5450 | 0.8430 | 0.8949 | 41.6 |
| cls_only + positive auto | 0.5390 | 0.8369 | 0.8962 | 37.2 |
| low_mse0.05 + positive auto | 0.5364 | 0.8305 | 0.8946 | 42.0 |
| target64 + positive auto | 0.5518 | 0.8450 | 0.8967 | 39.8 |
| no_pos | 0.6004 | 0.8306 | 0.9156 | 40.6 |
| cls_no_pos | 0.5803 | 0.8415 | 0.9143 | 39.0 |
| mse010_no_pos | 0.5872 | 0.8417 | 0.9134 | 38.0 |
| low_mse_no_pos | 0.5898 | 0.8427 | 0.9143 | 37.0 |
| target64_no_pos | 0.5883 | 0.8364 | 0.9130 | 39.8 |
| smooth05_no_pos | 0.6005 | 0.8344 | 0.9127 | 41.4 |
| dropout25_no_pos | 0.5840 | 0.8329 | 0.9124 | 38.8 |
| hidden512_no_pos | 0.5966 | 0.8547 | 0.9133 | 41.0 |

Main observations:

- Removing positive-class reweighting is the dominant improvement for unseen
  single-drug test AUPRC.
- `smooth05_no_pos` is numerically the highest mean AUPRC on
  `pert_stratified_5fold`, but the margin over `no_pos` is only `0.00007`.
- `hidden512_no_pos` gives the best AUROC, but not the best AUPRC.
- Increasing target protein length from 32 to 64 did not help AUPRC.
- Classification-only training is efficient but weaker than keeping the small
  MSE auxiliary loss.

## Results: graph feature iteration

The no-graph `no_pos` model above became the baseline for graph-feature
iteration. All graph runs explicitly used PPI + PDI + DDI through the compressed
cache described above.

Mean test metrics over five folds:

| method | graph feature | AUPRC | AUROC | ACC | fit seconds/fold |
| --- | --- | ---: | ---: | ---: | ---: |
| no_graph_no_pos baseline | off | 0.6004 | 0.8306 | 0.9156 | 40.6 |
| graph64_zero_no_pos | zero ablation | 0.5982 | 0.8417 | 0.9157 | 41.4 |
| graph64_no_pos | real, dim64, scale0.1 | 0.6014 | 0.8450 | 0.9146 | 40.8 |
| graph64_scale025_no_pos | real, dim64, scale0.25 | 0.6035 | 0.8446 | 0.9141 | 42.2 |
| graph64_scale05_no_pos | real, dim64, scale0.5 | 0.6004 | 0.8463 | 0.9158 | 40.0 |
| graph128_zero_no_pos | zero ablation | 0.5919 | 0.8404 | 0.9133 | 40.8 |
| graph128_no_pos | real, dim128, scale0.1 | 0.6025 | 0.8515 | 0.9136 | 41.2 |
| graph128_struct_no_pos | structural RP | 0.5725 | 0.8296 | 0.9122 | 40.8 |
| graph128_struct_zero_no_pos | structural zero ablation | 0.5867 | 0.8448 | 0.9122 | 38.6 |
| graph128_struct_drugcat_logit05 | structural + drug concat + logit0.5 | 0.6372 | 0.8951 | 0.9133 | 41.6 |
| graph128_struct_drugcat_logit05_zero | zero ablation | 0.5982 | 0.8510 | 0.9130 | 43.2 |
| graph128_struct_drugcat_logit1 | structural + drug concat + logit1.0 | 0.6422 | 0.8945 | 0.9180 | 41.2 |
| graph128_struct_drugcat_logit1_zero | zero ablation | 0.6004 | 0.8488 | 0.9138 | 40.4 |
| graph128_struct_drugcat_logit2 | structural + drug concat + logit2.0 | 0.6564 | 0.9003 | 0.9174 | 41.0 |
| graph128_struct_drugcat_logit2_zero | zero ablation | 0.5988 | 0.8493 | 0.9136 | 41.4 |

Graph ablation conclusions:

- `graph128_no_pos` vs `graph128_zero_no_pos`: AUPRC `+0.01064`, AUROC
  `+0.01106`. This is the cleanest without-graph ablation evidence that the
  PPI/PDI/DDI feature itself contributes useful signal.
- `graph128_no_pos` is retained as baseline3: it is the first explicit
  PPI/PDI/DDI graph-feature baseline.
- Structural random projection alone was not useful: `graph128_struct_no_pos`
  underperformed its zero ablation. This variant is rejected.
- Strong graph utilization is useful when graph features are fused into the drug
  encoder and directly into the response logit. The best setting is
  `graph128_struct_drugcat_logit2`.
- `graph128_struct_drugcat_logit2` vs its zero ablation: AUPRC `+0.05757`
  absolute (`+9.61%` relative to zero), AUROC `+0.05103`. This meets the
  requested `>5%` graph-ablation target even under an absolute five-point
  interpretation.
- Graph feature compression did not hurt efficiency. The selected boosted graph
  model still finishes a full 50-epoch fold in about `40-42` seconds on two GPUs.

## Comparison to 8-GPU reference

User-provided standard 8-GPU single unseen-drug 5-fold result:

- mean AUROC: `0.81438`
- mean AUPRC: `0.55630`
- fold AUPRC: `0.48825, 0.53805, 0.63009, 0.48014, 0.64499`

Selected boosted graph two-GPU result
`graph128_struct_drugcat_logit2_no_pos` on the same single unseen-drug 5-fold
setting:

| fold | AUPRC | AUROC | ACC | full seconds |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.5616 | 0.8418 | 0.9096 | 42 |
| 1 | 0.7659 | 0.9349 | 0.9362 | 41 |
| 2 | 0.5579 | 0.8915 | 0.8943 | 41 |
| 3 | 0.7354 | 0.9192 | 0.9391 | 41 |
| 4 | 0.6610 | 0.9142 | 0.9078 | 40 |
| mean | 0.6564 | 0.9003 | 0.9174 | 41 |

Delta against the 8-GPU reference:

- AUPRC: `+0.10007`, or `+10.01` points.
- AUROC: `+0.08594`, or `+8.59` points.

Both AUPRC and AUROC exceed the requested three-point improvement threshold.
The model also has clear efficiency gain: it uses two GPUs and finishes a full
fold in about `38-44` seconds, including test from the best checkpoint.

## Results: pert_id_5fold confirmation

The best stratified candidates were rechecked on `pert_id_5fold_fold*`.

| method | AUPRC | AUROC | ACC | fit seconds/fold |
| --- | ---: | ---: | ---: | ---: |
| no_pos | 0.5547 | 0.8421 | 0.9108 | 41.6 |
| smooth05_no_pos | 0.5346 | 0.8312 | 0.9101 | 36.4 |

This confirms `no_pos` as the more robust default. Label smoothing was not
retained as the standard recommendation because it hurt the second unseen-drug
split variant.

## Original-path timing probe

A one-epoch run of the original graph Transformer path was executed on the same
two GPUs with batch size 16:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --dataset-group ptv3 \
  --task-name ptv3_main_singledrug \
  --split-strategy pert_stratified_5fold_fold0 \
  --experiment-name original_graph_2gpu_full1epoch_b16 \
  --model-type attention_v10_hetero_cls_ee \
  --batch-size 16 \
  --max-epochs 1 \
  --accelerator gpu \
  --devices 2 \
  --strategy ddp_find_unused_parameters_true \
  --precision bf16-mixed \
  --num-workers 4 \
  --logger-backend none \
  --save-top-k 0 \
  --no-save-last-ckpt \
  --skip-test \
  --allow-nonfinite-monitor
```

Observed original-path timing:

- 385 train steps for one epoch.
- Training epoch finished in about `1:47`, around `3.57` steps/s at epoch end.
- Validation had 66 batches, around `15.35` batches/s.

Observed new-path timing:

- 49 train steps per epoch at batch size 128.
- Later epochs commonly reached about `70-85` steps/s.
- A full 50-epoch fold, including validation and test from the best checkpoint,
  usually completed in about `35-44` seconds.

Interpretation:

- On the same two GPUs, the new model completes a full 50-epoch fold faster than
  the original graph path completes one training epoch.
- Comparing 50 epochs to 50 epochs, the measured wall-clock speedup is roughly
  two orders of magnitude. The exact ratio is not a strict apples-to-apples
  model-quality comparison because the original probe used batch size 16 while
  the new path uses batch size 128.

## Independent graph-feature audit

On 2026-05-21, an independent subagent reviewed the graph-feature implementation,
run manifests, split summaries, graph cache metadata, and real-vs-zero ablation.
The audit found no evidence of fabricated metrics, test-selected checkpoints, or
direct label leakage.

The audit confirmed:

- Real graph runs build the cached graph matrix from PPI, PDI, and DDI sources.
- Zero graph ablation keeps the same architecture but feeds zero graph tensors
  with `graph_feature_mask=0`.
- Checked train/valid/test row overlap, sample-id overlap, and perturbation-id
  train-test overlap are all zero.
- The graph cache contains `pdi_direct`, `pdi_ppi`, `ddi_context`,
  `pdi_struct`, `pdi_ppi_struct`, `ddi_struct`, `pdi_stats`, and `ddi_stats`.

The original distributed 5-fold summary is:

- real graph: AUPRC `0.656369`, AUROC `0.900316`, ACC `0.917409`
- zero graph: AUPRC `0.598797`, AUROC `0.849282`, ACC `0.913561`
- gap: AUPRC `+0.057573`, AUROC `+0.051034`, ACC `+0.003848`

A single-GPU reevaluation of the saved best checkpoints was then run to avoid
DDP evaluation padding. It produced:

- real graph: AUPRC `0.656371`, AUROC `0.900312`, ACC `0.917402`
- zero graph: AUPRC `0.598797`, AUROC `0.849261`, ACC `0.913553`
- gap: AUPRC `+0.057574`, AUROC `+0.051051`, ACC `+0.003849`

Important reporting caveat: the graph features are global/transductive graph
features. Held-out perturbation nodes still have PPI/PDI/DDI-derived features.
This is not label leakage, but the final paper should describe the setting as
graph-assisted/transductive unseen-drug generalization unless the benchmark
definition explicitly permits known test-drug graph features.

## 2026-05-21 graph-jump and protein-concat iterations

Two extra architecture families were implemented and tested after baseline3:

- `PCEP` (`--protein-concat-mode pcep`): a low-cost protein-conditioned
  expression pooling module. It uses the ordered full expression axis and a
  fixed random projection of each protein embedding to compute context-dependent
  protein gates, then adds the pooled protein-expression hidden state to the
  control-expression branch. This restores a lightweight per-protein
  expression/protein-embedding interaction without returning to full protein
  token self-attention.
- `graph multihop / selective jump`: optional multi-hop graph feature cache
  blocks were added for `PDI-PPI^2`, `DDI^2`, `DDI-PDI`, and `DDI-PDI-PPI`
  contexts, plus matching structural random-projection blocks. A selective
  fusion mode can encode each graph block separately and combine them with
  context-conditioned softmax or sparsemax gates.

Implementation paths:

- `new_version/graph_feature_utils.py`
- `new_version/fast_delta_model.py`
- `new_version/train.py`
- `new_version/run_single_unseen_sweep.sh`
- `new_version/run_single_unseen_5fold.sh`

Smoke tests:

- Python compile passed for the changed modules.
- Single-GPU dry run passed for `graph-multihop + selective sparse jump + PCEP`.
- Two-GPU DDP smoke fit/test passed for the same configuration.

Full 5-fold experiments were run on two H200 GPUs with 50 epochs, batch size 128,
and matched real/zero graph ablations. Combined summary:

| Method | AUPRC | AUROC | ACC | AUPRC gap vs zero | Mean fit sec/fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline3 `graph128_struct_drugcat_logit2_no_pos` | 0.656369 | 0.900316 | 0.917409 | +0.057573 | 41.0 |
| baseline3 zero | 0.598797 | 0.849282 | 0.913561 | - | 41.4 |
| selective multihop sparse jump | 0.619492 | 0.894001 | 0.907998 | +0.011594 | 63.0 |
| selective multihop sparse jump zero | 0.607898 | 0.839606 | 0.915338 | - | 61.6 |
| selective multihop sparse jump + PCEP | 0.623436 | 0.888482 | 0.908126 | +0.050996 | 66.0 |
| selective multihop sparse jump + PCEP zero | 0.572440 | 0.839290 | 0.911943 | - | 64.4 |
| baseline3 + PCEP | 0.651715 | 0.899814 | 0.911120 | +0.059653 | 46.8 |
| baseline3 + PCEP zero | 0.592063 | 0.838600 | 0.913877 | - | 45.8 |
| multihop concat | 0.612670 | 0.888461 | 0.907267 | +0.020400 | 40.6 |
| multihop concat zero | 0.592269 | 0.835043 | 0.914503 | - | 41.6 |

Interpretation:

- Baseline3 remains the best default for effect and efficiency.
- `baseline3 + PCEP` is a reasonable interpretability-oriented ablation: it
  keeps a graph ablation gap above five AUPRC points and only adds about six
  seconds per fold, but its real-graph AUPRC is slightly below baseline3.
- The tested multi-hop graph extensions did not improve the main metric. They
  make the graph claim stronger methodologically, but current evidence does not
  support using them as the standard baseline.
- Result table saved at
  `new_version/runtime_logs/20260521_graphjump_combined_summary.tsv`.

Additional PCEP GPU timing check:

- Two-GPU DDP PCEP with `batch_size=128` per GPU averaged `46.8` seconds per
  fold across five folds.
- One-GPU PCEP with `batch_size=128` took `63` seconds on fold0 because it runs
  about twice as many optimizer steps per epoch.
- One-GPU PCEP with `batch_size=256` took `40` seconds on fold0 and matched the
  two-GPU global batch size more closely.

Practical recommendation: for this small fast model on H200, use one GPU with
`batch_size=256` for routine PCEP/baseline3 iterations when memory is available.
Use two GPUs only when keeping the exact existing two-GPU recipe is more
important than GPU efficiency, or when running multiple folds in parallel by
assigning one fold per GPU.

## 2026-05-21 baseline4 single-GPU ablation

Based on the timing check above, `baseline3 + PCEP` was promoted to
`baseline4` for a matched single-GPU recipe:

- one GPU per run (`--devices 1`);
- `batch_size=256`, matching the previous two-GPU global batch size;
- PPI + PDI + DDI compressed graph feature enabled;
- PCEP enabled with `protein_concat_dim=64` and `protein_concat_topk=512`;
- `mse_weight=0.25`, `positive_weight=none`, target protein length 32.

A new launcher was added:

- `new_version/run_baseline4_1gpu_parallel.sh`

It runs one single-GPU experiment per process and assigns jobs across
`GPU_IDS=0,1`, so the two H200 GPUs are used as two independent workers rather
than as DDP ranks.

Full `ptv3_main_singledrug` `pert_stratified_5fold` results at
`gpu=1, batch_size=256, max_epochs=50`:

| method | AUPRC | AUROC | ACC | fit seconds/fold |
| --- | ---: | ---: | ---: | ---: |
| baseline1 `no_pos` | 0.600407 | 0.830623 | 0.915606 | 40.6 |
| baseline2 `graph128_no_pos` | 0.602530 | 0.851481 | 0.913622 | 41.2 |
| baseline3 `graph128_struct_drugcat_logit2_no_pos` | 0.656369 | 0.900316 | 0.917409 | 41.0 |
| baseline4 `baseline3 + PCEP`, `mse_weight=0.25` | 0.666491 | 0.903489 | 0.915167 | 41.8 |
| baseline4 w/o graph feature | 0.561250 | 0.835845 | 0.912883 | 42.0 |
| baseline4 w/o MSE loss | 0.644541 | 0.888019 | 0.913273 | 41.2 |

Baseline4 ablation conclusions:

- Graph feature contribution is strong under the final single-GPU setting:
  baseline4 minus w/o graph is `+0.105241` AUPRC and `+0.067644` AUROC.
- The MSE auxiliary loss remains useful: baseline4 minus w/o MSE is
  `+0.021950` AUPRC and `+0.015470` AUROC.
- Against baseline3, baseline4 improves AUPRC by `+0.010122` and AUROC by
  `+0.003173` while using one GPU per run.
- Against the user-provided 8-GPU reference, baseline4 improves AUPRC by
  `+0.110191` and AUROC by `+0.089109`.

An MSE-weight sweep was also run under the same single-GPU baseline4 recipe,
with real graph features enabled:

| mse_weight | AUPRC | AUROC | ACC | fit seconds/fold |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.649057 | 0.887431 | 0.915543 | 42.4 |
| 0.10 | 0.656212 | 0.898468 | 0.916110 | 42.4 |
| 0.25 | 0.666491 | 0.903489 | 0.915167 | 41.8 |
| 0.50 | 0.658686 | 0.893493 | 0.912557 | 41.4 |

`mse_weight=0.25` is retained for baseline4 because it has the best 5-fold mean
AUPRC and AUROC among the tested values.

## Recommendation

Use baseline4 as the default unseen single-drug recipe when one-GPU-per-fold
parallelism is acceptable:

```bash
EXP_PREFIX=baseline4_b256 \
GPU_IDS=0,1 \
METHODS="baseline4 baseline4_zero baseline4_no_mse" \
BATCH_SIZE=256 \
MAX_EPOCHS=50 \
bash new_version/run_baseline4_1gpu_parallel.sh
```

Use `graph128_struct_drugcat_logit2_no_pos` as the fallback recipe when matching
the earlier two-GPU DDP baseline3 exactly:

```bash
GPU_IDS=0,1 \
DEVICES=2 \
FOLDS="0 1 2 3 4" \
MAX_EPOCHS=50 \
BATCH_SIZE=128 \
LEARNING_RATE=3e-4 \
GRAPH_FEATURE_MODE=real \
GRAPH_FEATURE_DIM=128 \
GRAPH_STRUCTURAL_RP=1 \
GRAPH_DRUG_CONCAT=1 \
GRAPH_LOGIT_SCALE=2.0 \
bash new_version/run_single_unseen_5fold.sh
```

For the matched without-graph ablation:

```bash
METHODS="graph128_struct_drugcat_logit2_no_pos graph128_struct_drugcat_logit2_zero_no_pos" \
EXP_PREFIX=graphboost_confirm \
bash new_version/run_single_unseen_sweep.sh
```

Current conclusion: baseline4 is the strongest tested unseen single-drug
configuration. It explicitly uses PPI + PDI + DDI, keeps PCEP for lightweight
per-protein expression/embedding interaction, has a large graph ablation gap,
and runs efficiently with one GPU per fold. The 8-GPU run is still useful only
if the final report must match the user's standard execution policy.
