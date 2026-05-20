# 2026-05-08 16:52 HKT Train Pipeline / Dimension / GPU Review

## Scope

Reviewed all files under `docs/`, then checked the current training stack against:

1. data process pipeline contracts;
2. `train.py`, `dataset/training_ready_dataset.py`, and the attention/graph implementation in `model/training_ready_models.py`;
3. metrics code in `model/training_ready_lightning.py`;
4. tensor and artifact dimensions used by the training path.

There is no standalone `attention.py` file in this checkout. The active attention implementation is in `model/training_ready_models.py`.

## Code Change

- Patched `model/training_ready_lightning.py::safe_mean` to return `NaN` cleanly when all metric values are non-finite.
- This removes `RuntimeWarning: Mean of empty slice` during small validation/test runs where AUROC/PCC-style metrics are not defined.

## Data Pipeline Checks

Commands run in `flow_v2`:

```bash
python utils/01_validate_standardized_outputs.py
python utils/03_validate_training_ready_outputs.py
python utils/09_build_data_splits.py --help
```

Results:

- Stage-1 standardized validation passed.
- Stage-2 training-ready validation passed.
- Split builder CLI loads correctly.

Key validated stage-2 shapes:

- `ptv3_main_singledrug`: processed / feature `18568 x 10982`
- `ptv3_main_doubledrug`: processed `2196 x 9202`, feature `20764 x 11092`
- `ptv3_extra_doubledrug_guomics`: processed / feature `9004 x 11092`
- `ptv3_extra_doubledrug_nc`: processed / feature `16412 x 11343`
- `ptv3_extra_doubledrug_nature`: processed / feature `23415 x 11343`
- `ptv1_aivc`: processed / feature `15002 x 5576`
- `ptv1_extra_singledrug`: processed / feature `186 x 5576`

## Dimension Checks

Checked current PTV3 global artifacts:

- `protein_index`: `11345`
- `pert_index`: `6113`
- protein embedding: `(11345, 1280)`
- drug embedding: `(6113, 2048)`
- PDI matrix: `(6113, 11345)` using `[drug, protein]`
- DDI matrix: `(6113, 6113)`
- PPI matrix: `(11345, 11345)`

Checked representative task artifacts:

- `ptv3_main_singledrug`: rows `18568`, expression `(18568, 10982)`, ordered proteins `10982`
- `ptv3_main_doubledrug`: rows `20764`, expression `(20764, 11092)`, ordered proteins `11092`
- `ptv3_extra_doubledrug_guomics`: rows `9004`, expression `(9004, 11092)`, ordered proteins `11092`

For all checked tasks, `expression_row_index` matched feature-table row order and max perturbation index was within the global drug embedding range.

## Train / Dataset / Attention Checks

Commands run:

```bash
python -m py_compile train.py dataset/training_ready_dataset.py model/training_ready_models.py model/training_ready_lightning.py infer.py scripts/0507_training_stack_smoke.py
python train.py --help
python scripts/0507_training_stack_smoke.py
```

Results:

- Compile checks passed.
- `train.py --help` loads correctly.
- Synthetic training stack smoke passed.
- Dataset emits the current no-group contract: one `control` row and one `perturb` row per item.
- Real-data dry run for `ptv3_main_doubledrug` returned:
  - expression `(1, 11092)`
  - response logits `(1, 1)`
  - synergy logits `(1, 1)`

## GPU Checks

Inside the default sandbox, CUDA was not visible and `nvidia-smi` could not communicate with the driver.

Unsandboxed GPU check succeeded:

- GPU: NVIDIA H200
- memory: `143771 MiB`
- no running GPU process at check time

GPU train smoke checks:

- `attention_v10_hetero_cls_ee`, default hidden/layer settings, `ptv3_main_doubledrug`, batch size `1`, one train batch + one validation batch:
  - GPU available and used: true
  - sanity validation passed
  - training batch passed
  - validation metric aggregation passed
  - checkpoint callback and run manifest flow completed

Baseline model check:

- `baseline_emb_v3`, `ptv3_main_doubledrug`, batch size `1`, one train batch + one validation batch on CPU passed.
- This confirms the default external Geneformer embedding path exists and covers the current feature row indices.

## Metrics Review

The metric stack supports:

- finite-mask expression MSE;
- response AUROC/AUPRC/ACC;
- synergy AUROC/AUPRC/ACC;
- expression MSE/MAE/PCC/R2/direction metrics;
- top-50 metrics;
- delta metrics;
- MMD and energy distance.

Expected caveat: one-sample or one-class validation slices still produce `NaN` for undefined metrics such as AUROC/AUPRC/MMD. After the `safe_mean` patch these undefined values are reported without warning noise.

## Remaining Caveats

1. This review verified one-batch trainability, not a full multi-epoch training run.
2. Full-size graph training is viable on the visible H200 for batch size `1`; larger batch sizes still need memory monitoring because the model uses full attention over about `11k` protein tokens.
3. `baseline_emb_v3` still relies on the external Geneformer embedding file row order matching the current feature table. The smoke test checks availability and index coverage, not semantic row-order provenance.
4. The sandbox hides CUDA devices; GPU training checks require unsandboxed execution.
