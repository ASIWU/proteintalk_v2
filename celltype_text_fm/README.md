# Cell-Type Text Foundation Experiment

This folder keeps the cell-type semantic embedding experiment independent from
the root `fast_delta` training code.

The experiment uses SapBERT to turn each `cell_type` label into a frozen
biomedical entity embedding. The row-level embedding is passed through the
existing fast model `prior_features` channel, so it is fused as an extra
cell-type semantic feature without editing the baseline model files.

Default smoke run:

```bash
source /mnt/shared-storage-user/wuhao/miniconda3/etc/profile.d/conda.sh
conda activate flow_v2
CUDA_VISIBLE_DEVICES=0 python celltype_text_fm/train_text_celltype.py \
  --split-strategy cell_5fold_fold2 \
  --dry-run-batches 1
```

Two-GPU bottleneck run:

```bash
EXP_PREFIX=20260525_celltype_text_v1 bash celltype_text_fm/run_bottleneck_2gpu.sh
```

The generated `.npz` embedding cache is intentionally ignored by git.
