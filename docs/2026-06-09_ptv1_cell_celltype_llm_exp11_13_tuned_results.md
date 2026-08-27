# 2026-06-09 PTV1 Graph + Cell/Cell-Type LLM exp11-13 Tuning Results

## Summary

This is the completed PTV1 migration and tuning run for the graph + frozen Cell LLM + frozen `cell_type` LLM architecture. The run uses the newly generated API-based PTV1 `cell_type` embedding artifact and evaluates exp11, exp12, and exp13 valid/oracle transfer.

- Base prefix: `20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype`
- Generated report: `logs/20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype_cell_celltype_tune_results.md`
- Generated TSV: `outputs/2026-06/2026-06-09/20260609_ptv1_cell_celltype_llm_tune_v2_api_celltype_cell_celltype_tune_results.tsv`
- Dataset group: `ptv1`
- Required policy: graph enabled, Cell LLM frozen, `cell_type` LLM frozen
- Candidates: `13`
- exp12 completed folds: `65/65`
- exp13 scored rows: `218` for every valid and oracle result

Final selections:

| target | selection rule | selected candidate | AUPRC | AUROC | nAUPRC | epoch/folds | notes |
|---|---|---|---:|---:|---:|---|---|
| exp11 | random-split valid AUPRC | `mse075_drop010` | 0.914806 | 0.955703 | 3.123631 | epoch 3 | formal exp11 selection |
| exp12 | unseen-drug 5-fold mean AUPRC | `mse075_drop010` | 0.624837 | 0.760972 | 2.663382 | folds 0-4 | formal exp12 selection |
| exp13 valid | exp11 validation-selected checkpoint inferred on extra | `mse050_target_pdi` | 0.601089 | 0.627653 | 1.169976 | rows 218 | best direct transfer among exp11 candidates |
| exp13 oracle | best extra AUPRC across saved exp11 epoch checkpoints | `mse050_lr1e4` | 0.636390 | 0.652502 | 1.238688 | epoch 16, rows 218 | diagnostic only; uses exp13 labels for checkpoint choice |

The exp13 oracle is not a model-selection rule. It answers the diagnostic question: "if every saved exp11 checkpoint is evaluated on exp13, what is the best exp13 AUPRC and AUROC?"

## Data And Embeddings

PTV1 has Cell labels and a coarse breast cancer `cell_type` label. The training-ready PTV1 rows assign real rows to `cell_type=BREAST` and keep `no=0` as the reserved missing row.

Training-ready audit:

| task | feature rows | expression shape | relevant labels | metric rows |
|---|---:|---|---|---:|
| `ptv1_aivc` | 15002 | `(15002, 5576)` | Cell indices 1-19; `cell_type=BREAST` | exp11/12 splits |
| `ptv1_extra_singledrug` | 222 | `(222, 5576)` | Cell indices 2,7,11,18; `cell_type=BREAST` | 218 |

The extra feature table includes 218 labeled self rows plus 4 matched-control rows. exp13 metrics are computed on the 218 labeled self rows.

Embedding artifacts:

| artifact | shape | aligned index | validation |
|---|---:|---|---|
| `data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz` | `(20, 4096)` | `Cell_index` | row 0 zero; rows 1-19 finite/nonzero; exp13 uses Cell indices 2,7,11,18 |
| `data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz` | `(2, 4096)` | `cell_type_index` | row 0 zero; row 1 `BREAST` finite/nonzero and normalized |

The `cell_type` artifact was generated through the repository `.env` API configuration after enabling `proxy_on2`. Metadata:

- description model: `gpt-5.4`
- embedding model: `Qwen/Qwen3-Embedding-8B`
- embedding dim: `4096`
- normalized: `true`
- sidecar: `data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.json`

The Python client removed SOCKS `ALL_PROXY` during the call because `socksio` is unavailable, while preserving the HTTP(S) proxy exported by `proxy_on2`.

## Prompt Policy

The implemented PTV1 `cell_type` prompt asks for one compact biomedical description of a coarse cancer tissue or lineage label used in PTV1 breast-cancer drug-response proteomics. It requests:

- breast cancer lineage;
- common molecular context;
- why the tissue label may affect drug perturbation response;
- 2-4 sentences, no bullets;
- no specific cell-line names.

For PTV1, the only nonzero `cell_type` label is `BREAST`, so the final embedding matrix has the reserved `no` zero vector and one generated `BREAST` vector.

## Run Policy

All candidates used:

- `GRAPH_FEATURE_MODE=real`
- `GRAPH_STRUCTURAL_RP=1`
- `GRAPH_DRUG_CONCAT=1`
- `GRAPH_LOGIT_SCALE=2.0`
- `CELL_LLM_MODE=frozen`
- `CELL_LLM_EMBEDDING_PATH=data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz`
- `CELL_TYPE_LLM_MODE=frozen`
- `CELL_TYPE_LLM_EMBEDDING_PATH=data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz`
- dose covariates `pert_dose1 pert_dose2`

No graph-off candidate is included in this run or report.

## Candidate Results

Rows are sorted by exp11 validation AUPRC, matching the generated report. `exp13 valid` is direct transfer from the exp11 validation-selected checkpoint. `exp13 oracle` is the best extra-test AUPRC over saved exp11 epoch checkpoints.

| candidate | exp11 AUPRC/AUROC (epoch) | exp12 mean AUPRC/AUROC | exp13 valid AUPRC/AUROC | exp13 oracle AUPRC/AUROC (epoch) | key params |
|---|---:|---:|---:|---:|---|
| `mse075_drop010` | 0.914806 / 0.955703 (3) | 0.624837 / 0.760972 | 0.594929 / 0.608701 | 0.603497 / 0.625337 (43) | LR 2e-4, dropout 0.10, MSE 0.75, target all |
| `mse050_target_ppi` | 0.913594 / 0.956849 (8) | 0.582255 / 0.724622 | 0.554254 / 0.583263 | 0.610779 / 0.623568 (35) | LR 2e-4, dropout 0.15, MSE 0.50, target pdi_ppi |
| `mse025_drop010` | 0.912541 / 0.954455 (3) | 0.607178 / 0.736127 | 0.568297 / 0.596572 | 0.624805 / 0.624537 (13) | LR 2e-4, dropout 0.10, MSE 0.25, target all |
| `mse050_lr1e4` | 0.911542 / 0.953449 (9) | 0.545716 / 0.690694 | 0.589180 / 0.602300 | 0.636390 / 0.652502 (16) | LR 1e-4, dropout 0.15, MSE 0.50, target all |
| `mse075` | 0.909997 / 0.955298 (3) | 0.609285 / 0.750467 | 0.589721 / 0.608112 | 0.623006 / 0.635234 (43) | LR 2e-4, dropout 0.15, MSE 0.75, target all |
| `focal_mse050` | 0.906650 / 0.949274 (6) | 0.598775 / 0.757722 | 0.584374 / 0.620999 | 0.628615 / 0.656124 (24) | LR 2e-4, dropout 0.15, MSE 0.50, focal |
| `pos_auto_mse050` | 0.895472 / 0.948408 (9) | 0.554776 / 0.712706 | 0.570109 / 0.587643 | 0.634038 / 0.638772 (24) | LR 2e-4, dropout 0.15, MSE 0.50, positive weight auto |
| `baseline_mse050` | 0.894591 / 0.948631 (10) | 0.561074 / 0.729005 | 0.568359 / 0.599014 | 0.608345 / 0.625042 (25) | LR 2e-4, dropout 0.15, MSE 0.50, target all |
| `mse050_lr3e4` | 0.893222 / 0.944168 (7) | 0.578748 / 0.737705 | 0.567037 / 0.601541 | 0.600804 / 0.608027 (39) | LR 3e-4, dropout 0.15, MSE 0.50, target all |
| `mse050_label_smooth005` | 0.887296 / 0.947939 (4) | 0.562418 / 0.737836 | 0.575820 / 0.593455 | 0.619319 / 0.615103 (29) | LR 2e-4, dropout 0.15, MSE 0.50, label smoothing 0.05 |
| `mse050_target_pdi` | 0.883922 / 0.939407 (16) | 0.581198 / 0.717900 | 0.601089 / 0.627653 | 0.614654 / 0.618725 (23) | LR 2e-4, dropout 0.15, MSE 0.50, target pdi |
| `mse050_drop010` | 0.878462 / 0.943994 (2) | 0.582352 / 0.720315 | 0.594112 / 0.608196 | 0.610921 / 0.630264 (18) | LR 2e-4, dropout 0.10, MSE 0.50, target all |
| `mse025` | 0.817553 / 0.919685 (10) | 0.582385 / 0.729455 | 0.577579 / 0.616872 | 0.614181 / 0.609880 (21) | LR 2e-4, dropout 0.15, MSE 0.25, target all |

## Interpretation

The same candidate now wins exp11 and exp12:

- exp11 best is `mse075_drop010`, with random-split AUPRC/AUROC `0.914806 / 0.955703`.
- exp12 best is also `mse075_drop010`, with unseen-drug 5-fold mean AUPRC/AUROC `0.624837 / 0.760972`.

exp13 transfer ranks candidates differently:

- The exp11-selected `mse075_drop010` transfers to exp13 at AUPRC/AUROC `0.594929 / 0.608701`.
- The best exp13 valid-transfer row is `mse050_target_pdi` at `0.601089 / 0.627653`.
- The all-checkpoint exp13 oracle diagnostic is `mse050_lr1e4` at `0.636390 / 0.652502`, selected from epoch 16.

The oracle result is useful for understanding checkpoint sensitivity, but it should not replace exp11 validation selection because it uses exp13 labels to choose the checkpoint.

## Validation

Completed checks:

- `python utils/ptv1/07_build_ptv1_cell_llm_embeddings.py --dataset-group ptv1 --output data/training_ready/ptv1/derived/cell_llm_embedding_qwen3_4096_v2.npz --validate-only`
- `python utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py --dataset-group ptv1 --output data/training_ready/ptv1/derived/cell_type_llm_embedding_qwen3_4096_v3.npz --validate-only`
- `bash -n scripts/ptv1/*.sh scripts/ptv3_experiment_common.sh`
- `python -m py_compile train.py infer.py utils/ptv1/08_build_ptv1_cell_type_llm_embeddings.py scripts/ptv1/report_ptv1_cell_celltype_llm_tune_results.py`
- structured TSV audit: 13 candidates, 65 exp12 folds, policy ok for all candidates, exp13 valid rows 218, exp13 oracle rows 218
- manifest audit: 13 exp11 manifests and 65 exp12 manifests with `run_status=fit_completed`, graph `real`, Cell LLM `frozen`, `cell_type` LLM `frozen`, and PTV1 `cell_type` embedding v3

All official rows use graph `real`; no `GRAPH_FEATURE_MODE=off` candidate is present.
