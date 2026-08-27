# 2026-06-03 19:39 HKT PTV1 Readiness Review

## Scope

Reviewed current PTV1 readiness after the completed PTV3 exp01-exp08 tuning suite, focusing on:

- `docs/Data_Process_1.md` through `docs/Data_Process_4.md`
- `docs/Training_guideline.md`
- recent `docs/` result summaries
- recent `data/review_summary/` records
- `data/rawdata/ptv1`
- `data/rawdata/ptv1_extra_singledrug`
- current standardized/training-ready PTV1 artifacts
- PTV1 split artifacts
- training entrypoint support for `--dataset-group ptv1`

## Findings

1. PTV1 table artifacts are present and internally valid.
   - `utils/03_validate_training_ready_outputs.py` passed.
   - `ptv1_aivc`: processed and feature shapes are `15002 x 5576`.
   - `ptv1_extra_singledrug`: processed and feature shapes are `186 x 5576`.

2. PTV1 main split artifacts are present.
   - `ptv1_aivc` has `fixed_experiment_type`, `random`, `pert_id_5fold_fold0..4`, and `all_train_subset_test`.
   - `fixed_experiment_type` is parsed from `data/rawdata/ptv1/experiment_type_list`.
   - The five `pert_id_5fold` splits have zero `pert_id1` overlap between train/valid/test.

3. PTV1 label encoding is compatible with training code.
   - `ptv1_aivc` uses `Y/N` in `PRISM1st_label_total`; `encode_response_label` supports this.
   - `ptv1_extra_singledrug` uses numeric `1.0/0.0` in `PRISM2nd_label_total`; the same encoder supports this.

4. PTV1 extra single-drug mapping is mostly ready.
   - `ptv1_extra_singledrug` has `182` perturb rows and `4` matched control rows.
   - There are `84` unique extra drugs.
   - Non-control extra rows have no missing SMILES.
   - `22` non-control extra rows have empty target lists after mapping, which is currently allowed but should be reported in experiment notes.

5. Current PTV1 is not yet directly trainable with the selected latest architecture.
   - `data/training_ready/ptv1/derived` currently only has `uniprot_ids.txt` and `uniprot_ids.audit.json`.
   - Missing required defaults:
     - `protein_embedding_esm.pkl`
     - `drug_embedding_morgan_2048.pkl`
     - `ppi_matrix.npy`
     - `pdi_matrix.npy`
     - `ddi_matrix.npy`
   - A minimal `train.py --dataset-group ptv1 --task-name ptv1_aivc ... --graph-feature-mode off` smoke fails at `protein_embedding_esm.pkl` lookup.

6. PTV1 cannot be solved by simple PTV3 artifact slicing.
   - PTV1 protein index has `5578` entries; `96` are not in PTV3 protein index.
   - PTV1 pert index has `148` entries; only `85` overlap PTV3, while `63` main PTV1 `#` drugs are PTV1-specific.
   - Existing PTV3 FASTA/cache can help regenerate PTV1 protein embeddings, but PTV1 still needs its own derived outputs.

7. PTV1 experiment runner scripts are not yet present.
   - Current `scripts/exp_01` through `scripts/exp_08` and shared `ptv3_experiment_common.sh` are hardwired to `--dataset-group ptv3`.
   - PTV1 needs either a new common runner or parameterized reuse before running random split, unseen-drug 5-fold, and extra-single inference cleanly.

## Verification Commands

```bash
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python utils/03_validate_training_ready_outputs.py
/mnt/shared-storage-user/wuhao/miniconda3/envs/flow_v2/bin/python train.py --dataset-group ptv1 --task-name ptv1_aivc --split-strategy random --task-head response --model-type fast_delta --max-epochs 1 --limit-train-batches 1 --limit-val-batches 1 --limit-test-batches 1 --batch-size 2 --logger-backend none --checkpoint-dir /tmp/proteintalk_v2_ptv1_smoke_ckpt --log-dir /tmp/proteintalk_v2_ptv1_smoke_logs --experiment-name ptv1_smoke_missing_derived --graph-feature-mode off --skip-test
```

## Conclusion

PTV1 data and splits are close, but PTV1 is not yet ready for the selected graph-enabled training suite until the PTV1 derived embeddings/graphs and PTV1 experiment launcher scripts are added.
