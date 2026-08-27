# Organoid exp09 inference feasibility reassessment

- Review time: 2026-07-10 11:32:09 +0800
- Scope: reassess exp09 inference on the two organoid baseline-proteome matrices after adding `data/rawdata/rna_seq/260709_samp_inf.csv`
- Execution policy checked: CPU preparation on the local shell; GPU inference only in the `gpu2` tmux session
- Raw files were not modified and no GPU job was launched.

## Reviewed inputs and contracts

- Workflow documentation:
  - `docs/Data_Process_1.md`
  - `docs/Data_Process_2.md`
  - `docs/Data_Process_3.md`
  - `docs/Data_Process_4.md`
  - `docs/Training_guideline.md`
- Organoid matrices:
  - `data/rawdata/organoid/260702_B260625QC_merged_replicates.csv`
  - `data/rawdata/organoid/260702_CAC260627QC_merged_replicates.csv`
- Sample metadata:
  - `data/rawdata/rna_seq/260709_samp_inf.csv`
- exp09 implementation and checkpoint evidence:
  - `scripts/exp_09_unified_all_train_valid_oracle.sh`
  - `scripts/ptv3_experiment_common.sh`
  - `infer.py`
  - `train.py`
  - `dataset/training_ready_fast_dataset.py`
  - `utils/13_build_patient_validation_inference_tasks.py`
  - `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/run_manifest.json`

## Organoid input findings

| source | instrument meaning | protein rows | sample columns | missing fraction | exp09-axis overlap |
|---|---|---:|---:|---:|---:|
| B | user-defined as QE_HF | 7,416 | 13 | 13.42% | 7,359 / 11,092 |
| CAC | user-defined as 480_FAIMS | 8,282 | 13 | 11.62% | 8,190 / 11,092 |

- Both matrices contain non-negative raw abundance values and are compatible with the documented finite-value `log1p` transformation.
- The B and CAC protein sets share 7,060 proteins. B has 356 B-only proteins and CAC has 1,222 CAC-only proteins.
- Per sample, B contains 6,188-6,674 measured proteins and CAC contains 7,147-7,605 measured proteins before checkpoint-axis filtering.
- After matching the common exp09 protein axis, the paired B/CAC log1p Pearson correlations for sample suffixes 1-13 range from 0.8962 to 0.9094 (mean 0.9021). This strongly supports that the same numeric suffix denotes the same biological sample across instruments.
- Missing checkpoint-axis proteins can remain `NaN` in the task artifact; the existing fast dataset/model path converts non-finite expression values to zero at model input. Proteins outside the fixed exp09 axis must be ignored for checkpoint-compatible inference.

## Sample metadata findings

- `260709_samp_inf.csv` contains exactly 13 unique `samp_ID` values, 1 through 13, and 13 unique `pat_ID` values.
- Every row has a Chinese cancer label and an English tissue label:
  - Lung / lung cancer: 10
  - Pancreas / pancreatic cancer: 2
  - Colon / colorectal cancer: 1
- These are tissue/tumor types, not fine-grained cell types.
- The exp09 PTV3 categorical space already contains `LUNG`, `PANCREAS`, and `COLON`; the validated cell-type LLM artifact has corresponding rows. No new tissue-type category or LLM API call is required.
- The metadata does not explicitly contain the full `B260625_PTV2_n` or `CAC260627_PTV2_n` column names. Joining `samp_ID=n` to the `PTV2_n` suffix is strongly supported by exact 1-13 coverage and cross-instrument expression correlation, but remains a mapping rule that should be confirmed by the data owner.
- The final numeric metadata field has no header. It is not required for inference and must remain an audit-only field until its meaning is provided.

## Checkpoint and feature compatibility

- The selected exp09 run exists and is complete. It was trained on merged single-drug and native double-drug rows with `task_head=unified`.
- Key active features are checkpoint-compatible: real control expression, PCEP, Morgan embedding, PPI/PDI/DDI graph features, target proteins, categorical covariates, frozen Cell LLM, and frozen cell-type LLM.
- The two instrument categories can potentially map to existing exp09 categories:
  - `CAC -> 480_FAIMS` is an exact semantic/category match.
  - `B/QE_HF -> QE` is the likely compatible mapping, but the equivalence must be confirmed because `QE_HF` is not itself an existing category.
- The new organoid/patient identifiers are not in the 74-category exp09 `Cell` space. A checkpoint-compatible conservative path is to preserve `pat_ID` as output/audit metadata while mapping categorical `Cell_index` and Cell-LLM input to `no`. Tissue information still enters through the existing `cell_type` categorical and cell-type LLM paths.
- New `Cell_plate` and `batch` values are also outside the checkpoint category spaces and would need to map to `no`; extending those learned categorical embedding sizes would be incompatible with the existing checkpoint.

## Drug scope

The union was computed from the main and extra PTV3 tasks used by exp01-exp09:

- main single and double tasks;
- extra single mat1-4 tasks for QE/480_FAIMS;
- extra double Nature, NC, and Guomics tasks.

Results:

- 3,217 unique drugs.
- All 3,217 are present in the exp09 `pert_index`.
- All 3,217 have non-zero Morgan embeddings.
- All 3,217 have finite, non-zero cached PPI/PDI/DDI-derived graph features.
- All 3,217 have SMILES metadata.
- 2,174 / 3,217 have a non-empty explicit target-protein list; missing target lists do not block inference because Morgan and graph features remain available.
- The reviewed double-drug tasks contain 13,014 unique observed unordered drug pairs.

Expected perturbation prediction counts across 13 samples and two instruments:

- all 3,217 single drugs: 83,642 rows;
- the 13,014 observed double-drug pairs: 338,364 rows;
- singles plus observed pairs: 422,006 rows;
- every possible pair among 3,217 drugs: 134,496,336 rows, which is a materially different and much larger task than observed-pair inference.

## Unified-head output semantics

- A single-drug query must use `pert_id1 == pert_id2`; the unified score is interpreted as sensitivity probability.
- A double-drug query must use distinct drug slots; the unified score is interpreted as synergy probability.
- In the current `infer.py` unified-head path, `pred_response_prob` and `pred_synergy_prob` are both populated from the same unified-head probability. They are not two independent predictions for one row. Reports must use one `pred_task_prob` plus an explicit `prediction_type` (`sensitivity` or `synergy`) derived from the query row.

## Checkpoint selection evidence

The exp09 directory contains epochs 0-49 and `last.ckpt`:

- prespecified reference-epoch policy: epoch 5 for extra single sensitivity and epoch 2 for extra double synergy;
- oracle best on the extra sets: epoch 2 for single and epoch 8 for double;
- `last.ckpt`: final epoch checkpoint, used as the exp31 fine-tuning initializer but not selected as the exp09 extra-inference reference checkpoint.

Using oracle-selected epochs for a new external benchmark would inherit selection on the extra evaluation sets. The desired checkpoint policy must be declared before organoid inference.

## Remaining decisions / blockers to a scientifically defined run

The data and model are technically sufficient, but the following choices cannot be inferred safely:

1. Confirm that `samp_ID=n` maps to both `B260625_PTV2_n` and `CAC260627_PTV2_n`.
2. Confirm that `QE_HF` should use the checkpoint category `QE`.
3. Clarify whether the earlier scope of 3,217 single-drug sensitivity queries still applies, or whether double-drug synergy is now also required.
4. If synergy is required, choose observed 13,014 pairs versus all 5,172,936 possible pairs.
5. Supply an exposure policy for `pert_time`, `pert_dose1`, and `pert_dose2`, which are active exp09 covariates. One prediction per sample-drug requires either fixed values or an explicitly defined aggregation over historical conditions.
6. Confirm the checkpoint policy: prespecified reference epochs, final `last.ckpt`, or another named checkpoint. Oracle epochs should only be used if their evaluation-set selection is intentionally accepted.
7. Confirm that unseen `Cell`, `Cell_plate`, and `batch` categories may map to `no`, while retaining `pat_ID` and source identifiers as audit columns.

## Feasibility conclusion

The existing data supports checkpoint-compatible exp09 inference after a copy-on-write training-ready task is built. No model architecture change, drug-index extension, graph rebuild, or tissue-type LLM regeneration is required. The run should not be launched until the query semantics and active covariate values listed above are confirmed.

The `gpu2` tmux session exists and currently presents an idle `flow_v2` shell. No command was sent to it during this review.

## Follow-up clarification (2026-07-10 11:36:21 +0800)

The user confirmed the following settings:

- map the B/QE_HF source to the checkpoint category `QE`;
- run only the 3,217 single-drug sensitivity queries, not double-drug synergy;
- use the single-drug prespecified reference checkpoint, `epoch=5.ckpt`;
- use the maximum exposure time, with the exact missing-time policy still to be clarified.

Historical-condition audit across the 3,217-drug scope found:

- 2,175 drugs have at least one numeric `pert_time` value;
- 2,174 drugs have a per-drug maximum time of 24;
- 1 drug has a per-drug maximum time of 6;
- 1,042 drugs have no numeric historical time;
- at each drug's maximum observed time, 2,028 drugs have one numeric dose, 147 have multiple numeric doses, and 1,042 have no numeric dose.

Therefore, `use maximum time` still requires an explicit choice between setting all queries to the global maximum `pert_time=24` versus using each drug's own maximum with a fallback for the 1,042 missing-time drugs. A dose policy is also still required.
