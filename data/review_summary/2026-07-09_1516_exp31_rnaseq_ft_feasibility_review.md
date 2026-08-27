# 2026-07-09 15:16 HKT Exp31 RNA-seq Fine-tune Feasibility Review

## Scope

- Reviewed the exp31 request to use `data/rawdata/rna_seq` for RNA baseline based fine-tuning and benchmark.
- Read the current data-process and training docs:
  - `docs/Data_Process_1.md`
  - `docs/Data_Process_2.md`
  - `docs/Data_Process_3.md`
  - `docs/Data_Process_4.md`
  - `docs/Training_guideline.md`
- Inspected current PTV3 training/inference entry points:
  - `train.py`
  - `infer.py`
  - `dataset/training_ready_fast_dataset.py`
  - `scripts/ptv3_experiment_common.sh`
  - `scripts/exp_09_unified_all_train_valid_oracle.sh`
  - patient validation builders under `utils/13_build_patient_validation_inference_tasks.py` and `utils/14_build_patient_validation_features.py`

## Raw Data Audit

- Raw files present:
  - `data/rawdata/rna_seq/260617_2015_BFnm3954_MOESM10_ESM_sub.csv`
  - `data/rawdata/rna_seq/260618pdx_pct_sample_info_with_smiles_check_comboAB.csv`
- Sample info shape: `2321 x 32`.
- RNA matrix shape: `22665 x 194`.
- RNA matrix columns:
  - metadata columns: `Sample`, `uniprot_ID`;
  - 178 sample columns.
- Unique `sample_id` in sample info: `178`.
- Sample ID overlap with RNA sample columns: `178 / 178`.
- `uniprot_ID` non-null rows: `17404`; unique non-null UniProt IDs: `17404`; unmapped RNA rows: `5261`.
- Overlap with existing PTV3 axes:
  - main single axis: `10060 / 10982` existing proteins covered by RNA;
  - main double axis: `10159 / 11092` existing proteins covered by RNA.

## Label and Split Feasibility

- All four requested label columns are present and non-null for all 2321 treatment rows.
- Treatment rows:
  - single-like rows: `1839`;
  - double-like rows: `482`.
- Cancer type row counts:
  - `BRCA`: `492`;
  - `PDAC`: `552`;
  - `CRC`: `456`;
  - `CM`: `420`;
  - `NSCLC`: `399`;
  - missing cancer type: `2`.
- BRCA adaptation subset:
  - `38` unique samples;
  - `417` single rows;
  - `75` double rows.
- Non-BRCA zero-shot subset:
  - `139` unique samples;
  - `1420` single rows;
  - `407` double rows.
- Each requested label has both positive and negative rows in BRCA and non-BRCA, for both single and double rows.
- BRCA positive counts:
  - `sensitive_label_early_CRPR_vs_SDPD`: single `31/417`, double `23/75`;
  - `sensitive_label_late_CRPR_vs_SDPD`: single `23/417`, double `21/75`;
  - `disease_control_label_early_CRPRSD_vs_PD`: single `151/417`, double `63/75`;
  - `disease_control_label_late_CRPRSD_vs_PD`: single `48/417`, double `38/75`.

## Drug Feature Audit

- `smiles_a` is present for all rows.
- `smiles_b` is present for all 482 double rows and absent for single rows.
- Unique raw SMILES:
  - `smiles_a`: `24`;
  - `smiles_b`: `8`;
  - combined unique SMILES: `25`.
- `ptv_component_guomics_ids` parsed to 29 unique component IDs; all 29 are present in the existing PTV3 `pert_index`.
- Exact raw SMILES string matching to existing `pertid_to_smiles` covers only `12 / 25`, so component IDs should be preferred where available.
- `ptv_treatment_match_status`:
  - `all_components_matched`: `1771` rows;
  - `partial_component_match`: `62` rows;
  - `unmatched`: `488` rows.
- The 488 unmatched rows are five single-drug treatments: `CGM097`, `CLR457`, `HDM201`, `LGH447`, and `abraxane`.
- The two partially matched combo treatments are:
  - `BYL719 + LGH447`;
  - `abraxane + gemcitabine`.

## Model Support Findings

- Current model input contract supports:
  - baseline/control expression;
  - two drug slots;
  - response, synergy, or unified binary head;
  - custom task label and mask keys;
  - `--checkpoint-path` initialization;
  - no-MSE training through `--no-mse-loss`.
- The data do not contain post-treatment perturb expression. Therefore exp31 fine-tuning should be label-only/no-MSE unless a separate target-expression policy is explicitly requested.
- `FastProteinTalkLightning._mse_loss` masks non-finite targets and returns zero when all perturb expression values are missing, but relying on this implicit behavior would make the experiment semantics unclear.
- `train.py --checkpoint-path` uses normal `load_state_dict`; `--allow-partial-checkpoint-load` does not handle tensor size mismatches. A strict fine-tune run therefore needs checkpoint-compatible model dimensions.
- `infer.py` supports protein-axis alignment to checkpoint axes for inference-only tasks, but `train.py` does not provide the same checkpoint-axis remapping for fine-tuning.

## Main Feasibility Conclusion

- The raw data support building an exp31 benchmark in principle.
- The benchmark is not ready to run as-is because exp31 training-ready artifacts, split manifests, feature artifacts, runner scripts, and reports do not exist yet.
- The largest design constraint is checkpoint compatibility during fine-tuning:
  - if exp31 uses an intersection-only RNA protein axis, existing checkpoint loading will likely fail due model size mismatches;
  - if exp31 keeps the checkpoint protein axis and fills RNA-missing proteins with `NaN`, checkpoint loading can remain compatible, while MSE should be disabled.

## User Decisions Needed

1. Starting checkpoint: use exp09 unified single+double checkpoint, exp07/exp08 separately, or another checkpoint.
2. Protein axis policy: preserve checkpoint axis with RNA-missing proteins as `NaN`, or use RNA/PTV3 intersection and add model state-transfer code for size mismatches.
3. Task formulation: train one unified binary response task over single+combo rows, or train separate single and combo tasks.
4. Label tasks: run all four requested labels as four independent exp31 panels, or select one primary label.
5. Unmatched drugs: keep them with generated SMILES-based pert IDs and empty/derived target lists, or drop/resolve them manually.
6. `CM` cancer type mapping: likely maps to `SKIN`, but this needs explicit confirmation.
7. Missing cancer type rows: drop the two rows with missing `cancer_type`, or keep as a separate unknown benchmark group.

