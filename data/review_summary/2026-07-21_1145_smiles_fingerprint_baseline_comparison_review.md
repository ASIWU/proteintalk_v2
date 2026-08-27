# SMILES Fingerprint Baseline Comparison Review

- Review time: 2026-07-21 11:45 HKT (UTC+08:00)
- Scope: compare the active SMILES-to-drug-fingerprint path with `/root/tmp/proteintalk_v2/baseline/get_ptv_all_drug_fp.py`.
- No production code or derived artifact was changed by this review.

## Conclusion

The implementations are not parameter-equivalent. Both generate 2,048-bit RDKit Morgan bit fingerprints with the generator defaults, but the baseline uses radius 3 while the current training-ready pipeline uses radius 2. Therefore the current artifact does not reproduce the baseline fingerprint values.

## Code Evidence

- Baseline: `rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)`; valid fingerprints are exported through `ToBitString()`, and invalid/missing inputs are filled with 2,048 zeros.
- Current main builder: `AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)` with CLI defaults `radius=2` and `n_bits=2048`; output is a `float32` NumPy matrix aligned to `global_meta.json["pert_index"]`.
- README command: explicitly passes `--radius 2 --n-bits 2048`.
- Current PTV1 builder defaults to `--drug-radius 2 --drug-n-bits 2048`.
- Exp31 added-drug fingerprint generation defaults to radius 2 and 2,048 bits.
- Patient-validation feature rebuilding and the main DDI builder also use radius 2 and 2,048 bits.

## Input And Fallback Differences

- Baseline reads one Excel row at a time from `Smiles_no_chiral` and produces `fp_0` through `fp_2047` columns.
- The current builder reads the pert-ID-to-SMILES map from `global_meta.json`, strips surrounding whitespace, deduplicates/aligned rows by `pert_index`, and saves a pickle payload.
- Current stage-1 PTV3 standardization prefers `Smiles_with_chiral`, then `smiles`, then `Smiles_no_chiral`; the baseline reads only `Smiles_no_chiral`.
- The baseline returns `None` for invalid/missing inputs and later replaces it with zeros. The current builder fingerprints an empty RDKit molecule for missing, invalid, or special inputs, which also produces an all-zero bit vector, and records the fallback reason.
- The current PTV3 metadata has three empty-SMILES IDs: `control`, `PC`, and `no`; the artifact records all three fallback reasons.

## Runtime Verification

Using `flow_v2`, the current PTV3 artifact and all 6,131 `pert_index` rows were checked against freshly generated fingerprints:

- artifact: `data/training_ready/ptv3/derived/drug_embedding_morgan_2048.pkl`
- shape and dtype: `(6131, 2048)`, `float32`
- stored payload parameters: radius 2, 2,048 bits
- rows matching fresh radius-2 fingerprints: `6131 / 6131`
- rows matching fresh radius-3 fingerprints: `79 / 6131`
- rows whose radius-2 and radius-3 fingerprints differ: `6052 / 6131`
- valid SMILES rows: 6,128; empty-SMILES fallback rows: 3

At a fixed radius of 3, `AllChem.GetMorganGenerator` and `rdFingerprintGenerator.GetMorganGenerator` matched on all 6,131 rows. `DataStructs.ConvertToNumpyArray` and conversion from `ToBitString()` also matched on all rows. This isolates the observed fingerprint mismatch to the radius setting (with input-source policy remaining a separate possible difference), not the RDKit namespace or vector-export method.

## Impact

Changing the current pipeline from radius 2 to radius 3 would change the model input representation for nearly all drugs and would make existing checkpoints incompatible in meaning with regenerated drug embeddings and DDI features. Any such change should be treated as a new, versioned data/model experiment rather than overwriting current artifacts.
