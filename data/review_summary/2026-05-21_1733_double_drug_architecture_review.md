# Double-Drug Architecture Review

Time: 2026-05-21 17:33 Asia/Hong_Kong

Scope:
- Reviewed current root baseline4 fast path for single-drug and double-drug handling.
- Files inspected: `train.py`, `infer.py`, `dataset/training_ready_fast_dataset.py`, `model/fast_delta_model.py`, `model/fast_lightning.py`, `scripts/exp_07_extra_single_all_train_infer.sh`, `scripts/exp_08_extra_double_all_train_infer.sh`, `scripts/ptv3_experiment_common.sh`, and training-ready manifests/tables.
- No training/model/data-processing code was modified.

Findings:
1. Single and double use the same model architecture class (`FastDeltaDrugResponseModel`) and same two-slot drug-pair encoder, but they are trained as separate checkpoints by the experiment scripts.
2. Single-drug rows use two identical drug slots (`pert_index1 == pert_index2`) for all non-control single-drug anchors; they do not use `drug + no`.
3. Primary double-drug rows use two different drug slots (`pert_index1 != pert_index2`) for all primary non-control double-drug anchors checked.
4. The pair encoder is symmetric: each slot is encoded with the same drug encoder, then fused through mean, absolute difference, and product. It is not raw ordered concatenation.
5. `ptv3_main_doubledrug` includes merged single-drug auxiliary rows in the training table. Their synergy labels are cleared, so they are ignored by the synergy BCE loss, but they still contribute to expression MSE loss.
6. In the all-train double run inspected, the train split had 19,777 rows: 17,986 rows with missing synergy labels and only 1,791 active synergy-labeled rows. With `mse_weight=0.25`, this can make double training heavily dominated by single-drug expression reconstruction.
7. Extra double-drug label priors are much more imbalanced than the main double training labels. The inspected extra datasets had low positive rates, which contributes to very low AUPRC and distribution mismatch risk.

Conclusion:
- The current implementation is architecturally consistent with a two-slot unified drug-pair model, but the double-drug training setup has a high-risk imbalance: most double-task training rows are auxiliary single-drug rows that only train the MSE branch. This is a plausible explanation for weak extra double-drug performance and should be tested before changing the standard baseline.
