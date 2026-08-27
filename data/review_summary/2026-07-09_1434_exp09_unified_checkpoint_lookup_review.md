# 2026-07-09 14:34 HKT exp09 Unified Checkpoint Lookup Review

- Checked the user's recollection of a single model trained on both single-drug sensitivity and double-drug synergy data.
- Confirmed the matching workflow is `scripts/exp_09_unified_all_train_valid_oracle.sh`, which trains `ptv3_main_doubledrug` with `split_strategy=all_train_subset_test` and `task_head=unified`.
- Confirmed unified label semantics in `train.py`, `infer.py`, and `model/fast_lightning.py`: valid synergy labels take priority; otherwise PRISM response labels are used, with the unified task using the response-logit path for the active binary prediction.
- Confirmed exp09 checkpoint directories still exist for `20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra` and `20260615_2010_exp09_lr1e5_v1_unified_all_single_double_for_extra`, each retaining epoch checkpoints `0..49` plus `last.ckpt`.
- Clarified that the earlier `20260610_ptv01_08_posweight_combo_selected_v1` exp07/exp08 results came from two separate extra checkpoints, while exp09 is the one-run unified alternative.
