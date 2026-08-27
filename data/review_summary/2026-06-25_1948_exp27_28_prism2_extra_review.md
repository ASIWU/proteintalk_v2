# 2026-06-25 19:48 HKT exp27-exp28 PRISM2 Extra Review

- Reviewed exp_07/exp_08 mechanics and mirrored them for the PRISM2 branch without overwriting existing exp_07/exp_08 artifacts.
- Added:
  - `scripts/exp_27_extra_single_prism2_all_train_infer.sh`
  - `scripts/exp_28_extra_double_prism2aux_all_train_infer.sh`
  - `scripts/run_exp_27_28_prism2_extra.sh`
- Verified shell syntax with `bash -n`.
- Ran exp_27/exp_28 in tmux session `gpu2` using `flow_v2` and `TRAINING_READY_ROOT=data/training_ready_prism2_main`.
- Runtime summary `logs/20260625_1940_prism2_exp27_28_nowandb_runtime_summary.tsv` has 11 rows after the header: 2 train rows and 9 inference rows, all status `0`.
- Manifest checks:
  - exp_27 uses `ptv3_main_singledrug_prism2`, `all_train_subset_test`, response head, and `PRISM2nd_label_total`.
  - exp_28 uses `ptv3_main_doubledrug_prism2aux`, `all_train_subset_test`, synergy head, and `PRISM2nd_label_total` as the auxiliary response key.
- Results were written to `docs/2026-06-25_exp27_28_prism2_extra_results.md`.
