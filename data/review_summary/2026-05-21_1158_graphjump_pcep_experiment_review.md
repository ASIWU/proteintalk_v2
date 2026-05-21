# 2026-05-21 11:58 HKT GraphJump and PCEP Experiment Review

Scope: review the new `new_version` architecture changes and full 5-fold experiment results for low-cost protein concat and multi-hop graph fusion.

Code changes reviewed:
- `new_version/graph_feature_utils.py`: added optional multi-hop PPI/PDI/DDI graph feature cache blocks.
- `new_version/fast_delta_model.py`: added PCEP protein-conditioned expression pooling and selective graph jump fusion.
- `new_version/train.py`: added CLI and model construction support for PCEP, multi-hop graph features, and graph jump gates.
- `new_version/run_single_unseen_sweep.sh` and `new_version/run_single_unseen_5fold.sh`: added experiment switches and method presets.

Verification:
- `python -m py_compile` passed for the changed Python modules.
- Single-GPU dry run passed for `graph-multihop + selective sparse jump + PCEP`.
- Two-GPU DDP smoke training/testing passed for the same configuration.
- Two full sweeps were completed on 2 GPUs:
  - `20260521_graphjump_v1`: selective sparse multi-hop jump, with and without PCEP, real and zero graph ablations.
  - `20260521_graphjump_v2`: baseline3+PCEP and multihop-concat, real and zero graph ablations.

Main results:
- baseline3 remains best: AUPRC 0.656369, AUROC 0.900316.
- baseline3+PCEP: AUPRC 0.651715, AUROC 0.899814, graph AUPRC gap +0.059653.
- selective sparse multi-hop jump: AUPRC 0.619492, graph AUPRC gap +0.011594.
- selective sparse multi-hop jump + PCEP: AUPRC 0.623436, graph AUPRC gap +0.050996.
- multihop concat: AUPRC 0.612670, graph AUPRC gap +0.020400.

Conclusion:
- Use baseline3 as the default standard baseline.
- Keep PCEP as an optional interpretability-oriented ablation; it restores lightweight per-protein expression/protein-embedding interaction without large speed loss.
- Do not promote the tested multi-hop/selective jump variants to the main baseline without further redesign, because they reduce the main unseen single-drug AUPRC.
