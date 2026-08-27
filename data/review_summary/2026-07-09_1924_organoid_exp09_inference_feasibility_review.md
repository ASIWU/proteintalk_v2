# 2026-07-09 19:24 HKT Organoid Exp09 Inference Feasibility Review

## Scope

- Reviewed `docs/Data_Process_1.md` through `docs/Data_Process_4.md` and `docs/Training_guideline.md`.
- Reviewed exp09 runner/checkpoint contract and the current inference path in `infer.py`, `scripts/exp_09_unified_all_train_valid_oracle.sh`, `scripts/ptv3_experiment_common.sh`, and `dataset/training_ready_fast_dataset.py`.
- Inspected organoid raw inputs under `data/rawdata/organoid`.
- No model code, raw data, or training-ready artifacts were modified.

## Organoid Raw Data

- `260702_B260625QC_merged_replicates.csv`
  - Shape: 7416 proteins x 13 sample columns, plus `Protein.Group` and `Genes`.
  - Sample columns: `B260625_PTV2_1` through `B260625_PTV2_13`.
  - Nonnegative raw abundance values; missing fraction about 0.134.
  - 7359 proteins overlap the exp09 checkpoint protein axis.
- `260702_CAC260627QC_merged_replicates.csv`
  - Shape: 8282 proteins x 13 sample columns, plus `Protein.Group` and `Genes`.
  - Sample columns: `CAC260627_PTV2_1` through `CAC260627_PTV2_13`.
  - Nonnegative raw abundance values; missing fraction about 0.116.
  - 8190 proteins overlap the exp09 checkpoint protein axis.
- B and CAC matrices share 7060 proteins; B-only proteins: 356; CAC-only proteins: 1222.

## Exp09 Checkpoint Contract

- Checked checkpoint root: `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra`.
- `last.ckpt` and epoch checkpoints are present.
- `run_manifest.json` shows:
  - task: `ptv3_main_doubledrug`
  - task head: `unified`
  - split: `all_train_subset_test`
  - status: `fit_completed`
  - pair settings: `PAIR_FUSION_MODE=dual`, `PAIR_TYPE_FEATURES=1`, `USE_DDI=1`, `GRAPH_PAIR_ADD_SCALE=0.5`
  - Cell LLM: frozen, indexed by `Cell_index`
  - Cell-type LLM: frozen, indexed by `cell_type_index`
  - protein concat mode: `pcep`
  - protein axis: `data/training_ready/ptv3/tasks/ptv3_main_doubledrug/feature_ordered_protein_index.json`, length 11092

## Drug Scope Audit

- Explicit exp01-exp09 task list from existing scripts:
  - `ptv3_main_singledrug`
  - `ptv3_main_doubledrug`
  - six extra single-drug tasks
  - three extra double-drug tasks
- Unique drugs across this explicit task list: 3217.
- All 3217 drug IDs exist in `data/training_ready/ptv3/global_meta.json`.
- Unordered drug pairs observed across these tasks: 15984.

## Feasibility

- The current data and model support building organoid inference-only tasks without modifying raw data or model code.
- Required CPU preprocessing would need to create a copy-on-write training-ready root for organoid, with two tasks for the two devices.
- Required GPU execution can use `infer.py` with the exp09 checkpoint in the `gpu2` tmux session.

## Open Decisions

- Drug query scope must be confirmed:
  - exp09 trained main single+double drugs only: about 2175 drugs;
  - explicit exp01-exp09 main+extra drugs: 3217 drugs;
  - all `global_meta` pert IDs: 6130 non-`no` drugs.
- Query type must be confirmed:
  - single-drug sensitivity only;
  - existing exp01-exp09 single rows plus existing double pairs for synergy;
  - all pairwise combinations, which would be much larger and is not implied by current exp01-exp09 tasks.
- New organoid sample covariate handling must be confirmed:
  - conservative path: map new `Cell` and `cell_type` values to existing `no` indices;
  - extended Cell LLM path: build organoid sample embeddings and infer with `--cell-llm-index-column cell_llm_index`.
