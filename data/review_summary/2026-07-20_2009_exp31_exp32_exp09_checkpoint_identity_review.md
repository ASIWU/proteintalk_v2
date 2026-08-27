# 2026-07-20 20:09 HKT Exp31/Exp32 Exp09 Checkpoint Identity Review

## Scope

- Rechecked the exact exp09 checkpoint used to initialize the four Exp31 RNA-seq PDX fine-tuning runs summarized in `docs/2026-07-09_exp31_rnaseq_pdx_ft_benchmark_results.md`.
- Rechecked the exact exp09 checkpoint used for the two formal Exp32 organoid sensitivity inference tasks summarized in `docs/2026-07-10_exp32_organoid_exp09_single_sensitivity_results.md`.
- Verified current checkpoint existence, same-run provenance, numbered checkpoint coverage, sizes, and selected SHA-256 identities.
- This was a read-only review; no experiment, checkpoint, code, data, or result artifact was changed.

## Exact Checkpoint Identity

- Both experiments reference the same parent exp09 training run:
  `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra`.
- They do not reference the same exact checkpoint file:
  - all four Exp31 fine-tuning `run_manifest.json` files record `args.checkpoint_path=.../last.ckpt`;
  - all four Exp31 exp09 zero-shot inference manifests also record `checkpoint_path=.../last.ckpt`;
  - both formal Exp32 inference manifests and the Exp32 summary record `checkpoint_path=.../epoch=5.ckpt`.
- Exp32 is direct unlabeled inference, not an additional fine-tuning run.

## Current File Existence and Hashes

- Exp31 initialization file `last.ckpt` exists and is non-empty (`389,369,531` bytes).
- Exp32 inference file `epoch=5.ckpt` exists and is non-empty (`389,369,531` bytes).
- SHA-256:
  - `epoch=5.ckpt`: `a5734682c37d807b2b6d1bb5ea66ad908107f45e15605aa65c40f6bab9b6db97`;
  - `last.ckpt`: `e75a2314bb11b698acc48739891578e7de0fddb293118f3e3c0fdc3419a39c2d`;
  - `epoch=49.ckpt`: `e75a2314bb11b698acc48739891578e7de0fddb293118f3e3c0fdc3419a39c2d`.
- Therefore `epoch=5.ckpt` and `last.ckpt` are different model states, while `last.ckpt` is byte-identical to `epoch=49.ckpt`.

## Same-run Checkpoint Completeness

- The parent exp09 run manifest exists and records:
  - `run_status=fit_completed`;
  - `max_epochs=50`;
  - `save_last_ckpt=true`;
  - training arguments include `save_every_n_epochs=1`.
- The directory currently contains every numbered checkpoint from `epoch=0.ckpt` through `epoch=49.ckpt`, with no gaps and no unexpected numbered checkpoint names.
- All 50 numbered checkpoint files are regular, non-empty files.
- `last.ckpt` is also present and non-empty, so the directory contains 51 `.ckpt` files total: 50 numbered epochs plus `last.ckpt`.

## Conclusion

- Same exp09 training batch: yes.
- Same exact checkpoint file: no.
- Exp31 checkpoint still exists: yes (`last.ckpt`, equivalent to `epoch=49.ckpt`).
- Exp32 checkpoint still exists: yes (`epoch=5.ckpt`).
- All checkpoints saved from that exp09 training batch still exist: yes, epochs 0–49 plus `last.ckpt`.
