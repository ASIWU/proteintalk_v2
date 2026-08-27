# Exp33 Alias, Slot, Dose, and Score Follow-up Review

- Review time: 2026-07-27 11:33 HKT (UTC+08:00)
- Scope: follow-up to the Exp33 feasibility review after the user confirmed the common Exp09 epoch-2 checkpoint, 24-hour treatment time, clipped/ceil dose semantics, mixed-time unique baseline rule, and acceptance of the available score semantics.
- This review did not modify code, raw data, standardized data, checkpoints, or prediction outputs.

## Confirmed Configuration

- Checkpoint:
  `checkpoints/20260615_1101_exp09_selectedref_v1_unified_all_single_double_for_extra/epoch=2.ckpt`
- Treatment time: `24`
- Dose preprocessing: numeric values above 10 are clipped to 10, then numeric dose is mapped to the stringified `ceil(dose)` categorical index.
- Baseline: exact within-tissue cell match to the supplied `*_control_unique.csv`, with the already reviewed model metadata recovery, protein-axis alignment, finite-value `log1p`, and missing-value preservation.

## Why 48 Query Drugs Have Ambiguous Internal IDs

The ambiguity is an ID-provenance issue, not a chemistry-parsing failure.

- The maintained perturbation registry contains several source-specific namespaces, including numeric main-task IDs, `L9200_*` IDs, and deterministic external-source IDs such as `extid::*` and `extsmiles::*`.
- The same chemical can therefore retain several internal IDs when it appeared in several source datasets. The IDs preserve the source records instead of forcing all records with the same chemical structure into one ID.
- The Exp33 query files contain drug names and SMILES but do not provide the source-specific internal `pert_id`. A SMILES match can identify the chemical, but it cannot reconstruct which historical source alias the query author intended.
- Exact name/SMILES resolution is intentionally unique-only in the maintained standardization path. When several registry rows share an exact chemical key, silently taking the first ID would make the result depend on registry ordering.

The complete checkpoint-registry audit found 48 ambiguous name/SMILES records:

- role occurrence: 30 appear only as A, 11 only as B, and 7 in both roles;
- each record has between 3 and 9 full-registry candidate IDs;
- all candidates for every one of the 48 records have identical checkpoint Morgan, PDI, DDI, and graph-feature rows;
- explicit `target_protein_list` is identical within only 12/48 alias groups and differs within 36/48 groups.

Consequently, alias selection does not change the audited structure/STITCH-derived feature rows, but it can change the target-list encoder input. It also determines the stored `pert_id`, `pert_index`, and same-drug identity semantics.

This refines and corrects the preliminary statement in
`2026-07-27_1110_exp33_double_drug_virtual_screen_feasibility_review.md` that an alias could change PDI/graph features. The exhaustive candidate-row comparison shows that PDI/graph features are identical for all 48 groups in this checkpoint; the material feature ambiguity is the explicit target list.

Examples of distinct aliases for the same queried chemical include:

- 5-FU: numeric ID `113`, `L9200_531`, and external-source aliases;
- AZD-7762: numeric ID `107`, `L9200_2573`, and external-source aliases;
- Crizotinib: numeric ID `114`, `L9200_1252`, and external-source aliases;
- Olaparib: numeric ID `105`, `L9200_826`, and external-source aliases.

## Recommended Deterministic Alias Rule

For each ambiguous query structure:

1. Among exact checkpoint-registry structure matches, choose the candidate with the largest occurrence count in native double-drug rows of the exact Exp09 training-ready task.
2. If every candidate has zero native-double occurrence, choose the candidate with the largest occurrence count across the complete Exp09 training-ready task.
3. Break any remaining tie by the smaller global `pert_index`.

Audit result:

- 47/48 structures obtain a unique winner from native-double occurrence;
- Paclitaxel is the only fallback and resolves to numeric ID `142` by total Exp09 occurrence;
- all 48 obtain a deterministic unique winner with no unresolved tie;
- this rule differs from a total-training-count-only rule for 41/48 groups, primarily because the latter favors numeric aliases learned through auxiliary single-drug rows, whereas the proposed rule favors the aliases actually seen by the native double-drug task.

This is the most task-aligned rule available without an authoritative query-to-`pert_id` table. It still requires explicit user approval before formal Exp33 construction.

## A/B Slot Semantics

The recommended mapping is:

| query field | model field | meaning |
|---|---|---|
| drug A name/SMILES | `pert_id1` | anchor drug |
| `drug_A_concentration_uM` | `pert_dose1` | anchor exposure |
| drug B name/SMILES | `pert_id2` | library drug |
| `assumed_combo_IC50_B_uM` | `pert_dose2` | proposed library exposure |

The order must be retained for three independent reasons:

1. Maintained double-drug standardization maps raw Anchor fields to slot 1 and Library fields to slot 2.
2. The Exp33 query design has a comparatively small A-side anchor set and a broad, fixed B-side library, matching that training convention.
3. The checkpoint uses `pair_fusion_mode=dual`, whose pair representation includes the symmetric mean, absolute difference, and product together with ordered `drug1` and `drug2` pieces. Separate slot-1 and slot-2 dose embeddings are also active. Swapping A/B is therefore not guaranteed to preserve a prediction.

The A-side dose meaning is unambiguous. The B-side field still has one semantic question:

- If `assumed_combo_IC50_B_uM` is the concentration at which B is intended to be evaluated in that row, it should be `pert_dose2`.
- If it is only an estimated/robustness-analysis statistic and was not intended as a B exposure condition, feeding it to the dose embedding would encode an outcome-derived or hypothetical value as treatment dose. In that case Exp33 needs a fixed B-dose policy or should omit B-dose variation.

Formal construction therefore still needs confirmation that `assumed_combo_IC50_B_uM` represents the intended B concentration condition.

## What the Checkpoint Sees After Dose Binning

The model does not consume the full continuous dose value. It consumes a categorical index:

- `0 -> 0`
- `0.05 -> 1`
- `0.1247 -> 1`
- `1.367 -> 2`
- `9.5 -> 10`
- `20 -> clip to 10 -> 10`

Raw dose columns should remain in the final result for provenance, but rows identical over cell, selected A/B IDs, time, baseline metadata, and the two dose-bin indices are model-equivalent.

| tissue | raw rows | unique model-equivalent keys | raw/key ratio |
|---|---:|---:|---:|
| colon | 403,200 | 218,736 | 1.843 |
| lung | 806,400 | 462,210 | 1.745 |
| pancreas | 1,317,120 | 443,982 | 2.967 |
| total | 2,526,720 | 1,124,928 | 2.246 |

Inference can therefore be run once for each of the 1,124,928 unique model keys and joined back to all 2,526,720 raw rows without changing model results. This avoids redundant compute while preserving the full requested screening table.

## Unified-score Semantics

The epoch-2 checkpoint uses `task_head=unified` with the training-label policy
`unified_synergy_first_else_response`:

- when a training row has a valid synergy label, that label supervises the active response logit;
- otherwise the response label supervises the same active response logit.

Current inference copies `pred_response_prob` into `pred_synergy_prob` for a unified checkpoint, and `pred_task_prob` also selects that same probability. These three columns are therefore not three independent predictions:

`pred_task_prob == pred_response_prob == pred_synergy_prob`

For Exp33, the primary ranking column should be named `pred_unified_combo_prob` and sourced from `pred_task_prob`. If compatibility columns are retained, their alias relationship must be documented rather than presenting them as independent response and synergy estimates.

## Remaining Decisions

Only two semantic approvals remain before implementation:

1. Approve the native-double-frequency-first alias-selection rule.
2. Confirm that `assumed_combo_IC50_B_uM` is the intended B-side concentration condition and may be used as `pert_dose2`.

