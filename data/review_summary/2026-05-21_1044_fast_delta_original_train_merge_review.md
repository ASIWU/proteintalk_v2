# 2026-05-21 10:44 HKT FastDelta Original Train Merge Review

Scope: review whether the `new_version` FastDelta graph-feature model can be merged into the original `train.py` by changing only model definitions.

Conclusion:
- Changing only `model/` is not sufficient.
- The original training path uses a nested batch contract (`control`/`perturb`) and only knows PDI as a graph-model artifact.
- The new FastDelta path uses a flat batch contract and requires PPI/PDI/DDI graph-feature cache construction, graph ablation modes, and additional graph/model hyperparameters.

Required merge areas:
- `model/training_ready_models.py`: register a new model name and expose/build the FastDelta model or an adapter.
- `dataset/training_ready_dataset.py`: emit `graph_features`, `graph_feature_mask`, optional `ddi_value`, and the label/mask names expected by FastDelta, or provide a model adapter that translates the original nested batch.
- `train.py`: add PPI/DDI default paths, graph-feature CLI args, graph cache construction, model builder arguments, manifest fields, and the correct strategy defaults.
- `model/training_ready_lightning.py`: either support the FastDelta flat batch/loss path or route the new model through the existing nested-batch metrics contract.
- Optional helper module: move `new_version/graph_feature_utils.py` into a stable import location so original `train.py` can build cached graph features.
