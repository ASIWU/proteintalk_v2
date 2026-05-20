#!/usr/bin/env python3
"""Focused smoke checks for the training-ready train/infer stack."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.training_ready_dataset import (
    ProteinTalkDataset,
    TrainingReadyArtifacts,
    encode_response_label,
    encode_synergy_label,
)
from model.training_ready_lightning import ProteinTalkLightning
from model.training_ready_models import ModelArtifacts, SELECTED_MODEL_NAMES, build_model


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def check_label_encoding() -> None:
    expected_response = [
        (1, (1.0, 0.0)),
        (0, (0.0, 0.0)),
        (1.0, (1.0, 0.0)),
        (0.0, (0.0, 0.0)),
        ("1", (1.0, 0.0)),
        ("0", (0.0, 0.0)),
        ("1.0", (1.0, 0.0)),
        ("0.0", (0.0, 0.0)),
        ("Y", (1.0, 0.0)),
        ("N", (0.0, 0.0)),
        ("sensitive", (1.0, 0.0)),
        ("non-responsive", (0.0, 0.0)),
    ]
    for value, expected in expected_response:
        assert encode_response_label(value) == expected, (value, encode_response_label(value), expected)
    assert encode_synergy_label("syn") == (1.0, 0.0)
    assert encode_synergy_label("non-syn") == (0.0, 0.0)


def make_artifacts(root: Path) -> TrainingReadyArtifacts:
    task_dir = root / "task"
    task_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "sample_id": ["ctrl_a", "ctrl_b", "pert_a"],
            "control": ["ctrl_a", "ctrl_a", "ctrl_a"],
            "is_control": [True, True, False],
            "source_row_role": ["self", "self", "self"],
            "feature_membership": ["primary", "primary", "primary"],
            "machineID_new_index": [0, 0, 0],
            "Cell_plate_index": [0, 0, 0],
            "Cell_index": [0, 0, 0],
            "cell_type_index": [0, 0, 0],
            "batch_index": [0, 0, 0],
            "pert_time_index": [0, 0, 0],
            "pert_index1": [2, 2, 0],
            "pert_index2": [2, 2, 1],
            "target_protein_list": ["[]", "[]", "[0, 1]"],
            "PRISM1st_label_total": ["N", "N", 1.0],
            "synergy": ["N", "N", "syn"],
        }
    )
    df.to_pickle(task_dir / "feature_table.pkl")
    np.save(task_dir / "feature_expression_matrix.npy", np.arange(12, dtype=np.float32).reshape(3, 4))
    dump_json(task_dir / "feature_ordered_protein_index.json", [0, 1, 2, 3])
    dump_json(task_dir / "feature_sample_ids.json", df["sample_id"].tolist())
    meta = {
        "special_values": {"pert_index": {"no": 2}},
        "protein_index": {"P0": 0, "P1": 1, "P2": 2, "P3": 3},
        "value_to_index": {
            "machineID_new": {"0": 0},
            "Cell_plate": {"0": 0},
            "Cell": {"0": 0},
            "cell_type": {"0": 0},
            "batch": {"0": 0},
            "pert_time": {"0": 0},
            "pert_dose": {"0": 0},
        },
    }
    dump_json(root / "global_meta.json", meta)
    return TrainingReadyArtifacts(task_dir, root / "global_meta.json")


def check_deterministic_eval_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = make_artifacts(Path(tmp))
        dataset = ProteinTalkDataset(
            artifacts,
            indices=[2],
            row_to_set_index={0: 0, 1: 0, 2: 0},
            set_info={0: {"control": [1, 0], "perturb": [2]}},
            mode="eval",
            batch_cov_list=["machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time"],
            drug_mode="index",
            target_protein_max_length=2,
        )
        first = dataset[0]
        second = dataset[0]
        assert int(first["control"]["row_index"]) == 0
        assert int(second["control"]["row_index"]) == 0
        assert first["control"]["expressions_hvg"].shape == (4,)
        assert first["perturb"]["pert_id"].shape == (2,)
        assert first["perturb"]["target_protein_list"].shape == (2,)
        assert float(first["perturb"]["PRISM1st_label_total"]) == 1.0
        assert float(first["perturb"]["sensitive_label_mask"]) == 0.0


def check_graph_forward_modes() -> None:
    assert SELECTED_MODEL_NAMES == {"attention_v10_hetero_cls_ee", "baseline_emb_v3"}
    rng = np.random.default_rng(0)
    protein = rng.normal(size=(4, 8)).astype("float32")
    drug = rng.normal(size=(3, 6)).astype("float32")
    pdi = np.ones((3, 4), dtype="float32")
    artifacts = ModelArtifacts(
        protein_embedding=protein,
        drug_embedding=drug,
        ordered_protein_index=[0, 1, 2, 3],
        pdi_matrix=pdi,
    )
    batch = {
        "control": {"expressions_hvg": torch.randn(1, 4)},
        "perturb": {
            "expressions_hvg": torch.randn(1, 4),
            "pert_id": torch.tensor([[0, 1]]),
            "machineID_new": torch.zeros(1, dtype=torch.long),
            "Cell_plate": torch.zeros(1, dtype=torch.long),
            "Cell": torch.zeros(1, dtype=torch.long),
            "cell_type": torch.zeros(1, dtype=torch.long),
            "batch": torch.zeros(1, dtype=torch.long),
            "pert_time": torch.zeros(1, dtype=torch.long),
            "target_protein_list": torch.tensor([[0, 1]]),
            "PRISM1st_label_total": torch.zeros(1),
            "synergy": torch.zeros(1),
            "sensitive_label_mask": torch.zeros(1),
            "synergy_label_mask": torch.zeros(1),
        },
    }
    for use_target, gate in [(False, "concat"), (True, "concat"), (True, "gate")]:
        model = build_model(
            "attention_v10_hetero_cls_ee",
            artifacts=artifacts,
            topk_genes=4,
            batch_cov_list=["machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time"],
            batch_cov_category_sizes=[1, 1, 1, 1, 1, 1],
            hidden_dim=8,
            perturb_fusion_mode="add",
            target_protein_max_length=2,
            dropout=0.0,
            num_heads=2,
            num_layers=1,
            use_target=use_target,
            target_protein_fusion_model=gate,
        )
        lightning = ProteinTalkLightning(model)
        output = lightning(batch)
        assert tuple(output[0].shape) == (1, 4)
        assert tuple(output[1].shape) == (1, 1)
        assert tuple(output[2].shape) == (1, 1)


def check_tiny_trainer_fit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = make_artifacts(Path(tmp))
        dataset_kwargs = {
            "artifacts": artifacts,
            "indices": [2],
            "row_to_set_index": {0: 0, 1: 0, 2: 0},
            "set_info": {0: {"control": [0], "perturb": [2]}},
            "batch_cov_list": ["machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time"],
            "drug_mode": "index",
            "target_protein_max_length": 2,
        }
        train_dataset = ProteinTalkDataset(mode="train", **dataset_kwargs)
        valid_dataset = ProteinTalkDataset(mode="eval", **dataset_kwargs)
        protein = np.ones((4, 8), dtype="float32")
        drug = np.ones((3, 6), dtype="float32")
        pdi = np.ones((3, 4), dtype="float32")
        model = build_model(
            "attention_v10_hetero_cls_ee",
            artifacts=ModelArtifacts(
                protein_embedding=protein,
                drug_embedding=drug,
                ordered_protein_index=[0, 1, 2, 3],
                pdi_matrix=pdi,
            ),
            topk_genes=4,
            batch_cov_list=["machineID_new", "Cell_plate", "Cell", "cell_type", "batch", "pert_time"],
            batch_cov_category_sizes=[1, 1, 1, 1, 1, 1],
            hidden_dim=8,
            perturb_fusion_mode="add",
            target_protein_max_length=2,
            dropout=0.0,
            num_heads=2,
            num_layers=1,
        )
        lightning = ProteinTalkLightning(model)
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="cpu",
            devices=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
        )
        trainer.fit(
            lightning,
            DataLoader(train_dataset, batch_size=1),
            DataLoader(valid_dataset, batch_size=1),
        )


def main() -> None:
    check_label_encoding()
    check_deterministic_eval_dataset()
    check_graph_forward_modes()
    check_tiny_trainer_fit()
    print("training stack smoke passed")


if __name__ == "__main__":
    main()
