#!/usr/bin/env python3
"""Regression tests for compact expression-row indirection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.training_ready_fast_dataset import FastProteinTalkDataset, FastTrainingReadyArtifacts


class ExpressionRowIndexTest(unittest.TestCase):
    def write_artifacts(
        self,
        root: Path,
        *,
        expression: np.ndarray,
        expression_row_index: list[object] | None,
    ) -> tuple[Path, Path]:
        task_dir = root / "tasks" / "test_task"
        task_dir.mkdir(parents=True)
        row_count = len(expression) if expression_row_index is None else len(expression_row_index)
        table = pd.DataFrame(
            {
                "sample_id": [f"row_{idx}" for idx in range(row_count)],
                "control": ["row_0"] * row_count,
                "is_control": [idx == 0 for idx in range(row_count)],
                "pert_index1": [0] * row_count,
                "pert_index2": [0] * row_count,
                "target_protein_list": ["[]"] * row_count,
            }
        )
        if expression_row_index is not None:
            table["expression_row_index"] = expression_row_index
        table.to_pickle(task_dir / "feature_table.pkl")
        np.save(task_dir / "feature_expression_matrix.npy", np.asarray(expression, dtype=np.float32))
        (task_dir / "feature_ordered_protein_index.json").write_text("[0, 1]", encoding="utf-8")
        (task_dir / "feature_sample_ids.json").write_text(
            json.dumps(table["sample_id"].tolist()), encoding="utf-8"
        )
        meta_path = root / "global_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "protein_index": {"P0": 0, "P1": 1},
                    "pert_index": {"drug": 0, "no": 1},
                    "special_values": {"pert_index": {"no": 1}},
                    "value_to_index": {},
                }
            ),
            encoding="utf-8",
        )
        return task_dir, meta_path

    def test_legacy_artifact_uses_identity_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir, meta_path = self.write_artifacts(
                Path(temp_dir),
                expression=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                expression_row_index=None,
            )
            artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
            np.testing.assert_array_equal(artifacts.expression_row_indices, np.asarray([0, 1]))

    def test_compact_control_and_nan_sentinel_are_dereferenced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir, meta_path = self.write_artifacts(
                Path(temp_dir),
                expression=np.asarray([[1.0, 2.0], [5.0, 6.0], [np.nan, np.nan]], dtype=np.float32),
                expression_row_index=[0, 2, 1],
            )
            artifacts = FastTrainingReadyArtifacts.load(task_dir, meta_path)
            dataset = FastProteinTalkDataset(
                artifacts=artifacts,
                indices=[1],
                row_to_set_index={0: 0, 1: 0},
                set_info={0: {"control": [0], "perturb": [1]}},
                mode="eval",
                drug_embedding_matrix=np.zeros((2, 4), dtype=np.float32),
                batch_cov_list=[],
                target_protein_max_length=2,
                effective_key1="response",
            )
            item = dataset[0]
            np.testing.assert_array_equal(item["control_expression"], np.asarray([1.0, 2.0]))
            self.assertTrue(np.isnan(item["perturb_expression"]).all())

    def test_invalid_expression_row_indices_fail_fast(self) -> None:
        invalid_cases = {
            "missing": [0, None],
            "non_integer": [0, 1.5],
            "negative": [0, -1],
            "out_of_bounds": [0, 2],
        }
        for case, values in invalid_cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                task_dir, meta_path = self.write_artifacts(
                    Path(temp_dir),
                    expression=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    expression_row_index=values,
                )
                with self.assertRaises(ValueError):
                    FastTrainingReadyArtifacts.load(task_dir, meta_path)


if __name__ == "__main__":
    unittest.main()
