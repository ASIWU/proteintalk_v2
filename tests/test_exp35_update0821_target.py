from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils/35_build_update0821_target_training_ready.py"
INPUT_PATH = REPO_ROOT / "data/rawdata/update_0821/260820ptv_drug_cell_predict_target.csv"
SPEC = importlib.util.spec_from_file_location("exp35_builder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
exp35 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp35)


class Exp35TargetContractTest(unittest.TestCase):
    def test_target_parser_normalizes_and_deduplicates(self) -> None:
        self.assertEqual(
            exp35.parse_target_cell(" p01116; P01111 ;P01116 "),
            ["P01111", "P01116"],
        )

    def test_target_parser_rejects_empty_values(self) -> None:
        for value in ("", " ; ", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "target"):
                exp35.parse_target_cell(value)

    def test_target_parser_rejects_invalid_accession(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid UniProt"):
            exp35.parse_target_cell("KRAS; P01116")

    def test_actual_input_contract_and_targets(self) -> None:
        frame = exp35.read_and_validate_input(INPUT_PATH)
        self.assertEqual(len(frame), 14)
        self.assertFalse(frame.duplicated(["cell_in_ptvdrug", "drug_name"]).any())
        target_sets = {
            drug: json.loads(str(group["manual_target_uniprot"].iloc[0]))
            for drug, group in frame.groupby("drug_name_norm", sort=False)
        }
        self.assertEqual(
            target_sets,
            {
                "daraxonrasib": ["P01111", "P01112", "P01116"],
                "zoldonrasib": ["P01116"],
            },
        )

    def test_target_contract_maps_to_checkpoint_axis(self) -> None:
        frame = exp35.read_and_validate_input(INPUT_PATH)
        meta = exp35.base.load_json(REPO_ROOT / "data/training_ready/ptv3/global_meta.json")
        uniprot, indices = exp35.target_contract(frame, meta)
        self.assertEqual(
            uniprot,
            {
                "daraxonrasib": ["P01111", "P01112", "P01116"],
                "zoldonrasib": ["P01116"],
            },
        )
        self.assertEqual(
            indices, {"daraxonrasib": [1374, 1375, 1376], "zoldonrasib": [1376]}
        )

    def test_conflicting_drug_targets_fail(self) -> None:
        frame = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
        frame.loc[1, "target"] = "P01116"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflict.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "conflicting target sets"):
                exp35.read_and_validate_input(path)

    def test_duplicate_cell_drug_key_fails(self) -> None:
        frame = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
        frame.loc[1, ["Cell_name", "cell_in_ptvdrug"]] = frame.loc[
            0, ["Cell_name", "cell_in_ptvdrug"]
        ].to_numpy()
        # Keep the row textually distinct so the base whole-row duplicate
        # check does not mask Exp35's stricter cell-drug key check.
        frame.loc[1, "target"] = "P01111;P01112;P01116"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate cell-drug keys"):
                exp35.read_and_validate_input(path)


if __name__ == "__main__":
    unittest.main()
