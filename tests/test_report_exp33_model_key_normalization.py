#!/usr/bin/env python3
"""Regression tests for exp33 raw-to-unique model-key dtype normalization."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.report_exp33_vc_doubledrug_epoch2 import (
    KEY_COLUMNS,
    normalize_model_key_columns,
)


class Exp33ModelKeyNormalizationTest(unittest.TestCase):
    def test_string_feature_time_merges_with_integer_raw_time(self) -> None:
        lookup = pd.DataFrame(
            {
                "tissue": ["colon"],
                "source_cell": ["DLD1"],
                "pert_id1": ["101"],
                "pert_id2": ["L9200_1"],
                "pert_dose1_index": [np.int64(2)],
                "pert_dose2_index": [np.int64(7)],
                "pert_time": ["24"],
                "model_key_id": ["exp33::colon::0000000"],
                "pred_unified_combo_prob": [0.75],
            }
        )
        raw = pd.DataFrame(
            {
                "tissue": ["colon"],
                "source_cell": ["DLD1"],
                "pert_id1": ["101"],
                "pert_id2": ["L9200_1"],
                "pert_dose1_index": [2],
                "pert_dose2_index": [7],
                "pert_time": [24],
            }
        )

        normalized_lookup = normalize_model_key_columns(lookup, context="lookup")
        normalized_raw = normalize_model_key_columns(raw, context="raw")
        merged = normalized_raw.merge(
            normalized_lookup,
            on=KEY_COLUMNS,
            how="left",
            validate="many_to_one",
            sort=False,
        )

        self.assertEqual(merged.loc[0, "model_key_id"], "exp33::colon::0000000")
        self.assertEqual(float(merged.loc[0, "pred_unified_combo_prob"]), 0.75)
        for column in ("pert_dose1_index", "pert_dose2_index", "pert_time"):
            self.assertEqual(normalized_lookup[column].dtype, np.dtype("int64"))
            self.assertEqual(normalized_raw[column].dtype, np.dtype("int64"))

    def test_non_integer_numeric_key_fails_fast(self) -> None:
        frame = pd.DataFrame(
            {
                "tissue": ["lung"],
                "source_cell": ["NCIH23"],
                "pert_id1": ["1"],
                "pert_id2": ["2"],
                "pert_dose1_index": [1],
                "pert_dose2_index": [1.5],
                "pert_time": [24],
            }
        )
        with self.assertRaisesRegex(ValueError, "non-integer"):
            normalize_model_key_columns(frame, context="invalid")


if __name__ == "__main__":
    unittest.main()
