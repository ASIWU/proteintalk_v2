from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "utils/34_build_update0819_ood_training_ready.py"
SPEC = importlib.util.spec_from_file_location("exp34_builder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
exp34 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp34)


def test_exp34_id_is_canonical_structure_stable() -> None:
    assert exp34.exp34_pert_id("CCO") == exp34.exp34_pert_id("OCC")
    assert exp34.exp34_pert_id("CCO").startswith("exp34smiles::")


def test_cell_alias_normalization_uses_model_alias_shape() -> None:
    assert exp34.canonical_text("AsPC-1") == "ASPC_1"
    assert exp34.canonical_text("ASPC1") == "ASPC1"


def test_similarity_transfer_changes_target_list_only() -> None:
    meta = {
        "pertid_to_smiles": {"known": "CCO", "no": ""},
        "pertid_to_target_protein_list": {"known": [7, 9], "no": []},
    }
    query = pd.DataFrame(
        [
            {
                "drug_name": "query",
                "drug_name2": "query-alias",
                "pert_id": "exp34smiles::query",
                "smiles": "OCC",
            }
        ]
    )
    audit = exp34.audit_drug_similarity(query_drugs=query, meta=meta, threshold=0.5)
    row = audit.iloc[0]
    assert row["maximum_morgan_tanimoto"] == 1.0
    assert row["transferred_target_indices"] == "[7,9]"
    assert bool(row["only_target_list_transferred"])


def test_similarity_below_threshold_does_not_transfer_target() -> None:
    meta = {
        "pertid_to_smiles": {"known": "CCN", "no": ""},
        "pertid_to_target_protein_list": {"known": [7], "no": []},
    }
    query = pd.DataFrame(
        [
            {
                "drug_name": "query",
                "drug_name2": "query-alias",
                "pert_id": "exp34smiles::query",
                "smiles": "CCO",
            }
        ]
    )
    audit = exp34.audit_drug_similarity(query_drugs=query, meta=meta, threshold=1.0)
    row = audit.iloc[0]
    assert row["threshold_candidate_count"] == 0
    assert row["transferred_target_indices"] == "[]"


def test_tied_identical_structures_with_conflicting_targets_fail() -> None:
    meta = {
        "pertid_to_smiles": {"known_a": "CCO", "known_b": "OCC"},
        "pertid_to_target_protein_list": {"known_a": [7], "known_b": [9]},
    }
    query = pd.DataFrame(
        [
            {
                "drug_name": "query",
                "drug_name2": "query-alias",
                "pert_id": "exp34smiles::query",
                "smiles": "CCO",
            }
        ]
    )
    with pytest.raises(ValueError, match="disagree on targets"):
        exp34.audit_drug_similarity(query_drugs=query, meta=meta, threshold=0.5)


def test_mechanism_target_contract() -> None:
    assert exp34.MECHANISM_TARGETS == {
        "daraxonrasib": ["P01111", "P01112", "P01116"],
        "zoldonrasib": ["P01116"],
    }
