#!/usr/bin/env python3
"""Build an unambiguous Exp32 drug-ID to raw-metadata mapping.

The raw inputs contain three differently named SMILES fields.  In particular,
the lower-case ``smiles`` field is a legacy raw representation and is not a
promise of either chiral or non-chiral normalization.  This utility therefore
keeps that field explicitly labelled as legacy and separately publishes the
exact canonical SMILES used by Exp32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = (
    REPO_ROOT
    / "outputs/2026-07/2026-07-20"
    / "20260720_exp32_organoid_exp09_epoch2_single_sensitivity"
)
DEFAULT_AUDIT_MAPPING = Path(f"{DEFAULT_PREFIX}_drug_id_mapping_audit.csv")
DEFAULT_SCOPE = REPO_ROOT / "data/training_ready_exp32_organoid/ptv3/exp32_organoid_drug_scope.csv"
DEFAULT_PREDICTIONS = Path(f"{DEFAULT_PREFIX}_predictions.csv")
DEFAULT_STANDARDIZED_ROOT = REPO_ROOT / "data/standardized/ptv3/tasks"


TASK_SPECS = [
    {
        "task": "ptv3_main_singledrug",
        "kind": "main_single",
        "id_columns": ["pert_id"],
        "name_columns": ["drugname"],
        "legacy_smiles_columns": ["smiles"],
        "no_chiral_columns": ["Smiles_no_chiral"],
        "with_chiral_columns": ["Smiles_with_chiral"],
    },
    {
        "task": "ptv3_main_doubledrug",
        "kind": "main_double",
        "id_columns": ["pert_id1", "pert_id2"],
        "name_columns": ["pert_name"],
        "legacy_smiles_columns": [],
        "no_chiral_columns": ["Smiles1_no_chiral", "Smiles2_no_chiral"],
        "with_chiral_columns": ["Smiles1_with_chiral", "Smiles2_with_chiral"],
    },
]

for task in (
    "ptv3_extra_singledrug_mat1_480_faims",
    "ptv3_extra_singledrug_mat1_qe",
    "ptv3_extra_singledrug_mat2_480_faims",
    "ptv3_extra_singledrug_mat2_qe",
    "ptv3_extra_singledrug_mat3_qe",
    "ptv3_extra_singledrug_mat4_qe",
):
    TASK_SPECS.append(
        {
            "task": task,
            "kind": "extra_single",
            "id_columns": ["drug_ID"],
            "name_columns": ["drug_name"],
            "legacy_smiles_columns": ["smiles"],
            "no_chiral_columns": ["Smiles_no_chiral"],
            "with_chiral_columns": ["Smiles_with_chiral"],
        }
    )

TASK_SPECS.extend(
    [
        {
            "task": "ptv3_extra_doubledrug_guomics",
            "kind": "extra_double_guomics",
            "id_columns": ["pert_id1", "pert_id2"],
            "name_columns": ["Anchor_name", "Library_name"],
            "legacy_smiles_columns": ["smiles1", "smiles2"],
            "no_chiral_columns": ["Smiles1_no_chiral", "Smiles2_no_chiral"],
            "with_chiral_columns": ["Smiles1_with_chiral", "Smiles2_with_chiral"],
        },
        {
            "task": "ptv3_extra_doubledrug_nc",
            "kind": "extra_double_nc",
            "id_columns": [],
            "name_columns": ["anchor_name", "library_name"],
            "legacy_smiles_columns": ["smiles1", "smiles2"],
            "no_chiral_columns": ["Smiles1_no_chiral", "Smiles2_no_chiral"],
            "with_chiral_columns": ["Smiles1_with_chiral", "Smiles2_with_chiral"],
        },
        {
            "task": "ptv3_extra_doubledrug_nature",
            "kind": "extra_double_nature",
            "id_columns": ["anchor_ID", "lib_ID"],
            "name_columns": ["Anchor.Name", "Library.Name"],
            "legacy_smiles_columns": ["smiles1", "smiles2"],
            "no_chiral_columns": ["Smiles1_no_chiral", "Smiles2_no_chiral"],
            "with_chiral_columns": ["Smiles1_with_chiral", "Smiles2_with_chiral"],
        },
    ]
)


def parse_json_list(value: object) -> list[str]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON list, found {type(parsed).__name__}")
    return [str(item).strip() for item in parsed if str(item).strip()]


def first_json_value(value: object) -> str:
    values = parse_json_list(value)
    return values[0] if values else ""


def read_id_set(path: Path, column: str) -> set[str]:
    frame = pd.read_csv(path, usecols=[column], dtype=str, keep_default_na=False)
    return set(frame[column].str.strip())


def build_mapping(audit: pd.DataFrame) -> pd.DataFrame:
    match_fields: list[str] = []
    for row in audit.itertuples(index=False):
        exact = str(row.exp32_smiles).strip()
        if exact == str(row.Smiles_with_chiral).strip():
            match_fields.append("raw_Smiles_with_chiral")
        elif exact == str(row.Smiles_no_chiral).strip():
            match_fields.append("raw_Smiles_no_chiral")
        elif exact == str(row.smiles).strip():
            match_fields.append("raw_legacy_smiles")
        else:
            candidates = {
                "raw_Smiles_with_chiral": parse_json_list(row.Smiles_with_chiral_all),
                "raw_Smiles_no_chiral": parse_json_list(row.Smiles_no_chiral_all),
                "raw_legacy_smiles": parse_json_list(row.smiles_all),
            }
            matches = [name for name, values in candidates.items() if exact in values]
            if not matches:
                raise ValueError(f"{row.drug_id}: Exp32 SMILES is absent from all raw SMILES fields")
            match_fields.append("|".join(matches))

    mapping = pd.DataFrame(
        {
            "drug_id": audit["drug_id"],
            "raw_drug_id": audit["raw_id_all"].map(first_json_value),
            "raw_drug_name": audit["drug_name"],
            "raw_legacy_smiles": audit["smiles"],
            "raw_Smiles_no_chiral": audit["Smiles_no_chiral"],
            "raw_Smiles_with_chiral": audit["Smiles_with_chiral"],
            "exp32_model_smiles": audit["exp32_smiles"],
            "exp32_model_smiles_exact_raw_field_match": match_fields,
            "raw_source_tasks": audit["raw_source_tasks"],
            "raw_source_files": audit["raw_source_files"],
            "raw_record_count": pd.to_numeric(audit["raw_record_count"], errors="raise").astype(int),
        }
    )
    return mapping


def build_source_inventory(standardized_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in TASK_SPECS:
        info_path = standardized_root / str(spec["task"]) / "info.csv"
        info = pd.read_csv(
            info_path,
            usecols=["source_file_info"],
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
        sources = sorted({item.strip() for item in info["source_file_info"] if item.strip()})
        if len(sources) != 1:
            raise ValueError(f"{spec['task']}: expected exactly one raw source, found {sources}")
        raw_path = REPO_ROOT / sources[0]
        header = pd.read_csv(raw_path, nrows=0).columns.tolist()
        first_column = str(header[0])
        raw_rows = len(pd.read_csv(raw_path, usecols=[first_column], low_memory=False))
        if raw_rows != len(info):
            raise ValueError(f"{spec['task']}: standardized/raw row mismatch {len(info)} != {raw_rows}")

        requested_columns = [
            *spec["id_columns"],
            *spec["name_columns"],
            *spec["legacy_smiles_columns"],
            *spec["no_chiral_columns"],
            *spec["with_chiral_columns"],
        ]
        missing = [column for column in requested_columns if column not in header]
        if missing:
            raise ValueError(f"{raw_path}: expected raw columns are missing: {missing}")
        rows.append(
            {
                "source_task": spec["task"],
                "source_kind": spec["kind"],
                "raw_source_file": sources[0],
                "raw_row_count": raw_rows,
                "raw_id_columns_used": "|".join(spec["id_columns"]),
                "raw_drug_name_columns": "|".join(spec["name_columns"]),
                "raw_legacy_smiles_columns": "|".join(spec["legacy_smiles_columns"]),
                "raw_no_chiral_columns": "|".join(spec["no_chiral_columns"]),
                "raw_with_chiral_columns": "|".join(spec["with_chiral_columns"]),
                "standardization_priority": "Smiles*_with_chiral > smiles* > Smiles*_no_chiral",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-mapping", type=Path, default=DEFAULT_AUDIT_MAPPING)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--standardized-root", type=Path, default=DEFAULT_STANDARDIZED_ROOT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    args = parser.parse_args()

    audit = pd.read_csv(args.audit_mapping, dtype=str, keep_default_na=False, low_memory=False)
    if len(audit) != 3217 or audit["drug_id"].nunique() != 3217:
        raise ValueError("audit mapping must contain exactly 3,217 unique drug IDs")
    audit_ids = set(audit["drug_id"].str.strip())
    if audit_ids != read_id_set(args.scope, "pert_id"):
        raise ValueError("audit-mapping drug IDs differ from the Exp32 scope")
    if audit_ids != read_id_set(args.predictions, "drug_id"):
        raise ValueError("audit-mapping drug IDs differ from the Exp32 predictions")

    mapping = build_mapping(audit)
    inventory = build_source_inventory(args.standardized_root)
    mapping_path = Path(f"{args.output_prefix}_drug_id_rawdata_mapping.csv")
    inventory_path = Path(f"{args.output_prefix}_drug_id_rawdata_source_inventory.csv")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    print(f"mapping={mapping_path}")
    print(f"mapping_rows={len(mapping)} mapping_columns={len(mapping.columns)}")
    print(f"source_inventory={inventory_path}")
    print(f"source_rows={len(inventory)}")
    print("model_smiles_match_fields=" + json.dumps(mapping[
        "exp32_model_smiles_exact_raw_field_match"
    ].value_counts().to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
