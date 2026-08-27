#!/usr/bin/env python3
"""Build a PTV3 fast-task view with PTV1 cell_5fold labels and splits.

The derived task reuses PTV3 feature rows/expression matrices, but replaces
the response label with PTV1's cell_5fold phenotype label on matching
``(Cell, pert_id1, pert_id2)`` experiment keys. PTV1 ``-1`` labels are written
as missing values so ProteinTalk v3 masks them in the response BCE.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SOURCE_TASK = Path("data/training_ready/ptv3/tasks/ptv3_main_singledrug")
DEFAULT_SOURCE_SPLIT = Path("data/training_ready/ptv3/splits/ptv3_main_singledrug")
DEFAULT_PTV1_ROOT = Path("baseline/ptv2_benchmark_260514/data/1_4EGHv2biorep_bind_unified/cell_5fold")
DEFAULT_OUTPUT_TASK = Path("data/training_ready/ptv3/tasks/ptv3_main_singledrug_ptv1_cell_5fold")
DEFAULT_OUTPUT_SPLIT = Path("data/training_ready/ptv3/splits/ptv3_main_singledrug_ptv1_cell_5fold")


@dataclass(frozen=True)
class Ptv1Record:
    key: tuple[str, str, str]
    label: float
    experiment_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-task", type=Path, default=DEFAULT_SOURCE_TASK)
    parser.add_argument("--source-split", type=Path, default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--ptv1-root", type=Path, default=DEFAULT_PTV1_ROOT)
    parser.add_argument("--output-task", type=Path, default=DEFAULT_OUTPUT_TASK)
    parser.add_argument("--output-split", type=Path, default=DEFAULT_OUTPUT_SPLIT)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def parse_experiment_type(value: str) -> tuple[str, str, str]:
    if "_" not in value:
        raise ValueError(f"cannot parse PTV1 experiment_type without cell separator: {value!r}")
    cell, rest = value.split("_", 1)
    parts = [part.lstrip("#") for part in rest.strip().split()]
    if len(parts) < 2:
        raise ValueError(f"cannot parse PTV1 drug pair from experiment_type: {value!r}")
    return str(cell), str(parts[0]), str(parts[1])


def load_ptv1_split(ptv1_root: Path, fold: int, split: str) -> list[Ptv1Record]:
    prefix = ptv1_root / f"fold{fold}" / split / f"cell_5fold_fold{fold}_"
    loo = pd.read_csv(str(prefix) + "loo_label.csv", header=None)
    pheno = pd.read_csv(str(prefix) + "pheno.csv", header=None)
    labels = loo[0].astype(str).to_numpy()
    times = loo[1].to_numpy()

    first_index_by_experiment: dict[str, int] = {}
    times_by_experiment: dict[str, set[int]] = {}
    for row_idx, (experiment_type, time_value) in enumerate(zip(labels, times)):
        first_index_by_experiment.setdefault(experiment_type, row_idx)
        try:
            time_int = int(time_value)
        except (TypeError, ValueError):
            continue
        times_by_experiment.setdefault(experiment_type, set()).add(time_int)

    records: list[Ptv1Record] = []
    for experiment_type in sorted(times_by_experiment):
        if "#" not in experiment_type:
            continue
        if not {6, 24}.issubset(times_by_experiment[experiment_type]):
            continue
        first_idx = first_index_by_experiment[experiment_type]
        raw_label = pd.to_numeric(pd.Series([pheno.iloc[first_idx, 0]]), errors="coerce").iloc[0]
        label = float(raw_label) if pd.notna(raw_label) else -1.0
        records.append(
            Ptv1Record(
                key=parse_experiment_type(experiment_type),
                label=label,
                experiment_type=experiment_type,
            )
        )
    return records


def normalized_ptv3_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ("Cell", "pert_id1", "pert_id2"):
        if column not in result.columns:
            raise KeyError(f"source feature table is missing required column {column!r}")
        result[column] = result[column].astype(str)
    result["_ptv1_key"] = list(zip(result["Cell"], result["pert_id1"], result["pert_id2"]))
    return result


def ptv1_label_to_ptv3_label(label: float) -> str | float:
    if label == 1.0:
        return "sensitive"
    if label == 0.0:
        return "non-responsive"
    return np.nan


def build_split_set_info(
    *,
    anchor_indices: list[int],
    row_to_set: dict[int, int],
    all_set_info: dict[int, dict[str, list[int]]],
) -> dict[int, dict[str, list[int]]]:
    required_set_ids = {row_to_set[int(row_idx)] for row_idx in anchor_indices}
    return {set_id: all_set_info[set_id] for set_id in sorted(required_set_ids)}


def clean_output_dir(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    source_task = args.source_task.resolve()
    source_split = args.source_split.resolve()
    ptv1_root = args.ptv1_root.resolve()
    output_task = args.output_task.resolve()
    output_split = args.output_split.resolve()

    clean_output_dir(output_task, force=args.force)
    clean_output_dir(output_split, force=args.force)
    output_task.mkdir(parents=True, exist_ok=True)
    output_split.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(source_task / "feature_table.csv", low_memory=False).reset_index(drop=True)
    source_df = normalized_ptv3_key_frame(source_df)
    source_row_to_set = {int(key): int(value) for key, value in load_pickle(source_split / "row_to_set_index.pkl").items()}
    source_set_info_raw = load_pickle(source_split / "set_info.pkl")
    source_set_info = {
        int(key): {
            "control": [int(item) for item in value["control"]],
            "perturb": [int(item) for item in value["perturb"]],
        }
        for key, value in source_set_info_raw.items()
    }

    ptv1_by_fold_split: dict[str, dict[str, list[Ptv1Record]]] = {}
    label_by_key: dict[tuple[str, str, str], float] = {}
    experiment_by_key: dict[tuple[str, str, str], str] = {}
    split_summary: dict[str, Any] = {}
    for fold in args.folds:
        fold_key = f"fold{fold}"
        ptv1_by_fold_split[fold_key] = {}
        split_summary[fold_key] = {}
        for split in ("train", "valid", "test"):
            records = load_ptv1_split(ptv1_root, fold, split)
            ptv1_by_fold_split[fold_key][split] = records
            valid_records = [record for record in records if record.label in {0.0, 1.0}]
            split_summary[fold_key][split] = {
                "raw_experiments": len(records),
                "valid_label_experiments": len(valid_records),
                "positive_experiments": int(sum(1 for record in valid_records if record.label == 1.0)),
            }
            for record in records:
                previous = label_by_key.get(record.key)
                if previous is not None and previous != record.label:
                    raise ValueError(f"conflicting PTV1 labels for {record.key}: {previous} vs {record.label}")
                label_by_key[record.key] = record.label
                experiment_by_key.setdefault(record.key, record.experiment_type)

    perturb_rows_by_key: dict[tuple[str, str, str], list[int]] = {}
    source_perturb_rows: set[int] = set()
    for row_idx, key in enumerate(source_df["_ptv1_key"]):
        if key not in label_by_key:
            continue
        if str(source_df.at[row_idx, "pert_id1"]) == "control":
            continue
        if row_idx not in source_row_to_set:
            continue
        perturb_rows_by_key.setdefault(key, []).append(row_idx)
        source_perturb_rows.add(row_idx)

    missing_keys = sorted(set(label_by_key) - set(perturb_rows_by_key))
    if missing_keys:
        raise ValueError(f"{len(missing_keys)} PTV1 keys are missing from PTV3 features; examples={missing_keys[:10]}")

    source_control_rows: set[int] = set()
    for source_row in source_perturb_rows:
        set_idx = source_row_to_set[source_row]
        source_control_rows.update(source_set_info[set_idx]["control"])

    selected_source_rows = sorted(source_perturb_rows | source_control_rows)
    source_to_new = {source_row: new_row for new_row, source_row in enumerate(selected_source_rows)}
    selected_df = source_df.iloc[selected_source_rows].drop(columns=["_ptv1_key"]).reset_index(drop=True)
    selected_df["feature_row_index"] = np.arange(len(selected_df), dtype=np.int64)
    selected_df["expression_row_index"] = np.arange(len(selected_df), dtype=np.int64)
    selected_df["processed_row_index"] = np.arange(len(selected_df), dtype=np.int64)

    new_row_to_set: dict[int, int] = {}
    new_set_info: dict[int, dict[str, list[int]]] = {}
    for source_row in sorted(source_perturb_rows):
        new_row = source_to_new[source_row]
        source_set_idx = source_row_to_set[source_row]
        new_row_to_set[new_row] = source_set_idx
        current = new_set_info.setdefault(source_set_idx, {"control": [], "perturb": []})
        current["perturb"].append(new_row)
        for control_row in source_set_info[source_set_idx]["control"]:
            if control_row in source_to_new:
                current["control"].append(source_to_new[control_row])
    for set_idx, info in new_set_info.items():
        info["control"] = sorted(set(info["control"]))
        info["perturb"] = sorted(set(info["perturb"]))
        if not info["control"]:
            raise ValueError(f"set {set_idx} has no remapped control rows")

    for key, source_rows in perturb_rows_by_key.items():
        label_value = ptv1_label_to_ptv3_label(label_by_key[key])
        experiment_type = experiment_by_key[key]
        for source_row in source_rows:
            new_row = source_to_new[source_row]
            selected_df.at[new_row, "PRISM1st_label_total"] = label_value
            selected_df.at[new_row, "ptv1_cell_5fold_label"] = label_by_key[key]
            selected_df.at[new_row, "ptv1_cell_5fold_experiment_type"] = experiment_type

    control_mask = selected_df["pert_id1"].astype(str).eq("control")
    selected_df.loc[control_mask, "PRISM1st_label_total"] = np.nan
    selected_df.loc[control_mask, "ptv1_cell_5fold_label"] = np.nan
    selected_df.loc[control_mask, "ptv1_cell_5fold_experiment_type"] = ""

    selected_df.to_csv(output_task / "feature_table.csv", index=False)
    try:
        selected_df.to_parquet(output_task / "feature_table.parquet", index=False)
    except Exception as exc:  # pragma: no cover - optional parquet backend
        print(f"[warn] failed to write parquet feature table: {exc}")

    sample_ids = selected_df["sample_id"].astype(str).tolist()
    dump_json(output_task / "feature_sample_ids.json", sample_ids)
    for file_name in ("feature_ordered_protein_index.json", "feature_ordered_protein_uniprot.json"):
        shutil.copy2(source_task / file_name, output_task / file_name)

    expression_path = source_task / "feature_expression_matrix.npy"
    try:
        source_expression = np.load(expression_path, mmap_mode="r")
    except OSError as exc:
        print(f"[warn] mmap load failed for {expression_path}: {exc}; falling back to normal np.load")
        source_expression = np.load(expression_path)
    expression_subset = np.asarray(source_expression[selected_source_rows], dtype=np.float32)
    np.save(output_task / "feature_expression_matrix.npy", expression_subset, allow_pickle=False)

    dump_pickle(output_split / "row_to_set_index.pkl", new_row_to_set)
    dump_pickle(output_split / "set_info.pkl", new_set_info)
    split_manifest: dict[str, Any] = {
        "source_task": str(source_task),
        "source_split": str(source_split),
        "ptv1_root": str(ptv1_root),
        "task": str(output_task),
        "split": str(output_split),
        "folds": list(args.folds),
        "ptv1_summary": split_summary,
        "selected_rows": {
            "total": len(selected_source_rows),
            "perturb": len(source_perturb_rows),
            "control": len(source_control_rows),
        },
    }

    for fold in args.folds:
        strategy = f"ptv1_cell_5fold_fold{fold}"
        fold_key = f"fold{fold}"
        split_manifest[strategy] = {}
        for split in ("train", "valid", "test"):
            anchor_indices: list[int] = []
            valid_anchor_indices: list[int] = []
            for record in ptv1_by_fold_split[fold_key][split]:
                rows = [source_to_new[row] for row in perturb_rows_by_key[record.key]]
                anchor_indices.extend(rows)
                if record.label in {0.0, 1.0}:
                    valid_anchor_indices.extend(rows)
            anchor_indices = sorted(set(anchor_indices))
            valid_anchor_indices = sorted(set(valid_anchor_indices))
            dump_pickle(output_split / f"{split}_indices_{strategy}.pkl", anchor_indices)
            if split == "valid":
                dump_pickle(output_split / f"val_indices_{strategy}.pkl", anchor_indices)
            set_info = build_split_set_info(
                anchor_indices=anchor_indices,
                row_to_set=new_row_to_set,
                all_set_info=new_set_info,
            )
            dump_pickle(output_split / f"{split}_set_info_{strategy}.pkl", set_info)
            if split == "valid":
                dump_pickle(output_split / f"val_set_info_{strategy}.pkl", set_info)
            split_manifest[strategy][split] = {
                "rows": len(anchor_indices),
                "valid_label_rows": len(valid_anchor_indices),
                "positive_label_rows": int(
                    sum(
                        1
                        for row_idx in valid_anchor_indices
                        if selected_df.at[row_idx, "PRISM1st_label_total"] == "sensitive"
                    )
                ),
                "experiments": split_summary[fold_key][split]["raw_experiments"],
                "valid_label_experiments": split_summary[fold_key][split]["valid_label_experiments"],
                "positive_experiments": split_summary[fold_key][split]["positive_experiments"],
            }

    dump_json(output_split / "split_manifest.json", split_manifest)
    dump_json(
        output_task / "feature_loading_manifest.json",
        {
            "derived_from": str(source_task),
            "label_source": str(ptv1_root),
            "row_count": int(len(selected_df)),
            "expression_shape": [int(dim) for dim in expression_subset.shape],
            "sample_id_count": len(sample_ids),
        },
    )
    print(
        "Built PTV3/PTV1 cell_5fold task: "
        f"rows={len(selected_df)} perturb={len(source_perturb_rows)} controls={len(source_control_rows)} "
        f"task={output_task} split={output_split}"
    )
    for fold in args.folds:
        strategy = f"ptv1_cell_5fold_fold{fold}"
        print(strategy, split_manifest[strategy])


if __name__ == "__main__":
    main()
