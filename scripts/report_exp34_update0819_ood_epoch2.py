#!/usr/bin/env python3
"""Validate and report Exp34 update-0819 OOD single-drug predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = "20260819_exp34_update0819_ood_epoch2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/2026-08/2026-08-19"
DEFAULT_TRAINING_READY_ROOT = Path("/tmp/proteintalk_exp34_update0819_ood_runtime")
EXPECTED_CHECKPOINT_SHA256 = "7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076"
EXPECTED_AXIS_SIZE = 11_092
TASKS = {
    "target_none": "ptv3_exp34_update0819_ood_target_none",
    "target_mechanism": "ptv3_exp34_update0819_ood_target_mechanism",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_new(paths: list[Path], *, allow_existing: bool) -> None:
    if allow_existing:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace Exp34 report outputs: {existing}")


def validate_task(
    *,
    scenario: str,
    task_name: str,
    run_dir: Path,
    training_ready_root: Path,
    smoke: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[int], dict[str, Any]]:
    prediction_dir = run_dir / task_name
    task_dir = training_ready_root / "ptv3/tasks" / task_name
    prediction_path = prediction_dir / "predictions.parquet"
    expression_path = prediction_dir / "expression_pred.npy"
    manifest_path = prediction_dir / "run_manifest.json"
    metrics_path = prediction_dir / "metrics.json"
    table_path = task_dir / "feature_table.parquet"
    matrix_path = task_dir / "feature_expression_matrix.npy"
    for path in (prediction_path, expression_path, manifest_path, metrics_path, table_path, matrix_path):
        require(path.exists(), f"missing Exp34 artifact: {path}")

    manifest = load_json(manifest_path)
    require(manifest.get("task_name") == task_name, f"{task_name}: task mismatch")
    require(manifest.get("task_head") == "unified", f"{task_name}: task head mismatch")
    require(manifest.get("split_strategy") == "test_only", f"{task_name}: split mismatch")
    require(manifest.get("split_name") == "test", f"{task_name}: split name mismatch")
    require(manifest.get("save_expression_pred") is True, f"{task_name}: expression not saved")
    require(
        manifest.get("limit_batches") == (1 if smoke else None),
        f"{task_name}: limit_batches mismatch",
    )
    require(
        sha256_file(manifest["checkpoint_path"]) == EXPECTED_CHECKPOINT_SHA256,
        f"{task_name}: checkpoint SHA mismatch",
    )
    validation = manifest.get("checkpoint_config_validation") or {}
    require(validation.get("architecture_matches") is True, f"{task_name}: architecture mismatch")
    expected_mismatch_keys = {
        "meta_path",
        "drug_embedding_path",
        "pdi_matrix_path",
        "ddi_matrix_path",
    }
    mismatches = manifest.get("checkpoint_config_mismatches") or []
    require(
        {str(item.get("key")) for item in mismatches} == expected_mismatch_keys,
        f"{task_name}: unexpected checkpoint mismatch set",
    )
    require(
        validation.get("mismatch_count") == len(expected_mismatch_keys)
        and all(item.get("category") == "path" for item in mismatches),
        f"{task_name}: checkpoint mismatches are not the four expected paths",
    )
    require(manifest.get("protein_axis_matches_checkpoint") is True, f"{task_name}: axis mismatch")
    require(
        (manifest.get("cell_llm_summary") or {}).get("index_column") == "cell_llm_index",
        f"{task_name}: Cell LLM index mismatch",
    )
    require(
        (manifest.get("cell_type_llm_summary") or {}).get("index_column")
        == "cell_type_llm_index",
        f"{task_name}: cell-type LLM index mismatch",
    )

    predictions = pd.read_parquet(prediction_path)
    expected_rows = 2 if smoke else 14
    require(len(predictions) == expected_rows, f"{task_name}: expected {expected_rows} predictions")
    probability = pd.to_numeric(predictions["pred_task_prob"], errors="raise").to_numpy(
        dtype=np.float64
    )
    require(np.isfinite(probability).all(), f"{task_name}: non-finite probability")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), f"{task_name}: invalid probability")
    for alias in ("pred_response_prob", "pred_synergy_prob"):
        values = pd.to_numeric(predictions[alias], errors="raise").to_numpy(dtype=np.float64)
        require(np.array_equal(values, probability), f"{task_name}: unified alias mismatch")

    table = pd.read_parquet(table_path)
    rows = table.iloc[predictions["feature_row_index"].astype(int).tolist()].reset_index(drop=True)
    require(rows["sample_id"].astype(str).tolist() == predictions["sample_id"].astype(str).tolist(), f"{task_name}: row alignment mismatch")
    require(rows["target_scenario"].eq(scenario).all(), f"{task_name}: target scenario mismatch")
    require(rows["ood_drug"].astype(bool).all(), f"{task_name}: non-OOD query found")
    require((rows["pert_index1"] == rows["pert_index2"]).all(), f"{task_name}: single-drug slot mismatch")
    require(rows["pert_time_norm"].astype(str).eq("24").all(), f"{task_name}: time mismatch")
    require(rows["pert_dose1_norm"].astype(str).eq("10").all(), f"{task_name}: dose mismatch")
    if scenario == "target_none":
        require(rows["target_protein_list"].astype(str).eq("[]").all(), "target_none contains targets")

    expression = np.asarray(np.load(expression_path), dtype=np.float32)
    require(expression.shape == (expected_rows, EXPECTED_AXIS_SIZE), f"{task_name}: expression shape mismatch")
    require(np.isfinite(expression).all(), f"{task_name}: non-finite expression prediction")
    feature_matrix = np.asarray(np.load(matrix_path), dtype=np.float32)
    sample_to_row = {str(item): index for index, item in enumerate(table["sample_id"].astype(str))}
    control_raw = np.stack(
        [feature_matrix[sample_to_row[str(control_id)]] for control_id in rows["control"]]
    ).astype(np.float32)
    require(control_raw.shape == expression.shape, f"{task_name}: control matrix shape mismatch")
    control = np.nan_to_num(control_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    require(np.isfinite(control).all(), f"{task_name}: sanitized control is non-finite")
    uniprot = [str(item) for item in load_json(task_dir / "feature_ordered_protein_uniprot.json")]
    protein_index = [int(item) for item in load_json(task_dir / "feature_ordered_protein_index.json")]
    require(len(uniprot) == EXPECTED_AXIS_SIZE == len(protein_index), f"{task_name}: axis length mismatch")

    combined = rows[
        [
            "input_row_index",
            "Cell_name_input",
            "cell_in_ptvdrug_input",
            "Cell",
            "cell_type",
            "drug_name",
            "drug_name2",
            "pert_id1",
            "pert_index1",
            "smiles",
            "target_protein_list",
            "target_scenario",
            "maximum_morgan_tanimoto",
            "source_checkpoint_sample_id",
            "source_checkpoint_feature_row_index",
            "machineID_new",
            "Cell_plate",
            "batch",
            "pert_time_norm",
            "pert_dose1_norm",
        ]
    ].copy()
    combined["input_row_index"] = pd.to_numeric(
        combined["input_row_index"], errors="raise"
    ).astype(np.int64)
    combined["sample_id"] = predictions["sample_id"].astype(str).tolist()
    combined["pred_task_prob"] = probability
    combined["pred_response_prob"] = probability
    combined["prediction_task"] = task_name
    combined["expression_pred_path"] = str(expression_path.resolve())
    combined["expression_row"] = np.arange(len(combined), dtype=np.int64)
    return combined, expression, control, uniprot, protein_index, manifest


def top_expression_changes(
    *, rows: pd.DataFrame, expression: np.ndarray, control: np.ndarray, uniprot: list[str], top_k: int = 20
) -> pd.DataFrame:
    delta = expression - control
    output: list[dict[str, Any]] = []
    for row_index, row in rows.reset_index(drop=True).iterrows():
        order = np.argsort(np.abs(delta[row_index]))[::-1][:top_k]
        for rank, column in enumerate(order, start=1):
            output.append(
                {
                    "target_scenario": row["target_scenario"],
                    "input_row_index": int(row["input_row_index"]),
                    "Cell": row["Cell"],
                    "drug_name": row["drug_name"],
                    "rank": rank,
                    "uniprot": uniprot[int(column)],
                    "predicted_expression": float(expression[row_index, column]),
                    "control_expression": float(control[row_index, column]),
                    "predicted_delta": float(delta[row_index, column]),
                }
            )
    return pd.DataFrame(output)


def report(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    prefix = str(args.prefix)
    run_dir = Path(args.run_dir) if args.run_dir else output_dir / prefix
    training_ready_root = Path(args.training_ready_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions_csv": output_dir / f"{prefix}_predictions.csv",
        "predictions_parquet": output_dir / f"{prefix}_predictions.parquet",
        "comparison_csv": output_dir / f"{prefix}_target_branch_comparison.csv",
        "comparison_parquet": output_dir / f"{prefix}_target_branch_comparison.parquet",
        "top_changes_csv": output_dir / f"{prefix}_expression_top_changes.csv",
        "branch_top_csv": output_dir / f"{prefix}_target_branch_expression_top_differences.csv",
        "summary_json": output_dir / f"{prefix}_summary.json",
        "results_md": output_dir / f"{prefix}_results.md",
        "similarity_audit": output_dir / f"{prefix}_drug_similarity_audit.csv",
        "control_audit": output_dir / f"{prefix}_fixed_control_audit.csv",
        "drug_registry": output_dir / f"{prefix}_drug_registry.csv",
        "build_summary": output_dir / f"{prefix}_build_summary.json",
        "axis_uniprot": output_dir / f"{prefix}_expression_axis_uniprot.json",
        "axis_index": output_dir / f"{prefix}_expression_axis_protein_index.json",
    }
    ensure_new(list(paths.values()), allow_existing=bool(args.allow_existing))

    scenario_data: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    manifests: dict[str, Any] = {}
    uniprot_axis: list[str] | None = None
    protein_index_axis: list[int] | None = None
    top_parts: list[pd.DataFrame] = []
    for scenario, task_name in TASKS.items():
        rows, expression, control, uniprot, protein_index, manifest = validate_task(
            scenario=scenario,
            task_name=task_name,
            run_dir=run_dir,
            training_ready_root=training_ready_root,
            smoke=bool(args.smoke),
        )
        if uniprot_axis is None:
            uniprot_axis = uniprot
            protein_index_axis = protein_index
        else:
            require(uniprot_axis == uniprot and protein_index_axis == protein_index, "branch axes differ")
        scenario_data[scenario] = (rows, expression, control)
        manifests[scenario] = manifest
        top_parts.append(
            top_expression_changes(
                rows=rows, expression=expression, control=control, uniprot=uniprot
            )
        )

    combined = pd.concat([scenario_data[item][0] for item in TASKS], ignore_index=True)
    key = ["input_row_index", "Cell", "drug_name"]
    none_rows, none_expression, _ = scenario_data["target_none"]
    mechanism_rows, mechanism_expression, _ = scenario_data["target_mechanism"]
    require(none_rows[key].equals(mechanism_rows[key]), "target branches are not row-aligned")
    comparison = none_rows[
        key
        + [
            "Cell_name_input",
            "cell_in_ptvdrug_input",
            "drug_name2",
            "pert_id1",
            "maximum_morgan_tanimoto",
            "source_checkpoint_sample_id",
        ]
    ].copy()
    comparison["target_none_list"] = none_rows["target_protein_list"].tolist()
    comparison["target_mechanism_list"] = mechanism_rows["target_protein_list"].tolist()
    comparison["target_none_prob"] = none_rows["pred_task_prob"].to_numpy(dtype=np.float64)
    comparison["target_mechanism_prob"] = mechanism_rows["pred_task_prob"].to_numpy(dtype=np.float64)
    comparison["target_probability_delta"] = (
        comparison["target_mechanism_prob"] - comparison["target_none_prob"]
    )
    expression_difference = mechanism_expression - none_expression
    comparison["target_expression_difference_l2"] = np.linalg.norm(
        expression_difference, axis=1
    )
    comparison["target_expression_difference_mean_abs"] = np.mean(
        np.abs(expression_difference), axis=1
    )

    branch_top: list[dict[str, Any]] = []
    assert uniprot_axis is not None and protein_index_axis is not None
    for row_index, row in comparison.iterrows():
        order = np.argsort(np.abs(expression_difference[row_index]))[::-1][:20]
        for rank, column in enumerate(order, start=1):
            branch_top.append(
                {
                    "input_row_index": int(row["input_row_index"]),
                    "Cell": row["Cell"],
                    "drug_name": row["drug_name"],
                    "rank": rank,
                    "uniprot": uniprot_axis[int(column)],
                    "mechanism_minus_none_expression": float(
                        expression_difference[row_index, column]
                    ),
                }
            )

    combined.to_csv(paths["predictions_csv"], index=False)
    combined.to_parquet(paths["predictions_parquet"], index=False)
    comparison.to_csv(paths["comparison_csv"], index=False)
    comparison.to_parquet(paths["comparison_parquet"], index=False)
    pd.concat(top_parts, ignore_index=True).to_csv(paths["top_changes_csv"], index=False)
    pd.DataFrame(branch_top).to_csv(paths["branch_top_csv"], index=False)
    dump_json(paths["axis_uniprot"], uniprot_axis)
    dump_json(paths["axis_index"], protein_index_axis)

    group = training_ready_root / "ptv3"
    for source_name, destination in (
        ("exp34_update0819_drug_similarity_audit.csv", paths["similarity_audit"]),
        ("exp34_update0819_fixed_control_audit.csv", paths["control_audit"]),
        ("exp34_update0819_drug_registry.csv", paths["drug_registry"]),
        ("exp34_update0819_ood_build_summary.json", paths["build_summary"]),
    ):
        shutil.copy2(group / source_name, destination)

    summary = {
        "prefix": prefix,
        "smoke": bool(args.smoke),
        "run_dir": str(run_dir.resolve()),
        "runtime_training_ready_root": str(training_ready_root.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "rows": {
            "combined_predictions": int(len(combined)),
            "branch_comparison": int(len(comparison)),
        },
        "fixed_settings": {
            "pert_time_hours": 24,
            "dose_uM": 10,
            "control_policy": "minimum checkpoint feature_row_index per Cell",
            "all_drugs_ood": True,
            "target_only_sensitivity_analysis": True,
            "new_pdi_rows_all_zero": True,
        },
        "probability_ranges": {
            scenario: {
                "min": float(data[0]["pred_task_prob"].min()),
                "max": float(data[0]["pred_task_prob"].max()),
                "mean": float(data[0]["pred_task_prob"].mean()),
            }
            for scenario, data in scenario_data.items()
        },
        "target_probability_delta": {
            "min": float(comparison["target_probability_delta"].min()),
            "max": float(comparison["target_probability_delta"].max()),
            "mean": float(comparison["target_probability_delta"].mean()),
            "mean_abs": float(comparison["target_probability_delta"].abs().mean()),
        },
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "notes": [
            "Inference-only OOD predictions; no labels are available.",
            "AUROC/AUPRC and expression reconstruction metrics are not valid for this run.",
            "Both branches share Morgan, DDI, zero-PDI and graph features; only target_protein_list differs.",
        ],
    }
    dump_json(paths["summary_json"], summary)
    markdown = f"""# Exp34 Update-0819 OOD Single-Drug Inference

- Checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA256}`
- Conditions: `24 h`, `10 μM`, single-drug same-slot encoding
- Control policy: minimum checkpoint `feature_row_index` per Cell
- Prediction rows: {len(combined)} ({len(comparison)} cell-drug pairs × 2 target scenarios)
- Drug policy: both drugs remain new OOD IDs with newly generated Morgan/DDI/graph features
- PDI policy: both new PDI rows remain zero in both scenarios

## Target sensitivity

- `target_none`: no historical target transferred because both maximum Tanimoto values are below 0.5.
- `target_mechanism`: daraxonrasib uses KRAS/NRAS/HRAS; zoldonrasib uses KRAS.
- Mean probability difference (`mechanism - none`): {comparison['target_probability_delta'].mean():.6f}
- Mean absolute probability difference: {comparison['target_probability_delta'].abs().mean():.6f}

## Interpretation

These are chemical-OOD extrapolations without ground-truth labels. AUROC/AUPRC and the automatically emitted expression reconstruction metrics are not scientifically valid for this run. Use the branch delta as a target-encoding sensitivity check, not as an uncertainty calibration.
"""
    paths["results_md"].write_text(markdown, encoding="utf-8")
    print(f"[exp34-report] predictions: {paths['predictions_parquet']}")
    print(f"[exp34-report] comparison: {paths['comparison_csv']}")
    print(f"[exp34-report] summary: {paths['summary_json']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--training-ready-root", default=str(DEFAULT_TRAINING_READY_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report(parse_args())
