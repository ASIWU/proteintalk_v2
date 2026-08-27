#!/usr/bin/env python3
"""Validate, export, and regression-check Exp35 manual-target inference."""

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
DEFAULT_PREFIX = "20260821_exp35_update0821_target_epoch2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/2026-08/2026-08-21"
DEFAULT_RUNTIME_ROOT = Path("/tmp/proteintalk_exp35_update0821_target_runtime")
DEFAULT_OLD_ROOT = (
    REPO_ROOT / "outputs/2026-08/2026-08-19/20260819_exp34_update0819_ood_epoch2"
)
DEFAULT_OLD_OUTPUT_DIR = REPO_ROOT / "outputs/2026-08/2026-08-19"
TASK_NAME = "ptv3_exp35_update0821_manual_target"
OLD_TASK_NAME = "ptv3_exp34_update0819_ood_target_mechanism"
EXPECTED_CHECKPOINT_SHA256 = "7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076"
EXPECTED_AXIS_SIZE = 11_092
DECISION_THRESHOLD = 0.5
ARTIFACT_STEM = "exp35_update0821_target"


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
        raise FileExistsError(f"refusing to replace Exp35 outputs: {existing}")


def validate_run(
    *, run_dir: Path, runtime_root: Path, smoke: bool
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, list[str], list[int], dict[str, Any]]:
    prediction_dir = run_dir / TASK_NAME
    task_dir = runtime_root / "ptv3/tasks" / TASK_NAME
    paths = {
        "predictions": prediction_dir / "predictions.parquet",
        "expression": prediction_dir / "expression_pred.npy",
        "run_manifest": prediction_dir / "run_manifest.json",
        "metrics": prediction_dir / "metrics.json",
        "feature_table": task_dir / "feature_table.parquet",
        "feature_expression": task_dir / "feature_expression_matrix.npy",
        "axis_uniprot": task_dir / "feature_ordered_protein_uniprot.json",
        "axis_index": task_dir / "feature_ordered_protein_index.json",
    }
    for path in paths.values():
        require(path.exists(), f"missing Exp35 artifact: {path}")

    manifest = load_json(paths["run_manifest"])
    require(manifest.get("task_name") == TASK_NAME, "task name mismatch")
    require(manifest.get("task_head") == "unified", "task head mismatch")
    require(manifest.get("split_strategy") == "test_only", "split strategy mismatch")
    require(manifest.get("split_name") == "test", "split name mismatch")
    require(manifest.get("save_expression_pred") is True, "expression output was not saved")
    require(
        manifest.get("limit_batches") == (1 if smoke else None), "limit_batches mismatch"
    )
    require(
        sha256_file(manifest["checkpoint_path"]) == EXPECTED_CHECKPOINT_SHA256,
        "checkpoint SHA mismatch",
    )
    validation = manifest.get("checkpoint_config_validation") or {}
    require(validation.get("architecture_matches") is True, "checkpoint architecture mismatch")
    require(manifest.get("protein_axis_matches_checkpoint") is True, "protein axis mismatch")

    predictions = pd.read_parquet(paths["predictions"])
    expected_rows = 2 if smoke else 14
    require(len(predictions) == expected_rows, f"expected {expected_rows} predictions")
    probability = pd.to_numeric(predictions["pred_task_prob"], errors="raise").to_numpy(
        dtype=np.float64
    )
    require(np.isfinite(probability).all(), "prediction probability contains non-finite values")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), "probability outside [0,1]")

    table = pd.read_parquet(paths["feature_table"])
    feature_rows = predictions["feature_row_index"].astype(int).tolist()
    rows = table.iloc[feature_rows].reset_index(drop=True)
    require(
        rows["sample_id"].astype(str).tolist() == predictions["sample_id"].astype(str).tolist(),
        "prediction-feature row alignment mismatch",
    )
    require(rows["target_scenario"].astype(str).eq("manual_target").all(), "scenario mismatch")
    require(rows["ood_drug"].astype(bool).all(), "non-OOD query found")
    require((rows["pert_index1"] == rows["pert_index2"]).all(), "single-drug slot mismatch")
    require(rows["pert_time_norm"].astype(str).eq("24").all(), "time mismatch")
    require(rows["pert_dose1_norm"].astype(str).eq("10").all(), "dose mismatch")
    expected_targets = {
        "daraxonrasib": "[1374,1375,1376]",
        "zoldonrasib": "[1376]",
    }
    for drug_name, target_list in expected_targets.items():
        require(
            rows.loc[rows["drug_name"].astype(str).eq(drug_name), "target_protein_list"]
            .astype(str)
            .eq(target_list)
            .all(),
            f"{drug_name}: target list mismatch",
        )

    expression = np.asarray(np.load(paths["expression"]), dtype=np.float32)
    require(expression.shape == (expected_rows, EXPECTED_AXIS_SIZE), "expression shape mismatch")
    require(np.isfinite(expression).all(), "predicted expression contains non-finite values")
    feature_expression = np.asarray(np.load(paths["feature_expression"]), dtype=np.float32)
    sample_to_row = {str(value): index for index, value in enumerate(table["sample_id"].astype(str))}
    control_feature_rows = np.asarray(
        [sample_to_row[str(control_id)] for control_id in rows["control"]], dtype=np.int64
    )
    control_raw = feature_expression[control_feature_rows].astype(np.float32)
    control = np.nan_to_num(control_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    require(control.shape == expression.shape, "matched control shape mismatch")
    require(np.isfinite(control).all(), "sanitized matched control is non-finite")

    uniprot = [str(item) for item in load_json(paths["axis_uniprot"])]
    protein_index = [int(item) for item in load_json(paths["axis_index"])]
    require(len(uniprot) == len(protein_index) == EXPECTED_AXIS_SIZE, "axis length mismatch")

    metadata_columns = [
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
    combined = rows[metadata_columns].copy()
    # The task table also contains control rows whose input_row_index is null,
    # so parquet materializes this query-only identifier as float.  Restore the
    # contractual integer type before exporting and comparing with Exp34.
    combined["input_row_index"] = pd.to_numeric(
        combined["input_row_index"], errors="raise"
    ).astype(np.int64)
    combined["sample_id"] = predictions["sample_id"].astype(str).tolist()
    combined["pred_task_prob"] = probability
    combined["pred_sensitivity_prob"] = probability
    combined["decision_threshold"] = DECISION_THRESHOLD
    combined["predicted_sensitive"] = (probability >= DECISION_THRESHOLD).astype(np.int64)
    combined["predicted_response_label"] = np.where(
        probability >= DECISION_THRESHOLD, "sensitive", "non-responsive"
    )
    combined["prediction_task"] = TASK_NAME
    combined["expression_pred_path"] = str(paths["expression"].resolve())
    combined["expression_row"] = np.arange(expected_rows, dtype=np.int64)

    control_metadata = pd.DataFrame(
        {
            "matched_control_sample_id": rows["control"].astype(str).tolist(),
            "matched_control_feature_row_index": control_feature_rows,
            "matched_control_expression_row_index": table.iloc[control_feature_rows][
                "expression_row_index"
            ].astype(int).tolist(),
            "matched_control_original_nan_count": np.isnan(control_raw).sum(axis=1).astype(int),
            "control_missing_value_policy": "nan_to_zero_as_model_input",
        }
    )
    return combined, expression, control, control_metadata, uniprot, protein_index, manifest


def wide_proteome(
    metadata: pd.DataFrame, matrix: np.ndarray, uniprot: list[str]
) -> pd.DataFrame:
    proteins = pd.DataFrame(matrix, columns=uniprot)
    return pd.concat([metadata.reset_index(drop=True), proteins], axis=1)


def regression_against_exp34(
    *, combined: pd.DataFrame, expression: np.ndarray, control: np.ndarray, uniprot: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    old_root = Path(args.old_run_root)
    old_prediction_path = Path(args.old_output_dir) / "20260819_exp34_update0819_ood_epoch2_predictions.parquet"
    old_expression_path = old_root / OLD_TASK_NAME / "expression_pred.npy"
    old_control_path = Path(args.old_output_dir) / "20260819_exp34_update0819_ood_epoch2_target_mechanism_matched_control_proteome.csv"
    for path in (old_prediction_path, old_expression_path, old_control_path):
        require(path.exists(), f"missing Exp34 regression artifact: {path}")
    old_predictions = pd.read_parquet(old_prediction_path)
    old_predictions = old_predictions.loc[
        old_predictions["target_scenario"].astype(str).eq("target_mechanism")
    ].sort_values("input_row_index").reset_index(drop=True)
    current = combined.sort_values("input_row_index").reset_index(drop=True)
    key = ["input_row_index", "Cell", "drug_name"]
    require(current[key].equals(old_predictions[key]), "Exp34 regression keys do not align")
    probability_error = np.abs(
        current["pred_sensitivity_prob"].to_numpy(dtype=np.float64)
        - old_predictions["pred_task_prob"].to_numpy(dtype=np.float64)
    )
    old_expression = np.asarray(np.load(old_expression_path), dtype=np.float32)
    require(old_expression.shape == expression.shape, "Exp34 expression shape mismatch")
    expression_error = np.abs(expression - old_expression)
    old_control = pd.read_csv(old_control_path, usecols=uniprot).to_numpy(dtype=np.float32)
    require(old_control.shape == control.shape, "Exp34 control shape mismatch")
    control_error = np.abs(control - old_control)
    probability_max = float(probability_error.max(initial=0.0))
    expression_max = float(expression_error.max(initial=0.0))
    control_max = float(control_error.max(initial=0.0))
    require(probability_max <= 1e-7, f"Exp34 probability regression failed: {probability_max}")
    require(expression_max <= 1e-6, f"Exp34 expression regression failed: {expression_max}")
    # Exp34 only retained this matrix as decimal CSV.  A float32 -> decimal ->
    # float32 round trip can move values by one ULP, so exact byte equality is
    # not a meaningful invariant for that reference artifact.
    require(control_max <= 1e-6, f"Exp34 matched-control regression failed: {control_max}")
    return {
        "reference": "20260819_exp34_update0819_ood_epoch2 target_mechanism",
        "probability_max_abs_error": probability_max,
        "expression_max_abs_error": expression_max,
        "matched_control_max_abs_error": control_max,
        "matched_control_within_1e-6": True,
    }


def report(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    prefix = str(args.prefix)
    run_dir = Path(args.run_dir) if args.run_dir else output_dir / prefix
    runtime_root = Path(args.training_ready_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions_csv": output_dir / f"{prefix}_predictions.csv",
        "predictions_parquet": output_dir / f"{prefix}_predictions.parquet",
        "perturbed_csv": output_dir / f"{prefix}_perturbed_proteome.csv",
        "control_csv": output_dir / f"{prefix}_matched_control_proteome.csv",
        "summary_json": output_dir / f"{prefix}_summary.json",
        "results_md": output_dir / f"{prefix}_results.md",
        "axis_uniprot": output_dir / f"{prefix}_expression_axis_uniprot.json",
        "axis_index": output_dir / f"{prefix}_expression_axis_protein_index.json",
        "similarity_audit": output_dir / f"{prefix}_drug_similarity_audit.csv",
        "control_audit": output_dir / f"{prefix}_fixed_control_audit.csv",
        "drug_registry": output_dir / f"{prefix}_drug_registry.csv",
        "build_summary": output_dir / f"{prefix}_build_summary.json",
    }
    ensure_new(list(paths.values()), allow_existing=bool(args.allow_existing))
    combined, expression, control, control_meta, uniprot, protein_index, manifest = validate_run(
        run_dir=run_dir, runtime_root=runtime_root, smoke=bool(args.smoke)
    )

    prediction_metadata = combined[
        [
            "target_scenario",
            "expression_row",
            "input_row_index",
            "Cell_name_input",
            "cell_in_ptvdrug_input",
            "Cell",
            "cell_type",
            "drug_name",
            "drug_name2",
            "pert_id1",
            "target_protein_list",
            "pred_sensitivity_prob",
            "predicted_response_label",
            "source_checkpoint_sample_id",
        ]
    ].copy()
    control_metadata = pd.concat(
        [
            prediction_metadata.drop(columns=["source_checkpoint_sample_id"]),
            control_meta,
        ],
        axis=1,
    )
    combined.to_csv(paths["predictions_csv"], index=False)
    combined.to_parquet(paths["predictions_parquet"], index=False)
    wide_proteome(prediction_metadata, expression, uniprot).to_csv(
        paths["perturbed_csv"], index=False
    )
    wide_proteome(control_metadata, control, uniprot).to_csv(paths["control_csv"], index=False)
    dump_json(paths["axis_uniprot"], uniprot)
    dump_json(paths["axis_index"], protein_index)

    group = runtime_root / "ptv3"
    for source_name, destination in (
        (f"{ARTIFACT_STEM}_drug_similarity_audit.csv", paths["similarity_audit"]),
        (f"{ARTIFACT_STEM}_fixed_control_audit.csv", paths["control_audit"]),
        (f"{ARTIFACT_STEM}_drug_registry.csv", paths["drug_registry"]),
        (f"{ARTIFACT_STEM}_build_summary.json", paths["build_summary"]),
    ):
        shutil.copy2(group / source_name, destination)

    regression: dict[str, Any] | None = None
    if not args.smoke:
        regression = regression_against_exp34(
            combined=combined, expression=expression, control=control, uniprot=uniprot, args=args
        )
    label_counts = combined["predicted_response_label"].value_counts().to_dict()
    summary = {
        "status": "complete",
        "prefix": prefix,
        "smoke": bool(args.smoke),
        "task_name": TASK_NAME,
        "run_dir": str(run_dir.resolve()),
        "runtime_training_ready_root": str(runtime_root.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_sha256": load_json(paths["build_summary"])["input_sha256"],
        "rows": len(combined),
        "protein_count": EXPECTED_AXIS_SIZE,
        "decision_threshold": DECISION_THRESHOLD,
        "label_semantics": "sensitive versus non-responsive; non-responsive is not a clinical resistance claim",
        "label_counts": label_counts,
        "probability": {
            "min": float(combined["pred_sensitivity_prob"].min()),
            "max": float(combined["pred_sensitivity_prob"].max()),
            "mean": float(combined["pred_sensitivity_prob"].mean()),
        },
        "matched_control_policy": "nan_to_zero_as_model_input",
        "regression_against_exp34": regression,
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "run_manifest": str((run_dir / TASK_NAME / "run_manifest.json").resolve()),
        "inference_manifest_generated_at": manifest.get("generated_at"),
    }
    dump_json(paths["summary_json"], summary)
    regression_text = "not run for smoke" if regression is None else json.dumps(regression)
    paths["results_md"].write_text(
        f"""# Exp35 Update-0821 Manual-Target Inference

- Samples: `{len(combined)}`
- Proteins: `{EXPECTED_AXIS_SIZE}`
- Checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA256}`
- Conditions: `24 h`, `10 μM`
- Target source: update-0821 CSV `target` column
- Decision threshold: `{DECISION_THRESHOLD}`
- Label counts: `{label_counts}`
- Exp34 mechanism regression: `{regression_text}`

`non-responsive` is the checkpoint's negative response class and must not be
interpreted as a validated clinical resistance call. These are unlabeled
chemical-OOD predictions.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--training-ready-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--old-run-root", default=str(DEFAULT_OLD_ROOT))
    parser.add_argument("--old-output-dir", default=str(DEFAULT_OLD_OUTPUT_DIR))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report(parse_args())
