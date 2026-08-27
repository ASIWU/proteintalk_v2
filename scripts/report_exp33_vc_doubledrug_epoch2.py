#!/usr/bin/env python3
"""Validate and report exp33 epoch-2 double-drug virtual-screen predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_READY_ROOT = REPO_ROOT / "data/training_ready_exp33_vc_doubledrug"
RAW_ROOT = REPO_ROOT / "data/rawdata/vc_doubledrug"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/2026-07/2026-07-27"
DEFAULT_PREFIX = "20260727_exp33_vc_doubledrug_epoch2"
EXPECTED_CHECKPOINT_SHA256 = "7a0786467279c079478a34dcd323cb61aaf6b2bbf68fc8a0be9d959d578cb076"
EXPECTED_RAW_ROWS = 2_526_720
EXPECTED_UNIQUE_ROWS = 1_124_928
KEY_COLUMNS = [
    "tissue",
    "source_cell",
    "pert_id1",
    "pert_id2",
    "pert_dose1_index",
    "pert_dose2_index",
    "pert_time",
]
TISSUE_SPECS = {
    "colon": {
        "task_name": "ptv3_exp33_vc_colon_double",
        "query_file": "PTV2_virtual_screen_colon_IC50_robustness_7col_260722.csv",
        "raw_rows": 403_200,
        "unique_rows": 218_736,
    },
    "lung": {
        "task_name": "ptv3_exp33_vc_lung_double",
        "query_file": "PTV2_virtual_screen_lung_IC50_robustness_7col_260722.csv",
        "raw_rows": 806_400,
        "unique_rows": 462_210,
    },
    "pancreas": {
        "task_name": "ptv3_exp33_vc_pancreas_double",
        "query_file": "PTV2_virtual_screen_pancreas_IC50_robustness_7col_260722.csv",
        "raw_rows": 1_317_120,
        "unique_rows": 443_982,
    },
}


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def ensure_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace existing exp33 report outputs: {existing}")


def clipped_dose_bucket(values: pd.Series, *, column: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    require(
        np.isfinite(numeric).all() and np.all(numeric >= 0),
        f"{column} contains non-finite or negative values",
    )
    return np.ceil(np.clip(numeric, 0.0, 10.0)).astype(np.int64)


def normalize_model_key_columns(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Return model-key columns with identical explicit dtypes for safe merging."""
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    require(not missing, f"{context}: missing model-key columns: {missing}")
    normalized = frame.copy()
    for column in ("tissue", "source_cell", "pert_id1", "pert_id2"):
        require(
            normalized[column].notna().all(),
            f"{context}: {column} contains missing values",
        )
        normalized[column] = normalized[column].astype(str).str.strip()
        require(
            normalized[column].ne("").all(),
            f"{context}: {column} contains empty values",
        )
    for column in ("pert_dose1_index", "pert_dose2_index", "pert_time"):
        numeric = pd.to_numeric(normalized[column], errors="raise").to_numpy(dtype=np.float64)
        require(
            np.isfinite(numeric).all(),
            f"{context}: {column} contains non-finite values",
        )
        require(
            np.equal(numeric, np.floor(numeric)).all(),
            f"{context}: {column} contains non-integer values",
        )
        normalized[column] = numeric.astype(np.int64)
    return normalized


def validate_task_predictions(
    *,
    tissue: str,
    run_dir: Path,
    training_ready_root: Path,
    smoke: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = TISSUE_SPECS[tissue]
    task_name = str(spec["task_name"])
    task_dir = training_ready_root / "ptv3/tasks" / task_name
    prediction_dir = run_dir / task_name
    prediction_path = prediction_dir / "predictions.parquet"
    manifest_path = prediction_dir / "run_manifest.json"
    metrics_path = prediction_dir / "metrics.json"
    for path in (prediction_path, manifest_path, metrics_path, task_dir / "feature_table.parquet"):
        require(path.exists(), f"missing exp33 task artifact: {path}")

    manifest = load_json(manifest_path)
    require(manifest.get("task_name") == task_name, f"{task_name}: manifest task mismatch")
    require(manifest.get("task_head") == "unified", f"{task_name}: task_head is not unified")
    require(manifest.get("split_strategy") == "test_only", f"{task_name}: split is not test_only")
    require(manifest.get("split_name") == "test", f"{task_name}: split_name is not test")
    require(manifest.get("save_expression_pred") is False, f"{task_name}: expression predictions were saved")
    require(
        manifest.get("limit_batches") == (1 if smoke else None),
        f"{task_name}: unexpected limit_batches for {'smoke' if smoke else 'full'} run",
    )
    require(
        Path(str(manifest["checkpoint_path"])).name == "epoch=2.ckpt",
        f"{task_name}: checkpoint is not epoch=2.ckpt",
    )
    require(
        sha256_file(manifest["checkpoint_path"]) == EXPECTED_CHECKPOINT_SHA256,
        f"{task_name}: checkpoint SHA-256 mismatch",
    )
    validation = manifest.get("checkpoint_config_validation") or {}
    require(validation.get("mismatch_count") == 0, f"{task_name}: checkpoint config mismatch")
    require(validation.get("architecture_matches") is True, f"{task_name}: architecture mismatch")
    require(validation.get("artifact_paths_match") is True, f"{task_name}: artifact path mismatch")
    cell_llm_summary = manifest.get("cell_llm_summary") or {}
    cell_type_llm_summary = manifest.get("cell_type_llm_summary") or {}
    require(
        cell_llm_summary.get("index_column") == "cell_llm_index",
        f"{task_name}: Cell LLM did not use cell_llm_index",
    )
    require(
        cell_type_llm_summary.get("index_column") == "cell_type_llm_index",
        f"{task_name}: cell-type LLM did not use cell_type_llm_index",
    )

    predictions = pd.read_parquet(prediction_path)
    required_prediction_columns = {
        "feature_row_index",
        "sample_id",
        "pred_task_prob",
        "pred_response_prob",
        "pred_synergy_prob",
    }
    require(
        required_prediction_columns.issubset(predictions.columns),
        f"{task_name}: prediction columns are incomplete",
    )
    expected_rows = 256 if smoke else int(spec["unique_rows"])
    require(len(predictions) == expected_rows, f"{task_name}: unexpected prediction row count")
    probability = pd.to_numeric(predictions["pred_task_prob"], errors="raise").to_numpy(
        dtype=np.float64
    )
    require(np.isfinite(probability).all(), f"{task_name}: non-finite probability")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), f"{task_name}: probability outside [0,1]")
    for duplicate in ("pred_response_prob", "pred_synergy_prob"):
        values = pd.to_numeric(predictions[duplicate], errors="raise").to_numpy(dtype=np.float64)
        require(
            np.array_equal(values, probability),
            f"{task_name}: {duplicate} is not the exact unified-head alias",
        )

    table_columns = [
        "feature_row_index",
        "sample_id",
        "model_key_id",
        "model_key_rank",
        "tissue",
        "source_cell",
        "Cell",
        "cell_type",
        "machineID_new",
        "Cell_index",
        "true_Cell_index",
        "cell_llm_index",
        "cell_type_index",
        "cell_type_llm_index",
        "cell_seen_in_checkpoint_train",
        "pert_id1",
        "pert_id2",
        "pert_index1",
        "pert_index2",
        "pert_dose1_index",
        "pert_dose2_index",
        "pert_time",
        "raw_dose1_representative_uM",
        "raw_dose2_representative_uM",
        "is_control",
    ]
    table = pd.read_parquet(task_dir / "feature_table.parquet", columns=table_columns)
    query = table.loc[~table["is_control"].astype(bool)].copy()
    expected_query = query.iloc[:expected_rows]
    require(
        predictions["feature_row_index"].astype(np.int64).tolist()
        == expected_query["feature_row_index"].astype(np.int64).tolist(),
        f"{task_name}: predictions are not the expected feature-row prefix",
    )
    require(
        predictions["sample_id"].astype(str).tolist()
        == expected_query["sample_id"].astype(str).tolist(),
        f"{task_name}: prediction sample IDs differ from feature table",
    )
    enriched = expected_query.drop(columns=["is_control"]).reset_index(drop=True)
    enriched["pred_unified_combo_prob"] = probability
    require(enriched["model_key_id"].is_unique, f"{task_name}: duplicate model_key_id")
    audit = {
        "tissue": tissue,
        "task_name": task_name,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "run_manifest_path": str(manifest_path.resolve()),
        "rows": len(enriched),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
        "checkpoint_config_mismatches": 0,
        "cell_llm_index_column": "cell_llm_index",
        "cell_type_llm_index_column": "cell_type_llm_index",
        "unified_probability_aliases_exact": True,
    }
    return enriched, audit


def write_markdown(path: Path, *, prefix: str, summary: dict[str, Any], smoke: bool) -> None:
    lines = [
        f"# {prefix} {'Smoke' if smoke else 'Formal'} Results",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        f"- Checkpoint SHA-256: `{summary['checkpoint_sha256']}`",
        "- Inference only; no training or fine-tuning.",
        "- Score: `pred_unified_combo_prob`, sourced only from unified-head `pred_task_prob`.",
        "- No AUROC/AUPRC and no top-ranking table are reported because the screen is unlabeled.",
        "",
        "## Counts",
        "",
        "| tissue | unique predictions | full raw rows |",
        "|---|---:|---:|",
    ]
    for tissue in TISSUE_SPECS:
        task = summary["tasks"][tissue]
        lines.append(
            f"| {tissue} | {task['rows']:,} | "
            f"{task.get('full_raw_rows', TISSUE_SPECS[tissue]['raw_rows'] if not smoke else 0):,} |"
        )
    lines.extend(
        [
            f"| total | {summary['unique_prediction_rows']:,} | "
            f"{summary.get('full_prediction_rows', 0):,} |",
            "",
            "## Covariate contract",
            "",
            "- Each cell uses its recovered `QE` or `480_FAIMS` machine value.",
            "- `Cell_plate=no` and `batch=no` are explicit checkpoint-training OOD categories.",
            "- The 18 train-seen cells use their true `Cell_index`; the 10 unseen cells use "
            "`Cell_index=0` while retaining their real nonzero `cell_llm_index`.",
            "- `cell_type`, 24h, and all realized dose buckets are checkpoint-train-supported.",
            "- Baselines are the supplied same-tissue mixed-6h/24h `*_control_unique.csv` rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_if_present(source: Path, target: Path) -> str | None:
    if not source.exists():
        return None
    if target.exists():
        raise FileExistsError(f"refusing to replace existing file: {target}")
    shutil.copy2(source, target)
    return str(target.resolve())


def write_full_predictions(
    *,
    unique_predictions: pd.DataFrame,
    output_parquet: Path,
    output_csv: Path,
    drug_audit: pd.DataFrame,
    build_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    parquet_temp = output_parquet.with_suffix(output_parquet.suffix + ".partial")
    csv_temp = output_csv.with_suffix(output_csv.suffix + ".partial")
    ensure_new([output_parquet, output_csv, parquet_temp, csv_temp])
    smiles_to_id = dict(
        zip(drug_audit["query_smiles"].astype(str), drug_audit["selected_pert_id"].astype(str), strict=True)
    )
    raw_hashes_after: dict[str, Any] = {}
    per_tissue: dict[str, dict[str, Any]] = {}
    parquet_writer: pq.ParquetWriter | None = None
    parquet_schema: pa.Schema | None = None
    total_rows = 0
    try:
        with csv_temp.open("w", encoding="utf-8", newline="") as csv_handle:
            wrote_header = False
            for tissue, spec in TISSUE_SPECS.items():
                raw_path = RAW_ROOT / str(spec["query_file"])
                expected_hash = build_summary["raw_inputs"][f"{tissue}_query"]["sha256_before"]
                require(
                    sha256_file(raw_path) == expected_hash,
                    f"{tissue}: raw query hash changed before report mapping",
                )
                lookup = unique_predictions.loc[
                    unique_predictions["tissue"].eq(tissue),
                    [
                        *KEY_COLUMNS,
                        "model_key_id",
                        "pred_unified_combo_prob",
                    ],
                ].copy()
                lookup = normalize_model_key_columns(
                    lookup,
                    context=f"{tissue} unique prediction lookup",
                )
                require(
                    len(lookup) == int(spec["unique_rows"]) and not lookup.duplicated(KEY_COLUMNS).any(),
                    f"{tissue}: unique lookup does not satisfy the model-key contract",
                )
                tissue_rows = 0
                score_min = 1.0
                score_max = 0.0
                for chunk_start, chunk in enumerate(
                    pd.read_csv(raw_path, chunksize=100_000, low_memory=False)
                ):
                    start_row = tissue_rows
                    tissue_rows += len(chunk)
                    key_frame = pd.DataFrame(
                        {
                            "tissue": tissue,
                            "source_cell": chunk["cell"].astype(str).str.strip(),
                            "pert_id1": chunk["drug_A_smiles"].astype(str).str.strip().map(smiles_to_id),
                            "pert_id2": chunk["drug_B_smiles"].astype(str).str.strip().map(smiles_to_id),
                            "pert_dose1_index": clipped_dose_bucket(
                                chunk["drug_A_concentration_uM"],
                                column="drug_A_concentration_uM",
                            ),
                            "pert_dose2_index": clipped_dose_bucket(
                                chunk["assumed_combo_IC50_B_uM"],
                                column="assumed_combo_IC50_B_uM",
                            ),
                            "pert_time": 24,
                        }
                    )
                    key_frame = normalize_model_key_columns(
                        key_frame,
                        context=f"{tissue} raw rows {start_row}:{tissue_rows}",
                    )
                    require(
                        key_frame[["pert_id1", "pert_id2"]].notna().all().all(),
                        f"{tissue}: raw row has no selected structural drug mapping",
                    )
                    mapped = key_frame.merge(
                        lookup,
                        on=KEY_COLUMNS,
                        how="left",
                        validate="many_to_one",
                        sort=False,
                    )
                    require(
                        mapped[["model_key_id", "pred_unified_combo_prob"]].notna().all().all(),
                        f"{tissue}: raw rows failed model-key prediction mapping",
                    )
                    score = mapped["pred_unified_combo_prob"].to_numpy(dtype=np.float64)
                    require(
                        np.isfinite(score).all() and np.all((score >= 0.0) & (score <= 1.0)),
                        f"{tissue}: mapped score is invalid",
                    )
                    score_min = min(score_min, float(score.min()))
                    score_max = max(score_max, float(score.max()))
                    output = chunk.copy()
                    output.insert(0, "source_tissue", tissue)
                    output.insert(1, "source_file", raw_path.name)
                    output.insert(
                        2,
                        "source_row_index",
                        np.arange(start_row, tissue_rows, dtype=np.int64),
                    )
                    output["model_key_id"] = mapped["model_key_id"].to_numpy()
                    output["pred_unified_combo_prob"] = score
                    output.to_csv(csv_handle, index=False, header=not wrote_header)
                    wrote_header = True
                    arrow_table = pa.Table.from_pandas(output, preserve_index=False)
                    if parquet_writer is None:
                        parquet_schema = arrow_table.schema
                        parquet_writer = pq.ParquetWriter(
                            parquet_temp,
                            parquet_schema,
                            compression="zstd",
                        )
                    elif arrow_table.schema != parquet_schema:
                        arrow_table = arrow_table.cast(parquet_schema)
                    parquet_writer.write_table(arrow_table)
                    total_rows += len(output)
                require(
                    tissue_rows == int(spec["raw_rows"]),
                    f"{tissue}: full mapped row count mismatch",
                )
                raw_hash = sha256_file(raw_path)
                require(raw_hash == expected_hash, f"{tissue}: raw query changed during report mapping")
                raw_hashes_after[tissue] = {
                    "path": str(raw_path.resolve()),
                    "sha256": raw_hash,
                    "unchanged_from_build": True,
                }
                per_tissue[tissue] = {
                    "full_raw_rows": tissue_rows,
                    "unique_model_keys": len(lookup),
                    "probability_min": score_min,
                    "probability_max": score_max,
                    "source_order_preserved": True,
                    "all_rows_mapped": True,
                }
        if parquet_writer is not None:
            parquet_writer.close()
            parquet_writer = None
        require(total_rows == EXPECTED_RAW_ROWS, "combined full prediction row count mismatch")
        os.replace(parquet_temp, output_parquet)
        os.replace(csv_temp, output_csv)
    except Exception:
        if parquet_writer is not None:
            parquet_writer.close()
        for path in (parquet_temp, csv_temp):
            if path.exists():
                path.unlink()
        raise
    return raw_hashes_after, per_tissue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-ready-root", default=str(TRAINING_READY_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_ready_root = Path(args.training_ready_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)
    run_dir = Path(args.run_dir) if args.run_dir else output_dir / prefix
    build_summary_path = (
        training_ready_root / "ptv3/exp33_vc_doubledrug_build_summary.json"
    )
    build_summary = load_json(build_summary_path)
    require(
        build_summary["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "build summary checkpoint SHA-256 mismatch",
    )

    unique_parts: list[pd.DataFrame] = []
    task_audits: dict[str, Any] = {}
    for tissue in TISSUE_SPECS:
        enriched, audit = validate_task_predictions(
            tissue=tissue,
            run_dir=run_dir,
            training_ready_root=training_ready_root,
            smoke=args.smoke,
        )
        unique_parts.append(enriched)
        task_audits[tissue] = audit
    unique_predictions = pd.concat(unique_parts, ignore_index=True)
    require(
        unique_predictions["model_key_id"].is_unique,
        "combined prediction table contains duplicate model_key_id",
    )

    checkpoint_path = Path(
        load_json(run_dir / TISSUE_SPECS["colon"]["task_name"] / "run_manifest.json")[
            "checkpoint_path"
        ]
    )
    summary: dict[str, Any] = {
        "generated_at": iso_now(),
        "prefix": prefix,
        "mode": "smoke" if args.smoke else "full",
        "run_dir": str(run_dir.resolve()),
        "training_ready_root": str(training_ready_root.resolve()),
        "build_summary_path": str(build_summary_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "score_column": "pred_unified_combo_prob",
        "score_source_column": "pred_task_prob",
        "unique_prediction_rows": len(unique_predictions),
        "tasks": task_audits,
        "metrics_reported": [],
        "ranking_outputs_generated": False,
        "acceptance_checks": {
            "probabilities_finite_in_unit_interval": True,
            "unified_alias_columns_exact": True,
            "checkpoint_config_mismatch_count_zero": True,
            "Cell_LLM_uses_separate_cell_llm_index": True,
            "cell_type_LLM_uses_true_cell_type_llm_index": True,
            "no_expression_predictions_saved": True,
            "no_AUROC_AUPRC": True,
            "no_top_ranking": True,
        },
    }

    log_source = REPO_ROOT / "logs" / f"{prefix}.log"
    runtime_source = REPO_ROOT / "logs" / f"{prefix}_runtime_summary.tsv"
    if args.smoke:
        summary_path = output_dir / f"{prefix}_summary.json"
        markdown_path = output_dir / f"{prefix}_results.md"
        ensure_new([summary_path, markdown_path])
        summary["expected_smoke_rows_per_task"] = 256
        summary["acceptance_checks"]["three_one_batch_smokes"] = (
            len(unique_predictions) == 3 * 256
        )
        summary["runtime_log"] = copy_if_present(
            log_source, output_dir / f"{prefix}_runtime.log"
        )
        summary["runtime_tsv"] = copy_if_present(
            runtime_source, output_dir / f"{prefix}_runtime_summary.tsv"
        )
        dump_json(summary_path, summary)
        write_markdown(markdown_path, prefix=prefix, summary=summary, smoke=True)
        print(f"[exp33-report] smoke passed: {summary_path}")
        return

    require(
        len(unique_predictions) == EXPECTED_UNIQUE_ROWS,
        "combined unique prediction count is not 1,124,928",
    )
    unique_path = output_dir / f"{prefix}_unique_model_predictions.parquet"
    full_parquet_path = output_dir / f"{prefix}_predictions.parquet"
    full_csv_path = output_dir / f"{prefix}_predictions.csv"
    drug_output_path = output_dir / f"{prefix}_drug_id_mapping_audit.csv"
    covariate_output_path = output_dir / f"{prefix}_covariate_audit.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    markdown_path = output_dir / f"{prefix}_results.md"
    runtime_log_path = output_dir / f"{prefix}_runtime.log"
    runtime_tsv_path = output_dir / f"{prefix}_runtime_summary.tsv"
    ensure_new(
        [
            unique_path,
            full_parquet_path,
            full_csv_path,
            drug_output_path,
            covariate_output_path,
            summary_path,
            markdown_path,
            runtime_log_path,
            runtime_tsv_path,
        ]
    )
    unique_predictions.to_parquet(unique_path, index=False)

    build_group_root = training_ready_root / "ptv3"
    drug_source_path = build_group_root / "exp33_vc_doubledrug_drug_id_mapping_audit.csv"
    covariate_source_path = build_group_root / "exp33_vc_doubledrug_covariate_audit.csv"
    drug_audit = pd.read_csv(drug_source_path, low_memory=False)
    require(len(drug_audit) == 227, "drug mapping audit does not contain 227 rows")
    require(
        drug_audit["selected_priority_tier"].value_counts().to_dict()
        == {"L9200": 176, "numeric": 51},
        "drug mapping audit ID-tier counts changed",
    )
    shutil.copy2(drug_source_path, drug_output_path)
    shutil.copy2(covariate_source_path, covariate_output_path)

    raw_hashes_after, mapping_audits = write_full_predictions(
        unique_predictions=unique_predictions,
        output_parquet=full_parquet_path,
        output_csv=full_csv_path,
        drug_audit=drug_audit,
        build_summary=build_summary,
    )
    for tissue in TISSUE_SPECS:
        summary["tasks"][tissue].update(mapping_audits[tissue])
    summary["full_prediction_rows"] = EXPECTED_RAW_ROWS
    summary["raw_input_hashes_after_report"] = raw_hashes_after
    summary["outputs"] = {
        "unique_model_predictions_parquet": str(unique_path.resolve()),
        "predictions_parquet": str(full_parquet_path.resolve()),
        "predictions_csv": str(full_csv_path.resolve()),
        "drug_id_mapping_audit_csv": str(drug_output_path.resolve()),
        "covariate_audit_csv": str(covariate_output_path.resolve()),
        "runtime_log": copy_if_present(log_source, runtime_log_path),
        "runtime_tsv": copy_if_present(runtime_source, runtime_tsv_path),
        "results_markdown": str(markdown_path.resolve()),
    }
    summary["output_sha256"] = {
        key: sha256_file(path)
        for key, path in {
            "unique_model_predictions_parquet": unique_path,
            "predictions_parquet": full_parquet_path,
            "predictions_csv": full_csv_path,
            "drug_id_mapping_audit_csv": drug_output_path,
            "covariate_audit_csv": covariate_output_path,
        }.items()
    }
    summary["acceptance_checks"].update(
        {
            "unique_predictions_1124928": len(unique_predictions) == EXPECTED_UNIQUE_ROWS,
            "full_predictions_2526720": summary["full_prediction_rows"] == EXPECTED_RAW_ROWS,
            "all_raw_rows_mapped": all(
                item["all_rows_mapped"] for item in mapping_audits.values()
            ),
            "same_model_key_has_one_score": unique_predictions["model_key_id"].is_unique,
            "raw_input_hashes_unchanged": all(
                item["unchanged_from_build"] for item in raw_hashes_after.values()
            ),
            "source_file_and_row_order_preserved": all(
                item["source_order_preserved"] for item in mapping_audits.values()
            ),
        }
    )
    dump_json(summary_path, summary)
    write_markdown(markdown_path, prefix=prefix, summary=summary, smoke=False)
    print(f"[exp33-report] full report passed: {summary_path}")
    print(f"[exp33-report] unique predictions: {unique_path} ({len(unique_predictions)} rows)")
    print(f"[exp33-report] full predictions: {full_parquet_path} ({EXPECTED_RAW_ROWS} rows)")


if __name__ == "__main__":
    main()
