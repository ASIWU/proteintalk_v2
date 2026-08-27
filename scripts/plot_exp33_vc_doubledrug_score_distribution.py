#!/usr/bin/env python3
"""Plot formal exp33 double-drug score distributions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs/2026-07/2026-07-27"
PREFIX = "20260727_exp33_vc_doubledrug_epoch2"
DEFAULT_FULL_INPUT = OUTPUT_ROOT / f"{PREFIX}_predictions.parquet"
DEFAULT_UNIQUE_INPUT = OUTPUT_ROOT / f"{PREFIX}_unique_model_predictions.parquet"
DEFAULT_PNG = OUTPUT_ROOT / f"{PREFIX}_score_distribution.png"
DEFAULT_PDF = OUTPUT_ROOT / f"{PREFIX}_score_distribution.pdf"
DEFAULT_SUMMARY = OUTPUT_ROOT / f"{PREFIX}_score_distribution_summary.csv"

TISSUE_ORDER = ("colon", "lung", "pancreas")
TISSUE_COLORS = {
    "colon": "#0072B2",
    "lung": "#D55E00",
    "pancreas": "#009E73",
}
EXPECTED_FULL_ROWS = 2_526_720
EXPECTED_UNIQUE_ROWS = 1_124_928


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-input", type=Path, default=DEFAULT_FULL_INPUT)
    parser.add_argument("--unique-input", type=Path, default=DEFAULT_UNIQUE_INPUT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def validated_scores(
    path: Path,
    *,
    tissue_column: str,
    expected_rows: int,
) -> pd.DataFrame:
    table = pd.read_parquet(
        path,
        columns=[tissue_column, "pred_unified_combo_prob"],
    ).rename(columns={tissue_column: "tissue", "pred_unified_combo_prob": "score"})
    if len(table) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows:,} rows, found {len(table):,}")
    table["tissue"] = table["tissue"].astype(str)
    observed_tissues = set(table["tissue"].unique())
    if observed_tissues != set(TISSUE_ORDER):
        raise ValueError(
            f"{path}: expected tissues {TISSUE_ORDER}, found {sorted(observed_tissues)}"
        )
    scores = pd.to_numeric(table["score"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"{path}: scores must be finite and within [0, 1]")
    table["score"] = scores
    return table


def summary_row(scope: str, tissue: str, values: np.ndarray) -> dict[str, object]:
    return {
        "scope": scope,
        "tissue": tissue,
        "rows": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    cumulative = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
    return ordered, cumulative


def main() -> None:
    args = parse_args()
    targets = (args.output_png, args.output_pdf, args.output_summary)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing distribution outputs: {existing}")

    full = validated_scores(
        args.full_input,
        tissue_column="source_tissue",
        expected_rows=EXPECTED_FULL_ROWS,
    )
    unique = validated_scores(
        args.unique_input,
        tissue_column="tissue",
        expected_rows=EXPECTED_UNIQUE_ROWS,
    )

    summary_records = [
        summary_row("full_raw_rows", "all", full["score"].to_numpy(dtype=np.float64)),
        summary_row("unique_model_keys", "all", unique["score"].to_numpy(dtype=np.float64)),
    ]
    for tissue in TISSUE_ORDER:
        values = full.loc[full["tissue"].eq(tissue), "score"].to_numpy(dtype=np.float64)
        summary_records.append(summary_row("full_raw_rows", tissue, values))
    summary = pd.DataFrame(summary_records)

    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, (hist_ax, ecdf_ax) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.1),
        constrained_layout=True,
    )
    bins = np.linspace(0.0, 1.0, 101)
    full_scores = full["score"].to_numpy(dtype=np.float64)
    unique_scores = unique["score"].to_numpy(dtype=np.float64)

    hist_ax.hist(
        full_scores,
        bins=bins,
        density=True,
        color="#6C757D",
        alpha=0.28,
        label=f"Full raw rows (n={len(full_scores):,})",
    )
    hist_ax.hist(
        full_scores,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="#343A40",
    )
    hist_ax.hist(
        unique_scores,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        linestyle="--",
        color="#7B2CBF",
        label=f"Unique model keys (n={len(unique_scores):,})",
    )
    full_mean = float(np.mean(full_scores))
    full_median = float(np.median(full_scores))
    hist_ax.axvline(
        full_mean,
        color="#C1121F",
        linewidth=1.4,
        linestyle="--",
        label=f"Full mean = {full_mean:.3f}",
    )
    hist_ax.axvline(
        full_median,
        color="#1D3557",
        linewidth=1.4,
        linestyle=":",
        label=f"Full median = {full_median:.3f}",
    )
    hist_ax.set_xlim(0.0, 1.0)
    hist_ax.set_xlabel("pred_unified_combo_prob")
    hist_ax.set_ylabel("Density")
    hist_ax.set_title("Overall distribution")
    hist_ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    hist_ax.legend(frameon=False, fontsize=9)
    hist_ax.text(
        0.99,
        0.02,
        "Full rows preserve source-row multiplicity",
        transform=hist_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color="#555555",
    )

    tissue_lines: list[str] = []
    for tissue in TISSUE_ORDER:
        values = full.loc[full["tissue"].eq(tissue), "score"].to_numpy(dtype=np.float64)
        x, y = ecdf(values)
        ecdf_ax.plot(
            x,
            y,
            color=TISSUE_COLORS[tissue],
            linewidth=2.0,
            label=tissue,
        )
        tissue_lines.append(
            f"{tissue}: n={len(values):,}, "
            f"mean={np.mean(values):.3f}, median={np.median(values):.3f}"
        )
    ecdf_ax.set_xlim(0.0, 1.0)
    ecdf_ax.set_ylim(0.0, 1.0)
    ecdf_ax.set_xlabel("pred_unified_combo_prob")
    ecdf_ax.set_ylabel("Cumulative fraction")
    ecdf_ax.set_title("Full raw-row ECDF by tissue")
    ecdf_ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ecdf_ax.legend(title="Tissue", frameon=False, loc="lower right")
    ecdf_ax.text(
        0.02,
        0.98,
        "\n".join(tissue_lines),
        transform=ecdf_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.92,
        },
    )

    figure.suptitle(
        "Exp33 double-drug virtual-screen score distributions",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"[plot] wrote {args.output_png}")
    print(f"[plot] wrote {args.output_pdf}")
    print(f"[plot] wrote {args.output_summary}")


if __name__ == "__main__":
    main()
