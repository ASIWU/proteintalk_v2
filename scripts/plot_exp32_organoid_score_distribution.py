#!/usr/bin/env python3
"""Plot exp32 organoid predicted sensitivity-score distributions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_predictions.parquet"
DEFAULT_PNG = REPO_ROOT / "outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_score_distribution.png"
DEFAULT_PDF = REPO_ROOT / "outputs/2026-07/2026-07-10/20260710_exp32_organoid_exp09_single_sensitivity_score_distribution.pdf"

DEVICE_ORDER = ("B", "CAC")
DEVICE_COLORS = {"B": "#0072B2", "CAC": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_scores(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        table = pd.read_parquet(path, columns=["device", "pred_sensitivity_prob"])
    else:
        table = pd.read_csv(path, usecols=["device", "pred_sensitivity_prob"], low_memory=False)
    if set(table["device"].astype(str).unique()) != set(DEVICE_ORDER):
        raise ValueError(f"expected devices {DEVICE_ORDER}, got {sorted(table['device'].astype(str).unique())}")
    scores = pd.to_numeric(table["pred_sensitivity_prob"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("pred_sensitivity_prob must be finite and within [0, 1]")
    return table.assign(pred_sensitivity_prob=scores)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    cumulative = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
    return ordered, cumulative


def main() -> None:
    args = parse_args()
    table = load_scores(args.input)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

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
    figure, (hist_ax, ecdf_ax) = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

    linear_bins = np.linspace(0.0, 1.0, 101)
    summary_lines: list[str] = []
    for device in DEVICE_ORDER:
        values = table.loc[table["device"].astype(str).eq(device), "pred_sensitivity_prob"].to_numpy(
            dtype=np.float64
        )
        color = DEVICE_COLORS[device]
        mean = float(np.mean(values))
        median = float(np.median(values))
        summary_lines.append(f"{device}: mean={mean:.4f}, median={median:.4f}, n={len(values):,}")

        hist_ax.hist(
            values,
            bins=linear_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.20,
            color=color,
        )
        hist_ax.hist(
            values,
            bins=linear_bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=device,
        )
        hist_ax.axvline(mean, color=color, linewidth=1.3, linestyle="--", alpha=0.9)
        hist_ax.axvline(median, color=color, linewidth=1.3, linestyle=":", alpha=0.9)

        x, y = ecdf(values)
        ecdf_ax.plot(x, y, color=color, linewidth=2.0, label=device)

    hist_ax.set_yscale("log")
    hist_ax.set_xlim(0.0, 1.0)
    hist_ax.set_xlabel("Predicted sensitivity probability")
    hist_ax.set_ylabel("Density (log scale)")
    hist_ax.set_title("Full score distribution")
    hist_ax.grid(axis="y", which="both", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    hist_ax.legend(title="Device", frameon=False, loc="upper right")
    hist_ax.text(
        0.985,
        0.67,
        "Dashed: mean\nDotted: median",
        transform=hist_ax.transAxes,
        ha="right",
        va="top",
        color="#555555",
        fontsize=9,
    )

    ecdf_ax.set_xscale("log")
    ecdf_ax.set_xlim(1e-5, 1.0)
    ecdf_ax.set_ylim(0.0, 1.0)
    ecdf_ax.set_xlabel("Predicted sensitivity probability (log scale)")
    ecdf_ax.set_ylabel("Cumulative fraction")
    ecdf_ax.set_title("ECDF highlighting low scores")
    ecdf_ax.grid(which="both", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ecdf_ax.legend(title="Device", frameon=False, loc="lower right")
    ecdf_ax.text(
        0.03,
        0.97,
        "\n".join(summary_lines),
        transform=ecdf_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
    )

    figure.suptitle(
        "Exp32 organoid single-drug sensitivity score distributions",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"[plot] wrote {args.output_png}")
    print(f"[plot] wrote {args.output_pdf}")


if __name__ == "__main__":
    main()
