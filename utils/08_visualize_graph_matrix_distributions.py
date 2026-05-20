#!/usr/bin/env python3
"""Visualize value distributions for PPI, PDI, and DDI matrices.

The graph matrices are large, so this script loads one matrix at a time and
plots both an all-value sample and a nonzero-value sample. Summary JSON/CSV
files include exact shape/count/min/max/mean/std plus sampled percentiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DERIVED_DIR = REPO_ROOT / "data" / "training_ready" / "ptv3" / "derived"
DEFAULT_OUTPUT_DIR = DEFAULT_DERIVED_DIR / "graph_value_distributions"
MATRIX_NAMES = ("ppi", "pdi", "ddi")
PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100)

# Matplotlib tries to write a cache under /root by default in this environment.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    path: Path


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_matrix(path: Path, *, mmap: bool) -> np.ndarray:
    if mmap:
        try:
            return np.load(path, mmap_mode="r")
        except OSError as exc:
            print(f"Warning: mmap failed for {path} ({exc}); falling back to normal np.load.")
    return np.load(path)


def choose_sample(flat_values: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    total = int(flat_values.size)
    if total == 0:
        return np.asarray([], dtype=np.float32)
    if sample_size <= 0 or sample_size >= total:
        return np.asarray(flat_values)
    indices = rng.choice(total, size=sample_size, replace=False)
    return np.asarray(flat_values[indices])


def choose_nonzero_sample(
    flat_values: np.ndarray,
    nonzero_indices: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = int(nonzero_indices.size)
    if total == 0:
        return np.asarray([], dtype=np.float32)
    if sample_size <= 0 or sample_size >= total:
        return np.asarray(flat_values[nonzero_indices])
    positions = rng.choice(total, size=sample_size, replace=False)
    return np.asarray(flat_values[nonzero_indices[positions]])


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.size == 0:
        return {
            "sample_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "percentiles": {},
        }

    quantiles = np.percentile(values, PERCENTILES)
    return {
        "sample_count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "percentiles": {
            percentile_label(percentile): float(value)
            for percentile, value in zip(PERCENTILES, quantiles)
        },
    }


def percentile_label(value: float) -> str:
    text = f"{value:g}".replace(".", "_")
    return f"p{text}"


def plot_distribution(
    *,
    name: str,
    all_values: np.ndarray,
    nonzero_values: np.ndarray,
    all_sampled: bool,
    nonzero_sampled: bool,
    bins: int,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Visualization requires matplotlib. Install it in the active environment.") from exc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    all_title = "all values"
    if all_sampled:
        all_title += f" sample n={len(all_values):,}"
    else:
        all_title += f" exact n={len(all_values):,}"
    axes[0].hist(all_values, bins=bins, range=(0.0, 1.0), color="#4C78A8", log=True)
    axes[0].set_title(f"{name.upper()} {all_title}")
    axes[0].set_xlabel("matrix value")
    axes[0].set_ylabel("count (log scale)")
    axes[0].grid(alpha=0.25)

    nonzero_title = "nonzero values"
    if nonzero_sampled:
        nonzero_title += f" sample n={len(nonzero_values):,}"
    else:
        nonzero_title += f" exact n={len(nonzero_values):,}"
    if len(nonzero_values):
        axes[1].hist(nonzero_values, bins=bins, range=(0.0, 1.0), color="#F58518", log=True)
        quantiles = np.percentile(nonzero_values, [50, 95, 99])
        for label, value in zip(("p50", "p95", "p99"), quantiles):
            axes[1].axvline(value, linestyle="--", linewidth=1, label=f"{label}={value:.3f}")
        axes[1].legend()
    axes[1].set_title(f"{name.upper()} {nonzero_title}")
    axes[1].set_xlabel("matrix value")
    axes[1].set_ylabel("count (log scale)")
    axes[1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_nonzero_overlay(
    samples_by_name: dict[str, np.ndarray],
    *,
    bins: int,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Visualization requires matplotlib. Install it in the active environment.") from exc

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for name, values in samples_by_name.items():
        if len(values) == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            range=(0.0, 1.0),
            histtype="step",
            linewidth=1.6,
            density=True,
            label=f"{name.upper()} nonzero",
        )
    ax.set_title("Nonzero Value Distribution Comparison")
    ax.set_xlabel("matrix value")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def matrix_report(
    spec: MatrixSpec,
    *,
    rng: np.random.Generator,
    output_dir: Path,
    sample_size: int,
    nonzero_sample_size: int,
    bins: int,
    mmap: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    matrix = load_matrix(spec.path, mmap=mmap)
    flat_values = np.asarray(matrix).ravel()
    nonzero_indices = np.flatnonzero(flat_values)

    all_sample = choose_sample(flat_values, sample_size, rng)
    nonzero_sample = choose_nonzero_sample(flat_values, nonzero_indices, nonzero_sample_size, rng)
    all_sampled = 0 < sample_size < flat_values.size
    nonzero_sampled = 0 < nonzero_sample_size < nonzero_indices.size

    zero_count = int(flat_values.size - nonzero_indices.size)
    row_nonzero_count = int(np.count_nonzero(np.any(np.asarray(matrix) != 0, axis=1)))
    col_nonzero_count = int(np.count_nonzero(np.any(np.asarray(matrix) != 0, axis=0)))

    report = {
        "name": spec.name,
        "path": str(spec.path),
        "shape": [int(value) for value in matrix.shape],
        "dtype": str(matrix.dtype),
        "total_count": int(flat_values.size),
        "zero_count": zero_count,
        "nonzero_count": int(nonzero_indices.size),
        "zero_fraction": float(zero_count / flat_values.size) if flat_values.size else None,
        "nonzero_fraction": float(nonzero_indices.size / flat_values.size) if flat_values.size else None,
        "nonzero_row_count": row_nonzero_count,
        "nonzero_col_count": col_nonzero_count,
        "finite_all": bool(np.isfinite(flat_values).all()),
        "exact_min": float(np.min(flat_values)) if flat_values.size else None,
        "exact_max": float(np.max(flat_values)) if flat_values.size else None,
        "exact_mean": float(np.mean(flat_values)) if flat_values.size else None,
        "exact_std": float(np.std(flat_values)) if flat_values.size else None,
        "all_values": numeric_summary(all_sample),
        "nonzero_values": numeric_summary(nonzero_sample),
        "all_values_sampled": all_sampled,
        "nonzero_values_sampled": nonzero_sampled,
    }

    dump_json(output_dir / f"{spec.name}_distribution_summary.json", report)
    plot_distribution(
        name=spec.name,
        all_values=all_sample,
        nonzero_values=nonzero_sample,
        all_sampled=all_sampled,
        nonzero_sampled=nonzero_sampled,
        bins=bins,
        output_path=output_dir / f"{spec.name}_distribution.png",
    )
    return report, nonzero_sample


def write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "shape",
        "zero_fraction",
        "nonzero_count",
        "nonzero_row_count",
        "nonzero_col_count",
        "exact_min",
        "exact_max",
        "exact_mean",
        "exact_std",
        "nonzero_p50",
        "nonzero_p95",
        "nonzero_p99",
        "nonzero_p99_5",
        "nonzero_p99_9",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            percentiles = report["nonzero_values"]["percentiles"]
            writer.writerow(
                {
                    "name": report["name"],
                    "shape": "x".join(str(value) for value in report["shape"]),
                    "zero_fraction": report["zero_fraction"],
                    "nonzero_count": report["nonzero_count"],
                    "nonzero_row_count": report["nonzero_row_count"],
                    "nonzero_col_count": report["nonzero_col_count"],
                    "exact_min": report["exact_min"],
                    "exact_max": report["exact_max"],
                    "exact_mean": report["exact_mean"],
                    "exact_std": report["exact_std"],
                    "nonzero_p50": percentiles.get("p50"),
                    "nonzero_p95": percentiles.get("p95"),
                    "nonzero_p99": percentiles.get("p99"),
                    "nonzero_p99_5": percentiles.get("p99_5"),
                    "nonzero_p99_9": percentiles.get("p99_9"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PPI/PDI/DDI matrix value distributions.")
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ppi-npy", type=Path, default=None, help="Override PPI matrix path.")
    parser.add_argument("--pdi-npy", type=Path, default=None, help="Override PDI matrix path.")
    parser.add_argument("--ddi-npy", type=Path, default=None, help="Override DDI matrix path.")
    parser.add_argument(
        "--matrices",
        nargs="+",
        choices=MATRIX_NAMES,
        default=list(MATRIX_NAMES),
        help="Matrices to visualize.",
    )
    parser.add_argument("--sample-size", type=int, default=2_000_000, help="All-value sample size per matrix.")
    parser.add_argument(
        "--nonzero-sample-size",
        type=int,
        default=2_000_000,
        help="Nonzero-value sample size per matrix.",
    )
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument(
        "--mmap",
        action="store_true",
        help="Try NumPy memory mapping. Disabled by default because some shared filesystems reject mmap.",
    )
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[MatrixSpec]:
    paths = {
        "ppi": args.ppi_npy or args.derived_dir / "ppi_matrix.npy",
        "pdi": args.pdi_npy or args.derived_dir / "pdi_matrix.npy",
        "ddi": args.ddi_npy or args.derived_dir / "ddi_matrix.npy",
    }
    specs = []
    for name in args.matrices:
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"{name.upper()} matrix not found: {path}")
        specs.append(MatrixSpec(name=name, path=path))
    return specs


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    reports: list[dict[str, Any]] = []
    nonzero_samples: dict[str, np.ndarray] = {}

    for spec in build_specs(args):
        print(f"Processing {spec.name.upper()}: {spec.path}")
        report, nonzero_sample = matrix_report(
            spec,
            rng=rng,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            nonzero_sample_size=args.nonzero_sample_size,
            bins=args.bins,
            mmap=args.mmap,
        )
        reports.append(report)
        nonzero_samples[spec.name] = nonzero_sample
        print(
            f"  shape={tuple(report['shape'])} zero_fraction={report['zero_fraction']:.6f} "
            f"nonzero_count={report['nonzero_count']}"
        )

    write_summary_csv(args.output_dir / "graph_matrix_distribution_summary.csv", reports)
    dump_json(args.output_dir / "graph_matrix_distribution_summary.json", reports)
    plot_nonzero_overlay(
        nonzero_samples,
        bins=args.bins,
        output_path=args.output_dir / "nonzero_distribution_overlay.png",
    )
    print(f"Wrote plots and summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
