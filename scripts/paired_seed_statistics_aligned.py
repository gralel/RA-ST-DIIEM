"""Matched five-seed tests against the revised global-energy final model."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

SEEDS = range(44, 49)
METHODS = (
    "Multi PC-BiLSTM", "TEM1DFormer", "DnCNN2D", "Single PC-BiLSTM",
    "Spatial PC-BiLSTM", "RA anchor-only", "Direct residual no-diffusion",
    "temdnet_style", "temnlnet_style", "joint_gated_moe", "elementwise",
)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    permuted = [
        abs(float((differences * np.asarray(signs)).mean()))
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def percentile_bootstrap_ci(differences: np.ndarray, seed: int = 20261013) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(20000, differences.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(rows: list[dict[str, object]], key: str) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index][key]))
    running = 0.0
    m = len(rows)
    adjusted = [1.0] * m
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(rows[index][key]))
        running = max(running, candidate)
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row[f"{key}_holm"] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary_path.read_text(encoding="utf-8"))
    rows = []
    for method in METHODS:
        for outcome in ("overall", "challenging"):
            differences = np.asarray([
                summary[method]["metrics"][outcome]["by_seed"][str(seed)]
                - summary["global_energy"]["metrics"][outcome]["by_seed"][str(seed)]
                for seed in SEEDS
            ])
            low, high = percentile_bootstrap_ci(differences)
            rows.append({
                "comparator": method,
                "metric": outcome,
                "n_matched_seeds": differences.size,
                "mean_difference_comparator_minus_aligned_final": float(differences.mean()),
                "seed_bootstrap_ci95_low": low,
                "seed_bootstrap_ci95_high": high,
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
            })
    holm_adjust(rows, "exact_sign_flip_p")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_seed_statistics_aligned.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out / "paired_seed_statistics_aligned.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["metric"] == "overall":
            print(f"{row['comparator']}: delta={row['mean_difference_comparator_minus_aligned_final']:.5f}; p={row['exact_sign_flip_p']:.4f}; Holm={row['exact_sign_flip_p_holm']:.4f}")


if __name__ == "__main__":
    main()
