"""Paired confidence intervals that resample independent clean TEM models.

Sample-level errors are first averaged across matched training seeds.  Repeated
noise realizations belonging to the same clean model are then kept together as
a cluster, preventing pseudo-replication of the 365 independent test models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_model_ids(path: Path) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray([str(row["model_name"]) for row in rows], dtype=object)


def load_seed_mean(pattern: str, seeds: list[int], scenario: str) -> np.ndarray:
    arrays = []
    for seed in seeds:
        path = Path(pattern.format(seed=seed)) / "sample_metrics" / f"{scenario}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        arrays.append(np.asarray(np.load(path)["rrmse"], dtype=np.float64))
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched sample counts: {sorted(lengths)}")
    return np.stack(arrays).mean(axis=0)


def cluster_means(values: np.ndarray, model_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(model_ids)
    means = np.asarray([values[model_ids == model_id].mean() for model_id in unique])
    return unique, means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator-pattern", required=True, help="Run directory pattern containing {seed}.")
    parser.add_argument("--final-pattern", required=True, help="Run directory pattern containing {seed}.")
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--scenario", required=True, help="NPZ stem, e.g. combined_test or low_snr_gaussian.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46, 47, 48])
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--limit", type=int, help="Optional smoke-test prefix; omit for publication statistics.")
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparator = load_seed_mean(args.comparator_pattern, args.seeds, args.scenario)
    final = load_seed_mean(args.final_pattern, args.seeds, args.scenario)
    model_ids = read_model_ids(args.recipes)
    if args.limit is not None:
        model_ids = model_ids[: args.limit]
    if not (comparator.size == final.size == model_ids.size):
        raise ValueError(f"Length mismatch: comparator={comparator.size}, final={final.size}, ids={model_ids.size}")

    ids, differences = cluster_means(comparator - final, model_ids)
    rng = np.random.default_rng(args.bootstrap_seed)
    indices = rng.integers(0, differences.size, size=(args.draws, differences.size))
    bootstrap = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    result = {
        "scenario": args.scenario,
        "matched_training_seeds": args.seeds,
        "n_test_samples": int(model_ids.size),
        "n_independent_test_models": int(ids.size),
        "estimand": "mean model-cluster RRMSE difference (comparator minus RA-ST-DiffTEM)",
        "mean_difference": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_draws": int(args.draws),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
