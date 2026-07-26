from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator


SCRIPT_PATH = Path(__file__).absolute()
WORKSPACE = SCRIPT_PATH.parents[2]
DATA_ROOT = WORKSPACE / "数据" / "data_paper_fixed_scenarios_25k"
REAL_INTERP_DIR = WORKSPACE / "实测" / "插值" / "插值869"
OUT_DIR = WORKSPACE / "实测" / "插值" / "插值方法评估"

EPS = 1.0e-12
SEQ_LEN = 869
RECEIVER_COUNT = 11


@dataclass(frozen=True)
class Candidate:
    name: str
    domain: str
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def to_log10_amplitude(array: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.asarray(array, dtype=np.float32)) + EPS).astype(np.float32)


def canonicalize_11t(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape == (RECEIVER_COUNT, SEQ_LEN):
            return arr[None]
        if arr.shape == (SEQ_LEN, RECEIVER_COUNT):
            return arr.T[None]
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected 2D/3D TEM array, got {arr.shape}")
    if arr.shape[1] == RECEIVER_COUNT:
        return arr
    if arr.shape[2] == RECEIVER_COUNT:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"{name}: expected one axis to be {RECEIVER_COUNT}, got {arr.shape}")


def interp_linear(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    return np.interp(x_dst, x_src, y_src).astype(np.float32)


def interp_pchip(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    if x_src.size < 4:
        return interp_linear(x_src, y_src, x_dst)
    return np.asarray(PchipInterpolator(x_src, y_src, extrapolate=False)(x_dst), dtype=np.float32)


def interp_akima(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    if x_src.size < 5:
        return interp_pchip(x_src, y_src, x_dst)
    vals = Akima1DInterpolator(x_src, y_src, extrapolate=False)(x_dst)
    if not np.all(np.isfinite(vals)):
        vals = interp_pchip(x_src, y_src, x_dst)
    return np.asarray(vals, dtype=np.float32)


def interp_cubic(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    if x_src.size < 4:
        return interp_pchip(x_src, y_src, x_dst)
    vals = CubicSpline(x_src, y_src, bc_type="natural", extrapolate=False)(x_dst)
    if not np.all(np.isfinite(vals)):
        vals = interp_pchip(x_src, y_src, x_dst)
    return np.asarray(vals, dtype=np.float32)


def interp_bounded_cubic(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Natural cubic constrained to the local physical range of adjacent gates."""
    vals = interp_cubic(x_src, y_src, x_dst).astype(np.float64)
    x = np.asarray(x_src, dtype=np.float64).reshape(-1)
    y = np.asarray(y_src, dtype=np.float64).reshape(-1)
    dst = np.asarray(x_dst, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(x, dst, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)
    lo = np.minimum(y[idx], y[idx + 1])
    hi = np.maximum(y[idx], y[idx + 1])
    return np.clip(vals, lo, hi).astype(np.float32)


def source_coords_from_real_mat(path: Path) -> np.ndarray:
    mat = loadmat(path)
    if "normalized_gate_40" in mat:
        coords = np.asarray(mat["normalized_gate_40"], dtype=np.float64).reshape(-1)
    elif "original_time_s" in mat and "log10_time_interp" in mat:
        source_log = np.log10(np.asarray(mat["original_time_s"], dtype=np.float64).reshape(-1))
        target_log = np.asarray(mat["log10_time_interp"], dtype=np.float64).reshape(-1)
        coords = (source_log - float(target_log[0])) / float(target_log[-1] - target_log[0])
    else:
        raise KeyError(f"{path}: cannot infer real source gate coordinates")
    coords = coords[np.isfinite(coords)]
    coords = np.unique(np.clip(coords, 0.0, 1.0))
    if coords.size < 2:
        raise ValueError(f"{path}: too few finite source coordinates")
    return coords.astype(np.float64)


def real_gate_patterns() -> dict[str, np.ndarray]:
    return {
        "dbz_40_normalized": source_coords_from_real_mat(REAL_INTERP_DIR / "dbz_11x869.mat"),
        "W21_40_logtime": source_coords_from_real_mat(REAL_INTERP_DIR / "taochong_W21_selected_11x869.mat"),
        "W23_40_logtime": source_coords_from_real_mat(REAL_INTERP_DIR / "taochong_W23_selected_11x869.mat"),
        "W25_39_logtime": source_coords_from_real_mat(REAL_INTERP_DIR / "taochong_W25_selected_11x869.mat"),
    }


def load_eval_arrays(max_samples_per_file: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    paths = [
        ("train_noisy", DATA_ROOT / "train" / "train_data.npy"),
        ("train_clean", DATA_ROOT / "train" / "train_clean_base.npy"),
        ("test_noisy", DATA_ROOT / "test" / "test_data.npy"),
        ("light_gaussian", DATA_ROOT / "test_scenarios" / "light_gaussian" / "light_gaussian_data.npy"),
        ("low_snr_gaussian", DATA_ROOT / "test_scenarios" / "low_snr_gaussian" / "low_snr_gaussian_data.npy"),
        ("em_interference", DATA_ROOT / "test_scenarios" / "em_interference" / "em_interference_data.npy"),
        ("drift", DATA_ROOT / "test_scenarios" / "drift" / "drift_data.npy"),
        ("local_receiver_corruption", DATA_ROOT / "test_scenarios" / "local_receiver_corruption" / "local_receiver_corruption_data.npy"),
        ("mixed_complex_noise", DATA_ROOT / "test_scenarios" / "mixed_complex_noise" / "mixed_complex_noise_data.npy"),
    ]
    out: list[tuple[str, np.ndarray]] = []
    for label, path in paths:
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r")
        arr = canonicalize_11t(arr, str(path))
        count = min(int(arr.shape[0]), max_samples_per_file)
        if count < int(arr.shape[0]):
            idx = np.sort(rng.choice(int(arr.shape[0]), size=count, replace=False))
            sample = np.asarray(arr[idx], dtype=np.float32)
        else:
            sample = np.asarray(arr[:count], dtype=np.float32)
        out.append((label, sample))
    if not out:
        raise FileNotFoundError(f"No evaluation arrays found under {DATA_ROOT}")
    return out


def make_candidates(include_bounded_cubic: bool = False) -> list[Candidate]:
    candidates = [
        Candidate("linear_physical_then_log", "physical", interp_linear),
        Candidate("pchip_physical_then_log", "physical", interp_pchip),
        Candidate("akima_physical_then_log", "physical", interp_akima),
        Candidate("cubic_physical_then_log", "physical", interp_cubic),
        Candidate("linear_log_magnitude", "log_magnitude", interp_linear),
        Candidate("pchip_log_magnitude", "log_magnitude", interp_pchip),
        Candidate("akima_log_magnitude", "log_magnitude", interp_akima),
        Candidate("cubic_log_magnitude", "log_magnitude", interp_cubic),
    ]
    if include_bounded_cubic:
        candidates.insert(4, Candidate("bounded_cubic_physical_then_log", "physical", interp_bounded_cubic))
    return candidates


def reconstruct_log(
    sample_physical: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    candidate: Candidate,
) -> np.ndarray:
    n, r, t = sample_physical.shape
    source_physical = np.empty((n, r, source_coords.size), dtype=np.float32)
    for i in range(n):
        for j in range(r):
            source_physical[i, j] = interp_linear(target_coords, sample_physical[i, j], source_coords)

    out = np.empty((n, r, t), dtype=np.float32)
    if candidate.domain == "physical":
        for i in range(n):
            for j in range(r):
                rec = candidate.fn(source_coords, source_physical[i, j], target_coords)
                out[i, j] = to_log10_amplitude(rec)
    elif candidate.domain == "log_magnitude":
        source_log = to_log10_amplitude(source_physical)
        for i in range(n):
            for j in range(r):
                out[i, j] = candidate.fn(source_coords, source_log[i, j], target_coords)
    else:
        raise ValueError(candidate.domain)
    return out


def robust_percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64).reshape(-1), q))


def summarize_errors(true_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred_log - true_log, dtype=np.float64)
    abs_err = np.abs(err)
    diff_true = np.diff(true_log.astype(np.float64), axis=-1)
    diff_pred = np.diff(pred_log.astype(np.float64), axis=-1)
    d2_true = np.diff(true_log.astype(np.float64), n=2, axis=-1)
    d2_pred = np.diff(pred_log.astype(np.float64), n=2, axis=-1)
    t = true_log.shape[-1]
    slices = {
        "early": slice(0, t // 3),
        "mid": slice(t // 3, 2 * t // 3),
        "late": slice(2 * t // 3, t),
    }
    out = {
        "rmse_log": float(math.sqrt(float(np.mean(err * err)))),
        "mae_log": float(np.mean(abs_err)),
        "p95_abs_log": robust_percentile(abs_err, 95.0),
        "max_abs_log": float(np.max(abs_err)),
        "diff1_std_ratio": float(np.std(diff_pred) / max(float(np.std(diff_true)), 1.0e-12)),
        "diff2_std_ratio": float(np.std(d2_pred) / max(float(np.std(d2_true)), 1.0e-12)),
        "diff1_mae": float(np.mean(np.abs(diff_pred - diff_true))),
        "diff2_mae": float(np.mean(np.abs(d2_pred - d2_true))),
    }
    for key, sl in slices.items():
        local = err[..., sl]
        out[f"{key}_rmse_log"] = float(math.sqrt(float(np.mean(local * local))))
    return out


def score_row(row: dict[str, float]) -> float:
    # RMSE dominates, but over-smoothing is explicitly penalized because real-field
    # denoising depends on preserving noise-pattern statistics seen during training.
    diff1_penalty = abs(math.log(max(row["diff1_std_ratio"], 1.0e-6)))
    diff2_penalty = abs(math.log(max(row["diff2_std_ratio"], 1.0e-6)))
    return float(row["rmse_log"] + 0.08 * diff1_penalty + 0.04 * diff2_penalty)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_coords = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float64)
    patterns = real_gate_patterns()
    datasets = load_eval_arrays(max_samples_per_file=args.max_samples_per_file, seed=args.seed)
    candidates = make_candidates(include_bounded_cubic=args.include_bounded_cubic)

    rows: list[dict[str, object]] = []
    for pattern_name, source_coords in patterns.items():
        for dataset_name, sample in datasets:
            true_log = to_log10_amplitude(sample)
            for candidate in candidates:
                pred_log = reconstruct_log(sample, source_coords, target_coords, candidate)
                metrics = summarize_errors(true_log, pred_log)
                row: dict[str, object] = {
                    "pattern": pattern_name,
                    "source_gate_count": int(source_coords.size),
                    "dataset": dataset_name,
                    "sample_count": int(sample.shape[0]),
                    "candidate": candidate.name,
                    "domain": candidate.domain,
                    **metrics,
                }
                row["score"] = score_row(metrics)
                rows.append(row)

    csv_path = out_dir / "interpolation_benchmark_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict[str, float]] = {}
    for candidate in {str(row["candidate"]) for row in rows}:
        c_rows = [row for row in rows if row["candidate"] == candidate]
        aggregate[candidate] = {
            "mean_score": float(np.mean([float(row["score"]) for row in c_rows])),
            "mean_rmse_log": float(np.mean([float(row["rmse_log"]) for row in c_rows])),
            "mean_p95_abs_log": float(np.mean([float(row["p95_abs_log"]) for row in c_rows])),
            "mean_diff1_std_ratio": float(np.mean([float(row["diff1_std_ratio"]) for row in c_rows])),
            "mean_diff2_std_ratio": float(np.mean([float(row["diff2_std_ratio"]) for row in c_rows])),
        }
    ranked = sorted(aggregate.items(), key=lambda kv: kv[1]["mean_score"])

    by_pattern: dict[str, list[tuple[str, float]]] = {}
    for pattern_name in patterns:
        local: list[tuple[str, float]] = []
        for candidate in aggregate:
            c_rows = [
                row for row in rows
                if row["candidate"] == candidate and row["pattern"] == pattern_name
            ]
            local.append((candidate, float(np.mean([float(row["score"]) for row in c_rows]))))
        by_pattern[pattern_name] = sorted(local, key=lambda item: item[1])

    summary = {
        "purpose": "Choose a 40-gate-to-869 interpolation method that best matches the existing synthetic/noisy 11x869 training distribution.",
        "datasets": [{"name": name, "shape": list(arr.shape)} for name, arr in datasets],
        "patterns": {name: int(coords.size) for name, coords in patterns.items()},
        "ranking": [{"candidate": name, **metrics} for name, metrics in ranked],
        "best_by_pattern": {
            name: [{"candidate": c, "mean_score": s} for c, s in local[:3]]
            for name, local in by_pattern.items()
        },
        "csv": str(csv_path),
    }
    json_path = out_dir / "interpolation_benchmark_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Interpolation benchmark",
        "",
        "Lower score is better. Score = log RMSE plus penalties for first/second-difference over-smoothing.",
        "",
        "## Overall ranking",
        "",
        "| Rank | Candidate | Score | RMSE log | P95 abs log | diff1 ratio | diff2 ratio |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, (name, metrics) in enumerate(ranked, start=1):
        md_lines.append(
            f"| {idx} | {name} | {metrics['mean_score']:.6f} | "
            f"{metrics['mean_rmse_log']:.6f} | {metrics['mean_p95_abs_log']:.6f} | "
            f"{metrics['mean_diff1_std_ratio']:.3f} | {metrics['mean_diff2_std_ratio']:.3f} |"
        )
    md_lines.extend(["", "## Best by real gate pattern", ""])
    for pattern_name, local in by_pattern.items():
        best = ", ".join(f"{c} ({s:.6f})" for c, s in local[:3])
        md_lines.append(f"- {pattern_name}: {best}")
    md_path = out_dir / "interpolation_benchmark_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("Best overall:", ranked[0][0], ranked[0][1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 40-gate to 869-gate interpolation methods against existing 11x869 training data.")
    parser.add_argument("--max-samples-per-file", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--include-bounded-cubic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
