"""Fit the optional output-strength calibrator from validation predictions.

The input manifest lists validation-only arrays and clean-model identifiers.
No test scenario labels or test targets are used in fitting or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.output_strength_calibrator import feature_matrix, receiver_feature_matrix  # noqa: E402

CONFIG = {"max_iter": 60, "max_leaf_nodes": 6, "min_samples_leaf": 80, "l2_regularization": 1.0}
GRID = np.arange(.35, 1.201, .05)
RECEIVER_CONFIG = {"max_iter": 100, "max_leaf_nodes": 20, "min_samples_leaf": 120, "l2_regularization": 1.0}
RECEIVER_GRID = np.arange(0.0, 1.201, 0.05, dtype=np.float32)
EPS = 1e-12


def log_magnitude(array: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.asarray(array, dtype=np.float32)) + EPS).astype(np.float32)


def sample_rrmse(noisy: np.ndarray, pred: np.ndarray, clean: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    if not np.isscalar(alpha):
        alpha = np.asarray(alpha, dtype=np.float32)
        alpha = alpha[:, None, None] if alpha.ndim == 1 else alpha[:, :, None]
    estimated_log = noisy + alpha * (pred - noisy)
    estimated = np.power(10.0, estimated_log.astype(np.float64))
    target = np.power(10.0, clean.astype(np.float64))
    return np.sqrt(np.mean((estimated - target) ** 2, axis=(1, 2)) / (np.mean(target ** 2, axis=(1, 2)) + EPS))


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receiver-wise", action="store_true", help="Fit one output-strength coefficient per receiver")
    args = parser.parse_args()
    config = RECEIVER_CONFIG if args.receiver_wise else CONFIG
    grid = RECEIVER_GRID if args.receiver_wise else GRID
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a nonempty datasets list")
    base = args.manifest.resolve().parent
    prepared = []
    for entry in entries:
        if entry.get("role") not in {"validation", "matched_validation"}:
            raise ValueError(f"Only validation data may be fitted: {entry.get('name')}")
        noisy = log_magnitude(np.load(resolve(base, entry["noisy"])))
        clean = log_magnitude(np.load(resolve(base, entry["clean"])))
        pred = np.load(resolve(base, entry["prediction"])).astype(np.float32)
        records = [json.loads(line) for line in resolve(base, entry["model_records_jsonl"]).read_text(encoding="utf-8").splitlines()]
        groups = [r["model_name"] for r in records]
        if noisy.shape != clean.shape or noisy.shape != pred.shape or len(groups) != len(noisy):
            raise ValueError(f"Shape/model-name mismatch for {entry.get('name')}")
        if args.receiver_wise:
            clean_amp = np.power(10.0, clean.astype(np.float64))
            delta = pred - noisy
            trace_errors = np.stack([
                np.mean((np.power(10.0, (noisy + float(alpha) * delta).astype(np.float64)) - clean_amp) ** 2, axis=2)
                for alpha in grid
            ], axis=2)
            target_alpha = grid[np.argmin(trace_errors, axis=2)].reshape(-1)
            selected_features = receiver_feature_matrix(noisy, pred)
            feature_groups = np.repeat(groups, noisy.shape[1])
        else:
            risks = np.stack([sample_rrmse(noisy, pred, clean, float(a)) for a in grid], axis=1)
            target_alpha = grid[np.argmin(risks, axis=1)]
            selected_features = feature_matrix(noisy, pred)
            feature_groups = groups
        prepared.append({"name": entry["name"], "noisy": noisy, "clean": clean, "pred": pred,
                         "features": selected_features, "target_alpha": target_alpha,
                         "groups": feature_groups, "baseline": sample_rrmse(noisy, pred, clean, 1.0)})
    x = np.concatenate([d["features"] for d in prepared])
    y = np.concatenate([d["target_alpha"] for d in prepared])
    groups = np.array([name for d in prepared for name in d["groups"]])
    if len(set(groups)) < 5:
        raise ValueError("At least five distinct clean models are needed for grouped CV")
    predictions = np.empty(len(x), dtype=np.float32)
    for train_idx, valid_idx in GroupKFold(5).split(x, y, groups):
        model = HistGradientBoostingRegressor(**config, random_state=43)
        model.fit(x[train_idx], y[train_idx])
        predictions[valid_idx] = np.clip(model.predict(x[valid_idx]), grid[0], grid[-1])
    summary = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "scikit_learn_version": sklearn_version,
               "configuration": config, "receiver_wise": args.receiver_wise,
               "training_rows": int(len(x)), "clean_model_count": len(set(groups)), "datasets": {}}
    cursor = 0
    for d in prepared:
        stop = cursor + len(d["target_alpha"])
        alpha = predictions[cursor:stop]
        if args.receiver_wise:
            alpha = alpha.reshape(d["noisy"].shape[:2])
        cv = sample_rrmse(d["noisy"], d["pred"], d["clean"], alpha)
        summary["datasets"][d["name"]] = {"count": int(len(cv)), "baseline_rrmse": float(d["baseline"].mean()),
                                           "group_cv_rrmse": float(cv.mean())}
        cursor = stop
    final = HistGradientBoostingRegressor(**config, random_state=43)
    final.fit(x, y)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.output_dir / "output_strength_calibrator.joblib")
    (args.output_dir / "group_cv.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
