"""Apply the validation-fitted receiver calibration to frozen model outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.output_strength_calibrator import apply_receiver_calibrator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy", type=Path, required=True, help="Physical-domain noisy .npy array")
    parser.add_argument("--prediction", type=Path, required=True, help="Frozen log10-magnitude prediction .npy array")
    parser.add_argument("--calibrator", type=Path, required=True, help="Trusted local fitted .joblib file")
    parser.add_argument("--output", type=Path, required=True, help="Output log10-magnitude .npy path")
    parser.add_argument("--window", type=int, default=101, help="Validation-selected odd smoothing window")
    args = parser.parse_args()

    noisy = np.load(args.noisy).astype(np.float32)
    prediction = np.load(args.prediction).astype(np.float32)
    if noisy.shape != prediction.shape or noisy.ndim != 3:
        raise ValueError("noisy and prediction must have matching [samples, receivers, gates] shapes")
    noisy_log10 = np.log10(np.abs(noisy) + 1e-12)
    regressor = joblib.load(args.calibrator)
    corrected, _ = apply_receiver_calibrator(
        noisy_log10, prediction, regressor, early_only=True, savgol_window=args.window
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, corrected)
    print(f"Saved {len(corrected)} calibrated predictions to {args.output}")


if __name__ == "__main__":
    main()
