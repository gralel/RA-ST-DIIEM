from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pywt


EPS = 1e-12


def to_log_magnitude(array: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(array).astype(np.float64) + EPS)


def metrics(pred_log: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    # Match the learned-method evaluator: both signals retain the log transform's
    # additive epsilon when mapped back to magnitude space.
    prediction = np.power(10.0, pred_log)
    target = np.abs(clean).astype(np.float64) + EPS
    error = prediction - target
    mse = np.mean(error**2, axis=(1, 2))
    signal = np.mean(target**2, axis=(1, 2))
    rrmse = np.sqrt(mse / (signal + EPS))
    snr = 10.0 * np.log10(np.sum(target**2, axis=(1, 2)) / (np.sum(error**2, axis=(1, 2)) + EPS))
    return {
        "rrmse": float(rrmse.mean()),
        "rrmse_std_over_samples": float(rrmse.std(ddof=1)),
        "snr": float(snr.mean()),
    }


def wavelet_denoise(log_data: np.ndarray, scale: float, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    output = np.empty_like(log_data, dtype=np.float64)
    for sample_index in range(log_data.shape[0]):
        coeffs = pywt.wavedec(log_data[sample_index], wavelet=wavelet, level=level, axis=-1, mode="symmetric")
        finest = coeffs[-1]
        sigma = np.median(np.abs(finest), axis=-1, keepdims=True) / 0.6744897501960817
        threshold = scale * sigma * np.sqrt(2.0 * np.log(log_data.shape[-1]))
        filtered = [coeffs[0]] + [pywt.threshold(detail, threshold, mode="soft") for detail in coeffs[1:]]
        output[sample_index] = pywt.waverec(filtered, wavelet=wavelet, axis=-1, mode="symmetric")[..., : log_data.shape[-1]]
    return output


def pca_denoise(log_data: np.ndarray, rank: int) -> np.ndarray:
    mean = log_data.mean(axis=-1, keepdims=True)
    centered = log_data - mean
    u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(rank, u.shape[-1])
    reconstructed = np.einsum(
        "nrk,nk,nkt->nrt",
        u[:, :, :rank],
        singular[:, :rank],
        vh[:, :rank, :],
        optimize=True,
    )
    return reconstructed + mean


def load_pair(data_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    noisy = np.load(data_path, mmap_mode="r")
    clean = np.load(label_path, mmap_mode="r")
    if noisy.shape != clean.shape or noisy.ndim != 3:
        raise ValueError(f"unexpected pair shapes: {noisy.shape}, {clean.shape}")
    return noisy, clean


def scenario_paths(data_root: Path) -> list[tuple[str, Path, Path]]:
    manifest = json.loads((data_root / "test_scenario_manifest.json").read_text(encoding="utf-8"))
    paths = []
    for item in manifest["scenarios"]:
        slug = item["slug"]
        directory = data_root / "test_scenarios" / slug
        paths.append((slug, directory / f"{slug}_data.npy", directory / f"{slug}_labels.npy"))
    return paths


def tune(
    name: str,
    candidates: list[object],
    denoiser_factory: Callable[[object], Callable[[np.ndarray], np.ndarray]],
    noisy: np.ndarray,
    clean: np.ndarray,
) -> tuple[object, list[dict[str, object]]]:
    log_noisy = to_log_magnitude(noisy)
    rows = []
    for candidate in candidates:
        prediction = denoiser_factory(candidate)(log_noisy)
        result = metrics(prediction, clean)
        rows.append({"method": name, "candidate": candidate, **result})
        print(f"[validation] {name} {candidate}: RRMSE={result['rrmse']:.6f}")
    best = min(rows, key=lambda row: float(row["rrmse"]))["candidate"]
    return best, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and evaluate conventional TEM denoising baselines.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_noisy, val_clean = load_pair(
        args.data_root / "val" / "val_data.npy",
        args.data_root / "val" / "val_labels.npy",
    )
    methods: dict[str, tuple[object, Callable[[np.ndarray], np.ndarray]]] = {}
    tuning_rows: list[dict[str, object]] = []

    wavelet_best, rows = tune(
        "Wavelet-db4",
        [0.25 * index for index in range(1, 17)] + [5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0, 32.0],
        lambda value: lambda data: wavelet_denoise(data, float(value)),
        val_noisy,
        val_clean,
    )
    methods["Wavelet-db4"] = (wavelet_best, lambda data: wavelet_denoise(data, float(wavelet_best)))
    tuning_rows.extend(rows)

    pca_best, rows = tune(
        "Receiver-PCA",
        list(range(1, 11)),  # Rank 11 is the unchanged input, not a denoising reconstruction.
        lambda value: lambda data: pca_denoise(data, int(value)),
        val_noisy,
        val_clean,
    )
    methods["Receiver-PCA"] = (pca_best, lambda data: pca_denoise(data, int(pca_best)))
    tuning_rows.extend(rows)

    (args.output_dir / "validation_selection.json").write_text(
        json.dumps(tuning_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows_out: list[dict[str, object]] = []
    test_sets = [("overall", args.data_root / "test" / "test_data.npy", args.data_root / "test" / "test_labels.npy")]
    test_sets.extend(scenario_paths(args.data_root))
    for method, (parameter, denoiser) in methods.items():
        for scenario, data_path, label_path in test_sets:
            noisy, clean = load_pair(data_path, label_path)
            result = metrics(denoiser(to_log_magnitude(noisy)), clean)
            row = {"method": method, "selected_parameter": parameter, "scenario": scenario, **result}
            rows_out.append(row)
            print(f"[test] {method} {scenario}: RRMSE={result['rrmse']:.6f}")

    with (args.output_dir / "conventional_baselines.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    (args.output_dir / "conventional_baselines.json").write_text(
        json.dumps(rows_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
