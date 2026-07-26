from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import PchipInterpolator


SCRIPT_PATH = Path(__file__).absolute()
WORKSPACE = SCRIPT_PATH.parents[2]
SOURCE_DIR = WORKSPACE / "实测" / "插值" / "插值869"
OUT_DIR = WORKSPACE / "实测" / "插值" / "PCHIP_物理域插值869"
FIG_DIR = OUT_DIR / "曲线图"

SEQ_LEN = 869
RECEIVER_COUNT = 11
EPS = 1.0e-12


def to_log10_amplitude(array: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.asarray(array, dtype=np.float32)) + EPS).astype(np.float32)


def pchip_1d(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    x = np.asarray(x_src, dtype=np.float64).reshape(-1)
    y = np.asarray(y_src, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.full(np.asarray(x_dst).shape, np.nan, dtype=np.float32)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    dst = np.asarray(x_dst, dtype=np.float64).reshape(-1)
    if x.size < 4:
        vals = np.interp(dst, x, y)
    else:
        vals = PchipInterpolator(x, y, extrapolate=True)(dst)
        if not np.all(np.isfinite(vals)):
            vals = np.interp(dst, x, y)
    return np.asarray(vals, dtype=np.float32)


def pchip_rows(raw_time_by_receiver: np.ndarray, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_time_by_receiver, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"expected 2D raw array, got {raw.shape}")
    return np.stack([pchip_1d(x_src, raw[:, j], x_dst) for j in range(raw.shape[1])], axis=1)


def selected_receiver_indices(count: int) -> np.ndarray:
    if count <= RECEIVER_COUNT:
        return np.arange(count)
    return np.unique(np.round(np.linspace(0, count - 1, RECEIVER_COUNT)).astype(int))


def save_plot(
    label: str,
    x_raw: np.ndarray,
    raw_time_by_receiver: np.ndarray,
    x_interp: np.ndarray,
    interp_time_by_receiver: np.ndarray,
    x_label: str,
    out_path: Path,
) -> None:
    raw = np.asarray(raw_time_by_receiver, dtype=np.float64)
    interp = np.asarray(interp_time_by_receiver, dtype=np.float64)
    idx = selected_receiver_indices(raw.shape[1])

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=180)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(idx)))
    for color, j in zip(colors, idx):
        y_raw = np.log10(np.abs(raw[:, j]) + EPS)
        y_interp = np.log10(np.abs(interp[:, j]) + EPS)
        ax.plot(x_interp, y_interp, color=color, linewidth=1.35, alpha=0.92)
        ax.scatter(x_raw, y_raw, color=color, s=12, alpha=0.72, edgecolors="none")

    ax.set_title(f"{label} measured curves: raw gates and PCHIP 869-grid interpolation")
    ax.set_xlabel(x_label)
    ax.set_ylabel("log10(abs(voltage) + 1e-12)")
    ax.grid(True, linewidth=0.45, alpha=0.28)
    ax.margins(x=0.015)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_summary(rows: list[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), dpi=170)
    for ax, row in zip(axes.reshape(-1), rows):
        raw = np.asarray(row["raw"], dtype=np.float64)
        interp = np.asarray(row["interp"], dtype=np.float64)
        x_raw = np.asarray(row["x_raw"], dtype=np.float64)
        x_interp = np.asarray(row["x_interp"], dtype=np.float64)
        idx = selected_receiver_indices(raw.shape[1])
        colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(idx)))
        for color, j in zip(colors, idx):
            ax.plot(x_interp, np.log10(np.abs(interp[:, j]) + EPS), color=color, linewidth=0.95, alpha=0.9)
            ax.scatter(x_raw, np.log10(np.abs(raw[:, j]) + EPS), color=color, s=7, alpha=0.6, edgecolors="none")
        ax.set_title(str(row["label"]))
        ax.grid(True, linewidth=0.35, alpha=0.24)
        ax.set_xlabel(str(row["x_label"]))
        ax.set_ylabel("log10 abs V")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def process_dbz(path: Path) -> dict[str, object]:
    mat = loadmat(path)
    raw = np.asarray(mat["raw_voltage_40x11"], dtype=np.float32)
    x_src = np.asarray(mat["normalized_gate_40"], dtype=np.float64).reshape(-1)
    x_dst = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float64)
    interp = pchip_rows(raw, x_src, x_dst)
    x_model_physical = interp.T.astype(np.float32)
    payload = {
        "raw_voltage_40x11": raw.astype(np.float32),
        "raw_log10_11x40": to_log10_amplitude(raw.T),
        "x_model_physical_11x869": x_model_physical,
        "x_model_physical_869x11": interp.astype(np.float32),
        "x_model_log10_11x869": to_log10_amplitude(x_model_physical),
        "x_model_magnitude_11x869": np.abs(x_model_physical).astype(np.float32),
        "x_model_magnitude_869x11": np.abs(interp).astype(np.float32),
        "normalized_gate_40": x_src.reshape(1, -1).astype(np.float32),
        "normalized_gate_869": x_dst.reshape(1, -1).astype(np.float32),
        "receiver_index_1based": np.arange(1, RECEIVER_COUNT + 1, dtype=np.int32).reshape(1, -1),
        "interpolation_method": "PCHIP physical-domain interpolation over normalized ordered gate coordinate, then log10(abs(x)+1e-12). Chosen as shape-preserving and no-overshoot conservative field interpolation.",
        "shape_note": "x_model_physical_11x869 matches train_data.npy physical-domain layout; x_model_log10_11x869 is direct ST-DiffTEM tensor.",
    }
    out_path = OUT_DIR / "dbz_11x869_pchip.mat"
    savemat(out_path, payload)
    save_plot("dbz", x_src, raw, x_dst, interp, "normalized gate coordinate", FIG_DIR / "dbz_pchip_measured_curves")
    return {
        "label": "dbz",
        "mat": str(out_path),
        "raw": raw,
        "interp": interp,
        "x_raw": x_src,
        "x_interp": x_dst,
        "x_label": "normalized gate",
    }


def process_taochong(path: Path, profile: str) -> dict[str, object]:
    mat = loadmat(path)
    raw = np.asarray(mat["original_voltage"], dtype=np.float32)
    source_time = np.asarray(mat["original_time_s"], dtype=np.float64).reshape(-1)
    source_log_time = np.log10(source_time)
    target_log_time = np.asarray(mat["log10_time_interp"], dtype=np.float64).reshape(-1)
    target_time = np.power(10.0, target_log_time)
    interp = pchip_rows(raw, source_log_time, target_log_time)
    x_model_physical = interp.T.astype(np.float32)

    payload = {
        "x_model_physical_11x869": x_model_physical,
        "x_model_physical_869x11": interp.astype(np.float32),
        "x_model_log10_11x869": to_log10_amplitude(x_model_physical),
        "x_model_magnitude_11x869": np.abs(x_model_physical).astype(np.float32),
        "x_model_magnitude_869x11": np.abs(interp).astype(np.float32),
        "time_interp_s": target_time.reshape(1, -1).astype(np.float64),
        "log10_time_interp": target_log_time.reshape(1, -1).astype(np.float64),
        "original_time_s": source_time.reshape(-1, 1).astype(np.float64),
        "original_voltage": raw.astype(np.float32),
        "original_width_s": np.asarray(mat["original_width_s"], dtype=np.float32),
        "original_error_bar": np.asarray(mat["original_error_bar"], dtype=np.float32),
        "original_mask": np.asarray(mat["original_mask"], dtype=np.float32),
        "selected_station_m": np.asarray(mat["selected_station_m"], dtype=np.float32),
        "selected_files": np.asarray(mat["selected_files"], dtype=object),
        "location_xyz": np.asarray(mat["location_xyz"], dtype=np.float32),
        "profile_name": profile,
        "interpolation_method": "PCHIP physical-domain interpolation over measured log10(time_s), then log10(abs(x)+1e-12). Chosen as shape-preserving and no-overshoot conservative field interpolation.",
        "shape_note": "x_model_physical_11x869 matches train_data.npy physical-domain layout; x_model_log10_11x869 is direct ST-DiffTEM tensor.",
    }
    out_path = OUT_DIR / f"taochong_{profile}_selected_11x869_pchip.mat"
    savemat(out_path, payload)
    save_plot(profile, source_time, raw, target_time, interp, "time (s)", FIG_DIR / f"{profile}_pchip_measured_curves")
    return {
        "label": profile,
        "mat": str(out_path),
        "raw": raw,
        "interp": interp,
        "x_raw": source_time,
        "x_interp": target_time,
        "x_label": "time (s)",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        process_dbz(SOURCE_DIR / "dbz_11x869.mat"),
        process_taochong(SOURCE_DIR / "taochong_W21_selected_11x869.mat", "W21"),
        process_taochong(SOURCE_DIR / "taochong_W23_selected_11x869.mat", "W23"),
        process_taochong(SOURCE_DIR / "taochong_W25_selected_11x869.mat", "W25"),
    ]
    plot_summary(rows, FIG_DIR / "all_pchip_measured_curves")

    manifest = {
        "chosen_algorithm": "PCHIP physical-domain interpolation, then log10(abs(x)+1e-12)",
        "reason": "PCHIP is shape-preserving and avoids cubic overshoot, making it the most conservative choice for measured-field curves used in underground-distribution comparison.",
        "mat_files": [row["mat"] for row in rows],
        "figure_dir": str(FIG_DIR),
    }
    (OUT_DIR / "README_PCHIP_物理域插值说明.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote PCHIP mat files to {OUT_DIR}")
    print(f"Wrote measured curve figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
