from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


WORKSPACE = Path(__file__).absolute().parents[2]
PROCESSED_DIR = WORKSPACE / "补充" / "processed_11x869"
OUT_DIR = PROCESSED_DIR / "comparison_figures"
EPS = 1.0e-12


DATASETS = [
    ("dbz", "dbz_11x869.mat"),
    ("W21", "taochong_W21_selected_11x869.mat"),
    ("W23", "taochong_W23_selected_11x869.mat"),
    ("W25", "taochong_W25_selected_11x869.mat"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10.5,
            "axes.titlesize": 15.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    mat = sio.loadmat(path)
    out: dict[str, np.ndarray] = {}

    if "x_model_magnitude_869x11" in mat:
        out["raw_869"] = np.asarray(mat["x_model_magnitude_869x11"], dtype=np.float64)
    elif "x_model_magnitude_11x869" in mat:
        out["raw_869"] = np.asarray(mat["x_model_magnitude_11x869"], dtype=np.float64).T
    else:
        raise KeyError(f"{path} has no x_model_magnitude_869x11 or x_model_magnitude_11x869")

    if "model_magnitude_869x11" in mat:
        out["denoised_869"] = np.asarray(mat["model_magnitude_869x11"], dtype=np.float64)
    elif "model_magnitude_11x869" in mat:
        out["denoised_869"] = np.asarray(mat["model_magnitude_11x869"], dtype=np.float64).T
    else:
        raise KeyError(f"{path} has no model_magnitude_869x11 or model_magnitude_11x869")

    if "normalized_gate_869" in mat:
        out["time_869"] = np.asarray(mat["normalized_gate_869"], dtype=np.float64).reshape(-1)
    elif "time_interp_s" in mat:
        out["time_869"] = np.asarray(mat["time_interp_s"], dtype=np.float64).reshape(-1)
    else:
        out["time_869"] = np.arange(out["raw_869"].shape[0], dtype=np.float64)

    if "raw_voltage_40x11" in mat and "model_magnitude_40x11" in mat:
        out["raw_original"] = np.abs(np.asarray(mat["raw_voltage_40x11"], dtype=np.float64))
        out["denoised_original"] = np.asarray(mat["model_magnitude_40x11"], dtype=np.float64)
    elif "original_voltage" in mat and "model_magnitude_original_gates" in mat:
        out["raw_original"] = np.abs(np.asarray(mat["original_voltage"], dtype=np.float64))
        out["denoised_original"] = np.asarray(mat["model_magnitude_original_gates"], dtype=np.float64)

    if "original_time_s" in mat:
        out["time_original"] = np.asarray(mat["original_time_s"], dtype=np.float64).reshape(-1)
    elif "normalized_gate_40" in mat:
        out["time_original"] = np.asarray(mat["normalized_gate_40"], dtype=np.float64).reshape(-1)
    elif "raw_original" in out:
        out["time_original"] = np.arange(out["raw_original"].shape[0], dtype=np.float64)

    return out


def gate_indices(n_time: int, max_lines: int) -> np.ndarray:
    if n_time <= max_lines:
        return np.arange(n_time, dtype=int)
    return np.unique(np.round(np.linspace(0, n_time - 1, max_lines)).astype(int))


def finite_positive_range(*arrays: np.ndarray) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64)
        mask = np.isfinite(a) & (a > 0)
        if np.any(mask):
            vals.append(a[mask])
    if not vals:
        return EPS, 1.0
    merged = np.concatenate(vals)
    lo, hi = np.percentile(merged, [0.5, 99.5])
    return max(float(lo) * 0.55, EPS), max(float(hi) * 1.8, EPS * 10)


def plot_curve_family(raw: np.ndarray, denoised: np.ndarray, title: str, out_path: Path, max_lines: int) -> None:
    raw = np.asarray(raw, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    if raw.shape != denoised.shape:
        raise ValueError(f"{title}: shape mismatch {raw.shape} vs {denoised.shape}")
    if raw.ndim != 2 or raw.shape[1] != 11:
        raise ValueError(f"{title}: expected time x 11 matrix, got {raw.shape}")

    idx = gate_indices(raw.shape[0], max_lines=max_lines)
    stations = np.arange(1, raw.shape[1] + 1, dtype=float)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, 20))
    ymin, ymax = finite_positive_range(raw[idx], denoised[idx])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2), sharey=True)
    panels = [("Raw TEM data", raw), ("Denoised TEM data", denoised)]
    for ax, (panel_title, data) in zip(axes, panels):
        for line_no, gate_id in enumerate(idx):
            ax.plot(
                stations,
                np.maximum(data[gate_id], EPS),
                color=colors[line_no % len(colors)],
                linewidth=0.72 if raw.shape[0] > 80 else 1.1,
                alpha=0.42 if raw.shape[0] > 80 else 0.82,
            )
        ax.set_yscale("log")
        ax.set_xlim(1, 11)
        ax.set_xticks(np.arange(1, 12, 1))
        ax.set_ylim(ymin, ymax)
        ax.grid(True, which="major", color="#C9D1DA", linewidth=0.65)
        ax.grid(True, which="minor", color="#E8EDF2", linewidth=0.35, alpha=0.75)
        ax.set_xlabel("Station index")
        ax.set_title(panel_title, pad=8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    axes[0].set_ylabel(r"$dB/dt$ magnitude (V/Am$^2$)")
    fig.suptitle(title, y=0.985, fontsize=11)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.16, top=0.80, wspace=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=360, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_pseudosection(raw: np.ndarray, denoised: np.ndarray, title: str, out_path: Path) -> None:
    from matplotlib.colors import LogNorm

    raw = np.asarray(raw, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    stations = np.arange(1, raw.shape[1] + 1, dtype=float)
    time_idx = np.arange(raw.shape[0], dtype=float)
    x_edges = np.r_[0.5, np.arange(1.5, raw.shape[1] + 0.5, 1.0), raw.shape[1] + 0.5]
    y_edges = np.r_[-0.5, np.arange(0.5, raw.shape[0] - 0.5, 1.0), raw.shape[0] - 0.5]
    if y_edges.size != raw.shape[0] + 1:
        y_edges = np.linspace(-0.5, raw.shape[0] - 0.5, raw.shape[0] + 1)
    if x_edges.size != raw.shape[1] + 1:
        x_edges = np.linspace(0.5, raw.shape[1] + 0.5, raw.shape[1] + 1)

    vmin, vmax = finite_positive_range(raw, denoised)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True)
    for ax, panel_title, data in zip(axes, ["Raw", "Denoised"], [raw, denoised]):
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.maximum(data, EPS),
            shading="auto",
            cmap="YlGnBu",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Station index")
        ax.set_xticks(stations)
        ax.invert_yaxis()
    axes[0].set_ylabel("Interpolated gate index")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.86, pad=0.02)
    cbar.set_label(r"$dB/dt$ magnitude (V/Am$^2$)")
    fig.suptitle(title, y=1.02, fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_change_diagnostics(raw: np.ndarray, denoised: np.ndarray, title: str, out_path: Path, max_lines: int) -> None:
    raw = np.maximum(np.asarray(raw, dtype=np.float64), EPS)
    denoised = np.maximum(np.asarray(denoised, dtype=np.float64), EPS)
    if raw.shape != denoised.shape:
        raise ValueError(f"{title}: shape mismatch {raw.shape} vs {denoised.shape}")
    ratio = denoised / raw
    log_delta = np.log10(ratio)
    idx = gate_indices(raw.shape[0], max_lines=max_lines)
    stations = np.arange(1, raw.shape[1] + 1, dtype=float)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, 20))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    ax = axes[0]
    for line_no, gate_id in enumerate(idx):
        ax.plot(
            stations,
            ratio[gate_id],
            color=colors[line_no % len(colors)],
            linewidth=0.72 if raw.shape[0] > 80 else 1.1,
            alpha=0.38 if raw.shape[0] > 80 else 0.82,
        )
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.axhline(0.5, color="#555555", linewidth=0.55, linestyle=":", alpha=0.55)
    ax.axhline(2.0, color="#555555", linewidth=0.55, linestyle=":", alpha=0.55)
    ax.set_yscale("log")
    ax.set_xlim(1, 11)
    ax.set_xticks(np.arange(1, 12, 1))
    ax.set_xlabel("Station index")
    ax.set_ylabel("Denoised / Raw")
    ax.set_title("Gate-wise ratio")
    ax.grid(True, which="major", color="#C9D1DA", linewidth=0.6)
    ax.grid(True, which="minor", color="#E8EDF2", linewidth=0.32, alpha=0.7)

    vmax = float(np.nanpercentile(np.abs(log_delta), 99.0))
    vmax = max(vmax, 0.05)
    mesh = axes[1].imshow(
        log_delta,
        aspect="auto",
        origin="upper",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[0.5, 11.5, raw.shape[0] - 0.5, -0.5],
    )
    axes[1].set_xlabel("Station index")
    axes[1].set_ylabel("Gate index")
    axes[1].set_xticks(np.arange(1, 12, 1))
    axes[1].set_title(r"$\log_{10}(\mathrm{Denoised}/\mathrm{Raw})$")
    cbar = fig.colorbar(mesh, ax=axes[1], shrink=0.88, pad=0.02)
    cbar.set_label("log10 ratio")

    fig.suptitle(title, y=0.99, fontsize=12)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.84, wspace=0.20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=360, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_readable_change_summary(
    datasets: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(datasets), 3, figsize=(14.2, 3.1 * len(datasets)))
    if len(datasets) == 1:
        axes = np.asarray([axes])

    for row_idx, (name, raw, denoised) in enumerate(datasets):
        raw = np.maximum(np.asarray(raw, dtype=np.float64), EPS)
        denoised = np.maximum(np.asarray(denoised, dtype=np.float64), EPS)
        ratio = denoised / raw
        pct = (ratio - 1.0) * 100.0
        log_delta = np.log10(ratio)
        gates = np.arange(1, raw.shape[0] + 1)
        stations = np.arange(1, raw.shape[1] + 1)

        ax = axes[row_idx, 0]
        selected = np.unique(np.round(np.linspace(0, raw.shape[0] - 1, min(8, raw.shape[0]))).astype(int))
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(selected)))
        for color, gate_id in zip(colors, selected):
            ax.plot(stations, raw[gate_id], color=color, linewidth=1.2, alpha=0.85)
            ax.plot(stations, denoised[gate_id], color=color, linewidth=1.2, linestyle="--", alpha=0.95)
        ax.set_yscale("log")
        ax.set_xlim(1, 11)
        ax.set_xticks(np.arange(1, 12, 1))
        ax.grid(True, which="major", color="#C9D1DA", linewidth=0.58)
        ax.grid(True, which="minor", color="#E8EDF2", linewidth=0.32, alpha=0.72)
        ax.set_title("Raw solid / Denoised dashed")
        ax.set_xlabel("Station index")
        ax.set_ylabel(f"{name}\n" + r"$|dB/dt|$")

        ax = axes[row_idx, 1]
        avg_abs_pct = np.nanmean(np.abs(pct), axis=1)
        signed_pct = np.nanmedian(pct, axis=1)
        ax.plot(gates, avg_abs_pct, color="#D62728", linewidth=1.4, label="mean abs %")
        ax.plot(gates, signed_pct, color="#1F77B4", linewidth=1.0, label="median signed %")
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xlim(1, raw.shape[0])
        ax.grid(True, color="#D7DEE6", linewidth=0.5)
        ax.set_title("Change by gate")
        ax.set_xlabel("Original gate index")
        ax.set_ylabel("Change (%)")
        if row_idx == 0:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[row_idx, 2]
        vmax = float(np.nanpercentile(np.abs(log_delta), 98.0))
        vmax = max(vmax, 0.05)
        mesh = ax.imshow(
            log_delta,
            aspect="auto",
            origin="upper",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            extent=[0.5, 11.5, raw.shape[0] + 0.5, 0.5],
        )
        ax.set_title("Red = larger, blue = smaller")
        ax.set_xlabel("Station index")
        ax.set_ylabel("Original gate index")
        ax.set_xticks(np.arange(1, 12, 1))
        fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.015, label="log10 ratio")

    fig.suptitle("Readable model-change summary on original gates", y=0.995, fontsize=13)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.055, top=0.945, wspace=0.25, hspace=0.38)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=360, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def roughness(arr: np.ndarray, axis: int) -> float:
    diff = np.diff(np.log10(np.maximum(np.asarray(arr, dtype=np.float64), EPS)), n=2, axis=axis)
    return float(np.nanmean(np.abs(diff))) if diff.size else 0.0


def metrics_for(name: str, raw: np.ndarray, denoised: np.ndarray, domain: str) -> dict[str, float | str | int]:
    raw = np.maximum(np.asarray(raw, dtype=np.float64), EPS)
    denoised = np.maximum(np.asarray(denoised, dtype=np.float64), EPS)
    ratio = denoised / raw
    return {
        "dataset": name,
        "domain": domain,
        "time_gates": int(raw.shape[0]),
        "stations": int(raw.shape[1]),
        "median_model_raw_ratio": float(np.nanmedian(ratio)),
        "p05_model_raw_ratio": float(np.nanpercentile(ratio, 5)),
        "p95_model_raw_ratio": float(np.nanpercentile(ratio, 95)),
        "frac_ratio_outside_0p5_2": float(np.nanmean((ratio < 0.5) | (ratio > 2.0))),
        "temporal_roughness_raw": roughness(raw, axis=0),
        "temporal_roughness_denoised": roughness(denoised, axis=0),
        "spatial_roughness_raw": roughness(raw, axis=1),
        "spatial_roughness_denoised": roughness(denoised, axis=1),
    }


def plot_combined_curve_family(
    loaded: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
    out_path: Path,
    max_lines: int,
) -> None:
    fig, axes = plt.subplots(len(loaded), 2, figsize=(12.8, 3.15 * len(loaded)), sharex=True)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, 20))
    if len(loaded) == 1:
        axes = np.asarray([axes])

    for row_idx, (name, raw, denoised) in enumerate(loaded):
        raw = np.asarray(raw, dtype=np.float64)
        denoised = np.asarray(denoised, dtype=np.float64)
        idx = gate_indices(raw.shape[0], max_lines=max_lines)
        stations = np.arange(1, raw.shape[1] + 1, dtype=float)
        ymin, ymax = finite_positive_range(raw[idx], denoised[idx])
        for col_idx, (panel_title, data) in enumerate([("Raw", raw), ("Denoised", denoised)]):
            ax = axes[row_idx, col_idx]
            for line_no, gate_id in enumerate(idx):
                ax.plot(
                    stations,
                    np.maximum(data[gate_id], EPS),
                    color=colors[line_no % len(colors)],
                    linewidth=0.58 if raw.shape[0] > 80 else 0.9,
                    alpha=0.34 if raw.shape[0] > 80 else 0.78,
                )
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(1, 11)
            ax.set_xticks(np.arange(1, 12, 1))
            ax.grid(True, which="major", color="#C9D1DA", linewidth=0.58)
            ax.grid(True, which="minor", color="#E8EDF2", linewidth=0.32, alpha=0.72)
            if row_idx == 0:
                ax.set_title(panel_title, pad=7)
            if col_idx == 0:
                ax.set_ylabel(f"{name}\n" + r"$|dB/dt|$")
            if row_idx == len(loaded) - 1:
                ax.set_xlabel("Station index")
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
    fig.suptitle(title, y=0.992, fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.992, bottom=0.055, top=0.935, wspace=0.09, hspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=360, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    combined_869: list[tuple[str, np.ndarray, np.ndarray]] = []
    combined_original: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, filename in DATASETS:
        data = load_dataset(PROCESSED_DIR / filename)
        combined_869.append((name, data["raw_869"], data["denoised_869"]))
        plot_curve_family(
            data["raw_869"],
            data["denoised_869"],
            f"{name}: interpolated 869-gate model grid",
            OUT_DIR / f"{name}_raw_vs_denoised_curve_family_869.png",
            max_lines=96,
        )
        plot_pseudosection(
            data["raw_869"],
            data["denoised_869"],
            f"{name}: interpolated 869-gate pseudosection",
            OUT_DIR / f"{name}_raw_vs_denoised_pseudosection_869.png",
        )
        plot_change_diagnostics(
            data["raw_869"],
            data["denoised_869"],
            f"{name}: model change on 869-gate grid",
            OUT_DIR / f"{name}_denoised_over_raw_diagnostics_869.png",
            max_lines=96,
        )
        rows.append(metrics_for(name, data["raw_869"], data["denoised_869"], "interp869"))
        if "raw_original" in data and "denoised_original" in data:
            combined_original.append((name, data["raw_original"], data["denoised_original"]))
            plot_curve_family(
                data["raw_original"],
                data["denoised_original"],
                f"{name}: sampled back to original gates",
                OUT_DIR / f"{name}_raw_vs_denoised_curve_family_original_gates.png",
                max_lines=80,
            )
            plot_change_diagnostics(
                data["raw_original"],
                data["denoised_original"],
                f"{name}: model change sampled back to original gates",
                OUT_DIR / f"{name}_denoised_over_raw_diagnostics_original_gates.png",
                max_lines=80,
            )
            rows.append(metrics_for(name, data["raw_original"], data["denoised_original"], "original_gates"))

    plot_combined_curve_family(
        combined_869,
        "All processed records: interpolated 869-gate model grid",
        OUT_DIR / "all_raw_vs_denoised_curve_family_869.png",
        max_lines=96,
    )
    plot_combined_curve_family(
        combined_original,
        "All processed records: sampled back to original gates",
        OUT_DIR / "all_raw_vs_denoised_curve_family_original_gates.png",
        max_lines=80,
    )
    if combined_original:
        plot_readable_change_summary(
            combined_original,
            OUT_DIR / "readable_model_change_summary_original_gates.png",
        )

    with (OUT_DIR / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"written: {OUT_DIR}")
    for row in rows:
        print(
            f"{row['dataset']} {row['domain']}: gates={row['time_gates']} "
            f"median_ratio={row['median_model_raw_ratio']:.3f} "
            f"outside={row['frac_ratio_outside_0p5_2']:.3f}"
        )


if __name__ == "__main__":
    main()
