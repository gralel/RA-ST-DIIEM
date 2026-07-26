from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_processed_11x869_comparison import DATASETS, EPS, PROCESSED_DIR, finite_positive_range, load_dataset


OUT_DIR = PROCESSED_DIR / "comparison_figures"


def centers_to_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=np.float64)
    mid = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mid[0] - x[0])
    last = x[-1] + (x[-1] - mid[-1])
    return np.r_[first, mid, last]


def positive_time_edges(t: np.ndarray) -> np.ndarray | None:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.size < 2 or not np.all(np.isfinite(t)) or np.any(t <= 0) or np.any(np.diff(t) <= 0):
        return None
    return np.r_[t[0] / np.sqrt(t[1] / t[0]), np.sqrt(t[:-1] * t[1:]), t[-1] * np.sqrt(t[-1] / t[-2])]


def load_station_axis(mat_path: Path, n_stations: int) -> tuple[np.ndarray, str]:
    mat = sio.loadmat(mat_path)
    if "selected_station_m" in mat:
        station = np.asarray(mat["selected_station_m"], dtype=np.float64).reshape(-1)
        if station.size == n_stations and np.all(np.isfinite(station)):
            return station, "Station (m)"
    return np.arange(1, n_stations + 1, dtype=np.float64), "Station index"


def choose_trace_stations(raw: np.ndarray, denoised: np.ndarray) -> list[int]:
    raw = np.maximum(np.asarray(raw, dtype=np.float64), EPS)
    denoised = np.maximum(np.asarray(denoised, dtype=np.float64), EPS)
    score = np.nanmedian(np.abs(np.log10(denoised / raw)), axis=0)
    order = np.argsort(score)
    choices = [int(order[len(order) // 2]), int(order[-1])]
    out: list[int] = []
    for idx in choices:
        if idx not in out:
            out.append(idx)
    return out


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def apply_story_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_trace(ax: plt.Axes, x: np.ndarray, y: np.ndarray, use_log_x: bool, *args, **kwargs) -> None:
    if use_log_x:
        ax.loglog(x, y, *args, **kwargs)
    else:
        ax.semilogy(x, y, *args, **kwargs)


def plot_story(name: str, mat_path: Path, out_path: Path) -> None:
    data = load_dataset(mat_path)
    if "raw_original" not in data or "denoised_original" not in data:
        raise ValueError(f"{name}: no original-gate raw/denoised arrays found in {mat_path}")

    raw = np.maximum(data["raw_original"], EPS)
    denoised = np.maximum(data["denoised_original"], EPS)
    times = np.asarray(data.get("time_original", np.arange(raw.shape[0])), dtype=np.float64).reshape(-1)
    station, station_label = load_station_axis(mat_path, raw.shape[1])

    x_edges = centers_to_edges(station)
    t_edges = positive_time_edges(times)
    use_log_time = t_edges is not None
    if t_edges is None:
        y_centers = np.arange(1, raw.shape[0] + 1, dtype=np.float64)
        y_edges = centers_to_edges(y_centers)
        time_label = "Original gate index"
    else:
        y_centers = times
        y_edges = t_edges
        time_label = "Time (s)"

    vmin, vmax = finite_positive_range(raw, denoised)
    log_delta = np.log10(denoised / raw)
    diff_lim = float(np.nanpercentile(np.abs(log_delta[np.isfinite(log_delta)]), 98))
    diff_lim = min(max(diff_lim, 0.15), 1.0)

    fig = plt.figure(figsize=(11.8, 6.15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 1.0], hspace=0.33, wspace=0.25)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    trace_ax = fig.add_subplot(gs[1, 0:2])
    gate_ax = fig.add_subplot(gs[1, 2])

    heatmaps = [
        ("Raw measured record", raw, "a"),
        ("RA-ST-DiffTEM output", denoised, "b"),
    ]
    for ax, (title, arr, label) in zip(axes[:2], heatmaps):
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            arr,
            shading="auto",
            cmap="YlGnBu",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_title(title, pad=8)
        ax.set_xlabel(station_label)
        if use_log_time:
            ax.set_yscale("log")
        ax.invert_yaxis()
        panel_label(ax, f"({label})")
    axes[0].set_ylabel(time_label)
    axes[1].set_yticklabels([])
    cbar = fig.colorbar(mesh, ax=axes[:2], shrink=0.93, pad=0.012)
    cbar.set_label(r"$|dB/dt|$ magnitude")
    cbar.ax.tick_params(labelsize=7.0)

    mesh_diff = axes[2].pcolormesh(
        x_edges,
        y_edges,
        np.clip(log_delta, -diff_lim, diff_lim),
        shading="auto",
        cmap="coolwarm",
        vmin=-diff_lim,
        vmax=diff_lim,
    )
    axes[2].set_title(r"Log-amplitude correction $\Delta\log_{10}|V|$", pad=8)
    axes[2].set_xlabel(station_label)
    if use_log_time:
        axes[2].set_yscale("log")
    axes[2].invert_yaxis()
    axes[2].set_yticklabels([])
    panel_label(axes[2], "(c)")
    cbar_diff = fig.colorbar(mesh_diff, ax=axes[2], shrink=0.93, pad=0.015)
    cbar_diff.set_label(f"clipped at +/-{diff_lim:.2f} dex")
    cbar_diff.ax.tick_params(labelsize=7.0)

    colors = ["#0072B2", "#D55E00", "#009E73"]
    trace_ids = choose_trace_stations(raw, denoised)
    for color, col in zip(colors, trace_ids):
        label = f"{station[col]:.0f} m" if station_label.endswith("(m)") else f"Station {col + 1}"
        plot_trace(
            trace_ax,
            times if use_log_time else y_centers,
            raw[:, col],
            use_log_time,
            "o",
            color=color,
            markersize=3.0,
            alpha=0.42,
            label=f"Raw {label}",
        )
        plot_trace(
            trace_ax,
            times if use_log_time else y_centers,
            denoised[:, col],
            use_log_time,
            "-",
            color=color,
            linewidth=2.0,
            label=f"Output {label}",
        )
    trace_ax.set_title("Two representative decay traces", pad=8)
    trace_ax.set_xlabel(time_label)
    trace_ax.set_ylabel(r"$|dB/dt|$ magnitude")
    trace_ax.grid(True, which="major", color="#D6DEE6", linewidth=0.7)
    trace_ax.grid(True, which="minor", color="#EEF2F6", linewidth=0.35)
    trace_ax.legend(ncol=2, frameon=False, loc="upper right")
    panel_label(trace_ax, "(d)")

    abs_delta = np.abs(log_delta)
    q25_abs = np.nanpercentile(abs_delta, 25, axis=1)
    q75_abs = np.nanpercentile(abs_delta, 75, axis=1)
    median_abs = np.nanmedian(abs_delta, axis=1)
    x_gate = times if use_log_time else y_centers
    gate_ax.fill_between(x_gate, q25_abs, q75_abs, color="#E69F00", alpha=0.22, linewidth=0, label="IQR")
    gate_ax.plot(x_gate, median_abs, color="#D55E00", linewidth=1.9, label="median")
    gate_ax.axhline(0.10, color="#4D4D4D", linewidth=0.8, linestyle="--", alpha=0.72, label="0.1 dex")
    if use_log_time:
        gate_ax.set_xscale("log")
    gate_ax.set_title(r"Gate-wise correction magnitude", pad=8)
    gate_ax.set_xlabel(time_label)
    gate_ax.set_ylabel(r"$|\Delta\log_{10}|$ (dex)")
    ylim = float(np.nanpercentile(abs_delta[np.isfinite(abs_delta)], 98))
    ylim = min(max(ylim, 0.2), 1.0)
    gate_ax.set_ylim(0.0, ylim * 1.08)
    gate_ax.grid(True, which="major", color="#D6DEE6", linewidth=0.7)
    gate_ax.grid(True, which="minor", color="#EEF2F6", linewidth=0.35)
    gate_ax.legend(frameon=False, loc="best")
    panel_label(gate_ax, "(e)")

    fig.suptitle(f"{name}: measured TEM record, model output, and correction", y=0.982, fontsize=10.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=360, facecolor="white", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_story_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, filename in DATASETS:
        mat_path = PROCESSED_DIR / filename
        out_path = OUT_DIR / f"real_field_story_{name}_original_gates.png"
        plot_story(name, mat_path, out_path)
        written.append(out_path)

    inversion_cache = (
        Path.cwd()
        / "RA_ST_DiffTEM_TGRS_initial_draft_package"
        / "field_outputs"
        / "field_inversion_w21_20260521"
        / "W21_raw_vs_denoised_1d_tem_inversion_with_forward_check.npz"
    )
    print("written story figures:")
    for path in written:
        print(path)
    if inversion_cache.exists():
        print(f"inversion cache exists: {inversion_cache}")
    else:
        print(f"inversion cache not found: {inversion_cache}")
        print("Run: python 模型代码/training/run_field_tem_1d_inversion.py --dataset W21")


if __name__ == "__main__":
    main()
