from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_field_tem_1d_inversion import time_edges_from_centers, station_edges_from_centers


SCRIPT_PATH = Path(__file__).absolute()
WORKSPACE = SCRIPT_PATH.parents[2]
RESULT_DIR = (
    WORKSPACE
    / "RA_ST_DiffTEM_TGRS_initial_draft_package"
    / "field_outputs"
    / "field_inversion_w21_20260521"
)
NPZ_PATH = RESULT_DIR / "W21_raw_vs_denoised_1d_tem_inversion_with_forward_check.npz"
OUT_PATH = RESULT_DIR / "W21_field_inversion_consistency.png"
PAPER_FIG_DIR = WORKSPACE / "RA_ST_DiffTEM_TGRS_initial_draft_package" / "figures" / "nature_final"


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
        }
    )


def normalized_residual(forward: np.ndarray, observed: np.ndarray, floor_fraction: float = 0.03) -> np.ndarray:
    floor = floor_fraction * max(float(np.nanmax(np.abs(observed))), 1.0e-30)
    return (forward - observed) / (np.abs(observed) + floor)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.3},
    )


def main() -> None:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)

    set_plot_style()
    data = np.load(NPZ_PATH)
    station_x = np.asarray(data["station_x"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    x_smooth = np.asarray(data["x_smooth"], dtype=np.float64)
    z_smooth = np.asarray(data["z_smooth"], dtype=np.float64)
    rho_raw = np.asarray(data["rho_raw_smooth"], dtype=np.float64)
    rho_denoised = np.asarray(data["rho_denoised_smooth"], dtype=np.float64)

    residual_raw = normalized_residual(data["forward_raw"], data["observed_used_raw"])
    residual_denoised = normalized_residual(data["forward_denoised"], data["observed_used_denoised"])

    rho_all = np.r_[rho_raw.ravel(), rho_denoised.ravel()]
    finite_rho = np.isfinite(rho_all) & (rho_all > 0.0)
    vmin, vmax = np.nanpercentile(rho_all[finite_rho], [2, 98])
    vmin = max(float(vmin), 1.0e-3)
    vmax = max(float(vmax), vmin * 1.01)

    res_all = np.r_[residual_raw.ravel(), residual_denoised.ravel()]
    finite_res = np.isfinite(res_all)
    res_lim = float(np.nanpercentile(np.abs(res_all[finite_res]), 98))
    res_lim = min(max(res_lim, 0.5), 3.0)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), constrained_layout=True)
    section_axes = axes[0]
    residual_axes = axes[1]

    for ax, title, arr, label in [
        (section_axes[0], "Raw stitched 1D inversion", rho_raw, "(a)"),
        (section_axes[1], "RA-ST-DiffTEM stitched 1D inversion", rho_denoised, "(b)"),
    ]:
        pm = ax.pcolormesh(
            x_smooth,
            z_smooth,
            arr,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("Station distance along profile (m)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        for x in station_x:
            ax.axvline(x, color="white", linestyle=":", linewidth=0.45, alpha=0.55)
        ax.scatter(station_x, np.zeros_like(station_x), marker="v", s=20, color="black", zorder=3)
        panel_label(ax, label)
    cbar = fig.colorbar(pm, ax=section_axes, shrink=0.82, pad=0.012)
    cbar.set_label(r"Resistivity ($\Omega$m)")

    t_edges = time_edges_from_centers(times)
    x_edges = station_edges_from_centers(station_x)
    for ax, title, arr, label in [
        (residual_axes[0], "Raw forward residual", residual_raw, "(c)"),
        (residual_axes[1], "RA-ST-DiffTEM forward residual", residual_denoised, "(d)"),
    ]:
        pm_res = ax.pcolormesh(
            x_edges,
            t_edges,
            np.clip(arr, -res_lim, res_lim),
            shading="auto",
            cmap="coolwarm",
            vmin=-res_lim,
            vmax=res_lim,
        )
        ax.set_title(title)
        ax.set_xlabel("Station distance along profile (m)")
        ax.set_ylabel("Time (s)")
        ax.set_yscale("log")
        ax.invert_yaxis()
        for x in station_x:
            ax.axvline(x, color="black", linestyle=":", linewidth=0.45, alpha=0.38)
        panel_label(ax, label)
    cbar_res = fig.colorbar(pm_res, ax=residual_axes, shrink=0.82, pad=0.012)
    cbar_res.set_label("Normalized residual")

    mean_raw = float(np.nanmean(data["nrms_raw"]))
    mean_denoised = float(np.nanmean(data["nrms_denoised"]))
    fig.suptitle(
        f"W21 stitched 1D inversion; mean NRMS raw {mean_raw:.2f}, RA-ST-DiffTEM {mean_denoised:.2f}",
        fontsize=10.5,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=360, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_FIG_DIR / "fig10_field_inversion_consistency.png").write_bytes(OUT_PATH.read_bytes())
    (PAPER_FIG_DIR / "fig10_field_inversion_consistency.pdf").write_bytes(OUT_PATH.with_suffix(".pdf").read_bytes())
    print(OUT_PATH)
    print(OUT_PATH.with_suffix(".pdf"))
    print(PAPER_FIG_DIR / "fig10_field_inversion_consistency.pdf")


if __name__ == "__main__":
    main()
