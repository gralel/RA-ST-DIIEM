from __future__ import annotations

import argparse
import csv
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy.io as sio
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).absolute()
WORKSPACE = SCRIPT_PATH.parents[2]
PROCESSED_DIR = WORKSPACE / "补充" / "processed_11x869"
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "RA_ST_DiffTEM_TGRS_initial_draft_package"
    / "field_outputs"
    / "field_inversion_w21_20260521"
)

DATASET_FILES = {
    "dbz": "dbz_11x869.mat",
    "W21": "taochong_W21_selected_11x869.mat",
    "W23": "taochong_W23_selected_11x869.mat",
    "W25": "taochong_W25_selected_11x869.mat",
}


@dataclass
class TEMConfig:
    data_kind: str = "dbdt"
    receiver_orientation: str = "z"

    sign_multiplier: float = 1.0
    data_scale: float = 1.0
    use_auto_calibration: bool = True
    calibration_rho_min: float = 10.0
    calibration_rho_max: float = 5000.0
    calibration_n_rho: int = 31

    station_spacing_m: float = 50.0
    loop_width_m: float = 200.0
    loop_height_m: float = 200.0
    source_current_a: float = 10.0
    source_height_m: float = 0.0
    receiver_height_m: float = 0.0

    n_layers: int = 16
    min_layer_thickness_m: float = 5.0
    max_layer_thickness_m: float = 60.0
    starting_resistivity_ohm_m: float = 300.0
    min_resistivity_ohm_m: float = 1.0
    max_resistivity_ohm_m: float = 10000.0

    relative_error: float = 0.10
    floor_fraction_of_max: float = 0.03
    max_iterations: int = 40
    alpha_s: float = 1.0e-1
    alpha_x: float = 30.0
    beta0_ratio: float = 1.0e1
    target_chifact: float = 2.0

    dpi: int = 300
    save_each_station_fit: bool = False


def import_simpeg() -> dict[str, Any]:
    try:
        from discretize import TensorMesh
        import simpeg
        import simpeg.electromagnetics.time_domain as tdem
        from simpeg import data, data_misfit, directives, inversion, inverse_problem
        from simpeg import maps, optimization, regularization
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "SimPEG/discretize is not installed in this environment. "
            "Use the isolated tem-simpeg environment requested by the field workflow."
        ) from err

    print("SimPEG version:", getattr(simpeg, "__version__", "unknown"))
    print("Simulation class:", tdem.Simulation1DLayered)
    return {
        "TensorMesh": TensorMesh,
        "tdem": tdem,
        "maps": maps,
        "data": data,
        "data_misfit": data_misfit,
        "inverse_problem": inverse_problem,
        "regularization": regularization,
        "optimization": optimization,
        "directives": directives,
        "inversion": inversion,
    }


def station_edges_from_centers(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=np.float64)
    mid = 0.5 * (x[:-1] + x[1:])
    return np.r_[x[0] - (mid[0] - x[0]), mid, x[-1] + (x[-1] - mid[-1])]


def time_edges_from_centers(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    return np.r_[t[0] / np.sqrt(t[1] / t[0]), np.sqrt(t[:-1] * t[1:]), t[-1] * np.sqrt(t[-1] / t[-2])]


def build_layer_thicknesses(cfg: TEMConfig) -> np.ndarray:
    return np.logspace(
        np.log10(cfg.min_layer_thickness_m),
        np.log10(cfg.max_layer_thickness_m),
        cfg.n_layers - 1,
    )


def layer_edges_and_centers(layer_thicknesses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plot_thk = np.r_[layer_thicknesses, layer_thicknesses[-1]]
    z_edges = np.r_[0.0, np.cumsum(plot_thk)]
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    return z_edges, z_centers


def default_station_axis(n_stations: int, cfg: TEMConfig) -> np.ndarray:
    return (np.arange(n_stations, dtype=np.float64) - (n_stations - 1) / 2.0) * cfg.station_spacing_m


def load_field_pair(dataset: str, processed_dir: Path, cfg: TEMConfig) -> dict[str, Any]:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unknown dataset {dataset!r}. Choices: {sorted(DATASET_FILES)}")

    mat_path = processed_dir / DATASET_FILES[dataset]
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    mat = sio.loadmat(mat_path)
    if dataset == "dbz":
        raw = np.asarray(mat["raw_voltage_40x11"], dtype=np.float64)
        denoised = np.asarray(mat["model_signed_by_raw_40x11"], dtype=np.float64)
        times = np.arange(1, raw.shape[0] + 1, dtype=np.float64)
        station_x = default_station_axis(raw.shape[1], cfg)
        station_label = "Station distance x (m, nominal)"
        time_label = "Original gate index"
        has_physical_time = False
    else:
        raw = np.asarray(mat["original_voltage"], dtype=np.float64)
        if "model_signed_by_raw_original_gates" in mat:
            denoised = np.asarray(mat["model_signed_by_raw_original_gates"], dtype=np.float64)
        else:
            mag = np.asarray(mat["model_magnitude_original_gates"], dtype=np.float64)
            denoised = np.sign(raw) * np.abs(mag)
        times = np.asarray(mat["original_time_s"], dtype=np.float64).reshape(-1)
        if "selected_station_m" in mat:
            station_x = np.asarray(mat["selected_station_m"], dtype=np.float64).reshape(-1)
        else:
            station_x = default_station_axis(raw.shape[1], cfg)
        station_label = "Station distance along profile (m)"
        time_label = "Time (s)"
        has_physical_time = True

    if raw.shape != denoised.shape:
        raise ValueError(f"{dataset}: raw shape {raw.shape} != denoised shape {denoised.shape}")
    if raw.shape[0] != times.size:
        raise ValueError(f"{dataset}: time count {times.size} != data rows {raw.shape[0]}")
    if raw.shape[1] != station_x.size:
        raise ValueError(f"{dataset}: station count {station_x.size} != data columns {raw.shape[1]}")

    row_mask = (
        np.isfinite(times)
        & np.all(np.isfinite(raw), axis=1)
        & np.all(np.isfinite(denoised), axis=1)
        & np.all(np.abs(raw) > 0.0, axis=1)
        & np.all(np.abs(denoised) > 0.0, axis=1)
    )
    if has_physical_time:
        row_mask &= times > 0.0
    if int(row_mask.sum()) < 8:
        raise ValueError(f"{dataset}: only {int(row_mask.sum())} valid gates remain after filtering.")

    raw = raw[row_mask, :]
    denoised = denoised[row_mask, :]
    times = times[row_mask]

    if not np.all(np.diff(times) > 0):
        order = np.argsort(times)
        times = times[order]
        raw = raw[order, :]
        denoised = denoised[order, :]

    return {
        "dataset": dataset,
        "mat_path": mat_path,
        "raw": raw,
        "denoised": denoised,
        "times": times,
        "station_x": station_x,
        "station_label": station_label,
        "time_label": time_label,
        "has_physical_time": has_physical_time,
        "valid_gate_count": int(row_mask.sum()),
        "dropped_gate_count": int(row_mask.size - row_mask.sum()),
    }


def build_survey_for_one_sounding(simpeg_mods: dict[str, Any], times: np.ndarray, cfg: TEMConfig):
    tdem = simpeg_mods["tdem"]
    receiver_location = np.array([[0.0, 0.0, cfg.receiver_height_m]])
    source_location = np.array([0.0, 0.0, cfg.source_height_m])
    equivalent_radius = np.sqrt(cfg.loop_width_m * cfg.loop_height_m / np.pi)

    if cfg.data_kind.lower() in {"dbdt", "dbzdt", "db_dt", "db/dt"}:
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_location,
            times,
            orientation=cfg.receiver_orientation,
        )
    elif cfg.data_kind.lower() in {"b", "bz", "bfield"}:
        rx = tdem.receivers.PointMagneticFluxDensity(
            receiver_location,
            times,
            orientation=cfg.receiver_orientation,
        )
    else:
        raise ValueError("cfg.data_kind must be 'dbdt' or 'b'.")

    src = tdem.sources.CircularLoop(
        receiver_list=[rx],
        location=source_location,
        waveform=tdem.sources.StepOffWaveform(),
        current=cfg.source_current_a,
        radius=equivalent_radius,
    )
    return tdem.Survey([src])


def make_regularization(regularization, mesh, mapping, cfg: TEMConfig, m0: np.ndarray):
    try:
        reg = regularization.WeightedLeastSquares(
            mesh,
            mapping=mapping,
            alpha_s=cfg.alpha_s,
            alpha_x=cfg.alpha_x,
        )
    except Exception:
        reg = regularization.Tikhonov(
            mesh,
            mapping=mapping,
            alpha_s=cfg.alpha_s,
            alpha_x=cfg.alpha_x,
        )

    for attr in ("reference_model", "mref"):
        try:
            if hasattr(reg, attr):
                setattr(reg, attr, m0)
        except Exception:
            pass
    return reg


def make_optimizer(optimization, cfg: TEMConfig, lower: np.ndarray, upper: np.ndarray):
    last_err: Exception | None = None
    for kwargs in (
        {"maxIter": cfg.max_iterations, "maxIterLS": 20, "lower": lower, "upper": upper},
        {"maxIter": cfg.max_iterations, "lower": lower, "upper": upper},
        {"lower": lower, "upper": upper},
    ):
        try:
            opt = optimization.ProjectedGNCG(**kwargs)
            break
        except Exception as err:
            last_err = err
    else:
        raise RuntimeError("Could not initialize ProjectedGNCG.") from last_err

    for attr, value in {
        "cg_maxiter": 30,
        "cg_rtol": 1.0e-3,
        "maxIterCG": 30,
        "tolCG": 1.0e-3,
        "maxIter": cfg.max_iterations,
        "maxIterLS": 20,
    }.items():
        try:
            if hasattr(opt, attr):
                setattr(opt, attr, value)
        except Exception:
            pass
    return opt


def make_directives(directives, cfg: TEMConfig) -> list[Any]:
    directive_list: list[Any] = []
    if hasattr(directives, "UpdateSensitivityWeights"):
        try:
            directive_list.append(directives.UpdateSensitivityWeights())
        except Exception:
            pass
    try:
        directive_list.append(directives.BetaEstimate_ByEig(beta0_ratio=cfg.beta0_ratio))
    except Exception:
        pass
    try:
        directive_list.append(directives.BetaSchedule(coolingFactor=2, coolingRate=1))
    except Exception:
        pass
    try:
        directive_list.append(directives.TargetMisfit(chifact=cfg.target_chifact))
    except Exception:
        pass
    try:
        directive_list.append(directives.UpdatePreconditioner())
    except Exception:
        pass
    return directive_list


def simulate_uniform_halfspace_response(
    simpeg_mods: dict[str, Any],
    times: np.ndarray,
    layer_thicknesses: np.ndarray,
    cfg: TEMConfig,
    rho_ohm_m: float,
) -> np.ndarray:
    maps = simpeg_mods["maps"]
    tdem = simpeg_mods["tdem"]
    survey = build_survey_for_one_sounding(simpeg_mods, times, cfg)
    n_model = layer_thicknesses.size + 1
    sigma_map = maps.ExpMap(nP=n_model)
    simulation = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=layer_thicknesses,
        sigmaMap=sigma_map,
    )
    model = np.log((1.0 / rho_ohm_m) * np.ones(n_model))
    return simulation.dpred(model)


def calibrate_sign_scale_and_starting_rho(
    dbz: np.ndarray,
    times: np.ndarray,
    layer_thicknesses: np.ndarray,
    cfg: TEMConfig,
    simpeg_mods: dict[str, Any],
) -> dict[str, float]:
    obs_all = np.asarray(dbz, dtype=np.float64).reshape(-1)
    finite_obs = np.isfinite(obs_all) & (np.abs(obs_all) > 0)
    if int(finite_obs.sum()) < 5:
        raise ValueError("Too few valid observations for calibration.")

    obs = obs_all[finite_obs]
    n_stations = dbz.shape[1]
    rho_grid = np.logspace(
        np.log10(cfg.calibration_rho_min),
        np.log10(cfg.calibration_rho_max),
        cfg.calibration_n_rho,
    )
    best = {
        "score": np.inf,
        "rho": float(cfg.starting_resistivity_ohm_m),
        "scale": float(cfg.data_scale),
        "sign": float(cfg.sign_multiplier),
    }

    print("[calibration] searching uniform half-space response...")
    for rho in rho_grid:
        try:
            pred_one = simulate_uniform_halfspace_response(
                simpeg_mods,
                times,
                layer_thicknesses,
                cfg,
                float(rho),
            ).reshape(-1)
        except Exception as err:
            print(f"  skip rho={rho:.3g} Ohm m: {err}")
            continue

        pred = np.tile(pred_one, n_stations)[finite_obs]
        finite = np.isfinite(pred) & (np.abs(pred) > 0.0) & np.isfinite(obs)
        if int(finite.sum()) < 5:
            continue

        p = pred[finite]
        d = obs[finite]
        log_scale = np.median(np.log10(np.abs(p) + 1.0e-30) - np.log10(np.abs(d) + 1.0e-30))
        scale = float(10.0**log_scale)
        for sign in (1.0, -1.0):
            d_cal = sign * scale * d
            floor = 0.03 * max(float(np.nanmax(np.abs(p))), 1.0e-30)
            rel_mis = (p - d_cal) / (np.abs(p) + floor)
            log_mis = np.log10(np.abs(p) + 1.0e-30) - np.log10(np.abs(d_cal) + 1.0e-30)
            score = float(np.nanmedian(np.abs(rel_mis)) + 0.25 * np.nanmedian(np.abs(log_mis)))
            if score < best["score"]:
                best = {"score": score, "rho": float(rho), "scale": scale, "sign": float(sign)}

    cfg.starting_resistivity_ohm_m = best["rho"]
    cfg.data_scale = best["scale"]
    cfg.sign_multiplier = best["sign"]
    print(
        "[calibration] "
        f"rho={cfg.starting_resistivity_ohm_m:.6g} Ohm m, "
        f"scale={cfg.data_scale:.6e}, sign={cfg.sign_multiplier:+.0f}, "
        f"score={best['score']:.6g}"
    )
    return best


def invert_one_station(
    station_id: int,
    dobs_raw: np.ndarray,
    times: np.ndarray,
    layer_thicknesses: np.ndarray,
    cfg: TEMConfig,
    simpeg_mods: dict[str, Any],
    out_dir: Path,
    dataset_name: str,
) -> dict[str, np.ndarray | float]:
    TensorMesh = simpeg_mods["TensorMesh"]
    maps = simpeg_mods["maps"]
    tdem = simpeg_mods["tdem"]
    data = simpeg_mods["data"]
    data_misfit = simpeg_mods["data_misfit"]
    inverse_problem = simpeg_mods["inverse_problem"]
    regularization = simpeg_mods["regularization"]
    optimization = simpeg_mods["optimization"]
    directives = simpeg_mods["directives"]
    inversion = simpeg_mods["inversion"]

    dobs_raw = np.asarray(dobs_raw, dtype=np.float64).reshape(-1)
    dobs = cfg.sign_multiplier * cfg.data_scale * dobs_raw
    abs_d = np.abs(dobs)
    data_floor = cfg.floor_fraction_of_max * max(float(np.nanmax(abs_d)), 1.0e-30)
    uncertainties = cfg.relative_error * abs_d + data_floor

    survey = build_survey_for_one_sounding(simpeg_mods, times, cfg)
    data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)
    n_model = layer_thicknesses.size + 1
    model_cell_widths = np.r_[layer_thicknesses, layer_thicknesses[-1]]
    reg_mesh = TensorMesh([model_cell_widths], "0")

    m0 = np.log((1.0 / cfg.starting_resistivity_ohm_m) * np.ones(n_model))
    lower = np.log((1.0 / cfg.max_resistivity_ohm_m) * np.ones(n_model))
    upper = np.log((1.0 / cfg.min_resistivity_ohm_m) * np.ones(n_model))
    sigma_map = maps.ExpMap(nP=n_model)

    simulation = tdem.Simulation1DLayered(
        survey=survey,
        thicknesses=layer_thicknesses,
        sigmaMap=sigma_map,
    )
    dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
    try:
        dmis.W = 1.0 / uncertainties
    except Exception:
        pass

    reg = make_regularization(regularization, reg_mesh, maps.IdentityMap(nP=n_model), cfg, m0)
    opt = make_optimizer(optimization, cfg, lower, upper)
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    inv = inversion.BaseInversion(inv_prob, make_directives(directives, cfg))

    print(f"[{dataset_name}] station {station_id + 1:02d}: SimPEG 1D inversion")
    recovered_model = inv.run(m0)
    sigma_rec = np.exp(recovered_model)
    rho_rec = 1.0 / sigma_rec
    forward_response = simulation.dpred(recovered_model)
    weighted_residual = (forward_response - dobs) / uncertainties
    phi_d = float(np.sum(weighted_residual**2))
    nrms = float(np.sqrt(np.mean(weighted_residual**2)))

    if cfg.save_each_station_fit:
        save_forward_check_grid(
            observed_used=dobs[:, None],
            forward_response=forward_response[:, None],
            times=times,
            station_x=np.array([float(station_id + 1)]),
            title=f"{dataset_name} station {station_id + 1}: forward check",
            label=dataset_name,
            out_path=out_dir / f"{dataset_name}_station_{station_id + 1:02d}_forward_check.png",
            cfg=cfg,
        )

    return {
        "rho": rho_rec,
        "sigma": sigma_rec,
        "model_log_sigma": recovered_model,
        "forward_response": forward_response,
        "observed_used": dobs,
        "observed_raw": dobs_raw,
        "uncertainties": uncertainties,
        "phi_d": phi_d,
        "nrms": nrms,
    }


def invert_dataset_1d_stitched(
    dbz_data: np.ndarray,
    dataset_name: str,
    times: np.ndarray,
    layer_thicknesses: np.ndarray,
    cfg: TEMConfig,
    simpeg_mods: dict[str, Any],
    out_dir: Path,
) -> dict[str, np.ndarray]:
    n_times, n_stations = dbz_data.shape
    n_model = layer_thicknesses.size + 1
    rho_section = np.full((n_model, n_stations), np.nan, dtype=np.float64)
    sigma_section = np.full_like(rho_section, np.nan)
    forward_response_all = np.full((n_times, n_stations), np.nan, dtype=np.float64)
    observed_used_all = np.full_like(forward_response_all, np.nan)
    uncertainties_all = np.full_like(forward_response_all, np.nan)
    phi_d = np.full(n_stations, np.nan, dtype=np.float64)
    nrms = np.full(n_stations, np.nan, dtype=np.float64)

    print(f"[{dataset_name}] shape={dbz_data.shape}")
    for station_id in range(n_stations):
        result = invert_one_station(
            station_id,
            dbz_data[:, station_id],
            times,
            layer_thicknesses,
            cfg,
            simpeg_mods,
            out_dir,
            dataset_name,
        )
        rho_section[:, station_id] = np.asarray(result["rho"], dtype=np.float64)
        sigma_section[:, station_id] = np.asarray(result["sigma"], dtype=np.float64)
        forward_response_all[:, station_id] = np.asarray(result["forward_response"], dtype=np.float64)
        observed_used_all[:, station_id] = np.asarray(result["observed_used"], dtype=np.float64)
        uncertainties_all[:, station_id] = np.asarray(result["uncertainties"], dtype=np.float64)
        phi_d[station_id] = float(result["phi_d"])
        nrms[station_id] = float(result["nrms"])

    print(f"[{dataset_name}] mean NRMS={float(np.nanmean(nrms)):.4g}")
    return {
        "rho_section": rho_section,
        "sigma_section": sigma_section,
        "forward_response_all": forward_response_all,
        "observed_used_all": observed_used_all,
        "uncertainties_all": uncertainties_all,
        "phi_d": phi_d,
        "nrms": nrms,
    }


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


def save_forward_check_grid(
    observed_used: np.ndarray,
    forward_response: np.ndarray,
    times: np.ndarray,
    station_x: np.ndarray,
    title: str,
    label: str,
    out_path: Path,
    cfg: TEMConfig,
) -> None:
    n_stations = observed_used.shape[1]
    n_cols = min(4, n_stations)
    n_rows = int(np.ceil(n_stations / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.15 * n_cols, 2.55 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for idx in range(n_stations):
        ax = axes[idx]
        ax.loglog(times, np.abs(observed_used[:, idx]), "o-", linewidth=1.0, markersize=2.4, label=f"{label} data")
        ax.loglog(times, np.abs(forward_response[:, idx]), "s--", linewidth=1.0, markersize=2.2, label="1D forward")
        ax.set_title(f"Station {idx + 1}, x={station_x[idx]:.0f} m")
        ax.grid(True, which="major", color="#D6DEE6", linewidth=0.55)
        ax.grid(True, which="minor", color="#EEF2F6", linewidth=0.3)
        if idx % n_cols == 0:
            ax.set_ylabel("Calibrated amplitude")
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time (s)")
    for idx in range(n_stations, axes.size):
        axes[idx].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=1.02, fontsize=10.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_residual_pseudosection(
    observed_used: np.ndarray,
    forward_response: np.ndarray,
    times: np.ndarray,
    station_x: np.ndarray,
    out_path: Path,
    cfg: TEMConfig,
) -> np.ndarray:
    floor = cfg.floor_fraction_of_max * max(float(np.nanmax(np.abs(observed_used))), 1.0e-30)
    residual = (forward_response - observed_used) / (np.abs(observed_used) + floor)
    finite = np.isfinite(residual)
    lim = float(np.nanpercentile(np.abs(residual[finite]), 98)) if np.any(finite) else 1.0
    lim = min(max(lim, 0.5), 5.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.35), constrained_layout=True)
    pm = ax.pcolormesh(
        station_edges_from_centers(station_x),
        time_edges_from_centers(times),
        np.clip(residual, -lim, lim),
        shading="auto",
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
    )
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Station distance along profile (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Denoised-data forward residual of stitched 1D inversions")
    cbar = fig.colorbar(pm, ax=ax)
    cbar.set_label("Normalized residual")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return residual


def smooth_log_rho_section(
    rho_section: np.ndarray,
    station_x: np.ndarray,
    z_centers: np.ndarray,
    nx_smooth: int = 300,
    nz_smooth: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_safe = np.where(rho_section > 0.0, rho_section, np.nan)
    log_rho = np.log10(rho_safe)
    if np.any(~np.isfinite(log_rho)):
        median_val = float(np.nanmedian(log_rho))
        log_rho = np.where(np.isfinite(log_rho), log_rho, median_val)

    x_new = np.linspace(float(np.min(station_x)), float(np.max(station_x)), nx_smooth)
    z_new = np.linspace(float(np.min(z_centers)), float(np.max(z_centers)), nz_smooth)
    zz, xx = np.meshgrid(z_new, x_new, indexing="ij")
    interp_func = RegularGridInterpolator(
        (z_centers, station_x),
        log_rho,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    points = np.column_stack([zz.ravel(), xx.ravel()])
    log_interp = interp_func(points).reshape(nz_smooth, nx_smooth)
    log_smooth = gaussian_filter(log_interp, sigma=(1.0, 2.0))
    return 10.0**log_smooth, x_new, z_new


def save_inversion_comparison(
    rho_raw: np.ndarray,
    rho_denoised: np.ndarray,
    station_x: np.ndarray,
    z_edges: np.ndarray,
    out_path: Path,
    cfg: TEMConfig,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    rho_raw_smooth, x_new, z_new = smooth_log_rho_section(rho_raw, station_x, z_centers)
    rho_denoised_smooth, _, _ = smooth_log_rho_section(rho_denoised, station_x, z_centers)
    log_ratio = np.log10(rho_denoised_smooth / rho_raw_smooth)

    rho_all = np.r_[rho_raw.ravel(), rho_denoised.ravel()]
    finite = np.isfinite(rho_all) & (rho_all > 0.0)
    vmin, vmax = np.nanpercentile(rho_all[finite], [2, 98])
    vmin = max(float(vmin), 1.0e-3)
    vmax = max(float(vmax), vmin * 1.01)
    diff_lim = float(np.nanpercentile(np.abs(log_ratio[np.isfinite(log_ratio)]), 98))
    diff_lim = min(max(diff_lim, 0.05), 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.95), constrained_layout=True)
    panels = [
        ("Raw inversion", rho_raw_smooth),
        ("RA-ST-DiffTEM inversion", rho_denoised_smooth),
    ]
    for ax, (title, arr) in zip(axes[:2], panels):
        pm = ax.pcolormesh(
            x_new,
            z_new,
            arr,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("Station distance along profile (m)")
        ax.invert_yaxis()
        for x in station_x:
            ax.axvline(x, color="white", linestyle=":", linewidth=0.45, alpha=0.55)
        ax.scatter(station_x, np.zeros_like(station_x), marker="v", s=20, color="black", zorder=3)
    axes[0].set_ylabel("Depth (m)")
    cbar = fig.colorbar(pm, ax=axes[:2], shrink=0.86, pad=0.01)
    cbar.set_label(r"Resistivity ($\Omega$m)")

    pm2 = axes[2].pcolormesh(
        x_new,
        z_new,
        np.clip(log_ratio, -diff_lim, diff_lim),
        shading="auto",
        cmap="coolwarm",
        vmin=-diff_lim,
        vmax=diff_lim,
    )
    axes[2].set_title(r"$\log_{10}(\rho_{denoised}/\rho_{raw})$")
    axes[2].set_xlabel("Station distance along profile (m)")
    axes[2].invert_yaxis()
    for x in station_x:
        axes[2].axvline(x, color="black", linestyle=":", linewidth=0.45, alpha=0.38)
    axes[2].scatter(station_x, np.zeros_like(station_x), marker="v", s=20, color="black", zorder=3)
    cbar2 = fig.colorbar(pm2, ax=axes[2], shrink=0.86, pad=0.01)
    cbar2.set_label("dex")
    fig.suptitle(f"{dataset_name} stitched 1D TEM inversion sections", fontsize=10.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return rho_raw_smooth, rho_denoised_smooth, log_ratio, x_new, z_new


def save_inversion_consistency_figure(
    rho_raw_smooth: np.ndarray,
    rho_denoised_smooth: np.ndarray,
    residual_denoised: np.ndarray,
    x_smooth: np.ndarray,
    z_smooth: np.ndarray,
    times: np.ndarray,
    station_x: np.ndarray,
    dataset_name: str,
    out_path: Path,
    cfg: TEMConfig,
) -> None:
    rho_all = np.r_[rho_raw_smooth.ravel(), rho_denoised_smooth.ravel()]
    finite = np.isfinite(rho_all) & (rho_all > 0.0)
    vmin, vmax = np.nanpercentile(rho_all[finite], [2, 98])
    vmin = max(float(vmin), 1.0e-3)
    vmax = max(float(vmax), vmin * 1.01)

    finite_res = np.isfinite(residual_denoised)
    res_lim = float(np.nanpercentile(np.abs(residual_denoised[finite_res]), 98)) if np.any(finite_res) else 1.0
    res_lim = min(max(res_lim, 0.5), 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.05), constrained_layout=True)
    for ax, title, arr in [
        (axes[0], "Raw stitched 1D inversion", rho_raw_smooth),
        (axes[1], "RA-ST-DiffTEM stitched 1D inversion", rho_denoised_smooth),
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
        ax.invert_yaxis()
        for x in station_x:
            ax.axvline(x, color="white", linestyle=":", linewidth=0.45, alpha=0.55)
        ax.scatter(station_x, np.zeros_like(station_x), marker="v", s=20, color="black", zorder=3)
    axes[0].set_ylabel("Depth (m)")
    cbar = fig.colorbar(pm, ax=axes[:2], shrink=0.86, pad=0.012)
    cbar.set_label(r"Resistivity ($\Omega$m)")

    pm_res = axes[2].pcolormesh(
        station_edges_from_centers(station_x),
        time_edges_from_centers(times),
        np.clip(residual_denoised, -res_lim, res_lim),
        shading="auto",
        cmap="coolwarm",
        vmin=-res_lim,
        vmax=res_lim,
    )
    axes[2].set_title("Denoised forward residual")
    axes[2].set_xlabel("Station distance along profile (m)")
    axes[2].set_ylabel("Time (s)")
    axes[2].set_yscale("log")
    axes[2].invert_yaxis()
    for x in station_x:
        axes[2].axvline(x, color="black", linestyle=":", linewidth=0.45, alpha=0.38)
    cbar_res = fig.colorbar(pm_res, ax=axes[2], shrink=0.86, pad=0.012)
    cbar_res.set_label("Normalized residual")

    fig.suptitle(f"{dataset_name} field inversion consistency", fontsize=10.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def section_lateral_roughness(rho_section: np.ndarray) -> float:
    log_rho = np.log10(np.where(rho_section > 0.0, rho_section, np.nan))
    if log_rho.shape[1] < 3:
        return float("nan")
    diff2 = np.diff(log_rho, n=2, axis=1)
    return float(np.nanmedian(np.abs(diff2)))


def write_summary_csv(
    out_path: Path,
    raw_inv: dict[str, np.ndarray],
    denoised_inv: dict[str, np.ndarray],
    residual_denoised: np.ndarray,
    calibration: dict[str, float],
    dataset_info: dict[str, Any],
) -> None:
    rows = [
        ("dataset", dataset_info["dataset"]),
        ("valid_gates", dataset_info["valid_gate_count"]),
        ("dropped_gates", dataset_info["dropped_gate_count"]),
        ("mean_nrms_raw", float(np.nanmean(raw_inv["nrms"]))),
        ("mean_nrms_denoised", float(np.nanmean(denoised_inv["nrms"]))),
        ("median_nrms_raw", float(np.nanmedian(raw_inv["nrms"]))),
        ("median_nrms_denoised", float(np.nanmedian(denoised_inv["nrms"]))),
        ("lateral_roughness_raw", section_lateral_roughness(raw_inv["rho_section"])),
        ("lateral_roughness_denoised", section_lateral_roughness(denoised_inv["rho_section"])),
        ("denoised_residual_median_abs", float(np.nanmedian(np.abs(residual_denoised)))),
        ("denoised_residual_p95_abs", float(np.nanpercentile(np.abs(residual_denoised), 95))),
        ("calibration_sign", calibration["sign"]),
        ("calibration_scale", calibration["scale"]),
        ("calibration_starting_rho_ohm_m", calibration["rho"]),
        ("calibration_score", calibration["score"]),
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    set_plot_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TEMConfig(max_iterations=args.max_iterations)
    dataset_info = load_field_pair(args.dataset, Path(args.processed_dir), cfg)
    if not dataset_info["has_physical_time"]:
        raise ValueError(
            "The selected dataset has no physical time axis. Use W21/W23/W25 for inversion; "
            "dbz remains a model-output diagnostic unless real gate times are provided."
        )

    raw = dataset_info["raw"]
    denoised = dataset_info["denoised"]
    times = dataset_info["times"]
    station_x = dataset_info["station_x"]
    print(
        f"Loaded {args.dataset}: raw={raw.shape}, denoised={denoised.shape}, "
        f"valid gates={dataset_info['valid_gate_count']}, dropped={dataset_info['dropped_gate_count']}"
    )

    layer_thicknesses = build_layer_thicknesses(cfg)
    z_edges, z_centers = layer_edges_and_centers(layer_thicknesses)
    simpeg_mods = import_simpeg()

    cfg_compare = copy.deepcopy(cfg)
    calibration = calibrate_sign_scale_and_starting_rho(
        raw,
        times,
        layer_thicknesses,
        cfg_compare,
        simpeg_mods,
    )

    raw_inv = invert_dataset_1d_stitched(
        raw,
        "raw",
        times,
        layer_thicknesses,
        cfg_compare,
        simpeg_mods,
        out_dir,
    )
    denoised_inv = invert_dataset_1d_stitched(
        denoised,
        "denoised",
        times,
        layer_thicknesses,
        cfg_compare,
        simpeg_mods,
        out_dir,
    )

    save_forward_check_grid(
        raw_inv["observed_used_all"],
        raw_inv["forward_response_all"],
        times,
        station_x,
        f"{args.dataset} raw data vs forward response of stitched 1D inversions",
        "Raw",
        out_dir / f"{args.dataset}_raw_all_stations_forward_check.png",
        cfg_compare,
    )
    save_forward_check_grid(
        denoised_inv["observed_used_all"],
        denoised_inv["forward_response_all"],
        times,
        station_x,
        f"{args.dataset} RA-ST-DiffTEM output vs forward response of stitched 1D inversions",
        "Denoised",
        out_dir / f"{args.dataset}_denoised_all_stations_forward_check.png",
        cfg_compare,
    )
    residual_denoised = save_residual_pseudosection(
        denoised_inv["observed_used_all"],
        denoised_inv["forward_response_all"],
        times,
        station_x,
        out_dir / f"{args.dataset}_denoised_forward_residual_pseudosection.png",
        cfg_compare,
    )
    rho_raw_smooth, rho_denoised_smooth, log_ratio, x_new, z_new = save_inversion_comparison(
        raw_inv["rho_section"],
        denoised_inv["rho_section"],
        station_x,
        z_edges,
        out_dir / f"{args.dataset}_raw_vs_denoised_inversion_sections.png",
        cfg_compare,
        args.dataset,
    )
    save_inversion_consistency_figure(
        rho_raw_smooth,
        rho_denoised_smooth,
        residual_denoised,
        x_new,
        z_new,
        times,
        station_x,
        args.dataset,
        out_dir / f"{args.dataset}_field_inversion_consistency.png",
        cfg_compare,
    )

    np.savez(
        out_dir / f"{args.dataset}_raw_vs_denoised_1d_tem_inversion_with_forward_check.npz",
        raw=raw,
        denoised=denoised,
        times=times,
        station_x=station_x,
        layer_thicknesses=layer_thicknesses,
        z_edges=z_edges,
        z_centers=z_centers,
        rho_raw=raw_inv["rho_section"],
        rho_denoised=denoised_inv["rho_section"],
        rho_raw_smooth=rho_raw_smooth,
        rho_denoised_smooth=rho_denoised_smooth,
        log_ratio=log_ratio,
        x_smooth=x_new,
        z_smooth=z_new,
        forward_raw=raw_inv["forward_response_all"],
        forward_denoised=denoised_inv["forward_response_all"],
        observed_used_raw=raw_inv["observed_used_all"],
        observed_used_denoised=denoised_inv["observed_used_all"],
        uncertainties_raw=raw_inv["uncertainties_all"],
        uncertainties_denoised=denoised_inv["uncertainties_all"],
        nrms_raw=raw_inv["nrms"],
        nrms_denoised=denoised_inv["nrms"],
        phi_d_raw=raw_inv["phi_d"],
        phi_d_denoised=denoised_inv["phi_d"],
        residual_denoised=residual_denoised,
        data_scale=cfg_compare.data_scale,
        sign_multiplier=cfg_compare.sign_multiplier,
        starting_resistivity_ohm_m=cfg_compare.starting_resistivity_ohm_m,
        source_mat_path=str(dataset_info["mat_path"]),
    )
    write_summary_csv(
        out_dir / f"{args.dataset}_inversion_summary.csv",
        raw_inv,
        denoised_inv,
        residual_denoised,
        calibration,
        dataset_info,
    )

    print("Saved:")
    for path in [
        out_dir / f"{args.dataset}_raw_vs_denoised_1d_tem_inversion_with_forward_check.npz",
        out_dir / f"{args.dataset}_raw_vs_denoised_inversion_sections.png",
        out_dir / f"{args.dataset}_denoised_forward_residual_pseudosection.png",
        out_dir / f"{args.dataset}_inversion_summary.csv",
    ]:
        print(" ", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stitched 1D SimPEG inversion for preprocessed field TEM data."
    )
    parser.add_argument("--dataset", default="W21", choices=sorted(DATASET_FILES))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-iterations", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
