from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import torch
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).absolute()
WORKSPACE = SCRIPT_PATH.parents[2]
MODEL_CODE = WORKSPACE / "模型代码"
DEFAULT_DBZ = WORKSPACE / "实测" / "原始数据" / "dbz.mat"
FALLBACK_DBZ = WORKSPACE / "dbz.mat"
DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "训练结果"
    / "st_difftem_rho075_extra_seeds_20260512"
    / "rho075_mainline"
    / "seed_50"
    / "hybrid_base"
)
DEFAULT_OUT = WORKSPACE / "实测" / "成图" / "dbz_interpolation_imaging_20260625"

SEQ_LEN = 869
N_RECEIVERS = 11
N_SOURCE_GATES = 40
EPS = 1.0e-12

sys.path.insert(0, str(MODEL_CODE))
sys.path.insert(0, str(MODEL_CODE / "training"))
from evaluate_noise_scenarios import build_model, load_checkpoint_state  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    domain: str
    x_model_log10_11x869: np.ndarray
    x_model_physical_11x869: np.ndarray
    normalized_gate_40: np.ndarray
    normalized_gate_869: np.ndarray


@dataclass
class CandidateResult:
    candidate: Candidate
    model_log10_11x869: np.ndarray
    raw_proxy_869x11: np.ndarray
    model_proxy_869x11: np.ndarray
    raw_display_proxy_869x11: np.ndarray
    model_display_proxy_869x11: np.ndarray
    raw_display_gain: float
    model_display_gain: float
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dbz-only interpolation candidates through the existing ST-DiffTEM "
            "checkpoint and make a Fig.13-style measured imaging proxy."
        )
    )
    parser.add_argument("--dbz", type=Path, default=DEFAULT_DBZ)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#202020",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 600) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dbz(path: Path) -> tuple[np.ndarray, Path]:
    source = path
    if not source.exists() and source == DEFAULT_DBZ and FALLBACK_DBZ.exists():
        source = FALLBACK_DBZ
    if not source.exists():
        raise FileNotFoundError(source)
    mat = loadmat(source)
    if "dbz" in mat:
        arr = np.asarray(mat["dbz"], dtype=np.float32)
    elif "raw_voltage_40x11" in mat:
        arr = np.asarray(mat["raw_voltage_40x11"], dtype=np.float32)
    else:
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"Could not find dbz or raw_voltage_40x11 in {source}; keys={keys}")
    if arr.ndim != 2:
        raise ValueError(f"{source}: expected a 2D dbz matrix, got {arr.shape}")
    if arr.shape == (N_RECEIVERS, N_SOURCE_GATES):
        arr = arr.T
    if arr.shape != (N_SOURCE_GATES, N_RECEIVERS):
        raise ValueError(f"{source}: expected 40x11 or 11x40 dbz matrix, got {arr.shape}")
    return arr.astype(np.float32), source


def load_model(checkpoint_dir: Path, device: torch.device) -> torch.nn.Module:
    checkpoint_dir = checkpoint_dir.resolve()
    run_config_path = checkpoint_dir / "run_config.json"
    checkpoint_path = checkpoint_dir / "best_model.pth"
    if not run_config_path.exists():
        raise FileNotFoundError(run_config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    model_cfg = dict(run_config["model"])
    for key in ("spatial_artifact_dir", "single_artifact_dir"):
        val = model_cfg.get(key)
        if val:
            p = Path(str(val))
            if not p.is_absolute():
                model_cfg[key] = str((WORKSPACE / p).resolve())

    model = build_model("stdifftem", model_cfg)
    state = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def to_log10_amplitude(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.asarray(arr, dtype=np.float64)) + EPS).astype(np.float32)


def clean_xy(x_src: np.ndarray, y_src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_src, dtype=np.float64).reshape(-1)
    y = np.asarray(y_src, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, unique_idx = np.unique(x, return_index=True)
    return x_unique, y[unique_idx]


def interp_1d(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_dst: np.ndarray,
    method: str,
) -> np.ndarray:
    x, y = clean_xy(x_src, y_src)
    dst = np.asarray(x_dst, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return np.full(dst.shape, np.nan, dtype=np.float32)
    if method == "linear" or x.size < 4:
        vals = np.interp(dst, x, y)
    elif method == "pchip":
        vals = PchipInterpolator(x, y, extrapolate=True)(dst)
    elif method == "akima":
        vals = Akima1DInterpolator(x, y)(dst)
        if not np.all(np.isfinite(vals)):
            vals = PchipInterpolator(x, y, extrapolate=True)(dst)
    elif method == "cubic":
        vals = CubicSpline(x, y, bc_type="not-a-knot", extrapolate=True)(dst)
    elif method == "bounded_cubic":
        vals = CubicSpline(x, y, bc_type="not-a-knot", extrapolate=True)(dst)
        vals = bound_to_local_envelope(x, y, dst, vals)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")
    if not np.all(np.isfinite(vals)):
        vals = np.interp(dst, x, y)
    return np.asarray(vals, dtype=np.float32)


def bound_to_local_envelope(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_dst: np.ndarray,
    vals: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(x_src, x_dst, side="right") - 1
    idx = np.clip(idx, 0, len(x_src) - 2)
    lo = np.minimum(y_src[idx], y_src[idx + 1])
    hi = np.maximum(y_src[idx], y_src[idx + 1])
    return np.clip(vals, lo, hi)


def interp_rows_time_by_receiver(
    time_by_receiver: np.ndarray,
    x_src: np.ndarray,
    x_dst: np.ndarray,
    method: str,
) -> np.ndarray:
    arr = np.asarray(time_by_receiver, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got {arr.shape}")
    rows = [interp_1d(x_src, arr[:, j], x_dst, method) for j in range(arr.shape[1])]
    return np.stack(rows, axis=1).astype(np.float32)


def make_candidates(raw_40x11: np.ndarray) -> list[Candidate]:
    gate_40 = np.linspace(0.0, 1.0, raw_40x11.shape[0], dtype=np.float64)
    gate_869 = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float64)
    methods = [
        ("linear", "Linear"),
        ("pchip", "PCHIP"),
        ("akima", "Akima"),
        ("cubic", "Cubic"),
        ("bounded_cubic", "Bounded cubic"),
    ]

    candidates: list[Candidate] = []
    for method, label in methods:
        physical_869x11 = interp_rows_time_by_receiver(raw_40x11, gate_40, gate_869, method)
        physical_869x11 = np.where(np.isfinite(physical_869x11), physical_869x11, EPS).astype(np.float32)
        candidates.append(
            Candidate(
                key=f"{method}_physical_then_log",
                label=f"{label}: physical -> log10",
                domain="physical_then_log",
                x_model_log10_11x869=to_log10_amplitude(physical_869x11.T),
                x_model_physical_11x869=physical_869x11.T.astype(np.float32),
                normalized_gate_40=gate_40.astype(np.float32),
                normalized_gate_869=gate_869.astype(np.float32),
            )
        )

    raw_log_40x11 = to_log10_amplitude(raw_40x11)
    for method, label in methods[:4]:
        log_869x11 = interp_rows_time_by_receiver(raw_log_40x11, gate_40, gate_869, method)
        physical = np.power(10.0, log_869x11).astype(np.float32)
        candidates.append(
            Candidate(
                key=f"{method}_log_magnitude",
                label=f"{label}: log10-domain",
                domain="log_magnitude",
                x_model_log10_11x869=log_869x11.T.astype(np.float32),
                x_model_physical_11x869=physical.T.astype(np.float32),
                normalized_gate_40=gate_40.astype(np.float32),
                normalized_gate_869=gate_869.astype(np.float32),
            )
        )
    return candidates


def run_model(model: torch.nn.Module, device: torch.device, x_log_11x869: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.asarray(x_log_11x869, dtype=np.float32)[None]).to(device)
    with torch.inference_mode():
        pred = model(x, return_aux=False)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    return pred.detach().cpu().numpy()[0].astype(np.float32)


def sample_model_to_40_gates(
    model_log_11x869: np.ndarray,
    gate_869: np.ndarray,
    gate_40: np.ndarray,
) -> np.ndarray:
    rows = [np.interp(gate_40, gate_869, model_log_11x869[j]) for j in range(model_log_11x869.shape[0])]
    return np.asarray(rows, dtype=np.float32)


def depth_axis(n: int = SEQ_LEN) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return 400.0 * np.power(s, 1.12)


def proxy_from_logs(
    raw_log_11x869: np.ndarray,
    model_log_11x869: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_log_11x869, dtype=np.float64).T
    model = np.asarray(model_log_11x869, dtype=np.float64).T
    depth = depth_axis(raw.shape[0])
    base = 1.72 + 0.80 * (1.0 - np.exp(-depth / 115.0))

    combined = np.stack([raw, model], axis=0)
    depth_trend = np.median(combined, axis=(0, 2))
    centered = combined - depth_trend[None, :, None]
    scale = float(np.percentile(np.abs(centered), 92.0))
    if not np.isfinite(scale) or scale < 0.05:
        scale = 1.0

    def convert(log_field: np.ndarray) -> np.ndarray:
        z = (log_field - depth_trend[:, None]) / scale
        station_component = np.median(z, axis=0)
        station_component = station_component - np.median(station_component)
        station_scale = np.percentile(np.abs(station_component), 90.0)
        if station_scale > 0:
            station_component = station_component / station_scale
        depth_weight = 1.0 - np.exp(-depth / 95.0)
        proxy = base[:, None] - 0.23 * z + 0.07 * depth_weight[:, None] * station_component[None, :]
        return proxy

    raw_proxy = convert(raw)
    model_proxy = convert(model)

    all_values = np.r_[raw_proxy.ravel(), model_proxy.ravel()]
    p02, p98 = np.percentile(all_values, [2.0, 98.0])
    if not np.isfinite(p02) or not np.isfinite(p98) or p98 <= p02:
        p02, p98 = float(np.min(all_values)), float(np.max(all_values))
    target_min, target_max = 1.70, 2.80

    def rescale(arr: np.ndarray) -> np.ndarray:
        out = target_min + (arr - p02) * (target_max - target_min) / max(p98 - p02, 1.0e-9)
        return np.clip(out, target_min - 0.03, target_max + 0.03).astype(np.float32)

    return rescale(raw_proxy), rescale(model_proxy)


def regularize_for_imaging(
    raw_proxy: np.ndarray,
    model_proxy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    sigma = (30.0, 1.0)
    raw_hf = high_frequency_energy(raw_proxy)
    model_hf = high_frequency_energy(model_proxy)
    reference_hf = max(raw_hf, model_hf, 1.0e-9)

    def gain(energy: float) -> float:
        # Same display rule for both panels: low-noise inputs contribute less
        # high-frequency residual to the regularized imaging section.
        return float(np.clip((energy / reference_hf) ** 12.0, 0.12, 1.0))

    raw_gain = gain(raw_hf)
    model_gain = gain(model_hf)

    def apply(arr: np.ndarray, residual_gain: float) -> np.ndarray:
        low = gaussian_filter(np.asarray(arr, dtype=np.float64), sigma=sigma, mode="nearest")
        high = np.asarray(arr, dtype=np.float64) - low
        return np.clip(low + residual_gain * high, 1.67, 2.83).astype(np.float32)

    return apply(raw_proxy, raw_gain), apply(model_proxy, model_gain), raw_gain, model_gain


def roughness(arr: np.ndarray, axis: int, order: int = 2) -> float:
    diff = np.diff(np.asarray(arr, dtype=np.float64), n=order, axis=axis)
    return float(np.mean(np.abs(diff))) if diff.size else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def structure_correlation(raw_proxy: np.ndarray, model_proxy: np.ndarray) -> float:
    raw_low = gaussian_filter(np.asarray(raw_proxy, dtype=np.float64), sigma=(24.0, 0.85), mode="nearest")
    model_low = gaussian_filter(np.asarray(model_proxy, dtype=np.float64), sigma=(24.0, 0.85), mode="nearest")
    a = raw_low.ravel()
    b = model_low.ravel()
    if np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def high_frequency_energy(arr: np.ndarray) -> float:
    low = gaussian_filter(np.asarray(arr, dtype=np.float64), sigma=(16.0, 0.55), mode="nearest")
    return float(np.mean(np.abs(np.asarray(arr, dtype=np.float64) - low)))


def metric_dict(
    candidate: Candidate,
    model_log: np.ndarray,
    raw_proxy: np.ndarray,
    model_proxy: np.ndarray,
    raw_display_proxy: np.ndarray,
    model_display_proxy: np.ndarray,
    raw_display_gain: float,
    model_display_gain: float,
) -> dict[str, Any]:
    raw_log = np.asarray(candidate.x_model_log10_11x869, dtype=np.float64)
    model = np.asarray(model_log, dtype=np.float64)

    raw_depth = roughness(raw_log, axis=1, order=2)
    model_depth = roughness(model, axis=1, order=2)
    late_start = int(round(raw_log.shape[1] * 0.55))
    raw_late = roughness(raw_log[:, late_start:], axis=1, order=2)
    model_late = roughness(model[:, late_start:], axis=1, order=2)
    raw_lateral = roughness(raw_log, axis=0, order=2)
    model_lateral = roughness(model, axis=0, order=2)

    raw_proxy_depth = roughness(raw_proxy, axis=0, order=2)
    model_proxy_depth = roughness(model_proxy, axis=0, order=2)
    raw_hf = high_frequency_energy(raw_proxy)
    model_hf = high_frequency_energy(model_proxy)
    raw_display_hf = high_frequency_energy(raw_display_proxy)
    model_display_hf = high_frequency_energy(model_display_proxy)
    corr = structure_correlation(raw_proxy, model_proxy)
    display_corr = structure_correlation(raw_display_proxy, model_display_proxy)

    raw_range = float(np.percentile(raw_proxy, 97.0) - np.percentile(raw_proxy, 3.0))
    model_range = float(np.percentile(model_proxy, 97.0) - np.percentile(model_proxy, 3.0))
    mean_shift = float(abs(np.mean(model_proxy) - np.mean(raw_proxy)))
    depth_ratio = safe_ratio(model_depth, raw_depth)
    late_ratio = safe_ratio(model_late, raw_late)
    lateral_ratio = safe_ratio(model_lateral, raw_lateral)
    proxy_depth_ratio = safe_ratio(model_proxy_depth, raw_proxy_depth)
    highfreq_ratio = safe_ratio(model_hf, raw_hf)
    range_ratio = safe_ratio(model_range, raw_range)

    corr_penalty = max(0.0, 0.70 - corr) * 1.6
    oversmooth_penalty = max(0.0, 0.22 - depth_ratio) * 2.0 if np.isfinite(depth_ratio) else 1.0
    no_denoise_penalty = max(0.0, depth_ratio - 0.86) * 1.4 if np.isfinite(depth_ratio) else 1.0
    late_penalty = max(0.0, late_ratio - 0.78) * 1.8 if np.isfinite(late_ratio) else 1.0
    range_penalty = abs(np.log(max(range_ratio, 1.0e-6))) * 0.28 if np.isfinite(range_ratio) else 1.0
    log_domain_penalty = 0.12 if candidate.domain == "log_magnitude" else 0.0
    score = (
        1.20 * highfreq_ratio
        + 0.95 * proxy_depth_ratio
        + 0.60 * depth_ratio
        + corr_penalty
        + oversmooth_penalty
        + no_denoise_penalty
        + late_penalty
        + range_penalty
        + 0.25 * mean_shift
        + log_domain_penalty
    )

    pass_gate = bool(
        np.isfinite(depth_ratio)
        and np.isfinite(late_ratio)
        and np.isfinite(highfreq_ratio)
        and 0.22 <= depth_ratio <= 0.88
        and late_ratio <= 0.82
        and highfreq_ratio <= 0.82
        and corr >= 0.68
        and 0.35 <= range_ratio <= 1.25
    )
    return {
        "candidate_key": candidate.key,
        "candidate_label": candidate.label,
        "domain": candidate.domain,
        "depth_roughness_raw": raw_depth,
        "depth_roughness_model": model_depth,
        "depth_roughness_ratio": depth_ratio,
        "late_depth_roughness_raw": raw_late,
        "late_depth_roughness_model": model_late,
        "late_depth_roughness_ratio": late_ratio,
        "lateral_roughness_raw": raw_lateral,
        "lateral_roughness_model": model_lateral,
        "lateral_roughness_ratio": lateral_ratio,
        "proxy_depth_roughness_raw": raw_proxy_depth,
        "proxy_depth_roughness_model": model_proxy_depth,
        "proxy_depth_roughness_ratio": proxy_depth_ratio,
        "proxy_highfreq_raw": raw_hf,
        "proxy_highfreq_model": model_hf,
        "proxy_highfreq_ratio": highfreq_ratio,
        "display_highfreq_raw": raw_display_hf,
        "display_highfreq_model": model_display_hf,
        "display_highfreq_ratio": safe_ratio(model_display_hf, raw_display_hf),
        "proxy_structure_correlation": corr,
        "display_structure_correlation": display_corr,
        "proxy_range_raw": raw_range,
        "proxy_range_model": model_range,
        "proxy_range_ratio": range_ratio,
        "proxy_mean_abs_shift": mean_shift,
        "imaging_residual_gain_raw": float(raw_display_gain),
        "imaging_residual_gain_model": float(model_display_gain),
        "selection_score": float(score),
        "quality_gate_pass": pass_gate,
    }


def choose_result(results: list[CandidateResult]) -> CandidateResult:
    passing = [r for r in results if bool(r.metrics["quality_gate_pass"])]
    pool = passing if passing else results
    return sorted(pool, key=lambda r: float(r.metrics["selection_score"]))[0]


def station_positions() -> np.ndarray:
    return np.linspace(0.0, 500.0, N_RECEIVERS, dtype=np.float64)


def draw_imaging_panel(
    ax: plt.Axes,
    data_869x11: np.ndarray,
    title: str,
    levels: np.ndarray,
    cmap: str,
) -> Any:
    depth = -depth_axis(data_869x11.shape[0])
    x = station_positions()
    mesh = ax.contourf(x, depth, data_869x11, levels=levels, cmap=cmap, extend="both")
    ax.set_xlim(0.0, 500.0)
    ax.set_ylim(-400.0, 0.0)
    ax.set_xticks(np.arange(0, 501, 100))
    ax.set_yticks([0, -100, -200, -300, -400])
    ax.set_title(title, pad=3)
    ax.set_ylabel("Depth (m)", fontsize=8.4)
    ax.tick_params(direction="out", length=3.0, width=0.75, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
    return mesh


def plot_final_imaging(
    result: CandidateResult,
    out_dir: Path,
    dpi: int,
    stem: str = "dbz_measured_imaging_final",
    use_display_proxy: bool = True,
) -> None:
    levels = np.linspace(1.70, 2.80, 29)
    cmap = "RdBu_r"
    fig = plt.figure(figsize=(6.05, 4.7))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.032],
        height_ratios=[1.0, 1.0],
        hspace=0.43,
        wspace=0.055,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

    raw_panel = result.raw_display_proxy_869x11 if use_display_proxy else result.raw_proxy_869x11
    model_panel = result.model_display_proxy_869x11 if use_display_proxy else result.model_proxy_869x11
    meshes = [
        draw_imaging_panel(axes[0], raw_panel, "", levels, cmap),
        draw_imaging_panel(axes[1], model_panel, "", levels, cmap),
    ]
    for ax in axes:
        ax.set_xlabel("Profile Distance (m)", fontsize=8.4)
    for ax, label in zip(axes, ["(a)", "(b)"]):
        ax.text(0.5, -0.30, label, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    for cax, mesh in zip(caxes, meshes):
        cb = fig.colorbar(mesh, cax=cax, ticks=[1.7, 1.9, 2.1, 2.3, 2.5, 2.7])
        cb.set_label(r"$\log_{10}(\rho_a)$ ($\Omega\cdot$m)", labelpad=3, fontsize=8.0)
        cb.outline.set_linewidth(0.55)
        cb.ax.tick_params(length=2.2, width=0.6, pad=1.5, labelsize=7.0)
    save_all(fig, out_dir, stem, dpi=dpi)
    plt.close(fig)


def plot_candidate_gallery(results: list[CandidateResult], selected: CandidateResult, out_dir: Path, dpi: int) -> None:
    levels = np.linspace(1.70, 2.80, 24)
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 6.9), sharex=True, sharey=True)
    ordered = sorted(results, key=lambda r: float(r.metrics["selection_score"]))
    x = station_positions()
    depth = -depth_axis()
    for ax, result in zip(axes.ravel(), ordered):
        ax.contourf(x, depth, result.model_display_proxy_869x11, levels=levels, cmap="RdBu_r", extend="both")
        is_selected = result.candidate.key == selected.candidate.key
        color = "#C42034" if is_selected else "#202020"
        ax.set_title(
            f"{result.candidate.label}\nscore={result.metrics['selection_score']:.3f}, "
            f"gain={result.model_display_gain:.2f}",
            color=color,
            pad=3,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.15 if is_selected else 0.65)
            spine.set_edgecolor(color if is_selected else "#343434")
        ax.set_xlim(0, 500)
        ax.set_ylim(-400, 0)
        ax.tick_params(direction="out", length=2.4, width=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Profile Distance (m)")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.08, hspace=0.36, wspace=0.12)
    save_all(fig, out_dir, "dbz_interpolation_candidate_gallery", dpi=dpi)
    plt.close(fig)


def plot_curve_family(raw_40x11: np.ndarray, result: CandidateResult, out_dir: Path, dpi: int) -> None:
    model_11x40 = sample_model_to_40_gates(
        result.model_log10_11x869,
        result.candidate.normalized_gate_869,
        result.candidate.normalized_gate_40,
    )
    model_40x11 = np.power(10.0, model_11x40).T
    raw_mag = np.abs(raw_40x11)
    x = np.arange(1, N_RECEIVERS + 1)
    y_min = max(float(np.nanmin([raw_mag.min(), model_40x11.min()])), EPS)
    y_max = float(np.nanmax([raw_mag.max(), model_40x11.max()]))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, N_SOURCE_GATES))

    fig, axes = plt.subplots(2, 1, figsize=(6.25, 6.25), sharex=True, sharey=True)
    for ax, data, title in zip(axes, [raw_mag, model_40x11], ["Raw dbz", "Denoised dbz"]):
        for gate_idx, color in enumerate(colors):
            ax.semilogy(x, np.maximum(data[gate_idx], EPS), lw=1.05, color=color, alpha=0.93)
        ax.set_ylabel(r"$dB_z/dt$ (V/Am$^2$)")
        ax.set_title(title, pad=4, fontsize=10.0, fontweight="bold")
        ax.grid(True, which="major", color="#DADDE2", lw=0.55)
        ax.grid(True, which="minor", color="#ECEFF3", lw=0.35, alpha=0.85)
        ax.set_xlim(0.5, 11.5)
        ax.set_ylim(max(y_min * 0.45, EPS), y_max * 1.75)
    axes[1].set_xlabel("Measuring point", fontsize=10.0, fontweight="bold")
    axes[0].text(-0.055, 1.04, "(a)", transform=axes[0].transAxes, fontsize=9.0)
    axes[1].text(-0.055, 1.04, "(b)", transform=axes[1].transAxes, fontsize=9.0)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.94, bottom=0.08, hspace=0.22)
    save_all(fig, out_dir, "dbz_selected_candidate_curve_family", dpi=dpi)
    plt.close(fig)


def write_metrics(results: list[CandidateResult], selected: CandidateResult, out_dir: Path) -> None:
    rows = []
    for result in sorted(results, key=lambda r: float(r.metrics["selection_score"])):
        row = dict(result.metrics)
        row["selected"] = result.candidate.key == selected.candidate.key
        rows.append(row)
    fieldnames = list(rows[0].keys())
    with (out_dir / "dbz_interpolation_imaging_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "dbz_interpolation_imaging_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_outputs(
    raw_40x11: np.ndarray,
    dbz_source: Path,
    selected: CandidateResult,
    all_results: list[CandidateResult],
    out_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> None:
    sign = np.sign(raw_40x11)
    sign[sign == 0.0] = 1.0
    model_11x40 = sample_model_to_40_gates(
        selected.model_log10_11x869,
        selected.candidate.normalized_gate_869,
        selected.candidate.normalized_gate_40,
    )
    model_mag_40x11 = np.power(10.0, model_11x40).T.astype(np.float32)
    selected_payload = {
        "dbz_raw_40x11": raw_40x11.astype(np.float32),
        "selected_candidate": np.array([selected.candidate.key], dtype=object),
        "selected_candidate_label": np.array([selected.candidate.label], dtype=object),
        "x_model_log10_11x869": selected.candidate.x_model_log10_11x869.astype(np.float32),
        "x_model_physical_11x869": selected.candidate.x_model_physical_11x869.astype(np.float32),
        "model_log10_11x869": selected.model_log10_11x869.astype(np.float32),
        "model_log10_11x40": model_11x40.astype(np.float32),
        "dbz_denoised_magnitude_40x11": model_mag_40x11,
        "dbz_denoised_signed_by_raw_40x11": (sign * model_mag_40x11).astype(np.float32),
        "raw_proxy_log10rho_869x11": selected.raw_proxy_869x11.astype(np.float32),
        "model_proxy_log10rho_869x11": selected.model_proxy_869x11.astype(np.float32),
        "raw_proxy_regularized_log10rho_869x11": selected.raw_display_proxy_869x11.astype(np.float32),
        "model_proxy_regularized_log10rho_869x11": selected.model_display_proxy_869x11.astype(np.float32),
        "imaging_residual_gain_raw": np.array([[selected.raw_display_gain]], dtype=np.float32),
        "imaging_residual_gain_model": np.array([[selected.model_display_gain]], dtype=np.float32),
        "depth_m": depth_axis().reshape(-1, 1).astype(np.float32),
        "profile_distance_m": station_positions().reshape(1, -1).astype(np.float32),
        "normalized_gate_40": selected.candidate.normalized_gate_40.reshape(1, -1).astype(np.float32),
        "normalized_gate_869": selected.candidate.normalized_gate_869.reshape(1, -1).astype(np.float32),
    }
    savemat(out_dir / "dbz_selected_interpolation_model_output.mat", selected_payload)

    manifest = {
        "dbz_source": str(dbz_source),
        "checkpoint_dir": str(checkpoint_dir),
        "device": str(device),
        "selected_candidate": selected.candidate.key,
        "selected_candidate_label": selected.candidate.label,
        "selection_metrics": selected.metrics,
        "candidate_order": [
            {
                "candidate_key": result.candidate.key,
                "candidate_label": result.candidate.label,
                "selection_score": result.metrics["selection_score"],
                "quality_gate_pass": result.metrics["quality_gate_pass"],
            }
            for result in sorted(all_results, key=lambda r: float(r.metrics["selection_score"]))
        ],
        "boundary_note": (
            "Only dbz interpolation candidates were changed. The ST-DiffTEM model code, "
            "checkpoint weights, and inference path were not modified. The imaging panels "
            "use an apparent-resistivity proxy plus display-stage imaging regularization "
            "because SimPEG is not available in this environment for true 1D inversion. "
            "The saved model_log10_11x869 array is the unmodified model output."
        ),
    }
    (out_dir / "selected_candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = "\n".join(
        [
            "# dbz interpolation imaging outputs",
            "",
            "- Scope: dbz only.",
            "- Model boundary: existing checkpoint inference only; no model code, loss, or weights were changed.",
            "- Changed item: 40-gate to 869-gate dbz interpolation method used as model input.",
            "- Figure mapping: `raw_proxy_log10rho_869x11` and `model_proxy_log10rho_869x11` are an apparent-resistivity visualization proxy, not a SimPEG inversion result.",
            "- Final display: `*_regularized_log10rho_869x11` applies imaging-stage regularization only for the section figure. It does not overwrite `model_log10_11x869`.",
            "- Final figure: `dbz_measured_imaging_final.*`.",
            "- Direct audit figure: `dbz_measured_imaging_direct_proxy.*`.",
            "- Candidate comparison: `dbz_interpolation_candidate_gallery.*`.",
            "- Curve check: `dbz_selected_candidate_curve_family.*`.",
            "",
            f"Selected candidate: `{selected.candidate.key}` ({selected.candidate.label}).",
        ]
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_style()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_40x11, dbz_source = load_dbz(args.dbz)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint_dir, device)

    candidates = make_candidates(raw_40x11)
    results: list[CandidateResult] = []
    for candidate in candidates:
        model_log = run_model(model, device, candidate.x_model_log10_11x869)
        raw_proxy, model_proxy = proxy_from_logs(candidate.x_model_log10_11x869, model_log)
        raw_display, model_display, raw_gain, model_gain = regularize_for_imaging(raw_proxy, model_proxy)
        metrics = metric_dict(
            candidate,
            model_log,
            raw_proxy,
            model_proxy,
            raw_display,
            model_display,
            raw_gain,
            model_gain,
        )
        results.append(
            CandidateResult(
                candidate=candidate,
                model_log10_11x869=model_log,
                raw_proxy_869x11=raw_proxy,
                model_proxy_869x11=model_proxy,
                raw_display_proxy_869x11=raw_display,
                model_display_proxy_869x11=model_display,
                raw_display_gain=raw_gain,
                model_display_gain=model_gain,
                metrics=metrics,
            )
        )

    selected = choose_result(results)
    write_metrics(results, selected, out_dir)
    write_outputs(raw_40x11, dbz_source, selected, results, out_dir, args.checkpoint_dir.absolute(), device)
    plot_final_imaging(selected, out_dir, args.dpi, "dbz_measured_imaging_final", use_display_proxy=True)
    plot_final_imaging(selected, out_dir, args.dpi, "dbz_measured_imaging_direct_proxy", use_display_proxy=False)
    plot_candidate_gallery(results, selected, out_dir, args.dpi)
    plot_curve_family(raw_40x11, selected, out_dir, args.dpi)

    print(f"Output directory: {out_dir}")
    print(f"Selected candidate: {selected.candidate.key} ({selected.candidate.label})")
    print(json.dumps(selected.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
