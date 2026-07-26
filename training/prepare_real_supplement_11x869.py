from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import PchipInterpolator

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


SCRIPT_PATH = Path(__file__).absolute()
WORKSPACE = SCRIPT_PATH.parents[2]
MODEL_CODE = WORKSPACE / "模型代码"
SUPPLEMENT_DIR = WORKSPACE / "补充"
DATA8_DIR = SUPPLEMENT_DIR / "data_8_extracted"
DBZ_PATH = SUPPLEMENT_DIR / "dbz.mat"
OUT_DIR = SUPPLEMENT_DIR / "processed_11x869"
TRAIN_DATA_PATH = WORKSPACE / "数据" / "data_paper_fixed_scenarios_25k" / "train" / "train_data.npy"
CHECKPOINT_DIR = (
    WORKSPACE
    / "训练结果"
    / "st_difftem_rho075_extra_seeds_20260512"
    / "rho075_mainline"
    / "seed_51"
    / "hybrid_base"
)
CHECKPOINT_ROOT = CHECKPOINT_DIR.parents[1]

SEQ_LEN = 869
RECEIVER_COUNT = 11
EPS = 1.0e-12
REFERENCE_QUANTILES = (1.0, 5.0, 50.0, 95.0, 99.0)

sys.path.insert(0, str(MODEL_CODE))
sys.path.insert(0, str(MODEL_CODE / "training"))


@dataclass(frozen=True)
class Sounding:
    profile: str
    station: float
    path: Path
    time_s: np.ndarray
    width_s: np.ndarray
    voltage: np.ndarray
    error_bar: np.ndarray
    mask: np.ndarray
    location_xyz: np.ndarray


@dataclass
class PreparedWindow:
    key: str
    profile: str
    records: list[Sounding]
    x_physical_11x869: np.ndarray
    x_log10_11x869: np.ndarray
    time_869_s: np.ndarray
    log10_time_869: np.ndarray
    original_time_s: np.ndarray
    original_voltage: np.ndarray
    original_width_s: np.ndarray
    original_error_bar: np.ndarray
    original_mask: np.ndarray
    model_log10_11x869: np.ndarray | None = None
    fallback_mean: float | None = None
    metrics: dict[str, Any] | None = None


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1.0e-12 or float(np.std(y)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def to_log10_amplitude(array: np.ndarray) -> np.ndarray:
    """Same model-input transform used by train_st_difftem.py/evaluate_noise_scenarios.py."""
    return np.log10(np.abs(np.asarray(array, dtype=np.float32)) + EPS).astype(np.float32)


def build_train_distribution_reference(train_data_path: Path, max_samples: int = 1024) -> dict[str, Any] | None:
    if not train_data_path.exists():
        return None
    arr = np.load(train_data_path, mmap_mode="r")
    if arr.ndim != 3:
        return None
    count = int(arr.shape[0])
    if count <= 0:
        return None
    take = min(count, int(max_samples))
    indices = np.unique(np.round(np.linspace(0, count - 1, take)).astype(np.int64))
    sample = np.asarray(arr[indices], dtype=np.float32)
    if sample.shape[1] != RECEIVER_COUNT and sample.shape[2] == RECEIVER_COUNT:
        sample = np.transpose(sample, (0, 2, 1))
    if sample.shape[1] != RECEIVER_COUNT:
        return None
    log_sample = to_log10_amplitude(sample)
    return {
        "path": str(train_data_path),
        "sample_count": int(sample.shape[0]),
        "quantiles": np.percentile(log_sample.reshape(-1), REFERENCE_QUANTILES).astype(np.float64),
        "start_quantiles": np.percentile(log_sample[:, :, 0].reshape(-1), REFERENCE_QUANTILES).astype(np.float64),
        "end_quantiles": np.percentile(log_sample[:, :, -1].reshape(-1), REFERENCE_QUANTILES).astype(np.float64),
    }


def train_distribution_distance(log_arr: np.ndarray, reference: dict[str, Any] | None) -> tuple[float, dict[str, float]]:
    if reference is None:
        return 0.0, {}
    log_arr = np.asarray(log_arr, dtype=np.float64)
    all_q = np.percentile(log_arr.reshape(-1), REFERENCE_QUANTILES)
    start_q = np.percentile(log_arr[:, 0].reshape(-1), REFERENCE_QUANTILES)
    end_q = np.percentile(log_arr[:, -1].reshape(-1), REFERENCE_QUANTILES)
    ref_all = np.asarray(reference["quantiles"], dtype=np.float64)
    ref_start = np.asarray(reference["start_quantiles"], dtype=np.float64)
    ref_end = np.asarray(reference["end_quantiles"], dtype=np.float64)
    all_dist = float(np.mean(np.abs(all_q - ref_all)))
    start_dist = float(np.mean(np.abs(start_q - ref_start)))
    end_dist = float(np.mean(np.abs(end_q - ref_end)))
    score = 0.60 * all_dist + 0.25 * start_dist + 0.15 * end_dist
    return float(score), {
        "train_distribution_score": float(score),
        "log_q01": float(all_q[0]),
        "log_q05": float(all_q[1]),
        "log_q50": float(all_q[2]),
        "log_q95": float(all_q[3]),
        "log_q99": float(all_q[4]),
        "train_ref_log_q01": float(ref_all[0]),
        "train_ref_log_q05": float(ref_all[1]),
        "train_ref_log_q50": float(ref_all[2]),
        "train_ref_log_q95": float(ref_all[3]),
        "train_ref_log_q99": float(ref_all[4]),
    }


def _prep_interp_xy(x_src: np.ndarray, y_src: np.ndarray):
    x = np.asarray(x_src, dtype=np.float64).reshape(-1)
    y = np.asarray(y_src, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError("Need at least two finite source samples for interpolation.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    return unique_x, y[unique_idx]


def interp_linear(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation — preserves original data character including noise."""
    x, y = _prep_interp_xy(x_src, y_src)
    dst = np.asarray(x_dst, dtype=np.float64).reshape(-1)
    return np.interp(dst, x, y).astype(np.float32)


def interp_pchip(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """PCHIP (shape-preserving cubic) interpolation — smooth, for back-sampling model output."""
    x, y = _prep_interp_xy(x_src, y_src)
    dst = np.asarray(x_dst, dtype=np.float64).reshape(-1)
    if x.size < 4:
        return np.interp(dst, x, y).astype(np.float32)
    try:
        vals = PchipInterpolator(x, y, extrapolate=False)(dst)
        if not np.all(np.isfinite(vals)):
            vals = np.interp(dst, x, y)
    except Exception:
        vals = np.interp(dst, x, y)
    return np.asarray(vals, dtype=np.float32)


def interp_rows_linear(rows: np.ndarray, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Linear-interpolate each row. Preserves noise character for model input."""
    return np.stack([interp_linear(x_src, row, x_dst) for row in rows], axis=0).astype(np.float32)


def interp_rows_pchip(rows: np.ndarray, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """PCHIP-interpolate each row. For back-sampling model output to original gates."""
    return np.stack([interp_pchip(x_src, row, x_dst) for row in rows], axis=0).astype(np.float32)


def load_model(checkpoint_dir: Path):
    if torch is None:
        raise RuntimeError("PyTorch is required for model inference, but torch is not installed.")
    from evaluate_noise_scenarios import build_model, load_checkpoint_state

    run_config = json.loads((checkpoint_dir / "run_config.json").read_text(encoding="utf-8"))
    model = build_model("stdifftem", dict(run_config["model"]))
    model.load_state_dict(load_checkpoint_state(checkpoint_dir / "best_model.pth"), strict=True)
    model.eval()
    return model


def checkpoint_label(checkpoint_dir: Path) -> str:
    if checkpoint_dir.parent.name.startswith("seed_"):
        return checkpoint_dir.parent.name
    return checkpoint_dir.name


def available_checkpoints(checkpoint_root: Path | None, fallback_checkpoint: Path) -> list[Path]:
    if checkpoint_root is not None and checkpoint_root.exists():
        dirs = sorted(checkpoint_root.glob("seed_*/hybrid_base"))
        dirs = [d for d in dirs if (d / "run_config.json").exists() and (d / "best_model.pth").exists()]
        if dirs:
            return dirs
    return [fallback_checkpoint]


def infer_model(model, windows: list[np.ndarray], batch_size: int = 16) -> tuple[list[np.ndarray], list[float | None]]:
    if not windows:
        return [], []
    preds: list[np.ndarray] = []
    fallback_means: list[float | None] = []
    with torch.no_grad():
        for offset in range(0, len(windows), batch_size):
            batch = np.asarray(windows[offset : offset + batch_size], dtype=np.float32)
            pred, aux = model(torch.from_numpy(batch), return_aux=True)
            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            preds.extend([pred_np[i] for i in range(pred_np.shape[0])])
            fallback = None
            if isinstance(aux, dict) and "w_single" in aux and "rho_max" in aux:
                fallback_arr = (aux["w_single"] / aux["rho_max"]).detach().cpu().numpy()
                fallback = [float(np.mean(fallback_arr[i])) for i in range(fallback_arr.shape[0])]
            if fallback is None:
                fallback_means.extend([None] * pred_np.shape[0])
            else:
                fallback_means.extend(fallback)
    return preds, fallback_means


def parse_sounding(path: Path) -> Sounding:
    text = path.read_text(errors="ignore")
    profile_match = re.search(r"/PROFILE:\s*(\S+)", text)
    location_match = re.search(r"/LOCATION:\s*([^\n]+)", text)
    rows: list[list[float]] = []
    start = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("INDEX,"):
            start = True
            continue
        if not start or not stripped or stripped.startswith("/END"):
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append([float(parts[i]) for i in range(6)])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    arr = np.asarray(rows, dtype=np.float64)
    location = np.full(3, np.nan, dtype=np.float64)
    if location_match:
        try:
            location = np.asarray([float(v.strip()) for v in location_match.group(1).split(",")[:3]], dtype=np.float64)
        except ValueError:
            pass
    return Sounding(
        profile=profile_match.group(1) if profile_match else path.stem.split("_", 1)[0],
        station=float(path.stem.split("_", 1)[1]),
        path=path,
        time_s=arr[:, 1].astype(np.float64),
        width_s=arr[:, 2].astype(np.float64),
        voltage=arr[:, 3].astype(np.float64),
        error_bar=arr[:, 4].astype(np.float64),
        mask=arr[:, 5].astype(np.float64),
        location_xyz=location,
    )


def load_profiles(data_dir: Path) -> dict[str, list[Sounding]]:
    profiles: dict[str, list[Sounding]] = {}
    for path in sorted(data_dir.glob("*.txt")):
        rec = parse_sounding(path)
        profiles.setdefault(rec.profile, []).append(rec)
    for profile in profiles:
        profiles[profile].sort(key=lambda item: item.station)
    return profiles


def align_original_gates(records: list[Sounding]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_times = sorted({float(v) for rec in records for v in np.round(rec.time_s, 12)})
    time_s = np.asarray(all_times, dtype=np.float64)
    n_t = time_s.size
    n_r = len(records)
    voltage = np.full((n_t, n_r), np.nan, dtype=np.float32)
    width_s = np.full((n_t, n_r), np.nan, dtype=np.float32)
    error_bar = np.full((n_t, n_r), np.nan, dtype=np.float32)
    mask = np.full((n_t, n_r), np.nan, dtype=np.float32)
    idx = {float(t): i for i, t in enumerate(time_s)}
    for j, rec in enumerate(records):
        for k, t in enumerate(np.round(rec.time_s, 12)):
            i = idx[float(t)]
            voltage[i, j] = rec.voltage[k]
            width_s[i, j] = rec.width_s[k]
            error_bar[i, j] = rec.error_bar[k]
            mask[i, j] = rec.mask[k]
    return time_s, voltage, width_s, error_bar, mask


def prepare_data8_window(profile: str, records: list[Sounding]) -> PreparedWindow:
    min_log_t = max(float(np.log10(np.nanmin(rec.time_s))) for rec in records)
    max_log_t = min(float(np.log10(np.nanmax(rec.time_s))) for rec in records)
    if not np.isfinite(min_log_t) or not np.isfinite(max_log_t) or min_log_t >= max_log_t:
        raise ValueError(f"{profile}: invalid overlapping time range")
    log_time_869 = np.linspace(min_log_t, max_log_t, SEQ_LEN, dtype=np.float64)
    physical_rows = []
    for rec in records:
        source_log_t = np.log10(rec.time_s)
        # Match the paper dataset path: keep the measured physical-domain signal
        # first, then apply train_st_difftem.py::to_log10_amplitude.
        physical_rows.append(interp_linear(source_log_t, rec.voltage, log_time_869))
    x_physical = np.asarray(physical_rows, dtype=np.float32)
    x_log = to_log10_amplitude(x_physical)
    time_s, voltage, width_s, error_bar, mask = align_original_gates(records)
    station_start = int(round(records[0].station))
    station_end = int(round(records[-1].station))
    key = f"{profile}_{station_start}_{station_end}"
    return PreparedWindow(
        key=key,
        profile=profile,
        records=records,
        x_physical_11x869=x_physical,
        x_log10_11x869=x_log,
        time_869_s=np.power(10.0, log_time_869).astype(np.float64),
        log10_time_869=log_time_869.astype(np.float64),
        original_time_s=time_s,
        original_voltage=voltage,
        original_width_s=width_s,
        original_error_bar=error_bar,
        original_mask=mask,
    )


def contiguous_windows(records: list[Sounding], width: int = RECEIVER_COUNT) -> list[list[Sounding]]:
    windows: list[list[Sounding]] = []
    for start in range(0, len(records) - width + 1):
        group = records[start : start + width]
        diffs = np.diff([rec.station for rec in group])
        if np.all(np.isclose(diffs, 25.0, atol=1.0e-6)):
            windows.append(group)
    if windows:
        return windows
    return [records[start : start + width] for start in range(0, len(records) - width + 1)]


def score_window(
    window: PreparedWindow,
    model_log10_11x869: np.ndarray | None = None,
    fallback_mean: float | None = None,
    train_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_log = np.asarray(window.x_log10_11x869, dtype=np.float32)
    if model_log10_11x869 is None:
        model_log10_11x869 = window.model_log10_11x869
    model_log = np.asarray(model_log10_11x869, dtype=np.float32) if model_log10_11x869 is not None else raw_log
    ratio = np.power(10.0, model_log - raw_log)
    raw_d2 = np.diff(raw_log, n=2, axis=1)
    model_d2 = np.diff(model_log, n=2, axis=1)
    raw_spatial = np.diff(raw_log, n=2, axis=0)
    model_spatial = np.diff(model_log, n=2, axis=0)
    raw_rough = float(np.mean(np.abs(raw_d2))) if raw_d2.size else 0.0
    model_rough = float(np.mean(np.abs(model_d2))) if model_d2.size else 0.0
    spatial_raw = float(np.mean(np.abs(raw_spatial))) if raw_spatial.size else 0.0
    spatial_model = float(np.mean(np.abs(model_spatial))) if model_spatial.size else 0.0
    median_ratio = float(np.median(ratio))
    p05 = float(np.percentile(ratio, 5.0))
    p95 = float(np.percentile(ratio, 95.0))
    rough_ratio = model_rough / raw_rough if raw_rough > 0.0 else float("nan")
    spatial_ratio = spatial_model / spatial_raw if spatial_raw > 0.0 else float("nan")
    corr = finite_corr(raw_log, model_log)
    finite_error = window.original_error_bar[np.isfinite(window.original_error_bar)]
    error_median = float(np.median(finite_error)) if finite_error.size else 0.0
    if not np.isfinite(error_median):
        error_median = 0.0
    mask_keep_fraction = float(np.nanmean(window.original_mask > 0.0))
    values = window.original_voltage[np.isfinite(window.original_voltage)]
    positive_fraction = float(np.mean(values > 0.0)) if values.size else float("nan")
    sign_stability = max(positive_fraction, 1.0 - positive_fraction) if np.isfinite(positive_fraction) else float("nan")
    frac_outside = float(np.mean((ratio < 0.5) | (ratio > 2.0)))
    train_dist_score, train_dist_fields = train_distribution_distance(raw_log, train_reference)

    amp_penalty = abs(np.log10(max(median_ratio, EPS)))
    spread_penalty = abs(np.log10(max(p95 / max(p05, EPS), 1.0)))
    temporal_penalty = 0.1
    if np.isfinite(rough_ratio):
        # The PCHIP-interpolated raw traces can be nearly perfectly smooth, so
        # the temporal second-difference ratio is diagnostic only. Keep it from
        # dominating the window choice.
        temporal_penalty = 0.05 * min(abs(np.log10(max(rough_ratio, EPS))), 3.0)
    spatial_penalty = 1.0
    if np.isfinite(spatial_ratio):
        spatial_penalty = abs(np.log10(max(spatial_ratio, EPS)))
    corr_penalty = max(0.0, 0.90 - corr) * 2.0 if np.isfinite(corr) else 2.0
    sign_penalty = max(0.0, 0.97 - sign_stability) if np.isfinite(sign_stability) else 0.5
    mask_penalty = max(0.0, 1.0 - mask_keep_fraction)
    score = (
        2.0 * amp_penalty
        + 0.35 * spread_penalty
        + 1.5 * frac_outside
        + 1.2 * train_dist_score
        + temporal_penalty
        + spatial_penalty
        + corr_penalty
        + 0.08 * np.log1p(max(error_median, 0.0))
        + sign_penalty
        + mask_penalty
    )
    stations = [rec.station for rec in window.records] or [float(i) for i in range(1, raw_log.shape[0] + 1)]
    out = {
        "key": window.key,
        "profile": window.profile,
        "station_start": float(stations[0]),
        "station_end": float(stations[-1]),
        "receiver_count": int(len(stations)),
        "time_gates_original_union": int(window.original_time_s.size),
        "time_min_s": float(window.time_869_s[0]),
        "time_max_s": float(window.time_869_s[-1]),
        "median_model_raw_ratio": median_ratio,
        "p05_model_raw_ratio": p05,
        "p95_model_raw_ratio": p95,
        "frac_ratio_outside_0p5_2": frac_outside,
        "roughness_raw": raw_rough,
        "roughness_model": model_rough,
        "roughness_ratio": float(rough_ratio),
        "spatial_roughness_raw": spatial_raw,
        "spatial_roughness_model": spatial_model,
        "spatial_roughness_ratio": float(spatial_ratio),
        "structure_correlation": corr,
        "error_bar_median": error_median,
        "mask_keep_fraction": mask_keep_fraction,
        "positive_fraction": positive_fraction,
        "sign_stability": sign_stability,
        "fallback_mean": window.fallback_mean if fallback_mean is None else fallback_mean,
        "selection_score": float(score),
        "preprocess_pipeline": "physical_domain_interpolation_then_log10_abs_eps",
    }
    out.update(train_dist_fields)
    return out


def process_dbz(
    dbz_path: Path,
    model=None,
    checkpoint_dir: Path | None = None,
    train_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mat = loadmat(dbz_path)
    if "dbz" not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        raise KeyError(f"dbz not found in {dbz_path}; available keys: {keys}")
    raw = np.asarray(mat["dbz"], dtype=np.float32)
    if raw.shape == (RECEIVER_COUNT, 40):
        raw = raw.T
    if raw.ndim != 2 or raw.shape[1] != RECEIVER_COUNT:
        raise ValueError(f"{dbz_path}: expected 40x11 or 11x40 dbz matrix, got {raw.shape}")

    raw_physical_11x40 = raw.T.astype(np.float32)
    raw_log_11x40 = to_log10_amplitude(raw_physical_11x40)
    coord_40 = np.linspace(0.0, 1.0, raw_physical_11x40.shape[1], dtype=np.float64)
    coord_869 = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float64)
    x_physical = interp_rows_linear(raw_physical_11x40, coord_40, coord_869)
    x_log = to_log10_amplitude(x_physical)
    payload: dict[str, Any] = {
        "raw_voltage_40x11": raw.astype(np.float32),
        "raw_log10_11x40": raw_log_11x40,
        "x_model_physical_11x869": x_physical.astype(np.float32),
        "x_model_physical_869x11": x_physical.T.astype(np.float32),
        "x_model_log10_11x869": x_log,
        "x_model_magnitude_11x869": np.abs(x_physical).astype(np.float32),
        "x_model_magnitude_869x11": np.abs(x_physical).T.astype(np.float32),
        "normalized_gate_40": coord_40.reshape(1, -1).astype(np.float32),
        "normalized_gate_869": coord_869.reshape(1, -1).astype(np.float32),
        "receiver_index_1based": np.arange(1, RECEIVER_COUNT + 1, dtype=np.int32).reshape(1, -1),
        "interpolation_method": "TRAIN-MATCHED: LINEAR interpolation in the physical voltage domain over normalized ordered gate coordinate, then log10(abs(x)+1e-12), matching train_st_difftem.py preprocessing. dbz.mat has no physical time vector.",
        "shape_note": "x_model_physical_11x869 matches the train_data.npy physical-domain layout; x_model_log10_11x869 is the direct ST-DiffTEM tensor.",
    }
    metrics: dict[str, Any] | None = None
    if model is not None:
        preds, fallback = infer_model(model, [x_log])
        model_log = preds[0]
        model_40 = interp_rows_pchip(model_log, coord_869, coord_40)
        sign = np.sign(raw)
        sign[sign == 0.0] = 1.0
        payload.update(
            {
                "model_log10_11x869": model_log,
                "model_magnitude_11x869": np.power(10.0, model_log).astype(np.float32),
                "model_magnitude_869x11": np.power(10.0, model_log).T.astype(np.float32),
                "model_log10_11x40": model_40,
                "model_magnitude_40x11": np.power(10.0, model_40).T.astype(np.float32),
                "model_signed_by_raw_40x11": (sign * np.power(10.0, model_40).T).astype(np.float32),
                "fallback_mean": np.asarray([[fallback[0] if fallback[0] is not None else np.nan]], dtype=np.float32),
            }
        )
        tmp = PreparedWindow(
            key="dbz",
            profile="dbz",
            records=[],
            x_physical_11x869=x_physical,
            x_log10_11x869=x_log,
            time_869_s=coord_869,
            log10_time_869=coord_869,
            original_time_s=coord_40,
            original_voltage=raw,
            original_width_s=np.full_like(raw, np.nan),
            original_error_bar=np.full_like(raw, np.nan),
            original_mask=np.ones_like(raw),
            model_log10_11x869=model_log,
            fallback_mean=fallback[0],
        )
        metrics = score_window(tmp, train_reference=train_reference)
        metrics["interpolation_method"] = "linear_normalized_gate_physical_then_log10_abs"
        if checkpoint_dir is not None:
            metrics["checkpoint"] = checkpoint_label(checkpoint_dir)
            metrics["checkpoint_path"] = str(checkpoint_dir)
            payload["selected_checkpoint"] = checkpoint_label(checkpoint_dir)
            payload["selected_checkpoint_path"] = str(checkpoint_dir)
        payload["selection_score"] = np.asarray([[metrics["selection_score"]]], dtype=np.float32)

    return {"payload": payload, "metrics": metrics}


def sample_model_to_original_gates(window: PreparedWindow) -> tuple[np.ndarray, np.ndarray]:
    if window.model_log10_11x869 is None:
        n_t, n_r = window.original_voltage.shape
        return np.full((n_t, n_r), np.nan, dtype=np.float32), np.full((n_t, n_r), np.nan, dtype=np.float32)
    out_mag = np.full_like(window.original_voltage, np.nan, dtype=np.float32)
    out_signed = np.full_like(window.original_voltage, np.nan, dtype=np.float32)
    for j, rec in enumerate(window.records):
        valid = np.isfinite(window.original_voltage[:, j])
        if not np.any(valid):
            continue
        target_log_t = np.log10(window.original_time_s[valid])
        pred_log = interp_pchip(window.log10_time_869, window.model_log10_11x869[j], target_log_t)
        mag = np.power(10.0, pred_log).astype(np.float32)
        raw_vals = window.original_voltage[valid, j]
        sign = np.sign(raw_vals)
        sign[sign == 0.0] = 1.0
        out_mag[valid, j] = mag
        out_signed[valid, j] = (sign * mag).astype(np.float32)
    return out_mag, out_signed


def mat_string_vector(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object).reshape(1, -1)


def save_data8_window(window: PreparedWindow, out_dir: Path) -> None:
    stations = np.asarray([rec.station for rec in window.records], dtype=np.float32).reshape(1, -1)
    locations = np.asarray([rec.location_xyz for rec in window.records], dtype=np.float32)
    model_mag_orig, model_signed_orig = sample_model_to_original_gates(window)
    payload: dict[str, Any] = {
        "x_model_physical_11x869": window.x_physical_11x869.astype(np.float32),
        "x_model_physical_869x11": window.x_physical_11x869.T.astype(np.float32),
        "x_model_log10_11x869": window.x_log10_11x869.astype(np.float32),
        "x_model_magnitude_11x869": np.abs(window.x_physical_11x869).astype(np.float32),
        "x_model_magnitude_869x11": np.abs(window.x_physical_11x869).T.astype(np.float32),
        "time_interp_s": window.time_869_s.reshape(1, -1).astype(np.float64),
        "log10_time_interp": window.log10_time_869.reshape(1, -1).astype(np.float64),
        "original_time_s": window.original_time_s.reshape(-1, 1).astype(np.float64),
        "original_voltage": window.original_voltage.astype(np.float32),
        "original_width_s": window.original_width_s.astype(np.float32),
        "original_error_bar": window.original_error_bar.astype(np.float32),
        "original_mask": window.original_mask.astype(np.float32),
        "selected_station_m": stations,
        "selected_files": mat_string_vector([rec.path.name for rec in window.records]),
        "location_xyz": locations,
        "profile_name": window.profile,
        "interpolation_method": "TRAIN-MATCHED: LINEAR interpolation in the measured physical voltage domain over log10(time_s), then log10(abs(x)+1e-12), matching train_st_difftem.py preprocessing.",
        "shape_note": "x_model_physical_11x869 matches the train_data.npy physical-domain layout; x_model_log10_11x869 is the direct ST-DiffTEM tensor.",
    }
    if window.model_log10_11x869 is not None:
        payload.update(
            {
                "model_log10_11x869": window.model_log10_11x869.astype(np.float32),
                "model_magnitude_11x869": np.power(10.0, window.model_log10_11x869).astype(np.float32),
                "model_magnitude_869x11": np.power(10.0, window.model_log10_11x869).T.astype(np.float32),
                "model_magnitude_original_gates": model_mag_orig,
                "model_signed_by_raw_original_gates": model_signed_orig,
                "fallback_mean": np.asarray([[window.fallback_mean if window.fallback_mean is not None else np.nan]], dtype=np.float32),
            }
        )
    if window.metrics is not None:
        payload["selection_score"] = np.asarray([[window.metrics["selection_score"]]], dtype=np.float32)
        if "checkpoint" in window.metrics:
            payload["selected_checkpoint"] = str(window.metrics["checkpoint"])
        if "checkpoint_path" in window.metrics:
            payload["selected_checkpoint_path"] = str(window.metrics["checkpoint_path"])
    stem = f"taochong_{window.profile}_selected_11x869"
    savemat(out_dir / f"{stem}.mat", payload)
    np.save(out_dir / f"{stem}_x_model_physical_11x869.npy", window.x_physical_11x869.astype(np.float32))
    np.save(out_dir / f"{stem}_x_model_physical_batch_1x11x869.npy", window.x_physical_11x869[None].astype(np.float32))
    np.save(out_dir / f"{stem}_x_model_log10_11x869.npy", window.x_log10_11x869.astype(np.float32))
    np.save(out_dir / f"{stem}_x_model_batch_1x11x869.npy", window.x_log10_11x869[None].astype(np.float32))
    np.savetxt(out_dir / f"{stem}_time_interp_s.txt", window.time_869_s, fmt="%.12e")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(out_dir: Path, selected_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Real supplement data converted to 11x869",
        "",
        "All source files remain unchanged. The arrays named `x_model_physical_11x869` follow the same physical-domain layout as `train_data.npy`: 11 receivers by 869 time gates. The arrays named `x_model_log10_11x869` are produced by the exact training/evaluation transform `log10(abs(x)+1e-12)` and are the direct ST-DiffTEM tensors.",
        "",
        "Interpolation is performed before the log transform, in the measured physical-voltage domain. For `data_8_extracted`, interpolation uses LINEAR over measured `log10(time_s)`. For `dbz.mat`, no physical time vector is stored, so interpolation uses LINEAR over the normalized ordered gate coordinate. After interpolation, the same training/evaluation preprocessing is applied: `log10(abs(x)+1e-12)`.",
        "",
        "Generated files:",
        "",
        "- `dbz_11x869.mat`: processed `dbz.mat`, including the model input and, when a checkpoint is available, model output.",
        "- `taochong_W*_selected_11x869.mat`: one selected contiguous 11-station window per W21/W23/W25 profile.",
        "- `*_x_model_physical_11x869.npy`: train-style physical-domain arrays before the log transform.",
        "- `*_x_model_log10_11x869.npy` and `*_x_model_batch_1x11x869.npy`: direct NumPy model tensors after the training/evaluation log transform.",
        "- `window_diagnostics.csv`: all candidate windows and their no-reference model diagnostics.",
        "- `selected_windows.json`: selected windows and scores.",
        "",
        "Selected windows:",
        "",
    ]
    for row in selected_rows:
        checkpoint = row.get("checkpoint", "no-model")
        lines.append(
            f"- {row['profile']}: stations {row['station_start']:.0f}-{row['station_end']:.0f}, "
            f"score={row['selection_score']:.4f}, corr={row['structure_correlation']:.4f}, "
            f"median_ratio={row['median_model_raw_ratio']:.4f}, checkpoint={checkpoint}"
        )
    lines.append("")
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real supplement TEM data to fixed 11x869 model inputs.")
    parser.add_argument("--supplement-dir", type=Path, default=SUPPLEMENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--no-model", action="store_true", help="Only prepare inputs; skip RA-ST-DiffTEM inference.")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supplement_dir = args.supplement_dir
    data8_dir = supplement_dir / "data_8_extracted"
    dbz_path = supplement_dir / "dbz.mat"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_reference = build_train_distribution_reference(args.train_data)
    if train_reference is not None:
        q = train_reference["quantiles"]
        print(
            "train reference:",
            f"samples={train_reference['sample_count']}",
            f"log_q05={q[1]:.3f}",
            f"log_q50={q[2]:.3f}",
            f"log_q95={q[3]:.3f}",
        )

    checkpoint_items: list[tuple[Path, Any]] = []
    if not args.no_model:
        checkpoints = available_checkpoints(args.checkpoint_root, args.checkpoint_dir)
        print("checkpoints:", ", ".join(checkpoint_label(p) for p in checkpoints))
        for checkpoint_dir in checkpoints:
            checkpoint_items.append((checkpoint_dir, load_model(checkpoint_dir)))

    selected_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    if checkpoint_items:
        dbz_results = [
            process_dbz(dbz_path, model=model, checkpoint_dir=checkpoint_dir, train_reference=train_reference)
            for checkpoint_dir, model in checkpoint_items
        ]
        dbz_result = min(
            dbz_results,
            key=lambda item: float(item["metrics"]["selection_score"]) if item["metrics"] is not None else float("inf"),
        )
        for item in dbz_results:
            if item["metrics"] is not None:
                row = item["metrics"]
                row["selected"] = item is dbz_result
                all_rows.append(row)
        if dbz_result["metrics"] is not None:
            dbz_metrics = dict(dbz_result["metrics"])
            dbz_metrics["selected"] = True
            selected_rows.append(dbz_metrics)
    else:
        dbz_result = process_dbz(dbz_path, model=None, train_reference=train_reference)

    savemat(out_dir / "dbz_11x869.mat", dbz_result["payload"])
    np.save(out_dir / "dbz_x_model_physical_11x869.npy", dbz_result["payload"]["x_model_physical_11x869"].astype(np.float32))
    np.save(out_dir / "dbz_x_model_physical_batch_1x11x869.npy", dbz_result["payload"]["x_model_physical_11x869"][None].astype(np.float32))
    np.save(out_dir / "dbz_x_model_log10_11x869.npy", dbz_result["payload"]["x_model_log10_11x869"].astype(np.float32))
    np.save(out_dir / "dbz_x_model_batch_1x11x869.npy", dbz_result["payload"]["x_model_log10_11x869"][None].astype(np.float32))

    profiles = load_profiles(data8_dir)
    prepared: list[PreparedWindow] = []
    for profile, records in sorted(profiles.items()):
        for group in contiguous_windows(records):
            prepared.append(prepare_data8_window(profile, group))

    selected_candidates: dict[str, tuple[PreparedWindow, np.ndarray | None, float | None, dict[str, Any]]] = {}
    if prepared and checkpoint_items:
        input_windows = [w.x_log10_11x869 for w in prepared]
        for checkpoint_dir, model in checkpoint_items:
            label = checkpoint_label(checkpoint_dir)
            preds, fallback_means = infer_model(model, input_windows, batch_size=args.batch_size)
            for window, pred, fallback in zip(prepared, preds, fallback_means):
                metrics = score_window(
                    window,
                    model_log10_11x869=pred,
                    fallback_mean=fallback,
                    train_reference=train_reference,
                )
                metrics["checkpoint"] = label
                metrics["checkpoint_path"] = str(checkpoint_dir)
                metrics["selected"] = False
                all_rows.append(metrics)
                current = selected_candidates.get(window.profile)
                if current is None or float(metrics["selection_score"]) < float(current[3]["selection_score"]):
                    selected_candidates[window.profile] = (window, pred, fallback, metrics)
    else:
        for window in prepared:
            metrics = score_window(window, train_reference=train_reference)
            metrics["selected"] = False
            all_rows.append(metrics)
            current = selected_candidates.get(window.profile)
            if current is None or float(metrics["selection_score"]) < float(current[3]["selection_score"]):
                selected_candidates[window.profile] = (window, None, None, metrics)

    selected: dict[str, PreparedWindow] = {}
    for profile, (window, pred, fallback, metrics) in sorted(selected_candidates.items()):
        window.model_log10_11x869 = pred
        window.fallback_mean = fallback
        metrics["selected"] = True
        window.metrics = metrics
        selected[profile] = window
        selected_rows.append(dict(metrics))
        save_data8_window(window, out_dir)

    combined: dict[str, Any] = {
        "dbz_x_model_physical_11x869": dbz_result["payload"]["x_model_physical_11x869"].astype(np.float32),
        "dbz_x_model_log10_11x869": dbz_result["payload"]["x_model_log10_11x869"].astype(np.float32),
    }
    if "model_log10_11x869" in dbz_result["payload"]:
        combined["dbz_model_log10_11x869"] = dbz_result["payload"]["model_log10_11x869"].astype(np.float32)
    for profile, window in selected.items():
        combined[f"{profile}_x_model_log10_11x869"] = window.x_log10_11x869.astype(np.float32)
        combined[f"{profile}_x_model_physical_11x869"] = window.x_physical_11x869.astype(np.float32)
        combined[f"{profile}_stations"] = np.asarray([rec.station for rec in window.records], dtype=np.float32)
        combined[f"{profile}_time_interp_s"] = window.time_869_s.astype(np.float64)
        if window.model_log10_11x869 is not None:
            combined[f"{profile}_model_log10_11x869"] = window.model_log10_11x869.astype(np.float32)
    np.savez_compressed(out_dir / "real_supplement_selected_11x869.npz", **combined)

    write_csv(out_dir / "window_diagnostics.csv", all_rows)
    out_dir.joinpath("selected_windows.json").write_text(json.dumps(selected_rows, indent=2), encoding="utf-8")
    write_readme(out_dir, selected_rows)

    print(f"written: {out_dir}")
    for row in selected_rows:
        checkpoint = row.get("checkpoint", "no-model")
        print(
            f"{row['profile']}: stations {row['station_start']:.0f}-{row['station_end']:.0f}, "
            f"score={row['selection_score']:.4f}, checkpoint={checkpoint}, shape=11x869"
        )


if __name__ == "__main__":
    main()
