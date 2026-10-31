"""Optional input-dependent output-strength calibration for log TEM responses.

The regressor is fitted separately on validation data. At inference this module
uses only the noisy input and the frozen denoiser's prediction, never a clean
target or a scenario label.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def feature_matrix(noisy_log10: np.ndarray, predicted_log10: np.ndarray) -> np.ndarray:
    """Return one 19-feature row per receiver-time sample."""
    noisy = np.asarray(noisy_log10, dtype=np.float32)
    pred = np.asarray(predicted_log10, dtype=np.float32)
    if noisy.shape != pred.shape or noisy.ndim != 3:
        raise ValueError("Expected matching [samples, receivers, gates] arrays")
    length = noisy.shape[-1]
    first = round(length / 3)
    second = round(2 * length / 3)
    delta = pred - noisy
    abs_delta = np.abs(delta)
    noisy_physical = np.power(10.0, noisy.astype(np.float64))
    pred_physical = np.power(10.0, pred.astype(np.float64))
    change = np.abs(pred_physical - noisy_physical)
    signal_scale = np.sqrt(np.mean(noisy_physical ** 2, axis=(1, 2))) + EPS
    values = [
        noisy.mean((1, 2)), noisy.std((1, 2)),
        pred.mean((1, 2)), pred.std((1, 2)),
        abs_delta.mean((1, 2)), abs_delta.std((1, 2)),
        np.quantile(abs_delta, .5, axis=(1, 2)),
        np.quantile(abs_delta, .9, axis=(1, 2)),
        np.quantile(abs_delta, .99, axis=(1, 2)),
        abs_delta[..., :first].mean((1, 2)),
        abs_delta[..., first:second].mean((1, 2)),
        abs_delta[..., second:].mean((1, 2)),
        np.abs(np.diff(noisy, axis=-1)).mean((1, 2)),
        np.abs(np.diff(pred, axis=-1)).mean((1, 2)),
        np.abs(np.diff(noisy, axis=1)).mean((1, 2)),
        np.abs(np.diff(pred, axis=1)).mean((1, 2)),
        change.mean((1, 2)) / signal_scale,
        np.sqrt(np.mean(change ** 2, axis=(1, 2))) / signal_scale,
        np.mean(noisy_physical < signal_scale[:, None, None] * .001, axis=(1, 2)),
    ]
    return np.column_stack(values).astype(np.float32)


def apply_calibrator(noisy_log10: np.ndarray, predicted_log10: np.ndarray, regressor) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fitted regressor and return calibrated prediction and weights."""
    noisy = np.asarray(noisy_log10, dtype=np.float32)
    pred = np.asarray(predicted_log10, dtype=np.float32)
    alpha = np.clip(regressor.predict(feature_matrix(noisy, pred)), .35, 1.20).astype(np.float32)
    corrected = noisy + alpha[:, None, None] * (pred - noisy)
    return corrected.astype(np.float32), alpha


def receiver_feature_matrix(noisy_log10: np.ndarray, predicted_log10: np.ndarray) -> np.ndarray:
    """Return one feature row per receiver, with whole-sample context."""
    noisy = np.asarray(noisy_log10, dtype=np.float32)
    pred = np.asarray(predicted_log10, dtype=np.float32)
    if noisy.shape != pred.shape or noisy.ndim != 3 or noisy.shape[1] < 2:
        raise ValueError("Expected matching [samples, >=2 receivers, gates] arrays")
    samples, receivers, gates = noisy.shape
    global_x = np.repeat(feature_matrix(noisy, pred), receivers, axis=0)
    delta = pred - noisy
    abs_delta = np.abs(delta)
    first, second = round(gates / 3), round(2 * gates / 3)
    neighbor_noisy = np.empty_like(noisy)
    neighbor_pred = np.empty_like(pred)
    for receiver in range(receivers):
        adjacent = [i for i in (receiver - 1, receiver + 1) if 0 <= i < receivers]
        neighbor_noisy[:, receiver] = noisy[:, adjacent].mean(axis=1)
        neighbor_pred[:, receiver] = pred[:, adjacent].mean(axis=1)
    noisy_amp = np.power(10.0, noisy.astype(np.float64))
    pred_amp = np.power(10.0, pred.astype(np.float64))
    scale = np.sqrt(np.mean(noisy_amp ** 2, axis=2)) + EPS
    local = [
        noisy.mean(2), noisy.std(2), pred.mean(2), pred.std(2),
        abs_delta.mean(2), abs_delta.std(2),
        np.quantile(abs_delta, .5, axis=2), np.quantile(abs_delta, .9, axis=2),
        abs_delta[..., :first].mean(2), abs_delta[..., second:].mean(2),
        np.abs(np.diff(noisy, axis=2)).mean(2),
        np.abs(np.diff(pred, axis=2)).mean(2),
        np.abs(noisy - neighbor_noisy).mean(2),
        np.abs(pred - neighbor_pred).mean(2),
        np.sqrt(np.mean((noisy_amp - pred_amp) ** 2, axis=2)) / scale,
    ]
    local_x = np.column_stack([v.reshape(-1) for v in local]).astype(np.float32)
    return np.column_stack([global_x, local_x]).astype(np.float32)


def apply_receiver_calibrator(
    noisy_log10: np.ndarray,
    predicted_log10: np.ndarray,
    regressor,
    *,
    early_only: bool = False,
    savgol_window: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-receiver strength; optionally preserve the final 30% of gates."""
    noisy = np.asarray(noisy_log10, dtype=np.float32)
    pred = np.asarray(predicted_log10, dtype=np.float32)
    alpha = np.clip(regressor.predict(receiver_feature_matrix(noisy, pred)), 0.0, 1.2)
    alpha = alpha.astype(np.float32).reshape(noisy.shape[:2])
    receiver_corrected = noisy + alpha[:, :, None] * (pred - noisy)
    if early_only:
        gates = pred.shape[2]
        start, end = int(.65 * gates), int(.70 * gates)
        weight = np.ones(gates, dtype=np.float32)
        weight[start:end] = np.linspace(1, 0, end - start, endpoint=False)
        weight[end:] = 0
        corrected = pred + weight[None, None, :] * (receiver_corrected - pred)
    else:
        corrected = receiver_corrected
    if savgol_window:
        smoothed = _savgol_cubic_numpy(corrected, savgol_window)
        if early_only:
            corrected = corrected + weight[None, None, :] * (smoothed - corrected)
        else:
            corrected = smoothed
    return corrected.astype(np.float32), alpha


def _savgol_cubic_numpy(values: np.ndarray, window: int) -> np.ndarray:
    """Cubic Savitzky--Golay smoothing with interpolated edges, without SciPy.

    The independent NumPy implementation avoids a SciPy/torch BLAS runtime
    conflict observed on Windows when both are imported during inference.
    """
    if window < 5 or window % 2 != 1 or window > values.shape[-1]:
        raise ValueError("window must be odd, at least 5, and no longer than the trace")
    half = window // 2
    positions = np.arange(-half, half + 1, dtype=np.float64)
    s0 = float(window)
    s2 = float(np.sum(positions ** 2))
    s4 = float(np.sum(positions ** 4))
    kernel = (s4 - s2 * positions ** 2) / (s0 * s4 - s2 ** 2)

    source = np.asarray(values, dtype=np.float32)
    flat = source.reshape(-1, source.shape[-1])
    result = np.empty_like(flat)
    fft_length = 1 << (flat.shape[-1] + window - 2).bit_length()
    kernel_fft = np.fft.rfft(kernel, n=fft_length)
    x = np.linspace(-1.0, 1.0, window)
    design = np.column_stack((np.ones(window), x, x * x, x * x * x))
    projection = np.linalg.solve(design.T @ design, design.T)
    left_fit = projection.T @ design[:half].T
    right_fit = projection.T @ design[-half:].T
    for begin in range(0, len(flat), 256):
        traces = flat[begin:begin + 256].astype(np.float64)
        convolution = np.fft.irfft(
            np.fft.rfft(traces, n=fft_length, axis=1) * kernel_fft[None, :],
            n=fft_length,
            axis=1,
        )
        block = convolution[:, half:half + flat.shape[-1]]
        block[:, :half] = traces[:, :window] @ left_fit
        block[:, -half:] = traces[:, -window:] @ right_fit
        result[begin:begin + len(traces)] = block.astype(np.float32)
    return result.reshape(source.shape)
