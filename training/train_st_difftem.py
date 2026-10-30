"""Train ST-DiffTEM diffusion variants on N x 11 x 869 numpy TEM data."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.st_difftem import (
    STDiffTEMNet,
    VALID_VARIANTS,
    FrozenSingleAdapter,
    FrozenSpatialAdapter,
)


EPS = 1e-12
SELECTION_METRICS = ("loss", "rmse", "rrmse", "snr")
DEFAULT_BASELINES_ROOT = Path(__file__).resolve().parents[2] / "训练结果" / "pcbilstm_baselines_suite"


@dataclass
class MetricPack:
    loss: float
    rmse: float
    rrmse: float
    snr: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.loss),
            "rmse": float(self.rmse),
            "rrmse": float(self.rrmse),
            "snr": float(self.snr),
        }


@dataclass
class LossConfig:
    lambda_rec: float = 1.0
    lambda_rel: float = 0.1
    relative_loss_mode: str = "global_energy"
    lambda_gate: float = 0.001
    lambda_residual_x0: float = 0.5
    lambda_anchor_residual: float = 0.5
    lambda_anchor_oracle: float = 0.2
    lambda_pred_oracle: float = 0.1
    lambda_router: float = 0.001
    lambda_router_oracle: float = 0.0
    router_oracle_temperature: float = 0.01
    oracle_margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda_rec": self.lambda_rec,
            "lambda_rel": self.lambda_rel,
            "relative_loss_mode": self.relative_loss_mode,
            "lambda_gate": self.lambda_gate,
            "lambda_residual_x0": self.lambda_residual_x0,
            "lambda_anchor_residual": self.lambda_anchor_residual,
            "lambda_anchor_oracle": self.lambda_anchor_oracle,
            "lambda_pred_oracle": self.lambda_pred_oracle,
            "lambda_router": self.lambda_router,
            "lambda_router_oracle": self.lambda_router_oracle,
            "router_oracle_temperature": self.router_oracle_temperature,
            "oracle_margin": self.oracle_margin,
            # Backward-compatible aliases for old configs/log readers.
            "lambda_baseline_protect": self.lambda_pred_oracle,
            "baseline_protect_margin": self.oracle_margin,
        }


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_workers: int = 4
    max_epochs: int = 140
    min_epochs: int = 50
    patience: int = 30
    warmup_epochs: int = 5
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    selection_metric: str = "rrmse"

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "warmup_epochs": self.warmup_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_accum_steps": self.grad_accum_steps,
            "selection_metric": self.selection_metric,
        }


class TEMResidualDataset(Dataset):
    def __init__(
        self,
        noisy_log10: np.ndarray,
        clean_log10: np.ndarray,
        spatial_base_log10: np.ndarray,
        single_base_log10: np.ndarray | None = None,
    ):
        self.noisy = torch.from_numpy(np.asarray(noisy_log10, dtype=np.float32))
        self.clean = torch.from_numpy(np.asarray(clean_log10, dtype=np.float32))
        self.spatial_base = torch.from_numpy(np.asarray(spatial_base_log10, dtype=np.float32))
        self.single_base = None if single_base_log10 is None else torch.from_numpy(np.asarray(single_base_log10, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def __getitem__(self, idx: int):
        if self.single_base is None:
            return self.noisy[idx], self.clean[idx], self.spatial_base[idx]
        return self.noisy[idx], self.clean[idx], self.spatial_base[idx], self.single_base[idx]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_device_ids(device_ids_arg: str | None, device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    visible_count = int(torch.cuda.device_count())
    if visible_count <= 0:
        raise RuntimeError("CUDA device was selected but no visible GPUs were found.")

    if device_ids_arg is None or str(device_ids_arg).strip() == "":
        return list(range(visible_count))

    raw = str(device_ids_arg).strip().lower()
    if raw == "all":
        return list(range(visible_count))

    device_ids: list[int] = []
    for part in str(device_ids_arg).split(","):
        item = part.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= visible_count:
            raise ValueError(f"Invalid GPU index {idx}; visible gpu count is {visible_count}.")
        if idx not in device_ids:
            device_ids.append(idx)
    if not device_ids:
        raise ValueError("No valid GPU ids were parsed from --device-ids.")
    return device_ids


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_wrap_data_parallel(
    model: nn.Module,
    device: torch.device,
    device_ids: list[int],
) -> tuple[nn.Module, list[int]]:
    if device.type != "cuda":
        return model.to(device), []

    resolved_ids = list(device_ids) if device_ids else [0]
    primary = int(resolved_ids[0])
    torch.cuda.set_device(primary)
    model = model.to(torch.device("cuda", primary))
    if len(resolved_ids) > 1:
        model = nn.DataParallel(model, device_ids=resolved_ids, output_device=primary)
    return model, resolved_ids


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_array(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32)


def load_array_memmap(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def canonicalize_batch_11t(batch: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(batch, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected 3D array (N,R,T), got {arr.shape}")
    if arr.shape[2] == 869:
        return arr
    if arr.shape[1] == 869:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"{name}: unsupported shape {arr.shape}, expected one time axis of length 869")


def to_log10_amplitude(array: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.asarray(array, dtype=np.float32)) + EPS).astype(np.float32)


def from_log10_amplitude_tensor(array: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.tensor(10.0, device=array.device, dtype=array.dtype), array)


def limit_samples(array: np.ndarray, max_samples: int | None, name: str) -> np.ndarray:
    arr = canonicalize_batch_11t(array, name)
    if max_samples is None or max_samples <= 0 or max_samples >= arr.shape[0]:
        return arr
    return arr[:max_samples]


def load_split_arrays(data_path: Path, label_path: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    noisy = limit_samples(load_array(data_path), limit, str(data_path))
    clean = limit_samples(load_array(label_path), limit, str(label_path))
    if noisy.shape != clean.shape:
        raise ValueError(f"Shape mismatch: {noisy.shape} vs {clean.shape}")
    return noisy, clean, tuple(int(v) for v in noisy.shape)


def compute_metrics_batch(
    pred_log10: torch.Tensor,
    target_log10: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = from_log10_amplitude_tensor(pred_log10)
    target = from_log10_amplitude_tensor(target_log10)
    err = pred - target
    mse_sample = torch.mean(err.pow(2), dim=(1, 2))
    rmse_sample = torch.sqrt(mse_sample + EPS)
    rrmse_sample = torch.sqrt(mse_sample / (torch.mean(target.pow(2), dim=(1, 2)) + EPS))
    signal_power = torch.sum(target.pow(2), dim=(1, 2))
    noise_power = torch.sum(err.pow(2), dim=(1, 2))
    snr_sample = 10.0 * torch.log10(signal_power / (noise_power + EPS))
    return rmse_sample, rrmse_sample, snr_sample


def compute_eval_loss(
    pred_log10: torch.Tensor,
    clean_log10: torch.Tensor,
    gate: torch.Tensor,
    loss_config: LossConfig,
) -> torch.Tensor:
    loss_mse = F.mse_loss(pred_log10, clean_log10)
    pred_amp = from_log10_amplitude_tensor(pred_log10)
    clean_amp = from_log10_amplitude_tensor(clean_log10)
    loss_rec = F.l1_loss(pred_amp, clean_amp)
    loss_rel = compute_relative_amplitude_loss(
        pred_amp,
        clean_amp,
        mode=loss_config.relative_loss_mode,
    )
    loss_gate = torch.mean(torch.abs(gate))
    return loss_mse + loss_config.lambda_rec * loss_rec + loss_config.lambda_rel * loss_rel + loss_config.lambda_gate * loss_gate


def compute_relative_amplitude_loss(
    pred_amp: torch.Tensor,
    clean_amp: torch.Tensor,
    mode: str = "global_energy",
) -> torch.Tensor:
    """Return a squared relative-amplitude loss.

    ``global_energy`` averages squared sample-wise RRMSE using the metric's
    global-energy denominator (apart from stabilizer scaling). It avoids
    assigning unbounded weight to late-time gates with near-zero targets.
    ``elementwise`` preserves the loss used by the originally submitted
    manuscript and is retained only for a controlled reviewer ablation.
    """
    squared_error = (pred_amp - clean_amp).pow(2)
    if mode == "global_energy":
        numerator = squared_error.sum(dim=(1, 2))
        denominator = clean_amp.pow(2).sum(dim=(1, 2)) + EPS
        return (numerator / denominator).mean()
    if mode == "elementwise":
        return torch.mean(squared_error / (clean_amp.pow(2) + EPS))
    raise ValueError(f"Unsupported relative loss mode: {mode!r}")


def _mean_or_zero(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return reference.new_tensor(0.0)
    return values.mean()


def compute_total_loss(
    pred_log10: torch.Tensor,
    clean_log10: torch.Tensor,
    aux: dict[str, Any],
    loss_config: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    noise_pred = aux["noise_pred"]
    noise_target = aux["noise_target"]
    gate = aux["gate"]
    residual_target = aux.get("residual_target")
    diffusion_residual_pred = aux.get("diffusion_residual_pred")
    anchor_residual_pred = aux.get("anchor_residual_pred")
    base_log = aux.get("base_log")
    anchor_log = aux.get("anchor_log", base_log)
    spatial_base_log = aux.get("spatial_base_log")
    single_base_log = aux.get("single_base_log")
    w_single = aux.get("w_single")
    residual_scale_factor = aux.get("residual_scale_factor")
    effective_residual_gate = aux.get("effective_residual_gate")
    prediction_type = str(aux.get("prediction_type", "noise"))
    disable_diffusion_residual = bool(
        aux.get(
            "disable_diffusion_residual",
            diffusion_residual_pred is None and anchor_residual_pred is None and noise_pred is None and noise_target is None,
        )
    )

    if disable_diffusion_residual:
        loss_diff = pred_log10.new_tensor(0.0)
    elif prediction_type == "x0":
        if residual_target is None or diffusion_residual_pred is None:
            raise RuntimeError("x0 diffusion loss requires residual_target and diffusion_residual_pred.")
        loss_diff = F.smooth_l1_loss(diffusion_residual_pred, residual_target, beta=0.05)
    else:
        if noise_pred is None or noise_target is None:
            raise RuntimeError("Training loss requires diffusion noise predictions and targets.")
        loss_diff = F.mse_loss(noise_pred, noise_target)

    if disable_diffusion_residual:
        loss_anchor_residual = pred_log10.new_tensor(0.0)
    elif residual_target is not None and anchor_residual_pred is not None:
        loss_anchor_residual = F.smooth_l1_loss(anchor_residual_pred, residual_target, beta=0.05)
    else:
        loss_anchor_residual = pred_log10.new_tensor(0.0)

    pred_sq = (pred_log10 - clean_log10).pow(2)
    spatial_sq = (spatial_base_log - clean_log10).pow(2) if spatial_base_log is not None else pred_sq
    single_sq = (single_base_log - clean_log10).pow(2) if single_base_log is not None else None
    oracle_err = torch.minimum(single_sq, spatial_sq) if single_sq is not None else spatial_sq
    margin = torch.as_tensor(loss_config.oracle_margin, dtype=pred_log10.dtype, device=pred_log10.device)
    loss_anchor_oracle = (
        torch.relu((anchor_log - clean_log10).pow(2) - oracle_err - margin).mean()
        if anchor_log is not None
        else pred_log10.new_tensor(0.0)
    )
    loss_pred_oracle = torch.relu(pred_sq - oracle_err - margin).mean()
    loss_router = w_single.mean() if w_single is not None else pred_log10.new_tensor(0.0)
    if w_single is not None and single_sq is not None:
        rho_aux = aux.get("rho_max", 0.5)
        if torch.is_tensor(rho_aux):
            rho_aux = rho_aux.detach().flatten()[0].item()
        rho_max = max(float(rho_aux), 1e-8)
        temperature = max(float(loss_config.router_oracle_temperature), 1e-8)
        single_receiver_err = single_sq.detach().mean(dim=-1, keepdim=True)
        spatial_receiver_err = spatial_sq.detach().mean(dim=-1, keepdim=True)
        router_target = torch.sigmoid((spatial_receiver_err - single_receiver_err) / temperature)
        w_single_norm = torch.clamp(w_single / rho_max, 0.0, 1.0)
        loss_router_oracle = F.mse_loss(w_single_norm, router_target)
    else:
        loss_router_oracle = pred_log10.new_tensor(0.0)

    eval_like_loss = compute_eval_loss(pred_log10, clean_log10, gate, loss_config)
    residual_term = loss_config.lambda_residual_x0 * loss_diff
    total = (
        eval_like_loss
        + residual_term
        + loss_config.lambda_anchor_residual * loss_anchor_residual
        + loss_config.lambda_anchor_oracle * loss_anchor_oracle
        + loss_config.lambda_pred_oracle * loss_pred_oracle
        + loss_config.lambda_router * loss_router
        + loss_config.lambda_router_oracle * loss_router_oracle
    )

    pred_amp = from_log10_amplitude_tensor(pred_log10)
    clean_amp = from_log10_amplitude_tensor(clean_log10)
    loss_mse = F.mse_loss(pred_log10, clean_log10)
    loss_rec = F.l1_loss(pred_amp, clean_amp)
    loss_rel = compute_relative_amplitude_loss(
        pred_amp,
        clean_amp,
        mode=loss_config.relative_loss_mode,
    )
    loss_gate = torch.mean(torch.abs(gate))
    receiver_gate = aux.get("receiver_gate")
    temporal_gate = aux.get("temporal_gate")
    loss_receiver_gate = torch.mean(torch.abs(receiver_gate)) if receiver_gate is not None else loss_gate
    loss_temporal_gate = torch.mean(torch.abs(temporal_gate)) if temporal_gate is not None else pred_log10.new_tensor(0.0)
    w_single_mean = float(w_single.mean().detach().item()) if w_single is not None else 0.0
    w_single_min = float(w_single.min().detach().item()) if w_single is not None else 0.0
    w_single_max = float(w_single.max().detach().item()) if w_single is not None else 0.0
    residual_scale_mean = (
        float(residual_scale_factor.mean().detach().item()) if residual_scale_factor is not None else 0.0
    )
    residual_scale_min = (
        float(residual_scale_factor.min().detach().item()) if residual_scale_factor is not None else 0.0
    )
    residual_scale_max = (
        float(residual_scale_factor.max().detach().item()) if residual_scale_factor is not None else 0.0
    )
    effective_gate_mean = (
        float(effective_residual_gate.mean().detach().item()) if effective_residual_gate is not None else 0.0
    )
    anchor_mse = float(F.mse_loss(anchor_log, clean_log10).detach().item()) if anchor_log is not None else None
    spatial_base_mse = float(F.mse_loss(spatial_base_log, clean_log10).detach().item()) if spatial_base_log is not None else None
    single_base_mse = float(F.mse_loss(single_base_log, clean_log10).detach().item()) if single_base_log is not None else None
    return total, {
        "loss_diff": float(loss_diff.detach().item()),
        "loss_anchor_residual": float(loss_anchor_residual.detach().item()),
        "loss_anchor_oracle": float(loss_anchor_oracle.detach().item()),
        "loss_pred_oracle": float(loss_pred_oracle.detach().item()),
        "loss_router": float(loss_router.detach().item()),
        "loss_router_oracle": float(loss_router_oracle.detach().item()),
        "loss_mse_log": float(loss_mse.detach().item()),
        "loss_rec": float(loss_rec.detach().item()),
        "loss_rel": float(loss_rel.detach().item()),
        "loss_gate": float(loss_gate.detach().item()),
        "loss_receiver_gate": float(loss_receiver_gate.detach().item()),
        "loss_temporal_gate": float(loss_temporal_gate.detach().item()),
        "w_single_mean": w_single_mean,
        "w_single_min": w_single_min,
        "w_single_max": w_single_max,
        "residual_scale_mean": residual_scale_mean,
        "residual_scale_min": residual_scale_min,
        "residual_scale_max": residual_scale_max,
        "effective_gate_mean": effective_gate_mean,
        "anchor_mse": anchor_mse,
        "spatial_base_mse": spatial_base_mse,
        "single_base_mse": single_base_mse,
    }


def unpack_base_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 3:
        noisy, clean, spatial_base = batch
        return noisy, clean, spatial_base, None
    if len(batch) == 4:
        noisy, clean, spatial_base, single_base = batch
        return noisy, clean, spatial_base, single_base
    raise ValueError(f"Unsupported batch structure of length {len(batch)}")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: LossConfig,
) -> MetricPack:
    model.eval()
    n_samples = 0
    loss_sum = 0.0
    rmse_sum = 0.0
    rrmse_sum = 0.0
    snr_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            noisy, clean, spatial_base, single_base = unpack_base_batch(batch)
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            spatial_base = spatial_base.to(device, non_blocking=True)
            single_base = None if single_base is None else single_base.to(device, non_blocking=True)
            pred, aux = model(
                noisy,
                spatial_base_log=spatial_base,
                single_base_log=single_base,
                return_aux=True,
            )
            batch_size = int(clean.shape[0])
            n_samples += batch_size
            loss_sum += float(compute_eval_loss(pred, clean, aux["gate"], loss_config).item()) * batch_size
            rmse, rrmse, snr = compute_metrics_batch(pred, clean)
            rmse_sum += float(rmse.sum().item())
            rrmse_sum += float(rrmse.sum().item())
            snr_sum += float(snr.sum().item())

    return MetricPack(
        loss=loss_sum / n_samples if n_samples else float("nan"),
        rmse=rmse_sum / n_samples if n_samples else float("nan"),
        rrmse=rrmse_sum / n_samples if n_samples else float("nan"),
        snr=snr_sum / n_samples if n_samples else float("nan"),
    )


def _metric_value(metrics: MetricPack, metric_name: str) -> float:
    if metric_name not in SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric: {metric_name}")
    return float(getattr(metrics, metric_name))


def _is_better_metric(current: MetricPack, best: MetricPack, metric_name: str) -> bool:
    current_value = _metric_value(current, metric_name)
    best_value = _metric_value(best, metric_name)
    if metric_name == "snr":
        return current_value > best_value
    return current_value < best_value


def resolve_artifact_dir(user_path: Path | None, default_name: str) -> Path:
    return user_path if user_path is not None else DEFAULT_BASELINES_ROOT / default_name


def prepare_base_cache(
    data_path: Path,
    cache_path: Path,
    adapter: nn.Module,
    batch_size: int,
    device: torch.device,
    limit: int | None,
) -> tuple[tuple[int, int, int], Path]:
    source = load_array_memmap(data_path)
    if source.ndim != 3:
        raise ValueError(f"{data_path}: expected 3D array, got {source.shape}")
    total_count = int(source.shape[0])
    count = total_count if limit is None or limit <= 0 or limit >= total_count else int(limit)

    sample0 = canonicalize_batch_11t(np.asarray(source[:1]), str(data_path))
    if sample0.shape[2] != 869:
        raise ValueError(f"{data_path}: expected (*,R,869), got {sample0.shape}")

    target_shape = (count, int(sample0.shape[1]), int(sample0.shape[2]))
    ensure_dir(cache_path.parent)
    needs_rebuild = True
    if cache_path.exists():
        try:
            existing = np.load(cache_path, mmap_mode="r")
            needs_rebuild = tuple(int(v) for v in existing.shape) != target_shape
            del existing
        except Exception:
            needs_rebuild = True
    if not needs_rebuild:
        return target_shape, cache_path

    adapter = adapter.to(device)
    adapter.eval()
    partial_path = cache_path.with_name(cache_path.name + ".partial")
    cache = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=target_shape)
    with torch.no_grad():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            batch = canonicalize_batch_11t(np.asarray(source[start:end]), f"{data_path}[{start}:{end}]")
            noisy_log = to_log10_amplitude(batch)
            noisy_tensor = torch.from_numpy(noisy_log).to(device)
            pred = adapter(noisy_tensor).detach().cpu().numpy().astype(np.float32)
            cache[start:end] = pred
    cache.flush()
    del cache
    os.replace(partial_path, cache_path)
    return target_shape, cache_path


def prepare_variant_caches(
    variant: str,
    split_name: str,
    data_path: Path,
    cache_dir: Path,
    spatial_adapter: FrozenSpatialAdapter,
    single_adapter: FrozenSingleAdapter | None,
    batch_size: int,
    device: torch.device,
    limit: int | None,
) -> dict[str, dict[str, Any] | None]:
    manifest: dict[str, dict[str, Any] | None] = {"spatial": None, "single": None}
    spatial_shape, spatial_path = prepare_base_cache(
        data_path=data_path,
        cache_path=cache_dir / f"{split_name}_spatial_base_log10.npy",
        adapter=spatial_adapter,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )
    manifest["spatial"] = {"shape": list(spatial_shape), "path": str(spatial_path.resolve())}

    if variant == "hybrid_base":
        if single_adapter is None:
            raise RuntimeError("Hybrid-base cache preparation requires single adapter.")
        single_shape, single_path = prepare_base_cache(
            data_path=data_path,
            cache_path=cache_dir / f"{split_name}_single_base_log10.npy",
            adapter=single_adapter,
            batch_size=batch_size,
            device=device,
            limit=limit,
        )
        manifest["single"] = {"shape": list(single_shape), "path": str(single_path.resolve())}
    return manifest


def load_test_scenario_specs(data_root: Path) -> list[dict[str, Any]]:
    manifest_path = data_root / "test_scenario_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return list(payload.get("scenarios", []))

    scenario_root = data_root / "test_scenarios"
    if not scenario_root.exists():
        return []
    specs: list[dict[str, Any]] = []
    for scenario_dir in sorted(scenario_root.iterdir()):
        if scenario_dir.is_dir():
            specs.append({"slug": scenario_dir.name, "name": scenario_dir.name.replace("_", " ")})
    return specs


def infer_data_root(test_data_path: Path | None) -> Path | None:
    if test_data_path is None:
        return None
    resolved = test_data_path.resolve()
    if resolved.parent.name != "test":
        return None
    return resolved.parent.parent


def prepare_test_scenario_caches(
    variant: str,
    data_root: Path | None,
    cache_dir: Path,
    spatial_adapter: FrozenSpatialAdapter,
    single_adapter: FrozenSingleAdapter | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, dict[str, Any] | None]]:
    if data_root is None:
        return {}

    scenario_root = data_root / "test_scenarios"
    if not scenario_root.exists():
        return {}

    scenario_manifest: dict[str, dict[str, dict[str, Any] | None]] = {}
    for spec in load_test_scenario_specs(data_root):
        slug = str(spec.get("slug", "")).strip()
        if not slug:
            continue
        data_path = scenario_root / slug / f"{slug}_data.npy"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing scenario data file for cache preparation: {data_path}")
        scenario_manifest[slug] = prepare_variant_caches(
            variant=variant,
            split_name=slug,
            data_path=data_path,
            cache_dir=cache_dir,
            spatial_adapter=spatial_adapter,
            single_adapter=single_adapter,
            batch_size=batch_size,
            device=device,
            limit=None,
        )
    return scenario_manifest


def load_recipe_rows(recipe_path: Path, expected_count: int, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in recipe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    if len(rows) != expected_count:
        raise ValueError(f"{recipe_path}: expected {expected_count} rows, got {len(rows)}")
    return rows


def build_challenging_indices(recipe_path: Path, expected_count: int, limit: int | None) -> list[int]:
    if not recipe_path.exists():
        return []
    rows = load_recipe_rows(recipe_path, expected_count, limit)
    indices: list[int] = []
    for idx, row in enumerate(rows):
        actual_snr = float(row.get("actual_snr_db", float("inf")))
        glitch_mode = row.get("glitch_mode")
        recipe_len = int(row.get("recipe_len", 0))
        if actual_snr <= 14.0 or glitch_mode in {"dropout", "bad_channel"} or recipe_len >= 4:
            indices.append(idx)
    return indices


def build_dataset(
    noisy: np.ndarray,
    clean: np.ndarray,
    spatial_cache_path: Path,
    single_cache_path: Path | None = None,
) -> TEMResidualDataset:
    spatial_base = np.asarray(np.load(spatial_cache_path), dtype=np.float32)
    if noisy.shape != clean.shape or noisy.shape != spatial_base.shape:
        raise ValueError(
            f"Dataset/base shape mismatch: noisy={noisy.shape}, clean={clean.shape}, spatial={spatial_base.shape}"
        )
    single_base = None
    if single_cache_path is not None:
        single_base = np.asarray(np.load(single_cache_path), dtype=np.float32)
        if single_base.shape != noisy.shape:
            raise ValueError(f"Single-base shape mismatch: single={single_base.shape}, noisy={noisy.shape}")
    return TEMResidualDataset(
        noisy_log10=to_log10_amplitude(noisy),
        clean_log10=to_log10_amplitude(clean),
        spatial_base_log10=spatial_base,
        single_base_log10=single_base,
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    challenging_val_loader: DataLoader | None,
    device: torch.device,
    output_dir: Path,
    config: TrainConfig,
    loss_config: LossConfig,
) -> tuple[MetricPack, MetricPack | None, list[dict[str, Any]], Path]:
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    warmup_epochs = max(0, min(config.warmup_epochs, max(config.max_epochs - 1, 0)))
    if warmup_epochs > 0:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, config.max_epochs - warmup_epochs),
                    eta_min=config.lr * 0.05,
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.max_epochs),
            eta_min=config.lr * 0.05,
        )

    best_metrics: MetricPack | None = None
    best_challenging: MetricPack | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    ensure_dir(output_dir)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss_total = 0.0
        train_diff_total = 0.0
        train_mse_total = 0.0
        train_rec_total = 0.0
        train_rel_total = 0.0
        train_gate_total = 0.0
        train_anchor_total = 0.0
        train_anchor_oracle_total = 0.0
        train_pred_oracle_total = 0.0
        train_router_total = 0.0
        train_router_oracle_total = 0.0
        train_receiver_gate_total = 0.0
        train_temporal_gate_total = 0.0
        train_w_single_mean_total = 0.0
        train_w_single_min_total = 0.0
        train_w_single_max_total = 0.0
        train_residual_scale_mean_total = 0.0
        train_residual_scale_min_total = 0.0
        train_residual_scale_max_total = 0.0
        train_effective_gate_mean_total = 0.0
        train_anchor_mse_total = 0.0
        train_spatial_base_mse_total = 0.0
        train_single_base_mse_total = 0.0
        train_single_base_mse_count = 0

        for step_idx, batch in enumerate(train_loader):
            noisy, clean, spatial_base, single_base = unpack_base_batch(batch)
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            spatial_base = spatial_base.to(device, non_blocking=True)
            single_base = None if single_base is None else single_base.to(device, non_blocking=True)
            pred, aux = model(
                noisy,
                clean_log=clean,
                spatial_base_log=spatial_base,
                single_base_log=single_base,
                return_aux=True,
            )
            total_loss, parts = compute_total_loss(pred, clean, aux, loss_config)
            loss = total_loss / config.grad_accum_steps
            loss.backward()

            if (step_idx + 1) % config.grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss_total += float(loss.item() * config.grad_accum_steps)
            train_diff_total += parts["loss_diff"]
            train_mse_total += parts["loss_mse_log"]
            train_rec_total += parts["loss_rec"]
            train_rel_total += parts["loss_rel"]
            train_gate_total += parts["loss_gate"]
            train_anchor_total += parts.get("loss_anchor_residual", 0.0)
            train_anchor_oracle_total += parts.get("loss_anchor_oracle", 0.0)
            train_pred_oracle_total += parts.get("loss_pred_oracle", 0.0)
            train_router_total += parts.get("loss_router", 0.0)
            train_router_oracle_total += parts.get("loss_router_oracle", 0.0)
            train_receiver_gate_total += parts.get("loss_receiver_gate", parts["loss_gate"])
            train_temporal_gate_total += parts.get("loss_temporal_gate", 0.0)
            train_w_single_mean_total += parts.get("w_single_mean", 0.0)
            train_w_single_min_total += parts.get("w_single_min", 0.0)
            train_w_single_max_total += parts.get("w_single_max", 0.0)
            train_residual_scale_mean_total += parts.get("residual_scale_mean", 0.0)
            train_residual_scale_min_total += parts.get("residual_scale_min", 0.0)
            train_residual_scale_max_total += parts.get("residual_scale_max", 0.0)
            train_effective_gate_mean_total += parts.get("effective_gate_mean", 0.0)
            train_anchor_mse_total += parts.get("anchor_mse", 0.0) or 0.0
            train_spatial_base_mse_total += parts.get("spatial_base_mse", 0.0) or 0.0
            if parts.get("single_base_mse") is not None:
                train_single_base_mse_total += float(parts["single_base_mse"])
                train_single_base_mse_count += 1

        scheduler.step()
        train_loss = train_loss_total / max(len(train_loader), 1)
        train_diff = train_diff_total / max(len(train_loader), 1)
        train_mse = train_mse_total / max(len(train_loader), 1)
        train_rec = train_rec_total / max(len(train_loader), 1)
        train_rel = train_rel_total / max(len(train_loader), 1)
        train_gate = train_gate_total / max(len(train_loader), 1)
        train_anchor = train_anchor_total / max(len(train_loader), 1)
        train_anchor_oracle = train_anchor_oracle_total / max(len(train_loader), 1)
        train_pred_oracle = train_pred_oracle_total / max(len(train_loader), 1)
        train_router = train_router_total / max(len(train_loader), 1)
        train_router_oracle = train_router_oracle_total / max(len(train_loader), 1)
        train_receiver_gate = train_receiver_gate_total / max(len(train_loader), 1)
        train_temporal_gate = train_temporal_gate_total / max(len(train_loader), 1)
        train_w_single_mean = train_w_single_mean_total / max(len(train_loader), 1)
        train_w_single_min = train_w_single_min_total / max(len(train_loader), 1)
        train_w_single_max = train_w_single_max_total / max(len(train_loader), 1)
        train_residual_scale_mean = train_residual_scale_mean_total / max(len(train_loader), 1)
        train_residual_scale_min = train_residual_scale_min_total / max(len(train_loader), 1)
        train_residual_scale_max = train_residual_scale_max_total / max(len(train_loader), 1)
        train_effective_gate_mean = train_effective_gate_mean_total / max(len(train_loader), 1)
        train_anchor_mse = train_anchor_mse_total / max(len(train_loader), 1)
        train_spatial_base_mse = train_spatial_base_mse_total / max(len(train_loader), 1)
        train_single_base_mse = (
            train_single_base_mse_total / train_single_base_mse_count
            if train_single_base_mse_count > 0
            else None
        )
        val_metrics = evaluate_model(model, val_loader, device, loss_config)
        challenging_metrics = evaluate_model(model, challenging_val_loader, device, loss_config) if challenging_val_loader is not None else None
        lr_now = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_diff": train_diff,
            "train_loss_mse_log": train_mse,
            "train_loss_rec": train_rec,
            "train_loss_rel": train_rel,
            "train_loss_gate": train_gate,
            "train_loss_anchor_residual": train_anchor,
            "train_loss_anchor_oracle": train_anchor_oracle,
            "train_loss_pred_oracle": train_pred_oracle,
            "train_loss_router": train_router,
            "train_loss_router_oracle": train_router_oracle,
            "train_receiver_gate": train_receiver_gate,
            "train_temporal_gate": train_temporal_gate,
            "w_single_mean": train_w_single_mean,
            "w_single_min": train_w_single_min,
            "w_single_max": train_w_single_max,
            "residual_scale_mean": train_residual_scale_mean,
            "residual_scale_min": train_residual_scale_min,
            "residual_scale_max": train_residual_scale_max,
            "effective_gate_mean": train_effective_gate_mean,
            "anchor_mse": train_anchor_mse,
            "spatial_base_mse": train_spatial_base_mse,
            "single_base_mse": train_single_base_mse,
            "val_loss": val_metrics.loss,
            "val_rmse": val_metrics.rmse,
            "val_rrmse": val_metrics.rrmse,
            "val_snr": val_metrics.snr,
            "lr": lr_now,
        }
        if challenging_metrics is not None:
            history_row.update(
                {
                    "challenging_val_loss": challenging_metrics.loss,
                    "challenging_val_rmse": challenging_metrics.rmse,
                    "challenging_val_rrmse": challenging_metrics.rrmse,
                    "challenging_val_snr": challenging_metrics.snr,
                }
            )
        history.append(history_row)
        print(
            f"[st_difftem:{unwrap_model(model).variant} epoch {epoch:03d}] "
            f"train={train_loss:.6f} diff={train_diff:.6f} mse={train_mse:.6f} "
            f"rec={train_rec:.6f} rel={train_rel:.6f} gate={train_gate:.6f} "
            f"anchor={train_anchor:.6f} anchor_oracle={train_anchor_oracle:.6f} "
            f"pred_oracle={train_pred_oracle:.6f} router={train_router:.6f} "
            f"router_oracle={train_router_oracle:.6f} | "
            f"val={val_metrics.loss:.6f} rmse={val_metrics.rmse:.6f} "
            f"rrmse={val_metrics.rrmse:.6f} snr={val_metrics.snr:.3f} | "
            f"lr={lr_now:.2e}"
        )
        single_base_mse_text = "na" if train_single_base_mse is None else f"{train_single_base_mse:.6f}"
        print(
            f"[st_difftem:{unwrap_model(model).variant} anchor epoch {epoch:03d}] "
            f"w_single_mean={train_w_single_mean:.6f} w_single_min={train_w_single_min:.6f} "
            f"w_single_max={train_w_single_max:.6f} residual_scale_mean={train_residual_scale_mean:.6f} "
            f"residual_scale_min={train_residual_scale_min:.6f} residual_scale_max={train_residual_scale_max:.6f} "
            f"effective_gate_mean={train_effective_gate_mean:.6f} anchor_mse={train_anchor_mse:.6f} "
            f"spatial_base_mse={train_spatial_base_mse:.6f} single_base_mse={single_base_mse_text}"
        )
        if challenging_metrics is not None:
            print(
                f"[st_difftem:{unwrap_model(model).variant} challenging_val epoch {epoch:03d}] "
                f"loss={challenging_metrics.loss:.6f} rmse={challenging_metrics.rmse:.6f} "
                f"rrmse={challenging_metrics.rrmse:.6f} snr={challenging_metrics.snr:.3f}"
            )

        selection_value = _metric_value(val_metrics, config.selection_metric)
        if best_metrics is None or _is_better_metric(val_metrics, best_metrics, config.selection_metric):
            best_metrics = val_metrics
            best_challenging = challenging_metrics
            best_state = {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
            print(
                f"[st_difftem:{unwrap_model(model).variant}] new best checkpoint by {config.selection_metric}="
                f"{selection_value:.6f}."
            )
        else:
            stale_epochs += 1

        if epoch >= config.min_epochs and stale_epochs >= config.patience:
            print(f"[st_difftem:{unwrap_model(model).variant}] early stopping triggered.")
            break

    if best_metrics is None or best_state is None:
        raise RuntimeError("Training produced no checkpoint.")

    unwrap_model(model).load_state_dict(best_state)
    checkpoint_path = output_dir / "best_model.pth"
    torch.save(
        {
            "state_dict": best_state,
            "best_metrics": best_metrics.to_dict(),
            "best_challenging_metrics": best_challenging.to_dict() if best_challenging is not None else None,
            "best_epoch": best_epoch,
            "history": history,
            "config": config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "variant": unwrap_model(model).variant,
        },
        checkpoint_path,
    )
    save_json(
        output_dir / "training_summary.json",
        {
            "variant": unwrap_model(model).variant,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics.to_dict(),
            "best_challenging_metrics": best_challenging.to_dict() if best_challenging is not None else None,
            "config": config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "history": history,
            "checkpoint": str(checkpoint_path.resolve()),
        },
    )
    return best_metrics, best_challenging, history, checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="spatial_base")
    parser.add_argument("--spatial-artifact-dir", type=Path)
    parser.add_argument("--single-artifact-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=140)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="rrmse")
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-rel", type=float, default=0.1)
    parser.add_argument(
        "--relative-loss-mode",
        choices=("global_energy", "elementwise"),
        default="global_energy",
        help=(
            "Relative-amplitude loss normalization. global_energy uses the "
            "metric's sample-wise energy basis but averages squared sample RRMSE; "
            "elementwise reproduces the "
            "original submission for ablation."
        ),
    )
    parser.add_argument("--lambda-gate", type=float, default=0.001)
    parser.add_argument("--lambda-residual-x0", type=float, default=0.5)
    parser.add_argument("--lambda-anchor-residual", type=float, default=0.5)
    parser.add_argument("--lambda-anchor-oracle", type=float, default=0.2)
    parser.add_argument("--lambda-pred-oracle", "--lambda-baseline-protect", dest="lambda_pred_oracle", type=float, default=0.1)
    parser.add_argument("--lambda-router", type=float, default=0.001)
    parser.add_argument("--lambda-router-oracle", type=float, default=0.0)
    parser.add_argument("--router-oracle-temperature", type=float, default=0.01)
    parser.add_argument("--oracle-margin", "--baseline-protect-margin", dest="oracle_margin", type=float, default=0.0)
    parser.add_argument("--condition-feature-dim", type=int, default=64)
    parser.add_argument("--condition-hidden-channels", type=int, default=32)
    parser.add_argument("--condition-spatial-layers", type=int, default=1)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-hidden-channels", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--step-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=869)
    parser.add_argument("--num-receivers", type=int, default=11)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--refine-scale", type=float, default=1.0)
    parser.add_argument("--sampler-seed", type=int, default=20260430)
    parser.add_argument("--prediction-type", choices=("noise", "x0"), default="x0")
    parser.add_argument("--residual-inference-mode", choices=("ddim_random", "anchor"), default="anchor")
    parser.add_argument("--residual-anchor-mode", choices=("zero", "observed"), default="zero")
    parser.add_argument("--residual-clip-value", type=float, default=0.5)
    parser.add_argument("--max-residual-gate", type=float, default=0.5)
    parser.add_argument("--rho-max", type=float, default=0.5)
    parser.add_argument("--residual-scale-by-w-single", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual-scale-floor", type=float, default=0.5)
    parser.add_argument("--disable-temporal-gate", action="store_true")
    parser.add_argument("--temporal-gate-init-bias", type=float, default=0.0)
    parser.add_argument("--disable-diffusion-residual", action="store_true")
    parser.add_argument("--direct-residual-nodiff", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--cache-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device-ids", type=str, help="Comma-separated CUDA device ids, e.g. '0,1'.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--prepare-cache-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    device_ids = parse_device_ids(args.device_ids, device)
    if device.type == "cuda" and device_ids:
        device = torch.device("cuda", int(device_ids[0]))

    output_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(output_dir / "cache")
    spatial_dir = resolve_artifact_dir(args.spatial_artifact_dir, "spatial_receiver_pcbilstm")
    single_dir = resolve_artifact_dir(args.single_artifact_dir, "single_receiver_pcbilstm")

    train_noisy, train_clean, train_shape = load_split_arrays(args.train_data, args.train_labels, args.train_limit)
    val_noisy, val_clean, val_shape = load_split_arrays(args.val_data, args.val_labels, args.val_limit)
    test_noisy = test_clean = None
    test_shape: tuple[int, int, int] | None = None
    if args.test_data and args.test_labels:
        test_noisy, test_clean, test_shape = load_split_arrays(args.test_data, args.test_labels, args.test_limit)

    for shape_name, shape in (("train", train_shape), ("val", val_shape), ("test", test_shape)):
        if shape is None:
            continue
        if shape[1] != args.num_receivers:
            raise ValueError(f"{shape_name}: expected {args.num_receivers} receivers, got {shape[1]}")
        if shape[2] != args.seq_len:
            raise ValueError(f"{shape_name}: expected seq_len {args.seq_len}, got {shape[2]}")

    spatial_adapter, spatial_run_config = FrozenSpatialAdapter.from_artifact_dir(spatial_dir)
    single_adapter = None
    single_run_config: dict[str, Any] | None = None
    if args.variant == "hybrid_base":
        single_adapter, single_run_config = FrozenSingleAdapter.from_artifact_dir(single_dir)

    cache_manifest = {
        "variant": args.variant,
        "train": prepare_variant_caches(
            variant=args.variant,
            split_name="train",
            data_path=args.train_data,
            cache_dir=cache_dir,
            spatial_adapter=spatial_adapter,
            single_adapter=single_adapter,
            batch_size=args.cache_batch_size,
            device=device,
            limit=args.train_limit,
        ),
        "val": prepare_variant_caches(
            variant=args.variant,
            split_name="val",
            data_path=args.val_data,
            cache_dir=cache_dir,
            spatial_adapter=spatial_adapter,
            single_adapter=single_adapter,
            batch_size=args.cache_batch_size,
            device=device,
            limit=args.val_limit,
        ),
        "test": None,
    }
    if args.test_data and args.test_labels:
        cache_manifest["test"] = prepare_variant_caches(
            variant=args.variant,
            split_name="test",
            data_path=args.test_data,
            cache_dir=cache_dir,
            spatial_adapter=spatial_adapter,
            single_adapter=single_adapter,
            batch_size=args.cache_batch_size,
            device=device,
            limit=args.test_limit,
        )
    cache_manifest["scenarios"] = prepare_test_scenario_caches(
        variant=args.variant,
        data_root=infer_data_root(args.test_data),
        cache_dir=cache_dir,
        spatial_adapter=spatial_adapter,
        single_adapter=single_adapter,
        batch_size=args.cache_batch_size,
        device=device,
    )

    save_json(cache_dir / "cache_manifest.json", cache_manifest)
    if args.prepare_cache_only:
        print(f"Prepared {args.variant} base cache under: {cache_dir}")
        return

    train_dataset = build_dataset(
        noisy=train_noisy,
        clean=train_clean,
        spatial_cache_path=Path(cache_manifest["train"]["spatial"]["path"]),
        single_cache_path=Path(cache_manifest["train"]["single"]["path"]) if cache_manifest["train"]["single"] else None,
    )
    val_dataset = build_dataset(
        noisy=val_noisy,
        clean=val_clean,
        spatial_cache_path=Path(cache_manifest["val"]["spatial"]["path"]),
        single_cache_path=Path(cache_manifest["val"]["single"]["path"]) if cache_manifest["val"]["single"] else None,
    )
    challenging_indices = build_challenging_indices(
        args.val_data.resolve().parent / "val_recipes.jsonl",
        expected_count=val_shape[0],
        limit=args.val_limit,
    )
    challenging_val_loader = None
    challenging_subset = Subset(val_dataset, challenging_indices) if challenging_indices else None

    pin_memory = device.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    if challenging_subset is not None and len(challenging_subset) > 0:
        challenging_val_loader = DataLoader(challenging_subset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    spatial_model_config = dict(spatial_run_config.get("model", {}))
    single_model_config = dict((single_run_config or {}).get("model", {}))
    model = STDiffTEMNet(
        seq_len=args.seq_len,
        num_receivers=args.num_receivers,
        variant=args.variant,
        condition_feature_dim=args.condition_feature_dim,
        condition_hidden_channels=args.condition_hidden_channels,
        condition_spatial_layers=args.condition_spatial_layers,
        gate_hidden_dim=args.gate_hidden_dim,
        diffusion_hidden_channels=args.diffusion_hidden_channels,
        diffusion_steps=args.diffusion_steps,
        sample_steps=args.sample_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        step_dim=args.step_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        refine_scale=args.refine_scale,
        sampler_seed=args.sampler_seed,
        prediction_type=args.prediction_type,
        residual_inference_mode=args.residual_inference_mode,
        residual_anchor_mode=args.residual_anchor_mode,
        residual_clip_value=args.residual_clip_value,
        max_residual_gate=args.max_residual_gate,
        rho_max=args.rho_max,
        residual_scale_by_w_single=args.residual_scale_by_w_single,
        residual_scale_floor=args.residual_scale_floor,
        use_temporal_gate=not args.disable_temporal_gate,
        temporal_gate_init_bias=args.temporal_gate_init_bias,
        disable_diffusion_residual=args.disable_diffusion_residual,
        direct_residual_nodiff=args.direct_residual_nodiff,
        spatial_artifact_dir=str(spatial_dir),
        spatial_model_config=spatial_model_config,
        single_artifact_dir=str(single_dir) if args.variant == "hybrid_base" else None,
        single_model_config=single_model_config if args.variant == "hybrid_base" else None,
    ).to(device)
    model, device_ids = maybe_wrap_data_parallel(model, device, device_ids)

    loss_config = LossConfig(
        lambda_rec=args.lambda_rec,
        lambda_rel=args.lambda_rel,
        relative_loss_mode=args.relative_loss_mode,
        lambda_gate=args.lambda_gate,
        lambda_residual_x0=args.lambda_residual_x0,
        lambda_anchor_residual=args.lambda_anchor_residual,
        lambda_anchor_oracle=args.lambda_anchor_oracle,
        lambda_pred_oracle=args.lambda_pred_oracle,
        lambda_router=args.lambda_router,
        lambda_router_oracle=args.lambda_router_oracle,
        router_oracle_temperature=args.router_oracle_temperature,
        oracle_margin=args.oracle_margin,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        selection_metric=args.selection_metric,
    )

    save_json(
        output_dir / "run_config.json",
        {
            "variant": args.variant,
            "train_data": str(args.train_data.resolve()),
            "train_labels": str(args.train_labels.resolve()),
            "val_data": str(args.val_data.resolve()),
            "val_labels": str(args.val_labels.resolve()),
            "test_data": str(args.test_data.resolve()) if args.test_data else None,
            "test_labels": str(args.test_labels.resolve()) if args.test_labels else None,
            "device": str(device),
            "device_ids": device_ids,
            "data_parallel": bool(len(device_ids) > 1),
            "seed": args.seed,
            "model": unwrap_model(model).get_config(),
            "train_config": train_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "cache": cache_manifest,
            "challenging_val": {
                "recipe_path": str((args.val_data.resolve().parent / "val_recipes.jsonl")),
                "sample_count": len(challenging_indices),
                "indices_path": str((output_dir / "challenging_val_indices.json").resolve()),
                "rules": {
                    "actual_snr_db_le": 14.0,
                    "glitch_mode_in": ["dropout", "bad_channel"],
                    "recipe_len_ge": 4,
                },
            },
            "dataset_shapes": {
                "train": train_shape,
                "val": val_shape,
                "test": test_shape,
            },
            "spatial_base": {
                "artifact_dir": str(spatial_dir.resolve()),
                "run_config_path": str((spatial_dir / "run_config.json").resolve()),
                "checkpoint_path": str((spatial_dir / "best_model.pth").resolve()),
                "model": spatial_model_config,
            },
            "single_base": {
                "artifact_dir": str(single_dir.resolve()) if args.variant == "hybrid_base" else None,
                "run_config_path": str((single_dir / "run_config.json").resolve()) if args.variant == "hybrid_base" else None,
                "checkpoint_path": str((single_dir / "best_model.pth").resolve()) if args.variant == "hybrid_base" else None,
                "model": single_model_config if args.variant == "hybrid_base" else None,
            },
        },
    )
    save_json(output_dir / "challenging_val_indices.json", challenging_indices)

    best_metrics, best_challenging, _, checkpoint_path = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        challenging_val_loader=challenging_val_loader,
        device=device,
        output_dir=output_dir,
        config=train_config,
        loss_config=loss_config,
    )
    if device.type == "cuda":
        print(f"Using CUDA device ids: {device_ids if device_ids else [0]}")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Best validation metrics: {best_metrics.to_dict()}")
    if best_challenging is not None:
        print(f"Best challenging validation metrics: {best_challenging.to_dict()}")

    if test_noisy is not None and test_clean is not None and cache_manifest["test"] is not None:
        test_dataset = build_dataset(
            noisy=test_noisy,
            clean=test_clean,
            spatial_cache_path=Path(cache_manifest["test"]["spatial"]["path"]),
            single_cache_path=Path(cache_manifest["test"]["single"]["path"]) if cache_manifest["test"]["single"] else None,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        test_metrics = evaluate_model(model, test_loader, device, loss_config)
        save_json(
            output_dir / "test_metrics.json",
            {
                "variant": args.variant,
                "test_shape": test_shape,
                "metrics": test_metrics.to_dict(),
            },
        )
        print(f"Test metrics: {test_metrics.to_dict()}")


if __name__ == "__main__":
    main()
