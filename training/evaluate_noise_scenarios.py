"""Evaluate PC-BiLSTM checkpoints on fixed noise scenarios.

This script reads the paper dataset layout:
- data_root/test/test_data.npy
- data_root/test/test_labels.npy
- data_root/test_scenarios/<slug>/<slug>_data.npy
- data_root/test_scenarios/<slug>/<slug>_labels.npy
- data_root/test_scenario_manifest.json

Outputs:
- scenario_metrics.json
- scenario_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.multi_receiver_pcbilstm import MultiReceiverPCBiLSTM
from models.single_receiver_pcbilstm import IndependentReceiverPCBiLSTM
from models.spatial_receiver_pcbilstm import SpatialReceiverPCBiLSTM
from models.st_difftem import STDiffTEMNet

EPS = 1e-12


@dataclass
class MetricPack:
    loss: float
    rmse: float
    rrmse: float
    snr: float

    def to_dict(self) -> dict[str, float]:
        return {
            'loss': float(self.loss),
            'rmse': float(self.rmse),
            'rrmse': float(self.rrmse),
            'snr': float(self.snr),
        }


class TEMDataset(Dataset):
    def __init__(
        self,
        noisy_log10: np.ndarray,
        clean_log10: np.ndarray,
        spatial_base_log10: np.ndarray | None = None,
        single_base_log10: np.ndarray | None = None,
    ):
        self.noisy = torch.from_numpy(np.asarray(noisy_log10, dtype=np.float32))
        self.clean = torch.from_numpy(np.asarray(clean_log10, dtype=np.float32))
        self.spatial_base = None if spatial_base_log10 is None else torch.from_numpy(np.asarray(spatial_base_log10, dtype=np.float32))
        self.single_base = None if single_base_log10 is None else torch.from_numpy(np.asarray(single_base_log10, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def __getitem__(self, idx: int):
        if self.spatial_base is None:
            return self.noisy[idx], self.clean[idx]
        if self.single_base is None:
            return self.noisy[idx], self.clean[idx], self.spatial_base[idx]
        return self.noisy[idx], self.clean[idx], self.spatial_base[idx], self.single_base[idx]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_device_ids(device_ids_arg: str | None, device: torch.device) -> list[int]:
    if device.type != 'cuda':
        return []
    visible_count = int(torch.cuda.device_count())
    if visible_count <= 0:
        raise RuntimeError('CUDA device was selected but no visible GPUs were found.')

    if device_ids_arg is None or str(device_ids_arg).strip() == '':
        return list(range(visible_count))

    raw = str(device_ids_arg).strip().lower()
    if raw == 'all':
        return list(range(visible_count))

    device_ids: list[int] = []
    for part in str(device_ids_arg).split(','):
        item = part.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= visible_count:
            raise ValueError(f'Invalid GPU index {idx}; visible gpu count is {visible_count}.')
        if idx not in device_ids:
            device_ids.append(idx)
    if not device_ids:
        raise ValueError('No valid GPU ids were parsed from --device-ids.')
    return device_ids


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_wrap_data_parallel(
    model: nn.Module,
    device: torch.device,
    device_ids: list[int],
) -> tuple[nn.Module, list[int]]:
    if device.type != 'cuda':
        return model.to(device), []

    resolved_ids = list(device_ids) if device_ids else [0]
    primary = int(resolved_ids[0])
    torch.cuda.set_device(primary)
    model = model.to(torch.device('cuda', primary))
    if len(resolved_ids) > 1:
        model = nn.DataParallel(model, device_ids=resolved_ids, output_device=primary)
    return model, resolved_ids


def canonicalize_11t(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f'{name}: expected 3D array (N,11,T), got {arr.shape}')
    if arr.shape[1] == 11:
        return arr
    if arr.shape[2] == 11:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f'{name}: unsupported shape {arr.shape}, expected one axis to be 11')


def limit_samples(array: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or max_samples >= array.shape[0]:
        return array
    return array[:max_samples]


def to_log10_amplitude(array: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.asarray(array, dtype=np.float32)) + EPS).astype(np.float32)


def build_dataset(
    data_path: Path,
    label_path: Path,
    limit: int | None,
    spatial_base_cache_path: Path | None = None,
    single_base_cache_path: Path | None = None,
) -> tuple[TEMDataset, tuple[int, int, int]]:
    noisy = limit_samples(canonicalize_11t(np.load(data_path), str(data_path)), limit)
    clean = limit_samples(canonicalize_11t(np.load(label_path), str(label_path)), limit)
    if noisy.shape != clean.shape:
        raise ValueError(f'Shape mismatch: {noisy.shape} vs {clean.shape}')
    spatial_base_log10 = None
    if spatial_base_cache_path is not None and spatial_base_cache_path.exists():
        spatial_base_log10 = limit_samples(canonicalize_11t(np.load(spatial_base_cache_path), str(spatial_base_cache_path)), limit)
        if spatial_base_log10.shape != noisy.shape:
            raise ValueError(f'Spatial base cache shape mismatch: {spatial_base_log10.shape} vs {noisy.shape}')
    single_base_log10 = None
    if single_base_cache_path is not None and single_base_cache_path.exists():
        single_base_log10 = limit_samples(canonicalize_11t(np.load(single_base_cache_path), str(single_base_cache_path)), limit)
        if single_base_log10.shape != noisy.shape:
            raise ValueError(f'Single base cache shape mismatch: {single_base_log10.shape} vs {noisy.shape}')
    dataset = TEMDataset(
        to_log10_amplitude(noisy),
        to_log10_amplitude(clean),
        spatial_base_log10=spatial_base_log10,
        single_base_log10=single_base_log10,
    )
    return dataset, tuple(int(v) for v in noisy.shape)


def compute_metrics_batch(pred_log10: torch.Tensor, target_log10: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = torch.pow(10.0, pred_log10)
    target = torch.pow(10.0, target_log10)
    err = pred - target
    mse_sample = torch.mean(err.pow(2), dim=(1, 2))
    rmse_sample = torch.sqrt(mse_sample + EPS)
    rrmse_sample = torch.sqrt(mse_sample / (torch.mean(target.pow(2), dim=(1, 2)) + EPS))
    signal_power = torch.sum(target.pow(2), dim=(1, 2))
    noise_power = torch.sum(err.pow(2), dim=(1, 2))
    snr_sample = 10.0 * torch.log10(signal_power / (noise_power + EPS))
    return rmse_sample, rrmse_sample, snr_sample


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    oracle_margin: float = 0.0,
) -> tuple[MetricPack, dict[str, Any] | None]:
    criterion = nn.MSELoss()
    model.eval()
    losses: list[float] = []
    rmses: list[float] = []
    rrmses: list[float] = []
    snrs: list[float] = []
    analysis_sums: dict[str, float] | None = None
    analysis_counts: dict[str, int] | None = None
    analysis_extrema: dict[str, float] | None = None

    with torch.no_grad():
        for batch in loader:
            spatial_base = None
            single_base = None
            if len(batch) == 3:
                noisy, clean, spatial_base = batch
                spatial_base = spatial_base.to(device, non_blocking=True)
            elif len(batch) == 4:
                noisy, clean, spatial_base, single_base = batch
                spatial_base = spatial_base.to(device, non_blocking=True)
                single_base = single_base.to(device, non_blocking=True)
            else:
                noisy, clean = batch
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            if model_type == 'stdifftem':
                pred, aux = model(
                    noisy,
                    spatial_base_log=spatial_base,
                    single_base_log=single_base,
                    return_aux=True,
                )
                if analysis_sums is None:
                    analysis_sums = {
                        'w_single_sum': 0.0,
                        'residual_scale_sum': 0.0,
                        'effective_gate_sum': 0.0,
                        'anchor_mse_sum': 0.0,
                        'spatial_base_mse_sum': 0.0,
                        'single_base_mse_sum': 0.0,
                        'oracle_regret_sum': 0.0,
                    }
                    analysis_counts = {
                        'w_single': 0,
                        'residual_scale': 0,
                        'effective_gate': 0,
                        'mse': 0,
                        'single_mse': 0,
                    }
                    analysis_extrema = {
                        'w_single_min': float('inf'),
                        'w_single_max': float('-inf'),
                    }
                assert analysis_sums is not None
                assert analysis_counts is not None
                assert analysis_extrema is not None
                anchor_log = aux.get('anchor_log', aux.get('base_log'))
                spatial_log = aux.get('spatial_base_log')
                single_log = aux.get('single_base_log')
                w_single = aux.get('w_single')
                residual_scale_factor = aux.get('residual_scale_factor')
                effective_residual_gate = aux.get('effective_residual_gate')
                pred_sq = (pred - clean).pow(2)
                spatial_sq = (spatial_log - clean).pow(2) if spatial_log is not None else pred_sq
                single_sq = (single_log - clean).pow(2) if single_log is not None else None
                oracle_err = torch.minimum(single_sq, spatial_sq) if single_sq is not None else spatial_sq
                margin = torch.as_tensor(oracle_margin, dtype=pred.dtype, device=pred.device)
                analysis_sums['oracle_regret_sum'] += float(torch.relu(pred_sq - oracle_err - margin).sum().item())
                analysis_sums['anchor_mse_sum'] += float((anchor_log - clean).pow(2).sum().item()) if anchor_log is not None else 0.0
                analysis_sums['spatial_base_mse_sum'] += float(spatial_sq.sum().item())
                if single_log is not None and single_sq is not None:
                    analysis_sums['single_base_mse_sum'] += float(single_sq.sum().item())
                    analysis_counts['single_mse'] += int(single_sq.numel())
                analysis_counts['mse'] += int(pred_sq.numel())
                if w_single is not None:
                    analysis_sums['w_single_sum'] += float(w_single.sum().item())
                    analysis_counts['w_single'] += int(w_single.numel())
                    analysis_extrema['w_single_min'] = min(analysis_extrema['w_single_min'], float(w_single.min().item()))
                    analysis_extrema['w_single_max'] = max(analysis_extrema['w_single_max'], float(w_single.max().item()))
                if residual_scale_factor is not None:
                    analysis_sums['residual_scale_sum'] += float(residual_scale_factor.sum().item())
                    analysis_counts['residual_scale'] += int(residual_scale_factor.numel())
                if effective_residual_gate is not None:
                    analysis_sums['effective_gate_sum'] += float(effective_residual_gate.sum().item())
                    analysis_counts['effective_gate'] += int(effective_residual_gate.numel())
            else:
                pred = model(noisy)
            losses.append(float(criterion(pred, clean).item()))
            rmse, rrmse, snr = compute_metrics_batch(pred, clean)
            rmses.append(float(rmse.mean().item()))
            rrmses.append(float(rrmse.mean().item()))
            snrs.append(float(snr.mean().item()))

    metrics = MetricPack(
        loss=float(np.mean(losses)) if losses else float('nan'),
        rmse=float(np.mean(rmses)) if rmses else float('nan'),
        rrmse=float(np.mean(rrmses)) if rrmses else float('nan'),
        snr=float(np.mean(snrs)) if snrs else float('nan'),
    )
    if analysis_sums is None or analysis_counts is None or analysis_extrema is None:
        return metrics, None
    w_single_mean = (
        analysis_sums['w_single_sum'] / analysis_counts['w_single']
        if analysis_counts['w_single'] > 0
        else 0.0
    )
    analysis = {
        'w_single_mean': float(w_single_mean),
        'w_single_min': float(analysis_extrema['w_single_min']) if analysis_counts['w_single'] > 0 else 0.0,
        'w_single_max': float(analysis_extrema['w_single_max']) if analysis_counts['w_single'] > 0 else 0.0,
        'residual_scale_mean': (
            float(analysis_sums['residual_scale_sum'] / analysis_counts['residual_scale'])
            if analysis_counts['residual_scale'] > 0
            else 0.0
        ),
        'effective_gate_mean': (
            float(analysis_sums['effective_gate_sum'] / analysis_counts['effective_gate'])
            if analysis_counts['effective_gate'] > 0
            else 0.0
        ),
        'anchor_mse': float(analysis_sums['anchor_mse_sum'] / analysis_counts['mse']) if analysis_counts['mse'] > 0 else float('nan'),
        'spatial_base_mse': float(analysis_sums['spatial_base_mse_sum'] / analysis_counts['mse']) if analysis_counts['mse'] > 0 else float('nan'),
        'single_base_mse': (
            float(analysis_sums['single_base_mse_sum'] / analysis_counts['single_mse'])
            if analysis_counts['single_mse'] > 0
            else None
        ),
        'oracle_regret': float(analysis_sums['oracle_regret_sum'] / analysis_counts['mse']) if analysis_counts['mse'] > 0 else float('nan'),
    }
    return metrics, analysis


def load_run_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def build_model(model_type: str, model_cfg: dict[str, Any]) -> nn.Module:
    common_kwargs = {
        'seq_len': int(model_cfg.get('seq_len', 869)),
        'num_receivers': int(model_cfg.get('num_receivers', 11)),
        'filters': int(model_cfg.get('filters', 16)),
        'lstm_hidden': int(model_cfg.get('lstm_hidden', 32)),
        'lstm_layers': int(model_cfg.get('lstm_layers', 2)),
        'num_heads': int(model_cfg.get('num_heads', 4)),
    }
    if model_type == 'single':
        return IndependentReceiverPCBiLSTM(**common_kwargs)
    if model_type == 'multi':
        return MultiReceiverPCBiLSTM(**common_kwargs)
    if model_type == 'spatial':
        return SpatialReceiverPCBiLSTM(
            **common_kwargs,
            spatial_layers=int(model_cfg.get('spatial_layers', 2)),
            dropout=float(model_cfg.get('dropout', 0.1)),
            use_graph_bias=bool(model_cfg.get('use_graph_bias', True)),
            residual=bool(model_cfg.get('residual', True)),
        )
    if model_type == 'stdifftem':
        base_gate_init_bias = model_cfg.get('base_gate_init_bias')
        prediction_type = str(model_cfg.get('prediction_type', 'noise'))
        return STDiffTEMNet(
            seq_len=int(model_cfg.get('seq_len', 869)),
            num_receivers=int(model_cfg.get('num_receivers', 11)),
            variant=str(model_cfg.get('variant', 'spatial_base')),
            condition_feature_dim=int(model_cfg.get('condition_feature_dim', 64)),
            condition_hidden_channels=int(model_cfg.get('condition_hidden_channels', 32)),
            condition_spatial_layers=int(model_cfg.get('condition_spatial_layers', 1)),
            gate_hidden_dim=int(model_cfg.get('gate_hidden_dim', 64)),
            diffusion_hidden_channels=int(model_cfg.get('diffusion_hidden_channels', 128)),
            diffusion_steps=int(model_cfg.get('diffusion_steps', 16)),
            sample_steps=int(model_cfg.get('sample_steps', 8)),
            beta_start=float(model_cfg.get('beta_start', 1e-4)),
            beta_end=float(model_cfg.get('beta_end', 2e-2)),
            step_dim=int(model_cfg.get('step_dim', 64)),
            num_heads=int(model_cfg.get('num_heads', 4)),
            dropout=float(model_cfg.get('dropout', 0.1)),
            refine_scale=float(model_cfg.get('refine_scale', 1.0)),
            sampler_seed=int(model_cfg.get('sampler_seed', 20260430)),
            prediction_type=prediction_type,
            residual_inference_mode=str(model_cfg.get('residual_inference_mode', 'ddim_random')),
            residual_anchor_mode=str(model_cfg.get('residual_anchor_mode', 'zero')),
            residual_clip_value=float(model_cfg.get('residual_clip_value', 0.0)),
            max_residual_gate=float(model_cfg.get('max_residual_gate', 1.0)),
            rho_max=float(model_cfg.get('rho_max', 0.5)),
            residual_scale_by_w_single=bool(model_cfg.get('residual_scale_by_w_single', True)),
            residual_scale_floor=float(model_cfg.get('residual_scale_floor', 0.5)),
            use_temporal_gate=bool(model_cfg.get('use_temporal_gate', False)),
            temporal_gate_init_bias=float(model_cfg.get('temporal_gate_init_bias', 0.0)),
            base_gate_init_bias=float(base_gate_init_bias) if base_gate_init_bias is not None else 4.0,
            residual_gate_init_bias=float(model_cfg.get('residual_gate_init_bias', -4.0)),
            zero_init_gates=bool(model_cfg.get('zero_init_gates', True)),
            disable_diffusion_residual=bool(model_cfg.get('disable_diffusion_residual', False)),
            direct_residual_nodiff=bool(model_cfg.get('direct_residual_nodiff', False)),
            spatial_artifact_dir=model_cfg.get('spatial_artifact_dir'),
            spatial_model_config=dict(model_cfg.get('spatial_model_config', {})),
            single_artifact_dir=model_cfg.get('single_artifact_dir'),
            single_model_config=dict(model_cfg.get('single_model_config', {})),
        )
    raise ValueError(f'Unsupported model_type: {model_type}')


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(payload, dict) and 'state_dict' in payload:
        return payload['state_dict']
    if isinstance(payload, dict):
        return payload
    raise ValueError(f'Unsupported checkpoint format: {checkpoint_path}')


def default_run_config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.resolve().parent / 'run_config.json'


def scenario_paths(scenario_dir: Path) -> tuple[Path, Path]:
    slug = scenario_dir.name
    return scenario_dir / f'{slug}_data.npy', scenario_dir / f'{slug}_labels.npy'


def load_scenario_specs(data_root: Path) -> list[dict[str, Any]]:
    manifest_path = data_root / 'test_scenario_manifest.json'
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        return list(payload.get('scenarios', []))

    scenario_root = data_root / 'test_scenarios'
    specs: list[dict[str, Any]] = []
    for scenario_dir in sorted(scenario_root.iterdir()):
        if scenario_dir.is_dir():
            specs.append({'slug': scenario_dir.name, 'name': scenario_dir.name.replace('_', ' ')})
    return specs


def resolve_base_cache_paths(
    base_cache_dir: Path | None,
    stem: str,
) -> tuple[Path | None, Path | None]:
    if base_cache_dir is None:
        return None, None

    spatial_candidates = [
        base_cache_dir / f'{stem}_spatial_base_log10.npy',
        base_cache_dir / f'{stem}_base_log10.npy',
    ]
    single_candidates = [
        base_cache_dir / f'{stem}_single_base_log10.npy',
    ]

    spatial_path = next((path for path in spatial_candidates if path.exists()), None)
    single_path = next((path for path in single_candidates if path.exists()), None)
    return spatial_path, single_path


def require_base_cache_paths(
    base_cache_dir: Path,
    stem: str,
    require_single: bool,
) -> tuple[Path, Path | None]:
    spatial_path, single_path = resolve_base_cache_paths(base_cache_dir, stem)
    missing: list[str] = []
    if spatial_path is None:
        missing.append(str((base_cache_dir / f'{stem}_spatial_base_log10.npy').resolve()))
    if require_single and single_path is None:
        missing.append(str((base_cache_dir / f'{stem}_single_base_log10.npy').resolve()))
    if missing:
        raise FileNotFoundError(
            'Missing precomputed ST-DiffTEM base cache(s). '
            'Run train_st_difftem.py again so the scenario caches are generated:\n' + '\n'.join(missing)
        )
    return spatial_path, single_path


def evaluate_scenario(
    model: nn.Module,
    model_type: str,
    device: torch.device,
    data_path: Path,
    label_path: Path,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    oracle_margin: float,
    spatial_base_cache_path: Path | None = None,
    single_base_cache_path: Path | None = None,
) -> dict[str, Any]:
    dataset, shape = build_dataset(
        data_path,
        label_path,
        limit,
        spatial_base_cache_path=spatial_base_cache_path,
        single_base_cache_path=single_base_cache_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    metrics, analysis = evaluate_model(model, loader, device, model_type, oracle_margin=oracle_margin)
    result = {
        'shape': list(shape),
        'sample_count': int(shape[0]),
        'metrics': metrics.to_dict(),
    }
    if analysis is not None:
        result['analysis'] = analysis
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        'model_type',
        'scenario_slug',
        'scenario_name',
        'sample_count',
        'loss',
        'rmse',
        'rrmse',
        'snr',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-type', choices=('single', 'multi', 'spatial', 'stdifftem'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-config', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--base-cache-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--device-ids', type=str, help="Comma-separated CUDA device ids, e.g. '0,1'. Default uses all visible GPUs when CUDA is selected.")
    parser.add_argument('--scenario-limit', type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    device_ids = parse_device_ids(args.device_ids, device)
    if device.type == 'cuda' and device_ids:
        device = torch.device('cuda', int(device_ids[0]))
    run_config_path = args.run_config if args.run_config is not None else default_run_config_path(args.checkpoint)
    run_config = load_run_config(run_config_path)
    model_cfg = dict(run_config.get('model', {}))
    loss_cfg = dict(run_config.get('loss_config', {}))
    oracle_margin = float(loss_cfg.get('oracle_margin', loss_cfg.get('baseline_protect_margin', 0.0)))

    model = build_model(args.model_type, model_cfg)
    state_dict = load_checkpoint_state(args.checkpoint)
    state_dict_load: dict[str, Any] | None = None
    if args.model_type == 'stdifftem':
        incompatible = model.load_state_dict(state_dict, strict=False)
        state_dict_load = {
            'strict': False,
            'missing_keys': list(incompatible.missing_keys),
            'unexpected_keys': list(incompatible.unexpected_keys),
        }
    else:
        model.load_state_dict(state_dict)
    model, device_ids = maybe_wrap_data_parallel(model, device, device_ids)
    require_single_cache = args.model_type == 'stdifftem' and str(model_cfg.get('variant', 'spatial_base')) == 'hybrid_base'

    output_dir = args.output_dir if args.output_dir is not None else args.checkpoint.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scenario_results: dict[str, Any] = {}

    combined_test_data = args.data_root / 'test' / 'test_data.npy'
    combined_test_labels = args.data_root / 'test' / 'test_labels.npy'
    combined_test_result = None
    if combined_test_data.exists() and combined_test_labels.exists():
        if args.model_type == 'stdifftem' and args.base_cache_dir is not None:
            combined_spatial_cache, combined_single_cache = require_base_cache_paths(args.base_cache_dir, 'test', require_single_cache)
        else:
            combined_spatial_cache, combined_single_cache = resolve_base_cache_paths(args.base_cache_dir, 'test')
        combined_test_result = evaluate_scenario(
            model=model,
            model_type=args.model_type,
            device=device,
            data_path=combined_test_data,
            label_path=combined_test_labels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.scenario_limit,
            oracle_margin=oracle_margin,
            spatial_base_cache_path=combined_spatial_cache,
            single_base_cache_path=combined_single_cache,
        )

    for spec in load_scenario_specs(args.data_root):
        slug = str(spec['slug'])
        scenario_name = str(spec.get('name', slug))
        scenario_dir = args.data_root / 'test_scenarios' / slug
        data_path, label_path = scenario_paths(scenario_dir)
        if not data_path.exists() or not label_path.exists():
            raise FileNotFoundError(f'Missing scenario files for {slug}: {data_path} / {label_path}')

        if args.model_type == 'stdifftem' and args.base_cache_dir is not None:
            scenario_spatial_cache, scenario_single_cache = require_base_cache_paths(args.base_cache_dir, slug, require_single_cache)
        else:
            scenario_spatial_cache, scenario_single_cache = resolve_base_cache_paths(args.base_cache_dir, slug)
        result = evaluate_scenario(
            model=model,
            model_type=args.model_type,
            device=device,
            data_path=data_path,
            label_path=label_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.scenario_limit,
            oracle_margin=oracle_margin,
            spatial_base_cache_path=scenario_spatial_cache,
            single_base_cache_path=scenario_single_cache,
        )
        scenario_results[slug] = {
            'scenario_name': scenario_name,
            **result,
        }
        rows.append(
            {
                'model_type': args.model_type,
                'scenario_slug': slug,
                'scenario_name': scenario_name,
                'sample_count': int(result['sample_count']),
                **result['metrics'],
            }
        )
        print(
            f"[{args.model_type}:{slug}] "
            f"rmse={result['metrics']['rmse']:.6f} "
            f"rrmse={result['metrics']['rrmse']:.6f} "
            f"snr={result['metrics']['snr']:.3f}"
            + (
                f" w_single_mean={result['analysis']['w_single_mean']:.6f}"
                if 'analysis' in result
                else ""
            )
        )

    summary = {
        'model_type': args.model_type,
        'checkpoint': str(args.checkpoint.resolve()),
        'run_config_path': str(run_config_path.resolve()) if run_config_path.exists() else None,
        'data_root': str(args.data_root.resolve()),
        'device': str(device),
        'device_ids': device_ids,
        'data_parallel': bool(len(device_ids) > 1),
        'state_dict_load': state_dict_load,
        'combined_test': combined_test_result,
        'scenarios': scenario_results,
    }
    save_json(output_dir / 'scenario_metrics.json', summary)
    write_csv(output_dir / 'scenario_metrics.csv', rows)
    print(f"Scenario metrics written to: {output_dir}")


if __name__ == '__main__':
    main()
