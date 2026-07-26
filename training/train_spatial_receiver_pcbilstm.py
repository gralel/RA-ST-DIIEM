"""Train the spatial receiver PC-BiLSTM baseline on N x 11 x 869 numpy data.

Expected arrays:
- noisy data:  [N, 11, 869] or [N, 869, 11]
- clean label: [N, 11, 869] or [N, 869, 11]

Training is performed in log10 amplitude space. This variant extends the
multi-receiver PC-BiLSTM with explicit receiver-channel spatial attention.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.spatial_receiver_pcbilstm import SpatialReceiverPCBiLSTM


EPS = 1e-12
SELECTION_METRICS = ("loss", "rmse", "rrmse", "snr")


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


class TEMDataset(Dataset):
    def __init__(self, noisy_log10: np.ndarray, clean_log10: np.ndarray):
        self.noisy = torch.from_numpy(np.asarray(noisy_log10, dtype=np.float32))
        self.clean = torch.from_numpy(np.asarray(clean_log10, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


@dataclass
class TrainConfig:
    batch_size: int = 128
    num_workers: int = 8
    max_epochs: int = 120
    min_epochs: int = 40
    patience: int = 25
    warmup_epochs: int = 5
    lr: float = 3e-4
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


@dataclass
class LossConfig:
    profile: str = "mse_log"
    lambda_rec: float = 1.0
    lambda_rel: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "lambda_rec": self.lambda_rec,
            "lambda_rel": self.lambda_rel,
        }


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


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_array(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32)


def canonicalize_11t(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected 3D array (N,11,T), got {arr.shape}")
    if arr.shape[1] == 11:
        return arr
    if arr.shape[2] == 11:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"{name}: unsupported shape {arr.shape}, expected one axis to be 11")


def limit_samples(array: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or max_samples >= array.shape[0]:
        return array
    return array[:max_samples]


def to_log10_amplitude(array: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.asarray(array, dtype=np.float32)) + EPS).astype(np.float32)


def from_log10_amplitude(array: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(array, dtype=np.float32)).astype(np.float32)


def from_log10_amplitude_tensor(array: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.tensor(10.0, dtype=array.dtype, device=array.device), array)


def restore_negative_signal(pred_log10: np.ndarray) -> np.ndarray:
    return -from_log10_amplitude(pred_log10)


def compute_training_loss(
    pred_log10: torch.Tensor,
    clean_log10: torch.Tensor,
    loss_config: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_mse = nn.functional.mse_loss(pred_log10, clean_log10)
    if loss_config.profile == "mse_log":
        return loss_mse, {
            "loss_mse_log": float(loss_mse.detach().item()),
            "loss_rec": 0.0,
            "loss_rel": 0.0,
        }

    pred_amp = from_log10_amplitude_tensor(pred_log10)
    clean_amp = from_log10_amplitude_tensor(clean_log10)
    loss_rec = nn.functional.l1_loss(pred_amp, clean_amp)
    loss_rel = torch.mean((pred_amp - clean_amp).pow(2) / (clean_amp.pow(2) + EPS))
    total = loss_mse + loss_config.lambda_rec * loss_rec + loss_config.lambda_rel * loss_rel
    return total, {
        "loss_mse_log": float(loss_mse.detach().item()),
        "loss_rec": float(loss_rec.detach().item()),
        "loss_rel": float(loss_rel.detach().item()),
    }


def build_dataset(
    data_path: Path,
    label_path: Path,
    limit: int | None,
) -> tuple[TEMDataset, tuple[int, int, int]]:
    noisy = limit_samples(canonicalize_11t(load_array(data_path), str(data_path)), limit)
    clean = limit_samples(canonicalize_11t(load_array(label_path), str(label_path)), limit)
    if noisy.shape != clean.shape:
        raise ValueError(f"Shape mismatch: {noisy.shape} vs {clean.shape}")
    dataset = TEMDataset(to_log10_amplitude(noisy), to_log10_amplitude(clean))
    return dataset, tuple(int(v) for v in noisy.shape)


def compute_metrics_batch(
    pred_log10: torch.Tensor,
    target_log10: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    loss_config: LossConfig,
) -> MetricPack:
    model.eval()

    losses: list[float] = []
    rmses: list[float] = []
    rrmses: list[float] = []
    snrs: list[float] = []

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(noisy)
            total_loss, _ = compute_training_loss(pred, clean, loss_config)
            losses.append(float(total_loss.item()))
            rmse, rrmse, snr = compute_metrics_batch(pred, clean)
            rmses.append(float(rmse.mean().item()))
            rrmses.append(float(rrmse.mean().item()))
            snrs.append(float(snr.mean().item()))

    return MetricPack(
        loss=float(np.mean(losses)),
        rmse=float(np.mean(rmses)),
        rrmse=float(np.mean(rrmses)),
        snr=float(np.mean(snrs)),
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


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    config: TrainConfig,
    loss_config: LossConfig,
) -> tuple[MetricPack, list[dict[str, Any]], Path]:
    optimizer = optim.AdamW(
        model.parameters(),
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
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    ensure_dir(output_dir)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss_total = 0.0
        train_mse_total = 0.0
        train_rec_total = 0.0
        train_rel_total = 0.0

        for step_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(noisy)
            total_loss, parts = compute_training_loss(pred, clean, loss_config)
            loss = total_loss / config.grad_accum_steps
            loss.backward()

            if (step_idx + 1) % config.grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss_total += float(loss.item() * config.grad_accum_steps)
            train_mse_total += parts["loss_mse_log"]
            train_rec_total += parts["loss_rec"]
            train_rel_total += parts["loss_rel"]

        scheduler.step()
        train_loss = train_loss_total / max(len(train_loader), 1)
        train_mse = train_mse_total / max(len(train_loader), 1)
        train_rec = train_rec_total / max(len(train_loader), 1)
        train_rel = train_rel_total / max(len(train_loader), 1)
        val_metrics = evaluate_model(model, val_loader, device, loss_config)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_loss_mse_log": train_mse,
                "train_loss_rec": train_rec,
                "train_loss_rel": train_rel,
                "val_loss": val_metrics.loss,
                "val_rmse": val_metrics.rmse,
                "val_rrmse": val_metrics.rrmse,
                "val_snr": val_metrics.snr,
                "lr": lr_now,
            }
        )
        print(
            f"[spatial_receiver_pcbilstm epoch {epoch:03d}] "
            f"train={train_loss:.6f} mse={train_mse:.6f} rec={train_rec:.6f} rel={train_rel:.6f} | "
            f"val={val_metrics.loss:.6f} rmse={val_metrics.rmse:.6f} "
            f"rrmse={val_metrics.rrmse:.6f} snr={val_metrics.snr:.3f} | "
            f"lr={lr_now:.2e}"
        )

        if best_metrics is None or _is_better_metric(val_metrics, best_metrics, config.selection_metric):
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
            print(
                f"[spatial_receiver_pcbilstm] new best checkpoint by {config.selection_metric}="
                f"{_metric_value(val_metrics, config.selection_metric):.6f}."
            )
        else:
            stale_epochs += 1

        if epoch >= config.min_epochs and stale_epochs >= config.patience:
            print("[spatial_receiver_pcbilstm] early stopping triggered.")
            break

    if best_metrics is None or best_state is None:
        raise RuntimeError("Training produced no checkpoint.")

    unwrap_model(model).load_state_dict(best_state)
    checkpoint_path = output_dir / 'best_model.pth'
    torch.save(
        {
            "state_dict": best_state,
            "best_metrics": best_metrics.to_dict(),
            "best_epoch": best_epoch,
            "history": history,
            "config": config.to_dict(),
            "loss_config": loss_config.to_dict(),
        },
        checkpoint_path,
    )
    save_json(
        output_dir / 'training_summary.json',
        {
            "best_epoch": best_epoch,
            "best_metrics": best_metrics.to_dict(),
            "config": config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "history": history,
            "checkpoint": str(checkpoint_path.resolve()),
        },
    )
    return best_metrics, history, checkpoint_path


def predict_numpy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for noisy, _ in loader:
            noisy = noisy.to(device, non_blocking=True)
            pred = model(noisy).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 11, 869), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--train-labels', type=Path, required=True)
    parser.add_argument('--val-data', type=Path, required=True)
    parser.add_argument('--val-labels', type=Path, required=True)
    parser.add_argument('--test-data', type=Path)
    parser.add_argument('--test-labels', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-epochs', type=int, default=120)
    parser.add_argument('--min-epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--selection-metric', choices=SELECTION_METRICS, default='rrmse')
    parser.add_argument('--loss-profile', choices=('mse_log', 'enhanced'), default='mse_log')
    parser.add_argument('--lambda-rec', type=float, default=1.0)
    parser.add_argument('--lambda-rel', type=float, default=0.1)
    parser.add_argument('--filters', type=int, default=16)
    parser.add_argument('--lstm-hidden', type=int, default=32)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=869)
    parser.add_argument('--num-receivers', type=int, default=11)
    parser.add_argument('--spatial-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no-graph-bias', action='store_true')
    parser.add_argument('--no-residual', action='store_true')
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--val-limit', type=int)
    parser.add_argument('--test-limit', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--device-ids', type=str, help="Comma-separated CUDA device ids, e.g. '0,1'. Default uses all visible GPUs when CUDA is selected.")
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--save-test-predictions', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    device_ids = parse_device_ids(args.device_ids, device)
    if device.type == 'cuda' and device_ids:
        device = torch.device('cuda', int(device_ids[0]))
    pin_memory = device.type == 'cuda'
    loader_kwargs: dict[str, Any] = {
        'num_workers': args.num_workers,
        'pin_memory': pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4

    train_dataset, train_shape = build_dataset(args.train_data, args.train_labels, args.train_limit)
    val_dataset, val_shape = build_dataset(args.val_data, args.val_labels, args.val_limit)

    for shape_name, shape in (("train", train_shape), ("val", val_shape)):
        if shape[1] != args.num_receivers:
            raise ValueError(f"{shape_name}: expected {args.num_receivers} receivers, got {shape[1]}")
        if shape[2] != args.seq_len:
            raise ValueError(f"{shape_name}: expected seq_len {args.seq_len}, got {shape[2]}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = SpatialReceiverPCBiLSTM(
        seq_len=args.seq_len,
        num_receivers=args.num_receivers,
        filters=args.filters,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        num_heads=args.num_heads,
        spatial_layers=args.spatial_layers,
        dropout=args.dropout,
        use_graph_bias=not args.no_graph_bias,
        residual=not args.no_residual,
    ).to(device)

    model, device_ids = maybe_wrap_data_parallel(model, device, device_ids)

    output_dir = ensure_dir(args.output_dir)
    loss_config = LossConfig(
        profile=args.loss_profile,
        lambda_rec=args.lambda_rec,
        lambda_rel=args.lambda_rel,
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
        output_dir / 'run_config.json',
        {
            'train_data': str(args.train_data.resolve()),
            'train_labels': str(args.train_labels.resolve()),
            'val_data': str(args.val_data.resolve()),
            'val_labels': str(args.val_labels.resolve()),
            'test_data': str(args.test_data.resolve()) if args.test_data else None,
            'test_labels': str(args.test_labels.resolve()) if args.test_labels else None,
            'device': str(device),
            'device_ids': device_ids,
            'data_parallel': bool(len(device_ids) > 1),
            'seed': args.seed,
            'model': {
                'seq_len': args.seq_len,
                'num_receivers': args.num_receivers,
                'filters': args.filters,
                'lstm_hidden': args.lstm_hidden,
                'lstm_layers': args.lstm_layers,
                'num_heads': args.num_heads,
                'spatial_layers': args.spatial_layers,
                'dropout': args.dropout,
                'use_graph_bias': not args.no_graph_bias,
                'residual': not args.no_residual,
            },
            'train_config': TrainConfig(
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
            ).to_dict(),
            'loss_config': loss_config.to_dict(),
            'dataset_shapes': {
                'train': train_shape,
                'val': val_shape,
            },
        },
    )

    best_metrics, _, checkpoint_path = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        config=train_config,
        loss_config=loss_config,
    )
    if device.type == 'cuda':
        print(f"Using CUDA device ids: {device_ids if device_ids else [0]}")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Best validation metrics: {best_metrics.to_dict()}")

    if args.test_data and args.test_labels:
        test_dataset, test_shape = build_dataset(args.test_data, args.test_labels, args.test_limit)
        if test_shape[1] != args.num_receivers or test_shape[2] != args.seq_len:
            raise ValueError(f"test: expected (*,{args.num_receivers},{args.seq_len}), got {test_shape}")
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        test_metrics = evaluate_model(model, test_loader, device, loss_config)
        save_json(
            output_dir / 'test_metrics.json',
            {
                'test_shape': test_shape,
                'metrics': test_metrics.to_dict(),
            },
        )
        print(f"Test metrics: {test_metrics.to_dict()}")

        if args.save_test_predictions:
            pred_log10 = predict_numpy(model, test_loader, device)
            np.save(output_dir / 'test_pred_log10.npy', pred_log10.astype(np.float32))
            np.save(output_dir / 'test_pred_amplitude.npy', from_log10_amplitude(pred_log10))
            np.save(output_dir / 'test_pred_negative_signal.npy', restore_negative_signal(pred_log10))
            print(f"Saved test predictions under: {output_dir}")


if __name__ == '__main__':
    main()
