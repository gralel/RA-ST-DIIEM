"""Train external 11x869 TEM denoising baselines and evaluate fixed scenarios."""

from __future__ import annotations

import argparse
import csv
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

from models.external_baselines import build_external_baseline


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


@dataclass
class TrainConfig:
    batch_size: int = 128
    num_workers: int = 4
    max_epochs: int = 120
    min_epochs: int = 40
    patience: int = 25
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


class TEMDataset(Dataset):
    def __init__(self, noisy_log10: np.ndarray, clean_log10: np.ndarray):
        self.noisy = torch.from_numpy(np.asarray(noisy_log10, dtype=np.float32))
        self.clean = torch.from_numpy(np.asarray(clean_log10, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


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
    ids: list[int] = []
    for part in str(device_ids_arg).split(","):
        item = part.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= visible_count:
            raise ValueError(f"Invalid GPU index {idx}; visible gpu count is {visible_count}.")
        if idx not in ids:
            ids.append(idx)
    if not ids:
        raise ValueError("No valid GPU ids were parsed from --device-ids.")
    return ids


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


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def canonicalize_11t(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected 3D array (N,11,T), got {arr.shape}")
    if arr.shape[1] == 11:
        return arr
    if arr.shape[2] == 11:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"{name}: unsupported shape {arr.shape}, expected one axis to be 11")


def limit_samples(array: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or limit <= 0 or limit >= array.shape[0]:
        return array
    return array[:limit]


def to_log10_amplitude(array: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(np.asarray(array, dtype=np.float32)) + EPS).astype(np.float32)


def build_dataset(data_path: Path, label_path: Path, limit: int | None = None) -> tuple[TEMDataset, tuple[int, int, int]]:
    noisy = limit_samples(canonicalize_11t(np.load(data_path), str(data_path)), limit)
    clean = limit_samples(canonicalize_11t(np.load(label_path), str(label_path)), limit)
    if noisy.shape != clean.shape:
        raise ValueError(f"Shape mismatch: {noisy.shape} vs {clean.shape}")
    return TEMDataset(to_log10_amplitude(noisy), to_log10_amplitude(clean)), tuple(int(v) for v in noisy.shape)


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


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> MetricPack:
    criterion = nn.MSELoss()
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
            losses.append(float(criterion(pred, clean).item()))
            rmse, rrmse, snr = compute_metrics_batch(pred, clean)
            rmses.append(float(rmse.mean().item()))
            rrmses.append(float(rrmse.mean().item()))
            snrs.append(float(snr.mean().item()))
    return MetricPack(
        loss=float(np.mean(losses)) if losses else float("nan"),
        rmse=float(np.mean(rmses)) if rmses else float("nan"),
        rrmse=float(np.mean(rrmses)) if rrmses else float("nan"),
        snr=float(np.mean(snrs)) if snrs else float("nan"),
    )


def metric_value(metrics: MetricPack, metric_name: str) -> float:
    if metric_name not in SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric: {metric_name}")
    return float(getattr(metrics, metric_name))


def is_better(current: MetricPack, best: MetricPack, metric_name: str) -> bool:
    if metric_name == "snr":
        return metric_value(current, metric_name) > metric_value(best, metric_name)
    return metric_value(current, metric_name) < metric_value(best, metric_name)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    config: TrainConfig,
    model_name: str,
) -> tuple[MetricPack, Path]:
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    warmup = max(0, min(config.warmup_epochs, max(config.max_epochs - 1, 0)))
    if warmup > 0:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, config.max_epochs - warmup),
                    eta_min=config.lr * 0.05,
                ),
            ],
            milestones=[warmup],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.max_epochs),
            eta_min=config.lr * 0.05,
        )
    criterion = nn.MSELoss()
    best_metrics: MetricPack | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss_total = 0.0
        for step_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(noisy)
            loss = criterion(pred, clean) / config.grad_accum_steps
            loss.backward()
            if (step_idx + 1) % config.grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            train_loss_total += float(loss.item() * config.grad_accum_steps)

        scheduler.step()
        train_loss = train_loss_total / max(len(train_loader), 1)
        val_metrics = evaluate_model(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics.loss,
                "val_rmse": val_metrics.rmse,
                "val_rrmse": val_metrics.rrmse,
                "val_snr": val_metrics.snr,
                "lr": lr_now,
            }
        )
        print(
            f"[{model_name} epoch {epoch:03d}] train={train_loss:.6f} | "
            f"val={val_metrics.loss:.6f} rmse={val_metrics.rmse:.6f} "
            f"rrmse={val_metrics.rrmse:.6f} snr={val_metrics.snr:.3f} | lr={lr_now:.2e}"
        )
        if best_metrics is None or is_better(val_metrics, best_metrics, config.selection_metric):
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu() for k, v in unwrap_model(model).state_dict().items()}
            best_epoch = epoch
            stale = 0
            print(f"[{model_name}] new best by {config.selection_metric}={metric_value(val_metrics, config.selection_metric):.6f}")
        else:
            stale += 1
        if epoch >= config.min_epochs and stale >= config.patience:
            print(f"[{model_name}] early stopping triggered.")
            break

    if best_metrics is None or best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    unwrap_model(model).load_state_dict(best_state)
    checkpoint_path = output_dir / "best_model.pth"
    torch.save(
        {
            "state_dict": best_state,
            "best_metrics": best_metrics.to_dict(),
            "best_epoch": best_epoch,
            "history": history,
            "config": config.to_dict(),
        },
        checkpoint_path,
    )
    save_json(
        output_dir / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "best_metrics": best_metrics.to_dict(),
            "config": config.to_dict(),
            "history": history,
            "checkpoint": str(checkpoint_path.resolve()),
        },
    )
    return best_metrics, checkpoint_path


def load_scenario_specs(data_root: Path) -> list[dict[str, Any]]:
    manifest_path = data_root / "test_scenario_manifest.json"
    if manifest_path.exists():
        return list(json.loads(manifest_path.read_text(encoding="utf-8")).get("scenarios", []))
    scenario_root = data_root / "test_scenarios"
    return [{"slug": p.name, "name": p.name.replace("_", " ")} for p in sorted(scenario_root.iterdir()) if p.is_dir()]


def scenario_paths(data_root: Path, slug: str) -> tuple[Path, Path]:
    scenario_dir = data_root / "test_scenarios" / slug
    return scenario_dir / f"{slug}_data.npy", scenario_dir / f"{slug}_labels.npy"


def evaluate_and_write_scenarios(
    model: nn.Module,
    device: torch.device,
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    model_name: str,
) -> None:
    loader_kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda"}
    combined = None
    test_data = data_root / "test" / "test_data.npy"
    test_labels = data_root / "test" / "test_labels.npy"
    if test_data.exists() and test_labels.exists():
        dataset, shape = build_dataset(test_data, test_labels, limit)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
        metrics = evaluate_model(model, loader, device)
        combined = {"shape": list(shape), "sample_count": int(shape[0]), "metrics": metrics.to_dict()}
        save_json(output_dir / "test_metrics.json", {"test_shape": list(shape), "metrics": metrics.to_dict()})
        print(f"[{model_name}:combined_test] rrmse={metrics.rrmse:.6f} snr={metrics.snr:.3f}")

    rows: list[dict[str, Any]] = []
    scenario_results: dict[str, Any] = {}
    for spec in load_scenario_specs(data_root):
        slug = str(spec["slug"])
        data_path, label_path = scenario_paths(data_root, slug)
        dataset, shape = build_dataset(data_path, label_path, limit)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
        metrics = evaluate_model(model, loader, device)
        scenario_results[slug] = {
            "scenario_name": str(spec.get("name", slug)),
            "shape": list(shape),
            "sample_count": int(shape[0]),
            "metrics": metrics.to_dict(),
        }
        rows.append(
            {
                "model_type": model_name,
                "scenario_slug": slug,
                "scenario_name": str(spec.get("name", slug)),
                "sample_count": int(shape[0]),
                **metrics.to_dict(),
            }
        )
        print(f"[{model_name}:{slug}] rmse={metrics.rmse:.6f} rrmse={metrics.rrmse:.6f} snr={metrics.snr:.3f}")

    save_json(
        output_dir / "scenario_metrics.json",
        {
            "model_type": model_name,
            "data_root": str(data_root.resolve()),
            "combined_test": combined,
            "scenarios": scenario_results,
        },
    )
    with (output_dir / "scenario_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_type", "scenario_slug", "scenario_name", "sample_count", "loss", "rmse", "rrmse", "snr"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tem1dformer869", "dncnn2d"), required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--data-root", type=Path, help="If set, write ten-scenario metrics after training.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="rrmse")
    parser.add_argument("--embed-size", type=int, default=96)
    parser.add_argument("--transformer-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-channels-2d", type=int, default=64)
    parser.add_argument("--depth-2d", type=int, default=8)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--scenario-limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device-ids", type=str)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    device_ids = parse_device_ids(args.device_ids, device)
    if device.type == "cuda" and device_ids:
        device = torch.device("cuda", int(device_ids[0]))
    output_dir = ensure_dir(args.output_dir)
    pin_memory = device.type == "cuda"
    loader_kwargs: dict[str, Any] = {"num_workers": args.num_workers, "pin_memory": pin_memory}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_dataset, train_shape = build_dataset(args.train_data, args.train_labels, args.train_limit)
    val_dataset, val_shape = build_dataset(args.val_data, args.val_labels, args.val_limit)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model_cfg = {
        "num_receivers": 11,
        "seq_len": 869,
        "embed_size": args.embed_size,
        "transformer_depth": args.transformer_depth,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "hidden_channels": args.hidden_channels_2d,
        "depth": args.depth_2d,
    }
    model = build_external_baseline(args.model, model_cfg)
    model, device_ids = maybe_wrap_data_parallel(model, device, device_ids)
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
            "model_type": "external_baseline",
            "model_name": args.model,
            "model": model_cfg,
            "train_config": train_config.to_dict(),
            "train_data": str(args.train_data.resolve()),
            "train_labels": str(args.train_labels.resolve()),
            "val_data": str(args.val_data.resolve()),
            "val_labels": str(args.val_labels.resolve()),
            "test_data": str(args.test_data.resolve()) if args.test_data else None,
            "test_labels": str(args.test_labels.resolve()) if args.test_labels else None,
            "seed": args.seed,
            "device": str(device),
            "device_ids": device_ids,
            "data_parallel": bool(len(device_ids) > 1),
            "dataset_shapes": {"train": list(train_shape), "val": list(val_shape)},
        },
    )
    best_metrics, checkpoint_path = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        config=train_config,
        model_name=args.model,
    )
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Best validation metrics: {best_metrics.to_dict()}")

    if args.test_data and args.test_labels:
        test_dataset, test_shape = build_dataset(args.test_data, args.test_labels, args.test_limit)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        test_metrics = evaluate_model(model, test_loader, device)
        save_json(output_dir / "test_metrics.json", {"test_shape": list(test_shape), "metrics": test_metrics.to_dict()})
        print(f"Test metrics: {test_metrics.to_dict()}")

    if args.data_root is not None:
        evaluate_and_write_scenarios(
            model=model,
            device=device,
            data_root=args.data_root,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.scenario_limit,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
