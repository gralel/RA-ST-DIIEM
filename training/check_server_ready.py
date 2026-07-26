"""Server-side integrity and readiness checks for the TEM baseline project."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from models.single_receiver_pcbilstm import IndependentReceiverPCBiLSTM
from models.multi_receiver_pcbilstm import MultiReceiverPCBiLSTM
from models.spatial_receiver_pcbilstm import SpatialReceiverPCBiLSTM
from models.st_difftem import STDiffTEMNet

SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
MODEL_CODE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / '数据' / 'data_paper_fixed_scenarios_25k'
EXPECTED_SCENARIOS = {
    'light_gaussian': 150,
    'low_snr_gaussian': 350,
    'colored_background': 200,
    'em_interference': 200,
    'impulse_spike': 200,
    'drift': 200,
    'local_receiver_corruption': 350,
    'dropout_missing_segment': 350,
    'mixed_complex_noise': 350,
    'unseen_noise_combination': 150,
}
EXPECTED_SPLITS = {
    'train': 20000,
    'val': 2500,
    'test': 2500,
}
EXPECTED_RECEIVERS = 11
EXPECTED_SEQ_LEN = 869


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


class IntegrityError(RuntimeError):
    pass


def canonicalize_11t(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise IntegrityError(f'{name}: expected 3D array, got {arr.shape}')
    if arr.shape[1] == EXPECTED_RECEIVERS:
        return arr
    if arr.shape[2] == EXPECTED_RECEIVERS:
        return np.transpose(arr, (0, 2, 1))
    raise IntegrityError(f'{name}: expected one axis to be {EXPECTED_RECEIVERS}, got {arr.shape}')


def check_exists(path: Path, label: str) -> CheckResult:
    ok = path.exists()
    return CheckResult(label, ok, str(path))


def check_split_pair(data_root: Path, split: str, full_finite: bool, finite_chunk: int) -> list[CheckResult]:
    results: list[CheckResult] = []
    split_dir = data_root / split
    data_path = split_dir / f'{split}_data.npy'
    label_path = split_dir / f'{split}_labels.npy'
    for label, path in ((f'{split}:data_file', data_path), (f'{split}:label_file', label_path)):
        results.append(check_exists(path, label))
        if not path.exists():
            return results

    data = canonicalize_11t(np.load(data_path, mmap_mode='r'), str(data_path))
    labels = canonicalize_11t(np.load(label_path, mmap_mode='r'), str(label_path))
    if data.shape != labels.shape:
        raise IntegrityError(f'{split}: data/label shape mismatch {data.shape} vs {labels.shape}')
    if data.shape[1] != EXPECTED_RECEIVERS or data.shape[2] != EXPECTED_SEQ_LEN:
        raise IntegrityError(f'{split}: expected (*,{EXPECTED_RECEIVERS},{EXPECTED_SEQ_LEN}), got {data.shape}')

    expected_count = EXPECTED_SPLITS.get(split)
    detail = f'shape={tuple(int(v) for v in data.shape)}'
    if expected_count is not None and data.shape[0] != expected_count:
        detail += f' expected_first_dim={expected_count}'
        results.append(CheckResult(f'{split}:shape', False, detail))
    else:
        results.append(CheckResult(f'{split}:shape', True, detail))

    sample_count = int(min(4, data.shape[0]))
    if sample_count > 0:
        data_probe = np.asarray(data[:sample_count])
        label_probe = np.asarray(labels[:sample_count])
        probe_ok = np.isfinite(data_probe).all() and np.isfinite(label_probe).all()
        results.append(CheckResult(f'{split}:probe_finite', bool(probe_ok), f'checked_first={sample_count}'))

    if full_finite:
        finite_ok = True
        for start in range(0, data.shape[0], finite_chunk):
            data_block = np.asarray(data[start:start + finite_chunk])
            label_block = np.asarray(labels[start:start + finite_chunk])
            if not np.isfinite(data_block).all() or not np.isfinite(label_block).all():
                finite_ok = False
                break
        results.append(CheckResult(f'{split}:full_finite', finite_ok, f'chunk={finite_chunk}'))

    return results


def check_scenario_manifest(data_root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    manifest_path = data_root / 'test_scenario_manifest.json'
    results.append(check_exists(manifest_path, 'scenario_manifest'))
    if not manifest_path.exists():
        return results

    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    scenarios = payload.get('scenarios', [])
    seen: dict[str, int] = {}
    for row in scenarios:
        slug = str(row['slug'])
        seen[slug] = int(row['sample_count'])

    missing = sorted(set(EXPECTED_SCENARIOS).difference(seen))
    extra = sorted(set(seen).difference(EXPECTED_SCENARIOS))
    count_match = all(seen.get(slug) == count for slug, count in EXPECTED_SCENARIOS.items())
    ok = (not missing) and (not extra) and count_match
    results.append(
        CheckResult(
            'scenario_manifest_content',
            ok,
            f'missing={missing} extra={extra} counts={seen}',
        )
    )

    scenario_root = data_root / 'test_scenarios'
    results.append(check_exists(scenario_root, 'test_scenarios_dir'))
    if not scenario_root.exists():
        return results

    for slug, expected_count in EXPECTED_SCENARIOS.items():
        scenario_dir = scenario_root / slug
        data_path = scenario_dir / f'{slug}_data.npy'
        label_path = scenario_dir / f'{slug}_labels.npy'
        recipe_path = scenario_dir / f'{slug}_recipes.jsonl'
        for label, path in (
            (f'scenario:{slug}:dir', scenario_dir),
            (f'scenario:{slug}:data_file', data_path),
            (f'scenario:{slug}:label_file', label_path),
            (f'scenario:{slug}:recipe_file', recipe_path),
        ):
            results.append(check_exists(path, label))
        if not (data_path.exists() and label_path.exists()):
            continue

        data = canonicalize_11t(np.load(data_path, mmap_mode='r'), str(data_path))
        labels = canonicalize_11t(np.load(label_path, mmap_mode='r'), str(label_path))
        ok = data.shape == labels.shape and data.shape[0] == expected_count and data.shape[1] == EXPECTED_RECEIVERS and data.shape[2] == EXPECTED_SEQ_LEN
        results.append(CheckResult(f'scenario:{slug}:shape', ok, f'shape={tuple(int(v) for v in data.shape)} expected_count={expected_count}'))

    return results


def check_training_scripts() -> list[CheckResult]:
    paths = [
        TRAINING_DIR / 'train_single_receiver_pcbilstm.py',
        TRAINING_DIR / 'train_multi_receiver_pcbilstm.py',
        TRAINING_DIR / 'train_spatial_receiver_pcbilstm.py',
        TRAINING_DIR / 'train_st_difftem.py',
        TRAINING_DIR / 'evaluate_noise_scenarios.py',
        TRAINING_DIR / 'run_pcbilstm_baselines_suite.py',
        TRAINING_DIR / 'run_st_difftem_suite.py',
        MODEL_CODE_ROOT / 'models' / 'single_receiver_pcbilstm.py',
        MODEL_CODE_ROOT / 'models' / 'multi_receiver_pcbilstm.py',
        MODEL_CODE_ROOT / 'models' / 'spatial_receiver_pcbilstm.py',
        MODEL_CODE_ROOT / 'models' / 'st_difftem.py',
    ]
    return [check_exists(path, f'script:{path.name}') for path in paths]


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


def check_torch_and_models(device_arg: str, device_ids_arg: str | None) -> list[CheckResult]:
    results: list[CheckResult] = []
    if torch is None:
        results.append(CheckResult('torch_import', False, repr(TORCH_IMPORT_ERROR)))
        return results

    device = torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            results.append(CheckResult('cuda_available', False, 'CUDA requested but not available'))
            return results
        device = torch.device('cuda')
    elif device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device_ids = parse_device_ids(device_ids_arg, device)
    if device.type == 'cuda' and device_ids:
        primary = int(device_ids[0])
        torch.cuda.set_device(primary)
        device = torch.device('cuda', primary)

    results.append(CheckResult('torch_import', True, f'version={torch.__version__} device={device} device_ids={device_ids}'))

    dummy = torch.randn(2, EXPECTED_RECEIVERS, EXPECTED_SEQ_LEN, device=device)
    models = {
        'single': IndependentReceiverPCBiLSTM(seq_len=EXPECTED_SEQ_LEN, num_receivers=EXPECTED_RECEIVERS),
        'multi': MultiReceiverPCBiLSTM(seq_len=EXPECTED_SEQ_LEN, num_receivers=EXPECTED_RECEIVERS),
        'spatial': SpatialReceiverPCBiLSTM(seq_len=EXPECTED_SEQ_LEN, num_receivers=EXPECTED_RECEIVERS),
        'stdifftem': STDiffTEMNet(seq_len=EXPECTED_SEQ_LEN, num_receivers=EXPECTED_RECEIVERS),
    }
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        ok = tuple(out.shape) == (2, EXPECTED_RECEIVERS, EXPECTED_SEQ_LEN) and torch.isfinite(out).all().item()
        results.append(CheckResult(f'model_forward:{name}', bool(ok), f'output_shape={tuple(out.shape)}'))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--device-ids', type=str, help="Comma-separated CUDA device ids, e.g. '0,1'. Default uses all visible GPUs when CUDA is selected.")
    parser.add_argument('--full-finite', action='store_true')
    parser.add_argument('--finite-chunk', type=int, default=256)
    parser.add_argument('--strict', action='store_true', help='Exit with code 1 if any check fails.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[CheckResult] = []
    try:
        results.extend(check_training_scripts())
        for split in ('train', 'val', 'test'):
            results.extend(check_split_pair(args.data_root, split, args.full_finite, args.finite_chunk))
        results.extend(check_scenario_manifest(args.data_root))
        results.extend(check_torch_and_models(args.device, args.device_ids))
    except Exception as exc:
        results.append(CheckResult('unexpected_exception', False, f'{type(exc).__name__}: {exc}'))

    fail_count = 0
    for row in results:
        flag = 'PASS' if row.ok else 'FAIL'
        print(f'[{flag}] {row.name}: {row.detail}')
        if not row.ok:
            fail_count += 1

    print(f'Checked {len(results)} items, failures={fail_count}')
    if args.strict and fail_count > 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

