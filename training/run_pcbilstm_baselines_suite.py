"""Train and compare single/multi/spatial PC-BiLSTM baselines on the paper dataset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
MODEL_CODE_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / '数据' / 'data_paper_fixed_scenarios_25k'
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / '训练结果' / 'pcbilstm_baselines_suite'

MODEL_SPECS = {
    'single': {
        'name': 'single_receiver_pcbilstm',
        'train_script': TRAINING_DIR / 'train_single_receiver_pcbilstm.py',
    },
    'multi': {
        'name': 'multi_receiver_pcbilstm',
        'train_script': TRAINING_DIR / 'train_multi_receiver_pcbilstm.py',
    },
    'spatial': {
        'name': 'spatial_receiver_pcbilstm',
        'train_script': TRAINING_DIR / 'train_spatial_receiver_pcbilstm.py',
    },
}


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def run_command(command: list[str], cwd: Path) -> None:
    printable = ' '.join(f'"{item}"' if ' ' in item else item for item in command)
    print(f'[run] {printable}')
    subprocess.run(command, cwd=str(cwd), check=True)


def add_optional_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--artifacts-root', type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument('--models', nargs='+', choices=tuple(MODEL_SPECS.keys()), default=['single', 'multi', 'spatial'])
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--device-ids', type=str, help="Comma-separated CUDA device ids, e.g. '0,1'. Default uses all visible GPUs when CUDA is selected.")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-epochs', type=int, default=120)
    parser.add_argument('--min-epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--selection-metric', choices=('loss', 'rmse', 'rrmse', 'snr'), default='rrmse')
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
    parser.add_argument('--scenario-limit', type=int)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--save-test-predictions', action='store_true')
    return parser.parse_args()


def training_paths(data_root: Path) -> dict[str, Path]:
    return {
        'train_data': data_root / 'train' / 'train_data.npy',
        'train_labels': data_root / 'train' / 'train_labels.npy',
        'val_data': data_root / 'val' / 'val_data.npy',
        'val_labels': data_root / 'val' / 'val_labels.npy',
        'test_data': data_root / 'test' / 'test_data.npy',
        'test_labels': data_root / 'test' / 'test_labels.npy',
    }


def ensure_dataset_exists(data_root: Path) -> None:
    missing = [str(path) for path in training_paths(data_root).values() if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing dataset files:\n' + '\n'.join(missing))
    scenario_manifest = data_root / 'test_scenario_manifest.json'
    if not scenario_manifest.exists():
        raise FileNotFoundError(f'Missing test scenario manifest: {scenario_manifest}')


def build_train_command(model_key: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    spec = MODEL_SPECS[model_key]
    data_paths = training_paths(args.data_root)
    command = [
        str(args.python),
        str(spec['train_script']),
        '--train-data', str(data_paths['train_data']),
        '--train-labels', str(data_paths['train_labels']),
        '--val-data', str(data_paths['val_data']),
        '--val-labels', str(data_paths['val_labels']),
        '--test-data', str(data_paths['test_data']),
        '--test-labels', str(data_paths['test_labels']),
        '--output-dir', str(output_dir),
        '--batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers),
        '--max-epochs', str(args.max_epochs),
        '--min-epochs', str(args.min_epochs),
        '--patience', str(args.patience),
        '--warmup-epochs', str(args.warmup_epochs),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--grad-accum-steps', str(args.grad_accum_steps),
        '--selection-metric', str(args.selection_metric),
        '--filters', str(args.filters),
        '--lstm-hidden', str(args.lstm_hidden),
        '--lstm-layers', str(args.lstm_layers),
        '--num-heads', str(args.num_heads),
        '--seq-len', str(args.seq_len),
        '--num-receivers', str(args.num_receivers),
        '--device', str(args.device),
        '--seed', str(args.seed),
    ]
    if args.device_ids is not None:
        command.extend(['--device-ids', str(args.device_ids)])
    add_optional_arg(command, '--train-limit', args.train_limit)
    add_optional_arg(command, '--val-limit', args.val_limit)
    add_optional_arg(command, '--test-limit', args.test_limit)
    if args.save_test_predictions:
        command.append('--save-test-predictions')
    if model_key == 'spatial':
        command.extend(['--spatial-layers', str(args.spatial_layers), '--dropout', str(args.dropout)])
        if args.no_graph_bias:
            command.append('--no-graph-bias')
        if args.no_residual:
            command.append('--no-residual')
    return command


def build_eval_command(model_key: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        str(args.python),
        str(TRAINING_DIR / 'evaluate_noise_scenarios.py'),
        '--model-type', model_key,
        '--checkpoint', str(output_dir / 'best_model.pth'),
        '--run-config', str(output_dir / 'run_config.json'),
        '--data-root', str(args.data_root),
        '--output-dir', str(output_dir),
        '--batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers),
        '--device', str(args.device),
    ]
    if args.device_ids is not None:
        command.extend(['--device-ids', str(args.device_ids)])
    add_optional_arg(command, '--scenario-limit', args.scenario_limit)
    return command


def collect_model_summary(model_key: str, output_dir: Path) -> dict[str, Any]:
    training_summary = json.loads((output_dir / 'training_summary.json').read_text(encoding='utf-8'))
    test_metrics = json.loads((output_dir / 'test_metrics.json').read_text(encoding='utf-8'))
    scenario_metrics = json.loads((output_dir / 'scenario_metrics.json').read_text(encoding='utf-8'))
    return {
        'model_key': model_key,
        'model_name': MODEL_SPECS[model_key]['name'],
        'output_dir': str(output_dir.resolve()),
        'training_summary': training_summary,
        'test_metrics': test_metrics,
        'scenario_metrics': scenario_metrics,
    }


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        'model_key',
        'model_name',
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


def main() -> None:
    args = parse_args()
    ensure_dataset_exists(args.data_root)
    args.artifacts_root.mkdir(parents=True, exist_ok=True)

    suite_rows: list[dict[str, Any]] = []
    suite_models: list[dict[str, Any]] = []

    for model_key in args.models:
        output_dir = args.artifacts_root / MODEL_SPECS[model_key]['name']
        output_dir.mkdir(parents=True, exist_ok=True)

        run_command(build_train_command(model_key, args, output_dir), cwd=PROJECT_ROOT)
        run_command(build_eval_command(model_key, args, output_dir), cwd=PROJECT_ROOT)

        model_summary = collect_model_summary(model_key, output_dir)
        suite_models.append(model_summary)

        for scenario_slug, scenario_payload in model_summary['scenario_metrics']['scenarios'].items():
            suite_rows.append(
                {
                    'model_key': model_key,
                    'model_name': MODEL_SPECS[model_key]['name'],
                    'scenario_slug': scenario_slug,
                    'scenario_name': scenario_payload['scenario_name'],
                    'sample_count': int(scenario_payload['sample_count']),
                    **scenario_payload['metrics'],
                }
            )

    suite_summary = {
        'project_root': str(PROJECT_ROOT.resolve()),
        'data_root': str(args.data_root.resolve()),
        'artifacts_root': str(args.artifacts_root.resolve()),
        'models': suite_models,
        'comparison_rows': suite_rows,
    }
    save_json(args.artifacts_root / 'suite_summary.json', suite_summary)
    write_comparison_csv(args.artifacts_root / 'scenario_comparison.csv', suite_rows)
    save_json(args.artifacts_root / 'scenario_comparison.json', suite_rows)
    print(f"Suite summary written to: {args.artifacts_root}")


if __name__ == '__main__':
    main()
