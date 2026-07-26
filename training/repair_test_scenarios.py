"""Rebuild missing test_scenarios files from the combined test split.

This is useful when uploading to a server and some per-scenario files are missing,
while the combined test arrays and recipe log are present.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / '数据' / 'data_paper_fixed_scenarios_25k'


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--force', action='store_true', help='Overwrite existing per-scenario files.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root

    combined_root = data_root / 'test'
    scenario_root = data_root / 'test_scenarios'
    manifest_path = data_root / 'test_scenario_manifest.json'
    recipes_path = combined_root / 'test_recipes.jsonl'
    data_path = combined_root / 'test_data.npy'
    labels_path = combined_root / 'test_labels.npy'
    clean_base_path = combined_root / 'test_clean_base.npy'
    model_names_path = combined_root / 'test_clean_model_names.json'

    for path in (manifest_path, recipes_path, data_path, labels_path):
        ensure_exists(path)

    manifest = load_json(manifest_path)
    scenario_specs = list(manifest.get('scenarios', []))
    if not scenario_specs:
        raise ValueError(f'No scenarios found in manifest: {manifest_path}')

    records_by_slug: dict[str, list[dict[str, Any]]] = {}
    indices_by_slug: dict[str, list[int]] = {}
    for spec in scenario_specs:
        slug = str(spec['slug'])
        records_by_slug[slug] = []
        indices_by_slug[slug] = []

    with recipes_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            slug = row.get('scenario_slug')
            sample_idx = row.get('sample_idx')
            if slug is None:
                continue
            slug = str(slug)
            if slug not in records_by_slug:
                records_by_slug[slug] = []
                indices_by_slug[slug] = []
            records_by_slug[slug].append(row)
            indices_by_slug[slug].append(int(sample_idx))

    data = np.load(data_path, mmap_mode='r')
    labels = np.load(labels_path, mmap_mode='r')

    if data.shape != labels.shape:
        raise ValueError(f'Combined test data/labels shape mismatch: {data.shape} vs {labels.shape}')

    scenario_root.mkdir(parents=True, exist_ok=True)

    for spec in scenario_specs:
        slug = str(spec['slug'])
        expected_count = int(spec['sample_count'])
        rows = records_by_slug.get(slug, [])
        indices = indices_by_slug.get(slug, [])
        if len(indices) != expected_count:
            raise ValueError(f'{slug}: expected {expected_count} records, found {len(indices)} in test_recipes.jsonl')

        scenario_dir = scenario_root / slug
        scenario_dir.mkdir(parents=True, exist_ok=True)

        out_data = scenario_dir / f'{slug}_data.npy'
        out_labels = scenario_dir / f'{slug}_labels.npy'
        out_recipes = scenario_dir / f'{slug}_recipes.jsonl'
        out_clean_base = scenario_dir / f'{slug}_clean_base.npy'
        out_model_names = scenario_dir / f'{slug}_clean_model_names.json'

        if (not args.force) and out_data.exists() and out_labels.exists() and out_recipes.exists():
            print(f'[skip] {slug} already exists')
            continue

        scenario_data = np.asarray(data[indices], dtype=np.float32)
        scenario_labels = np.asarray(labels[indices], dtype=np.float32)
        np.save(out_data, scenario_data)
        np.save(out_labels, scenario_labels)

        with out_recipes.open('w', encoding='utf-8') as handle:
            for local_idx, row in enumerate(rows):
                row_copy = dict(row)
                row_copy['sample_idx'] = int(local_idx)
                handle.write(json.dumps(row_copy, ensure_ascii=False) + '\n')

        if clean_base_path.exists():
            shutil.copy2(clean_base_path, out_clean_base)
        if model_names_path.exists():
            shutil.copy2(model_names_path, out_model_names)

        print(f'[rebuilt] {slug}: {expected_count} samples')

    print(f'Repair completed under: {scenario_root}')


if __name__ == '__main__':
    main()
