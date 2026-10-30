from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


ORIGINAL_SPACING_M = 50.0


def parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("at least one receiver index is required")
    if indices != sorted(set(indices)):
        raise argparse.ArgumentTypeError("receiver indices must be unique and increasing")
    if indices[0] < 0 or indices[-1] >= 11:
        raise argparse.ArgumentTypeError("receiver indices must lie in [0, 10]")
    return indices


def subset_array(source: Path, destination: Path, indices: list[int]) -> dict[str, object]:
    array = np.load(source, mmap_mode="r")
    original_shape = list(array.shape)
    if array.ndim == 3 and array.shape[1] == 11:
        selected = np.asarray(array[:, indices, :])
        np.save(destination, selected)
        action = "receiver_subset"
        output_shape = list(selected.shape)
    else:
        shutil.copy2(source, destination)
        action = "copied_unchanged"
        output_shape = original_shape
    return {
        "action": action,
        "original_shape": original_shape,
        "output_shape": output_shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a receiver-subset view of the fixed 11-receiver TEM benchmark."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--indices", type=parse_indices, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        destination = output_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".npy":
            record = subset_array(source, destination, args.indices)
            record["relative_path"] = relative.as_posix()
            records.append(record)
        else:
            shutil.copy2(source, destination)

    positions_m = [float((index - 5) * ORIGINAL_SPACING_M) for index in args.indices]
    adjacent_spacings_m = [
        positions_m[index + 1] - positions_m[index]
        for index in range(len(positions_m) - 1)
    ]
    metadata = {
        "receiver_indices_zero_based": args.indices,
        "receiver_indices_one_based": [index + 1 for index in args.indices],
        "receiver_count": len(args.indices),
        "receiver_positions_m_relative_to_center": positions_m,
        "adjacent_spacings_m": adjacent_spacings_m,
        "uniform_spacing_m": (
            adjacent_spacings_m[0]
            if adjacent_spacings_m
            and all(value == adjacent_spacings_m[0] for value in adjacent_spacings_m)
            else None
        ),
        "files": records,
    }
    (output_root / "receiver_geometry_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
