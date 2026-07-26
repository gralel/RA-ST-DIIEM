"""Summarize A/B/C supplement experiment scenario metrics across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


CHALLENGING = [
    "low_snr_gaussian",
    "dropout_missing_segment",
    "local_receiver_corruption",
    "mixed_complex_noise",
    "unseen_noise_combination",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_run(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "scenario_metrics.json"
    if not path.exists():
        return None
    payload = load_json(path)
    row: dict[str, float] = {}
    combined = payload.get("combined_test")
    if combined and combined.get("metrics"):
        row["overall"] = float(combined["metrics"]["rrmse"])
    scenarios = payload.get("scenarios", {})
    for slug, item in scenarios.items():
        metrics = item.get("metrics", {})
        if "rrmse" in metrics:
            row[str(slug)] = float(metrics["rrmse"])
    if all(slug in row for slug in CHALLENGING):
        row["challenging_avg"] = float(mean(row[slug] for slug in CHALLENGING))
    return row


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def collect_group(name: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        run_dir = seed_dir / "hybrid_base" if (seed_dir / "hybrid_base").exists() else seed_dir
        metrics = collect_run(run_dir)
        if metrics is None:
            continue
        rows.append({"group": name, "seed": seed_dir.name, **metrics})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=Path("训练结果/external_baselines_20260513"))
    parser.add_argument(
        "--direct-root",
        type=Path,
        default=Path("训练结果/residual_refiner_ablation_20260513/direct_residual_nodiff"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("训练结果/abc_supplement_summary_20260513"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_runs: list[dict[str, Any]] = []
    all_runs.extend(collect_group("tem1dformer869", args.external_root / "tem1dformer869"))
    all_runs.extend(collect_group("dncnn2d", args.external_root / "dncnn2d"))
    all_runs.extend(collect_group("direct_residual_nodiff", args.direct_root))

    metric_keys = sorted({key for row in all_runs for key in row.keys() if key not in {"group", "seed"}})
    summary_rows: list[dict[str, Any]] = []
    for group in sorted({row["group"] for row in all_runs}):
        group_rows = [row for row in all_runs if row["group"] == group]
        out: dict[str, Any] = {"group": group, "n": len(group_rows)}
        for key in metric_keys:
            vals = [float(row[key]) for row in group_rows if key in row]
            avg, std = summarize(vals)
            out[f"{key}_mean"] = avg
            out[f"{key}_std"] = std
        summary_rows.append(out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "abc_runs.json").write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "abc_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with (args.output_dir / "abc_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Wrote summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
