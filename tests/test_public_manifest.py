"""The shipped scenario manifest must load through the evaluation entry point."""

from pathlib import Path

from training.evaluate_noise_scenarios import load_scenario_specs


def test_public_scenario_manifest_loads() -> None:
    metadata = Path(__file__).resolve().parents[1] / "data" / "metadata"
    scenarios = load_scenario_specs(metadata)
    assert len(scenarios) == 10
    assert len({row["slug"] for row in scenarios}) == 10
    assert sum(row["sample_count"] for row in scenarios) == 2500
    assert all("data_path" not in row and "labels_path" not in row for row in scenarios)
