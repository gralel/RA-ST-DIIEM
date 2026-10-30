"""Run Spatial+enhanced-loss and ST-DiffTEM variants, then aggregate results."""

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
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "数据" / "data_paper_fixed_scenarios_25k"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "训练结果" / "pcbilstm_baselines_suite"
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "训练结果" / "st_difftem_suite"
DEFAULT_SPATIAL_ENHANCED_ROOT = PROJECT_ROOT / "训练结果" / "spatial_enhanced_loss"

BASELINE_SPECS = {
    "single": "single_receiver_pcbilstm",
    "multi": "multi_receiver_pcbilstm",
    "spatial": "spatial_receiver_pcbilstm",
}
DIFFUSION_VARIANTS = ("spatial_base", "hybrid_base")
CHALLENGING_SCENARIOS = {
    "low_snr_gaussian",
    "local_receiver_corruption",
    "dropout_missing_segment",
    "mixed_complex_noise",
    "unseen_noise_combination",
}


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], cwd: Path) -> None:
    printable = " ".join(f'"{item}"' if " " in item else item for item in command)
    print(f"[run] {printable}")
    subprocess.run(command, cwd=str(cwd), check=True)


def add_optional_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def training_paths(data_root: Path) -> dict[str, Path]:
    return {
        "train_data": data_root / "train" / "train_data.npy",
        "train_labels": data_root / "train" / "train_labels.npy",
        "val_data": data_root / "val" / "val_data.npy",
        "val_labels": data_root / "val" / "val_labels.npy",
        "test_data": data_root / "test" / "test_data.npy",
        "test_labels": data_root / "test" / "test_labels.npy",
    }


def ensure_dataset_exists(data_root: Path) -> None:
    missing = [str(path) for path in training_paths(data_root).values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))


def ensure_baselines_exist(baseline_root: Path) -> None:
    missing: list[str] = []
    # The reliability model consumes the independent and spatial experts only.
    # The multi-receiver baseline is optional for reduced-geometry ablations.
    for model_key in ("single", "spatial"):
        model_dir = BASELINE_SPECS[model_key]
        root = baseline_root / model_dir
        for name in ("training_summary.json", "test_metrics.json", "scenario_metrics.json"):
            path = root / name
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing baseline artifacts:\n" + "\n".join(missing))


def parse_variants(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one diffusion variant must be provided.")
    for item in items:
        if item not in DIFFUSION_VARIANTS:
            raise ValueError(f"Unsupported diffusion variant: {item}")
    unique: list[str] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--artifacts-root", "--output-root", dest="artifacts_root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--spatial-enhanced-root", type=Path, default=DEFAULT_SPATIAL_ENHANCED_ROOT)
    parser.add_argument(
        "--skip-spatial-enhanced",
        action="store_true",
        help="Skip the unrelated spatial enhanced-loss ablation.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--variants", type=str, default="spatial_base,hybrid_base")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device-ids", type=str, help="Comma-separated CUDA device ids, e.g. '0,1'.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--selection-metric", choices=("loss", "rmse", "rrmse", "snr"), default="rrmse")
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-rel", type=float, default=0.1)
    parser.add_argument(
        "--relative-loss-mode",
        choices=("global_energy", "elementwise"),
        default="global_energy",
    )
    parser.add_argument("--lambda-gate", type=float, default=0.001)
    parser.add_argument("--lambda-residual-x0", type=float, default=0.5)
    parser.add_argument("--lambda-anchor-residual", type=float, default=0.5)
    parser.add_argument("--lambda-anchor-oracle", type=float, default=0.2)
    parser.add_argument("--lambda-pred-oracle", "--lambda-baseline-protect", dest="lambda_pred_oracle", type=float, default=0.1)
    parser.add_argument("--lambda-router", type=float, default=0.001)
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
    parser.add_argument("--cache-batch-size", type=int, default=256)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--scenario-limit", type=int)
    parser.add_argument("--robust-score-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--prepare-cache-only", action="store_true")
    return parser.parse_args()


def build_spatial_enhanced_train_command(args: argparse.Namespace) -> list[str]:
    data_paths = training_paths(args.data_root)
    command = [
        str(args.python),
        str(TRAINING_DIR / "train_spatial_receiver_pcbilstm.py"),
        "--train-data", str(data_paths["train_data"]),
        "--train-labels", str(data_paths["train_labels"]),
        "--val-data", str(data_paths["val_data"]),
        "--val-labels", str(data_paths["val_labels"]),
        "--test-data", str(data_paths["test_data"]),
        "--test-labels", str(data_paths["test_labels"]),
        "--output-dir", str(args.spatial_enhanced_root),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--max-epochs", str(args.max_epochs),
        "--min-epochs", str(args.min_epochs),
        "--patience", str(args.patience),
        "--warmup-epochs", str(args.warmup_epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--selection-metric", str(args.selection_metric),
        "--loss-profile", "enhanced",
        "--lambda-rec", str(args.lambda_rec),
        "--lambda-rel", str(args.lambda_rel),
        "--relative-loss-mode", str(args.relative_loss_mode),
        "--seq-len", str(args.seq_len),
        "--num-receivers", str(args.num_receivers),
        "--num-heads", str(args.num_heads),
        "--dropout", str(args.dropout),
        "--device", str(args.device),
        "--seed", str(args.seed),
    ]
    if args.device_ids is not None:
        command.extend(["--device-ids", str(args.device_ids)])
    add_optional_arg(command, "--train-limit", args.train_limit)
    add_optional_arg(command, "--val-limit", args.val_limit)
    add_optional_arg(command, "--test-limit", args.test_limit)
    return command


def build_eval_command(
    args: argparse.Namespace,
    model_type: str,
    checkpoint: Path,
    run_config: Path,
    output_dir: Path,
    base_cache_dir: Path | None = None,
) -> list[str]:
    command = [
        str(args.python),
        str(TRAINING_DIR / "evaluate_noise_scenarios.py"),
        "--model-type", model_type,
        "--checkpoint", str(checkpoint),
        "--run-config", str(run_config),
        "--data-root", str(args.data_root),
        "--output-dir", str(output_dir),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", str(args.device),
    ]
    if base_cache_dir is not None:
        command.extend(["--base-cache-dir", str(base_cache_dir)])
    if args.device_ids is not None:
        command.extend(["--device-ids", str(args.device_ids)])
    add_optional_arg(command, "--scenario-limit", args.scenario_limit)
    return command


def build_st_difftem_train_command(
    args: argparse.Namespace,
    variant: str,
    output_dir: Path,
) -> list[str]:
    data_paths = training_paths(args.data_root)
    command = [
        str(args.python),
        str(TRAINING_DIR / "train_st_difftem.py"),
        "--train-data", str(data_paths["train_data"]),
        "--train-labels", str(data_paths["train_labels"]),
        "--val-data", str(data_paths["val_data"]),
        "--val-labels", str(data_paths["val_labels"]),
        "--test-data", str(data_paths["test_data"]),
        "--test-labels", str(data_paths["test_labels"]),
        "--output-dir", str(output_dir),
        "--variant", variant,
        "--spatial-artifact-dir", str(args.baseline_root / BASELINE_SPECS["spatial"]),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--max-epochs", str(args.max_epochs),
        "--min-epochs", str(args.min_epochs),
        "--patience", str(args.patience),
        "--warmup-epochs", str(args.warmup_epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--selection-metric", str(args.selection_metric),
        "--lambda-rec", str(args.lambda_rec),
        "--lambda-rel", str(args.lambda_rel),
        "--lambda-gate", str(args.lambda_gate),
        "--lambda-residual-x0", str(args.lambda_residual_x0),
        "--lambda-anchor-residual", str(args.lambda_anchor_residual),
        "--lambda-anchor-oracle", str(args.lambda_anchor_oracle),
        "--lambda-pred-oracle", str(args.lambda_pred_oracle),
        "--lambda-router", str(args.lambda_router),
        "--oracle-margin", str(args.oracle_margin),
        "--condition-feature-dim", str(args.condition_feature_dim),
        "--condition-hidden-channels", str(args.condition_hidden_channels),
        "--condition-spatial-layers", str(args.condition_spatial_layers),
        "--gate-hidden-dim", str(args.gate_hidden_dim),
        "--diffusion-hidden-channels", str(args.diffusion_hidden_channels),
        "--diffusion-steps", str(args.diffusion_steps),
        "--sample-steps", str(args.sample_steps),
        "--beta-start", str(args.beta_start),
        "--beta-end", str(args.beta_end),
        "--step-dim", str(args.step_dim),
        "--num-heads", str(args.num_heads),
        "--seq-len", str(args.seq_len),
        "--num-receivers", str(args.num_receivers),
        "--dropout", str(args.dropout),
        "--refine-scale", str(args.refine_scale),
        "--sampler-seed", str(args.sampler_seed),
        "--prediction-type", str(args.prediction_type),
        "--residual-inference-mode", str(args.residual_inference_mode),
        "--residual-anchor-mode", str(args.residual_anchor_mode),
        "--residual-clip-value", str(args.residual_clip_value),
        "--max-residual-gate", str(args.max_residual_gate),
        "--rho-max", str(args.rho_max),
        "--residual-scale-floor", str(args.residual_scale_floor),
        "--temporal-gate-init-bias", str(args.temporal_gate_init_bias),
        "--cache-batch-size", str(args.cache_batch_size),
        "--device", str(args.device),
        "--seed", str(args.seed),
    ]
    if args.residual_scale_by_w_single:
        command.append("--residual-scale-by-w-single")
    else:
        command.append("--no-residual-scale-by-w-single")
    if args.disable_temporal_gate:
        command.append("--disable-temporal-gate")
    if args.disable_diffusion_residual:
        command.append("--disable-diffusion-residual")
    if args.direct_residual_nodiff:
        command.append("--direct-residual-nodiff")
    if variant == "hybrid_base":
        command.extend(["--single-artifact-dir", str(args.baseline_root / BASELINE_SPECS["single"])])
    if args.device_ids is not None:
        command.extend(["--device-ids", str(args.device_ids)])
    add_optional_arg(command, "--train-limit", args.train_limit)
    add_optional_arg(command, "--val-limit", args.val_limit)
    add_optional_arg(command, "--test-limit", args.test_limit)
    if args.prepare_cache_only:
        command.append("--prepare-cache-only")
    return command


def collect_model_summary(model_key: str, model_name: str, output_dir: Path) -> dict[str, Any]:
    training_summary = load_json(output_dir / "training_summary.json")
    test_metrics = load_json(output_dir / "test_metrics.json")
    scenario_metrics = load_json(output_dir / "scenario_metrics.json")
    return {
        "model_key": model_key,
        "model_name": model_name,
        "output_dir": str(output_dir.resolve()),
        "training_summary": training_summary,
        "test_metrics": test_metrics,
        "scenario_metrics": scenario_metrics,
    }


def iter_scenario_rows(model_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_slug, scenario_payload in model_summary["scenario_metrics"]["scenarios"].items():
        rows.append(
            {
                "model_key": model_summary["model_key"],
                "model_name": model_summary["model_name"],
                "scenario_slug": scenario_slug,
                "scenario_name": scenario_payload["scenario_name"],
                "sample_count": int(scenario_payload["sample_count"]),
                **scenario_payload["metrics"],
            }
        )
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average_rows(rows: list[dict[str, Any]], weighted: bool) -> dict[str, float]:
    metrics = ("loss", "rmse", "rrmse", "snr")
    if not rows:
        return {metric: float("nan") for metric in metrics}
    if not weighted:
        return {metric: float(sum(float(row[metric]) for row in rows) / len(rows)) for metric in metrics}
    total_weight = float(sum(int(row["sample_count"]) for row in rows))
    return {
        metric: float(sum(float(row[metric]) * int(row["sample_count"]) for row in rows) / total_weight)
        for metric in metrics
    }


def compute_target_robust_summary(
    model_summary: dict[str, Any],
    robust_score_alpha: float,
) -> dict[str, float | None]:
    scenarios = dict(model_summary["scenario_metrics"].get("scenarios", {}))
    target_payloads = [scenarios[slug] for slug in CHALLENGING_SCENARIOS if slug in scenarios]
    if not target_payloads:
        return {
            "mean_target_rrmse": None,
            "max_target_rrmse": None,
            "robust_score": None,
            "target_mean_oracle_regret": None,
        }
    target_rrmses = [float(payload["metrics"]["rrmse"]) for payload in target_payloads]
    oracle_regrets = [
        float(payload["analysis"]["oracle_regret"])
        for payload in target_payloads
        if payload.get("analysis", {}).get("oracle_regret") is not None
    ]
    mean_target_rrmse = float(sum(target_rrmses) / len(target_rrmses))
    max_target_rrmse = float(max(target_rrmses))
    return {
        "mean_target_rrmse": mean_target_rrmse,
        "max_target_rrmse": max_target_rrmse,
        "robust_score": float(mean_target_rrmse + robust_score_alpha * max_target_rrmse),
        "target_mean_oracle_regret": (
            float(sum(oracle_regrets) / len(oracle_regrets))
            if oracle_regrets
            else None
        ),
    }


def _empty_robust_fields() -> dict[str, float | None]:
    return {
        "mean_target_rrmse": None,
        "max_target_rrmse": None,
        "robust_score": None,
        "target_mean_oracle_regret": None,
    }


def build_method_averages(
    model_summary: dict[str, Any],
    scenario_rows: list[dict[str, Any]],
    robust_score_alpha: float,
) -> list[dict[str, Any]]:
    overall_metrics = dict(model_summary["test_metrics"]["metrics"])
    robust_summary = compute_target_robust_summary(model_summary, robust_score_alpha)
    rows = [
        {
            "model_key": model_summary["model_key"],
            "model_name": model_summary["model_name"],
            "average_type": "overall_test",
            **overall_metrics,
            **_empty_robust_fields(),
        }
    ]
    rows.append(
        {
            "model_key": model_summary["model_key"],
            "model_name": model_summary["model_name"],
            "average_type": "scenario_macro_average",
            **average_rows(scenario_rows, weighted=False),
            **_empty_robust_fields(),
        }
    )
    challenging_rows = [row for row in scenario_rows if row["scenario_slug"] in CHALLENGING_SCENARIOS]
    rows.append(
        {
            "model_key": model_summary["model_key"],
            "model_name": model_summary["model_name"],
            "average_type": "challenging_macro_average",
            **average_rows(challenging_rows, weighted=False),
            **_empty_robust_fields(),
        }
    )
    rows.append(
        {
            "model_key": model_summary["model_key"],
            "model_name": model_summary["model_name"],
            "average_type": "challenging_micro_average",
            **average_rows(challenging_rows, weighted=True),
            **_empty_robust_fields(),
        }
    )
    rows.append(
        {
            "model_key": model_summary["model_key"],
            "model_name": model_summary["model_name"],
            "average_type": "target_robust_summary",
            "loss": None,
            "rmse": None,
            "rrmse": None,
            "snr": None,
            **robust_summary,
        }
    )
    return rows


def index_rows(rows: list[dict[str, Any]], key_field: str, value_field: str | None = None) -> dict[str, dict[str, Any] | Any]:
    mapping: dict[str, dict[str, Any] | Any] = {}
    for row in rows:
        if value_field is None:
            mapping[str(row[key_field])] = row
        else:
            mapping[str(row[key_field])] = row[value_field]
    return mapping


def rename_rows(rows: list[dict[str, Any]], old_key: str, new_key: str, new_name: str) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        if row_copy["model_key"] == old_key:
            row_copy["source_model_key"] = old_key
            row_copy["source_model_name"] = row_copy["model_name"]
            row_copy["model_key"] = new_key
            row_copy["model_name"] = new_name
        remapped.append(row_copy)
    return remapped


def select_best_diffusion_variant(
    diffusion_summaries: dict[str, dict[str, Any]],
    robust_score_alpha: float,
) -> dict[str, Any]:
    def metric_value(summary: dict[str, Any]) -> float:
        robust_summary = compute_target_robust_summary(summary, robust_score_alpha)
        robust_score = robust_summary["robust_score"]
        if robust_score is None:
            return float("inf")
        return float(robust_score)

    ordered = sorted(
        diffusion_summaries.values(),
        key=metric_value,
        reverse=False,
    )
    best = ordered[0]
    best_robust = compute_target_robust_summary(best, robust_score_alpha)
    return {
        "selected_variant_key": best["model_key"],
        "selected_model_name": best["model_name"],
        "selection_metric": "robust_score",
        "robust_score_alpha": robust_score_alpha,
        "selected_metric_value": metric_value(best),
        "selected_overall_rrmse": float(best["test_metrics"]["metrics"]["rrmse"]),
        **best_robust,
        "candidates": [
            {
                "variant_key": summary["model_key"],
                "model_name": summary["model_name"],
                "metric_value": metric_value(summary),
                "overall_rrmse": float(summary["test_metrics"]["metrics"]["rrmse"]),
                **compute_target_robust_summary(summary, robust_score_alpha),
            }
            for summary in ordered
        ],
    }


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    ensure_dataset_exists(args.data_root)
    ensure_baselines_exist(args.baseline_root)
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    args.spatial_enhanced_root.mkdir(parents=True, exist_ok=True)

    if not args.prepare_cache_only and not args.skip_spatial_enhanced:
        run_command(build_spatial_enhanced_train_command(args), cwd=PROJECT_ROOT)
        run_command(
            build_eval_command(
                args,
                model_type="spatial",
                checkpoint=args.spatial_enhanced_root / "best_model.pth",
                run_config=args.spatial_enhanced_root / "run_config.json",
                output_dir=args.spatial_enhanced_root,
            ),
            cwd=PROJECT_ROOT,
        )

    diffusion_dirs = {variant: args.artifacts_root / variant for variant in variants}
    for variant, output_dir in diffusion_dirs.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        run_command(build_st_difftem_train_command(args, variant, output_dir), cwd=PROJECT_ROOT)

    if args.prepare_cache_only:
        save_json(
            args.artifacts_root / "suite_summary.json",
            {
                "prepare_cache_only": True,
                "variants": variants,
                "artifacts_root": str(args.artifacts_root.resolve()),
                "spatial_enhanced_root": str(args.spatial_enhanced_root.resolve()),
            },
        )
        print(f"Prepared diffusion caches under: {args.artifacts_root}")
        return

    for variant, output_dir in diffusion_dirs.items():
        run_command(
            build_eval_command(
                args,
                model_type="stdifftem",
                checkpoint=output_dir / "best_model.pth",
                run_config=output_dir / "run_config.json",
                output_dir=output_dir,
                base_cache_dir=output_dir / "cache",
            ),
            cwd=PROJECT_ROOT,
        )

    model_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    method_averages: list[dict[str, Any]] = []

    for model_key, model_name in BASELINE_SPECS.items():
        if not (args.baseline_root / model_name / "test_metrics.json").exists():
            continue
        summary = collect_model_summary(model_key, model_name, args.baseline_root / model_name)
        model_summaries.append(summary)
        scenario_rows = iter_scenario_rows(summary)
        comparison_rows.extend(scenario_rows)
        method_averages.extend(build_method_averages(summary, scenario_rows, args.robust_score_alpha))
    if not args.skip_spatial_enhanced:
        spatial_enhanced_summary = collect_model_summary("spatial_enhanced", "spatial_enhanced_loss", args.spatial_enhanced_root)
        model_summaries.append(spatial_enhanced_summary)
        spatial_enhanced_rows = iter_scenario_rows(spatial_enhanced_summary)
        comparison_rows.extend(spatial_enhanced_rows)
        method_averages.extend(build_method_averages(spatial_enhanced_summary, spatial_enhanced_rows, args.robust_score_alpha))

    diffusion_summaries: dict[str, dict[str, Any]] = {}
    for variant, output_dir in diffusion_dirs.items():
        model_key = f"stdifftem_{variant}"
        model_name = f"stdifftem_{variant}"
        summary = collect_model_summary(model_key, model_name, output_dir)
        diffusion_summaries[model_key] = summary
        model_summaries.append(summary)
        scenario_rows = iter_scenario_rows(summary)
        comparison_rows.extend(scenario_rows)
        method_averages.extend(build_method_averages(summary, scenario_rows, args.robust_score_alpha))

    selection = select_best_diffusion_variant(diffusion_summaries, args.robust_score_alpha)
    best_diffusion_key = str(selection["selected_variant_key"])

    paper_rows = [
        row for row in comparison_rows
        if row["model_key"] in {"single", "multi", "spatial", "spatial_enhanced", best_diffusion_key}
    ]
    paper_averages = [
        row for row in method_averages
        if row["model_key"] in {"single", "multi", "spatial", "spatial_enhanced", best_diffusion_key}
    ]
    paper_rows = rename_rows(paper_rows, best_diffusion_key, "stdifftem", "stdifftem")
    paper_averages = rename_rows(paper_averages, best_diffusion_key, "stdifftem", "stdifftem")

    comparison_fieldnames = [
        "model_key",
        "model_name",
        "scenario_slug",
        "scenario_name",
        "sample_count",
        "loss",
        "rmse",
        "rrmse",
        "snr",
        "mean_target_rrmse",
        "max_target_rrmse",
        "robust_score",
        "target_mean_oracle_regret",
    ]
    average_fieldnames = [
        "model_key",
        "model_name",
        "average_type",
        "loss",
        "rmse",
        "rrmse",
        "snr",
        "mean_target_rrmse",
        "max_target_rrmse",
        "robust_score",
        "target_mean_oracle_regret",
    ]
    write_rows_csv(args.artifacts_root / "combined_scenario_comparison.csv", comparison_rows, comparison_fieldnames)
    write_rows_csv(args.artifacts_root / "method_averages.csv", method_averages, average_fieldnames)
    write_rows_csv(args.artifacts_root / "paper_scenario_comparison.csv", paper_rows, comparison_fieldnames + ["source_model_key", "source_model_name"])
    write_rows_csv(args.artifacts_root / "paper_method_averages.csv", paper_averages, average_fieldnames + ["source_model_key", "source_model_name"])
    save_json(args.artifacts_root / "combined_scenario_comparison.json", comparison_rows)
    save_json(args.artifacts_root / "method_averages.json", method_averages)
    save_json(args.artifacts_root / "paper_scenario_comparison.json", paper_rows)
    save_json(args.artifacts_root / "paper_method_averages.json", paper_averages)
    save_json(
        args.artifacts_root / "suite_summary.json",
        {
            "project_root": str(PROJECT_ROOT.resolve()),
            "data_root": str(args.data_root.resolve()),
            "baseline_root": str(args.baseline_root.resolve()),
            "artifacts_root": str(args.artifacts_root.resolve()),
            "spatial_enhanced_root": str(args.spatial_enhanced_root.resolve()),
            "variants": variants,
            "robust_score_alpha": args.robust_score_alpha,
            "selection": selection,
            "models": model_summaries,
            "comparison_rows": comparison_rows,
            "method_averages": method_averages,
            "paper_rows": paper_rows,
            "paper_method_averages": paper_averages,
        },
    )
    print(f"Suite summary written to: {args.artifacts_root}")


if __name__ == "__main__":
    main()
