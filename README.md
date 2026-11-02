# RA-ST-DiffTEM

Official research code and a compact public data sample for **RA-ST-DiffTEM:
Reliability-Aware Complementary Anchoring for Multi-Receiver Transient
Electromagnetic Denoising**.

The repository contains the proposed model, PCBiLSTM and external baselines,
training/evaluation programs, and a 20-sample demonstration set covering all
ten fixed test scenarios. Each sample has 11 receivers and 869 time gates.
Path-free benchmark split, noise-recipe, scenario, and receiver-geometry
manifests are included under `data/metadata/`.

## Installation

From this project directory, in an activated Python environment:

```bash
python -m pip install -e ".[dev]"
```

PyTorch installation can be adjusted for the CUDA version on your machine.

## Quick check

```bash
python scripts/inspect_demo_data.py
python -m pytest
```

The demo archive contains:

- `noisy`: noisy multi-receiver TEM observations, shape `(20, 11, 869)`;
- `clean`: paired clean targets, shape `(20, 11, 869)`;
- `scenario`: scenario identifier for each pair.

## Training and evaluation

The original experiment entry points are retained in `training/`. Run:

```bash
python training/train_st_difftem.py --help
python training/evaluate_noise_scenarios.py --help
```

Paths are command-line arguments; no private workstation paths are required.
See [data/README.md](data/README.md) for the full benchmark layout.

The revision's additional comparisons have runnable entry points:

```bash
python scripts/evaluate_conventional_baselines.py --data-root data/full --output-dir artifacts/conventional
bash training/run_extended_baselines.sh
```

The extended-baseline launcher reads `DATA_ROOT` (default `data/full`),
`OUT_ROOT`, `CODE_ROOT`, `PYTHON`, and `DEVICE_IDS` (default `0`, one GPU) from
the environment. On Windows, the extended baselines and receiver-geometry
ablation can be launched with the commands below. Use `-DeviceIds` to select
another GPU:

```powershell
./training/run_extended_baselines.ps1 -DataRoot data/full -OutputRoot artifacts/extended_baselines
./training/run_receiver_geometry_ablation.ps1 -DataRoot data/full -OutputRoot artifacts/receiver_geometry
```

`scripts/prepare_receiver_geometry_ablation.py` creates the receiver subsets
and writes only relative input-file paths in its generated manifest.
These commands require the complete benchmark arrays and start training or
evaluation; the 20-sample demo is only for format checks.

The saved sample-weighted seed summary supports a lightweight check of the
reported paired statistics without training:

```bash
python scripts/paired_seed_statistics_aligned.py \
  --summary-path data/metadata/completed_revision_summary_sample_weighted.json \
  --output-dir artifacts/paired_statistics
```

`scripts/model_cluster_bootstrap.py` reproduces clean-model-cluster confidence
intervals when the full per-sample metric files are available. It accepts the
comparator and final run-directory patterns, the matching test recipe manifest,
the scenario name, and an output path as command-line arguments.

## Final output calibration

`models/output_strength_calibrator.py` contains the output-strength calibration
stage of the final reported RA-ST-DiffTEM pipeline. It acts on frozen
log-magnitude network predictions. The default predicts one
denoising strength for each receiver-time matrix; `--receiver-wise` fits a
separate strength for every receiver trace. Both use the noisy input and model prediction;
clean targets and noise-scenario names are used only while fitting on
validation data. The manuscript uses the receiver-wise stage for its final
synthetic benchmark results and reports the network without calibration as an
ablation. The preserved measured-field image uses the network output before
this calibration stage.

Install the calibration dependencies with `python -m pip install -e ".[calibration]"`.
Supply a JSON manifest containing a `datasets` list. Each entry needs `name`,
`role` (`validation` or `matched_validation`), `noisy`, `clean`, `prediction`,
and `model_records_jsonl` paths. `noisy` and `clean` are physical-domain `.npy`
arrays; `prediction` is the frozen model's log10-magnitude `.npy` output. Each
JSONL record needs the corresponding `model_name` so five-fold validation can
keep all corruptions of one clean model in the same fold. Then run:

```bash
python scripts/fit_output_strength_calibrator.py \
  --manifest artifacts/calibration_manifest.json \
  --output-dir artifacts/output_calibration \
  --receiver-wise
```

The script writes a fitted `output_strength_calibrator.joblib` and grouped-CV
summary. `models.output_strength_calibrator.apply_calibrator` applies the
sample-level model if `--receiver-wise` is omitted; `apply_receiver_calibrator`
applies the receiver-wise model shown above.
For the reported final synthetic pipeline, use
`apply_receiver_calibrator(noisy_log, model_output_log, regressor,
early_only=True, savgol_window=101)`. The calibration and polynomial smoothing
taper between 65% and 70% of the gates, leaving the final 30% equal to the
frozen model output. This protects the late-time log-MAE measured in the paper.
The same operation is available as a command-line step after saving the frozen
model's log-domain predictions:

```bash
python scripts/apply_receiver_calibration.py \
  --noisy data/test_scenarios/light_gaussian/light_gaussian_data.npy \
  --prediction artifacts/light_gaussian_prediction_log10.npy \
  --calibrator artifacts/output_calibration/output_strength_calibrator.joblib \
  --output artifacts/light_gaussian_calibrated_log10.npy
```

Load only a trusted local `.joblib` file. Fit and inference use
scikit-learn 1.6.1 for the reported calibration results.
The full synthetic dataset and pretrained model weights are required to
reproduce publication-scale experiments; the 20-sample demo is insufficient.

## Reproducibility scope

The checked benchmark comprises 25,000 paired synthetic samples: 20,000 for
training, 2,500 for validation, and 2,500 for testing. The fixed-scenario test
suite contains ten noise conditions. The small sample committed here is meant
for format checks and smoke tests, not for reproducing paper-level scores.
Trained checkpoints are not bundled with this compact code package; reproducing
the reported scores requires the full benchmark data and the stated training runs.

## License and citation

Code is released under the MIT License. Data files in `data/` are released
under CC BY 4.0; see `DATA_LICENSE`.

If this work is useful, please cite it using `CITATION.cff`.
