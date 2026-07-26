# RA-ST-DiffTEM

Official research code and a compact public data sample for **RA-ST-DiffTEM:
Reliability-Aware Complementary Anchoring for Multi-Receiver Transient
Electromagnetic Denoising**.

The repository contains the proposed model, PCBiLSTM and external baselines,
training/evaluation programs, and a 20-sample demonstration set covering all
ten fixed test scenarios. Each sample has 11 receivers and 869 time gates.

## Installation

```bash
git clone https://github.com/OWNER/RA-ST-DiffTEM.git
cd RA-ST-DiffTEM
python -m venv .venv
pip install -e .
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

## Reproducibility scope

The checked benchmark comprises 25,000 paired synthetic samples: 20,000 for
training, 2,500 for validation, and 2,500 for testing. The fixed-scenario test
suite contains ten noise conditions. The small sample committed here is meant
for format checks and smoke tests, not for reproducing paper-level scores.

## License and citation

Code is released under the MIT License. Data files in `data/` are released
under CC BY 4.0; see `DATA_LICENSE`.

If this work is useful, please cite it using `CITATION.cff`.
