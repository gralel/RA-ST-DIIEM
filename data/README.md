# Data

## Committed demonstration data

`sample/fixed_scenarios_demo.npz` contains two paired examples from each fixed
noise scenario. Load it with:

```python
import numpy as np
d = np.load("data/sample/fixed_scenarios_demo.npz")
noisy, clean, scenario = d["noisy"], d["clean"], d["scenario"]
```

## Benchmark metadata

The `metadata/` directory includes the clean-model split assignments,
train/validation/test synthetic-noise recipe rows, the eight held-out recipe
combinations, fixed test-scenario definitions, and receiver subsets used in the
geometry ablation. It also includes the sample-weighted five-seed result summary
used by the paired-sign and bootstrap statistics script. Recipe rows are ordered like the corresponding paired arrays
in the full benchmark. Absolute workstation paths and measured-field records
are excluded from these public manifests.

These files document the experimental partitions and settings. They do not
contain the complete noisy-clean arrays or trained checkpoints, so they cannot
by themselves reproduce the reported scores.

## Full benchmark layout

The complete benchmark is approximately 2.25 GB and is intentionally excluded
from Git history because GitHub rejects ordinary files larger than 100 MB.
After downloading the full archive, place it under `data/full/`:

```text
data/full/
  test_scenario_manifest.json
  train/
  val/
  test/
  test_scenarios/
    colored_background/
    drift/
    dropout_missing_segment/
    em_interference/
    impulse_spike/
    light_gaussian/
    local_receiver_corruption/
    low_snr_gaussian/
    mixed_complex_noise/
    unseen_noise_combination/
```

Copy `metadata/test_scenario_manifest.json` to `data/full/test_scenario_manifest.json`
if it is not already included in the full archive. The conventional-baseline
and neural evaluators read its `scenarios` list for the ten scenario names and
counts, then locate arrays under `data/full/test_scenarios/<slug>/`. The public
manifest omits private workstation paths.

The public release should use split GitHub Release assets or an archival
service such as Zenodo. A permanent download URL can then be added here
without changing the code.

## Dataset summary

- 25,000 paired samples
- 11 receivers
- 869 time gates
- train/validation/test: 20,000/2,500/2,500
- generation seed: 20260428
- no model-level overlap across the reported splits

Data are licensed under CC BY 4.0.
