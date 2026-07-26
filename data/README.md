# Data

## Committed demonstration data

`sample/fixed_scenarios_demo.npz` contains two paired examples from each fixed
noise scenario. Load it with:

```python
import numpy as np
d = np.load("data/sample/fixed_scenarios_demo.npz")
noisy, clean, scenario = d["noisy"], d["clean"], d["scenario"]
```

## Full benchmark layout

The complete benchmark is approximately 2.25 GB and is intentionally excluded
from Git history because GitHub rejects ordinary files larger than 100 MB.
After downloading the full archive, place it under `data/full/`:

```text
data/full/
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
