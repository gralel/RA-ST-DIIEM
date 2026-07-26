from pathlib import Path

import numpy as np


path = Path(__file__).parents[1] / "data" / "sample" / "fixed_scenarios_demo.npz"
with np.load(path) as data:
    for key in data.files:
        value = data[key]
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    assert data["noisy"].shape == data["clean"].shape == (20, 11, 869)
    assert len(set(data["scenario"].tolist())) == 10
