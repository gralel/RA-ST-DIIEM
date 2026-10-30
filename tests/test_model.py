import torch

from models.st_difftem import STDiffTEMNet
from models.external_baselines import build_external_baseline


def test_model_import_and_construction():
    model = STDiffTEMNet()
    assert isinstance(model, torch.nn.Module)


def test_reviewer_requested_external_baselines_forward_and_backward():
    x = torch.randn(2, 3, 64)
    common = {
        "seq_len": 64,
        "num_receivers": 3,
        "hidden_channels": 8,
        "depth": 3,
        "filters": 4,
        "lstm_hidden": 8,
        "lstm_layers": 1,
        "num_heads": 4,
        "spatial_layers": 1,
        "gate_hidden": 8,
    }
    for name in ("temdnet_style", "temnlnet_style", "joint_gated_moe"):
        model = build_external_baseline(name, common)
        prediction = model(x)
        assert prediction.shape == x.shape
        prediction.square().mean().backward()
        assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
