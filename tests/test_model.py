import torch

from models.st_difftem import STDiffTEMNet


def test_model_import_and_construction():
    model = STDiffTEMNet()
    assert isinstance(model, torch.nn.Module)
