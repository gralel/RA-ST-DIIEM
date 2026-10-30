from __future__ import annotations

import torch

from training.train_st_difftem import compute_relative_amplitude_loss


def test_global_energy_relative_loss_matches_squared_sample_rrmse() -> None:
    target = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ]
    )
    prediction = target + torch.tensor(
        [
            [[0.1, -0.2], [0.3, -0.4]],
            [[0.2, -0.1], [0.4, -0.3]],
        ]
    )

    error_energy = (prediction - target).pow(2).sum(dim=(1, 2))
    signal_energy = target.pow(2).sum(dim=(1, 2))
    expected = (error_energy / (signal_energy + 1e-12)).mean()

    actual = compute_relative_amplitude_loss(prediction, target)

    torch.testing.assert_close(actual, expected)


def test_global_energy_does_not_overweight_a_near_zero_gate() -> None:
    target = torch.tensor([[[1.0, 1e-9]]])
    prediction = torch.tensor([[[1.0, 1e-4]]])

    global_energy = compute_relative_amplitude_loss(
        prediction,
        target,
        mode="global_energy",
    )
    elementwise = compute_relative_amplitude_loss(
        prediction,
        target,
        mode="elementwise",
    )

    assert torch.isfinite(global_energy)
    assert global_energy < 1e-6
    assert elementwise > 1.0


def test_unknown_relative_loss_mode_is_rejected() -> None:
    target = torch.ones(1, 1, 2)

    try:
        compute_relative_amplitude_loss(target, target, mode="unknown")
    except ValueError as exc:
        assert "Unsupported relative loss mode" in str(exc)
    else:
        raise AssertionError("unknown relative loss mode should raise ValueError")
