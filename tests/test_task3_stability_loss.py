import math

import pytest
import torch

from kron_reduce.evaluation.task3_stability_loss import loss_of_sync_margin


def test_loss_of_sync_requires_five_genuine_points_and_both_thresholds():
    path = torch.zeros((4, 16, 20), dtype=torch.float64)
    omega_s = torch.full((4, 1, 1), 2 * math.pi * 60, dtype=path.dtype)
    path[:, 11:, 0] = 1.1 * math.pi
    path[:3, 11:, 10] = .03 * omega_s[:3, 0]
    path[:3, 11:, 11] = -.03 * omega_s[:3, 0]
    path[2, 11:, 0] = math.pi
    eq = torch.zeros((4, 10), dtype=path.dtype)
    path.requires_grad_()
    actual = loss_of_sync_margin(path, eq, torch.ones_like(eq), omega_s, torch.tensor([5, 4, 5, 5]))
    torch.testing.assert_close(actual, path.new_tensor([.1, -1., 0., -1.]))
    gradient, = torch.autograd.grad(actual.sum(), path)
    assert torch.count_nonzero(gradient[1, 15:]) == 0


def test_loss_of_sync_gradient_for_window_crossing_the_origin():
    path = torch.zeros((1, 12, 20), dtype=torch.float64)
    omega_s = path.new_full((1, 1, 1), 2 * math.pi * 60)
    path[:, 7:, 0] = 1.2 * math.pi
    path[:, 11, 0] = 1.1 * math.pi
    path[:, 7:, 10] = .03 * omega_s
    path[:, 7:, 11] = -.03 * omega_s
    path.requires_grad_()
    eq = path.new_zeros((1, 10))
    margin = loss_of_sync_margin(path, eq, torch.ones_like(eq), omega_s, torch.tensor([1]))
    gradient, = torch.autograd.grad(margin.sum(), path)
    torch.testing.assert_close(margin, path.new_tensor([.1]))
    torch.testing.assert_close(gradient[0, 11, 0], path.new_tensor(1. / math.pi))


def test_loss_of_sync_rejects_invalid_mass_and_nonfinite_paths():
    path = torch.zeros((1, 12, 20), dtype=torch.float64)
    eq = path.new_zeros((1, 10))
    omega_s = path.new_full((1, 1, 1), 2 * math.pi * 60)
    counts = torch.tensor([1])
    with pytest.raises(ValueError, match="positive mass and frequency"):
        loss_of_sync_margin(path, eq, -torch.ones_like(eq), omega_s, counts)
    path[0, 11, 10] = float("nan")
    with pytest.raises(ValueError, match="paths and equilibrium must be finite"):
        loss_of_sync_margin(path, eq, torch.ones_like(eq), omega_s, counts)
