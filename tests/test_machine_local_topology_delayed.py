from __future__ import annotations

import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology_delayed import (
    adam_direction,
    opposed_direction,
)


def test_opposed_direction_has_unit_spread_and_zero_center_of_inertia() -> None:
    mass = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    fast = (1, 7)

    direction = opposed_direction(mass, fast)
    fast_mass_fraction = mass[list(fast)].sum() / mass.sum()

    assert direction.shape == (10,)
    assert np.allclose(direction[list(fast)], 1.0 - fast_mass_fraction)
    assert np.allclose(direction[[i for i in range(10) if i not in fast]], -fast_mass_fraction)
    assert direction[fast[0]] - direction[0] == pytest.approx(1.0)
    assert np.dot(mass, direction) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    "fast",
    [(), (1,), (1, 1), (1, 2, 3), (-1, 2), (0, 10), (1.0, 2), (True, 2)],
)
def test_opposed_direction_rejects_invalid_fast_indexes(fast: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        opposed_direction(np.ones(10), fast)


@pytest.mark.parametrize(
    "mass",
    [np.ones(9), np.zeros(10), np.r_[np.ones(9), -1.0], np.r_[np.ones(9), np.nan], np.r_[np.ones(9), np.inf]],
)
def test_opposed_direction_rejects_invalid_masses(mass: np.ndarray) -> None:
    with pytest.raises(ValueError):
        opposed_direction(mass, (1, 2))


def test_adam_direction_uses_clipped_first_step_signs_and_preserves_zeros() -> None:
    gradient = torch.tensor([3.0, -4.0, 0.0], dtype=torch.float64)

    direction = adam_direction(gradient)
    clipped = gradient / torch.linalg.vector_norm(gradient)
    expected = clipped / (clipped.abs() + 1e-8)

    assert isinstance(direction, torch.Tensor)
    assert torch.isfinite(direction).all()
    assert torch.allclose(direction, expected)
    assert direction[0] > 0
    assert direction[1] < 0
    assert direction[2] == 0


def test_adam_direction_of_zero_gradient_is_finite_zero() -> None:
    direction = adam_direction(torch.zeros(10, dtype=torch.float64))

    assert torch.isfinite(direction).all()
    assert torch.count_nonzero(direction) == 0
