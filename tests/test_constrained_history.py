import math

import pytest
import torch

from kron_reduce.models.differentiable_history import (
    UnresolvedHistoryError,
    solve_hidden_ball,
    solve_hidden_batch,
)


def _spend(*_):
    pass


def test_inactive_ball_matches_unconstrained_batch_solution():
    design = torch.tensor([[[1., 0.], [0., 2.], [1., 1.]]], dtype=torch.float64)
    rhs = torch.tensor([[2., -1., .5]], dtype=torch.float64)
    expected = solve_hidden_batch(design, rhs, spend=_spend)["hidden_start"]

    result = solve_hidden_ball(design, rhs, 100., spend=_spend)

    assert torch.allclose(result["hidden_start"], expected, atol=1e-12, rtol=0)
    assert torch.equal(result["multipliers"], torch.zeros_like(result["multipliers"]))


def test_anisotropic_active_ball_satisfies_kkt_and_beats_radial_projection():
    design = torch.diag(torch.tensor([1., 3.], dtype=torch.float64)).unsqueeze(0)
    rhs = torch.tensor([[2., 6.]], dtype=torch.float64)
    radius = .8
    result = solve_hidden_ball(design, rhs, radius, spend=_spend)
    hidden = result["hidden_start"][0]
    multiplier = result["multipliers"][0]
    unconstrained = solve_hidden_batch(design, rhs, spend=_spend)["hidden_start"][0]
    radial = unconstrained * (radius / torch.linalg.vector_norm(unconstrained))

    assert torch.linalg.vector_norm(hidden) == pytest.approx(radius, abs=1e-8)
    assert multiplier > 0
    gradient = design[0].T @ (design[0] @ hidden - rhs[0])
    assert torch.linalg.vector_norm(gradient + multiplier * hidden) < 1e-7
    def objective(value):
        return torch.linalg.vector_norm(design[0] @ value - rhs[0]) ** 2

    assert objective(hidden) < objective(radial)


def test_tiny_positive_multiplier_at_the_condition_limit():
    design = torch.diag(torch.tensor([1., 1e-8], dtype=torch.float64)).unsqueeze(0)
    rhs = design.new_tensor([[.5, math.sqrt(.75)*(1.+1e-9)*1e-8]])
    result = solve_hidden_ball(design, rhs, 1., spend=_spend)
    assert 0. < result["multipliers"][0] < 1e-24
    assert result["hidden_start"].norm() == pytest.approx(1., abs=1e-10)


@pytest.mark.parametrize("design", [
    torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]], dtype=torch.float64),
    torch.diag(torch.tensor([1., 1e-9], dtype=torch.float64)).unsqueeze(0),
])
def test_unresolvable_rank_or_condition_is_rejected(design):
    rhs = torch.ones((1, design.shape[1]), dtype=torch.float64)
    with pytest.raises(UnresolvedHistoryError):
        solve_hidden_ball(design, rhs, 1., spend=_spend)
