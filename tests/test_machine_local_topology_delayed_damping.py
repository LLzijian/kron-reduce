import math

import pytest

from kron_reduce.evaluation.machine_local_topology_delayed_damping import minimize_damping


def test_bounded_profile_refines_interior_within_budget():
    calls = []

    def objective(value):
        calls.append(value)
        return .5 + (value - .213)**2

    optimum, status = minimize_damping(objective, .3, 9, math.log(2), 24)
    assert status == "refined"
    assert optimum == pytest.approx(.213, abs=1e-5)
    assert calls[4] == .3
    assert len(calls) <= 33


def test_boundary_does_not_refine_outside_bounds():
    calls = []

    def objective(value):
        calls.append(value)
        return value

    optimum, status = minimize_damping(objective, 0., 9, math.log(2), 24)
    assert status == "grid_boundary"
    assert optimum == -math.log(2)
    assert len(calls) == 9


def test_invalid_profile_has_no_fitted_value():
    assert minimize_damping(lambda _: math.inf, 0., 9, 1., 24) == (None, "no_finite_grid_value")
