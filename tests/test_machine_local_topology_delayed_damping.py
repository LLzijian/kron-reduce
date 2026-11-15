import math

import pytest
import torch

from kron_reduce.evaluation.machine_local_topology_delayed_damping import fit_profile, minimize_damping


class _Budget:
    def __init__(self):
        self.counts = {}

    def spend(self, name):
        self.counts[name] = self.counts.get(name, 0) + 1


class _Model:
    log_damping_factor = torch.tensor(0.0)


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


def test_fit_profile_restores_selected_value_and_records_objective_calls():
    model, budget = _Model(), _Budget()
    result = fit_profile(model, lambda: (float(model.log_damping_factor) - .15) ** 2,
                         {"grid_points": 9, "log_half_width": 1., "max_refinement_calls": 24}, budget)
    assert result["status"] == "refined"
    assert float(model.log_damping_factor) == pytest.approx(.15, abs=1e-4)
    assert result["selected_train_loss"] < result["start_train_loss"]
    assert budget.counts == {"fits": 1, "objective_calls": len(result["trace"])}
