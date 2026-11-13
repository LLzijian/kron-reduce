from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kron_reduce.evaluation import machine_local_stiffness_calibration as calibration
from kron_reduce.evaluation.machine_local_synthetic import fixture
from kron_reduce.models.nonlinear_history import tangent_rhs


def _spend(*_args):
    pass


def test_nonunit_calibration_uses_same_stiffness_in_closure_rhs_and_tangent():
    base = fixture(backbone="sine")
    log_factor = np.log(1.37)
    model = calibration.calibrated_model(base, log_factor)
    closure = model.closure()
    expected_stiffness = np.full(45, .132*np.exp(log_factor))
    np.testing.assert_allclose(model.stiffness.detach().numpy(), expected_stiffness, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(closure.edge_stiffness, model.stiffness.detach().numpy())

    n = model.n_nodes
    mass = model.mass.detach().numpy()
    project = np.eye(n)-np.outer(mass, np.ones(n))/mass.sum()
    q = project @ np.linspace(-.04, .05, n)
    v = project @ np.linspace(.03, -.02, n)
    z = np.linspace(-.1, .12, model.latent_dimension)
    equilibrium = project @ np.linspace(.02, -.03, n)
    power = project @ np.linspace(-.1, .1, n)
    state = torch.tensor(np.r_[q, v, z][None], dtype=torch.float64)
    power_t = torch.tensor(power[None], dtype=torch.float64)
    equilibrium_t = torch.tensor(equilibrium[None], dtype=torch.float64)
    np.testing.assert_allclose(model.rhs(state, power_t, equilibrium_t)[0].detach().numpy(),
                               closure.rhs(state[0].numpy(), power, equilibrium), rtol=2e-13, atol=2e-13)

    direction = torch.linspace(-.3, .4, 30, dtype=torch.float64)[None, :, None]
    _, tangent = tangent_rhs(model, state, direction, power_t, equilibrium_t, spend=_spend)
    epsilon = 1e-6
    plus = model.rhs(state+epsilon*direction[:, :, 0], power_t, equilibrium_t)
    minus = model.rhs(state-epsilon*direction[:, :, 0], power_t, equilibrium_t)
    torch.testing.assert_close(tangent[:, :, 0], (plus-minus)/(2*epsilon), rtol=2e-7, atol=2e-9)


def test_fit_selects_first_best_finite_visited_and_never_spends_over_call_limit(monkeypatch):
    calls = []
    values = {-.2: 2., .1: None, .3: 1., .4: 1.}

    def evaluate(x):
        calls.append(x)
        return {"log_factor": x, "loss": values[x]}

    class Budget:
        count = 0

        def spend(self, _key):
            self.count += 1

    def optimizer(loss, **_kwargs):
        for point in (-.2, .1, .3, .4):
            loss(point)
        return SimpleNamespace(success=True, status=0, message="done", nfev=4)

    monkeypatch.setattr(calibration, "minimize_scalar", optimizer)
    budget = Budget()
    result = calibration.fit_scalar(evaluate, budget)
    assert calls == [-.2, .1, .3, .4]
    assert result["selected"] == {"log_factor": .3, "loss": 1.}
    assert result["optimizer_calls"] == result["nfev"] == 4
    assert budget.count == 5  # fit reservation plus the four objective allocations

    def overrun(loss, **_kwargs):
        for point in range(calibration.MAX_CALLS):
            loss(0.)
        with pytest.raises(RuntimeError, match="exceeded its frozen scope"):
            loss(0.)
        return SimpleNamespace(success=True, status=0, message="bounded", nfev=calibration.MAX_CALLS)

    monkeypatch.setattr(calibration, "minimize_scalar", overrun)
    result = calibration.fit_scalar(lambda x: {"log_factor": x, "loss": 1.}, Budget())
    assert result["optimizer_calls"] == calibration.MAX_CALLS
