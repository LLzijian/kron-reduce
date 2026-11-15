import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology import energy_residual, make_model
from kron_reduce.evaluation.machine_local_topology_diagnosis import crossed_origin
from kron_reduce.evaluation.machine_local_topology_truth import (
    initial_cases,
    parameters,
    truth_rollout,
)
from kron_reduce.models.machine_local_topology import (
    TopologyMemoryModel,
    _ball,
    predict_from_topology_history,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


@pytest.mark.parametrize(("use_a", "use_z"), [(False, False), (True, False), (False, True), (True, True)])
def test_origin_replacement_keeps_visible_state_and_crosses_latent_blocks(use_a, use_z):
    estimated = torch.ones((2, 39), dtype=torch.float64)
    truth = torch.zeros_like(estimated)
    origin = crossed_origin(estimated, truth, exact_a=use_a, exact_z=use_z)
    torch.testing.assert_close(origin[:, :20], truth[:, :20])
    torch.testing.assert_close(origin[:, 20:30], (truth if use_z else estimated)[:, 20:30])
    torch.testing.assert_close(origin[:, 30:], (truth if use_a else estimated)[:, 30:])


@pytest.mark.parametrize("support", ["T0M0", "T1M0", "T0M1", "T1M1"])
def test_topology_memory_energy_and_tangent(support):
    model = make_model(parameters(), support, "cpu")
    rng = np.random.default_rng(17)
    width = 20+model.latent_dimension
    x = model._tensor(rng.normal(0., .1, (2, width)))
    for visible in (x[:, :10], x[:, 10:20]):
        visible -= ((visible@model.mass)/model.mass.sum())[:, None]
    equilibrium = model._tensor(rng.normal(0., .08, (2, 10)))
    p = model._tensor(rng.normal(0., .03, (2, 10))) @ model.power_projection.T
    a = model.node_correction(x)
    torch.testing.assert_close(a.sum(-1), x.new_zeros(2), atol=1e-12, rtol=0)
    torch.testing.assert_close(model.edge_stiffness(x), model.stiffness*torch.exp(a@model.incidence.abs()))
    assert energy_residual(model, x, p, equilibrium, lambda *_: None) < 1e-12

    direction = model._tensor(rng.normal(size=(2, width)))
    field, tangent = model._rhs(p, equilibrium, tangent=True)(x, direction[..., None])
    torch.testing.assert_close(field, model.rhs(x, p, equilibrium))
    epsilon = 1e-6
    numerical = (model.rhs(x+epsilon*direction, p, equilibrium)
                 -model.rhs(x-epsilon*direction, p, equilibrium))/(2*epsilon)
    torch.testing.assert_close(tangent[..., 0], numerical, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(field[:, 20+model.memory_dimension:],
                               torch.zeros_like(field[:, 20+model.memory_dimension:]))


def test_initialization_ball_preserves_interior_and_bounds_exterior():
    x = torch.tensor([[.3, .4], [6., 8.], [0., 0.]], dtype=torch.float64, requires_grad=True)
    result = _ball(x, 1.)
    torch.testing.assert_close(result, x.new_tensor([[.3, .4], [.6, .8], [0., 0.]]))
    result.sum().backward()
    assert torch.isfinite(x.grad).all()


@torch.no_grad()
def test_exact_memory_ball_uses_absolute_state_and_monotone_history_cost():
    model = make_model(parameters(), "T0M1", "cpu")
    x = model.mass.new_zeros((2, 30))
    x[0, 20:] = .1
    x[1, 20] = .001
    zeros = model.mass.new_zeros((2, 10))
    def spend(*_):
        pass

    radius = .1
    measured = model.rollout(x, zeros, zeros, 10, max_step=.0025, spend=spend)
    result = predict_from_topology_history(
        model, measured[:, :, :10], measured[:, :, 10:20], zeros, zeros, model.mass.new_ones(20),
        2, max_step=.0025, spend=spend, memory_radius=radius, exact_memory_ball=True)
    diagnostic = result["history_ball"]
    assert (diagnostic["after"] <= diagnostic["before"]).all()
    assert (diagnostic["multipliers"][:, 0] > 0).all()
    assert (diagnostic["multipliers"][:, 1] == 0).all()
    torch.testing.assert_close(result["hidden_start"].norm(dim=1)[0], x.new_tensor(radius))
    torch.testing.assert_close(result["hidden_start"][1], x[1, 20:], atol=1e-10, rtol=0)
    torch.testing.assert_close(result["forecast"][:, 0, :20], measured[:, -1, :20])


def test_shared_coefficient_gradient_through_causal_forecast():
    mass, k, d, g, tau, eq = params = parameters()
    cases = initial_cases()
    q, v, p, a, z = (cases[key][:1] for key in ("q", "v", "power", "a", "z"))
    def no_spend(*_):
        pass

    truth = truth_rollout(params, q, v, p, a, z, count=12, max_step=.0025, spend=no_spend)
    empty = np.zeros((0, 0))
    backbone = NonlinearPassiveClosure(mass, k, d, np.zeros((10, 0)), empty, empty)
    model = TopologyMemoryModel(backbone, topology=True, port=g*.98, tau=tau*1.02, train_scales=True)
    with torch.no_grad():
        model.log_stiffness_factor.fill_(.005)
        model.log_damping_factor.fill_(-.005)

    def loss():
        predicted = predict_from_topology_history(model, truth[:, :11, :10], truth[:, :11, 10:20], p,
                                                  eq[None], np.ones(20), 2, max_step=.005, spend=no_spend)
        error = predicted["forecast"][:, 1:, :20]-model._tensor(truth[:, 11:, :20])
        return error.square().mean()

    named_variables = dict(model.named_parameters())
    expected_groups = {
        "log_stiffness_factor", "log_damping_factor", "port_parameter", "rate_parameter",
    }
    assert set(named_variables) == expected_groups
    variables = tuple(named_variables.values())
    gradients = torch.autograd.grad(loss(), variables)
    initial = [x.detach().clone() for x in variables]
    epsilon = 2e-6
    for variable, value, gradient in zip(variables, initial, gradients, strict=True):
        direction = torch.linspace(.7, 1., variable.numel(), dtype=variable.dtype).reshape(variable.shape)
        assert gradient.shape == variable.shape
        assert torch.isfinite(gradient).all() and torch.linalg.norm(gradient) > 0
        with torch.no_grad():
            variable.copy_(value+epsilon*direction)
        plus = float(loss().detach())
        with torch.no_grad():
            variable.copy_(value-epsilon*direction)
        minus = float(loss().detach())
        with torch.no_grad():
            variable.copy_(value)
        analytic = float((direction*gradient).sum())
        assert analytic == pytest.approx((plus-minus)/(2*epsilon), rel=2e-3, abs=1e-10)
