import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology_truth import parameters, truth_rollout
from kron_reduce.models.machine_local_topology import (
    TopologyMemoryModel,
    predict_from_topology_history,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


def _make_t0m1(params=None):
    params = list(parameters() if params is None else params)
    mass, stiffness, damping, port, tau, _equilibrium = params
    empty = np.zeros((0, 0), dtype=np.float64)
    backbone = NonlinearPassiveClosure(
        mass, stiffness, damping, np.zeros((len(mass), 0)), empty, empty
    )
    return TopologyMemoryModel(
        backbone, topology=False, port=port, tau=tau
    )


def _initial_inputs(model, params=None, *, coupling=0.0):
    params = list(parameters() if params is None else params)
    mass = np.asarray(params[0], dtype=np.float64)
    n = len(mass)
    projection = np.eye(n) - np.outer(mass, np.ones(n)) / mass.sum()
    q = (0.026 * np.sin(0.63 * np.arange(n))) @ projection
    v = (0.038 * np.cos(0.41 * np.arange(n))) @ projection
    z = 0.025 * np.sin(0.29 * np.arange(n) + 0.2)
    power = 0.014 * np.sin(0.37 * np.arange(n) + 0.1)
    power -= power.mean()
    equilibrium = np.linspace(-0.025, 0.031, n, dtype=np.float64)
    params[-1] = equilibrium.copy()
    state_parts = [q, v, z]
    if model.dynamic_coupling is not None:
        state_parts.append(np.array([coupling], dtype=np.float64))
    initial = model._tensor(np.concatenate(state_parts)[None, :])
    power_tensor = model._tensor(power[None, :])
    equilibrium_tensor = model._tensor(equilibrium[None, :])
    return params, q[None, :], v[None, :], z[None, :], power[None, :], equilibrium, initial, power_tensor, equilibrium_tensor


@pytest.fixture
def dynamic_model():
    return _make_t0m1()


def test_dynamic_log_stiffness_is_bounded_by_declared_limit(dynamic_model):
    model = dynamic_model
    limit = 0.31
    model.set_dynamic_coupling(0.2, stiffness=1.7, limit=limit)
    params, _q, _v, _z, _power, _eq, initial, _p, _e = _initial_inputs(model)
    shifts = []
    for value in (-1e6, 0.0, 1e6):
        state = initial.clone()
        state[:, -1] = value
        ratio = model.edge_stiffness(state) / model.effective_stiffness
        expected = np.exp(2.0 * limit * np.tanh(value / limit))
        torch.testing.assert_close(ratio, torch.full_like(ratio, expected), rtol=2e-15, atol=0.0)
        shifts.append(float(ratio[0, 0]))
    assert shifts[0] >= np.exp(-2.0 * limit)
    assert shifts[-1] <= np.exp(2.0 * limit)


def test_zero_mobility_matches_plain_rollout_and_causal_predictor_bitwise():
    plain = _make_t0m1()
    dynamic = _make_t0m1()
    dynamic.set_dynamic_coupling(0.0, stiffness=1.7, limit=0.31)
    _params, _q, _v, _z, _power, _eq, initial, power, equilibrium = _initial_inputs(plain)
    dynamic_initial = torch.cat((initial, torch.zeros((1, 1), dtype=torch.float64)), dim=1)
    plain_path = plain.rollout(initial, power, equilibrium, 10, max_step=0.005)
    dynamic_path = dynamic.rollout(dynamic_initial, power, equilibrium, 10, max_step=0.005)
    assert torch.equal(dynamic_path[..., :-1], plain_path)
    assert torch.count_nonzero(dynamic_path[..., -1]) == 0

    scale = torch.ones(2 * plain.n_nodes, dtype=torch.float64)

    def no_spend(*_args):
        pass

    arguments = (plain_path[:, :, :plain.n_nodes], plain_path[:, :, plain.n_nodes:2*plain.n_nodes],
                 power, equilibrium, scale, 2)
    plain_result = predict_from_topology_history(
        plain, *arguments, max_step=0.005, spend=no_spend
    )
    dynamic_result = predict_from_topology_history(
        dynamic, *arguments, max_step=0.005, spend=no_spend
    )
    assert dynamic_result["hidden_start"].shape == (1, plain.memory_dimension)
    assert torch.equal(dynamic_result["hidden_start"], plain_result["hidden_start"])
    for key in ("history", "forecast"):
        assert torch.equal(dynamic_result[key][..., :-1], plain_result[key]), key
        assert torch.count_nonzero(dynamic_result[key][..., -1]) == 0


def test_nonzero_coupling_tangent_matches_full_state_finite_difference(dynamic_model):
    model = dynamic_model
    model.set_dynamic_coupling(0.18, stiffness=2.3, limit=0.27)
    _params, _q, _v, _z, _power, _eq, initial, power, equilibrium = _initial_inputs(
        model, coupling=0.14
    )
    state = initial.clone()
    direction = torch.tensor(
        [[0.13, -0.08, 0.06, -0.11, 0.17, -0.05, 0.09, -0.12, 0.04, -0.03,
          0.11, -0.07, 0.08, -0.05, 0.04, -0.09, 0.06, 0.02, -0.13, 0.10,
          0.07, -0.04, 0.09, -0.02, 0.05, -0.08, 0.12, -0.03, 0.06, -0.11, 0.15]],
        dtype=torch.float64,
    )
    field, tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])
    epsilon = 1e-6
    plus = model.rhs(state + epsilon * direction, power, equilibrium)
    minus = model.rhs(state - epsilon * direction, power, equilibrium)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(tangent[..., 0], finite_difference, rtol=4e-7, atol=4e-9)
    assert torch.isfinite(field).all() and torch.isfinite(tangent).all()


def test_dynamic_coupling_dissipation_accounts_for_nonzero_input_work(dynamic_model):
    model = dynamic_model
    mobility, stiffness, limit = 0.16, 2.1, 0.29
    model.set_dynamic_coupling(mobility, stiffness=stiffness, limit=limit)
    _params, _q, _v, _z, _power, _eq, state, power, equilibrium = _initial_inputs(
        model, coupling=0.08
    )
    state = state.detach().requires_grad_(True)
    field = model.rhs(state, power, equilibrium)
    storage = model.storage(state, equilibrium).sum()
    gradient = torch.autograd.grad(storage, state)[0]
    energy_rate = (gradient * field).sum(dim=1)

    n = model.n_nodes
    velocity = state[:, n:2*n]
    memory = state[:, 2*n:3*n]
    input_work = (velocity * power).sum(dim=1)
    damping_loss = torch.einsum("bi,ij,bj->b", velocity, model.effective_damping, velocity)
    memory_loss = (memory.square() / model.tau).sum(dim=1)
    coupling_gradient = gradient[:, -1]
    expected_rate = input_work - damping_loss - memory_loss - mobility * coupling_gradient.square()
    torch.testing.assert_close(energy_rate, expected_rate, rtol=3e-11, atol=3e-12)
    assert torch.all(input_work.abs() > 1e-5)


def test_dynamic_coupling_short_rollout_matches_independent_numpy_truth(dynamic_model):
    model = dynamic_model
    dynamic = (0.12, 1.9, 0.26)
    model.set_dynamic_coupling(dynamic[0], stiffness=dynamic[1], limit=dynamic[2])
    params, q, v, z, power, equilibrium, initial, power_tensor, equilibrium_tensor = _initial_inputs(model)
    predicted = model.rollout(initial, power_tensor, equilibrium_tensor, 3, max_step=0.00125)

    def no_spend(*_args):
        pass

    truth = truth_rollout(
        tuple(params), q, v, power, np.zeros_like(q), z, count=3, max_step=0.00125,
        spend=no_spend, dynamic_coupling=dynamic,
    )
    np.testing.assert_allclose(predicted.detach().cpu().numpy(), truth, rtol=3e-10, atol=3e-12)


def test_observer_integrates_c_from_zero_and_carries_its_origin_value(dynamic_model):
    model = dynamic_model
    model.set_dynamic_coupling(0.2, stiffness=2.0, limit=0.3)
    _params, _q, _v, _z, _power, _eq, initial, power, equilibrium = _initial_inputs(model)
    measured = model.rollout(initial, power, equilibrium, 10, max_step=0.005)
    scale = torch.ones(2 * model.n_nodes, dtype=torch.float64)

    def no_spend(*_args):
        pass

    result = predict_from_topology_history(
        model,
        measured[:, :, :model.n_nodes],
        measured[:, :, model.n_nodes:2*model.n_nodes],
        power,
        equilibrium,
        scale,
        3,
        max_step=0.005,
        spend=no_spend,
    )
    history_c = result["history"][..., -1]
    forecast_c = result["forecast"][..., -1]
    assert result["hidden_start"].shape == (1, model.memory_dimension)
    torch.testing.assert_close(history_c[:, 0], torch.zeros_like(history_c[:, 0]), rtol=0.0, atol=0.0)
    assert torch.any(history_c[:, -1].abs() > 1e-10)
    assert torch.equal(forecast_c[:, 0], history_c[:, -1])


@pytest.mark.parametrize(
    "mobility,stiffness,limit",
    [(-0.1, 1.0, 0.2), (0.1, 0.0, 0.2), (0.1, 1.0, 0.0), (float("nan"), 1.0, 0.2)],
)
def test_dynamic_coupling_rejects_invalid_parameters(dynamic_model, mobility, stiffness, limit):
    with pytest.raises(ValueError, match="dynamic coupling"):
        dynamic_model.set_dynamic_coupling(mobility, stiffness=stiffness, limit=limit)
