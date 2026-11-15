import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters, truth_rollout
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _model_and_params():
    params = list(parameters())
    return make_model(tuple(params), "T0M1", "cpu"), params


def _inputs(model, params, *, xi=None, nonzero_reference=True):
    mass = np.asarray(params[0], dtype=np.float64)
    n = len(mass)
    projection = np.eye(n) - np.outer(mass, np.ones(n)) / mass.sum()
    q = (0.022 * np.sin(0.61 * np.arange(n) + .1)) @ projection
    v = (0.034 * np.cos(0.43 * np.arange(n) + .2)) @ projection
    z = 0.018 * np.sin(0.31 * np.arange(n) + .3)
    power = 0.012 * np.sin(0.37 * np.arange(n) + .4)
    power -= power.mean()
    equilibrium = (np.linspace(-0.035, 0.041, n) if nonzero_reference
                   else np.zeros(n, dtype=np.float64))
    params[-1] = equilibrium.copy()
    parts = [q, v, z]
    if model.dynamic_edge_basis is not None:
        parts.append(np.zeros(n-1, dtype=np.float64) if xi is None else np.asarray(xi, dtype=np.float64))
    initial = model._tensor(np.concatenate(parts)[None, :])
    return (
        q[None, :], v[None, :], z[None, :], power[None, :], equilibrium,
        initial, model._tensor(power[None, :]), model._tensor(equilibrium[None, :]),
    )


def _coi_projection(mass):
    n = len(mass)
    return np.eye(n) - np.outer(mass, np.ones(n)) / np.sum(mass)


def test_spatial_coupling_log_products_are_exact_and_bounded_per_edge():
    model, params = _model_and_params()
    n = model.n_nodes
    edge_basis = model.incidence.abs().T @ model.basis[:, :n-1]
    edge_norm = torch.linalg.vector_norm(edge_basis, dim=1)
    limit = float(np.log(2.0) / edge_norm.max())
    model.set_dynamic_coupling(0.1, stiffness=3.2, limit=limit, spatial=True)

    _q, _v, _z, _power, _eq, initial, _p, _e = _inputs(model, params, xi=np.zeros(n-1))
    zero_log = torch.log(model.edge_stiffness(initial) / model.effective_stiffness)
    torch.testing.assert_close(zero_log, torch.zeros_like(zero_log), rtol=0.0, atol=0.0)

    xi = torch.linspace(-4.0, 5.0, n-1, dtype=torch.float64)[None, :]
    state = initial.clone()
    state[:, 2*n+model.memory_dimension:] = xi
    radial = xi / torch.sqrt(1.0 + (xi / limit).square().sum(dim=1, keepdim=True))
    expected_log = radial @ edge_basis.T
    actual_log = torch.log(model.edge_stiffness(state) / model.effective_stiffness)
    torch.testing.assert_close(actual_log, expected_log, rtol=2e-14, atol=2e-14)
    assert torch.all(actual_log.abs() <= np.log(2.0) + 2e-14)
    assert torch.all(model.edge_stiffness(state) / model.effective_stiffness >= 0.5 - 2e-14)
    assert torch.all(model.edge_stiffness(state) / model.effective_stiffness <= 2.0 + 2e-14)


def test_zero_mobility_and_zero_spatial_state_match_plain_prediction_bitwise():
    plain, params = _model_and_params()
    spatial, _ = _model_and_params()
    n = plain.n_nodes
    edge_basis = spatial.incidence.abs().T @ spatial.basis[:, :n-1]
    limit = float(np.log(2.0) / torch.linalg.vector_norm(edge_basis, dim=1).max())
    spatial.set_dynamic_coupling(0.0, stiffness=3.2, limit=limit, spatial=True)
    _q, _v, _z, _power, _eq, initial, power, equilibrium = _inputs(plain, params)
    spatial_initial = torch.cat((initial, torch.zeros((1, n-1), dtype=torch.float64)), dim=1)

    plain_path = plain.rollout(initial, power, equilibrium, 10, max_step=0.005)
    spatial_path = spatial.rollout(spatial_initial, power, equilibrium, 10, max_step=0.005)
    assert torch.equal(spatial_path[..., :3*n], plain_path)
    assert torch.count_nonzero(spatial_path[..., 3*n:]) == 0

    def no_spend(*_args):
        pass

    scale = torch.ones(2*n, dtype=torch.float64)
    q_history = plain_path[:, :, :n]
    v_history = plain_path[:, :, n:2*n]
    plain_result = predict_from_topology_history(
        plain, q_history, v_history, power, equilibrium, scale, 2,
        max_step=0.005, spend=no_spend,
    )
    spatial_result = predict_from_topology_history(
        spatial, q_history, v_history, power, equilibrium, scale, 2,
        max_step=0.005, spend=no_spend,
    )
    assert spatial.latent_dimension == 19
    assert spatial_result["hidden_start"].shape == (1, n)
    torch.testing.assert_close(
        spatial_result["hidden_start"], plain_result["hidden_start"], rtol=0.0, atol=5e-16
    )
    torch.testing.assert_close(
        spatial_result["history"][..., :3*n], plain_result["history"], rtol=0.0, atol=5e-16
    )
    torch.testing.assert_close(
        spatial_result["forecast"][..., :3*n], plain_result["forecast"], rtol=0.0, atol=5e-16
    )
    assert torch.count_nonzero(spatial_result["history"][..., 3*n:]) == 0
    assert torch.count_nonzero(spatial_result["forecast"][..., 3*n:]) == 0


def test_spatial_dynamic_coupling_tangent_matches_finite_difference_with_nonzero_ref_and_damping():
    model, params = _model_and_params()
    n = model.n_nodes
    edge_basis = model.incidence.abs().T @ model.basis[:, :n-1]
    limit = float(np.log(2.0) / torch.linalg.vector_norm(edge_basis, dim=1).max())
    model.set_dynamic_coupling(0.13, stiffness=2.8, limit=limit, spatial=True)
    model.set_angle_damping(0.41, spatial_mix=0.63)
    xi = np.linspace(-0.7, 0.8, n-1)
    _q, _v, _z, _power, _eq, state, power, equilibrium = _inputs(model, params, xi=xi)
    direction = torch.linspace(-0.16, 0.19, state.shape[1], dtype=torch.float64)[None, :]
    field, tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])
    epsilon = 1e-6
    plus = model.rhs(state + epsilon*direction, power, equilibrium)
    minus = model.rhs(state - epsilon*direction, power, equilibrium)
    finite_difference = (plus - minus) / (2.0*epsilon)
    torch.testing.assert_close(tangent[..., 0], finite_difference, rtol=6e-7, atol=5e-9)
    assert torch.isfinite(field).all() and torch.isfinite(tangent).all()


def test_spatial_dynamic_coupling_energy_balance_with_external_work():
    model, params = _model_and_params()
    n = model.n_nodes
    edge_basis = model.incidence.abs().T @ model.basis[:, :n-1]
    limit = float(np.log(2.0) / torch.linalg.vector_norm(edge_basis, dim=1).max())
    mobility, stiffness = 0.14, 2.6
    model.set_dynamic_coupling(mobility, stiffness=stiffness, limit=limit, spatial=True)
    xi = np.linspace(-0.3, 0.5, n-1)
    _q, _v, _z, _power, _eq, state, power, equilibrium = _inputs(model, params, xi=xi)
    state = state.detach().requires_grad_(True)
    field = model.rhs(state, power, equilibrium)
    storage = model.storage(state, equilibrium).sum()
    gradient = torch.autograd.grad(storage, state)[0]
    energy_rate = (gradient * field).sum(dim=1)

    velocity = state[:, n:2*n]
    memory = state[:, 2*n:3*n]
    input_work = (velocity * power).sum(dim=1)
    damping_loss = torch.einsum("bi,ij,bj->b", velocity, model.effective_damping, velocity)
    memory_loss = (memory.square() / model.tau).sum(dim=1)
    coupling_loss = mobility * gradient[:, 3*n:].square().sum(dim=1)
    expected_rate = input_work - damping_loss - memory_loss - coupling_loss
    torch.testing.assert_close(energy_rate, expected_rate, rtol=5e-10, atol=5e-11)
    assert torch.all(input_work.abs() > 1e-5)


def test_spatial_coupling_short_rollout_matches_independent_numpy_truth():
    model, params = _model_and_params()
    n = model.n_nodes
    edge_basis = model.incidence.abs().T @ model.basis[:, :n-1]
    limit = float(np.log(2.0) / torch.linalg.vector_norm(edge_basis, dim=1).max())
    mobility, stiffness = 0.12, 2.4
    model.set_dynamic_coupling(mobility, stiffness=stiffness, limit=limit, spatial=True)
    xi = np.linspace(-0.025, 0.031, n-1)
    q, v, z, power_np, _equilibrium_np, initial, power, equilibrium = _inputs(
        model, params, xi=xi
    )
    model_path = model.rollout(initial, power, equilibrium, 3, max_step=0.005)
    truth_path = truth_rollout(
        tuple(params), q, v, power_np, np.zeros_like(q), z,
        count=3, max_step=0.005, spend=lambda *_args: None,
        dynamic_coupling=(mobility, stiffness, limit), coupling_initial=xi[None, :],
        spatial_coupling=True,
    )
    torch.testing.assert_close(
        model_path, torch.as_tensor(truth_path), rtol=3e-9, atol=2e-11
    )


def test_nonzero_spatial_coordinate_is_carried_from_history_to_forecast():
    model, params = _model_and_params()
    n = model.n_nodes
    edge_basis = model.incidence.abs().T @ model.basis[:, :n-1]
    limit = float(np.log(2.0) / torch.linalg.vector_norm(edge_basis, dim=1).max())
    model.set_dynamic_coupling(0.2, stiffness=2.1, limit=limit, spatial=True)
    xi = np.linspace(-0.04, 0.05, n-1)
    _q, _v, _z, _power, _eq, initial, power, equilibrium = _inputs(model, params, xi=xi)
    path = model.rollout(initial, power, equilibrium, 10, max_step=0.005)
    scale = torch.ones(2*n, dtype=torch.float64)
    result = predict_from_topology_history(
        model, path[:, :11, :n], path[:, :11, n:2*n], power, equilibrium, scale, 2,
        max_step=0.005, spend=lambda *_args: None,
    )
    xi_history = result["history"][..., 3*n:]
    xi_forecast = result["forecast"][..., 3*n:]
    assert torch.count_nonzero(xi_history[:, 0]) == 0
    torch.testing.assert_close(xi_forecast[:, 0], xi_history[:, -1], rtol=0.0, atol=0.0)
    assert torch.linalg.vector_norm(xi_forecast[:, -1]) > 0.0


@pytest.mark.parametrize("spatial", [False, True])
def test_dynamic_coupling_setter_only_accepts_t0m1(spatial):
    params = list(parameters())
    other = make_model(tuple(params), "T1M1", "cpu")
    with pytest.raises(ValueError, match="dynamic coupling"):
        other.set_dynamic_coupling(0.1, stiffness=2.0, limit=0.2, spatial=spatial)
