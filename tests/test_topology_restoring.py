import numpy as np
import pytest
import torch

from kron_reduce.models.machine_local_topology import TopologyMemoryModel
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters, truth_rollout
from kron_reduce.models.machine_local_topology import predict_from_topology_history


@pytest.fixture
def restoring_model():
    mass = np.array([1.2, 1.8, 1.5], dtype=np.float64)
    stiffness = np.array([1.1, 0.9, 1.3], dtype=np.float64)
    damping = np.array(
        [[1.3, -0.7, -0.6], [-0.7, 1.1, -0.4], [-0.6, -0.4, 1.0]],
        dtype=np.float64,
    )
    empty = np.zeros((0, 0), dtype=np.float64)
    backbone = NonlinearPassiveClosure(
        mass, stiffness, damping, np.zeros((3, 0)), empty, empty
    )
    return TopologyMemoryModel(backbone, topology=True, port=None, tau=None)


def _coi(value, mass):
    return value - ((value @ mass) / mass.sum())[:, None]


def _state_inputs(model, *, near_equilibrium=False):
    equilibrium = torch.tensor([[0.13, -0.08, 0.025], [-0.06, 0.09, -0.015]], dtype=torch.float64)
    q = torch.tensor([[0.0, 0.0, 0.0], [0.021, -0.013, 0.006]], dtype=torch.float64)
    if near_equilibrium:
        q[1] *= 1e-3
    q = _coi(q, model.mass)
    velocity = _coi(
        torch.tensor([[0.0, 0.0, 0.0], [0.17, -0.09, 0.04]], dtype=torch.float64),
        model.mass,
    )
    topology = torch.tensor([[0.0, 0.0], [0.07, -0.035]], dtype=torch.float64)
    state = torch.cat((q, velocity, topology), dim=1)
    direction = torch.tensor(
        [[0.16, -0.21, 0.07, 0.13, -0.04, -0.12, 0.11, -0.09],
         [-0.08, 0.14, 0.19, -0.15, 0.06, 0.10, -0.07, 0.12]],
        dtype=torch.float64,
    )
    power = torch.zeros((2, 3), dtype=torch.float64)
    return state, direction, power, equilibrium


def test_zero_restoring_shape_matches_plain_rhs_tangent_and_storage(restoring_model):
    model = restoring_model
    state, direction, power, equilibrium = _state_inputs(model)
    plain_rhs = model._rhs(power, equilibrium)
    plain_field, plain_tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])
    plain_storage = model.storage(state, equilibrium)

    model.set_restoring_shape(0.0)
    field, tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])
    torch.testing.assert_close(field, plain_rhs(state), rtol=0.0, atol=0.0)
    torch.testing.assert_close(field, plain_field, rtol=0.0, atol=0.0)
    torch.testing.assert_close(tangent, plain_tangent, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.storage(state, equilibrium), plain_storage, rtol=0.0, atol=0.0)


def test_restoring_shape_preserves_nonzero_equilibrium_and_tangent(restoring_model):
    model = restoring_model
    state, direction, power, equilibrium = _state_inputs(model)
    plain_tangent = model._rhs(power, equilibrium, tangent=True)(
        state, direction[..., None]
    )[1]
    model.set_restoring_shape(0.23)
    field, tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])

    # Relative angle and velocity are zero in the first row, despite nonzero
    # reference angles; the modified conservative force keeps that equilibrium.
    torch.testing.assert_close(field[0], torch.zeros_like(field[0]), rtol=0.0, atol=1e-14)

    torch.testing.assert_close(tangent[0], plain_tangent[0], rtol=0.0, atol=2e-13)

    near_state, near_direction, near_power, near_equilibrium = _state_inputs(model, near_equilibrium=True)
    near_field, near_tangent = model._rhs(near_power, near_equilibrium, tangent=True)(
        near_state, near_direction[..., None]
    )
    epsilon = 1e-6
    plus = model.rhs(near_state + epsilon * near_direction, near_power, near_equilibrium)
    minus = model.rhs(near_state - epsilon * near_direction, near_power, near_equilibrium)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(near_tangent[..., 0], finite_difference, rtol=3e-7, atol=3e-9)
    assert torch.isfinite(near_field).all() and torch.isfinite(near_tangent).all()


def test_restoring_shape_storage_has_exact_dissipation_identity(restoring_model):
    model = restoring_model
    model.set_restoring_shape(-0.19)
    state, _direction, power, equilibrium = _state_inputs(model)
    state = state.detach().requires_grad_(True)
    field = model.rhs(state, power, equilibrium)
    energy = model.storage(state, equilibrium).sum()
    gradient = torch.autograd.grad(energy, state)[0]
    energy_rate = (gradient * field).sum(dim=1)
    velocity = state[:, model.n_nodes:2 * model.n_nodes]
    expected_rate = -(velocity * (velocity @ model.effective_damping.T)).sum(dim=1)
    torch.testing.assert_close(energy_rate, expected_rate, rtol=2e-12, atol=2e-12)


def test_zero_restoring_shape_preserves_causal_history_and_forecast_bitwise():
    base = list(parameters())
    mass = base[0]
    equilibrium = np.linspace(-0.08, 0.09, len(mass), dtype=np.float64)
    base[-1] = equilibrium
    model = make_model(tuple(base), "T0M1", "cpu")
    projection = np.eye(len(mass)) - np.outer(mass, np.ones(len(mass))) / mass.sum()
    q0 = (0.025 * np.sin(0.61 * np.arange(len(mass)))) @ projection
    v0 = (0.04 * np.cos(0.47 * np.arange(len(mass)))) @ projection
    power = 0.018 * np.sin(0.39 * np.arange(len(mass)))
    power -= power.mean()
    z0 = np.linspace(-0.045, 0.055, len(mass), dtype=np.float64)
    initial = model._tensor(np.concatenate((q0, v0, z0))[None, :])
    power_tensor = model._tensor(power[None, :])
    equilibrium_tensor = model._tensor(equilibrium[None, :])
    measured = model.rollout(initial, power_tensor, equilibrium_tensor, 10, max_step=0.005)

    def no_spend(*_args):
        pass

    def predict():
        return predict_from_topology_history(
            model,
            measured[:, :, :10],
            measured[:, :, 10:20],
            power_tensor,
            equilibrium_tensor,
            model.mass.new_ones(20),
            2,
            max_step=0.005,
            spend=no_spend,
        )

    baseline = predict()
    model.set_restoring_shape(0.18)
    model.set_restoring_shape(0.0)
    reset = predict()
    assert torch.equal(reset["history"], baseline["history"])
    assert torch.equal(reset["forecast"], baseline["forecast"])


def test_nonzero_equilibrium_numpy_truth_force_matches_model_rollout():
    params = list(parameters())
    mass = params[0]
    equilibrium = np.linspace(-0.09, 0.08, len(mass), dtype=np.float64)
    params[-1] = equilibrium
    model = make_model(tuple(params), "T0M1", "cpu")
    alpha = -0.17
    model.set_restoring_shape(alpha)
    projection = np.eye(len(mass)) - np.outer(mass, np.ones(len(mass))) / mass.sum()
    q = ((0.023 * np.sin(0.73 * np.arange(len(mass)))) @ projection)[None, :]
    v = ((0.031 * np.cos(0.52 * np.arange(len(mass)))) @ projection)[None, :]
    power = (0.015 * np.sin(0.31 * np.arange(len(mass))))[None, :]
    power -= power.mean(axis=1, keepdims=True)
    a = np.zeros_like(q)
    z = np.linspace(-0.035, 0.045, len(mass), dtype=np.float64)[None, :]
    state = model._tensor(np.concatenate((q, v, z), axis=1))
    power_tensor = model._tensor(power)
    equilibrium_tensor = model._tensor(equilibrium[None, :])
    predicted = model.rollout(state, power_tensor, equilibrium_tensor, 2, max_step=0.0025)

    def no_spend(*_args):
        pass

    truth = truth_rollout(
        tuple(params), q, v, power, a, z, count=2, max_step=0.0025,
        spend=no_spend, restoring_alpha=alpha,
    )
    np.testing.assert_allclose(predicted.detach().cpu().numpy(), truth, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("alpha", [-0.250001, 0.250001, float("nan"), float("inf")])
def test_restoring_shape_rejects_values_outside_declared_range(restoring_model, alpha):
    with pytest.raises(ValueError, match="restoring shape"):
        restoring_model.set_restoring_shape(alpha)
