import numpy as np
import pytest
import torch

from kron_reduce.models.machine_local_topology import (
    TopologyMemoryModel,
    predict_from_topology_history,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


@pytest.fixture
def angle_model():
    # A deterministic two-machine system with one nonzero edge and a PSD D0.
    mass = np.array([1.2, 1.8], dtype=np.float64)
    stiffness = np.array([1.4], dtype=np.float64)
    damping = np.array([[0.45, -0.45], [-0.45, 0.45]], dtype=np.float64)
    empty = np.zeros((0, 0), dtype=np.float64)
    backbone = NonlinearPassiveClosure(
        mass, stiffness, damping, np.zeros((2, 0)), empty, empty
    )
    return TopologyMemoryModel(backbone, topology=True, port=None, tau=None)


def _inputs(model):
    state = torch.tensor(
        [[0.13, -0.07, 0.31, -0.19, 0.08], [-0.09, 0.12, -0.22, 0.27, -0.11]],
        dtype=torch.float64,
    )
    direction = torch.tensor(
        [[0.21, -0.16, -0.08, 0.17, 0.13], [-0.12, 0.18, 0.23, -0.14, 0.09]],
        dtype=torch.float64,
    )
    power = torch.zeros((2, 2), dtype=torch.float64)
    equilibrium = torch.tensor([[0.04, -0.03], [-0.02, 0.05]], dtype=torch.float64)
    return state, direction, power, equilibrium


def test_zero_angle_damping_restores_exact_plain_rhs_and_tangent(angle_model):
    model = angle_model
    state, direction, power, equilibrium = _inputs(model)
    keys_before = set(model.state_dict())
    rhs_plain = model._rhs(power, equilibrium)
    rhs_tangent_plain = model._rhs(power, equilibrium, tangent=True)
    expected = rhs_plain(state)
    expected_field, expected_tangent = rhs_tangent_plain(state, direction[..., None])

    model.set_angle_damping(0.0)
    actual_field, actual_tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])

    torch.testing.assert_close(actual_field, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)
    torch.testing.assert_close(actual_tangent, expected_tangent, rtol=0, atol=0)
    assert set(model.state_dict()) == keys_before
    assert "angle_damping_rho" not in model.state_dict()


def test_zero_reset_preserves_causal_predictor_bitwise(angle_model):
    mass = np.array([1.2, 1.8, 1.5], dtype=np.float64)
    damping = np.eye(3, dtype=np.float64) - np.ones((3, 3), dtype=np.float64) / 3.0
    empty = np.zeros((0, 0), dtype=np.float64)
    backbone = NonlinearPassiveClosure(
        mass, np.array([1.1, 0.9, 1.3]), damping, np.zeros((3, 0)), empty, empty
    )
    model = TopologyMemoryModel(backbone, topology=True, port=None, tau=None)
    q0 = torch.tensor([[0.11, -0.073, 0.02]], dtype=torch.float64)
    v0 = torch.tensor([[0.24, -0.16, 0.09]], dtype=torch.float64)
    q0 -= ((q0 @ model.mass) / model.mass.sum())[:, None]
    v0 -= ((v0 @ model.mass) / model.mass.sum())[:, None]
    initial = torch.cat((q0, v0, torch.tensor([[0.06, -0.04]], dtype=torch.float64)), dim=1)
    power = torch.zeros((1, 3), dtype=torch.float64)
    equilibrium = torch.zeros_like(power)
    measured = model.rollout(initial, power, equilibrium, 10, max_step=0.005)
    scale = torch.ones(6, dtype=torch.float64)

    def no_spend(*_):
        pass

    def predict():
        return predict_from_topology_history(
            model,
            measured[:, :, :3],
            measured[:, :, 3:6],
            power,
            equilibrium,
            scale,
            2,
            max_step=0.005,
            spend=no_spend,
        )

    baseline = predict()
    model.set_angle_damping(0.51)
    model.set_angle_damping(0.0)
    reset = predict()
    for key in ("hidden_start", "history", "forecast", "singular"):
        if baseline[key] is None:
            assert reset[key] is None
        else:
            assert torch.equal(reset[key], baseline[key]), key


def test_angle_damping_power_identity_and_full_state_tangent(angle_model):
    model = angle_model
    rho = 0.42
    model.set_angle_damping(rho)
    state, direction, power, equilibrium = _inputs(model)
    n = model.n_nodes
    edge = model.incidence
    relative_angle = state[:, :n] @ edge
    s = relative_angle.square().mean(dim=-1)
    damping = model.effective_damping
    velocity = state[:, n:2*n]
    # For COI velocity, the power projection leaves v^T M a unchanged. The
    # added acceleration must therefore remove exactly rho*s/(1+s) v^T D0 v
    # from the baseline dissipation, matching D(q)=(1-rho*s/(1+s))D0.
    velocity = velocity - (velocity @ model.mass / model.mass.sum())[:, None]
    state[:, n:2*n] = velocity
    model.set_angle_damping(0.0)
    plain_rhs = model._rhs(power, equilibrium)(state)
    model.set_angle_damping(rho)
    field, tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])
    observed_change = torch.einsum(
        "bi,i,bi->b", velocity, model.mass, field[:, n:2*n] - plain_rhs[:, n:2*n]
    )
    expected_change = rho * s / (1.0 + s) * torch.einsum("bi,ij,bj->b", velocity, damping, velocity)
    torch.testing.assert_close(observed_change, expected_change, rtol=2e-13, atol=2e-14)
    epsilon = 1e-6
    plus = model.rhs(state + epsilon * direction, power, equilibrium)
    minus = model.rhs(state - epsilon * direction, power, equilibrium)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(tangent[..., 0], finite_difference, rtol=2e-7, atol=2e-9)
    assert torch.isfinite(field).all() and torch.isfinite(tangent).all()


@pytest.mark.parametrize("rho", [-1e-12, 2.0 / 3.0 + 1e-12, float("nan"), float("inf")])
def test_angle_damping_rejects_values_outside_passive_range(angle_model, rho):
    with pytest.raises(ValueError):
        angle_model.set_angle_damping(rho)
