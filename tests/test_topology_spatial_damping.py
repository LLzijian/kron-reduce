import numpy as np
import pytest
import torch

from kron_reduce.models.machine_local_topology import (
    TopologyMemoryModel,
    predict_from_topology_history,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


@pytest.fixture
def spatial_model():
    mass = np.array([1.2, 1.8, 1.5], dtype=np.float64)
    stiffness = np.array([1.1, 0.9, 1.3], dtype=np.float64)
    # Weighted three-node Laplacian with asymmetric edge weights.
    damping = np.array(
        [[1.3, -0.7, -0.6], [-0.7, 1.1, -0.4], [-0.6, -0.4, 1.0]],
        dtype=np.float64,
    )
    empty = np.zeros((0, 0), dtype=np.float64)
    backbone = NonlinearPassiveClosure(
        mass, stiffness, damping, np.zeros((3, 0)), empty, empty
    )
    return TopologyMemoryModel(backbone, topology=True, port=None, tau=None)


def _state_and_direction(model):
    q = torch.tensor([[0.31, -0.08, -0.23], [-0.12, 0.27, -0.04]], dtype=torch.float64)
    v = torch.tensor([[0.23, -0.17, 0.05], [-0.11, 0.19, -0.09]], dtype=torch.float64)
    q -= ((q @ model.mass) / model.mass.sum())[:, None]
    v -= ((v @ model.mass) / model.mass.sum())[:, None]
    latent = torch.tensor([[0.06, -0.04], [-0.03, 0.08]], dtype=torch.float64)
    state = torch.cat((q, v, latent), dim=1)
    direction = torch.tensor(
        [[0.16, -0.21, 0.07, 0.13, -0.04, -0.12, 0.11, -0.09],
         [-0.08, 0.14, 0.19, -0.15, 0.06, 0.10, -0.07, 0.12]],
        dtype=torch.float64,
    )
    power = torch.zeros((2, 3), dtype=torch.float64)
    equilibrium = torch.tensor([[0.04, -0.02, 0.01], [-0.03, 0.01, 0.02]], dtype=torch.float64)
    return state, direction, power, equilibrium


def test_spatial_mix_zero_is_bitwise_global_angle_law(spatial_model):
    model = spatial_model
    state, direction, power, equilibrium = _state_and_direction(model)
    keys = set(model.state_dict())

    model.set_angle_damping(0.51)
    default_field, default_tangent = model._rhs(power, equilibrium, tangent=True)(
        state, direction[..., None]
    )
    model.set_angle_damping(0.51, spatial_mix=0.0)
    explicit_field, explicit_tangent = model._rhs(power, equilibrium, tangent=True)(
        state, direction[..., None]
    )

    assert torch.equal(explicit_field, default_field)
    assert torch.equal(explicit_tangent, default_tangent)
    assert set(model.state_dict()) == keys


def test_spatial_mix_zero_reset_preserves_causal_rollout_bitwise(spatial_model):
    model = spatial_model
    state, _direction, _power, equilibrium = _state_and_direction(model)
    initial = state[:1].clone()
    power = torch.tensor([[0.025, -0.01, -0.015]], dtype=torch.float64)
    equilibrium = equilibrium[:1]

    model.set_angle_damping(0.43, spatial_mix=0.0)
    observed = model.rollout(initial, power, equilibrium, 10, max_step=0.005)
    scale = torch.ones(6, dtype=torch.float64)

    def forecast():
        return predict_from_topology_history(model, observed[:, :, :3], observed[:, :, 3:6], power, equilibrium,
                                             scale, 5, max_step=0.005, spend=lambda *_: None)["forecast"]

    global_rollout = forecast()
    model.set_angle_damping(0.43, spatial_mix=0.8)
    spatial_rollout = forecast()
    model.set_angle_damping(0.43, spatial_mix=0.0)
    reset_rollout = forecast()

    assert not torch.equal(spatial_rollout, global_rollout)
    assert torch.equal(reset_rollout, global_rollout)


def test_spatial_and_global_laws_have_exact_dissipation_without_increase(spatial_model):
    model = spatial_model
    state, _direction, power, equilibrium = _state_and_direction(model)
    n = model.n_nodes
    q, velocity = state[:, :n], state[:, n:2*n]
    damping = model.effective_damping
    edge_angle = q @ model.incidence
    edge_energy = edge_angle.square()
    local_s = (q[:, :, None] - q[:, None, :]).square().sum(dim=-1) / (n - 1)
    root = model._damping_root()

    model.set_angle_damping(0.0)
    baseline = model._rhs(power, equilibrium)(state)
    baseline_power = torch.einsum("bi,ij,bj->b", velocity, damping, velocity)

    rho = 0.60
    model.set_angle_damping(rho, spatial_mix=0.0)
    global_rhs = model._rhs(power, equilibrium)(state)
    mean_s = edge_energy.mean(dim=-1)
    expected_global_change = rho * mean_s / (1.0 + mean_s) * torch.einsum(
        "bi,ij,bj->b", velocity, damping, velocity
    )
    global_change = torch.einsum(
        "bi,i,bi->b", velocity, model.mass, global_rhs[:, n:2*n] - baseline[:, n:2*n]
    )
    torch.testing.assert_close(global_change, expected_global_change, rtol=2e-13, atol=2e-14)

    model.set_angle_damping(rho, spatial_mix=1.0)
    spatial_rhs = model._rhs(power, equilibrium)(state)
    root_velocity = velocity @ root
    expected_spatial_change = rho * torch.einsum(
        "bi,bi->b", root_velocity.square(), local_s / (1.0 + local_s)
    )
    spatial_change = torch.einsum(
        "bi,i,bi->b", velocity, model.mass, spatial_rhs[:, n:2*n] - baseline[:, n:2*n]
    )
    torch.testing.assert_close(spatial_change, expected_spatial_change, rtol=2e-13, atol=2e-14)
    assert torch.all(spatial_change >= 0.0)
    assert torch.all(spatial_change <= (2.0 / 3.0) * baseline_power)
    assert torch.linalg.vector_norm(global_rhs - spatial_rhs) > 1e-4


def test_spatial_angle_damping_full_state_tangent_matches_finite_difference(spatial_model):
    model = spatial_model
    model.set_angle_damping(0.48, spatial_mix=0.7)
    state, direction, power, equilibrium = _state_and_direction(model)
    field, tangent = model._rhs(power, equilibrium, tangent=True)(state, direction[..., None])
    epsilon = 1e-6
    plus = model.rhs(state + epsilon * direction, power, equilibrium)
    minus = model.rhs(state - epsilon * direction, power, equilibrium)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(tangent[..., 0], finite_difference, rtol=3e-7, atol=3e-9)
    assert torch.isfinite(field).all() and torch.isfinite(tangent).all()


@pytest.mark.parametrize("spatial_mix", [-1e-12, 1.0 + 1e-12, float("nan"), float("inf")])
def test_spatial_mix_rejects_values_outside_convex_range(spatial_model, spatial_mix):
    with pytest.raises(ValueError, match="spatial damping mix"):
        spatial_model.set_angle_damping(0.3, spatial_mix=spatial_mix)
