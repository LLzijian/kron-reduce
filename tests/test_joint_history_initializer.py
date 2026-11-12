import numpy as np
import pytest
import torch

from kron_reduce.models.joint_history_initializer import (
    estimate_joint_history,
    initial_state_basis,
)


class _QuadraticHistoryModel:
    backbone_kind = "linear"
    n_nodes = 2
    latent_dimension = 2

    def __init__(self, *, rank_deficient=False):
        self.mass = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self.rank_deficient = rank_deficient
        self.calls = []

    def rollout(self, initial, power, equilibrium, count, *, max_step, spend):
        self.calls.append((initial.copy(), power.copy(), equilibrium.copy(), count, max_step))
        spend("rollouts")
        t = np.linspace(0.0, 1.0, count + 1)[None, :, None]
        q0, v0, hidden = initial[:, None, :2], initial[:, None, 2:4], initial[:, None, 4:]
        states = np.empty((len(initial), count + 1, 6), dtype=np.float64)
        states[:, :, :2] = q0 + t*v0 + .5*t*t*hidden
        states[:, :, 2:4] = v0 + t*hidden
        states[:, :, 4:] = hidden
        if self.rank_deficient:
            # Erase one response direction while retaining the baseline rows.
            states[len(initial)-4] = 0.0
        return torch.tensor(states, dtype=torch.float64)


def _problem(*, rank_deficient=False, seed=42):
    rng = np.random.default_rng(seed)
    model = _QuadraticHistoryModel(rank_deficient=rank_deficient)
    batch = 3
    basis = initial_state_basis(model.mass.numpy())
    coordinates = rng.normal(scale=0.1, size=(batch, basis.shape[1]))
    state = coordinates @ basis.T
    t = np.linspace(0.0, 1.0, 11)[None, :, None]
    q0, v0, hidden = state[:, None, :2], state[:, None, 2:4], state[:, None, 4:]
    q = q0 + t*v0 + .5*t*t*hidden
    v = v0 + t*hidden
    power = np.zeros((batch, 2))
    equilibrium = np.zeros_like(power)
    scale = np.array([0.7, 1.1, 0.8, 1.3])
    return model, q, v, power, equilibrium, scale, state, basis


def _estimate(problem):
    model, q, v, power, equilibrium, scale, *_ = problem
    spent = []
    result = estimate_joint_history(
        model, q, v, power, equilibrium, scale,
        max_step=0.01, spend=lambda operation, amount=1: spent.append((operation, amount)),
    )
    return result, spent


def test_householder_basis_has_orthonormal_mass_orthogonal_blocks():
    mass = np.array([1.0, 2.0, 3.0])
    basis = initial_state_basis(mass)

    assert basis.shape == (9, 7)
    np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(mass @ basis[:3, :2], np.zeros(2), atol=1e-14)
    np.testing.assert_allclose(mass @ basis[3:6, 2:4], np.zeros(2), atol=1e-14)
    np.testing.assert_array_equal(basis[6:, 4:], np.eye(3))
    np.testing.assert_array_equal(basis[:3, 2:], np.zeros((3, 5)))
    np.testing.assert_array_equal(basis[3:6, :2], np.zeros((3, 2)))


def test_joint_history_exact_recovery_resets_visible_endpoint_and_counts_budget():
    problem = _problem()
    model, q, v, _, _, _, state, _ = problem
    result, spent = _estimate(problem)

    assert result["resolved"] is True
    np.testing.assert_allclose(result["history"][:, 0, :4], np.concatenate((q[:, 0], v[:, 0]), axis=1),
                               atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result["history"][:, 0, 4:], state[:, 4:], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result["origin"][:, :2], q[:, -1], atol=0, rtol=0)
    np.testing.assert_allclose(result["origin"][:, 2:4], v[:, -1], atol=0, rtol=0)
    np.testing.assert_allclose(result["origin"][:, 4:], result["history"][:, -1, 4:], atol=0, rtol=0)
    assert len(model.calls) == 1
    initial, rollout_power, rollout_equilibrium, count, max_step = model.calls[0]
    assert initial.shape == (len(q) + 4, 6)
    assert count == 10
    assert max_step == 0.01
    assert rollout_power.shape == rollout_equilibrium.shape == (len(q) + 4, 2)
    np.testing.assert_array_equal(rollout_power[len(q):], np.zeros((4, 2)))
    assert spent == [("rollouts", 1), ("svd", 1), ("matrix_products", 3),
                     ("matrix_products", 11)]
    assert sum(amount for operation, amount in spent if operation == "matrix_products") == 14


def test_frame_zero_is_estimated_instead_of_hard_fixed():
    problem = _problem()
    result, _ = _estimate(problem)
    changed = list(problem)
    changed_q = problem[1].copy()
    changed_q[:, 0, :] += 0.02*problem[-1][:2, 0][None, :]
    changed[1] = changed_q
    changed_result, _ = _estimate(tuple(changed))

    assert not np.allclose(changed_result["initial_coordinates"], result["initial_coordinates"])
    assert not np.allclose(changed_result["history"][:, 0, :2], result["history"][:, 0, :2])


def test_rank_deficient_joint_observation_has_no_estimated_origin_or_history():
    result, spent = _estimate(_problem(rank_deficient=True))

    assert result["resolved"] is False
    assert result["rank"] < result["initial_basis"].shape[1]
    assert "origin" not in result
    assert "history" not in result
    assert "hidden_start" not in result
    assert spent == [("rollouts", 1), ("svd", 1)]


@pytest.mark.parametrize("case", ["shape", "scale", "nonfinite", "empty"])
def test_joint_observer_rejects_invalid_inputs(case):
    problem = list(_problem())
    if case == "shape":
        problem[1] = problem[1][:, :-1]
    elif case == "scale":
        problem[5][0] = 0.0
    elif case == "nonfinite":
        problem[2][0, 1, 0] = np.nan
    elif case == "empty":
        problem[1] = problem[1][:0]
        problem[2] = problem[2][:0]
        problem[3] = problem[3][:0]
        problem[4] = problem[4][:0]
    model, q, v, power, equilibrium, scale, *_ = problem
    with pytest.raises(ValueError, match="invalid causal joint observation problem"):
        estimate_joint_history(model, q, v, power, equilibrium, scale,
                               max_step=0.01, spend=lambda *_: None)
    assert model.calls == []


@pytest.mark.parametrize("backbone,latent", [("sine", 2), ("linear", 1)])
def test_joint_observer_rejects_model_structure_before_rollout(backbone, latent):
    problem = list(_problem())
    problem[0].backbone_kind = backbone
    problem[0].latent_dimension = latent
    model, q, v, power, equilibrium, scale, *_ = problem

    with pytest.raises(ValueError, match="joint observer requires a linear model"):
        estimate_joint_history(model, q, v, power, equilibrium, scale,
                               max_step=0.01, spend=lambda *_: None)
    assert model.calls == []


@pytest.mark.parametrize("mass", [[1.0], [1.0, 0.0], [1.0, np.nan]])
def test_initial_state_basis_rejects_invalid_masses(mass):
    with pytest.raises(ValueError, match="positive finite masses are required"):
        initial_state_basis(mass)
