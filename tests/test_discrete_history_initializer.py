import numpy as np
import pytest
import torch

from kron_reduce.models.discrete_history_initializer import (
    estimate_history,
    solve_history_state,
)


def _exact_problem(seed=31):
    rng = np.random.default_rng(seed)
    batch, frames, visible, latent = 3, 11, 6, 2
    response = rng.normal(size=(frames, visible, latent))
    baseline = rng.normal(scale=0.02, size=(batch, frames, visible))
    hidden = rng.normal(scale=0.1, size=(batch, latent))
    observed = baseline + np.einsum("tir,br->bti", response, hidden)
    scale = np.linspace(0.7, 1.4, visible)
    return observed, baseline, response, scale, hidden


def test_history_state_exact_recovery_and_budget_counts():
    observed, baseline, response, scale, expected = _exact_problem()
    spent = []
    result = solve_history_state(
        observed, baseline, response, scale,
        spend=lambda operation, amount=1: spent.append((operation, amount)),
    )

    assert result["resolved"] is True
    np.testing.assert_allclose(result["hidden_start"], expected, rtol=1e-12, atol=1e-12)
    assert spent == [("svd", 1), ("matrix_products", 3)]


def test_frame_zero_does_not_affect_history_state_solution():
    observed, baseline, response, scale, _ = _exact_problem()
    reference = solve_history_state(
        observed, baseline, response, scale, spend=lambda *_: None,
    )
    changed_observed, changed_baseline, changed_response = (
        value.copy() for value in (observed, baseline, response)
    )
    changed_observed[:, 0] += 1000
    changed_baseline[:, 0] -= 2000
    changed_response[0] *= -500
    changed = solve_history_state(
        changed_observed, changed_baseline, changed_response, scale,
        spend=lambda *_: None,
    )
    np.testing.assert_array_equal(changed["hidden_start"], reference["hidden_start"])


def test_rank_deficient_observation_is_rejected_without_estimate():
    observed, baseline, response, scale, _ = _exact_problem()
    response[:, :, 1] = response[:, :, 0]
    observed = baseline + np.einsum("tir,br->bti", response, np.zeros((len(observed), 2)))
    spent = []
    result = solve_history_state(
        observed, baseline, response, scale,
        spend=lambda operation, amount=1: spent.append((operation, amount)),
    )
    assert result["resolved"] is False
    assert result["rank"] < response.shape[2]
    assert "hidden_start" not in result
    assert spent == [("svd", 1)]


def test_ill_conditioned_full_rank_observation_is_rejected_without_estimate():
    observed, baseline, response, scale, _ = _exact_problem()
    response[:] = 0
    response[1:, 0, 0] = 1
    response[1:, 1, 1] = 1e-10
    observed = baseline.copy()
    result = solve_history_state(
        observed, baseline, response, scale, spend=lambda *_: None,
    )
    assert result["rank"] == response.shape[2]
    assert result["condition_number"] > 1e8
    assert result["resolved"] is False
    assert "hidden_start" not in result


@pytest.mark.parametrize("case", ["scale", "nonfinite", "shape"])
def test_history_state_rejects_invalid_inputs(case):
    observed, baseline, response, scale, _ = _exact_problem()
    if case == "scale":
        scale[0] = 0
    elif case == "nonfinite":
        observed[0, 1, 0] = np.nan
    else:
        observed = observed[:, :-1]
    with pytest.raises(ValueError, match="invalid eleven-frame observation problem"):
        solve_history_state(observed, baseline, response, scale, spend=lambda *_: None)


class _LinearHistoryModel:
    backbone_kind = "linear"
    n_nodes = 2
    latent_dimension = 2

    def __init__(self):
        self.calls = []

    def rollout(self, initial, power, equilibrium, count, *, max_step, spend):
        self.calls.append((initial.copy(), power.copy(), equilibrium.copy(), count, max_step))
        spend("rollouts")
        states = np.repeat(initial[:, None, :], count + 1, axis=1)
        for t in range(count + 1):
            states[2:, t, :2] = t * np.eye(2)
        return torch.tensor(states, dtype=torch.float64)


def _history_inputs():
    model = _LinearHistoryModel()
    batch, n, r = 2, model.n_nodes, model.latent_dimension
    q = np.zeros((batch, 11, n))
    v = np.zeros_like(q)
    z = np.array([[0.2, -0.1], [-0.05, 0.16]])
    response_qv = np.zeros((11, 2*n, r))
    response_qv[:, :2, :] = np.arange(11)[:, None, None] * np.eye(r)[None]
    observed = np.einsum("tir,br->bti", response_qv, z)
    q, v = observed[:, :, :n], observed[:, :, n:]
    power = np.zeros((batch, n))
    equilibrium = np.zeros_like(power)
    return model, q, v, power, equilibrium, np.ones(2*n), z


def test_estimate_history_uses_only_history_rollout_and_resets_visible_endpoint():
    model, q, v, power, equilibrium, scale, expected = _history_inputs()
    spent = []
    result = estimate_history(
        model, q, v, power, equilibrium, scale, max_step=0.01,
        spend=lambda operation, amount=1: spent.append((operation, amount)),
    )

    assert result["resolved"] is True
    np.testing.assert_allclose(result["hidden_start"], expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(result["origin"][:, :2], q[:, -1])
    np.testing.assert_array_equal(result["origin"][:, 2:4], v[:, -1])
    np.testing.assert_array_equal(result["origin"][:, 4:], result["history"][:, -1, 4:])
    assert len(model.calls) == 1
    assert model.calls[0][3:] == (10, 0.01)
    assert spent == [("rollouts", 1), ("svd", 1), ("matrix_products", 3),
                     ("matrix_products", 11)]
    with pytest.raises(TypeError):
        estimate_history(
            model, q, v, power, equilibrium, scale, max_step=0.01,
            spend=lambda *_: None, future=np.zeros((2, 4)),
        )


def test_estimate_history_rejects_nonlinear_model_before_rollout():
    model, q, v, power, equilibrium, scale, _ = _history_inputs()
    model.backbone_kind = "sine"
    with pytest.raises(ValueError, match="requires a linear backbone"):
        estimate_history(
            model, q, v, power, equilibrium, scale, max_step=0.01,
            spend=lambda *_: None,
        )
    assert model.calls == []
