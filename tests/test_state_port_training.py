import numpy as np
import pytest
from scipy.linalg import helmert

from kron_reduce.evaluation.state_port_training import loss, rollout
from kron_reduce.models.state_port import StatePortClosure


def model(kind="state"):
    n, r = 3, 2
    basis = helmert(n, full=False).T
    return StatePortClosure(
        mass=np.array([1.0, 1.3, 1.7]),
        edge_stiffness=np.array([0.8, 1.1, 0.6]),
        damping=0.15 * basis @ basis.T,
        coupling=0.2 * basis[:, :r],
        latent_skew=np.array([[0.0, 0.08], [-0.08, 0.0]]),
        latent_dissipation=0.25 * np.eye(r),
        delta=np.zeros((n - 1, r)),
        port_kind=kind,
    )


def batch_inputs(m=None, batch=2):
    m = model() if m is None else m
    q = np.array([[0.12, -0.04, 0.0], [-0.08, 0.03, 0.02]])[:batch]
    v = np.array([[0.02, -0.01, 0.0], [-0.01, 0.03, -0.02]])[:batch]
    q = q - (q @ m.mass)[:, None] / m.mass.sum()
    v = v - (v @ m.mass)[:, None] / m.mass.sum()
    z = np.array([[0.3, -0.2], [-0.1, 0.25]])[:batch]
    initial = np.concatenate((q, v, z), axis=1)
    power = np.array([[0.06, -0.02, -0.04], [-0.03, 0.05, -0.02]])[:batch]
    equilibrium = np.array([[0.04, -0.02, 0.01], [-0.03, 0.02, 0.01]])[:batch]
    return initial, power, equilibrium


def test_rollout_preserves_full_origin_and_matches_independent_rows():
    torch = pytest.importorskip("torch")
    closure = model()
    initial, power, equilibrium = batch_inputs(closure)
    path = rollout(closure, initial, power, equilibrium, np.array([3, 3]), max_step=0.005)
    assert path.shape == (2, 4, 8)
    np.testing.assert_array_equal(path[:, 0].detach().numpy(), initial)
    for index in range(2):
        single = rollout(
            closure, initial[index : index + 1], power[index : index + 1],
            equilibrium[index : index + 1], np.array([3]), max_step=0.005,
        )
        torch.testing.assert_close(path[index], single[0], rtol=1e-13, atol=1e-13)
    assert torch.max(torch.abs(path[:, :, :3] @ torch.tensor(closure.mass))) < 1e-12
    assert torch.max(torch.abs(path[:, :, 3:6] @ torch.tensor(closure.mass))) < 1e-12


def test_rollout_stops_short_rows_and_pads_with_last_state():
    closure = model()
    initial, power, equilibrium = batch_inputs(closure)
    transitions = []

    def record(transition, *, batch_size, state, error=None, **_details):
        transitions.append((transition, batch_size, state.copy(), error))

    path = rollout(
        closure, initial, power, equilibrium, np.array([1, 3]), max_step=0.01, calls=record
    ).detach().numpy()
    assert [row[1] for row in transitions if row[0] == "attempted"] == [2] * 4 + [1] * 8
    assert all(row[0] == "returned" for row in transitions if row[0] != "attempted")
    np.testing.assert_array_equal(path[0, 2:], np.repeat(path[0, 1:2], 2, axis=0))
    assert path[1, -1].shape == (8,)


def test_rollout_records_failed_rhs_state(monkeypatch):
    closure = model()
    initial, power, equilibrium = batch_inputs(closure)
    events = []

    def fail(self, delta=None):
        def rhs(state, _power, _equilibrium):
            raise FloatingPointError("synthetic RHS failure")
        return rhs

    monkeypatch.setattr(StatePortClosure, "torch_vector_field", fail)
    def record(transition, *, batch_size, state, error=None, **_details):
        events.append((transition, batch_size, state, error))

    with pytest.raises(FloatingPointError, match="synthetic RHS failure"):
        rollout(closure, initial, power, equilibrium, np.array([1, 2]), max_step=0.01, calls=record)
    assert [row[0] for row in events] == ["attempted", "failed"]
    assert events[0][1] == events[1][1] == 2
    np.testing.assert_array_equal(events[1][2], initial)
    assert isinstance(events[1][3], FloatingPointError)


def test_rollout_stops_when_a_substep_diverges(monkeypatch):
    closure = model()
    initial, power, equilibrium = batch_inputs(closure)
    calls = []

    def divergent(self, delta=None):
        def rhs(state, _power, _equilibrium):
            value = torch.zeros_like(state)
            value[:, 0] = 1e15
            return value
        import torch
        return rhs

    monkeypatch.setattr(StatePortClosure, "torch_vector_field", divergent)
    with pytest.raises(FloatingPointError, match="divergent"):
        rollout(closure, initial, power, equilibrium, np.array([1, 1]), max_step=0.01,
                calls=lambda transition, **_kwargs: calls.append(transition))
    assert calls == ["attempted", "returned"] * 4


def test_future_loss_ignores_padding_and_preserves_event_weights():
    torch = pytest.importorskip("torch")
    path = torch.zeros((2, 4, 8), dtype=torch.float64)
    target = torch.zeros((2, 4, 6), dtype=torch.float64)
    path[0, 2:, :6] = 7.0
    target[0, 2:] = -11.0
    path[1, 1:, :6] = 1.0
    counts = np.array([1, 3])
    weights = torch.tensor([2.0, 3.0], dtype=torch.float64)
    scale = torch.ones(6, dtype=torch.float64)
    score = loss(path, target, weights, scale, counts)
    weighted_score, event_scores = loss(path, target, weights, scale, counts, return_events=True)
    altered = target.clone()
    altered[0, 2:] = 1234.0
    torch.testing.assert_close(score, loss(path, altered, weights, scale, counts))
    torch.testing.assert_close(score, weighted_score)
    torch.testing.assert_close(event_scores, torch.tensor([0.0, 1.0], dtype=torch.float64))
    # Event 0 contributes zero; event 1 has unit normalized error at every valid time.
    assert score.item() == pytest.approx(3.0)


def test_theta_and_state_gradients_match_finite_differences():
    torch = pytest.importorskip("torch")
    closure = model()
    initial, power, equilibrium = batch_inputs(closure)
    counts = np.array([3, 2])
    target = torch.zeros((2, 4, 6), dtype=torch.float64)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    scale = torch.ones(6, dtype=torch.float64)
    theta = torch.tensor([[0.2, -0.1], [0.07, 0.16]], dtype=torch.float64, requires_grad=True)
    coupling_scale = np.linalg.norm(closure.coupling) / np.sqrt(theta.numel())
    delta = coupling_scale * torch.tanh(theta)
    state = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    path = rollout(
        closure, state, power, equilibrium, counts, max_step=0.01, delta=delta
    )
    objective = loss(path, target, weights, scale, counts)
    theta_grad, state_grad = torch.autograd.grad(objective, (theta, state))
    assert torch.isfinite(theta_grad).all() and torch.isfinite(state_grad).all()

    eps = 1e-6
    plus, minus = theta.detach().numpy().copy(), theta.detach().numpy().copy()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    def value(theta_value, state_value=initial):
        value_delta = coupling_scale * torch.tanh(torch.tensor(theta_value, dtype=torch.float64))
        value_path = rollout(
            closure, state_value, power, equilibrium, counts, max_step=0.01, delta=value_delta
        )
        return loss(value_path, target, weights, scale, counts)

    delta_fd = (value(plus) - value(minus)) / (2 * eps)
    assert theta_grad[0, 0].item() == pytest.approx(delta_fd.item(), rel=3e-5, abs=2e-8)

    state_plus, state_minus = initial.copy(), initial.copy()
    state_plus[1, 0] += eps
    state_minus[1, 0] -= eps
    state_fd = (value(theta.detach().numpy(), state_plus) - value(theta.detach().numpy(), state_minus)) / (2 * eps)
    assert state_grad[1, 0].item() == pytest.approx(state_fd.item(), rel=3e-5, abs=2e-8)


def test_training_helpers_reject_bad_alignment_and_runtime_inputs():
    torch = pytest.importorskip("torch")
    closure = model()
    initial, power, equilibrium = batch_inputs(closure)
    with pytest.raises(ValueError, match="align by sample"):
        rollout(closure, initial, power, equilibrium, np.array([1.0, 2.0]), max_step=0.01)
    with pytest.raises(ValueError, match="align by sample"):
        rollout(closure, initial, power[:1], equilibrium, np.array([1, 2]), max_step=0.01)
    with pytest.raises(ValueError, match="max_step must be finite"):
        rollout(closure, initial, power, equilibrium, np.array([1, 2]), max_step=0.02)
    with pytest.raises(ValueError, match="CPU float64"):
        rollout(closure, torch.tensor(initial, dtype=torch.float32), power, equilibrium,
                np.array([1, 2]), max_step=0.01)
    with pytest.raises(ValueError, match="aligned finite inputs"):
        loss(torch.zeros((2, 3, 8), dtype=torch.float64),
             torch.zeros((2, 2, 6), dtype=torch.float64),
             torch.ones(2, dtype=torch.float64),
             torch.ones(6, dtype=torch.float64), np.array([1, 2]))
