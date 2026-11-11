import numpy as np
import pytest
from scipy.linalg import helmert

from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.state_port import StatePortClosure


def closure(kind="state", delta=None):
    n, r = 3, 2
    basis = helmert(n, full=False).T
    return StatePortClosure(
        mass=np.array([1.0, 1.4, 2.0]),
        edge_stiffness=np.array([0.8, 1.1, 0.6]),
        damping=0.2 * (np.eye(n) - np.ones((n, n)) / n),
        coupling=0.15 * basis[:, :r],
        latent_skew=np.array([[0.0, 0.1], [-0.1, 0.0]]),
        latent_dissipation=0.3 * np.eye(r),
        delta=np.zeros((n - 1, r)) if delta is None else delta,
        port_kind=kind,
    )


def inputs():
    mass = np.array([1.0, 1.4, 2.0])
    project = lambda value: value - np.dot(mass, value) / mass.sum()
    return (
        np.r_[project(np.array([0.12, -0.07, 0.01])), project(np.array([0.2, -0.1, 0.0])), [0.4, -0.3]],
        np.array([0.2, -0.1, -0.1]),
        project(np.array([0.1, -0.04, -0.01])),
    )


@pytest.mark.parametrize("kind", ["constant", "state"])
def test_zero_delta_matches_base_rhs_energy_and_rollout(kind):
    model = closure(kind)
    base = NonlinearPassiveClosure(
        model.mass, model.edge_stiffness, model.damping, model.coupling,
        model.latent_skew, model.latent_dissipation,
    )
    state, power, equilibrium = inputs()
    np.testing.assert_array_equal(model.rhs(state, power, equilibrium), base.rhs(state, power, equilibrium))
    assert model.energy_identity(state, power, equilibrium)["identity_error"] < 1e-12
    np.testing.assert_allclose(
        model.rollout(*[state[:3], state[3:6], state[6:], np.array([0.0, 0.01]), power, equilibrium]),
        base.rollout(state[:3], state[3:6], state[6:], np.array([0.0, 0.01]), power, equilibrium),
        rtol=0, atol=0,
    )


def test_invalid_parameters_are_rejected_and_delta_is_read_only():
    with pytest.raises(ValueError, match="delta must have shape"):
        closure(delta=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        closure(delta=np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="port_kind"):
        closure(kind="learned")
    model = closure()
    assert not model.delta.flags.writeable


@pytest.mark.parametrize("kind", ["constant", "state"])
def test_torch_rhs_matches_numpy(kind):
    torch = pytest.importorskip("torch")
    delta = np.array([[0.2, -0.1], [0.07, 0.16]])
    model = closure(kind, delta)
    state, power, equilibrium = inputs()
    rhs = model.torch_vector_field(torch.tensor(delta, dtype=torch.float64))
    got = rhs(torch.tensor(state), power, equilibrium).detach().numpy()
    np.testing.assert_allclose(got, model.rhs(state, power, equilibrium), rtol=1e-13, atol=1e-13)


def test_state_port_autograd_includes_state_and_delta_derivatives():
    torch = pytest.importorskip("torch")
    model = closure("state")
    state0, power, equilibrium = inputs()
    delta = torch.tensor([[0.2, -0.1], [0.07, 0.16]], dtype=torch.float64, requires_grad=True)
    rhs = model.torch_vector_field(delta)
    state = torch.tensor(state0, requires_grad=True)
    direction = torch.arange(1, state.numel() + 1, dtype=torch.float64) / 10
    objective = rhs(state, power, equilibrium) @ direction
    state_grad, delta_grad = torch.autograd.grad(objective, (state, delta))
    assert torch.isfinite(state_grad).all() and torch.isfinite(delta_grad).all()

    eps = 1e-6
    state_plus, state_minus = state0.copy(), state0.copy()
    state_plus[0] += eps
    state_minus[0] -= eps
    delta0 = delta.detach().numpy()
    rhs_state = model.torch_vector_field(torch.tensor(delta0, dtype=torch.float64))
    finite_state = (
        rhs_state(torch.tensor(state_plus), power, equilibrium) @ direction
        - rhs_state(torch.tensor(state_minus), power, equilibrium) @ direction
    ) / (2 * eps)
    plus, minus = delta0.copy(), delta0.copy()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    finite_delta = (
        model.torch_vector_field(torch.tensor(plus, dtype=torch.float64))(torch.tensor(state0), power, equilibrium) @ direction
        - model.torch_vector_field(torch.tensor(minus, dtype=torch.float64))(torch.tensor(state0), power, equilibrium) @ direction
    ) / (2 * eps)
    assert state_grad[0].item() == pytest.approx(finite_state.item(), rel=2e-6, abs=2e-8)
    assert delta_grad[0, 0].item() == pytest.approx(finite_delta.item(), rel=2e-6, abs=2e-8)


@pytest.mark.parametrize("kind", ["constant", "state"])
def test_batched_torch_rhs_matches_stacked_single_rhs(kind):
    torch = pytest.importorskip("torch")
    delta = np.array([[0.2, -0.1], [0.07, 0.16]])
    model = closure(kind, delta)
    states = np.stack([inputs()[0], inputs()[0] + np.array([.03, -.02, .01, .01, .02, -.03, .1, -.04])])
    powers = np.array([[.2, -.1, -.1], [-.1, .04, .06]])
    equilibria = np.array([[.1, -.04, -.01], [-.02, .03, -.01]])
    field = model.torch_vector_field(torch.tensor(delta, dtype=torch.float64))
    batched = field(torch.tensor(states), powers, equilibria).detach().numpy()
    singles = np.stack([
        field(torch.tensor(state), power, equilibrium).detach().numpy()
        for state, power, equilibrium in zip(states, powers, equilibria, strict=True)
    ])
    np.testing.assert_allclose(batched, singles, rtol=1e-13, atol=1e-13)


def test_batched_torch_rhs_keeps_state_and_shared_delta_gradients_and_rejects_bad_inputs():
    torch = pytest.importorskip("torch")
    model = closure("state")
    delta0 = np.array([[0.2, -0.1], [0.07, 0.16]])
    delta = torch.tensor(delta0, dtype=torch.float64, requires_grad=True)
    field = model.torch_vector_field(delta)
    states0 = np.stack([inputs()[0], inputs()[0] + np.array([.03, -.02, .01, .01, .02, -.03, .1, -.04])])
    states = torch.tensor(states0, dtype=torch.float64, requires_grad=True)
    powers = np.array([[.2, -.1, -.1], [-.1, .04, .06]])
    equilibria = np.array([[.1, -.04, -.01], [-.02, .03, -.01]])
    direction = torch.arange(1, states.numel() + 1, dtype=torch.float64).reshape_as(states) / 10
    objective = (field(states, powers, equilibria) * direction).sum()
    state_grad, delta_grad = torch.autograd.grad(objective, (states, delta))
    assert torch.isfinite(state_grad).all() and torch.isfinite(delta_grad).all()

    eps = 1e-6
    plus, minus = states0.copy(), states0.copy()
    plus[1, 0] += eps
    minus[1, 0] -= eps
    fixed_delta = torch.tensor(delta0, dtype=torch.float64)
    field_fixed = model.torch_vector_field(fixed_delta)
    state_fd = (
        (field_fixed(torch.tensor(plus, dtype=torch.float64), powers, equilibria) * direction).sum()
        - (field_fixed(torch.tensor(minus, dtype=torch.float64), powers, equilibria) * direction).sum()
    ) / (2 * eps)
    delta_plus, delta_minus = delta0.copy(), delta0.copy()
    delta_plus[0, 0] += eps
    delta_minus[0, 0] -= eps
    delta_fd = (
        (model.torch_vector_field(torch.tensor(delta_plus, dtype=torch.float64))(
            torch.tensor(states0, dtype=torch.float64), powers, equilibria) * direction).sum()
        - (model.torch_vector_field(torch.tensor(delta_minus, dtype=torch.float64))(
            torch.tensor(states0, dtype=torch.float64), powers, equilibria) * direction).sum()
    ) / (2 * eps)
    assert state_grad[1, 0].item() == pytest.approx(state_fd.item(), rel=2e-6, abs=2e-8)
    assert delta_grad[0, 0].item() == pytest.approx(delta_fd.item(), rel=2e-6, abs=2e-8)

    with pytest.raises(ValueError, match="wrong dimension"):
        field(torch.zeros((2, 7), dtype=torch.float64), powers, equilibria)
    with pytest.raises(ValueError, match="batch, nodes"):
        field(torch.tensor(states0, dtype=torch.float64), powers[:1], equilibria)
    with pytest.raises(ValueError, match="CPU float64"):
        field(torch.tensor(states0, dtype=torch.float32), powers, equilibria)


def test_one_sided_port_update_breaks_energy_identity():
    model = closure("state", np.array([[0.2, -0.1], [0.07, 0.16]]))
    state, power, equilibrium = inputs()
    n = model.n_nodes
    q, v, z = state[:n], state[n : 2 * n], state[2 * n :]
    increment = model._increment(q)
    correct = model.rhs(state, power, equilibrium)
    one_sided = correct.copy()
    one_sided[2 * n :] -= increment.T @ (model._projections[1] @ v)
    b = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1]], dtype=float)
    gradient = np.sin(b.T @ (q + equilibrium)) - np.sin(b.T @ equilibrium)
    energy_gradient = np.r_[b @ (model.edge_stiffness * gradient), model.mass * v, z]
    expected = v @ power - v @ model.damping @ v - z @ model.latent_dissipation @ z
    assert abs(energy_gradient @ correct - expected) < 1e-12
    broken_rate = energy_gradient @ one_sided
    assert abs(broken_rate - expected) > 1e-5
