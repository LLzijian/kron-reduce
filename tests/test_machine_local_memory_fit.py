"""NumPy-only checks for the exact memory convolution and tau sensitivities."""

import numpy as np
import pytest

from kron_reduce.evaluation import machine_local_memory_fit as module


def _linear_nodes(module, times, slope):
    nodes = np.asarray(module.GAUSS_NODES)
    frame_times = times[:-1, None] + np.diff(times)[:, None] * nodes[None, :]
    return frame_times[..., None] * np.asarray(slope)[None, None, :]


def test_linear_displacement_has_analytic_exponential_memory_response():
    times = np.arange(9, dtype=float) * 0.01
    slope = np.linspace(-0.4, 0.5, 10)
    tau = np.linspace(0.08, 0.3, 10)
    h, dh, energy, denergy = module.memory_terms(_linear_nodes(module, times, slope), times, tau)
    elapsed = times[:, None]
    rise = -np.expm1(-elapsed / tau[None, :])
    expected = slope[None, :] * (tau[None, :] * elapsed - tau[None, :]**2 * rise)
    expected_derivative = slope[None, :] * (
        elapsed - 2 * tau[None, :] * rise + elapsed * np.exp(-elapsed / tau[None, :])
    )
    np.testing.assert_allclose(h, expected, rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(dh, expected_derivative, rtol=2e-9, atol=2e-11)
    assert energy.shape == h.shape
    assert denergy.shape == h.shape


def test_tau_derivatives_match_central_differences_for_general_cubic_path():
    times = np.arange(9, dtype=float) * 0.01
    phase = np.asarray(module.GAUSS_NODES)[None, :, None]
    interval = np.arange(len(times) - 1, dtype=float)[:, None, None]
    machine = np.arange(10, dtype=float)[None, None, :]
    q_nodes = 0.01 * (interval + phase) ** 2 * np.cos(0.2 * machine + phase)
    tau = np.linspace(0.09, 0.24, 10)
    h, dh, energy, denergy = module.memory_terms(q_nodes, times, tau)
    eps = 1e-6
    plus = module.memory_terms(q_nodes, times, tau + eps)
    minus = module.memory_terms(q_nodes, times, tau - eps)
    np.testing.assert_allclose(dh, (plus[0] - minus[0]) / (2 * eps), rtol=2e-6, atol=2e-9)
    np.testing.assert_allclose(denergy, (plus[2] - minus[2]) / (2 * eps), rtol=2e-6, atol=2e-9)
    assert np.isfinite(h).all() and np.isfinite(dh).all()
    assert np.isfinite(energy).all() and np.isfinite(denergy).all()


def test_zero_displacement_has_zero_memory_and_tau_derivative():
    times = np.arange(9, dtype=float) * 0.01
    q_nodes = np.zeros((len(times) - 1, 8, 10))
    h, dh, energy, denergy = module.memory_terms(q_nodes, times, np.full(10, 0.15))
    np.testing.assert_array_equal(h, np.zeros((len(times), 10)))
    np.testing.assert_array_equal(dh, np.zeros((len(times), 10)))
    assert np.all(energy[1:] > 0) and np.all(denergy[1:] > 0)


@pytest.mark.parametrize("intervals", [4, 17])
def test_memory_terms_supports_variable_length_observation_windows(intervals):
    times = np.arange(intervals + 1, dtype=float) * module.DT
    q_nodes = np.ones((intervals, len(module.GAUSS_NODES), 10)) * 0.02
    result = module.memory_terms(q_nodes, times, np.linspace(0.1, 0.4, 10))
    assert all(value.shape == (intervals + 1, 10) for value in result)
    assert all(np.isfinite(value).all() for value in result)


def test_variable_projection_log_tau_gradient_matches_centered_difference():
    from kron_reduce.models.machine_local_equation import JointMachineLocalModel
    from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence

    n, count = 10, 401
    incidence = complete_incidence(n)
    stiffness = np.linspace(0.2, 0.5, incidence.shape[1])
    edge_damping = np.full(incidence.shape[1], 0.03)
    backbone = NonlinearPassiveClosure(
        np.linspace(1.0, 2.0, n), stiffness, (incidence * edge_damping) @ incidence.T,
        np.zeros((n, 0)), np.zeros((0, 0)), np.zeros((0, 0)), "sine",
    )
    model = JointMachineLocalModel(
        backbone, port=np.linspace(0.2, 0.4, n), tau=np.linspace(0.1, 0.5, n),
    )
    events = []
    for seed, event_count in ((13, count), (31, count // 2 + 1)):
        rng = np.random.default_rng(seed)
        t = np.arange(event_count) * module.DT
        phase = rng.uniform(-np.pi, np.pi, (4, n))
        frequency = rng.uniform(0.3, 4.0, (4, n))
        amplitude = rng.uniform(0.003, 0.03, (4, n))
        angle = t[:, None, None] * frequency[None] + phase[None]
        q = np.sum(amplitude[None] * np.sin(angle), axis=1)
        v = np.sum((amplitude*frequency)[None] * np.cos(angle), axis=1)
        mass = model.mass.numpy()
        q -= ((q @ mass) / mass.sum())[:, None]
        v -= ((v @ mass) / mass.sum())[:, None]
        p0 = rng.uniform(-0.03, 0.03, n)
        p0 -= p0.mean()
        assert np.max(np.abs(q @ mass)) < 1e-12
        assert np.max(np.abs(v @ mass)) < 1e-12
        assert abs(p0.sum()) < 1e-12
        events.append({"x": np.concatenate((q, v), axis=1), "equilibrium": np.zeros(n), "p0": p0})

    class Budget:
        def spend(self, *_):
            return None

    problem = module.MomentumProblem(model, events, np.ones(2 * n))
    # Keep the first physical coefficient on its lower bound at every perturbation.
    problem.lower[0], problem.upper[0] = 0.099, 0.1
    log_tau = np.log(model.tau.detach().numpy())
    value, gradient, linear, condition = problem.solve(log_tau, Budget())
    assert 0.099 - 1e-12 <= linear[0] <= 0.1 + 1e-12
    eps = 2e-5
    direction = np.linspace(-0.7, 0.8, n)
    plus = problem.solve(log_tau + eps * direction, Budget())[0]
    minus = problem.solve(log_tau - eps * direction, Budget())[0]
    finite = (plus - minus) / (2 * eps)
    analytic = float(gradient @ direction)
    assert np.isfinite(value) and np.isfinite(condition) and np.isfinite(analytic) and np.isfinite(finite)
    assert finite == pytest.approx(analytic, rel=3e-3, abs=2e-9)


def test_momentum_fit_stops_cleanly_at_objective_call_cap(monkeypatch):
    from kron_reduce.models.machine_local_equation import JointMachineLocalModel
    from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence

    n = 10
    incidence = complete_incidence(n)
    stiffness = np.linspace(0.2, 0.5, incidence.shape[1])
    edge_damping = np.full(incidence.shape[1], 0.03)
    backbone = NonlinearPassiveClosure(
        np.linspace(1.0, 2.0, n), stiffness, (incidence * edge_damping) @ incidence.T,
        np.zeros((n, 0)), np.zeros((0, 0)), np.zeros((0, 0)), "sine",
    )
    model = JointMachineLocalModel(
        backbone, port=np.linspace(0.2, 0.4, n), tau=np.linspace(0.1, 0.5, n),
    )
    event = {"x": np.zeros((12, 20)), "equilibrium": np.zeros(n), "p0": np.zeros(n)}
    linear = np.r_[np.ones(incidence.shape[1]), np.ones(n-1), np.full(n, 0.1)]

    class Budget:
        def __init__(self):
            self.counts = {}

        def spend(self, name, amount=1):
            self.counts[name] = self.counts.get(name, 0) + amount

    class FakeProblem:
        def __init__(self, *_):
            self.calls = 0

        def solve(self, x, budget):
            self.calls += 1
            budget.spend("momentum_objectives")
            displacement = x - np.log(model.tau.detach().numpy()) - 0.1
            return float(displacement @ displacement), 2 * displacement, linear, 1.0

    monkeypatch.setattr(module, "MomentumProblem", FakeProblem)
    budget = Budget()
    result = module.momentum_fit(
        model, [event], np.ones(20), budget, max_calls=1, monitor=lambda _: (0.25, [0.25]),
    )
    assert result["stop_reason"] == "objective_call_cap"
    assert len(result["train_trace"]) == 1
    assert result["selected_val_nmse"] == pytest.approx(0.25)
    assert result["val_trace"][0]["val_event_nmse"] == [0.25]
    assert budget.counts["momentum_objectives"] == 1
