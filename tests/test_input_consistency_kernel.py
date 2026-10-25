"""Synthetic evidence for the separately registered A17 numerical kernel."""
import numpy as np
import pytest

from kron_reduce.evaluation.input_consistency import (
    linear_stage_power,
    path_preflight,
    reset_visible_origin,
    rollout_fixed_input,
    solve_saved_decomposition,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.physics.coordinates import coi_project


def direct(function):
    return function()


def closure():
    mass = np.linspace(.1, 1.3, 10)
    p = np.eye(10) - np.ones((10, 10)) / 10
    coupling = np.column_stack((np.arange(10) - 4.5, (-1.) ** np.arange(10))) * .01
    return NonlinearPassiveClosure(mass, np.full(45, .04), .03 * p,
                                   coupling, np.array([[0., .6], [-.6, 0.]]),
                                   np.diag([.2, .4]))


@pytest.mark.parametrize("scale", [.0001, .03, 1.])
@pytest.mark.parametrize("step", [.005, .0025])
def test_constant_path_is_bitwise_original_with_nonuniform_clock(scale, step):
    model = closure()
    clock = np.array([0., .009999999999999953, .021, .03999999999999998, .05])
    q = coi_project(np.linspace(-scale, scale, 10), model.mass)
    v = coi_project(np.sin(np.arange(10)) * scale, model.mass)
    z = np.array([.7, -.3]) * scale
    power = np.linspace(-.02, .02, 10)
    eq = np.linspace(-.05, .05, 10)
    original = model.rollout(q, v, z, clock, power, eq, max_step=step)
    count = {"preflight": 0, "rhs": 0}

    def dispatch(name, function):
        count[name] += 1
        return function()

    actual = rollout_fixed_input(
        model, q, v, z, clock, power, eq, max_step=step, linear=False,
        preflight_dispatch=lambda f: dispatch("preflight", f),
        rhs_dispatch=lambda f: dispatch("rhs", f),
        power_dispatch=lambda f: pytest.fail("constant input must bypass PL arithmetic"))
    assert np.array_equal(actual["path"], original)
    assert count == {"preflight": 1, "rhs": 4 * actual["rk4_updates"]}


def test_pl_input_every_rk4_stage_and_model_generated_states():
    model = closure()
    clock = np.r_[0., np.cumsum(np.linspace(.009, .011, 10))]
    pattern = np.arange(10, dtype=float) - 4.5
    powers = np.arange(11)[:, None] * pattern[None, :] * .001
    calls, states, rhs_powers = [], [], []

    def evaluate(function):
        value = function()
        calls.append(value.copy())
        return value

    class RecordingClosure:
        n_nodes, latent_dimension = 10, 2
        mass = model.mass

        def rhs(self, state, power, equilibrium):
            states.append(state.copy())
            rhs_powers.append(power.copy())
            return model.rhs(state, power, equilibrium)

    result = rollout_fixed_input(
        RecordingClosure(), np.zeros(10), np.zeros(10), np.array([.2, -.1]),
        clock, powers, np.zeros(10), max_step=.005, linear=True,
        preflight_dispatch=direct, rhs_dispatch=direct, power_dispatch=evaluate)
    expected = []
    for j, interval in enumerate(np.diff(clock)):
        n = max(1, int(np.ceil(interval / .005 - 1e-12)))
        for k in range(n):
            expected.extend([linear_stage_power(powers[j], powers[j+1], x, n)
                             for x in (k, k+.5, k+.5, k+1)])
    assert len(calls) == 4 * result["rk4_updates"]
    assert np.array_equal(calls, expected)
    assert np.array_equal(rhs_powers, expected)
    assert np.array_equal(calls[-1], powers[-1])
    assert np.any(np.stack(states)[1:, :20] != 0)
    assert np.array_equal(powers, np.arange(11)[:, None] * pattern[None, :] * .001)


def test_saved_decomposition_single_new_solution_without_any_factorization(monkeypatch):
    rng = np.random.default_rng(56)
    matrix = rng.normal(size=(100, 2))
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    b, impulse = rng.normal(size=100), rng.normal(size=100) * .03
    expected = np.linalg.lstsq(matrix, b + impulse, rcond=None)[0]
    threshold = float(64 * np.finfo(float).eps * 100 * s[0])
    for method in ("svd", "solve", "lstsq", "pinv", "inv", "qr", "eig", "eigvalsh"):
        monkeypatch.setattr(np.linalg, method, lambda *a, **k: pytest.fail("new factorization/solver"))
    calls = []

    def dispatch(function):
        calls.append("ls")
        return function()

    out = solve_saved_decomposition(U=U, singular=s, Vh=Vh, b=b,
                                   weighted_impulse=impulse, rank=2,
                                   threshold=threshold, natural_norm=0., dispatch=dispatch)
    assert calls == ["ls"]
    assert np.allclose(out["z0"], expected, rtol=0, atol=1e-15)
    assert np.array_equal(out["rhs"], b + impulse)
    with pytest.raises(ValueError, match="rank"):
        solve_saved_decomposition(U=U, singular=s, Vh=Vh, b=b,
                                 weighted_impulse=impulse, rank=1,
                                 threshold=threshold, natural_norm=0., dispatch=dispatch)
    assert calls == ["ls"]


def test_reset_uses_own_hidden_state_and_reports_observation_energy_change():
    model = closure()
    history = np.zeros((11, 22))
    history[-1, 20:] = [.4, -.6]
    observed = np.r_[coi_project(np.linspace(-.1, .1, 10), model.mass), np.zeros(10)]
    before = history.copy()
    reset = reset_visible_origin(model, history, observed, np.zeros(10))
    assert np.array_equal(reset["origin"][:20], observed)
    assert np.array_equal(reset["origin"][20:], history[-1, 20:])
    assert np.array_equal(history, before)
    assert reset["reset_energy_change"] == reset["energy_after"] - reset["energy_before"]
    assert reset["reset_energy_change"] != 0


@pytest.mark.parametrize("bad", ["coi", "power", "clock", "future_points"])
def test_preflight_rejects_before_any_rhs(bad):
    model = closure()
    q, v, z = np.zeros(10), np.zeros(10), np.zeros(2)
    clock, power = np.linspace(0, .1, 11), np.zeros((11, 10))
    if bad == "coi":
        q[0] = 1.
    elif bad == "power":
        power[0, 0] = 1.
    elif bad == "clock":
        clock[1] = clock[0]
    else:
        clock, power = np.linspace(0, .11, 12), np.zeros((12, 10))
    with pytest.raises(ValueError):
        path_preflight(model, q, v, z, clock, power, np.zeros(10), .005, linear=True)


def test_numerical_failure_retains_prefix_without_another_rhs():
    model = closure()
    calls = []

    def failing_rhs(function):
        calls.append(1)
        return np.full(22, np.nan)

    with pytest.raises(FloatingPointError) as caught:
        rollout_fixed_input(model, np.zeros(10), np.zeros(10), np.zeros(2),
                            np.linspace(0, .1, 11), np.zeros((11, 10)), np.zeros(10),
                            max_step=.005, linear=True, preflight_dispatch=direct,
                            rhs_dispatch=failing_rhs, power_dispatch=direct)
    assert len(calls) == 4
    partial = caught.value.partial_record
    assert partial["output_prefix"].shape == (1, 22)
    assert partial["completed_updates"] == 0


@pytest.mark.parametrize("failure", ["rhs", "input", "sink"])
def test_dispatch_failure_keeps_returned_stages_and_underlying_record(failure):
    model = closure()
    seen = {"rhs": 0, "input": 0}
    original = {"upstream": "must remain unchanged"}
    error = OSError("injected persistence failure")
    error.partial_record = original

    def dispatch(kind, function):
        seen[kind] += 1
        if kind == failure and seen[kind] == 3:
            raise error
        return function()

    def update_sink(_):
        if failure == "sink":
            raise error

    with pytest.raises(OSError) as caught:
        rollout_fixed_input(
            model, np.zeros(10), np.zeros(10), np.zeros(2), np.linspace(0, .1, 11),
            np.zeros((11, 10)), np.zeros(10), max_step=.005, linear=True,
            preflight_dispatch=direct, rhs_dispatch=lambda f: dispatch("rhs", f),
            power_dispatch=lambda f: dispatch("input", f), update_sink=update_sink)
    assert caught.value is error and caught.value.partial_record is original
    partial = error.input_consistency_partial_record
    assert partial["output_prefix"].shape == (1, 22)
    assert partial["current"]["substep_index"] == 0
    assert set(partial["current"]["stages"]) >= {"k1", "k2"}
    assert partial["current"]["stages"]["k2"]["returned_rhs"].shape == (22,)
    assert seen["rhs"] == (4 if failure == "sink" else 3 if failure == "rhs" else 2)


def test_reset_subtraction_overflow_retains_both_finite_energies():
    class LargeEnergy:
        def energy(self, state, equilibrium):
            return 1e308 if state[0] else -1e308

    observed = np.zeros(20)
    observed[0] = 1
    with pytest.raises(FloatingPointError) as caught:
        reset_visible_origin(LargeEnergy(), np.zeros((11, 22)), observed, np.zeros(10))
    partial = caught.value.partial_record
    assert partial["energy_before"] == -1e308
    assert partial["energy_after"] == 1e308
    assert np.isinf(partial["reset_energy_change"])
