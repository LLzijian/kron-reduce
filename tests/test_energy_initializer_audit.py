from __future__ import annotations

import json

import numpy as np
import pytest

from kron_reduce.evaluation import energy_initializer_audit as audit
from kron_reduce.evaluation.constant_c_observability import Operations
from kron_reduce.evaluation.energy_constrained_initializer import EnergyBallNumericalError


class MockOps:
    def __init__(self):
        self.partial = {}
        self.calls = []

    def call(self, category, function, *, tag=""):
        self.calls.append((category, tag))
        return function()


def _context(*, sine=False, baseline_paths=None, baseline_refinement=None):
    observed = np.arange(11 * 20, dtype=float).reshape(11, 20)
    return {
        "case_id": "S100_clean" if sine else "L100_clean",
        "sine": sine,
        "observed": observed,
        "power": np.zeros(10),
        "transition": np.eye(23),
        "baseline_paths": baseline_paths,
        "baseline_refinement": baseline_refinement,
        "unconstrained": np.array([4.0, -2.0]),
    }


def _tolerances(refinement=1.0):
    return {"refinement": refinement}


def _model():
    return type("MockModel", (), {
        "energy": staticmethod(lambda state, _equilibrium: float(np.asarray(state) @ np.asarray(state))),
    })()


def test_inactive_reuses_saved_paths_without_any_propagation(monkeypatch):
    baseline = {"history": np.zeros((11, 22)), "reset": np.zeros(22),
                "future": np.zeros((801, 22))}
    context = _context(baseline_paths=baseline, baseline_refinement=0.0)
    solution = {"status": "inactive", "z": context["unconstrained"].copy()}
    ops = MockOps()

    def forbidden(*_args, **_kwargs):
        pytest.fail("inactive candidate must reuse saved paths without propagation")

    monkeypatch.setattr(audit, "exact_path", forbidden)
    monkeypatch.setattr(audit, "rk_path", forbidden)
    result = audit.build_candidate_paths(
        context, _model(), solution, "fixed_radius", ops, _tolerances(),
    )

    assert result["paths"] is baseline
    assert result["reused_unconstrained"] is True
    assert result["refinement_max"] == 0.0
    assert result["new_linear_candidates"] == result["new_sine_candidates"] == 0
    assert ops.calls == []


def test_unresolved_candidate_returns_no_paths_and_never_propagates(monkeypatch):
    context = _context()
    ops = MockOps()

    def forbidden(*_args, **_kwargs):
        pytest.fail("unresolved candidate must not enter propagation")

    monkeypatch.setattr(audit, "exact_path", forbidden)
    monkeypatch.setattr(audit, "rk_path", forbidden)
    result = audit.build_candidate_paths(
        context, _model(), {"status": "unresolved", "z": None},
        "oracle_radius", ops, _tolerances(),
    )

    assert result["paths"] is None
    assert result["reused_unconstrained"] is False
    assert result["new_linear_candidates"] == result["new_sine_candidates"] == 0
    assert ops.calls == []


def test_zero_candidate_keeps_observed_visible_endpoint_and_propagated_hidden_origin(monkeypatch):
    context = _context()
    z = np.zeros(2)
    calls = []

    def fake_exact(_transition, initial, intervals, _ops):
        initial = np.array(initial, copy=True)
        calls.append((intervals, initial))
        if intervals == 10:
            path = np.repeat(initial[None, :], 11, axis=0)
            path[-1, 20:] = [7.0, -3.0]
            return path
        assert intervals == 800
        future = np.repeat(initial[None, :], 801, axis=0)
        return future

    monkeypatch.setattr(audit, "exact_path", fake_exact)
    monkeypatch.setattr(audit, "check_path", lambda *_args, **_kwargs: {"checked": True})
    result = audit.build_candidate_paths(
        context, _model(), {"status": "zero_baseline", "z": z},
        "zero", MockOps(), _tolerances(),
    )
    path = result["paths"]

    np.testing.assert_array_equal(calls[0][1][:20], context["observed"][0])
    np.testing.assert_array_equal(calls[0][1][20:], z)
    np.testing.assert_array_equal(path["history"][0, :20], context["observed"][0])
    np.testing.assert_array_equal(path["history"][0, 20:], z)
    np.testing.assert_array_equal(path["reset"][:20], context["observed"][-1])
    np.testing.assert_array_equal(path["reset"][20:], [7.0, -3.0])
    np.testing.assert_array_equal(calls[1][1], path["reset"])
    np.testing.assert_array_equal(path["future"][0], path["reset"])
    assert result["new_linear_candidates"] == 1
    assert result["new_sine_candidates"] == 0


def test_sine_fine_and_coarse_terminal_states_feed_their_own_future(monkeypatch):
    context = _context(sine=True)
    calls = []

    def fake_rk(_model_value, _power, initial, intervals, step, _ops, *, role, method):
        initial = np.array(initial, copy=True)
        calls.append((role, step, method, initial))
        if role == "history":
            path = np.repeat(initial[None, :], 11, axis=0)
            path[-1, 20:] = [1.0 + step, 2.0 + step]
            return path
        assert intervals == 800
        return np.repeat(initial[None, :], 801, axis=0)

    monkeypatch.setattr(audit, "rk_path", fake_rk)
    monkeypatch.setattr(audit, "check_path", lambda *_args, **_kwargs: {"checked": True})
    result = audit.build_candidate_paths(
        context, _model(), {"status": "active", "z": np.array([0.25, -0.5])},
        "oracle_radius", MockOps(), _tolerances(refinement=1.0),
    )

    paths = result["paths"]
    assert [path["step"] for path in paths] == [0.00125, 0.000625]
    assert len(calls) == 4
    for index, path in enumerate(paths):
        history_call, future_call = calls[2 * index:2 * index + 2]
        assert history_call[:3] == ("history", path["step"], "oracle_radius")
        assert future_call[:3] == ("future", path["step"], "oracle_radius")
        np.testing.assert_array_equal(future_call[3], path["reset"])
        np.testing.assert_array_equal(path["reset"][20:], path["history"][-1, 20:])
        np.testing.assert_array_equal(path["future"][0], path["reset"])
    assert not np.array_equal(paths[0]["reset"][20:], paths[1]["reset"][20:])
    assert result["new_linear_candidates"] == 0
    assert result["new_sine_candidates"] == 1
    assert result["refinement_max"] < 1.0


def test_solver_numerical_failure_preserves_diagnostics_without_fallback(monkeypatch):
    context = {
        "radius_oracle": 0.75,
        "matrix": np.eye(2),
        "right": np.array([1.0, 2.0]),
        "svd": {"u": np.eye(2), "s": np.ones(2), "vh": np.eye(2),
                "rank": 2, "threshold": 1e-12},
        "unconstrained": np.array([1.0, 2.0]),
    }
    diagnostics = {
        "status": "numerical_failure", "iterations": 64,
        "secular_evaluations": 80, "trace": [{"label": "upper", "multiplier": 4.0}],
        "bracket_final": [1.0, 5.0], "z": None,
    }
    error = EnergyBallNumericalError("bounded mock failure", diagnostics)
    monkeypatch.setattr(audit, "solve_energy_ball", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    ops = MockOps()

    with pytest.raises(EnergyBallNumericalError, match="bounded mock failure") as raised:
        audit.solve_candidate(context, "oracle_radius", ops)

    assert raised.value is error
    assert ops.partial["solver_failure"] == error.diagnostics
    assert ops.partial["solver_failure"]["trace"]
    assert ops.partial["solver_failure"]["z"] is None


def test_failed_scope_persists_solver_diagnostics(tmp_path, monkeypatch):
    context = {
        "radius_oracle": 0.75,
        "matrix": np.eye(2),
        "right": np.array([1.0, 2.0]),
        "svd": {"u": np.eye(2), "s": np.ones(2), "vh": np.eye(2),
                "rank": 2, "threshold": 1e-12},
        "unconstrained": np.array([1.0, 2.0]),
    }
    diagnostics = {
        "status": "numerical_failure", "iterations": 64,
        "secular_evaluations": 80, "trace": [{"label": "upper", "multiplier": 4.0}],
        "bracket_final": [1.0, 5.0], "z": None,
    }
    monkeypatch.setattr(
        audit, "solve_energy_ball",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            EnergyBallNumericalError("persist mock failure", diagnostics)),
    )
    ops = Operations(tmp_path / "journal", {
        "operator_limits": {"probe": 1},
        "scope_limits": {"diagnostic-case": {"probe": 1}},
    })

    with pytest.raises(EnergyBallNumericalError, match="persist mock failure"), ops.scope(
            "diagnostic-case"):
        audit.solve_candidate(context, "oracle_radius", ops)
    ops.journal.close()

    saved = json.loads((tmp_path / "partial-diagnostic-case.json").read_text(encoding="utf-8"))
    saved_diagnostics = saved["returned_operations"]["solver_failure"]
    assert saved_diagnostics["status"] == "numerical_failure"
    assert saved_diagnostics["trace"] == diagnostics["trace"]
    assert saved_diagnostics["bracket_final"] == diagnostics["bracket_final"]
    assert saved_diagnostics["z"] is None


def test_solver_candidate_uses_only_radius_and_saved_integral_inputs(monkeypatch):
    context = {
        "radius_oracle": 0.75,
        "matrix": np.eye(2),
        "right": np.array([1.0, 2.0]),
        "svd": {"u": np.eye(2), "s": np.ones(2), "vh": np.eye(2),
                "rank": 2, "threshold": 1e-12},
        "unconstrained": np.array([1.0, 2.0]),
    }
    captured = {}

    def fake_solver(matrix, right, svd, radius, *, unconstrained):
        captured.update(matrix=matrix, right=right, svd=svd,
                        radius=radius, unconstrained=unconstrained)
        return {"z": np.array([0.5, 0.5]), "status": "active"}

    monkeypatch.setattr(audit, "solve_energy_ball", fake_solver)
    result = audit.solve_candidate(context, "oracle_radius", MockOps())

    assert captured["matrix"] is context["matrix"]
    assert captured["right"] is context["right"]
    assert captured["svd"] is context["svd"]
    assert captured["unconstrained"] is context["unconstrained"]
    assert captured["radius"] == 0.75
    assert result["radius_id"] == "oracle_radius"


