import numpy as np
import pytest
from scipy.linalg import helmert

from kron_reduce.evaluation import state_port_correctness as evaluation
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.state_port import StatePortClosure
from kron_reduce.workflows import state_port_synthetic as workflow


def test_workflow_requires_operator_approval_before_reading_policy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: calls.append("clean"))
    monkeypatch.setattr(workflow, "_registered_experiment", lambda _root: calls.append("registered") or {})
    monkeypatch.setattr(workflow, "load_state", lambda *_args: calls.append("ledger") or {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: False)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: pytest.fail("policy read before approval"))
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: pytest.fail("binding checked before approval"))
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: pytest.fail("policy read before approval"))

    with pytest.raises(RuntimeError, match="EXPLICIT_STAGE_APPROVAL_REQUIRED"):
        workflow.authorize(tmp_path)
    assert calls == ["clean", "registered", "ledger"]


def test_workflow_rejects_approval_without_active_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    monkeypatch.setattr(workflow, "_registered_experiment", lambda _root: {})
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: True)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "test-digest")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {})

    with pytest.raises(RuntimeError, match="AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(tmp_path)


def test_generic_runner_rejects_bounded_synthetic_experiment(tmp_path, monkeypatch):
    from project_harness import runner

    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        workflow.EXPERIMENT: {"execution_mode": workflow.MODE},
    }})
    with pytest.raises(RuntimeError, match="BOUNDED_SYNTHETIC_CLI_ONLY"):
        runner.run_experiment(tmp_path, workflow.EXPERIMENT, workflow.STAGE)


def test_call_budget_refuses_over_limit_before_next_operation():
    calls = evaluation.Calls({"probe": 1})
    calls.add("probe")
    with pytest.raises(RuntimeError, match="budget exhausted: probe"):
        calls.add("probe")
    assert calls.counts["probe"] == 1
    assert calls.returned["probe"] == 0


def _small_parameters(calls):
    n, r = 3, 2
    basis = helmert(n, full=False).T
    return {
        "mass": np.array([1.0, 1.4, 2.0]),
        "edge_stiffness": np.array([0.8, 1.1, 0.6]),
        "damping": 0.2 * (np.eye(n) - np.ones((n, n)) / n),
        "coupling": 0.15 * basis[:, :r],
        "latent_skew": np.array([[0.0, 0.1], [-0.1, 0.0]]),
        "latent_dissipation": 0.3 * np.eye(r),
        "calls": calls,
    }


def test_small_counted_models_record_construction_and_rhs_failures():
    calls = evaluation.Calls({"model_constructions": 2, "numpy_rhs": 3})
    parameters = _small_parameters(calls)
    baseline = evaluation._Baseline(**parameters)
    port = evaluation._Port(**parameters, delta=np.zeros((2, 2)), port_kind="state")
    assert calls.counts["model_constructions"] == calls.returned["model_constructions"] == 2

    state = np.array([0.1, -0.04, -0.02, 0.03, -0.01, -0.008, 0.2, -0.1])
    power = np.array([0.1, -0.04, -0.06])
    equilibrium = np.array([0.05, -0.02, -0.01])
    np.testing.assert_allclose(
        port.rhs(state, power, equilibrium),
        baseline.rhs(state, power, equilibrium),
        rtol=0,
        atol=0,
    )
    assert calls.counts["numpy_rhs"] == calls.returned["numpy_rhs"] == 2

    with pytest.raises(ValueError):
        port.rhs(np.zeros(7), power, equilibrium)
    assert calls.counts["numpy_rhs"] == 3
    assert calls.returned["numpy_rhs"] == 2


def test_algebra_gate_is_builtin_bool_and_writes_strict_summary_json(tmp_path):
    n, r = 10, 2
    basis = helmert(n, full=False).T
    common = {
        "mass": np.ones(n),
        "edge_stiffness": np.ones(n * (n - 1) // 2),
        "damping": 0.2 * basis @ basis.T,
        "coupling": 0.15 * basis[:, :r],
        "latent_skew": np.zeros((r, r)),
        "latent_dissipation": 0.3 * np.eye(r),
    }
    models = {
        "baseline": NonlinearPassiveClosure(**common),
        "state": StatePortClosure(**common, delta=np.zeros((n - 1, r)), port_kind="state"),
    }
    record = evaluation._algebra(
        models, {"zero": np.zeros(2 * n + r)}, np.zeros(n), np.zeros(n),
        np.zeros((n - 1, r)), 1.0,
    )
    assert type(record["pass"]) is bool
    path = tmp_path / "algebra-summary.json"
    evaluation.write_json(path, {"gates": {"algebra": record["pass"]}, "algebra": record})
    assert '"algebra": true' in path.read_text(encoding="utf-8")
