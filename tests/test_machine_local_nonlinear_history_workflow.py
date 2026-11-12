from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from kron_reduce.evaluation import machine_local_nonlinear_history as evaluation
from kron_reduce.workflows import machine_local_nonlinear_history as workflow


def _registration():
    name, _ = workflow.STAGES["s0"]
    return {
        "module": workflow.MODULE,
        "workflow": workflow.WORKFLOW,
        "stage": "s0",
        "config": name + ".toml",
        "execution_mode": "bounded_synthetic_cli_only",
        "dataset": "synthetic_benchmark_registry_v1",
        "data_policy": "analytic_synthetic_only",
        "reads_validation_performance": False,
        "reads_test_performance": False,
    }


def _authorize_fixture(tmp_path, monkeypatch, *, status="active", approved=True, auth_changes=None):
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    monkeypatch.setattr(workflow, "load_manifest", lambda _root: {"experiments": {
        workflow.STAGES["s0"][0]: _registration(),
    }})
    monkeypatch.setattr(workflow, "sys", SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: approved)

    bindings = {name: "source-digest" for name in workflow.REQUIRED_BINDINGS}
    expected_scope = workflow.scope("s0")
    auth = {
        **expected_scope,
        "status": status,
        "plan_path": workflow.PLAN,
        "plan_sha256": "plan-digest",
        "protocol_path": workflow.PROTOCOL,
        "protocol_sha256": "protocol-digest",
        "review_path": workflow.REVIEW,
        "review_sha256": "review-digest",
        "implementation_bindings": bindings,
    }
    auth.update(auth_changes or {})
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "policy-digest")
    verified = []
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: verified.append(_args))
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {
        "bounded_synthetic_evaluations": {expected_scope["authorization"]: auth},
    })
    plan = {
        "implementation_bindings": bindings,
        "protocol_sha256": "protocol-digest",
        "protocol_path": workflow.PROTOCOL,
        "stages": {"s0": {"scope": expected_scope}},
    }
    review = {
        "approved": True,
        "plan_path": workflow.PLAN,
        "plan_sha256": "plan-digest",
        "implementation_bindings": bindings,
    }
    monkeypatch.setattr(workflow, "_read_json", lambda _root, path: plan if path == workflow.PLAN else review)
    return plan, auth, verified


def test_workflow_identity_scope_and_nonlinear_history_bindings_are_frozen():
    assert workflow.WORKFLOW == "machine_local_nonlinear_history_v1"
    assert workflow.MODULE == "kron_reduce.workflows.machine_local_nonlinear_history"
    assert workflow.STAGES == {"s0": ("machine_local_nonlinear_history_v1", 0)}
    assert workflow.scope("s0") == {
        "authorization": "machine_local_nonlinear_history_v1_20261113",
        "experiment": "machine_local_nonlinear_history_v1",
        "workflow": workflow.WORKFLOW,
        "stage": "s0",
        "module": workflow.MODULE,
        "dataset": "synthetic_benchmark_registry_v1",
        "data_policy": "analytic_synthetic_only",
        "max_executions": 1,
        "retry_allowed": False,
        "real_data_reads": 0,
        "fit_attempts": 0,
        "optimizer_updates": 0,
    }
    assert {
        "src/kron_reduce/models/nonlinear_history.py",
        "src/kron_reduce/evaluation/trajectory_stability.py",
        "src/kron_reduce/evaluation/machine_local_nonlinear_history.py",
        "tests/test_machine_local_nonlinear_history_workflow.py",
    }.issubset(workflow.REQUIRED_BINDINGS)


def test_authorize_accepts_matching_active_approved_scope(tmp_path, monkeypatch):
    plan, auth, verified = _authorize_fixture(tmp_path, monkeypatch)

    result = workflow.authorize(tmp_path, "s0")

    assert result is plan
    assert auth["status"] == "active"
    assert len(verified) == 4 + len(workflow.REQUIRED_BINDINGS)


def test_authorize_rejects_missing_approval_before_policy_read(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    monkeypatch.setattr(workflow, "load_manifest", lambda _root: {"experiments": {
        workflow.STAGES["s0"][0]: _registration(),
    }})
    monkeypatch.setattr(workflow, "sys", SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: False)
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("unapproved scope read policy"))

    with pytest.raises(RuntimeError, match="MACHINE_NONLINEAR_HISTORY_STAGE_APPROVAL_REQUIRED"):
        workflow.authorize(tmp_path, "s0")


def test_authorize_rejects_consumed_authorization(tmp_path, monkeypatch):
    _plan, _auth, _verified = _authorize_fixture(tmp_path, monkeypatch, status="consumed")

    with pytest.raises(RuntimeError, match="MACHINE_NONLINEAR_HISTORY_AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(tmp_path, "s0")


@pytest.mark.parametrize("train_result", ["numerical_failure", "high_rmse"])
def test_train_failure_or_high_error_does_not_skip_val_test_and_keeps_stability(
    tmp_path, monkeypatch, train_result,
):
    class Budget:
        @staticmethod
        def scope(_name):
            return nullcontext()

    calls = []

    def evaluator(_model, paths, classes, _scale, _hscale, _budget, _directory, key):
        role = key.rsplit("_", 1)[-1]
        calls.append((role, paths[:, 0, 0].copy()))
        if role == "train" and train_result == "numerical_failure":
            raise FloatingPointError("mock numerical failure")
        return {
            "status": "PASS",
            "main_nrmse": 10. if role == "train" else 0.,
            "evaluated_events": len(paths),
            "allocated_events": len(paths),
            "failed_events": 0,
            "stability": {"accuracy": 0.5, "unresolved": 1, "resolved": 1},
            "classes_seen": len(classes),
        }

    monkeypatch.setattr(evaluation, "write_json", lambda *_args: None)
    paths = np.zeros((6, 811, 30))
    paths[:, 0, 0] = np.arange(6)
    truth_classes = [
        {"label": "unstable" if i % 2 == 0 else "stable", "trigger_index": None}
        for i in range(6)
    ]
    rows = evaluation.evaluate_all(
        object(), paths, truth_classes, np.ones(20), np.ones(20), Budget(), tmp_path,
        "candidate", evaluator=evaluator,
    )

    assert [role for role, _ in calls] == ["train", "val", "test"]
    np.testing.assert_array_equal(calls[0][1], [0, 1])
    np.testing.assert_array_equal(calls[1][1], [2, 3])
    np.testing.assert_array_equal(calls[2][1], [4, 5])
    if train_result == "numerical_failure":
        assert rows["train"]["status"] == "NUMERICAL_FAILURE"
        assert rows["train"]["failed_events"] == 2
        assert rows["train"]["stability"]["unresolved"] == 2
        assert rows["train"]["stability"]["numerical_failures"] == 2
    else:
        assert rows["train"]["main_nrmse"] == 10.
    for role in ("val", "test"):
        assert rows[role]["stability"] == {"accuracy": 0.5, "unresolved": 1, "resolved": 1}
        assert rows[role]["classes_seen"] == 2
