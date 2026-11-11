import pytest

from kron_reduce.workflows import kiki_state_port_train as workflow


def test_workflow_requires_operator_approval_before_reading_policy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: calls.append("clean"))
    monkeypatch.setattr(workflow, "_registered_experiment", lambda _root: calls.append("registered") or {})
    monkeypatch.setattr(workflow, "load_state", lambda *_args: calls.append("ledger") or {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: False)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: pytest.fail("policy read before approval"))
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: pytest.fail("binding checked before approval"))
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: pytest.fail("policy read before approval"))

    with pytest.raises(RuntimeError, match="STATE_PORT_EXPLICIT_STAGE_APPROVAL_REQUIRED"):
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

    with pytest.raises(RuntimeError, match="STATE_PORT_AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(tmp_path)


def test_generic_runner_rejects_saved_train_experiment(tmp_path, monkeypatch):
    from project_harness import runner

    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        workflow.EXPERIMENT: {"execution_mode": workflow.MODE},
    }})
    with pytest.raises(RuntimeError, match="BOUNDED_HISTORY_CACHE_AUDIT_CLI_ONLY"):
        runner.run_experiment(tmp_path, workflow.EXPERIMENT, workflow.STAGE)
