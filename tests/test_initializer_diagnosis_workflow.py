import json
from types import SimpleNamespace

import pytest

from kron_reduce.workflows import machine_local_initializer_diagnosis as workflow


def _registration():
    name, _ = workflow.STAGES["diagnosis"]
    return {
        "module": workflow.MODULE,
        "workflow": workflow.WORKFLOW,
        "stage": "diagnosis",
        "config": name + ".toml",
        "execution_mode": "bounded_synthetic_cli_only",
        "dataset": "synthetic_benchmark_registry_v1",
        "data_policy": "analytic_synthetic_only",
        "reads_validation_performance": False,
        "reads_test_performance": False,
    }


def _authorization_prefix(monkeypatch, *, registered=True):
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    if registered:
        name, _ = workflow.STAGES["diagnosis"]
        experiments = {name: _registration()}
    else:
        experiments = {}
    monkeypatch.setattr(workflow, "load_manifest", lambda _root: {"experiments": experiments})


def test_authorize_rejects_registry_mismatch_before_policy_read(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch, registered=False)
    monkeypatch.setattr(
        workflow, "load_policy", lambda *_: pytest.fail("unregistered run read policy")
    )
    with pytest.raises(RuntimeError, match="MACHINE_INITIALIZER_REGISTRATION_MISMATCH"):
        workflow.authorize(tmp_path, "diagnosis")


def test_authorize_rejects_consumed_scope(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch)
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: True)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "digest")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    authorization = {
        **workflow.scope("diagnosis"),
        "status": "consumed",
        "plan_path": workflow.PLAN,
        "protocol_path": workflow.PROTOCOL,
        "review_path": workflow.REVIEW,
        "implementation_bindings": {},
    }
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {
        "bounded_synthetic_evaluations": {authorization["authorization"]: authorization},
    })
    with pytest.raises(RuntimeError, match="MACHINE_INITIALIZER_AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(tmp_path, "diagnosis")


def test_generic_runner_refuses_before_policy_or_data_access(tmp_path, monkeypatch):
    from project_harness import runner

    touched = []
    name, _ = workflow.STAGES["diagnosis"]
    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        name: {"execution_mode": "bounded_synthetic_cli_only"},
    }})
    for attr in ("validate_run_policy", "load_policy", "verify_data", "_run_experiment"):
        monkeypatch.setattr(
            runner, attr, lambda *args, _attr=attr, **kwargs: touched.append(_attr)
        )
    with pytest.raises(RuntimeError, match="BOUNDED_SYNTHETIC_CLI_ONLY"):
        runner.run_experiment(tmp_path, name, "diagnosis")
    assert touched == []


def test_source_change_fails_and_reservation_cannot_be_reused(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    result_root = root / "results"
    result_root.mkdir(parents=True)
    common = tmp_path / "git-common"
    common.mkdir()
    experiment = _registration()
    plan = {"stages": {"diagnosis": {"operation_limits": {}}}}
    monkeypatch.setattr(workflow, "registered", lambda *_args: experiment)
    monkeypatch.setattr(workflow, "authorize", lambda *_args: plan)
    monkeypatch.setattr(workflow, "subprocess", SimpleNamespace(run=lambda *_args, **_kwargs: SimpleNamespace(
        stdout=str(common),
    )))
    commits = iter(("a" * 40, "b" * 40))
    monkeypatch.setattr(workflow, "git_head", lambda _root: next(commits))
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "plan-digest")
    monkeypatch.setattr(workflow, "_verify_loaded_runtime", lambda *_args: None)
    calls = []

    def execute(*, result_dir, **_kwargs):
        calls.append("execute")
        return {"scientific_pass": False}

    monkeypatch.setattr(
        workflow.importlib, "import_module", lambda _name: SimpleNamespace(execute=execute)
    )
    first = result_root / "run-1"
    first.mkdir()
    with pytest.raises(RuntimeError, match="MACHINE_INITIALIZER_SOURCE_CHANGED"):
        workflow.run(
            root=root, experiment=experiment, stage="diagnosis",
            result_dir=first, run_id="run-1",
        )

    reservation = common / "machine_local_initializer_budgets" / workflow.scope("diagnosis")["authorization"]
    failed = json.loads((reservation / "failed.json").read_text(encoding="utf-8"))
    assert failed["retry_allowed"] is False
    second = result_root / "run-2"
    second.mkdir()
    with pytest.raises(FileExistsError):
        workflow.run(
            root=root, experiment=experiment, stage="diagnosis",
            result_dir=second, run_id="run-2",
        )
    assert calls == ["execute"]
