import json
from types import SimpleNamespace

import pytest

from kron_reduce.workflows import machine_local_joint_observer as workflow


def _registration():
    name, _ = workflow.STAGES["fixed"]
    return {
        "module": workflow.MODULE,
        "workflow": workflow.WORKFLOW,
        "stage": "fixed",
        "config": name + ".toml",
        "execution_mode": "bounded_synthetic_cli_only",
        "dataset": "synthetic_benchmark_registry_v1",
        "data_policy": "analytic_synthetic_only",
        "reads_validation_performance": False,
        "reads_test_performance": False,
    }


def _authorization_prefix(monkeypatch, *, entry=None):
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    name, _ = workflow.STAGES["fixed"]
    experiments = {} if entry is None else {name: entry}
    monkeypatch.setattr(workflow, "load_manifest", lambda _root: {"experiments": experiments})
    monkeypatch.setattr(workflow, "sys", SimpleNamespace(version_info=(3, 12)))


def test_joint_workflow_and_runtime_binding_identities_are_fixed():
    assert workflow.WORKFLOW == "machine_local_joint_observer_v1"
    assert workflow.MODULE == "kron_reduce.workflows.machine_local_joint_observer"
    assert workflow.PLAN == "configs/proposals/machine_local_joint_observer_v1.json"
    assert workflow.PROTOCOL == "docs/protocols/machine_local_joint_observer_v1_zh.md"
    assert workflow.REVIEW == "docs/research/machine_local_joint_observer_review_20261112.json"
    assert workflow.STAGES == {"fixed": ("machine_local_joint_observer_v1", 0)}
    assert workflow.scope("fixed")["fit_attempts"] == 0
    assert workflow.scope("fixed")["optimizer_updates"] == 0
    assert {
        "src/kron_reduce/models/joint_history_initializer.py",
        "src/kron_reduce/models/machine_local.py",
        "src/kron_reduce/evaluation/machine_local_synthetic.py",
        "src/kron_reduce/evaluation/machine_local_initializer_diagnosis.py",
        "src/kron_reduce/evaluation/machine_local_observer.py",
        "src/kron_reduce/evaluation/machine_local_joint_observer.py",
        "src/kron_reduce/workflows/machine_local_joint_observer.py",
        "tests/test_joint_history_initializer.py",
        "tests/test_joint_observer_workflow.py",
    }.issubset(workflow.REQUIRED_BINDINGS)


@pytest.mark.parametrize("bad_field", ["module", "workflow", "stage", "execution_mode"])
def test_registered_rejects_wrong_joint_observer_identity_before_policy_read(tmp_path, monkeypatch, bad_field):
    entry = _registration()
    entry[bad_field] = "wrong-joint-observer-identity"
    _authorization_prefix(monkeypatch, entry=entry)
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("invalid registration read policy"))
    with pytest.raises(RuntimeError, match="MACHINE_JOINT_OBSERVER_REGISTRATION_MISMATCH"):
        workflow.registered(tmp_path, "fixed")


def test_authorize_rejects_unregistered_stage_before_policy_read(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch)
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("unregistered run read policy"))
    with pytest.raises(RuntimeError, match="MACHINE_JOINT_OBSERVER_REGISTRATION_MISMATCH"):
        workflow.authorize(tmp_path, "fixed")


def test_authorize_requires_explicit_stage_approval(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch, entry=_registration())
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: False)
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("unapproved stage read policy"))
    with pytest.raises(RuntimeError, match="MACHINE_JOINT_OBSERVER_STAGE_APPROVAL_REQUIRED"):
        workflow.authorize(tmp_path, "fixed")


def test_authorize_rejects_consumed_authorization(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch, entry=_registration())
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: True)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "digest")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    authorization = {
        **workflow.scope("fixed"),
        "status": "consumed",
        "plan_path": workflow.PLAN,
        "protocol_path": workflow.PROTOCOL,
        "review_path": workflow.REVIEW,
        "implementation_bindings": {},
    }
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {
        "bounded_synthetic_evaluations": {authorization["authorization"]: authorization},
    })
    with pytest.raises(RuntimeError, match="MACHINE_JOINT_OBSERVER_AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(tmp_path, "fixed")


def test_authorize_rejects_incomplete_source_bindings(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch, entry=_registration())
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: True)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "digest")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    authorization = {
        **workflow.scope("fixed"),
        "status": "active",
        "plan_path": workflow.PLAN,
        "protocol_path": workflow.PROTOCOL,
        "review_path": workflow.REVIEW,
        "implementation_bindings": {},
    }
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {
        "bounded_synthetic_evaluations": {authorization["authorization"]: authorization},
    })
    with pytest.raises(RuntimeError, match="MACHINE_JOINT_OBSERVER_BINDINGS_INCOMPLETE"):
        workflow.authorize(tmp_path, "fixed")


def test_generic_runner_rejects_bounded_synthetic_cli_before_data_or_policy_access(tmp_path, monkeypatch):
    from project_harness import runner

    name, _ = workflow.STAGES["fixed"]
    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        name: {"execution_mode": "bounded_synthetic_cli_only"},
    }})
    touched = []
    for attr in ("validate_run_policy", "load_policy", "verify_data", "_run_experiment"):
        monkeypatch.setattr(runner, attr, lambda *args, _attr=attr, **kwargs: touched.append(_attr))
    with pytest.raises(RuntimeError, match="BOUNDED_SYNTHETIC_CLI_ONLY"):
        runner.run_experiment(tmp_path, name, "fixed")
    assert touched == []


def test_run_rejects_wrong_experiment_identity_before_authorization(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    result_dir = results / "run-1"
    result_dir.mkdir()
    monkeypatch.setattr(workflow, "registered", lambda *_args: _registration())
    monkeypatch.setattr(workflow, "authorize", lambda *_args: pytest.fail("wrong identity reached authorization"))
    with pytest.raises(RuntimeError, match="MACHINE_JOINT_OBSERVER_RUN_IDENTITY_MISMATCH"):
        workflow.run(root=root, experiment={"module": "wrong"}, stage="fixed",
                     result_dir=result_dir, run_id="run-1")


def test_failed_run_consumes_joint_reservation_and_cannot_be_retried(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    common = tmp_path / "git-common"
    common.mkdir()
    experiment = _registration()
    plan = {"stages": {"fixed": {"operation_limits": {}}}}
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

    monkeypatch.setattr(workflow, "importlib", SimpleNamespace(
        import_module=lambda _name: SimpleNamespace(execute=execute),
    ))
    first = results / "run-1"
    first.mkdir()
    with pytest.raises(RuntimeError, match="MACHINE_JOINT_OBSERVER_SOURCE_CHANGED"):
        workflow.run(root=root, experiment=experiment, stage="fixed", result_dir=first, run_id="run-1")

    reservation = common / "machine_local_joint_observer_budgets" / workflow.scope("fixed")["authorization"]
    failed = json.loads((reservation / "failed.json").read_text(encoding="utf-8"))
    assert failed["retry_allowed"] is False
    second = results / "run-2"
    second.mkdir()
    with pytest.raises(FileExistsError):
        workflow.run(root=root, experiment=experiment, stage="fixed", result_dir=second, run_id="run-2")
    assert calls == ["execute"]
