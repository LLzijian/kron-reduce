import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kron_reduce.evaluation import machine_local_clean_learning as evaluation
from kron_reduce.workflows import machine_local_clean_learning as workflow


def _registration():
    name, _ = workflow.STAGES["learning"]
    return {
        "module": workflow.MODULE,
        "workflow": workflow.WORKFLOW,
        "stage": "learning",
        "config": name + ".toml",
        "execution_mode": "bounded_synthetic_cli_only",
        "dataset": "synthetic_benchmark_registry_v1",
        "data_policy": "analytic_synthetic_only",
        "reads_validation_performance": False,
        "reads_test_performance": False,
    }


def _authorization_prefix(monkeypatch, *, entry=None):
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    name, _ = workflow.STAGES["learning"]
    experiments = {} if entry is None else {name: entry}
    monkeypatch.setattr(workflow, "load_manifest", lambda _root: {"experiments": experiments})
    monkeypatch.setattr(workflow, "sys", SimpleNamespace(version_info=(3, 12)))


def test_clean_learning_workflow_and_runtime_binding_identities_are_fixed():
    assert workflow.WORKFLOW == "machine_local_clean_learning_v1"
    assert workflow.MODULE == "kron_reduce.workflows.machine_local_clean_learning"
    assert workflow.PLAN == "configs/proposals/machine_local_clean_learning_v1.json"
    assert workflow.PROTOCOL == "docs/protocols/machine_local_clean_learning_v1_zh.md"
    assert workflow.REVIEW == "docs/research/machine_local_clean_learning_review_20261113.json"
    assert workflow.STAGES == {"learning": ("machine_local_clean_learning_v1", 1)}
    assert workflow.scope("learning")["fit_attempts"] == 1
    assert workflow.scope("learning")["optimizer_updates"] == 240
    assert {
        "src/kron_reduce/models/differentiable_history.py",
        "src/kron_reduce/models/machine_local.py",
        "src/kron_reduce/evaluation/machine_local_synthetic.py",
        "src/kron_reduce/evaluation/machine_local_observer.py",
        "src/kron_reduce/evaluation/machine_local_clean_learning.py",
        "src/kron_reduce/workflows/machine_local_clean_learning.py",
        "tests/test_differentiable_history.py",
        "tests/test_clean_learning_workflow.py",
    }.issubset(workflow.REQUIRED_BINDINGS)


def test_train_precision_failure_still_scores_val_and_test(tmp_path, monkeypatch):
    class Budget:
        @staticmethod
        def scope(_name):
            return nullcontext()

    calls = []

    def evaluate(_model, _inputs, truth, _scale, _budget, _directory, key):
        role = key.rsplit("_", 1)[-1]
        calls.append(role)
        if role == "train":
            raise torch.linalg.LinAlgError("mock precision failure")
        return {"status": "PASS", "evaluated_events": len(truth), "accuracy_pass": False}

    monkeypatch.setattr(evaluation, "write_json", lambda *_args: None)
    inputs = (np.zeros((64, 11, 10)), np.zeros((64, 11, 10)), np.zeros((64, 10)),
              np.zeros((64, 10)), np.ones(20))
    rows = evaluation.evaluate_all(
        object(), inputs, np.zeros((64, 811, 30)), np.ones(20), Budget(), tmp_path,
        "candidate", evaluator=evaluate,
    )

    assert calls == ["train", "val", "test"]
    assert rows["train"]["status"] == "NUMERICAL_FAILURE"
    assert rows["val"]["evaluated_events"] == 16 and rows["val"]["accuracy_pass"] is False
    assert rows["test"]["evaluated_events"] == 16 and rows["test"]["accuracy_pass"] is False


@pytest.mark.parametrize("bad_field", ["module", "workflow", "stage", "execution_mode"])
def test_registered_rejects_wrong_clean_learning_identity_before_policy_read(tmp_path, monkeypatch, bad_field):
    entry = _registration()
    entry[bad_field] = "wrong-clean-learning-identity"
    _authorization_prefix(monkeypatch, entry=entry)
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("invalid registration read policy"))
    with pytest.raises(RuntimeError, match="MACHINE_CLEAN_LEARNING_REGISTRATION_MISMATCH"):
        workflow.registered(tmp_path, "learning")


def test_authorize_rejects_unregistered_stage_before_policy_read(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch)
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("unregistered run read policy"))
    with pytest.raises(RuntimeError, match="MACHINE_CLEAN_LEARNING_REGISTRATION_MISMATCH"):
        workflow.authorize(tmp_path, "learning")


def test_authorize_requires_explicit_stage_approval(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch, entry=_registration())
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: False)
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("unapproved stage read policy"))
    with pytest.raises(RuntimeError, match="MACHINE_CLEAN_LEARNING_STAGE_APPROVAL_REQUIRED"):
        workflow.authorize(tmp_path, "learning")


def test_authorize_rejects_consumed_authorization(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch, entry=_registration())
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: True)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "digest")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    authorization = {
        **workflow.scope("learning"),
        "status": "consumed",
        "plan_path": workflow.PLAN,
        "protocol_path": workflow.PROTOCOL,
        "review_path": workflow.REVIEW,
        "implementation_bindings": {},
    }
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {
        "bounded_synthetic_evaluations": {authorization["authorization"]: authorization},
    })
    with pytest.raises(RuntimeError, match="MACHINE_CLEAN_LEARNING_AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(tmp_path, "learning")


def test_authorize_rejects_incomplete_source_bindings(tmp_path, monkeypatch):
    _authorization_prefix(monkeypatch, entry=_registration())
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: True)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "digest")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    authorization = {
        **workflow.scope("learning"),
        "status": "active",
        "plan_path": workflow.PLAN,
        "protocol_path": workflow.PROTOCOL,
        "review_path": workflow.REVIEW,
        "implementation_bindings": {},
    }
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {
        "bounded_synthetic_evaluations": {authorization["authorization"]: authorization},
    })
    with pytest.raises(RuntimeError, match="MACHINE_CLEAN_LEARNING_BINDINGS_INCOMPLETE"):
        workflow.authorize(tmp_path, "learning")


def test_generic_runner_rejects_bounded_synthetic_cli_before_data_or_policy_access(tmp_path, monkeypatch):
    from project_harness import runner

    name, _ = workflow.STAGES["learning"]
    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        name: {"execution_mode": "bounded_synthetic_cli_only"},
    }})
    touched = []
    for attr in ("validate_run_policy", "load_policy", "verify_data", "_run_experiment"):
        monkeypatch.setattr(runner, attr, lambda *args, _attr=attr, **kwargs: touched.append(_attr))
    with pytest.raises(RuntimeError, match="BOUNDED_SYNTHETIC_CLI_ONLY"):
        runner.run_experiment(tmp_path, name, "learning")
    assert touched == []


def test_run_rejects_wrong_experiment_identity_before_authorization(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    result_dir = results / "run-1"
    result_dir.mkdir()
    monkeypatch.setattr(workflow, "registered", lambda *_args: _registration())
    monkeypatch.setattr(workflow, "authorize", lambda *_args: pytest.fail("wrong identity reached authorization"))
    with pytest.raises(RuntimeError, match="MACHINE_CLEAN_LEARNING_RUN_IDENTITY_MISMATCH"):
        workflow.run(root=root, experiment={"module": "wrong"}, stage="learning",
                     result_dir=result_dir, run_id="run-1")


def test_failed_run_consumes_clean_learning_reservation_and_cannot_be_retried(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    common = tmp_path / "git-common"
    common.mkdir()
    experiment = _registration()
    plan = {"stages": {"learning": {"operation_limits": {}}}}
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
    with pytest.raises(RuntimeError, match="MACHINE_CLEAN_LEARNING_SOURCE_CHANGED"):
        workflow.run(root=root, experiment=experiment, stage="learning", result_dir=first, run_id="run-1")

    reservation = common / "machine_local_clean_learning_budgets" / workflow.scope("learning")["authorization"]
    failed = json.loads((reservation / "failed.json").read_text(encoding="utf-8"))
    assert failed["retry_allowed"] is False
    second = results / "run-2"
    second.mkdir()
    with pytest.raises(FileExistsError):
        workflow.run(root=root, experiment=experiment, stage="learning", result_dir=second, run_id="run-2")
    assert calls == ["execute"]





