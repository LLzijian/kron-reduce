"""Inactive A17 entry points must stop before any source access or reservation."""
from __future__ import annotations

import pytest

from kron_reduce.workflows import kiki_input_consistency as workflow
from project_harness import runner


@pytest.mark.parametrize("status", [None, "implementation_review_pending", "consumed", "user_approved"])
def test_authorization_refuses_missing_or_incomplete_exact_review_before_io(tmp_path, monkeypatch, status):
    monkeypatch.setattr(workflow, "validate_run_policy", lambda *args: None)
    monkeypatch.setattr(workflow, "_verify_development_text_bindings", lambda *args: None)
    monkeypatch.setattr(workflow, "load_policy", lambda root: {
        "development_evaluations": {workflow.AUTH: {"status": status}}})
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *args: pytest.fail("inactive scope must not verify files"))
    monkeypatch.setattr(workflow, "assert_clean", lambda *args: pytest.fail("inactive scope must stop before runtime"))
    with pytest.raises(PermissionError, match="inactive"):
        workflow.authorization(tmp_path, {}, {})


def test_generic_runner_refuses_before_policy_or_generic_data_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_manifest", lambda root: {"experiments": {
        workflow.EXPERIMENT: {"execution_mode": workflow.MODE}}})
    monkeypatch.setattr(runner, "validate_run_policy", lambda *args, **kwargs:
        pytest.fail("generic runner must reject before policy/data verification"))
    monkeypatch.setattr(runner, "verify_data", lambda *args, **kwargs:
        pytest.fail("generic dataset verification would exceed the finite input contract"))
    with pytest.raises(RuntimeError, match="BOUNDED_HISTORY_CACHE_AUDIT_CLI_ONLY"):
        runner.run_experiment(tmp_path, workflow.EXPERIMENT, workflow.STAGE)
    assert not (tmp_path / "results").exists()


def test_campaign_budget_is_exclusive_before_any_run_work(tmp_path, monkeypatch):
    common = tmp_path / "common-git"
    common.mkdir()
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *args, **kwargs: str(common))
    monkeypatch.setattr(workflow, "git_head", lambda *args: "synthetic-commit")
    config = {"read_plan_sha256": "a" * 64, "compute_plan_sha256": "b" * 64}
    directory, receipt = workflow.reserve(tmp_path, "first-synthetic-run", config)
    assert receipt["spent"] is True and receipt["retry_allowed"] is False
    assert (directory / "reservation.json").is_file()
    with pytest.raises(FileExistsError):
        workflow.reserve(tmp_path, "second-synthetic-run", config)
    assert not (directory / "completed.json").exists()
