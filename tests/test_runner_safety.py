import json
import types

import pytest
from filelock import FileLock

from project_harness import runner
from project_harness.loop import approve, record_stage, run_one


def setup_runner(tmp_path, monkeypatch, implementation, *, approvals=(), registered_stage="experiment"):
    workflow_dir = tmp_path / "configs/workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "example.json").write_text(json.dumps({
        "workflow": "example", "initial_stage": "scaffold",
        "stages": ["scaffold", "experiment"], "approval_required": list(approvals),
    }), encoding="utf-8")
    monkeypatch.setattr(runner, "load_manifest", lambda _: {"experiments": {
        "sample": {"workflow": "example", "module": "fake", "dataset": "kiki",
                   "stage": registered_stage}}})
    monkeypatch.setattr(runner, "git_head", lambda _: "abcdef123456")
    monkeypatch.setattr(runner.importlib, "import_module", lambda _: types.SimpleNamespace(run=implementation))


def test_runner_and_manual_mutations_share_workflow_lock(tmp_path, monkeypatch):
    setup_runner(tmp_path, monkeypatch, lambda **_: None, approvals=["experiment"])
    locks = tmp_path / ".project_state/locks"
    locks.mkdir(parents=True)
    with FileLock(locks / "example.lock"):
        with pytest.raises(RuntimeError, match="WORKFLOW_BUSY"):
            runner.run_experiment(tmp_path, "sample", "experiment")
        with pytest.raises(RuntimeError, match="WORKFLOW_BUSY"):
            approve(tmp_path, "experiment", "example")
        with pytest.raises(RuntimeError, match="WORKFLOW_BUSY"):
            record_stage(tmp_path, "experiment", {"status": "PASS"}, "example")
        with pytest.raises(RuntimeError, match="WORKFLOW_BUSY"):
            run_one(tmp_path, "example")
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / ".project_state/example_state.json").exists()


@pytest.mark.parametrize(
    ("registered_stage", "error"),
    [("experiment", "requires explicit approval"), ("other", "registered for stage 'other'")],
)
def test_runner_checks_stage_and_approval_before_data_access(tmp_path, monkeypatch, registered_stage, error):
    setup_runner(tmp_path, monkeypatch, lambda **_: pytest.fail("unexpected dispatch"),
                 approvals=["experiment"], registered_stage=registered_stage)
    monkeypatch.setattr(runner, "load_policy", lambda _: {"primary_real_dataset": "kiki"})
    monkeypatch.setattr(runner, "verify_data", lambda *_, **__: pytest.fail("unexpected data access"))
    state_dir = tmp_path / ".project_state"
    state_dir.mkdir()
    (state_dir / "example_state.json").write_text(json.dumps({
        "workflow": "example", "current_stage": "scaffold",
        "approvals": {"experiment": {"actor": "explicit_operator_action"}}, "history": [],
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match=error):
        runner.run_experiment(tmp_path, "sample", "experiment")
    assert not (tmp_path / "results").exists()


def test_failed_attempt_keeps_traceback_and_releases_lock(tmp_path, monkeypatch):
    def fail(**_):
        raise ValueError("intentional failure")
    setup_runner(tmp_path, monkeypatch, fail)
    for _ in range(2):
        with pytest.raises(ValueError, match="intentional failure"):
            runner.run_experiment(tmp_path, "sample", "experiment")
    failures = list((tmp_path / "results").glob("*/RUN_FAILED.json"))
    assert len(failures) == 2
    for path in failures:
        evidence = json.loads(path.read_text())
        assert evidence["error_type"] == "ValueError"
        assert "intentional failure" in evidence["traceback"]
        assert evidence["automatic_retry"] is False
