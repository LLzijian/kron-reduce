import json
import types

import pytest
from filelock import FileLock

from project_harness import runner


def setup_runner(tmp_path, monkeypatch, implementation):
    workflow_dir = tmp_path / "configs/workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "example.json").write_text(json.dumps({"stages": ["scaffold", "experiment"]}))
    monkeypatch.setattr(runner, "load_manifest", lambda _: {"experiments": {
        "sample": {"workflow": "example", "module": "fake"}}})
    monkeypatch.setattr(runner, "next_stage", lambda *_: "experiment")
    monkeypatch.setattr(runner, "git_head", lambda _: "abcdef123456")
    monkeypatch.setattr(runner, "record_stage", lambda *args: None)
    monkeypatch.setattr(runner.importlib, "import_module", lambda _: types.SimpleNamespace(run=implementation))


def test_parallel_runner_is_blocked_before_creating_result(tmp_path, monkeypatch):
    setup_runner(tmp_path, monkeypatch, lambda **_: None)
    locks = tmp_path / ".project_state/locks"
    locks.mkdir(parents=True)
    with FileLock(locks / "example.lock"), pytest.raises(RuntimeError, match="WORKFLOW_BUSY"):
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
