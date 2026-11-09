from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kron_reduce.workflows import energy_confirmation_synthetic as workflow


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _approved_repo(root: Path, *, status: str = "active") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Codex Test")
    _git(root, "config", "user.email", "codex-test@example.invalid")
    _write(root, ".gitignore", ".project_state/\nresults/\n")
    for relative in workflow.REQUIRED_BINDINGS:
        _write(root, relative, f"bound test fixture: {relative}\n")
    _write(root, workflow.PROTOCOL, "test protocol\n")
    _write(root, "configs/project_manifest.json", json.dumps({
        "project": "kron-reduce",
        "experiments": {workflow.EXPERIMENT: {
            "module": workflow.MODULE,
            "workflow": workflow.WORKFLOW,
            "stage": workflow.STAGE,
            "config": Path(workflow.CONFIG).name,
            "execution_mode": workflow.MODE,
            "dataset": workflow.DATASET,
            "data_policy": workflow.DATA_POLICY,
            "reads_validation_performance": False,
            "reads_test_performance": False,
        }},
    }, ensure_ascii=False, indent=2) + "\n")
    bindings = {relative: _digest(root / relative) for relative in workflow.REQUIRED_BINDINGS}
    protocol_sha = _digest(root / workflow.PROTOCOL)
    review_path = "docs/research/energy_confirmation_review.json"
    plan = {
        "scope": workflow.SCOPE,
        "protocol_path": workflow.PROTOCOL,
        "protocol_sha256": protocol_sha,
        "implementation_bindings": bindings,
        "cases": ["analytic_case_a"],
    }
    _write(root, workflow.PLAN, json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    plan_sha = _digest(root / workflow.PLAN)
    review = {
        "approved": True,
        "scope": workflow.SCOPE,
        "plan_path": workflow.PLAN,
        "plan_sha256": plan_sha,
        "protocol_path": workflow.PROTOCOL,
        "protocol_sha256": protocol_sha,
        "implementation_bindings": bindings,
    }
    _write(root, review_path, json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    authorization = {
        "status": status,
        **workflow.SCOPE,
        "plan_path": workflow.PLAN,
        "plan_sha256": plan_sha,
        "protocol_path": workflow.PROTOCOL,
        "protocol_sha256": protocol_sha,
        "review_path": review_path,
        "review_sha256": _digest(root / review_path),
        "implementation_bindings": bindings,
    }
    _write(root, "configs/research_policy.json", json.dumps({
        "bounded_synthetic_evaluations": {workflow.AUTH: authorization},
    }, ensure_ascii=False, indent=2) + "\n")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=Codex Test", "-c", "user.email=codex-test@example.invalid",
         "commit", "-q", "-m", "test analytic synthetic authorization")
    return root


def test_authorize_accepts_exact_active_scope(tmp_path):
    root = _approved_repo(tmp_path / "repo")
    assert workflow.authorize(root)["cases"] == ["analytic_case_a"]


@pytest.mark.parametrize("status", ["inactive", "consumed"])
def test_authorize_rejects_non_active_policy(tmp_path, status):
    root = _approved_repo(tmp_path / "repo", status=status)
    with pytest.raises(RuntimeError, match="AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(root)


def test_authorize_rejects_committed_binding_tamper(tmp_path):
    root = _approved_repo(tmp_path / "repo")
    bound = "src/kron_reduce/evaluation/energy_confirmation.py"
    _write(root, bound, "tampered after approval\n")
    _git(root, "add", bound)
    _git(root, "-c", "user.name=Codex Test", "-c", "user.email=codex-test@example.invalid",
         "commit", "-q", "-m", "tamper bound kernel")
    with pytest.raises((RuntimeError, OSError)):
        workflow.authorize(root)


def _dispatch(monkeypatch, kernel_execute):
    monkeypatch.setattr(
        workflow.importlib, "import_module",
        lambda _name: SimpleNamespace(execute=kernel_execute),
    )
    monkeypatch.setattr(workflow, "_verify_loaded_runtime", lambda _root, _plan: {})
    ids = itertools.count()

    def dispatch(root, experiment_name, stage, *, experiment):
        assert experiment_name == workflow.EXPERIMENT
        assert stage == workflow.STAGE
        run_id = f"energy-confirmation-test-{next(ids)}"
        result_dir = root / "results" / run_id
        result_dir.mkdir(parents=True, exist_ok=False)
        return workflow.run(
            root=root, experiment=experiment, stage=stage,
            result_dir=result_dir, run_id=run_id,
        )

    monkeypatch.setattr(workflow, "_run_experiment", dispatch)


def test_one_use_success_writes_completed_and_second_run_is_rejected(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo")
    calls = []

    def succeed(*, result_dir, **_kwargs):
        calls.append("kernel")
        payload = result_dir / "summary.json"
        payload.write_text('{"status":"PASS"}\n', encoding="utf-8")
        (result_dir / "artifact_hashes.json").write_text(
            json.dumps({"summary.json": _digest(payload)}) + "\n", encoding="utf-8",
        )
        return {"status": "PASS"}

    _dispatch(monkeypatch, succeed)
    workflow.execute_registered(root)
    with pytest.raises(FileExistsError, match=workflow.AUTH):
        workflow.execute_registered(root)

    common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    budget = common / "energy_confirmation_budgets" / workflow.AUTH
    assert json.loads((budget / "reservation.json").read_text(encoding="utf-8"))["spent"] is True
    assert json.loads((budget / "completed.json").read_text(encoding="utf-8"))["status"] == "complete"
    assert calls == ["kernel"]


def test_failed_kernel_spends_reservation_and_cannot_be_retried(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo")
    calls = []

    def fail(**_kwargs):
        calls.append("kernel")
        raise ValueError("injected failure")

    _dispatch(monkeypatch, fail)
    with pytest.raises(ValueError, match="injected failure"):
        workflow.execute_registered(root)
    with pytest.raises(FileExistsError, match=workflow.AUTH):
        workflow.execute_registered(root)

    common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    budget = common / "energy_confirmation_budgets" / workflow.AUTH
    failure = json.loads((budget / "failed.json").read_text(encoding="utf-8"))
    assert failure["retry_allowed"] is False
    assert calls == ["kernel"]


def test_generic_runner_rejects_before_policy_or_data_access(tmp_path, monkeypatch):
    from project_harness import runner

    touched = []
    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        workflow.EXPERIMENT: {"execution_mode": workflow.MODE},
    }})
    for name in ("validate_run_policy", "load_policy", "_verify_development_text_bindings",
                 "verify_data", "_run_experiment"):
        if hasattr(runner, name):
            monkeypatch.setattr(runner, name, lambda *args, _name=name, **kwargs: touched.append(_name))
    with pytest.raises(RuntimeError, match="BOUNDED_SYNTHETIC_CLI_ONLY"):
        runner.run_experiment(tmp_path, workflow.EXPERIMENT, workflow.STAGE)
    assert touched == []


def test_direct_run_rejects_wrong_stage_before_kernel_import(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo")
    imports = []
    monkeypatch.setattr(workflow.importlib, "import_module", lambda name: imports.append(name))
    with pytest.raises(RuntimeError, match="DIRECT_RUN_IDENTITY_MISMATCH"):
        workflow.run(
            root=root, experiment=workflow._registered_experiment(root), stage="wrong-stage",
            result_dir=root, run_id="wrong-run",
        )
    assert imports == []
