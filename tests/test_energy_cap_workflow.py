from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kron_reduce.workflows import kiki_energy_cap_transfer as workflow


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


def _approved_repo(root: Path, monkeypatch, *, status="user_approved") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Codex Test")
    _git(root, "config", "user.email", "codex-test@example.invalid")
    _write(root, ".gitignore", ".project_state/\nresults/\n")
    for relative in workflow.REQUIRED_BINDINGS:
        _write(root, relative, f"bound test fixture: {relative}\n")
    _write(root, workflow.PROTOCOL, "test transfer protocol\n")
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
            "spatial_node_count": 10,
            "spatial_clustering": False,
            "input_contract_id": "measured_history_100ms_zoh_v1",
        }},
    }, ensure_ascii=False, indent=2) + "\n")
    bindings = {relative: _digest(root / relative) for relative in workflow.REQUIRED_BINDINGS}
    protocol_sha = _digest(root / workflow.PROTOCOL)
    review_path = "docs/research/kiki_energy_cap_transfer_review.json"
    plan = {
        "scope": workflow.SCOPE,
        "protocol_path": workflow.PROTOCOL,
        "protocol_sha256": protocol_sha,
        "implementation_bindings": bindings,
        "fixture": "artificial authorization only",
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
    approval = {
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
        "development_evaluations": {workflow.AUTH: approval},
    }, ensure_ascii=False, indent=2) + "\n")
    # Policy validation is covered by its own project-harness tests. This
    # authorization fixture isolates the workflow's registration and binding checks.
    monkeypatch.setattr(workflow, "validate_run_policy", lambda *_args, **_kwargs: None)
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=Codex Test", "-c", "user.email=codex-test@example.invalid",
         "commit", "-q", "-m", "test saved-train authorization")
    return root


def test_authorize_accepts_user_approved_exact_binding_fixture(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo", monkeypatch)
    plan = workflow.authorize(root)
    assert plan["scope"] == workflow.SCOPE


def test_authorize_rejects_consumed_approval(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo", monkeypatch, status="consumed")
    with pytest.raises(RuntimeError, match="AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(root)


def test_authorize_rejects_committed_implementation_binding_tamper(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo", monkeypatch)
    bound = "src/kron_reduce/evaluation/energy_cap_transfer.py"
    _write(root, bound, "tampered fixture source\n")
    _git(root, "add", bound)
    _git(root, "-c", "user.name=Codex Test", "-c", "user.email=codex-test@example.invalid",
         "commit", "-q", "-m", "tamper bound implementation")

    with pytest.raises(RuntimeError, match="BOUND_FILE_WORKTREE_MISMATCH"):
        workflow.authorize(root)


def _dispatch(monkeypatch, kernel_execute):
    monkeypatch.setattr(
        workflow.importlib, "import_module",
        lambda _name: SimpleNamespace(execute=kernel_execute),
    )
    monkeypatch.setattr(workflow, "_verify_loaded_runtime", lambda _root, _plan: {"runtime": "mock"})
    run_ids = itertools.count()

    def dispatch(root, experiment_name, stage, *, experiment):
        assert experiment_name == workflow.EXPERIMENT
        assert stage == workflow.STAGE
        run_id = f"energy-cap-test-{next(run_ids)}"
        result_dir = root / "results" / run_id
        result_dir.mkdir(parents=True, exist_ok=False)
        return workflow.run(root=root, experiment=experiment, stage=stage,
                            result_dir=result_dir, run_id=run_id)

    monkeypatch.setattr(workflow, "_run_experiment", dispatch)


def test_success_spends_one_use_reservation_and_second_dispatch_is_blocked(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo", monkeypatch)
    kernel_calls = []

    def succeed(*, result_dir, **_kwargs):
        kernel_calls.append("execute")
        summary = result_dir / "summary.json"
        summary.write_text('{"scientific_verdict":"NO_GO"}\n', encoding="utf-8")
        (result_dir / "artifact_hashes.json").write_text(
            json.dumps({"summary.json": _digest(summary)}) + "\n", encoding="utf-8",
        )
        return {"scientific_verdict": "NO_GO"}

    _dispatch(monkeypatch, succeed)
    result = workflow.execute_registered(root)
    with pytest.raises(FileExistsError, match=workflow.AUTH):
        workflow.execute_registered(root)

    common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    budget = common / "energy_cap_transfer_budgets" / workflow.AUTH
    reservation = json.loads((budget / "reservation.json").read_text(encoding="utf-8"))
    completed = json.loads((budget / "completed.json").read_text(encoding="utf-8"))
    assert reservation["spent"] is True
    assert completed["scientific_verdict"] == "NO_GO"
    assert result["scientific_verdict"] == "NO_GO"
    assert kernel_calls == ["execute"]


def test_failed_kernel_also_spends_reservation_without_retry(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo", monkeypatch)
    calls = []

    def fail(**_kwargs):
        calls.append("execute")
        raise ValueError("mock kernel failure")

    _dispatch(monkeypatch, fail)
    with pytest.raises(ValueError, match="mock kernel failure"):
        workflow.execute_registered(root)
    with pytest.raises(FileExistsError, match=workflow.AUTH):
        workflow.execute_registered(root)

    common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    budget = common / "energy_cap_transfer_budgets" / workflow.AUTH
    assert json.loads((budget / "reservation.json").read_text(encoding="utf-8"))["spent"] is True
    assert json.loads((budget / "failed.json").read_text(encoding="utf-8"))["retry_allowed"] is False
    assert calls == ["execute"]


def test_direct_entry_rejects_wrong_stage_before_kernel_import(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    imports = []
    monkeypatch.setattr(workflow.importlib, "import_module", lambda name: imports.append(name))

    with pytest.raises(RuntimeError, match="DIRECT_RUN_IDENTITY_MISMATCH"):
        workflow.run(root=root, experiment={}, stage="wrong-stage",
                     result_dir=root, run_id="wrong-run")
    assert imports == []


def test_generic_runner_rejects_saved_train_mode_before_policy_or_source_checks(tmp_path, monkeypatch):
    from project_harness import runner

    touched = []
    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        workflow.EXPERIMENT: {"execution_mode": workflow.MODE},
    }})
    for name in ("validate_run_policy", "load_policy", "_verify_development_text_bindings",
                 "verify_data", "_run_experiment"):
        if hasattr(runner, name):
            monkeypatch.setattr(runner, name, lambda *args, _name=name, **kwargs: touched.append(_name))

    with pytest.raises(RuntimeError, match="BOUNDED_HISTORY_CACHE_AUDIT_CLI_ONLY"):
        runner.run_experiment(tmp_path, workflow.EXPERIMENT, workflow.STAGE)
    assert touched == []
