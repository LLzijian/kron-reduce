from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kron_reduce.workflows import constant_c_synthetic as workflow

REVIEW_PATH = "docs/research/constant_c_100ms_synthetic_review_20261109.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, relative: str, value: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _approved_repo(root: Path, *, status: str = "active", missing_binding: bool = False,
                    commit_changed_binding: bool = False,
                    review_binding_mismatch: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Codex Test")
    _git(root, "config", "user.email", "codex-test@example.invalid")
    _write(root, ".gitignore", ".project_state/\nresults/\n")

    bindings = {relative: "bound test fixture\n" for relative in workflow.REQUIRED_BINDINGS}
    if missing_binding:
        bindings.pop("src/kron_reduce/models/causal_latent.py")
    for relative, contents in bindings.items():
        _write(root, relative, contents)
    _write(root, workflow.PROTOCOL, "test protocol\n")
    experiment = {
        "module": workflow.MODULE,
        "workflow": workflow.WORKFLOW,
        "stage": workflow.STAGE,
        "config": Path(workflow.CONFIG).name,
        "execution_mode": workflow.MODE,
        "dataset": workflow.DATASET,
        "data_policy": workflow.DATA_POLICY,
        "reads_validation_performance": False,
        "reads_test_performance": False,
    }
    _write(root, "configs/project_manifest.json", json.dumps({"project": "kron-reduce", "experiments": {
        workflow.EXPERIMENT: experiment,
    }}, indent=2) + "\n")

    implementation_bindings = {
        relative: _digest(root / relative) for relative in bindings
    }
    if commit_changed_binding:
        _write(root, "src/kron_reduce/workflows/constant_c_synthetic.py", "committed changed binding\n")
    plan = {
        "scope": workflow.SCOPE,
        "protocol_path": workflow.PROTOCOL,
        "protocol_sha256": _digest(root / workflow.PROTOCOL),
        "implementation_bindings": implementation_bindings,
    }
    plan_text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    _write(root, workflow.PLAN, plan_text)
    plan_sha = _digest(root / workflow.PLAN)
    review = {
        "approved": True,
        "scope": workflow.SCOPE,
        "plan_path": workflow.PLAN,
        "plan_sha256": plan_sha,
        "protocol_path": workflow.PROTOCOL,
        "protocol_sha256": plan["protocol_sha256"],
        "implementation_bindings": {} if review_binding_mismatch else implementation_bindings,
    }
    _write(root, REVIEW_PATH, json.dumps(review, ensure_ascii=False, indent=2) + "\n")

    authorization = {
        "status": status,
        **workflow.SCOPE,
        "plan_path": workflow.PLAN,
        "plan_sha256": plan_sha,
        "review_path": REVIEW_PATH,
        "review_sha256": _digest(root / REVIEW_PATH),
        "protocol_path": workflow.PROTOCOL,
        "protocol_sha256": plan["protocol_sha256"],
        "implementation_bindings": implementation_bindings,
    }
    _write(root, "configs/research_policy.json", json.dumps({
        "bounded_synthetic_evaluations": {workflow.AUTH: authorization},
    }, indent=2) + "\n")

    _git(root, "add", "-A")
    _git(root, "-c", "user.name=Codex Test", "-c", "user.email=codex-test@example.invalid",
         "commit", "-q", "-m", "test authorization")
    return root


@pytest.mark.parametrize("case", ["inactive", "missing_binding", "changed_binding", "dirty"])
def test_authorize_fails_closed_before_dispatch(tmp_path, monkeypatch, case):
    root = _approved_repo(
        tmp_path / "repo",
        status="consumed" if case == "inactive" else "active",
        missing_binding=case == "missing_binding",
        commit_changed_binding=case == "changed_binding",
    )
    if case == "dirty":
        _write(root, "untracked.txt", "dirty\n")
    called = []
    monkeypatch.setattr(workflow.importlib, "import_module", lambda *_: called.append(True))
    with pytest.raises(RuntimeError):
        workflow.authorize(root)
    assert called == []


def test_authorize_rejects_non_312_before_checkout_access(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.sys, "version_info", (3, 11, 0))
    with pytest.raises(RuntimeError, match="REQUIRES_PYTHON_312"):
        workflow.authorize(tmp_path)


def test_policy_plan_and_review_bindings_must_match(tmp_path):
    root = _approved_repo(tmp_path / "repo", review_binding_mismatch=True)
    with pytest.raises(RuntimeError, match="REVIEW_BINDING_MISMATCH"):
        workflow.authorize(root)


def _dispatch_through_registered_runner(monkeypatch, *, kernel_execute):
    monkeypatch.setattr(workflow.importlib, "import_module", lambda _name: SimpleNamespace(execute=kernel_execute))
    monkeypatch.setattr(workflow, "_verify_loaded_runtime", lambda _root, _plan: {})
    run_ids = itertools.count()

    def dispatch(root, experiment_name, stage, *, experiment):
        assert experiment_name == workflow.EXPERIMENT
        assert stage == workflow.STAGE
        run_id = f"test-run-{next(run_ids)}"
        result_dir = root / "results" / run_id
        result_dir.mkdir(parents=True, exist_ok=False)
        return workflow.run(root=root, experiment=experiment, stage=stage,
                            result_dir=result_dir, run_id=run_id)

    monkeypatch.setattr(workflow, "_run_experiment", dispatch)


def test_shared_reservation_allows_only_one_run(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo")
    calls = []

    def succeed(*, result_dir, **_kwargs):
        calls.append("called")
        payload = result_dir / "result.json"
        payload.write_text("{}\n", encoding="utf-8")
        (result_dir / "artifact_hashes.json").write_text(
            json.dumps({"result.json": _digest(payload)}) + "\n", encoding="utf-8"
        )
        return {"status": "PASS"}

    _dispatch_through_registered_runner(monkeypatch, kernel_execute=succeed)

    workflow.execute_registered(root)
    with pytest.raises(FileExistsError, match=workflow.AUTH):
        workflow.execute_registered(root)
    assert calls == ["called"]
    common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    budget = common / "constant_c_synthetic_budgets" / workflow.AUTH
    assert (budget / "reservation.json").is_file()
    completed = json.loads((budget / "completed.json").read_text(encoding="utf-8"))
    assert completed["artifact_manifest_sha256"] == _digest(root / "results/test-run-0/artifact_hashes.json")


def test_failed_run_spends_reservation_without_retry(tmp_path, monkeypatch):
    root = _approved_repo(tmp_path / "repo")
    calls = []

    def fail(**_kwargs):
        calls.append("called")
        raise ValueError("injected kernel failure")

    _dispatch_through_registered_runner(monkeypatch, kernel_execute=fail)
    with pytest.raises(ValueError, match="injected kernel failure"):
        workflow.execute_registered(root)
    with pytest.raises(FileExistsError, match=workflow.AUTH):
        workflow.execute_registered(root)
    common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    budget = common / "constant_c_synthetic_budgets" / workflow.AUTH
    assert calls == ["called"]
    assert (budget / "reservation.json").is_file()
    assert json.loads((budget / "failed.json").read_text(encoding="utf-8"))["retry_allowed"] is False


def test_generic_runner_rejects_dedicated_execution_mode_before_dispatch(tmp_path, monkeypatch):
    from project_harness import runner

    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        workflow.EXPERIMENT: {"execution_mode": workflow.MODE},
    }})
    with pytest.raises(RuntimeError, match="BOUNDED_SYNTHETIC_CLI_ONLY"):
        runner.run_experiment(tmp_path, workflow.EXPERIMENT, workflow.STAGE)


def _runtime_fixture(root: Path, monkeypatch):
    constraints = """numpy==2.5.2
scipy==1.18.0
torch==2.13.0
filelock==3.32.2
h5py==3.16.0
"""
    module_files = {
        "src/kron_reduce/evaluation/constant_c_observability.py":
            "kron_reduce.evaluation.constant_c_observability",
        "project_harness/evidence.py": "project_harness.evidence",
    }
    bindings = {}
    for relative in module_files:
        _write(root, relative, f"bound source for {relative}\n")
        bindings[relative] = _digest(root / relative)
    _write(root, "requirements/constraints-py312.txt", constraints)
    bindings["requirements/constraints-py312.txt"] = _digest(
        root / "requirements/constraints-py312.txt"
    )
    for relative, module_name in module_files.items():
        monkeypatch.setitem(
            workflow.sys.modules,
            module_name,
            SimpleNamespace(__file__=str(root / relative)),
        )
    versions = {
        "numpy": "2.5.2",
        "scipy": "1.18.0",
        "torch": "2.13.0+cpu",
        "filelock": "3.32.2",
        "h5py": "3.16.0",
    }
    monkeypatch.setattr(workflow, "version", lambda name: versions[name])
    return {"implementation_bindings": bindings}, module_files, versions


def test_loaded_runtime_checks_bound_module_paths_hashes_and_versions(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    plan, _module_files, versions = _runtime_fixture(root, monkeypatch)
    assert workflow._verify_loaded_runtime(root, plan) == versions


def test_loaded_runtime_rejects_module_path_offset(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    plan, module_files, _versions = _runtime_fixture(root, monkeypatch)
    relative, module_name = next(iter(module_files.items()))
    outside = tmp_path / "same_name.py"
    outside.write_text((root / relative).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setitem(workflow.sys.modules, module_name, SimpleNamespace(__file__=str(outside)))
    with pytest.raises(RuntimeError, match="LOADED_MODULE_PATH_MISMATCH"):
        workflow._verify_loaded_runtime(root, plan)


def test_loaded_runtime_rejects_dependency_version_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    plan, _module_files, _versions = _runtime_fixture(root, monkeypatch)
    monkeypatch.setattr(workflow, "version", lambda name: "2.5.1" if name == "numpy" else {
        "scipy": "1.18.0",
        "torch": "2.13.0+cpu",
        "filelock": "3.32.2",
        "h5py": "3.16.0",
    }[name])
    with pytest.raises(RuntimeError, match="RUNTIME_VERSION_MISMATCH: numpy"):
        workflow._verify_loaded_runtime(root, plan)
