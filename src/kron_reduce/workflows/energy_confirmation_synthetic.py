"""One-use entry point for the analytic synthetic energy confirmation."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from kron_reduce.workflows.constant_c_synthetic import (
    _read_json,
    _verify_loaded_runtime,
    _write_new_json,
)
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy
from project_harness.runner import _run_experiment

AUTH = "constant_c_energy_confirmation_v1_20261109"
EXPERIMENT = WORKFLOW = "constant_c_energy_confirmation_v1"
STAGE = "synthetic_confirmation"
MODULE = "kron_reduce.workflows.energy_confirmation_synthetic"
DATASET = "synthetic_benchmark_registry_v1"
DATA_POLICY = "analytic_synthetic_only"
MODE = "bounded_synthetic_cli_only"
PLAN = "configs/proposals/constant_c_energy_confirmation_v1.json"
PROTOCOL = "docs/protocols/constant_c_energy_confirmation_v1_zh.md"
CONFIG = "configs/experiments/constant_c_energy_confirmation_v1.toml"
SCOPE = {
    "authorization": AUTH,
    "experiment": EXPERIMENT,
    "workflow": WORKFLOW,
    "stage": STAGE,
    "module": MODULE,
    "dataset": DATASET,
    "data_policy": DATA_POLICY,
    "max_executions": 1,
    "retry_allowed": False,
    "real_data_reads": 0,
    "fit_attempts": 0,
    "optimizer_updates": 0,
}
REQUIRED_BINDINGS = frozenset({
    CONFIG,
    PROTOCOL,
    "configs/project_manifest.json",
    "configs/workflows/constant_c_energy_confirmation_v1.json",
    "requirements/constraints-py312.txt",
    "src/kron_reduce/data/energy_saved_inputs.py",
    "src/kron_reduce/data/history_contract.py",
    "src/kron_reduce/evaluation/energy_confirmation.py",
    "src/kron_reduce/evaluation/energy_constrained_initializer.py",
    "src/kron_reduce/evaluation/energy_initializer_audit.py",
    "src/kron_reduce/evaluation/constant_c_observability.py",
    "src/kron_reduce/evaluation/scoped_audit_journal.py",
    "src/kron_reduce/models/causal_latent.py",
    "src/kron_reduce/models/learned_latent.py",
    "src/kron_reduce/models/nonlinear_latent.py",
    "src/kron_reduce/physics/coordinates.py",
    "src/kron_reduce/workflows/constant_c_synthetic.py",
    "src/kron_reduce/workflows/energy_confirmation_synthetic.py",
    "tests/test_energy_confirmation.py",
    "tests/test_energy_confirmation_workflow.py",
    "tests/test_energy_initializer_audit.py",
    "tests/test_energy_constrained_initializer.py",
    "project_harness/evidence.py",
    "project_harness/git_state.py",
    "project_harness/loop.py",
    "project_harness/manifest.py",
    "project_harness/policy.py",
    "project_harness/runner.py",
})


def _registered_experiment(root: Path) -> dict:
    entry = load_manifest(root).get("experiments", {}).get(EXPERIMENT)
    expected = {
        "module": MODULE,
        "workflow": WORKFLOW,
        "stage": STAGE,
        "config": Path(CONFIG).name,
        "execution_mode": MODE,
        "dataset": DATASET,
        "data_policy": DATA_POLICY,
        "reads_validation_performance": False,
        "reads_test_performance": False,
    }
    if not isinstance(entry, dict) or any(
        type(entry.get(key)) is not type(value) or entry[key] != value
        for key, value in expected.items()
    ):
        raise RuntimeError("ENERGY_CONFIRMATION_EXPERIMENT_REGISTRATION_MISMATCH")
    return entry


def authorize(root: Path) -> dict:
    """Require the committed, exact one-use synthetic authorization."""
    root = Path(root).resolve(strict=True)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("ENERGY_CONFIRMATION_REQUIRES_PYTHON_312")
    assert_clean(root)
    _registered_experiment(root)
    policy_digest = sha256_file(root / "configs/research_policy.json")
    verify_bound_file(root, "configs/research_policy.json", policy_digest)
    authorization = load_policy(root).get("bounded_synthetic_evaluations", {}).get(AUTH)
    if not isinstance(authorization, dict) or authorization.get("status") != "active":
        raise RuntimeError("ENERGY_CONFIRMATION_AUTHORIZATION_NOT_ACTIVE")
    for key, value in SCOPE.items():
        if type(authorization.get(key)) is not type(value) or authorization[key] != value:
            raise RuntimeError(f"ENERGY_CONFIRMATION_SCOPE_MISMATCH: {key}")
    if authorization.get("plan_path") != PLAN or authorization.get("protocol_path") != PROTOCOL:
        raise RuntimeError("ENERGY_CONFIRMATION_BOUND_PATH_MISMATCH")
    review_path = authorization.get("review_path")
    if (not isinstance(review_path, str) or not review_path.startswith("docs/research/")
            or not review_path.endswith(".json")):
        raise RuntimeError("ENERGY_CONFIRMATION_REVIEW_PATH_INVALID")
    bindings = authorization.get("implementation_bindings")
    if (not isinstance(bindings, dict) or not REQUIRED_BINDINGS.issubset(bindings)
            or any(not isinstance(path, str) or not isinstance(digest, str)
                   for path, digest in bindings.items())):
        raise RuntimeError("ENERGY_CONFIRMATION_BINDINGS_INCOMPLETE")
    for relative, digest in [
        (PLAN, authorization.get("plan_sha256")),
        (PROTOCOL, authorization.get("protocol_sha256")),
        (review_path, authorization.get("review_sha256")),
        *bindings.items(),
    ]:
        verify_bound_file(root, relative, digest)

    plan, review = (_read_json(root, PLAN, authorization["plan_sha256"]),
                    _read_json(root, review_path, authorization["review_sha256"]))
    expected = {
        "scope": SCOPE,
        "protocol_path": PROTOCOL,
        "protocol_sha256": authorization["protocol_sha256"],
        "implementation_bindings": bindings,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise RuntimeError("ENERGY_CONFIRMATION_PLAN_BINDING_MISMATCH")
    if (review.get("approved") is not True
            or review.get("plan_path") != PLAN
            or review.get("plan_sha256") != authorization["plan_sha256"]
            or any(review.get(key) != value for key, value in expected.items())):
        raise RuntimeError("ENERGY_CONFIRMATION_REVIEW_BINDING_MISMATCH")
    return plan


def _reserve(root: Path, run_id: str) -> tuple[Path, str]:
    common = Path(subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()).resolve(strict=True)
    directory = common / "energy_confirmation_budgets" / AUTH
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    commit = git_head(root)
    _write_new_json(directory / "reservation.json", {
        "authorization": AUTH,
        "run_id": run_id,
        "source_commit": commit,
        "plan_sha256": sha256_file(root / PLAN),
        "spent": True,
        "retry_allowed": False,
    })
    return directory, commit


def run(*, root, experiment, stage, result_dir, run_id):
    root = Path(root).resolve(strict=True)
    if stage != STAGE or experiment != _registered_experiment(root):
        raise RuntimeError("ENERGY_CONFIRMATION_DIRECT_RUN_IDENTITY_MISMATCH")
    plan = authorize(root)
    result_dir = Path(result_dir).resolve(strict=True)
    if result_dir.parent != (root / "results").resolve(strict=True) or result_dir.name != run_id:
        raise RuntimeError("ENERGY_CONFIRMATION_RESULT_DIRECTORY_MISMATCH")
    if any(result_dir.iterdir()):
        raise RuntimeError("ENERGY_CONFIRMATION_RESULT_DIRECTORY_NOT_EMPTY")

    budget_dir, source_commit = _reserve(root, run_id)
    try:
        kernel = importlib.import_module("kron_reduce.evaluation.energy_confirmation")
        _verify_loaded_runtime(root, plan)
        summary = kernel.execute(plan=plan, root=root, result_dir=result_dir)
        if not isinstance(summary, dict):
            raise TypeError("ENERGY_CONFIRMATION_SUMMARY_INVALID")
        authorize(root)
        if git_head(root) != source_commit:
            raise RuntimeError("ENERGY_CONFIRMATION_SOURCE_HEAD_CHANGED")
        verify_artifact_directory(result_dir, strict=True)
        _write_new_json(budget_dir / "completed.json", {
            "run_id": run_id,
            "authorization_spent": True,
            "status": "complete",
            "source_commit": source_commit,
            "artifact_manifest_sha256": sha256_file(result_dir / "artifact_hashes.json"),
        })
        return summary
    except BaseException as error:
        try:
            _write_new_json(budget_dir / "failed.json", {
                "run_id": run_id,
                "authorization_spent": True,
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_allowed": False,
            })
        except OSError as persist_error:
            error.add_note(f"Could not persist failure marker: {persist_error}")
        raise


def execute_registered(root: Path):
    root = Path(root).resolve(strict=True)
    authorize(root)
    experiment = _registered_experiment(root)
    with workflow_lock(root, WORKFLOW):
        return _run_experiment(root, EXPERIMENT, STAGE, experiment=experiment)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the one-use analytic synthetic confirmation")
    parser.add_argument("--root", type=Path, default=project_root())
    print(execute_registered(parser.parse_args(argv).root))


if __name__ == "__main__":
    main()
