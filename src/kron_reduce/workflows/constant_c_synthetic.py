"""One-use harness for the registered constant-C synthetic study."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy
from project_harness.runner import _run_experiment

AUTH = "constant_c_100ms_synthetic_v1_20261109"
EXPERIMENT = "constant_c_100ms_synthetic_v1"
WORKFLOW = "constant_c_100ms_synthetic_v1"
STAGE = "synthetic_audit"
MODULE = "kron_reduce.workflows.constant_c_synthetic"
DATASET = "synthetic_benchmark_registry_v1"
DATA_POLICY = "analytic_synthetic_only"
MODE = "bounded_synthetic_cli_only"
PLAN = "configs/proposals/constant_c_100ms_synthetic_v1.json"
PROTOCOL = "docs/protocols/constant_c_100ms_synthetic_v1_zh.md"
CONFIG = "configs/experiments/constant_c_100ms_synthetic_v1.toml"

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
    "configs/experiments/constant_c_100ms_synthetic_v1.toml",
    "configs/project_manifest.json",
    "configs/workflows/constant_c_100ms_synthetic_v1.json",
    PROTOCOL,
    "requirements/constraints-py312.txt",
    "src/kron_reduce/evaluation/constant_c_observability.py",
    "src/kron_reduce/evaluation/scoped_audit_journal.py",
    "src/kron_reduce/data/history_contract.py",
    "src/kron_reduce/models/causal_latent.py",
    "src/kron_reduce/models/learned_latent.py",
    "src/kron_reduce/models/nonlinear_latent.py",
    "src/kron_reduce/physics/coordinates.py",
    "src/kron_reduce/workflows/constant_c_synthetic.py",
    "tests/test_constant_c_synthetic_guard.py",
    "project_harness/evidence.py",
    "project_harness/git_state.py",
    "project_harness/loop.py",
    "project_harness/manifest.py",
    "project_harness/policy.py",
    "project_harness/runner.py",
})


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    try:
        value = json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"CONSTANT_C_SYNTHETIC_JSON_INVALID: {relative}") from error
    if not isinstance(value, dict):
        raise TypeError(f"CONSTANT_C_SYNTHETIC_JSON_INVALID: {relative}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_identity(record: dict[str, Any], *, label: str) -> None:
    for key, expected in SCOPE.items():
        actual = record.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(f"CONSTANT_C_SYNTHETIC_{label}_SCOPE_MISMATCH: {key}")


def _registered_experiment(root: Path) -> dict[str, Any]:
    experiment = load_manifest(root).get("experiments", {}).get(EXPERIMENT)
    expected = {
        "module": MODULE,
        "workflow": WORKFLOW,
        "stage": STAGE,
        "config": Path(CONFIG).name,
        "execution_mode": MODE,
        "dataset": DATASET,
        "data_policy": DATA_POLICY,
    }
    if not isinstance(experiment, dict) or any(experiment.get(key) != value for key, value in expected.items()):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_EXPERIMENT_REGISTRATION_MISMATCH")
    if experiment.get("reads_validation_performance") is not False or experiment.get("reads_test_performance") is not False:
        raise RuntimeError("CONSTANT_C_SYNTHETIC_READ_SCOPE_MISMATCH")
    return experiment


def authorize(root: Path) -> dict[str, Any]:
    """Validate committed policy, protocol, plan, review and source bindings."""
    root = Path(root).resolve(strict=True)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_REQUIRES_PYTHON_312")
    assert_clean(root)
    experiment = _registered_experiment(root)
    _ = experiment

    policy_path = root / "configs/research_policy.json"
    head_policy = subprocess.run(
        ["git", "show", "HEAD:configs/research_policy.json"],
        cwd=root, capture_output=True, check=False,
    )
    if (head_policy.returncode != 0
            or hashlib.sha256(head_policy.stdout).hexdigest() != _sha256_file(policy_path)):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_POLICY_NOT_COMMITTED")

    policy = load_policy(root)
    entries = policy.get("bounded_synthetic_evaluations", {})
    authorization = entries.get(AUTH) if isinstance(entries, dict) else None
    if not isinstance(authorization, dict) or authorization.get("status") != "active":
        raise RuntimeError("CONSTANT_C_SYNTHETIC_AUTHORIZATION_NOT_ACTIVE")
    _require_identity(authorization, label="AUTHORIZATION")

    if authorization.get("plan_path") != PLAN or authorization.get("protocol_path") != PROTOCOL:
        raise RuntimeError("CONSTANT_C_SYNTHETIC_BOUND_PATH_MISMATCH")
    plan_sha = authorization.get("plan_sha256")
    review_path = authorization.get("review_path")
    review_sha = authorization.get("review_sha256")
    protocol_sha = authorization.get("protocol_sha256")
    if (not isinstance(review_path, str) or not review_path.startswith("docs/research/")
            or not review_path.endswith(".json")):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_REVIEW_PATH_INVALID")
    bindings = authorization.get("implementation_bindings")
    if (not isinstance(bindings, dict) or not REQUIRED_BINDINGS.issubset(bindings)
            or any(not isinstance(path, str) or not isinstance(digest, str)
                   for path, digest in bindings.items())):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_BINDINGS_INCOMPLETE")
    for relative, digest in (
        (PLAN, plan_sha), (PROTOCOL, protocol_sha), (review_path, review_sha),
    ):
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise TypeError("CONSTANT_C_SYNTHETIC_BOUND_FILE_MISSING")
        verify_bound_file(root, relative, digest)
    for relative, digest in bindings.items():
        verify_bound_file(root, relative, digest)

    plan = _read_json(root, PLAN)
    review = _read_json(root, review_path)
    if (plan.get("protocol_path") != PROTOCOL or plan.get("protocol_sha256") != protocol_sha
            or plan.get("implementation_bindings") != bindings):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_PLAN_BINDING_MISMATCH")
    if (review.get("approved") is not True or review.get("implementation_bindings") != bindings
            or review.get("plan_path") != PLAN or review.get("plan_sha256") != plan_sha
            or review.get("protocol_path") != PROTOCOL or review.get("protocol_sha256") != protocol_sha):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_REVIEW_BINDING_MISMATCH")
    if not isinstance(review.get("scope"), dict) or review["scope"] != SCOPE:
        raise RuntimeError("CONSTANT_C_SYNTHETIC_REVIEW_SCOPE_MISMATCH")
    if not isinstance(plan.get("scope"), dict) or plan["scope"] != SCOPE:
        raise RuntimeError("CONSTANT_C_SYNTHETIC_PLAN_SCOPE_MISMATCH")
    return plan


def _verify_loaded_runtime(root: Path, plan: dict[str, Any]) -> dict[str, str]:
    """Bind loaded Python modules and installed packages to the approved runtime."""
    root = Path(root).resolve(strict=True)
    bindings = plan.get("implementation_bindings")
    if not isinstance(bindings, dict):
        raise TypeError("CONSTANT_C_SYNTHETIC_RUNTIME_BINDINGS_INVALID")

    for relative, digest in bindings.items():
        if not relative.startswith(("src/", "project_harness/")):
            continue
        if not relative.endswith(".py"):
            continue
        bound_path = (root / relative).resolve(strict=True)
        if _sha256_file(bound_path) != digest:
            raise RuntimeError(f"CONSTANT_C_SYNTHETIC_LOADED_SOURCE_HASH_MISMATCH: {relative}")
        module_name = relative.removesuffix(".py").replace("/", ".")
        module_name = module_name.removeprefix("src.")
        if module_name.endswith(".__init__"):
            module_name = module_name.removesuffix(".__init__")
        module = sys.modules.get(module_name)
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError(f"CONSTANT_C_SYNTHETIC_BOUND_MODULE_NOT_LOADED: {relative}")
        try:
            loaded_path = Path(module.__file__).resolve(strict=True)
        except (OSError, TypeError) as error:
            raise RuntimeError(f"CONSTANT_C_SYNTHETIC_LOADED_MODULE_PATH_INVALID: {relative}") from error
        if loaded_path != bound_path:
            raise RuntimeError(f"CONSTANT_C_SYNTHETIC_LOADED_MODULE_PATH_MISMATCH: {relative}")
        if _sha256_file(loaded_path) != digest:
            raise RuntimeError(f"CONSTANT_C_SYNTHETIC_LOADED_MODULE_HASH_MISMATCH: {relative}")

    constraints_relative = "requirements/constraints-py312.txt"
    constraints_path = (root / constraints_relative).resolve(strict=True)
    constraints_digest = bindings.get(constraints_relative)
    if not isinstance(constraints_digest, str) or _sha256_file(constraints_path) != constraints_digest:
        raise RuntimeError("CONSTANT_C_SYNTHETIC_RUNTIME_CONSTRAINTS_NOT_BOUND")
    pins = {}
    for raw_line in constraints_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise RuntimeError("CONSTANT_C_SYNTHETIC_RUNTIME_CONSTRAINT_FORMAT_INVALID")
        name, pinned = (part.strip() for part in line.split("==", maxsplit=1))
        pins[name.lower().replace("_", "-")] = pinned

    actual_versions = {}
    for name in ("numpy", "scipy", "torch", "filelock", "h5py"):
        pinned = pins.get(name)
        if not pinned:
            raise RuntimeError(f"CONSTANT_C_SYNTHETIC_RUNTIME_PIN_MISSING: {name}")
        try:
            installed = version(name)
        except Exception as error:
            raise RuntimeError(f"CONSTANT_C_SYNTHETIC_RUNTIME_PACKAGE_MISSING: {name}") from error
        compatible = installed == pinned or (name == "torch" and installed == f"{pinned}+cpu")
        if not compatible:
            raise RuntimeError(
                f"CONSTANT_C_SYNTHETIC_RUNTIME_VERSION_MISMATCH: {name} {installed} != {pinned}"
            )
        actual_versions[name] = installed
    return actual_versions


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reserve(root: Path, run_id: str, plan: dict[str, Any]) -> tuple[Path, str]:
    common_dir = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    common = Path(common_dir).resolve(strict=True)
    directory = common / "constant_c_synthetic_budgets" / AUTH
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    commit = git_head(root)
    reservation = {
        "authorization": AUTH,
        "run_id": run_id,
        "source_commit": commit,
        "plan_sha256": _sha256_file(root / PLAN),
        "spent": True,
        "retry_allowed": False,
    }
    _write_new_json(directory / "reservation.json", reservation)
    return directory, commit


def run(*, root, experiment, stage, result_dir, run_id):
    """Run only the exact registered stage after spending its shared reservation."""
    root = Path(root).resolve(strict=True)
    if stage != STAGE or experiment != _registered_experiment(root):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_DIRECT_RUN_IDENTITY_MISMATCH")
    plan = authorize(root)
    result_dir = Path(result_dir).resolve(strict=True)
    if result_dir.parent != (root / "results").resolve(strict=True) or result_dir.name != run_id:
        raise RuntimeError("CONSTANT_C_SYNTHETIC_RESULT_DIRECTORY_MISMATCH")
    if any(result_dir.iterdir()):
        raise RuntimeError("CONSTANT_C_SYNTHETIC_RESULT_DIRECTORY_NOT_EMPTY")
    budget_dir, source_commit = _reserve(root, run_id, plan)
    try:
        kernel = importlib.import_module("kron_reduce.evaluation.constant_c_observability")
        _verify_loaded_runtime(root, plan)
        summary = kernel.execute(plan=plan, result_dir=result_dir)
        if not isinstance(summary, dict):
            raise TypeError("CONSTANT_C_SYNTHETIC_SUMMARY_INVALID")
        authorize(root)
        if git_head(root) != source_commit:
            raise RuntimeError("CONSTANT_C_SYNTHETIC_SOURCE_HEAD_CHANGED")
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


def execute_registered(root: Path) -> Path:
    root = Path(root).resolve(strict=True)
    authorize(root)
    experiment = _registered_experiment(root)
    with workflow_lock(root, WORKFLOW):
        return _run_experiment(root, EXPERIMENT, STAGE, experiment=experiment)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the registered one-use constant-C synthetic study")
    parser.add_argument("--root", type=Path, default=project_root())
    print(execute_registered(parser.parse_args(argv).root))


if __name__ == "__main__":
    main()
