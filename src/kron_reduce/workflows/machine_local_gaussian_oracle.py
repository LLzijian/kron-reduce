"""One-use entry point for the known-distribution Gaussian information diagnosis."""
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
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy
from project_harness.runner import _run_experiment

WORKFLOW = "machine_local_gaussian_oracle_v1"
MODULE = "kron_reduce.workflows.machine_local_gaussian_oracle"
PLAN = "configs/proposals/machine_local_gaussian_oracle_v1.json"
PROTOCOL = "docs/protocols/machine_local_gaussian_oracle_v1_zh.md"
REVIEW = "docs/research/machine_local_gaussian_oracle_review_20261112.json"
STAGES = {"fixed": ("machine_local_gaussian_oracle_v1", 0)}
REQUIRED_BINDINGS = frozenset({
    "src/kron_reduce/models/joint_history_initializer.py",
    "src/kron_reduce/models/machine_local.py",
    "src/kron_reduce/evaluation/machine_local_synthetic.py",
    "src/kron_reduce/evaluation/machine_local_initializer_diagnosis.py",
    "src/kron_reduce/evaluation/machine_local_gaussian_oracle.py",
    "src/kron_reduce/evaluation/machine_local_observer.py",
    "src/kron_reduce/workflows/machine_local_gaussian_oracle.py",
    "configs/project_manifest.json", "project_harness/policy.py",
    "tests/test_joint_history_initializer.py", "tests/test_gaussian_oracle_workflow.py",
    "tests/test_gaussian_state_posterior.py", "tests/test_gaussian_oracle_risks.py",
    "src/kron_reduce/evaluation/gaussian_state_posterior.py",
    PROTOCOL, "docs/research/machine_local_gaussian_oracle_plan_20261112_zh.md",
})


def scope(stage):
    name, fits = STAGES[stage]
    return {"authorization": name+"_20261112", "experiment": name, "workflow": WORKFLOW,
            "stage": stage, "module": MODULE, "dataset": "synthetic_benchmark_registry_v1",
            "data_policy": "analytic_synthetic_only", "max_executions": 1, "retry_allowed": False,
            "real_data_reads": 0, "fit_attempts": fits, "optimizer_updates": 0}


def registered(root, stage):
    name, _ = STAGES[stage]
    entry = load_manifest(root).get("experiments", {}).get(name)
    expected = {"module": MODULE, "workflow": WORKFLOW, "stage": stage, "config": name+".toml",
                "execution_mode": "bounded_synthetic_cli_only", "dataset": "synthetic_benchmark_registry_v1",
                "data_policy": "analytic_synthetic_only", "reads_validation_performance": False,
                "reads_test_performance": False}
    if not isinstance(entry, dict) or any(type(entry.get(k)) is not type(v) or entry[k] != v for k, v in expected.items()):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_REGISTRATION_MISMATCH")
    return entry


def authorize(root, stage):
    if stage not in STAGES or sys.version_info[:2] != (3, 12):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_STAGE_OR_RUNTIME_INVALID")
    root = Path(root).resolve(strict=True)
    assert_clean(root)
    registered(root, stage)
    state = load_state(root, WORKFLOW)
    if not approval_recorded(state, stage):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_STAGE_APPROVAL_REQUIRED")
    expected = scope(stage)
    policy_digest = sha256_file(root/"configs/research_policy.json")
    verify_bound_file(root, "configs/research_policy.json", policy_digest)
    auth = load_policy(root).get("bounded_synthetic_evaluations", {}).get(expected["authorization"])
    if not isinstance(auth, dict) or auth.get("status") != "active":
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_AUTHORIZATION_NOT_ACTIVE")
    if any(type(auth.get(k)) is not type(v) or auth[k] != v for k, v in expected.items()):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_SCOPE_MISMATCH")
    if (auth.get("plan_path"), auth.get("protocol_path"), auth.get("review_path")) != (PLAN, PROTOCOL, REVIEW):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_AUTHORITY_PATH_MISMATCH")
    bindings = auth.get("implementation_bindings")
    if not isinstance(bindings, dict) or not REQUIRED_BINDINGS.issubset(bindings):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_BINDINGS_INCOMPLETE")
    for path, digest in [(PLAN, auth.get("plan_sha256")), (PROTOCOL, auth.get("protocol_sha256")),
                         (REVIEW, auth.get("review_sha256")), *bindings.items()]:
        verify_bound_file(root, path, digest)
    plan, review = _read_json(root, PLAN, auth["plan_sha256"]), _read_json(root, REVIEW, auth["review_sha256"])
    if (plan.get("implementation_bindings") != bindings or plan.get("protocol_sha256") != auth["protocol_sha256"]
            or plan.get("protocol_path") != PROTOCOL or plan["stages"][stage].get("scope") != expected
            or review.get("approved") is not True or review.get("plan_path") != PLAN
            or review.get("plan_sha256") != auth["plan_sha256"] or review.get("implementation_bindings") != bindings):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_REVIEW_MISMATCH")
    return plan


def run(*, root, experiment, stage, result_dir, run_id):
    root, result_dir = Path(root).resolve(strict=True), Path(result_dir).resolve(strict=True)
    if stage not in STAGES or experiment != registered(root, stage):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_RUN_IDENTITY_MISMATCH")
    plan = authorize(root, stage)
    if result_dir.parent != (root/"results").resolve(strict=True) or result_dir.name != run_id or any(result_dir.iterdir()):
        raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_RESULT_DIRECTORY_INVALID")
    common = Path(subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root, capture_output=True, text=True, check=True).stdout.strip()).resolve(strict=True)
    reservation = common/"machine_local_gaussian_oracle_budgets"/scope(stage)["authorization"]
    reservation.parent.mkdir(parents=True, exist_ok=True)
    reservation.mkdir(exist_ok=False)
    commit = git_head(root)
    _write_new_json(reservation/"reservation.json", {"run_id": run_id, "source_commit": commit,
        "plan_sha256": sha256_file(root/PLAN), "spent": True, "retry_allowed": False,
        "operation_limits": plan["stages"][stage]["operation_limits"]})
    try:
        kernel = importlib.import_module("kron_reduce.evaluation.machine_local_gaussian_oracle")
        _verify_loaded_runtime(root, plan)
        summary = kernel.execute(root=root, result_dir=result_dir, stage=stage, plan=plan)
        authorize(root, stage)
        if git_head(root) != commit:
            raise RuntimeError("MACHINE_GAUSSIAN_ORACLE_SOURCE_CHANGED")
        verify_artifact_directory(result_dir, strict=True)
        _write_new_json(reservation/"completed.json", {"run_id": run_id, "spent": True,
            "source_commit": commit, "scientific_pass": summary["scientific_pass"],
            "artifact_manifest_sha256": sha256_file(result_dir/"artifact_hashes.json")})
        return summary
    except BaseException as error:
        _write_new_json(reservation/"failed.json", {"run_id": run_id, "spent": True,
            "error_type": type(error).__name__, "message": str(error), "retry_allowed": False})
        raise


def execute_registered(root, stage):
    root = Path(root).resolve(strict=True)
    with workflow_lock(root, WORKFLOW):
        authorize(root, stage)
        return _run_experiment(root, STAGES[stage][0], stage, experiment=registered(root, stage))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args(argv)
    print(execute_registered(args.root, args.stage))


if __name__ == "__main__":
    main()
