"""Registered one-use run for joint equation recovery on synthetic truth."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy
from project_harness.runner import _run_experiment

NAME = "machine_local_equation_v1"
AUTH = NAME+"_20261114"
STAGE = "synthetic"
MODULE = "kron_reduce.workflows.machine_local_equation"
PLAN = f"configs/proposals/{NAME}.json"
REVIEW = "docs/research/machine_local_equation_review_20261114.json"


def authorize(root):
    assert_clean(root)
    entry = load_manifest(root)["experiments"][NAME]
    expected = {"module": MODULE, "workflow": NAME, "stage": STAGE, "dataset": "synthetic_benchmark_registry_v1",
                "execution_mode": "bounded_synthetic_cli_only", "data_policy": "analytic_synthetic_only"}
    if (sys.version_info[:2] != (3, 12) or any(entry.get(k) != v for k, v in expected.items())
            or not approval_recorded(load_state(root, NAME), STAGE)):
        raise RuntimeError("synthetic runtime, registration or explicit loop approval differs")
    auth = load_policy(root)["bounded_synthetic_evaluations"][AUTH]
    scope = {"status": "active", "experiment": NAME, "workflow": NAME, "stage": STAGE,
             "real_data_reads": 0, "max_executions": 1, "retry_allowed": False,
             "fit_attempts": 15, "optimizer_updates": 2160, "plan_path": PLAN, "review_path": REVIEW}
    if any(type(auth.get(key)) is not type(value) or auth[key] != value for key, value in scope.items()):
        raise RuntimeError("active joint equation scope differs")
    verify_bound_file(root, PLAN, auth["plan_sha256"])
    verify_bound_file(root, REVIEW, auth["review_sha256"])
    plan, review = (json.loads((root/path).read_text(encoding="utf-8")) for path in (PLAN, REVIEW))
    bindings = plan["implementation_bindings"]
    if (plan["authorization"] != AUTH or plan["dataset"] != expected["dataset"]
            or review.get("approved") is not True or review.get("plan_path") != PLAN
            or review.get("plan_sha256") != auth["plan_sha256"] or review.get("implementation_bindings") != bindings
            or auth["operation_limits"] != plan["operation_limits"]
            or plan["operation_limits"]["fits"] != scope["fit_attempts"]
            or plan["operation_limits"]["optimizer_updates"] != scope["optimizer_updates"]):
        raise RuntimeError("exact reviewed joint equation plan required")
    for path, digest in bindings.items():
        verify_bound_file(root, path, digest)
    pins = dict(line.strip().split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#"))
    for package in ("numpy", "torch", "scipy"):
        expected_version = pins[package] + ("+cu130" if package == "torch" else "")
        if version(package) != expected_version:
            raise RuntimeError(f"unregistered runtime package: {package}")
    return entry, plan


def run(*, root, experiment, stage, result_dir, run_id):
    entry, plan = authorize(root)
    if experiment != entry or stage != STAGE or result_dir.parent != root/"results" or result_dir.name != run_id or any(result_dir.iterdir()):
        raise RuntimeError("joint equation result directory or stage differs")
    _cuda_preflight()
    common = Path(subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=root,
        capture_output=True, text=True, check=True).stdout.strip()).resolve(strict=True)
    marker = common/"machine_local_equation_budgets"/(AUTH+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "plan_sha256": sha256_file(root/PLAN),
                   "retry_allowed": False}, stream)
    from kron_reduce.evaluation.machine_local_equation import execute

    summary = execute(root, result_dir, plan)
    if git_head(root) != commit:
        raise RuntimeError("joint equation source changed during execution")
    verify_artifact_directory(result_dir, strict=True)
    return summary


def main():
    root = project_root()
    with workflow_lock(root, NAME):
        entry, _ = authorize(root)
        _cuda_preflight()
        print(_run_experiment(root, NAME, STAGE, experiment=entry))


def _cuda_preflight():
    import torch

    if (os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or torch.version.cuda != "13.0" or not torch.cuda.is_available()):
        raise RuntimeError("registered deterministic CUDA 13.0 inference runtime is unavailable")


if __name__ == "__main__":
    main()
