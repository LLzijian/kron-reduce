"""One-use fixed-support TASK3 fit and frozen GPU evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from kron_reduce.workflows.machine_local_equation import _cuda_preflight
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "kiki_machine_local_topology_v1"
MODULE = "kron_reduce.workflows.kiki_machine_local_topology"
CALIBRATION = "machine_local_topology_transfer_v1_20261115"


def authorize(root, stage):
    assert_clean(root)
    name = f"kiki_machine_local_topology_{stage}_v1"
    entry = load_manifest(root)["experiments"][name]
    if (sys.version_info[:2] != (3, 12) or stage not in ("fit", "test")
            or entry["workflow"] != WORKFLOW or entry["stage"] != stage or entry["module"] != MODULE
            or entry["execution_mode"] != "bounded_kiki_official_cli_only"
            or not approval_recorded(load_state(root, WORKFLOW), stage)):
        raise RuntimeError("registered TASK3 topology stage and explicit stage approval required")
    validate_run_policy(root, name, entry)
    policy = load_policy(root)
    authorization = name + "_20261115"
    category = "development_evaluations" if stage == "fit" else "official_test_evaluations"
    auth = policy[category][authorization]
    scope = {"status": "active", "experiment": name, "fit_attempts": 2 if stage == "fit" else 0,
             "max_executions": 1, "retry_allowed": False}
    if any(type(auth.get(key)) is not type(value) or auth[key] != value for key, value in scope.items()):
        raise RuntimeError("active single-use TASK3 scope differs")
    for field in ("plan", "review"):
        verify_bound_file(root, auth[f"{field}_path"], auth[f"{field}_sha256"])
    plan, review = (json.loads((root / auth[f"{field}_path"]).read_text(encoding="utf-8"))
                    for field in ("plan", "review"))
    if (plan["stage"] != stage or plan["device"] != "cuda" or auth["operation_limits"] != plan["operation_limits"]
            or review.get("approved") is not True or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan.get("implementation_bindings")):
        raise RuntimeError("exact TASK3 topology implementation and budget review required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)

    transfer = policy["bounded_synthetic_evaluations"][CALIBRATION]
    for key in ("run_id", "result_manifest_sha256", "result_receipt_sha256", "result_review_sha256"):
        if transfer[key] != plan["transfer"][key]:
            raise RuntimeError("registered synthetic transfer gate differs")
    if transfer["status"] != "consumed" or transfer.get("scientific_pass") is not True:
        raise RuntimeError("completed synthetic transfer gate must pass")
    for key in ("result_receipt", "result_review"):
        verify_bound_file(root, transfer[key + "_path"], transfer[key + "_sha256"])

    source = plan["source_contract"]
    verify_bound_file(root, source["path"], source["sha256"])
    source_plan = json.loads((root / source["path"]).read_text(encoding="utf-8"))
    if plan["backbone"] != source_plan["backbone"]:
        raise RuntimeError("starting backbone differs from the registered TASK3 source")
    if stage == "test":
        fit_auth = policy["development_evaluations"]["kiki_machine_local_topology_fit_v1_20261115"]
        if fit_auth["status"] != "consumed" or plan["fit_plan_sha256"] != fit_auth["plan_sha256"]:
            raise RuntimeError("test requires the completed paired fit")
        verify_bound_file(root, fit_auth["plan_path"], fit_auth["plan_sha256"])
        fit_plan = json.loads((root / fit_auth["plan_path"]).read_text(encoding="utf-8"))
        if plan["settings"] != fit_plan["settings"]:
            raise RuntimeError("test settings differ from the frozen fit plan")
        if plan["frozen_model_manifest"] != auth["frozen_model_manifest"]:
            raise RuntimeError("test must use the frozen fit artifact")
        binding = auth["frozen_model_manifest"]
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise RuntimeError("frozen model manifest is missing or changed")
    for path, digest in source_plan["external_code"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("TASK3 scoring source changed")
    plan["sources"] = source_plan["sources"] | {"allowed_roles": ["train", "val"] if stage == "fit" else ["test"]}
    plan["task3_root"] = source_plan["task3_root"]
    pins = dict(line.split("==") for line in (root / "requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if any(version(package) != pins[package] + ("+cu130" if package == "torch" else "")
           for package in ("numpy", "scipy", "torch", "filelock", "h5py")):
        raise RuntimeError("registered Python 3.12 and Torch CUDA runtime required")
    _cuda_preflight()
    return name, entry, plan, authorization


def run(*, root, experiment, stage, result_dir, run_id):
    _, entry, plan, authorization = authorize(root, stage)
    if experiment != entry or result_dir != root / "results" / run_id or any(result_dir.iterdir()):
        raise RuntimeError("TASK3 topology dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common / "kiki_machine_local_topology_budgets" / (authorization + ".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "retry_allowed": False}, stream)
    from kron_reduce.evaluation.kiki_machine_local_topology import execute

    result = execute(root, result_dir, plan, stage)
    if git_head(root) != commit:
        raise RuntimeError("TASK3 topology source changed during execution")
    verify_artifact_directory(result_dir, strict=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "test"), required=True)
    stage = parser.parse_args().stage
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        name, entry, _, _ = authorize(root, stage)
        print(_run_experiment(root, name, stage, experiment=entry))


if __name__ == "__main__":
    main()
