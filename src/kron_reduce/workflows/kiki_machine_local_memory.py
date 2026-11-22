"""One-use TASK3 momentum calibration and frozen GPU evaluation."""
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
from project_harness.report_storage import locate_report
from project_harness.runner import _run_experiment

WORKFLOW = "kiki_machine_local_memory_v1"
MODULE = "kron_reduce.workflows.kiki_machine_local_memory"
CALIBRATION = "machine_local_memory_calibration_v1_20261114"


def authorize(root, stage):
    assert_clean(root)
    name = f"kiki_machine_local_memory_{stage}_v1"
    entry = load_manifest(root)["experiments"][name]
    if (sys.version_info[:2] != (3, 12) or stage not in ("fit", "test")
            or entry["workflow"] != WORKFLOW or entry["stage"] != stage or entry["module"] != MODULE
            or entry["execution_mode"] != "bounded_kiki_official_cli_only"
            or not approval_recorded(load_state(root, WORKFLOW), stage)):
        raise RuntimeError("registered TASK3 momentum stage and explicit stage approval required")
    validate_run_policy(root, name, entry)
    policy = load_policy(root)
    authorization = name+"_20261114"
    category = "development_evaluations" if stage == "fit" else "official_test_evaluations"
    auth = policy[category][authorization]
    scope = {"status": "active", "experiment": name, "fit_attempts": 4 if stage == "fit" else 0,
             "max_executions": 1, "retry_allowed": False}
    if any(type(auth.get(key)) is not type(value) or auth[key] != value for key, value in scope.items()):
        raise RuntimeError("active single-use TASK3 scope differs")
    for field in ("plan", "review"):
        verify_bound_file(root, auth[f"{field}_path"], auth[f"{field}_sha256"])
    plan, review = (json.loads((root/locate_report(root, auth[f"{field}_path"], auth[f"{field}_sha256"])).read_text(encoding="utf-8")) for field in ("plan", "review"))
    if (plan["stage"] != stage or auth["operation_limits"] != plan["operation_limits"]
            or review.get("approved") is not True or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan["implementation_bindings"]):
        raise RuntimeError("exact TASK3 momentum implementation and budget review required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)

    calibration = policy["bounded_synthetic_evaluations"][CALIBRATION]
    for key in ("run_id", "result_manifest_sha256", "result_receipt_sha256", "result_review_sha256"):
        if calibration[key] != plan["calibration"][key]:
            raise RuntimeError("registered synthetic continuation gate differs")
    if calibration["status"] != "consumed" or calibration["scientific_pass"] is not True:
        raise RuntimeError("completed synthetic coefficient recovery gate must pass")
    for key in ("result_receipt", "result_review"):
        verify_bound_file(root, calibration[key+"_path"], calibration[key+"_sha256"])

    source = plan["source_contract"]
    verify_bound_file(root, source["path"], source["sha256"])
    source_plan = json.loads((root/locate_report(root, source["path"], source["sha256"])).read_text(encoding="utf-8"))
    if stage == "fit" and plan["backbone"] != source_plan["backbone"]:
        raise RuntimeError("starting backbone differs from the registered TASK3 source")
    if stage == "test" and plan["frozen_model_manifest"] != auth["frozen_model_manifest"]:
        raise RuntimeError("test must use the frozen fit artifact")
    for path, digest in source_plan["external_code"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("TASK3 scoring source changed")
    plan["sources"] = source_plan["sources"] | {"allowed_roles": ["train", "val"] if stage == "fit" else ["test"]}
    plan["task3_root"] = source_plan["task3_root"]
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if any(version(p) != pins[p]+("+cu130" if p == "torch" else "")
           for p in ("numpy", "scipy", "torch", "filelock", "h5py")):
        raise RuntimeError("registered Python 3.12 and Torch CUDA runtime required")
    _cuda_preflight()
    return name, entry, plan, authorization


def run(*, root, experiment, stage, result_dir, run_id):
    _, entry, plan, authorization = authorize(root, stage)
    if experiment != entry or result_dir != root/"results"/run_id or any(result_dir.iterdir()):
        raise RuntimeError("TASK3 momentum dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                         cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"kiki_machine_local_memory_budgets"/(authorization+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "retry_allowed": False}, stream)
    from kron_reduce.evaluation.kiki_machine_local_memory import execute

    result = execute(root, result_dir, plan, stage)
    if git_head(root) != commit:
        raise RuntimeError("TASK3 momentum source changed during execution")
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
