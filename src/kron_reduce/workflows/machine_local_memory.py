"""Run one registered machine-local memory study with the workflow lock held."""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from kron_reduce.workflows.machine_local_equation import _cuda_preflight
from project_harness.evidence import sha256_file
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy
from project_harness.runner import _run_experiment

STUDIES = {"diagnosis": "machine_local_memory_diagnosis_v1", "calibration": "machine_local_memory_calibration_v1"}
MODULE = "kron_reduce.workflows.machine_local_memory"
DATA_POLICY = "analytic_synthetic_only"


def registration(root, stage):
    assert_clean(root)
    name = STUDIES[stage]
    entry = load_manifest(root)["experiments"][name]
    if (entry.get("module") != MODULE or entry.get("stage") != stage or entry.get("workflow") != name
            or entry.get("dataset") != "synthetic_benchmark_registry_v1"
            or entry.get("data_policy") != DATA_POLICY or entry.get("execution_mode") != "bounded_synthetic_cli_only"
            or not approval_recorded(load_state(root, name), stage)):
        raise RuntimeError("exact synthetic registration and explicit stage approval required")
    authorization = name+"_20261114"
    auth = load_policy(root)["bounded_synthetic_evaluations"][authorization]
    scope = {"status": "active", "experiment": name, "workflow": name, "stage": stage, "real_data_reads": 0,
             "max_executions": 1, "retry_allowed": False, "plan_path": f"configs/proposals/{name}.json"}
    if any(type(auth.get(key)) is not type(value) or auth[key] != value for key, value in scope.items()):
        raise RuntimeError("active one-use memory study scope differs")
    for field in ("plan", "review"):
        verify_bound_file(root, auth[f"{field}_path"], auth[f"{field}_sha256"])
    plan, review = (json.loads((root/auth[f"{field}_path"]).read_text(encoding="utf-8")) for field in ("plan", "review"))
    if (plan["authorization"] != authorization or plan["stage"] != stage
            or plan["operation_limits"] != auth["operation_limits"] or not review.get("approved")
            or review.get("plan_path") != auth["plan_path"] or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan["implementation_bindings"]):
        raise RuntimeError("require independent review of the exact implementation and budget")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)
    if stage == "calibration":
        previous = load_policy(root)["bounded_synthetic_evaluations"][STUDIES["diagnosis"]+"_20261114"]
        required = plan["diagnosis"]
        if (previous["status"] != "consumed" or previous["scientific_pass"] is not True
                or previous["run_id"] != required["run_id"]
                or previous["result_manifest_sha256"] != required["artifact_manifest_sha256"]):
            raise RuntimeError("valid completed coefficient-substitution diagnosis required")
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if sys.version_info[:2] != (3, 12) or any(version(p) != pins[p]+("+cu130" if p == "torch" else "")
                                              for p in ("numpy", "scipy", "torch")):
        raise RuntimeError("registered Python 3.12 and Torch CUDA runtime required")
    _cuda_preflight()
    return name, entry, plan


def run(*, root, experiment, stage, result_dir, run_id):
    name, entry, plan = registration(root, stage)
    if experiment != entry or result_dir != root/"results"/run_id or any(result_dir.iterdir()):
        raise RuntimeError("memory study dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                         cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"machine_local_memory_budgets"/(plan["authorization"]+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "retry_allowed": False}, stream)
    import torch

    from kron_reduce.evaluation.constant_c_observability import write_json
    from kron_reduce.evaluation.machine_local_synthetic import Budget

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kernel = importlib.import_module(f"kron_reduce.evaluation.{name.removesuffix('_v1')}")
    started = time.perf_counter()
    budget = Budget(result_dir/"operations.jsonl", plan["operation_limits"])
    try:
        result = kernel.execute(root, result_dir, plan, budget)
    finally:
        budget.stream.close()
    if git_head(root) != commit:
        raise RuntimeError("memory study source changed during execution")
    evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit, "data_policy": DATA_POLICY,
                "promotion_decision": "KEEP_BASELINE", "scientific_pass": result["scientific_pass"],
                "claim_boundary": result["claim_boundary"]}
    write_json(result_dir/"summary.json", result | evidence | {"elapsed_seconds": time.perf_counter()-started,
        "operation_counts": dict(budget.counts), "runtime": {"python": sys.version, "cuda": torch.version.cuda,
        "score_device": torch.cuda.get_device_name(), **{p: version(p) for p in ("numpy", "scipy", "torch")}}})
    write_json(result_dir/"stage_evidence.json", evidence)
    write_json(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
               for p in sorted(result_dir.rglob("*")) if p.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STUDIES)
    stage = parser.parse_args().stage
    root = project_root()
    with workflow_lock(root, STUDIES[stage]):
        name, entry, _ = registration(root, stage)
        print(_run_experiment(root, name, stage, experiment=entry))


if __name__ == "__main__":
    main()
