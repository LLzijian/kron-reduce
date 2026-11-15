"""Registered one-use run for the topology and memory engineering gate on synthetic truth."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy
from project_harness.runner import _run_experiment

MODULE = "kron_reduce.workflows.machine_local_topology"
STUDIES = {"s0": "machine_local_topology_v1", "closeout": "machine_local_topology_closeout_v1",
           "learning": "machine_local_topology_learning_v1", "diagnosis": "machine_local_topology_diagnosis_v1",
           "refinement": "machine_local_topology_refinement_v1", "transfer": "machine_local_topology_transfer_v1",
           "margin": "machine_local_topology_margin_v1", "margin_v2": "machine_local_topology_margin_v2",
           "delayed": "machine_local_topology_delayed_v1",
           "delayed_origin": "machine_local_topology_delayed_origin_v1"}


def authorize(root, stage):
    assert_clean(root)
    name = STUDIES[stage]
    auth_id = name+"_20261115"
    entry = load_manifest(root)["experiments"][name]
    expected = {"module": MODULE, "workflow": name, "stage": stage, "dataset": "synthetic_benchmark_registry_v1",
                "execution_mode": "bounded_synthetic_cli_only", "data_policy": "analytic_synthetic_only"}
    if (sys.version_info[:2] != (3, 12) or any(entry.get(k) != v for k, v in expected.items())
            or not approval_recorded(load_state(root, name), stage)):
        raise RuntimeError("synthetic runtime, registration or explicit loop approval differs")
    policy = load_policy(root)["bounded_synthetic_evaluations"]
    auth = policy[auth_id]
    plan_path, review_path = auth["plan_path"], auth["review_path"]
    scope = {"status": "active", "experiment": name, "workflow": name, "stage": stage,
             "real_data_reads": 0, "max_executions": 1, "retry_allowed": False,
             "fit_attempts": {"learning": 4, "refinement": 2, "margin": 2, "margin_v2": 2, "delayed": 2}.get(stage, 0)}
    if any(type(auth.get(key)) is not type(value) or auth[key] != value for key, value in scope.items()):
        raise RuntimeError("active topology and memory scope differs")
    verify_bound_file(root, plan_path, auth["plan_sha256"])
    verify_bound_file(root, review_path, auth["review_sha256"])
    plan, review = (json.loads((root/path).read_text(encoding="utf-8")) for path in (plan_path, review_path))
    bindings = plan["implementation_bindings"]
    if (plan["authorization"] != auth_id or plan["dataset"] != expected["dataset"] or plan["stage"] != stage
            or review.get("approved") is not True or review.get("plan_path") != plan_path
            or review.get("plan_sha256") != auth["plan_sha256"] or review.get("implementation_bindings") != bindings
            or auth["operation_limits"] != plan["operation_limits"]):
        raise RuntimeError("exact reviewed topology and memory plan required")
    for path, digest in bindings.items():
        verify_bound_file(root, path, digest)
    if stage in ("margin", "margin_v2", "delayed") and (plan["operation_limits"]["fits"] != scope["fit_attempts"]
            or plan["settings"]["updates"]*scope["fit_attempts"] != plan["operation_limits"]["optimizer_updates"]):
        raise RuntimeError("paired margin training requires the exact update budget")
    if stage == "margin_v2":
        previous = policy[STUDIES["margin"]+"_20261115"]
        if (previous["status"] != "consumed" or previous["failed_record_sha256"] != plan["failed_record_sha256"]):
            raise RuntimeError("repaired margin run requires the archived serialization failure")
    if stage == "delayed_origin":
        previous = policy[STUDIES["delayed"]+"_20261115"]
        source = plan["source"]
        if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                or previous["run_id"] != source["run_id"]
                or previous["result_manifest_sha256"] != source["manifest_sha256"]
                or previous["result_receipt_sha256"] != source["receipt_sha256"]):
            raise RuntimeError("origin controls require the verified frozen delayed cohort")
        verify_bound_file(root, previous["result_receipt_path"], source["receipt_sha256"])
    if stage == "closeout":
        previous = policy[STUDIES["s0"]+"_20261115"]
        if (previous["status"] != "consumed" or previous["run_id"] != plan["source"]["run_id"]
                or previous["failed_record_sha256"] != plan["source"]["files"]["RUN_FAILED.json"]):
            raise RuntimeError("closeout requires the exact consumed serialization failure")
    if stage == "learning":
        previous = policy[STUDIES["closeout"]+"_20261115"]
        if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                or previous["result_receipt_path"] != plan["prerequisite"]["path"]
                or previous["result_receipt_sha256"] != plan["prerequisite"]["sha256"]
                or plan["operation_limits"]["fits"] != scope["fit_attempts"]
                or plan["settings"]["updates"]*scope["fit_attempts"] != plan["operation_limits"]["optimizer_updates"]):
            raise RuntimeError("learning requires the completed S0 engineering gate and exact update budget")
        verify_bound_file(root, previous["result_receipt_path"], previous["result_receipt_sha256"])
    if stage == "transfer":
        for key, supports in (("learning", {"T0M0", "T1M0"}), ("refinement", {"T0M1", "T1M1"})):
            previous = policy[STUDIES[key]+"_20261115"]
            source = plan["sources"][key]
            if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                    or (key == "refinement" and previous["scientific_pass"] is not True)
                    or previous["run_id"] != source["run_id"]
                    or previous["result_manifest_sha256"] != source["manifest_sha256"]
                    or previous["result_receipt_sha256"] != source["receipt_sha256"]
                    or set(source["equation_sha256"]) != supports):
                raise RuntimeError("transfer requires the exact frozen learning and refinement evidence")
            verify_bound_file(root, previous["result_receipt_path"], source["receipt_sha256"])
    if stage in ("diagnosis", "refinement"):
        previous = policy[STUDIES["learning"]+"_20261115"]
        source = plan["source"]
        if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                or previous["run_id"] != source["run_id"]
                or previous["result_manifest_sha256"] != source["manifest_sha256"]
                or previous["result_receipt_sha256"] != source["receipt_sha256"]):
            raise RuntimeError("diagnosis requires the exact frozen learning evidence")
        verify_bound_file(root, previous["result_receipt_path"], source["receipt_sha256"])
    if stage == "refinement":
        previous = policy[STUDIES["diagnosis"]+"_20261115"]
        if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                or previous["result_receipt_sha256"] != plan["diagnosis_receipt_sha256"]
                or plan["operation_limits"]["fits"] != scope["fit_attempts"]
                or plan["settings"]["updates"]*scope["fit_attempts"] != plan["operation_limits"]["optimizer_updates"]):
            raise RuntimeError("refinement requires the completed factorial diagnosis and exact update budget")
        verify_bound_file(root, previous["result_receipt_path"], previous["result_receipt_sha256"])
    pins = dict(line.strip().split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#"))
    for package in ("numpy", "torch", "scipy"):
        expected_version = pins[package] + ("+cu130" if package == "torch" else "")
        if version(package) != expected_version:
            raise RuntimeError(f"unregistered runtime package: {package}")
    return entry, plan, auth


def run(*, root, experiment, stage, result_dir, run_id):
    entry, plan, auth = authorize(root, stage)
    if experiment != entry or result_dir.parent != root/"results" or result_dir.name != run_id or any(result_dir.iterdir()):
        raise RuntimeError("topology and memory result directory or stage differs")
    _cuda_preflight()
    common = Path(subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=root,
        capture_output=True, text=True, check=True).stdout.strip()).resolve(strict=True)
    marker = common/"machine_local_topology_budgets"/(plan["authorization"]+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "plan_sha256": auth["plan_sha256"],
                   "retry_allowed": False}, stream)
    import torch

    from kron_reduce.evaluation.constant_c_observability import write_json
    from kron_reduce.evaluation.machine_local_synthetic import Budget
    from kron_reduce.evaluation.machine_local_topology import closeout, execute
    from kron_reduce.evaluation.machine_local_topology_delayed import execute as delayed
    from kron_reduce.evaluation.machine_local_topology_delayed_origin import (
        execute as delayed_origin,
    )
    from kron_reduce.evaluation.machine_local_topology_diagnosis import execute as diagnose
    from kron_reduce.evaluation.machine_local_topology_learning import execute as learn
    from kron_reduce.evaluation.machine_local_topology_margin import execute as margin
    from kron_reduce.evaluation.machine_local_topology_transfer import execute as transfer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    budget = Budget(result_dir/"operations.jsonl", plan["operation_limits"])
    try:
        kernel = {"closeout": closeout, "learning": learn, "s0": execute, "diagnosis": diagnose,
                  "refinement": learn, "transfer": transfer, "margin": margin, "margin_v2": margin,
                  "delayed": delayed, "delayed_origin": delayed_origin}[stage]
        summary = kernel(root, result_dir, plan, budget)
    finally:
        budget.stream.close()
    evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit,
                "data_policy": "analytic_synthetic_only", "promotion_decision": "KEEP_BASELINE",
                **{key: summary[key] for key in ("engineering_pass", "scientific_pass", "claim_boundary")}}
    write_json(result_dir/"summary.json", summary | evidence | {
        "operation_counts": dict(budget.counts), "elapsed_seconds": time.perf_counter()-started,
        "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "numpy": version("numpy"), "scipy": version("scipy")},
    })
    write_json(result_dir/"stage_evidence.json", evidence)
    write_json(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
               for p in sorted(result_dir.rglob("*")) if p.is_file()})
    if git_head(root) != commit:
        raise RuntimeError("topology and memory source changed during execution")
    verify_artifact_directory(result_dir, strict=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STUDIES, default="s0")
    stage = parser.parse_args().stage
    root = project_root()
    name = STUDIES[stage]
    with workflow_lock(root, name):
        entry, _, _ = authorize(root, stage)
        _cuda_preflight()
        print(_run_experiment(root, name, stage, experiment=entry))


def _cuda_preflight():
    import torch

    if (os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or torch.version.cuda != "13.0" or not torch.cuda.is_available()):
        raise RuntimeError("registered deterministic CUDA 13.0 inference runtime is unavailable")


if __name__ == "__main__":
    main()
