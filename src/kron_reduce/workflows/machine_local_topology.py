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
           "delayed_origin": "machine_local_topology_delayed_origin_v1",
           "delayed_damping": "machine_local_topology_delayed_damping_v1",
           "ball_history": "machine_local_topology_ball_history_v1",
           "ball_history_no_prior": "machine_local_topology_ball_history_no_prior_v1",
           "momentum": "machine_local_topology_momentum_v1",
           "angle_damping": "machine_local_topology_angle_damping_v1",
           "spatial_damping": "machine_local_topology_spatial_damping_v1",
           "restoring_shape": "machine_local_topology_restoring_shape_v1",
           "dynamic_coupling": "machine_local_topology_dynamic_coupling_v1",
           "spatial_coupling": "machine_local_topology_spatial_coupling_v1",
           "spatial_coupling_causal": "machine_local_topology_spatial_coupling_causal_v1",
           "topology_memory_rom": "topology_memory_rom_synthetic_v1",
           "topology_memory_rom_causal": "topology_memory_rom_causal_v1",
           "topology_memory_rom_coupling_strength": "topology_memory_rom_coupling_strength_v1",
           "topology_memory_rom_branch_init": "topology_memory_rom_branch_init_v1"}


def authorize(root, stage):
    assert_clean(root)
    name = STUDIES[stage]
    recent = ("dynamic_coupling", "spatial_coupling", "spatial_coupling_causal", "topology_memory_rom",
              "topology_memory_rom_causal")
    suffix = "_20261117" if stage in ("topology_memory_rom_coupling_strength", "topology_memory_rom_branch_init") else (
        "_20261116" if stage in recent else "_20261115")
    auth_id = name+suffix
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
             "fit_attempts": {"learning": 4, "refinement": 2, "margin": 2, "margin_v2": 2,
                              "delayed": 2, "delayed_damping": 1, "angle_damping": 4,
                              "spatial_damping": 2, "restoring_shape": 6, "dynamic_coupling": 6,
                              "spatial_coupling": 6, "spatial_coupling_causal": 6,
                              "topology_memory_rom": 40}.get(stage, 0)}
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
    if stage in ("delayed_origin", "delayed_damping", "ball_history"):
        previous = policy[STUDIES["delayed"]+"_20261115"]
        source = plan["source"]
        if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                or previous["run_id"] != source["run_id"]
                or previous["result_manifest_sha256"] != source["manifest_sha256"]
                or previous["result_receipt_sha256"] != source["receipt_sha256"]):
            raise RuntimeError("origin controls require the verified frozen delayed cohort")
        verify_bound_file(root, previous["result_receipt_path"], source["receipt_sha256"])
    if stage == "ball_history_no_prior":
        previous = policy[STUDIES["ball_history"]+"_20261115"]
        source = plan["source"]
        if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                or previous["run_id"] != source["run_id"]
                or previous["result_manifest_sha256"] != source["manifest_sha256"]
                or previous["result_receipt_sha256"] != source["receipt_sha256"]):
            raise RuntimeError("no-prior comparison requires the verified complete observer cohort")
        verify_bound_file(root, previous["result_receipt_path"], source["receipt_sha256"])
        evidence = json.loads((root/previous["result_receipt_path"]).read_bytes())
        if evidence["gates"]["clean_recovery"] is not True:
            raise RuntimeError("the frozen no-prior candidate failed clean recovery")
    if stage in ("momentum", "angle_damping", "spatial_damping", "restoring_shape", "dynamic_coupling",
                 "spatial_coupling", "spatial_coupling_causal"):
        previous = policy[STUDIES["ball_history"]+"_20261115"]
        source = plan["source"]
        if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                or previous["run_id"] != source["run_id"]
                or previous["result_manifest_sha256"] != source["manifest_sha256"]
                or previous["result_receipt_sha256"] != source["receipt_sha256"]):
            raise RuntimeError("momentum diagnosis requires the verified complete known-truth cohort")
        verify_bound_file(root, previous["result_receipt_path"], source["receipt_sha256"])
    if stage == "angle_damping" and plan["operation_limits"]["fits"] != scope["fit_attempts"]:
        raise RuntimeError("angle damping requires four bounded scalar fits")
    if stage == "topology_memory_rom":
        prior = load_policy(root)["development_evaluations"]["kiki_topology_memory_balance_v1_20261116"]
        receipt = {"path": prior["result_receipt_path"], "sha256": prior["result_receipt_sha256"]}
        if (prior["status"] != "consumed" or prior["engineering_pass"] is not True
                or plan["diagnosis_receipt"] != receipt or plan["operation_limits"]["fits"] != scope["fit_attempts"]):
            raise RuntimeError("topology-memory identification requires the completed train diagnosis and paired supports")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
    if stage == "topology_memory_rom_causal":
        prior = policy[STUDIES["topology_memory_rom"]+"_20261116"]
        source = {"run_id": prior["run_id"], "manifest_sha256": prior["result_manifest_sha256"],
                  "plan_path": prior["plan_path"], "plan_sha256": prior["plan_sha256"],
                  "receipt_path": prior["result_receipt_path"], "receipt_sha256": prior["result_receipt_sha256"]}
        if (prior["status"] != "consumed" or prior["engineering_pass"] is not True
                or prior["scientific_pass"] is not False or plan["source"] != source
                or "fits" in plan["operation_limits"]):
            raise RuntimeError("history prior diagnosis requires the closed synthetic comparison and zero fits")
        verify_bound_file(root, source["receipt_path"], source["receipt_sha256"])
        receipt = json.loads((root/source["receipt_path"]).read_bytes())
        if receipt["selected_by_truth"]["S0"]["gates"]["initial_state_recovery"]["status"] != "FAIL":
            raise RuntimeError("the history prior diagnosis requires the registered fixed-coefficient recovery failure")
    if stage == "topology_memory_rom_coupling_strength":
        previous = policy[STUDIES["topology_memory_rom_causal"]+"_20261116"]
        receipt = {"path": previous["result_receipt_path"], "sha256": previous["result_receipt_sha256"]}
        if (previous["status"] != "consumed" or previous["scientific_pass"] is not True
                or plan["causal_receipt"] != receipt or "fits" in plan["operation_limits"]):
            raise RuntimeError("the coupling comparison requires completed fixed-coefficient clean recovery and zero fits")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
    if stage == "topology_memory_rom_branch_init":
        previous = policy[STUDIES["topology_memory_rom_coupling_strength"]+"_20261117"]
        source = {"run_id": previous["run_id"], "manifest_sha256": previous["result_manifest_sha256"],
                  "receipt_path": previous["result_receipt_path"], "receipt_sha256": previous["result_receipt_sha256"],
                  "plan_path": previous["plan_path"], "plan_sha256": previous["plan_sha256"]}
        if (previous["status"] != "consumed" or previous["engineering_pass"] is not True
                or previous["scientific_pass"] is not False or plan["source"] != source or "fits" in plan["operation_limits"]):
            raise RuntimeError("the branch initializer requires the closed known-coefficient comparison and zero fits")
        verify_bound_file(root, source["receipt_path"], source["receipt_sha256"])
        verify_bound_file(root, source["plan_path"], source["plan_sha256"])
        original = json.loads((root/source["plan_path"]).read_bytes())
        same = ("ids", "cohorts", "speed_rad_s", "tau_range", "gates", "seed", "excitation", "truth")
        if (any(plan[key] != original[key] for key in same)
                or plan["settings"] != original["settings"] | {"branch_log_coordinates": True}):
            raise RuntimeError("the comparison may change only the branch history solve coordinates")
    if stage == "spatial_damping":
        prior = policy[STUDIES["angle_damping"]+"_20261115"]
        receipt = {"path": prior["result_receipt_path"], "sha256": prior["result_receipt_sha256"]}
        if (prior["status"] != "consumed" or prior["scientific_pass"] is not True
                or receipt != plan["angle_receipt"] or plan["operation_limits"]["fits"] != scope["fit_attempts"]):
            raise RuntimeError("spatial damping requires the completed global-damping study and two scalar fits")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
    if stage == "restoring_shape":
        prior = policy[STUDIES["spatial_damping"]+"_20261115"]
        receipt = {"path": prior["result_receipt_path"], "sha256": prior["result_receipt_sha256"]}
        if (prior["status"] != "consumed" or prior["scientific_pass"] is not True
                or receipt != plan["spatial_receipt"] or plan["operation_limits"]["fits"] != scope["fit_attempts"]
                or plan["truth_alpha"] != [-.125, 0., .125]):
            raise RuntimeError("restoring shape requires the completed spatial study and the signed paired design")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
    if stage == "dynamic_coupling":
        prior = policy[STUDIES["restoring_shape"]+"_20261115"]
        receipt = {"path": prior["result_receipt_path"], "sha256": prior["result_receipt_sha256"]}
        if (prior["status"] != "consumed" or prior["scientific_pass"] is not True
                or receipt != plan["static_receipt"] or plan["operation_limits"]["fits"] != scope["fit_attempts"]
                or plan["operation_limits"]["train_candidate_evaluations"] != 30):
            raise RuntimeError("dynamic coupling requires the verified static control and exact paired grid budget")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
    if stage in ("spatial_coupling", "spatial_coupling_causal"):
        prior = policy[STUDIES["dynamic_coupling"]+"_20261116"]
        receipt = {"path": prior["result_receipt_path"], "sha256": prior["result_receipt_sha256"]}
        grid = [0., .25, .5, 1., 2.]
        if (prior["status"] != "consumed" or prior["scientific_pass"] is not True
                or receipt != plan["common_receipt"] or plan["grids"] != {"dynamic": grid, "spatial": grid}
                or plan["operation_limits"]["fits"] != 6 or plan["operation_limits"]["train_candidate_evaluations"] != 30):
            raise RuntimeError("spatial coupling requires the verified common control and exact paired grid budget")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
    if stage == "spatial_coupling_causal":
        prior = policy[STUDIES["spatial_coupling"]+"_20261116"]
        receipt = {"path": prior["result_receipt_path"], "sha256": prior["result_receipt_sha256"]}
        if (prior["status"] != "consumed" or prior["engineering_pass"] is not True
                or prior["scientific_pass"] is not False or receipt != plan["first_spatial_receipt"]
                or plan["radius"] != 3. or plan["prior_weight"] != 0.
                or plan["gates"]["memory_relative_error"] != .001):
            raise RuntimeError("spatial causal comparison requires the archived first comparison and signed observer")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
    if stage == "delayed_damping":
        diagnosis = policy[STUDIES["delayed_origin"]+"_20261115"]
        if (diagnosis["status"] != "consumed" or diagnosis["engineering_pass"] is not True
                or diagnosis["result_receipt_sha256"] != plan["diagnosis_receipt_sha256"]):
            raise RuntimeError("damping fit requires the verified origin diagnosis")
        path = diagnosis["result_receipt_path"]
        verify_bound_file(root, path, plan["diagnosis_receipt_sha256"])
        if "damping" not in json.loads((root/path).read_bytes())["sufficient_interventions"]:
            raise RuntimeError("damping correction did not pass the registered diagnosis")
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
    from kron_reduce.evaluation.machine_local_topology_ball_history import execute as ball_history
    from kron_reduce.evaluation.machine_local_topology_ball_history_no_prior import (
        execute as ball_history_no_prior,
    )
    from kron_reduce.evaluation.machine_local_topology_delayed import execute as delayed
    from kron_reduce.evaluation.machine_local_topology_delayed_damping import (
        execute as delayed_damping,
    )
    from kron_reduce.evaluation.machine_local_topology_delayed_origin import (
        execute as delayed_origin,
    )
    from kron_reduce.evaluation.machine_local_topology_diagnosis import execute as diagnose
    from kron_reduce.evaluation.machine_local_topology_learning import execute as learn
    from kron_reduce.evaluation.machine_local_topology_margin import execute as margin
    from kron_reduce.evaluation.machine_local_topology_transfer import execute as transfer
    from kron_reduce.evaluation.topology_angle_damping_synthetic import (
        execute as angle_damping,
    )
    from kron_reduce.evaluation.topology_angle_damping_synthetic import (
        execute_spatial,
    )
    from kron_reduce.evaluation.topology_dynamic_coupling import execute as dynamic_coupling
    from kron_reduce.evaluation.topology_momentum_synthetic import execute as momentum
    from kron_reduce.evaluation.topology_restoring_synthetic import execute as restoring_shape

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    budget = Budget(result_dir/"operations.jsonl", plan["operation_limits"])
    try:
        kernels = {"closeout": closeout, "learning": learn, "s0": execute, "diagnosis": diagnose,
                  "refinement": learn, "transfer": transfer, "margin": margin, "margin_v2": margin,
                  "delayed": delayed, "delayed_origin": delayed_origin, "delayed_damping": delayed_damping,
                  "ball_history": ball_history, "ball_history_no_prior": ball_history_no_prior,
                  "momentum": momentum, "angle_damping": angle_damping, "spatial_damping": execute_spatial,
                  "restoring_shape": restoring_shape, "dynamic_coupling": dynamic_coupling,
                  "spatial_coupling": dynamic_coupling, "spatial_coupling_causal": dynamic_coupling}
        if stage == "topology_memory_rom":
            from kron_reduce.evaluation.topology_memory_rom_synthetic import execute as rom

            kernels[stage] = rom
        if stage == "topology_memory_rom_causal":
            from kron_reduce.evaluation.topology_memory_rom_causal import execute as rom_causal

            kernels[stage] = rom_causal
        if stage in ("topology_memory_rom_coupling_strength", "topology_memory_rom_branch_init"):
            from kron_reduce.evaluation.topology_memory_rom_coupling_strength import (
                execute as rom_coupling_strength,
            )

            kernels[stage] = rom_coupling_strength
        kernel = kernels[stage]
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
