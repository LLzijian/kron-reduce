"""Run the registered joint ES-SINDy gate or its frozen TASK3 comparison."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from kron_reduce.workflows.machine_local_topology import _cuda_preflight
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

MODULE = "kron_reduce.workflows.joint_energy_sindy"
SYNTHETIC = "joint_energy_sindy_synthetic_v1"
SYNTHETIC_V2 = "joint_energy_sindy_synthetic_v2"
SYNTHETIC_RESTART = "joint_energy_sindy_synthetic_v2_restart"
TASK3 = "kiki_joint_energy_sindy_v1"


def _experiment_name(stage, requested=None):
    if stage == "synthetic":
        name = requested or SYNTHETIC
        if name not in (SYNTHETIC, SYNTHETIC_V2, SYNTHETIC_RESTART):
            raise RuntimeError("unregistered joint ES-SINDy synthetic experiment")
    else:
        if stage not in ("fit", "test"):
            raise RuntimeError("unknown joint ES-SINDy stage")
        name = f"kiki_joint_energy_sindy_{stage}_v1"
        if requested is not None and requested != name:
            raise RuntimeError("a synthetic experiment does not authorize a TASK3 run")
    return name


def authorize(root, stage, experiment_name=None):
    assert_clean(root)
    analytic = stage == "synthetic"
    name = _experiment_name(stage, experiment_name)
    entries = load_manifest(root)["experiments"]
    entry = entries[name]
    workflow = entry["workflow"]
    mode = "bounded_synthetic_cli_only" if analytic else "bounded_kiki_official_cli_only"
    expected = {"module": MODULE, "workflow": name if analytic else TASK3,
                "stage": stage, "execution_mode": mode}
    if (sys.version_info[:2] != (3, 12) or any(entry.get(k) != v for k, v in expected.items())
            or not approval_recorded(load_state(root, workflow), stage)):
        raise RuntimeError("registered joint ES-SINDy stage, Python 3.12 and explicit stage approval required")
    validate_run_policy(root, name, entry)
    policy = load_policy(root)
    category, authority_key = {
        "synthetic": ("bounded_synthetic_evaluations", "synthetic_authorization_id"),
        "fit": ("development_evaluations", "development_authorization_id"),
        "test": ("official_test_evaluations", "test_authorization"),
    }[stage]
    authorization = entry[authority_key]
    auth = policy[category][authorization]
    attempts = 0 if stage == "test" else 1
    scope = {"status": "active", "experiment": name, "workflow": workflow,
             "stage": stage, "fit_attempts": attempts,
             "max_executions": 1, "retry_allowed": False}
    if any(type(auth.get(k)) is not type(v) or auth[k] != v for k, v in scope.items()):
        raise RuntimeError("active one-use joint ES-SINDy scope differs")
    for field in ("plan", "review"):
        verify_bound_file(root, auth[f"{field}_path"], auth[f"{field}_sha256"])
    plan, review = (json.loads((root/auth[f"{field}_path"]).read_bytes()) for field in ("plan", "review"))
    if (plan["stage"] != stage or plan["workflow"] != workflow
            or plan["experiment"] != name or plan["authorization"] != authorization
            or plan["plan_path"] != auth["plan_path"]
            or plan["operation_limits"] != auth["operation_limits"]
            or plan["operation_limits"]["fits"] != attempts
            or plan["operation_limits"]["optimizer_updates"] != (plan["settings"]["updates"] if attempts else 0)
            or review.get("approved") is not True or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan["implementation_bindings"]):
        raise RuntimeError("exact reviewed joint ES-SINDy budget and implementation required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)
    if not analytic:
        from kron_reduce.data.task3_full import REFERENCE, source_plan

        gate_id = entries[SYNTHETIC]["synthetic_authorization_id"]
        gate = policy["bounded_synthetic_evaluations"][gate_id]
        if gate["status"] != "consumed" or gate.get("scientific_pass") is not True:
            raise RuntimeError("joint synthetic structure and misspecification gate must pass before TASK3")
        bound = {"path": gate["result_receipt_path"], "sha256": gate["result_receipt_sha256"]}
        if plan["synthetic_receipt"] != bound:
            raise RuntimeError("joint synthetic receipt differs")
        verify_bound_file(root, bound["path"], bound["sha256"])
        source = plan["source_contract"]
        verify_bound_file(root, source["path"], source["sha256"])
        source = json.loads((root/source["path"]).read_bytes())
        sources = {k: v for k, v in source["sources"].items() if k != "allowed_roles"}
        files = sources["files"]
        files["reference"] = files["reference"] | {"path": str((root/REFERENCE).resolve())}
        if any(sha256_file(Path(path)) != digest for path, digest in source["external_code"].items()):
            raise RuntimeError("frozen TASK3 scoring source differs")
        if sources != source_plan(root):
            raise RuntimeError("frozen TASK3 source roster differs")
        plan["sources"] = sources | {"allowed_roles": ["train", "val"] if stage == "fit" else ["test"]}
        plan["task3_root"] = source["task3_root"]
        if stage == "test":
            fit_id = entries["kiki_joint_energy_sindy_fit_v1"]["development_authorization_id"]
            fit = policy["development_evaluations"][fit_id]
            if (fit["status"] != "consumed" or plan["fit_plan_sha256"] != fit["plan_sha256"]
                    or plan["frozen_model_manifest"] != auth["frozen_model_manifest"]):
                raise RuntimeError("joint ES-SINDy test requires the exact frozen one-fit result")
    elif entry["dataset"] != "synthetic_benchmark_registry_v1" or auth.get("real_data_reads") != 0:
        raise RuntimeError("synthetic gate accepts only analytic truth")
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    packages = ("numpy", "scipy", "torch", "filelock", "h5py")
    if name in (SYNTHETIC_V2, SYNTHETIC_RESTART):
        device = plan["settings"].get("device")
        if device not in ("cpu", "cuda"):
            raise RuntimeError("synthetic v2 must explicitly register cpu or cuda")
        for package in packages:
            allowed = {pins[package]}
            if package == "torch":
                allowed.update((pins[package]+"+cu130", pins[package]+"+cpu"))
            if version(package) not in allowed:
                raise RuntimeError(f"joint ES-SINDy v2 requires the pinned {package} base version")
        if device == "cuda":
            _cuda_preflight()
    else:
        if any(version(package) != pins[package]+("+cu130" if package == "torch" else "")
               for package in packages):
            raise RuntimeError("joint ES-SINDy requires the pinned deterministic CUDA environment")
        _cuda_preflight()
    return name, entry, plan, authorization, auth


def run(*, root, experiment, stage, result_dir, run_id):
    requested = experiment.get("workflow") if stage == "synthetic" else None
    name, entry, plan, authorization, auth = authorize(root, stage, requested)
    if experiment != entry or result_dir != root/"results"/run_id or any(result_dir.iterdir()):
        raise RuntimeError("joint ES-SINDy dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"joint_energy_sindy_budgets"/(authorization+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "plan_sha256": auth["plan_sha256"],
                   "retry_allowed": False}, stream)
    if stage == "synthetic":
        import torch

        from kron_reduce.evaluation.constant_c_observability import write_json
        from kron_reduce.evaluation.joint_energy_sindy_synthetic import execute
        from kron_reduce.evaluation.machine_local_synthetic import Budget

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        started = time.perf_counter()
        limits = {key: float("inf") if limit is None else limit
                  for key, limit in plan["operation_limits"].items()}
        budget = Budget(result_dir/"operations.jsonl", limits)
        try:
            summary = execute(root, result_dir, plan, budget)
        finally:
            budget.stream.close()
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit,
                    "data_policy": "analytic_synthetic_only", "promotion_decision": "KEEP_BASELINE",
                    **{key: summary[key] for key in ("engineering_pass", "scientific_pass", "claim_boundary")}}
        device = plan["settings"].get("device", "cuda")
        write_json(result_dir/"summary.json", summary | evidence | {
            "operation_counts": dict(budget.counts), "elapsed_seconds": time.perf_counter()-started,
            "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                        **{package: version(package) for package in ("numpy", "scipy", "filelock", "h5py")}}})
        write_json(result_dir/"stage_evidence.json", evidence)
        write_json(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
                   for p in sorted(result_dir.rglob("*")) if p.is_file()})
    else:
        from kron_reduce.evaluation.kiki_joint_energy_sindy import execute

        summary = execute(root, result_dir, plan, stage)
    if git_head(root) != commit:
        raise RuntimeError("joint ES-SINDy source changed during execution")
    verify_artifact_directory(result_dir, strict=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("synthetic", "fit", "test"))
    parser.add_argument("--experiment", help="Registered experiment; synthetic defaults to v1")
    args = parser.parse_args()
    stage = args.stage
    root = project_root()
    name = _experiment_name(stage, args.experiment)
    entry = load_manifest(root)["experiments"][name]
    with workflow_lock(root, entry["workflow"]):
        name, entry, _, _, _ = authorize(root, stage, name)
        print(_run_experiment(root, name, stage, experiment=entry))


if __name__ == "__main__":
    main()
