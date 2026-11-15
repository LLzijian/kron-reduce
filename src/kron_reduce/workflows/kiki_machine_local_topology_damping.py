"""Run a bounded TASK3 damping fit and a separately authorized frozen test."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from kron_reduce.workflows.kiki_machine_local_topology_radius import _bound_json, _result_file
from kron_reduce.workflows.machine_local_equation import _cuda_preflight
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "kiki_machine_local_topology_damping_v1"
PREFIX = "kiki_machine_local_topology_damping"
MODULE = "kron_reduce.workflows.kiki_machine_local_topology_damping"
CALIBRATION = "machine_local_topology_delayed_damping_v1_20261115"


def _frozen(root, binding):
    manifest = json.loads(_result_file(root, binding).read_bytes())
    if (manifest["phase"] != "freeze" or manifest["fitting_closed"] is not True
            or manifest["selection_closed"] is not True
            or manifest["input_contract"] != "measured_history_100ms_zoh_v1"):
        raise RuntimeError("TASK3 damping requires a closed model freeze")
    index, prep = (_result_file(root, {"path": manifest[f"{key}_path"], "sha256": manifest[f"{key}_sha256"]})
                   for key in ("checkpoint", "preprocessing"))
    _result_file(root, json.loads(index.read_bytes())["T0M1"])
    return manifest, json.loads(prep.read_bytes())


def _saved_reference(root, policy, stage):
    category, roles = ("development_evaluations", ("train", "val")) if stage == "fit" else (
        "official_test_evaluations", ("test",))
    parent = policy[category][f"kiki_machine_local_topology_{stage}_v1_20261115"]
    if parent["status"] != "consumed":
        raise RuntimeError("the saved T0M1 reference stage is not complete")
    receipt = _bound_json(root, {"path": parent["result_receipt_path"], "sha256": parent["result_receipt_sha256"]})
    records = {role: {"path": f"results/{receipt['run_id']}/T0M1/{role}.json",
                      "sha256": receipt["records"]["T0M1"][role]["sha256"]} for role in roles}
    return receipt, records


def authorize(root, stage):
    assert_clean(root)
    name = f"{PREFIX}_{stage}_v1"
    entry = load_manifest(root)["experiments"][name]
    if (sys.version_info[:2] != (3, 12) or entry["workflow"] != WORKFLOW or entry["stage"] != stage
            or entry["module"] != MODULE or entry["execution_mode"] != "bounded_kiki_official_cli_only"
            or not approval_recorded(load_state(root, WORKFLOW), stage)):
        raise RuntimeError("registered damping stage and explicit approval required")
    validate_run_policy(root, name, entry)
    policy = load_policy(root)
    category = "development_evaluations" if stage == "fit" else "official_test_evaluations"
    authorization = name+"_20261115"
    auth = policy[category][authorization]
    scope = {"status": "active", "experiment": name, "fit_attempts": int(stage == "fit"),
             "max_executions": 1, "retry_allowed": False}
    if any(type(auth.get(k)) is not type(v) or auth[k] != v for k, v in scope.items()):
        raise RuntimeError("active single-use damping authority differs")
    plan, review = (_bound_json(root, {"path": auth[f"{key}_path"], "sha256": auth[f"{key}_sha256"]})
                    for key in ("plan", "review"))
    if (plan["stage"] != stage or plan["device"] != "cuda" or plan["fit_attempts"] != scope["fit_attempts"]
            or plan["cohort_counts"] != ({"train": 346, "val": 107} if stage == "fit" else {"test": 76})
            or plan["source_cache_open_limit"] != (3 if stage == "fit" else 77)
            or auth["operation_limits"] != plan["operation_limits"] or review.get("approved") is not True
            or review.get("plan_path") != auth["plan_path"] or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan["implementation_bindings"]):
        raise RuntimeError("exact damping plan, budget and review required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)

    gate = policy["bounded_synthetic_evaluations"][CALIBRATION]
    if (gate["status"] != "consumed" or gate["scientific_pass"] is not True
            or any(gate[key] != digest for key, digest in plan["calibration"].items())):
        raise RuntimeError("the causal damping calibration must pass")
    for key in ("result_receipt", "result_review"):
        verify_bound_file(root, gate[key+"_path"], gate[key+"_sha256"])

    original, reference = _saved_reference(root, policy, stage)
    if reference != plan["reference"]:
        raise RuntimeError("saved complete-cohort reference differs")
    for binding in reference.values():
        _result_file(root, binding)
    original_binding = {k: original["frozen_model_manifest"][k] for k in ("path", "sha256")}
    if original_binding != plan["original_model_manifest"]:
        raise RuntimeError("original fit and reference freeze differ")
    model, scales = _frozen(root, plan["model_manifest"])
    if (scales["memory_radius"] != plan["memory_radius"] or model["backbone"] != plan["backbone"]
            or model["settings"]["prior_weight"] != plan["settings"]["prior_weight"]
            or plan["settings"]["prior_weight"] != 1e-6):
        raise RuntimeError("model basis, original radius or history prior differs")
    if stage == "fit":
        if plan["model_manifest"] != original_binding or model["training"]["groups"]["T0M1"]["selected_update"] != 12:
            raise RuntimeError("fit must start from original T0M1 update 12")
    else:
        fit = policy["development_evaluations"][PREFIX+"_fit_v1_20261115"]
        if (fit["status"] != "consumed" or plan["model_manifest"] != auth["frozen_model_manifest"]
                or model["fit_plan_sha256"] != fit["plan_sha256"] or model["settings"] != plan["settings"]):
            raise RuntimeError("test requires exactly the frozen damping endpoint")
        verify_bound_file(root, fit["plan_path"], fit["plan_sha256"])
        receipt = _bound_json(root, {"path": fit["result_receipt_path"], "sha256": fit["result_receipt_sha256"]})
        if plan["model_manifest"] != {k: receipt["frozen_model_manifest"][k] for k in ("path", "sha256")}:
            raise RuntimeError("test freeze differs from the fit receipt")

    source = _bound_json(root, plan["source_contract"])
    if source["backbone"] != plan["backbone"]:
        raise RuntimeError("TASK3 backbone differs from its source contract")
    for path, digest in source["external_code"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("TASK3 scoring source changed")
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if any(version(p) != pins[p]+("+cu130" if p == "torch" else "") for p in ("numpy", "scipy", "torch", "filelock", "h5py")):
        raise RuntimeError("registered Python 3.12 and Torch CUDA runtime required")
    _cuda_preflight()
    roles = ["train", "val"] if stage == "fit" else ["test"]
    runtime = plan | {"sources": source["sources"] | {"allowed_roles": roles}, "task3_root": source["task3_root"]}
    return name, entry, runtime, authorization


def run(*, root, experiment, stage, result_dir, run_id):
    _, entry, plan, authorization = authorize(root, stage)
    if experiment != entry or result_dir != root/"results"/run_id or any(result_dir.iterdir()):
        raise RuntimeError("damping dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"kiki_machine_local_topology_budgets"/(authorization+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "retry_allowed": False}, stream)
    from kron_reduce.evaluation.kiki_machine_local_topology_damping import execute

    result = execute(root, result_dir, plan, stage)
    if git_head(root) != commit:
        raise RuntimeError("damping checkout changed during execution")
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
