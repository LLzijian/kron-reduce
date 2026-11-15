"""One-run observed-path momentum diagnosis on the official complete TASK3 train."""
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

NAME = "kiki_machine_local_topology_momentum_v1"
STAGE = "train_momentum"
MODULE = "kron_reduce.workflows.kiki_topology_momentum"
ORIGIN_TRANSFER = "kiki_machine_local_topology_origin_transfer_v1"
ANGLE_FIT = "kiki_machine_local_topology_angle_damping_fit_v1"
ANGLE_SCALE = "kiki_machine_local_topology_angle_scale_v1"
SPATIAL_FIT = "kiki_machine_local_topology_spatial_damping_fit_v1"
MODAL = "kiki_machine_local_topology_modal_damping_v1"
RESTORING = "kiki_machine_local_topology_restoring_shape_fit_v1"
DYNAMIC = "kiki_machine_local_topology_dynamic_coupling_v1"
GPU_STUDIES = (ANGLE_SCALE, MODAL, DYNAMIC)
CONTRASTS = {NAME: "force_blocks", ORIGIN_TRANSFER: "origin_transfer", ANGLE_FIT: "angle_damping_fit",
             ANGLE_SCALE: "angle_scale", SPATIAL_FIT: "spatial_damping_fit", MODAL: "modal_damping",
             RESTORING: "restoring_shape_fit", DYNAMIC: "dynamic_coupling"}


def _authority(name):
    return name+("_20261116" if name == DYNAMIC else "_20261115")


def authorize(root, name=NAME):
    assert_clean(root)
    contrast = CONTRASTS[name]
    authority = _authority(name)
    entry = load_manifest(root)["experiments"][name]
    scope = {"module": MODULE, "workflow": name, "stage": STAGE, "execution_mode": "bounded_kiki_official_cli_only",
             "development_authorization_id": authority, "reads_validation_performance": False, "reads_test_performance": False}
    if (sys.version_info[:2] != (3, 12) or any(entry.get(k) != v for k,v in scope.items())
            or not approval_recorded(load_state(root, name), STAGE)):
        raise RuntimeError("registered train-only momentum stage and explicit loop approval required")
    validate_run_policy(root, name, entry)
    policy = load_policy(root)
    auth = policy["development_evaluations"][authority]
    expected = {"status": "active", "experiment": name,
                "fit_attempts": {ANGLE_FIT: 2, SPATIAL_FIT: 1, RESTORING: 2}.get(name, 0), "max_executions": 1,
                "retry_allowed": False, "train_cases": 346, "source_cache_open_limit": 2}
    if any(type(auth.get(k)) is not type(v) or auth[k] != v for k,v in expected.items()):
        raise RuntimeError("one-run train-only momentum authority differs")
    plan, review = (_bound_json(root, {"path": auth[f"{key}_path"], "sha256": auth[f"{key}_sha256"]})
                    for key in ("plan","review"))
    if (plan["stage"] != STAGE or plan["cohort_counts"] != {"train":346} or plan["source_cache_open_limit"] != 2
            or plan.get("contrast", "force_blocks") != contrast
            or plan["operation_limits"] != auth["operation_limits"] or review.get("approved") is not True
            or review.get("plan_path") != auth["plan_path"] or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan["implementation_bindings"]):
        raise RuntimeError("exact train-only plan and review required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)
    known_truth = {ANGLE_FIT: "angle_damping", ANGLE_SCALE: "angle_damping", SPATIAL_FIT: "spatial_damping",
                   MODAL: "spatial_damping", RESTORING: "restoring_shape"}
    synthetic = ("machine_local_topology_dynamic_coupling_v1_20261116" if name == DYNAMIC
                 else f"machine_local_topology_{known_truth.get(name, 'momentum')}_v1_20261115")
    truth = policy["bounded_synthetic_evaluations"][synthetic]
    if (truth["status"] != "consumed" or truth["scientific_pass"] is not True
            or plan["synthetic_receipt"] != {"path":truth["result_receipt_path"],"sha256":truth["result_receipt_sha256"]}):
        raise RuntimeError("known-truth momentum gate required")
    synthetic_result = _bound_json(root, plan["synthetic_receipt"])
    if name == DYNAMIC and (plan["coupling"] != {"rate": 1., "limit": .34657359027997265}
                            or synthetic_result["populations"]["dynamic"]["fit"]["dynamic"]["value"] != 1.):
        raise RuntimeError("fixed observed-path coupling direction must match the known-truth rate")

    anchor = policy["development_evaluations"]["kiki_machine_local_topology_fit_v1_20261115"]
    if anchor["status"] != "consumed":
        raise RuntimeError("original T0M1 fit must be closed")
    receipt = _bound_json(root,{"path":anchor["result_receipt_path"],"sha256":anchor["result_receipt_sha256"]})
    binding = {k:receipt["frozen_model_manifest"][k] for k in ("path","sha256")}
    if plan["anchor_freeze"] != binding:
        raise RuntimeError("original T0M1 freeze differs")
    freeze = json.loads(_result_file(root,binding).read_bytes())
    checkpoint, prep = (_result_file(root, {"path":freeze[k+"_path"],"sha256":freeze[k+"_sha256"]})
                        for k in ("checkpoint","preprocessing"))
    preprocessing = json.loads(prep.read_bytes())
    if (freeze["phase"] != "freeze" or freeze["fitting_closed"] is not True
            or freeze["training"]["groups"]["T0M1"]["selected_update"] != 12
            or json.loads(checkpoint.read_bytes())["T0M1"] != plan["checkpoint"]
            or preprocessing["memory_radius"] != plan["memory_radius"]):
        raise RuntimeError("original model, preprocessing or radius differs")
    _result_file(root,plan["checkpoint"])
    source = _bound_json(root,plan["source_contract"])
    if source["backbone"] != freeze["backbone"] or source["backbone"] != plan["backbone"]:
        raise RuntimeError("frozen model backbone differs")
    _result_file(root,plan["backbone"])
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    variant = "+cu130" if name in GPU_STUDIES else "+cpu"
    if any(version(p) != pins[p]+(variant if p=="torch" else "") for p in ("torch","numpy","scipy","filelock","h5py")):
        raise RuntimeError("registered train diagnostic runtime required")
    if name in (*GPU_STUDIES, RESTORING):
        group = "angle_damping" if name == ANGLE_SCALE else "spatial_damping"
        prior = policy["official_test_evaluations"][f"kiki_machine_local_topology_{group}_test_v1_20261115"]
        fit = policy["development_evaluations"][f"kiki_machine_local_topology_{group}_fit_v1_20261115"]
        audit = policy["saved_evidence_audits"][f"kiki_machine_local_topology_{group}_saved_audit_v1_20261115"]
        record = {"path": prior["result_receipt_path"], "sha256": prior["result_receipt_sha256"]}
        audited = {"path": audit["result_receipt_path"], "sha256": audit["result_receipt_sha256"]}
        anchor_key = "anchor_angle_receipt" if name == ANGLE_SCALE else "anchor_spatial_receipt"
        parameter, required = ("angle_damping_rho", 2./3.) if name == ANGLE_SCALE else ("spatial_damping_mix", 1.)
        if (any(row["status"] != "consumed" for row in (prior, fit, audit)) or audit["mismatches"] != 0
                or _bound_json(root, audited)["run_id"] != prior["run_id"] or plan[anchor_key] != record
                or _bound_json(root, record)[parameter] != required):
            raise RuntimeError("closed damping comparison and independent scores required")
        if name in GPU_STUDIES:
            _cuda_preflight()
    return entry, plan | {"sources":source["sources"] | {"allowed_roles":["train"]},
                          "history_scale": preprocessing["history"]}


def run(*, root, experiment, stage, result_dir, run_id):
    name = experiment["workflow"]
    entry, plan = authorize(root, name)
    if (entry != experiment or stage != STAGE or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("train momentum dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"kiki_machine_local_topology_budgets"/(_authority(name)+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id":run_id,"source_commit":commit,"retry_allowed":False},stream)

    import torch

    from kron_reduce.evaluation.constant_c_observability import write_json
    from kron_reduce.evaluation.kiki_topology_momentum import execute
    from kron_reduce.evaluation.machine_local_synthetic import Budget

    torch.set_num_threads(1)
    if name in GPU_STUDIES:
        torch.use_deterministic_algorithms(True)
    budget = Budget(result_dir/"operations.jsonl", plan["operation_limits"])
    try:
        result = execute(root,result_dir,plan,budget)
    finally:
        budget.stream.close()
    evidence = {"status":"PASS","process_status":"PASS","source_commit":commit,"data_policy":"kiki_official_train_test_v1",
                "promotion_decision":"KEEP_BASELINE","scientific_pass":result["scientific_pass"],
                "claim_boundary":"Observed-path diagnosis over the complete official TASK3 train."}
    runtime = {"python":sys.version, "torch":version("torch"), "numpy":version("numpy"), "scipy":version("scipy")}
    if name in GPU_STUDIES:
        runtime.update(executable=sys.executable, cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name())
    write_json(result_dir/"summary.json", result | evidence | {"operation_counts":dict(budget.counts), "runtime":runtime})
    write_json(result_dir/"stage_evidence.json", evidence)
    write_json(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix():sha256_file(p)
               for p in sorted(result_dir.rglob("*")) if p.is_file()})
    if git_head(root) != commit:
        raise RuntimeError("train momentum source changed during execution")
    verify_artifact_directory(result_dir,strict=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--origin-transfer", action="store_const", const=ORIGIN_TRANSFER, dest="study")
    options.add_argument("--angle-damping-fit", action="store_const", const=ANGLE_FIT, dest="study")
    options.add_argument("--angle-scale", action="store_const", const=ANGLE_SCALE, dest="study")
    options.add_argument("--spatial-damping-fit", action="store_const", const=SPATIAL_FIT, dest="study")
    options.add_argument("--modal-damping", action="store_const", const=MODAL, dest="study")
    options.add_argument("--restoring-shape-fit", action="store_const", const=RESTORING, dest="study")
    options.add_argument("--dynamic-coupling", action="store_const", const=DYNAMIC, dest="study")
    name = parser.parse_args().study or NAME
    root = project_root()
    with workflow_lock(root,name):
        entry, _ = authorize(root, name)
        print(_run_experiment(root,name,STAGE,experiment=entry))


if __name__ == "__main__":
    main()
