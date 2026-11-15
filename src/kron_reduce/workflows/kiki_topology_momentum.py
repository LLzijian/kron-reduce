"""One-run observed-path momentum diagnosis on the official complete TASK3 train."""
from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from kron_reduce.workflows.kiki_machine_local_topology_radius import _bound_json, _result_file
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

NAME = "kiki_machine_local_topology_momentum_v1"
STAGE = "train_momentum"
AUTH = NAME+"_20261115"
MODULE = "kron_reduce.workflows.kiki_topology_momentum"


def authorize(root):
    assert_clean(root)
    entry = load_manifest(root)["experiments"][NAME]
    scope = {"module": MODULE, "workflow": NAME, "stage": STAGE, "execution_mode": "bounded_kiki_official_cli_only",
             "development_authorization_id": AUTH, "reads_validation_performance": False, "reads_test_performance": False}
    if (sys.version_info[:2] != (3, 12) or any(entry.get(k) != v for k,v in scope.items())
            or not approval_recorded(load_state(root, NAME), STAGE)):
        raise RuntimeError("registered train-only momentum stage and explicit loop approval required")
    validate_run_policy(root, NAME, entry)
    policy = load_policy(root)
    auth = policy["development_evaluations"][AUTH]
    expected = {"status": "active", "experiment": NAME, "fit_attempts": 0, "max_executions": 1,
                "retry_allowed": False, "train_cases": 346, "source_cache_open_limit": 2}
    if any(type(auth.get(k)) is not type(v) or auth[k] != v for k,v in expected.items()):
        raise RuntimeError("one-run train-only momentum authority differs")
    plan, review = (_bound_json(root, {"path": auth[f"{key}_path"], "sha256": auth[f"{key}_sha256"]})
                    for key in ("plan","review"))
    if (plan["stage"] != STAGE or plan["cohort_counts"] != {"train":346} or plan["source_cache_open_limit"] != 2
            or plan["operation_limits"] != auth["operation_limits"] or review.get("approved") is not True
            or review.get("plan_path") != auth["plan_path"] or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan["implementation_bindings"]):
        raise RuntimeError("exact train-only plan and review required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)
    truth = policy["bounded_synthetic_evaluations"]["machine_local_topology_momentum_v1_20261115"]
    if (truth["status"] != "consumed" or truth["scientific_pass"] is not True
            or plan["synthetic_receipt"] != {"path":truth["result_receipt_path"],"sha256":truth["result_receipt_sha256"]}):
        raise RuntimeError("known-truth momentum gate required")
    _bound_json(root, plan["synthetic_receipt"])

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
    if (freeze["phase"] != "freeze" or freeze["fitting_closed"] is not True
            or freeze["training"]["groups"]["T0M1"]["selected_update"] != 12
            or json.loads(checkpoint.read_bytes())["T0M1"] != plan["checkpoint"]
            or json.loads(prep.read_bytes())["memory_radius"] != plan["memory_radius"]):
        raise RuntimeError("original model, preprocessing or radius differs")
    _result_file(root,plan["checkpoint"])
    source = _bound_json(root,plan["source_contract"])
    if source["backbone"] != freeze["backbone"] or source["backbone"] != plan["backbone"]:
        raise RuntimeError("frozen model backbone differs")
    _result_file(root,plan["backbone"])
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if any(version(p) != pins[p]+("+cpu" if p=="torch" else "") for p in ("torch","numpy","scipy","filelock","h5py")):
        raise RuntimeError("registered CPU runtime required")
    return entry, plan | {"sources":source["sources"] | {"allowed_roles":["train"]}}


def run(*, root, experiment, stage, result_dir, run_id):
    entry, plan = authorize(root)
    if (entry != experiment or stage != STAGE or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("train momentum dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"kiki_machine_local_topology_budgets"/(AUTH+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id":run_id,"source_commit":commit,"retry_allowed":False},stream)

    import torch

    from kron_reduce.evaluation.constant_c_observability import write_json
    from kron_reduce.evaluation.kiki_topology_momentum import execute
    from kron_reduce.evaluation.machine_local_synthetic import Budget

    torch.set_num_threads(1)
    budget = Budget(result_dir/"operations.jsonl", plan["operation_limits"])
    try:
        result = execute(root,result_dir,plan,budget)
    finally:
        budget.stream.close()
    evidence = {"status":"PASS","process_status":"PASS","source_commit":commit,"data_policy":"kiki_official_train_test_v1",
                "promotion_decision":"KEEP_BASELINE","scientific_pass":result["scientific_pass"],
                "claim_boundary":"Observed-path diagnosis over the complete official TASK3 train."}
    write_json(result_dir/"summary.json", result | evidence | {"operation_counts":dict(budget.counts),
        "runtime":{"python":sys.version,"torch":version("torch"),"numpy":version("numpy"),"scipy":version("scipy")}})
    write_json(result_dir/"stage_evidence.json", evidence)
    write_json(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix():sha256_file(p)
               for p in sorted(result_dir.rglob("*")) if p.is_file()})
    if git_head(root) != commit:
        raise RuntimeError("train momentum source changed during execution")
    verify_artifact_directory(result_dir,strict=True)
    return result


def main():
    root = project_root()
    with workflow_lock(root,NAME):
        entry, _ = authorize(root)
        print(_run_experiment(root,NAME,STAGE,experiment=entry))


if __name__ == "__main__":
    main()
