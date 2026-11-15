"""One-use frozen TASK3 comparison of the memory initialization bound."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from kron_reduce.workflows.machine_local_equation import _cuda_preflight
from project_harness.evidence import _safe_relative, sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "kiki_machine_local_topology_radius_v2"
EXPERIMENT = "kiki_machine_local_topology_radius_test_v2"
AUTHORIZATION = EXPERIMENT + "_20261115"
MODULE = "kron_reduce.workflows.kiki_machine_local_topology_radius"
PRIOR_WORKFLOW = "kiki_machine_local_topology_no_prior_v1"
PRIOR_EXPERIMENT = "kiki_machine_local_topology_no_prior_test_v1"
PRIOR_AUTHORIZATION = PRIOR_EXPERIMENT+"_20261115"


def _bound_json(root, binding):
    verify_bound_file(root, binding["path"], binding["sha256"])
    return json.loads((root/binding["path"]).read_bytes())


def _result_file(root, binding):
    # Run artifacts live outside Git and are pinned by their archived digests.
    path = _safe_relative(root, binding["path"])
    if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root/"results")
            or sha256_file(path) != binding["sha256"]):
        raise RuntimeError(f"frozen result artifact differs: {binding['path']}")
    return path


def radius_freeze(anchor, binding, checkpoint):
    if (anchor["input_contract"] != "measured_history_100ms_zoh_v1"
            or anchor["phase"] != "freeze" or anchor["selection_closed"] is not True
            or anchor["fitting_closed"] is not True
            or anchor["training"]["groups"]["T0M1"]["selected_update"] != 12
            or anchor["settings"]["prior_weight"] != 1e-6):
        raise RuntimeError("radius comparison requires the exact selected T0M1 anchor")
    return {"phase": "freeze", "fitting_closed": True, "selection_closed": True,
            "input_contract": anchor["input_contract"], "anchor_fit_manifest": binding,
            "checkpoint_path": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"],
            "preprocessing_path": anchor["preprocessing_path"],
            "preprocessing_sha256": anchor["preprocessing_sha256"],
            "settings": anchor["settings"] | {"prior_weight": 4e-6}, "memory_radius_multiplier": 2.}


def prior_freeze(anchor, binding, checkpoint):
    frozen = radius_freeze(anchor, binding, checkpoint)
    frozen["memory_radius_multiplier"] = 1.
    frozen["settings"] = anchor["settings"] | {"prior_weight": 0., "matched_train_witnesses": True}
    return frozen


def _reference(root, policy, plan, *, no_prior=False):
    receipts, references = {}, {}
    for stage, roles, category in (("fit", ("train", "val"), "development_evaluations"),
                                   ("test", ("test",), "official_test_evaluations")):
        original = policy[category][f"kiki_machine_local_topology_{stage}_v1_20261115"]
        if original["status"] != "consumed":
            raise RuntimeError("radius comparison requires consumed full-cohort reference stages")
        receipt = _bound_json(root, {"path": original["result_receipt_path"],
                                     "sha256": original["result_receipt_sha256"]})
        receipts[stage] = receipt
        for role in roles:
            references[role] = {"path": f"results/{receipt['run_id']}/T0M1/{role}.json",
                                "sha256": receipt["records"]["T0M1"][role]["sha256"]}
    if plan["reference"] != references:
        raise RuntimeError("radius comparison reference differs from the saved full-cohort records")
    for binding in references.values():
        _result_file(root, binding)
    binding = {key: receipts["fit"]["frozen_model_manifest"][key] for key in ("path", "sha256")}
    if any(receipts["test"]["frozen_model_manifest"][key] != binding[key] for key in binding):
        raise RuntimeError("test and development reference models differ")
    audit = policy["saved_evidence_audits"]["kiki_machine_local_topology_saved_audit_v1_20261115"]
    if audit["status"] != "consumed" or audit["mismatches"] != 0:
        raise RuntimeError("radius comparison requires the completed paired arithmetic audit")
    recomputed = _bound_json(root, {"path": audit["result_receipt_path"], "sha256": audit["result_receipt_sha256"]})
    if any(recomputed[f"{stage}_run_id"] != receipts[stage]["run_id"] for stage in receipts):
        raise RuntimeError("saved score audit bound different reference runs")
    anchor = json.loads(_result_file(root, binding).read_bytes())
    index = _result_file(root, {"path": anchor["checkpoint_path"], "sha256": anchor["checkpoint_sha256"]})
    checkpoints = json.loads(index.read_bytes())
    freeze = _bound_json(root, plan["frozen_model_manifest"])
    build = prior_freeze if no_prior else radius_freeze
    if freeze != build(anchor, binding, checkpoints["T0M1"]) or plan["backbone"] != anchor["backbone"]:
        raise RuntimeError("initialization intervention differs from the original T0M1 freeze")
    for key in ("checkpoint", "preprocessing"):
        _result_file(root, {"path": freeze[f"{key}_path"], "sha256": freeze[f"{key}_sha256"]})


def _execution_plan(plan, source):
    sources = {key: source["sources"][key] for key in ("files", "rosters")}
    return plan | {"sources": sources, "task3_root": source["task3_root"]}


def authorize(root, *, no_prior=False):
    assert_clean(root)
    workflow, experiment, auth_id = ((PRIOR_WORKFLOW, PRIOR_EXPERIMENT, PRIOR_AUTHORIZATION) if no_prior
                                     else (WORKFLOW, EXPERIMENT, AUTHORIZATION))
    entry = load_manifest(root)["experiments"][experiment]
    scope = {"workflow": workflow, "stage": "test", "module": MODULE,
             "execution_mode": "bounded_kiki_official_cli_only", "test_authorization": auth_id}
    if (sys.version_info[:2] != (3, 12) or any(entry.get(key) != val for key, val in scope.items())
            or not approval_recorded(load_state(root, workflow), "test")):
        raise RuntimeError("registered TASK3 initialization stage and explicit test approval required")
    validate_run_policy(root, experiment, entry)
    policy = load_policy(root)
    auth = policy["official_test_evaluations"][auth_id]
    plan, review = (_bound_json(root, {"path": auth[f"{key}_path"], "sha256": auth[f"{key}_sha256"]})
                    for key in ("plan", "review"))
    if (plan["stage"] != "test" or plan["device"] != "cuda" or plan["fit_attempts"] != 0
            or plan["cohort_counts"] != {"train": 346, "val": 107, "test": 76}
            or plan["source_cache_open_limit"] != 80 or plan["frozen_model_manifest"] != auth["frozen_model_manifest"]
            or plan["operation_limits"] != auth["operation_limits"] or review.get("approved") is not True
            or review.get("plan_path") != auth["plan_path"] or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan["implementation_bindings"]):
        raise RuntimeError("exact radius evaluation implementation and scientific review required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)
    if no_prior:
        synthetic = policy["bounded_synthetic_evaluations"]["machine_local_topology_ball_history_no_prior_v1_20261115"]
        support = plan["radial_zero_prior_support"]
        review_binding = {"path": synthetic["result_review_path"], "sha256": synthetic["result_review_sha256"]}
        if (synthetic["status"] != "consumed" or synthetic["engineering_pass"] is not True
                or support != review_binding or plan["group"] != "no_prior"
                or _bound_json(root, support)["radial_zero_prior_support"]["passed"] is not True):
            raise RuntimeError("verified synthetic evidence for the radial zero-prior control required")
    _reference(root, policy, plan, no_prior=no_prior)
    source = _bound_json(root, plan["source_contract"])
    if plan["backbone"] != source["backbone"]:
        raise RuntimeError("radius evaluation backbone differs from the registered source")
    for path, digest in source["external_code"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("TASK3 scoring source changed")
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if any(version(p) != pins[p]+("+cu130" if p == "torch" else "")
           for p in ("numpy", "scipy", "torch", "filelock", "h5py")):
        raise RuntimeError("registered Python 3.12 and Torch CUDA runtime required")
    _cuda_preflight()
    return entry, _execution_plan(plan, source)


def run(*, root, experiment, stage, result_dir, run_id):
    no_prior = experiment.get("workflow") == PRIOR_WORKFLOW
    entry, plan = authorize(root, no_prior=no_prior)
    if stage != "test" or experiment != entry or result_dir != root/"results"/run_id or any(result_dir.iterdir()):
        raise RuntimeError("TASK3 radius dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"kiki_machine_local_topology_budgets"/((PRIOR_AUTHORIZATION if no_prior else AUTHORIZATION)+".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "retry_allowed": False}, stream)
    from kron_reduce.evaluation.kiki_machine_local_topology_radius import execute

    result = execute(root, result_dir, plan, stage)
    if git_head(root) != commit:
        raise RuntimeError("radius evaluation source changed during execution")
    verify_artifact_directory(result_dir, strict=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-prior", action="store_true")
    no_prior = parser.parse_args().no_prior
    root = project_root()
    workflow, experiment = (PRIOR_WORKFLOW, PRIOR_EXPERIMENT) if no_prior else (WORKFLOW, EXPERIMENT)
    with workflow_lock(root, workflow):
        entry, _ = authorize(root, no_prior=no_prior)
        print(_run_experiment(root, experiment, "test", experiment=entry))


if __name__ == "__main__":
    main()
