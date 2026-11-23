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
from project_harness.report_storage import locate_report
from project_harness.runner import _run_experiment

WORKFLOW = "kiki_machine_local_topology_radius_v2"
EXPERIMENT = "kiki_machine_local_topology_radius_test_v2"
AUTHORIZATION = EXPERIMENT + "_20261115"
MODULE = "kron_reduce.workflows.kiki_machine_local_topology_radius"
PRIOR_WORKFLOW = "kiki_machine_local_topology_no_prior_v1"
PRIOR_EXPERIMENT = "kiki_machine_local_topology_no_prior_test_v1"
PRIOR_AUTHORIZATION = PRIOR_EXPERIMENT+"_20261115"
MOMENTUM_WORKFLOW = "kiki_machine_local_topology_momentum_damping_v1"
MOMENTUM_EXPERIMENT = "kiki_machine_local_topology_momentum_damping_test_v1"
MOMENTUM_AUTHORIZATION = MOMENTUM_EXPERIMENT+"_20261115"
ANGLE_WORKFLOW = "kiki_machine_local_topology_angle_damping_v1"
ANGLE_EXPERIMENT = "kiki_machine_local_topology_angle_damping_test_v1"
ANGLE_AUTHORIZATION = ANGLE_EXPERIMENT+"_20261115"
SPATIAL_WORKFLOW = "kiki_machine_local_topology_spatial_damping_v1"
SPATIAL_EXPERIMENT = "kiki_machine_local_topology_spatial_damping_test_v1"
SPATIAL_AUTHORIZATION = SPATIAL_EXPERIMENT+"_20261115"
RESTORING_WORKFLOW = "kiki_machine_local_topology_restoring_shape_v1"
RESTORING_EXPERIMENT = "kiki_machine_local_topology_restoring_shape_test_v1"
RESTORING_AUTHORIZATION = RESTORING_EXPERIMENT+"_20261115"
COMPARISONS = {"R2": (WORKFLOW, EXPERIMENT, AUTHORIZATION),
               "no_prior": (PRIOR_WORKFLOW, PRIOR_EXPERIMENT, PRIOR_AUTHORIZATION),
               "momentum_damping": (MOMENTUM_WORKFLOW, MOMENTUM_EXPERIMENT, MOMENTUM_AUTHORIZATION),
               "angle_damping": (ANGLE_WORKFLOW, ANGLE_EXPERIMENT, ANGLE_AUTHORIZATION),
               "spatial_damping": (SPATIAL_WORKFLOW, SPATIAL_EXPERIMENT, SPATIAL_AUTHORIZATION),
               "restoring_shape": (RESTORING_WORKFLOW, RESTORING_EXPERIMENT, RESTORING_AUTHORIZATION)}


def _bound_json(root, binding):
    verify_bound_file(root, binding["path"], binding["sha256"])
    path = locate_report(root, binding["path"], binding["sha256"])
    return json.loads((root/path).read_bytes())


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


def momentum_damping_freeze(anchor, binding, checkpoint, multiplier):
    frozen = radius_freeze(anchor, binding, checkpoint)
    frozen["memory_radius_multiplier"] = 1.
    frozen["settings"] = anchor["settings"] | {"damping_multiplier": multiplier, "matched_train_witnesses": True}
    return frozen


def angle_damping_freeze(anchor, binding, checkpoint, rho):
    if not 0. <= rho <= 2./3.:
        raise ValueError("angle damping rho must be between zero and two-thirds")
    frozen = radius_freeze(anchor, binding, checkpoint)
    frozen["memory_radius_multiplier"] = 1.
    frozen["settings"] = anchor["settings"] | {"angle_damping_rho": rho, "matched_train_witnesses": True}
    return frozen


def spatial_damping_freeze(anchor, binding, checkpoint, mix):
    if not 0. < mix <= 1.:
        raise ValueError("spatial comparison requires a positive fitted mix no greater than one")
    frozen = angle_damping_freeze(anchor, binding, checkpoint, 2./3.)
    frozen["settings"]["spatial_damping_mix"] = mix
    return frozen


def restoring_shape_freeze(anchor, binding, checkpoint, alpha):
    if not -.25 <= alpha <= .25 or alpha == 0.:
        raise ValueError("restoring comparison requires a nonzero shape coefficient within its fitted bounds")
    frozen = spatial_damping_freeze(anchor, binding, checkpoint, 1.)
    frozen["settings"]["restoring_shape_alpha"] = alpha
    return frozen


def _reference(root, policy, plan, build):
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
    if freeze != build(anchor, binding, checkpoints["T0M1"]) or plan["backbone"] != anchor["backbone"]:
        raise RuntimeError("initialization intervention differs from the original T0M1 freeze")
    for key in ("checkpoint", "preprocessing"):
        _result_file(root, {"path": freeze[f"{key}_path"], "sha256": freeze[f"{key}_sha256"]})


def _execution_plan(plan, source):
    sources = {key: source["sources"][key] for key in ("files", "rosters")}
    return plan | {"sources": sources, "task3_root": source["task3_root"]}


def authorize(root, *, comparison="R2"):
    assert_clean(root)
    workflow, experiment, auth_id = COMPARISONS[comparison]
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
            or plan.get("group", "R2") != comparison
            or plan["operation_limits"] != auth["operation_limits"] or review.get("approved") is not True
            or review.get("plan_path") != auth["plan_path"] or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan["implementation_bindings"]):
        raise RuntimeError("exact radius evaluation implementation and scientific review required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)
    build = radius_freeze
    if comparison == "no_prior":
        synthetic = policy["bounded_synthetic_evaluations"]["machine_local_topology_ball_history_no_prior_v1_20261115"]
        support = plan["radial_zero_prior_support"]
        review_binding = {"path": synthetic["result_review_path"], "sha256": synthetic["result_review_sha256"]}
        if (synthetic["status"] != "consumed" or synthetic["engineering_pass"] is not True
                or support != review_binding or plan["group"] != "no_prior"
                or _bound_json(root, support)["radial_zero_prior_support"]["passed"] is not True):
            raise RuntimeError("verified synthetic evidence for the radial zero-prior control required")
        build = prior_freeze
    if comparison == "momentum_damping":
        diagnostic = policy["development_evaluations"]["kiki_machine_local_topology_momentum_v1_20261115"]
        receipt_binding = {"path": diagnostic["result_receipt_path"], "sha256": diagnostic["result_receipt_sha256"]}
        result_binding = {"path": diagnostic["result_review_path"], "sha256": diagnostic["result_review_sha256"]}
        if (diagnostic["status"] != "consumed" or diagnostic["scientific_pass"] is not True
                or plan["momentum_receipt"] != receipt_binding or plan["momentum_review"] != result_binding
                or _bound_json(root, result_binding)["prospective_evaluation_support"]["supported"] is not True):
            raise RuntimeError("verified train-only momentum direction evidence required")
        directions = _bound_json(root, receipt_binding)["supported_directions"]
        if (len(directions) != 1 or directions[0]["direction"] != "damping"
                or plan["damping_multiplier"] != directions[0]["multiplier"] or plan["damping_multiplier"] <= 0.):
            raise RuntimeError("registered damping factor differs from the sole train direction")

        def build(anchor, binding, checkpoint):
            return momentum_damping_freeze(anchor, binding, checkpoint, plan["damping_multiplier"])

    if comparison == "angle_damping":
        fit = policy["development_evaluations"]["kiki_machine_local_topology_angle_damping_fit_v1_20261115"]
        if fit["status"] != "consumed" or fit["engineering_pass"] is not True:
            raise RuntimeError("qualified complete train angle damping fit required")
        receipt, review = ({"path": fit[f"result_{key}_path"], "sha256": fit[f"result_{key}_sha256"]}
                           for key in ("receipt", "review"))
        if plan["angle_fit_receipt"] != receipt or plan["angle_fit_review"] != review:
            raise RuntimeError("angle damping fit binding differs")
        matched = _bound_json(root, receipt)
        rho = matched["train_fits"]["angle"]
        if (matched["fit_supported"] is not True or matched["allocated"] != 346
                or _bound_json(root, review)["selected_rho"] != rho or plan["angle_damping_rho"] != rho):
            raise RuntimeError("frozen angle damping coefficient differs from full train fit")

        def build(anchor, binding, checkpoint):
            return angle_damping_freeze(anchor, binding, checkpoint, rho)

    if comparison == "spatial_damping":
        fit = policy["development_evaluations"]["kiki_machine_local_topology_spatial_damping_fit_v1_20261115"]
        if fit["status"] != "consumed" or fit["engineering_pass"] is not True:
            raise RuntimeError("qualified complete train spatial damping fit required")
        receipt, review = ({"path": fit[f"result_{key}_path"], "sha256": fit[f"result_{key}_sha256"]}
                           for key in ("receipt", "review"))
        if plan["spatial_fit_receipt"] != receipt or plan["spatial_fit_review"] != review:
            raise RuntimeError("spatial damping fit binding differs")
        matched = _bound_json(root, receipt)
        mix = matched["train_fit"]["value"]
        if (matched["fit_supported"] is not True or matched["allocated"] != 346 or not 0. < mix <= 1.
                or _bound_json(root, review)["selected_mix"] != mix or plan["spatial_damping_mix"] != mix):
            raise RuntimeError("frozen spatial mix differs from complete train fit")

        def build(anchor, binding, checkpoint):
            return spatial_damping_freeze(anchor, binding, checkpoint, mix)

    if comparison == "restoring_shape":
        fit = policy["development_evaluations"]["kiki_machine_local_topology_restoring_shape_fit_v1_20261115"]
        if fit["status"] != "consumed" or fit["engineering_pass"] is not True:
            raise RuntimeError("qualified complete train restoring-shape fit required")
        receipt, review = ({"path": fit[f"result_{key}_path"], "sha256": fit[f"result_{key}_sha256"]}
                           for key in ("receipt", "review"))
        if plan["restoring_fit_receipt"] != receipt or plan["restoring_fit_review"] != review:
            raise RuntimeError("restoring-shape fit binding differs")
        matched = _bound_json(root, receipt)
        signed_alpha = matched["train_fit"]["shape"]["value"]
        if (matched["fit_supported"] is not True or matched["allocated"] != 346
                or _bound_json(root, review)["selected_alpha"] != signed_alpha
                or plan["restoring_shape_alpha"] != signed_alpha):
            raise RuntimeError("frozen restoring coefficient differs from complete train fit")

        def build(anchor, binding, checkpoint):
            return restoring_shape_freeze(anchor, binding, checkpoint, signed_alpha)

    _reference(root, policy, plan, build)
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
    comparison, auth_id = next((group, identity[2]) for group, identity in COMPARISONS.items()
                               if identity[0] == experiment.get("workflow"))
    entry, plan = authorize(root, comparison=comparison)
    if stage != "test" or experiment != entry or result_dir != root/"results"/run_id or any(result_dir.iterdir()):
        raise RuntimeError("TASK3 radius dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"kiki_machine_local_topology_budgets"/(auth_id+".json")
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
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--no-prior", action="store_const", const="no_prior", dest="comparison")
    options.add_argument("--momentum-damping", action="store_const", const="momentum_damping", dest="comparison")
    options.add_argument("--angle-damping", action="store_const", const="angle_damping", dest="comparison")
    options.add_argument("--spatial-damping", action="store_const", const="spatial_damping", dest="comparison")
    options.add_argument("--restoring-shape", action="store_const", const="restoring_shape", dest="comparison")
    comparison = parser.parse_args().comparison or "R2"
    root = project_root()
    workflow, experiment, _ = COMPARISONS[comparison]
    with workflow_lock(root, workflow):
        entry, _ = authorize(root, comparison=comparison)
        print(_run_experiment(root, experiment, "test", experiment=entry))


if __name__ == "__main__":
    main()
