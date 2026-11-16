"""One-use fixed-support TASK3 fit and frozen GPU evaluation."""
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

WORKFLOW = "kiki_machine_local_topology_v1"
SPATIAL_WORKFLOW = "kiki_machine_local_topology_spatial_relaxation_v1"
HARMONIC_WORKFLOW = "kiki_es_sindy_v1"
MODULE = "kron_reduce.workflows.kiki_machine_local_topology"
CALIBRATION = "machine_local_topology_transfer_v1_20261115"


def _spatial_prerequisites(root, plan, policy):
    development = policy["development_evaluations"]
    synthetic = policy["bounded_synthetic_evaluations"]["machine_local_topology_spatial_coupling_causal_v1_20261116"]
    observed = development["kiki_machine_local_topology_spatial_coupling_v1_20261116"]
    for row, key, passed in ((synthetic, "spatial_synthetic", True), (observed, "spatial_observed_path", False)):
        receipt = {"path": row["result_receipt_path"], "sha256": row["result_receipt_sha256"]}
        if (row["status"] != "consumed" or row["engineering_pass"] is not True
                or row["scientific_pass"] is not passed or plan[key] != receipt):
            raise RuntimeError("spatial learning requires the exact saved synthetic and observed-path comparisons")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
    prior = development["kiki_machine_local_topology_fit_v1_20261115"]
    receipt = _bound_json(root, {"path": prior["result_receipt_path"], "sha256": prior["result_receipt_sha256"]})
    start = plan["paired_start"]
    freeze_ref = {key: receipt["frozen_model_manifest"][key] for key in ("path", "sha256")}
    if prior["status"] != "consumed" or start["freeze"] != freeze_ref:
        raise RuntimeError("paired initialization requires the original frozen T0M1")
    if plan["stage"] == "fit":
        freeze = json.loads(_result_file(root, freeze_ref).read_bytes())
        checkpoint_ref = {"path": freeze["checkpoint_path"], "sha256": freeze["checkpoint_sha256"]}
        checkpoint = json.loads(_result_file(root, checkpoint_ref).read_bytes())
        preprocessing = {"path": freeze["preprocessing_path"], "sha256": freeze["preprocessing_sha256"]}
        if (start["checkpoint"] != checkpoint["T0M1"] or freeze["backbone"] != plan["backbone"]
                or start["preprocessing"] != preprocessing
                or freeze["training"]["groups"]["T0M1"]["selected_update"] != 12):
            raise RuntimeError("both fits require the exact shared T0M1 anchor at update 12")
        _result_file(root, checkpoint["T0M1"])
        _result_file(root, preprocessing)
    interface = _bound_json(root, plan["interface_check"])
    if (interface["passed"] is not True or interface["real_data_reads"] != 0
            or interface["synthetic_fixture_batch"] != plan["settings"]["train_batch"]):
        raise RuntimeError("the differentiable spatial coupling fixture must pass before training")


def authorize(root, stage, *, workflow=WORKFLOW):
    assert_clean(root)
    spatial = workflow == SPATIAL_WORKFLOW
    harmonic = workflow == HARMONIC_WORKFLOW
    series = workflow.removesuffix("_v1")
    name = f"{series}_{stage}_v1"
    entry = load_manifest(root)["experiments"][name]
    if (sys.version_info[:2] != (3, 12) or stage not in ("fit", "test")
            or entry["workflow"] != workflow or entry["stage"] != stage or entry["module"] != MODULE
            or entry["execution_mode"] != "bounded_kiki_official_cli_only"
            or not approval_recorded(load_state(root, workflow), stage)):
        raise RuntimeError("registered TASK3 topology stage and explicit stage approval required")
    validate_run_policy(root, name, entry)
    policy = load_policy(root)
    suffix = "_20261117" if harmonic else "_20261116" if spatial else "_20261115"
    authorization = (entry["development_authorization_id"] if harmonic and stage == "fit" else name + suffix)
    category = "development_evaluations" if stage == "fit" else "official_test_evaluations"
    auth = policy[category][authorization]
    scope = {"status": "active", "experiment": name, "fit_attempts": (1 if harmonic else 2) if stage == "fit" else 0,
             "max_executions": 1, "retry_allowed": False}
    if any(type(auth.get(key)) is not type(value) or auth[key] != value for key, value in scope.items()):
        raise RuntimeError("active single-use TASK3 scope differs")
    for field in ("plan", "review"):
        verify_bound_file(root, auth[f"{field}_path"], auth[f"{field}_sha256"])
    plan, review = (json.loads((root / auth[f"{field}_path"]).read_text(encoding="utf-8"))
                    for field in ("plan", "review"))
    if (plan["stage"] != stage or plan["device"] != "cuda" or auth["operation_limits"] != plan["operation_limits"]
            or review.get("approved") is not True or review.get("plan_sha256") != auth["plan_sha256"]
            or review.get("implementation_bindings") != plan.get("implementation_bindings")):
        raise RuntimeError("exact TASK3 topology implementation and budget review required")
    for path, digest in plan["implementation_bindings"].items():
        verify_bound_file(root, path, digest)

    if spatial:
        _spatial_prerequisites(root, plan, policy)

    if harmonic:
        synthetic = policy["bounded_synthetic_evaluations"]["es_sindy_synthetic_v1_20261117"]
        receipt = {"path": synthetic["result_receipt_path"], "sha256": synthetic["result_receipt_sha256"]}
        if (synthetic["status"] != "consumed" or synthetic["scientific_pass"] is not True
                or plan["harmonic_synthetic"] != receipt or plan["operation_limits"]["fits"] != scope["fit_attempts"]):
            raise RuntimeError("harmonic training requires the exact completed synthetic gate and fit scope")
        verify_bound_file(root, receipt["path"], receipt["sha256"])
        start = plan["harmonic_start"]
        if stage == "fit":
            anchor = json.loads(_result_file(root, start["freeze"]).read_bytes())
            checkpoint = {"path": anchor["checkpoint_path"], "sha256": anchor["checkpoint_sha256"]}
            index = json.loads(_result_file(root, checkpoint).read_bytes())
            if (index["T0M1"] != start["checkpoint"] or anchor["backbone"] != plan["backbone"]
                    or anchor["training"]["groups"]["T0M1"]["selected_update"] != 12
                    or start["preprocessing"] != {"path": anchor["preprocessing_path"], "sha256": anchor["preprocessing_sha256"]}):
                raise RuntimeError("harmonic training requires the frozen T0M1 anchor and preprocessing")
            _result_file(root, start["checkpoint"])
            _result_file(root, start["preprocessing"])

    transfer = policy["bounded_synthetic_evaluations"][CALIBRATION]
    for key in ("run_id", "result_manifest_sha256", "result_receipt_sha256", "result_review_sha256"):
        if transfer[key] != plan["transfer"][key]:
            raise RuntimeError("registered synthetic transfer gate differs")
    if transfer["status"] != "consumed" or transfer.get("scientific_pass") is not True:
        raise RuntimeError("completed synthetic transfer gate must pass")
    for key in ("result_receipt", "result_review"):
        verify_bound_file(root, transfer[key + "_path"], transfer[key + "_sha256"])

    source = plan["source_contract"]
    verify_bound_file(root, source["path"], source["sha256"])
    source_plan = json.loads((root / source["path"]).read_text(encoding="utf-8"))
    if plan["backbone"] != source_plan["backbone"]:
        raise RuntimeError("starting backbone differs from the registered TASK3 source")
    if stage == "test":
        fit_name = f"{series}_fit_v1"
        fit_id = (load_manifest(root)["experiments"][fit_name]["development_authorization_id"]
                  if harmonic else fit_name + suffix)
        fit_auth = policy["development_evaluations"][fit_id]
        if fit_auth["status"] != "consumed" or plan["fit_plan_sha256"] != fit_auth["plan_sha256"]:
            raise RuntimeError("test requires the completed paired fit")
        verify_bound_file(root, fit_auth["plan_path"], fit_auth["plan_sha256"])
        fit_plan = json.loads((root / fit_auth["plan_path"]).read_text(encoding="utf-8"))
        if plan["settings"] != fit_plan["settings"]:
            raise RuntimeError("test settings differ from the frozen fit plan")
        if spatial:
            receipt = {"path": fit_auth["result_receipt_path"], "sha256": fit_auth["result_receipt_sha256"]}
            saved = _bound_json(root, receipt)["frozen_model_manifest"]
            review_ref = {"path": fit_auth["result_review_path"], "sha256": fit_auth["result_review_sha256"]}
            fit_review = _bound_json(root, review_ref)
            if (plan["paired_start"] != fit_plan["paired_start"]
                    or any(value != fit_auth[key] for key, value in plan["fit_result"].items())
                    or fit_review["verified"] is not True or fit_review["run_id"] != fit_auth["run_id"]
                    or saved["path"] != plan["frozen_model_manifest"]["path"]
                    or saved["sha256"] != plan["frozen_model_manifest"]["sha256"]):
                raise RuntimeError("test requires the verified receipt and exact spatial fit freeze")
        if plan["frozen_model_manifest"] != auth["frozen_model_manifest"]:
            raise RuntimeError("test must use the frozen fit artifact")
        binding = auth["frozen_model_manifest"]
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise RuntimeError("frozen model manifest is missing or changed")
    for path, digest in source_plan["external_code"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("TASK3 scoring source changed")
    sources = source_plan["sources"]
    if harmonic:
        from kron_reduce.data.task3_full import REFERENCE, source_plan as current_source_plan

        reference = sources["files"]["reference"] | {"path": str((root / REFERENCE).resolve())}
        sources = {key: value for key, value in sources.items() if key != "allowed_roles"}
        sources["files"] = sources["files"] | {"reference": reference}
        if sources != current_source_plan(root):
            raise RuntimeError("TASK3 reference rebinding changed the frozen source contract")
    plan["sources"] = sources | {"allowed_roles": ["train", "val"] if stage == "fit" else ["test"]}
    plan["task3_root"] = source_plan["task3_root"]
    pins = dict(line.split("==") for line in (root / "requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if any(version(package) != pins[package] + ("+cu130" if package == "torch" else "")
           for package in ("numpy", "scipy", "torch", "filelock", "h5py")):
        raise RuntimeError("registered Python 3.12 and Torch CUDA runtime required")
    _cuda_preflight()
    return name, entry, plan, authorization


def run(*, root, experiment, stage, result_dir, run_id):
    _, entry, plan, authorization = authorize(root, stage, workflow=experiment["workflow"])
    if experiment != entry or result_dir != root / "results" / run_id or any(result_dir.iterdir()):
        raise RuntimeError("TASK3 topology dispatch identity differs")
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common / "kiki_machine_local_topology_budgets" / (authorization + ".json")
    marker.parent.mkdir(exist_ok=True)
    commit = git_head(root)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "retry_allowed": False}, stream)
    from kron_reduce.evaluation.kiki_machine_local_topology import execute

    result = execute(root, result_dir, plan, stage)
    if git_head(root) != commit:
        raise RuntimeError("TASK3 topology source changed during execution")
    verify_artifact_directory(result_dir, strict=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "test"), required=True)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--spatial-relaxation", action="store_true")
    variant.add_argument("--es-sindy", action="store_true")
    args = parser.parse_args()
    stage = args.stage
    workflow = HARMONIC_WORKFLOW if args.es_sindy else SPATIAL_WORKFLOW if args.spatial_relaxation else WORKFLOW
    root = project_root()
    with workflow_lock(root, workflow):
        name, entry, _, _ = authorize(root, stage, workflow=workflow)
        print(_run_experiment(root, name, stage, experiment=entry))


if __name__ == "__main__":
    main()
