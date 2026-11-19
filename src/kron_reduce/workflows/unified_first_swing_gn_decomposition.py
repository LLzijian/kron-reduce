"""Registered one-use GN prior and radius comparison on the frozen M1 equation."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.workflows.machine_local_equation import _cuda_preflight
from kron_reduce.workflows.unified_first_swing import _archive_asset
from kron_reduce.workflows.unified_first_swing import _plan as _v1_plan
from kron_reduce.workflows.unified_first_swing_probes import _plan as _v2_plan
from project_harness.evidence import _safe_relative, sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "unified_first_swing_gn_decomposition_v3"
FIXTURE = "unified_first_swing_gn_decomposition_fixture_v3"
AUTHORIZATION = f"{WORKFLOW}_20261119"
CONFIG = f"configs/experiments/{WORKFLOW}.toml"


def _plan(root):
    path = root/CONFIG
    verify_bound_file(root, CONFIG, sha256_file(path))
    plan = tomllib.loads(path.read_text(encoding="utf-8"))
    for name, digest in plan["bindings"].items():
        verify_bound_file(root, name, digest)
    return plan


def _synthetic(root, directory, run_id, commit):
    temporary = root/"scratch"/"one_off"/run_id/"pytest"
    temporary.parent.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--basetemp", str(temporary), "tests/test_first_swing_gn_grid.py",
               "tests/test_first_swing_probes.py", "tests/test_unified_first_swing.py"]
    child = subprocess.run(command, cwd=root, capture_output=True, check=False)
    (directory/"pytest.log").write_bytes(child.stdout+child.stderr)
    if child.returncode:
        raise RuntimeError("GN comparison fixtures failed; see pytest.log")
    write_json(directory/"stage_evidence.json", {"status": "PASS", "source_commit": commit,
        "plan_sha256": sha256_file(root/CONFIG), "real_data_reads": 0, "torch": version("torch"),
        "claim_boundary": "Known-truth fixed and soft GN, gating and forecast interface fixtures."})


def _authorize(root, experiment, plan):
    validate_run_policy(root, WORKFLOW, experiment)
    state = load_state(root, WORKFLOW)
    authorizations = load_policy(root)["development_evaluations"]
    scope = authorizations.get(AUTHORIZATION, {})
    expected = {"status": "active", "experiment": WORKFLOW, "allowed_roles": ["train", "val"],
                "train_cases": 346, "val_cases": 107, "test_cases": 0, "fit_attempts": 0,
                "source_cache_open_limit": 3, "max_executions": 1, "retry_allowed": False,
                "plan_path": CONFIG, "plan_sha256": sha256_file(root/CONFIG),
                "operation_limits": plan["operation_limits"],
                "split_reference_sha256": experiment["split_reference_sha256"]}
    if (experiment["development_authorization_id"] != AUTHORIZATION or plan["development_authorization_id"] != AUTHORIZATION
            or not approval_recorded(state, "probe") or state["current_stage"] != "synthetic"
            or any(type(scope.get(k)) is not type(v) or scope[k] != v for k, v in expected.items())):
        raise PermissionError("the exact single-use train/val scope and passed synthetic stage are required")
    fixture = next(row["evidence"]["artifact_manifest"] for row in state["history"] if row.get("to") == "synthetic")
    directory = _safe_relative(root, f"results/{fixture['run_id']}")
    verify_artifact_directory(directory, strict=True, expected_manifest_sha256=fixture["sha256"])
    saved = json.loads((directory/"stage_evidence.json").read_bytes())
    if (saved["status"], saved["plan_sha256"], saved["real_data_reads"]) != ("PASS", expected["plan_sha256"], 0):
        raise RuntimeError("synthetic results do not bind this exact plan")
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if sys.version_info[:2] != (3, 12) or any(version(name) != pins[name]+("+cu130" if name == "torch" else "")
            for name in ("torch", "numpy", "scipy", "h5py", "filelock")):
        raise RuntimeError("GN comparison requires the frozen deterministic CUDA float64 runtime")
    _cuda_preflight()
    return authorizations


def _baseline(root, plan, authorizations):
    v1, v2 = _v1_plan(root), _v2_plan(root)
    references = {}
    for mode, previous, previous_plan in (("fixed", "unified_first_swing_v1", v1),
                                           ("adaptive", "unified_first_swing_probes_v2", v2)):
        prior = authorizations[previous_plan["development_authorization_id"]]
        path = plan[f"{mode}_receipt"]
        if (prior["status"] != "consumed" or prior["result_receipt_path"] != path
                or prior["result_receipt_sha256"] != plan["bindings"][path]
                or prior["plan_sha256"] != sha256_file(root/f"configs/experiments/{previous}.toml")):
            raise RuntimeError("GN controls must come from the exact consumed runs")
        receipt = json.loads((root/path).read_bytes())
        if (receipt["run_id"], receipt["artifact_manifest_sha256"]) != (prior["run_id"], prior["result_manifest_sha256"]):
            raise RuntimeError("upstream receipt is inconsistent with the consumed scope")
        directory = _safe_relative(root, f"results/{receipt['run_id']}")
        hashes = verify_artifact_directory(directory, strict=True, expected_manifest_sha256=receipt["artifact_manifest_sha256"])
        references[mode] = (directory, receipt, hashes)
    fixed, receipt, hashes = references["fixed"]
    adaptive, prior_receipt, _ = references["adaptive"]
    freeze = json.loads((fixed/"freeze.json").read_bytes())
    adaptive_freeze = json.loads((adaptive/"freeze.json").read_bytes())
    checkpoint = receipt["checkpoints"]["first_swing"]
    if (hashes["freeze.json"] != receipt["frozen_model_manifest"]["sha256"]
            or hashes["freeze-first_swing.npz"] != checkpoint["sha256"]
            or (fixed/"freeze-first_swing.npz").relative_to(root).as_posix() != checkpoint["path"]
            or prior_receipt["baseline_run_id"] != receipt["run_id"]
            or not freeze["fitting_closed"] or not freeze["selection_closed"]
            or freeze["training"]["first_swing"]["selected_update"] != 0
            or freeze["settings"] != v1["settings"]
            or any(adaptive_freeze.get(k) != v for k, v in freeze.items())):
        raise RuntimeError("both controls must use the identical frozen M1 update-zero equation")
    source = json.loads((root/v1["source_contract"]).read_bytes())
    if freeze["backbone"] != source["backbone"]:
        raise RuntimeError("GN comparison backbone differs from the frozen TASK3 source")
    for name, digest in source["external_code"].items():
        if sha256_file(Path(name)) != digest:
            raise RuntimeError("frozen TASK3 scoring source changed")
    backbone = _archive_asset(root, source["backbone"], v1["anchor_archives"]["backbone"])
    preprocessing = _safe_relative(root, freeze["preprocessing_path"])
    if sha256_file(preprocessing) != freeze["preprocessing_sha256"]:
        raise RuntimeError("frozen preprocessing changed")
    baseline = {"freeze": freeze, "checkpoint": root/checkpoint["path"],
        "run_ids": {mode: item[1]["run_id"] for mode, item in references.items()},
        "reference": adaptive_freeze["adaptive_prior_reference"],
        "without_witness": json.loads((fixed/"train-witnesses.json").read_bytes())["without_witness"],
        "residuals": json.loads((adaptive/"adaptive-priors.json").read_bytes()),
        "witness": json.loads((adaptive/"train-witnesses.json").read_bytes()),
        "peaks": json.loads((adaptive/"val-speed-peaks.json").read_bytes()),
        "scores": {role: {mode: json.loads((directory/f"{role}.json").read_bytes())
                   for mode, directory in (("fixed", fixed/"first_swing"), ("adaptive", adaptive))}
                   for role in ("train", "val")}}
    return source, backbone, json.loads(preprocessing.read_bytes()), baseline


def _evaluate(root, directory, plan, source, backbone, scales, baseline, commit):
    import torch

    from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
    from kron_reduce.data.task3_full import Task3FullReader
    from kron_reduce.evaluation.first_swing_gn_decomposition import execute
    from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model
    from kron_reduce.evaluation.machine_local_synthetic import Budget
    from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(Path(source["task3_root"])/"src"))
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"])
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    try:
        budget.spend("checkpoint_loads")
        equation = LearnedNonlinearModel.load(io.BytesIO(backbone)).closure
        model = make_model(equation, "T0M1", train_scales=True, device="cuda")
        model.set_angle_damping(2./3., spatial_mix=1.)
        model.requires_grad_(False)
        budget.spend("checkpoint_loads")
        load_snapshot(model, baseline["checkpoint"])
        reader = Task3FullReader(root, source["sources"] | {"allowed_roles": ["train", "val"]}, journal)
        result = execute(model, reader, directory, budget, scales, plan, baseline)
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit,
            "data_policy": "kiki_official_train_test_v1", "promotion_decision": "KEEP_BASELINE",
            "scientific_pass": bool(result["passing_new_modes"]),
            "claim_boundary": "Frozen-equation GN comparison on the complete TASK3 train and val cohorts."}
        write_json(directory/"summary.json", evidence | result | {
            "operations": {key: budget.counts[key] for key in plan["operation_limits"]},
            "baseline_run_ids": baseline["run_ids"], "plan_sha256": sha256_file(root/CONFIG),
            "runtime": {"python": sys.version, "executable": sys.executable, "torch": version("torch"),
                        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name()}})
        write_json(directory/"stage_evidence.json", evidence)
    finally:
        budget.stream.close()
        journal.close()


def run(*, root, experiment, stage, result_dir, run_id):
    assert_clean(root)
    name = {"synthetic": FIXTURE, "probe": WORKFLOW}.get(stage)
    if (name is None or load_manifest(root)["experiments"][name] != experiment
            or experiment["stage"] != stage or experiment["workflow"] != WORKFLOW
            or experiment["module"] != __spec__.name or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("invalid GN comparison dispatch")
    commit, plan = git_head(root), _plan(root)
    try:
        if stage == "synthetic":
            _synthetic(root, result_dir, run_id, commit)
        else:
            authorizations = _authorize(root, experiment, plan)
            source, backbone, scales, baseline = _baseline(root, plan, authorizations)
            common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                                  cwd=root, text=True).strip()).resolve(strict=True)
            marker = common/"unified_first_swing_budgets"/f"{AUTHORIZATION}.json"
            marker.parent.mkdir(exist_ok=True)
            write_json(marker, {"run_id": run_id, "source_commit": commit, "retry_allowed": False})
            _evaluate(root, result_dir, plan, source, backbone, scales, baseline, commit)
        if git_head(root) != commit:
            raise RuntimeError("source commit changed during the GN comparison")
    finally:
        write_json(result_dir/"artifact_hashes.json", {path.relative_to(result_dir).as_posix(): sha256_file(path)
                   for path in sorted(result_dir.rglob("*")) if path.is_file()})


def main():
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        experiment = load_manifest(root)["experiments"][WORKFLOW]
        print(_run_experiment(root, WORKFLOW, "probe", experiment=experiment))


if __name__ == "__main__":
    main()
