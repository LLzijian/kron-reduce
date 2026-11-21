"""Registered one-use M1 geometry and frozen-initializer diagnostics."""

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
from project_harness.evidence import _safe_relative, sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.report_storage import locate_report
from project_harness.runner import _run_experiment

WORKFLOW = "unified_first_swing_probes_v2"
FIXTURE = "unified_first_swing_probes_fixture_v2"
AUTHORIZATION = f"{WORKFLOW}_20261119"
CONFIG = f"configs/experiments/{WORKFLOW}.toml"


def _plan(root):
    path = root/CONFIG
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    verify_bound_file(root, CONFIG, sha256_file(path))
    for name, digest in value["bindings"].items():
        verify_bound_file(root, name, digest)
    return value


def _synthetic(root, result_dir, run_id):
    temporary = root/"scratch"/"one_off"/run_id/"pytest"
    temporary.parent.mkdir(parents=True, exist_ok=False)
    tests = ["tests/test_unified_first_swing.py", "tests/test_first_swing_geometry.py",
             "tests/test_first_swing_probes.py"]
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--basetemp", str(temporary), *tests]
    result = subprocess.run(command, cwd=root, capture_output=True, check=False)
    (result_dir/"pytest.log").write_bytes(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError("M1 probe interface fixtures failed; see pytest.log")
    write_json(result_dir/"stage_evidence.json", {"status": "PASS", "source_commit": git_head(root),
        "plan_sha256": sha256_file(root/CONFIG), "real_data_reads": 0, "torch": version("torch"),
        "claim_boundary": "Known-truth causal-history and diagnostic interface fixtures."})


def _preflight(root, experiment, plan):
    validate_run_policy(root, WORKFLOW, experiment)
    state, policy = load_state(root, WORKFLOW), load_policy(root)
    scope = policy["development_evaluations"].get(AUTHORIZATION, {})
    expected = {"status": "active", "experiment": WORKFLOW, "allowed_roles": ["train", "val"],
                "train_cases": 346, "val_cases": 107, "test_cases": 0, "fit_attempts": 0,
                "source_cache_open_limit": 3, "max_executions": 1, "retry_allowed": False,
                "plan_path": CONFIG, "plan_sha256": sha256_file(root/CONFIG),
                "operation_limits": plan["operation_limits"]}
    if (experiment.get("development_authorization_id") != AUTHORIZATION or not approval_recorded(state, "probe")
            or scope.get("split_reference_sha256") != experiment["split_reference_sha256"]
            or any(type(scope.get(k)) is not type(v) or scope[k] != v for k, v in expected.items())):
        raise PermissionError("the exact one-use M1 diagnostic scope and stage approval are required")
    if state["current_stage"] != "synthetic":
        raise RuntimeError("M1 probes require their passed synthetic interface stage")
    stage = next(row["evidence"] for row in reversed(state["history"]) if row.get("to") == "synthetic")
    fixture = stage["artifact_manifest"]
    fixture_dir = _safe_relative(root, f"results/{fixture['run_id']}")
    verify_artifact_directory(fixture_dir, strict=True, expected_manifest_sha256=fixture["sha256"])
    saved = json.loads((fixture_dir/"stage_evidence.json").read_bytes())
    if (saved["status"], saved["plan_sha256"], saved["real_data_reads"]) != ("PASS", expected["plan_sha256"], 0):
        raise RuntimeError("synthetic results do not bind this exact plan")

    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if sys.version_info[:2] != (3, 12) or any(version(name) != pins[name]+("+cu130" if name == "torch" else "")
            for name in ("torch", "numpy", "scipy", "h5py", "filelock")):
        raise RuntimeError("diagnostics require the frozen CUDA float64 runtime")
    _cuda_preflight()

    v1 = _v1_plan(root)
    prior = policy["development_evaluations"]["unified_first_swing_v1_20261119"]
    receipt_path = plan["baseline_receipt"]
    if (prior["status"] != "consumed" or prior["result_receipt_path"] != receipt_path
            or prior["result_receipt_sha256"] != plan["bindings"][receipt_path]):
        raise RuntimeError("probe baseline must be the consumed and archived M1 v1 result")
    receipt = json.loads((root/locate_report(root, receipt_path, plan["bindings"][receipt_path])).read_bytes())
    directory = _safe_relative(root, f"results/{receipt['run_id']}")
    hashes = verify_artifact_directory(directory, strict=True, expected_manifest_sha256=receipt["artifact_manifest_sha256"])
    freeze = json.loads((directory/"freeze.json").read_bytes())
    checkpoint = receipt["checkpoints"]["first_swing"]
    if (directory/"freeze.json").relative_to(root).as_posix() != receipt["frozen_model_manifest"]["path"] or (
            hashes["freeze.json"] != receipt["frozen_model_manifest"]["sha256"]
            or hashes["freeze-first_swing.npz"] != checkpoint["sha256"]
            or (directory/"freeze-first_swing.npz").relative_to(root).as_posix() != checkpoint["path"]
            or not freeze["fitting_closed"] or not freeze["selection_closed"]
            or freeze["training"]["first_swing"]["selected_update"] != 0
            or freeze["settings"] != v1["settings"] or freeze["fit_plan_sha256"] != prior["plan_sha256"]):
        raise RuntimeError("probe must use the exact frozen v1 update-zero equation and settings")

    source = json.loads((root/v1["source_contract"]).read_bytes())
    if freeze["backbone"] != source["backbone"]:
        raise RuntimeError("probe backbone differs from the frozen TASK3 source")
    for path, digest in source["external_code"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("frozen TASK3 scoring source changed")
    baseline = {"freeze": freeze, "checkpoint": root/checkpoint["path"],
        "gradients": json.loads((directory/"initial_gradients.json").read_bytes()),
        "scores": {role: json.loads((directory/"first_swing"/f"{role}.json").read_bytes()) for role in ("train", "val")},
        "val_forecast": directory/"first_swing"/"val_predictions.npz",
        "witness": json.loads((directory/"train-witnesses.json").read_bytes()), "run_id": receipt["run_id"]}
    backbone = _archive_asset(root, source["backbone"], v1["anchor_archives"]["backbone"])
    preprocessing = _safe_relative(root, freeze["preprocessing_path"])
    if sha256_file(preprocessing) != freeze["preprocessing_sha256"]:
        raise RuntimeError("frozen preprocessing changed")
    return source, backbone, json.loads(preprocessing.read_bytes()), baseline


def run(*, root, experiment, stage, result_dir, run_id):
    assert_clean(root)
    name = {"synthetic": FIXTURE, "probe": WORKFLOW}.get(stage)
    if (name is None or load_manifest(root)["experiments"][name] != experiment
            or experiment["stage"] != stage or experiment["workflow"] != WORKFLOW
            or experiment["module"] != __spec__.name or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("invalid M1 probes dispatch")
    commit, plan = git_head(root), _plan(root)
    try:
        if stage == "synthetic":
            _synthetic(root, result_dir, run_id)
        else:
            source, backbone, scales, baseline = _preflight(root, experiment, plan)
            common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                                  cwd=root, text=True).strip()).resolve(strict=True)
            marker = common/"unified_first_swing_budgets"/f"{AUTHORIZATION}.json"
            marker.parent.mkdir(exist_ok=True)
            write_json(marker, {"run_id": run_id, "source_commit": commit, "retry_allowed": False})
            _evaluate(root, result_dir, plan, source, backbone, scales, baseline, commit)
        if git_head(root) != commit:
            raise RuntimeError("source commit changed during the diagnostic")
    finally:
        write_json(result_dir/"artifact_hashes.json", {path.relative_to(result_dir).as_posix(): sha256_file(path)
                   for path in sorted(result_dir.rglob("*")) if path.is_file()})


def _evaluate(root, directory, plan, source, backbone, scales, baseline, commit):
    import torch

    from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
    from kron_reduce.data.task3_full import Task3FullReader
    from kron_reduce.evaluation.first_swing_probes import execute
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
        budget.spend("checkpoint_loads")
        load_snapshot(model, baseline["checkpoint"])
        reader = Task3FullReader(root, source["sources"] | {"allowed_roles": ["train", "val"]}, journal)
        result = execute(model, reader, directory, budget, baseline["freeze"]["settings"], scales, plan, baseline)
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit,
            "data_policy": "kiki_official_train_test_v1", "promotion_decision": "KEEP_BASELINE",
            "claim_boundary": "Frozen M1 geometry and initializer probes on TASK3 train/val."}
        write_json(directory/"summary.json", evidence | result | {"operations": dict(budget.counts),
            "baseline_run_id": baseline["run_id"], "plan_sha256": sha256_file(root/CONFIG),
            "runtime": {"python": sys.version, "executable": sys.executable,
                        "torch": version("torch"), "cuda_build": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()}})
        write_json(directory/"stage_evidence.json", evidence)
    finally:
        budget.stream.close()
        journal.close()


def main():
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        experiment = load_manifest(root)["experiments"][WORKFLOW]
        print(_run_experiment(root, WORKFLOW, "probe", experiment=experiment))


if __name__ == "__main__":
    main()
