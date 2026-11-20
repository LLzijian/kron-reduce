"""One-use TASK3 train/val discovery after the registered CUDA synthetic gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.workflows.machine_local_equation import _cuda_preflight
from kron_reduce.workflows.unified_first_swing import _archive_asset
from kron_reduce.workflows.unified_first_swing_v4_fit import _file, _receipt
from project_harness.evidence import _safe_relative, sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, approve, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment, run_experiment

WORKFLOW = "unified_discovery_m3_v1"
FIXTURE = "unified_discovery_m3_fixture_v1"
CONFIG = f"configs/experiments/{WORKFLOW}.toml"


def _plan(root):
    path = root/CONFIG
    verify_bound_file(root, CONFIG, sha256_file(path))
    plan = tomllib.loads(path.read_text(encoding="utf-8"))
    for name, digest in plan["bindings"].items():
        verify_bound_file(root, name, digest)
    return plan | {"plan_path": CONFIG}


def _runtime(root):
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if sys.version_info[:2] != (3, 12) or any(version(key) != pins[key]+("+cu130" if key == "torch" else "")
            for key in ("torch", "numpy", "scipy", "h5py", "filelock")):
        raise RuntimeError("discovery requires the registered CUDA float64 runtime")
    _cuda_preflight()


def _synthetic(root, result_dir, run_id, commit):
    _runtime(root)
    temp = root/"scratch"/"one_off"/run_id/"pytest"
    temp.parent.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-s", "--basetemp", str(temp),
               "tests/test_unified_discovery_m3_model.py", "tests/test_unified_discovery_m3_selection.py"]
    check = subprocess.run(command, cwd=root, capture_output=True, check=False)
    (result_dir/"pytest.log").write_bytes(check.stdout+check.stderr)
    if check.returncode:
        raise RuntimeError("M3 synthetic fixtures failed; see pytest.log")
    write_json(result_dir/"stage_evidence.json", {"status": "PASS", "source_commit": commit,
        "plan_sha256": sha256_file(root/CONFIG), "real_data_reads": 0,
        "claim_boundary": "All 32 physical supports and 74 shared coordinates on synthetic trajectories."})


def _scope(root, experiment, plan):
    validate_run_policy(root, WORKFLOW, experiment)
    policy = load_policy(root)
    key = plan["development_authorization_id"]
    scope = policy["development_evaluations"][key]
    expected = {"status": "active", "experiment": WORKFLOW, "authority": "explicit_operator_m3_train_val_only",
        "allowed_roles": ["train", "val"], "train_cases": 346, "val_cases": 107, "test_cases": 0,
        "source_cache_open_limit": 3, "max_executions": 1, "retry_allowed": False,
        "fit_attempts": plan["operation_limits"]["fits"], "plan_path": CONFIG, "plan_sha256": sha256_file(root/CONFIG),
        "split_reference_sha256": plan["split_reference_sha256"], "operation_limits": plan["operation_limits"],
        "scope_limits": plan["scope_limits"]}
    if (key != experiment["development_authorization_id"] or experiment["reads_test_performance"] is not False
            or experiment["split_reference_sha256"] != expected["split_reference_sha256"]
            or any(type(scope.get(k)) is not type(v) or scope[k] != v for k, v in expected.items())):
        raise PermissionError("the exact active M3 train/val-only scope is required")
    state = load_state(root, WORKFLOW)
    if state["current_stage"] != "synthetic":
        raise RuntimeError("M3 discovery requires the passed synthetic stage")
    synthetic = next(row["evidence"] for row in state["history"] if row.get("to") == "synthetic")
    fixture = synthetic["artifact_manifest"]
    directory = _safe_relative(root, f"results/{fixture['run_id']}")
    verify_artifact_directory(directory, strict=True, expected_manifest_sha256=fixture["sha256"])
    if (synthetic["plan_sha256"], synthetic["real_data_reads"]) != (expected["plan_sha256"], 0):
        raise RuntimeError("synthetic gate is not bound to this exact plan")
    return key, policy, state


def _assets(root, plan, policy):
    source = json.loads((root/plan["source_contract"]).read_bytes())
    for path, digest in source["external_code"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("frozen TASK3 scoring source changed")
    if source["backbone"]["sha256"] != plan["a06_sha256"]:
        raise RuntimeError("the 45/9 anchor differs from A06")
    v4 = _receipt(root, policy["development_evaluations"][plan["v4_authorization"]], plan["v4_receipt"], plan)
    model = v4["parameter_checkpoints"]["soft_2r"]
    if v4["training"]["soft_2r"]["selected_update"] != 0 or model["sha256"] != plan["v4_memory_sha256"]:
        raise RuntimeError("memory coordinates must come from the frozen V4 update-zero equation")
    freeze = json.loads(_file(root, v4["frozen_model_manifest"]).read_bytes())
    preprocess = _file(root, {"path": freeze["preprocessing_path"], "sha256": freeze["preprocessing_sha256"]})
    reference = v4["training"]["hard_r"]["selected_witness"]
    if any(row["qualified"] != 33 or abs(row["signed_median"]-2.55102) > 1e-5
           or abs(row["absolute_median"]-2.62025) > 1e-5 for row in reference.values()):
        raise RuntimeError("the reference is not the frozen hard-R witness baseline")
    return source, {"backbone": _archive_asset(root, source["backbone"], plan["backbone_archive"]),
        "memory_path": _file(root, model), "preprocessing": json.loads(preprocess.read_bytes()), "witness_reference": reference}


def run(*, root, experiment, stage, result_dir, run_id):
    assert_clean(root)
    expected = {"synthetic": FIXTURE, "discovery": WORKFLOW}.get(stage)
    if (expected is None or load_manifest(root)["experiments"][expected] != experiment
            or experiment["stage"] != stage or experiment["workflow"] != WORKFLOW
            or experiment["module"] != __spec__.name or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("invalid M3 dispatch")
    commit, plan = git_head(root), _plan(root)
    try:
        if stage == "synthetic":
            _synthetic(root, result_dir, run_id, commit)
        else:
            key, policy, state = _scope(root, experiment, plan)
            if not approval_recorded(state, "discovery"):
                raise PermissionError("M3 requires the explicit operator discovery approval")
            _runtime(root)
            source, assets = _assets(root, plan, policy)
            common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                                  cwd=root, text=True).strip()).resolve(strict=True)
            marker = common/"unified_discovery_budgets"/f"{key}.json"
            marker.parent.mkdir(exist_ok=True)
            write_json(marker, {"run_id": run_id, "source_commit": commit, "retry_allowed": False})
            from kron_reduce.evaluation.unified_discovery_m3 import execute

            execute(root, result_dir, plan, source, assets, commit)
        if git_head(root) != commit:
            raise RuntimeError("source commit changed during discovery")
    finally:
        write_json(result_dir/"artifact_hashes.json", {path.relative_to(result_dir).as_posix(): sha256_file(path)
                   for path in sorted(result_dir.rglob("*")) if path.is_file()})


def main():
    root = project_root()
    assert_clean(root)
    plan = _plan(root)
    state = load_state(root, WORKFLOW)
    if state["current_stage"] == "scaffold":
        print(run_experiment(root, FIXTURE, "synthetic"), flush=True)
    entry = load_manifest(root)["experiments"][WORKFLOW]
    _, _, state = _scope(root, entry, plan)
    if not approval_recorded(state, "discovery"):
        approve(root, "discovery", WORKFLOW)
    with workflow_lock(root, WORKFLOW):
        print(_run_experiment(root, WORKFLOW, "discovery", experiment=entry), flush=True)
