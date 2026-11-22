"""Registered M1 synthetic check and one-use TASK3 train/val dispatch."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.workflows.machine_local_equation import _cuda_preflight
from project_harness.archive_storage import archive_path
from project_harness.evidence import _safe_relative, sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "unified_first_swing_v1"
FIXTURE = "unified_first_swing_fixture_v1"
AUTHORIZATION = "unified_first_swing_v1_20261119"
CONFIG = f"configs/experiments/{WORKFLOW}.toml"


def _archive_asset(root, binding, archive):
    original = _safe_relative(root, binding["path"])
    if original.is_file():
        payload = original.read_bytes()
    else:
        member = original.relative_to(root/"results").as_posix()
        with ZipFile(archive_path(root, archive)) as backup:
            payload = backup.read(member)
    if hashlib.sha256(payload).hexdigest() != binding["sha256"]:
        raise RuntimeError(f"frozen starting asset changed: {binding['path']}")
    return payload


def _plan(root):
    plan_path = root/CONFIG
    plan = tomllib.loads(plan_path.read_text(encoding="utf-8"))
    verify_bound_file(root, CONFIG, sha256_file(plan_path))
    for path, digest in plan["bindings"].items():
        verify_bound_file(root, path, digest)
    return plan | {"plan_path": CONFIG}


def _fixture(root, directory, plan, run_id):
    scratch = root/"scratch"/"one_off"/run_id/"pytest"
    scratch.parent.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(scratch),
               "tests/test_unified_first_swing.py"]
    child = subprocess.run(command, cwd=root, capture_output=True, check=False)
    (directory/"pytest.log").write_bytes(child.stdout+child.stderr)
    result = {"status": "PASS" if child.returncode == 0 else "FAIL", "real_data_reads": 0,
              "source_commit": git_head(root), "claim_boundary": "Known-truth and engineering fixtures.",
              "plan_sha256": sha256_file(root/plan["plan_path"]), "python": sys.version,
              "executable": sys.executable, "torch": version("torch"), "pytest": version("pytest")}
    write_json(directory/"synthetic.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("first-swing synthetic checks failed; see pytest.log")
    write_json(directory/"stage_evidence.json", result)


def _fit_preflight(root, experiment, plan):
    validate_run_policy(root, WORKFLOW, experiment)
    state = load_state(root, WORKFLOW)
    authorization = load_policy(root)["development_evaluations"].get(AUTHORIZATION, {})
    signed = {"status": "active", "experiment": WORKFLOW, "fit_attempts": 2,
              "max_executions": 1, "retry_allowed": False, "allowed_roles": ["train", "val"],
              "train_cases": 346, "val_cases": 107, "test_cases": 0, "source_cache_open_limit": 3,
              "plan_path": CONFIG, "plan_sha256": sha256_file(root/CONFIG),
              "operation_limits": plan["operation_limits"]}
    if (experiment.get("development_authorization_id") != AUTHORIZATION or not approval_recorded(state, "fit")
            or any(type(authorization.get(key)) is not type(value) or authorization[key] != value
                   for key, value in signed.items())):
        raise PermissionError("explicit M1 fit approval and the exact one-use scientific scope are required")
    if state["current_stage"] != "synthetic":
        raise RuntimeError("first-swing fit requires completed synthetic evidence")
    evidence = next(item["evidence"] for item in reversed(state["history"])
                    if item.get("action") == "advance" and item.get("to") == "synthetic")
    binding = evidence["artifact_manifest"]
    directory = _safe_relative(root, f"results/{binding['run_id']}")
    manifest = directory/"artifact_hashes.json"
    if binding["path"] != manifest.name or sha256_file(manifest) != binding["sha256"]:
        raise RuntimeError("first-swing synthetic receipt changed")
    verify_artifact_directory(directory, strict=True)
    fixture = json.loads((directory/"synthetic.json").read_bytes())
    if fixture["status"] != "PASS" or fixture["real_data_reads"] != 0 or fixture["plan_sha256"] != signed["plan_sha256"]:
        raise RuntimeError("first-swing synthetic evidence differs from the proposed fit")

    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if sys.version_info[:2] != (3, 12) or any(
            version(name) != pins[name]+("+cu130" if name == "torch" else "")
            for name in ("torch", "numpy", "scipy", "h5py", "filelock")):
        raise RuntimeError("first-swing fit requires the pinned Python 3.12 CUDA runtime")
    _cuda_preflight()

    source = json.loads((root/plan["source_contract"]).read_bytes())
    for path, digest in source["external_code"].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("frozen TASK3 scoring code changed")
    historical_fit = json.loads((root/plan["anchor_contract"]).read_bytes())
    anchor = historical_fit["paired_start"]
    if source["backbone"] != historical_fit["backbone"]:
        raise RuntimeError("frozen TASK3 backbone bindings disagree")
    archives = plan["anchor_archives"]
    assets = {name: _archive_asset(root, anchor[name], archives["topology"])
              for name in ("freeze", "checkpoint", "preprocessing")}
    frozen = json.loads(assets.pop("freeze"))
    if (frozen["phase"] != "freeze" or not frozen["fitting_closed"] or not frozen["selection_closed"]
            or frozen["training"]["groups"]["T0M1"]["selected_update"] != 12
            or frozen["backbone"] != source["backbone"]
            or frozen["preprocessing_sha256"] != anchor["preprocessing"]["sha256"]):
        raise RuntimeError("paired start must be the exact original T0M1 update 12")
    assets["backbone"] = _archive_asset(root, source["backbone"], archives["backbone"])
    return source, assets


def run(*, root, experiment, stage, result_dir, run_id):
    assert_clean(root)
    manifest = load_manifest(root)["experiments"]
    expected = FIXTURE if stage == "synthetic" else WORKFLOW if stage == "fit" else None
    if (expected is None or manifest[expected] != experiment or experiment["workflow"] != WORKFLOW
            or experiment["stage"] != stage or experiment["module"] != __spec__.name
            or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("invalid first-swing dispatch")
    commit = git_head(root)
    plan = _plan(root)
    try:
        if stage == "synthetic":
            _fixture(root, result_dir, plan, run_id)
        else:
            source, assets = _fit_preflight(root, experiment, plan)
            common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                                  cwd=root, text=True).strip()).resolve(strict=True)
            marker = common/"unified_first_swing_budgets"/f"{AUTHORIZATION}.json"
            marker.parent.mkdir(exist_ok=True)
            write_json(marker, {"run_id": run_id, "source_commit": commit, "retry_allowed": False})
            from kron_reduce.evaluation.kiki_first_swing import execute

            execute(root, result_dir, plan, source, assets)
        if git_head(root) != commit:
            raise RuntimeError("first-swing source commit changed during execution")
    finally:
        write_json(result_dir/"artifact_hashes.json", {path.relative_to(result_dir).as_posix(): sha256_file(path)
                   for path in sorted(result_dir.rglob("*")) if path.is_file()})


def main():
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        experiment = load_manifest(root)["experiments"][WORKFLOW]
        print(_run_experiment(root, WORKFLOW, "fit", experiment=experiment))


if __name__ == "__main__":
    main()
