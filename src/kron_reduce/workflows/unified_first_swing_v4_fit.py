"""Bounded first-swing fit followed by separately registered frozen TASK3 reporting."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import tomllib
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.workflows.machine_local_equation import _cuda_preflight
from kron_reduce.workflows.unified_first_swing import _archive_asset
from project_harness.evidence import _safe_relative, sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.report_storage import locate_report
from project_harness.runner import _run_experiment

WORKFLOW = "unified_first_swing_v4_fit"
STAGES = {"synthetic": "unified_first_swing_v4_fixture", "fit": WORKFLOW, "test": "unified_first_swing_v4_test"}
CONFIG = f"configs/experiments/{WORKFLOW}.toml"


def _plan(root):
    path = root/CONFIG
    verify_bound_file(root, CONFIG, sha256_file(path))
    plan = tomllib.loads(path.read_text(encoding="utf-8"))
    for name, digest in plan["bindings"].items():
        verify_bound_file(root, name, digest)
    return plan | {"plan_path": CONFIG}


def _synthetic(root, directory, run_id, commit):
    temporary = root/"scratch"/"one_off"/run_id/"pytest"
    temporary.parent.mkdir(parents=True, exist_ok=False)
    child = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
        str(temporary), "tests/test_first_swing_v4_model.py", "tests/test_first_swing_v4_fit.py",
        "tests/test_first_swing_gn_grid.py", "tests/test_unified_first_swing.py"], cwd=root, capture_output=True, check=False)
    (directory/"pytest.log").write_bytes(child.stdout+child.stderr)
    if child.returncode:
        raise RuntimeError("first-swing v4 synthetic fixtures failed; see pytest.log")
    write_json(directory/"stage_evidence.json", {"status": "PASS", "source_commit": commit,
        "plan_sha256": sha256_file(root/CONFIG), "real_data_reads": 0,
        "claim_boundary": "Known-truth causal parity, differentiability, dissipation and checkpoint fixtures."})


def _file(root, binding):
    path = _safe_relative(root, locate_report(root, binding["path"], binding["sha256"]))
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise RuntimeError(f"frozen file differs from its binding: {binding['path']}")
    return path


def _json(path):
    return json.loads(path.read_bytes())


def _receipt(root, declared, path, plan):
    if (declared["status"] != "consumed" or declared["result_receipt_path"] != path
            or declared["result_receipt_sha256"] != plan["bindings"][path]):
        raise RuntimeError("upstream reference receipt must be the exact consumed scope")
    receipt = _json(_file(root, {"path": path, "sha256": plan["bindings"][path]}))
    if (receipt["run_id"], receipt["artifact_manifest_sha256"]) != (declared["run_id"], declared["result_manifest_sha256"]):
        raise RuntimeError("upstream result manifest differs from the consumed scope")
    return receipt


def _authorize(root, entry, stage, plan):
    validate_run_policy(root, STAGES[stage], entry)
    state, policy = load_state(root, WORKFLOW), load_policy(root)
    is_fit = stage == "fit"
    name = plan[f"{stage}_authorization"]
    category = "development_evaluations" if is_fit else "official_test_evaluations"
    scope = policy[category][name]
    limits = plan[f"{stage}_operation_limits"]
    required = {"status": "active", "experiment": STAGES[stage], "allowed_roles": ["train", "val"] if is_fit else ["test"],
        "fit_attempts": limits["fits"], "max_executions": 1, "retry_allowed": False, "plan_path": CONFIG,
        "plan_sha256": sha256_file(root/CONFIG), "split_reference_sha256": entry["split_reference_sha256"],
        "train_cases": 346 if is_fit else 0, "val_cases": 107 if is_fit else 0, "test_cases": 0 if is_fit else 76,
        "source_cache_open_limit": 3 if is_fit else 77, "operation_limits": limits}
    if (entry["development_authorization_id" if is_fit else "test_authorization"] != name
            or state["current_stage"] != ("synthetic" if is_fit else "fit") or not approval_recorded(state, stage)
            or any(type(scope.get(k)) is not type(v) or scope[k] != v for k, v in required.items())):
        raise PermissionError("the exact active stage and single-use scope are required")
    precursor = next(row["evidence"] for row in state["history"] if row.get("to") == state["current_stage"])
    artifact = precursor["artifact_manifest"]
    directory = _safe_relative(root, f"results/{artifact['run_id']}")
    verify_artifact_directory(directory, strict=True, expected_manifest_sha256=artifact["sha256"])
    if is_fit and (precursor["plan_sha256"], precursor["real_data_reads"]) != (required["plan_sha256"], 0):
        raise RuntimeError("the passed synthetic fixture does not bind this exact plan")
    if not is_fit:
        trained = policy["development_evaluations"][plan["fit_authorization"]]
        frozen = scope["frozen_model_manifest"]
        fit_receipt = _file(root, {"path": trained["result_receipt_path"], "sha256": trained["result_receipt_sha256"]})
        if (trained["status"], trained["run_id"], trained["result_manifest_sha256"]) != (
                "consumed", artifact["run_id"], artifact["sha256"]) or (
                frozen != _json(fit_receipt)["frozen_model_manifest"] or
                _file(root, frozen) != directory/"freeze.json"):
            raise RuntimeError("test authority must bind the frozen and closed fit")
    pins = dict(line.split("==") for line in (root/"requirements/constraints-py312.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    if sys.version_info[:2] != (3, 12) or any(version(key) != pins[key]+("+cu130" if key == "torch" else "")
            for key in ("torch", "numpy", "scipy", "h5py", "filelock")):
        raise RuntimeError("registered Python 3.12 and deterministic CUDA float64 runtime required")
    _cuda_preflight()
    return name, scope, policy


def _assets(root, stage, plan, scope, policy):
    source = _json(root/plan["source_contract"])
    for name, digest in source["external_code"].items():
        if sha256_file(Path(name)) != digest:
            raise RuntimeError("TASK3 scoring source changed")
    backbone = _archive_asset(root, source["backbone"], plan["backbone_archive"])
    if stage == "fit":
        old = policy["development_evaluations"]
        base = _receipt(root, old["unified_first_swing_v1_20261119"], plan["anchor_receipt"], plan)
        prior = _receipt(root, old["unified_first_swing_gn_decomposition_v3_20261119"], plan["initializer_receipt"], plan)
        anchor = _json(_file(root, base["frozen_model_manifest"]))
        checkpoint = _file(root, base["checkpoints"]["first_swing"])
        if (anchor["training"]["first_swing"]["selected_update"] != 0
                or prior["baseline_run_ids"]["fixed"] != base["run_id"]
                or not anchor["fitting_closed"] or not anchor["selection_closed"]):
            raise RuntimeError("v4 must start from the same frozen update-zero equation as v3")
        frozen = anchor
        extra = {"checkpoint": checkpoint, "baseline": prior}
    else:
        path = _file(root, scope["frozen_model_manifest"])
        frozen = _json(path)
        alpha = frozen["settings"].get("alpha_fs")
        if (frozen["phase"] != "freeze" or not frozen["fitting_closed"] or not frozen["selection_closed"]
                or frozen["fit_plan_sha256"] != sha256_file(root/CONFIG)
                or frozen["settings"] != plan["settings"] | {"alpha_fs": alpha}):
            raise RuntimeError("test requires the exact closed parameter and evaluation contract")
        index = _json(_file(root, {"path": frozen["checkpoint_path"], "sha256": frozen["checkpoint_sha256"]}))
        if set(index) != {"hard_r", "soft_2r"}:
            raise RuntimeError("the frozen pair differs from the registered comparison")
        extra = {"freeze_path": path, "checkpoints": {key: _file(root, value) for key, value in index.items()},
                 "settings": frozen["settings"]}
    if frozen["backbone"] != source["backbone"] or frozen["input_contract"] != "measured_history_100ms_zoh_v1":
        raise RuntimeError("model backbone or input contract differs from the frozen reference")
    prep = _json(_file(root, {"path": frozen["preprocessing_path"], "sha256": frozen["preprocessing_sha256"]}))
    return source, backbone, prep, extra


def _execute(root, directory, stage, plan, source, backbone, scales, extra, commit):
    import torch

    from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
    from kron_reduce.data.task3_full import Task3FullReader
    from kron_reduce.evaluation.first_swing_balanced_fit import (
        GROUPS,
        compact_score,
        fit,
        gradient_calibration,
        score,
    )
    from kron_reduce.evaluation.kiki_machine_local_topology import freeze_pair, scales_from_train
    from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model
    from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
    from kron_reduce.evaluation.machine_local_synthetic import Budget
    from kron_reduce.models.first_swing_fit import (
        enable_first_swing_rho,
        load_first_swing_scoring_view,
    )
    from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(Path(source["task3_root"])/"src"))
    started = time.perf_counter()
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    budget = Budget(directory/"operations.jsonl", plan[f"{stage}_operation_limits"])
    with closing(journal), closing(budget.stream):
        budget.spend("checkpoint_loads")
        equation = LearnedNonlinearModel.load(io.BytesIO(backbone)).closure
        roles = ["train", "val"] if stage == "fit" else ["test"]
        reader = Task3FullReader(root, source["sources"] | {"allowed_roles": roles}, journal)
        models = {group: make_model(equation, "T0M1", train_scales=True, device="cuda") for group in GROUPS}
        views = {group: make_model(equation, "T0M1", train_scales=True, device="cuda") for group in GROUPS}
        for group in models:
            (directory/group).mkdir()
        if stage == "fit":
            budget.spend("checkpoint_loads")
            old = load_snapshot(models["hard_r"], extra["checkpoint"])
            models["soft_2r"].load_state_dict(snapshot(old))
            for model in models.values():
                enable_first_swing_rho(model)
                model.requires_grad_(True)
            settings = plan["settings"]
            train = reader.load("train")
            if not np.allclose(reader.static["mass"], equation.mass, rtol=0, atol=1e-12):
                raise RuntimeError("train inertia differs from the frozen equation")
            derived = scales_from_train(train, equation.mass)
            if set(derived) != set(scales) or any(not np.allclose(derived[key], scales[key], rtol=1e-13, atol=1e-14)
                                                 for key in scales):
                raise RuntimeError("train preprocessing differs from the frozen anchor")
            write_json(directory/"preprocessing.json", scales)
            try:
                calib = gradient_calibration(models["soft_2r"], train, scales, settings, budget)
            except NUMERICAL_ERRORS as error:
                calib = {"status": "FAIL", "alpha_fs": None, "message": str(error)}
            write_json(directory/"initial-gradients.json", calib)
            settings = settings | {"alpha_fs": calib["alpha_fs"]}
            freeze_plan = plan | {"settings": settings, "backbone": source["backbone"]}
            initial = freeze_pair(root, directory, "initial", models, freeze_plan, None)
            val = reader.load("val", freeze_path=initial)
            history = {group: fit(model, views[group], train, val, scales, settings, budget, directory/group, group,
                                  calib, extra["baseline"])
                       for group, model in models.items()}
            freeze = freeze_pair(root, directory, "freeze", models, freeze_plan, history)
            populations = {"train": train, "val": val}
            fitted = history["soft_2r"]
            completed = all(row["completed_updates"] == settings["updates"] for row in history.values())
            science = completed and fitted["selected_update"] > 0 and fitted["selected_val"]["joint_pass"]
            extra_output = {"training": history, "gradients": calib, "scientific_pass": science,
                            "target_fp_scientific_pass": science and fitted["selected_val"]["target_fp_pass"]}
        else:
            for group, model in models.items():
                enable_first_swing_rho(model)
                budget.spend("checkpoint_loads")
                load_snapshot(model, extra["checkpoints"][group])
            freeze, settings = extra["freeze_path"], extra["settings"]
            populations = {"test": reader.load("test", freeze_path=freeze)}
            extra_output = {"holdout_scope": "previously_exposed_fixed_comparison_reporting_only"}
        if not np.allclose(reader.static["mass"], equation.mass, rtol=0, atol=1e-12):
            raise RuntimeError("TASK3 inertia differs from the frozen equation")

        records = {}
        for group, model in models.items():
            load_first_swing_scoring_view(model, views[group])
            records[group] = {role: compact_score(score(views[group], events, scales, budget, directory/group, role,
                                                       settings, group, save=True)) for role, events in populations.items()}
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit,
                    "data_policy": "kiki_official_train_test_v1", "promotion_decision": "KEEP_BASELINE",
                    "plan_sha256": sha256_file(root/CONFIG), "claim_boundary": "Fixed TASK3 cohorts and the causal 100 ms contract."}
        binding = {"path": freeze.relative_to(root).as_posix(), "sha256": sha256_file(freeze)}
        write_json(directory/"summary.json", evidence | extra_output | {"records": records, "frozen_model_manifest": binding,
            "operations": {key: budget.counts[key] for key in budget.limits}, "wall_seconds": time.perf_counter()-started,
            "runtime": {"python": sys.version, "executable": sys.executable, "torch": version("torch"),
                        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name()}})
        write_json(directory/"stage_evidence.json", evidence | {"frozen_model_manifest": binding})


def run(*, root, experiment, stage, result_dir, run_id):
    assert_clean(root)
    if (stage not in STAGES or load_manifest(root)["experiments"][STAGES[stage]] != experiment
            or experiment["stage"] != stage or experiment["workflow"] != WORKFLOW or experiment["module"] != __spec__.name
            or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("invalid first-swing v4 dispatch")
    commit, plan = git_head(root), _plan(root)
    try:
        if stage == "synthetic":
            _synthetic(root, result_dir, run_id, commit)
        else:
            name, scope, policy = _authorize(root, experiment, stage, plan)
            source, backbone, scales, extra = _assets(root, stage, plan, scope, policy)
            common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                                  cwd=root, text=True).strip()).resolve(strict=True)
            marker = common/"unified_first_swing_budgets"/f"{name}.json"
            marker.parent.mkdir(exist_ok=True)
            write_json(marker, {"run_id": run_id, "source_commit": commit, "retry_allowed": False})
            _execute(root, result_dir, stage, plan, source, backbone, scales, extra, commit)
        if git_head(root) != commit:
            raise RuntimeError("source commit changed during the bounded v4 stage")
    finally:
        write_json(result_dir/"artifact_hashes.json", {path.relative_to(result_dir).as_posix(): sha256_file(path)
                   for path in sorted(result_dir.rglob("*")) if path.is_file()})


def main():
    argument = argparse.ArgumentParser()
    argument.add_argument("stage", choices=["fit", "test"])
    stage = argument.parse_args().stage
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        name = STAGES[stage]
        entry = load_manifest(root)["experiments"][name]
        print(_run_experiment(root, name, stage, experiment=entry))


if __name__ == "__main__":
    main()
