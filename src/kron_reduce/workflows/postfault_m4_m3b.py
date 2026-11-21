"""Registered post-fault analysis, three matched refits, and gated fixed-test reporting."""

from __future__ import annotations

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
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model
from kron_reduce.models.first_swing_fit import enable_first_swing_rho, load_first_swing_scoring_view
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from kron_reduce.models.unified_discovery_m3 import UnifiedDiscoveryModel
from kron_reduce.workflows.unified_discovery_m3 import _runtime
from kron_reduce.workflows.unified_first_swing import _archive_asset
from kron_reduce.workflows.unified_first_swing_v4_fit import _file, _receipt
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, approve, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "postfault_m4_m3b_v1"
TEST = "unified_discovery_m3b_test_v1"
EXPERIMENTS = {"m4": "postfault_m4_v1", "fit": "unified_discovery_m3b_v1", "test": TEST}
CONFIG = f"configs/experiments/{WORKFLOW}.toml"
INPUT = "measured_history_100ms_zoh_v1"


def read(path):
    return json.loads(path.read_bytes())


def bind(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _plan(root):
    config = root/CONFIG
    verify_bound_file(root, CONFIG, sha256_file(config))
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    for name, sha in plan["bindings"].items():
        verify_bound_file(root, name, sha)
    return plan | {"plan_path": CONFIG}


def _entry_evidence(root, workflow, stage):
    state = load_state(root, workflow)
    evidence = next(row["evidence"] for row in state["history"] if row.get("to") == stage)
    artifact = evidence["artifact_manifest"]
    verify_artifact_directory(root/"results"/artifact["run_id"], strict=True, expected_manifest_sha256=artifact["sha256"])
    return evidence


def _scope(root, stage, plan, entry):
    experiment = EXPERIMENTS[stage]
    validate_run_policy(root, experiment, entry)
    policy = load_policy(root)
    test = stage == "test"
    category = "official_test_evaluations" if test else "development_evaluations"
    key = plan[f"{stage}_authorization"]
    declared = policy[category][key]
    expected = {"status": "active", "experiment": experiment, "max_executions": 1, "retry_allowed": False,
                "plan_path": CONFIG, "plan_sha256": sha256_file(root/CONFIG),
                "split_reference_sha256": plan["split_reference_sha256"],
                "fit_attempts": 3 if stage == "fit" else 0,
                "allowed_roles": ["test"] if test else ["train", "val"],
                "train_cases": 0 if test else 346, "val_cases": 0 if test else 107,
                "test_cases": 76 if test else 0, "source_cache_open_limit": {"m4": 0, "fit": 3, "test": 77}[stage]}
    if stage != "m4":
        expected["operation_limits"] = plan["test_operation_limits" if test else "operation_limits"]
    if stage == "m4":
        expected["result_archive_open_limit"] = 8
    if stage == "fit":
        expected["scope_limits"] = plan["scope_limits"]
    gate = "test_authorization" if test else "development_authorization_id"
    state = load_state(root, entry["workflow"])
    before = "m4" if stage == "fit" else "scaffold"
    if (entry.get(gate) != key or state["current_stage"] != before
            or any(type(declared.get(k)) is not type(v) or declared[k] != v for k, v in expected.items())):
        raise PermissionError("the exact active follow-up scope is required")
    if stage == "fit":
        evidence = _entry_evidence(root, WORKFLOW, "m4")
        if evidence["test_trajectory_reads"] != 0 or evidence["plan_sha256"] != expected["plan_sha256"]:
            raise PermissionError("fit requires the completed frozen M4 analysis")
    if test:
        evidence = _entry_evidence(root, WORKFLOW, "fit")
        parent = policy["development_evaluations"][plan["fit_authorization"]]
        receipt = _receipt(root, parent, parent["result_receipt_path"],
                           {"bindings": {parent["result_receipt_path"]: parent["result_receipt_sha256"]}})
        if (not evidence["scientific_pass"] or evidence["scientific_verdict"] != "ACCEPT"
                or not receipt["scientific_pass"] or receipt["run_id"] != evidence["result_dir"]
                or declared["frozen_model_manifest"] != evidence["frozen_model_manifest"]
                or receipt["frozen_model_manifest"] != evidence["frozen_model_manifest"]):
            raise PermissionError("fixed test requires an accepted, closed and hashed M3b winner")
    return key, policy


def _bound_archived(root, receipt, relative):
    return _file(root, {"path": f"results/{receipt['run_id']}/{relative}",
                        "sha256": receipt["artifact_bindings"][relative]})


def _upstream(root, policy, plan):
    return {name: _receipt(root, policy["development_evaluations"][plan[f"{name}_scope"]],
                           plan[f"{name}_receipt"], plan) for name in ("m3", "v4")}


def _unified(backbone, path, support):
    model = UnifiedDiscoveryModel(backbone, port=np.ones(10), tau=np.ones(10), support=tuple(support))
    return load_snapshot(model, path)


def _m4(root, directory, source, backbone, refs, evidence):
    from kron_reduce.evaluation.postfault_m4 import (
        archive_diagnostics,
        archived_cohort,
        energies,
        separation_cuts,
    )
    from kron_reduce.evaluation.postfault_m4_graph import evaluate_graph, plot_topology
    from kron_reduce.evaluation.postfault_m4_modes import evaluate_modes

    v4, m3 = refs["v4"], refs["m3"]
    hard_binding = v4["parameter_checkpoints"]["hard_r"]
    hard = enable_first_swing_rho(make_model(backbone, "T0M1", train_scales=True))
    load_snapshot(hard, _file(root, hard_binding))
    hard = load_first_swing_scoring_view(hard, make_model(backbone, "T0M1", train_scales=True))
    m3_freeze = read(_file(root, m3["frozen_model_manifest"]))
    if not m3_freeze["fitting_closed"] or not m3_freeze["selection_closed"] or m3_freeze["input_contract"] != INPUT:
        raise ValueError("M4 requires the exact closed M3 diagnostic snapshot")
    fixed = _unified(backbone, _file(root, m3["frozen_parameters"]), m3_freeze["support"])
    models = {"hard_r": hard, "m3": fixed}
    scores = {}
    stable_val = None
    with closing(AppendOnlyAuditJournal(directory/"access.jsonl")) as journal:
        for role in ("train", "val"):
            roster = source["sources"]["rosters"][role]
            artifacts = {}
            for model, ref, folder in (("hard_r", v4, "hard_r"), ("m3", m3, "curriculum_2b")):
                for kind, suffix in (("paths", "_predictions.npz"), ("metrics", ".json")):
                    path = _bound_archived(root, ref, f"{folder}/{role}{suffix}")
                    journal.append({"stage": "archived_result_open", "role": role, "model": model, "kind": kind, **bind(root, path)})
                    artifacts[model, kind] = path
            with (np.load(artifacts["hard_r", "paths"], allow_pickle=False) as old,
                  np.load(artifacts["m3", "paths"], allow_pickle=False) as new):
                events, forecasts = archived_cohort(roster, old, new, role)
            diagnostics = archive_diagnostics(roster, read(artifacts["hard_r", "metrics"]), read(artifacts["m3", "metrics"]))
            journal.append({"stage": "roster_verified", "role": role, "count": len(events), "ids": [event["id"] for event in events]})
            scores[role] = {step: {name: {"energies": energies(model, events, diagnostics[step][name][0]),
                                    "cuts": separation_cuts(events, forecasts[name][step], diagnostics[step][name][1])}
                             for name, model in models.items()} for step in diagnostics}
            if role == "val":
                stable_val = [e for e in events if e["label"] == "stable"]
    if stable_val is None or len(stable_val) != 95:
        raise RuntimeError("M4 stable val roster is incomplete")
    modes = evaluate_modes(models, stable_val)
    graph = evaluate_graph(models, stable_val)
    plot_topology(graph, directory/"top10_topology")
    write_json(directory/"modes.json", modes)
    write_json(directory/"graph.json", graph)
    write_json(directory/"event_physics.json", scores)
    write_json(directory/"summary.json", evidence | {"zero_training": True, "new_rollouts": 0,
        "original_data_cache_opens": 0, "result_archive_opens": 8, "test_trajectory_reads": 0,
        "frozen_models": {"hard_r": hard_binding, "m3": m3["frozen_parameters"]}, "modes": modes["summary"], "runtime": _versions()})
    write_json(directory/"stage_evidence.json", evidence | {"test_trajectory_reads": 0, "original_data_cache_opens": 0})


def _fit(root, directory, plan, source, backbone, refs, evidence):
    from kron_reduce.evaluation.unified_discovery_m3b import execute

    m3, v4 = refs["m3"], refs["v4"]
    path = _bound_archived(root, m3, "backbone_54d.npz")
    if sha256_file(path) != plan["backbone_sha256"]:
        raise ValueError("M3b requires the selected M3 54-dimensional backbone")
    assets = {"model": _unified(backbone, path, ("G2",)),
              "preprocessing": read(_bound_archived(root, m3, "preprocessing.json")),
              "witness_reference": v4["training"]["hard_r"]["selected_witness"],
              "screening": read(_bound_archived(root, m3, "screening_table.json"))}
    execute(root, directory, plan, source, assets, evidence["source_commit"])


def _test(root, directory, plan, source, backbone, policy, evidence):
    from kron_reduce.data.task3_full import Task3FullReader
    from kron_reduce.evaluation.first_swing_balanced_fit import trajectory_groups
    from kron_reduce.evaluation.kiki_machine_local_full import evaluate, inputs
    from kron_reduce.evaluation.machine_local_synthetic import Budget
    from kron_reduce.models.unified_discovery_m3 import predict_m3

    scope = policy["official_test_evaluations"][plan["test_authorization"]]
    freeze_path = _file(root, scope["frozen_model_manifest"])
    frozen = read(freeze_path)
    if (not frozen["selection_closed"] or not frozen["fitting_closed"] or frozen["decision"] != "SELECTED"
            or not all(frozen["gates"].values()) or frozen["settings"] != plan["settings"]
            or frozen["plan_sha256"] != evidence["plan_sha256"] or frozen["input_contract"] != INPUT
            or frozen["support"] not in (["G2"], ["G2", "G5"])):
        raise PermissionError("fixed-test evaluation model or contract differs from the accepted winner")
    weights = _file(root, {"path": frozen["checkpoint_path"], "sha256": frozen["checkpoint_sha256"]})
    scales = read(_file(root, {"path": frozen["preprocessing_path"], "sha256": frozen["preprocessing_sha256"]}))
    prior = policy["official_test_evaluations"][plan["v4_test_scope"]]
    v4_path = plan["v4_test_receipt"]
    verify_bound_file(root, v4_path, plan["v4_test_receipt_sha256"])
    archived = _receipt(root, prior, v4_path, {"bindings": {v4_path: plan["v4_test_receipt_sha256"]}})
    v4 = read(_bound_archived(root, archived, "summary.json"))["records"]
    budget = Budget(directory/"operations.jsonl", plan["test_operation_limits"])
    with closing(AppendOnlyAuditJournal(directory/"access.jsonl")) as journal, closing(budget.stream):
        budget.spend("checkpoint_loads")
        model = _unified(backbone, weights, frozen["support"]).to("cuda")
        reader = Task3FullReader(root, source["sources"] | {"allowed_roles": ["test"]}, journal)
        events = reader.load("test", freeze_path=freeze_path)
        if (len(events) != 76 or sum(e["label"] == "unstable" for e in events) != 9
                or not np.allclose(reader.static["mass"], backbone.mass, rtol=0, atol=1e-12)):
            raise RuntimeError("fixed test roster or inertia differs from TASK3")

        def predict(batch, step):
            result = predict_m3(model, *inputs(batch, scales["history"]), 800, max_step=step, spend=budget.spend,
                                memory_radius=scales["memory_radius"], prior_weight=plan["settings"]["prior_weight"])
            return result | {"gn_singular": result["singular"][None]}

        result = evaluate(model, events, scales["history"], budget, directory, "test", plan["settings"], predictor=predict)
    record = {k: result[k] for k in ("stability", "stable_complete", "unstable_prefix", "task3_six_term_score", "numerical_pass")}
    record["stable_errors"] = trajectory_groups(result)
    evidence = evidence | {"frozen_model_manifest": scope["frozen_model_manifest"], "test_trajectory_reads": 76,
                           "promotion_decision": "KEEP_BASELINE"}
    write_json(directory/"summary.json", evidence | {"m3b": record, "v4": v4, "support": frozen["support"],
        "operations": {k: budget.counts[k] for k in budget.limits}, "runtime": _versions()})
    write_json(directory/"stage_evidence.json", evidence)


def _versions():
    return {"python": sys.version, "executable": sys.executable, "torch": version("torch"),
            "scipy": version("scipy"), "numpy": version("numpy"), "cuda_build": torch.version.cuda}


def run(*, root, experiment, stage, result_dir, run_id):
    assert_clean(root)
    name = EXPERIMENTS[stage]
    workflow = TEST if stage == "test" else WORKFLOW
    if (load_manifest(root)["experiments"][name] != experiment or experiment["stage"] != stage
            or experiment["workflow"] != workflow or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("invalid frozen follow-up dispatch")
    commit = git_head(root)
    plan = _plan(root)
    key, policy = _scope(root, stage, plan, experiment)
    if not approval_recorded(load_state(root, workflow), stage):
        raise PermissionError("explicit operator stage approval required")
    if stage != "m4":
        _runtime(root)
    elif sys.version_info[:2] != (3, 12):
        raise RuntimeError("M4 requires the pinned Python 3.12 analysis runtime")
    source = read(root/plan["source_contract"])
    for file, sha in source["external_code"].items():
        if sha256_file(Path(file)) != sha:
            raise RuntimeError("frozen TASK3 scoring source changed")
    sys.path.insert(0, str(Path(source["task3_root"])/"src"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if source["backbone"]["sha256"] != plan["a06_sha256"]:
        raise ValueError("the physical anchor differs from A06")
    backbone = LearnedNonlinearModel.load(io.BytesIO(_archive_asset(root, source["backbone"], plan["backbone_archive"]))).closure
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"postfault_followup_budgets"/f"{key}.json"
    marker.parent.mkdir(exist_ok=True)
    write_json(marker, {"run_id": run_id, "source_commit": commit, "retry_allowed": False})
    evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit, "plan_sha256": sha256_file(root/CONFIG),
                "data_policy": "kiki_official_train_test_v1", "claim_boundary": "Post-fault TASK3 with the frozen 100 ms input contract."}
    start = time.perf_counter()
    try:
        if stage == "test":
            _test(root, result_dir, plan, source, backbone, policy, evidence)
        else:
            assets = _upstream(root, policy, plan)
            if stage == "m4":
                _m4(root, result_dir, source, backbone, assets, evidence)
            else:
                _fit(root, result_dir, plan, source, backbone, assets, evidence)
        if git_head(root) != commit:
            raise RuntimeError("source commit changed during frozen follow-up")
        write_json(result_dir/"runtime.json", _versions() | {"wall_seconds": time.perf_counter()-start})
    finally:
        write_json(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
                   for p in sorted(result_dir.rglob("*")) if p.is_file()})


def main(stage):
    root = project_root()
    assert_clean(root)
    plan = _plan(root)
    experiment = load_manifest(root)["experiments"][EXPERIMENTS[stage]]
    workflow = experiment["workflow"]
    _scope(root, stage, plan, experiment)
    if not approval_recorded(load_state(root, workflow), stage):
        approve(root, stage, workflow)
    with workflow_lock(root, workflow):
        print(_run_experiment(root, EXPERIMENTS[stage], stage, experiment=experiment), flush=True)
