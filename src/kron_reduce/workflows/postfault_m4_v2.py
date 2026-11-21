"""Zero-rollout follow-up on three frozen TASK3 equations and archived cohorts."""

from __future__ import annotations

import io
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
from kron_reduce.evaluation.postfault_m4 import archive_diagnostics, archived_cohort, energies
from kron_reduce.models.first_swing_fit import enable_first_swing_rho, load_first_swing_scoring_view
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from kron_reduce.workflows.postfault_m4_m3b import INPUT, _bound_archived, _unified, read
from kron_reduce.workflows.unified_first_swing import _archive_asset
from kron_reduce.workflows.unified_first_swing_v4_fit import _file, _receipt
from project_harness.evidence import sha256_file
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, approve, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "postfault_m4_v2"
CONFIG = f"configs/experiments/{WORKFLOW}.toml"
ROLES = ("train", "val")


def read_plan(root, relative):
    verify_bound_file(root, relative, sha256_file(root/relative))
    plan = tomllib.loads((root/relative).read_text(encoding="utf-8"))
    for name, digest in plan["bindings"].items():
        verify_bound_file(root, name, digest)
    return plan


def prior_receipt(root, plan, key):
    policy = load_policy(root)["development_evaluations"]
    return _receipt(root, policy[plan[f"{key}_scope"]], plan[f"{key}_receipt"], plan)


def frozen_models(root, plan, device="cpu"):
    """Construct the exact three scoring views without fitting or reading an event."""
    source = read(root/plan["source_contract"])
    v4 = prior_receipt(root, plan, "v4")
    m3 = prior_receipt(root, plan, "m3b")
    if source["backbone"]["sha256"] != plan["a06_sha256"] or not m3["scientific_pass"]:
        raise ValueError("the evaluation requires A06 and the accepted M3b freeze")
    backbone = LearnedNonlinearModel.load(
        io.BytesIO(_archive_asset(root, source["backbone"], plan["backbone_archive"]))).closure
    old = read(_file(root, v4["frozen_model_manifest"]))
    new = read(_file(root, m3["frozen_model_manifest"]))
    if (any(z["input_contract"] != INPUT or not z["selection_closed"] or not z["fitting_closed"] for z in (old, new))
            or old["backbone"] != source["backbone"] or new["backbone_anchor"] != source["backbone"]
            or new["support"] != ["G2"] or new["decision"] != "SELECTED"):
        raise ValueError("the closed model identities differ from the frozen evaluation contract")
    weights = {key: v4["parameter_checkpoints"][key] for key in ("hard_r", "soft_2r")} | {"m3": m3["frozen_parameters"]}
    if (any(weights[k]["sha256"] != plan["v4_parameters_sha256"] for k in ("hard_r", "soft_2r"))
            or weights["m3"]["sha256"] != plan["m3b_parameters_sha256"]
            or new["checkpoint_sha256"] != weights["m3"]["sha256"]
            or new["checkpoint_path"] != weights["m3"]["path"]):
        raise ValueError("a frozen parameter file differs from the pre-registered identity")
    prep = {"path": old["preprocessing_path"], "sha256": old["preprocessing_sha256"]}
    later = {"path": new["preprocessing_path"], "sha256": new["preprocessing_sha256"]}
    if prep["sha256"] != plan["preprocessing_sha256"]:
        raise ValueError("the registered V4 preprocessing changed")
    scales = read(_file(root, prep))
    if read(_file(root, later)) != scales:
        raise ValueError("the M3b preprocessing differs from the shared V4 scales")
    if old["settings"]["prior_weight"] != plan["prior_weight"] or new["settings"]["prior_weight"] != plan["prior_weight"]:
        raise ValueError("the fixed-prior initializers require their frozen prior weight")
    models = {}
    for key, binding in weights.items():
        path = _file(root, binding)
        if key == "m3":
            models[key] = _unified(backbone, path, new["support"]).to(device)
        else:
            fitted = load_snapshot(enable_first_swing_rho(make_model(backbone, "T0M1", train_scales=True)), path)
            models[key] = load_first_swing_scoring_view(
                fitted, make_model(backbone, "T0M1", train_scales=True)).to(device)
    return source, models, scales, {"v4": v4, "m3b": m3, "weights": weights, "preprocessing": prep}


def _archive(root, journal, receipt, relative, model, role, field):
    path = _bound_archived(root, receipt, relative)
    journal.append({"stage": "archived_artifact_open", "model": model, "role": role, "field": field,
                    "path": path.relative_to(root).as_posix(), "sha256": receipt["artifact_bindings"][relative]})
    return path


def _profiles(root, journal, source, models, refs):
    profiles = {}
    names = {"hard_r": (refs["v4"], "hard_r"), "soft_2r": (refs["v4"], "soft_2r"),
             "m3": (refs["m3b"], "support-G2")}
    for role in ROLES:
        roster = source["sources"]["rosters"][role]
        archives = {}
        for model, (receipt, directory) in names.items():
            for field, suffix in (("paths", "_predictions.npz"), ("metrics", ".json")):
                archives[model, field] = _archive(root, journal, receipt, f"{directory}/{role}{suffix}", model, role, field)
        with (np.load(archives["hard_r", "paths"], allow_pickle=False) as hard,
              np.load(archives["soft_2r", "paths"], allow_pickle=False) as soft,
              np.load(archives["m3", "paths"], allow_pickle=False) as current):
            events, _ = archived_cohort(roster, hard, current, role)
            if any(not np.array_equal(hard[field], soft[field]) for field in
                   ("ids", "truth", "lengths", "mass", "equilibrium", "labels")):
                raise ValueError("the two V4 archives use different truth or roster")
            if any(not np.allclose(hard[key][:, 0], soft[key][:, 0], atol=1e-12, rtol=0)
                   for key in ("fine", "refined")):
                raise ValueError("the V4 initializer changed the visible forecast origin")
        m3_metrics = read(archives["m3", "metrics"])
        diagnostics = archive_diagnostics(roster, read(archives["hard_r", "metrics"]), m3_metrics)
        alternate = archive_diagnostics(roster, read(archives["soft_2r", "metrics"]), m3_metrics)
        profiles[role] = {}
        for step, values in diagnostics.items():
            selected = values | {"soft_2r": alternate[step]["hard_r"]}
            profiles[role][step] = {k: energies(m, events, selected[k][0]) for k, m in models.items()}
        journal.append({"stage": "roster_verified", "role": role, "count": len(events), "ids": [e["id"] for e in events]})
    return profiles


def _authorized(root, plan, entry, relative):
    validate_run_policy(root, WORKFLOW, entry)
    policy = load_policy(root)["development_evaluations"]
    key = plan["authorization"]
    scope = policy[key]
    expected = {"status": "active", "experiment": WORKFLOW, "plan_path": relative, "plan_sha256": sha256_file(root/relative),
                "max_executions": 1, "retry_allowed": False, "allowed_roles": ["train", "val"], "fit_attempts": 0,
                "train_cases": 346, "val_cases": 107, "test_cases": 0, "archive_artifact_opens": 13,
                "original_data_cache_opens": 0, "new_rollouts": 0, "gradients": 0, "optimizer_updates": 0}
    if (entry["development_authorization_id"] != key or load_state(root, WORKFLOW)["current_stage"] != "scaffold"
            or any(type(scope.get(k)) is not type(v) or scope[k] != v for k, v in expected.items())):
        raise PermissionError("the explicit archived-only M4 v2 scope is required")
    return key


def reserve(root, scope, run_id, commit):
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common/"m4v2_m5_budgets"/f"{scope}.json"
    marker.parent.mkdir(exist_ok=True)
    write_json(marker, {"run_id": run_id, "source_commit": commit, "retry_allowed": False})


def run(*, root, experiment, stage, result_dir, run_id):
    from kron_reduce.evaluation.postfault_m4_v2 import (
        plot_energy_decomposition,
        plot_slow_coherency,
        separate_storage,
        slow_coherency,
    )

    assert_clean(root)
    if (stage != "analysis" or load_manifest(root)["experiments"][WORKFLOW] != experiment
            or result_dir != root/"results"/run_id or any(result_dir.iterdir())):
        raise RuntimeError("invalid M4 v2 dispatch")
    commit = git_head(root)
    relative = f"configs/experiments/{experiment['config']}"
    plan = read_plan(root, relative)
    key = _authorized(root, plan, experiment, relative)
    if not approval_recorded(load_state(root, WORKFLOW), stage) or sys.version_info[:2] != (3, 12):
        raise PermissionError("M4 v2 requires the recorded operator approval and Python 3.12")
    started = time.perf_counter()
    torch.set_num_threads(1)
    source, models, _, refs = frozen_models(root, plan)
    archived = prior_receipt(root, plan, "m4")
    reserve(root, key, run_id, commit)
    try:
        with closing(AppendOnlyAuditJournal(result_dir/"access.jsonl")) as ledger:
            path = _archive(root, ledger, archived, "graph.json", "prior_m4", "val", "coherency_reference")
            graph = read(path)
            for k, model in (("hard_r", models["hard_r"]), ("m3", models["m3"])):
                if not np.array_equal([r["stiffness"][k] for r in graph["edge_rows"]], model.effective_stiffness.tolist()):
                    raise ValueError("archived graph weights differ from the bound models")
            profiles = _profiles(root, ledger, source, models, refs)
        slow = slow_coherency(graph, models["m3"].mass.numpy())
        energy = separate_storage(profiles)
        plot_slow_coherency(slow, result_dir/"slow_coherency_comparison")
        plot_energy_decomposition(energy, result_dir/"energy_decomposition_auroc")
        write_json(result_dir/"slow_coherency.json", slow)
        write_json(result_dir/"energy_decomposition.json", energy)
        counts = {"fits": 0, "gradients": 0, "optimizer_updates": 0, "new_rollouts": 0, "new_simulations": 0,
                  "original_data_cache_opens": 0, "test_trajectory_reads": 0, "archive_artifact_opens": 13}
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit,
                    "plan_sha256": sha256_file(root/relative), "claim_boundary": "Sealed TASK3 post-fault train/val only.",
                    **counts}
        with closing(AppendOnlyAuditJournal(result_dir/"operations.jsonl")) as ledger:
            ledger.append({"stage": "complete", **counts})
        write_json(result_dir/"summary.json", evidence | {"frozen_models": refs["weights"],
                   "runtime": {"python": sys.version, "numpy": version("numpy"), "scipy": version("scipy"),
                               "torch": version("torch"), "wall_seconds": time.perf_counter()-started}})
        write_json(result_dir/"stage_evidence.json", evidence)
        if git_head(root) != commit:
            raise RuntimeError("source commit changed during M4 v2")
    finally:
        write_json(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
                   for p in sorted(result_dir.rglob("*")) if p.is_file()})


def main():
    root = project_root()
    assert_clean(root)
    entry = load_manifest(root)["experiments"][WORKFLOW]
    relative = f"configs/experiments/{entry['config']}"
    plan = read_plan(root, relative)
    _authorized(root, plan, entry, relative)
    approve(root, "analysis", WORKFLOW)
    with workflow_lock(root, WORKFLOW):
        print(_run_experiment(root, WORKFLOW, "analysis", experiment=entry), flush=True)
