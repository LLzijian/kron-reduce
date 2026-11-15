"""Fixed-weight, paired TASK3 memory-radius comparison for T0M1."""

from __future__ import annotations

import math
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import COUNTS, Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_r20_r21 import read_json, verified
from kron_reduce.evaluation.kiki_machine_local_topology import score
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head


def _assets(root, frozen):
    return {key: verified(root, {"path": frozen[key+"_path"], "sha256": frozen[key+"_sha256"]})
            for key in ("checkpoint", "preprocessing")}


def _freeze(root, plan):
    path = verified(root, plan["frozen_model_manifest"])
    frozen = read_json(path)
    assets = _assets(root, frozen)
    scales = read_json(assets["preprocessing"])
    if set(scales) != {"history", "target", "memory_radius"}:
        raise RuntimeError("invalid frozen preprocessing")
    scales["memory_radius"] *= frozen["memory_radius_multiplier"]
    if (not math.isfinite(scales["memory_radius"]) or scales["memory_radius"] <= 0
            or any(np.shape(scales[key]) != (20,) or not np.isfinite(scales[key]).all()
                   or not np.all(np.asarray(scales[key]) > 0) for key in ("history", "target"))):
        raise RuntimeError("invalid doubled memory radius or scales")
    return path, frozen, assets, scales


def _reference(root, plan):
    if set(plan["reference"]) != set(COUNTS):
        raise RuntimeError("all three saved T0M1 reference scores are required")
    return {role: read_json(verified(root, plan["reference"][role])) for role in COUNTS}


def _validate_result(result, reference, events, radius, *, multiplier=2., group="R2"):
    ids = [event["id"] for event in events]
    expected_truth = [(event["id"], event["label"]) for event in events]
    expected_initializations = {(event_id, step) for event_id in ids for step in (.0025, .00125)}
    for name, row in (("candidate", result), ("reference", reference)):
        stability = row["stability"]
        actual_initializations = [(item["id"], item["step"]) for item in row["initialization"]]
        if (stability["allocated"] != len(ids) or len(row["stability_events"]) != len(ids)
                or [(item["id"], item["truth_label"]) for item in row["stability_events"]] != expected_truth
                or len(set(actual_initializations)) != len(actual_initializations)
                or not set(actual_initializations) <= expected_initializations
                or any(not math.isfinite(item["hidden_start_energy"])
                       or item["hidden_start_energy"] < 0 for item in row["initialization"])
                or (name == "reference" and set(actual_initializations) != expected_initializations)):
            raise RuntimeError(f"{name} does not match the complete frozen cohort")
        if (stability["tp"] + stability["tn"] + stability["fp"] + stability["fn"]
                + stability["unresolved"] + stability["uncovered"] != len(ids)):
            raise RuntimeError(f"{name} stability counts do not partition the cohort")

    def cap_counts(row, limit):
        by_step = {}
        for step in (.0025, .00125):
            values = [item for item in row["initialization"] if item["step"] == step]
            assessed = len(values)
            at_radius = sum(math.sqrt(2*item["hidden_start_energy"]) >= limit*(1-1e-7)
                            for item in values)
            by_step[str(step)] = {"allocated": len(ids), "assessed": assessed,
                                 "missing": len(ids)-assessed, "at_radius": at_radius,
                                 "fraction_of_assessed": at_radius/assessed if assessed else None}
        return by_step

    return {"reference": cap_counts(reference, radius/multiplier), group: cap_counts(result, radius)}


def execute(root, directory, plan, stage):
    """Score one immutable candidate against saved baseline scores; never fit or select."""
    if stage != "test" or "allowed_roles" in plan["sources"]:
        raise RuntimeError("radius comparison only permits the frozen one-stage test")
    root, directory = Path(root).resolve(), Path(directory)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(plan["task3_root"])/"src"))
    freeze_path, frozen, assets, scales = _freeze(root, plan)
    group_name = plan.get("group", "R2")
    references = _reference(root, plan)
    started = time.perf_counter()
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"])
    development_journal = AppendOnlyAuditJournal(directory/"development-access.jsonl")
    test_journal = AppendOnlyAuditJournal(directory/"test-access.jsonl")
    try:
        development = Task3FullReader(root, plan["sources"] | {"allowed_roles": ["train", "val"]},
                                      development_journal)
        held_out = Task3FullReader(root, plan["sources"] | {"allowed_roles": ["test"]}, test_journal)
        budget.spend("checkpoint_loads")
        backbone = LearnedNonlinearModel.load(verified(root, plan["backbone"])).closure
        model = make_model(backbone, "T0M1", train_scales=True, device="cuda")
        budget.spend("checkpoint_loads")
        load_snapshot(model, assets["checkpoint"])
        if group_name == "momentum_damping":
            factor = frozen["settings"]["damping_multiplier"]
            original_damping = model.effective_damping.clone()
            with torch.no_grad():
                model.log_damping_factor.add_(math.log(factor))
            torch.testing.assert_close(model.effective_damping, factor*original_damping, rtol=1e-13, atol=1e-13)
        roles = {"train": development.load("train"),
                 "val": development.load("val", freeze_path=freeze_path),
                 "test": held_out.load("test", freeze_path=freeze_path)}
        if (not np.allclose(development.static["mass"], backbone.mass, rtol=0, atol=1e-12)
                or not np.array_equal(development.static["mass"], held_out.static["mass"])):
            raise RuntimeError("TASK3 mass differs from frozen model")
        group = directory/group_name
        group.mkdir()
        records, cap_counts, train_witnesses = {}, {}, None
        for role, events in roles.items():
            if len(events) != COUNTS[role]:
                raise RuntimeError(f"incomplete {role} cohort")
            result = score(model, events, scales, budget, group, role, frozen["settings"])
            if role == "train" and frozen["settings"].get("matched_train_witnesses"):
                witnesses = result["matched_train_witnesses"]
                train_witnesses = {"allocated": len(witnesses),
                                   "truth_triggered": sum(row["truth"]["triggered"] is True for row in witnesses),
                                   "predicted": {str(step): sum(row["prediction"][str(step)] is not None
                                                                and row["prediction"][str(step)]["triggered"] is True
                                                                for row in witnesses) for step in (.0025, .00125)}}
            cap_counts[role] = _validate_result(result, references[role], events, scales["memory_radius"],
                                                multiplier=frozen["memory_radius_multiplier"], group=group_name)
            records[role] = {key: result[key] for key in
                             ("summary", "unstable_prefix", "stable_complete", "counts",
                              "stability", "task3_six_term_score")}
        val, old_val = records["val"], references["val"]
        stable8 = val["stable_complete"]["8"].get("mean")
        reference_stable8 = old_val["stable_complete"]["8"].get("mean")
        s = val["stability"]
        stable_coverage = all(row["valid"] is True and row["n_events"] == 95
                              and row["qualified"] == 95 and row["unresolved"] == 0
                              for row in (val["stable_complete"]["8"],
                                          old_val["stable_complete"]["8"]))
        val_guard = bool(val["task3_six_term_score"] is not None
                         and math.isfinite(val["task3_six_term_score"])
                         and val["task3_six_term_score"] <= .5 and s["fp"] == 0
                         and s["unresolved"] <= int(group_name == "R2")
                         and s["uncovered"] == s["numerical_failures"] == 0
                         and 107-s["tp"]-s["tn"] <= 3
                         and stable_coverage
                         and stable8 is not None and reference_stable8 is not None
                         and math.isfinite(stable8) and math.isfinite(reference_stable8)
                         and stable8 <= 1.05*reference_stable8)
        train_guard = None
        if train_witnesses is not None:
            train = records["train"]["stability"]
            train_guard = (train_witnesses["allocated"] == train_witnesses["truth_triggered"] == 35
                           and min(train_witnesses["predicted"].values()) > 6 and train["tp"] >= 24 and train["fp"] <= 2
                           and train["unresolved"] <= 8 and train["uncovered"] == train["numerical_failures"] == 0)
        test = records["test"]["stability"]
        test_goal_met = bool(test["allocated"] == test["resolved"] == 76 and test["accuracy"] is not None
                             and test["numerical_failures"] == 0
                             and test["tp"]+test["tn"] >= 69)
        evidence = {"status": "PASS", "process_status": "PASS", "engineering_pass": True,
                    "source_commit": git_head(root), "data_policy": "kiki_official_train_test_v1",
                    "promotion_decision": "KEEP_BASELINE", "scientific_pass": test_goal_met,
                    "claim_boundary": "Frozen one-factor comparison on previously exposed TASK3 cohorts."}
        summary = evidence | {"group": group_name, "records": {group_name: records},
                              "reference": {role: {key: references[role][key] for key in
                                      ("summary", "unstable_prefix", "stable_complete", "counts",
                                       "stability", "task3_six_term_score")}
                                            for role in COUNTS},
                              "radius_freeze": plan["frozen_model_manifest"],
                              "cap_counts": cap_counts, "val_guard": val_guard,
                              "test_goal_met": test_goal_met,
                              "operations": dict(budget.counts),
                              "wall_seconds": time.perf_counter()-started,
                              "runtime": {"python": sys.version, "executable": sys.executable,
                                  "device": "cuda", "cuda_build": torch.version.cuda,
                                  "gpu": torch.cuda.get_device_name(),
                                  **{package: version(package) for package in
                                     ("torch", "numpy", "scipy", "filelock", "h5py")}}}
        if train_guard is not None:
            summary.update(train_guard=train_guard, train_witnesses=train_witnesses)
        if group_name == "momentum_damping":
            summary["damping_multiplier"] = frozen["settings"]["damping_multiplier"]
        write_json(directory/"summary.json", summary)
        write_json(directory/"stage_evidence.json", evidence)
        return summary
    finally:
        development_journal.close()
        test_journal.close()
        budget.stream.close()
        write_json(directory/"artifact_hashes.json", {path.relative_to(directory).as_posix(): sha256_file(path)
                   for path in sorted(directory.rglob("*")) if path.is_file()})
