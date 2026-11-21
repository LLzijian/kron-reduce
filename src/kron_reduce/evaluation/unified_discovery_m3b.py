"""Three bounded refits from the frozen M3 54-coordinate endpoint."""

from __future__ import annotations

import math
import sys
import time
from collections import Counter
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_first_swing import STEPS, _witnesses
from kron_reduce.evaluation.kiki_machine_local_topology import scales_from_train
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.evaluation.unified_discovery_m3 import (
    BASIS,
    MEMORY,
    _bind,
    _eligible,
    _finalize,
    _stage_fit,
    _write_model,
)
from kron_reduce.evaluation.unified_discovery_m3_scores import equivalent, quality
from kron_reduce.models.unified_discovery_m3 import MODULE_PARAMETERS, MODULES
from project_harness.evidence import sha256_file

SUPPORTS = (("G2", "G3", "G5"), ("G2", "G5"), ("G2",))
STEP_KEYS = tuple(map(str, STEPS))
RHO_START = 2./3.


def _screened_starts(screening):
    records = screening["modules"]
    if set(records) != set(MODULES):
        raise ValueError("M3b requires the archived comparison of all five M3 modules")
    summaries, starts = {}, {}
    for module in MODULES:
        row = records[module]
        changes = row.get("relative_rms_change", {})
        if set(changes) != set(STEP_KEYS) or any(
                not math.isfinite(change["unstable"]) for change in changes.values()):
            raise ValueError(f"incomplete archived dual-step screening for {module}")
        matched = row["status"] == "RETAINED" and all(change["unstable"] < 0. for change in changes.values())
        parameters = row["best"]["parameters"]
        if set(parameters) != set(MODULE_PARAMETERS[module]) or not all(math.isfinite(v) for v in parameters.values()):
            raise ValueError(f"invalid archived screening parameters for {module}")
        summaries[module] = {"archive_status": row["status"], "relative_rms_change": changes,
                             "archive_parameters": parameters, "strict_dual_step_improvement": matched}
        if matched:
            starts[module] = {"rho": RHO_START} if module == "G2" else parameters
    if tuple(starts) != SUPPORTS[0]:
        raise ValueError("archived screening does not produce the registered G2/G3/G5 support")
    return starts, summaries


def _train_detection(candidate, minimum):
    counts = {dt: row["stability"]["tp"] for dt, row in candidate["train"].items()}
    if set(counts) != set(STEP_KEYS):
        raise ValueError("the final train detection gate requires both declared steps")
    passed = all(count >= minimum for count in counts.values())
    gate = candidate["gate"]
    rank = gate["rank"]
    if not passed:
        rank = [max(rank[0], 2), *rank[1:]]
    candidate["gate"] = gate | {"gates": gate["gates"] | {"train_detection": passed},
        "rank": rank, "passed": gate["passed"] and passed, "relaxed_pass": gate["relaxed_pass"] and passed,
        "train_detection": {"threshold": minimum, "confirmed_by_step": counts, "passed": passed}}
    return candidate


def _max_fp(candidate):
    return max(row["stability"]["fp"] for row in candidate["gate"]["val"].values())


def _choose_finalists(candidates, tolerance):
    finalists = [candidates[support] for support in SUPPORTS[1:]]
    eligible = [row for row in finalists if _eligible(row)]
    if not eligible:
        return min(finalists, key=quality), False
    best = min(eligible, key=quality)
    simple = finalists[1]
    if (best is not simple and _eligible(simple) and _max_fp(simple) == _max_fp(best)
            and equivalent(simple, best, tolerance)):
        return simple, True
    return best, True


def _deletion(parent, child, removed, tolerance):
    near = equivalent(child, parent, tolerance)
    no_fp_increase = _max_fp(child) <= _max_fp(parent)
    parent_passed, child_passed = _eligible(parent), _eligible(child)
    return {"parent": parent["name"], "child": child["name"], "removed": removed,
            "parent_passed": parent_passed, "child_passed": child_passed, "within_error_band": near,
            "no_fp_increase": no_fp_increase,
            "eligible_deletion": child_passed and (not parent_passed or (near and no_fp_increase))}


def execute(root, directory, plan, source, assets, commit):
    settings, schedule = plan["settings"], plan["schedule"]
    if (settings["val_interval"], settings["max_fp"], settings["diagnostic_fp"], settings["min_train_tp"],
            schedule["support_updates"], schedule["learning_rate"], schedule["backbone_learning_rate"]) != (
            3, 2, 2, 33, 6, .001, .0002):
        raise ValueError("M3b fit schedule or thresholds differ from the registered paired comparison")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(Path(source["task3_root"])/"src"))
    started = time.perf_counter()
    starts, screened = _screened_starts(assets["screening"])
    scales, reference = assets["preprocessing"], assets["witness_reference"]
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"], plan["scope_limits"])
    with closing(journal), closing(budget.stream):
        with budget.scope("anchors"):
            budget.spend("checkpoint_loads")
            anchor = assets["model"].to(device="cuda", dtype=torch.float64).eval()
            if (anchor.support != ("G2",) or float(anchor.rho.detach()) != RHO_START or anchor.stiffness.numel() != 45
                    or anchor.damping_log_factor.numel() != 9 or min(anchor.physical_parameters()["port_gain"]) < .01):
                raise ValueError("M3b requires the feasible G2 endpoint from the 54-coordinate curriculum")
        write_json(directory/"screening_reuse.json", {"modules": screened, "module_initials": starts,
                   "new_screening_profiles": 0})
        reader = Task3FullReader(root, source["sources"] | {"allowed_roles": ["train", "val"]}, journal)
        train = reader.load("train")
        if Counter(e["label"] for e in train) != {"stable": 311, "unstable": 35}:
            raise RuntimeError("the frozen train class sizes changed")
        mass = anchor.mass.detach().cpu().numpy()
        if not np.allclose(reader.static["mass"], mass, rtol=0, atol=1e-12):
            raise RuntimeError("TASK3 inertia differs from the archived M3 model")
        derived = scales_from_train(train, mass)
        if set(scales) != set(derived) or any(
                not np.allclose(derived[key], scales[key], rtol=1e-13, atol=1e-14) for key in scales):
            raise RuntimeError("normalization differs from the archived train preprocessing")
        write_json(directory/"preprocessing.json", scales)
        prep = _bind(root, directory/"preprocessing.json")
        initial = _write_model(root, directory/"initial.npz", anchor)
        write_json(directory/"initial.json", {"checkpoint_path": initial["path"], "checkpoint_sha256": initial["sha256"],
            "preprocessing_path": prep["path"], "preprocessing_sha256": prep["sha256"], "support": ["G2"],
            "source_commit": commit, "fitting_closed": False, "selection_closed": False})
        val = reader.load("val", freeze_path=directory/"initial.json")
        if Counter(e["label"] for e in val) != {"stable": 95, "unstable": 12}:
            raise RuntimeError("the frozen val class sizes changed")
        rows, without_witness = _witnesses(train)
        anchors = {name: torch.zeros_like(value) for name, value in anchor.named_parameters() if name in BASIS}
        candidates = {}
        for support in SUPPORTS:
            name = "support-"+"-".join(support)
            path = directory/name
            modules = {name: value for group in support for name, value in starts[group].items()}
            model = anchor.with_support(support, modules)
            groups = ((BASIS+MEMORY, schedule["backbone_learning_rate"]), (tuple(modules), schedule["learning_rate"]))
            with budget.scope(name):
                fit = _stage_fit(model, train, val, rows, scales, reference, settings, budget, path, groups, anchors,
                                 schedule["support_updates"])
                if fit["active_coefficients"] != 74+len(modules):
                    raise RuntimeError("the fit did not expose the registered 74 base and active module coordinates")
                candidate = _finalize(root, name, model, fit, train, val, scales, reference, settings, budget, path,
                                      path/"checkpoint.npz")
            write_json(path/"fit.json", fit)
            candidates[support] = _train_detection(candidate, settings["min_train_tp"])

        selected, science = _choose_finalists(candidates, settings["parsimony_tolerance"])
        decision = "SELECTED" if science else "KEEP_BASELINE"
        comparisons = [_deletion(candidates[parent], candidates[child], removed, settings["parsimony_tolerance"])
                       for parent, child, removed in zip(SUPPORTS[:-1], SUPPORTS[1:], ("G3", "G5"), strict=True)]
        write_json(directory/"pruning_table.json", {"full_support": list(SUPPORTS[0]),
            "models": list(candidates.values()), "comparisons": comparisons,
            "finalists": [candidates[support]["name"] for support in SUPPORTS[1:]],
            "chosen": selected["name"], "decision": decision, "without_witness": without_witness})
        parameter_path = directory/"freeze_m3b.npz"
        with parameter_path.open("xb") as stream:
            stream.write((root/selected["checkpoint"]["path"]).read_bytes())
        frozen = _bind(root, parameter_path)
        if frozen["sha256"] != selected["checkpoint"]["sha256"]:
            raise RuntimeError("the final parameter bytes differ from the selected candidate")
        freeze = directory/"freeze.json"
        plan_digest = sha256_file(root/plan["plan_path"])
        write_json(freeze, {"checkpoint_path": frozen["path"], "checkpoint_sha256": frozen["sha256"],
            "preprocessing_path": prep["path"], "preprocessing_sha256": prep["sha256"],
            "initial_checkpoint": initial, "candidate": selected["name"], "support": selected["support"],
            "selected_update": selected["selected_update"], "physical_parameters": selected["physical_parameters"],
            "gates": selected["gate"]["gates"], "settings": settings, "schedule": schedule,
            "backbone_anchor": source["backbone"], "m3_curriculum_checkpoint_sha256": plan["backbone_sha256"],
            "source_contract_sha256": sha256_file(root/plan["source_contract"]),
            "source_commit": commit, "plan_sha256": plan_digest, "split_reference_sha256": plan["split_reference_sha256"],
            "selection_closed": True, "fitting_closed": True, "decision": decision, "scoring_steps_s": list(STEPS),
            "input_contract": "measured_history_100ms_zoh_v1", "test_trajectory_reads": 0})
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit,
            "scientific_pass": bool(science), "scientific_verdict": "ACCEPT" if science else "REJECT",
            "data_policy": "kiki_official_train_test_v1", "split_reference_sha256": plan["split_reference_sha256"],
            "promotion_decision": "KEEP_BASELINE", "discovery_decision": decision, "plan_sha256": plan_digest,
            "test_trajectory_reads": 0, "frozen_model_manifest": _bind(root, freeze),
            "claim_boundary": "Complete TASK3 train and val paired comparison and freeze."}
        write_json(directory/"summary.json", evidence | {"selected": selected, "full_support": list(SUPPORTS[0]),
            "module_initials": starts, "new_screening_profiles": 0, "support_fits": len(candidates),
            "operations": {key: budget.counts[key] for key in budget.limits}, "operation_scopes": budget.scopes,
            "wall_seconds": time.perf_counter()-started, "runtime": {"python": sys.version, "executable": sys.executable,
            "torch": version("torch"), "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name()}})
        write_json(directory/"stage_evidence.json", evidence)
