"""Preregistered screening, backbone curriculum, and backward elimination."""

from __future__ import annotations

import io
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
from kron_reduce.evaluation.kiki_first_swing import _witnesses
from kron_reduce.evaluation.kiki_machine_local_topology import scales_from_train
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.evaluation.unified_discovery_m3_fit import train_stage
from kron_reduce.evaluation.unified_discovery_m3_scores import (
    checkpoint,
    equivalent,
    quality,
    score,
    screening,
    witness_scores,
)
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from kron_reduce.models.unified_discovery_m3 import UnifiedDiscoveryModel
from project_harness.evidence import sha256_file

BASIS = ("edge_log_factor", "damping_log_factor")
MEMORY = ("port_parameter", "rate_parameter")


def _bind(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _write_model(root, path, model):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()})
    return _bind(root, path)


def _edge_changes(model):
    relative = (model.effective_stiffness/model.stiffness-1.).detach().cpu().numpy()
    incidence = model.incidence.detach().cpu().numpy()
    edges = [{"edge_index": i, "machines": np.flatnonzero(incidence[:, i]).tolist(), "relative_change": float(value)}
             for i, value in enumerate(relative)]
    top = sorted(edges, key=lambda row: (-abs(row["relative_change"]), row["edge_index"]))[:10]
    return {"all_edges": edges, "top10": top, "machine_9_in_top10": sum(9 in row["machines"] for row in top),
            "median_absolute_change": float(np.median(np.abs(relative))),
            "maximum_absolute_change": float(np.abs(relative).max())}


def _stage_fit(model, train, val, rows, scales, reference, settings, budget, directory, groups, anchors, updates):
    named = dict(model.named_parameters())
    active = [name for names, _ in groups for name in names]
    parameter_groups = [{"params": tuple(named[name] for name in names), "lr": rate} for names, rate in groups if names]
    targets = {name: anchors.get(name, named[name]).detach().clone() for name in active}

    def check(update):
        evaluation = score(model, val, scales, settings, budget)
        witnesses = witness_scores(model, rows, scales, settings, budget)
        return checkpoint(evaluation, witnesses, reference, settings, update) | {
            "edge_changes": _edge_changes(model), "physical_parameters": model.physical_parameters()}

    result = train_stage(model, train, scales, settings | {"updates": updates}, budget, directory,
                         parameter_groups=parameter_groups, anchor=targets, check=check)
    result["active_coefficients"] = sum(named[name].numel() for name in active)
    result["scheduled_updates"] = updates
    result["edge_changes"] = _edge_changes(model)
    return result


def _finalize(root, name, model, fit, train, val, scales, reference, settings, budget, path, checkpoint_path):
    model.requires_grad_(False).eval()
    binding = _write_model(root, checkpoint_path, model)
    scores = {role: score(model, events, scales, settings, budget, directory=path)
              for role, events in (("train", train), ("val", val))}
    witnesses = fit["selected_check"]["witness"]
    gate = checkpoint(scores["val"], witnesses, reference, settings, fit["selected_update"], train=scores["train"])
    for dt, fresh in gate["val"].items():
        earlier = fit["selected_check"]["val"][dt]
        metrics = ((fresh["stable_errors"][key], earlier["stable_errors"][key])
                   for key in ("mean_8s", "median_8s", "tn_mean_8s"))
        if (fresh["stability"] != earlier["stability"] or any(
                a != b and (a is None or b is None or not np.isclose(a, b, rtol=1e-9, atol=1e-11)) for a, b in metrics)):
            raise RuntimeError("frozen val score differs from its selected checkpoint")
    return {"name": name, "support": list(model.support), "checkpoint": binding, "gate": gate,
            "train": scores["train"], "physical_parameters": model.physical_parameters(),
            "selected_update": fit["selected_update"], "completed_updates": fit["completed_updates"],
            "scheduled_updates": fit["scheduled_updates"], "active_coefficients": fit["active_coefficients"],
            "stop_reason": fit["stop_reason"]}


def _eligible(candidate):
    return candidate["gate"]["passed"] and candidate["completed_updates"] == candidate["scheduled_updates"]


def _choose(candidates, tolerance):
    eligible = [row for row in candidates if _eligible(row)]
    if not eligible:
        return min(candidates, key=quality), False
    best = min(eligible, key=quality)
    while True:
        subset = [row for row in eligible if set(row["support"]) < set(best["support"])
                  and len(row["support"]) == len(best["support"])-1 and equivalent(row, best, tolerance)]
        if not subset:
            return best, True
        best = min(subset, key=quality)


def execute(root, directory, plan, source, assets, commit):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(Path(source["task3_root"])/"src"))
    started = time.perf_counter()
    settings, schedule = plan["settings"], plan["schedule"]
    reference = assets["witness_reference"]
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"], plan["scope_limits"])
    with closing(journal), closing(budget.stream):
        with budget.scope("anchors"):
            budget.spend("checkpoint_loads")
            backbone = LearnedNonlinearModel.load(io.BytesIO(assets["backbone"])).closure
            budget.spend("checkpoint_loads")
            with np.load(assets["memory_path"], allow_pickle=False) as memory:
                anchor = UnifiedDiscoveryModel(backbone, port=np.exp(memory["port_parameter"]), tau=np.full(10, .5),
                                               support=("G2",), module_values={"rho": 2./3.}).to("cuda")
                with torch.no_grad():
                    for name in MEMORY:
                        value = memory[name]
                        if value.dtype != np.float64 or value.shape != (10,) or not np.isfinite(value).all():
                            raise ValueError("invalid V4 memory coordinate")
                        getattr(anchor, name).copy_(anchor._tensor(value))
            if (anchor.stiffness.numel() != 45 or anchor.damping_log_factor.numel() != 9
                    or min(anchor.physical_parameters()["port_gain"]) < .01):
                raise ValueError("the registered 74-dimensional anchor is infeasible")
            anchors = {name: value.detach().clone() for name, value in anchor.named_parameters() if name in BASIS+MEMORY}
        reader = Task3FullReader(root, source["sources"] | {"allowed_roles": ["train", "val"]}, journal)
        train = reader.load("train")
        if Counter(e["label"] for e in train) != {"stable": 311, "unstable": 35}:
            raise RuntimeError("the frozen train class sizes changed")
        if not np.allclose(reader.static["mass"], backbone.mass, rtol=0, atol=1e-12):
            raise RuntimeError("TASK3 inertia differs from A06")
        scales = assets["preprocessing"]
        derived = scales_from_train(train, backbone.mass)
        if set(scales) != set(derived) or any(not np.allclose(derived[k], scales[k], rtol=1e-13, atol=1e-14) for k in scales):
            raise RuntimeError("normalization differs from the frozen train preprocessing")
        write_json(directory/"preprocessing.json", scales)
        initial = _write_model(root, directory/"initial.npz", anchor)
        write_json(directory/"initial.json", {"checkpoint_path": initial["path"], "checkpoint_sha256": initial["sha256"],
                   "preprocessing_path": _bind(root, directory/"preprocessing.json")["path"],
                   "preprocessing_sha256": sha256_file(directory/"preprocessing.json"), "support": ["G2"],
                   "source_commit": commit, "fitting_closed": False, "selection_closed": False})
        with budget.scope("screening"):
            screen = screening(anchor, train, scales, settings, plan["search"], budget, directory)
        val = reader.load("val", freeze_path=directory/"initial.json")
        if Counter(e["label"] for e in val) != {"stable": 95, "unstable": 12}:
            raise RuntimeError("the frozen val class sizes changed")
        rows, without_witness = _witnesses(train)
        backbone_model = anchor.with_support(("G2",))
        high, low = schedule["learning_rate"], schedule["backbone_learning_rate"]
        with budget.scope("curriculum_2a"):
            stage_a = _stage_fit(backbone_model, train, val, rows, scales, reference, settings, budget,
                                 directory/"curriculum_2a", ((BASIS, high),), anchors, schedule["curriculum_updates"])
            _write_model(root, directory/"backbone_54d.npz", backbone_model)
        with budget.scope("curriculum_2b"):
            stage_b = _stage_fit(backbone_model, train, val, rows, scales, reference, settings, budget,
                                 directory/"curriculum_2b", ((BASIS, low), (MEMORY, high)), anchors, schedule["curriculum_updates"])
            backbone_row = _finalize(root, "backbone_74d", backbone_model, stage_b, train, val, scales, reference, settings,
                                     budget, directory/"curriculum_2b", directory/"backbone_74d.npz")
        if (stage_a["active_coefficients"], stage_b["active_coefficients"]) != (54, 74):
            raise RuntimeError("the configured curriculum did not expose 54 then 74 coefficients")
        write_json(directory/"curriculum.json", {"stage_2a": stage_a, "stage_2b": stage_b, "backbone": backbone_row,
                   "without_witness": without_witness, "witness_reference": reference})

        starts = {module: row["best"]["parameters"] for module, row in screen["modules"].items() if row["status"] == "RETAINED"}
        support_fits = {}

        def fit_support(support):
            if support in support_fits:
                return support_fits[support]
            if len(support_fits) >= schedule["maximum_support_fits"]:
                raise RuntimeError("registered unique support budget exhausted")
            name = "support-"+("-".join(support) or "none")
            path = directory/name
            module_parameters = {name: value for module in support for name, value in starts[module].items()}
            model = backbone_model.with_support(support, module_parameters)
            groups = ((BASIS+MEMORY, low), (tuple(module_parameters), high))
            trust = {k: anchors[k] for k in BASIS} | {k: v.detach().clone() for k, v in backbone_model.named_parameters() if k in MEMORY}
            print(f"M3 fitting {name}", flush=True)
            fit = _stage_fit(model, train, val, rows, scales, reference, settings, budget, path, groups, trust,
                             schedule["support_updates"])
            candidate = _finalize(root, name, model, fit, train, val, scales, reference, settings, budget, path, path/"checkpoint.npz")
            write_json(path/"fit.json", fit)
            support_fits[support] = candidate
            return candidate

        with budget.scope("pruning"):
            full = tuple(sorted(starts))
            parent = fit_support(full)
            comparisons, path = [], [parent["name"]]
            while parent["support"]:
                eligible = []
                for module in parent["support"]:
                    child = fit_support(tuple(key for key in parent["support"] if key != module))
                    near = equivalent(child, parent, settings["parsimony_tolerance"])
                    acceptable = _eligible(child) and (not _eligible(parent) or near)
                    comparisons.append({"parent": parent["name"], "child": child["name"], "removed": module,
                                        "within_error_band": near, "eligible_to_continue": acceptable})
                    if acceptable:
                        eligible.append(child)
                if not eligible:
                    break
                parent = min(eligible, key=quality)
                path.append(parent["name"])
        candidates = [backbone_row, *support_fits.values()]
        selected, gate_pass = _choose(candidates, settings["parsimony_tolerance"])
        science = gate_pass and stage_a["completed_updates"] == schedule["curriculum_updates"]
        decision = "SELECTED" if science else "KEEP_BASELINE"
        write_json(directory/"pruning_table.json", {"full_support": list(full), "path": path, "comparisons": comparisons,
                   "models": candidates, "chosen": selected["name"], "decision": decision})
        parameter_path = directory/"freeze_m3.npz"
        payload = (root/selected["checkpoint"]["path"]).read_bytes()
        with parameter_path.open("xb") as stream:
            stream.write(payload)
        frozen = _bind(root, parameter_path)
        if frozen["sha256"] != selected["checkpoint"]["sha256"]:
            raise RuntimeError("the final parameter bytes differ from the selected candidate")
        freeze = directory/"freeze.json"
        calibration = _bind(root, directory/"preprocessing.json")
        write_json(freeze, {"checkpoint_path": frozen["path"], "checkpoint_sha256": frozen["sha256"],
            "preprocessing_path": calibration["path"], "preprocessing_sha256": calibration["sha256"],
            "candidate": selected["name"], "support": selected["support"], "selected_update": selected["selected_update"],
            "physical_parameters": selected["physical_parameters"], "backbone_anchor": source["backbone"],
            "source_commit": commit, "plan_sha256": sha256_file(root/plan["plan_path"]),
            "split_reference_sha256": plan["split_reference_sha256"], "selection_closed": True, "fitting_closed": True,
            "decision": decision, "scoring_steps_s": [.0025, .00125], "input_contract": "measured_history_100ms_zoh_v1",
            "test_trajectory_reads": 0})
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit, "scientific_pass": bool(science),
            "data_policy": "kiki_official_train_test_v1", "promotion_decision": "KEEP_BASELINE", "discovery_decision": decision,
            "plan_sha256": sha256_file(root/plan["plan_path"]), "test_trajectory_reads": 0,
            "frozen_model_manifest": _bind(root, freeze), "claim_boundary": "Complete TASK3 train and val discovery and freeze."}
        write_json(directory/"summary.json", evidence | {"selected": selected, "full_support": list(full),
            "screening_status": {m: row["status"] for m, row in screen["modules"].items()},
            "curriculum": {phase: {k: row[k] for k in ("active_coefficients", "completed_updates", "selected_update", "stop_reason")}
                           for phase, row in (("2a", stage_a), ("2b", stage_b))},
            "support_fits": len(support_fits), "operations": {name: budget.counts[name] for name in budget.limits},
            "operation_scopes": budget.scopes, "wall_seconds": time.perf_counter()-started,
            "runtime": {"python": sys.version, "executable": sys.executable, "torch": version("torch"),
                        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name()}})
        write_json(directory/"stage_evidence.json", evidence)
