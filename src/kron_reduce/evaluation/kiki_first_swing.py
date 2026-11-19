"""First paired TASK3 fit and frozen train/val evidence."""

from __future__ import annotations

import io
import json
import math
import sys
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.first_swing_loss import (
    objective_terms,
    observed_velocity_increment,
    training_weights,
)
from kron_reduce.evaluation.kiki_machine_local_full import inputs, train_updates
from kron_reduce.evaluation.kiki_machine_local_topology import freeze_pair, scales_from_train, score
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model
from kron_reduce.evaluation.kiki_topology_momentum import (
    MAX_FRAME,
    ORIGIN_FRAME,
    _separation_window,
)
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.models.machine_local_topology import predict_from_topology_history
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from project_harness.git_state import git_head

GROUPS = ("rollout_control", "first_swing")
STEPS = (.0025, .00125)


def checkpoint_metrics(result, settings, update):
    stability, stable = result["stability"], result["stable_complete"]["8"]
    error = stable.get("mean")
    six = result["task3_six_term_score"]
    numerical = stability["uncovered"]+stability["numerical_failures"]
    valid = (stability["allocated"] == 107 and numerical == 0 and stable["qualified"] == 95
             and error is not None and math.isfinite(error) and six is not None and math.isfinite(six))
    passed = (valid and stability["tp"] >= settings["min_val_tp"] and stability["fp"] <= settings["max_val_fp"]
              and stability["unsettled_predictions"] <= settings["max_val_unsettled"]
              and error <= settings["max_val_stable_nrmse"])
    rank = (0 if passed else 1 if valid else 2, numerical, -stability["tp"], stability["unsettled_predictions"],
            stability["fp"], error if error is not None and math.isfinite(error) else None,
            six if six is not None and math.isfinite(six) else None, update)
    return {"passed": passed, "rank": list(rank), "stability": stability,
            "stable_8s": stable, "task3_six_term_score": six}


def _gradient_audit(model, events, scales, budget, settings, weights):
    names = ("weighted_rollout", "one_sided_velocity", "weak_momentum")
    parameters = tuple(model.parameters())
    accumulated = {key: torch.cat([torch.zeros_like(p).flatten() for p in parameters]) for key in names}
    for offset in range(0, len(events), settings["train_batch"]):
        batch = events[offset:offset+settings["train_batch"]]
        objective, terms = objective_terms(model, batch, len(events), scales, budget, settings, weights=weights)
        for key in names:
            if not terms[key].requires_grad:
                continue
            budget.spend("gradients")
            gradients = torch.autograd.grad(terms[key], parameters, retain_graph=key != names[-1], allow_unused=True)
            accumulated[key] += torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                            for g, p in zip(gradients, parameters, strict=True)])
        del objective, terms
    if not all(torch.isfinite(g).all() for g in accumulated.values()):
        raise FloatingPointError("nonfinite first-swing gradient audit")
    norms = {name: float(g.norm()) for name, g in accumulated.items()}
    combined = (accumulated[names[0]]+settings["lambda_asym"]*accumulated[names[1]]
                +settings["lambda_momentum"]*accumulated[names[2]])
    return {name: {"norm": norms[name],
                   "cosine_to_rollout": float(g@accumulated[names[0]])/(norms[name]*norms[names[0]])
                   if norms[name] and norms[names[0]] else None,
                   "directional_derivative_against_weighted_first_swing_gradient": -float(g@combined)}
            for name, g in accumulated.items()}


def _fit(model, train, val, scales, budget, directory, settings, weights):
    bounds = (math.log(.5), math.log(2.))

    def objective(candidate, batch, population, preprocessing, ledger):
        return objective_terms(candidate, batch, population, preprocessing, ledger, settings, weights=weights)

    updates = train_updates(model, train, scales, budget, directory, settings,
                            stiffness_bounds=bounds, damping_bounds=bounds, objective_fn=objective,
                            minimum_port=settings["minimum_port"], learning_rate=settings["learning_rate"],
                            gradient_clip=1.)
    best = None
    status = {"status": "COMPLETED", "completed_updates": 0, "message": None}
    with closing(updates):
        for update in range(settings["updates"]+1):
            if update:
                try:
                    step = next(updates)
                    status["completed_updates"] = step["update"]
                except StopIteration as stopped:
                    status = stopped.value
                    break
            if update % settings["val_interval"] and update != settings["updates"]:
                continue
            budget.spend("val_checks")
            row = checkpoint_metrics(score(model, val, scales, budget, directory, "val", settings, save=False),
                                     settings, update)
            write_json(directory/f"val-check-{update:03d}.json", row)
            rank = tuple(math.inf if value is None else value for value in row["rank"])
            if best is None or rank < best[0]:
                best = (rank, snapshot(model), row)
    model.load_state_dict(best[1])
    model.requires_grad_(False).eval()
    return status | {"selected_update": best[0][-1], "selected_val": best[2],
                     "stop_reason": "update_cap" if status["status"] == "COMPLETED" else "numerical_failure"}


def _witnesses(train):
    rows, missing = [], []
    for event in train:
        if event["label"] != "unstable":
            continue
        path = event["x"][:MAX_FRAME]
        window = _separation_window(event, path)
        if window is None:
            missing.append(event["id"])
        else:
            rows.append((event, window))
    if len(rows) != 33 or len(missing) != 2:
        raise RuntimeError("frozen train separation witness population changed")
    return rows, missing


@torch.no_grad()
def _witness_scores(model, rows, scales, budget, settings):
    events = [event for event, _ in rows]
    lengths = np.array([window["witness_end"]-ORIGIN_FRAME for _, window in rows])
    count = int(lengths.max())
    data = [event["x"][ORIGIN_FRAME:window["witness_end"]+1] for event, window in rows]
    path = model._tensor(np.stack([np.pad(value, ((0, count+1-len(value)), (0, 0)), mode="edge") for value in data]))
    q, v, power, equilibrium, scale = map(model._tensor, inputs(events, scales["history"]))
    results = {}
    for step in STEPS:
        try:
            origin = predict_from_topology_history(model, q, v, power, equilibrium, scale, 0, max_step=step,
                spend=budget.spend, memory_radius=scales["memory_radius"], prior_weight=settings["prior_weight"])
            impulse = observed_velocity_increment(model, path, origin["history"][:, -1, 20:], power, equilibrium,
                                                 lengths, step=step, spend=budget.spend)
            residual = (path[..., 10:]-path[:, :1, 10:]-impulse).cpu().numpy()
            details = []
            for index, (event, window) in enumerate(rows):
                high, low = window["machines"]
                signed = float(residual[index, lengths[index], high]-residual[index, lengths[index], low])
                details.append({"id": event["id"], **window, "signed_residual_rad_s": signed})
            values = [row["signed_residual_rad_s"] for row in details]
            results[str(step)] = {"qualified": len(details), "rows": details, "positive": sum(x > 0. for x in values),
                                  "signed_median": float(np.median(values)),
                                  "absolute_median": float(np.median(np.abs(values)))}
        except NUMERICAL_ERRORS as error:
            results[str(step)] = {"qualified": 0, "error": str(error), "signed_median": None,
                                  "absolute_median": None, "rows": [{"id": event["id"], **window,
                                  "signed_residual_rad_s": None} for event, window in rows]}
    return results


def _witness_gate(reference, candidate, settings):
    return all(base["qualified"] == trial["qualified"] == 33
               and abs(base["signed_median"]-settings["reference_witness_median"]) <= settings["witness_reference_tolerance"]
               and abs(trial["signed_median"]) <= settings["witness_relative_limit"]*abs(base["signed_median"])
               and trial["absolute_median"] <= base["absolute_median"]
               for base, trial in zip(reference.values(), candidate.values(), strict=True))


def execute(root, directory, plan, source, assets):
    """Run only after the one-use development scope is reserved."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(source["task3_root"])/"src"))
    started = time.perf_counter()
    plan = plan | {"backbone": source["backbone"]}
    settings = plan["settings"]
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"])
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    try:
        budget.spend("checkpoint_loads")
        backbone = LearnedNonlinearModel.load(io.BytesIO(assets["backbone"])).closure
        budget.spend("checkpoint_loads")
        anchor = make_model(backbone, "T0M1", train_scales=True, device="cuda")
        anchor.set_angle_damping(2./3., spatial_mix=1.)
        load_snapshot(anchor, io.BytesIO(assets["checkpoint"]))
        if torch.any(anchor.port_parameter.exp() < settings["minimum_port"]):
            raise RuntimeError("the frozen anchor violates the registered port floor")
        reader = Task3FullReader(root, source["sources"] | {"allowed_roles": ["train", "val"]}, journal)
        train = reader.load("train")
        if not np.allclose(reader.static["mass"], backbone.mass, rtol=0, atol=1e-12):
            raise RuntimeError("TASK3 and frozen T0M1 inertia differ")
        scales = json.loads(assets["preprocessing"])
        derived = scales_from_train(train, backbone.mass)
        if set(scales) != set(derived) or any(not np.allclose(scales[key], derived[key], rtol=1e-13, atol=1e-14)
                                              for key in derived):
            raise RuntimeError("train normalization differs from the shared frozen anchor")
        weights = training_weights(train, settings["large_event_weight"], settings["history_kinetic_fraction"])
        write_json(directory/"preprocessing.json", scales)
        write_json(directory/"training_population.json", {"weights": weights, "history_points": 11,
            "weak_window_eligible": [sum(len(event["x"])-11 >= count for event in train) for count in (20, 40)]})
        start = snapshot(anchor)
        models = {}
        for group in GROUPS:
            model = make_model(backbone, "T0M1", train_scales=True, device="cuda")
            model.set_angle_damping(2./3., spatial_mix=1.)
            model.load_state_dict(start)
            models[group] = model
            (directory/group).mkdir()
        val = reader.load("val", freeze_path=freeze_pair(root, directory, "initial", models, plan, None))
        write_json(directory/"initial_gradients.json",
                   _gradient_audit(models["first_swing"], train, scales, budget, settings, weights))
        training = {group: _fit(model, train, val, scales, budget, directory/group, settings,
                                weights if group == "first_swing" else None) for group, model in models.items()}
        freeze = freeze_pair(root, directory, "freeze", models, plan, training)

        populations = {"train": train, "val": val}
        records = {}
        for group, model in models.items():
            records[group] = {}
            for role, events in populations.items():
                result = score(model, events, scales, budget, directory/group, role, settings)
                records[group][role] = {key: result[key] for key in
                                        ("stable_complete", "unstable_prefix", "stability", "task3_six_term_score")}
                records[group][role]["max_origin_memory_energy"] = max(
                    (row["hidden_origin_energy"] for row in result["initialization"]), default=None)
            records[group]["port_gains"] = model.port_parameter.detach().exp().tolist()
        witness_rows, without_witness = _witnesses(train)
        witnesses = {name: _witness_scores(model, witness_rows, scales, budget, settings)
                     for name, model in {"frozen_anchor": anchor, **models}.items()}
        paired_effect = {}
        for step in map(str, STEPS):
            control, trial = (witnesses[group][step] for group in GROUPS)
            change = [abs(b["signed_residual_rad_s"])-abs(a["signed_residual_rad_s"])
                      for a, b in zip(control["rows"], trial["rows"], strict=True)
                      if a["signed_residual_rad_s"] is not None and b["signed_residual_rad_s"] is not None]
            paired_effect[step] = {"qualified": len(change), "improved": sum(x < 0 for x in change),
                                   "median_absolute_residual_change_rad_s": float(np.median(change)) if change else None}
        write_json(directory/"train-witnesses.json", {"without_witness": without_witness, "comparisons": witnesses,
                   "first_swing_minus_rollout_control": paired_effect})
        witness_pass = _witness_gate(witnesses["frozen_anchor"], witnesses["first_swing"], settings)
        fit_pass = all(row["status"] == "COMPLETED" and row["completed_updates"] == settings["updates"]
                       for row in training.values())
        candidate = training["first_swing"]
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": git_head(root),
                    "data_policy": "kiki_official_train_test_v1", "promotion_decision": "KEEP_BASELINE",
                    "claim_boundary": "Fixed TASK3 train and val, causal first-swing objective ablation.",
                    "scientific_pass": (fit_pass and candidate["selected_update"] > 0
                                        and candidate["selected_val"]["passed"] and witness_pass)}
        write_json(directory/"summary.json", evidence | {"freeze_path": freeze.name, "records": records,
            "training": training, "witness_pass": witness_pass, "paired_witness_effect": paired_effect,
            "operations": dict(budget.counts),
            "wall_seconds": time.perf_counter()-started, "runtime": {"python": sys.version,
            "executable": sys.executable, "gpu": torch.cuda.get_device_name(), "cuda_build": torch.version.cuda,
            **{package: version(package) for package in ("torch", "numpy", "scipy", "filelock", "h5py")}}})
        write_json(directory/"stage_evidence.json", evidence)
    finally:
        budget.stream.close()
        journal.close()
