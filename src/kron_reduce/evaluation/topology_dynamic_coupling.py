"""Paired synthetic selection of dynamic coupling and static restoring force."""
from __future__ import annotations

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_topology import energy_residual, make_model, scoring, simulate
from kron_reduce.evaluation.machine_local_topology_margin import matched_margins
from kron_reduce.evaluation.machine_local_topology_truth import parameters, truth_rollout
from kron_reduce.evaluation.topology_angle_damping_synthetic import _source_population, _stable_supported, _truth_classes
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _model(params, family, value, settings):
    model = make_model(params, "T0M1", "cuda").eval()
    model.set_angle_damping(2./3., spatial_mix=1.)
    if family == "dynamic":
        kappa = settings["stiffness"]
        model.set_dynamic_coupling(value/kappa, stiffness=kappa, limit=settings["limit"])
    else:
        model.set_restoring_shape(value)
    return model


@torch.no_grad()
def _fit(params, train, power, equilibrium, history_scale, target_scale, family, grid, settings, observer, budget):
    budget.spend("fits")
    attempts = []
    for value in grid:
        budget.spend("train_candidate_evaluations")
        model = _model(params, family, value, settings)
        target = model._tensor(train[:, 10:, :20])
        try:
            result = observer(model, train[:, :11, :10], train[:, :11, 10:20], power, equilibrium,
                              history_scale, 300, max_step=.0025, spend=budget.spend)["forecast"]
            error = ((result[:, 1:, :20]-target[:, 1:301])/model._tensor(target_scale)).square()
            loss = float(.5*(error[:, :100].mean()+error.mean()))
            if not np.isfinite(loss):
                raise FloatingPointError("nonfinite train candidate loss")
            attempts.append({"value": value, "loss": loss})
        except (FloatingPointError, torch.linalg.LinAlgError) as error:
            attempts.append({"value": value, "loss": None, "failure": str(error)})
    valid = [row for row in attempts if row["loss"] is not None]
    winner = min(valid, key=lambda row: (row["loss"], abs(row["value"]), row["value"])) if valid else None
    return {"value": winner["value"] if winner else 0., "qualified": len(valid) == len(grid), "attempts": attempts,
            "control_fallback": winner is None}


def _evaluate(params, family, value, settings, observer, directory, tag, cases, truth, labels,
              zero, power, eq, target_scale, history_scale, budget):
    model = _model(params, family, value, settings)
    p, equilibrium = model._tensor(power), model._tensor(eq)
    paths, histories, failures = simulate(model, truth, zero, p, equilibrium, history_scale, budget,
                                          oracle=False, observer=observer)
    score = scoring(model, cases, paths, histories, truth, zero, eq, params[0], target_scale, history_scale, labels, False)
    matched_margins(model, paths, truth, labels, equilibrium, score)
    score["failures"] = failures
    write_json(directory/f"{tag}-{family}-scores.json", score)
    np.savez_compressed(directory/f"{tag}-{family}-predictions.npz", forecasts=paths, histories=histories)
    return score


def _loss_by_role(model, role):
    rows = [row for row in model["events"] if row["role"] == role]
    if not rows or not all(row["qualified"] for row in rows):
        return None
    return float(np.mean([.5*(row["trajectory"]["1"]["nmse"]+row["trajectory"]["3"]["nmse"]) for row in rows]))


def scientific_checks(models, fit, truth, floor, limits):
    expected = "dynamic" if truth["rate"] else "static"
    candidate = models[expected]
    supported = [row for row in candidate["events"] if row["role"] in ("train", "val")
                 and not row["id"].startswith("exterior-")]
    checks = {"candidate_selection": fit[expected]["value"] == (truth["rate"] or truth["alpha"]),
              "feasible_trajectory": len(supported) == 27 and all(row["truth_label"] is not None and row["truth_agreement"]
                  and row["trajectory"]["8"]["nrmse"] <= limits["feasible_nrmse"] for row in supported)}
    if not truth["rate"] and not truth["alpha"]:
        checks["null_dynamic"] = fit["dynamic"]["value"] == 0.
        return checks
    other = models["static" if expected == "dynamic" else "dynamic"]
    losses = sorted(row["loss"] for row in fit[expected]["attempts"] if row["loss"] is not None)
    pairs = [(_loss_by_role(candidate, role), _loss_by_role(other, role), floor[role]) for role in ("train", "val")]
    stable, alternative = (_stable_supported(model, "val") for model in (candidate, other))
    roles = [(candidate["roles"][role], other["roles"][role]) for role in ("train", "val")]
    checks.update({
        "grid_separation": len(losses) == 5 and losses[0] <= limits["grid_ratio"]*losses[1] and losses[1] > 0.,
        "mechanism_separation": all(a is not None and b is not None and a <= limits["mechanism_ratio"]*b
                                   and b-a > limits["floor_factor"]*f for a, b, f in pairs),
        "stable_improvement": stable is not None and alternative is not None and stable <= limits["stable_ratio"]*alternative,
        "matched_witness": all(
            x["qualified"] == x["allocated"] > 0 and x["predicted_triggers"] >= y["predicted_triggers"]
            for x, y in ((a["matched_margin"]["unstable"], b["matched_margin"]["unstable"]) for a, b in roles)),
        "no_additional_false_alarm": all(a["stability"]["fp"] <= b["stability"]["fp"] for a, b in roles),
    })
    if truth["rate"]:
        checks["coupling_recovery"] = len(supported) == 27 and all(
            row.get("coupling", {}).get("coordinate_max_error") is not None
            and row["coupling"]["coordinate_max_error"] <= limits["coupling_error"] for row in supported)
    return checks


def execute(root, directory, plan, budget):
    source, cases = _source_population(root, plan, budget)
    settings = plan["settings"]
    params = parameters()
    mass, _, _, _, _, reference = params
    initial, power, eq = (source[key] for key in ("refined", "power", "equilibrium"))
    if not np.array_equal(eq, np.broadcast_to(reference, power.shape)):
        raise ValueError("synthetic dynamic equilibrium differs")
    q, v, z = initial[:, 0, :10], initial[:, 0, 10:20], initial[:, 0, 20:]
    zero = np.zeros_like(power)
    train = np.array(cases["roles"]) == "train"
    populations, freeze = {}, {}

    def observer(*args, **kwargs):
        return predict_from_topology_history(*args, **kwargs, memory_radius=plan["radius"],
                                             prior_weight=plan["prior_weight"])

    for tag, truth in plan["truths"].items():
        kappa, limit = settings["stiffness"], settings["limit"]
        dynamic = (truth["rate"]/kappa, kappa, limit) if truth["rate"] else None
        c = np.full(len(q), truth["start"]) if dynamic is not None else None
        fine, refined = (truth_rollout(params, q, v, power, zero, z, count=810, max_step=step, spend=budget.spend,
                        angle_damping_rho=2./3., spatial_mix=1., restoring_alpha=truth["alpha"],
                        dynamic_coupling=dynamic, coupling_initial=c) for step in (.00125, .000625))
        if truth["start"]:
            hscale, scale, fit = freeze["dynamic"]
        else:
            scale = np.sqrt(np.mean(refined[train, 11:311, :20]**2, axis=(0, 1))).clip(min=1e-5)
            hscale = np.sqrt(np.mean(refined[train, :11, :20]**2, axis=(0, 1))).clip(min=1e-5)
            fit = {family: _fit(params, refined[train], power[train], eq[train], hscale, scale, family, grid,
                                settings, observer, budget) for family, grid in plan["grids"].items()}
            freeze[tag] = hscale, scale, fit
        write_json(directory/f"{tag}-freeze.json", {"truth": truth, "fit": fit, "history_scale": hscale.tolist(),
                   "target_scale": scale.tolist(), "training_ids": [case for case, flag in zip(cases["ids"], train) if flag],
                   "fixed_start_audit": bool(truth["start"])})
        labels = _truth_classes(fine, refined, eq, mass, scale)
        step_error = ((fine[:, 11:311, :20]-refined[:, 11:311, :20])/scale)**2
        floor = {}
        for role in ("train", "val", "test"):
            error = step_error[np.array(cases["roles"]) == role]
            floor[role] = float(.5*(error[:, :100].mean()+error.mean()))
        write_json(directory/f"{tag}-truth-labels.json", labels)
        np.savez_compressed(directory/f"{tag}-truth.npz", fine=fine, refined=refined, power=power, equilibrium=eq)
        known = _model(params, "dynamic" if truth["rate"] else "static", truth["rate"] or truth["alpha"], settings)
        energy = energy_residual(known, known._tensor(refined[train, 10]), known._tensor(power[train]),
                                 known._tensor(eq[train]), budget.spend)
        models = {family: _evaluate(params, family, row["value"], settings, observer, directory, tag, cases, refined,
                                    labels, zero, power, eq, scale, hscale, budget)
                  for family, row in fit.items() if not truth["start"] or family == "dynamic"}
        dev = [row for model in models.values() for row in model["events"] if row["role"] != "test"]
        dynamic_dev = [row for row in models["dynamic"]["events"] if row["role"] != "test"]
        engineering = (energy <= 1e-10 and all(row["qualified"] for row in dev)
                       and all(row["qualified"] for row in fit.values())
                       and all("coupling" in row
                               and row["coupling"]["max_abs_start"] == row["coupling"]["origin_continuity_error"] == 0.
                               for row in dynamic_dev))
        checks = {} if truth["start"] else scientific_checks(models, fit, truth, floor, plan["gates"])
        populations[tag] = {"fit": fit, "models": {key: model["roles"] for key, model in models.items()},
                            "engineering_pass": engineering, "energy_error": energy, "gates": checks,
                            "truth_step_mse": floor,
                            "model_mse_1_3": {family: {role: _loss_by_role(model, role) for role in model["roles"]}
                                              for family, model in models.items()},
                            "fixed_start_audit": bool(truth["start"])}
        selected = {key: row["value"] for key, row in fit.items()}
        print(f"{tag}: selected {selected}, gates {checks}", flush=True)
    primary = [row for row in populations.values() if not row["fixed_start_audit"]]
    engineering = all(row["engineering_pass"] for row in primary)
    return {"engineering_pass": engineering, "scientific_pass": engineering and all(all(x["gates"].values()) for x in primary),
            "populations": populations, "claim_boundary": "Finite-grid comparison of common dynamic coupling against static restoring shape."}
