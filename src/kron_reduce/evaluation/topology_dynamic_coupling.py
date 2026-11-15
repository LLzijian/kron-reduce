"""Paired synthetic selection of dynamic coupling and static restoring force."""
from __future__ import annotations

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_topology import (
    _coupling_log_ratios,
    energy_residual,
    make_model,
    scoring,
    simulate,
)
from kron_reduce.evaluation.machine_local_topology_margin import matched_margins
from kron_reduce.evaluation.machine_local_topology_truth import (
    _incidence,
    parameters,
    truth_rollout,
)
from kron_reduce.evaluation.topology_angle_damping_synthetic import (
    _source_population,
    _stable_supported,
    _truth_classes,
)
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _model(params, family, value, settings):
    model = make_model(params, "T0M1", "cuda").eval()
    model.set_angle_damping(2./3., spatial_mix=1.)
    if family in ("dynamic", "spatial"):
        kappa = settings["stiffness"]
        limit = settings["spatial_limit"] if family == "spatial" else settings["limit"]
        model.set_dynamic_coupling(value/kappa, stiffness=kappa, limit=limit, spatial=family == "spatial")
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
              zero, power, eq, target_scale, history_scale, budget, *, truth_log_ratios=None,
              memory_recovery=False):
    model = _model(params, family, value, settings)
    p, equilibrium = model._tensor(power), model._tensor(eq)
    paths, histories, failures = simulate(model, truth, zero, p, equilibrium, history_scale, budget,
                                          oracle=False, observer=observer)
    score = scoring(model, cases, paths, histories, truth, zero, eq, params[0], target_scale, history_scale, labels, False,
                    truth_log_ratios=truth_log_ratios)
    if memory_recovery:
        end = 20+model.memory_dimension
        target = truth[:, (0, 10), 20:end]
        scale = np.maximum(.5, np.linalg.norm(target[:, 0], axis=-1))
        errors = np.linalg.norm(histories[:, :, (0, 10), 20:end]-target, axis=-1)/scale[:, None]
        for index, row in enumerate(score["events"]):
            values = errors[:, index]
            row["memory_relative_errors"] = values.tolist() if np.isfinite(values).all() else None
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
    if "spatial" in models:
        expected = "spatial" if truth["spatial"] else "dynamic"
    else:
        expected = "dynamic" if truth["rate"] else "static"
    candidate = models[expected]
    memory_limit = limits.get("memory_relative_error")
    all_feasible = memory_limit is not None
    support_size = 36 if all_feasible else 27
    supported = [row for row in candidate["events"] if row["role"] in ("train", "val")
                 and (all_feasible or not row["id"].startswith("exterior-"))]
    checks = {"candidate_selection": fit[expected]["value"] == (truth["rate"] or truth["alpha"]),
              "feasible_trajectory": len(supported) == support_size and all(row["truth_label"] is not None and row["truth_agreement"]
                  and row["trajectory"]["8"]["nrmse"] <= limits["feasible_nrmse"] for row in supported)}
    if all_feasible:
        checks["memory_recovery"] = len(supported) == support_size and all(
            row["memory_relative_errors"] is not None
            and max(map(max, row["memory_relative_errors"])) <= memory_limit for row in supported)
    if not truth["rate"] and not truth["alpha"]:
        checks["null_dynamic"] = all(row["value"] == 0. for row in fit.values())
        return checks
    others = [model for name, model in models.items() if name != expected]
    losses = sorted(row["loss"] for row in fit[expected]["attempts"] if row["loss"] is not None)
    pairs = [(_loss_by_role(candidate, role), _loss_by_role(other, role), floor[role])
             for other in others for role in ("train", "val")]
    stable = _stable_supported(candidate, "val", all_feasible=all_feasible)
    alternatives = [_stable_supported(other, "val", all_feasible=all_feasible) for other in others]
    roles = [(candidate["roles"][role], other["roles"][role]) for other in others for role in ("train", "val")]
    checks.update({
        "grid_separation": len(losses) == 5 and losses[0] <= limits["grid_ratio"]*losses[1] and losses[1] > 0.,
        "mechanism_separation": all(a is not None and b is not None and a <= limits["mechanism_ratio"]*b
                                   and b-a > limits["floor_factor"]*f for a, b, f in pairs),
        "stable_improvement": stable is not None and all(
            alternative is not None and stable <= limits["stable_ratio"]*alternative for alternative in alternatives),
        "matched_witness": all(
            x["qualified"] == x["allocated"] > 0 and x["predicted_triggers"] >= y["predicted_triggers"]
            for x, y in ((a["matched_margin"]["unstable"], b["matched_margin"]["unstable"]) for a, b in roles)),
        "no_additional_false_alarm": all(a["stability"]["fp"] <= b["stability"]["fp"] for a, b in roles),
    })
    if truth["rate"]:
        tolerance = limits["spatial_coupling_error"] if expected == "spatial" else limits["coupling_error"]
        checks["coupling_recovery"] = len(supported) == support_size and all(
            row.get("coupling", {}).get("coordinate_max_error") is not None
            and row["coupling"]["coordinate_max_error"] <= tolerance for row in supported)
    if "spatial" in models:
        checks["edge_recovery"] = all(row.get("coupling", {}).get("log_ratio_max_error") is not None
                                      and row["coupling"]["log_ratio_max_error"] <= limits["edge_error"] for row in supported)
    return checks


def _truth_geometry(truth, log_ratio, eq, params, roles):
    mass, stiffness = params[:2]
    incidence = _incidence()
    projection = np.eye(len(mass))-mass[:, None]/mass.sum()
    response = (projection @ incidence)/mass[:, None]
    ref = eq[:, None, :] @ incidence
    q = truth[:, 11:311, :len(mass)] @ incidence
    base_force = stiffness*(np.sin(ref+q)-np.sin(ref))
    base = base_force @ response.T
    log_ratio = log_ratio[:, 1:301]
    change = (base_force*np.expm1(log_ratio)) @ response.T
    base_norm = np.sum(base**2, axis=-1, keepdims=True)
    component = np.divide(np.sum(change*base, axis=-1, keepdims=True), base_norm,
                          out=np.zeros_like(base_norm), where=base_norm > 0.)
    perpendicular = change-component*base
    geometry = {}
    for role in ("train", "val", "test"):
        mask = np.asarray(roles) == role
        size = float(np.sum(change[mask]**2))
        geometry[role] = {"orthogonal_fraction": float(np.sum(perpendicular[mask]**2))/size if size else None,
                          "log_ratio_rms": float(np.sqrt(np.mean(log_ratio[mask]**2)))}
    return geometry


def execute(root, directory, plan, budget):
    source, cases = _source_population(root, plan, budget)
    settings = plan["settings"]
    params = parameters()
    mass, _, _, _, _, reference = params
    initial, power, eq = (source[key] for key in ("refined", "power", "equilibrium"))
    if not np.array_equal(eq, np.broadcast_to(reference, power.shape)):
        raise ValueError("synthetic dynamic equilibrium differs")
    q, v, z = initial[:, 0, :10], initial[:, 0, 10:20], initial[:, 0, 20:]
    memory_recovery = "memory_relative_error" in plan["gates"]
    if memory_recovery and np.linalg.norm(z, axis=-1).max() > plan["radius"]/2.+1e-12:
        raise ValueError("known truth exceeds the signed half-radius support")
    zero = np.zeros_like(power)
    train = np.array(cases["roles"]) == "train"
    compare_spatial = "spatial" in plan["grids"]
    basis = None
    if compare_spatial:
        basis = _model(params, "spatial", 0., settings).dynamic_edge_basis.cpu().numpy()
        if (np.max(np.abs(basis.sum(axis=0))) > 1e-12 or not np.isclose(
                np.linalg.norm(basis, axis=1).max()*settings["spatial_limit"], np.log(2.), rtol=0., atol=1e-12)):
            raise ValueError("spatial log redistribution basis violates zero sum or the frozen edge bound")
    populations, freeze = {}, {}

    def observer(*args, **kwargs):
        return predict_from_topology_history(*args, **kwargs, memory_radius=plan["radius"],
                                             prior_weight=plan["prior_weight"])

    for tag, truth in plan["truths"].items():
        spatial_truth = truth.get("spatial", False)
        kappa = settings["stiffness"]
        limit = settings["spatial_limit"] if spatial_truth else settings["limit"]
        dynamic = (truth["rate"]/kappa, kappa, limit) if truth["rate"] else None
        c = np.full(len(q), truth["start"]) if dynamic is not None and not spatial_truth else None
        fine, refined = (truth_rollout(params, q, v, power, zero, z, count=810, max_step=step, spend=budget.spend,
                        angle_damping_rho=2./3., spatial_mix=1., restoring_alpha=truth["alpha"],
                        dynamic_coupling=dynamic, coupling_initial=c, spatial_coupling=spatial_truth)
                        for step in (.00125, .000625))
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
        family = "spatial" if spatial_truth else "dynamic" if truth["rate"] or compare_spatial else "static"
        known = _model(params, family, truth["rate"] or truth["alpha"], settings)
        truth_log = None
        if compare_spatial:
            truth_log = (_coupling_log_ratios(refined[:, 10:, 30:], limit, basis if spatial_truth else None)
                         if truth["rate"] else np.zeros((len(q), 801, 1)))
            energy_state = refined[train, 10]
            if not truth["rate"]:
                energy_state = np.pad(energy_state, ((0, 0), (0, 1)))
        else:
            energy_state = refined[train, 10]
        energy = energy_residual(known, known._tensor(energy_state), known._tensor(power[train]),
                                 known._tensor(eq[train]), budget.spend)
        models = {family: _evaluate(params, family, row["value"], settings, observer, directory, tag, cases, refined,
                                    labels, zero, power, eq, scale, hscale, budget, truth_log_ratios=truth_log,
                                    memory_recovery=memory_recovery)
                  for family, row in fit.items() if not truth["start"] or family == "dynamic"}
        if compare_spatial:
            models["baseline"] = _evaluate(params, "baseline", 0., settings, observer, directory, tag, cases, refined,
                                           labels, zero, power, eq, scale, hscale, budget)
        dev = [row for model in models.values() for row in model["events"] if row["role"] != "test"]
        dynamic_dev = [row for name in ("dynamic", "spatial") if name in models
                       for row in models[name]["events"] if row["role"] != "test"]
        engineering = (energy <= 1e-10 and all(row["qualified"] for row in dev)
                       and all(row["qualified"] for row in fit.values())
                       and all("coupling" in row
                               and row["coupling"]["max_abs_start"] == row["coupling"]["origin_continuity_error"] == 0.
                               for row in dynamic_dev))
        checks = {} if truth["start"] else scientific_checks(models, fit, truth, floor, plan["gates"])
        geometry = {}
        if compare_spatial:
            invariants = {"dynamic": "log_spatial_rms_max", "spatial": "edge_log_sum_max_abs"}
            checks["zero_sum_and_common_invariants"] = all(
                row.get("coupling", {}).get(key) is not None and row["coupling"][key] <= 1e-12
                and row["coupling"]["min_multiplier"] >= .5-1e-12 and row["coupling"]["max_multiplier"] <= 2.+1e-12
                for name, key in invariants.items() for row in models[name]["events"] if row["role"] != "test")
            geometry = _truth_geometry(refined, truth_log, eq, params, cases["roles"])
            if truth["rate"]:
                checks["truth_excitation"] = all(geometry[role]["log_ratio_rms"] >= plan["gates"]["excitation"]
                                                 for role in ("train", "val"))
            if spatial_truth:
                checks["orthogonal_force"] = all(geometry[role]["orthogonal_fraction"] is not None
                                                 and geometry[role]["orthogonal_fraction"] >= plan["gates"]["orthogonal"]
                                                 for role in ("train", "val"))
        populations[tag] = {"fit": fit, "models": {key: model["roles"] for key, model in models.items()},
                            "engineering_pass": engineering, "energy_error": energy, "gates": checks,
                            "truth_step_mse": floor,
                            "truth_geometry": geometry,
                            "model_mse_1_3": {family: {role: _loss_by_role(model, role) for role in model["roles"]}
                                              for family, model in models.items()},
                            "fixed_start_audit": bool(truth["start"])}
        selected = {key: row["value"] for key, row in fit.items()}
        print(f"{tag}: selected {selected}, gates {checks}", flush=True)
    primary = [row for row in populations.values() if not row["fixed_start_audit"]]
    engineering = all(row["engineering_pass"] for row in primary)
    return {"engineering_pass": engineering, "scientific_pass": engineering and all(all(x["gates"].values()) for x in primary),
            "populations": populations,
            "claim_boundary": ("Finite-grid comparison of common scale and zero-sum log stiffness redistribution."
                               if compare_spatial else "Finite-grid comparison of common dynamic coupling against static restoring shape.")}
