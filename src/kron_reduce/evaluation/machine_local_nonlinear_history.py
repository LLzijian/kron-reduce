"""Clean nonlinear history S0, including frozen train/val/test stability scores."""
from __future__ import annotations

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, scores, snapshot
from kron_reduce.evaluation.machine_local_observer import execute as execute_observer
from kron_reduce.evaluation.machine_local_observer import known_model
from kron_reduce.evaluation.machine_local_synthetic import fixture
from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.evaluation.trajectory_stability import classify, summarize
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_history import history_with_tangent, predict_from_history
from project_harness.evidence import sha256_file

ROLES = {"train": slice(0, 2), "val": slice(2, 4), "test": slice(4, 6)}


def make_inputs(base, budget):
    projection = np.eye(10)-np.outer(base.mass, np.ones(10))/base.mass.sum()
    initial = []
    for machine, phase in zip((0, 3, 6), (0., .4, .8), strict=True):
        q = .03*np.sin(np.arange(10)+phase) @ projection
        z = .05*np.cos(np.arange(10)+phase)
        for kick in (0., 16.):
            v = .02*np.cos(np.arange(10)+phase)
            v[machine] += kick
            initial.append(np.r_[q, v @ projection, z])
    paths = []
    clock = np.arange(811)*.01
    for step in (.00125, .000625):
        values = []
        # Match the reference integrator's interval-wise ceil, including clock roundoff.
        rhs_count = 4*sum(max(1, int(np.ceil(interval/step-1e-12))) for interval in np.diff(clock))
        for state in initial:
            budget.spend("rollouts")
            budget.spend("numpy_rhs", rhs_count)
            values.append(base.rollout(state[:10], state[10:20], state[20:], clock,
                                       np.zeros(10), np.zeros(10), max_step=step))
        paths.append(np.stack(values))
    reference = paths[1]
    scale = np.maximum(np.sqrt(np.mean(reference[:2, 11:311, :20]**2, axis=(0, 1))), 1e-5)
    hscale = np.maximum(np.sqrt(np.mean(reference[:2, :11, :20]**2, axis=(0, 1))), 1e-5)
    return reference, paths[0], scale, hscale


def inputs_for(paths, hscale):
    return (paths[:, :11, :10], paths[:, :11, 10:20], np.zeros((len(paths), 10)),
            np.zeros((len(paths), 10)), hscale)


def gradient_check(model, paths, scale, hscale, budget, directory):
    def objective():
        result = predict_from_history(model, *inputs_for(paths[:2], hscale), 300,
                                      max_step=.005, spend=budget.spend)
        return loss(result["forecast"], paths[:2, 10:311, :20], np.full(2, .5), scale, np.full(2, 300))
    value = objective()
    params = (model.port_parameter, model.rate_parameter)
    budget.spend("gradients")
    gradients = torch.autograd.grad(value, params)
    rng, rows = np.random.default_rng(2026111304), []
    for label, parameter, grad in zip(("port", "rate"), params, gradients, strict=True):
        direction = torch.tensor(rng.normal(size=tuple(parameter.shape)), dtype=torch.float64)
        direction /= torch.linalg.vector_norm(direction)
        saved, analytic = parameter.detach().clone(), float((grad*direction).sum())
        for eps in (1e-5, 1e-6):
            values = []
            try:
                for sign in (1., -1.):
                    with torch.no_grad():
                        parameter.copy_(saved+sign*eps*direction)
                        values.append(float(objective()))
            finally:
                with torch.no_grad():
                    parameter.copy_(saved)
            difference = (values[0]-values[1])/(2*eps)
            rows.append({"parameter": label, "direction": direction.tolist(), "epsilon": eps,
                         "plus_minus_losses": values, "analytic": analytic, "finite_difference": difference,
                         "pass": abs(difference-analytic) <= 1e-9+1e-5*abs(analytic)})
    result = {"pass": all(r["pass"] for r in rows), "checks": rows}
    write_json(directory/"gradient.json", result)
    return result


def tangent_check(model, paths, budget, directory):
    initial = model._tensor(paths[:1, 0])
    zeros = torch.zeros((1, 10), dtype=torch.float64)
    direction = torch.cos(torch.arange(10, dtype=torch.float64))[None]
    direction /= torch.linalg.vector_norm(direction)
    history, tangent = history_with_tangent(model, initial, zeros, zeros, max_step=.005, spend=budget.spend)
    budget.spend("gradients")
    actual, derivative = torch.autograd.functional.jvp(
        lambda z: model.rollout(torch.cat((initial[:, :20], z), dim=1), zeros, zeros, 10,
                                max_step=.005, spend=budget.spend), initial[:, 20:], direction)
    predicted = (tangent @ direction[:, None, :, None]).squeeze(-1)
    values = {"history_max_difference": float(torch.max(torch.abs(history-actual)).detach()),
              "tangent_max_difference": float(torch.max(torch.abs(predicted-derivative)).detach())}
    values["pass"] = all(v <= 1e-10 for v in values.values())
    np.savez_compressed(directory/"tangent.npz", history=history.detach().numpy(), actual=actual.numpy(),
                        analytic=predicted.detach().numpy(), autograd=derivative.numpy(), direction=direction.numpy())
    write_json(directory/"tangent.json", values)
    return values


def class_for(path, forecast, mass):
    return classify(path[:11, :10], path[:11, 10:20], forecast[:, :20], np.zeros(10), mass)


def evaluate(model, paths, truth_classes, scale, hscale, budget, directory, key):
    with torch.no_grad():
        fine = predict_from_history(model, *inputs_for(paths, hscale), 800, max_step=.0025, spend=budget.spend)
        refined = predict_from_history(model, *inputs_for(paths, hscale), 800, max_step=.00125, spend=budget.spend)
    arrays = {prefix+name: value.detach().numpy() for prefix, result in (("fine_", fine), ("refined_", refined))
              for name, value in result.items()}
    np.savez_compressed(directory/(key+".npz"), **arrays)
    row = scores(arrays["fine_forecast"], arrays["fine_history"], paths, scale)
    events, closure = [], model.closure()
    for i, path in enumerate(paths):
        a, b = (class_for(path, arrays[k+"_forecast"][i], closure.mass) for k in ("fine", "refined"))
        step_error = float(np.max(np.abs((arrays["fine_forecast"][i, :, :20]-arrays["refined_forecast"][i, :, :20])/scale)))
        time_ok = (a["trigger_index"] is None and b["trigger_index"] is None
                   or a["trigger_index"] is not None and b["trigger_index"] is not None
                   and abs(a["trigger_index"]-b["trigger_index"]) <= 2)
        events.append({"truth_label": truth_classes[i]["label"], "predicted_label": a["label"],
            "prospective_eligible": truth_classes[i]["trigger_index"] is None or truth_classes[i]["trigger_index"] > 10,
            "qualified": a["finite_full_path"] and b["finite_full_path"] and a["label"] == b["label"] and time_ok and step_error <= 1e-5,
            "numerical_failure": False, "unsettled": a["label"] is None, "prediction": a, "refined_prediction": b,
            "step_difference": step_error})
    row["stability"] = summarize(events)
    row["stability_events"] = events
    row["step_difference"] = max(e["step_difference"] for e in events)
    row["coi_error"] = max(float(np.max(np.abs(arrays["fine_forecast"][:, :, sl]@closure.mass)))
                            for sl in (slice(0, 10), slice(10, 20)))
    budget.spend("numpy_rhs", len(paths))
    row["energy_identity_error"] = max(closure.energy_identity(x, np.zeros(10), np.zeros(10))["identity_error"]
                                       for x in arrays["fine_forecast"][:, 0])
    row["history_max_normalized_error"] = float(np.max(np.abs((arrays["fine_history"][:, :, :20]-paths[:, :11, :20])/hscale)))
    row.update(status="PASS", evaluated_events=len(paths), allocated_events=len(paths), failed_events=0)
    return row


def evaluate_all(model, paths, truth_classes, scale, hscale, budget, directory, name, evaluator=evaluate):
    rows = {}
    for role, ix in ROLES.items():
        key = name+"_"+role
        try:
            with budget.scope(key):
                row = evaluator(model, paths[ix], truth_classes[ix], scale, hscale, budget, directory, key)
        except NUMERICAL_ERRORS as error:
            events = [{"truth_label": c["label"], "predicted_label": None, "qualified": False,
                       "prospective_eligible": c["trigger_index"] is None or c["trigger_index"] > 10,
                       "numerical_failure": True} for c in truth_classes[ix]]
            row = {"status": "NUMERICAL_FAILURE", "message": str(error), "allocated_events": len(events),
                   "evaluated_events": 0, "failed_events": len(events), "stability": summarize(events), "stability_events": events}
        rows[role] = row
        write_json(directory/(key+".json"), row)
        print(f"{key}: NRMSE3s={row.get('nrmse_3s_all')} stability={row['stability']['accuracy']}", flush=True)
    return rows


def validate(root, directory, budget, plan):
    base = fixture(backbone="sine")
    with budget.scope("truth"):
        paths, coarse, scale, hscale = make_inputs(base, budget)
        classes = [class_for(path, path[10:], base.mass) for path in paths]
        coarse_classes = [class_for(path, path[10:], base.mass) for path in coarse]
        difference = float(np.max(np.abs((paths[:, :, :20]-coarse[:, :, :20])/scale)))
        coverage = all({c["label"] for c in classes[ix]} == {"stable", "unstable"} for ix in ROLES.values())
        truth_pass = (coverage and difference <= 1e-6
            and all(a["label"] == b["label"] and a["trigger_index"] == b["trigger_index"]
                    and (a["trigger_index"] is None or a["trigger_index"] > 10)
                    for a, b in zip(classes, coarse_classes, strict=True)))
        np.savez_compressed(directory/"inputs.npz", paths=paths, coarse_paths=coarse, scale=scale, history_scale=hscale,
                            mass=base.mass, roles=np.repeat(["train", "val", "test"], 2))
    write_json(directory/"truth.json", {"pass": truth_pass, "class_coverage_pass": coverage,
        "step_difference": difference, "classes": classes, "coarse_classes": coarse_classes})
    known = known_model(base, "local")
    perturbed = DiagonalMemoryModel(base, port_kind="local", port=np.full(10, .4), tau=np.geomspace(.06, 1.6, 10),
                                    encoder=np.zeros((10, 80)), feature_scale=np.ones(80))
    for name, model in (("known", known), ("perturbed", perturbed)):
        model.encoder.requires_grad_(False)
        np.savez_compressed(directory/(name+"_parameters.npz"), **{k: v.numpy() for k, v in snapshot(model).items()})
    write_json(directory/"frozen_inputs_and_models.json", {p.name: sha256_file(p) for p in directory.glob("*.npz")})
    try:
        with budget.scope("tangent"):
            tangent = tangent_check(perturbed, paths, budget, directory)
    except NUMERICAL_ERRORS as error:
        tangent = {"pass": False, "message": str(error)}
        write_json(directory/"tangent.json", tangent)
    try:
        with budget.scope("gradient"):
            gradient = gradient_check(perturbed, paths, scale, hscale, budget, directory)
    except NUMERICAL_ERRORS as error:
        gradient = {"pass": False, "message": str(error)}
        write_json(directory/"gradient.json", gradient)
    records = {name: evaluate_all(model, paths, classes, scale, hscale, budget, directory, name)
               for name, model in (("known", known), ("perturbed", perturbed))}
    numerical = all(r["status"] == "PASS" and r["step_difference"] <= 1e-5 and r["coi_error"] <= 1e-10
                    and r["energy_identity_error"] <= 1e-10 for rows in records.values() for r in rows.values())
    recovery = all(r["status"] == "PASS" and r["main_nrmse"] <= .001 and r["nrmse_3s_all"] <= .001
                   and max(r["hidden_start_rmse"], r["hidden_origin_rmse"]) <= 1e-4
                   and r["history_max_normalized_error"] <= 1e-7 for r in records["known"].values())
    stability = all(r["stability"]["accuracy"] == 1. and r["stability"]["balanced_accuracy"] == 1.
                    for r in records["known"].values())
    passed = bool(truth_pass and tangent["pass"] and gradient["pass"] and numerical and recovery and stability)
    return {"scientific_pass": passed, "truth_pass": truth_pass, "tangent_pass": tangent["pass"],
        "gradient_pass": gradient["pass"], "numerical_pass": numerical, "recovery_pass": recovery,
        "stability_pass": stability, "evaluation_complete": True, "records": records, "fit_attempts": 0,
        "noise_injection": 0., "t0_allowed": False,
        "next_route": "PREPARE_TASK3_NONLINEAR_PILOT" if passed else "REVIEW_NONLINEAR_HISTORY_FAILURE",
        "claim_boundary": "Six fixed clean sine-system cases verify a differentiable nonlinear initializer and known-dynamics recovery. Perturbed dynamics are descriptive controls. No parameter fitting or real-data evaluation."}


def execute(*, root, result_dir, stage, plan):
    return execute_observer(root=root, result_dir=result_dir, stage=stage, plan=plan, validator=validate)
