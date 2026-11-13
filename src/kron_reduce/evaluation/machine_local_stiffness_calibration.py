"""One bounded stiffness calibration with a complete known-truth evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.machine_local_hidden_oracle import (
    ROLES,
    STEPS,
    describe,
    predict_batches,
)
from kron_reduce.evaluation.machine_local_nonlinear_history import class_for
from kron_reduce.evaluation.machine_local_observer import execute as execute_observer
from kron_reduce.evaluation.machine_local_observer import known_model
from kron_reduce.evaluation.machine_local_synthetic import fixture
from kron_reduce.evaluation.trajectory_stability import summarize
from kron_reduce.models.machine_local_forecast import FrozenMachineLocalForecaster
from project_harness.evidence import sha256_file

BOUNDS = (math.log(.5), math.log(2.))
TRUE_LOG_FACTOR = math.log(10/11)
XATOL = 1e-8
MAX_CALLS = 80
FIT_STEP = .005


def make_inputs(base, budget):
    i, j = np.arange(10), np.arange(64)
    phase = i[None, :] + 2*np.pi*(j[:, None]+.25)/64
    mass = base.mass
    projection = np.eye(10)-np.outer(mass, np.ones(10))/mass.sum()
    q, v, z = .03*np.sin(phase), .02*np.cos(phase), .05*np.cos(phase)
    v[j, (j//2) % 10] += 16*(j % 2)
    states = np.c_[q @ projection, v @ projection, z]
    clock, zeros = np.arange(811)*.01, np.zeros(10)
    paths = []
    for step in (.00125, .000625):
        rhs_count = 4*sum(max(1, int(np.ceil(h/step-1e-12))) for h in np.diff(clock))
        values = []
        for state in states:
            budget.spend("rollouts")
            budget.spend("numpy_rhs", rhs_count)
            values.append(base.rollout(state[:10], state[10:20], state[20:], clock,
                                       zeros, zeros, max_step=step))
        paths.append(np.stack(values))
        print(f"truth step={step}: {len(values)}/64", flush=True)
    coarse, reference = paths
    scale = np.maximum(np.sqrt(np.mean(reference[:32, 11:311, :20]**2, axis=(0, 1))), 1e-5)
    hscale = np.maximum(np.sqrt(np.mean(reference[:32, :11, :20]**2, axis=(0, 1))), 1e-5)
    return reference, coarse, scale, hscale


def calibrated_model(base, log_factor):
    # One effective stiffness buffer feeds force, tangent, closure and export.
    stiffness = np.full_like(base.edge_stiffness, .132*math.exp(log_factor))
    return known_model(replace(base, edge_stiffness=stiffness), "local").requires_grad_(False)


def causal_predict(model, paths, hscale, count, step, spend):
    zeros, clock = np.zeros((len(paths), 10)), np.arange(-10, 1)*.01
    histories = [CausalStateHistory(clock, path[:11, :10], path[:11, 10:20], power)
                 for path, power in zip(paths, zeros, strict=True)]
    forecaster = FrozenMachineLocalForecaster(model, hscale, max_step=step)
    try:
        return forecaster.predict_batch(histories, zeros, count, spend=spend)
    except ValueError as error:
        if str(error) == "model inputs must be finite":
            raise FloatingPointError("nonfinite intermediate RK4 state") from error
        raise


def event_nmse(forecasts, paths, scale):
    return np.mean(((forecasts[:, 1:301, :20]-paths[:, 11:311, :20])/scale)**2, axis=(1, 2))


def objective(base, paths, scale, hscale, log_factor, budget):
    model = calibrated_model(base, log_factor)
    losses = []
    for offset in (0, 16):
        try:
            part = paths[offset:offset+16]
            forecast = causal_predict(model, part, hscale, 300, FIT_STEP, budget.spend)["forecast"].numpy()
            losses.extend(event_nmse(forecast, part, scale).tolist())
        except NUMERICAL_ERRORS as error:
            return {"log_factor": log_factor, "gamma": math.exp(log_factor), "loss": None,
                    "event_nmse": losses, "failed_batch_offset": offset, "error": str(error)}
    loss = float(np.mean(losses))
    return {"log_factor": log_factor, "gamma": math.exp(log_factor),
            "loss": loss if math.isfinite(loss) else None,
            "event_nmse": [x if math.isfinite(x) else None for x in losses],
            **({"error": "nonfinite objective"} if not math.isfinite(loss) else {})}


def fit_scalar(evaluate, budget):
    visited = []

    def loss(log_factor):
        log_factor = float(log_factor)
        if len(visited) >= MAX_CALLS or not BOUNDS[0] <= log_factor <= BOUNDS[1]:
            raise RuntimeError("bounded search exceeded its frozen scope")
        budget.spend("optimizer_objective_calls")
        row = evaluate(log_factor)
        visited.append(row)
        return row["loss"] if row["loss"] is not None else math.inf

    budget.spend("fits")
    result = minimize_scalar(loss, bounds=BOUNDS, method="bounded",
                             options={"xatol": XATOL, "maxiter": MAX_CALLS})
    selected = min((row for row in visited if row["loss"] is not None),
                   key=lambda row: row["loss"], default=None)
    return {"success": bool(result.success), "status": int(result.status), "message": str(result.message),
            "nfev": int(result.nfev), "optimizer_calls": len(visited), "selected": selected, "visited": visited}


def predict_for_scoring(model, paths, hscale, step, _oracle, spend):
    result = causal_predict(model, paths, hscale, 800, step, spend)
    arrays = {key: result[key].numpy() for key in ("forecast", "history")}
    for target, source in (("iterates", "iterates"), ("increments", "gn_hidden_start"),
                           ("singular", "gn_singular"), ("residuals", "iteration_residuals")):
        arrays[target] = result[source].numpy().swapaxes(0, 1)
    return arrays


def gn_diagnostics(arrays, paths, hscale, row):
    for prefix in STEPS:
        arrays[prefix+"_final_residual"] = (paths[:, 1:11, :20]-arrays[prefix+"_history"][:, 1:11, :20])/hscale
    for index, event in enumerate(row["events"]):
        diagnostics = {}
        for prefix in STEPS:
            history = arrays[prefix+"_history"][index]
            if not np.isfinite(history).all():
                diagnostics[prefix] = None
                continue
            residual = arrays[prefix+"_final_residual"][index]
            before = arrays[prefix+"_residuals"][index]
            reset = paths[index, 10, :20]-history[10, :20]
            singular = arrays[prefix+"_singular"][index]
            rank_threshold = 64*np.finfo(np.float64).eps*200*singular[:, :1]
            diagnostics[prefix] = {
                "residual_rms_each_iterate": np.r_[np.sqrt(np.mean(before**2, axis=1)),
                                                   np.sqrt(np.mean(residual**2))].tolist(),
                "increment_norm": np.linalg.norm(arrays[prefix+"_increments"][index], axis=1).tolist(),
                "hidden_norm_each_iterate": np.linalg.norm(arrays[prefix+"_iterates"][index], axis=1).tolist(),
                "singular_values": singular.tolist(),
                "rank_each_update": np.count_nonzero(singular > rank_threshold, axis=1).tolist(),
                "condition_each_update": (singular[:, 0]/singular[:, -1]).tolist(),
                "visible_reset_q_v": reset.tolist(),
            }
        event["gn"] = diagnostics
        event["hidden_recovery_pass"] = bool(event["qualified"] and
            max(event["hidden_start_rmse"], event["hidden_origin_rmse"]) <= 1e-4)
        event["history_fit_pass"] = bool(event["qualified"] and event["history_max_normalized_error"] <= 1e-7)
    row["hidden_recovery_pass"] = all(e["hidden_recovery_pass"] for e in row["events"])
    row["history_fit_pass"] = all(e["history_fit_pass"] for e in row["events"])
    row["prediction_3s_pass"] = bool(row["horizons"] is not None and row["horizons"]["3"]["pooled_nrmse"] <= .001)


def diagnose_scalar(fit, profile, reference, train_losses):
    selected = fit["selected"]
    if selected is None:
        return {"optimizer_valid": False, "comparison_available": False,
                "profile_valid_points": sum(row["loss"] is not None for row in profile)}
    log_factor, loss = selected["log_factor"], selected["loss"]
    finite = [row["loss"] for row in [*fit["visited"], *profile, reference] if row["loss"] is not None]
    eps = float(64*np.finfo(np.float64).eps*max(1., *finite))
    boundary = min(log_factor-BOUNDS[0], BOUNDS[1]-log_factor) <= 10*XATOL
    contradiction = any(row["loss"] is not None and row["loss"] < loss-eps for row in profile)
    gamma, true_gamma = math.exp(log_factor), math.exp(TRUE_LOG_FACTOR)
    difference = None if reference["loss"] is None else reference["loss"]-loss
    delta = None
    if difference is not None and all(value is not None for steps in train_losses.values() for value in steps.values()):
        pairs = [(loss, train_losses["fitted"]), (reference["loss"], train_losses["known"])]
        delta = max(eps, *(x for coarse, steps in pairs for x in
            (abs(coarse-steps["fine"]), abs(steps["fine"]-steps["refined"]))))
    profile_valid_points = sum(row["loss"] is not None for row in profile)
    return {"optimizer_valid": bool(fit["success"] and not boundary and not contradiction and profile_valid_points == 9),
            "boundary": boundary, "profile_contradiction": bool(contradiction),
            "profile_valid_points": profile_valid_points,
            "gamma": gamma, "gamma_error": gamma-true_gamma, "gamma_absolute_error": abs(gamma-true_gamma),
            "gamma_relative_error": (gamma-true_gamma)/true_gamma,
            "log_factor_error": log_factor-TRUE_LOG_FACTOR,
            "within_optimizer_scale": abs(log_factor-TRUE_LOG_FACTOR) <= 10*XATOL,
            "objective_roundoff": eps, "same_step_reference_minus_fitted": difference,
            "local_discretization_indicator": delta, "comparison_available": delta is not None,
            "same_step_objective_consistent": None if delta is None else abs(difference) <= delta,
            "profile_points_within_indicator": None if delta is None else [
                row["log_factor"] for row in profile if row["loss"] is not None and abs(row["loss"]-loss) <= delta],
            "train_losses_by_step": train_losses}


def paired_differences(records):
    comparisons = {}
    for group in ("fitted", "known"):
        comparisons[group+"_minus_mismatch"] = {}
        for role in ROLES:
            left, right = (records[key][role]["events"] for key in (group, "mismatch"))
            horizons = {}
            for seconds in ("1", "2", "3", "5", "8"):
                values = [a["trajectory"][seconds]["nmse"]-b["trajectory"][seconds]["nmse"]
                          if a["qualified"] and b["qualified"] else None
                          for a, b in zip(left, right, strict=True)]
                paired = [v for v in values if v is not None]
                horizons[seconds] = {"per_event": values, "paired_count": len(paired),
                    "allocated": len(values), "mean": float(np.mean(paired)) if len(paired) == len(values) else None}
            comparisons[group+"_minus_mismatch"][role] = horizons
    return comparisons


def validate(root, directory, budget, plan):
    base = fixture(backbone="sine")
    with budget.scope("truth"):
        paths, coarse, scale, hscale = make_inputs(base, budget)
        classes = [class_for(p, p[10:], base.mass) for p in paths]
        other = [class_for(p, p[10:], base.mass) for p in coarse]
        difference = float(np.max(np.abs((paths[:, :, :20]-coarse[:, :, :20])/scale)))
        coverage = all({c["label"] for c in classes[ix]} == {"stable", "unstable"} for ix in ROLES.values())
        truth_pass = bool(coverage and difference <= 1e-6 and np.isfinite(paths).all()
            and all(a["label"] == b["label"] and a["trigger_index"] == b["trigger_index"]
                    and (a["trigger_index"] is None or a["trigger_index"] > 10)
                    for a, b in zip(classes, other, strict=True)))
        np.savez_compressed(directory/"inputs.npz", paths=paths, coarse_paths=coarse, scale=scale,
            history_scale=hscale, mass=base.mass, ids=np.array([f"stiffness_calibration_v1:{j:03d}" for j in range(64)]),
            roles=np.array(["train"]*32+["val"]*16+["test"]*16))
        write_json(directory/"truth.json", {"pass": truth_pass, "class_coverage_pass": coverage,
            "step_difference": difference, "classes": classes, "coarse_classes": other,
            "inputs_sha256": sha256_file(directory/"inputs.npz")})
    summary = {"truth_pass": truth_pass, "scientific_pass": False, "evaluation_complete": False,
        "next_route": "REVIEW_SCALAR_CALIBRATION",
        "claim_boundary": "One bounded scalar calibration on 64 clean analytic sine-L10 cases."}
    if not truth_pass:
        return summary
    train = paths[ROLES["train"]]

    def evaluate(log_factor):
        row = objective(base, train, scale, hscale, log_factor, budget)
        with (directory/"objectives.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"scope": budget.active, **row}, allow_nan=False)+"\n")
        return row

    with budget.scope("fit"):
        fit = fit_scalar(evaluate, budget)
    write_json(directory/"fit.json", fit)
    factors = {"mismatch": 0., "known": TRUE_LOG_FACTOR,
               "fitted": None if fit["selected"] is None else fit["selected"]["log_factor"]}
    models, checkpoints = {}, {}
    for group, log_factor in factors.items():
        if log_factor is None:
            continue
        models[group] = calibrated_model(base, log_factor)
        checkpoint = directory/(group+".pt")
        FrozenMachineLocalForecaster(models[group], hscale, max_step=STEPS["fine"]).save_checkpoint(checkpoint)
        checkpoints[group] = {"log_factor": log_factor, "gamma": math.exp(log_factor),
                              "checkpoint": checkpoint.name, "sha256": sha256_file(checkpoint)}
    write_json(directory/"freeze.json", {"selection": fit["selected"], "models": checkpoints,
        "inputs_sha256": sha256_file(directory/"inputs.npz"), "protocol_sha256": plan["protocol_sha256"],
        "fit_step_s": FIT_STEP, "evaluation_steps_s": STEPS, "evaluation_horizons_s": [1, 2, 3, 5, 8],
        "stability_window_s": 8})
    profile = []
    with budget.scope("profile"):
        for log_factor in np.linspace(*BOUNDS, 9):
            budget.spend("profile_objective_calls")
            profile.append(evaluate(float(log_factor)))
    with budget.scope("same_step_reference"):
        budget.spend("reference_objective_calls")
        reference = evaluate(TRUE_LOG_FACTOR)
    write_json(directory/"train_diagnostics.json", {"profile": profile, "same_step_reference": reference})
    records, train_losses = {}, {}
    for group in factors:
        records[group] = {}
        for role, ix in ROLES.items():
            key, part = group+"_"+role, paths[ix]
            with budget.scope(key):
                if group in models:
                    arrays, failures = predict_batches(models[group], part, hscale, False, budget,
                                                       predictor=predict_for_scoring)
                else:
                    arrays = {prefix+"_"+field: np.full((len(part), points, 30), np.nan)
                              for prefix in STEPS for field, points in (("forecast", 801), ("history", 11))}
                    failures = [{"reason": "no_finite_optimizer_point"}]
                row = describe(arrays, part, classes[ix], scale, hscale, base.mass, False)
                gn_diagnostics(arrays, part, hscale, row)
                row["failures"] = failures
                if group not in models:
                    for event in row["events"]:
                        event.update(numerical_failure=False, unsettled=False, uncovered=True)
                    row["stability"].update(summarize([
                        event | {"prospective_eligible": False} for event in row["events"]]))
                else:
                    row["stability"]["unsettled_predictions"] = sum(
                        e["qualified"] and e["unsettled"] for e in row["events"])
                np.savez_compressed(directory/(key+".npz"), **arrays)
                write_json(directory/(key+".json"), row)
                if role == "train" and group != "mismatch":
                    train_losses[group] = {
                        prefix: float(np.mean(event_nmse(arrays[prefix+"_forecast"], part, scale)))
                        if row["numerical_pass"] else None
                        for prefix in STEPS}
            records[group][role] = row
            print(f"{key}: evaluated {len(part)} events", flush=True)
    scalar = diagnose_scalar(fit, profile, reference, train_losses)
    numerical = all(row["numerical_pass"] for roles in records.values() for row in roles.values())
    known = all(row["recovery_pass"] for row in records["known"].values())
    comparisons = paired_differences(records)
    write_json(directory/"paired_differences.json", comparisons)
    summary.update(evaluation_complete=True, technical_valid=numerical and known,
        scientific_pass=numerical and known and scalar["optimizer_valid"],
        scientific_pass_meaning="valid diagnostic execution, evaluated separately from fitted recovery",
        matched_control_valid=known, fit=fit, scalar=scalar, records=records,
        paired_differences=comparisons)
    return summary


def execute(*, root, result_dir, stage, plan):
    return execute_observer(root=root, result_dir=result_dir, stage=stage, plan=plan, validator=validate)
