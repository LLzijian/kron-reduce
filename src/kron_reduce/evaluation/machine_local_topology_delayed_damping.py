"""Fit a single positive damping scale with a causal train-only profile."""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.machine_local_topology import make_model, scoring, simulate
from kron_reduce.evaluation.machine_local_topology_delayed_origin import (
    checked_cohort,
    load_arrays,
    source_files,
    sufficient,
    support,
)
from kron_reduce.evaluation.machine_local_topology_margin import (
    fit_counts,
    matched_margins,
    save_equation,
)
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def minimize_damping(evaluate, center, grid_points, half_width, refinements):
    grid = np.linspace(center-half_width, center+half_width, grid_points)
    grid[grid_points//2] = center
    values = [evaluate(float(point)) for point in grid]
    finite = [i for i, value in enumerate(values) if math.isfinite(value)]
    if not finite:
        return None, "no_finite_grid_value"
    i = min(finite, key=lambda j: (values[j], abs(grid[j]-center), j))
    if i == 0 or i == grid_points-1:
        return float(grid[i]), "grid_boundary"
    if not all(math.isfinite(values[j]) for j in (i-1, i+1)):
        return float(grid[i]), "invalid_bracket"
    result = minimize_scalar(evaluate, bounds=(float(grid[i-1]), float(grid[i+1])), method="bounded",
                             options={"maxiter": refinements, "xatol": 1e-5})
    if result.success and math.isfinite(result.fun):
        return (float(result.x) if result.fun < values[i] else float(grid[i])), "refined"
    return float(grid[i]), "refinement_failed"


@torch.no_grad()
def fit_scale(model, visible, p, eq, lengths, scale, hscale, settings, budget):
    center = float(model.log_damping_factor)
    history = visible[:, :11]
    weights = visible.new_full((len(visible),), 1./len(visible))
    trace = []

    def evaluate(alpha):
        budget.spend("objective_calls")
        model.log_damping_factor.fill_(alpha)
        try:
            prediction = predict_from_topology_history(model, history[:, :, :10], history[:, :, 10:],
                                                       p, eq, hscale, 300, max_step=.005, spend=budget.spend)
            value = float(loss(prediction["forecast"], visible[:, 10:], weights, scale, lengths))
            if not math.isfinite(value):
                raise FloatingPointError("damping objective is nonfinite")
        except NUMERICAL_ERRORS as error:
            trace.append({"alpha": alpha, "scale": math.exp(alpha), "train_loss": None, "failure": str(error)})
            return math.inf
        trace.append({"alpha": alpha, "scale": math.exp(alpha), "train_loss": value})
        return value

    budget.spend("fits")
    optimum, status = minimize_damping(evaluate, center, settings["grid_points"], settings["log_half_width"],
                                       settings["max_refinement_calls"])
    frozen = center if optimum is None else optimum
    model.log_damping_factor.fill_(frozen)
    start = next(row["train_loss"] for row in trace if row["alpha"] == center)
    selected = next(row["train_loss"] for row in trace if row["alpha"] == frozen)
    return {"status": status, "trace": trace, "start_scale": math.exp(center), "selected_scale": math.exp(frozen),
            "start_train_loss": start, "selected_train_loss": selected,
            "valid": status in ("refined", "grid_boundary") and all(row["train_loss"] is not None for row in trace)
            and selected < start}


def execute(root, directory, plan, budget):
    source = source_files(root, plan["source"])
    truth = load_arrays(source["truth.npz"], ("fine", "refined", "ids", "roles", "power", "equilibrium"), budget)
    saved = load_arrays(source["trajectory-forecasts.npz"], ("forecasts", "histories"), budget)
    reference = json.loads(source["trajectory-scores.json"].read_bytes())
    _, labels = json.loads(source["truth-classes.json"].read_bytes())
    scales = json.loads(source["preprocessing.json"].read_bytes())
    checked_cohort(plan, truth, saved, labels, reference)
    params = parameters()
    mass, *_, eq0 = params
    path, power, eq = (truth[key] for key in ("refined", "power", "equilibrium"))
    if not np.array_equal(eq, np.broadcast_to(eq0, eq.shape)) or np.any(power):
        raise ValueError("saved delayed truth inputs differ")
    train = np.flatnonzero(truth["roles"] == "train")
    lengths = fit_counts([labels[i] for i in train])
    scale, hscale = (np.asarray(scales[key], dtype=float) for key in ("target", "history"))
    if (lengths.tolist() != scales["train_lengths"]
            or any(x.shape != (20,) or not np.isfinite(x).all() or np.any(x <= 0) for x in (scale, hscale))):
        raise ValueError("saved train support or scales differ")
    budget.spend("npz_decodes")
    model = load_snapshot(make_model(params, "T0M1", "cuda", train_scales=True), source["trajectory-equation.npz"])
    p, equilibrium = model._tensor(power), model._tensor(eq)
    result = fit_scale(model, model._tensor(path[train, :311, :20]), p[train], equilibrium[train], lengths,
                       scale, hscale, plan["settings"], budget)
    write_json(directory/"damping-profile.json", result)
    save_equation(model, directory, "candidate")
    a = np.zeros_like(power)
    cases = {key: plan[key] for key in ("ids", "roles")}
    predictions, histories, failures = simulate(model, path, a, p, equilibrium, hscale, budget, oracle=False)
    score = scoring(model, cases, predictions, histories, path, a, eq, mass, scale, hscale, labels, False)
    matched_margins(model, predictions, path, labels, equilibrium, score)
    score["failures"] = failures
    np.savez_compressed(directory/"candidate-forecasts.npz", forecasts=predictions, histories=histories)
    write_json(directory/"candidate-scores.json", score)
    write_json(directory/"reference-scores.json", reference)
    pairs = support(score, reference)
    best = sufficient(score, reference, pairs, plan["gates"])
    return {"engineering_pass": result["valid"], "scientific_pass": result["valid"] and best,
            "claim_boundary": "Single damping scale fitted to visible synthetic train trajectories; causal evaluation.",
            "fit": {name: value for name, value in result.items() if name != "trace"},
            "paired_witnesses": pairs, "models": {"reference": reference["roles"], "candidate": score["roles"]}}
