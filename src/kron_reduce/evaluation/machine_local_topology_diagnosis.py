"""Cross frozen shared coefficients with estimated and exact forecast states."""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_equation import STEPS
from kron_reduce.evaluation.machine_local_topology import make_model, pack, scoring
from kron_reduce.evaluation.machine_local_topology_learning import memory_kernel_error
from kron_reduce.evaluation.machine_local_topology_truth import (
    initial_cases,
    parameters,
    truth_rollout,
)
from kron_reduce.evaluation.trajectory_stability import classify
from kron_reduce.models.machine_local_topology import predict_from_topology_history
from project_harness.evidence import sha256_file

SUPPORTS = ("T0M1", "T1M1")
ROLES = ("train", "val")
NUMERICAL_ERRORS = (FloatingPointError, torch.linalg.LinAlgError)


def selected_model(root, source, support, params):
    relative = Path(source["run_id"])/support/"selected-equation.json"
    path = (root/"results"/relative).resolve(strict=True)
    if (not path.is_relative_to(root/"results")
            or sha256_file(path) != source["equation_sha256"][support]):
        raise RuntimeError("frozen S1 equation differs from the registered source")
    state = json.loads(path.read_bytes())
    mass, k, d, _, _, eq = params
    g, tau = (np.asarray(state[key], dtype=float) for key in ("port_amplitude", "tau"))
    if g.shape != mass.shape or tau.shape != mass.shape:
        raise ValueError("invalid selected memory coefficients")
    model = make_model((mass, k, d, g, tau, eq), support, "cuda", train_scales=True)
    with torch.no_grad():
        model.log_stiffness_factor.fill_(math.log(state["stiffness_scale"]))
        model.log_damping_factor.fill_(math.log(state["damping_scale"]))
    return model.requires_grad_(False).eval()


def crossed_origin(estimate, truth, *, exact_a, exact_z):
    return torch.cat((truth[:, :20], (truth if exact_z else estimate)[:, 20:30],
                      (truth if exact_a else estimate)[:, 30:]), dim=1)


def estimate_origin(model, truth, power, eq, scale, budget):
    history = np.full((len(power), 11, 20+model.latent_dimension), np.nan)
    origin = np.full((len(power), history.shape[-1]), np.nan)
    failures = []

    def estimate(rows):
        result = predict_from_topology_history(model, truth[rows, :11, :10], truth[rows, :11, 10:20],
                                               power[rows], eq[rows], scale, 0, max_step=.00125,
                                               spend=budget.spend)
        history[rows] = result["history"].cpu().numpy()
        origin[rows] = result["forecast"][:, 0].cpu().numpy()

    with torch.no_grad():
        try:
            estimate(slice(None))
        except NUMERICAL_ERRORS as error:
            failures.append({"batch": True, "error": str(error)})
            for i in range(len(power)):
                try:
                    estimate(slice(i, i+1))
                except NUMERICAL_ERRORS as individual:
                    failures.append({"index": i, "error": str(individual)})
    return model._tensor(origin), np.broadcast_to(history, (2, *history.shape)), failures


def rollout(model, origins, power, eq, budget):
    paths = np.full((2, len(power), 801, origins.shape[-1]), np.nan)
    failures = []
    for j, step in enumerate(STEPS):
        state = origins[j]
        valid = torch.isfinite(state).all(dim=1).cpu().numpy()
        indices = np.flatnonzero(valid)
        for i in np.flatnonzero(~valid):
            failures.append({"index": int(i), "step": step, "error": "causal origin unavailable"})
        if not len(indices):
            continue
        try:
            paths[j, indices] = model.rollout(state[indices], power[indices], eq[indices], 800,
                                             max_step=step, spend=budget.spend).cpu().numpy()
        except NUMERICAL_ERRORS as error:
            failures.append({"step": step, "batch": True, "error": str(error)})
            for i in indices:
                try:
                    p = slice(i, i+1)
                    paths[j, p] = model.rollout(state[p], power[p], eq[p], 800, max_step=step,
                                                spend=budget.spend).cpu().numpy()
                except NUMERICAL_ERRORS as individual:
                    failures.append({"index": int(i), "step": step, "error": str(individual)})
    return paths, failures


def execute(root, directory, plan, budget):
    source = plan["source"]
    source_dir = (root/"results"/source["run_id"]).resolve(strict=True)
    if source_dir.parent != root/"results" or sha256_file(source_dir/"artifact_hashes.json") != source["manifest_sha256"]:
        raise RuntimeError("S1 manifest identity differs")
    cases, params = initial_cases(), parameters()
    mass, _, _, g, tau, eq = params
    rows = np.flatnonzero(np.isin(cases["roles"], ROLES))
    ids = [cases["ids"][i] for i in rows]
    roles = np.array(cases["roles"])[rows].tolist()
    if ids != plan["ids"] or roles != ["train"]*6+["val"]*3:
        raise RuntimeError("S1 train and val roster differs")
    q, v, power, a, z = (cases[key][rows] for key in ("q", "v", "power", "a", "z"))
    equilibrium = np.broadcast_to(eq, power.shape).copy()
    records = {}
    for support in SUPPORTS:
        model = selected_model(root, source, support, params)
        known = make_model(params, support, "cuda")
        factor = a if model.topology_dimension else np.zeros_like(a)
        truth = truth_rollout(params, q, v, power, factor, z, count=810, max_step=.000625, spend=budget.spend)
        train = truth[np.array(roles) == "train", :, :20]
        scale = np.maximum(np.sqrt(np.mean(train[:, 11:]**2, axis=(0, 1))), 1e-4)
        hscale = np.maximum(np.sqrt(np.mean(train[:, :11]**2, axis=(0, 1))), 1e-4)
        p, e = model._tensor(power), model._tensor(equilibrium)
        expected = pack(model, truth[:, 10], factor)
        start, history, failures = estimate_origin(model, truth, p, e, hscale, budget)
        controls = {}
        with torch.no_grad():
            a_switch = (False, True) if model.topology_dimension else (False,)
            for label, fixed in (("fitted", model), ("truth", known)):
                for exact_a, exact_z in product(a_switch, (False, True)):
                    suffix = ("truth_both" if exact_a and exact_z else "truth_coupling" if exact_a
                              else "truth_memory" if exact_z else "estimated")
                    origin = crossed_origin(start, expected, exact_a=exact_a, exact_z=exact_z)
                    controls[f"{label}_{suffix}"] = rollout(fixed, origin[None].expand(2, -1, -1), p, e, budget)
        classes = [classify(t[:11, :10], t[:11, 10:20], t[10:, :20], equilibrium[i], mass)
                   for i, t in enumerate(truth)]
        roster = {"ids": ids, "roles": roles}
        result = {key: scoring(model, roster, path, history, truth, factor, equilibrium, mass, scale, hscale,
                               classes, key != "fitted_estimated") | {"failures": errors}
                  for key, (path, errors) in controls.items()}
        for key, (path, _) in controls.items():
            np.savez_compressed(directory/f"{support}-{key}.npz", forecast=path)
        np.savez_compressed(directory/f"{support}-inputs.npz", truth=truth, a=factor, power=power, equilibrium=equilibrium,
                            history=history, scale=scale, hscale=hscale, ids=np.array(ids))
        records[support] = {"controls": result, "memory_kernel_error": memory_kernel_error(model, g, tau),
                            "initialization_failures": failures}
        write_json(directory/f"{support}.json", records[support])
        print(f"{support}: complete paired train/val forecasts", flush=True)
    pass_gate = all(row["qualified"] == row["allocated"] for record in records.values()
                    for trial in record["controls"].values() for row in trial["roles"].values())
    return {"populations": records, "engineering_pass": pass_gate, "scientific_pass": False,
            "claim_boundary": "Exact or estimated forecast states with frozen coefficients on synthetic train and val."}
