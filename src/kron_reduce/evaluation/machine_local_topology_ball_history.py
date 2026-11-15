"""Known-truth comparison of clipped and ball-constrained history initialization."""

from __future__ import annotations

import json
from functools import partial

import numpy as np

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_equation import align_classes
from kron_reduce.evaluation.machine_local_topology import make_model, scoring, simulate
from kron_reduce.evaluation.machine_local_topology_delayed_origin import load_arrays, source_files
from kron_reduce.evaluation.machine_local_topology_margin import matched_margins
from kron_reduce.evaluation.machine_local_topology_truth import (
    initial_cases,
    parameters,
    truth_rollout,
)
from kron_reduce.evaluation.trajectory_stability import classify
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _population(root, plan, params, budget):
    source = source_files(root, plan["source"])
    delayed = load_arrays(source["truth.npz"], ("fine", "refined", "ids", "roles", "power", "equilibrium"), budget)
    _, labels = json.loads(source["truth-classes.json"].read_bytes())
    if (delayed["ids"].tolist() != plan["delayed_ids"] or delayed["roles"].tolist() != plan["delayed_roles"]
            or len(labels) != len(plan["delayed_ids"]) or any(not row["reference_qualified"] for row in labels)):
        raise ValueError("delayed truth roster differs")
    count = len(labels)
    shapes = {"fine": (count, 811, 30), "refined": (count, 811, 30), "power": (count, 10), "equilibrium": (count, 10)}
    if (any(delayed[k].shape != shape or delayed[k].dtype != np.float64 or not np.isfinite(delayed[k]).all()
            for k, shape in shapes.items()) or np.any(delayed["power"]) or np.any(delayed["equilibrium"])
            or np.any(delayed["refined"][:, 0, 20:])):
        raise ValueError("delayed truth arrays or matched inputs differ")

    nominal = initial_cases()
    q, v, power = (np.tile(nominal[key], (2, 1)) for key in ("q", "v", "power"))
    z = nominal["z"]/np.linalg.norm(nominal["z"], axis=1, keepdims=True)
    z = np.concatenate([radius*z for radius in plan["truth_radii"]])
    fresh_ids = [f"{family}-{key}" for family in ("interior", "exterior") for key in nominal["ids"]]
    ids = fresh_ids + delayed["ids"].tolist()
    roles = nominal["roles"]*2 + delayed["roles"].tolist()
    if ids != plan["ids"] or roles != plan["roles"]:
        raise ValueError("synthetic observer roster differs")
    fresh = [truth_rollout(params, q, v, power, np.zeros_like(q), z, count=810, max_step=h, spend=budget.spend)
             for h in (.00125, .000625)]
    fine, truth = (np.concatenate((x, delayed[key])) for x, key in zip(fresh, ("fine", "refined"), strict=True))
    power = np.concatenate((power, delayed["power"]))
    equilibrium = np.broadcast_to(params[-1], power.shape).copy()
    classes = []
    for i in range(len(truth)):
        coarse, refined = (classify(x[i, :11, :10], x[i, :11, 10:20], x[i, 10:, :20], equilibrium[i], params[0])
                           for x in (fine, truth))
        qualified = (refined["label"] is not None and align_classes(coarse, refined)
                     and (refined["trigger_index"] is None or refined["trigger_index"] > 10))
        classes.append(refined | {"reference_qualified": bool(qualified)})
    if any((a["label"], a["trigger_index"]) != (b["label"], b["trigger_index"])
           for a, b in zip(classes[len(fresh_ids):], labels, strict=True)):
        raise ValueError("saved delayed labels differ from direct reconstruction")
    return fine, truth, power, equilibrium, {"ids": ids, "roles": roles}, classes


def _gates(plan, classes, records, diagnostics):
    limits = plan["gates"]
    exact, no_prior, radial = (records[k]["events"] for k in ("exact", "exact_no_prior", "radial"))
    dev = [i for i, row in enumerate(exact) if row["role"] in ("train", "val")]
    interior = [i for i in dev if exact[i]["id"].startswith("interior-")]
    exterior = [i for i in dev if exact[i]["id"].startswith("exterior-")]
    feasible = [i for i in dev if i not in exterior]
    diagnostics_dev = [
        (row, j) for row in diagnostics["exact"] for j, i in enumerate(row["indexes"]) if i in dev]
    observed_rank = []
    for row, j in diagnostics_dev:
        for spectrum in row["observed_singular"]:
            threshold = float(64*np.finfo(float).eps*200*spectrum[j][0])
            observed_rank.append(sum(sigma > threshold for sigma in spectrum[j]))
    detail_coverage = len(diagnostics_dev) == 2*len(dev)
    monotone = detail_coverage and all(after[j] <= before[j] for row, j in diagnostics_dev
                                      for before, after in zip(row["before"], row["after"], strict=True))
    engineering = (all(classes[i]["reference_qualified"] for i in dev)
                   and all(row["qualified"] for name, record in records.items() if name != "wrong_damping_oracle"
                           for row in record["events"] if row["role"] in ("train", "val"))
                   and monotone and all(rank == 10 for rank in observed_rank)
                   and all(records["oracle"]["events"][i]["truth_agreement"]
                           and records["oracle"]["events"][i]["trajectory"]["8"]["nrmse"] <= limits["oracle_nrmse"]
                           for i in dev))

    def history_cost(row):
        return 200*(row["history_nrmse"]**2+2*plan["prior_weight"]*row["memory_start_energy"]/plan["radius"]**2)

    eps = limits["objective_tolerance"]
    differences = [history_cost(radial[i])-history_cost(exact[i]) for i in exterior] if engineering else []
    boundary = (bool(differences) and min(differences) >= -eps and max(differences) >= eps
                and all(abs(2*exact[i]["memory_start_energy"]-plan["radius"]**2) <= 1e-9 for i in exterior))
    clean = engineering and all(no_prior[i]["truth_agreement"]
                                and no_prior[i]["trajectory"]["8"]["nrmse"] <= limits["nrmse"] for i in feasible)
    clean = clean and all(no_prior[i]["memory_start_error"]/plan["truth_radii"][0] <= limits["latent_relative"]
                          and no_prior[i]["memory_origin_error"]/plan["truth_radii"][0] <= limits["latent_relative"]
                          for i in interior)
    actual_prior = engineering and all(exact[i]["truth_agreement"] and exact[i]["trajectory"]["8"]["nrmse"]
                                       <= limits["nrmse"] for i in feasible)
    return bool(engineering), {"monotone_history": bool(monotone), "min_observed_rank": min(observed_rank, default=None),
                              "clean_recovery": bool(clean), "regularized_visible_future": bool(actual_prior),
                              "active_bound_improvement": bool(boundary), "active_history_improvements": differences}


def execute(root, directory, plan, budget):
    params = parameters()
    fine, truth, power, equilibrium, cases, classes = _population(root, plan, params, budget)
    train = np.array(cases["roles"]) == "train"
    hscale = np.sqrt(np.mean(truth[train, :11, :20]**2, axis=(0, 1))).clip(min=1e-5)
    scale = np.sqrt(np.mean(truth[train, 11:311, :20]**2, axis=(0, 1))).clip(min=1e-5)
    step_gap = np.max(np.abs((fine[:, :, :20]-truth[:, :, :20])/scale), axis=(1, 2))
    for row, gap in zip(classes, step_gap, strict=True):
        row["reference_step_gap"] = float(gap) if np.isfinite(gap) else None
        row["reference_qualified"] &= bool(gap <= plan["gates"]["reference_step_gap"])
    write_json(directory/"preprocessing.json", {"history": hscale.tolist(), "target": scale.tolist()})
    write_json(directory/"truth-classes.json", classes)
    np.savez_compressed(directory/"truth.npz", fine=fine, refined=truth, power=power, equilibrium=equilibrium,
                        ids=cases["ids"], roles=cases["roles"])
    known = make_model(params, "T0M1", "cuda").eval()
    controls = {"oracle": (True, False, 0.), "radial": (False, False, plan["prior_weight"]),
                "exact": (False, True, plan["prior_weight"]), "exact_no_prior": (False, True, 0.),
                "wrong_damping_oracle": (True, False, 0.)}
    records, diagnostics = {}, {}
    for name, (oracle, exact, prior) in controls.items():
        wrong = name == "wrong_damping_oracle"
        subset = slice(-len(plan["delayed_ids"]), None) if wrong else slice(None)
        model = (make_model((*params[:2], plan["wrong_damping_factor"]*params[2], *params[3:]), "T0M1", "cuda").eval()
                 if wrong else known)
        target, p, eq, labels = truth[subset], power[subset], equilibrium[subset], classes[subset]
        rows = {key: value[subset] for key, value in cases.items()}
        a = np.zeros_like(p)
        observer = partial(predict_from_topology_history, memory_radius=plan["radius"],
                           prior_weight=prior, exact_memory_ball=exact)
        details = [] if exact else None
        future, history, failures = simulate(model, target, a, model._tensor(p), model._tensor(eq), hscale, budget,
                                             oracle=oracle, observer=observer, diagnostics=details)
        score = scoring(model, rows, future, history, target, a, eq, params[0], scale, hscale, labels, oracle)
        matched_margins(model, future, target, labels, model._tensor(eq), score)
        score.update(failures=failures)
        records[name] = score
        if exact:
            diagnostics[name] = details
            write_json(directory/f"{name}-iterations.json", details)
        np.savez_compressed(directory/f"{name}-predictions.npz", forecasts=future, histories=history)
        write_json(directory/f"{name}-scores.json", score)
        print(f"Scored {name} on {len(target)} frozen synthetic events", flush=True)

    engineering, gates = _gates(plan, classes, records, diagnostics)
    science = engineering and all(gates[k] for k in ("clean_recovery", "regularized_visible_future",
                                                      "active_bound_improvement"))
    return {"engineering_pass": engineering, "scientific_pass": science, "gates": gates,
            "models": {name: record["roles"] for name, record in records.items()},
            "claim_boundary": "Frozen known-truth test of initialization geometry and damping mismatch."}
