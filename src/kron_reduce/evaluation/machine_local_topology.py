"""Paired synthetic checks of event-conditioned coupling and passive memory."""
from __future__ import annotations

import json

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_equation import ROLES, STEPS, align_classes
from kron_reduce.evaluation.machine_local_hidden_oracle import trajectory_scores
from kron_reduce.evaluation.machine_local_topology_truth import (
    initial_cases,
    parameters,
    truth_rollout,
)
from kron_reduce.evaluation.trajectory_stability import classify, summarize
from kron_reduce.models.machine_local_topology import (
    TopologyMemoryModel,
    predict_from_topology_history,
)
from kron_reduce.models.nonlinear_history import history_with_tangent
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from project_harness.evidence import sha256_file

SUPPORTS = ("T0M0", "T1M0", "T0M1", "T1M1")
FORECAST_COUNT = 800


def make_model(params, support, device, *, train_scales=False):
    mass, k, d, g, tau, _ = params
    empty = np.zeros((0, 0))
    backbone = NonlinearPassiveClosure(mass, k, d, np.zeros((len(mass), 0)), empty, empty)
    return TopologyMemoryModel(backbone, topology=support[1] == "1", port=g if support[-1] == "1" else None,
                               tau=tau if support[-1] == "1" else None,
                               train_scales=train_scales).to(device).requires_grad_(train_scales)


def pack(model, truth, a):
    basis = model.basis[:, :model.topology_dimension].cpu().numpy()
    return model._tensor(np.concatenate((truth[:, :20+model.memory_dimension], a@basis), axis=-1))


def initial_spectrum(model, start, power, equilibrium, scale, spend):
    if not model.latent_dimension:
        return None
    with torch.no_grad():
        _, tangent = history_with_tangent(model, start, power, equilibrium, max_step=.00125, spend=spend)
        design = (tangent[:, 1:, :20]/scale[None, None, :, None]).flatten(1, 2)
        design[:, :, model.memory_dimension:] *= .6

        def spectrum(matrix):
            if matrix.shape[-1] == 0:
                return [None]*len(matrix)
            spend("svd", len(matrix))
            return torch.linalg.svdvals(matrix).cpu().tolist()

        return list(zip(spectrum(design), spectrum(design[:, :, :model.memory_dimension]),
                        spectrum(design[:, :, model.memory_dimension:]), strict=True))


def energy_residual(model, state, power, equilibrium, spend):
    with torch.enable_grad():
        x = state.detach().requires_grad_(True)
        spend("gradients")
        gradient, = torch.autograd.grad(model.storage(x, equilibrium).sum(), x)
        spend("torch_rhs")
        derivative = model.rhs(x, power, equilibrium)
    v = x[:, 10:20]
    z = x[:, 20:20+model.memory_dimension]
    expected = (v*(power-v@model.effective_damping.T)).sum(-1)-(z.square()/model.tau).sum(-1)
    terms = gradient*derivative
    return float(((terms.sum(-1)-expected).abs()/terms.abs().sum(-1).clamp_min(1.)).max().detach())


def simulate(model, truth, a, power, equilibrium, scale, budget, *, oracle):
    size = len(truth)
    shape = (size, 801, 20+model.latent_dimension)
    paths, histories, errors = [], [], []
    for step in STEPS:
        future, history = np.full(shape, np.nan), np.full((size, 11, shape[-1]), np.nan)

        def forecast(rows, step=step, future=future, history=history):
            with torch.no_grad():
                if oracle:
                    origin = pack(model, truth[rows, 10], a[rows])
                    result = model.rollout(origin, power[rows], equilibrium[rows], FORECAST_COUNT,
                                           max_step=step, spend=budget.spend)
                    future[rows] = result.cpu().numpy()
                else:
                    result = predict_from_topology_history(model, truth[rows, :11, :10], truth[rows, :11, 10:20],
                                                           power[rows], equilibrium[rows], scale, FORECAST_COUNT,
                                                           max_step=step, spend=budget.spend)
                    future[rows] = result["forecast"].cpu().numpy()
                    history[rows] = result["history"].cpu().numpy()

        try:
            forecast(slice(None))
        except (FloatingPointError, torch.linalg.LinAlgError) as error:
            errors.append({"step": step, "batch": True, "error": str(error)})
            for i in range(size):
                try:
                    forecast(slice(i, i+1))
                except (FloatingPointError, torch.linalg.LinAlgError) as individual:
                    errors.append({"step": step, "index": i, "error": str(individual)})
        paths.append(future)
        histories.append(history)
    return np.stack(paths), np.stack(histories), errors


def scoring(model, cases, paths, histories, truth, a, equilibrium, mass, scale, hscale, truth_classes, oracle):
    events = []
    basis = model.basis[:, :model.topology_dimension].cpu().numpy()
    edge_basis = model.incidence.abs().cpu().numpy()
    k_ratio = (model.effective_stiffness/model.stiffness).detach().cpu().numpy()
    for i, target in enumerate(truth):
        c = truth_classes[i]
        history = histories[0, i]
        fine, refined = paths[:, i]
        args = (target[:11, :10], target[:11, 10:20])
        pred, pred_refined = (classify(*args, x[:, :20], equilibrium[i], mass) for x in (fine, refined))
        finite = np.isfinite(paths[:, i]).all()
        gap = float(np.max(np.abs((fine[:, :20]-refined[:, :20])/scale))) if finite else None
        coi = float(np.max(np.abs(paths[:, i, :, :20].reshape(2, 801, 2, 10)@mass))) if finite else None
        reference_ok = c.get("reference_qualified", True)
        qualified = bool(reference_ok and finite and gap <= 1e-5 and coi <= 1e-10 and align_classes(pred, pred_refined))
        row = {"id": cases["ids"][i], "role": cases["roles"][i], "truth_label": c["label"],
               "predicted_label": pred["label"], "qualified": qualified, "numerical_failure": not qualified,
               "prospective_eligible": reference_ok and (c["trigger_index"] is None or c["trigger_index"] > 10),
               "unsettled": qualified and pred["label"] is None, "step_gap": gap, "coi_error": coi,
               "trajectory": trajectory_scores(fine, target, scale) if qualified else None,
               "truth_agreement": qualified and align_classes(c, pred)}
        if not oracle and np.isfinite(history).all():
            r = model.memory_dimension
            estimated = history[0, 20+r:]@basis.T
            row.update(coupling_error=float(np.linalg.norm(estimated-a[i])),
                       history_nrmse=float(np.sqrt(np.mean(((history[1:, :20]-target[1:11, :20])/hscale)**2))),
                       edge_relative_error=float(np.max(np.abs(k_ratio*np.exp((estimated-a[i])@edge_basis)-1.))),
                       memory_start_error=float(np.linalg.norm(history[0, 20:20+r]-target[0, 20:20+r])),
                       memory_origin_error=float(np.linalg.norm(history[-1, 20:20+r]-target[10, 20:20+r])),
                       memory_start_energy=float(.5*np.sum(history[0, 20:20+r]**2)),
                       memory_origin_energy=float(.5*np.sum(history[-1, 20:20+r]**2)))
        events.append(row)

    roles = {}
    for role in ROLES:
        rows = [row for row in events if row["role"] == role]
        if not rows:
            continue
        valid = [row for row in rows if row["qualified"]]
        stability = summarize(rows)
        for key, good, total in (
            ("recall_on_resolved", stability["tp"], stability["tp"]+stability["fn"]),
            ("miss_rate_on_resolved", stability["fn"], stability["tp"]+stability["fn"]),
            ("false_alarm_rate_on_resolved", stability["fp"], stability["fp"]+stability["tn"]),
        ):
            stability[key] = good/total if total else None
        roles[role] = {"allocated": len(rows), "qualified": len(valid), "stability": stability,
                       "pooled_nrmse": {str(h): float(np.sqrt(np.mean([x["trajectory"][str(h)]["nmse"] for x in valid])))
                                        if len(valid) == len(rows) else None for h in (1, 2, 3, 5, 8)}}
    return {"roles": roles, "events": events}


def null_checks(params, cases, budget):
    outputs = {}
    for support in ("T1M0", "T0M1"):
        probe = params if support == "T1M0" else (*params[:4], np.full(10, .2), params[-1])
        model = make_model(probe, support, "cuda")
        start = model.mass.new_zeros((1, 20+model.latent_dimension))
        zero = model.mass.new_zeros((1, 10))
        if support == "T0M1":
            start[0, :10] = model._tensor(cases["q"][0])
            start[0, 10:20] = model._tensor(cases["v"][0])
        equilibrium = model._tensor(probe[-1][None, :])
        scale = model.mass.new_ones(20)
        (joint, _, _), = initial_spectrum(model, start, zero, equilibrium, scale, budget.spend)
        rank = int(np.count_nonzero(np.asarray(joint) > 64*np.finfo(float).eps*200*joint[0]))
        outputs[support] = {"rank": rank, "columns": model.latent_dimension, "singular": joint,
                            "pass": rank == (0 if support == "T1M0" else 9)}
    return outputs


def execute(root, directory, plan, budget):
    cases, params = initial_cases(), parameters()
    n = len(cases["ids"])
    mass = params[0]
    p = cases["power"]
    equilibrium = np.broadcast_to(params[-1], p.shape).copy()
    models = {support: make_model(params, support, "cuda") for support in SUPPORTS}
    records = {}
    for support, model in models.items():
        a = cases["a"] if model.topology_dimension else np.zeros_like(cases["a"])
        z = cases["z"][:, :model.memory_dimension]
        paths = [truth_rollout(params, cases["q"], cases["v"], p, a, z, count=810, max_step=step,
                               spend=budget.spend) for step in (.00125, .000625)]
        fine, truth = paths
        train = truth[np.array(cases["roles"]) == "train", :, :20]
        scale = np.maximum(np.sqrt(np.mean(train[:, 11:]**2, axis=(0, 1))), 1e-4)
        hscale = np.maximum(np.sqrt(np.mean(train[:, :11]**2, axis=(0, 1))), 1e-4)
        start = pack(model, truth[:, 0], a)
        power, eq = model._tensor(p), model._tensor(equilibrium)
        spectra = initial_spectrum(model, start, power, eq, model._tensor(hscale), budget.spend)
        balance = energy_residual(model, start, power, eq, budget.spend)
        classes = [[classify(t[:11, :10], t[:11, 10:20], t[10:, :20], equilibrium[i], mass)
                    for i, t in enumerate(path)] for path in paths]
        truth_gap = float(np.max(np.abs((fine[:, :, :20]-truth[:, :, :20])/scale)))
        truth_ok = truth_gap <= 1e-6 and all(align_classes(x, y) for x, y in zip(*classes, strict=True))
        results = {}
        candidates = [("oracle", model, True), ("causal", model, False)]
        if support == "T1M1":
            candidates.extend((other, models[other], False) for other in SUPPORTS[:-1])
        for name, candidate, oracle in candidates:
            paths, histories, failures = simulate(candidate, truth, a, power, eq, hscale, budget, oracle=oracle)
            results[name] = scoring(candidate, cases, paths, histories, truth, a, equilibrium, mass, scale, hscale,
                                    classes[1], oracle) | {"failures": failures}
            np.savez_compressed(directory/f"{support}-{name}.npz", forecasts=paths, histories=histories)
        np.savez_compressed(directory/f"{support}-truth.npz", fine=fine, refined=truth, a=a, power=p, equilibrium=equilibrium,
                            scale=scale, history_scale=hscale, ids=np.array(cases["ids"]))
        oracle = results["oracle"]["events"]
        causal = results["causal"]["events"]
        known = [row for row in causal if row["role"] in ("train", "val")]
        observed = [s for s, role in zip(spectra, cases["roles"], strict=True) if role != "test"] if spectra else []
        gates = {
            "physical": balance <= 1e-10,
            "oracle": truth_ok and all(x["qualified"] and x["trajectory"]["8"]["nrmse"] <= 1e-6
                                       for x in oracle if x["role"] != "test"),
            "observability": all(j[-1]/np.sqrt(200) >= 1e-6 and j[0]/j[-1] <= 1e8 for j, _, _ in observed),
            "causal": all(x["qualified"] and x["trajectory"]["8"]["nrmse"] <= 1e-3
                          and x.get("history_nrmse", np.inf) <= 1e-7
                          and x.get("edge_relative_error", np.inf) <= 1e-3
                          and x.get("memory_start_error", np.inf) <= 1e-3
                          and x.get("memory_origin_error", np.inf) <= 1e-3 for x in known),
        }
        records[support] = {"gates": gates, "energy_residual_max": balance, "truth_step_gap": truth_gap,
                            "true_start_spectra": None if not spectra else dict(zip(cases["ids"], spectra, strict=True)),
                            "results": results}
        write_json(directory/f"{support}.json", records[support])
        print(f"{support}: {n} cases in each control; {gates}", flush=True)
    negative = null_checks(params, cases, budget)
    passed = all(all(row["gates"].values()) for row in records.values()) and all(row["pass"] for row in negative.values())
    return {"engineering_pass": passed, "scientific_pass": False, "populations": records, "negative_controls": negative,
            "claim_boundary": "Analytic known-coefficient clean histories and paired supports; no shared-coefficient training."}


def closeout(root, directory, plan, budget):
    source = plan["source"]
    data = root/"results"/source["run_id"]
    for relative, digest in source["files"].items():
        if sha256_file(data/relative) != digest:
            raise RuntimeError(f"original S0 artifact differs: {relative}")
    records = {name: json.loads((data/f"{name}.json").read_text(encoding="utf-8")) for name in SUPPORTS}
    negatives = null_checks(parameters(), initial_cases(), budget)
    write_json(directory/"source.json", source)
    return {"engineering_pass": all(all(row["gates"].values()) for row in records.values())
            and all(row["pass"] for row in negatives.values()), "scientific_pass": False,
            "populations": records, "negative_controls": negatives, "source": source["run_id"],
            "claim_boundary": "Verified saved S0 positive records and isolated negative spectra; no shared-coefficient fitting."}
