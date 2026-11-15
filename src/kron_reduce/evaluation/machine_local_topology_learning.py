"""Fit shared coefficients of four known-support topology and memory models."""
from __future__ import annotations

import json
import math
from copy import deepcopy

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_equation import align_classes
from kron_reduce.evaluation.machine_local_topology import (
    SUPPORTS,
    energy_residual,
    make_model,
    pack,
    scoring,
    simulate,
)
from kron_reduce.evaluation.machine_local_topology_truth import (
    initial_cases,
    parameters,
    truth_rollout,
)
from kron_reduce.evaluation.trajectory_stability import classify
from kron_reduce.models.machine_local_topology import predict_from_topology_history
from project_harness.evidence import sha256_file


def objective(model, batch, spend):
    q, v, power, eq, hscale, target, scale = batch
    result = predict_from_topology_history(model, q, v, power, eq, hscale, target.shape[1], max_step=.005, spend=spend)
    errors = ((result["forecast"][:, 1:, :20]-target)/scale).square().mean(dim=2)
    if not torch.isfinite(errors).all():
        raise FloatingPointError("nonfinite causal trajectory loss")
    singular = result["singular"]
    condition = None if singular is None else (singular[:, 0]/singular[:, -1]).max()
    hidden = result["hidden_start"].detach()
    memory = model.memory_dimension
    limited = {"memory_on_bound": int((hidden[:, :memory].norm(dim=1) >= 1-1e-12).sum()),
               "coupling_on_bound": int((hidden[:, memory:].norm(dim=1) >= .6-1e-12).sum()),
               "max_last_gn_condition": None if condition is None else float(condition)}
    return errors, limited


def horizon_loss(errors, short):
    return errors.mean(dim=1) if short is None else .5*(errors[:, :short].mean(dim=1)+errors.mean(dim=1))


def gradient_check(model, batch, budget, short):
    params = dict(model.named_parameters())
    rows = {}
    try:
        errors, baseline = objective(model, batch, budget.spend)
        budget.spend("gradients")
        gradients = torch.autograd.grad(horizon_loss(errors, short).mean(), tuple(params.values()))
        with torch.no_grad():
            for (name, parameter), gradient in zip(params.items(), gradients, strict=True):
                value = parameter.clone()
                direction = torch.linspace(.7, 1., value.numel(), dtype=value.dtype).reshape_as(value)
                analytic = float((gradient*direction).sum())
                differences, projection = [], []
                try:
                    for step in (1e-5, 5e-6):
                        scores = []
                        for sign in (1, -1):
                            parameter.copy_(value+sign*step*direction)
                            errors, limited = objective(model, batch, budget.spend)
                            scores.append(float(horizon_loss(errors, short).mean()))
                            projection.append(limited)
                        differences.append((scores[0]-scores[1])/(2*step))
                finally:
                    parameter.copy_(value)
                rows[name] = {"analytic": analytic if math.isfinite(analytic) else None, "central": differences,
                              "side_diagnostics": projection, "pass": math.isfinite(analytic) and all(
                                  abs(analytic-d) <= max(1e-8, 2e-3*abs(d)) for d in differences)}
    except NUMERICAL_ERRORS as error:
        return {"pass": False, "failure": str(error), "directions": rows}
    return {"pass": all(row["pass"] for row in rows.values()), "start": baseline, "directions": rows}


def fit(model, batches, settings, budget, directory):
    """Select the earliest complete-val minimum including the common start."""
    budget.spend("fits")
    train, val = batches
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], foreach=False, fused=False)
    before = snapshot(model)
    best, reference, stale = None, None, 0
    trace, val_trace = [], []
    completed, failure = 0, None
    reason = "update_cap"
    train_short = settings.get("train_short_frames", 50)
    val_short = settings.get("val_short_frames")
    gradient = gradient_check(model, train, budget, train_short)
    write_json(directory/"gradient.json", gradient)

    def check(update):
        nonlocal best, reference, stale
        budget.spend("val_checks")
        try:
            with torch.no_grad():
                errors, limited = objective(model, val, budget.spend)
                scores = horizon_loss(errors, val_short)
            score = float(scores.mean())
            if best is None or score < best[0]:
                best = (score, update, snapshot(model))
            if reference is None or score < reference-max(1e-12, 1e-3*reference):
                reference, stale = score, 0
            else:
                stale += 1
            row = {"update": update, "val_nmse": score, "val_event_nmse": scores.tolist(),
                   "stale": stale, "equation": readout(model)} | limited
        except NUMERICAL_ERRORS as error:
            stale = 0
            row = {"update": update, "val_nmse": None, "failure": str(error)}
        val_trace.append(row)
        write_json(directory/f"val-{update:03d}.json", row)
        print(f"{directory.name} update {update}: val={row['val_nmse']}", flush=True)

    check(0)
    if not gradient["pass"]:
        reason, failure = "gradient_check", gradient.get("failure", "shared parameter gradient disagreement")
    for update in range(1, settings["updates"]+1 if gradient["pass"] else 1):
        optimizer.zero_grad(set_to_none=True)
        try:
            errors, limited = objective(model, train, budget.spend)
            loss = horizon_loss(errors, train_short).mean()
            budget.spend("gradients")
            loss.backward()
            if any(p.grad is None for p in model.parameters()):
                raise FloatingPointError("missing shared parameter gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite shared parameter gradient")
            budget.spend("optimizer_updates")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                model.load_state_dict(before)
                raise FloatingPointError("nonfinite shared parameters")
        except NUMERICAL_ERRORS as error:
            reason, failure = "numerical_failure", str(error)
            break
        completed = update
        before = snapshot(model)
        trace.append({"update": update, "train_nmse_before_update": float(loss.detach()),
                      "gradient_norm": float(norm)} | limited)
        if update % settings["val_interval"] == 0:
            check(update)
            if update >= settings["min_updates"] and stale >= settings["val_patience"]:
                reason = "val_plateau"
                break
    if completed != val_trace[-1]["update"]:
        check(completed)
    if best is not None:
        model.load_state_dict(best[2])
    record = {"completed_updates": completed, "selected_update": best[1] if best else completed,
              "selected_val_nmse": best[0] if best else None, "reason": reason, "failure": failure,
              "train_trace": trace, "val_trace": val_trace}
    write_json(directory/"selection.json", record)
    return record


def readout(model):
    return {
        "stiffness_scale": math.exp(float(model.log_stiffness_factor.detach())),
        "damping_scale": math.exp(float(model.log_damping_factor.detach())),
        "port_amplitude": model.port_parameter.detach().exp().tolist() if model.memory_dimension else [],
        "tau": model.tau.detach().tolist(),
    }


def selected_model(root, source, support, params, *, device):
    path = (root/"results"/source["run_id"]/support/"selected-equation.json").resolve(strict=True)
    if (not path.is_relative_to(root/"results")
            or sha256_file(path) != source["equation_sha256"][support]):
        raise RuntimeError("frozen S1 equation differs from the registered source")
    state = json.loads(path.read_bytes())
    mass, k, d, _, _, eq = params
    g, tau = (np.asarray(state[key], dtype=float) for key in ("port_amplitude", "tau"))
    if g.shape != mass.shape or tau.shape != mass.shape:
        raise ValueError("invalid selected memory coefficients")
    model = make_model((mass, k, d, g, tau, eq), support, device, train_scales=True)
    with torch.no_grad():
        model.log_stiffness_factor.fill_(math.log(state["stiffness_scale"]))
        model.log_damping_factor.fill_(math.log(state["damping_scale"]))
    return model


def memory_kernel_error(model, g, tau):
    if not model.memory_dimension:
        return None
    projection = model.power_projection.detach().cpu().numpy()
    t = np.geomspace(.01, 3., 32)

    def kernel(c, rates):
        return np.einsum("ij,tj,kj->tik", c, np.exp(-t[:, None]/rates), c)

    truth = kernel(projection*g, tau)
    estimate = kernel(model.coupling.detach().cpu().numpy(), model.tau.detach().cpu().numpy())
    return float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))


def execute(root, directory, plan, budget):
    cases, params = initial_cases(), parameters()
    mass, k, damping, g, tau, eq = params
    power = cases["power"]
    equilibrium = np.broadcast_to(eq, power.shape).copy()
    roles = np.array(cases["roles"])
    if ({r: int((roles == r).sum()) for r in ("train", "val", "test")} != plan["cohorts"]
            or cases["ids"] != plan["ids"]):
        raise RuntimeError("synthetic roster differs from registration")
    settings = plan["settings"]
    source = plan.get("source")
    if source is not None and sha256_file(root/"results"/source["run_id"]/"artifact_hashes.json") != source["manifest_sha256"]:
        raise RuntimeError("S1 manifest identity differs")
    count = settings.get("future_frames", 100)
    supports = SUPPORTS if source is None else tuple(source["equation_sha256"])
    records = {}
    for support in supports:
        output = directory/support
        output.mkdir()
        if source is None:
            start = settings["start"]
            anchor = mass, k, damping, np.full(10, start["port"]), np.geomspace(*start["tau_range"], 10), eq
            model = make_model(anchor, support, "cpu", train_scales=True)
            with torch.no_grad():
                model.log_stiffness_factor.fill_(math.log(start["stiffness"]))
                model.log_damping_factor.fill_(math.log(start["damping"]))
        else:
            model = selected_model(root, source, support, params, device="cpu")
        initial = deepcopy(model)
        a = cases["a"] if model.topology_dimension else np.zeros_like(cases["a"])
        z = cases["z"][:, :model.memory_dimension]
        fine, truth = (truth_rollout(params, cases["q"], cases["v"], power, a, z, count=810, max_step=step,
                                     spend=budget.spend) for step in (.00125, .000625))
        train = truth[roles == "train", :, :20]
        scale = np.maximum(np.sqrt(np.mean(train[:, 11:]**2, axis=(0, 1))), 1e-4)
        hscale = np.maximum(np.sqrt(np.mean(train[:, :11]**2, axis=(0, 1))), 1e-4)

        batches = []
        for role in ("train", "val"):
            rows = roles == role
            batches.append(tuple(model._tensor(x) for x in (truth[rows, :11, :10], truth[rows, :11, 10:20],
                power[rows], equilibrium[rows], hscale, truth[rows, 11:11+count, :20], scale)))
        fit_record = fit(model, batches, settings, budget, output)
        frozen = {"initial": initial.requires_grad_(False).eval(), "selected": model.requires_grad_(False).eval()}
        for name, candidate in frozen.items():
            write_json(output/f"{name}-equation.json", readout(candidate))
            np.savez_compressed(output/f"{name}-equation.npz", **{k: v.numpy() for k, v in snapshot(candidate).items()})
        np.savez_compressed(output/"truth.npz", fine=fine, refined=truth, scale=scale, history_scale=hscale,
                            ids=np.array(cases["ids"]), roles=roles, a=a, power=power, equilibrium=equilibrium)

        truth_classes = [[classify(t[:11, :10], t[:11, 10:20], t[10:, :20], equilibrium[i], mass)
                          for i, t in enumerate(p)] for p in (fine, truth)]
        development = roles != "test"
        truth_gap = float(np.max(np.abs((fine[development, :, :20]-truth[development, :, :20])/scale)))
        truth_ok = truth_gap <= 1e-6 and all(align_classes(c, d) for i, (c, d) in
                                           enumerate(zip(*truth_classes, strict=True)) if development[i])
        results = {}
        p, eq_batch = model._tensor(power).to("cuda"), model._tensor(equilibrium).to("cuda")
        for name, candidate in frozen.items():
            candidate.to("cuda")
            paths, histories, failures = simulate(candidate, truth, a, p, eq_batch, hscale, budget, oracle=False)
            result = scoring(candidate, cases, paths, histories, truth, a, equilibrium, mass, scale, hscale,
                             truth_classes[1], False)
            equation = readout(candidate)
            errors = {key: float(np.max(np.abs(np.asarray(equation[key])/target-1.))) if candidate.memory_dimension else None
                      for key, target in (("port_amplitude", g), ("tau", tau))}
            result.update(failures=failures, equation=equation, memory_kernel_error=memory_kernel_error(candidate, g, tau),
                          memory_parameter_max_relative_error=errors)
            np.savez_compressed(output/f"{name}-forecasts.npz", forecasts=paths, histories=histories)
            write_json(output/f"{name}-scores.json", result)
            results[name] = result
        balance = energy_residual(model, pack(model, truth[development, 0], a[development]),
                                  p[development], eq_batch[development], budget.spend)
        selected = results["selected"]
        known = [event for event in selected["events"] if event["role"] != "test"]
        limit = plan["gates"]
        gates = {
            "valid_fit": fit_record["failure"] is None and fit_record["selected_val_nmse"] is not None,
            "physical": balance <= 1e-10,
            "truth": truth_ok,
            "trajectory": all(x["qualified"] and x["trajectory"]["3"]["nrmse"] <= limit["nrmse_3s"]
                              and x["trajectory"]["8"]["nrmse"] <= limit["nrmse_8s"] for x in known),
            "edge": all(x.get("edge_relative_error", np.inf) <= limit["edge_relative_error"] for x in known),
            "scales": all(abs(selected["equation"][name]-1.) <= limit["scale_relative_error"]
                          for name in ("stiffness_scale", "damping_scale")),
            "memory_kernel": (not model.memory_dimension or selected["memory_kernel_error"] <= limit["kernel_relative_error"]),
            "memory_parameters": all(x is None or x <= limit["memory_parameter_max_relative_error"]
                                     for x in selected["memory_parameter_max_relative_error"].values()),
        }
        records[support] = {"gates": gates, "fit": {k: v for k, v in fit_record.items() if not k.endswith("_trace")},
                            "energy_residual_max": balance, "truth_step_max": truth_gap,
                            "results": {k: {field: row[field] for field in
                                             ("roles", "equation", "memory_kernel_error", "memory_parameter_max_relative_error")}
                                        for k, row in results.items()}}
        print(f"{support}: {gates}", flush=True)
    return {"populations": records, "engineering_pass": all(row["gates"]["physical"] and row["gates"]["truth"]
                                                            for row in records.values()),
            "scientific_pass": all(all(row["gates"].values()) for row in records.values()),
            "claim_boundary": "Fixed-support, clean synthetic shared-coefficient learning; previously scored S0 cohorts."}
