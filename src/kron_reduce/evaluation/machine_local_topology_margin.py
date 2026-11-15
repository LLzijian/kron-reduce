"""Paired known-truth fit of trajectory and five-point loss objectives."""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_equation import ROLES, STEPS, align_classes
from kron_reduce.evaluation.machine_local_topology import (
    energy_residual,
    make_model,
    pack,
    scoring,
    simulate,
)
from kron_reduce.evaluation.machine_local_topology_learning import readout
from kron_reduce.evaluation.machine_local_topology_transfer import transfer_cases
from kron_reduce.evaluation.machine_local_topology_truth import parameters, truth_rollout
from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.evaluation.task3_stability_loss import loss_of_sync_margin
from kron_reduce.evaluation.trajectory_stability import classify
from kron_reduce.models.machine_local_topology import predict_from_topology_history

MAX_FRAMES = 300
HISTORY_FRAMES = 11


def fit_counts(truth_rows):
    counts = []
    for row in truth_rows:
        label, hit = row.get("label"), row.get("trigger_index")
        if label == "stable" and hit is None:
            counts.append(MAX_FRAMES)
        elif label == "unstable" and type(hit) is int and HISTORY_FRAMES <= hit < HISTORY_FRAMES+MAX_FRAMES:
            counts.append(hit-HISTORY_FRAMES+1)
        else:
            raise ValueError("truth requires a stable path or an eligible five-point trigger within three seconds")
    return np.array(counts, dtype=np.int64)


def margin(path, equilibrium, mass, lengths):
    size = len(path)
    return loss_of_sync_margin(path, equilibrium, mass.expand(size, -1),
                               path.new_full((size, 1, 1), 2*math.pi*60),
                               torch.as_tensor(lengths, dtype=torch.int64, device=path.device))


def fit(model, visible, power, equilibrium, lengths, scale, hscale, settings, weight, budget):
    """Use the same complete train update for both objectives."""
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], foreach=False, fused=False)
    trace, failure = [], None
    history = visible[:, :HISTORY_FRAMES]
    target = visible[:, HISTORY_FRAMES-1:]
    weights = visible.new_full((len(visible),), 1./len(visible))
    target_margin = margin(visible, equilibrium, model.mass, lengths)
    budget.spend("fits")
    for update in range(1, settings["updates"]+1):
        before = snapshot(model)
        optimizer.zero_grad(set_to_none=True)
        try:
            forecast = predict_from_topology_history(
                model, history[:, :, :10], history[:, :, 10:], power, equilibrium, hscale, MAX_FRAMES,
                max_step=.005, spend=budget.spend,
            )["forecast"][..., :20]
            trajectory = loss(forecast, target, weights, scale, lengths)
            predicted_margin = margin(torch.cat((history, forecast[:, 1:]), dim=1), equilibrium, model.mass, lengths)
            margin_mse = (predicted_margin-target_margin).square().mean()
            objective = trajectory+weight*margin_mse
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite paired training loss")
            budget.spend("gradients")
            objective.backward()
            if any(p.grad is None for p in model.parameters()):
                raise FloatingPointError("missing shared parameter gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite shared parameter gradient")
            budget.spend("optimizer_updates")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite shared parameter update")
            if any(not torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values()):
                raise FloatingPointError("nonfinite optimizer state")
        except NUMERICAL_ERRORS as error:
            model.load_state_dict(before)
            failure = {"attempted_update": update, "reason": str(error)}
            break
        trace.append({"update": update, "trajectory_mse": float(trajectory.detach()),
                      "margin_mse": float(margin_mse.detach()), "gradient_norm": float(norm),
                      "objective": float(objective.detach())})
        if update == 1 or update % 5 == 0:
            print(f"margin weight {weight:g} update {update}: {trace[-1]['objective']:.6g}", flush=True)
    return {"completed_updates": len(trace), "trace": trace, "failure": failure}


def matched_margins(model, paths, truth, labels, equilibrium, results):
    for index, row in enumerate(results["events"]):
        row["matched_margin"] = None
        if not row["qualified"]:
            continue
        try:
            lengths = fit_counts([labels[index]])
        except ValueError:
            continue
        target = truth[index:index+1, :HISTORY_FRAMES+MAX_FRAMES, :20]
        observed = target[:, :HISTORY_FRAMES]
        pred = [np.concatenate((observed, path[index:index+1, 1:MAX_FRAMES+1, :20]), axis=1) for path in paths]
        eq = equilibrium[index:index+1]
        values = [float(margin(model._tensor(path), eq, model.mass, lengths)[0]) for path in (target, *pred)]
        row["matched_margin"] = {"frames": int(lengths[0]), "truth": values[0],
                                 "predicted": dict(zip(map(str, STEPS), values[1:], strict=True))}
    for role in ROLES:
        for label in ("stable", "unstable"):
            rows = [row for row in results["events"] if row["role"] == role and row["truth_label"] == label]
            matched = [row["matched_margin"] for row in rows if row["matched_margin"] is not None]
            errors = [(entry["predicted"][str(STEPS[0])]-entry["truth"])**2 for entry in matched]
            summary = {"allocated": len(rows), "qualified": len(matched),
                       "rmse": float(np.sqrt(np.mean(errors))) if matched and len(matched) == len(rows) else None,
                       "predicted_triggers": sum(entry["predicted"][str(STEPS[0])] > 0 for entry in matched),
                       "truth_triggers": sum(entry["truth"] > 0 for entry in matched),
                       "pooled_nrmse": {h: float(np.sqrt(np.mean([row["trajectory"][h]["nmse"] for row in rows])))
                                        if rows and all(row["qualified"] for row in rows) else None
                                        for h in ("3", "8")}}
            results["roles"][role].setdefault("matched_margin", {})[label] = summary


def compare(initial, reference, candidate, gates):
    ref, new = reference["roles"]["val"], candidate["roles"]["val"]
    baseline, result = ref["matched_margin"], new["matched_margin"]
    loss_0 = initial["roles"]["val"]["matched_margin"]["unstable"]["rmse"]
    loss_a, loss_b = (row["unstable"]["rmse"] for row in (baseline, result))
    trajectory_pairs = [(group["pooled_nrmse"][h], control["pooled_nrmse"][h])
                        for group, control in ((new, ref), (result["stable"], baseline["stable"]))
                        for h in ("3", "8")]
    old, final = ref["stability"], new["stability"]
    return {"matched_unstable": all(value is not None for value in (loss_0, loss_a, loss_b))
            and min(loss_0, loss_a) > 0. and loss_b <= gates["unstable_margin_ratio"]*min(loss_0, loss_a),
            "matched_triggers": result["unstable"]["predicted_triggers"] >= baseline["unstable"]["predicted_triggers"],
            "trajectory": all(a is not None and b is not None and a <= gates["trajectory_ratio"]*b
                              for a, b in trajectory_pairs),
            "coverage_and_stability": final["resolved"] == final["allocated"]
            and final["fp"] == 0 and final["tp"]+final["tn"] >= old["tp"]+old["tn"]}


def save_equation(model, directory, name):
    write_json(directory/f"{name}-equation.json", readout(model))
    np.savez_compressed(directory/f"{name}-equation.npz", **{k: v.cpu().numpy() for k, v in snapshot(model).items()})


def execute(root, directory, plan, budget):
    cases, params = transfer_cases(plan), parameters()
    mass, k, d, _, _, ref = params
    roles = np.array(cases["roles"])
    is_train, development = roles == "train", roles != "test"
    power, a = cases["power"], np.zeros_like(cases["a"])
    equilibrium = np.broadcast_to(ref, power.shape).copy()
    fine, truth = (truth_rollout(params, cases["q"], cases["v"], power, a, cases["z"], count=810,
                                 max_step=step, spend=budget.spend) for step in (.00125, .000625))
    train = truth[is_train, :, :20]
    hscale = np.maximum(np.sqrt(np.mean(train[:, :HISTORY_FRAMES]**2, axis=(0, 1))), 1e-4)
    classes = [[classify(t[:11, :10], t[:11, 10:20], t[10:, :20], ref, mass) for t in path]
               for path in (fine, truth)]
    labels = classes[1]
    np.savez_compressed(directory/"truth.npz", fine=fine, refined=truth, power=power, equilibrium=equilibrium,
                        ids=cases["ids"], roles=roles)
    boundary = "Paired analytic T0M1 fits with fixed endpoints on generated train, val and test."
    try:
        truth_counts = fit_counts([labels[i] for i in np.flatnonzero(development)])
        if any({labels[i]["label"] for i in np.flatnonzero(roles == role)} != {"stable", "unstable"}
               for role in ("train", "val")):
            raise ValueError("each development role requires stable and unstable truth")
    except ValueError as error:
        write_json(directory/"truth-classes.json", classes)
        return {"engineering_pass": False, "scientific_pass": False, "claim_boundary": boundary,
                "truth_failure": str(error)}
    lengths = truth_counts[:len(train)]
    scale = np.maximum(np.sqrt(np.mean([np.mean(t[11:11+n]**2, axis=0)
                                       for t, n in zip(train, lengths, strict=True)], axis=0)), 1e-4)
    truth_gap = np.max(np.abs((fine[:, :, :20]-truth[:, :, :20])/scale), axis=(1, 2))
    for coarse, refined, gap in zip(*classes, truth_gap, strict=True):
        refined.update(reference_qualified=bool(gap <= 1e-6 and align_classes(coarse, refined)),
                       reference_step_gap=float(gap) if np.isfinite(gap) else None)
    write_json(directory/"truth-classes.json", classes)
    write_json(directory/"preprocessing.json", {"target": scale.tolist(), "history": hscale.tolist(),
                                                "train_lengths": lengths.tolist()})
    if not all(labels[i]["reference_qualified"] for i in np.flatnonzero(development)):
        return {"engineering_pass": False, "scientific_pass": False, "claim_boundary": boundary,
                "truth_failure": "independent NumPy dual-step qualification"}

    settings, start = plan["settings"], plan["settings"]["start"]
    anchor = (mass, k, d, np.full(10, start["port"]), np.geomspace(*start["tau_range"], 10), ref)
    initial = make_model(anchor, "T0M1", "cuda", train_scales=True)
    with torch.no_grad():
        initial.log_stiffness_factor.fill_(math.log(start["stiffness_scale"]))
        initial.log_damping_factor.fill_(math.log(start["damping_scale"]))
    p, eq = initial._tensor(power), initial._tensor(equilibrium)
    train_visible = initial._tensor(train[:, :HISTORY_FRAMES+MAX_FRAMES])
    frozen, traces = {"initial": initial}, {}
    for name, weight in (("trajectory", 0.), ("margin", settings["margin_weight"])):
        model = deepcopy(initial)
        traces[name] = fit(model, train_visible, p[is_train], eq[is_train], lengths, scale, hscale,
                           settings, weight, budget)
        write_json(directory/f"{name}-fit.json", traces[name])
        frozen[name] = model
    for name, model in frozen.items():
        model.requires_grad_(False).eval()
        save_equation(model, directory, name)
    results = {}
    for name, model in frozen.items():
        paths, histories, failures = simulate(model, truth, a, p, eq, hscale, budget, oracle=False)
        result = scoring(model, cases, paths, histories, truth, a, equilibrium, mass, scale, hscale, labels, False)
        matched_margins(model, paths, truth, labels, eq, result)
        balance = energy_residual(model, pack(model, truth[development, 10], a[development]),
                                  p[development], eq[development], budget.spend)
        result.update(failures=failures, energy_residual_max=balance)
        np.savez_compressed(directory/f"{name}-forecasts.npz", forecasts=paths, histories=histories)
        write_json(directory/f"{name}-scores.json", result)
        results[name] = result
    engineering = all(result["energy_residual_max"] <= 1e-10 for result in results.values()) and all(
        row["completed_updates"] == settings["updates"] for row in traces.values())
    gates = compare(results["initial"], results["trajectory"], results["margin"], plan["gates"])
    return {"engineering_pass": engineering, "scientific_pass": engineering and all(gates.values()),
            "claim_boundary": boundary, "gates": gates,
            "models": {name: {"roles": result["roles"], "energy_residual_max": result["energy_residual_max"]}
                       for name, result in results.items()},
            "fit_updates": {name: row["completed_updates"] for name, row in traces.items()}}
