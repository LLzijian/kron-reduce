"""Matched TASK3 fits with calibrated first-swing gradients and fixed GN priors."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.first_swing_gn_decomposition import _distribution
from kron_reduce.evaluation.first_swing_loss import (
    momentum_window_loss,
    observed_velocity_increment,
    one_sided_velocity_loss,
)
from kron_reduce.evaluation.first_swing_probes import speed_peak_gap
from kron_reduce.evaluation.kiki_first_swing import STEPS, _witnesses
from kron_reduce.evaluation.kiki_machine_local_full import evaluate, inputs
from kron_reduce.evaluation.kiki_topology_momentum import ORIGIN_FRAME
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.models.first_swing_fit import (
    load_first_swing_scoring_view,
    predict_first_swing_fit,
)

GROUPS = {"hard_r": False, "soft_2r": True}


def _predict(model, events, scales, settings, budget, step, count, group):
    return predict_first_swing_fit(model, *inputs(events, scales["history"]), count, max_step=step,
        spend=budget.spend, memory_radius=scales["memory_radius"], prior_weight=settings["prior_weight"],
        soft_radius=GROUPS[group])


def robust_trajectory_loss(prediction, truth, scale, lengths, stable, delta):
    """Average the two event losses on the available 1 s and 3 s grids."""
    residual = (prediction[:, 1:]-truth[:, 1:])/scale
    square = residual.square()
    huber = torch.where(residual.abs() <= delta, square, 2*delta*residual.abs()-delta**2)
    frame_loss = torch.where(stable[:, None, None], huber, square).mean(dim=2)
    time = torch.arange(1, truth.shape[1], device=truth.device)
    loss = 0.
    for stop in (100, 300):
        valid = lengths.clamp(max=stop)
        loss = loss+(frame_loss*(time <= valid[:, None])).sum(dim=1)/(2*valid)
    return loss


def _terms(model, batch, population, scales, settings, budget, group):
    if not batch or any(event["role"] != "train" or event["label"] not in {"stable", "unstable"} for event in batch):
        raise ValueError("objective terms require labeled train events")
    lengths = np.array([min(300, len(event["x"])-11) for event in batch])
    count = int(lengths.max())
    truth = model._tensor(np.stack([np.pad(event["x"][10:11+n], ((0, count-n), (0, 0)), mode="edge")
                                    for event, n in zip(batch, lengths, strict=True)]))
    predicted = _predict(model, batch, scales, settings, budget, .005, count, group)
    visible = predicted["forecast"][..., :20]
    hidden = predicted["history"][:, -1, 20:]
    stable = torch.tensor([e["label"] == "stable" for e in batch], device=visible.device)
    lengths = torch.as_tensor(lengths, device=visible.device)
    scale = model._tensor(scales["target"])
    trajectory = robust_trajectory_loss(visible, truth, scale, lengths, stable, settings["huber_delta"]).sum()/population
    window = min(count, 40)
    power = model._tensor(np.stack([e["p0"] for e in batch]))
    equilibrium = model._tensor(np.stack([e["equilibrium"] for e in batch]))
    increment = observed_velocity_increment(model, truth[:, :window+1], hidden, power, equilibrium,
                                            lengths, step=.005, spend=budget.spend)
    weak = momentum_window_loss(truth[..., 10:], increment, scale[10:], lengths)
    one_sided = one_sided_velocity_loss(truth[..., 10:], visible[..., 10:], scale[10:], lengths)
    weights = torch.where(stable, 1., settings["unstable_first_swing_weight"]).to(visible.dtype)
    first_swing = (weights*(weak+.5*one_sided)).sum()/population
    limit = .5*(2.*scales["memory_radius"])**2
    guard = settings["energy_weight"]*(.5*hidden.square().sum(dim=1)/limit-1.).relu().square().sum()/population
    return {"trajectory": trajectory, "first_swing": first_swing, "weak_momentum": (weights*weak).sum()/population,
            "one_sided_velocity": (weights*one_sided).sum()/population, "memory_energy_guard": guard}


def _batches(events, size):
    for offset in range(0, len(events), size):
        yield offset, events[offset:offset+size]


def gradient_calibration(model, train, scales, settings, budget):
    names, parameters = zip(*((name, p) for name, p in model.named_parameters() if p.requires_grad), strict=True)
    if sum(p.numel() for p in parameters) != 23:
        raise RuntimeError("the paired fit requires 23 trainable coefficients")
    components = ("trajectory", "first_swing")
    gradients = {key: [torch.zeros_like(p) for p in parameters] for key in components}
    for _, batch in _batches(train, settings["train_batch"]):
        terms = _terms(model, batch, len(train), scales, settings, budget, "soft_2r")
        for i, key in enumerate(components):
            budget.spend("gradients")
            batch_gradient = torch.autograd.grad(terms[key], parameters, retain_graph=i == 0)
            for accumulated, individual in zip(gradients[key], batch_gradient, strict=True):
                accumulated.add_(individual)
        del terms
    norms = {key: float(torch.linalg.vector_norm(torch.cat([g.flatten() for g in rows])))
             for key, rows in gradients.items()}
    if not all(math.isfinite(value) and value > 0. for value in norms.values()):
        raise FloatingPointError("a step-zero gradient norm is nonfinite or zero")
    alpha = settings["gradient_ratio"]*norms["trajectory"]/norms["first_swing"]
    if not math.isfinite(alpha) or alpha <= 0.:
        raise FloatingPointError("the first-swing gradient coefficient is nonfinite or zero")
    vectors = {key: {name: row.flatten().tolist() for name, row in zip(names, rows, strict=True)}
               for key, rows in gradients.items()}
    damping = {key: vectors[key]["log_damping_factor"][0] for key in components}
    damping["scaled_first_swing"] = alpha*damping["first_swing"]
    damping["combined"] = damping["trajectory"]+damping["scaled_first_swing"]
    signs = {key: "positive" if value > 0. else "negative" if value < 0. else "zero" for key, value in damping.items()}
    print(f"Step-zero soft 2R: alpha={alpha:.8g}, scaled/trajectory={settings['gradient_ratio']}, "
          f"log damping partials {damping}", flush=True)
    return {"status": "PASS", "alpha_fs": alpha, "norms": norms, "gradient_vectors": vectors,
            "scaled_first_swing_norm": alpha*norms["first_swing"], "log_damping_partials": damping,
            "log_damping_signs": signs}


def trajectory_groups(result):
    stability = [row for row in result["stability_events"] if row["truth_label"] == "stable"]
    tn = {row["id"] for row in stability if row["qualified"] and row["predicted_label"] == "stable"}
    records = {row["event_id"]: row["rmse"] for row in result["records"]
               if row["horizon"] == "8" and row["label"] == "stable" and row["qualified"]}
    complete = len(records) == len(stability) and bool(stability)
    return {"allocated_stable": len(stability), "qualified_stable_8s": len(records),
            "mean_8s": float(np.mean(list(records.values()))) if complete else None,
            "median_8s": float(np.median(list(records.values()))) if complete else None,
            "tn_count": len(tn), "tn_qualified_8s": len(tn & records.keys()),
            "tn_mean_8s": float(np.mean([records[eid] for eid in sorted(tn)])) if tn and tn <= records.keys() else None}


def score(model, events, scales, budget, directory, role, settings, group, *, save):
    peak_rows = {str(step): {} for step in STEPS}

    def predict(batch, step):
        result = _predict(model, batch, scales, settings, budget, step, 800, group)
        traces = result["forecast"][..., :20]
        short = traces[:, :51].cpu().numpy()
        for index, event in enumerate(batch):
            if event["label"] == "unstable":
                peak_rows[str(step)][event["id"]] = speed_peak_gap(event, short[index])
        return result | {"forecast": traces, "gn_singular": result["singular"][None]}

    result = evaluate(model, events, scales["history"], budget, directory, role, settings, predictor=predict, save=save)
    unstable = [event["id"] for event in events if event["label"] == "unstable"]
    peaks = {}
    for dt, rows in peak_rows.items():
        pairs = {eid: rows.get(eid) for eid in unstable}
        peaks[dt] = {"rows": pairs, **_distribution([row["gap_rad_s"] if row else None for row in pairs.values()],
                                                   len(unstable))}
    extra = {"stable_errors": trajectory_groups(result), "first_500ms_peaks": peaks}
    if save:
        write_json(directory/f"{role}-first-swing.json", extra)
    return result | extra


@torch.no_grad()
def _witness_scores(model, rows, scales, settings, budget, group):
    events = [event for event, _ in rows]
    lengths = np.array([window["witness_end"]-ORIGIN_FRAME for _, window in rows])
    count = int(lengths.max())
    clips = [event["x"][ORIGIN_FRAME:window["witness_end"]+1] for event, window in rows]
    path = model._tensor(np.stack([np.pad(clip, ((0, count+1-len(clip)), (0, 0)), mode="edge") for clip in clips]))
    _, _, power, equilibrium, _ = inputs(events, scales["history"])
    records = {}
    for step in STEPS:
        error = None
        values = [None]*len(rows)
        try:
            origin = _predict(model, events, scales, settings, budget, step, 0, group)["history"][:, -1, 20:]
            impulses = observed_velocity_increment(model, path, origin, model._tensor(power), model._tensor(equilibrium),
                                                   lengths, step=step, spend=budget.spend)
            residual = (path[..., 10:]-path[:, :1, 10:]-impulses).cpu().numpy()
            for i, (_, window) in enumerate(rows):
                high, low = window["machines"]
                values[i] = float(residual[i, lengths[i], high]-residual[i, lengths[i], low])
        except NUMERICAL_ERRORS as failed:
            error = str(failed)
        records[str(step)] = _distribution(values, len(rows)) | {"numerical_error": error, "rows": [
            {"id": event["id"], **window, "signed_residual_rad_s": value}
            for (event, window), value in zip(rows, values, strict=True)]}
    return records


def checkpoint_metrics(result, witness, reference, settings, update):
    s, stable, peaks = (result[key] for key in ("stability", "stable_errors", "first_500ms_peaks"))
    numerical = s["uncovered"]+s["numerical_failures"]
    trajectory_covered = (s["allocated"] == 107 and numerical == 0 and stable["qualified_stable_8s"] == 95
                          and result["task3_six_term_score"] is not None
                          and all(x["qualified"] == 12 for x in peaks.values()))
    witness_covered = all(row["qualified"] == 33 for row in witness.values())
    valid = trajectory_covered and witness_covered
    val_pass = bool(trajectory_covered and s["tp"] == 12 and s["unsettled_predictions"] == 0
                    and s["fp"] <= settings["max_val_fp"] and stable["median_8s"] is not None
                    and stable["median_8s"] <= settings["max_val_stable_median"] and stable["tn_mean_8s"] is not None
                    and stable["tn_mean_8s"] <= settings["max_val_tn_mean"])
    witness_pass = witness_covered and all(abs(row["signed_median"]) <= settings["witness_relative_limit"]*abs(reference[dt])
                                            for dt, row in witness.items())
    joint = val_pass and witness_pass
    rank = [0 if joint else 1 if val_pass and valid else 2 if valid else 3, numerical, -s["tp"],
            s["unsettled_predictions"], s["fp"],
            max(row["positive_median"] for row in peaks.values()) if all(row["qualified"] == 12 for row in peaks.values()) else None,
            max(abs(row["signed_median"]) for row in witness.values()) if witness_covered else None,
            stable["median_8s"], stable["tn_mean_8s"], stable["mean_8s"], update]
    return {"update": update, "rank": rank, "val_pass": val_pass, "witness_pass": witness_pass, "joint_pass": joint,
            "target_fp_pass": joint and s["fp"] <= settings["target_val_fp"], "stability": s, "stable_errors": stable,
            "task3_six_term_score": result["task3_six_term_score"], "first_500ms_peaks": peaks, "witness": witness}


def _verify_start(check, baseline, group):
    key = "fixed_soft_2p0" if group == "soft_2r" else "fixed"
    old = baseline["scores"][key]["val"]
    if any(check["stability"][k] != old["stability"][k] for k in
           ("tp", "tn", "fp", "fn", "uncovered", "unsettled_predictions", "numerical_failures")):
        raise RuntimeError("step-zero classification differs from the archived initializer")
    for current, saved in ((check["stable_errors"]["mean_8s"], old["stable_8s"]["mean"]), *[
            (check[key_name][dt][metric], baseline[source][dt][key][metric])
            for dt in map(str, STEPS) for key_name, source, metric in
            (("first_500ms_peaks", "peaks", "positive_median"), ("witness", "witness", "signed_median"))]):
        if not np.isclose(current, saved, rtol=1e-9, atol=1e-11):
            raise RuntimeError("step-zero trajectory or witness score differs from the archived initializer")


def fit(model, scoring_view, train, val, scales, settings, budget, directory: Path, group, calibration, baseline):
    budget.spend("fits")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    anchor = tuple(p.detach().clone() for p in params)
    optimizer = torch.optim.Adam(params, lr=settings["learning_rate"], foreach=False, fused=False)
    reference = {dt: row["fixed"]["signed_median"] for dt, row in baseline["witness"].items()}
    rows, missing = _witnesses(train)
    best, checks, failures, completed = None, [], [], 0
    stop_reason = "update_cap"
    with (directory/"training.jsonl").open("x", encoding="utf-8") as stream:
        for update in range(settings["updates"]+1):
            if update:
                if calibration["alpha_fs"] is None:
                    stop_reason = "gradient_calibration_failure"
                    break
                previous = snapshot(model)
                optimizer.zero_grad(set_to_none=True)
                totals = {}
                try:
                    for offset, batch in _batches(train, settings["train_batch"]):
                        terms = _terms(model, batch, len(train), scales, settings, budget, group)
                        trust = settings["trust_weight"]*sum((p-a).square().sum() for p, a in zip(params, anchor, strict=True))
                        terms["anchor_penalty"] = trust*(len(batch)/len(train))
                        value = (terms["trajectory"] + calibration["alpha_fs"]*terms["first_swing"]
                                 + terms["memory_energy_guard"] + terms["anchor_penalty"])
                        terms["objective"] = value
                        if not all(torch.isfinite(term) for term in terms.values()):
                            raise FloatingPointError("nonfinite train loss")
                        budget.spend("gradients")
                        value.backward()
                        for key, term in terms.items():
                            totals[key] = totals.get(key, 0.)+float(term.detach())
                        del value, terms
                    if any(p.grad is None for p in params):
                        raise FloatingPointError("missing shared gradient")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(params, settings["gradient_clip"])
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("nonfinite shared gradient norm")
                    budget.spend("optimizer_updates")
                    optimizer.step()
                    if not all(torch.isfinite(x).all() for x in (*params, *[item for state in optimizer.state.values()
                            for item in state.values() if isinstance(item, torch.Tensor)])):
                        raise FloatingPointError("nonfinite Adam parameters or state")
                    with torch.no_grad():
                        model.port_parameter.clamp_min_(math.log(settings["minimum_port"]))
                        model.log_stiffness_factor.clamp_(math.log(.5), math.log(2.))
                        model.log_damping_factor.clamp_(math.log(.5), math.log(2.))
                    completed += 1
                except NUMERICAL_ERRORS as original:
                    model.load_state_dict(previous)
                    optimizer.zero_grad(set_to_none=True)
                    failures.append({"attempted_update": update, "batch_offset": offset, "error": str(original)})
                    stop_reason = "numerical_failure"
                    break
                totals.update(update=update, gradient_norm=float(gradient_norm),
                              rho=float(model.angle_damping_rho.detach()), log_damping=float(model.log_damping_factor.detach()))
                stream.write(json.dumps(totals, allow_nan=False)+"\n")
                stream.flush()
                print(f"{group} update {update}/{settings['updates']} train objective={totals['objective']:.6g}", flush=True)
            if update % settings["val_interval"] and update != settings["updates"]:
                continue
            budget.spend("val_checks")
            load_first_swing_scoring_view(model, scoring_view)
            result = score(scoring_view, val, scales, budget, directory, "val", settings, group, save=False)
            witness = _witness_scores(scoring_view, rows, scales, settings, budget, group)
            check = checkpoint_metrics(result, witness, reference, settings, update)
            if update == 0:
                _verify_start(check, baseline, group)
            write_json(directory/f"val-check-{update:03d}.json", check)
            checks.append({key: check[key] for key in ("update", "rank", "val_pass", "witness_pass", "joint_pass")})
            rank = tuple(math.inf if x is None else x for x in check["rank"])
            if best is None or rank < best[0]:
                best = (rank, snapshot(model), check)
            print(f"{group} check {update}: {check['stability']['tp']}/12, FP={check['stability']['fp']}, "
                  f"joint={check['joint_pass']}", flush=True)
    model.load_state_dict(best[1])
    model.requires_grad_(False).eval()
    return {"completed_updates": completed, "selected_update": best[2]["update"], "stop_reason": stop_reason,
            "selected_val": {k: v for k, v in best[2].items() if k != "witness"}, "selected_witness": best[2]["witness"],
            "without_witness": missing, "checks": checks, "numerical_failures": failures,
            "angle_damping_rho": float(model.angle_damping_rho), "port_gains": model.port_parameter.exp().tolist()}


def compact_score(result):
    return {key: result[key] for key in ("stability", "stable_complete", "unstable_prefix", "stable_errors",
                                         "task3_six_term_score", "first_500ms_peaks")} | {
        "max_origin_memory_energy": max((row["hidden_origin_energy"] for row in result["initialization"]), default=None)}
