"""Full-roster first-swing training with a fresh causal solve at every update."""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.first_swing_balanced_fit import robust_trajectory_loss
from kron_reduce.evaluation.first_swing_loss import (
    momentum_window_loss,
    observed_velocity_increment,
    one_sided_velocity_loss,
)
from kron_reduce.evaluation.kiki_machine_local_full import inputs
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.models.unified_discovery_m3 import predict_m3

TRAIN_STEP = .005


def _prepare_batches(model, events, scales, settings):
    if (not events or any(e["role"] != "train" or e["label"] not in {"stable", "unstable"} for e in events)
            or len({e["id"] for e in events}) != len(events)):
        raise ValueError("M3 fitting requires unique labeled train events")
    n = model.n_nodes
    if any(e["x"].ndim != 2 or e["x"].shape[1] != 2*n or len(e["x"]) < 12 for e in events):
        raise ValueError("train events require eleven history frames and at least one forecast frame")

    batches = []
    for offset in range(0, len(events), settings["train_batch"]):
        batch = events[offset:offset+settings["train_batch"]]
        lengths = [min(300, len(event["x"])-11) for event in batch]
        count = max(lengths)
        truth = np.stack([np.pad(event["x"][10:11+length], ((0, count-length), (0, 0)), mode="edge")
                          for event, length in zip(batch, lengths, strict=True)])
        q, v, power, equilibrium, history_scale = map(model._tensor, inputs(batch, scales["history"]))
        lengths = torch.tensor(lengths, device=q.device, dtype=torch.int64)
        stable = torch.tensor([event["label"] == "stable" for event in batch], device=q.device)
        weights = torch.where(stable, q.new_tensor(1.), q.new_tensor(settings["unstable_first_swing_weight"]))
        batches.append((q, v, power, equilibrium, history_scale, model._tensor(truth), lengths, stable, weights))
    return batches


def _terms(model, batch, population, target_scale, radius, settings, budget):
    q, v, power, equilibrium, history_scale, truth, lengths, stable, weights = batch
    n, count = model.n_nodes, truth.shape[1]-1
    result = predict_m3(model, q, v, power, equilibrium, history_scale, count, max_step=TRAIN_STEP,
                        spend=budget.spend, memory_radius=radius, prior_weight=settings["prior_weight"])
    visible = result["forecast"][..., :2*n]
    hidden = result["history"][:, -1, 2*n:]
    trajectory = robust_trajectory_loss(visible, truth, target_scale, lengths, stable,
                                       settings["huber_delta"]).sum()/population
    impulse = observed_velocity_increment(model, truth[:, :min(count, 40)+1], hidden, power, equilibrium,
                                          lengths, step=TRAIN_STEP, spend=budget.spend)
    weak = weights @ momentum_window_loss(truth[..., n:], impulse, target_scale[n:], lengths)/population
    asym = weights @ one_sided_velocity_loss(truth[..., n:], visible[..., n:], target_scale[n:], lengths)/population
    memory = hidden[:, :model.memory_dimension]
    excess = (.5*memory.square().sum(dim=1)/(.5*(2.*radius)**2)-1.).relu()
    energy_guard = settings["energy_weight"]*excess.square().sum()/population
    return {"trajectory": trajectory, "first_swing": weak+.5*asym, "weak_momentum": weak,
            "one_sided_velocity": asym, "memory_energy_guard": energy_guard}


def _calibrate(batch_terms, parameters, ratio, budget):
    keys = ("trajectory", "first_swing")
    params = tuple(parameters.values())
    gradients = {key: [torch.zeros_like(p) for p in params] for key in keys}
    for terms in batch_terms():
        for key in keys:
            if not torch.isfinite(terms[key]):
                raise FloatingPointError(f"nonfinite step-zero {key} loss")
            budget.spend("gradients")
            rows = torch.autograd.grad(terms[key], params, retain_graph=key == keys[0], allow_unused=True)
            for total, grad in zip(gradients[key], rows, strict=True):
                if grad is not None:
                    total.add_(grad)
        del terms

    vectors = {key: torch.cat([g.flatten() for g in rows]) for key, rows in gradients.items()}
    norms = {key: float(g.norm()) for key, g in vectors.items()}
    if not all(math.isfinite(value) and value > 0. for value in norms.values()):
        raise FloatingPointError("a step-zero gradient norm is nonfinite or zero")
    alpha = ratio*norms["trajectory"]/norms["first_swing"]
    if not math.isfinite(alpha) or alpha <= 0.:
        raise FloatingPointError("automatic first-swing weighting is nonfinite or zero")
    cosine = (vectors["trajectory"]/norms["trajectory"]) @ (vectors["first_swing"]/norms["first_swing"])
    return {"status": "PASS", "alpha_fs": alpha, "norms": norms,
            "scaled_first_swing_norm": alpha*norms["first_swing"], "target_ratio": ratio,
            "gradient_cosine": float(cosine),
            "gradient_vectors": {key: {name: g.flatten().tolist() for name, g in zip(parameters, rows, strict=True)}
                                 for key, rows in gradients.items()}}


def train_stage(model, train, scales, settings, budget, directory, *, parameter_groups, anchor, check):
    """Fit the given parameter groups and restore the best preregistered checkpoint."""
    updates, interval = settings["updates"], settings["val_interval"]
    if (type(updates) is not int or updates < 0 or type(interval) is not int or interval <= 0
            or type(settings["train_batch"]) is not int or settings["train_batch"] <= 0
            or not 1. <= settings["gradient_ratio"] <= 1.1):
        raise ValueError("invalid M3 update schedule, batch size, or gradient ratio")
    named = dict(model.named_parameters())
    grouped = [p for group in parameter_groups for p in group["params"]]
    group_ids = {id(p) for p in grouped}
    active = {name: p for name, p in named.items() if id(p) in group_ids}
    if (not grouped or len(grouped) != len(group_ids) or len(active) != len(group_ids)
            or any(not math.isfinite(g["lr"]) or g["lr"] <= 0. for g in parameter_groups)):
        raise ValueError("optimizer groups must contain distinct model parameters at positive learning rates")
    targets = {name: torch.as_tensor(anchor[name], dtype=p.dtype, device=p.device).detach().clone()
               for name, p in active.items()}
    if any(t.shape != active[name].shape or not torch.isfinite(t).all() for name, t in targets.items()):
        raise ValueError("fit anchors must be finite and match their parameter shapes")

    batches = _prepare_batches(model, train, scales, settings)
    target_scale = model._tensor(scales["target"])
    radius = scales["memory_radius"]

    def batch_terms():
        for batch in batches:
            yield _terms(model, batch, len(train), target_scale, radius, settings, budget)

    params = tuple(active.values())
    saved_flags = {name: p.requires_grad for name, p in named.items()}
    training = model.training
    directory.mkdir(parents=True, exist_ok=True)
    budget.spend("fits")
    for name, p in named.items():
        p.requires_grad_(name in active)
    model.train()
    best, checks, failures, completed = None, [], [], 0
    stop_reason = "update_cap"
    try:
        optimizer = torch.optim.Adam([dict(group) for group in parameter_groups], foreach=False, fused=False)
        try:
            calibration = _calibrate(batch_terms, active, settings["gradient_ratio"], budget)
        except NUMERICAL_ERRORS as error:
            calibration = {"status": "FAIL", "alpha_fs": None, "message": str(error)}
        write_json(directory/"gradient-calibration.json", calibration)
        alpha = calibration["alpha_fs"]
        print(f"{directory.name}: step-zero first-swing alpha={alpha}", flush=True)

        with (directory/"training.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for update in range(updates+1):
                if update:
                    if alpha is None:
                        stop_reason = "gradient_calibration_failure"
                        break
                    previous = snapshot(model)
                    optimizer.zero_grad(set_to_none=True)
                    totals = {}
                    batch_index = 0
                    try:
                        for batch_index, batch in enumerate(batches, start=1):
                            terms = _terms(model, batch, len(train), target_scale, radius, settings, budget)
                            trust = sum((p-targets[name]).square().sum() for name, p in active.items())
                            terms["anchor_penalty"] = settings["trust_weight"]*trust*len(batch[0])/len(train)
                            value = (terms["trajectory"]+alpha*terms["first_swing"]
                                     +terms["memory_energy_guard"]+terms["anchor_penalty"])
                            terms["objective"] = value
                            if not torch.isfinite(torch.stack(tuple(terms.values()))).all():
                                raise FloatingPointError("nonfinite train loss")
                            budget.spend("gradients")
                            value.backward()
                            for key in terms:
                                totals[key] = totals.get(key, 0.)+float(terms[key].detach())
                            del value, terms
                        if any(p.grad is None for p in params):
                            raise FloatingPointError("a selected parameter has no train gradient")
                        norm = torch.nn.utils.clip_grad_norm_(params, settings["gradient_clip"])
                        if not torch.isfinite(norm):
                            raise FloatingPointError("nonfinite full train gradient")
                        budget.spend("optimizer_updates")
                        optimizer.step()
                        state = (x for group_state in optimizer.state.values() for x in group_state.values()
                                 if isinstance(x, torch.Tensor))
                        if not all(torch.isfinite(x).all() for x in (*params, *state)):
                            raise FloatingPointError("nonfinite Adam parameter or optimizer state")
                        model.project_parameters_()
                    except NUMERICAL_ERRORS as error:
                        model.load_state_dict(previous)
                        optimizer.zero_grad(set_to_none=True)
                        failures.append({"attempted_update": update, "batch": batch_index, "error": str(error)})
                        stop_reason = "numerical_failure"
                        break
                    completed = update
                    totals.update(update=update, gradient_norm=float(norm),
                                  raw_parameters={name: p.detach().tolist() for name, p in active.items()})
                    stream.write(json.dumps(totals, allow_nan=False)+"\n")
                    stream.flush()
                    print(f"{directory.name}: update {update}/{updates} train objective={totals['objective']:.6g}",
                          flush=True)

                if update % interval and update != updates:
                    continue
                budget.spend("val_checks")
                model.eval()
                with torch.no_grad():
                    row = check(update)
                model.train()
                if row.get("update", update) != update or not row["rank"]:
                    raise ValueError("checkpoint callback returned an invalid update or empty rank")
                row = row | {"update": update, "rank": list(row["rank"])}
                rank = tuple(math.inf if value is None else value for value in row["rank"])
                write_json(directory/f"val-check-{update:03d}.json", row)
                checks.append({"update": update, "rank": row["rank"]})
                if best is None or rank < best[0]:
                    best = (rank, snapshot(model), row)
                print(f"{directory.name}: check {update} rank={row['rank']}", flush=True)

        model.load_state_dict(best[1])
        return {"completed_updates": completed, "selected_update": best[2]["update"], "selected_check": best[2],
                "stop_reason": stop_reason, "calibration": calibration, "checks": checks,
                "numerical_failures": failures}
    finally:
        for name, p in named.items():
            p.requires_grad_(saved_flags[name])
        model.train(training)
