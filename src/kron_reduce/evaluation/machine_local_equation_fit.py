"""Causal joint coefficient fitting and validation-only function selection."""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_full import inputs, predict_checked
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.models.machine_local_equation import JointMachineLocalModel

SUPPORTS = {"a": (False, False), "both": (True, True), "harmonic": (True, False),
            "damping": (False, True), "neither": (False, False)}
FAMILIES = ("harmonic_strength", "modulated_damping_strength")


def make_model(population, support):
    harmonic, damping = SUPPORTS[support]
    return JointMachineLocalModel(population["anchor"], port=population["port"], tau=population["tau"],
                                  harmonic=harmonic, modulated_damping=damping)


def objective_batches(model, population, role):
    events = population["roles"][role]
    args = tuple(model._tensor(x) for x in inputs(events, population["history_scale"]))
    scale = model._tensor(population["target_scale"])
    target = model._tensor(np.stack([e["x"][11:111] for e in events]))
    return args, target, scale


def objective(model, batch, budget):
    args, target, scale = batch
    prediction = predict_checked(model, *args, 100, max_step=.005, spend=budget.spend)
    errors = ((prediction["forecast"][:, 1:, :20]-target)/scale).square().mean(dim=2)
    by_event = .5*(errors[:, :50].mean(dim=1)+errors.mean(dim=1))
    value = by_event.mean()
    if not torch.isfinite(value):
        raise FloatingPointError("nonfinite joint objective")
    return value, by_event


def checkpoint(directory, model, optimizer, update):
    with (directory/f"checkpoint-{update:03d}.pt").open("xb") as stream:
        torch.save({"model": snapshot(model), "optimizer": optimizer.state_dict(), "update": update}, stream)


def fit(model, population, directory, budget, updates, *, shrink):
    directory.mkdir()
    train, val = (objective_batches(model, population, role) for role in ("train", "val"))
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=.01, foreach=False, fused=False)
    budget.spend("fits")
    completed = stagnant = 0
    comparator = best = None
    reason, failure = "update_cap", None
    rows = []
    with (directory/"training.jsonl").open("x", encoding="utf-8") as stream:
        for update in range(1, updates+1):
            before = snapshot(model)
            optimizer.zero_grad(set_to_none=True)
            try:
                value, _ = objective(model, train, budget)
                budget.spend("gradients")
                value.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError("invalid joint gradient")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=False)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient norm")
                budget.spend("optimizer_updates")
                optimizer.step()
                if (any(not torch.isfinite(p).all() for p in parameters)
                        or any(not torch.isfinite(v).all() for state in optimizer.state.values()
                               for v in state.values() if isinstance(v, torch.Tensor))):
                    raise FloatingPointError("nonfinite parameters or optimizer state")
                model.project_dictionary_(shrink)
            except NUMERICAL_ERRORS as error:
                checkpoint(directory, model, optimizer, -update)
                model.load_state_dict(before)
                reason, failure = "numerical_failure", str(error)
                break
            completed = update
            row = {"update": update, "train_nmse_before_update": float(value.detach()), "gradient_norm": float(norm)}
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()
            if update % 20:
                continue
            budget.spend("val_checks")
            try:
                with torch.no_grad():
                    val_value, val_events = objective(model, val, budget)
                score = float(val_value)
                row.update(val_nmse=score, val_event_nmse=val_events.tolist())
                if best is None or score < best[0]:
                    best = (score, update, snapshot(model))
                if comparator is None or score < comparator-1e-4*max(abs(comparator), 1e-12):
                    comparator, stagnant = score, 0
                else:
                    stagnant += 1
            except NUMERICAL_ERRORS as error:
                stagnant = 0
                row.update(val_nmse=None, val_failure=str(error))
            checkpoint(directory, model, optimizer, update)
            rows.append(row | {"stagnant_val_checks": stagnant})
            print(f"{directory.parent.name}/{directory.name} update {update}: train={row['train_nmse_before_update']:.6g}, val={row['val_nmse']}", flush=True)
            if update >= 80 and stagnant >= 3:
                reason = "val_plateau"
                break
    if best is not None:
        model.load_state_dict(best[2])
    row = {"completed_updates": completed, "selected_update": completed if best is None else best[1],
           "selected_val_nmse": None if best is None else best[0], "reason": reason, "failure": failure,
           "trace": rows}
    write_json(directory/"selection.json", row)
    return row


def refit_model(population, dense, support):
    model = make_model(population, support)
    shared = dense.state_dict()
    model.load_state_dict({k: shared[k] for k in model.state_dict()}, strict=True)
    return model


def choose_support(models, fits):
    options = [name for name in ("neither", "harmonic", "damping", "both")
               if fits[name]["selected_val_nmse"] is not None]
    if not options:
        return "both", None
    best = min(fits[name]["selected_val_nmse"] for name in options)
    allowance = max(1e-8, .01*best)

    def key(name):
        size = sum(float(getattr(models[name], k).detach()) > 0 for k in FAMILIES if getattr(models[name], k) is not None)
        return size, fits[name]["selected_val_nmse"]

    return min((name for name in options if fits[name]["selected_val_nmse"] <= best+allowance), key=key), allowance


def support_values(model):
    values = (getattr(model, name) for name in FAMILIES)
    return [0. if value is None else float(value.detach()) for value in values]


def freeze(model, directory):
    """Save the selected dynamics and explicit support before any formal scoring."""
    state = {name: value.cpu().numpy() for name, value in snapshot(model).items()}
    np.savez_compressed(directory/"frozen.npz", **state)
    scales = {"stiffness": model.effective_stiffness.detach().cpu().tolist(),
              "damping": model.effective_damping.detach().cpu().tolist(),
              "port_amplitude": model.port_parameter.detach().exp().cpu().tolist(),
              "tau": model.tau.detach().cpu().tolist(), "optional_strengths": support_values(model)}
    a, raw_b = scales["optional_strengths"]
    scales.update(harmonic_coefficient=a, modulated_damping_coefficient=8*raw_b,
                  declared_support=[name for name in FAMILIES if getattr(model, name) is not None],
                  nonzero_support=[name for name, value in zip(FAMILIES, (a, raw_b), strict=True) if value > 0])
    if any(not math.isfinite(x) for x in scales["optional_strengths"]):
        raise FloatingPointError("cannot freeze nonfinite function coefficients")
    write_json(directory/"equation.json", scales)
    return model.requires_grad_(False).eval()
