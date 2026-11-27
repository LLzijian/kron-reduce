"""Conditional full-train fit of the V3 initializer with frozen ES-ROM equations."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.noise_stress_fit_v2 import (
    ADAM_EPSILON,
    ADAM_MOMENTS,
    DIFFERENCE_STEP,
    HORIZONS,
    LEARNING_RATE,
    PERTURBATION_SEED,
    _errors,
    _train_bank,
)
from kron_reduce.evaluation.noise_stress_predict import STEPS, observation_inputs
from kron_reduce.models.noise_robustness_mhe_v3 import predict_mhe_v3
from project_harness.evidence import sha256_file

PARAMETERS = ("alpha", "c_var", "kappa_term", "huber_delta")
PARAMETER_LIMITS = ((1e-5, 10.0), (0.5, 6.0), (0.5, 6.0), (0.5, 4.0))
SCENARIO_SEED_NAMESPACE = "esrom_init_train_noise_v3"
TRAIN_COUNT = 346
EQUATION_PARAMETERS = 75


def _equation_binding(model):
    parameters = tuple(model.parameters())
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.asarray(value.detach().cpu().numpy(), dtype="<f8")
        digest.update(json.dumps((name, array.shape)).encode())
        digest.update(array.tobytes())
    return {"parameter_count": sum(p.numel() for p in parameters),
            "all_parameters_frozen": not any(p.requires_grad for p in parameters),
            "state_sha256": digest.hexdigest()}


def _candidate(log_params, initial, name):
    values = dict(zip(PARAMETERS, np.exp(log_params).tolist(), strict=True))
    return initial | {"name": name} | values


@torch.no_grad()
def _objective(model, events, indices, inputs, score_scale, memory_radius, batch, candidate, operations):
    breakdown = {}
    options = {key: candidate[key] for key in (*PARAMETERS, "gate_rad", "inverse_snr")}

    def spend(key, amount=1):
        operations[key] += amount

    for step in STEPS:
        terms = {}
        for label, horizon in HORIZONS.items():
            term = "stable_nrmse_3s" if label == "stable" else "positive_peak_deficit_rad_s_500ms"
            expected, measured, failures = len(indices[label]), [], []
            for start in range(0, expected, batch):
                selected = indices[label][start:start + batch]
                chunk = tuple(values[selected] for values in inputs[:-1]) + (inputs[-1],)
                rows = [events[i] for i in selected]
                spend("allocated_train_predictions", len(rows))
                try:
                    origin = predict_mhe_v3(
                        model, *chunk, 0, max_step=step, spend=spend, memory_radius=memory_radius, **options,
                    )["forecast"][:, 0]
                    if not torch.isfinite(origin).all():
                        raise FloatingPointError("nonfinite V3 forecast origin")
                    paths = model.rollout(origin, chunk[2], chunk[3], horizon, max_step=step, spend=spend)
                    values = _errors(label, rows, paths[:, :, :20].cpu().numpy(), score_scale)
                except NUMERICAL_ERRORS as error:
                    failures.append({"ids": [row["id"] for row in rows], "error": str(error)})
                    continue
                finite = np.isfinite(values)
                measured.extend(values[finite])
                if not finite.all():
                    failures.append({
                        "ids": [row["id"] for row, ok in zip(rows, finite, strict=True) if not ok],
                        "error": "uncovered truth horizon or nonfinite trajectory loss",
                    })
            # Dividing each term first avoids overflow in the sum of finite losses.
            mean = math.fsum(value / expected for value in measured) if expected and len(measured) == expected else None
            terms[term] = {"mean": mean, "evaluated": len(measured), "allocated": expected, "failures": failures}
        stable = terms["stable_nrmse_3s"]["mean"]
        peak = terms["positive_peak_deficit_rad_s_500ms"]["mean"]
        loss = stable + peak / float(score_scale[10]) if stable is not None and peak is not None else None
        breakdown[str(step)] = {"loss": loss if loss is None or math.isfinite(loss) else None, **terms}
    step_losses = [entry["loss"] for entry in breakdown.values()]
    loss = sum(value / len(STEPS) for value in step_losses) if all(x is not None for x in step_losses) else None
    return {"loss": loss, "steps": breakdown}


def fit_candidates(directory, train, models, scales, plan, initial, evaluate):
    """Fit after rejection at update zero and keep the first passing full-val checkpoint.

    ``evaluate(checkpoint, candidate, update)`` writes and returns its selection
    after the entire frozen val matrix. Its result only controls stopping.
    """
    limits = plan["conditional_fit"]
    budget, interval = limits["train_fit_budget"], limits["validation_interval"]
    namespace = limits["scenario_seed_namespace"]
    ids = [event["id"] for event in train]
    labels = np.asarray([event["label"] for event in train])
    if (initial["passed"] is not False or initial["optimizer_updates"] != 0
            or len(train) != TRAIN_COUNT or len(set(ids)) != TRAIN_COUNT or plan["counts"]["train"] != TRAIN_COUNT
            or {event["role"] for event in train} != {"train"} or set(labels) != set(HORIZONS)
            or type(budget) is not int or not 0 < budget <= 15 or interval != 3 or budget % interval
            or tuple(limits["allowed_parameters"]) != PARAMETERS
            or namespace != SCENARIO_SEED_NAMESPACE or limits["learning_rate"] != LEARNING_RATE):
        raise PermissionError("fit requires full train, zero-update rejection, and the registered V3 schedule")
    start = initial["candidate"]
    initial_params = np.asarray([start[key] for key in PARAMETERS], dtype=np.float64)
    if not np.isfinite(initial_params).all() or np.any(initial_params <= 0):
        raise ValueError("initial parameters must be positive and finite")
    log_params = np.log(initial_params)
    low, high = np.log(np.array(PARAMETER_LIMITS)).T
    low, high = low + DIFFERENCE_STEP, high - DIFFERENCE_STEP
    if not np.all((low <= log_params) & (log_params <= high)):
        raise ValueError("initial parameters do not support the registered symmetric probes")

    model = models["m3"]
    equations_before = _equation_binding(model)
    if equations_before["parameter_count"] != EQUATION_PARAMETERS or not equations_before["all_parameters_frozen"]:
        raise PermissionError("fit requires all 75 frozen ES-ROM equation parameters")
    score_scales = np.asarray(plan["score_scales"], dtype=np.float64)
    if score_scales.shape != (2,) or not np.isfinite(score_scales).all() or np.any(score_scales <= 0):
        raise ValueError("expected positive frozen angle and speed score scales")
    score_scale = np.repeat(score_scales, 10)
    batch = plan["trajectory_batch"]
    if type(batch) is not int or not 0 < batch <= 256:
        raise ValueError("the full-train rollout batch must be between 1 and 256")

    indices = {label: np.flatnonzero(labels == label) for label in HORIZONS}
    rng = np.random.default_rng(np.random.PCG64(PERTURBATION_SEED))
    first_moment, second_moment = np.zeros(len(PARAMETERS)), np.zeros(len(PARAMETERS))
    beta1, beta2 = ADAM_MOMENTS
    operations = Counter()
    training, checkpoints = [], []
    result, applied, status = initial, 0, "REJECT"

    for update in range(1, budget + 1):
        bank = _train_bank(ids, namespace, update)
        inputs = observation_inputs(train, bank, scales)
        signs = rng.choice([-1.0, 1.0], size=len(PARAMETERS))

        def objective(candidate, current_inputs=inputs):
            return _objective(model, train, indices, current_inputs, score_scale,
                              scales["memory_radius"], batch, candidate, operations)

        row = {"update": update, "sample_count": len(train), "noise_sha256": bank.sha256,
               "perturbation": signs.tolist(), "log_parameter_gradient": None, "loss": None, "applied": False}
        probes = {}
        for name, direction in (("lower", -signs), ("upper", signs)):
            trial = _candidate(log_params + DIFFERENCE_STEP * direction, start, name)
            probes[name] = {"candidate": trial, **objective(trial)}
        row["probes"] = probes
        lower, upper = (probes[name]["loss"] for name in ("lower", "upper"))
        if lower is not None and upper is not None:
            gradient = ((upper - lower) / (2 * DIFFERENCE_STEP)) * signs
            with np.errstate(over="ignore", invalid="ignore"):
                second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
            if np.isfinite(second_moment).all():
                first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
                adam_direction = (first_moment / (1.0 - beta1**update)) / (
                    np.sqrt(second_moment / (1.0 - beta2**update)) + ADAM_EPSILON)
                trial_params = np.clip(log_params - LEARNING_RATE * adam_direction, low, high)
                candidate = _candidate(trial_params, start, f"train_update_{update}")
                row.update(candidate=candidate, log_parameter_gradient=gradient.tolist(), **objective(candidate))
            else:
                row["error"] = "nonfinite Adam second moment"

        equations_after = _equation_binding(model)
        if equations_after != equations_before:
            status = "FROZEN_EQUATIONS_CHANGED"
        elif row["loss"] is None:
            status = "NUMERICAL_FAILURE"
        else:
            row["applied"], log_params, applied = True, trial_params, update
        training.append(row)
        write_json(directory / f"train-update-{update:02d}.json", row)
        print(f"v3 full train update {update}/{budget}: loss={row['loss']}", flush=True)
        if not row["applied"]:
            break
        if update % interval == 0:
            checkpoint = directory / f"val-update-{update:02d}"
            checkpoint.mkdir()
            result = evaluate(checkpoint, candidate.copy(), update)
            selection_file = checkpoint / "selection.json"
            if (type(result["passed"]) is not bool or result["candidate"] != candidate
                    or result["optimizer_updates"] != update or json.loads(selection_file.read_bytes()) != result):
                raise ValueError("the checkpoint selection does not match the fitted V3 candidate")
            checkpoints.append({"update": update, "passed": result["passed"], "candidate": candidate,
                                "metrics": result["evaluated"], "selection_sha256": sha256_file(selection_file)})
            equations_after = _equation_binding(model)
            if equations_after != equations_before:
                status = "FROZEN_EQUATIONS_CHANGED"
                break
            if result["passed"]:
                status = "ACCEPT"
                break

    selection = result | {"passed": status == "ACCEPT", "optimizer_updates": applied,
                          "optimizer_attempts": len(training), "selected_checkpoint_update": result["optimizer_updates"],
                          "train_fit_status": status}
    write_json(directory / "training_curves.json", {
        "train": training, "val": checkpoints, "operations": dict(operations), "status": status,
        "train_roster_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
        "frozen_equations": {"before": equations_before, "after": equations_after},
    })
    write_json(directory / "selection.json", selection)
    return selection
