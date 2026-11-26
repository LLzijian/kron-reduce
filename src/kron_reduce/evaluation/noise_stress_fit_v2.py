"""Conditional full-train fit of the V2 initializer with frozen ES-ROM equations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import torch

from kron_reduce.data.task3_measurement_noise import NoiseBank, NoiseDraw
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.first_swing_probes import speed_peak_gap
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.noise_stress_predict import STEPS, observation_inputs
from kron_reduce.models.noise_robustness_mhe_v2 import predict_mhe_v2
from project_harness.evidence import sha256_file

PARAMETERS = ("alpha", "tau_mis", "huber_delta")
PARAMETER_LIMITS = ((1e-5, 10.0), (0.005, 0.25), (0.5, 4.0))
DIFFERENCE_STEP = 0.05
PERTURBATION_SEED = 20261127
SCENARIO_SEED_NAMESPACE = "esrom_init_train_noise_v2"
ADAM_MOMENTS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
LEARNING_RATE = 0.08
HORIZONS = {"stable": 300, "unstable": 50}


def _train_bank(ids, namespace, update):
    draws, digest = {}, hashlib.sha256()
    for event_id in sorted(ids):
        phase = hashlib.sha256(f"{namespace}|train|scenario|{event_id}".encode()).digest()[0]
        stress = (phase + update - 1) % 4
        key = f"{namespace}|train|update{update}|{event_id}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "little")
        rng = np.random.default_rng(np.random.PCG64(seed))
        scale = (0.010, 0.050) if stress < 2 else (0.0033, 0.010)
        if stress == 1:
            delta, omega = (rng.laplace(0.0, s / np.sqrt(2.0), (11, 10)) for s in scale)
        else:
            delta, omega = (rng.normal(0.0, s, (11, 10)) for s in scale)
        endpoint = None
        if stress == 2:
            for noise, magnitude in ((delta, 0.1), (omega, 0.5)):
                noise += (rng.random((11, 10)) < 0.05) * rng.choice([-1.0, 1.0], (11, 10)) * magnitude
        elif stress == 3:
            endpoint, generator = int(rng.choice([0, 10])), rng.integers(0, 10)
            delta[endpoint, generator] += rng.choice([-0.1, 0.1])
            omega[endpoint, generator] += rng.choice([-0.5, 0.5])
        draw = NoiseDraw(delta, omega, np.zeros(10), endpoint)
        draws[event_id] = draw
        for values in (draw.eps_delta, draw.eps_omega, draw.eps_pm):
            digest.update(values.astype("<f8", copy=False).tobytes())
    return NoiseBank(draws, digest.hexdigest())


def _candidate(log_params, initial, name):
    return initial | {"name": name} | dict(zip(PARAMETERS, np.exp(log_params).tolist(), strict=True))


def _errors(label, events, paths, score_scale):
    if label == "stable":
        available = [i for i, event in enumerate(events) if len(event["x"]) >= 311]
        if not available:
            return np.full(len(events), np.nan)
        truth = np.stack([events[i]["x"][11:311] for i in available])
        result = np.full(len(events), np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            result[available] = np.sqrt(np.square((paths[available, 1:] - truth) / score_scale).mean(axis=(1, 2)))
        return result
    gaps = [speed_peak_gap(event, path) for event, path in zip(events, paths, strict=True)]
    return np.array([
        max(0.0, gap["gap_rad_s"]) if gap is not None and np.isfinite(gap["gap_rad_s"]) else np.nan
        for gap in gaps
    ])


@torch.no_grad()
def _objective(model, events, indices, inputs, score_scale, memory_radius, batch, candidate, operations):
    breakdown, step_losses = {}, []
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
                    initial = predict_mhe_v2(
                        model, *chunk, 0, max_step=step, spend=spend, memory_radius=memory_radius, **options,
                    )["forecast"][:, 0]
                    paths = model.rollout(initial, chunk[2], chunk[3], horizon, max_step=step, spend=spend)
                    values = _errors(label, rows, paths[:, :, :20].cpu().numpy(), score_scale)
                except NUMERICAL_ERRORS as error:
                    failures.append({"ids": [row["id"] for row in rows], "error": str(error)})
                    continue
                finite = np.isfinite(values)
                measured.extend(values[finite])
                if not finite.all():
                    failures.append({"ids": [row["id"] for row, ok in zip(rows, finite, strict=True) if not ok],
                                     "error": "uncovered truth horizon or nonfinite trajectory loss"})
            mean = float(np.mean(measured)) if expected == len(measured) and expected else None
            terms[term] = {"mean": mean, "evaluated": len(measured), "allocated": expected, "failures": failures}
        stable = terms["stable_nrmse_3s"]["mean"]
        peak = terms["positive_peak_deficit_rad_s_500ms"]["mean"]
        step_loss = stable + peak / score_scale[10] if stable is not None and peak is not None else None
        breakdown[str(step)] = {"loss": step_loss, **terms}
        step_losses.append(step_loss)
    loss = float(np.mean(step_losses)) if all(value is not None for value in step_losses) else None
    return {"loss": loss, "steps": breakdown}


def fit_candidates(directory, train, models, scales, plan, initial, evaluate):
    """Run a new fit after zero-retrain rejection, with full-val checkpoint callbacks.

    ``evaluate(checkpoint, candidate, update)`` writes and returns its selection
    after the entire frozen val matrix. A callback result only controls stopping.
    """
    limits = plan["conditional_fit"]
    budget, interval = limits["train_fit_budget"], limits["validation_interval"]
    namespace = limits["scenario_seed_namespace"]
    ids = [event["id"] for event in train]
    labels = [event["label"] for event in train]
    if (initial["passed"] is not False or initial["optimizer_updates"] != 0
            or len(train) != plan["counts"]["train"] or len(set(ids)) != len(train)
            or {event["role"] for event in train} != {"train"} or set(labels) != set(HORIZONS)
            or type(budget) is not int or not 0 < budget <= 15 or interval != 3 or budget % interval
            or tuple(limits["allowed_parameters"]) != PARAMETERS
            or namespace != SCENARIO_SEED_NAMESPACE or limits["learning_rate"] != LEARNING_RATE):
        raise PermissionError("fit requires full train, zero-retrain rejection, and the registered V2 schedule")
    start = initial["candidate"]
    initial_params = np.asarray([start[key] for key in PARAMETERS])
    low, high = np.log(np.array(PARAMETER_LIMITS)).T
    if (not np.isfinite(initial_params).all() or np.any(initial_params <= 0)
            or not np.all((np.log(initial_params) >= low + DIFFERENCE_STEP)
                          & (np.log(initial_params) <= high - DIFFERENCE_STEP))):
        raise ValueError("initial parameters do not support the registered symmetric probes")
    log_params = np.log(initial_params)
    if any(parameter.requires_grad for parameter in models["m3"].parameters()):
        raise PermissionError("the ES-ROM equation parameters must be frozen")
    score_scales = np.asarray(plan["score_scales"], dtype=np.float64)
    if score_scales.shape != (2,) or not np.isfinite(score_scales).all() or np.any(score_scales <= 0):
        raise ValueError("expected positive frozen angle and speed score scales")
    score_scale = np.repeat(score_scales, 10)
    batch = plan["trajectory_batch"]
    if type(batch) is not int or not 0 < batch <= 256:
        raise ValueError("the full-train rollout batch must be between 1 and 256")
    indices = {label: np.flatnonzero(np.asarray(labels) == label) for label in HORIZONS}
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
            return _objective(models["m3"], train, indices, current_inputs, score_scale,
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
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
            adam_direction = (first_moment / (1.0 - beta1**update)) / (
                np.sqrt(second_moment / (1.0 - beta2**update)) + ADAM_EPSILON)
            trial_params = np.clip(log_params - LEARNING_RATE * adam_direction,
                                   low + DIFFERENCE_STEP, high - DIFFERENCE_STEP)
            candidate = _candidate(trial_params, start, f"train_update_{update}")
            row.update(candidate=candidate, log_parameter_gradient=gradient.tolist(), **objective(candidate))
            if row["loss"] is not None:
                row["applied"], log_params, applied = True, trial_params, update
        training.append(row)
        write_json(directory / f"train-update-{update:02d}.json", row)
        print(f"v2 full train update {update}/{budget}: loss={row['loss']}", flush=True)
        if not row["applied"]:
            status = "NUMERICAL_FAILURE"
            break
        if update % interval == 0:
            checkpoint = directory / f"val-update-{update:02d}"
            checkpoint.mkdir()
            result = evaluate(checkpoint, candidate, update)
            selection_file = checkpoint / "selection.json"
            if (type(result["passed"]) is not bool or result["candidate"] != candidate
                    or result["optimizer_updates"] != update or json.loads(selection_file.read_bytes()) != result):
                raise ValueError("the checkpoint selection does not match the fitted V2 candidate")
            checkpoints.append({"update": update, "passed": result["passed"], "candidate": candidate,
                                "metrics": result["evaluated"], "selection_sha256": sha256_file(selection_file)})
            if result["passed"]:
                status = "ACCEPT"
                break
    selection = result | {"optimizer_updates": applied, "optimizer_attempts": len(training),
                          "selected_checkpoint_update": result["optimizer_updates"], "train_fit_status": status}
    write_json(directory / "training_curves.json", {"train": training, "val": checkpoints,
                                                   "operations": dict(operations), "status": status})
    write_json(directory / "selection.json", selection)
    return selection
