"""Conditional whole-train adaptation of the initializer, with frozen equations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

from kron_reduce.data.task3_measurement_noise import NoiseBank, NoiseDraw
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.noise_stress_matrix import (
    ROBUST,
    benchmark,
    evaluate_matrix,
    select_candidate,
)
from kron_reduce.evaluation.noise_stress_predict import (
    STEPS,
    initialize,
    observation_inputs,
    slice_inputs,
)
from project_harness.evidence import sha256_file

LOG_LIMITS = np.log([[1e-5, 10.], [.5, 4.]])
DIFFERENCE_STEP = .05


def _train_bank(events, seed_namespace, update):
    draws = {}
    for i, event in enumerate(events):
        key = f"{seed_namespace}|train|update{update}|{event['id']}"
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:16], "little")
        rng = np.random.default_rng(np.random.PCG64(seed))
        stress = (i + update) % 4
        scale = (0.010, 0.050) if stress < 2 else (0.0033, 0.010)
        if stress == 1:
            delta, omega = (rng.laplace(0., s / np.sqrt(2.), (11, 10)) for s in scale)
        else:
            delta, omega = (rng.normal(0., s, (11, 10)) for s in scale)
        if stress == 2:
            for noise, magnitude in ((delta, .1), (omega, .5)):
                noise += (rng.random((11, 10)) < .05) * rng.choice([-1., 1.], size=(11, 10)) * magnitude
        elif stress == 3:
            row, generator = rng.choice([0, 10]), rng.integers(0, 10)
            delta[row, generator] += rng.choice([-.1, .1])
            omega[row, generator] += rng.choice([-.5, .5])
        draws[event["id"]] = NoiseDraw(delta, omega, np.zeros(10))
    return NoiseBank(draws, "")


def _candidate(log_params, initial, name):
    alpha, huber = np.exp(log_params)
    return {"name": name, "alpha": float(alpha), "huber_delta": float(huber), "gate_rad": initial["gate_rad"]}


def _objective(model, inputs, truth, plan, scales, candidate, operations):
    """Full-list mean normalized squared error at the clean issued state."""
    scale = np.repeat(plan["score_scales"], 10)
    error = 0.

    def spend(key, amount=1):
        operations[key] += amount

    batch = 4 * plan["m3b_batch"]
    for step in STEPS:
        for start in range(0, len(truth), batch):
            stop = min(start + batch, len(truth))
            result = initialize(model, ROBUST, slice_inputs(inputs, start, stop), step, plan, scales, candidate, spend)
            origin = result["forecast"][:, 0, :20].cpu().numpy()
            error += np.square((origin - truth[start:stop]) / scale).sum()
    return float(error / (len(STEPS) * truth.size))


def adapt(directory, prior_dir, train, val, models, scales, plan):
    limits = plan["conditional_fit"]
    if (len(train) != plan["counts"]["train"] or len(val) != plan["counts"]["val"]
            or {event["role"] for event in train} != {"train"} or {event["role"] for event in val} != {"val"}
            or not 0 < limits["train_fit_budget"] <= 15):
        raise PermissionError("the complete train and val lists and a bounded schedule are required")
    initial = json.loads((prior_dir / "selection.json").read_bytes())["candidate"]
    log_params = np.log([initial["alpha"], initial["huber_delta"]])
    first_moment = np.zeros_like(log_params)
    second_moment = np.zeros_like(log_params)
    truth = np.stack([event["x"][10] for event in train])
    operations = Counter()
    checkpoints, training = [], []
    result = {"passed": False}
    for update in range(1, limits["train_fit_budget"] + 1):
        inputs = observation_inputs(train, _train_bank(train, limits["scenario_seed_namespace"], update), scales)
        gradient = np.empty(2)
        for index, delta in enumerate(np.eye(2) * DIFFERENCE_STEP):
            low, high = (_objective(models["m3"], inputs, truth, plan, scales,
                                    _candidate(log_params + sign * delta, initial, "trial"), operations)
                         for sign in (-1, 1))
            gradient[index] = (high - low) / (2 * DIFFERENCE_STEP)
        if not np.isfinite(gradient).all():
            raise FloatingPointError("invalid full-train initializer gradient")
        first_moment = .9 * first_moment + .1 * gradient
        second_moment = .999 * second_moment + .001 * gradient**2
        update_direction = (first_moment / (1 - .9**update)) / (
            np.sqrt(second_moment / (1 - .999**update)) + 1e-8)
        log_params = np.clip(log_params - limits["learning_rate"] * update_direction,
                             LOG_LIMITS[:, 0], LOG_LIMITS[:, 1])
        candidate = _candidate(log_params, initial, f"train_update_{update}")
        measured = _objective(models["m3"], inputs, truth, plan, scales, candidate, operations)
        if not np.isfinite(measured):
            raise FloatingPointError("invalid full-train initializer loss")
        training.append({"update": update, "sample_count": len(train), "loss": measured,
                         "log_parameter_gradient": gradient.tolist(), "candidate": candidate})
        write_json(directory / f"train-update-{update:02d}.json", training[-1])
        if update % limits["validation_interval"] == 0:
            checkpoint = directory / f"val-update-{update:02d}"
            checkpoint.mkdir()
            evaluate_matrix(checkpoint, {"val": val}, models, scales, plan, candidates=[candidate], ablation=True)
            result = select_candidate(checkpoint, plan, candidates=[candidate], reference_dir=prior_dir,
                                      optimizer_updates=update)
            checkpoints.append({"update": update, "passed": result["passed"], "candidate": candidate,
                                "selection_sha256": sha256_file(checkpoint / "selection.json")})
            if result["passed"]:
                break
    if not checkpoints or "candidate" not in result:
        raise RuntimeError("the registered full-val checkpoints did not execute")
    write_json(directory / "selection.json", result)
    write_json(directory / "training_curves.json", {"train": training, "val": checkpoints, "operations": dict(operations)})
    if result["passed"]:
        benchmark(directory, val, models, scales, plan, result["candidate"])
    return {"train_cases": len(train), "val_cases": len(val), "optimizer_updates": update,
            "scientific_verdict": "ACCEPT" if result["passed"] else "REJECT",
            "training_sha256": sha256_file(directory / "training_curves.json")}
