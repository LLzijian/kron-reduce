"""Paired pressure matrices using the frozen TASK3 scoring and V3 initialization."""

from __future__ import annotations

import time
from collections import Counter
from itertools import groupby

import numpy as np
import torch

from kron_reduce.data.task3_measurement_noise import generate_noise_bank
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.noise_stress_matrix import ORIG, _compare, _summary
from kron_reduce.evaluation.noise_stress_matrix_v2 import _energy, _forecasts, _score_predictions
from kron_reduce.evaluation.noise_stress_predict import (
    STEPS,
    infer,
    model_for,
    observation_inputs,
    slice_inputs,
)
from kron_reduce.evaluation.task3_noise_scores import paired_group_bootstrap, summarize_reps
from kron_reduce.models.noise_robustness_mhe_v3 import predict_mhe_v3, reblend_origin
from project_harness.evidence import sha256_file

V3 = "m3b_g2_robust_mhe_v3"
OPTIONS = ("gate_rad", "alpha", "c_var", "kappa_term", "huber_delta", "inverse_snr")


def _initialize(model, inputs, step, scales, candidate, spend):
    return predict_mhe_v3(model, *inputs, 0, max_step=step, spend=spend,
                          memory_radius=scales["memory_radius"],
                          **{key: candidate[key] for key in OPTIONS})


def _variant_groups(candidates, full_models, plan):
    if full_models:
        for name in plan["models"][:-1]:
            yield [(name, plan["legacy_l2"])]
    if len(candidates) == 1:
        yield [(V3, candidates[0])]
        return
    fit_keys = tuple(key for key in OPTIONS if key != "kappa_term")
    for _, rows in groupby(candidates, lambda candidate: tuple(candidate[key] for key in fit_keys)):
        yield [(f"{V3}__{row['name']}", row) for row in rows]


def _v3_forecasts(variants, model, inputs, plan, scales, scenario, operations):
    count = len(inputs[0])
    frames = plan["forecast_frames"] + 1
    batch = plan["m3b_batch"] if scenario == "clean" else plan["trajectory_batch"]
    results = {
        name: {
            "paths": {step: np.full((count, frames, 20), np.nan) for step in STEPS},
            "energy": {step: np.full((count, 3), np.nan) for step in STEPS},
            "diagnostics": {step: {} for step in STEPS}, "failures": [],
        }
        for name, _ in variants
    }
    _, first = variants[0]

    def spend(key, amount=1):
        operations[key] += amount

    for step in STEPS:
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            local = slice_inputs(inputs, start, stop)
            spend("allocated_event_predictions", len(variants) * (stop - start))
            try:
                initial = _initialize(model, local, step, scales, first, spend)
            except NUMERICAL_ERRORS as error:
                for result in results.values():
                    result["failures"].append({"step_s": step, "start": start, "stop": stop, "error": str(error)})
                continue
            variance = initial["mismatch_variance"]
            info = {key: initial[key].cpu().numpy() for key in (
                "joint_active", "kinematic_defect_rad", "sigma_hat", "map_lambda",
            )}
            info["max_mismatch_variance"] = variance.amax(dim=1).cpu().numpy()
            for name, candidate in variants:
                item = results[name]
                try:
                    origin = (initial["origin"] if candidate is first
                              else reblend_origin(initial, model, candidate["kappa_term"]))
                    path = model.rollout(origin, local[2], local[3], frames - 1, max_step=step, spend=spend)
                    energy = _energy(model, origin, local[3])
                except NUMERICAL_ERRORS as error:
                    item["failures"].append({"step_s": step, "start": start, "stop": stop, "error": str(error)})
                    continue
                item["paths"][step][start:stop] = path[:, :, :20].cpu().numpy()
                item["energy"][step][start:stop] = np.where(np.isfinite(energy), energy, np.nan)
                gain = variance / (variance + candidate["kappa_term"] * initial["channel_noise"].square() + 1e-12)
                for key, values in (info | {"mean_endpoint_gain": gain.mean(dim=1).cpu().numpy()}).items():
                    column = item["diagnostics"][step].setdefault(key, [None] * count)
                    column[start:stop] = [value.item() if np.isfinite(value) else None for value in values]
    return results


def evaluate_matrix(directory, cohorts, models, scales, plan, *, candidates, reference_noise=None):
    validation = set(cohorts) == {"val"}
    if not validation and (set(cohorts) != {"test", "blind_v2"} or len(candidates) != 1):
        raise PermissionError("joint holdout requires the two complete rosters and one frozen candidate")
    operations = Counter()
    rosters, manifest, bootstrap = {}, {}, {}
    groups = list(_variant_groups(candidates, not validation, plan))
    for roster, events in cohorts.items():
        if len(events) != plan["counts"][roster] or {event["role"] for event in events} != {roster}:
            raise PermissionError("noise stress requires a complete authorized roster")
        ids = [event["id"] for event in events]
        if not np.allclose(np.stack([e["mass"] for e in events]), models["m3"].mass.cpu().numpy(),
                           rtol=0, atol=1e-12):
            raise ValueError("event inertia differs from the frozen system")
        rosters[roster], manifest[roster] = {}, {}
        for scenario in plan["scenarios"]:
            banks = [generate_noise_bank(roster, scenario, rep, ids)
                     for rep in range(1 if scenario == "clean" else 5)]
            digests = {str(i): bank.sha256 for i, bank in enumerate(banks)}
            if validation and (reference_noise is None or digests != reference_noise["val"][scenario]):
                raise ValueError(f"the raw val noise differs from V1 for {scenario}")
            manifest[roster][scenario] = digests
            chunks = [observation_inputs(events, bank, scales) for bank in banks]
            inputs = tuple(np.concatenate([chunk[k] for chunk in chunks]) for k in range(4)) + (chunks[0][4],)
            endpoints = [[bank.draws[event_id].endpoint_index for event_id in ids] for bank in banks]
            condition, paired_bootstrap = {}, {}
            for variants in groups:
                is_v3 = variants[0][0].startswith(V3)
                predictions = (_v3_forecasts(variants, models["m3"], inputs, plan, scales, scenario, operations)
                               if is_v3 else _forecasts(variants, models, inputs, plan, scales, scenario, operations))
                for name, candidate in variants:
                    prediction = predictions[name]
                    scores = _score_predictions(directory, roster, scenario, name, events, endpoints, prediction)
                    condition[name] = {
                        "repeats": [_summary(score) for score in scores],
                        "summary": summarize_reps(scores),
                        "candidate": candidate if is_v3 else None,
                        "inference_failures": prediction["failures"],
                    }
                    if roster == "blind_v2" and name in (V3, ORIG):
                        paired_bootstrap[name] = scores
                    qualified = sum(score["pairwise"]["summary"]["qualified"] for score in scores)
                    print(f"{roster} {scenario} {name}: {qualified}/{len(events)*len(banks)} paired", flush=True)
            rosters[roster][scenario] = condition
            if paired_bootstrap:
                bootstrap[scenario] = paired_group_bootstrap(
                    paired_bootstrap[V3], paired_bootstrap[ORIG], seed=plan["bootstrap_seed"],
                )
    expected = (plan["counts"]["val"] * 46 * 2 * len(candidates) if validation
                else plan["limits"]["holdout_predictions"])
    if (operations["allocated_event_predictions"] != expected
            or validation and len(candidates) > 1 and expected != plan["limits"]["validation_predictions"]):
        raise RuntimeError("prediction matrix is incomplete or outside its budget")
    write_json(directory / "metrics.json", {"rosters": rosters, "operations": dict(operations)})
    write_json(directory / "noise_manifest.json", manifest)
    if bootstrap:
        write_json(directory / "blind_group_bootstrap.json", bootstrap)
    evidence = {"noise_manifest_sha256": sha256_file(directory / "noise_manifest.json"),
                "metrics_sha256": sha256_file(directory / "metrics.json"),
                "event_predictions": operations["allocated_event_predictions"]}
    return rosters, evidence


def select_candidate(directory, plan, scores, v1_scores, candidates, update=0):
    val = {scenario: v1_scores[scenario] | records for scenario, records in scores["val"].items()}
    labels = ([V3] if len(candidates) == 1 else [f"{V3}__{candidate['name']}" for candidate in candidates])
    reviews = [_compare(val, label, plan) for label in labels]

    def rank(index):
        mean = reviews[index]["mean_relative_improvement"]
        return reviews[index]["passed"], mean if mean is not None else -np.inf

    winner = max(range(len(candidates)), key=rank)
    result = {"passed": reviews[winner]["passed"], "candidate": candidates[winner],
              "model_key": labels[winner], "optimizer_updates": update,
              "evaluated": dict(zip(labels, reviews, strict=True))}
    write_json(directory / "selection.json", result)
    return result


def benchmark(directory, events, models, scales, plan, candidate):
    scenario, role = "gauss_dse_dynamic", events[0]["role"]
    bank = generate_noise_bank(role, scenario, 0, [event["id"] for event in events])
    inputs = observation_inputs(events, bank, scales)
    timings = {}
    for name in plan["models"]:
        model = model_for(models, name)
        timings[name] = {}
        for step in STEPS:
            timings[name][str(step)] = {}
            for batch in (1, plan["m3b_batch"]):
                local = slice_inputs(inputs, 0, batch)
                rows = {"init_s": [], "rollout_s": [], "total_s": []}
                for trial in range(plan["timing_repeats"] + 1):
                    torch.cuda.synchronize()
                    if name == V3:
                        start = time.perf_counter()
                        origin = _initialize(model, local, step, scales, candidate, lambda *_: None)["origin"]
                        torch.cuda.synchronize()
                        middle = time.perf_counter()
                        model.rollout(origin, local[2], local[3], plan["forecast_frames"], max_step=step)
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        measured = {"init_s": middle - start, "rollout_s": end - middle, "total_s": end - start}
                    else:
                        measured = infer(model, name, local, step, plan, scales,
                                         plan["legacy_l2"], lambda *_: None)[3]
                    if trial:
                        for key, value in measured.items():
                            rows[key].append(value)
                timings[name][str(step)][str(batch)] = rows
    write_json(directory / "latency.json", {"source_roster": role, "scenario": scenario, "warmup": 1,
                                           "by_model": timings})
