"""Paired pressure matrices for the frozen ES-ROM V2 initializer."""

from __future__ import annotations

import time
from collections import Counter
from itertools import groupby

import numpy as np
import torch

from kron_reduce.data.task3_measurement_noise import generate_noise_bank
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.noise_stress_matrix import ORIG, _compare, _summary, _write_case_scores
from kron_reduce.evaluation.noise_stress_predict import (
    STEPS,
    infer,
    model_for,
    observation_inputs,
    slice_inputs,
)
from kron_reduce.evaluation.task3_noise_scores import (
    paired_group_bootstrap,
    score_pair,
    summarize_reps,
)
from kron_reduce.models.noise_robustness_mhe_v2 import predict_mhe_v2, reblend_origin
from project_harness.evidence import sha256_file

V2 = "m3b_g2_robust_mhe_v2"
OPTIONS = ("gate_rad", "alpha", "tau_mis", "huber_delta", "inverse_snr")


def _initialize_v2(model, inputs, step, scales, candidate, spend):
    return predict_mhe_v2(model, *inputs, 0, max_step=step, spend=spend,
                          memory_radius=scales["memory_radius"],
                          **{key: candidate[key] for key in OPTIONS})


def _energy(model, origin, equilibrium):
    memory = .5 * origin[:, 20:].square().sum(dim=1)
    total = model.storage(origin, model._tensor(equilibrium))
    return torch.stack((total - memory, memory, total), dim=1).cpu().numpy()


def _pairs(candidates, full_models, plan):
    if full_models:
        for name in plan["models"][:-1]:
            yield [(name, plan["legacy_l2"])]
    if len(candidates) == 1:
        yield [(V2, candidates[0])]
        return
    solve_parameters = tuple(key for key in OPTIONS if key != "tau_mis")
    for _, group in groupby(candidates, lambda row: tuple(row[k] for k in solve_parameters)):
        yield [(f"{V2}__{x['name']}", x) for x in group]


def _forecasts(paired, models, inputs, plan, scales, scenario, operations):
    count = len(inputs[0])
    frames = plan["forecast_frames"] + 1
    results = {
        name: {
            "paths": {step: np.full((count, frames, 20), np.nan) for step in STEPS},
            "energy": {step: np.full((count, 3), np.nan) for step in STEPS},
            "diagnostics": {step: {} for step in STEPS}, "failures": [],
        }
        for name, _ in paired
    }
    first, parameters = paired[0]
    is_v2 = first.startswith(V2)
    model = model_for(models, first)
    batch = plan["trajectory_batch"]
    if scenario == "clean":
        batch = plan["v4_batch"] if first.startswith("v4_") else plan["m3b_batch"]

    def spend(key, amount=1):
        operations[key] += amount

    for step in STEPS:
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            local = slice_inputs(inputs, start, stop)
            initial = None
            if is_v2:
                try:
                    initial = _initialize_v2(model, local, step, scales, parameters, spend)
                except NUMERICAL_ERRORS as error:
                    for name, _ in paired:
                        results[name]["failures"].append({"step_s": step, "start": start, "stop": stop,
                                                          "error": str(error)})
            for name, candidate in paired:
                spend("allocated_event_predictions", stop - start)
                if is_v2 and initial is None:
                    continue
                item = results[name]
                try:
                    if is_v2:
                        origin = (initial["origin"] if candidate is parameters
                                  else reblend_origin(initial, model, candidate["tau_mis"]))
                        forecast = model.rollout(origin, local[2], local[3], frames - 1,
                                                 max_step=step, spend=spend)
                        path = forecast[:, :, :20].cpu().numpy()
                        energy = _energy(model, origin, local[3])
                        info = {k: initial[k].cpu().numpy() for k in (
                            "joint_active", "kinematic_defect_rad", "sigma_hat", "map_lambda",
                        )}
                    else:
                        path, energy, info, _ = infer(model, name, local, step, plan, scales, candidate, spend)
                except NUMERICAL_ERRORS as error:
                    item["failures"].append({"step_s": step, "start": start, "stop": stop, "error": str(error)})
                    continue
                item["paths"][step][start:stop] = path
                item["energy"][step][start:stop] = np.where(np.isfinite(energy), energy, np.nan)
                for key, values in info.items():
                    column = item["diagnostics"][step].setdefault(key, [None] * count)
                    column[start:stop] = [value.item() if np.isfinite(value) else None for value in values]
    return results


def _score_predictions(directory, roster, scenario, name, events, endpoints, prediction):
    count = len(events)
    scores = []
    for rep, endpoint in enumerate(endpoints):
        span = slice(rep * count, (rep + 1) * count)
        score = score_pair(
            events, {k: v[span] for k, v in prediction["paths"].items()},
            {k: v[span] for k, v in prediction["energy"].items()}, endpoint=endpoint,
        )
        for step, fields in prediction["diagnostics"].items():
            for key, values in fields.items():
                for row, value in zip(score["steps"][str(step)]["rows"], values[span], strict=True):
                    row[key] = value
        _write_case_scores(directory / "cases" / roster / scenario / name / f"rep{rep}.json.gz", score)
        scores.append(score)
    return scores


def evaluate_matrix(directory, cohorts, models, scales, plan, *, candidates, reference_noise=None):
    operations = Counter()
    rosters, manifest, bootstrap = {}, {}, {}
    validation = set(cohorts) == {"val"}
    names = list(_pairs(candidates, not validation, plan))
    for roster, events in cohorts.items():
        if roster not in ("val", "test", "blind_v2") or len(events) != plan["counts"][roster]:
            raise PermissionError("noise stress requires a complete authorized roster")
        ids = [event["id"] for event in events]
        inertia = models["m3"].mass.cpu().numpy()
        if not np.allclose(np.stack([e["mass"] for e in events]), inertia, rtol=0, atol=1e-12):
            raise ValueError("event inertia differs from the frozen system")
        rosters[roster], manifest[roster] = {}, {}
        for scenario in plan["scenarios"]:
            banks = [
                generate_noise_bank(roster, scenario, rep, ids)
                for rep in range(1 if scenario == "clean" else 5)
            ]
            digests = {str(i): bank.sha256 for i, bank in enumerate(banks)}
            if validation and (reference_noise is None or digests != reference_noise["val"][scenario]):
                raise ValueError(f"the raw val noise differs from V1 for {scenario}")
            manifest[roster][scenario] = digests
            chunks = [observation_inputs(events, bank, scales) for bank in banks]
            inputs = tuple(np.concatenate([c[k] for c in chunks]) for k in range(4)) + (chunks[0][4],)
            endpoints = [[bank.draws[event_id].endpoint_index for event_id in ids] for bank in banks]
            rosters[roster][scenario] = {}
            boot = {}
            for variant_pair in names:
                predictions = _forecasts(variant_pair, models, inputs, plan, scales, scenario, operations)
                for name, candidate in variant_pair:
                    prediction = predictions[name]
                    scores = _score_predictions(directory, roster, scenario, name, events, endpoints, prediction)
                    rosters[roster][scenario][name] = {
                        "repeats": [_summary(score) for score in scores],
                        "summary": summarize_reps(scores),
                        "candidate": candidate if name.startswith(V2) else None,
                        "inference_failures": prediction["failures"],
                    }
                    if roster == "blind_v2" and name in (V2, ORIG):
                        boot[name] = scores
                    qualified = sum(score["pairwise"]["summary"]["qualified"] for score in scores)
                    print(f"{roster} {scenario} {name}: {qualified}/{len(events)*len(banks)} paired", flush=True)
            if boot:
                bootstrap[scenario] = paired_group_bootstrap(boot[V2], boot[ORIG], seed=plan["bootstrap_seed"])
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
    labels = ([V2] if len(candidates) == 1 else [f"{V2}__{candidate['name']}" for candidate in candidates])
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


def benchmark(directory, val, model, scales, plan, candidate, legacy):
    scenario = "gauss_dse_dynamic"
    bank = generate_noise_bank("val", scenario, 0, [e["id"] for e in val])
    inputs = observation_inputs(val, bank, scales)
    timings = {}
    for step in STEPS:
        timings[str(step)] = {}
        for batch in (1, plan["m3b_batch"]):
            local = slice_inputs(inputs, 0, batch)
            rows = {"init_s": [], "rollout_s": [], "total_s": []}
            for trial in range(plan["timing_repeats"] + 1):
                torch.cuda.synchronize()
                started = time.perf_counter()
                initial = _initialize_v2(model, local, step, scales, candidate, lambda *_: None)
                torch.cuda.synchronize()
                initialized = time.perf_counter()
                model.rollout(initial["origin"], local[2], local[3], plan["forecast_frames"], max_step=step)
                torch.cuda.synchronize()
                stopped = time.perf_counter()
                if trial:
                    rows["init_s"].append(initialized - started)
                    rows["rollout_s"].append(stopped - initialized)
                    rows["total_s"].append(stopped - started)
            timings[str(step)][str(batch)] = rows
    write_json(directory / "latency.json", {
        "source_roster": "val", "scenario": scenario, "warmup": 1,
        "reused_v1_models": [name for name in plan["models"][:-1] if name in legacy["by_model"]],
        "by_model": {name: timings if name == V2 else legacy["by_model"][name] for name in plan["models"]},
    })
