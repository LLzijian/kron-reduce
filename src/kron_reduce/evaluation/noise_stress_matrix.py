"""Paired noise matrices and the preregistered validation decision."""

from __future__ import annotations

import gzip
import json
from collections import Counter

import numpy as np
import torch

from kron_reduce.data.task3_measurement_noise import generate_noise_bank
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.noise_stress_predict import (
    MODELS,
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
from project_harness.evidence import sha256_file

ROBUST = MODELS[-1]
ORIG = MODELS[2]
SG = MODELS[3]


def _write_case_scores(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output, gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as stream:
        stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _summary(score):
    return {k: v for k, v in score.items() if k not in ("steps", "pairwise")} | {
        "steps": {step: {k: v for k, v in details.items() if k != "rows"}
                  for step, details in score["steps"].items()},
        "pairwise": score["pairwise"]["summary"],
    }


def _forecasts(name, model, inputs, plan, scales, candidate, batch, operations):
    size = len(inputs[0])
    frames = plan["forecast_frames"] + 1
    paths = {step: np.full((size, frames, 20), np.nan) for step in STEPS}
    energy = {step: np.full((size, 3), np.nan) for step in STEPS}
    diagnostics, timings, failures = {step: {} for step in STEPS}, [], []

    def spend(key, amount=1):
        operations[key] += amount

    for step in STEPS:
        for start in range(0, size, batch):
            stop = min(start + batch, size)
            operations["allocated_event_predictions"] += stop - start
            try:
                pred, storage, evidence, timing = infer(
                    model, name, slice_inputs(inputs, start, stop), step, plan, scales, candidate, spend)
            except NUMERICAL_ERRORS as error:
                failures.append({"step_s": step, "start": start, "stop": stop, "error": str(error)})
                continue
            paths[step][start:stop] = pred
            energy[step][start:stop] = np.where(np.isfinite(storage), storage, np.nan)
            for key, values in evidence.items():
                if key not in diagnostics[step]:
                    diagnostics[step][key] = [None] * size
                diagnostics[step][key][start:stop] = [
                    value.item() if np.isfinite(value) else None for value in values]
            timings.append({"step_s": step, "batch": stop - start, **timing})
    return paths, energy, diagnostics, timings, failures


def evaluate_matrix(directory, cohorts, models, scales, plan, *, candidates, ablation=False):
    calculations = Counter()
    rosters, manifest, bootstrap = {}, {}, {}
    names = [(name, candidates[0]) for name in (MODELS if not ablation else (ROBUST,))]
    names += [(f"{ROBUST}__{x['name']}", x) for x in candidates[1:]]
    for roster, events in cohorts.items():
        if roster not in ("val", "test", "blind_v2") or len(events) != plan["counts"][roster]:
            raise PermissionError("noise stress evaluation requires a complete authorized roster")
        count = len(events)
        ids = [event["id"] for event in events]
        rosters[roster], manifest[roster] = {}, {}
        inertia = models["m3"].mass.cpu().numpy()
        if not np.allclose(np.stack([event["mass"] for event in events]), inertia, rtol=0, atol=1e-12):
            raise ValueError("event inertia differs from M5")
        for scenario in plan["scenarios"]:
            repetition_count = 1 if scenario == "clean" else 5
            banks = [generate_noise_bank(roster, scenario, rep, ids) for rep in range(repetition_count)]
            manifest[roster][scenario] = {str(rep): bank.sha256 for rep, bank in enumerate(banks)}
            chunks = [observation_inputs(events, bank, scales) for bank in banks]
            inputs = tuple(np.concatenate([chunk[k] for chunk in chunks]) for k in range(4)) + (chunks[0][4],)
            endpoint = [[bank.draws[canonical_id].endpoint_index for canonical_id in ids] for bank in banks]
            rosters[roster][scenario] = {}
            paired = {}
            for label, candidate in names:
                name = label.split("__", maxsplit=1)[0]
                if name == MODELS[4]:
                    candidate = plan["candidates"][0]
                model = model_for(models, name)
                batch = plan["v4_batch"] if name in MODELS[:2] else plan["m3b_batch"]
                if scenario != "clean":
                    batch = 4 * plan["m3b_batch"]
                paths, energy, details, timing, failures = _forecasts(
                    name, model, inputs, plan, scales, candidate, batch, calculations)
                scores = []
                for rep in range(repetition_count):
                    sl = slice(rep * count, (rep + 1) * count)
                    score = score_pair(events, {s: values[sl] for s, values in paths.items()},
                                       {s: values[sl] for s, values in energy.items()},
                                       endpoint=endpoint[rep])
                    for step in STEPS:
                        for key, values in details[step].items():
                            for row, value in zip(score["steps"][str(step)]["rows"], values[sl], strict=True):
                                row[key] = value
                    scores.append(score)
                    _write_case_scores(directory / "cases" / roster / scenario / label / f"rep{rep}.json.gz", score)
                if roster == "blind_v2" and label in (ORIG, ROBUST):
                    paired[label] = scores
                rosters[roster][scenario][label] = {
                    "repeats": [_summary(score) for score in scores], "summary": summarize_reps(scores),
                    "bulk_timing": timing, "inference_failures": failures,
                    "candidate": candidate if name in MODELS[4:] else None,
                }
                qualified = sum(score["pairwise"]["summary"]["qualified"] for score in scores)
                print(f"{roster} {scenario} {label}: {qualified}/{count*repetition_count} paired", flush=True)
                del scores, paths, energy, details
            if paired:
                bootstrap[scenario] = paired_group_bootstrap(paired[ROBUST], paired[ORIG], seed=plan["bootstrap_seed"])
    if not ablation:
        expected = plan["limits"]["validation_predictions" if set(cohorts) == {"val"} else "holdout_predictions"]
        if calculations["allocated_event_predictions"] != expected:
            raise RuntimeError("evaluation matrix does not fit the preregistered forecast allocation")
    write_json(directory / "metrics.json", {"rosters": rosters, "operations": dict(calculations)})
    write_json(directory / "noise_manifest.json", manifest)
    if bootstrap:
        write_json(directory / "blind_group_bootstrap.json", bootstrap)
    return {"noise_manifest_sha256": sha256_file(directory / "noise_manifest.json"),
            "metrics_sha256": sha256_file(directory / "metrics.json"),
            "event_predictions": calculations["allocated_event_predictions"]}


def _loss(rep, step, speed_scale):
    values = rep["steps"][step]["metrics"]
    terms = [values["stable_8s_nrmse"]["mean"], values["unstable_available_nrmse"]["mean"],
             values["positive_peak_deficit_rad_s"]["mean"], values["confirmed_recall"], values["false_alarm_rate"]]
    if any(term is None for term in terms):
        return None
    stable, unstable, peak, recall, fp_rate = terms
    return stable + unstable + peak / speed_scale + (1 - recall + fp_rate) / 2


def _compare(roster, name, plan):
    gate = plan["gate"]
    all_steps = {}
    improvements = []
    for step in map(str, STEPS):
        clean = roster["clean"][name]["repeats"][0]["steps"][step]["metrics"]
        reference = roster["clean"][ORIG]["repeats"][0]["steps"][step]["metrics"]
        stable, original = clean["stable_8s_nrmse"]["mean"], reference["stable_8s_nrmse"]["mean"]
        ratio = stable / original if stable is not None and original else None
        clean_pass = (clean["tp"] == gate["clean_confirmed_unstable"] and clean["fp"] <= reference["fp"]
                      and ratio is not None and ratio <= 1 + gate["maximum_clean_stable_relative_increase"])
        by_scenario = {}
        for scenario in gate["required_scenarios"]:
            values = {}
            for model in (name, ORIG, SG):
                measured = [_loss(row, step, plan["score_scales"][1])
                            for row in roster[scenario][model]["repeats"]]
                values[model] = float(np.mean(measured)) if all(value is not None for value in measured) else None
            loss = values[name]
            comparison = {m: 1 - loss / values[m] if loss is not None and values[m] else None for m in (ORIG, SG)}
            passed = all(value is not None and value >= gate["minimum_noisy_relative_improvement"]
                         for value in comparison.values())
            improvements.extend(value for value in comparison.values() if value is not None)
            by_scenario[scenario] = {"loss": values, "relative_improvement": comparison, "passed": passed}
        all_steps[step] = {"clean": {"tp": clean["tp"], "fp": clean["fp"], "reference_fp": reference["fp"],
                                    "stable_nrmse_ratio": ratio, "passed": clean_pass},
                           "noisy": by_scenario}
    numerical = all(rep["pairwise"]["unqualified"] == 0 for condition in roster.values()
                    for rep in condition[name]["repeats"])
    passed = numerical and all(row["clean"]["passed"] and all(x["passed"] for x in row["noisy"].values())
                               for row in all_steps.values())
    return {"passed": passed, "paired_numerical_pass": numerical, "details": all_steps,
            "mean_relative_improvement": float(np.mean(improvements)) if improvements else None}


def select_candidate(directory, plan, *, candidates=None, reference_dir=None, optimizer_updates=0):
    val = json.loads((directory / "metrics.json").read_bytes())["rosters"]["val"]
    if reference_dir is not None:
        base = json.loads((reference_dir / "metrics.json").read_bytes())["rosters"]["val"]
        val = {key: base[key] | val[key] for key in val}
    candidates = plan["candidates"] if candidates is None else candidates
    names = [ROBUST] + [f"{ROBUST}__{candidate['name']}" for candidate in candidates[1:]]
    reviews = [_compare(val, name, plan) for name in names]

    def rank(i):
        value = reviews[i]["mean_relative_improvement"]
        return reviews[i]["passed"], value if value is not None else -np.inf

    winner = max(range(len(names)), key=rank)
    result = {"passed": reviews[winner]["passed"], "candidate": candidates[winner], "model_key": names[winner],
              "optimizer_updates": optimizer_updates, "evaluated": dict(zip(names, reviews, strict=True))}
    write_json(directory / "selection.json", result)
    return result


def benchmark(directory, events, models, scales, plan, candidate):
    scenario = "gauss_dse_dynamic"
    inputs = observation_inputs(events, generate_noise_bank("val", scenario, 0, [e["id"] for e in events]), scales)
    timings = {}
    for name in MODELS:
        model = model_for(models, name)
        timings[name] = {}
        selected = plan["candidates"][0] if name == MODELS[4] else candidate
        for step in STEPS:
            timings[name][str(step)] = {}
            for batch in (1, plan["m3b_batch"]):
                curr = slice_inputs(inputs, 0, batch)
                values = {"init_s": [], "rollout_s": [], "total_s": []}
                for trial in range(plan["timing_repeats"] + 1):
                    torch.cuda.synchronize()
                    measured = infer(model, name, curr, step, plan, scales, selected, lambda *_: None)[3]
                    if trial:
                        for key, value in measured.items():
                            values[key].append(value)
                timings[name][str(step)][str(batch)] = values
    write_json(directory / "latency.json", {"source_roster": "val", "scenario": scenario, "by_model": timings})
