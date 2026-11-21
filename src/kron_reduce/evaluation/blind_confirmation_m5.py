"""Evaluate the sealed M5 roster once and reuse those predictions for post-fault analyses."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.first_swing_probes import speed_peak_gap
from kron_reduce.evaluation.kiki_machine_local_full import evaluate, inputs
from kron_reduce.evaluation.postfault_m4 import STEPS, energies, separation_cuts
from kron_reduce.evaluation.postfault_m4_modes import _assign, _medians, _ringdown, _spectrum
from kron_reduce.evaluation.postfault_m4_v2 import _group_stats
from kron_reduce.evaluation.trajectory_stability import summarize
from kron_reduce.models.first_swing_fit import predict_first_swing_fit
from kron_reduce.models.unified_discovery_m3 import predict_m3

MODEL_NAMES = ("v4_hard_r", "v4_soft_2r", "m3b_g2")
BOOTSTRAP_REPLICATES = 10_000


def _predict_once(name, model, events, scales, settings, budget):
    paths = {step: np.full((len(events), 801, 20), np.nan) for step in STEPS}
    index = {event["id"]: i for i, event in enumerate(events)}
    attempted = set()
    blocked_retries = Counter()

    def predict(batch, step):
        step_key = str(step)
        ids = [event["id"] for event in batch]
        keys = {(event_id, step_key) for event_id in ids}
        if keys & attempted:
            blocked_retries[step_key] += len(keys)
            raise FloatingPointError("blind prediction attempt already consumed")
        attempted.update(keys)
        options = {"max_step": step, "spend": budget.spend, "memory_radius": scales["memory_radius"],
                   "prior_weight": settings["prior_weight"]}
        history = inputs(batch, scales["history"])
        try:
            if name == "m3b_g2":
                result = predict_m3(model, *history, 800, **options)
            else:
                result = predict_first_swing_fit(model, *history, 800, **options, soft_radius=name == "v4_soft_2r")
        except ValueError as error:
            if str(error) in ("finite float64 observation matrices on the same device required",
                              "model inputs must be finite"):
                raise FloatingPointError(str(error)) from error
            raise
        result["forecast"] = result["forecast"].cpu()
        visible = result["forecast"].numpy()[:, :, :20]
        if (visible.shape != (len(batch), 801, 20) or visible.dtype != np.float64
                or not np.array_equal(visible[:, 0], np.concatenate((history[0][:, -1], history[1][:, -1]), axis=1))):
            raise ValueError("blind predictor must retain the exact measured forecast origin")
        paths[step_key][[index[event_id] for event_id in ids]] = visible
        return result | {"gn_singular": result["singular"][None]}

    return predict, paths, attempted, blocked_retries


def _classification(rows, *, engineering=False):
    if engineering:
        rows = [row | {"predicted_label": "unstable", "unsettled": False}
                if row["qualified"] and row["unsettled"] else row for row in rows]
    report = summarize(rows)
    # This legacy field combines numerical failures with qualified unsettled cases.
    report.pop("numerical_unresolved")
    unstable = sum(row["truth_label"] == "unstable" for row in rows)
    stable = len(rows)-unstable
    report.update(
        true_unstable=unstable, true_stable=stable,
        confirmed_recall=report["tp"]/unstable if unstable else None,
        confirmed_miss_rate=report["fn"]/unstable if unstable else None,
        not_confirmed_instability_rate=1.-report["tp"]/unstable if unstable else None,
        confirmed_false_alarm_rate=report["fp"]/stable if stable else None,
    )
    return report


def _stable8(scored, events):
    records = {row["event_id"]: row["rmse"] for row in scored["records"]
               if row["label"] == "stable" and row["horizon"] == "8" and row["qualified"]
               and np.isfinite(row["rmse"])}
    return np.array([records.get(event["id"], np.nan) for event in events])


def _complete_stat(values, statistic):
    valid = values[np.isfinite(values)]
    return {"allocated": len(values), "qualified": len(valid),
            "value": float(statistic(valid)) if len(valid) and len(valid) == len(values) else None}


def _score_dual(events, paths, main):
    from task3.c2.evaluation import evaluate as task3_evaluate
    from task3.c2.evaluation import numerical

    scoring = [event | {"x": event["raw_x"], "equilibrium_delta": event["equilibrium"],
                        "event_time_s": event.get("event_time_s")} for event in events]
    refined = task3_evaluate(paths[STEPS[1]], scoring, paths[STEPS[0]])
    valid_prefix = []
    for i, event in enumerate(events):
        count = min(50, len(event["x"])-11)
        valid_prefix.append(count > 0 and numerical(paths[STEPS[0]][i], paths[STEPS[1]][i], event, count)["passed"])
    report, vectors = {}, {}
    for step, scored, other in ((STEPS[0], main, refined), (STEPS[1], refined, main)):
        classifications, peaks = [], []
        for i, (event, a, b) in enumerate(zip(events, scored["classifications"], other["classifications"], strict=True)):
            labels = [("unstable" if row["predicted_los"] else "stable" if row["window_settled"] else None)
                      for row in (a, b)]
            qualified = bool(a["qualified"] and b["qualified"] and labels[0] == labels[1])
            predicted = labels[0] if qualified else None
            classifications.append({"id": event["id"], "truth_label": event["label"], "predicted_label": predicted,
                                    "prospective_eligible": True, "qualified": qualified, "c2_qualified": a["qualified"],
                                    "numerical_failure": not qualified, "unsettled": qualified and predicted is None})
            gap = speed_peak_gap(event, paths[step][i]) if event["label"] == "unstable" and valid_prefix[i] else None
            peaks.append({"id": event["id"], "label": event["label"], "qualified": gap is not None,
                          "available_points": min(50, len(event["x"])-11),
                          "signed_peak_deficit": gap, "positive_peak_deficit_rad_s": (
                              max(0., gap["gap_rad_s"]) if gap is not None else None)})
        stable8 = _stable8(scored, events)
        peak = np.array([np.nan if row["positive_peak_deficit_rad_s"] is None else row["positive_peak_deficit_rad_s"]
                         for row in peaks])
        unstable = np.array([event["label"] == "unstable" for event in events])
        tp = np.array([row["qualified"] and row["predicted_label"] == "unstable" for row in classifications])
        full_finite = np.all(np.isfinite(paths[step]), axis=(1, 2))
        invalid_metrics = [{key: row[key] for key in ("event_id", "horizon")} for row in scored["records"]
                           if not (row["qualified"] and np.isfinite(row["rmse"]))]
        report[step] = {
            "strict": _classification(classifications),
            "engineering": _classification(classifications, engineering=True),
            "classification_rows": classifications,
            "classification_qualified_both_and_equal": sum(row["qualified"] for row in classifications),
            "classification_c2": scored["classifications"], "counts_c2": scored["counts"],
            "full_path_finite": int(full_finite.sum()),
            "full_path_numerical_failure_ids": [event["id"] for i, event in enumerate(events) if not full_finite[i]],
            "unqualified_trajectory_records": invalid_metrics,
            "numerical_pass": (bool(full_finite.all()) and all(row["qualified"] for row in classifications)
                               and bool(scored["records"]) and not invalid_metrics),
            "trajectory": scored["summary"], "stable_trajectory": scored["stable_complete"],
            "unstable_available_trajectory": scored["unstable_prefix"], "trajectory_rows": scored["records"],
            "stable_mean_nrmse_8s": _complete_stat(stable8[~unstable], np.mean),
            "unstable_median_positive_peak_deficit_rad_s": _complete_stat(peak[unstable], np.median),
            "first_swing": [row for row in peaks if row["label"] == "unstable"],
        }
        vectors[step] = (tp, peak, stable8)
    return report, vectors


def _cluster_bootstrap(events, vectors, seed):
    unstable = np.array([event["label"] == "unstable" for event in events])
    observation_groups = np.array([event["group"] for event in events])
    groups = sorted(set(observation_groups.tolist()))
    groups_to_rows = [np.flatnonzero(observation_groups == group) for group in groups]
    indices = np.random.default_rng(seed).integers(len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
    samples = [np.concatenate([groups_to_rows[index] for index in draw]) for draw in indices]
    roster = np.arange(len(events))
    metric_names = ("confirmed_recall", "unstable_median_positive_peak_deficit_rad_s", "stable_mean_nrmse_8s")

    def metrics(values, rows):
        tp, peak, stable8 = values
        positive = rows[unstable[rows]]
        negative = rows[~unstable[rows]]
        return [float(tp[positive].mean()) if len(positive) else np.nan,
                float(np.median(peak[positive])) if len(positive) else np.nan,
                float(stable8[negative].mean()) if len(negative) else np.nan]

    result = {}
    for step in STEPS:
        by_model = {}
        baseline = vectors["v4_hard_r"][step]
        baseline_point = metrics(baseline, roster)
        baseline_samples = np.array([metrics(baseline, sample) for sample in samples])
        for name in MODEL_NAMES[1:]:
            candidate = vectors[name][step]
            point = np.asarray(metrics(candidate, roster))-baseline_point
            differences = np.array([metrics(candidate, sample) for sample in samples])-baseline_samples
            metrics_by_name = {}
            for key, value, draws in zip(metric_names, point, differences.T, strict=True):
                valid = draws[np.isfinite(draws)]
                complete = bool(np.isfinite(value))
                metrics_by_name[key] = {
                    "delta": float(value) if complete else None,
                    "percentile_ci95": (np.quantile(valid, [.025, .975]).tolist()
                                        if complete and len(valid) and len(groups) > 1 else None),
                    "valid_replicates": len(valid), "invalid_replicates": len(draws)-len(valid),
                }
            by_model[name] = metrics_by_name
        result[step] = by_model
    return {"seed": seed, "replicates": BOOTSTRAP_REPLICATES, "unit": "frozen_fault_group",
            "comparator": "v4_hard_r", "direction": "candidate_minus_comparator",
            "groups": {group: len(rows) for group, rows in zip(groups, groups_to_rows, strict=True)},
            "draw_index_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
            "by_step": result}


def _stored_energy(model, events, initialization, step):
    memory = {row["id"]: row["hidden_origin_energy"] for row in initialization
              if str(row["step"]) == step and np.isfinite(row["hidden_origin_energy"])}
    available = [event for event in events if event["id"] in memory]
    complete = len(available) == len(events)
    profile = {"allocated": len(events), "qualified": len(available), "summary": None, "rows": [],
               "missing_ids": [event["id"] for event in events if event["id"] not in memory]}
    positive = np.array([event["label"] == "unstable" for event in available])
    if positive.any() and not positive.all():
        data = energies(model, available, [memory[event["id"]] for event in available])
        for row in data["rows"]:
            row["mechanical"] = row["kinetic"]+row["potential"]
        stats = {key: _group_stats(np.array([row[key] for row in data["rows"]]), positive)
                 for key in ("kinetic", "potential", "mechanical", "memory", "total")}
        profile["summary" if complete else "qualified_subset_diagnostic"] = stats
        profile["rows"] = data["rows"]
    return profile


def _mode_evidence(models, events):
    cpu = {name: deepcopy(models[name]).cpu() for name in MODEL_NAMES}
    references, spectra, observed = {}, {}, {}
    for event in events:
        mass, equilibrium = (np.asarray(event[key], dtype=np.float64) for key in ("mass", "equilibrium"))
        identity = (mass.tobytes(), equilibrium.tobytes())
        if identity not in references:
            index = str(len(references))
            references[identity] = index
            spectra[index] = {name: _spectrum(cpu[name], equilibrium, mass) for name in MODEL_NAMES}
        if event["label"] != "stable":
            continue
        index = references[identity]
        speed = event["x"][10:811, 10:]
        row = {"reference_index": index, "available_forecast_frames": len(speed), "retained_modes": 0,
               "comparisons": {}}
        observed[event["id"]] = row
        if len(speed) != 801:
            row["status"] = "incomplete_window"
            continue
        if not np.isfinite(speed[50:]).all():
            row["status"] = "nonfinite_window"
            continue
        try:
            measured, audit = _ringdown(speed, mass)
        except np.linalg.LinAlgError as error:
            row.update(status="identification_failed", error=str(error))
            continue
        row.update(status="identified" if measured else "no_stabilized_mode", retained_modes=len(measured),
                   identification=audit)
        for name in MODEL_NAMES:
            modes = spectra[index][name][0]
            matches = _assign(measured, modes, frequency_limit=.25, mac_limit=.5)
            for pair in matches.values():
                pair["strong"] = bool(pair["relative_frequency_error"] <= .10 and pair["mac"] >= .80
                                      and abs(pair["damping_ratio_error"]) <= .05
                                      and modes[pair["candidate_index"]].decay_rate <= 0.)
            row["comparisons"][name] = {
                "matches": list(matches.values()), "dominant": matches.get(0),
                "unmatched_empirical_indices": sorted(set(range(len(measured)))-matches.keys()),
                "unmatched_analytic_indices": sorted(set(range(len(modes)))-{m["candidate_index"] for m in matches.values()}),
            }
    status = Counter(row["status"] for row in observed.values())
    shared = [row["comparisons"] for row in observed.values() if row["comparisons"]
              and all(row["comparisons"][name]["dominant"] is not None for name in MODEL_NAMES)]
    summary = {}
    for name in MODEL_NAMES:
        compared = [row["comparisons"][name] for row in observed.values() if row["comparisons"]]
        matches = [pair for row in compared for pair in row["matches"]]
        dominant = [row["dominant"] for row in compared if row["dominant"] is not None]
        strong = sum(pair["strong"] for pair in dominant)
        summary[name] = {"all_matched_event_modes": _medians(matches), "dominant_matched_events": _medians(dominant),
                         "dominant_strong_events": strong,
                         "dominant_strong_fraction_all_events": strong/len(observed) if observed else None,
                         "same_empirical_dominant_matched_by_all": _medians([row[name]["dominant"] for row in shared])}
    return {"role": "blind", "allocated_stable": len(observed), "status_counts": dict(status),
            "retained_event_modes": sum(row["retained_modes"] for row in observed.values()),
            "summary": summary, "events": observed,
            "references": {index: {name: item[1] for name, item in by_model.items()}
                           for index, by_model in spectra.items()}}


def _save_predictions(directory, events, paths, official):
    from task3.protocol import safe

    np.savez_compressed(directory/"blind_predictions.npz", ids=np.array([event["id"] for event in events]),
                        fine=paths[STEPS[0]], refined=paths[STEPS[1]],
                        truth=np.concatenate([event["raw_x"] for event in events]),
                        lengths=np.array([len(event["raw_x"]) for event in events]),
                        mass=np.stack([event["mass"] for event in events]),
                        equilibrium=np.stack([event["equilibrium"] for event in events]),
                        labels=np.array([event["label"] for event in events]))
    write_json(directory/"blind.json", safe(official))


def execute(root, directory, plan, source, events, models, scales, budget):
    """Consume one in-memory sealed roster with the three registered frozen models."""
    events = [event | {"event_time_s": event.get("event_time_s")} for event in events]
    ids = [event["id"] for event in events]
    if not 0 < len(events) <= 100 or len(set(ids)) != len(ids):
        raise ValueError("M5 requires a unique sealed roster of at most 100 events")
    if any(event["role"] != "blind" or event["label"] not in ("stable", "unstable")
           or not isinstance(event["group"], str) or not event["group"] for event in events):
        raise ValueError("M5 requires blind roles, frozen labels and nonempty fault group IDs")
    if set(models) != set(MODEL_NAMES) or models["m3b_g2"].support != ("G2",):
        raise ValueError("the three registered frozen models are required")
    for model in models.values():
        if (model.mass.device.type != "cuda" or model.mass.dtype != torch.float64 or model.topology_dimension
                or model.memory_dimension != 10 or any(p.requires_grad for p in model.parameters())):
            raise ValueError("blind forecasts require frozen CUDA float64 ten-memory models")
    sys.path.insert(0, str(Path(root)/source["task3_root"]/"src"))
    from task3.protocol import safe

    settings = plan["settings"]
    reports, vectors, physics = {}, {}, {}
    for name in MODEL_NAMES:
        model_directory = directory/name
        model_directory.mkdir()
        model = models[name]
        predict, paths, attempted, blocked = _predict_once(name, model, events, scales, settings, budget)
        main = evaluate(model, events, scales["history"], budget, model_directory, "blind", settings,
                        predictor=predict, save=False)
        if attempted != {(event_id, step) for event_id in ids for step in STEPS}:
            raise RuntimeError("one blind prediction attempt per event and step is required")
        rollouts_after_prediction = budget.counts["rollouts"]
        _save_predictions(model_directory, events, paths, main)
        dual, vectors[name] = _score_dual(events, paths, main)
        attempt_audit = {"event_step_attempts": len(attempted), "blocked_retries": dict(blocked),
                         "repeat_integrations": 0, "batch_and_event_failures": main["failures"]}
        write_json(model_directory/"dual_metrics.json", safe({"by_step": dual, "prediction_attempts": attempt_audit}))
        physics[name] = {step: {
            "origin_energy": _stored_energy(model, events, main["initialization"], step),
            "separation_cut": separation_cuts(events, paths[step], dual[step]["classification_rows"]),
        } for step in STEPS}
        if budget.counts["rollouts"] != rollouts_after_prediction:
            raise RuntimeError("post-processing repeated a blind rollout")
        fields = ("strict", "engineering", "classification_qualified_both_and_equal", "full_path_finite",
                  "full_path_numerical_failure_ids",
                  "numerical_pass", "stable_trajectory", "trajectory", "unstable_available_trajectory",
                  "stable_mean_nrmse_8s", "unstable_median_positive_peak_deficit_rad_s")
        reports[name] = {"by_step": {step: {key: dual[step][key] for key in fields} for step in STEPS},
                         "prediction_attempts": attempt_audit}

    rollouts_after_prediction = budget.counts["rollouts"]
    modal = _mode_evidence(models, events)
    bootstrap = _cluster_bootstrap(events, vectors, settings["bootstrap_seed"])
    if budget.counts["rollouts"] != rollouts_after_prediction:
        raise RuntimeError("physical evidence repeated a blind rollout")
    write_json(directory/"event_physics.json", physics)
    write_json(directory/"modes.json", modal)
    write_json(directory/"paired_bootstrap.json", bootstrap)
    report = {
        "role": "blind", "n_events": len(events), "labels": dict(Counter(event["label"] for event in events)),
        "primary_step": STEPS[0], "refined_step": STEPS[1], "models": reports,
        "numerical_pass": all(row["numerical_pass"] for model in reports.values() for row in model["by_step"].values()),
        "physics_new_rollouts": 0, "paired_bootstrap": bootstrap["by_step"], "modal_summary": modal["summary"],
    }
    report = safe(report)
    write_json(directory/"summary.json", report)
    return report
