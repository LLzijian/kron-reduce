"""Pure-array paired scoring for the TASK3 measurement-noise comparison.

The caller owns roster access and forecast generation. This module consumes
only in-memory events, paired paths, and origin-energy arrays.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping

import numpy as np

from kron_reduce.evaluation.first_swing_probes import speed_peak_gap
from kron_reduce.evaluation.postfault_m4 import _auc, _crossing_edges, _partition
from kron_reduce.physics.coordinates import coi_project

STEPS = ("0.0025", "0.00125")
SCALE_DELTA = 0.476753
SCALE_OMEGA = 1.003367
PARITY_SCALE_DELTA = 0.47675314588683515
PARITY_SCALE_OMEGA = 1.0033667249602924
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_GROUP_COUNT = 21
BOOTSTRAP_METRICS = (
    "confirmed_recall", "false_alarm_rate", "coverage", "stable_8s_nrmse",
    "unstable_available_nrmse", "positive_peak_deficit_rad_s", "exact_cut_rate",
    "cut_jaccard_mean", "total_energy_auroc", "stable_8s_nrmse_coverage",
    "unstable_available_nrmse_coverage", "positive_peak_deficit_coverage",
    "cut_coverage", "total_energy_coverage",
)


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _validate_paths(events, paths, energy):
    n = len(events)
    def normalize(mapping, name):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a step mapping")
        result = {}
        for key, value in mapping.items():
            try:
                normalized = str(float(key))
            except (TypeError, ValueError):
                normalized = str(key)
            if normalized not in STEPS or normalized in result:
                raise ValueError(f"{name} must contain exactly {STEPS}")
            result[normalized] = value
        if set(result) != set(STEPS):
            raise ValueError(f"{name} must contain exactly {STEPS}")
        return result

    paths = normalize(paths, "paths")
    energy = normalize(energy, "energy")
    path_arrays, energy_arrays = {}, {}
    for step in STEPS:
        path = np.asarray(paths[step], dtype=np.float64)
        if path.shape != (n, 801, 20):
            raise ValueError(f"path {step} must have shape (n, 801, 20)")
        values = np.asarray(energy[step], dtype=np.float64)
        if values.shape != (n, 3):
            raise ValueError(f"energy {step} must have shape (n, 3)")
        if np.isinf(values).any():
            raise ValueError("energy may use NaN for unavailable values but not infinity")
        path_arrays[step], energy_arrays[step] = path, values
    ids = [event.get("id") for event in events]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != n:
        raise ValueError("events must have unique nonempty IDs")
    for event in events:
        if event.get("label") not in ("stable", "unstable"):
            raise ValueError("events must carry stable/unstable labels")
        if not isinstance(event.get("group"), str) or not event["group"]:
            raise ValueError("events must carry nonempty fault groups")
        x = np.asarray(event.get("x"), dtype=np.float64)
        mass = np.asarray(event.get("mass"), dtype=np.float64)
        equilibrium = np.asarray(event.get("equilibrium"), dtype=np.float64)
        p0 = np.asarray(event.get("p0"), dtype=np.float64)
        if (x.ndim != 2 or x.shape[1] != 20 or mass.shape != (10,)
                or equilibrium.shape != (10,) or p0.shape != (10,)):
            raise ValueError(f"invalid projected event arrays: {event['id']}")
        if np.any(mass <= 0) or not np.isfinite(mass).all():
            raise ValueError(f"invalid inertia: {event['id']}")
        omega_s = event.get("omega_s")
        if (isinstance(omega_s, (bool, np.bool_)) or not np.isfinite(float(omega_s))
                or float(omega_s) <= 0):
            raise ValueError(f"invalid synchronous frequency: {event['id']}")
    return path_arrays, energy_arrays


def _endpoint_values(endpoint, events):
    if endpoint is None:
        return [None] * len(events)
    if isinstance(endpoint, Mapping):
        values = [endpoint.get(event["id"]) for event in events]
    elif np.isscalar(endpoint):
        values = [endpoint] * len(events)
    else:
        values = list(endpoint)
        if len(values) != len(events):
            raise ValueError("endpoint sequence must align with events")
    if any(value not in (None, 0, 10) for value in values):
        raise ValueError("endpoint strata must be None, 0, or 10")
    return [None if value is None else int(value) for value in values]


def _trigger(path, event, *, start=1):
    """Return the first prospective full-path index with five consecutive flags."""
    future = np.asarray(path, dtype=np.float64)
    finite = np.isfinite(future).all(axis=1)
    if not finite.all():
        stop = int(np.flatnonzero(~finite)[0])
        future = future[:stop]
    if start >= len(future):
        return None, None, bool(finite.all())
    angle = np.ptp(future[:, :10] + np.asarray(event["equilibrium"]), axis=1)
    speed = np.max(np.abs(coi_project(future[:, 10:], event["mass"])), axis=1) / float(event["omega_s"])
    flags = (angle > np.pi) & (speed > 0.02)
    prospective = flags[start:]
    if len(prospective) >= 5:
        windows = np.lib.stride_tricks.sliding_window_view(prospective, 5).all(axis=-1)
        hits = np.flatnonzero(windows)
        if len(hits):
            end = start + int(hits[0]) + 4
            return end, {"maximum_angle_spread_rad": float(np.max(angle)),
                         "maximum_coi_speed_pu": float(np.max(speed))}, bool(finite.all())
    return None, {"maximum_angle_spread_rad": float(np.max(angle)) if len(angle) else None,
                  "maximum_coi_speed_pu": float(np.max(speed)) if len(speed) else None}, bool(finite.all())


def _classify(path, event):
    hit, details, complete = _trigger(path, event, start=1)
    if not np.isfinite(path).all():
        return {"predicted_label": None, "trigger_index": hit, "window_settled": False,
                "finite_full_path": False, "details": details}
    angle = np.ptp(path[:, :10] + np.asarray(event["equilibrium"]), axis=1)
    speed = np.max(np.abs(coi_project(path[:, 10:], event["mass"])), axis=1) / float(event["omega_s"])
    settled = bool(hit is None and complete and np.all(angle[-100:] < np.pi)
                   and np.all(speed[-100:] < 0.005))
    label = "unstable" if hit is not None else "stable" if settled else None
    return {"predicted_label": label, "trigger_index": hit, "window_settled": settled,
            "finite_full_path": bool(complete), "details": details}


def _step_parity(coarse, fine, coarse_class, fine_class):
    finite = bool(np.isfinite(coarse).all() and np.isfinite(fine).all())
    same_los = (coarse_class["trigger_index"] is not None) == (fine_class["trigger_index"] is not None)
    same_label = coarse_class["predicted_label"] == fine_class["predicted_label"]
    trigger_ok, end = True, 800
    if same_los and coarse_class["trigger_index"] is not None:
        trigger_gap = abs(coarse_class["trigger_index"] - fine_class["trigger_index"])
        trigger_ok = trigger_gap <= 2
        end = max(0, max(coarse_class["trigger_index"], fine_class["trigger_index"]) - 20)
    else:
        trigger_gap = None
    error = coarse[:end + 1] - fine[:end + 1]
    value = (float(np.sqrt(0.5 * np.mean(
        (error[:, :10] / PARITY_SCALE_DELTA) ** 2 + (error[:, 10:] / PARITY_SCALE_OMEGA) ** 2
    ))) if finite else None)
    return {"qualified": bool(finite and same_los and same_label and trigger_ok
                              and value is not None and value <= 1e-3),
            "finite_full_paths": finite, "same_los_decision": same_los,
            "same_stability_label": same_label, "trigger_index_difference": trigger_gap,
            "trigger_times_within_20ms": trigger_ok,
            "normalized_joint_state_difference": value,
            "comparison_points_including_origin": end + 1,
            "normalized_joint_state_difference_limit": 1e-3}


def _nrmse(prediction, truth):
    if (not len(truth) or prediction.shape != truth.shape
            or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.sqrt(0.5 * np.mean(
            ((prediction[:, :10] - truth[:, :10]) / SCALE_DELTA) ** 2
            + ((prediction[:, 10:] - truth[:, 10:]) / SCALE_OMEGA) ** 2
        ))
    return float(value) if np.isfinite(value) else None


def _truth_trigger(event):
    x = np.asarray(event["x"], dtype=np.float64)
    if len(x) <= 10 or not np.isfinite(x[10:]).all():
        return None
    hit, _, _ = _trigger(x, event, start=10)
    return hit


def _cut_metrics(event, predicted_path, predicted_hit):
    truth_hit = _truth_trigger(event)
    truth_cut = prediction_cut = None
    exact_leading = exact_cut = jaccard = None
    if truth_hit is not None:
        truth_cut = _partition(event["x"], truth_hit, event)
    if predicted_hit is not None:
        full_path = np.concatenate((event["x"][:11], predicted_path[1:]), axis=0)
        prediction_cut = _partition(full_path, 10 + predicted_hit, event)
    if truth_cut is not None and prediction_cut is not None:
        truth_edges = _crossing_edges(truth_cut)
        prediction_edges = _crossing_edges(prediction_cut)
        union = truth_edges | prediction_edges
        exact_leading = truth_cut["leading"] == prediction_cut["leading"]
        exact_cut = truth_edges == prediction_edges
        jaccard = len(truth_edges & prediction_edges) / len(union) if union else 1.0
    return {"truth_trigger_index": truth_hit, "prediction_trigger_index": predicted_hit,
            "truth_cut": truth_cut, "prediction_cut": prediction_cut,
            "exact_leading": exact_leading, "exact_cut": exact_cut, "cut_jaccard": jaccard}


def _classification_summary(rows):
    counts = Counter(row["status"] for row in rows)
    tp = sum(row["status"] == "unstable" and row["label"] == "unstable" for row in rows)
    fn = sum(row["status"] == "stable" and row["label"] == "unstable" for row in rows)
    fp = sum(row["status"] == "unstable" and row["label"] == "stable" for row in rows)
    tn = sum(row["status"] == "stable" and row["label"] == "stable" for row in rows)
    unresolved, numerical, uncovered = counts["unresolved"], counts["numerical_failure"], counts["uncovered"]
    resolved, total = tp + tn + fp + fn, len(rows)
    # Fixed roster denominators include uncovered, unresolved, and numerical cases.
    positive = sum(row["label"] == "unstable" for row in rows)
    negative = sum(row["label"] == "stable" for row in rows)
    complete = unresolved == 0 and numerical == 0 and uncovered == 0
    return {
        "allocated": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "unresolved": unresolved, "numerical_failure": numerical, "uncovered": uncovered,
        "stable_unsettled": sum(row["label"] == "stable" and row["status"] == "unresolved" for row in rows),
        "false_settled": sum(row["label"] == "unstable" and row["status"] == "stable" for row in rows),
        "resolved": resolved, "coverage": _safe_ratio(resolved, total),
        "accuracy": _safe_ratio(tp + tn, total) if complete else None,
        "accuracy_on_resolved": _safe_ratio(tp + tn, resolved),
        "eligible_unstable": positive, "eligible_stable": negative,
        "confirmed_recall": _safe_ratio(tp, positive),
        "confirmed_miss_rate": _safe_ratio(fn, positive),
        "false_alarm_rate": _safe_ratio(fp, negative),
        "status_counts": dict(sorted(counts.items())),
    }


def _energy_summary(rows):
    result = {}
    for name in ("mechanical", "memory", "total"):
        available = [row for row in rows if row["energy"][name] is not None]
        positive = np.asarray([row["label"] == "unstable" for row in available], dtype=bool)
        values = np.asarray([row["energy"][name] for row in available], dtype=np.float64)
        result[name] = {"allocated": len(rows), "available": len(available),
                        "coverage": _safe_ratio(len(available), len(rows)),
                        "auroc": _auc(positive, values) if len(values) else None}
    return result


def _metrics(rows):
    summary = _classification_summary(rows)
    for field in ("stable_8s_nrmse", "unstable_available_nrmse", "positive_peak_deficit_rad_s",
                  "exact_cut", "cut_jaccard"):
        values = [row[field] for row in rows if row[field] is not None]
        summary[field] = {"allocated": len(rows), "available": len(values),
                          "coverage": _safe_ratio(len(values), len(rows)),
                          "mean": float(np.mean(values)) if values else None,
                          "median": float(np.median(values)) if values else None,
                          "exact_rate": float(np.mean(values)) if values and field == "exact_cut" else None}
    summary["energy"] = _energy_summary(rows)
    return summary


def _stratified_summary(rows):
    output = {"all": _metrics(rows)}
    for endpoint in (0, 10):
        output[str(endpoint)] = _metrics([row for row in rows if row["endpoint_index"] == endpoint])
    return output


def score_pair(events, paths, energy, *, endpoint=None) -> dict:
    """Score paired trajectories without file access or forecast generation."""
    events = list(events)
    path_arrays, energy_arrays = _validate_paths(events, paths, energy)
    endpoint_values = _endpoint_values(endpoint, events)
    classifications = {
        step: [_classify(path_arrays[step][i], event) for i, event in enumerate(events)]
        for step in STEPS
    }
    parity = [_step_parity(path_arrays[STEPS[0]][i], path_arrays[STEPS[1]][i],
                           classifications[STEPS[0]][i], classifications[STEPS[1]][i])
              for i in range(len(events))]
    result = {"steps": {}, "pairwise": {"rows": [], "summary": None},
              "protocol": {
                  "positive_class": "unstable",
                  "trigger": "angle range > pi AND projected speed > 0.02 pu for 5 consecutive prospective 10 ms points",
                  "window_settled": "complete finite path, no trigger, final 100 points angle range < pi and projected speed < 0.005 pu",
                  "dual_grid": {"joint_state_difference_limit": 1e-3, "trigger_index_tolerance": 2,
                                "state_scales": {"delta": PARITY_SCALE_DELTA, "omega": PARITY_SCALE_OMEGA}},
                  "nrmse_scales": {"delta": SCALE_DELTA, "omega": SCALE_OMEGA},
              }}
    for step in STEPS:
        rows = []
        for i, event in enumerate(events):
            path = path_arrays[step][i]
            classification = classifications[step][i]
            truth = np.asarray(event["x"], dtype=np.float64)[11:811]
            truth_count = len(truth)
            stable_nrmse = None
            unstable_nrmse = None
            if event["label"] == "stable" and truth_count >= 800:
                stable_nrmse = _nrmse(path[1:801], truth[:800])
            elif event["label"] == "unstable" and truth_count:
                count = min(800, truth_count)
                unstable_nrmse = _nrmse(path[1:1 + count], truth[:count])
            gap = speed_peak_gap(event, path) if event["label"] == "unstable" else None
            if gap is not None and not np.isfinite(gap["gap_rad_s"]):
                gap = None
            signed_gap = float(gap["gap_rad_s"]) if gap is not None else None
            positive_gap = max(0.0, signed_gap) if signed_gap is not None else None
            cuts = (_cut_metrics(event, path, classification["trigger_index"])
                    if event["label"] == "unstable" else {
                        "truth_trigger_index": None, "prediction_trigger_index": None,
                        "truth_cut": None, "prediction_cut": None, "exact_leading": None,
                        "exact_cut": None, "cut_jaccard": None})
            if truth_count == 0:
                status = "uncovered"
            elif not parity[i]["qualified"] or not classification["finite_full_path"]:
                status = "numerical_failure"
            elif classification["predicted_label"] is None:
                status = "unresolved"
            else:
                status = classification["predicted_label"]
            energy_values = energy_arrays[step][i]
            rows.append({
                "id": event["id"], "event_id": event["id"], "label": event["label"],
                "group": event["group"], "status": status, "predicted_label": classification["predicted_label"],
                "endpoint_index": endpoint_values[i], "event_time_s": event.get("event_time_s"),
                "truth_future_points": truth_count, "finite_full_path": classification["finite_full_path"],
                "window_settled": classification["window_settled"],
                "trigger_index_forecast": classification["trigger_index"],
                "trigger_time_from_origin_s": (float(classification["trigger_index"] * 0.01)
                                               if classification["trigger_index"] is not None else None),
                "classification_details": classification["details"], "step_parity": parity[i],
                "stable_8s_nrmse": stable_nrmse, "unstable_available_nrmse": unstable_nrmse,
                "speed_peak_gap": gap, "signed_peak_gap_rad_s": signed_gap,
                "positive_peak_deficit_rad_s": positive_gap, **cuts,
                "energy": {name: float(energy_values[j]) if np.isfinite(energy_values[j]) else None
                           for j, name in enumerate(("mechanical", "memory", "total"))},
            })
        result["steps"][step] = {
            "rows": rows, "classification": _classification_summary(rows), "metrics": _metrics(rows),
            "endpoint_strata": _stratified_summary(rows),
        }
    pair_rows = [{"id": event["id"], "label": event["label"], "group": event["group"],
                  "endpoint_index": endpoint_values[i], **parity[i]} for i, event in enumerate(events)]
    qualified = sum(row["qualified"] for row in pair_rows)
    result["pairwise"] = {
        "rows": pair_rows, "summary": {"allocated": len(pair_rows), "qualified": qualified,
                                        "unqualified": len(pair_rows) - qualified,
                                        "coverage": _safe_ratio(qualified, len(pair_rows))},
    }
    return result


_METRIC_DIRECTION = {
    "confirmed_recall": "higher", "false_alarm_rate": "lower", "coverage": "higher",
    "stable_8s_nrmse": "lower", "unstable_available_nrmse": "lower",
    "positive_peak_deficit_rad_s": "lower", "exact_cut_rate": "higher",
    "cut_jaccard_mean": "higher", "total_energy_auroc": "higher",
    "stable_8s_nrmse_coverage": "higher", "unstable_available_nrmse_coverage": "higher",
    "positive_peak_deficit_coverage": "higher", "cut_coverage": "higher",
    "total_energy_coverage": "higher",
}


def _scenario_step_metrics(rep_output, step):
    profile = rep_output["steps"][step]["metrics"]
    return {
        "confirmed_recall": profile["confirmed_recall"],
        "false_alarm_rate": profile["false_alarm_rate"],
        "coverage": profile["coverage"],
        "stable_8s_nrmse": profile["stable_8s_nrmse"]["mean"],
        "unstable_available_nrmse": profile["unstable_available_nrmse"]["mean"],
        "positive_peak_deficit_rad_s": profile["positive_peak_deficit_rad_s"]["median"],
        "exact_cut_rate": profile["exact_cut"]["exact_rate"],
        "cut_jaccard_mean": profile["cut_jaccard"]["mean"],
        "total_energy_auroc": profile["energy"]["total"]["auroc"],
        "stable_8s_nrmse_coverage": profile["stable_8s_nrmse"]["coverage"],
        "unstable_available_nrmse_coverage": profile["unstable_available_nrmse"]["coverage"],
        "positive_peak_deficit_coverage": profile["positive_peak_deficit_rad_s"]["coverage"],
        "cut_coverage": profile["exact_cut"]["coverage"],
        "total_energy_coverage": profile["energy"]["total"]["coverage"],
    }


_REPEAT_SUMMARY_DIRECTIONS = {
    "tp": "higher", "tn": "higher", "fp": "lower", "fn": "lower",
    "unresolved": "lower", "uncovered": "lower", "numerical_failure": "lower",
    "accuracy": "higher", "accuracy_on_resolved": "higher", "coverage": "higher",
    "confirmed_recall": "higher", "confirmed_miss_rate": "lower", "false_alarm_rate": "lower",
    "mechanical_energy_auroc": "higher", "memory_energy_auroc": "higher",
}


def _repeat_profile(rep_output, step, endpoint):
    rows = rep_output["steps"][step]["rows"]
    if endpoint != "all":
        endpoint_value = int(endpoint)
        rows = [row for row in rows if row["endpoint_index"] == endpoint_value]
    classification = _classification_summary(rows)
    metrics = _metrics(rows)
    profile = _scenario_step_metrics(rep_output, step) if endpoint == "all" else {
        "stable_8s_nrmse": metrics["stable_8s_nrmse"]["mean"],
        "unstable_available_nrmse": metrics["unstable_available_nrmse"]["mean"],
        "positive_peak_deficit_rad_s": metrics["positive_peak_deficit_rad_s"]["median"],
        "exact_cut_rate": metrics["exact_cut"]["exact_rate"],
        "cut_jaccard_mean": metrics["cut_jaccard"]["mean"],
        "total_energy_auroc": metrics["energy"]["total"]["auroc"],
        "stable_8s_nrmse_coverage": metrics["stable_8s_nrmse"]["coverage"],
        "unstable_available_nrmse_coverage": metrics["unstable_available_nrmse"]["coverage"],
        "positive_peak_deficit_coverage": metrics["positive_peak_deficit_rad_s"]["coverage"],
        "cut_coverage": metrics["exact_cut"]["coverage"],
        "total_energy_coverage": metrics["energy"]["total"]["coverage"],
    }
    profile.update({key: classification[key] for key in _REPEAT_SUMMARY_DIRECTIONS
                    if key in classification})
    profile["mechanical_energy_auroc"] = metrics["energy"]["mechanical"]["auroc"]
    profile["memory_energy_auroc"] = metrics["energy"]["memory"]["auroc"]
    return profile


def _summarize_repeat_values(values, metric, direction):
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    available_indices = np.flatnonzero(np.isfinite(array))
    available = array[available_indices]
    complete = len(available) == len(array)
    worst_offset = (int(np.argmin(available) if direction == "higher" else np.argmax(available))
                    if len(available) else None)
    return {
        "direction": direction, "replicates": len(array), "available_replicates": len(available),
        "coverage": _safe_ratio(len(available), len(array)),
        "mean": float(np.mean(available)) if complete else None,
        "sample_std": float(np.std(available, ddof=1)) if complete and len(available) > 1 else None,
        "worst_rep": (float(available[worst_offset]) if worst_offset is not None else None),
        "worst_rep_complete": complete,
        "worst_rep_index": (int(available_indices[worst_offset]) if worst_offset is not None else None),
        "valid_subset_mean": float(np.mean(available)) if len(available) else None,
        "valid_subset_std": float(np.std(available, ddof=1)) if len(available) > 1 else None,
    }


def summarize_reps(reps):
    """Summarize scores by step and endpoint, preserving per-metric coverage."""
    reps = list(reps)
    if not reps:
        raise ValueError("at least one replicate score is required")
    result = {"replicates": len(reps), "by_step": {}}
    for step in STEPS:
        by_endpoint = {}
        for endpoint in ("all", 0, 10):
            by_rep = [_repeat_profile(rep, step, endpoint) for rep in reps]
            metrics = set().union(*(row.keys() for row in by_rep))
            summary = {}
            for metric in metrics:
                direction = _REPEAT_SUMMARY_DIRECTIONS.get(
                    metric, _METRIC_DIRECTION.get(metric, "higher"))
                summary[metric] = _summarize_repeat_values(
                    [row.get(metric) for row in by_rep], metric, direction)
            by_endpoint[str(endpoint)] = summary
        result["by_step"][step] = by_endpoint
    return result


def _metric_values(rows):
    classification, metrics = _classification_summary(rows), _metrics(rows)
    return {
        "confirmed_recall": classification["confirmed_recall"],
        "false_alarm_rate": classification["false_alarm_rate"],
        "coverage": classification["coverage"],
        "stable_8s_nrmse": metrics["stable_8s_nrmse"]["mean"],
        "unstable_available_nrmse": metrics["unstable_available_nrmse"]["mean"],
        "positive_peak_deficit_rad_s": metrics["positive_peak_deficit_rad_s"]["median"],
        "exact_cut_rate": metrics["exact_cut"]["exact_rate"],
        "cut_jaccard_mean": metrics["cut_jaccard"]["mean"],
        "total_energy_auroc": metrics["energy"]["total"]["auroc"],
        "stable_8s_nrmse_coverage": metrics["stable_8s_nrmse"]["coverage"],
        "unstable_available_nrmse_coverage": metrics["unstable_available_nrmse"]["coverage"],
        "positive_peak_deficit_coverage": metrics["positive_peak_deficit_rad_s"]["coverage"],
        "cut_coverage": metrics["exact_cut"]["coverage"],
        "total_energy_coverage": metrics["energy"]["total"]["coverage"],
    }


def _group_bootstrap_data(rows, groups):
    if isinstance(rows, Mapping):
        rows = rows.values()
    group_index = {group: index for index, group in enumerate(groups)}
    data = {name: np.zeros(len(groups), dtype=np.float64) for name in (
        "total", "positive", "negative", "tp", "fp", "resolved",
        "stable_nrmse_sum", "stable_nrmse_count", "unstable_nrmse_sum", "unstable_nrmse_count",
        "peak_sum", "peak_count", "exact_sum", "exact_count", "jaccard_sum", "jaccard_count",
        "energy_count",
    )}
    peak_values, peak_groups, energy_pos, energy_neg = [], [], [[] for _ in groups], [[] for _ in groups]
    for row in rows:
        g = group_index[row["group"]]
        label, status = row["label"], row["status"]
        data["total"][g] += 1
        data["positive" if label == "unstable" else "negative"][g] += 1
        resolved = status in ("unstable", "stable")
        data["resolved"][g] += resolved
        data["tp"][g] += status == "unstable" and label == "unstable"
        data["fp"][g] += status == "unstable" and label == "stable"
        if row["stable_8s_nrmse"] is not None:
            data["stable_nrmse_sum"][g] += row["stable_8s_nrmse"]
            data["stable_nrmse_count"][g] += 1
        if row["unstable_available_nrmse"] is not None:
            data["unstable_nrmse_sum"][g] += row["unstable_available_nrmse"]
            data["unstable_nrmse_count"][g] += 1
        peak = row["positive_peak_deficit_rad_s"]
        if peak is not None:
            peak_values.append(float(peak))
            peak_groups.append(g)
            data["peak_sum"][g] += 1
            data["peak_count"][g] += 1
        if row["exact_cut"] is not None:
            data["exact_sum"][g] += bool(row["exact_cut"])
            data["exact_count"][g] += 1
        if row["cut_jaccard"] is not None:
            data["jaccard_sum"][g] += row["cut_jaccard"]
            data["jaccard_count"][g] += 1
        energy = row["energy"]["total"]
        if energy is not None:
            data["energy_count"][g] += 1
            (energy_pos if label == "unstable" else energy_neg)[g].append(float(energy))
    data["peak_values"] = np.asarray(peak_values, dtype=np.float64)
    data["peak_groups"] = np.asarray(peak_groups, dtype=np.intp)
    data["energy_pos"] = energy_pos
    data["energy_neg"] = energy_neg
    return data


def _bootstrap_metric_draws(group_draws, grouped_reps):
    """Evaluate resamples from group-level sufficient statistics in arrays."""
    batch, rep_count, group_count = group_draws.shape
    stats = {name: np.stack([rep[name] for rep in grouped_reps]) for name in (
        "total", "positive", "negative", "tp", "fp", "resolved",
        "stable_nrmse_sum", "stable_nrmse_count", "unstable_nrmse_sum", "unstable_nrmse_count",
        "peak_count", "exact_sum", "exact_count", "jaccard_sum", "jaccard_count", "energy_count",
    )}

    def weighted_sum(name):
        return np.einsum("brg,rg->b", group_draws, stats[name], optimize=True)

    total = weighted_sum("total")
    positive, negative = weighted_sum("positive"), weighted_sum("negative")
    tp, fp, resolved = weighted_sum("tp"), weighted_sum("fp"), weighted_sum("resolved")
    stable_n = weighted_sum("stable_nrmse_count")
    unstable_n = weighted_sum("unstable_nrmse_count")
    stable_rmse = np.divide(weighted_sum("stable_nrmse_sum"), stable_n,
                            out=np.full(batch, np.nan), where=stable_n > 0)
    unstable_rmse = np.divide(weighted_sum("unstable_nrmse_sum"), unstable_n,
                              out=np.full(batch, np.nan), where=unstable_n > 0)
    exact_n = weighted_sum("exact_count")
    exact_rate = np.divide(weighted_sum("exact_sum"), exact_n,
                           out=np.full(batch, np.nan), where=exact_n > 0)
    jaccard_n = weighted_sum("jaccard_count")
    jaccard_mean = np.divide(weighted_sum("jaccard_sum"), jaccard_n,
                              out=np.full(batch, np.nan), where=jaccard_n > 0)
    energy_n = weighted_sum("energy_count")
    energy_coverage = np.divide(energy_n, total, out=np.full(batch, np.nan), where=total > 0)

    peak_values, peak_rep_indices, peak_group_indices = [], [], []
    energy_positive, energy_negative = [], []
    for rep_index, rep in enumerate(grouped_reps):
        peak_values.extend(rep["peak_values"].tolist())
        peak_rep_indices.extend([rep_index] * len(rep["peak_values"]))
        peak_group_indices.extend(rep["peak_groups"].tolist())
        for group_index in range(group_count):
            energy_positive.append(rep["energy_pos"][group_index])
            energy_negative.append(rep["energy_neg"][group_index])
    peak_values = np.asarray(peak_values, dtype=np.float64)
    if len(peak_values):
        peak_rep_indices = np.asarray(peak_rep_indices, dtype=np.intp)
        peak_group_indices = np.asarray(peak_group_indices, dtype=np.intp)
        order = np.argsort(peak_values, kind="stable")
        weights = group_draws[:, peak_rep_indices[order], peak_group_indices[order]]
        cumulative = np.cumsum(weights, axis=1)
        peak_total = cumulative[:, -1]
        lower_rank = np.maximum(1, (peak_total + 1) // 2)
        upper_rank = np.maximum(1, (peak_total + 2) // 2)
        lower_index = np.argmax(cumulative >= lower_rank[:, None], axis=1)
        upper_index = np.argmax(cumulative >= upper_rank[:, None], axis=1)
        peak_median = 0.5 * (peak_values[order[lower_index]] + peak_values[order[upper_index]])
        peak_median[peak_total == 0] = np.nan
    else:
        peak_median = np.full(batch, np.nan)

    # A group-by-group pair matrix turns AUROC into one batched bilinear form.
    group_rep_count = rep_count * group_count
    auc_pairs = np.zeros((group_rep_count, group_rep_count), dtype=np.float64)
    pos_counts = np.zeros(group_rep_count, dtype=np.float64)
    neg_counts = np.zeros(group_rep_count, dtype=np.float64)
    for index, values in enumerate(energy_positive):
        pos_counts[index] = len(values)
    for index, values in enumerate(energy_negative):
        neg_counts[index] = len(values)
    for i, pos in enumerate(energy_positive):
        if not pos:
            continue
        pos = np.asarray(pos, dtype=np.float64)[:, None]
        for j, neg in enumerate(energy_negative):
            if neg:
                neg = np.asarray(neg, dtype=np.float64)[None, :]
                auc_pairs[i, j] = np.sum((pos > neg) + 0.5 * (pos == neg))
    flat_draws = group_draws.reshape(batch, group_rep_count)
    positive_weights = flat_draws * pos_counts[None, :]
    negative_weights = flat_draws * neg_counts[None, :]
    auc_denominator = positive_weights.sum(axis=1) * negative_weights.sum(axis=1)
    # auc_pairs stores the full sum of pairwise wins for each group pair.
    auc_numerator = np.einsum("bi,ij,bj->b", flat_draws, auc_pairs, flat_draws, optimize=True)
    energy_auc = np.divide(auc_numerator, auc_denominator,
                           out=np.full(batch, np.nan), where=auc_denominator > 0)

    return {
        "confirmed_recall": np.divide(tp, positive, out=np.full(batch, np.nan), where=positive > 0),
        "false_alarm_rate": np.divide(fp, negative, out=np.full(batch, np.nan), where=negative > 0),
        "coverage": np.divide(resolved, total, out=np.full(batch, np.nan), where=total > 0),
        "stable_8s_nrmse": stable_rmse,
        "unstable_available_nrmse": unstable_rmse,
        "positive_peak_deficit_rad_s": peak_median,
        "exact_cut_rate": exact_rate,
        "cut_jaccard_mean": jaccard_mean,
        "total_energy_auroc": energy_auc,
        "stable_8s_nrmse_coverage": np.divide(weighted_sum("stable_nrmse_count"), negative,
                                                out=np.full(batch, np.nan), where=negative > 0),
        "unstable_available_nrmse_coverage": np.divide(weighted_sum("unstable_nrmse_count"), positive,
                                                         out=np.full(batch, np.nan), where=positive > 0),
        "positive_peak_deficit_coverage": np.divide(weighted_sum("peak_count"), positive,
                                                      out=np.full(batch, np.nan), where=positive > 0),
        "cut_coverage": np.divide(exact_n, positive,
                                  out=np.full(batch, np.nan), where=positive > 0),
        "total_energy_coverage": energy_coverage,
    }


def paired_group_bootstrap(candidate_reps, original_reps, seed=20261126):
    """Paired 21-group bootstrap for candidate-minus-original metric deltas."""
    candidate_reps, original_reps = list(candidate_reps), list(original_reps)
    if not candidate_reps or len(candidate_reps) != len(original_reps):
        raise ValueError("candidate and original need the same nonzero replicate count")
    by_step, draw_hash = {}, hashlib.sha256()
    rng = np.random.default_rng(seed)
    base_group_draws = rng.multinomial(
        BOOTSTRAP_GROUP_COUNT, np.full(BOOTSTRAP_GROUP_COUNT, 1 / BOOTSTRAP_GROUP_COUNT),
        size=BOOTSTRAP_REPLICATES,
    ).astype(np.int16, copy=False)
    draw_hash.update(base_group_draws.astype("<i8", copy=False).tobytes())
    common_groups = None
    for step in STEPS:
        rep_pairs, canonical_groups = [], None
        for candidate, original in zip(candidate_reps, original_reps, strict=True):
            left = candidate["steps"][step]["rows"]
            right = original["steps"][step]["rows"]
            left_ids, right_ids = ({row["id"]: row for row in values} for values in (left, right))
            if set(left_ids) != set(right_ids):
                raise ValueError("paired bootstrap requires identical event IDs")
            if any(left_ids[event_id]["group"] != right_ids[event_id]["group"] for event_id in left_ids):
                raise ValueError("paired bootstrap event groups differ")
            groups = sorted({row["group"] for row in left})
            if len(groups) != BOOTSTRAP_GROUP_COUNT:
                raise ValueError("paired bootstrap requires exactly 21 fault groups per replicate")
            if canonical_groups is None:
                canonical_groups = groups
            elif groups != canonical_groups:
                raise ValueError("group roster differs between replicates")
            rep_pairs.append((left_ids, right_ids, groups))
        if common_groups is None:
            common_groups = canonical_groups
        elif canonical_groups != common_groups:
            raise ValueError("group roster differs between steps")
        draw_counts = np.broadcast_to(
            base_group_draws[:, None, :],
            (BOOTSTRAP_REPLICATES, len(rep_pairs), BOOTSTRAP_GROUP_COUNT),
        )
        point_left = [row for left, _, _ in rep_pairs for row in left.values()]
        point_right = [row for _, right, _ in rep_pairs for row in right.values()]
        left_point, right_point = _metric_values(point_left), _metric_values(point_right)
        point_delta = {
            metric: (float(left_point[metric] - right_point[metric])
                     if left_point[metric] is not None and right_point[metric] is not None else None)
            for metric in BOOTSTRAP_METRICS
        }
        left_grouped = [_group_bootstrap_data(left, groups) for left, _, groups in rep_pairs]
        right_grouped = [_group_bootstrap_data(right, groups) for _, right, groups in rep_pairs]
        left_draws = _bootstrap_metric_draws(draw_counts, left_grouped)
        right_draws = _bootstrap_metric_draws(draw_counts, right_grouped)
        metric_result = {}
        for metric in BOOTSTRAP_METRICS:
            draws = left_draws[metric] - right_draws[metric]
            valid = draws[np.isfinite(draws)]
            metric_result[metric] = {
                "candidate_minus_original": point_delta[metric],
                "percentile_ci95": (np.quantile(valid, [0.025, 0.975]).tolist()
                                    if point_delta[metric] is not None
                                    and len(valid) == BOOTSTRAP_REPLICATES else None),
                "valid_replicates": len(valid),
                "invalid_replicates": int(BOOTSTRAP_REPLICATES - len(valid)),
                "direction": _METRIC_DIRECTION[metric],
            }
        by_step[step] = metric_result
    return {
        "seed": int(seed), "replicates": BOOTSTRAP_REPLICATES,
        "noise_replicates": len(candidate_reps), "unit": "fault_group_resampled_within_rep",
        "groups_per_rep": BOOTSTRAP_GROUP_COUNT, "direction": "candidate_minus_original",
        "shared_group_draws_sha256": draw_hash.hexdigest(),
        "by_step": by_step,
    }
