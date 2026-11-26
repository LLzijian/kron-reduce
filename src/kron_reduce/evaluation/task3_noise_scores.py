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
SCALE_DELTA = 0.47675314588683515
SCALE_OMEGA = 1.0033667249602924
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_GROUP_COUNT = 21
BOOTSTRAP_METRICS = (
    "confirmed_recall", "false_alarm_rate", "coverage", "stable_8s_nrmse",
    "unstable_available_nrmse", "positive_peak_deficit_rad_s", "exact_cut_rate",
    "cut_jaccard_mean", "total_energy_auroc",
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
    for end in range(max(start + 4, 4), len(flags)):
        if np.all(flags[end - 4:end + 1]):
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
        (error[:, :10] / SCALE_DELTA) ** 2 + (error[:, 10:] / SCALE_OMEGA) ** 2
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
    positive = sum(row["label"] == "unstable" for row in rows if row["status"] != "uncovered")
    negative = sum(row["label"] == "stable" for row in rows if row["status"] != "uncovered")
    complete = unresolved == 0 and numerical == 0 and uncovered == 0
    return {
        "allocated": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "unresolved": unresolved, "numerical_failure": numerical, "uncovered": uncovered,
        "stable_unsettled": sum(row["label"] == "stable" and row["status"] == "unresolved" for row in rows),
        "false_settled": sum(row["label"] == "unstable" and row["status"] == "stable" for row in rows),
        "resolved": resolved, "coverage": _safe_ratio(resolved, total),
        "accuracy": _safe_ratio(tp + tn, total) if complete else None,
        "accuracy_on_resolved": _safe_ratio(tp + tn, resolved),
        "confirmed_recall": _safe_ratio(tp, positive) if complete else None,
        "confirmed_miss_rate": _safe_ratio(fn, positive) if complete else None,
        "false_alarm_rate": _safe_ratio(fp, negative) if complete else None,
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
                                "state_scales": {"delta": SCALE_DELTA, "omega": SCALE_OMEGA}},
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
    }


def summarize_reps(reps):
    """Summarize repeats and make incomplete metric coverage explicit."""
    reps = list(reps)
    if not reps:
        raise ValueError("at least one replicate score is required")
    result = {"replicates": len(reps), "by_step": {}}
    for step in STEPS:
        by_rep = [_scenario_step_metrics(rep, step) for rep in reps]
        summary = {}
        for metric in BOOTSTRAP_METRICS:
            values = np.asarray([np.nan if row[metric] is None else row[metric] for row in by_rep])
            available = values[np.isfinite(values)]
            complete = len(available) == len(values)
            direction = _METRIC_DIRECTION[metric]
            worst = ((float(np.min(available)) if direction == "higher" else float(np.max(available)))
                     if complete else None)
            summary[metric] = {
                "direction": direction, "replicates": len(values), "available_replicates": len(available),
                "coverage": _safe_ratio(len(available), len(values)),
                "mean": float(np.mean(available)) if complete else None,
                "sample_std": (float(np.std(available, ddof=1)) if complete and len(available) > 1
                               else 0.0 if complete else None),
                "worst_rep": worst,
                "valid_subset_mean": float(np.mean(available)) if len(available) else None,
                "valid_subset_std": (float(np.std(available, ddof=1)) if len(available) > 1
                                     else 0.0 if len(available) == 1 else None),
            }
        result["by_step"][step] = summary
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
    }


def paired_group_bootstrap(candidate_reps, original_reps, seed=20261126):
    """Paired 21-group bootstrap for candidate-minus-original metric deltas."""
    candidate_reps, original_reps = list(candidate_reps), list(original_reps)
    if not candidate_reps or len(candidate_reps) != len(original_reps):
        raise ValueError("candidate and original need the same nonzero replicate count")
    by_step = {}
    final_draw_indices = None
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
        rng = np.random.default_rng(seed)
        draw_indices = np.empty((BOOTSTRAP_REPLICATES, len(rep_pairs), BOOTSTRAP_GROUP_COUNT), dtype=np.int16)
        for rep_index in range(len(rep_pairs)):
            draw_indices[:, rep_index, :] = rng.integers(
                0, BOOTSTRAP_GROUP_COUNT,
                size=(BOOTSTRAP_REPLICATES, BOOTSTRAP_GROUP_COUNT), dtype=np.int16)
        final_draw_indices = draw_indices
        point_left = [row for left, _, _ in rep_pairs for row in left.values()]
        point_right = [row for _, right, _ in rep_pairs for row in right.values()]
        left_point, right_point = _metric_values(point_left), _metric_values(point_right)
        point_delta = {
            metric: (float(left_point[metric] - right_point[metric])
                     if left_point[metric] is not None and right_point[metric] is not None else None)
            for metric in BOOTSTRAP_METRICS
        }
        draws = {metric: np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=np.float64)
                 for metric in BOOTSTRAP_METRICS}
        for sample_index in range(BOOTSTRAP_REPLICATES):
            sampled_left, sampled_right = [], []
            for rep_index, (left, right, groups) in enumerate(rep_pairs):
                chosen = [groups[int(j)] for j in draw_indices[sample_index, rep_index]]
                for group in chosen:
                    sampled_left.extend(row for row in left.values() if row["group"] == group)
                    sampled_right.extend(row for row in right.values() if row["group"] == group)
            left_metrics, right_metrics = _metric_values(sampled_left), _metric_values(sampled_right)
            for metric in BOOTSTRAP_METRICS:
                a, b = left_metrics[metric], right_metrics[metric]
                if a is not None and b is not None:
                    draws[metric][sample_index] = float(a - b)
        metric_result = {}
        for metric in BOOTSTRAP_METRICS:
            valid = draws[metric][np.isfinite(draws[metric])]
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
        "draw_index_sha256": hashlib.sha256(final_draw_indices.astype("<i8").tobytes()).hexdigest(),
        "by_step": by_step,
    }
