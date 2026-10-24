"""Array-only identity, clock and rule checks for the frozen A15 comparison."""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from kron_reduce.evaluation.history_metrics import hierarchy_weights
from kron_reduce.evaluation.train_mechanism_audit import (
    _row_margin,
    _rule_and_bound,
    _trigger_times,
    audit_event,
)
from kron_reduce.evaluation.train_refinement_audit import (
    _coarse_fine_margin_failures,
    _diagnostic_margin_failures,
)

SEEDS = (324, 325, 326)
STEPS = (.005, .0025)
PARAMETERS = {
    "mass": (10,), "edge_stiffness": (45,), "damping": (10, 10),
    "coupling": (10, 2), "latent_skew": (2, 2), "latent_dissipation": (2, 2),
}
INPUTS = {
    "history": (258, 11, 20), "history_time": (258, 11),
    "constant_power": (258, 10), "equilibrium": (258, 10),
    "mass": (10,), "target_scale": (20,),
}


def finite(value, shape, name):
    array = np.asarray(value)
    if array.dtype != np.dtype("float64") or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: finite float64 array with shape {shape} required")
    return array


@np.errstate(over="raise", invalid="raise", divide="raise")
def refinement(coarse, fine, scale):
    if coarse.shape != fine.shape or coarse.ndim != 2 or coarse.shape[1] != 22:
        raise ValueError("refinement requires equal 22-channel paths")
    if not np.isfinite(coarse).all() or not np.isfinite(fine).all():
        raise ValueError("refinement paths must be finite")
    visible = float(np.max(np.abs((coarse[:, :20] - fine[:, :20]) / scale)))
    hidden = float(np.max(np.linalg.norm(coarse[:, 20:] - fine[:, 20:], axis=1)))
    bound = float(1e-5 * (1 + np.max(np.linalg.norm(fine[:, 20:], axis=1))))
    if not np.isfinite([visible, hidden, bound]).all():
        raise FloatingPointError("nonfinite derived refinement diagnostic")
    return {"visible_max_normalized": visible, "hidden_max_norm": hidden,
            "hidden_bound": bound, "qualified": visible <= 1e-5 and hidden <= bound}


def control_key(seed, step):
    return f"A09/seed-{seed}-h005/paths.npz" if step == .005 else f"A10/seed-{seed}-h0025/paths.npz"


def preflight_prerequisite(decoded, population):
    """Check every cached reset before the reader permits future truth access."""
    rows, weights = population["cache_rows"], np.asarray(population["weights"])
    if (len(rows) != 258 or len(population["allocated_slots"]) != 268
            or len(population["seed_event_slots"]) != 774
            or not np.array_equal(weights, hierarchy_weights(rows))):
        raise ValueError("A15 inherited population or original hierarchy weights differ")
    inputs = decoded["A14/permitted_inputs.npz"]
    if set(inputs) != set(INPUTS):
        raise ValueError("A15 input cache member set differs")
    inputs = {k: finite(inputs[k], shape, k) for k, shape in INPUTS.items()}
    mass, scale, times = inputs["mass"], inputs["target_scale"], inputs["history_time"]
    if (np.any(mass <= 0) or np.any(scale <= 0) or np.any(times[:, 0] >= 0)
            or np.any(times[:, -1] != 0) or np.any(np.diff(times, axis=1) <= 0)):
        raise ValueError("invalid inherited mass, scale or history clock")
    for part in (inputs["history"][:, :, :10], inputs["history"][:, :, 10:]):
        if np.max(np.abs(part @ mass)) > 1e-10:
            raise ValueError("observed history violates mass-weighted COI")
    parameters, resets, controls, baseline_refinement = {}, {}, {}, {}
    common_offsets = None
    for seed in SEEDS:
        raw = decoded[f"A14/static-{seed}.npz"]
        if set(raw) != {"parameters__" + k for k in PARAMETERS}:
            raise ValueError("A15 fixed parameter member set differs")
        parameters[seed] = {k: finite(raw["parameters__" + k], shape, k) for k, shape in PARAMETERS.items()}
        if not np.array_equal(parameters[seed]["mass"], mass):
            raise ValueError("fixed model mass differs from inherited inputs")
        for step in STEPS:
            source = decoded[control_key(seed, step)]
            offset = np.asarray(source["offsets"])
            if (set(source) != {"future", "offsets"} or offset.dtype != np.dtype("int64")
                    or offset.shape != (259,) or offset[0] != 0 or offset[-1] != 72405
                    or np.any(np.diff(offset) < 2) or np.any(np.diff(offset) > 301)):
                raise ValueError("old future offsets differ from registered integer counts")
            if common_offsets is None:
                common_offsets = offset
            elif not np.array_equal(offset, common_offsets):
                raise ValueError("old control offsets disagree")
            controls[seed, step] = finite(source["future"], (72405, 22), "old future")
            pair = []
            for index in range(258):
                saved = decoded[f"A14/seed-{seed}-event-{index:03d}/history-{step}.npz"]
                if set(saved) != {"reset", "baseline_reset"}:
                    raise ValueError("reset source members differ")
                candidate = finite(saved["reset"], (22,), "candidate reset")
                baseline = finite(saved["baseline_reset"], (22,), "baseline reset")
                if (not np.array_equal(candidate[:20], inputs["history"][index, -1])
                        or not np.array_equal(baseline[:20], candidate[:20])
                        or not np.array_equal(controls[seed, step][offset[index]], baseline)):
                    raise ValueError("cached reset does not match history end or old future start")
                pair.append(candidate)
            resets[seed, step] = np.stack(pair)
            resets[seed, step].flags.writeable = False
        baseline_refinement[str(seed)] = [refinement(
            controls[seed, .005][a:b], controls[seed, .0025][a:b], scale)
            for a, b in pairwise(common_offsets)]
    counts = np.diff(common_offsets) - 1
    if int(counts.sum()) != 72147:
        raise ValueError("A15 frozen future point count differs")
    return {"inputs": inputs, "parameters": parameters, "resets": resets,
            "controls": controls, "offsets": common_offsets, "counts": counts,
            "baseline_refinement": baseline_refinement}


def preflight_truth(truth, pre):
    if set(truth) != {"future", "forecast_time", "offsets", "omega_s"}:
        raise ValueError("A15 future truth members differ")
    future = finite(truth["future"], (72405, 20), "truth future")
    clock = finite(truth["forecast_time"], (72405,), "truth forecast_time")
    omega = finite(truth["omega_s"], (258,), "omega_s")
    offsets = np.asarray(truth["offsets"])
    if offsets.dtype != np.dtype("int64") or not np.array_equal(offsets, pre["offsets"]) or np.any(omega <= 0):
        raise ValueError("future truth offsets or omega_s differ")
    counts = []
    for index, (a, b) in enumerate(pairwise(offsets)):
        time = clock[a:b]
        n = b - a - 1
        if (time[0] != 0 or np.any(np.diff(time) <= 0)
                or not np.allclose(time, np.arange(n + 1) * .01, rtol=0, atol=1e-12)
                or not np.array_equal(future[a], pre["inputs"]["history"][index, -1])):
            raise ValueError("truth clock or common observed origin differs")
        steps = {str(h): int(sum(max(1, int(np.ceil(dt / h - 1e-12))) for dt in np.diff(time))) for h in STEPS}
        if steps != {"0.005": 2 * n, "0.0025": 4 * n}:
            raise ValueError("cached clock changes registered RK4 count")
        counts.append({"event_index": index, "future_points": int(n), "rk4_updates": steps})
    return {"qualified": True, "event_clock_counts": counts, "scored_points": 72147,
            "future_truth_frozen_after_resets": True}


@np.errstate(over="raise", invalid="raise", divide="raise")
def rule_audit(history, truth, path, eq, mass, omega_s, scale, history_time, forecast_time):
    value = audit_event(history=history, true_future=truth, pred_future=path[:, :20],
                        equilibrium_delta=eq, mass=mass, omega_s=omega_s, target_scale=scale)
    if history_time.shape != (11,) or forecast_time.shape != (len(truth),):
        raise ValueError("rule reporting clocks must align with the complete path")
    for name, future in (("truth", truth), ("prediction", path[:, :20])):
        margins = _row_margin(np.concatenate((history, future[1:])), eq, mass, omega_s)
        points = np.flatnonzero(margins > 0)
        future_points = points[points >= 11]
        value[f"{name}_first_point_violation"] = {
            "record_index": int(points[0]) if len(points) else None,
            "record_s": float(points[0] * .01) if len(points) else None,
            "forecast_s": float(points[0] * .01 - .1) if len(points) else None,
            "first_future_record_index": int(future_points[0]) if len(future_points) else None,
            "first_future_record_s": float(future_points[0] * .01) if len(future_points) else None,
            "first_future_forecast_s": float(future_points[0] * .01 - .1) if len(future_points) else None,
            "pre_record_onset_unknown": bool(len(points) and points[0] == 0),
        }
    failures = _diagnostic_margin_failures(value)
    value["margin_validity"] = {"qualified": not failures, "failures": failures}
    return value


@np.errstate(over="raise", invalid="raise", divide="raise")
def compare_rule_pair(history, coarse, fine, truth_diagnostic, eq, mass, omega_s, scale,
                      coarse_diagnostic, fine_diagnostic):
    margin = audit_event(history=history, true_future=coarse[:, :20], pred_future=fine[:, :20],
                        equilibrium_delta=eq, mass=mass, omega_s=omega_s, target_scale=scale)
    result = {key: margin[key] for key in ("history_rho", "future_rho", "full_rho")}
    representative = truth_diagnostic["representative_truth_window"]
    start, end = representative["start_index"], representative["end_index"]
    left, right = (np.concatenate((history, p[1:, :20])) for p in (coarse, fine))
    fixed = _rule_and_bound(left[start:end + 1], right[start:end + 1], eq, mass, omega_s, min_end=4)
    for rule in (fixed["truth"], fixed["prediction"]):
        _trigger_times(rule, index_offset=start)
    result["representative_truth_window"] = {
        "start_index": start, "end_index": end,
        "coarse_margin_same_window": fixed["truth"]["rho"],
        "fine_margin_same_window": fixed["prediction"]["rho"], "fixed_five_point_audit": fixed,
    }
    item = {"coarse_fine_rule_margin_comparison": result}
    failures = _coarse_fine_margin_failures(item, coarse_diagnostic, fine_diagnostic)
    coarse_rule, fine_rule = (v["future_rho"]["prediction"] for v in (coarse_diagnostic, fine_diagnostic))
    same = coarse_rule["triggered"] == fine_rule["triggered"] and type(coarse_rule["triggered"]) is bool
    timing = True
    if coarse_rule["triggered"] and fine_rule["triggered"]:
        timing = abs(coarse_rule["first_trigger_end_index"] - fine_rule["first_trigger_end_index"]) <= 2
    return {**item, "same_rule_decision": same, "trigger_within_two_points": timing,
            "margin_failures": failures, "qualified": same and timing and not failures}


def rule_counts(diagnostics):
    result = {"events": len(diagnostics), "truth_positive": 0, "positive_not_retained": 0,
              "truth_not_triggered_prediction_triggered": 0, "retained": 0, "unavailable_windows": 0}
    for row in diagnostics:
        truth, prediction = row["observed_rule_hit"], row["prediction_rule_hit"]
        if type(truth) is not bool or type(prediction) is not bool:
            result["unavailable_windows"] += 1
            continue
        result["truth_positive"] += int(truth)
        result["retained"] += int(truth and prediction)
        result["positive_not_retained"] += int(truth and not prediction)
        result["truth_not_triggered_prediction_triggered"] += int(not truth and prediction)
    return result


def compare_rule_counts(candidate_diagnostics, baseline_diagnostics):
    if len(candidate_diagnostics) != len(baseline_diagnostics):
        raise ValueError("rule comparisons require the same complete population")
    for left, right in zip(candidate_diagnostics, baseline_diagnostics, strict=True):
        if left["observed_rule_hit"] is not right["observed_rule_hit"]:
            raise ValueError("rule comparison changed the common truth window population")
    candidate, baseline = rule_counts(candidate_diagnostics), rule_counts(baseline_diagnostics)
    qualified = (candidate["events"] == 258 and candidate["unavailable_windows"] == 0
                 and baseline["unavailable_windows"] == 0
                 and all(candidate[k] <= baseline[k] for k in (
                     "positive_not_retained", "truth_not_triggered_prediction_triggered")))
    return {"candidate": candidate, "baseline": baseline, "D_pass": qualified,
            "strictly_more_retained": candidate["retained"] > baseline["retained"],
            "label_classification": False, "FSR_claimed": False}
