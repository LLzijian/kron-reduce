"""Frozen A17 future-only MSE comparisons on caller-supplied arrays.

No labels, instability rules, model calls, file access or population selection.
"""
from __future__ import annotations

import numpy as np

from kron_reduce.evaluation.history_metrics import weighted_cvar
from kron_reduce.evaluation.input_consistency import finite_array

ATOL = 1e-12
FIELDS = ("primary", "q", "v", "horizon1", "horizon3")


def _require_finite_record(record):
    def finite(value):
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return not isinstance(value, (float, np.floating)) or bool(np.isfinite(value))
    if not finite(record):
        error = FloatingPointError("nonfinite derived population/comparison metric")
        error.partial_record = record
        raise error
    return record


def future_event_scores(prediction, truth, clock, target_scale):
    pred = finite_array(prediction, "prediction")
    target = finite_array(truth, "truth")
    time = finite_array(clock, "future clock")
    scale = finite_array(target_scale, "target_scale", (20,))
    if (time.ndim != 1 or not 2 <= len(time) <= 301 or time[0] != 0
            or np.any(np.diff(time) <= 0) or time[-1] > 3 + ATOL
            or pred.shape != (len(time), 22) or target.shape != (len(time), 20)
            or np.any(scale <= 0) or not np.array_equal(pred[0, :20], target[0])):
        raise ValueError("future comparison must preserve the complete original origin/clock")
    error = (pred[1:, :20] - target[1:]) / scale
    if not np.isfinite(error).all():
        raise FloatingPointError("nonfinite normalized future error")
    horizons = {}
    for name, horizon in (("horizon1", 1.), ("horizon3", 3.)):
        mask = time[1:] <= horizon + ATOL
        if not np.any(mask):
            raise ValueError("both original future horizons must contain real targets")
        q = float(np.mean(error[mask, :10] ** 2))
        v = float(np.mean(error[mask, 10:] ** 2))
        horizons[name] = {"q": q, "v": v, "all": .5 * q + .5 * v,
                          "scored_points": int(np.sum(mask))}
    record = {"primary": .5 * (horizons["horizon1"]["all"] + horizons["horizon3"]["all"]),
              "q": .5 * (horizons["horizon1"]["q"] + horizons["horizon3"]["q"]),
              "v": .5 * (horizons["horizon1"]["v"] + horizons["horizon3"]["v"]),
              "horizon1": horizons["horizon1"]["all"],
              "horizon3": horizons["horizon3"]["all"],
              "future_points": len(time) - 1, "horizons": horizons}
    if not all(np.isfinite(record[name]) for name in FIELDS):
        error = FloatingPointError("nonfinite future event MSE")
        error.partial_record = record
        raise error
    return record


def population_scores(records, weights, counts):
    """Keep failures and original denominator; condition only the named 3 s group."""
    n = len(records)
    weight = finite_array(weights, "original weights", (n,))
    count = np.asarray(counts)
    if (not n or count.shape != (n,) or count.dtype.kind not in "iu"
            or np.any(count < 1) or np.any(count > 300) or np.any(weight <= 0)
            or abs(float(np.sum(weight)) - 1.) > ATOL):
        raise ValueError("fixed complete population/weights/counts required")
    present = np.array([row is not None for row in records], dtype=bool)
    for row, expected in zip(records, count, strict=True):
        if row is not None and (row["future_points"] != expected
                                or not all(np.isfinite(row[key]) and row[key] >= 0 for key in FIELDS)):
            raise ValueError("saved event score differs from original population")
    partial = {key: float(sum(weight[i] * row[key] for i, row in enumerate(records) if row is not None))
               for key in FIELDS}
    complete = bool(np.all(present))
    values = np.array([row["primary"] if row is not None else np.nan for row in records])
    groups = {}
    for name, mask in (("under_1s", count < 100), ("from_1s_under_3s", (count >= 100) & (count < 300)),
                       ("full_3s", count == 300)):
        groups[name] = {"events": int(np.sum(mask)), "original_weight": float(np.sum(weight[mask])),
                        "missing_weight": float(np.sum(weight[mask & ~present])),
                        "partial_primary_contribution_same_denominator":
                            float(sum(weight[i] * records[i]["primary"] for i in np.flatnonzero(mask & present)))}
    full = count == 300
    full_mass = float(np.sum(weight[full]))
    conditional = (float(sum(weight[i] * records[i]["horizon3"] for i in np.flatnonzero(full))) / full_mass
                   if full_mass > 0 and bool(np.all(present[full])) else None)
    return _require_finite_record({"complete": complete, "allocated_eligible_events": n,
            "completed_events": int(np.sum(present)), "missing_weight": float(np.sum(weight[~present])),
            "partial_contributions_same_denominator": partial,
            "metrics": partial if complete else None,
            "worst10_cvar": weighted_cvar(values, weight, .1) if complete else None,
            "complete_3s_horizon3_conditional_mse": conditional,
            "duration_groups": groups})


def paired_group(candidate, reference):
    """One seed/step comparison; six independent required groups are not averaged."""
    if type(candidate["complete"]) is not bool or type(reference["complete"]) is not bool:
        raise ValueError("population completeness must be an explicit boolean")
    _require_finite_record(candidate)
    _require_finite_record(reference)
    if not candidate["complete"] or not reference["complete"]:
        return {"complete": False, "pass": False, "reason": "original_population_incomplete",
                "conditions": None, "differences": None}
    c, r = candidate["metrics"], reference["metrics"]
    conditions = {"primary_strictly_lower": c["primary"] < r["primary"] - ATOL}
    differences = {name: c[name] - r[name] for name in FIELDS}
    for name in ("horizon1", "horizon3", "q", "v"):
        conditions[name + "_not_higher"] = c[name] <= r[name] + ATOL
    for name in ("worst10_cvar", "complete_3s_horizon3_conditional_mse"):
        cv, rv = candidate[name], reference[name]
        conditions[name + "_not_higher"] = cv is not None and rv is not None and cv <= rv + ATOL
        differences[name] = None if cv is None or rv is None else cv - rv
    return _require_finite_record({"complete": True, "pass": bool(all(conditions.values())),
                                   "conditions": conditions, "differences": differences, "comparison_atol": ATOL})


def campaign_verdict(V_pass, groups):
    """Exactly six comparisons against each reference, without seed selection."""
    expected = {(reference, seed, step) for reference in ("I", "E")
                for seed in (324, 325, 326) for step in (.005, .0025)}
    if type(V_pass) is not bool or set(groups) != expected:
        raise ValueError("all twelve frozen paired groups are required")
    conditions_expected = {"primary_strictly_lower", "horizon1_not_higher", "horizon3_not_higher",
                           "q_not_higher", "v_not_higher", "worst10_cvar_not_higher",
                           "complete_3s_horizon3_conditional_mse_not_higher"}
    for row in groups.values():
        if (not isinstance(row, dict) or type(row.get("pass")) is not bool
                or type(row.get("complete")) is not bool):
            raise ValueError("paired verdicts must use explicit complete/pass booleans")
        if row["complete"]:
            conditions = row.get("conditions")
            if (not isinstance(conditions, dict) or set(conditions) != conditions_expected
                    or any(type(value) is not bool for value in conditions.values())
                    or row["pass"] != all(conditions.values())):
                raise ValueError("paired verdict disagrees with its seven explicit conditions")
        elif row["pass"]:
            raise ValueError("an incomplete paired group cannot pass")
        _require_finite_record(row)
    GI = V_pass and all(groups[("I", seed, step)]["pass"] for seed in (324, 325, 326) for step in (.005, .0025))
    GE = V_pass and all(groups[("E", seed, step)]["pass"] for seed in (324, 325, 326) for step in (.005, .0025))
    return {"V_pass": V_pass, "G_I": GI, "G_E": GE,
            "scientific_verdict": "ACCEPT" if V_pass and GI and GE else "REJECT",
            "promotion_decision": "KEEP_BASELINE", "new_missed_instability_result_established": False}
