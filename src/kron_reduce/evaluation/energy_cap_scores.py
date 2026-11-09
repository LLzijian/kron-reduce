"""Pure-array scoring and frozen gates for the encoder energy-cap comparison.

Callers supply already authorized saved paths. No file, model, solver, workflow,
or trajectory-generation operation is invoked here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .encoder_energy_cap import trajectory_horizon_metrics
from .history_metrics import event_losses, weighted_cvar
from .train_mechanism_audit import _row_margin, _rule_and_bound, _trigger_times, audit_event
from .train_refinement_audit import (
    _coarse_fine_margin_failures,
    _diagnostic_margin_failures,
)

STEPS = ("0.005", "0.0025")
METHODS = ("E", "I", "C")
PROTECTIONS = ("horizon1", "horizon3", "q", "v", "cvar10", "complete3")


def _array(value, name, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with the required shape")
    return array


def _finite_tree(value):
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("nonfinite derived score")


def _number(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value)) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _rule_counts(truth, prediction):
    if len(truth) != len(prediction):
        raise ValueError("unaligned rule identities")
    unknown = sum(type(t) is not bool or type(p) is not bool
                  for t, p in zip(truth, prediction, strict=True))
    return {
        "miss": sum(t is True and p is not True for t, p in zip(truth, prediction, strict=True)),
        "extra": sum(t is False and p is True for t, p in zip(truth, prediction, strict=True)),
        "unknown": unknown, "rule_complete": unknown == 0,
    }


def score_history(observed, prediction, scale):
    """A14 history MSE over the ten noninitial points, without a benefit gate."""
    observed = _array(observed, "observed history", (11, 20))
    prediction = _array(prediction, "predicted history", (11, 22))
    scale = _array(scale, "scale", (20,))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        losses = event_losses(prediction[None, :, :], observed[None, :, :], scale)
    result = {key: float(value[0]) for key, value in losses.items()}
    _finite_tree(result)
    return result


def _first_violations(history, future, equilibrium, mass, omega_s):
    path = np.concatenate((history, future[1:]), axis=0)
    margins = _row_margin(path, equilibrium, mass, omega_s)
    result = {}
    for name, start in (("full", 0), ("future", 11)):
        hits = np.flatnonzero(margins[start:] > 0)
        index = int(hits[0] + start) if len(hits) else None
        result[name] = {
            "record_index": index,
            "record_time_s": None if index is None else index * .01,
            "forecast_time_s": None if index is None else index * .01 - .1,
            "pre_record_onset_unknown": index == 0,
        }
    return result


def _compare_truth_windows(history, coarse, fine, diagnostic, equilibrium, mass, omega_s):
    """Compare on the original truth's window, never the coarse path's chosen window."""
    left = np.concatenate((history, coarse[1:]), axis=0)
    right = np.concatenate((history, fine[1:]), axis=0)
    result = {}
    for name, lhs, rhs, min_end in (
        ("history_rho", left[:11], right[:11], 0),
        ("future_rho", left, right, 11),
        ("full_rho", left, right, 0),
    ):
        bound = _rule_and_bound(lhs, rhs, equilibrium, mass, omega_s, min_end=min_end)
        for rule in (bound["truth"], bound["prediction"]):
            _trigger_times(rule)
        result[name] = bound
    window = diagnostic["representative_truth_window"]
    start, end = window["start_index"], window["end_index"]
    fixed = _rule_and_bound(
        left[start:end + 1], right[start:end + 1], equilibrium, mass, omega_s, min_end=4,
    )
    for rule in (fixed["truth"], fixed["prediction"]):
        _trigger_times(rule, index_offset=start)
    result["representative_truth_window"] = {
        "start_index": start, "end_index": end,
        "coarse_margin_same_window": fixed["truth"]["rho"],
        "fine_margin_same_window": fixed["prediction"]["rho"],
        "fixed_five_point_audit": fixed,
    }
    return result


def score_pair(*, history, truth, coarse, fine, equilibrium, mass, omega_s, scale, clock):
    """Score one event/method's two future paths and their array refinement checks.

    ``truth`` has 20 channels; ``coarse`` and ``fine`` have 22. Paths include
    the common visible forecast origin. Physical/RHS checks belong to the caller.
    """
    truth = _array(truth, "truth")
    if truth.ndim != 2 or truth.shape[1] != 20 or not 2 <= len(truth) <= 301:
        raise ValueError("truth must have 20 channels and 1..300 scored future points")
    history = _array(history, "history", (11, 20))
    coarse = _array(coarse, "coarse", (len(truth), 22))
    fine = _array(fine, "fine", coarse.shape)
    scale = _array(scale, "scale", (20,))
    equilibrium = _array(equilibrium, "equilibrium", (10,))
    mass = _array(mass, "mass", (10,))
    clock = _array(clock, "clock", (len(truth),))
    if not np.array_equal(clock, np.arange(len(truth)) * .01):
        raise ValueError("clock must equal the frozen 10 ms forecast grid elementwise")
    if np.any(scale <= 0) or np.any(mass <= 0):
        raise ValueError("scale and mass must be positive")
    kwargs = {"history": history, "equilibrium_delta": equilibrium,
              "mass": mass, "omega_s": omega_s, "target_scale": scale}
    records = {}
    failures = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for step, path in zip(STEPS, (coarse, fine), strict=True):
            diagnostic = audit_event(true_future=truth, pred_future=path[:, :20], **kwargs)
            diagnostic["first_joint_violation"] = {
                "truth": _first_violations(history, truth, equilibrium, mass, omega_s),
                "prediction": _first_violations(
                    history, path[:, :20], equilibrium, mass, omega_s,
                ),
            }
            failures.extend(_diagnostic_margin_failures(diagnostic, step))
            records[step] = {
                "trajectory": trajectory_horizon_metrics(clock, truth, path, scale),
                "audit": diagnostic,
            }
        comparison = _compare_truth_windows(
            history, coarse[:, :20], fine[:, :20], records[STEPS[0]]["audit"],
            equilibrium, mass, omega_s,
        )
        visible = float(np.max(np.abs((coarse[:, :20] - fine[:, :20]) / scale)))
        hidden = float(np.max(np.linalg.norm(coarse[:, 20:] - fine[:, 20:], axis=1)))
        hidden_bound = 1e-5 * (1 + float(np.max(np.linalg.norm(fine[:, 20:], axis=1))))
    a, b = (records[step]["audit"] for step in STEPS)
    refinement = {
        "visible_max_normalized": visible, "hidden_max_norm": hidden,
        "hidden_bound": hidden_bound,
        "visible_qualified": visible <= 1e-5, "hidden_qualified": hidden <= hidden_bound,
        "coarse_fine_rule_margin_comparison": comparison,
    }
    failures.extend(_coarse_fine_margin_failures(refinement, a, b))
    ar, br = (d["future_rho"]["prediction"] for d in (a, b))
    same_trigger = type(ar["triggered"]) is bool and ar["triggered"] is br["triggered"]
    time_ok = bool(same_trigger)
    if same_trigger and ar["triggered"]:
        time_ok = abs(ar["first_trigger_end_index"] - br["first_trigger_end_index"]) <= 2
    refinement.update(
        trigger_consistent=bool(same_trigger), trigger_within_two_points=bool(time_ok),
        margin_failures=failures,
        qualified=bool(visible <= 1e-5 and hidden <= hidden_bound
                       and same_trigger and time_ok and not failures),
    )
    for record in records.values():
        record["array_qualified"] = refinement["qualified"]
    result = {"steps": records, "refinement": refinement, "future_points": len(truth) - 1}
    _finite_tree(result)
    return result


def aggregate_group(records, weights, counts, *, event_ids, expected_count=258):
    """Aggregate one seed/step/method, preserving every expected event and weight."""
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    if (len(records) != expected_count or len(event_ids) != expected_count
            or len(counts) != expected_count):
        raise ValueError("the complete expected population is required")
    ids = list(event_ids)
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("event identities must be nonempty, unique and aligned")
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
           or not 1 <= n <= 300 for n in counts):
        raise ValueError("counts must be integers in 1..300")
    counts = np.asarray(counts, dtype=int)
    weights = _array(weights, "weights", (expected_count,))
    if np.any(weights <= 0) or abs(math.fsum(weights) - 1) > 1e-12:
        raise ValueError("original weights must be positive and sum to one")
    values = {key: [] for key in ("M", "horizon1", "horizon3", "q", "v")}
    truth_hits, prediction_hits = [], []
    history_records = []
    for i, record in enumerate(records):
        trajectory, diagnostic = record["trajectory"], record["audit"]
        if (trajectory["available_future_point_count"] != counts[i]
                or diagnostic["future_points"] != counts[i]):
            raise ValueError("saved score and declared future point count disagree")
        horizons = trajectory["horizons"]
        for h in (1, 3):
            if horizons[str(h)]["point_count"] != min(int(counts[i]), 100 * h):
                raise ValueError("horizon score has a different observation population")
        first, third = (horizons[str(h)]["normalized_mse"] for h in (1, 3))
        for row in (first, third):
            for key in ("all", "q", "v"):
                _number(row[key], key)
            if not math.isclose(row["all"], .5 * (row["q"] + row["v"]),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("channel mean mismatch")
        scalar = {"M": .5 * (first["all"] + third["all"]),
                  "horizon1": first["all"], "horizon3": third["all"],
                  "q": .5 * (first["q"] + third["q"]),
                  "v": .5 * (first["v"] + third["v"])}
        if not math.isclose(scalar["M"], _number(diagnostic["event_mse"], "event_mse"),
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("trajectory score disagrees with the independent audit score")
        for key, value in scalar.items():
            values[key].append(value)
        truth_hits.append(diagnostic["future_rho"]["truth"]["triggered"])
        prediction_hits.append(diagnostic["future_rho"]["prediction"]["triggered"])
        if "history" in record:
            history_records.append(record["history"])
    arrays = {key: np.asarray(value) for key, value in values.items()}
    result = {key: float(weights @ value) for key, value in arrays.items()}
    complete = counts == 300
    complete_mass = float(np.sum(weights[complete]))
    result.update(
        cvar10=weighted_cvar(arrays["M"], weights),
        complete3=(float(weights[complete] @ arrays["horizon3"][complete]) / complete_mass
                   if complete_mass else None),
        complete3_count=int(np.count_nonzero(complete)), complete3_weight_mass=complete_mass,
        event_count=expected_count, event_ids=ids, weights=weights.tolist(), counts=counts.tolist(),
        weight_sum=float(math.fsum(weights)), event_main_mse=values["M"],
        truth_triggered=truth_hits, prediction_triggered=prediction_hits,
        array_qualified=all(record.get("array_qualified") is True for record in records),
    )
    result.update(_rule_counts(truth_hits, prediction_hits))
    result["nrmse"] = math.sqrt(result["M"])
    result["duration_groups"] = {}
    for name, mask in (("lt1s", counts < 100), ("1s_to_lt3s", (counts >= 100) & (counts < 300)),
                       ("complete3s", complete)):
        result["duration_groups"][name] = {
            "event_count": int(np.count_nonzero(mask)),
            "weight_mass": float(np.sum(weights[mask])),
            "main_mse_contribution": float(weights[mask] @ arrays["M"][mask]),
        }
    result["history"] = None
    if history_records:
        if len(history_records) != expected_count:
            raise ValueError("history scores must be present for the whole population or absent")
        history_values = {
            key: np.array([_number(row[key], f"history {key}") for row in history_records])
            for key in ("all", "q", "v")
        }
        result["history"] = {key: float(weights @ row) for key, row in history_values.items()}
        result["history"]["cvar10"] = weighted_cvar(history_values["all"], weights)
    _finite_tree(result)
    return result


def verdict(groups, *, valid, mechanism_exercised, expected_seeds=(324, 325, 326)):
    """Frozen V/GE/GI/D decision; caller supplies all model/physics/source validity."""
    if type(valid) is not bool or type(mechanism_exercised) is not bool:
        raise ValueError("valid and mechanism_exercised must be booleans")
    seeds = tuple(str(seed) for seed in expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or set(groups) != set(seeds):
        raise ValueError("all and only the fixed seeds are required")
    reference_population = None
    array_valid = True
    for seed in seeds:
        if set(groups[seed]) != set(STEPS):
            raise ValueError("both fixed steps are required")
        for step in STEPS:
            if set(groups[seed][step]) != set(METHODS):
                raise ValueError("E, I and C must all be retained")
            for row in groups[seed][step].values():
                population = [row[key] for key in ("event_ids", "weights", "counts")]
                if reference_population is None:
                    reference_population = population
                elif population != reference_population:
                    raise ValueError("group population/weights/order mismatch")
                if row["truth_triggered"] != groups[seeds[0]][STEPS[0]]["E"]["truth_triggered"]:
                    raise ValueError("truth rule identities differ across methods or seeds")
                for key in ("M", *PROTECTIONS):
                    if key != "complete3" or row[key] is not None:
                        _number(row[key], key)
                if any(type(row[key]) is not int or row[key] < 0
                       for key in ("miss", "extra", "unknown")):
                    raise ValueError("rule counts must be nonnegative integers")
                if (len(row["truth_triggered"]) != len(row["event_ids"])
                        or _rule_counts(row["truth_triggered"], row["prediction_triggered"])
                        != {key: row[key] for key in ("miss", "extra", "unknown", "rule_complete")}):
                    raise ValueError("rule counts disagree with retained event identities")
                array_valid &= row.get("array_qualified") is True
    comparisons = {}
    for reference in ("E", "I"):
        comparisons[reference] = {}
        for seed in seeds:
            c = [groups[seed][step]["C"] for step in STEPS]
            r = [groups[seed][step][reference] for step in STEPS]
            proxy = max(abs(c[0]["M"] - c[1]["M"]), abs(r[0]["M"] - r[1]["M"]))
            minimum_gain = min(rr["M"] - cc["M"] for cc, rr in zip(c, r))
            required = max(1e-12, 10 * proxy)
            per_step = {}
            for step, cc, rr in zip(STEPS, c, r, strict=True):
                protections = {
                    key: bool(cc[key] is not None and rr[key] is not None
                              and cc[key] <= rr[key] + 1e-12) for key in PROTECTIONS
                }
                one_percent = cc["M"] <= .99 * rr["M"]
                per_step[step] = {
                    "candidate_M": cc["M"], "reference_M": rr["M"],
                    "absolute_improvement": rr["M"] - cc["M"],
                    "relative_improvement": (1 - cc["M"] / rr["M"] if rr["M"] else None),
                    "one_percent": bool(one_percent), "protections": protections,
                    "qualified": bool(one_percent and all(protections.values())),
                }
            comparisons[reference][seed] = {
                "refinement_proxy": proxy, "required_numerical_effect": required,
                "minimum_improvement": minimum_gain, "effect_resolved": minimum_gain > required,
                "steps": per_step,
                "qualified": bool(minimum_gain > required
                                  and all(row["qualified"] for row in per_step.values())),
            }
    protection = {}
    for seed in seeds:
        protection[seed] = {}
        for step in STEPS:
            c, e = (groups[seed][step][method] for method in ("C", "E"))
            complete = all(groups[seed][step][method]["rule_complete"] is True
                           and groups[seed][step][method]["unknown"] == 0 for method in METHODS)
            protection[seed][step] = {
                "rule_complete": complete, "candidate_miss": c["miss"], "encoder_miss": e["miss"],
                "candidate_extra": c["extra"], "encoder_extra": e["extra"],
                "qualified": bool(complete and c["miss"] <= e["miss"] and c["extra"] <= e["extra"]),
            }
    ge = all(row["qualified"] for row in comparisons["E"].values())
    gi = all(row["qualified"] for row in comparisons["I"].values())
    d = all(row["qualified"] for per_seed in protection.values() for row in per_seed.values())
    v = bool(valid and array_valid)
    result = {
        "V": v, "mechanism_exercised": mechanism_exercised, "G_E": ge, "G_I": gi, "D": d,
        "comparisons": comparisons, "rule_protection": protection,
        "scientific_verdict": "ACCEPT" if v and mechanism_exercised and ge and d else "REJECT",
        "promotion_decision": "KEEP_BASELINE", "selected_model": None,
        "certified": False, "new_training_authorized": False, "val_evaluation_authorized": False,
    }
    _finite_tree(result)
    return result


__all__ = ["aggregate_group", "score_history", "score_pair", "verdict"]
