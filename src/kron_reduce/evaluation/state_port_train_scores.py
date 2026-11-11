"""Frozen train-only score gates for the paired state-port study."""
from __future__ import annotations

import math

import numpy as np

STEPS = ("0.005", "0.0025")
METHODS = ("baseline", "constant", "state")
PROTECTIONS = ("horizon1", "horizon3", "q", "v", "cvar10", "complete3")


def descriptive(paths, truth, counts, scale, rows):
    """Summarize saved paths: event loss, pooled normalized MSE, and physical RMSE.

    ``paths`` is nested as seed -> step -> method -> (events, time, state).
    ``truth`` is (events, time, 20); ``rows`` is the aligned frozen roster.
    """
    truth = np.asarray(truth, dtype=np.float64)
    counts = np.asarray(counts)
    scale = np.asarray(scale, dtype=np.float64)
    n = len(rows)
    if (truth.ndim != 3 or truth.shape[0] != n or truth.shape[2] != 20
            or counts.shape != (n,) or counts.dtype.kind not in "iu"
            or np.any(counts < 1) or np.any(counts > truth.shape[1] - 1)
            or scale.shape != (20,) or not np.isfinite(truth).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("truth, counts, scale and population are not aligned finite arrays")
    result = {}
    for seed, by_step in paths.items():
        result[str(seed)] = {}
        if set(by_step) != set(STEPS):
            raise ValueError("both fixed steps are required")
        for step, by_method in by_step.items():
            if set(by_method) != set(METHODS):
                raise ValueError("baseline, constant and state paths are required")
            result[str(seed)][step] = {}
            for method, values in by_method.items():
                path = np.asarray(values, dtype=np.float64)
                if (path.ndim != 3 or path.shape[0] != n or path.shape[1] < int(counts.max()) + 1
                        or path.shape[2] < 20 or not np.isfinite(path).all()):
                    raise ValueError("saved path shape or finiteness differs from the frozen population")
                event_ell, norm_sum, q_sum, v_sum = [], 0.0, 0.0, 0.0
                point_count = 0
                for index, count in enumerate(counts):
                    count = int(count)
                    error = path[index, 1:count + 1, :20] - truth[index, 1:count + 1]
                    normalized_sq = (error / scale) ** 2
                    per_time = normalized_sq.mean(axis=1)
                    h1 = float(per_time[:min(count, 100)].mean())
                    h3 = float(per_time[:min(count, 300)].mean())
                    event_ell.append(.5 * (h1 + h3))
                    norm_sum += float(normalized_sq.sum())
                    q_sum += float(np.square(error[:, :10]).sum())
                    v_sum += float(np.square(error[:, 10:20]).sum())
                    point_count += count
                result[str(seed)][step][method] = {
                    "event_ell_mean": float(np.mean(event_ell)),
                    "event_ell_median": float(np.median(event_ell)),
                    "event_ell": event_ell,
                    "pooled_normalized_mse": norm_sum / (20 * point_count),
                    "q_rmse_rad": math.sqrt(q_sum / (10 * point_count)),
                    "v_rmse_rad_per_s": math.sqrt(v_sum / (10 * point_count)),
                }
    return result


def paired_line_summary(groups, rows):
    """Summarize event-loss ``ell_R - ell_state`` by source and fault line.

    The line-based standard error follows the fixed source-stratified formula in
    the train protocol. It is descriptive and does not define an acceptance gate.
    """
    seeds = tuple(str(seed) for seed in (324, 325, 326))
    if set(groups) != set(seeds):
        raise ValueError("all and only the fixed seeds are required")
    if not rows:
        raise ValueError("the frozen event roster is empty")
    strata = []
    for row in rows:
        source, line = row.get("source_dataset"), row.get("fault_line")
        if not isinstance(source, str) or not source or not isinstance(line, str) or not line:
            raise ValueError("each roster row needs source_dataset and fault_line")
        strata.append((source, line))
    sources = sorted({source for source, _ in strata})
    line_keys = {source: sorted({line for src, line in strata if src == source})
                 for source in sources}
    if any(len(lines) < 2 for lines in line_keys.values()):
        estimable = False
    else:
        estimable = True

    result = {}
    for seed in seeds:
        if set(groups[seed]) != set(STEPS):
            raise ValueError("both fixed steps are required")
        result[seed] = {}
        for step in STEPS:
            if set(groups[seed][step]) != set(METHODS):
                raise ValueError("baseline, constant and state score groups are required")
            state = groups[seed][step]["state"].get("event_main_mse")
            if state is None or len(state) != len(rows):
                raise ValueError("state event losses do not match the roster")
            result[seed][step] = {}
            for reference in ("baseline", "constant"):
                ref = groups[seed][step][reference].get("event_main_mse")
                if ref is None or len(ref) != len(rows):
                    raise ValueError("reference event losses do not match the roster")
                differences = [_number(a, "reference event loss") - _number(b, "state event loss")
                               for a, b in zip(ref, state, strict=True)]
                line_means = {}
                for source in sources:
                    for line in line_keys[source]:
                        values = [d for d, key in zip(differences, strata, strict=True)
                                  if key == (source, line)]
                        if not values:
                            raise ValueError("source-line stratum has no events")
                        line_means[f"{source}/{line}"] = float(math.fsum(values) / len(values))
                source_means = {
                    source: float(math.fsum(line_means[f"{source}/{line}"] for line in line_keys[source])
                                  / len(line_keys[source]))
                    for source in sources
                }
                if estimable:
                    variance = math.fsum(
                        math.fsum((line_means[f"{source}/{line}"] - source_means[source]) ** 2
                                  for line in line_keys[source])
                        / (len(sources) ** 2 * len(line_keys[source]) * (len(line_keys[source]) - 1))
                        for source in sources
                    )
                    se = math.sqrt(variance)
                else:
                    se = None
                result[seed][step][reference] = {
                    "line_mean_ell_reference_minus_state": line_means,
                    "source_mean": source_means,
                    "balanced_mean": float(math.fsum(source_means.values()) / len(sources)),
                    "se": se,
                    "se_estimable": estimable,
                }
    return result


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _rule_counts(truth, prediction):
    if len(truth) != len(prediction):
        raise ValueError("unaligned rule identities")
    unknown = sum(type(t) is not bool or type(p) is not bool
                  for t, p in zip(truth, prediction, strict=True))
    return {
        "miss": sum(t is True and p is not True for t, p in zip(truth, prediction, strict=True)),
        "extra": sum(t is False and p is True for t, p in zip(truth, prediction, strict=True)),
        "unknown": unknown,
        "rule_complete": unknown == 0,
    }


def verdict(groups, *, valid: bool, expected_seeds=(324, 325, 326)):
    """Apply V, two paired trajectory gates, and two rule-protection gates."""
    if type(valid) is not bool:
        raise ValueError("valid must be a boolean")
    seeds = tuple(str(seed) for seed in expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or set(groups) != set(seeds):
        raise ValueError("all and only the fixed seeds are required")

    population = None
    reference_truth = None
    arrays_valid = True
    for seed in seeds:
        if set(groups[seed]) != set(STEPS):
            raise ValueError("both fixed steps are required")
        for step in STEPS:
            if set(groups[seed][step]) != set(METHODS):
                raise ValueError("baseline, constant and state groups are required")
            for row in groups[seed][step].values():
                identity = [row[key] for key in ("event_ids", "weights", "counts")]
                if population is None:
                    population = identity
                elif identity != population:
                    raise ValueError("group population, weights or order mismatch")
                if reference_truth is None:
                    reference_truth = row["truth_triggered"]
                elif row["truth_triggered"] != reference_truth:
                    raise ValueError("truth rule identities differ across methods or seeds")
                ids, weights, counts = identity
                if (not ids or any(not isinstance(event_id, str) or not event_id for event_id in ids)
                        or len(set(ids)) != len(ids) or len(weights) != len(ids)
                        or len(counts) != len(ids)):
                    raise ValueError("invalid event population identity")
                if (any(not isinstance(weight, (int, float)) or isinstance(weight, bool)
                        or not math.isfinite(weight) or weight <= 0 for weight in weights)
                        or abs(math.fsum(weights) - 1) > 1e-12):
                    raise ValueError("event weights must be positive and sum to one")
                if any(type(count) is not int or not 1 <= count <= 300 for count in counts):
                    raise ValueError("future counts must be integers in 1..300")
                for key in ("M", *PROTECTIONS):
                    if key == "complete3" and row[key] is None:
                        continue
                    _number(row[key], key)
                if any(type(row[key]) is not int or row[key] < 0
                       for key in ("miss", "extra", "unknown")):
                    raise ValueError("rule counts must be nonnegative integers")
                counts = _rule_counts(row["truth_triggered"], row["prediction_triggered"])
                if (len(row["truth_triggered"]) != len(row["event_ids"])
                        or counts != {key: row[key] for key in
                                      ("miss", "extra", "unknown", "rule_complete")}):
                    raise ValueError("rule counts disagree with retained event identities")
                if type(row["rule_complete"]) is not bool:
                    raise ValueError("rule_complete must be boolean")
                arrays_valid &= row.get("array_qualified") is True

    comparisons = {}
    for reference in ("baseline", "constant"):
        comparisons[reference] = {}
        for seed in seeds:
            candidate = [groups[seed][step]["state"] for step in STEPS]
            reference_rows = [groups[seed][step][reference] for step in STEPS]
            proxy = max(abs(candidate[0]["M"] - candidate[1]["M"]),
                        abs(reference_rows[0]["M"] - reference_rows[1]["M"]))
            minimum_gain = min(ref["M"] - cand["M"]
                               for cand, ref in zip(candidate, reference_rows, strict=True))
            required = max(1e-12, 10 * proxy)
            per_step = {}
            for step, cand, ref in zip(STEPS, candidate, reference_rows, strict=True):
                protections = {
                    key: bool(cand[key] is not None and ref[key] is not None
                              and cand[key] <= ref[key] + 1e-12)
                    for key in PROTECTIONS
                }
                one_percent = cand["M"] <= .99 * ref["M"]
                per_step[step] = {
                    "candidate_M": cand["M"], "reference_M": ref["M"],
                    "absolute_improvement": ref["M"] - cand["M"],
                    "relative_improvement": (1 - cand["M"] / ref["M"] if ref["M"] else None),
                    "one_percent": bool(one_percent), "protections": protections,
                    "qualified": bool(one_percent and all(protections.values())),
                }
            comparisons[reference][seed] = {
                "refinement_proxy": proxy, "required_numerical_effect": required,
                "minimum_improvement": minimum_gain,
                "effect_resolved": minimum_gain > required,
                "steps": per_step,
                "qualified": bool(minimum_gain > required and
                                  all(row["qualified"] for row in per_step.values())),
            }

    rule_protection = {}
    for seed in seeds:
        rule_protection[seed] = {}
        for step in STEPS:
            rows = {method: groups[seed][step][method] for method in METHODS}
            complete = all(row["rule_complete"] and row["unknown"] == 0 for row in rows.values())
            state = rows["state"]
            baseline, constant = rows["baseline"], rows["constant"]
            against_baseline = bool(state["miss"] <= baseline["miss"]
                                    and state["extra"] <= baseline["extra"])
            against_constant = bool(state["miss"] <= constant["miss"]
                                    and state["extra"] <= constant["extra"])
            rule_protection[seed][step] = {
                "rule_complete": complete,
                "state_miss": state["miss"], "baseline_miss": baseline["miss"],
                "constant_miss": constant["miss"],
                "state_extra": state["extra"], "baseline_extra": baseline["extra"],
                "constant_extra": constant["extra"],
                "against_baseline": bool(complete and against_baseline),
                "against_constant": bool(complete and against_constant),
                "qualified": bool(complete and against_baseline and against_constant),
            }

    gates = {
        "V": bool(valid and arrays_valid),
        "G_baseline": all(row["qualified"] for row in comparisons["baseline"].values()),
        "G_constant": all(row["qualified"] for row in comparisons["constant"].values()),
        "D_baseline": all(row["against_baseline"] for per_seed in rule_protection.values()
                          for row in per_seed.values()),
        "D_constant": all(row["against_constant"] for per_seed in rule_protection.values()
                          for row in per_seed.values()),
    }
    return {
        **gates,
        "comparisons": comparisons,
        "rule_protection": rule_protection,
        "scientific_verdict": "ACCEPT" if all(gates.values()) else "REJECT",
        "promotion_decision": "KEEP_BASELINE",
        "selected_model": None,
        "certified": False,
        "new_training_authorized": False,
        "val_evaluation_authorized": False,
    }


__all__ = ["descriptive", "paired_line_summary", "verdict"]
