"""Pure-array metrics for A15 saved future-transfer trajectories.

This module consumes arrays already produced elsewhere. It performs no file
I/O, model construction, prediction, or integration.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

DEFAULT_ATOL = 1e-12


class FutureMetricsError(ValueError):
    """Input arrays cannot support a complete, aligned future comparison."""


def hierarchy_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Equal source → source-local line → event mass, preserving row order."""
    if not rows:
        raise FutureMetricsError("population rows must be nonempty")
    pairs: list[tuple[str, str]] = []
    sources: set[str] = set()
    lines_by_source: dict[str, set[str]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    event_ids: set[str] = set()
    for row in rows:
        source, line, event_id = (row.get("source_dataset"), row.get("fault_line"),
                                  row.get("event_id"))
        if not all(isinstance(value, str) and value for value in (source, line, event_id)):
            raise FutureMetricsError("each row requires source_dataset, fault_line, and event_id")
        if event_id in event_ids:
            raise FutureMetricsError(f"duplicate event_id: {event_id}")
        event_ids.add(event_id)
        pair = (source, line)
        pairs.append(pair)
        sources.add(source)
        lines_by_source.setdefault(source, set()).add(line)
        counts[pair] += 1
    weights = np.asarray([
        1.0 / len(sources) / len(lines_by_source[source]) / counts[source, line]
        for source, line in pairs
    ], dtype=np.float64)
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise FutureMetricsError("hierarchical weights must be finite and sum to one")
    return weights


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FutureMetricsError(f"{name} must be a numeric array") from error
    if (ndim is not None and array.ndim != ndim) or not np.isfinite(array).all():
        raise FutureMetricsError(f"{name} must be finite" + (f" with ndim={ndim}" if ndim else ""))
    return array


def _validate_offsets(offsets: Any, n_events: int, total_rows: int) -> np.ndarray:
    raw = np.asarray(offsets)
    if raw.ndim != 1 or len(raw) != n_events + 1 or raw.dtype.kind not in "iu":
        raise FutureMetricsError("offsets must be an integer vector of length N+1")
    if raw.dtype.kind == "b":
        raise FutureMetricsError("boolean offsets are invalid")
    result = raw.astype(np.int64, copy=False)
    if result[0] != 0 or result[-1] != total_rows or np.any(np.diff(result) < 2):
        raise FutureMetricsError("offsets must partition every event into at least t0 and one future row")
    return result


def weighted_cvar(losses: Any, weights: Any, tail_mass: float = 0.1) -> float:
    """Worst-tail weighted mean with stable row-order ties and fractional mass."""
    value = _finite_array(losses, "losses", ndim=1)
    weight = _finite_array(weights, "weights", ndim=1)
    if value.size == 0 or value.shape != weight.shape or np.any(value < 0) or np.any(weight <= 0):
        raise FutureMetricsError("losses and positive weights must be nonempty aligned vectors")
    if isinstance(tail_mass, bool) or not math.isfinite(tail_mass) or not 0 < tail_mass <= 1:
        raise FutureMetricsError("tail_mass must be in (0, 1]")
    if not np.isclose(float(weight.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise FutureMetricsError("weights must sum to one; subset weights are not renormalized")
    remaining = float(tail_mass)
    total = 0.0
    for index in np.argsort(-value, kind="stable"):
        take = min(float(weight[index]), remaining)
        total += take * float(value[index])
        remaining -= take
        if remaining <= 8 * np.finfo(np.float64).eps * tail_mass:
            break
    if remaining > 8 * np.finfo(np.float64).eps * tail_mass:
        raise FutureMetricsError("weights do not cover the requested tail mass")
    return total / tail_mass


def _channel_mse(error: np.ndarray) -> dict[str, float]:
    q = float(np.mean(np.square(error[:, :10])))
    v = float(np.mean(np.square(error[:, 10:20])))
    result = {"q": q, "v": v, "all": 0.5 * (q + v)}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite derived channel MSE")
    return result


def _event_record(path: np.ndarray, observed: np.ndarray, scale: np.ndarray,
                  times: np.ndarray, event_index: int) -> dict[str, Any]:
    if times[0] != 0.0 or np.any(np.diff(times) <= 0):
        raise FutureMetricsError(f"event {event_index} times must start at zero and strictly increase")
    expected_clock = np.arange(len(times), dtype=np.float64) * 0.01
    if not np.allclose(times, expected_clock, rtol=0.0, atol=1e-12):
        raise FutureMetricsError(f"event {event_index} forecast times do not match the 0.01s output grid")
    if not np.array_equal(path[0, :20], observed[0]):
        raise FutureMetricsError(f"event {event_index} predicted t0 does not match observed origin")
    scored_error = (path[1:, :20] - observed[1:]) / scale[None, :]
    scored_times = times[1:]
    horizon_mse: dict[str, dict[str, float]] = {}
    horizon_counts: dict[str, int] = {}
    for name, horizon in (("1.0", 1.0), ("3.0", 3.0)):
        mask = scored_times <= horizon + 1e-12
        if not np.any(mask):
            raise FutureMetricsError(f"event {event_index} has no scored points in {name}s")
        horizon_mse[name] = _channel_mse(scored_error[mask])
        horizon_counts[name] = int(np.count_nonzero(mask))
    primary = {
        component: 0.5 * (horizon_mse["1.0"][component] + horizon_mse["3.0"][component])
        for component in ("all", "q", "v")
    }
    if not all(math.isfinite(value) for value in primary.values()):
        raise FloatingPointError("nonfinite derived event primary MSE")
    duration = float(times[-1])
    n3 = horizon_counts["3.0"]
    duration_group = "under_1s" if n3 < 100 else (
        "from_1s_to_under_3s" if n3 < 300 else "complete_3s"
    )
    return {"event_index": event_index, "duration_s": duration,
            "duration_group": duration_group, "horizon_point_counts": horizon_counts,
            "horizon_mse": horizon_mse, "primary_mse": primary}


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    return float(weights @ values)


def _duration_summaries(records: Sequence[Mapping[str, Any]], weights: np.ndarray,
                        component: str) -> dict[str, dict[str, float | int | None]]:
    names = ("under_1s", "from_1s_to_under_3s", "complete_3s")
    output = {}
    values = np.asarray([row["primary_mse"][component] for row in records], dtype=np.float64)
    for name in names:
        indices = [i for i, row in enumerate(records) if row["duration_group"] == name]
        mass = float(np.sum(weights[indices])) if indices else 0.0
        contribution = float(np.sum(weights[indices] * values[indices])) if indices else 0.0
        output[name] = {"event_count": len(indices), "original_weight_mass": mass,
                        "same_full_population_denominator_contribution": contribution,
                        "conditional_mse": contribution / mass if mass > 0 else None}
    return output


def _aggregate(records: Sequence[Mapping[str, Any]], weights: np.ndarray) -> dict[str, Any]:
    horizon = {}
    for name in ("1.0", "3.0"):
        horizon[name] = {
            component: _weighted(np.asarray([row["horizon_mse"][name][component] for row in records]), weights)
            for component in ("all", "q", "v")
        }
    primary = {
        component: _weighted(np.asarray([row["primary_mse"][component] for row in records]), weights)
        for component in ("all", "q", "v")
    }
    event_primary = np.asarray([row["primary_mse"]["all"] for row in records], dtype=np.float64)
    full_indices = [i for i, row in enumerate(records) if row["duration_group"] == "complete_3s"]
    full_mass = float(np.sum(weights[full_indices])) if full_indices else 0.0
    full_conditional = None
    if full_mass > 0:
        full_conditional = {
            component: float(weights[full_indices] @ np.asarray([
                row["horizon_mse"]["3.0"][component] for row in records
            ])[full_indices] / full_mass)
            for component in ("all", "q", "v")
        }
    return {
        "horizon_weighted_mse": horizon,
        "primary_weighted_mse": primary,
        "primary_worst10_weighted_cvar": weighted_cvar(event_primary, weights, 0.1),
        "complete_3s": {"event_count": len(full_indices), "original_weight_mass": full_mass,
                         "conditional_3s_mse": full_conditional},
        "duration_groups": {
            component: _duration_summaries(records, weights, component)
            for component in ("all", "q", "v")
        },
    }


def _conditions(candidate: Mapping[str, Any], baseline: Mapping[str, Any], atol: float,
                strict_margin: float) -> dict[str, bool]:
    tests = {
        "primary_strictly_lower": candidate["primary_weighted_mse"]["all"]
        < baseline["primary_weighted_mse"]["all"] - strict_margin,
    }
    for horizon in ("1.0", "3.0"):
        tests[f"h{horizon}_all_not_higher"] = (
            candidate["horizon_weighted_mse"][horizon]["all"]
            <= baseline["horizon_weighted_mse"][horizon]["all"] + atol
        )
    for component in ("q", "v"):
        tests[f"primary_{component}_not_higher"] = (
            candidate["primary_weighted_mse"][component]
            <= baseline["primary_weighted_mse"][component] + atol
        )
    tests["primary_worst10_cvar_not_higher"] = (
        candidate["primary_worst10_weighted_cvar"]
        <= baseline["primary_worst10_weighted_cvar"] + atol
    )
    candidate_full, baseline_full = candidate["complete_3s"], baseline["complete_3s"]
    tests["complete_3s_population_available"] = candidate_full["event_count"] > 0
    tests["complete_3s_conditional_not_higher"] = (
        candidate_full["conditional_3s_mse"] is not None
        and baseline_full["conditional_3s_mse"] is not None
        and candidate_full["conditional_3s_mse"]["all"]
        <= baseline_full["conditional_3s_mse"]["all"] + atol
    )
    return tests


@np.errstate(over="raise", invalid="raise", divide="raise")
def compare_future(candidate_paths: Any, baseline_paths: Any, observed: Any,
                   forecast_times: Any, offsets: Any, target_scale: Any,
                   weights: Any, *, atol: float = DEFAULT_ATOL,
                   strict_margin: float = DEFAULT_ATOL) -> dict[str, Any]:
    """Compare flattened ragged future paths against observed arrays.

    Paths are ``(total_points, 22)``; observed is ``(total_points, 20)`` and
    forecast_times shares the same flattened offsets, with an exact t0 at zero
    for each event.  Each event's first predicted visible state must exactly
    equal its first observed visible state. The t0 row is excluded from every
    loss. The primary event loss averages the true available <=1 s and <=3 s
    horizon MSEs with equal weight, and population CVaR uses original weights.
    """
    if (isinstance(atol, bool) or isinstance(strict_margin, bool)
            or not math.isfinite(atol) or atol < 0
            or not math.isfinite(strict_margin) or strict_margin < 0):
        raise FutureMetricsError("comparison tolerances must be finite and nonnegative")
    candidate = _finite_array(candidate_paths, "candidate_paths", ndim=2)
    baseline = _finite_array(baseline_paths, "baseline_paths", ndim=2)
    truth = _finite_array(observed, "observed", ndim=2)
    times = _finite_array(forecast_times, "forecast_times", ndim=1)
    scale = _finite_array(target_scale, "target_scale", ndim=1)
    weight = _finite_array(weights, "weights", ndim=1)
    if candidate.shape[1:] != (22,) or baseline.shape != candidate.shape:
        raise FutureMetricsError("candidate and baseline paths must align with shape (total_points, 22)")
    if truth.shape != (candidate.shape[0], 20) or times.shape != (candidate.shape[0],):
        raise FutureMetricsError("observed and forecast_times must align with flattened path rows")
    if scale.shape != (20,) or np.any(scale <= 0):
        raise FutureMetricsError("target_scale must contain 20 finite positive values")
    if weight.size == 0 or np.any(weight <= 0) or not np.isclose(weight.sum(), 1.0, rtol=0, atol=1e-12):
        raise FutureMetricsError("weights must be positive and sum to one")
    offsets_array = _validate_offsets(offsets, weight.size, candidate.shape[0])

    candidate_records, baseline_records = [], []
    try:
        for index, (start, end) in enumerate(pairwise(offsets_array)):
            if end - start > 301:
                raise FutureMetricsError(f"event {index} has more than 300 future output points")
            sl = slice(int(start), int(end))
            event_times = times[sl]
            candidate_records.append(_event_record(candidate[sl], truth[sl], scale, event_times, index))
            baseline_records.append(_event_record(baseline[sl], truth[sl], scale, event_times, index))
        candidate_summary = _aggregate(candidate_records, weight)
        baseline_summary = _aggregate(baseline_records, weight)
        _finite_derived_tree(candidate_summary)
        _finite_derived_tree(baseline_summary)
    except (FloatingPointError, OverflowError) as error:
        error.partial_record = {"candidate_events": candidate_records, "baseline_events": baseline_records}
        raise
    conditions = _conditions(candidate_summary, baseline_summary, atol, strict_margin)
    failures = [name for name, passed in conditions.items() if not passed]
    return {
        "event_count": int(weight.size),
        "original_weights": weight.tolist(),
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "candidate_events": candidate_records,
        "baseline_events": baseline_records,
        "conditions": conditions,
        "failure_reasons": failures,
        "I_pass": not failures,
    }


def _finite_derived_tree(value):
    if isinstance(value, Mapping):
        for child in value.values():
            _finite_derived_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_derived_tree(child)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError("nonfinite derived future aggregate")
