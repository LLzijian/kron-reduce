"""Pure NumPy replay of the frozen record-20 ms train-input audit rules.

This module accepts arrays supplied by a caller and performs no I/O, model
construction, prediction, integration, or label lookup.
"""
from __future__ import annotations

import numpy as np

DT_S = 0.01
ANGLE_LIMIT_RAD = np.pi
SPEED_LIMIT_PU = 0.02
HOLD_POINTS = 5
TAIL_POINTS = 100
TAIL_ANGLE_LIMIT_RAD = np.pi
TAIL_SPEED_LIMIT_PU = 0.005


def _empty_times():
    return {"index": None, "actual_relative_s": None, "nominal_relative_s": None}


def _relative_time(times: np.ndarray, index: int | None, origin: int) -> dict:
    if index is None:
        return _empty_times()
    return {
        "index": int(index),
        "actual_relative_s": float(times[index] - times[origin]),
        "nominal_relative_s": float((index - origin) * DT_S),
    }


def _invalid_result(n: int, reason: str) -> dict:
    return {
        "status": "invalid",
        "invalid_reason": reason,
        "record_points": n,
        "angle_spread_rad": None,
        "coi_speed_pu": None,
        "angle_violation": None,
        "speed_violation": None,
        "joint_violation": None,
        "first_angle_violation": _empty_times(),
        "first_speed_violation": _empty_times(),
        "first_joint_violation": _empty_times(),
        "first_joint_violation_reason": None,
        "pre_record_onset_unknown": None,
        "first_five_point_window": {
            "start": _empty_times(), "end": _empty_times(), "reason": None,
        },
        "origins": {"2": None, "10": None},
        "stable_tail_100": None,
        "stable_source_acceptance": False,
    }


def audit_record20_rule(t, delta, omega_pu, inertia) -> dict:
    """Audit a complete record using the source inertia-weighted COI rule.

    ``omega_pu`` remains in per-unit. The speed statistic is calculated in
    source order: subtract the inertia-weighted mean from each raw per-unit
    speed, then take the maximum absolute deviation. A record with bad clock
    or non-finite values is ``invalid``; it is never represented as a no-hit.
    Short but otherwise valid records retain their available pointwise
    statistics, while unavailable origins/windows/tails are ``None``.
    """
    times = np.asarray(t, dtype=np.float64)
    angles = np.asarray(delta, dtype=np.float64)
    speeds = np.asarray(omega_pu, dtype=np.float64)
    inertias = np.asarray(inertia, dtype=np.float64)
    if times.ndim != 1 or angles.ndim != 2 or speeds.ndim != 2 or inertias.ndim != 1:
        raise ValueError("t/delta/omega_pu/inertia must have ranks 1/2/2/1")
    n = times.shape[0]
    if angles.shape != speeds.shape or angles.shape != (n, inertias.size):
        raise ValueError("delta and omega_pu must have shape (len(t), len(inertia))")
    if inertias.size == 0 or not np.isfinite(inertias).all() or np.any(inertias <= 0):
        raise ValueError("inertia must contain finite positive values")
    if n == 0:
        return _invalid_result(n, "empty_record")
    if n > 1501:
        return _invalid_result(n, "record_exceeds_registered_1501_point_bound")
    if not (np.isfinite(times).all() and np.isfinite(angles).all() and np.isfinite(speeds).all()):
        return _invalid_result(n, "nonfinite_visible_record")
    expected = np.arange(n, dtype=np.float64) * DT_S
    if (times[0] != 0.0 or np.any(np.diff(times) <= 0.0)
            or np.any(np.abs(times - expected) > 1e-10)):
        return _invalid_result(n, "clock_invalid")

    # Keep the original raw-pu COI order and inertia weights. Do not project
    # a transformed rad/s array and divide it back to pu.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        speed_center = np.sum(speeds * inertias, axis=-1, keepdims=True) / np.sum(inertias)
        coi_speed = speeds - speed_center
        angle_spread = np.ptp(angles, axis=1)
        coi_speed_abs = np.max(np.abs(coi_speed), axis=1)
    if not (np.isfinite(speed_center).all() and np.isfinite(coi_speed).all()
            and np.isfinite(angle_spread).all() and np.isfinite(coi_speed_abs).all()):
        return _invalid_result(n, "nonfinite_derived_statistic")
    angle_bad = angle_spread > ANGLE_LIMIT_RAD
    speed_bad = coi_speed_abs > SPEED_LIMIT_PU
    joint_bad = angle_bad & speed_bad

    def first_true(values):
        found = np.flatnonzero(values)
        return None if found.size == 0 else int(found[0])

    angle_first = first_true(angle_bad)
    speed_first = first_true(speed_bad)
    joint_first = first_true(joint_bad)
    trigger_end = None
    trigger_start = None
    trigger_reason = "no_five_consecutive_joint_violations"
    if n >= HOLD_POINTS:
        run_counts = np.convolve(joint_bad.astype(np.int8), np.ones(HOLD_POINTS, dtype=np.int8), mode="valid")
        starts = np.flatnonzero(run_counts == HOLD_POINTS)
        if starts.size:
            trigger_start = int(starts[0])
            trigger_end = trigger_start + HOLD_POINTS - 1
            trigger_reason = None
    else:
        trigger_reason = "insufficient_record_points_for_five_point_window"

    origins = {}
    for origin in (2, 10):
        if n <= origin:
            origins[str(origin)] = None
            continue
        count = 0
        cursor = origin
        while cursor >= 0 and joint_bad[cursor]:
            count += 1
            cursor -= 1
        origins[str(origin)] = {
            "origin_index": origin,
            "continuous_joint_violations_ending_at_origin": count,
            "joint_violation_seen_by_origin": bool(np.any(joint_bad[:origin + 1])),
            "five_point_trigger_confirmed_by_origin": trigger_end is not None and trigger_end <= origin,
            "first_angle_violation": _relative_time(times, angle_first, origin),
            "first_speed_violation": _relative_time(times, speed_first, origin),
            "first_joint_violation": _relative_time(times, joint_first, origin),
            "five_point_window_start": _relative_time(times, trigger_start, origin),
            "five_point_window_end": _relative_time(times, trigger_end, origin),
        }

    if n < TAIL_POINTS:
        stable_tail = None
    else:
        tail_start = n - TAIL_POINTS
        tail_angle_ok = bool(np.all(angle_spread[tail_start:] < TAIL_ANGLE_LIMIT_RAD))
        tail_speed_ok = bool(np.all(coi_speed_abs[tail_start:] < TAIL_SPEED_LIMIT_PU))
        stable_tail = {
            "start_index": tail_start,
            "end_index": n - 1,
            "start_time_s": float(times[tail_start]),
            "end_time_s": float(times[-1]),
            "angle_below_pi": tail_angle_ok,
            "coi_speed_below_0_005_pu": tail_speed_ok,
            "both_conditions": tail_angle_ok and tail_speed_ok,
        }
    stable_acceptance = bool(
        n == 1501 and times[-1] >= 15.0 - 1e-8 and trigger_end is None
        and stable_tail is not None and stable_tail["both_conditions"]
    )
    return {
        "status": "valid",
        "invalid_reason": None,
        "record_points": n,
        "angle_spread_rad": angle_spread.tolist(),
        "coi_speed_pu": coi_speed_abs.tolist(),
        "angle_violation": angle_bad.tolist(),
        "speed_violation": speed_bad.tolist(),
        "joint_violation": joint_bad.tolist(),
        "first_angle_violation": _relative_time(times, angle_first, 0),
        "first_speed_violation": _relative_time(times, speed_first, 0),
        "first_joint_violation": _relative_time(times, joint_first, 0),
        "first_joint_violation_reason": None if joint_first is not None else "no_joint_violation",
        "pre_record_onset_unknown": joint_first == 0,
        "first_five_point_window": {
            "start": _relative_time(times, trigger_start, 0),
            "end": _relative_time(times, trigger_end, 0),
            "reason": trigger_reason,
        },
        "origins": origins,
        "stable_tail_100": stable_tail,
        "stable_source_acceptance": stable_acceptance,
    }
