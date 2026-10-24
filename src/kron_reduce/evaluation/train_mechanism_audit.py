"""Pure-NumPy, data-independent helpers for the proposed A09 train audit.

The caller supplies already authorized arrays. This module has no project data,
model, provider, or file-system imports. Paths use 20 visible channels: q[0:10]
in radians and omega[10:20] in rad/s. ``true_future`` and ``pred_future`` both
include their t0 point; that repeated point is excluded from future scoring and
the visible path is built from the 11 true history points plus future[1:].
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

PERSISTENCE = 5
DT_S = 0.01


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _visible(value: Any, name: str) -> np.ndarray:
    result = _finite_array(value, name, ndim=2)
    if result.shape[1:] != (20,):
        raise ValueError(f"{name} must have 20 columns (q10 then omega10)")
    return result


def _validate_inputs(
    history: Any,
    true_future: Any,
    pred_future: Any,
    equilibrium_delta: Any,
    mass: Any,
    omega_s: Any,
    target_scale: Any,
) -> tuple[np.ndarray, ...]:
    history = _visible(history, "history")
    truth = _visible(true_future, "true_future")
    pred = _visible(pred_future, "pred_future")
    if history.shape != (11, 20):
        raise ValueError("history must contain exactly 11 points")
    if truth.shape != pred.shape or truth.shape[0] < 1:
        raise ValueError("future arrays must have equal shape and include at least the t0 point")
    if not np.array_equal(truth[0], history[-1]) or not np.array_equal(pred[0], history[-1]):
        raise ValueError("true_future[0] and pred_future[0] must exactly equal history[-1]")
    if truth.shape[0] > 301:
        raise ValueError("future arrays may contain at most 300 scored points after t0")
    eq = _finite_array(equilibrium_delta, "equilibrium_delta", ndim=1)
    mass = _finite_array(mass, "mass", ndim=1)
    scale = _finite_array(target_scale, "target_scale", ndim=1)
    if eq.shape != (10,) or mass.shape != (10,) or scale.shape != (20,):
        raise ValueError("equilibrium_delta/mass/target_scale must have shapes (10,), (10,), (20,)")
    if np.any(mass <= 0) or np.any(scale <= 0):
        raise ValueError("mass and target_scale must be strictly positive")
    if isinstance(omega_s, (bool, np.bool_)) or not np.isscalar(omega_s):
        raise ValueError("omega_s must be a finite positive scalar")
    omega_s = float(omega_s)
    if not math.isfinite(omega_s) or omega_s <= 0:
        raise ValueError("omega_s must be a finite positive scalar")
    return history, truth, pred, eq, mass, omega_s, scale


def _coi_project(values: np.ndarray, mass: np.ndarray) -> np.ndarray:
    center = np.sum(values * mass[None, :], axis=1, keepdims=True) / np.sum(mass)
    return values - center


def _row_margin(path: np.ndarray, eq: np.ndarray, mass: np.ndarray, omega_s: float) -> np.ndarray:
    q = path[:, :10] + eq[None, :]
    v_coi = _coi_project(path[:, 10:], mass)
    angle_margin = np.ptp(q, axis=1) / math.pi - 1.0
    speed_margin = np.max(np.abs(v_coi), axis=1) / (0.02 * omega_s) - 1.0
    return np.minimum(angle_margin, speed_margin)


def _window_result(row_margin: np.ndarray, *, min_end: int) -> dict[str, Any]:
    windows: list[tuple[int, int, float]] = []
    for end in range(PERSISTENCE - 1, len(row_margin)):
        if end >= min_end:
            start = end - PERSISTENCE + 1
            windows.append((start, end, float(np.min(row_margin[start:end + 1]))))
    if not windows:
        return {
            "rho": None,
            "triggered": None,
            "first_trigger_end_index": None,
            "best_window_start_index": None,
            "best_window_end_index": None,
            "best_window_margin": None,
            "eligible_windows": 0,
            "arithmetic_boundary": None,
        }
    best = max(windows, key=lambda item: (item[2], -item[1]))
    rho = best[2]
    first = next((end for _start, end, margin in windows if margin > 0.0), None)
    return {
        "rho": rho,
        "triggered": bool(rho > 0.0),
        "first_trigger_end_index": first,
        "best_window_start_index": best[0],
        "best_window_end_index": best[1],
        "best_window_margin": rho,
        "eligible_windows": len(windows),
        "arithmetic_boundary": bool(abs(rho) <= 1e-12),
    }


def _window_points(result: dict[str, Any], npoints: int) -> np.ndarray:
    """Points participating in any complete window in a returned population."""
    if result["eligible_windows"] == 0:
        return np.zeros(npoints, dtype=bool)
    min_end = result.get("_min_end", 0)
    points = np.zeros(npoints, dtype=bool)
    for end in range(PERSISTENCE - 1, npoints):
        if end >= min_end:
            points[end - PERSISTENCE + 1:end + 1] = True
    return points


def _rule_and_bound(
    truth_path: np.ndarray,
    pred_path: np.ndarray,
    eq: np.ndarray,
    mass: np.ndarray,
    omega_s: float,
    *,
    min_end: int,
) -> dict[str, Any]:
    truth_rows = _row_margin(truth_path, eq, mass, omega_s)
    pred_rows = _row_margin(pred_path, eq, mass, omega_s)
    truth_rule = _window_result(truth_rows, min_end=min_end)
    pred_rule = _window_result(pred_rows, min_end=min_end)
    truth_rule["_min_end"] = min_end
    mask = _window_points(truth_rule, len(truth_path))
    if not np.any(mask):
        atom = vector = None
    else:
        dq = pred_path[mask, :10] - truth_path[mask, :10]
        dv = pred_path[mask, 10:] - truth_path[mask, 10:]
        delta_bound = 2.0 * float(np.max(np.abs(dq))) / math.pi
        speed_bound = float(np.max(np.abs(_coi_project(dv, mass)))) / (0.02 * omega_s)
        vector = max(delta_bound, speed_bound)
        q_t = truth_path[mask, :10] + eq[None, :]
        q_p = pred_path[mask, :10] + eq[None, :]
        v_t = _coi_project(truth_path[mask, 10:], mass)
        v_p = _coi_project(pred_path[mask, 10:], mass)
        angle_delta = np.abs(np.ptp(q_p, axis=1) - np.ptp(q_t, axis=1)) / math.pi
        speed_delta = (np.abs(np.max(np.abs(v_p), axis=1)
                               - np.max(np.abs(v_t), axis=1)) / (0.02 * omega_s))
        atom = max(float(np.max(angle_delta)), float(np.max(speed_delta)))
        rho_t, rho_p = truth_rule["rho"], pred_rule["rho"]
        tolerance = 1e-12 * max(1.0, abs(rho_t), abs(rho_p), vector)
        if abs(rho_p - rho_t) > atom + tolerance or atom > vector + tolerance:
            raise ArithmeticError("LOS margin Lipschitz bound failed on its fixed window population")
    truth_rule.pop("_min_end")
    return {
        "truth": truth_rule,
        "prediction": pred_rule,
        "E_atom": atom,
        "E_vector": vector,
        "bound_verified": None if atom is None else True,
    }


def _point_error_metrics(
    true_future: np.ndarray,
    pred_future: np.ndarray,
    target_scale: np.ndarray,
) -> dict[str, Any]:
    # Index zero is the shared origin; only points 1..N are scored.
    error = (pred_future[1:] - true_future[1:]) / target_scale[None, :]
    e2 = np.mean(error**2, axis=1)
    if not np.all(np.isfinite(e2)):
        raise ValueError("normalized point errors are non-finite")
    if not len(e2):
        return {"e2": e2, "mean": None, "maximum": None,
                "maximum_future_index": None, "maximum_to_mean_ratio": None}
    mean = float(np.mean(e2))
    maximum = float(np.max(e2))
    return {"e2": e2, "mean": mean, "maximum": maximum,
            "maximum_future_index": int(np.argmax(e2) + 1),
            "maximum_to_mean_ratio": maximum / mean if mean > 0 else None}


def audit_event(
    *,
    history: Any,
    true_future: Any,
    pred_future: Any,
    equilibrium_delta: Any,
    mass: Any,
    omega_s: Any,
    target_scale: Any,
) -> dict[str, Any]:
    """Audit one train event from authorized arrays, without labels or metadata.

    ``history`` is the 11 observed points at indices 0..10. Both future arrays
    include the shared origin at index 0; only future indices 1..N are scored.
    The LOS paths concatenate the real history with future[1:], so t0 is never
    duplicated. A representative window is selected from truth alone among
    windows ending at index >=11, with earliest-end tie breaking, and all
    prediction/error-density statistics use that exact window.
    """
    history, truth, pred, eq, mass, omega_s, scale = _validate_inputs(
        history, true_future, pred_future, equilibrium_delta, mass, omega_s, target_scale
    )
    nfuture = truth.shape[0] - 1
    truth_path = np.concatenate((history, truth[1:]), axis=0)
    pred_path = np.concatenate((history, pred[1:]), axis=0)
    if truth_path.shape != (11 + nfuture, 20):
        raise ArithmeticError("history/future concatenation has an unexpected shape")

    history_result = _rule_and_bound(
        truth_path[:11], pred_path[:11], eq, mass, omega_s, min_end=0
    )
    future_result = _rule_and_bound(
        truth_path, pred_path, eq, mass, omega_s, min_end=11
    )
    full_result = _rule_and_bound(
        truth_path, pred_path, eq, mass, omega_s, min_end=0
    )
    for result in (history_result, future_result, full_result):
        for rule in (result["truth"], result["prediction"]):
            _trigger_times(rule)
    representative = future_result["truth"]
    rep_start, rep_end = representative["best_window_start_index"], representative["best_window_end_index"]
    if rep_start is None:
        representative_info = None
        window_weight_mass = window_error_contribution = None
    else:
        # Future scoring index 1 corresponds to concatenated path index 11.
        selected_future = [k - 10 for k in range(rep_start, rep_end + 1) if k >= 11]
        n1, n3 = min(nfuture, 100), min(nfuture, 300)
        coefficients = np.array([
            0.5 / n1 * (k <= n1) + 0.5 / n3 * (k <= n3)
            for k in range(1, nfuture + 1)
        ], dtype=np.float64)
        e2 = _point_error_metrics(truth, pred, scale)["e2"]
        selection = np.zeros(nfuture, dtype=bool)
        for k in selected_future:
            selection[k - 1] = True
        window_weight_mass = float(np.sum(coefficients[selection]))
        window_error_contribution = float(np.sum(coefficients[selection] * e2[selection]))
        fixed_window = _rule_and_bound(
            truth_path[rep_start:rep_end + 1], pred_path[rep_start:rep_end + 1],
            eq, mass, omega_s, min_end=4,
        )
        representative_info = {
            "start_index": rep_start,
            "end_index": rep_end,
            "truth_margin": representative["best_window_margin"],
            "prediction_margin_same_window": fixed_window["prediction"]["rho"],
            "fixed_five_point_audit": fixed_window,
            "future_indices": selected_future,
            "limiting_condition": _limiting_condition(truth_path, eq, mass, omega_s, rep_start, rep_end),
        }
        for rule in (representative_info["fixed_five_point_audit"]["truth"],
                     representative_info["fixed_five_point_audit"]["prediction"]):
            _trigger_times(rule, index_offset=rep_start)

    point = _point_error_metrics(truth, pred, scale)
    n1, n3 = min(nfuture, 100), min(nfuture, 300)
    normalized_error = (pred[1:] - truth[1:]) / scale[None, :]
    physical_error = pred[1:] - truth[1:]
    horizon_scores = {}
    for name, count in (("1.0", n1), ("3.0", n3)):
        if count == 0:
            horizon_scores[name] = None
            continue
        norm = normalized_error[:count]
        physical = physical_error[:count]
        delta_norm = 0.5 * float(np.mean(norm[:, :10] ** 2))
        omega_norm = 0.5 * float(np.mean(norm[:, 10:] ** 2))
        horizon_scores[name] = {
            "scored_points": count,
            "normalized_mse": delta_norm + omega_norm,
            "normalized_delta_mse_contribution": delta_norm,
            "normalized_omega_mse_contribution": omega_norm,
            "physical_delta_mse_rad2": float(np.mean(physical[:, :10] ** 2)),
            "physical_omega_mse_rad2_per_s2": float(np.mean(physical[:, 10:] ** 2)),
        }
    first_mse = None if horizon_scores["1.0"] is None else horizon_scores["1.0"]["normalized_mse"]
    third_mse = None if horizon_scores["3.0"] is None else horizon_scores["3.0"]["normalized_mse"]
    event_mse = 0.5 * (first_mse + third_mse) if n1 and n3 else None
    if event_mse is not None and not math.isfinite(event_mse):
        raise ValueError("event MSE is non-finite")
    event_weights = np.array([
        0.5 / n1 * (k <= n1) + 0.5 / n3 * (k <= n3)
        for k in range(1, nfuture + 1)
    ])
    coefficient_total = float(np.sum(event_weights))
    if nfuture and not math.isclose(coefficient_total, 1.0, rel_tol=0.0, abs_tol=1e-14):
        raise ArithmeticError("original time-loss coefficients do not sum to one")
    window_density = None
    if window_weight_mass is not None and window_weight_mass > 0:
        window_density = window_error_contribution / window_weight_mass
    tail_mean = float(np.mean(point["e2"][100:])) if nfuture > 100 else None
    q_error = physical_error[:, :10]
    omega_error = _coi_project(physical_error[:, 10:], mass)
    max_q_error = float(np.max(np.abs(q_error))) if nfuture else None
    max_coi_omega_error = float(np.max(np.abs(omega_error))) if nfuture else None
    point_max_angle_error = np.max(np.abs(q_error), axis=1)
    point_max_coi_speed_error = np.max(np.abs(omega_error), axis=1) / omega_s

    return {
        "future_points": nfuture,
        "future_duration_s": nfuture * DT_S,
        "history_rho": history_result,
        "future_rho": future_result,
        "full_rho": full_result,
        "observed_rule_hit": future_result["truth"]["triggered"],
        "prediction_rule_hit": future_result["prediction"]["triggered"],
        "representative_truth_window": representative_info,
        "normalized_point_e2": point["e2"].tolist(),
        "one_second_mse": first_mse,
        "three_second_mse": third_mse,
        "horizon_scores": horizon_scores,
        "event_mse": event_mse,
        "event_nrmse": math.sqrt(event_mse) if event_mse is not None else None,
        "after_one_second_available_point_mean_e2": tail_mean,
        "ordinary_available_point_mean_e2": point["mean"],
        "maximum_e2": point["maximum"],
        "maximum_e2_future_index": point["maximum_future_index"],
        "maximum_to_mean_e2_ratio": point["maximum_to_mean_ratio"],
        "maximum_angle_error_rad": max_q_error,
        "point_max_angle_error_rad": point_max_angle_error.tolist(),
        "point_max_coi_speed_error_pu": point_max_coi_speed_error.tolist(),
        "maximum_angle_error_future_index": int(np.argmax(point_max_angle_error))+1 if nfuture else None,
        "maximum_coi_speed_error_future_index": int(np.argmax(point_max_coi_speed_error))+1 if nfuture else None,
        "maximum_coi_omega_error_rad_per_s": max_coi_omega_error,
        "maximum_coi_speed_error_pu": (
            None if max_coi_omega_error is None else max_coi_omega_error / omega_s
        ),
        "time_coefficients_sum": coefficient_total if nfuture else None,
        "representative_window_original_weight_mass_p": window_weight_mass,
        "representative_window_error_contribution_C": window_error_contribution,
        "representative_window_error_density_C_over_p": window_density,
        "representative_window_excess_over_event_mean_C_minus_pM": (
            None if window_density is None or event_mse is None
            else window_error_contribution - window_weight_mass * event_mse
        ),
    }


def _trigger_times(rule, *, index_offset=0):
    index = rule["first_trigger_end_index"]
    record_time = None if index is None else (index+index_offset)*DT_S
    rule.update(index_offset_to_record_grid=index_offset,
                first_trigger_record_s=record_time,
                first_trigger_forecast_s=None if record_time is None else record_time-.1)


def _limiting_condition(
    path: np.ndarray, eq: np.ndarray, mass: np.ndarray, omega_s: float, start: int, end: int
) -> str:
    q = path[start:end + 1, :10] + eq[None, :]
    v = _coi_project(path[start:end + 1, 10:], mass)
    angle_margin = np.ptp(q, axis=1) / math.pi - 1.0
    speed_margin = np.max(np.abs(v), axis=1) / (0.02 * omega_s) - 1.0
    angle = float(np.min(angle_margin))
    speed = float(np.min(speed_margin))
    if angle < speed:
        return "angle"
    if speed < angle:
        return "speed"
    return "tie"
