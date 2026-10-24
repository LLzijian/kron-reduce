"""Descriptive A14 residual statistics, using already computed arrays only."""
from __future__ import annotations

import numpy as np


def residual_statistics(result, target_scale):
    """Keep force, dynamics and kinematics separate; they are not additive parts."""
    out = {}
    c, scale = result["c"], np.asarray(target_scale)
    for order in (2, 4):
        force = result[f"force_residual{order}"]
        normalized = result[f"force_residual_scaled{order}"]
        average = result[f"force_residual_average_force{order}"]
        out[f"gauss{order}_force_max_abs_momentum"] = float(np.max(np.abs(force)))
        out[f"gauss{order}_force_max_abs_average_force"] = float(np.max(np.abs(average)))
        out[f"gauss{order}_force_max_abs_normalized"] = float(np.max(np.abs(normalized)))
        out[f"gauss{order}_force_weighted_squared_residual"] = float(np.sum((c[:, None] * normalized) ** 2) / 10)
        dynamic = result[f"dynamic_residual{order}"]
        scaled = result[f"dynamic_residual_scaled{order}"]
        dynamic_normalized = None if dynamic is None else dynamic.reshape(10, 10) * result["S"][None, :]
        out[f"gauss{order}_dynamic_max_abs_momentum"] = None if dynamic is None else float(np.max(np.abs(dynamic)))
        out[f"gauss{order}_dynamic_max_abs_normalized"] = None if dynamic_normalized is None else float(np.max(np.abs(dynamic_normalized)))
        out[f"gauss{order}_dynamic_max_abs_weighted_design"] = None if scaled is None else float(np.max(np.abs(scaled)))
        out[f"gauss{order}_dynamic_weighted_squared_residual"] = None if scaled is None else float(np.sum(scaled ** 2) / 10)
    kin = result["kin"]
    out["kinematic_max_abs_rad"] = float(np.max(np.abs(kin)))
    out["kinematic_max_abs_normalized"] = float(np.max(np.abs(kin / scale[:10])))
    out["kinematic_weighted_squared_residual"] = float(np.sum(
        (result["h"] / np.sum(result["h"]))[:, None] * (kin / scale[:10]) ** 2) / 10)
    if any(value is not None and not np.isfinite(value) for value in out.values()):
        raise FloatingPointError("nonfinite derived residual statistic")
    return out


def weighted_residual_summary(slots, weights, seeds=(324, 325, 326)):
    """Use the entire fixed population or report incomplete; never drop a failure."""
    weight = np.asarray(weights, dtype=float)
    if weight.shape != (258,) or not np.isfinite(weight).all() or np.any(weight <= 0) or not np.isclose(weight.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("fixed full-population weights are required")
    if len(slots) != 774 or any(s["status"] != "complete" for s in slots):
        return {"complete": False, "reason": "at_least_one_allocated_slot_unresolved_or_failed",
                "weighted_residuals": None, "success_subset_renormalized": False}
    results = {}
    for seed in seeds:
        rows = [s for s in slots if s["seed"] == seed]
        if len(rows) != 258 or [s["cache_index"] for s in rows] != list(range(258)):
            raise ValueError("mechanism summary requires frozen cache order")
        values = [s["residual_statistics"] for s in rows]
        if any(set(v) != set(values[0]) for v in values):
            raise ValueError("inconsistent diagnostic definitions")
        results[str(seed)] = {key: {"weighted_mean": float(weight @ np.array([v[key] for v in values])),
                                   "population_max": float(max(v[key] for v in values))}
                              for key in values[0]}
    return {"complete": True, "weighted_residuals": results, "events_per_seed": 258,
            "success_subset_renormalized": False, "force_and_dynamic_are_additive_components": False}
