"""Weak-form restoring-force diagnostics on a fixed spatial-damping anchor."""

from __future__ import annotations

import numpy as np

from kron_reduce.evaluation.topology_momentum import momentum_system, project_residual
from kron_reduce.evaluation.topology_spatial_momentum import _force_directions

COEFFICIENT_BOUNDS = (-0.25, 0.25)
SCALE_MIN = 1e-6 - 1.


def restoring_momentum(path, power, equilibrium, blocks, step, spend, *, root):
    """Project restoring and global-stiffness directions off one hidden-state span.

    The base residual uses the fixed spatial damping anchor (global-to-local
    coefficient one). Both candidate directions are then evaluated in the same
    per-case latent-memory column space.
    """
    intervals = len(path) - 1
    root = np.asarray(root, dtype=np.float64)
    damping = np.asarray(blocks[3], dtype=np.float64)
    if (root.shape != damping.shape or not np.isfinite(root).all()
            or not np.allclose(root, root.T, rtol=0.0, atol=1e-12)):
        raise ValueError("finite spatial damping root with matching dimensions required")
    if not np.allclose(root @ root, damping, rtol=1e-9, atol=1e-11):
        raise ValueError("spatial damping root does not reconstruct D0")
    spend("weak_intervals", intervals)
    operator, rhs, directions = momentum_system(path, power, equilibrium, blocks, step,
                                                 restoring=True)
    spatial4 = _force_directions(path, blocks, step, 4, root)
    spend("quad_check_intervals", intervals)
    spatial8 = _force_directions(path, blocks, step, 8, root)
    qgap = float(np.max(np.abs(spatial8-spatial4)) /
                 max(1.0, float(np.max(np.abs(spatial8)))))
    spatial_rhs = rhs - spatial4[:, 0] - spatial4[:, 1]

    spend("svd")
    projected = project_residual(operator, spatial_rhs)
    basis = projected["basis"]
    baseline = projected["perpendicular"]
    direction_vectors = {
        "restoring": directions["restoring"],
        "stiffness_scaling": directions["stiffness"],
    }
    names = ("restoring", "stiffness_scaling")
    columns = np.column_stack([direction_vectors[name] for name in names])
    projected_columns = columns - basis @ (basis.T @ columns)
    perpendicular = {name: projected_columns[:, index] for index, name in enumerate(names)}
    cross_directions = float(projected_columns[:, 0] @ projected_columns[:, 1])
    shape_norm2 = float(projected_columns[:, 0] @ projected_columns[:, 0])
    scale_norm2 = float(projected_columns[:, 1] @ projected_columns[:, 1])
    correlation = (cross_directions / np.sqrt(shape_norm2*scale_norm2)
                   if shape_norm2 > 0.0 and scale_norm2 > 0.0 else None)
    result = {
        "intervals": intervals,
        "rank": int(projected["rank"]),
        "spatial_sse": float(baseline @ baseline),
        "quadrature_gap": qgap,
        "directions": {
            name: {"sse": float(vector @ vector), "cross": float(vector @ baseline)}
            for name, vector in perpendicular.items()
        },
        "projected_direction_dot": cross_directions,
        "projected_direction_correlation": correlation,
    }
    values = [result["spatial_sse"], qgap, cross_directions]
    values.extend(item for row in result["directions"].values() for item in row.values())
    if correlation is not None:
        values.append(correlation)
    if not np.isfinite(values).all():
        raise FloatingPointError("restoring momentum projection is nonfinite")
    result["qualified"] = bool(result["rank"] == 10 and qgap <= 1e-10)
    return result


def fit_restoring_train(rows):
    """Fit both bounded signed directions independently on train rows."""
    if not rows or any(row.get("role") != "train" for row in rows):
        raise ValueError("restoring coefficients require train-only rows")
    if any(row["rank"] != 10 for row in rows):
        return None
    fits = {}
    for output_name, name in (("shape", "restoring"), ("scale", "stiffness_scaling")):
        curvature = sum(float(row["directions"][name]["sse"]) for row in rows)
        numerator = sum(float(row["directions"][name]["cross"]) for row in rows)
        if not np.isfinite([curvature, numerator]).all() or curvature <= 0.0:
            return None
        raw = numerator / curvature
        if not np.isfinite(raw):
            return None
        lower, upper = COEFFICIENT_BOUNDS if output_name == "shape" else (SCALE_MIN, np.inf)
        value = float(np.clip(raw, lower, upper))
        gradient = curvature * value - numerator
        boundary = "lower" if value == lower else "upper" if value == upper else "interior"
        fits[output_name] = {"raw": float(raw), "value": value, "curvature": float(curvature),
                             "gradient": float(gradient), "boundary": boundary}
    return fits


def restoring_sse(rows, fit):
    """Score base, shape-corrected, and stiffness-scale-corrected saved rows."""
    if not isinstance(fit, dict) or any(name not in fit or "value" not in fit[name]
                                        for name in ("shape", "scale")):
        raise ValueError("shape and scale fitted coefficients are required")
    alpha, beta = float(fit["shape"]["value"]), float(fit["scale"]["value"])
    lower, upper = COEFFICIENT_BOUNDS
    if not np.isfinite([alpha, beta]).all() or not lower <= alpha <= upper or beta < SCALE_MIN:
        raise ValueError("restoring coefficient must be bounded and stiffness multiplier positive")
    baseline = sum(float(row["spatial_sse"]) for row in rows)
    scores = {"base": float(baseline)}
    for output_name, direction_name, coefficient in (
            ("shape", "restoring", alpha), ("scale", "stiffness_scaling", beta)):
        cross = sum(float(row["directions"][direction_name]["cross"]) for row in rows)
        size = sum(float(row["directions"][direction_name]["sse"]) for row in rows)
        scores[output_name] = float(baseline - 2.0*coefficient*cross + coefficient**2*size)
    if not np.isfinite(list(scores.values())).all():
        raise FloatingPointError("restoring momentum score is nonfinite")
    for name in ("shape", "scale"):
        scores[name] = max(0.0, scores[name])
    return scores
