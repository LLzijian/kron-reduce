"""Observed-path scalar profiles for constant and angle-dependent damping."""

from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre

from kron_reduce.evaluation.topology_momentum import momentum_system, project_residual

_QUADRATURE = {n: roots_legendre(n) for n in (4, 8)}


def _angle_damping_direction(path, blocks, step, points, theta_radians=1.0):
    """Build the signed force direction for D(q) = D0 * (1 - rho*h(q))."""
    if not np.isfinite(theta_radians) or theta_radians <= 0.0:
        raise ValueError("angle scale must be finite and positive")
    mass, incidence, _stiffness, damping, _coupling, _tau = blocks
    n = len(mass)
    q = np.asarray(path, dtype=np.float64)[:, :n]
    v = np.asarray(path, dtype=np.float64)[:, n:]

    nodes, weights4 = _QUADRATURE[points]
    fraction = (nodes + 1.0) * 0.5
    quadrature_weights = weights4 * 0.5
    projection = np.eye(n) - mass[:, None] / mass.sum()
    damping_force = np.zeros((len(q) - 1, n), dtype=np.float64)
    dq, dv = np.diff(q, axis=0), np.diff(v, axis=0)
    for s, weight in zip(fraction, quadrature_weights, strict=True):
        q_s = q[:-1] + s * dq
        v_s = v[:-1] + s * dv
        relative = q_s @ incidence
        energy = np.mean(relative * relative, axis=1)
        h = energy / (energy + theta_radians**2)
        damping_force += weight * h[:, None] * (v_s @ damping.T)
    damping_force = step * damping_force @ projection.T
    return damping_force


def angle_scale_residuals(path, power, equilibrium, blocks, step, spend, starts):
    """Compare two fixed 100 ms memory starts and the free optimum on future rows.

    The two observed-path scales (0.0025 and 0.00125) use the supplied start
    coordinates unchanged.  The angle-dependent damping reduction is fixed at
    rho=2/3 for this comparison; only the future residual is scored.
    """
    path_array = np.asarray(path, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    if path_array.ndim != 2 or len(path_array) <= 11:
        raise ValueError("path must include the 100 ms history and at least one future interval")
    if starts.shape != (2, 10) or not np.isfinite(starts).all():
        raise ValueError("exactly two finite ten-dimensional history starts are required")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")

    intervals = len(path_array) - 1
    node_count = len(np.asarray(blocks[0]))
    if node_count != 10:
        raise ValueError("angle-scale comparison is defined for the ten-node T0M1 model")
    cutoff = 10 * node_count
    future_intervals = len(path_array) - 11
    spend("weak_intervals", intervals)
    operator, rhs, _ = momentum_system(path_array, power, equilibrium, blocks, step)
    if operator.shape[1] != starts.shape[1]:
        raise ValueError("momentum operator and fixed starts have incompatible dimensions")
    if not np.isfinite(operator).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("nonfinite projected momentum system")

    rho = 2.0 / 3.0
    scales = (1.0, 1.25)
    directions = {}
    gaps = {}
    for theta in scales:
        direction4 = _angle_damping_direction(path_array, blocks, step, 4, theta)
        spend("quad_check_intervals", intervals)
        direction8 = _angle_damping_direction(path_array, blocks, step, 8, theta)
        if not np.isfinite(direction4).all() or not np.isfinite(direction8).all():
            raise FloatingPointError("nonfinite angle damping force")
        denominator = max(1.0, float(np.max(np.abs(direction8))))
        gaps[str(theta)] = float(np.max(np.abs(direction8 - direction4)) / denominator)
        if not np.isfinite(gaps[str(theta)]):
            raise FloatingPointError("nonfinite quadrature gap")
        mass = np.asarray(blocks[0], dtype=np.float64)
        weights = 1.0 / (mass * step * np.sqrt(intervals))
        directions[theta] = (direction4 * weights).ravel()

    operator_future = operator[cutoff:]
    rhs_future = rhs[cutoff:]
    if len(rhs_future) != future_intervals * node_count:
        raise ValueError("future row count does not match the history boundary")
    normalization = intervals / future_intervals

    spend("svd")
    fitted = project_residual(operator_future, rhs_future)
    rank, basis = fitted["rank"], fitted["basis"]

    sse = {"0.0025": [], "0.00125": [], "free": []}
    for theta in scales:
        adjusted = rhs_future - rho * directions[theta][cutoff:]
        for observer, start in zip(("0.0025", "0.00125"), starts, strict=True):
            residual = adjusted - operator_future @ start
            score = float((residual @ residual) * normalization)
            if not np.isfinite(score):
                raise FloatingPointError("fixed-start future residual is nonfinite")
            sse[observer].append(score)
        perpendicular = adjusted - basis @ (basis.T @ adjusted)
        score = float((perpendicular @ perpendicular) * normalization)
        if not np.isfinite(score):
            raise FloatingPointError("free future residual is nonfinite")
        sse["free"].append(score)

    return {"rank": rank, "quadrature_gap": gaps, "sse": sse,
            "rho": rho, "theta_radians": list(scales),
            "future_intervals": future_intervals, "normalization": normalization}


def angle_momentum(path, power, equilibrium, blocks, step, spend, *, check_quadrature=False):
    """Return projected residual SSE, rank, and fitted damping directions.

    Both directions are projected against the same frozen latent-memory span.
    ``constant`` is the original ``D/D0 = 1 + alpha`` damping direction;
    ``angle`` is the signed reduction direction ``D(q)/D0 = 1 - rho*h(q)``.
    """
    spend("weak_intervals", len(path) - 1)
    operator, rhs, directions = momentum_system(path, power, equilibrium, blocks, step)
    angle_direction = _angle_damping_direction(path, blocks, step, 4)
    mass = np.asarray(blocks[0], dtype=np.float64)
    weights = 1.0 / (mass * step * np.sqrt(len(path) - 1))
    directions["angle_damping_reduction"] = (angle_direction * weights).ravel()

    spend("svd")
    fitted = project_residual(operator, rhs)
    perpendicular = fitted["perpendicular"]
    basis = fitted["basis"]
    result_directions = {}
    for name, raw in (("constant", directions["damping"]),
                      ("angle", directions["angle_damping_reduction"])):
        projected = raw - basis @ (basis.T @ raw)
        result_directions[name] = {
            "sse": float(projected @ projected),
            "cross": float(projected @ perpendicular),
        }
    result = {
        "orthogonal_sse": float(perpendicular @ perpendicular),
        "rank": int(fitted["rank"]),
        "directions": result_directions,
    }
    if check_quadrature:
        spend("quad_check_intervals", len(path)-1)
        finer = _angle_damping_direction(path, blocks, step, 8)
        result["quadrature_gap"] = float(np.max(np.abs(finer-angle_direction))/max(1., np.max(np.abs(finer))))
    return result


def fit_train_damping(rows):
    """Fit each one-dimensional force direction on complete train statistics."""
    if not rows or any(row["role"] != "train" for row in rows):
        raise ValueError("damping coefficients require train-only rows")
    if any(row["rank"] != 10 for row in rows):
        return None
    fit = {}
    for name, (lower, upper) in {"constant": (-1.+1e-6, np.inf), "angle": (0., 2./3.)}.items():
        size = sum(row["directions"][name]["sse"] for row in rows)
        cross = sum(row["directions"][name]["cross"] for row in rows)
        if not np.isfinite([size, cross]).all() or size <= 0.:
            return None
        fit[name] = float(np.clip(cross/size, lower, upper))
    return fit


def damping_sse(rows, fit):
    baseline = sum(row["orthogonal_sse"] for row in rows)
    output = {"events": len(rows), "original": baseline}
    for name, change in fit.items():
        output[name] = sum(max(0., row["orthogonal_sse"]-2.*change*row["directions"][name]["cross"]
                              +change**2*row["directions"][name]["sse"]) for row in rows)
    return output
