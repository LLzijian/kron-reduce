"""Observed-path scalar profiles for constant and angle-dependent damping."""

from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre

from kron_reduce.evaluation.topology_momentum import momentum_system, project_residual

_QUADRATURE = {n: roots_legendre(n) for n in (4, 8)}


def _angle_damping_direction(path, blocks, step, points):
    """Build the signed force direction for D(q) = D0 * (1 - rho*h(q))."""
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
        h = energy / (1.0 + energy)
        damping_force += weight * h[:, None] * (v_s @ damping.T)
    damping_force = step * damping_force @ projection.T
    return damping_force


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
