"""Pure fixed-origin momentum comparison for a precomputed modal damper."""

from __future__ import annotations

import math

import numpy as np

from kron_reduce.evaluation.topology_momentum import momentum_system, project_residual
from kron_reduce.evaluation.topology_spatial_momentum import (
    _force_directions,
    spatial_damping_root,
)

RHO = 2.0 / 3.0
SCALE = 1.0
HISTORY_INTERVALS = 10


def modal_probe(blocks):
    """Precompute spatial anchor and the deterministic highest mode once."""
    mass, incidence, stiffness, damping, _coupling, _tau = (
        np.asarray(value, dtype=np.float64) for value in blocks
    )
    n = len(mass)
    if (n != 10 or mass.shape != (n,) or incidence.ndim != 2 or incidence.shape[0] != n
            or stiffness.shape != (incidence.shape[1],) or damping.shape != (n, n)
            or not all(np.isfinite(value).all() for value in (mass, incidence, stiffness, damping))
            or np.any(mass <= 0.0) or np.any(stiffness <= 0.0)):
        raise ValueError("finite positive mass/stiffness and compatible topology are required")
    try:
        root = spatial_damping_root(blocks)
        inverse_sqrt_mass = 1.0 / np.sqrt(mass)
        laplacian = (incidence * stiffness[None, :]) @ incidence.T
        symmetric = inverse_sqrt_mass[:, None] * laplacian * inverse_sqrt_mass[None, :]
        values, vectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError("modal or damping-root eigensolve failed") from error
    if not np.isfinite(values).all() or len(values) < 2:
        raise FloatingPointError("modal spectrum is incomplete or nonfinite")
    positive_floor = 64.0 * np.finfo(float).eps * max(1.0, abs(float(values[-1])))
    positive_modes = int(np.count_nonzero(values > positive_floor))
    if positive_modes != 9 or values[-1] <= 0.0:
        raise FloatingPointError("reference stiffness must have exactly nine positive modes")
    gap = float(values[-1] - values[-2])
    relative_gap = gap / max(abs(float(values[-1])), np.finfo(float).tiny)
    if not math.isfinite(relative_gap) or relative_gap < 1e-8:
        raise FloatingPointError("highest modal eigenvalue relative gap is below 1e-8")

    mode = np.sqrt(mass) * vectors[:, -1]
    norm = float(np.linalg.norm(mode))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("highest modal vector has invalid norm")
    mode /= norm
    if not math.isclose(float(np.linalg.norm(mode)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise FloatingPointError("highest modal vector failed Euclidean normalization")
    pivot = int(np.argmax(np.abs(mode)))
    if mode[pivot] < 0.0:
        mode *= -1.0
    balance = float(np.sum(mode))
    if not math.isfinite(balance) or abs(balance) > 1e-10:
        raise FloatingPointError("highest modal vector does not satisfy the balance constraint")
    eta = 0.5 * float(mode @ damping @ mode)
    if not math.isfinite(eta) or eta <= 0.0:
        raise FloatingPointError("modal damping coefficient is not finite and positive")
    return {"root": root, "mode": mode, "eta": eta,
            "eigenvalue": float(values[-1]), "eigengap": gap,
            "relative_eigengap": relative_gap, "positive_modes": positive_modes,
            "mode_norm": float(np.linalg.norm(mode)),
            "balance_residual": balance, "sign_pivot": pivot,
            "rho": RHO, "scale": SCALE}


def modal_residuals(path, power, equilibrium, blocks, step, spend, starts, *, probe):
    """Score spatial anchor and fixed modal damper on future observed rows only."""
    path = np.asarray(path, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    mass = np.asarray(blocks[0], dtype=np.float64)
    n = len(mass)
    if path.ndim != 2 or path.shape[1] != 2*n or len(path) <= HISTORY_INTERVALS+1:
        raise ValueError("path must contain ten history intervals and future intervals")
    if starts.shape != (2, 10) or n != 10 or not np.isfinite(starts).all():
        raise ValueError("two finite ten-dimensional observed history starts are required")
    mode = np.asarray(probe["mode"], dtype=np.float64)
    root = np.asarray(probe["root"], dtype=np.float64)
    eta = float(probe["eta"])
    if (mode.shape != (n,) or root.shape != (n, n) or not np.isfinite(mode).all() or not np.isfinite(root).all()
            or not math.isfinite(eta) or eta <= 0.):
        raise ValueError("precomputed modal probe is invalid")

    full_intervals = len(path) - 1
    future_intervals = len(path) - HISTORY_INTERVALS - 1
    cutoff = HISTORY_INTERVALS * n
    spend("weak_intervals", full_intervals)
    operator, rhs, _ = momentum_system(path, power, equilibrium, blocks, step)
    spatial4 = _force_directions(path, blocks, step, 4, root)
    spend("quad_check_intervals", full_intervals)
    spatial8 = _force_directions(path, blocks, step, 8, root)
    qgap = float(np.max(np.abs(spatial8-spatial4)) / max(1.0, float(np.max(np.abs(spatial8)))))

    velocity = path[:, n:]
    projection = np.eye(n) - mass[:, None] / mass.sum()
    modal_axis = projection @ mode
    modal_velocity = 0.5 * (velocity[:-1] + velocity[1:]) @ mode
    modal_rows = (step * eta * modal_velocity)[:, None] * modal_axis[None, :]
    row_scale = 1.0 / (mass * step * np.sqrt(full_intervals))
    modal_direction = (modal_rows * row_scale[None, :]).ravel()
    spatial_rhs = rhs - spatial4[:, 0] - spatial4[:, 1]
    modal_rhs = spatial_rhs + modal_direction

    operator_future = operator[cutoff:]
    old_future, new_future = spatial_rhs[cutoff:], modal_rhs[cutoff:]
    if (len(old_future) != future_intervals*n or operator_future.shape[1] != starts.shape[1]
            or not np.isfinite(operator_future).all() or not np.isfinite(old_future).all()
            or not np.isfinite(new_future).all()):
        raise FloatingPointError("future-only modal residual inputs are malformed or nonfinite")

    spend("svd")
    fit = project_residual(operator_future, old_future)
    basis = fit["basis"]
    normalization = full_intervals / future_intervals
    sse = {"0.0025": [], "0.00125": [], "free": []}
    cross = {}
    direction_sse = {}
    for label, start in zip(("0.0025", "0.00125"), starts, strict=True):
        old_residual = old_future - operator_future @ start
        new_residual = new_future - operator_future @ start
        sse[label] = [float((old_residual @ old_residual) * normalization),
                      float((new_residual @ new_residual) * normalization)]
        cross[label] = float((old_residual @ modal_direction[cutoff:]) * normalization)
        direction_sse[label] = float((modal_direction[cutoff:] @ modal_direction[cutoff:]) * normalization)
    old_free = fit["perpendicular"]
    new_free = new_future - basis @ (basis.T @ new_future)
    direction_free = modal_direction[cutoff:] - basis @ (basis.T @ modal_direction[cutoff:])
    sse["free"] = [float((old_free @ old_free) * normalization),
                   float((new_free @ new_free) * normalization)]
    cross["free"] = float((old_free @ direction_free) * normalization)
    direction_sse["free"] = float((direction_free @ direction_free) * normalization)
    if (not all(np.isfinite(pair).all() for pair in sse.values())
            or not np.isfinite(list(cross.values()) + list(direction_sse.values())).all()
            or not math.isfinite(qgap)):
        raise FloatingPointError("modal future score is nonfinite")
    return {"rank": fit["rank"], "quadrature_gap": {"spatial_anchor": qgap},
            "sse": sse, "cross": cross, "direction_sse": direction_sse,
            "future_intervals": future_intervals,
            "normalization": normalization}
