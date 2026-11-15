"""Observed-path weak momentum comparison for global and spatial damping."""

from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre

from kron_reduce.evaluation.topology_momentum import momentum_system, project_residual

_QUADRATURE = {n: roots_legendre(n) for n in (4, 8)}
_RHO = 2.0 / 3.0


def spatial_damping_root(blocks):
    """Return the symmetric PSD square root of the frozen damping matrix D0."""
    damping = np.asarray(blocks[3], dtype=np.float64)
    if damping.ndim != 2 or damping.shape[0] != damping.shape[1] or not np.isfinite(damping).all():
        raise ValueError("finite square damping matrix required")
    eigenvalues, eigenvectors = np.linalg.eigh(damping)
    tolerance = 64.0 * np.finfo(float).eps * float(np.max(np.abs(eigenvalues)))
    if (float(np.min(eigenvalues)) < -tolerance or np.max(np.abs(damping-damping.T)) > tolerance
            or np.trace(damping) <= 0.):
        raise ValueError("spatial damping requires symmetric positive-semidefinite D0")
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    return 0.5 * (root + root.T)


def _force_directions(path, blocks, step, points, root):
    """Integrate global and local damping-reduction forces by Gauss quadrature."""
    mass, incidence, _stiffness, damping, _coupling, _tau = blocks
    n = len(mass)
    q = np.asarray(path, dtype=np.float64)[:, :n]
    v = np.asarray(path, dtype=np.float64)[:, n:]
    incident = np.abs(incidence)
    degree = incident.sum(axis=1)
    if np.any(degree <= 0.0):
        raise ValueError("every node must have positive topology degree")
    incident /= degree[:, None]

    nodes, weights = _QUADRATURE[points]
    projection = np.eye(n) - mass[:, None] / mass.sum()
    global_force = np.zeros((len(q) - 1, n), dtype=np.float64)
    local_force = np.zeros_like(global_force)
    trace_force = np.zeros_like(global_force)
    trace_weights = np.diag(damping)/np.trace(damping)
    dq, dv = np.diff(q, axis=0), np.diff(v, axis=0)
    for node, weight in zip((nodes + 1.0) * 0.5, weights * 0.5, strict=True):
        q_node = q[:-1] + node * dq
        v_node = v[:-1] + node * dv
        edge_sq = (q_node @ incidence) ** 2
        mean_sq = np.mean(edge_sq, axis=1, keepdims=True)
        global_strength = _RHO * mean_sq / (mean_sq + 1.0)
        node_index = edge_sq @ incident.T
        node_strength = _RHO * node_index / (node_index + 1.0)
        drag = v_node @ damping.T
        global_force += weight * global_strength * drag
        local_force += weight * ((v_node @ root)*node_strength) @ root
        trace_force += weight * (node_strength @ trace_weights)[:, None] * drag

    # These are positive force changes relative to D0: (D0 - D_candidate) v.
    scale = 1.0 / (mass * step * np.sqrt(len(q) - 1))
    global_force = step * global_force @ projection.T
    local_force = step * local_force @ projection.T
    trace_force = step * trace_force @ projection.T
    direction = ((local_force - global_force) * scale[None, :]).ravel()
    trace_direction = ((trace_force-global_force)*scale[None, :]).ravel()
    global_force = (global_force * scale[None, :]).ravel()
    return np.stack((global_force, direction, trace_direction), axis=1)


def spatial_momentum(path, power, equilibrium, blocks, step, spend, *, root=None):
    """Report the global-damping residual and spatial-minus-global direction.

    The train coefficient interpolates from global to local damping and is
    subsequently clipped to [0, 1]. Supply ``root`` from
    :func:`spatial_damping_root` to reuse the eigendecomposition across events.
    """
    intervals = len(path) - 1
    spend("weak_intervals", intervals)
    operator, rhs, _ = momentum_system(path, power, equilibrium, blocks, step)
    if root is None:
        root = spatial_damping_root(blocks)
        root_eigendecompositions = 1
    else:
        root_eigendecompositions = 0
    root = np.asarray(root, dtype=np.float64)
    damping = blocks[3]
    if (root.shape != damping.shape or not np.isfinite(root).all()
            or not np.allclose(root, root.T, rtol=0.0, atol=1e-12)):
        raise ValueError("finite damping root with matching dimensions required")
    if not np.allclose(root @ root, damping, rtol=1e-9, atol=1e-11):
        raise ValueError("provided damping root does not reconstruct D0")

    directions4 = _force_directions(path, blocks, step, 4, root)
    spend("quad_check_intervals", intervals)
    directions8 = _force_directions(path, blocks, step, 8, root)
    denominator = max(1.0, float(np.max(np.abs(directions8))))
    quadrature_gap = float(np.max(np.abs(directions8-directions4))/denominator)

    spend("svd")
    fitted = project_residual(operator, rhs)
    basis = fitted["basis"]
    baseline = fitted["perpendicular"]
    global_projected, direction_projected, trace_projected = (directions4-basis @ (basis.T @ directions4)).T
    global_residual = baseline - global_projected
    trace_residual = global_residual-trace_projected
    global_sse = float(global_residual @ global_residual)
    trace_sse = float(trace_residual @ trace_residual)
    direction_sse = float(direction_projected @ direction_projected)
    direction_cross = float(direction_projected @ global_residual)
    if not np.isfinite([global_sse, trace_sse, direction_sse, direction_cross, quadrature_gap]).all():
        raise FloatingPointError("spatial momentum evidence contains a nonfinite result")
    return {"global_sse": global_sse, "trace_sse": trace_sse, "direction_sse": direction_sse,
            "direction_cross": direction_cross, "rank": int(fitted["rank"]),
            "quadrature_gap": quadrature_gap,
            "root_eigendecompositions": root_eigendecompositions,
            "rho": _RHO, "scale": 1.0}


def fit_spatial_train(rows):
    """Fit the bounded global-to-local damping interpolation on train rows."""
    if not rows or any(row.get("role") != "train" for row in rows):
        raise ValueError("spatial damping coefficient requires train-only rows")
    if any(row["rank"] != 10 for row in rows):
        return None
    size = sum(float(row["direction_sse"]) for row in rows)
    cross = sum(float(row["direction_cross"]) for row in rows)
    if not np.isfinite([size, cross]).all() or size <= 0.0:
        return None
    raw = cross/size
    value = float(np.clip(raw, 0., 1.))
    return {"raw": raw, "value": value, "curvature": size, "gradient": size*value-cross,
            "boundary": "lower" if value == 0. else "upper" if value == 1. else "interior"}


def spatial_sse(rows, coefficient):
    """Apply one frozen interpolation coefficient to a set of observed rows."""
    if not np.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise ValueError("spatial interpolation coefficient must lie in [0, 1]")
    baseline = sum(float(row["global_sse"]) for row in rows)
    size = sum(float(row["direction_sse"]) for row in rows)
    cross = sum(float(row["direction_cross"]) for row in rows)
    value = baseline - 2.0 * coefficient * cross + coefficient**2 * size
    if not np.isfinite(value):
        raise ValueError("spatial momentum SSE is nonfinite")
    return max(0.0, float(value))
