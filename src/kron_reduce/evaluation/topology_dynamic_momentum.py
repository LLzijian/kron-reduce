"""Observed-path impulse of the common dynamic coupling state."""
from __future__ import annotations

import math

import numpy as np

from kron_reduce.evaluation.topology_momentum import momentum_system, project_residual
from kron_reduce.evaluation.topology_spatial_momentum import _force_directions

HISTORY_INTERVALS = 10
STARTS = ("0.0025", "0.00125", "free")
ANCHORS = ("constant", "spatial")


def coupling_impulse(path, equilibrium, blocks, step, *, rate, limit, substeps, spend):
    """Integrate c and the signed momentum impulse of the changing stiffness."""
    mass, incidence, stiffness = blocks[:3]
    path, equilibrium = np.asarray(path, dtype=np.float64), np.asarray(equilibrium, dtype=np.float64)
    n = len(mass)
    if (path.ndim != 2 or path.shape[1] != 2*n or len(path) < 2 or equilibrium.shape != (n,)
            or incidence.shape != (n, len(stiffness)) or min(mass) <= 0. or min(stiffness) <= 0.
            or not all(np.isfinite(x).all() for x in (path, equilibrium, mass, incidence, stiffness))
            or step != .01 or type(substeps) is not int or substeps not in (2, 4)
            or not math.isfinite(rate) or rate < 0. or not 0. < limit < math.log(np.finfo(float).max)/2.):
        raise ValueError("finite observed path, frozen topology and registered coupling clock required")
    intervals = len(path)-1
    spend("dynamic_intervals", intervals)
    spend("dynamic_rhs", 4*substeps*intervals)
    edges = path[:, :n] @ incidence
    ref = equilibrium @ incidence
    fractions = np.linspace(0., 1., 2*substeps+1)
    q = edges[:-1, None, :]+np.diff(edges, axis=0)[:, None, :]*fractions[None, :, None]
    shifted = np.sin(q+ref)-np.sin(ref)
    potential = (np.cos(ref)-np.cos(q+ref)-q*np.sin(ref)) @ stiffness
    projection = np.eye(n)-mass[:, None]/mass.sum()
    force = -(shifted*stiffness) @ (projection @ incidence).T
    h = step/substeps
    mobility = rate/float(stiffness.sum())
    coordinate = np.zeros(len(path))
    impulse = np.zeros((intervals, n))
    c = 0.

    def field(c, potential):
        s = math.tanh(c/limit)
        change = math.expm1(2.*limit*s)
        return -rate*c-mobility*2.*(1.-s*s)*(1.+change)*potential, change

    for i, (energy_nodes, force_nodes) in enumerate(zip(potential, force, strict=True)):
        for j in range(0, 2*substeps, 2):
            a, fa = field(c, energy_nodes[j])
            b, fb = field(c+.5*h*a, energy_nodes[j+1])
            d, fd = field(c+.5*h*b, energy_nodes[j+1])
            e, fe = field(c+h*d, energy_nodes[j+2])
            c += (h/6.)*(a+2.*b+2.*d+e)
            impulse[i] += (h/6.)*(fa*force_nodes[j]+2.*(fb+fd)*force_nodes[j+1]+fe*force_nodes[j+2])
        coordinate[i+1] = c
    if not np.isfinite(impulse).all() or not np.isfinite(coordinate).all():
        raise FloatingPointError("nonfinite dynamic coupling on observed path")
    return {"interval_impulses": impulse, "c": coordinate}


def coupling_residuals(path, power, equilibrium, blocks, step, spend, starts, *, root, rate, limit):
    """Compare the same coupling impulse under two damping laws and fixed causal origins."""
    mass = blocks[0]
    if len(path) <= HISTORY_INTERVALS+1 or starts.shape != (2, 2, len(mass)) or not np.isfinite(starts).all():
        raise ValueError("future observations and the four causal memory origins are required")
    coarse, fine = (coupling_impulse(path, equilibrium, blocks, step, rate=rate, limit=limit,
                                   substeps=count, spend=spend) for count in (2, 4))
    future = path[HISTORY_INTERVALS:]
    intervals = len(future)-1
    spend("weak_intervals", intervals)
    operator, rhs, _ = momentum_system(future, power, equilibrium, blocks, step)
    damping = _force_directions(future, blocks, step, 4, root)[:, :2].sum(axis=1)
    spend("quad_check_intervals", intervals)
    damping_refined = _force_directions(future, blocks, step, 8, root)[:, :2].sum(axis=1)
    scale = 1./(mass*step*math.sqrt(intervals))
    direction = (fine["interval_impulses"][HISTORY_INTERVALS:]*scale).ravel()
    discrepancy = ((fine["interval_impulses"]-coarse["interval_impulses"])[HISTORY_INTERVALS:]*scale).ravel()
    gaps = {"damping": float(np.abs(damping-damping_refined).max()),
            "impulse": float(np.abs(discrepancy).max()),
            "coordinate": float(np.abs(fine["c"]-coarse["c"]).max())}
    spend("svd")
    fit = project_residual(operator, rhs)
    basis = fit["basis"]
    sse = {}
    for name, base, fixed in zip(ANCHORS, (rhs, rhs-damping), starts, strict=True):
        residuals = [base-operator @ state for state in fixed]
        residuals.append(base-basis @ (basis.T @ base))
        projected_direction = direction-basis @ (basis.T @ direction)
        sse[name] = {}
        for kind, baseline, impulse in zip(STARTS, residuals, (direction, direction, projected_direction), strict=True):
            changed = baseline-impulse
            sse[name][kind] = [float(baseline @ baseline), float(changed @ changed)]
    if not all(np.isfinite(pair).all() for anchor in sse.values() for pair in anchor.values()):
        raise FloatingPointError("nonfinite weak momentum residual on observed path")
    c = fine["c"]
    multiplier = np.exp(2.*limit*np.tanh(c/limit))
    return {"rank": fit["rank"], "future_intervals": intervals, "gaps": gaps, "sse": sse,
            "coupling_origin": float(c[HISTORY_INTERVALS]), "multiplier_range": [float(multiplier.min()),
                                                                              float(multiplier.max())]}
