"""Observed-path impulses of common and zero-sum dynamic coupling."""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import helmert

from kron_reduce.evaluation.topology_momentum import momentum_system, project_residual
from kron_reduce.evaluation.topology_spatial_momentum import _force_directions

HISTORY_INTERVALS = 10
STARTS = ("0.0025", "0.00125", "free")
ANCHORS = ("constant", "spatial")


def _spatial_coupling_impulse(path, equilibrium, blocks, step, *, rate, limit, substeps, spend):
    """Integrate the zero-sum edge-coupling coordinates and force impulse."""
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
    edge_potential = np.cos(ref)-np.cos(q+ref)-q*np.sin(ref)
    projection = np.eye(n)-mass[:, None]/mass.sum()
    projected_edge_force = projection @ incidence
    edge_basis = np.abs(incidence).T @ helmert(n, full=False).T
    dimension = n-1
    kappa = float(stiffness.sum())
    mobility = rate/kappa
    h = step/substeps
    coordinate = np.zeros((len(path), dimension), dtype=np.float64)
    impulse = np.zeros((intervals, n), dtype=np.float64)
    xi = np.zeros(dimension, dtype=np.float64)

    def field(xi, potential_edges, shifted_edges):
        radial_scale = math.sqrt(1.+float(xi @ xi)/(limit*limit))
        radial = xi/radial_scale
        log_multiplier = edge_basis @ radial
        delta_multiplier = np.expm1(log_multiplier)
        multiplier = 1. + delta_multiplier
        effective_stiffness = stiffness*multiplier
        gradient_radial = edge_basis.T @ (effective_stiffness*potential_edges)
        jacobian_gradient = (gradient_radial/radial_scale
                             - xi*float(xi @ gradient_radial)/(limit*limit*radial_scale**3))
        dxi = -mobility*(kappa*xi+jacobian_gradient)
        d_impulse = -(shifted_edges*stiffness*delta_multiplier) @ projected_edge_force.T
        return np.concatenate((dxi, d_impulse))

    for index in range(intervals):
        state = np.concatenate((xi, np.zeros(n, dtype=np.float64)))
        for j in range(0, 2*substeps, 2):
            k1 = field(state[:dimension], edge_potential[index, j], shifted[index, j])
            k2 = field(state[:dimension]+.5*h*k1[:dimension], edge_potential[index, j+1],
                       shifted[index, j+1])
            k3 = field(state[:dimension]+.5*h*k2[:dimension], edge_potential[index, j+1],
                       shifted[index, j+1])
            k4 = field(state[:dimension]+h*k3[:dimension], edge_potential[index, j+2],
                       shifted[index, j+2])
            state += (h/6.)*(k1+2.*k2+2.*k3+k4)
        xi = state[:dimension]
        coordinate[index+1] = xi
        impulse[index] = state[dimension:]

    radial_scale = np.sqrt(1.+np.sum(coordinate*coordinate, axis=1, keepdims=True)/(limit*limit))
    radial = coordinate/radial_scale
    log_multipliers = radial @ edge_basis.T
    multipliers = np.exp(log_multipliers)
    edge_log_sum = log_multipliers.sum(axis=1)
    if not all(np.isfinite(x).all() for x in (impulse, coordinate, multipliers, edge_log_sum)):
        raise FloatingPointError("nonfinite spatial coupling on observed path")
    return {"interval_impulses": impulse, "c": coordinate,
            "multiplier_range": [float(multipliers.min()), float(multipliers.max())],
            "multiplier_mean": float(multipliers.mean()),
            "edge_log_sum": edge_log_sum,
            "edge_log_sum_max_abs": float(np.abs(edge_log_sum).max())}


def coupling_impulse(path, equilibrium, blocks, step, *, rate, limit, substeps, spend, spatial=False):
    """Integrate c and the signed momentum impulse of the changing stiffness."""
    if not isinstance(spatial, bool):
        raise TypeError("spatial must be a boolean")
    if spatial:
        return _spatial_coupling_impulse(path, equilibrium, blocks, step, rate=rate, limit=limit,
                                        substeps=substeps, spend=spend)
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


def coupling_residuals(path, power, equilibrium, blocks, step, spend, starts, *, root, rate, limit, spatial=False):
    """Compare the same coupling impulse under two damping laws and fixed causal origins."""
    mass = blocks[0]
    starts = np.asarray(starts, dtype=np.float64)
    if not isinstance(spatial, bool):
        raise TypeError("spatial must be a boolean")
    if (len(path) <= HISTORY_INTERVALS+1 or starts.shape != (2, 2, len(mass))
            or not np.isfinite(starts).all()):
        raise ValueError("future observations and the four causal memory origins are required")
    coarse, fine = (coupling_impulse(path, equilibrium, blocks, step, rate=rate, limit=limit,
                                   substeps=count, spend=spend, spatial=spatial) for count in (2, 4))
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
    coordinate_difference = fine["c"]-coarse["c"]
    coordinate_gap = (float(np.linalg.norm(coordinate_difference, axis=1).max()) if spatial
                      else float(np.abs(coordinate_difference).max()))
    gaps = {"damping": float(np.abs(damping-damping_refined).max()),
            "impulse": float(np.abs(discrepancy).max()), "coordinate": coordinate_gap}
    spend("svd")
    fit = project_residual(operator, rhs)
    basis = fit["basis"]
    projected_direction = direction-basis @ (basis.T @ direction)
    sse = {}
    for name, base, fixed in zip(ANCHORS, (rhs, rhs-damping), starts, strict=True):
        residuals = [base-operator @ state for state in fixed]
        residuals.append(base-basis @ (basis.T @ base))
        sse[name] = {}
        for kind, baseline, impulse in zip(STARTS, residuals, (direction, direction, projected_direction), strict=True):
            changed = baseline-impulse
            sse[name][kind] = [float(baseline @ baseline), float(changed @ changed)]
    if not all(np.isfinite(pair).all() for anchor in sse.values() for pair in anchor.values()):
        raise FloatingPointError("nonfinite weak momentum residual on observed path")
    c = fine["c"]
    if spatial:
        multiplier_range = [min(coarse["multiplier_range"][0], fine["multiplier_range"][0]),
                            max(coarse["multiplier_range"][1], fine["multiplier_range"][1])]
        coarse_zero_sum = coarse["edge_log_sum_max_abs"]
        fine_zero_sum = fine["edge_log_sum_max_abs"]
        combined_zero_sum = max(coarse_zero_sum, fine_zero_sum)
        combined_edge_log_sums = np.concatenate((coarse["edge_log_sum"], fine["edge_log_sum"]))
        geometry_qualified = (coarse_zero_sum <= 1e-12 and fine_zero_sum <= 1e-12
                              and multiplier_range[0] >= .5-1e-12
                              and multiplier_range[1] <= 2.+1e-12)
        gaps.update(coarse_zero_sum=coarse_zero_sum, fine_zero_sum=fine_zero_sum,
                    edge_log_sum=combined_zero_sum)
        result = {"rank": fit["rank"], "future_intervals": intervals, "gaps": gaps, "sse": sse,
                  "coupling_origin": c[HISTORY_INTERVALS].tolist(),
                  "direction_sse": {"causal": float(direction @ direction),
                                    "free": float(projected_direction @ projected_direction)},
                  "multiplier_range": fine["multiplier_range"], "multiplier_mean": fine["multiplier_mean"],
                  "coarse_multiplier_range": coarse["multiplier_range"],
                  "fine_multiplier_range": fine["multiplier_range"],
                  "combined_multiplier_range": multiplier_range,
                  "spatial_geometry_qualified": geometry_qualified,
                  "coarse_edge_log_sum_max_abs": coarse_zero_sum,
                  "fine_edge_log_sum_max_abs": fine_zero_sum,
                  "edge_log_sum_range": [float(combined_edge_log_sums.min()),
                                          float(combined_edge_log_sums.max())],
                  "edge_log_sum_max_abs": combined_zero_sum}
    else:
        multiplier = np.exp(2.*limit*np.tanh(c/limit))
        result = {"rank": fit["rank"], "future_intervals": intervals, "gaps": gaps, "sse": sse,
                  "coupling_origin": float(c[HISTORY_INTERVALS]),
                  "multiplier_range": [float(multiplier.min()), float(multiplier.max())]}
    return result
