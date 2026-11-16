"""Observed-path momentum equations for a fixed reciprocal diagonal memory port."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import hyp1f1

from kron_reduce.evaluation.energy_constrained_initializer import (
    EnergyBallNumericalError,
    solve_energy_ball,
)


def memory_integrals(velocity, coupling, rates, step, *, return_states=False):
    """Integrate each free mode and its zero-initial-state response to linear drive."""
    velocity, coupling, rates = (np.asarray(x, dtype=np.float64) for x in (velocity, coupling, rates))
    if (velocity.ndim != 2 or coupling.ndim != 2 or rates.ndim != 1 or len(velocity) < 2
            or min(coupling.shape) < 1 or coupling.shape != (velocity.shape[1], len(rates))
            or not all(np.isfinite(x).all() for x in (velocity, coupling, rates))
            or np.any(rates <= 0) or not math.isfinite(step) or step <= 0):
        raise ValueError("finite velocity, coupling, positive rates and a positive interval required")
    x = rates*step
    # phi_k(-x) = 1F1(1; k+1; -x)/k!, avoiding subtraction at small rates.
    phi1, phi2, phi3 = (hyp1f1(1., k+1., -x)/math.factorial(k) for k in (1, 2, 3))
    decay = np.exp(-x)
    drive = velocity @ coupling
    left, change = drive[:-1], np.diff(drive, axis=0)
    state = np.zeros_like(rates)
    integral = np.empty_like(left)
    states = np.zeros_like(drive) if return_states else None
    for i, (u, du) in enumerate(zip(left, change, strict=True)):
        integral[i] = step*(phi1*state+step*(phi2*u+phi3*du))
        state = decay*state+step*(phi1*u+phi2*du)
        if states is not None:
            states[i+1] = state
    free = step*phi1*np.exp(-np.arange(len(left))[:, None]*x)
    return (free, integral, states) if return_states else (free, integral)


def project_residual(operator, rhs):
    """Project onto the sampled memory span, retaining rank and minimum-norm diagnostics."""
    operator, rhs = (np.asarray(x, dtype=np.float64) for x in (operator, rhs))
    if (operator.ndim != 2 or rhs.shape != (operator.shape[0],) or not operator.shape[1]
            or operator.shape[0] < operator.shape[1] or not np.isfinite(operator).all() or not np.isfinite(rhs).all()):
        raise ValueError("finite tall momentum operator and right-hand side required")
    u, singular, vh = np.linalg.svd(operator, full_matrices=False)
    threshold = float(64*np.finfo(float).eps*max(operator.shape)*singular[0])
    rank = int((singular > threshold).sum())
    basis = u[:, :rank]
    component = basis.T @ rhs
    state = vh[:rank].T @ (component/singular[:rank])
    return {"state": state, "perpendicular": rhs-basis @ component, "rank": rank, "singular": singular,
            "threshold": threshold, "basis": basis, "u": u, "vh": vh}


def momentum_system(path, power, equilibrium, blocks, step, *, memory_inertia=1., restoring=False):
    """Assemble the mass- and time-scaled weak equation and signed force directions.

    With ``restoring=True``, also return the exact integrated reverse-restoring
    force direction while leaving the legacy right-hand side unchanged.
    """
    mass, incidence, stiffness, damping, coupling, tau = blocks
    path, power, equilibrium = (np.asarray(x, dtype=np.float64) for x in (path, power, equilibrium))
    n, edges = incidence.shape
    if (not isinstance(restoring, bool) or path.ndim != 2 or path.shape[1] != 2*n or len(path) < 2 or power.shape != (n,)
            or equilibrium.shape != (n,) or mass.shape != (n,) or stiffness.shape != (edges,)
            or damping.shape != (n, n) or coupling.shape != (n, len(tau)) or np.any(mass <= 0)
            or not all(np.isfinite(x).all() for x in (*blocks, path, power, equilibrium))
            or not math.isfinite(memory_inertia) or memory_inertia <= 0):
        raise ValueError("invalid observed path or fixed momentum blocks")
    q, v = path[:, :n], path[:, n:]
    projection = np.eye(n)-mass[:, None]/mass.sum()
    port = projection @ coupling
    edge_force = projection @ incidence
    ref_angle = equilibrium @ incidence
    angle = q @ incidence+ref_angle
    middle = .5*(angle[1:]+angle[:-1])
    change = np.diff(angle, axis=0)
    sine = step*(np.sin(middle)*np.sinc(change/(2*np.pi))-np.sin(ref_angle))
    harmonic = step*(np.sin(2*middle)*np.sinc(change/np.pi)-np.sin(2*ref_angle))
    free, driven = memory_integrals(v, coupling/memory_inertia, 1./(memory_inertia*tau), step)
    stiffness_force = -(sine*stiffness) @ edge_force.T
    damping_force = -(.5*step*(v[1:]+v[:-1])) @ (projection @ damping).T
    forcing = step*(projection @ power)+stiffness_force+damping_force-driven @ port.T
    weights = 1./(mass*step*math.sqrt(len(path)-1))
    operator = -free[:, None, :]*port[None, :, :]*weights[None, :, None]
    rhs = (np.diff(v, axis=0)*mass-forcing)*weights
    directions = {"stiffness": (stiffness_force*weights).ravel(), "damping": (damping_force*weights).ravel(),
                  "second_harmonic": ((-(harmonic*stiffness) @ edge_force.T)*weights).ravel()}
    if restoring:
        psi = np.cos(ref_angle)*harmonic-2.*np.cos(2.*ref_angle)*sine
        restoring_force = -(psi*stiffness) @ edge_force.T
        directions["restoring"] = (restoring_force*weights).ravel()
    return operator.reshape(rhs.size, -1), rhs.ravel(), directions


def inspect_momentum(path, power, equilibrium, blocks, step, spend, *, radius=None, memory_inertia=1.):
    """Report the hidden-state lower bound and projected candidate force statistics."""
    spend("weak_intervals", len(path)-1)
    operator, rhs, directions = momentum_system(path, power, equilibrium, blocks, step, memory_inertia=memory_inertia)
    spend("svd")
    projection = project_residual(operator, rhs)
    perpendicular, basis = projection["perpendicular"], projection["basis"]
    spectrum, rank = projection["singular"], projection["rank"]
    result = {"intervals": len(path)-1, "rank": rank, "singular": spectrum.tolist(),
              "condition": float(spectrum[0]/spectrum[-1]) if rank == operator.shape[1] else None,
              "rhs_sse": float(rhs@rhs), "orthogonal_sse": float(perpendicular@perpendicular),
              "algebraic_state": projection["state"].tolist(),
              "algebraic_norm": float(np.linalg.norm(projection["state"])), "directions": {}}
    for name, direction in directions.items():
        projected = direction-basis @ (basis.T @ direction)
        result["directions"][name] = {"sse": float(projected@projected), "cross": float(projected@perpendicular),
                                      "raw_sse": float(direction@direction)}
    if radius is not None:
        spend("ball_solves")
        svd = {"u": projection["u"], "s": spectrum, "vh": projection["vh"], "rank": rank,
               "threshold": projection["threshold"]}
        try:
            solved = solve_energy_ball(operator, rhs, svd, radius, projection["state"])
        except EnergyBallNumericalError as error:
            result["ball"] = {"status": "numerical_failure", "residual_sse": None, "reason": str(error)}
        else:
            residual = solved["residual_norm"]
            result["ball"] = {"status": solved["status"], "residual_sse": None if residual is None else residual**2,
                              "constraint_active": solved["constraint_active"],
                              "state": None if solved["z"] is None else solved["z"].tolist()}
    return result


def direction_changes(records, groups):
    """Fit each scalar to train statistics, then apply it to each unchanged group."""
    result = {}
    train = [row for row in records if row["role"] == "train"]
    for name in records[0]["directions"]:
        size = sum(row["directions"][name]["sse"] for row in train)
        cross = sum(row["directions"][name]["cross"] for row in train)
        change = cross/size if size > 0 else None
        scores = {}
        for group, rows in groups.items():
            base = sum(row["orthogonal_sse"] for row in rows)
            after = None if change is None else sum(max(0., row["orthogonal_sse"]
                         -2*change*row["directions"][name]["cross"]+change**2*row["directions"][name]["sse"])
                         for row in rows)
            scores[group] = {"events": len(rows), "baseline_sse": base, "corrected_sse": after,
                             "ratio": after/base if after is not None and base > 0 else None}
        result[name] = {"delta_multiplier": change, "by_group": scores}
    return result


def origin_transfer(path, power, equilibrium, blocks, step, spend, *, radius):
    """Score frozen history-only memory starts on subsequent observed intervals."""
    history_intervals = 10
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("a positive finite history radius is required")
    if len(path) <= history_intervals+1:
        return {"status": "missing_future", "intervals": 0}
    history = path[:history_intervals+1]
    fits = []
    for stride in (1, 2):
        observed = history[::stride]
        spend("weak_intervals", len(observed)-1)
        operator, rhs, _ = momentum_system(observed, power, equilibrium, blocks, stride*step)
        spend("svd")
        fits.append((operator, rhs, project_residual(operator, rhs)))
    operator, rhs, fine = fits[0]
    coarse = fits[1][2]
    ranks = [fit[2]["rank"] for fit in fits]
    intervals = len(path)-1-history_intervals
    result = {"ranks": ranks, "intervals": intervals}
    if any(rank != operator.shape[1] for rank in ranks):
        return result | {"status": "unresolved"}
    svd = {"u": fine["u"], "s": fine["singular"], "vh": fine["vh"], "rank": fine["rank"],
           "threshold": fine["threshold"]}
    spend("ball_solves")
    try:
        ball = solve_energy_ball(operator, rhs, svd, radius, fine["state"])
    except EnergyBallNumericalError as error:
        return result | {"status": "numerical_failure", "reason": str(error)}
    if ball["z"] is None:
        return result | {"status": ball["status"]}
    spend("weak_intervals", len(path)-1)
    operator, rhs, _ = momentum_system(path, power, equilibrium, blocks, step)
    # The full operator carries the free and driven memory across 100 ms. Only
    # rows after that boundary enter the score, and neither start is refit.
    cutoff = history_intervals*len(power)
    starts = (fine["state"], ball["z"], coarse["state"])
    residuals = rhs[cutoff:, None]-operator[cutoff:] @ np.column_stack(starts[:2])
    future_sse = np.sum(residuals**2, axis=0)*(len(path)-1)/intervals
    difference = np.linalg.norm(starts[0]-starts[2])/max(radius, np.linalg.norm(starts[0]))
    if not np.isfinite(future_sse).all() or not np.isfinite(difference):
        return result | {"status": "numerical_failure", "reason": "nonfinite origin contrast"}
    return result | {"status": "solved", "history_start": [state.tolist() for state in starts],
                     "constraint_active": ball["constraint_active"], "sampling_difference": float(difference),
                     "future_sse": dict(zip(("unconstrained", "constrained"), future_sse.tolist(), strict=True))}
