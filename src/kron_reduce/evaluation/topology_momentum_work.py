"""Per-interval impulse and work accounting on an observed machine path."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import hyp1f1

from kron_reduce.evaluation.topology_momentum import memory_integrals
from kron_reduce.evaluation.topology_spatial_momentum import _QUADRATURE

_RHO = 2.0 / 3.0


def account_path(path, power, equilibrium, blocks, step, start, root, points):
    """Account five signed impulses and their machine work along linear q/v paths.

    ``start`` is the hidden memory state at the first path sample.  The model's
    passive local angle-damping correction is fixed at rho=2/3.
    """
    mass, incidence, stiffness, damping, coupling, tau = (
        np.asarray(value, dtype=np.float64) for value in blocks
    )
    path, power, equilibrium, start, root = (
        np.asarray(value, dtype=np.float64)
        for value in (path, power, equilibrium, start, root)
    )
    n = len(mass)
    if (points not in _QUADRATURE or path.ndim != 2 or path.shape[1] != 2*n or len(path) < 2
            or incidence.ndim != 2 or incidence.shape[0] != n or stiffness.shape != (incidence.shape[1],)
            or damping.shape != (n, n) or coupling.ndim != 2 or coupling.shape[0] != n
            or tau.shape != (coupling.shape[1],) or power.shape != (n,) or equilibrium.shape != (n,)
            or start.shape != (coupling.shape[1],) or root.shape != (n, n)
            or not all(np.isfinite(value).all() for value in
                       (mass, incidence, stiffness, damping, coupling, tau, path, power, equilibrium, start, root))
            or np.any(mass <= 0) or np.any(tau <= 0) or not math.isfinite(step) or step <= 0
            or not np.allclose(root, root.T, rtol=0, atol=1e-12)
            or not np.allclose(root @ root, damping, rtol=1e-9, atol=1e-11)):
        raise ValueError("invalid path, blocks, memory start, damping root, or quadrature order")
    rates = 1.0 / tau

    q, velocity = path[:, :n], path[:, n:]
    intervals = len(path)-1
    projection = np.eye(n)-mass[:, None]/mass.sum()
    projected_coupling = projection @ coupling
    incident = np.abs(incidence)
    degree = incident.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("every machine must have positive topology degree")
    incident /= degree[:, None]
    nodes, weights = _QUADRATURE[points]
    fractions, weights = (nodes+1.)*.5, weights*.5
    names = ("applied", "restoring", "damping", "initial_memory", "driven_memory")
    impulses = {name: np.zeros((intervals, n)) for name in names}
    work = {name: np.zeros((intervals, n)) for name in names}
    free_integral, driven_integral, driven_state = memory_integrals(
        velocity, projected_coupling, rates, step, return_states=True)
    free_state = np.exp(-step*np.arange(len(path))[:, None]*rates)*start
    hidden_energy = .5*np.square(free_state+driven_state).sum(1)
    memory_dissipation = np.zeros(intervals)
    drive = velocity @ projected_coupling
    dq, dv, du = np.diff(q, axis=0), np.diff(velocity, axis=0), np.diff(drive, axis=0)
    reference = equilibrium @ incidence
    applied = projection @ power
    edge_force = projection @ incidence
    drag = projection @ damping
    projected_root = projection @ root
    for fraction, weight in zip(fractions, weights*step, strict=True):
        relative = (q[:-1]+fraction*dq) @ incidence
        vx = velocity[:-1]+fraction*dv
        force = (np.sin(reference+relative)-np.sin(reference))*stiffness
        local_angle = relative**2 @ incident.T
        local_strength = _RHO*local_angle/(1.+local_angle)
        s = fraction*step
        decay = np.exp(-rates*s)
        p1, p2 = hyp1f1(1., 2., -rates*s), .5*hyp1f1(1., 3., -rates*s)
        z_free = free_state[:-1]*decay
        z_driven = driven_state[:-1]*decay+s*(p1*drive[:-1]+fraction*p2*du)
        terms = (applied, -force @ edge_force.T, -vx @ drag.T+(local_strength*(vx @ root)) @ projected_root.T,
                 -z_free @ projected_coupling.T, -z_driven @ projected_coupling.T)
        for name, force_x in zip(names, terms, strict=True):
            impulses[name] += weight*force_x
            work[name] += weight*force_x*vx
        memory_dissipation += weight*np.square(z_free+z_driven) @ rates
    memory_quadrature = max(float(np.max(np.abs(a-b))) for a, b in (
        (impulses["initial_memory"], -(free_integral*start) @ projected_coupling.T),
        (impulses["driven_memory"], -driven_integral @ projected_coupling.T)))
    if not math.isfinite(memory_quadrature):
        raise FloatingPointError("nonfinite memory impulse quadrature")
    memory_balance = (np.diff(hidden_energy)+work["initial_memory"].sum(1)
                      +work["driven_memory"].sum(1)+memory_dissipation)

    total_impulse = sum(impulses.values())
    momentum_residual = mass[None, :]*np.diff(velocity, axis=0)-total_impulse
    kinetic_change = .5*mass[None, :]*(velocity[1:]**2-velocity[:-1]**2)
    kinetic_residual = kinetic_change-sum(work.values())
    return {"impulses": impulses, "work": work,
            "momentum_residual": momentum_residual, "kinetic_residual": kinetic_residual,
            "hidden_energy": hidden_energy, "memory_dissipation": memory_dissipation,
            "memory_balance_residual": memory_balance, "memory_impulse_quadrature_gap": memory_quadrature}
