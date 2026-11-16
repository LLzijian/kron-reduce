"""Independent NumPy truth solver for topology-memory ROM synthetic cases."""
from __future__ import annotations

import numpy as np

from kron_reduce.evaluation.machine_local_topology_truth import (
    _EDGES,
    _N,
    _helmert_basis,
    _incidence,
)

_LIMIT = np.log(2.)/2.
_RHO = 2./3.


def truth_rollout(params, q, v, power, z, *, count, max_step, spend,
                  topology="none", c=None, xi=None, gamma_s=0.0,
                  topology_dynamics=None, fifth_harmonic=0.0,
                  restoring_harmonics=(0.0, 0.0)):
    """Integrate fixed-input synthetic trajectories with topology and memory states.

    ``topology`` selects ``none``, ``common``, ``spatial`` or ``both``. When a
    topology state is active, ``topology_dynamics`` is ``(mobility, stiffness)``;
    each is a pair ordered as common, spatial. ``gamma_s=0`` gives constant memory
    ports. ``fifth_harmonic`` is a truth-only coefficient for the out-of-library
    fifth harmonic and is restricted to topology-off, constant-port cases. The
    optional ``restoring_harmonics`` contains signed second- and third-harmonic
    restoring coefficients. The fixed damping baseline uses the existing local
    angle law at rho=2/3.
    """
    mass, k0, damping, g, tau, equilibrium = (
        np.asarray(value, dtype=np.float64) for value in params
    )
    q, v, power, z = (np.asarray(value, dtype=np.float64) for value in (q, v, power, z))
    try:
        restoring_harmonics = np.asarray(restoring_harmonics, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("restoring_harmonics must contain two finite coefficients") from None
    if topology not in ("none", "common", "spatial", "both"):
        raise ValueError("topology must be none, common, spatial, or both")
    batch = len(q) if q.ndim else 0
    expected = (_N,)
    if (q.shape != (batch, _N) or v.shape != q.shape or power.shape != q.shape or z.shape != (batch, _N)
            or mass.shape != expected or k0.shape != (len(_EDGES),) or damping.shape != (_N, _N)
            or g.shape != expected or equilibrium.shape != expected or tau.shape != expected
            or not all(np.isfinite(value).all() for value in (mass, k0, damping, g, tau, equilibrium, q, v, power, z))
            or np.any(mass <= 0) or np.any(g <= 0) or np.any(tau <= .02) or np.any(tau >= 5.)
            or type(count) is not int or count < 1 or not np.isfinite(max_step) or max_step <= 0
            or not np.isfinite(gamma_s) or gamma_s < 0 or not np.isfinite(fifth_harmonic)
            or abs(fifth_harmonic) > .05 or restoring_harmonics.shape != (2,)
            or not np.isfinite(restoring_harmonics).all()):
        raise ValueError("invalid topology-memory truth inputs")
    if fifth_harmonic and (topology != "none" or gamma_s != 0.):
        raise ValueError("fifth-harmonic truth requires topology and port saturation to be off")
    if not callable(spend):
        raise TypeError("spend must be callable")

    has_common, has_spatial = topology in ("common", "both"), topology in ("spatial", "both")
    if has_common:
        c = np.asarray(c, dtype=np.float64)
        if c.ndim == 0 and batch == 1:
            c = c.reshape(1)
        if c.shape != (batch,) or not np.isfinite(c).all():
            raise ValueError("c must contain one finite common topology state per event")
    elif c is not None:
        raise ValueError("c is only valid when common topology is active")
    if has_spatial:
        xi = np.asarray(xi, dtype=np.float64)
        if xi.shape != (batch, _N-1) or not np.isfinite(xi).all():
            raise ValueError("xi must contain one finite zero-sum coordinate vector per event")
    elif xi is not None:
        raise ValueError("xi is only valid when spatial topology is active")
    if has_common or has_spatial:
        if (not isinstance(topology_dynamics, (tuple, list)) or len(topology_dynamics) != 2
                or any(np.asarray(values).shape != (2,) for values in topology_dynamics)):
            raise ValueError("topology_dynamics must contain (common, spatial) mobility and stiffness pairs")
        mobility, stiffness = (np.asarray(values, dtype=np.float64) for values in topology_dynamics)
        if (not np.isfinite(mobility).all() or not np.isfinite(stiffness).all()
                or np.any(mobility < 0.) or np.any(stiffness <= 0.)):
            raise ValueError("topology mobility must be nonnegative and stiffness positive")
    elif topology_dynamics is not None:
        raise ValueError("topology_dynamics requires an active topology state")

    incidence = _incidence()
    projection = np.eye(_N)-np.outer(mass, np.ones(_N))/mass.sum()
    zero_sum_basis = _helmert_basis(_N)
    edge_basis = np.abs(incidence).T @ zero_sum_basis
    edge_norm = float(np.max(np.linalg.norm(edge_basis, axis=1)))
    if not np.allclose(damping, damping.T, rtol=0., atol=1e-12):
        raise ValueError("local angle damping requires symmetric D0")
    spend("numpy_damping_roots")
    spectrum, modes = np.linalg.eigh(damping)
    tolerance = 64*np.finfo(float).eps*max(1., float(np.max(np.abs(spectrum))))
    if np.min(spectrum) < -tolerance:
        raise ValueError("local angle damping requires a positive-semidefinite baseline")
    root = (modes*np.sqrt(np.maximum(spectrum, 0.))) @ modes.T
    incident = np.abs(incidence)
    incident /= incident.sum(axis=1, keepdims=True)
    ref = equilibrium @ incidence
    tau = tau[:z.shape[1]]
    substeps = max(1, int(np.ceil(.01/max_step-1e-12)))
    h = .01/substeps

    state_parts = [q.copy(), v.copy(), z.copy()]
    if has_common:
        state_parts.append(c[:, None].copy())
    if has_spatial:
        state_parts.append(xi.copy())
    state = np.concatenate(state_parts, axis=1)
    c_size = int(has_common)
    spatial_size = (_N-1) if has_spatial else 0
    path = np.empty((batch, count+1, state.shape[1]), dtype=np.float64)
    path[:, 0] = state
    z_start = 2*_N
    c_start = z_start+z.shape[1]
    xi_start = c_start+c_size

    def rhs(x):
        xq, xv = x[:, :_N], x[:, _N:2*_N]
        xz = x[:, z_start:z_start+z.shape[1]]
        edge = xq @ incidence
        s_i = (edge*edge) @ incident.T
        h_i = 1./(1.+gamma_s*s_i)
        amplitude = g*h_i

        ell = np.zeros((batch, len(_EDGES)), dtype=np.float64)
        if has_common:
            xc = x[:, c_start]
            ell += _LIMIT*np.tanh(2.*xc/_LIMIT)[:, None]
        if has_spatial:
            xxi = x[:, xi_start:xi_start+spatial_size]
            denom = np.sqrt(1.+(edge_norm/_LIMIT)**2*np.sum(xxi*xxi, axis=1, keepdims=True))
            ell += (xxi @ edge_basis.T)/denom
        edge_k = k0[None, :]*np.exp(ell)
        angle = edge+ref
        base_phi = np.cos(ref)-np.cos(angle)-edge*np.sin(ref)
        base_force = np.sin(angle)-np.sin(ref)
        phi, restoring = base_phi, base_force
        for harmonic, coefficient in enumerate(restoring_harmonics, start=2):
            if coefficient:
                phase = harmonic*ref
                phi_m = (np.cos(phase)-np.cos(harmonic*angle))/harmonic-edge*np.sin(phase)
                force_m = np.sin(harmonic*angle)-np.sin(phase)
                linear = harmonic*np.cos(phase)
                phi = phi+coefficient*(np.cos(ref)*phi_m-linear*base_phi)
                restoring = restoring+coefficient*(np.cos(ref)*force_m-linear*base_force)
        if fifth_harmonic:
            phi5 = ((np.cos(5.*ref)[None, :]-np.cos(5.*(edge+ref)))/5.
                    - edge*np.sin(5.*ref)[None, :])
            phi += fifth_harmonic*phi5
            restoring += fifth_harmonic*(np.sin(5.*(edge+ref))-np.sin(5.*ref)[None, :])
        gradient_q = incidence @ (edge_k*restoring).T
        conservative = projection @ gradient_q

        # D(q)=D0^(1/2)[I-rho diag(s/(1+s))]D0^(1/2), using local edge support.
        root_v = xv @ root
        damping_force = (root @ root_v.T).T
        damping_force -= _RHO*((s_i/(1.+s_i))*root_v) @ root
        damping_force = (projection @ damping_force.T).T

        force = projection @ power.T
        force = force.T-conservative.T-damping_force
        force -= (xz*amplitude) @ projection.T
        dz = (xv @ projection)*amplitude-xz/tau
        result = [xv, force/mass[None, :], dz]

        if has_common or has_spatial:
            grad = np.zeros((batch, _N), dtype=np.float64)
            if has_common:
                common_derivative = 2.*(1.-np.tanh(2.*x[:, c_start]/_LIMIT)**2)
                grad[:, 0] = np.sum(edge_k*phi, axis=1)*common_derivative
            if has_spatial:
                xxi = x[:, xi_start:xi_start+spatial_size]
                denom = np.sqrt(1.+(edge_norm/_LIMIT)**2*np.sum(xxi*xxi, axis=1, keepdims=True))
                raw = edge_basis[None, :, :]
                xi_term = np.einsum("bei,bi->be", raw, xxi)
                shrinkage = (edge_norm/_LIMIT)**2*xi_term/denom**3
                dell = raw/denom[:, :, None]-shrinkage[..., None]*xxi[:, None, :]
                grad[:, -(_N-1):] = np.einsum("be,bei->bi", edge_k*phi, dell)
            if has_common:
                xc = x[:, c_start]
                dc = -mobility[0]*(grad[:, 0]+stiffness[0]*xc)
                result.append(dc[:, None])
            if has_spatial:
                xxi = x[:, xi_start:xi_start+spatial_size]
                dxi = -mobility[1]*(grad[:, -(_N-1):]+stiffness[1]*xxi)
                result.append(dxi)
        return np.concatenate(result, axis=1)

    for frame in range(1, count+1):
        for _ in range(substeps):
            spend("numpy_rhs", 4)
            k1 = rhs(state)
            k2 = rhs(state+.5*h*k1)
            k3 = rhs(state+.5*h*k2)
            k4 = rhs(state+h*k3)
            state += (h/6.)*(k1+2.*k2+2.*k3+k4)
            state[:, :_N] -= ((state[:, :_N]@mass)/mass.sum())[:, None]
            state[:, _N:2*_N] -= ((state[:, _N:2*_N]@mass)/mass.sum())[:, None]
        path[:, frame] = state
    return path
