"""Independent NumPy truth for paired machine-local topology and memory tests."""
from __future__ import annotations

import numpy as np

_N = 10
_EDGES = np.array([(i, j) for i in range(_N) for j in range(i + 1, _N)])


def parameters():
    """Return fixed mass, complete-graph stiffness, damping, port, rates and equilibrium."""
    mass = 1. + .1*np.arange(_N)
    k0 = np.full(len(_EDGES), .12)
    basis = np.eye(_N) - np.ones((_N, _N))/_N
    damping = (basis * .8) @ basis.T
    g = np.linspace(.45, .72, _N)
    tau = np.geomspace(.07, .65, _N)
    equilibrium = np.zeros(_N)
    return mass, k0, damping, g, tau, equilibrium


def truth_rollout(params, q, v, power, a, z, *, count, max_step, spend, angle_damping_rho=0., spatial_mix=0.,
                  restoring_alpha=0., dynamic_coupling=None, coupling_initial=None, spatial_coupling=False):
    """Integrate batched visible and optional latent states at a 10 ms frame rate."""
    mass, k0, damping, g, tau, equilibrium = (np.asarray(x, dtype=np.float64) for x in params)
    q, v, power, a, z = (np.asarray(x, dtype=np.float64) for x in (q, v, power, a, z))
    batch = len(q)
    if (q.shape != (batch, _N) or v.shape != q.shape or power.shape != q.shape
            or a.shape != q.shape or z.ndim != 2 or z.shape[0] != batch
            or mass.shape != (_N,) or damping.shape != (_N, _N) or g.shape != (_N,)
            or equilibrium.shape != (_N,) or k0.shape != (len(_EDGES),)
            or z.shape[1] > len(tau) or count < 1 or max_step <= 0
            or not 0. <= angle_damping_rho <= 2./3. or not 0. <= spatial_mix <= 1.
            or not -.25 <= restoring_alpha <= .25):
        raise ValueError("incompatible topology truth inputs")
    if not isinstance(spatial_coupling, bool):
        raise TypeError("spatial_coupling must be a boolean")
    if dynamic_coupling is None:
        if spatial_coupling:
            raise ValueError("spatial_coupling requires dynamic_coupling")
        if coupling_initial is not None:
            raise ValueError("coupling_initial requires dynamic_coupling")
        dynamic = None
    else:
        try:
            dynamic = tuple(float(x) for x in dynamic_coupling)
        except (TypeError, ValueError):
            raise ValueError("dynamic_coupling must be (mobility, stiffness, limit)") from None
        if (len(dynamic) != 3 or not np.isfinite(dynamic).all()
                or dynamic[0] < 0. or any(x <= 0. for x in dynamic[1:])):
            raise ValueError("dynamic coupling needs finite nonnegative mobility and positive stiffness and limit")
        if coupling_initial is None:
            c_initial = (np.zeros((batch, _N-1), dtype=np.float64) if spatial_coupling
                         else np.zeros(batch, dtype=np.float64))
        else:
            c_initial = np.asarray(coupling_initial, dtype=np.float64)
            expected_shape = (batch, _N-1) if spatial_coupling else (batch,)
            if not spatial_coupling and c_initial.ndim == 0 and batch == 1:
                c_initial = c_initial.reshape(1)
            if c_initial.shape != expected_shape or not np.isfinite(c_initial).all():
                detail = ("one finite zero-sum coordinate vector per event" if spatial_coupling
                          else "one finite scalar per event")
                raise ValueError(f"coupling_initial must be {detail}")
    if not all(np.isfinite(x).all() for x in (mass, k0, damping, g, tau, equilibrium, q, v, power, a, z)):
        raise ValueError("topology truth inputs must be finite")
    substeps = max(1, int(np.ceil(.01/max_step - 1e-12)))
    tau = tau[:z.shape[1]]
    h = .01/substeps
    projection = np.eye(_N) - np.outer(mass, np.ones(_N))/mass.sum()
    coupling = projection @ np.diag(g)
    incidence = _incidence()
    if spatial_mix and angle_damping_rho:
        spend("numpy_damping_roots")
        spectrum, modes = np.linalg.eigh(damping)
        tolerance = 64*np.finfo(float).eps*np.max(np.abs(spectrum))
        if np.min(spectrum) < -tolerance or np.max(np.abs(damping-damping.T)) > tolerance:
            raise ValueError("spatial truth requires a symmetric PSD damping matrix")
        root = (modes*np.sqrt(spectrum.clip(min=0.))) @ modes.T
    ref = equilibrium[_EDGES[:, 0]] - equilibrium[_EDGES[:, 1]]
    edge_k = k0[None, :] * np.exp(a[:, _EDGES[:, 0]] + a[:, _EDGES[:, 1]])
    if spatial_coupling:
        zero_sum_basis = _helmert_basis(_N)
        edge_basis = np.abs(incidence).T @ zero_sum_basis
    state_parts = [q.copy(), v.copy(), z.copy()]
    if dynamic is not None:
        state_parts.append(c_initial.copy() if spatial_coupling else c_initial[:, None].copy())
    state = np.concatenate(state_parts, axis=1)
    dynamic_dimension = ((_N-1) if spatial_coupling else 1) if dynamic is not None else 0
    path = np.empty((batch, count + 1, 2*_N + z.shape[1] + dynamic_dimension), dtype=np.float64)
    path[:, 0] = state

    def rhs(x):
        xq, xv = x[:, :_N], x[:, _N:2*_N]
        xz = x[:, 2*_N:2*_N + z.shape[1]]
        edge = xq[:, _EDGES[:, 0]] - xq[:, _EDGES[:, 1]]
        shifted = np.sin(edge + ref) - np.sin(ref)
        if restoring_alpha:
            harmonic = np.sin(2.*(edge+ref))-np.sin(2.*ref)
            shifted = shifted+restoring_alpha*(np.cos(ref)*harmonic-2.*np.cos(2.*ref)*shifted)
        effective_edge_k = edge_k
        if dynamic is not None and spatial_coupling:
            mobility, stiffness, limit = dynamic
            xi = x[:, 2*_N + z.shape[1]:]
            radial_scale = np.sqrt(1.+np.sum(xi*xi, axis=1, keepdims=True)/(limit*limit))
            radial = xi/radial_scale
            log_multiplier = radial @ edge_basis.T
            effective_edge_k = edge_k * np.exp(log_multiplier)
        elif dynamic is not None:
            mobility, stiffness, limit = dynamic
            xc = x[:, 2*_N + z.shape[1]]
            effective_edge_k = edge_k * np.exp(2.*limit*np.tanh(xc/limit))[:, None]
        force = power - (effective_edge_k*shifted) @ incidence.T
        drag = xv @ damping.T
        if angle_damping_rho:
            s = np.mean(edge**2, axis=1, keepdims=True)
            drag *= 1.-(1.-spatial_mix)*angle_damping_rho*s/(1.+s)
            if spatial_mix:
                differences = xq[:, :, None]-xq[:, None, :]
                local_s = np.sum(differences**2, axis=2)/(_N-1)
                drag -= ((xv @ root)*(spatial_mix*angle_damping_rho*local_s/(1.+local_s))) @ root
        force -= drag
        if xz.shape[1]:
            force -= xz @ coupling.T
            dz = xv @ coupling - xz/tau
        else:
            dz = xz
        result = [xv, (force @ projection.T)/mass, dz]
        if dynamic is not None:
            base_potential = np.cos(ref) - np.cos(edge + ref) - edge*np.sin(ref)
            if restoring_alpha:
                second = ((np.cos(2.*ref)-np.cos(2.*(edge+ref)))/2.
                          - edge*np.sin(2.*ref))
                base_potential = (base_potential + restoring_alpha*(
                    np.cos(ref)*second - 2.*np.cos(2.*ref)*base_potential))
            if spatial_coupling:
                gradient_radial = (effective_edge_k*base_potential) @ edge_basis
                xi_dot_gradient = np.sum(xi*gradient_radial, axis=1, keepdims=True)
                jacobian_gradient = (gradient_radial/radial_scale
                                     - xi*xi_dot_gradient/(limit*limit*radial_scale**3))
                dxi = -mobility*(stiffness*xi+jacobian_gradient)
                result.append(dxi)
            else:
                potential = np.sum(effective_edge_k*base_potential, axis=1)
                sech2 = 1. - np.tanh(xc/limit)**2
                dc = -mobility*(stiffness*xc + 2.*sech2*potential)
                result.append(dc[:, None])
        return np.concatenate(result, axis=1)

    for frame in range(1, count + 1):
        for _ in range(substeps):
            spend("numpy_rhs", 4)
            k1 = rhs(state)
            k2 = rhs(state + .5*h*k1)
            k3 = rhs(state + .5*h*k2)
            k4 = rhs(state + h*k3)
            state += (h/6.)*(k1 + 2*k2 + 2*k3 + k4)
            state[:, :_N] -= ((state[:, :_N] @ mass)/mass.sum())[:, None]
            state[:, _N:2*_N] -= ((state[:, _N:2*_N] @ mass)/mass.sum())[:, None]
        path[:, frame] = state
    return path


def _incidence():
    result = np.zeros((_N, len(_EDGES)))
    result[_EDGES[:, 0], np.arange(len(_EDGES))] = 1.
    result[_EDGES[:, 1], np.arange(len(_EDGES))] = -1.
    return result


def _helmert_basis(size):
    """Return scipy.linalg.helmert(size, full=False).T using NumPy only."""
    result = np.zeros((size, size-1), dtype=np.float64)
    for column in range(size-1):
        denominator = np.sqrt((column+1.)*(column+2.))
        result[:column+1, column] = 1./denominator
        result[column+1, column] = -(column+1.)/denominator
    return result


def initial_cases():
    """Return one deterministic shared 12-event design with train/val/test roles."""
    rng = np.random.default_rng(20261115)
    mass, *_ = parameters()
    projection = np.eye(_N) - np.outer(mass, np.ones(_N))/mass.sum()
    phase = 2*np.pi*np.arange(12)/12
    mode = np.arange(_N)
    q = (.22*np.sin(phase[:, None] + .71*mode) + .08*np.cos(2*phase[:, None] - .43*mode)) @ projection
    v = (.16*np.cos(phase[:, None] + .53*mode) + .06*np.sin(2*phase[:, None] + .37*mode)) @ projection
    q[::4] *= 1.55
    v[1::4] *= 1.5
    a = rng.normal(0., .095, size=(12, _N))
    a -= a.mean(axis=1, keepdims=True)
    a *= np.minimum(1., .20/np.max(np.abs(a), axis=1, keepdims=True))
    a *= np.minimum(1., .50/np.linalg.norm(a, axis=1, keepdims=True))
    p = rng.normal(0., .035, size=(12, _N)) @ projection.T
    z = rng.normal(0., .18, size=(12, _N))
    z *= np.minimum(1., .45/np.linalg.norm(z, axis=1, keepdims=True))
    return {
        "q": q, "v": v, "power": p, "a": a, "z": z,
        "ids": [f"topology-{i:02d}" for i in range(12)],
        "roles": ["train"]*6 + ["val"]*3 + ["test"]*3,
    }
