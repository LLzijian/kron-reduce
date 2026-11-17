"""Independent NumPy equations for the joint ES-SINDy synthetic gate."""
from __future__ import annotations

import numpy as np
from scipy.linalg import helmert

from kron_reduce.evaluation.machine_local_topology_truth import _incidence


def operators(mass):
    incidence = _incidence()
    node_edges = np.abs(incidence)
    basis = helmert(10, full=False).T
    spatial = node_edges.T @ basis
    spatial -= spatial.mean(axis=0)
    limit = np.log(2.)/2.
    radius = limit/np.linalg.norm(spatial, axis=1).max()
    projection = np.eye(10)-np.outer(mass, np.ones(10))/mass.sum()
    return incidence, node_edges/node_edges.sum(axis=1, keepdims=True), basis, spatial, limit, radius, projection


def port_and_edges(state, mass, coefficients):
    incidence, incident, basis, spatial, limit, radius, _ = operators(mass)
    edge = state[..., :10] @ incidence
    common, xi = state[..., 40:41], state[..., 41:]
    radial = xi/np.sqrt(1.+np.square(xi).sum(-1, keepdims=True)/radius**2)
    shared = limit*np.tanh(2.*common/limit)
    redistribution = radial @ spatial.T
    local = edge**2 @ incident.T
    amplitude = np.exp(coefficients["port_topology"]*np.tanh(common+xi@basis.T))
    amplitude /= 1.+coefficients["saturation"]*local
    ports = np.asarray(coefficients["ports"])*amplitude[..., None, :]
    return shared, redistribution, ports


def truth_rollout(mass, initial, power, equilibrium, coefficients, *, count, step, spend, fifth=0.):
    """RK4 with independent physical coefficients and a truth-only fifth harmonic."""
    incidence, incident, basis, spatial, limit, radius, projection = operators(mass)
    ref = equilibrium @ incidence
    tau, ports = (np.asarray(coefficients[key]) for key in ("tau", "ports"))
    decay_factor = np.asarray(coefficients["memory_decay_factor"])
    stiffness = np.asarray(coefficients["stiffness"])
    damping = np.asarray(coefficients["node_damping"])
    mobility, storage = (np.asarray(coefficients[key]) for key in ("mobility", "storage"))
    state = np.array(initial, dtype=np.float64, copy=True)
    output = np.empty((len(state), count+1, 50))
    output[:, 0] = state

    def rhs(x):
        q, v, z1, z2, common, xi = np.split(x, (10, 20, 30, 40, 41), axis=-1)
        edge = q@incidence
        angle = edge+ref
        base_energy = np.cos(ref)-np.cos(angle)-edge*np.sin(ref)
        base_force = np.sin(angle)-np.sin(ref)
        energy, restoring = base_energy, base_force
        for order, coefficient in zip((2, 3, 5), (*coefficients["harmonics"], fifth), strict=True):
            phase = order*ref
            s = (np.cos(phase)-np.cos(order*angle))/order-edge*np.sin(phase)
            force = np.sin(order*angle)-np.sin(phase)
            curvature = order*np.cos(phase)
            energy = energy+coefficient*(np.cos(ref)*s-curvature*base_energy)
            restoring = restoring+coefficient*(np.cos(ref)*force-curvature*base_force)
        c = np.tanh(2.*common/limit)
        scale = 1./np.sqrt(1.+np.sum(xi**2, axis=-1, keepdims=True)/radius**2)
        radial = xi*scale
        k = stiffness*np.exp(limit*c+radial@spatial.T)
        q2 = edge**2
        local = q2@incident.T
        amp = np.exp(coefficients["port_topology"]*np.tanh(common+xi@basis.T))/(1.+coefficients["saturation"]*local)
        amplitudes = ports*amp[:, None, :]
        velocity = v@projection
        memory = np.stack((z1, z2), axis=1)
        feedback = np.sum(amplitudes*memory, axis=1)
        flat_memory = memory.reshape(-1, 20)
        memory_dot = (amplitudes*velocity[:, None, :]-memory/tau).reshape(-1, 20)
        memory_dot -= (flat_memory@decay_factor)[:, None]*decay_factor
        drag = (damping*(1.-coefficients["angle_damping"]*local/(1.+local))*velocity)@projection.T
        drag += (coefficients["edge_damping"]*(q2/(1.+q2))*(v@incidence))@incidence.T
        drag += coefficients["speed_damping"]*v**3/(1.+v**2)
        accel = ((power-(k*restoring)@incidence.T-drag-feedback)@projection.T)/mass
        edge_energy = k*energy
        spatial_force = edge_energy@spatial
        dcommon = 2.*(1.-c**2)*edge_energy.sum(-1, keepdims=True)+storage[0]*common
        dxi = scale*(spatial_force-radial*np.sum(radial*spatial_force, axis=-1, keepdims=True)/radius**2)+storage[1]*xi
        return np.concatenate((v, accel, memory_dot, -mobility[0]*dcommon, -mobility[1]*dxi), axis=1)

    substeps = round(.01/step)
    h = .01/substeps
    for frame in range(1, count+1):
        for _ in range(substeps):
            spend("numpy_rhs", 4)
            k1 = rhs(state)
            k2 = rhs(state+.5*h*k1)
            k3 = rhs(state+.5*h*k2)
            k4 = rhs(state+h*k3)
            state += (h/6.)*(k1+2.*k2+2.*k3+k4)
            for offset in (0, 10):
                block = state[:, offset:offset+10]
                block -= (block@mass/mass.sum())[:, None]
        output[:, frame] = state
    if not np.isfinite(output).all():
        raise FloatingPointError("nonfinite independent joint truth")
    return output
