"""Deterministic synthetic populations for local equation discovery."""
from __future__ import annotations

import numpy as np
import torch

from kron_reduce.models.machine_local_equation import JointMachineLocalModel
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence
from kron_reduce.physics.coordinates import coi_basis


def _numpy_truth_rollout(initial, power, equilibrium, *, step, stiffness, damping,
                         coupling, tau, harmonic_weights, damping_strength,
                         mass, incidence, projection, budget):
    """Independent batched RK4 implementation of the physical equation."""
    n, r = len(mass), len(tau)
    output = np.empty((len(initial), 811, 2*n+r), dtype=np.float64)
    state = np.asarray(initial, dtype=np.float64).copy()
    output[:, 0] = state
    reference = equilibrium @ incidence
    inv_mass = 1./mass
    substeps = round(.01/step)
    h = .01/substeps
    damping_edge_scale = (8./incidence.shape[1])*damping_strength

    def rhs(x):
        budget.spend("numpy_rhs")
        q, v, z = x[:, :n], x[:, n:2*n], x[:, 2*n:]
        edge = q @ incidence
        angle = edge+reference
        edge_force = stiffness*(np.sin(angle)-np.sin(reference))
        if np.any(harmonic_weights):
            edge_force += harmonic_weights*(np.sin(2*angle)-np.sin(2*reference))
        force = power-edge_force @ incidence.T-v @ damping.T-z @ coupling.T
        if damping_strength:
            weights = damping_edge_scale*np.sin(edge)**2
            force -= (weights*(v @ incidence)) @ incidence.T
        acceleration = (force @ projection.T)*inv_mass
        return np.concatenate((v, acceleration, v @ coupling-z/tau), axis=1)

    for frame in range(1, 811):
        for _ in range(substeps):
            k1 = rhs(state)
            k2 = rhs(state+.5*h*k1)
            k3 = rhs(state+.5*h*k2)
            k4 = rhs(state+h*k3)
            state += (h/6.)*(k1+2*k2+2*k3+k4)
            q, v = state[:, :n], state[:, n:2*n]
            state[:, :n] = q-(q @ mass)[:, None]/mass.sum()
            state[:, n:2*n] = v-(v @ mass)[:, None]/mass.sum()
        output[:, frame] = state
    return output


def make_population(name, budget, *, train_only=False):
    """Build one of three known-truth equation families on a shared 32-case design.

    The event assignment is fixed: seeded permutation positions 0:16 are train,
    16:24 val, and 24:32 test. Excitations are generated jointly but independently
    across events; no future inputs are supplied.
    """
    if name not in {"null", "harmonic", "damping"}:
        raise ValueError("name must be null, harmonic, or damping")
    n, r = 10, 10
    rng = np.random.default_rng(20261114)
    mass = 1. + .1*np.arange(n)
    incidence = complete_incidence(n)
    basis = coi_basis(np.ones(n))
    stiffness = np.full(incidence.shape[1], .12)
    damping_coefficients = np.full(n-1, .8)
    damping = (basis*damping_coefficients) @ basis.T
    port = np.full(n, .4)
    tau = np.geomspace(.06, 1.6, r)
    anchor = NonlinearPassiveClosure(
        mass, stiffness, damping, np.zeros((n, 0)), np.zeros((0, 0)),
        np.zeros((0, 0)), "sine",
    )
    truth = JointMachineLocalModel(
        anchor, port=port, tau=tau, harmonic=True, modulated_damping=True,
    )
    def rate_coordinate(value):
        fraction = np.log(value/.02) / np.log(5./.02)
        return np.log(fraction/(1-fraction))

    with torch.no_grad():
        edge_log_factor = np.log1p(.25*np.cos(.71*np.arange(1, 46)))
        damping_log_factor = np.log1p(.2*np.sin(.63*np.arange(1, 10)))
        port_log_factor = np.log(.35+.025*np.arange(10))
        rate_parameter = rate_coordinate(np.geomspace(.05, 2., r))
        truth.edge_log_factor.copy_(truth._tensor(edge_log_factor))
        truth.damping_log_factor.copy_(truth._tensor(damping_log_factor))
        truth.port_parameter.copy_(truth._tensor(port_log_factor))
        truth.rate_parameter.copy_(truth._tensor(rate_parameter))
        truth.harmonic_strength.fill_(.6 if name == "harmonic" else 0.)
        truth.modulated_damping_strength.fill_(1. if name == "damping" else 0.)

    projection = np.eye(n)-np.outer(mass, np.ones(n))/mass.sum()
    order = rng.permutation(32)
    q = rng.uniform(-.6, .6, size=(32, n)) @ projection
    v = rng.uniform(-.65, .65, size=(32, n))
    fast = order[::4]
    v[fast, np.arange(len(fast))] += 16.
    v = v @ projection
    z = rng.uniform(-.35, .35, size=(32, r))
    power = .045*rng.normal(size=(32, n)) @ projection.T
    equilibrium = rng.uniform(-.15, .15, size=(32, n)) @ projection
    initial = np.concatenate((q, v, z), axis=1)

    # Build the reference coefficients independently of the Torch model state.
    true_stiffness = stiffness*np.exp(edge_log_factor)
    true_damping_coefficients = damping_coefficients*np.exp(damping_log_factor)
    true_damping = (basis*true_damping_coefficients) @ basis.T
    true_coupling = projection*(.35+.025*np.arange(n))[None, :]
    true_tau = np.geomspace(.05, 2., r)
    harmonic_strength = .6 if name == "harmonic" else 0.
    damping_strength = 1. if name == "damping" else 0.
    split_indices = {"train": order[:16], "val": order[16:24], "test": order[24:]}
    # Select before integration so a train-only diagnosis generates no held-out trajectories.
    chosen = order[:16] if train_only else np.arange(len(initial))
    if train_only:
        split_indices = {"train": np.arange(len(chosen))}
    initial, power, equilibrium = initial[chosen], power[chosen], equilibrium[chosen]
    paths = {}
    for label, step in (("fine", .00125), ("refined", .000625)):
        paths[label] = _numpy_truth_rollout(
            initial, power, equilibrium, step=step, stiffness=true_stiffness,
            damping=true_damping, coupling=true_coupling, tau=true_tau,
            harmonic_weights=harmonic_strength*stiffness, damping_strength=damping_strength,
            mass=mass, incidence=incidence, projection=projection, budget=budget,
        )
    if any(not np.isfinite(path).all() for path in paths.values()):
        raise FloatingPointError("nonfinite known-truth synthetic trajectory")

    # Train-only scales use 0.5 s and 1 s windows beginning at the 100 ms origin.
    train_indices = split_indices["train"]
    visible = paths["refined"][:, :, :2*n]
    target_scale = np.maximum(
        np.sqrt(np.mean(visible[train_indices, 11:111]**2, axis=(0, 1))), 1e-5
    )
    history_scale = np.maximum(
        np.sqrt(np.mean(visible[train_indices, :11]**2, axis=(0, 1))), 1e-5
    )
    roles, role_paths = {}, {}
    for role, indices in split_indices.items():
        role_paths[role] = {key: value[indices] for key, value in paths.items()}
        roles[role] = [
            {"id": f"{name}-{int(chosen[index]):02d}",
             "x": paths["refined"][index, :, :2*n],
             "p0": power[index], "equilibrium": equilibrium[index],
             "truth_index": int(chosen[index])}
            for index in indices
        ]
    return {
        "name": name, "anchor": anchor, "truth_model": truth,
        "port": port, "tau": tau, "stiffness": stiffness,
        "damping_coefficients": damping_coefficients,
        "truth_parameters": {
            "edge_log_factor": edge_log_factor.copy(),
            "damping_log_factor": damping_log_factor.copy(),
            "port_log_factor": port_log_factor.copy(),
            "rate_parameter": rate_parameter.copy(),
            "harmonic_strength": truth.harmonic_strength.detach().item(),
            "modulated_damping_strength": truth.modulated_damping_strength.detach().item(),
        },
        "paths": paths, "role_paths": role_paths, "roles": roles,
        "power": power, "equilibrium": equilibrium,
        "initial": initial, "history_scale": history_scale,
        "target_scale": target_scale, "assignment_seed": 20261114,
        "time_step_s": .01, "history_frames": 11,
    }
