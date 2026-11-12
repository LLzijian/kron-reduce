"""Recover the whole COI initial state instead of fixing a noisy first frame."""
from __future__ import annotations

import numpy as np
import torch


def initial_state_basis(mass):
    """Deterministic Householder coordinates perpendicular to the mass vector."""
    m = np.asarray(mass, dtype=np.float64)
    if m.ndim != 1 or len(m) < 2 or not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError("positive finite masses are required")
    w = m / np.linalg.norm(m)
    w[0] += 1.
    v = (np.eye(len(m)) - 2*np.outer(w, w)/np.sum(w*w))[:, 1:]
    n = len(m)
    basis = np.zeros((3*n, 3*n-2))
    basis[:n, :n-1] = v
    basis[n:2*n, n-1:2*n-2] = v
    basis[2*n:, 2*n-2:] = np.eye(n)
    return basis


def estimate_joint_history(model, q, v, power, equilibrium, scale, *, max_step, spend):
    """Use all eleven frames, including frame zero, for one joint state LS."""
    if model.backbone_kind != "linear" or model.latent_dimension != model.n_nodes:
        raise ValueError("joint observer requires a linear model with one hidden state per node")
    q, v, power, equilibrium, scale = [np.asarray(x, dtype=np.float64) for x in
                                      (q, v, power, equilibrium, scale)]
    n, batch = model.n_nodes, len(q)
    if (q.shape != (batch, 11, n) or not batch or v.shape != q.shape
            or power.shape != (batch, n) or equilibrium.shape != power.shape
            or scale.shape != (2*n,) or np.any(scale <= 0)
            or not all(np.isfinite(x).all() for x in (q, v, power, equilibrium, scale))):
        raise ValueError("invalid causal joint observation problem")
    basis = initial_state_basis(model.mass.numpy())
    coordinates = basis.shape[1]
    initial = np.zeros((batch+coordinates, 3*n))
    initial[batch:] = basis.T
    with torch.no_grad():
        history = model.rollout(initial, np.concatenate((power, np.zeros((coordinates, n)))),
            np.concatenate((equilibrium, np.zeros((coordinates, n)))), 10,
            max_step=max_step, spend=spend).numpy()
    baseline, response = history[:batch], history[batch:].transpose(1, 2, 0)
    observed = np.concatenate((q, v), axis=2)
    design = (response[:, :2*n]/scale[None, :, None]).reshape(-1, coordinates)
    rhs = ((observed-baseline[:, :, :2*n])/scale).reshape(batch, -1)
    spend("svd")
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    tolerance = 64*np.finfo(float).eps*max(design.shape)*singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0]/singular[-1]) if rank == coordinates else None
    resolved = rank == coordinates and condition <= 1e8
    result = {"resolved": resolved, "rank": rank, "rank_threshold": float(tolerance),
              "condition_number": condition, "design": design, "rhs": rhs,
              "u": u, "singular": singular, "vh": vh, "initial_basis": basis,
              "zero_history": baseline, "history_response": response}
    if resolved:
        spend("matrix_products", 3)
        estimate = ((rhs@u)/singular)@vh
        residual = estimate@design.T-rhs
        spend("matrix_products", 11)
        closed = baseline+np.einsum("tir,br->bti", response, estimate)
        if not np.isfinite(closed).all() or np.max(np.abs(closed)) > 1e12:
            raise FloatingPointError("nonfinite joint history estimate")
        origin = closed[:, -1].copy()
        origin[:, :n], origin[:, n:2*n] = q[:, -1], v[:, -1]
        result.update(initial_coordinates=estimate, hidden_start=closed[:, 0, 2*n:],
                      residual=residual, history=closed, origin=origin)
    return result
