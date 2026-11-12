"""Exact encoder fitting for a fixed linear dynamics model and causal reset."""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve


def causal_affine_maps(model, q, v, power, equilibrium, count, *, max_step, spend):
    """Return zero-hidden paths and their response to the history-start state."""
    if model.backbone_kind != "linear":
        raise ValueError("affine initialization requires a linear backbone")
    q, v, power, equilibrium = map(np.asarray, (q, v, power, equilibrium))
    n, r, batch = model.n_nodes, model.latent_dimension, len(q)
    if q.shape != (batch, 11, n) or v.shape != q.shape:
        raise ValueError("eleven history frames are required")
    initial = np.zeros((batch+r, 2*n+r))
    initial[:batch, :n], initial[:batch, n:2*n] = q[:, 0], v[:, 0]
    initial[batch:, 2*n:] = np.eye(r)
    inputs = np.concatenate((power, np.zeros((r, n))))
    eq = np.concatenate((equilibrium, np.zeros((r, n))))
    with torch.no_grad():
        history = model.rollout(initial, inputs, eq, 10, max_step=max_step, spend=spend)
        origin = history[:, -1].clone()
        origin[:, :2*n] = 0
        origin[:batch, :n] = torch.as_tensor(q[:, -1])
        origin[:batch, n:2*n] = torch.as_tensor(v[:, -1])
        future = model.rollout(origin, inputs, eq, count, max_step=max_step, spend=spend)
    h, f = history.numpy(), future.numpy()
    return h[:batch], f[:batch], h[batch:].transpose(1, 2, 0), f[batch:].transpose(1, 2, 0)


def solve_conditional_encoder(train_features, train_residual, response, weights, scale,
                              *, penalty, spend):
    """Minimize weighted future MSE plus penalty*||E||² using train arrays only."""
    f, y, d, w, scale = [np.asarray(x, dtype=np.float64) for x in (
        train_features, train_residual, response, weights, scale)]
    if (f.ndim != 2 or y.ndim != 3 or d.ndim != 3 or not len(f)
            or y.shape[0] != len(f) or y.shape[1:] != d.shape[:2]
            or w.shape != (d.shape[0],) or scale.shape != (d.shape[1],)
            or min(f.shape[1], d.shape[0], d.shape[1], d.shape[2]) < 1
            or not np.isfinite(penalty) or penalty <= 0
            or np.any(scale <= 0) or np.any(w < 0) or not np.isclose(w.sum(), 1.)
            or not all(np.isfinite(x).all() for x in (f, y, d, w, scale))):
        raise ValueError("invalid finite train-only conditional encoder problem")
    count, channels = len(f), d.shape[1]
    normalized_d, normalized_y = d/scale[None, :, None], y/scale[None, None, :]
    spend("matrix_products", 4)
    gf = f.T@f/count
    gz = np.einsum("t,tir,tis->rs", w, normalized_d, normalized_d)/channels
    projected = np.einsum("t,tir,bti->br", w, normalized_d, normalized_y)/channels
    rhs = projected.T@f/count
    system = np.kron(gz, gf.T)+penalty*np.eye(rhs.size)
    spend("linear_solves")
    encoder = solve(system, rhs.reshape(-1), assume_a="pos").reshape(rhs.shape)
    spend("matrix_products")
    residual = system@encoder.reshape(-1)-rhs.reshape(-1)
    bound = float(np.sum(residual**2)/penalty)
    if not np.isfinite(encoder).all() or not np.isfinite(bound):
        raise FloatingPointError("nonfinite conditional encoder solution")
    return {"encoder": encoder, "system": system, "rhs": rhs,
            "feature_gram": gf, "response_gram": gz,
            "stationarity_residual": residual, "optimality_gap_upper_bound": bound}
