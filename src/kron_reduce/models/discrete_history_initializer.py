"""Known-linear-model state recovery from the eleven causal observations."""
from __future__ import annotations

import numpy as np
import torch


def solve_history_state(visible, zero_history, response, scale, *, spend):
    """Solve all event states with one thin SVD; reject unresolved directions."""
    y, b, d, s = [np.asarray(x, dtype=np.float64) for x in
                  (visible, zero_history, response, scale)]
    if (y.ndim != 3 or y.shape[1] != 11 or not len(y) or b.shape != y.shape
            or d.ndim != 3 or d.shape[:2] != y.shape[1:] or d.shape[2] < 1
            or s.shape != (y.shape[2],) or np.any(s <= 0)
            or not all(np.isfinite(x).all() for x in (y, b, d, s))):
        raise ValueError("invalid eleven-frame observation problem")
    design = (d[1:] / s[None, :, None]).reshape(-1, d.shape[2])
    rhs = ((y[:, 1:] - b[:, 1:]) / s).reshape(len(y), -1)
    spend("svd")
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    tolerance = 64 * np.finfo(float).eps * max(design.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if rank == d.shape[2] else None
    resolved = rank == d.shape[2] and condition <= 1e8
    result = {"resolved": resolved, "rank": rank, "rank_threshold": float(tolerance),
              "condition_number": condition, "design": design, "rhs": rhs,
              "u": u, "singular": singular, "vh": vh}
    if resolved:
        spend("matrix_products", 3)
        zh = ((rhs @ u) / singular) @ vh
        residual = zh @ design.T - rhs
        if not np.isfinite(zh).all():
            raise FloatingPointError("nonfinite history estimate")
        result.update(hidden_start=zh, residual=residual)
    return result


def estimate_history(model, q, v, power, equilibrium, scale, *, max_step, spend):
    """Build the actual model's sampled response, with no future observations."""
    if model.backbone_kind != "linear":
        raise ValueError("discrete affine observer requires a linear backbone")
    q, v, power, equilibrium = map(np.asarray, (q, v, power, equilibrium))
    n, r, batch = model.n_nodes, model.latent_dimension, len(q)
    if q.shape != (batch, 11, n) or v.shape != q.shape:
        raise ValueError("eleven q/v history frames are required")
    initial = np.zeros((batch + r, 2*n + r))
    initial[:batch, :n], initial[:batch, n:2*n] = q[:, 0], v[:, 0]
    initial[batch:, 2*n:] = np.eye(r)
    with torch.no_grad():
        history = model.rollout(initial, np.concatenate((power, np.zeros((r, n)))),
            np.concatenate((equilibrium, np.zeros((r, n)))), 10,
            max_step=max_step, spend=spend).numpy()
    baseline, response = history[:batch], history[batch:].transpose(1, 2, 0)
    result = solve_history_state(np.concatenate((q, v), axis=2), baseline[:, :, :2*n],
                                response[:, :2*n], scale, spend=spend)
    result.update(zero_history=baseline, history_response=response)
    if result["resolved"]:
        spend("matrix_products", 11)
        closed = baseline + np.einsum("tir,br->bti", response, result["hidden_start"])
        origin = closed[:, -1].copy()
        origin[:, :n], origin[:, n:2*n] = q[:, -1], v[:, -1]
        result.update(history=closed, origin=origin)
    return result
