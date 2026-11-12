"""Differentiable clean-history hidden-state recovery for a linear backbone."""
from __future__ import annotations

import torch


class UnresolvedHistoryError(FloatingPointError):
    pass


def solve_hidden(design, rhs, *, spend):
    """Use reduced QR; only the numerical rank decision is detached."""
    if (design.ndim != 2 or rhs.ndim != 2 or design.shape[0] != rhs.shape[1]
            or design.shape[0] < design.shape[1] or min(design.shape) == 0 or not len(rhs)
            or any(x.dtype != torch.float64 or x.device.type != 'cpu'
                   or not torch.isfinite(x).all() for x in (design, rhs))):
        raise ValueError("finite CPU float64 observation matrices required")
    spend("qr")
    q, r = torch.linalg.qr(design, mode="reduced")
    spend("svd")
    singular = torch.linalg.svdvals(r.detach())
    tolerance = 64*torch.finfo(torch.float64).eps*max(design.shape)*singular[0]
    rank = int(torch.count_nonzero(singular > tolerance))
    if rank != design.shape[1] or singular[0]/singular[-1] > 1e8:
        raise UnresolvedHistoryError(f"hidden history rank/conditioning failed: rank={rank}")
    spend("matrix_products")
    projected = q.T@rhs.T
    spend("triangular_solves")
    hidden = torch.linalg.solve_triangular(r, projected, upper=True).T
    if not torch.isfinite(hidden).all():
        raise UnresolvedHistoryError("nonfinite hidden estimate")
    return {"hidden_start": hidden, "design": design, "rhs": rhs, "qr_q": q, "qr_r": r,
            "singular": singular, "rank": rank, "rank_threshold": float(tolerance),
            "condition_number": float(singular[0]/singular[-1])}


def predict_from_history(model, q, v, power, equilibrium, scale, count, *, max_step, spend):
    if model.backbone_kind != "linear":
        raise ValueError("differentiable discrete observer requires a linear backbone")
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    n, r, batch = model.n_nodes, model.latent_dimension, len(q)
    if (q.shape != (batch, 11, n) or not batch or v.shape != q.shape
            or power.shape != (batch, n) or equilibrium.shape != power.shape
            or scale.shape != (2*n,) or torch.any(scale <= 0)
            or not all(torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale))):
        raise ValueError("eleven finite observed frames and positive train scales required")
    initial = torch.zeros((batch+r, 2*n+r), dtype=torch.float64)
    initial[:batch, :n], initial[:batch, n:2*n] = q[:, 0], v[:, 0]
    initial[batch:, 2*n:] = torch.eye(r, dtype=torch.float64)
    zeros = torch.zeros((r, n), dtype=torch.float64)
    trajectories = model.rollout(initial, torch.cat((power, zeros)), torch.cat((equilibrium, zeros)),
                                  10, max_step=max_step, spend=spend)
    baseline, response = trajectories[:batch], trajectories[batch:].permute(1, 2, 0)
    observed = torch.cat((q, v), dim=2)
    design = (response[1:, :2*n]/scale[None, :, None]).reshape(-1, r)
    rhs = ((observed[:, 1:]-baseline[:, 1:, :2*n])/scale).reshape(batch, -1)
    result = solve_hidden(design, rhs, spend=spend)
    spend("matrix_products", 11)
    history = baseline+torch.einsum("tir,br->bti", response, result["hidden_start"])
    if not torch.isfinite(history).all() or torch.max(torch.abs(history)) > 1e12:
        raise UnresolvedHistoryError("nonfinite or divergent history estimate")
    origin = torch.cat((q[:, -1], v[:, -1], history[:, -1, 2*n:]), dim=1)
    future = model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
    result.update(history=history, forecast=future, zero_history=baseline, history_response=response)
    return result
