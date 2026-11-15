"""Differentiable clean-history hidden-state recovery for a linear backbone."""
from __future__ import annotations

import math

import torch


class UnresolvedHistoryError(FloatingPointError):
    pass


def solve_hidden(design, rhs, *, spend):
    """Use reduced QR; only the numerical rank decision is detached."""
    if (design.ndim != 2 or rhs.ndim != 2 or design.shape[0] != rhs.shape[1]
            or design.shape[0] < design.shape[1] or min(design.shape) == 0 or not len(rhs)
            or any(x.dtype != torch.float64 or x.device != design.device
                   or not torch.isfinite(x).all() for x in (design, rhs))):
        raise ValueError("finite float64 observation matrices on the same device required")
    hidden, q, r, singular, rank, tolerance, condition = _solve_qr(design, rhs.T, 1, spend)
    return {"hidden_start": hidden.T, "design": design, "rhs": rhs, "qr_q": q, "qr_r": r,
            "singular": singular, "rank": int(rank), "rank_threshold": float(tolerance),
            "condition_number": float(condition)}


def solve_hidden_batch(design, rhs, *, spend):
    """Solve one weighted history equation per event on the model device."""
    _check_batch(design, rhs)
    hidden, q, r, singular, *_ = _solve_qr(design, rhs.unsqueeze(-1), len(design), spend)
    return {"hidden_start": hidden.squeeze(-1), "design": design, "rhs": rhs,
            "qr_q": q, "qr_r": r, "singular": singular}


def _check_batch(design, rhs):
    if (design.ndim != 3 or rhs.shape != design.shape[:2] or min(design.shape) == 0
            or design.shape[-2] < design.shape[-1] or design.device != rhs.device
            or design.dtype != torch.float64 or rhs.dtype != torch.float64
            or not (torch.isfinite(design).all() & torch.isfinite(rhs).all())):
        raise ValueError("finite float64 observation matrices on the same device required")


def _check_rank(singular, shape):
    tolerance = 64*torch.finfo(torch.float64).eps*max(shape[-2:])*singular[..., 0]
    rank = torch.count_nonzero(singular > tolerance[..., None], dim=-1)
    condition = singular[..., 0]/singular[..., -1]
    if not ((rank == shape[-1]) & (condition <= 1e8)).all():
        raise UnresolvedHistoryError(f"hidden history rank/conditioning failed: minimum rank={int(rank.min())}")
    return rank, tolerance, condition


def _solve_qr(design, rhs, count, spend):
    spend("qr", count)
    q, r = torch.linalg.qr(design, mode="reduced")
    spend("svd", count)
    singular = torch.linalg.svdvals(r.detach())
    rank, tolerance, condition = _check_rank(singular, design.shape)
    spend("matrix_products", count)
    projected = q.mT @ rhs
    spend("triangular_solves", count)
    hidden = torch.linalg.solve_triangular(r, projected, upper=True)
    if not torch.isfinite(hidden).all():
        raise UnresolvedHistoryError("nonfinite hidden estimate")
    return hidden, q, r, singular, rank, tolerance, condition


@torch.no_grad()
def solve_hidden_ball(design, rhs, radius, *, spend):
    """Fit absolute hidden states in a common positive-radius ball."""
    _check_batch(design, rhs)
    if isinstance(radius, bool) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("the hidden-state radius must be finite and positive")
    count = len(design)
    spend("svd", count)
    u, singular, vh = torch.linalg.svd(design, full_matrices=False)
    _check_rank(singular, design.shape)
    spend("matrix_products", count)
    projected = (u.mT @ rhs.unsqueeze(-1)).squeeze(-1)
    scale = singular[:, :1]
    sigma = singular/scale
    target = projected/scale
    squares, normal_rhs = sigma.square(), sigma*target
    unconstrained = target/sigma
    active = torch.linalg.vector_norm(unconstrained, dim=1) > radius
    multiplier = torch.zeros_like(active, dtype=design.dtype)
    eps = torch.finfo(design.dtype).eps
    if active.any():
        sq, d = squares[active], normal_rhs[active]
        lower = (d.abs()/radius-sq).amax(dim=1).clamp_min(0.)
        upper = (1.+8.*eps)*torch.linalg.vector_norm(d, dim=1)/radius
        floor = sq[:, -1]
        log_lower, log_upper = (lower+floor).log(), (upper+floor).log()
        for _ in range(64):
            spend("secular_iterations", len(upper))
            # Bisection in log(mu + sigma_min^2) retains resolution near zero.
            middle = .5*(log_lower+log_upper)
            trial = (middle.exp()-floor).clamp_min(0.)
            coefficient = d/(sq+trial[:, None])
            magnitude = torch.linalg.vector_norm(coefficient, dim=1)
            outside = magnitude > radius
            log_lower = torch.where(outside, middle, log_lower)
            log_upper = torch.where(outside, log_upper, middle)
            upper = torch.where(outside, upper, trial)
        multiplier[active] = upper
    coefficients = torch.where(active[:, None], normal_rhs/(squares+multiplier[:, None]), unconstrained)
    spend("matrix_products", count)
    hidden = (vh.mT @ coefficients.unsqueeze(-1)).squeeze(-1)
    norm = torch.linalg.vector_norm(hidden, dim=1)
    scaled_design, scaled_rhs = design/scale[:, :, None], rhs/scale
    spend("matrix_products", 2*count)
    residual = scaled_design @ hidden.unsqueeze(-1)-scaled_rhs.unsqueeze(-1)
    gradient = (scaled_design.mT @ residual).squeeze(-1)+multiplier[:, None]*hidden
    denominator = norm*(1.+multiplier)+torch.linalg.vector_norm(scaled_rhs, dim=1)
    kkt = torch.linalg.vector_norm(gradient, dim=1)/denominator.clamp_min(torch.finfo(design.dtype).tiny)
    physical_multiplier = multiplier*scale[:, 0].square()
    if (not torch.isfinite(hidden).all() or not torch.isfinite(physical_multiplier).all()
            or not (norm <= radius*(1.+1e-10)).all() or not ((norm[active]/radius-1.).abs() <= 1e-10).all()
            or not (kkt <= 1e-10).all()):
        raise UnresolvedHistoryError("hidden ball solve failed its feasibility or KKT check")
    return {"hidden_start": hidden, "singular": singular, "multipliers": physical_multiplier, "kkt": kkt}


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
    initial = q.new_zeros((batch+r, 2*n+r))
    initial[:batch, :n], initial[:batch, n:2*n] = q[:, 0], v[:, 0]
    initial[batch:, 2*n:] = torch.eye(r, dtype=q.dtype, device=q.device)
    zeros = q.new_zeros((r, n))
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
