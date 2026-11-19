"""Batched fixed-parameter comparisons of memory initialization from 100 ms histories."""

from __future__ import annotations

import math

import torch

from kron_reduce.models.differentiable_history import solve_hidden_batch
from kron_reduce.models.machine_local_topology import _ball, _radial_coordinate
from kron_reduce.models.nonlinear_history import history_with_tangent


def _mode_controls(modes, zero_residual_squared, primary_r0, memory_radius):
    factors, limits, soft_rows = [], [], []
    names = set()
    for mode in modes:
        name, prior = mode["name"], mode["prior"]
        multiplier, reference = mode["radius_multiplier"], mode["reference"]
        if (not isinstance(name, str) or not name or name in names
                or prior not in ("fixed", "quadratic", "hinge")
                or not math.isfinite(multiplier) or multiplier < 1.
                or not math.isfinite(reference) or reference <= 0.):
            raise ValueError("invalid GN mode name, prior, radius multiplier, or residual reference")
        names.add(name)
        active = torch.full_like(zero_residual_squared, multiplier > 1., dtype=torch.bool)
        factor = torch.ones_like(zero_residual_squared)
        if prior == "quadratic":
            factor = (1.+zero_residual_squared/reference**2).reciprocal()
        elif prior == "hinge":
            if primary_r0 is None:
                raise ValueError("hinge modes require the primary-step history residual for every event")
            active = active & (primary_r0 > reference)
            excess = (primary_r0/reference-1.).clamp_min(0.)
            factor = (1.+excess.square()).reciprocal()
        factors.append(factor)
        soft_radius = torch.full_like(zero_residual_squared, memory_radius*multiplier)
        limits.append(torch.where(active, soft_radius, memory_radius))
        soft_rows.append(active)
    return torch.stack(factors), torch.stack(limits), torch.stack(soft_rows)


@torch.no_grad()
def predict_gn_grid(model, q, v, power, equilibrium, scale, *, max_step, spend, memory_radius,
                    prior_weight, modes, primary_r0, count=800):
    """Estimate each mode's memory and return mode-major free forecasts after the visible reset."""
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    n, r = model.n_nodes, model.memory_dimension
    if (not modes or not r or model.latent_dimension != r or model.topology_dimension
            or q.ndim != 3 or not len(q) or q.shape[1:] != (11, n) or v.shape != q.shape
            or power.shape != (len(q), n) or equilibrium.shape != power.shape or scale.shape != (2*n,)
            or max_step not in (.005, .0025, .00125) or type(count) is not int or count < 0
            or not math.isfinite(memory_radius) or memory_radius <= 0.
            or not math.isfinite(prior_weight) or prior_weight <= 0.
            or not all(x.dtype == torch.float64 and torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale))
            or torch.any(scale <= 0)):
        raise ValueError("memory-only GN requires finite eleven-frame histories, positive scales and a positive prior")
    if max(power.sum(1).abs().max(), (q@model.mass).abs().max(), (v@model.mass).abs().max()) > 1e-10:
        raise ValueError("history needs balanced power and COI visible states")
    batch, variants = len(q), len(modes)
    if primary_r0 is not None:
        primary_r0 = model._tensor(primary_r0)
        if primary_r0.shape != (batch,) or not torch.isfinite(primary_r0).all() or torch.any(primary_r0 < 0):
            raise ValueError("primary-step history residuals must be finite and nonnegative for every event")

    observed = torch.cat((q, v), dim=2)
    hidden = q.new_zeros((batch, r))

    def linearize(hidden):
        initial = torch.cat((observed[:, 0], hidden), dim=1)
        history, tangent = history_with_tangent(model, initial, power, equilibrium, max_step=max_step, spend=spend)
        design = (tangent[:, 1:, :2*n]/scale[None, None, :, None]).flatten(1, 2)
        residual = ((observed[:, 1:]-history[:, 1:, :2*n])/scale).flatten(1)
        return design, residual

    # All modes share the same zero-memory iterate and its physical-coordinate Jacobian.
    design, residual = linearize(hidden)
    zero_residual_squared = residual.square().mean(dim=1)
    factors, limits, soft = _mode_controls(modes, zero_residual_squared, primary_r0, memory_radius)
    soft_mask, radius = soft.reshape(-1, 1), limits.reshape(-1, 1)
    base_prior = math.sqrt(prior_weight*10*2*n)/q.new_full((r,), memory_radius)
    prior = factors.sqrt().reshape(-1, 1)*base_prior
    prior_rows = torch.diag_embed(prior)
    power, equilibrium = power.repeat(variants, 1), equilibrium.repeat(variants, 1)
    observed, hidden = observed.repeat(variants, 1, 1), hidden.repeat(variants, 1)
    design, residual = design.repeat(variants, 1, 1), residual.repeat(variants, 1)
    coordinate = hidden

    for iteration in range(5):
        if iteration:
            design, residual = linearize(hidden)
            _, radial_scale = _radial_coordinate(coordinate, radius)
            soft_design = radial_scale[..., None]*(design
                -(design @ hidden[..., None])*hidden[:, None, :]/radius[..., None].square())
            design = torch.where(soft_mask[..., None], soft_design, design)
        fit = solve_hidden_batch(torch.cat((design, prior_rows), dim=1),
                                 torch.cat((residual, -coordinate*prior), dim=1), spend=spend)
        proposal = coordinate+fit["hidden_start"]
        projected = _ball(proposal, memory_radius)
        soft_hidden, _ = _radial_coordinate(proposal, radius)
        hidden = torch.where(soft_mask, soft_hidden, projected)
        coordinate = torch.where(soft_mask, proposal, projected)

    history = model.rollout(torch.cat((observed[:, 0], hidden), dim=1), power, equilibrium, 10,
                            max_step=max_step, spend=spend)
    origin = torch.cat((observed[:, -1], history[:, -1, 2*n:]), dim=1)
    forecast = model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
    return {
        "forecast": forecast[..., :2*n].reshape(variants, batch, count+1, 2*n),
        "history": history.reshape(variants, batch, 11, 2*n+r),
        "hidden_start": hidden.reshape(variants, batch, r),
        "singular": fit["singular"].reshape(variants, batch, r),
        "zero_history_nrmse": zero_residual_squared.sqrt(),
        "prior_weight": prior_weight*factors,
        "radius_limit": limits,
        "soft_active": soft,
    }
