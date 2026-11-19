"""Differentiable fixed-prior initializers and trainable machine-wise angle damping."""

from __future__ import annotations

import math

import torch
from torch.nn.utils import parametrize

from kron_reduce.models.differentiable_history import solve_hidden_batch
from kron_reduce.models.machine_local_topology import (
    _radial_coordinate,
    predict_from_topology_history,
)
from kron_reduce.models.nonlinear_history import history_with_tangent


class _BoundedAngleDamping(torch.nn.Module):
    def forward(self, raw):
        return .95*raw.sigmoid()

    def right_inverse(self, physical):
        return torch.logit(physical/.95)


def enable_first_swing_rho(model):
    """Install a bounded scalar in the existing angle-damping field and history tangent."""
    if model.topology_dimension or model.latent_dimension != model.memory_dimension or not model.memory_dimension:
        raise ValueError("first-swing fitting requires fixed topology with local memory")
    model.set_angle_damping(2./3., spatial_mix=1.)
    model.angle_damping_rho = torch.nn.Parameter(model.mass.new_tensor(2./3.))
    parametrize.register_parametrization(model, "angle_damping_rho", _BoundedAngleDamping())
    return model


@torch.no_grad()
def load_first_swing_scoring_view(fitted, view):
    """Copy a fitted T0M1 into a frozen model with a scalar rho for CUDA graph capture."""
    if (not parametrize.is_parametrized(fitted, "angle_damping_rho") or parametrize.is_parametrized(view)
            or fitted.angle_damping_spatial_mix != 1. or fitted.restoring_shape != view.restoring_shape
            or any(m.topology_dimension or m.latent_dimension != m.memory_dimension
                   or m.memory_dimension != m.n_nodes or m.port_kind != "local" for m in (fitted, view))):
        raise ValueError("scoring requires matched local-memory T0M1 models and one fitted damping parameter")
    physical_rho = float(fitted.angle_damping_rho)
    if not 0. <= physical_rho <= .95:
        raise ValueError("invalid physical damping strength for scoring")
    state = fitted.state_dict()
    del state["parametrizations.angle_damping_rho.original"]
    view.load_state_dict(state, strict=True)
    # CUDA graph capture cannot branch on a device scalar inside the legacy field.
    view.angle_damping_rho = physical_rho
    view.angle_damping_spatial_mix = 1.
    return view.requires_grad_(False).eval()


def predict_first_swing_fit(model, q, v, power, equilibrium, scale, count, *, max_step, spend,
                            memory_radius, prior_weight, soft_radius):
    """Shoot five times from eleven causal frames, then reset visible states and freely forecast."""
    if type(soft_radius) is not bool:
        raise ValueError("soft_radius must be a boolean")
    if not soft_radius:
        return predict_from_topology_history(model, q, v, power, equilibrium, scale, count,
                                             max_step=max_step, spend=spend, memory_radius=memory_radius,
                                             prior_weight=prior_weight)
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    n, r = model.n_nodes, model.memory_dimension
    if (not r or model.latent_dimension != r or model.topology_dimension
            or q.ndim != 3 or not len(q) or q.shape[1:] != (11, n) or v.shape != q.shape
            or power.shape != (len(q), n) or equilibrium.shape != power.shape or scale.shape != (2*n,)
            or max_step not in (.005, .0025, .00125) or type(count) is not int or count < 0
            or not math.isfinite(memory_radius) or memory_radius <= 0.
            or not math.isfinite(prior_weight) or prior_weight <= 0.
            or not all(x.dtype == torch.float64 and torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale))
            or torch.any(scale <= 0)):
        raise ValueError("soft fixed-prior GN requires finite 100 ms histories and positive scales, radius and prior")
    if max(power.sum(1).abs().max(), (q@model.mass).abs().max(), (v@model.mass).abs().max()) > 1e-10:
        raise ValueError("history needs balanced power and COI visible states")

    observed = torch.cat((q, v), dim=2)
    radius = 2.*memory_radius
    prior = math.sqrt(prior_weight*10*2*n)/q.new_full((r,), memory_radius)
    prior_rows = torch.diag(prior).expand(len(q), -1, -1)
    coordinate = q.new_zeros((len(q), r))
    for _ in range(5):
        hidden, radial_scale = _radial_coordinate(coordinate, radius)
        initial = torch.cat((observed[:, 0], hidden), dim=1)
        history, tangent = history_with_tangent(model, initial, power, equilibrium, max_step=max_step, spend=spend)
        design = (tangent[:, 1:, :2*n]/scale[None, None, :, None]).flatten(1, 2)
        design = radial_scale[..., None]*(design-(design @ hidden[..., None])*hidden[:, None, :]/radius**2)
        residual = ((observed[:, 1:]-history[:, 1:, :2*n])/scale).flatten(1)
        fit = solve_hidden_batch(torch.cat((design, prior_rows), dim=1),
                                 torch.cat((residual, -coordinate*prior), dim=1), spend=spend)
        coordinate = coordinate+fit["hidden_start"]

    hidden, _ = _radial_coordinate(coordinate, radius)
    history = model.rollout(torch.cat((observed[:, 0], hidden), dim=1), power, equilibrium, 10,
                            max_step=max_step, spend=spend)
    origin = torch.cat((observed[:, -1], history[:, -1, 2*n:]), dim=1)
    forecast = model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
    return {"forecast": forecast, "history": history, "hidden_start": hidden, "singular": fit["singular"]}
