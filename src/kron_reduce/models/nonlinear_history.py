"""Fixed-iteration nonlinear shooting from an observed 100 ms history."""
from __future__ import annotations

import torch

from kron_reduce.models.differentiable_history import UnresolvedHistoryError, solve_hidden_batch


def tangent_rhs(model, state, tangent, power, equilibrium, *, spend):
    """Propagate the derivative with respect to the hidden initial state."""
    state, tangent, power, equilibrium = map(model._tensor, (state, tangent, power, equilibrium))
    n, r = model.n_nodes, model.latent_dimension
    if (state.ndim != 2 or state.shape[1] != 2*n+r or tangent.ndim != 3
            or tangent.shape[:2] != state.shape or power.shape != (len(state), n)
            or equilibrium.shape != power.shape):
        raise ValueError("state, tangent, power and equilibrium dimensions differ")
    if not all(torch.isfinite(x).all() for x in (state, tangent, power, equilibrium)):
        raise ValueError("model inputs must be finite")
    spend("tangent_rhs")
    spend("torch_rhs")
    return model._rhs(power, equilibrium, tangent=True)(state, tangent)


def history_with_tangent(model, initial, power, equilibrium, *, max_step, spend):
    initial, power, equilibrium = map(model._tensor, (initial, power, equilibrium))
    n, r, batch = model.n_nodes, model.latent_dimension, len(initial)
    if (initial.shape != (batch, 2*n+r) or power.shape != (batch, n)
            or equilibrium.shape != power.shape or not batch):
        raise ValueError("state, power and equilibrium dimensions differ")
    if not (torch.isfinite(initial).all() & torch.isfinite(power).all() & torch.isfinite(equilibrium).all()):
        raise ValueError("model inputs must be finite")
    tangent = initial.new_zeros((batch, 2*n+r, r))
    tangent[:, 2*n:] = torch.eye(r, dtype=initial.dtype, device=initial.device)
    field = model._rhs(power, equilibrium, tangent=True)
    mass_sum = model.mass.sum()
    state, states, tangents = initial, [initial], [tangent]
    substeps = round(.01/max_step)
    h = .01/substeps
    spend("rollouts")
    for _ in range(10):
        for _ in range(substeps):
            spend("tangent_rhs", 4)
            spend("torch_rhs", 4)
            a, sa = field(state, tangent)
            b, sb = field(state+h*a/2, tangent+h*sa/2)
            c, sc = field(state+h*b/2, tangent+h*sb/2)
            d, sd = field(state+h*c, tangent+h*sc)
            state = state+h*(a+2*b+2*c+d)/6
            tangent = tangent+h*(sa+2*sb+2*sc+sd)/6
            q, v = state[:, :n], state[:, n:2*n]
            state = torch.cat((q-(q@model.mass)[:, None]/mass_sum,
                               v-(v@model.mass)[:, None]/mass_sum, state[:, 2*n:]), dim=1)
            sq, sv = tangent[:, :n], tangent[:, n:2*n]
            tangent = torch.cat((sq-(sq*model.mass[None, :, None]).sum(1, keepdim=True)/mass_sum,
                                 sv-(sv*model.mass[None, :, None]).sum(1, keepdim=True)/mass_sum,
                                 tangent[:, 2*n:]), dim=1)
        states.append(state)
        tangents.append(tangent)
    history, response = torch.stack(states, dim=1), torch.stack(tangents, dim=1)
    if (not torch.isfinite(history).all() or not torch.isfinite(response).all()
            or torch.max(torch.abs(history)) > 1e12):
        raise UnresolvedHistoryError("nonfinite nonlinear history or sensitivity")
    return history, response


def predict_from_history(model, q, v, power, equilibrium, scale, count, *, max_step, spend):
    """Three full Gauss–Newton updates, followed by actual nonlinear warmup."""
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    n, r = model.n_nodes, model.latent_dimension
    if (model.backbone_kind != "sine" or q.ndim != 3 or not len(q)
            or q.shape[1:] != (11, n) or v.shape != q.shape
            or power.shape != (len(q), n) or equilibrium.shape != power.shape
            or scale.shape != (2*n,) or torch.any(scale <= 0)
            or max_step not in (.005, .0025, .00125)
            or type(count) is not int or count < 0
            or not all(torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale))):
        raise ValueError("finite sine-model 100 ms history and registered clock required")
    if (torch.max(torch.abs(power.sum(1))) > 1e-10
            or torch.max(torch.abs(q@model.mass)) > 1e-10
            or torch.max(torch.abs(v@model.mass)) > 1e-10):
        raise ValueError("history needs balanced power and COI visible states")
    observed = torch.cat((q, v), dim=2)
    hidden = q.new_zeros((len(q), r))
    traces = {key: [] for key in ("design", "rhs", "qr_q", "qr_r", "hidden_start", "singular")}
    iterates, residuals, histories, responses = [hidden], [], [], []
    for _ in range(3):
        initial = torch.cat((q[:, 0], v[:, 0], hidden), dim=1)
        history, response = history_with_tangent(model, initial, power, equilibrium,
                                                 max_step=max_step, spend=spend)
        histories.append(history)
        responses.append(response)
        design = (response[:, 1:, :2*n]/scale[None, None, :, None]).reshape(len(q), -1, r)
        rhs = ((observed[:, 1:]-history[:, 1:, :2*n])/scale).reshape(len(q), -1)
        solution = solve_hidden_batch(design, rhs, spend=spend)
        for key, trace in traces.items():
            trace.append(solution[key])
        hidden = hidden+traces["hidden_start"][-1]
        if not torch.isfinite(hidden).all() or torch.max(torch.abs(hidden)) > 1e12:
            raise UnresolvedHistoryError("nonfinite or divergent hidden update")
        iterates.append(hidden)
        residuals.append(rhs)
    initial = torch.cat((q[:, 0], v[:, 0], hidden), dim=1)
    history = model.rollout(initial, power, equilibrium, 10, max_step=max_step, spend=spend)
    origin = torch.cat((q[:, -1], v[:, -1], history[:, -1, 2*n:]), dim=1)
    forecast = model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
    return {"hidden_start": hidden, "history": history, "forecast": forecast,
            "iterates": torch.stack(iterates), "iteration_residuals": torch.stack(residuals),
            "iteration_histories": torch.stack(histories), "iteration_responses": torch.stack(responses),
            **{"gn_"+key: torch.stack(value) for key, value in traces.items()}}
