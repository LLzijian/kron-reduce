"""Fixed-iteration nonlinear shooting from an observed 100 ms history."""
from __future__ import annotations

import torch

from kron_reduce.models.differentiable_history import UnresolvedHistoryError, solve_hidden


def tangent_rhs(model, state, tangent, power, equilibrium, *, spend):
    """Propagate the derivative with respect to the hidden initial state."""
    n = model.n_nodes
    sq, sv, sz = tangent[:, :n], tangent[:, n:2*n], tangent[:, 2*n:]
    edge = (state[:, :n]+equilibrium) @ model.incidence
    stiffness = model.stiffness * (torch.cos(edge) if model.backbone_kind == "sine" else 1.)
    coupling = model.coupling
    force = (model.incidence @ (stiffness[..., None]*(model.incidence.T @ sq))
             + model.damping @ sv + coupling @ sz)
    derivative = torch.cat((sv, -(model.power_projection @ force)/model.mass[None, :, None],
                            coupling.T @ sv-sz/model.tau[None, :, None]), dim=1)
    spend("tangent_rhs")
    spend("torch_rhs")
    return model.rhs(state, power, equilibrium), derivative


def history_with_tangent(model, initial, power, equilibrium, *, max_step, spend):
    n, r, batch = model.n_nodes, model.latent_dimension, len(initial)
    tangent = torch.zeros((batch, 2*n+r, r), dtype=torch.float64)
    tangent[:, 2*n:] = torch.eye(r, dtype=torch.float64)
    state, states, tangents = initial, [initial], [tangent]
    substeps = round(.01/max_step)
    h = .01/substeps
    spend("rollouts")
    for _ in range(10):
        for _ in range(substeps):
            a, sa = tangent_rhs(model, state, tangent, power, equilibrium, spend=spend)
            b, sb = tangent_rhs(model, state+h*a/2, tangent+h*sa/2, power, equilibrium, spend=spend)
            c, sc = tangent_rhs(model, state+h*b/2, tangent+h*sb/2, power, equilibrium, spend=spend)
            d, sd = tangent_rhs(model, state+h*c, tangent+h*sc, power, equilibrium, spend=spend)
            state = state+h*(a+2*b+2*c+d)/6
            tangent = tangent+h*(sa+2*sb+2*sc+sd)/6
            q, v = state[:, :n], state[:, n:2*n]
            state = torch.cat((q-(q@model.mass)[:, None]/model.mass.sum(),
                               v-(v@model.mass)[:, None]/model.mass.sum(), state[:, 2*n:]), dim=1)
            sq, sv = tangent[:, :n], tangent[:, n:2*n]
            tangent = torch.cat((sq-(sq*model.mass[None, :, None]).sum(1, keepdim=True)/model.mass.sum(),
                                 sv-(sv*model.mass[None, :, None]).sum(1, keepdim=True)/model.mass.sum(),
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
    hidden = torch.zeros((len(q), r), dtype=torch.float64)
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
        solutions = [solve_hidden(a, b[None], spend=spend) for a, b in zip(design, rhs, strict=True)]
        for key, trace in traces.items():
            trace.append(torch.stack([item[key].squeeze(0) if key in ("rhs", "hidden_start")
                                      else item[key] for item in solutions]))
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
