"""Train-only first-swing targets for the fixed TASK3 causal predictor."""

from __future__ import annotations

import math

import numpy as np
import torch

from kron_reduce.evaluation.kiki_machine_local_full import inputs
from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def training_weights(events, weight, fraction):
    if not events or not 0. <= fraction <= 1. or not math.isfinite(weight) or weight < 1.:
        raise ValueError("invalid first-swing weighting")
    kinetic = {}
    for event in events:
        if event["role"] != "train" or event["id"] in kinetic or event["label"] not in {"stable", "unstable"}:
            raise ValueError("weighting requires unique labeled train events")
        mass = event["mass"]
        history_speed = event["x"][:11, len(mass):]
        if history_speed.shape != (11, len(mass)):
            raise ValueError("weighting requires eleven finite history frames")
        energy = .5*float(np.mean(history_speed**2 @ mass))
        if not math.isfinite(energy):
            raise ValueError("nonfinite history kinetic energy")
        kinetic[event["id"]] = energy
    top = set(sorted(kinetic, key=lambda identity: (-kinetic[identity], identity))[:math.ceil(fraction*len(events))])
    return {row["id"]: weight if row["id"] in top or row["label"] == "unstable" else 1. for row in events}


def one_sided_velocity_loss(observed, forecast, speed_scale, counts):
    speed_scale = torch.as_tensor(speed_scale, dtype=observed.dtype, device=observed.device)
    lengths = torch.as_tensor(counts, device=observed.device).clamp(max=50)
    time = torch.arange(1, min(51, observed.shape[1]), device=observed.device)
    truth, predicted = observed[:, 1:len(time)+1], forecast[:, 1:len(time)+1]
    error = (truth.sign()*(truth-predicted)/speed_scale).relu().square().mean(dim=-1)
    return (error*(time <= lengths[:, None])).sum(dim=1)/lengths.clamp_min(1)


def momentum_window_loss(observed, increment, speed_scale, counts):
    # Dividing momentum by machine mass gives velocity-increment units.
    speed_scale = torch.as_tensor(speed_scale, dtype=observed.dtype, device=observed.device)
    counts = torch.as_tensor(counts, device=observed.device)
    total, available = torch.zeros_like(counts, dtype=observed.dtype), torch.zeros_like(counts)
    for start, stop in ((0, 20), (20, 40)):
        if stop < increment.shape[1]:
            residual = observed[:, stop]-observed[:, start]-(increment[:, stop]-increment[:, start])
            total = total+(residual/speed_scale).square().mean(dim=1)*(counts >= stop)
            available = available+(counts >= stop)
    return total/available.clamp_min(1)


def observed_velocity_increment(model, visible, hidden, power, equilibrium, counts, *, step, spend):
    """Integrate the force and latent dynamics along the linearly interpolated observations."""
    n = model.n_nodes
    field = model._rhs(power, equilibrium)
    counts = torch.as_tensor(counts, device=visible.device)
    substeps = round(.01/step)
    h = .01/substeps
    state = torch.cat((visible.new_zeros((len(visible), n)), hidden), dim=1)
    outputs = [state[:, :n]]

    def derivative(observation, current):
        return field(torch.cat((observation, current[:, n:]), dim=1))[:, n:]

    for index in range(visible.shape[1]-1):
        left, right = visible[:, index], visible[:, index+1]
        previous = state
        spend("observed_rhs", 4*substeps)
        for j in range(substeps):
            first = derivative(torch.lerp(left, right, j/substeps), state)
            middle = torch.lerp(left, right, (j+.5)/substeps)
            second = derivative(middle, state+.5*h*first)
            third = derivative(middle, state+.5*h*second)
            fourth = derivative(torch.lerp(left, right, (j+1)/substeps), state+h*third)
            state = state+h/6.*(first+2.*second+2.*third+fourth)
        state = torch.where((counts > index)[:, None], state, previous)
        outputs.append(state[:, :n])
    result = torch.stack(outputs, dim=1)
    if not torch.isfinite(result).all() or not torch.isfinite(state).all():
        raise FloatingPointError("nonfinite observed-path momentum")
    return result


def objective_terms(model, batch, population, scales, budget, settings, *, weights=None):
    if any(event["role"] != "train" for event in batch):
        raise ValueError("first-swing targets are restricted to train")
    n, step = model.n_nodes, .005
    counts = np.array([min(300, len(event["x"])-11) for event in batch])
    count = int(counts.max())
    padded = np.stack([np.pad(event["x"][10:11+c], ((0, count-c), (0, 0)), mode="edge")
                       for event, c in zip(batch, counts, strict=True)])
    q, v, power, equilibrium, scale = map(model._tensor, inputs(batch, scales["history"]))
    result = predict_from_topology_history(model, q, v, power, equilibrium, scale, count,
        max_step=step, spend=budget.spend, memory_radius=scales["memory_radius"], prior_weight=settings["prior_weight"])
    forecast, observed = result["forecast"][..., :2*n], model._tensor(padded)
    hidden_origin = result["history"][:, -1, 2*n:]
    memory_energy = .5*hidden_origin[:, :model.memory_dimension].square().sum(dim=1)
    energy_limit = .5*(settings["origin_energy_radius_multiplier"]*scales["memory_radius"])**2
    energy_guard = settings["origin_energy_weight"]*(memory_energy/energy_limit-1.).relu().square().sum()/population
    rollout, event_mse = loss(forecast, observed, np.full(len(batch), 1./population),
                              scales["target"], counts, return_events=True)
    terms = {"train_nmse_1s_3s": rollout, "origin_energy_guard": energy_guard}
    if weights is None:
        return rollout+energy_guard, terms

    weights = model._tensor([weights[event["id"]] for event in batch])/population
    speed_scale = scales["target"][n:]
    window = min(40, count)
    increment = observed_velocity_increment(model, observed[:, :window+1], hidden_origin, power, equilibrium, counts,
                                            step=step, spend=budget.spend)
    asym = one_sided_velocity_loss(observed[..., n:], forecast[..., n:], speed_scale, counts)
    mom = momentum_window_loss(observed[..., n:], increment, speed_scale, counts)
    terms.update(weighted_rollout=weights@event_mse, one_sided_velocity=weights@asym, weak_momentum=weights@mom)
    objective = (terms["weighted_rollout"]+settings["lambda_asym"]*terms["one_sided_velocity"]
                 +settings["lambda_momentum"]*terms["weak_momentum"]+energy_guard)
    return objective, terms
