"""Training controls shared by synthetic and TASK3 joint energy discovery.

The default derivative follows the actual causal prediction algorithm, including
all five history updates. Historical stop-gradient/early-pruning behavior is an
explicit opt-in, never a silent fallback for a new experiment.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from kron_reduce.models.machine_local_topology import predict_from_topology_history


def training_controls(settings):
    gradient = settings.get("gradient_mode", "full")
    support = settings.get("support_policy", "dense")
    schedule = settings.get("learning_rate_policy", "constant")
    weight = float(settings.get("group_regularization_weight", 1e-4 if support == "legacy_threshold" else 0.))
    if gradient not in ("full", "stop_initialization"):
        raise ValueError("gradient_mode must be full or stop_initialization")
    if support not in ("dense", "legacy_threshold"):
        raise ValueError("support_policy must be dense or legacy_threshold")
    if schedule not in ("constant", "cosine"):
        raise ValueError("learning_rate_policy must be constant or cosine")
    if not math.isfinite(weight) or weight < 0.:
        raise ValueError("group regularization weight must be finite and nonnegative")
    if support == "dense" and weight:
        raise ValueError("dense discovery does not penalize heterogeneous raw group norms")
    return {"gradient_mode": gradient, "support_policy": support,
            "learning_rate_policy": schedule, "group_regularization_weight": weight}


def causal_rollout(model, q, v, power, equilibrium, scale, count, *, settings, spend, max_step=.005):
    """Re-estimate the origin under the current parameters, then freely forecast.

No future target is read here. In full mode, the QR solves, ball projections,
history propagation and forecast remain connected to the shared coefficients.
"""
    mode = training_controls(settings)["gradient_mode"]
    context = torch.no_grad() if mode == "stop_initialization" else nullcontext()
    with context:
        origin = predict_from_topology_history(
            model, q, v, power, equilibrium, scale, 0, max_step=max_step, spend=spend,
            memory_radius=settings["memory_radius"], prior_weight=settings["prior_weight"],
        )["forecast"][:, 0]
    return model.rollout(origin, model._tensor(power), model._tensor(equilibrium), count,
                         max_step=max_step, spend=spend)


def group_regularization(model, settings):
    weight = training_controls(settings)["group_regularization_weight"]
    if not weight:
        return None
    return weight*sum(model.group_norms().values())


def update_learning_rate(optimizer, settings, update):
    """Set the registered rate before an update; the final cosine step reaches min_lr."""
    controls = training_controls(settings)
    initial = float(settings["learning_rate"])
    minimum = float(settings.get("minimum_learning_rate", initial))
    if not 0. < minimum <= initial or not math.isfinite(initial):
        raise ValueError("learning rates must satisfy 0 < minimum <= initial")
    fraction = (update-1)/max(settings["updates"]-1, 1)
    value = (minimum+.5*(initial-minimum)*(1.+math.cos(math.pi*fraction))
             if controls["learning_rate_policy"] == "cosine" else initial)
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def prune_due(settings, update):
    return (training_controls(settings)["support_policy"] == "legacy_threshold"
            and update == settings["prune_update"])


def checkpoint_eligible(settings, update):
    return (training_controls(settings)["support_policy"] == "dense"
            or update >= settings["prune_update"])
