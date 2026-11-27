"""Causal ES-ROM initialization with empirical per-channel mismatch variances."""

from __future__ import annotations

import math

import torch

from kron_reduce.models.differentiable_history import solve_hidden_batch
from kron_reduce.models.noise_robustness_mhe import (
    PRIOR_WEIGHT,
    _check_model,
    _JointHistory,
    history_diagnostics,
)
from kron_reduce.models.noise_robustness_mhe_v2 import (
    LM_INITIAL,
    LM_ITERATIONS,
    LM_MAX,
    LM_MIN,
    LM_TRIALS,
    SIGNAL_ENERGY_FLOOR,
    _initial_state,
    _objective,
    _project_visible,
    _trial_history,
    _weighted_system,
    kinematic_cleaner,
)
from kron_reduce.models.unified_discovery_m3 import predict_m3

VISIBLE_PRIOR = 1e-4
KINEMATIC_ONLY_ITERATIONS = 3


def _mismatch_variance(clean, history, scale, c_var):
    """Positive excess mean-square discrepancy in normalized units, per channel."""
    weight = clean["weight"]
    difference = (clean["smooth"] - history[:, :, :20]) / scale
    discrepancy = (weight * difference.square()).sum(1) / weight.sum(1).clamp_min(1e-6)
    return (discrepancy - c_var * clean["channel_noise"].square()).clamp_min(0.)


def _endpoint_gain(variance, noise, kappa_term):
    return variance / (variance + kappa_term * noise.square() + 1e-12)


def _fuse_terminal(model, history, terminal, gain):
    visible = torch.lerp(history[:, -1, :20], terminal, gain)
    return torch.cat((_project_visible(visible, model.mass), history[:, -1, 20:]), dim=1)


@torch.no_grad()
def reblend_origin(result, model, kappa_term):
    """Change only the terminal gain of a completed fit and leave clean bypasses intact."""
    if not math.isfinite(kappa_term) or kappa_term <= 0:
        raise ValueError("kappa_term must be finite and positive")
    origin = result["origin"].clone()
    use = result["joint_active"]
    if use.any():
        gain = _endpoint_gain(result["mismatch_variance"][use], result["channel_noise"][use], kappa_term)
        origin[use] = _fuse_terminal(model, result["history"][use], result["terminal"][use], gain)
    return origin


def _fit_joint(model, observed, clean, power, equilibrium, scale, map_lambda, *,
               max_step, spend, memory_radius, huber_delta, c_var):
    integrator = _JointHistory(model, power, equilibrium, max_step, spend)
    lift, radius = integrator.lift, 2. * memory_radius
    anchor = _project_visible(clean["smooth"][:, 0], model.mass)
    center = torch.nn.functional.pad(anchor @ lift[:20, :18], (0, 10))
    coordinates = center
    visible_rows = math.sqrt(11. * VISIBLE_PRIOR) * lift[:20] / scale[:, None]
    identity = torch.eye(28, device=scale.device, dtype=scale.dtype)
    memory_rows = (220. * map_lambda).sqrt()[:, None, None] * identity[None, 18:] / memory_radius
    prior = torch.cat((visible_rows.expand(len(observed), -1, -1), memory_rows), dim=1)
    weight = clean["weight"]
    kinematic_scale = clean["kinematic_scale"]
    history, tangent = integrator(_initial_state(coordinates, lift, radius))
    residual = (observed - history[:, :, :20]) / scale
    damping = torch.full_like(map_lambda, LM_INITIAL)
    step_objectives, cutoffs = [], []

    for iteration in range(LM_ITERATIONS):
        robust_scale = kinematic_scale
        if iteration >= KINEMATIC_ONLY_ITERATIONS:
            variance = _mismatch_variance(clean, history, scale, c_var)
            robust_scale = (kinematic_scale.square() + variance).sqrt()
        cutoff = huber_delta * robust_scale[:, None]
        cutoffs.append(cutoff[:, 0])
        # Each damped trial competes with the incumbent under the same cutoff.
        # Only an accepted history can change the scale for the next iteration.
        start_loss = _objective(coordinates, center, prior, residual, weight, cutoff)
        loss = start_loss
        design, rhs = _weighted_system(coordinates, center, tangent, residual, scale, radius,
                                       prior, weight, cutoff)
        diagonal = design.square().sum(1).clamp_min(1e-12)
        pending = torch.ones_like(loss, dtype=torch.bool)
        start = coordinates
        for _ in range(LM_TRIALS):
            damped = torch.diag_embed((damping[:, None] * diagonal).sqrt())
            system = torch.cat((design, damped), dim=1)
            solution = solve_hidden_batch(system, torch.nn.functional.pad(rhs, (0, 28)), spend=spend)
            proposal = torch.where(pending[:, None], start + solution["hidden_start"], coordinates)
            trial_history, trial_tangent, finite = _trial_history(
                integrator, _initial_state(proposal, lift, radius), (history, tangent),
                model, power, equilibrium, max_step, spend)
            trial_residual = (observed - trial_history[:, :, :20]) / scale
            trial_loss = _objective(proposal, center, prior, trial_residual, weight, cutoff)
            accepted = pending & finite & torch.isfinite(trial_loss) & (trial_loss < loss)
            coordinates = torch.where(accepted[:, None], proposal, coordinates)
            history = torch.where(accepted[:, None, None], trial_history, history)
            tangent = torch.where(accepted[:, None, None, None], trial_tangent, tangent)
            residual = torch.where(accepted[:, None, None], trial_residual, residual)
            loss = torch.where(accepted, trial_loss, loss)
            damping = torch.where(accepted, damping / 3., torch.where(pending, damping * 10., damping))
            damping = damping.clamp(LM_MIN, LM_MAX)
            pending = pending & ~accepted
            if not pending.any():
                break
        step_objectives.append(torch.stack((start_loss, loss), dim=1))

    design, _ = _weighted_system(coordinates, center, tangent, residual, scale, radius, prior, weight, cutoff)
    spend("svd", len(observed))
    singular = torch.linalg.svdvals(design)
    return (history, torch.stack(step_objectives, dim=1), torch.stack(cutoffs, dim=1),
            singular[:, 0] / singular[:, -1])


@torch.no_grad()
def predict_mhe_v3(model, q, v, power, equilibrium, scale, count, *, max_step, spend,
                   memory_radius, gate_rad=.0003, alpha=.05, c_var=2.5, kappa_term=1.,
                   inverse_snr=True, huber_delta=1.5):
    """Fit eleven visible frames, fuse the terminal visible state, and forecast.

    For active joint fits, lm_step_objectives contains comparable losses before
    and after each LM iteration. lm_objectives records the initial value followed
    by each iteration's ending value under its own cutoff. On bypass rows the
    empirical variance and gain are diagnostic only and the GN forecast is used.
    """
    _check_model(model, max_step)
    if (type(count) is not int or count < 0 or type(inverse_snr) is not bool
            or not all(math.isfinite(x) and x > 0 for x in (memory_radius, kappa_term, huber_delta))
            or not all(math.isfinite(x) and x >= 0 for x in (alpha, c_var, gate_rad))):
        raise ValueError("invalid horizon, prior, variance multiplier, Huber cutoff or gate")
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    if (q.ndim != 3 or not len(q) or q.shape[1:] != (11, 10) or v.shape != q.shape
            or power.shape != (len(q), 10) or equilibrium.shape != power.shape or scale.shape != (20,)
            or not all(torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale))
            or (scale <= 0).any() or power.sum(dim=1).abs().max() > 1e-10
            or (q @ model.mass).abs().max() > 1e-10 or (v @ model.mass).abs().max() > 1e-10):
        raise ValueError("eleven COI frames, balanced held input, finite equilibrium and positive scales required")
    defect, _ = history_diagnostics(q, v, scale)
    joint = defect > gate_rad
    observed = torch.cat((q, v), dim=2)
    clean = kinematic_cleaner(observed, scale)
    variance = clean["sigma_hat"].square()
    if inverse_snr:
        variance = variance / clean["signal_energy"].clamp_min(SIGNAL_ENERGY_FLOOR)
    map_lambda = PRIOR_WEIGHT + torch.where(joint, alpha * variance, 0.)

    history = q.new_empty((len(q), 11, 30))
    forecast = q.new_empty((len(q), count + 1, 30))
    condition = q.new_empty((len(q),))
    step_objectives = q.new_full((len(q), LM_ITERATIONS, 2), torch.nan)
    cutoffs = q.new_full((len(q), LM_ITERATIONS, 20), torch.nan)
    bypass = ~joint
    if bypass.any():
        original = predict_m3(model, q[bypass], v[bypass], power[bypass], equilibrium[bypass], scale,
                              count, max_step=max_step, spend=spend, memory_radius=memory_radius,
                              prior_weight=PRIOR_WEIGHT)
        history[bypass], forecast[bypass] = original["history"], original["forecast"]
        singular = original["singular"]
        condition[bypass] = singular[:, 0] / singular[:, -1]
    if joint.any():
        filtered = {key: value[joint] for key, value in clean.items()}
        history[joint], step_objectives[joint], cutoffs[joint], condition[joint] = _fit_joint(
            model, observed[joint], filtered, power[joint], equilibrium[joint], scale, map_lambda[joint],
            max_step=max_step, spend=spend, memory_radius=memory_radius, huber_delta=huber_delta, c_var=c_var)

    mismatch = _mismatch_variance(clean, history, scale, c_var)
    gain = _endpoint_gain(mismatch, clean["channel_noise"], kappa_term)
    if joint.any():
        origin = _fuse_terminal(model, history[joint], clean["terminal"][joint], gain[joint])
        forecast[joint] = (model.rollout(origin, power[joint], equilibrium[joint], count,
                                         max_step=max_step, spend=spend) if count else origin[:, None])
    objectives = torch.cat((step_objectives[:, :1, 0], step_objectives[:, :, 1]), dim=1)
    return {"forecast": forecast, "history": history, "origin": forecast[:, 0],
            "hidden_start": history[:, 0, 20:], "joint_active": joint, "kinematic_defect_rad": defect,
            "sigma_hat": clean["sigma_hat"], "signal_energy": clean["signal_energy"],
            "map_lambda": map_lambda, "condition": condition, "terminal": clean["terminal"],
            "kinematic_scale": clean["kinematic_scale"], "channel_noise": clean["channel_noise"],
            "smooth": clean["smooth"], "outlier_weight": clean["weight"], "mismatch_variance": mismatch,
            "endpoint_gain": gain, "lm_objectives": objectives, "lm_step_objectives": step_objectives,
            "lm_cutoffs": cutoffs}
