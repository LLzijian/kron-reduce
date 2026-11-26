"""Robust initialization of the frozen G2 ES-ROM from eleven visible frames."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations

import torch

from kron_reduce.models.differentiable_history import UnresolvedHistoryError, solve_hidden_batch
from kron_reduce.models.machine_local_topology import _radial_coordinate
from kron_reduce.models.noise_robustness_mhe import (
    PRIOR_WEIGHT,
    _check_model,
    _JointHistory,
    _lower_median,
    _mad,
    history_diagnostics,
)
from kron_reduce.models.unified_discovery_m3 import predict_m3

KINEMATIC_HUBER = 1.5
TUKEY_CUTOFF = 4.685
KINEMATIC_SCALE_FLOOR = 1e-4
SIGNAL_ENERGY_FLOOR = .01
MODEL_NOISE = .05
VISIBLE_PRIOR = .01
LM_ITERATIONS = 7
LM_TRIALS = 3
LM_INITIAL = 1e-3
LM_MIN = 1e-9
LM_MAX = 1e9


@lru_cache(maxsize=4)
def _quadratic_design(device):
    """All three-frame interpolants, precomputed once on the fixed 10 ms clock."""
    t = torch.arange(-5, 6, dtype=torch.float64) / 5.
    design = torch.stack((torch.ones_like(t), t, t.square()), dim=1)
    triples = torch.tensor(list(combinations(range(11), 3)))
    selectors = torch.eye(11, dtype=t.dtype)[triples]
    operators = torch.linalg.solve(design[triples], selectors)
    return design.to(device), operators.reshape(-1, 11).to(device)


def _second_difference_noise(normalized):
    return _mad(normalized.diff(n=2, dim=1), dim=1) / math.sqrt(6.)


def _tukey(standardized):
    return (1. - (standardized / TUKEY_CUTOFF).square()).clamp_min(0.).square()


@torch.no_grad()
def kinematic_cleaner(observed, scale):
    """Fit each physical channel independently using only the eleven input frames.

    The polynomial and the cleaned observations retain the input units. All noise
    scales, fitted signal energy and threshold decisions use normalized channels.
    """
    if (observed.ndim != 3 or not len(observed) or observed.shape[1:] != (11, 20)
            or scale.shape != (20,) or observed.dtype != torch.float64 or scale.dtype != torch.float64
            or observed.device != scale.device or not torch.isfinite(observed).all()
            or not (torch.isfinite(scale) & (scale > 0)).all()):
        raise ValueError("eleven finite float64 frames, twenty channels and positive scales required")
    normalized = observed / scale
    design, operators = _quadratic_design(observed.device)
    # With up to two corrupted frames, a majority of the 165 exact interpolants
    # still use exclusively uncorrupted observations, including at either endpoint.
    all_fits = (operators @ normalized).reshape(len(observed), -1, 3, 20)
    coeff = _lower_median(all_fits, dim=1).squeeze(1)
    diff_noise = _second_difference_noise(normalized).clamp_min(KINEMATIC_SCALE_FLOOR)
    for stage in range(4):
        residual = normalized - design @ coeff
        sigma = torch.maximum(_mad(residual, dim=1), diff_noise)
        innovation = residual.abs() / sigma
        weight = (KINEMATIC_HUBER / innovation.clamp_min(KINEMATIC_HUBER)
                  if stage < 2 else _tukey(innovation))
        weighted_transpose = design.T * weight.transpose(1, 2)[:, :, None, :]
        rhs = weighted_transpose @ normalized.transpose(1, 2).unsqueeze(-1)
        gram = weighted_transpose @ design
        solution, info = torch.linalg.solve_ex(gram, rhs)
        solution = solution.squeeze(-1).transpose(1, 2)
        # A redescending fit can lose its third distinct frame. Retain the
        # previous robust polynomial for that channel if the new fit is undefined.
        valid = ((weight > 0).sum(1) >= 3) & (info == 0) & solution.isfinite().all(dim=1)
        coeff = torch.where(valid[:, None], solution, coeff)

    fitted = design @ coeff
    residual = normalized - fitted
    sigma = torch.maximum(_mad(residual, dim=1), diff_noise)
    weight = _tukey(residual.abs() / sigma)
    clean = torch.where(weight > 0, observed, fitted * scale)
    channel_noise = _second_difference_noise(clean / scale).squeeze(1)
    smooth = fitted * scale
    terminal = torch.lerp(smooth[:, -1], observed[:, -1], weight[:, -1])
    return {"smooth": smooth, "clean": clean, "terminal": terminal, "weight": weight,
            "kinematic_scale": sigma.squeeze(1), "channel_noise": channel_noise,
            "signal_energy": fitted.square().mean(dim=(1, 2)),
            "sigma_hat": channel_noise.square().mean(dim=1).sqrt()}


def _project_visible(visible, mass):
    blocks = visible.reshape(-1, 2, 10)
    return (blocks - (blocks @ mass)[..., None] / mass.sum()).reshape(-1, 20)


def _fuse_terminal(model, history, terminal, channel_noise, tau_mis):
    gain = tau_mis**2 / (tau_mis**2 + channel_noise.square())
    visible = torch.lerp(history[:, -1, :20], terminal, gain)
    origin = torch.cat((_project_visible(visible, model.mass), history[:, -1, 20:]), dim=1)
    return origin, gain


@torch.no_grad()
def reblend_origin(result, model, tau_mis):
    """Reissue an identical fit with a different terminal gain for a paired comparison."""
    if not math.isfinite(tau_mis) or tau_mis <= 0:
        raise ValueError("a positive finite normalized mismatch scale is required")
    origin = result["origin"].clone()
    use = result["joint_active"]
    if use.any():
        origin[use], _ = _fuse_terminal(model, result["history"][use], result["terminal"][use],
                                        result["channel_noise"][use], tau_mis)
    return origin


def _initial_state(coordinates, lift, radius):
    memory, _ = _radial_coordinate(coordinates[:, 18:], radius)
    return torch.cat((coordinates[:, :18] @ lift[:20, :18].T, memory), dim=1)


def _objective(coordinates, center, prior, residual, outlier_weight, cutoff):
    magnitude = residual.abs()
    quadratic = torch.minimum(magnitude, cutoff)
    huber = quadratic * (magnitude - .5 * quadratic)
    displacement = prior @ (coordinates - center).unsqueeze(-1)
    return ((outlier_weight * huber).sum(dim=(1, 2)) + .5 * displacement.square().sum(dim=(1, 2))) / 220.


def _weighted_system(coordinates, center, tangent, residual, scale, radius, prior,
                     outlier_weight, cutoff):
    weighted = (outlier_weight * cutoff / torch.maximum(residual.abs(), cutoff)).sqrt()
    design = (tangent[:, :, :20] / scale[None, None, :, None]).flatten(1, 2)
    memory, radial_scale = _radial_coordinate(coordinates[:, 18:], radius)
    columns = design[:, :, 18:]
    radial = columns - (columns @ memory.unsqueeze(-1)) * memory[:, None, :] / radius**2
    design = torch.cat((design[:, :, :18], radial_scale[..., None] * radial), dim=2)
    rhs = torch.cat(((weighted * residual).flatten(1), (prior @ (center - coordinates).unsqueeze(-1)).squeeze(-1)),
                    dim=1)
    return torch.cat((design * weighted.flatten(1)[..., None], prior), dim=1), rhs


def _trial_history(integrator, initial, incumbent, model, power, equilibrium, max_step, spend):
    """Localize a divergent proposal so its damping cannot affect other events."""
    try:
        history, tangent = integrator(initial)
        return history, tangent, torch.ones(len(initial), dtype=torch.bool, device=initial.device)
    except UnresolvedHistoryError:
        history, tangent = (value.clone() for value in incumbent)
        finite = torch.zeros(len(initial), dtype=torch.bool, device=initial.device)
        for i in range(len(initial)):
            one = _JointHistory(model, power[i:i+1], equilibrium[i:i+1], max_step, spend)
            try:
                history[i:i+1], tangent[i:i+1] = one(initial[i:i+1])
                finite[i] = True
            except UnresolvedHistoryError:
                continue
        return history, tangent, finite


def _fit_joint(model, observed, clean, power, equilibrium, scale, map_lambda, *,
               max_step, spend, memory_radius, huber_delta):
    integrator = _JointHistory(model, power, equilibrium, max_step, spend)
    lift, radius = integrator.lift, 2. * memory_radius
    anchor = _project_visible(clean["smooth"][:, 0], model.mass)
    visible = anchor @ lift[:20, :18]
    center = torch.nn.functional.pad(visible, (0, 10))
    coordinates = center
    visible_rows = math.sqrt(11. * VISIBLE_PRIOR) * lift[:20] / scale[:, None]
    identity = torch.eye(28, device=scale.device, dtype=scale.dtype)
    memory_rows = (220. * map_lambda).sqrt()[:, None, None] * identity[None, 18:] / memory_radius
    prior = torch.cat((visible_rows.expand(len(observed), -1, -1), memory_rows), dim=1)
    cutoff = huber_delta * (clean["kinematic_scale"].square() + MODEL_NOISE**2).sqrt()[:, None]
    outlier_weight = clean["weight"]
    history, tangent = integrator(_initial_state(coordinates, lift, radius))
    residual = (observed - history[:, :, :20]) / scale
    loss = _objective(coordinates, center, prior, residual, outlier_weight, cutoff)
    objectives = [loss]
    damping = torch.full_like(loss, LM_INITIAL)
    for _ in range(LM_ITERATIONS):
        design, rhs = _weighted_system(coordinates, center, tangent, residual, scale, radius,
                                       prior, outlier_weight, cutoff)
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
            trial_loss = _objective(proposal, center, prior, trial_residual, outlier_weight, cutoff)
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
        objectives.append(loss)
    # Report conditioning of the final prior-augmented observations before LM damping.
    design, _ = _weighted_system(coordinates, center, tangent, residual, scale, radius, prior,
                                 outlier_weight, cutoff)
    spend("svd", len(observed))
    singular = torch.linalg.svdvals(design)
    return history, torch.stack(objectives, dim=1), singular[:, 0] / singular[:, -1]


@torch.no_grad()
def predict_mhe_v2(model, q, v, power, equilibrium, scale, count, *, max_step, spend,
                   memory_radius, gate_rad=.0003, alpha=.05, tau_mis=.025, inverse_snr=True,
                   huber_delta=1.5):
    """Issue one 30-state physical trajectory from the causal measurement window."""
    _check_model(model, max_step)
    if (type(count) is not int or count < 0 or type(inverse_snr) is not bool
            or not all(math.isfinite(x) and x > 0 for x in (memory_radius, tau_mis, huber_delta))
            or not all(math.isfinite(x) and x >= 0 for x in (alpha, gate_rad))):
        raise ValueError("invalid horizon, prior, noise scale, Huber cutoff or gate")
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
    origin = q.new_empty((len(q), 30))
    condition = q.new_empty((len(q),))
    objectives = q.new_full((len(q), LM_ITERATIONS + 1), torch.nan)
    gain = q.new_ones((len(q), 20))
    if (~joint).any():
        use = ~joint
        original = predict_m3(model, q[use], v[use], power[use], equilibrium[use], scale, 0,
                              max_step=max_step, spend=spend, memory_radius=memory_radius,
                              prior_weight=PRIOR_WEIGHT)
        singular = original["singular"]
        history[use], origin[use] = original["history"], original["forecast"][:, 0]
        condition[use] = singular[:, 0] / singular[:, -1]
    if joint.any():
        filtered = {key: value[joint] for key, value in clean.items()}
        fitted, objectives[joint], condition[joint] = _fit_joint(
            model, observed[joint], filtered, power[joint], equilibrium[joint], scale,
            map_lambda[joint], max_step=max_step, spend=spend, memory_radius=memory_radius,
            huber_delta=huber_delta)
        history[joint] = fitted
        origin[joint], gain[joint] = _fuse_terminal(
            model, fitted, filtered["terminal"], filtered["channel_noise"], tau_mis)
    forecast = (model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
                if count else origin[:, None])
    return {"forecast": forecast, "history": history, "origin": origin, "hidden_start": history[:, 0, 20:],
            "joint_active": joint, "kinematic_defect_rad": defect, "sigma_hat": clean["sigma_hat"],
            "signal_energy": clean["signal_energy"], "map_lambda": map_lambda, "condition": condition,
            "terminal": clean["terminal"], "channel_noise": clean["channel_noise"],
            "endpoint_gain": gain, "lm_objectives": objectives}
