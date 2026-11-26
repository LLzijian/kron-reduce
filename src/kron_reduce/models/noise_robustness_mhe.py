"""Causal 100 ms moving-horizon initialization of the frozen G2 ES-ROM."""

from __future__ import annotations

import math

import torch

from kron_reduce.models.differentiable_history import UnresolvedHistoryError, solve_hidden_batch
from kron_reduce.models.machine_local_topology import _radial_coordinate
from kron_reduce.models.unified_discovery_m3 import predict_m3

PRIOR_WEIGHT = 1e-6
MAD_FACTOR = 1.4826
STEPS = (.0025, .00125)


def coi_lift(mass):
    """Lift independent physical (9 q, 9 v, 10 z) directions to the full state."""
    if mass.shape != (10,) or mass.dtype != torch.float64 or not torch.all(torch.isfinite(mass) & (mass > 0)):
        raise ValueError("ten positive float64 inertias required")
    # A Householder reflection sends the inertia vector to the first coordinate axis.
    identity = torch.eye(10, dtype=mass.dtype, device=mass.device)
    reflector = mass/torch.linalg.vector_norm(mass)+identity[:, 0]
    basis = identity[:, 1:]-2.*reflector[:, None]*reflector[None, 1:]/reflector.square().sum()
    return torch.block_diag(basis, basis, identity)


class _JointHistory:
    """Reuse one projected RK4 substep for all five linearizations of this batch."""

    def __init__(self, model, power, equilibrium, max_step, spend):
        self.mass = model.mass
        self.mass_sum = model.mass.sum()
        self.lift = coi_lift(model.mass)
        self.field = model._rhs(power, equilibrium, tangent=True)
        self.substeps = round(.01/max_step)
        self.h = .01/self.substeps
        self.spend = spend
        self.graph = None

    def _substep(self, state, tangent):
        h, field = self.h, self.field
        a, da = field(state, tangent)
        b, db = field(state+h*a/2., tangent+h*da/2.)
        c, dc = field(state+h*b/2., tangent+h*db/2.)
        d, dd = field(state+h*c, tangent+h*dc)
        state = state+h*(a+2.*b+2.*c+d)/6.
        tangent = tangent+h*(da+2.*db+2.*dc+dd)/6.
        q, v, z = state.split(10, dim=1)
        dq, dv, dz = tangent.split(10, dim=1)
        mass = self.mass
        q = q-(q @ mass)[:, None]/self.mass_sum
        v = v-(v @ mass)[:, None]/self.mass_sum
        dq = dq-(dq*mass[None, :, None]).sum(dim=1, keepdim=True)/self.mass_sum
        dv = dv-(dv*mass[None, :, None]).sum(dim=1, keepdim=True)/self.mass_sum
        return torch.cat((q, v, z), dim=1), torch.cat((dq, dv, dz), dim=1)

    def __call__(self, initial):
        tangent = self.lift.expand(len(initial), -1, -1)
        states, tangents = [initial], [tangent]
        if initial.is_cuda:
            if self.graph is None:
                self.static_state = initial.clone()
                self.static_tangent = tangent.contiguous().clone()
                self.graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream(device=initial.device)
                with torch.cuda.device(initial.device), torch.cuda.graph(self.graph, stream=stream):
                    state, derivative = self._substep(self.static_state, self.static_tangent)
                    self.static_state.copy_(state)
                    self.static_tangent.copy_(derivative)
            self.static_state.copy_(initial)
            self.static_tangent.copy_(tangent)
        state = initial
        for _ in range(10):
            for _ in range(self.substeps):
                if initial.is_cuda:
                    self.graph.replay()
                else:
                    state, tangent = self._substep(state, tangent)
            states.append(self.static_state.clone() if initial.is_cuda else state)
            tangents.append(self.static_tangent.clone() if initial.is_cuda else tangent)
        self.spend("rollouts")
        self.spend("tangent_rhs", 40*self.substeps)
        self.spend("torch_rhs", 40*self.substeps)
        states, tangents = torch.stack(states, dim=1), torch.stack(tangents, dim=1)
        if not torch.isfinite(tangents).all() or not (states.abs() <= 1e12).all():
            raise UnresolvedHistoryError("nonfinite or divergent joint ES-ROM history or tangent")
        return states, tangents


def _check_model(model, max_step):
    if (max_step not in STEPS or getattr(model, "support", None) != ("G2",)
            or model.n_nodes != 10 or model.memory_dimension != 10 or model.latent_dimension != 10
            or model.mass.dtype != torch.float64):
        raise ValueError("requires the ten-machine float64 G2 ES-ROM and a registered RK4 step")


@torch.no_grad()
def joint_history_with_tangent(model, initial, power, equilibrium, *, max_step, spend):
    """Differentiate the 11-frame RK4 history in the physical coordinates of coi_lift."""
    _check_model(model, max_step)
    initial, power, equilibrium = map(model._tensor, (initial, power, equilibrium))
    if (initial.ndim != 2 or not len(initial) or initial.shape[1] != 30
            or power.shape != (len(initial), 10) or equilibrium.shape != power.shape
            or not all(torch.isfinite(x).all() for x in (initial, power, equilibrium))
            or power.sum(-1).abs().max() > 1e-10 or (initial[:, :10] @ model.mass).abs().max() > 1e-10
            or (initial[:, 10:20] @ model.mass).abs().max() > 1e-10):
        raise ValueError("balanced power, COI states and finite inputs required")
    return _JointHistory(model, power, equilibrium, max_step, spend)(initial)


def _mad(values, dim):
    center = values.median(dim=dim, keepdim=True).values
    return MAD_FACTOR*(values-center).abs().median(dim=dim, keepdim=True).values


def history_diagnostics(q, v, scale):
    """Estimate causal kinematic inconsistency (rad) and dimensionless channel noise."""
    defect = q[:, 2:]-q[:, :-2]-(.01/3.)*(v[:, :-2]+4.*v[:, 1:-1]+v[:, 2:])
    observed = torch.cat((q, v), dim=-1)
    second_difference = (observed[:, 2:]-2.*observed[:, 1:-1]+observed[:, :-2])/scale
    channel_noise = _mad(second_difference, dim=1)/math.sqrt(6.)
    return defect.abs().amax(dim=(1, 2)), channel_noise


def _huber_weights(residual, channel_noise, delta):
    channel_scale = torch.maximum(_mad(residual, dim=1), channel_noise).clamp_min(1e-4)
    standardized = residual.abs()/channel_scale
    element_weight = delta/standardized.clamp_min(delta)
    frame_score = standardized.median(dim=2, keepdim=True).values
    frame_cutoff = frame_score.median(dim=1, keepdim=True).values+delta*_mad(frame_score, 1).clamp_min(1.)
    frame_weight = frame_cutoff/torch.maximum(frame_cutoff, frame_score)
    return element_weight*frame_weight


def _fit_joint(model, observed, power, equilibrium, scale, channel_noise, map_lambda, *,
               max_step, spend, memory_radius, method, huber_delta):
    integrator = _JointHistory(model, power, equilibrium, max_step, spend)
    basis = integrator.lift[:10, :9]
    visible = torch.cat((observed[:, 0, :10] @ basis, observed[:, 0, 10:] @ basis), dim=1)
    coordinate = observed.new_zeros((len(observed), 10))
    radius = 2.*memory_radius
    # Two COI constraints leave 198 independent visible coordinates across the
    # window. The 220 physical channel residuals retain machine-specific weights.
    prior = (220.*map_lambda).sqrt()[:, None]/memory_radius
    prior_rows = torch.nn.functional.pad(
        torch.eye(10, device=observed.device, dtype=observed.dtype)[None]*prior[..., None], (18, 0))
    for iteration in range(5):
        z, radial_scale = _radial_coordinate(coordinate, radius)
        initial = torch.cat((visible[:, :9] @ basis.T, visible[:, 9:] @ basis.T, z), dim=1)
        history, tangent = integrator(initial)
        residual = (observed-history[:, :, :20])/scale
        design = (tangent[:, :, :20]/scale[None, None, :, None]).flatten(1, 2)
        z_design = design[:, :, 18:]
        z_design = radial_scale[..., None]*(z_design-(z_design @ z[..., None])*z[:, None, :]/radius**2)
        design = torch.cat((design[:, :, :18], z_design), dim=-1)
        if method == "huber" and iteration:
            weight = _huber_weights(residual, channel_noise, huber_delta).sqrt().flatten(1)
            design = design*weight[..., None]
            rhs = residual.flatten(1)*weight
        else:
            rhs = residual.flatten(1)
        fit = solve_hidden_batch(torch.cat((design, prior_rows), dim=1),
                                 torch.cat((rhs, -coordinate*prior), dim=1), spend=spend)
        update = fit["hidden_start"]
        visible = visible+update[:, :18]
        coordinate = coordinate+update[:, 18:]
    z, _ = _radial_coordinate(coordinate, radius)
    initial = torch.cat((visible[:, :9] @ basis.T, visible[:, 9:] @ basis.T, z), dim=1)
    history = model.rollout(initial, power, equilibrium, 10, max_step=max_step, spend=spend)
    singular = fit["singular"]
    return history, history[:, -1], singular[:, 0]/singular[:, -1]


@torch.no_grad()
def predict_mhe(model, q, v, power, equilibrium, scale, count, *, max_step, spend,
                memory_radius, method="huber", huber_delta=1.5, alpha=.01, gate_rad=1e-4):
    """Initialize and forecast using only the observed 11 frames and one held power vector.

    gn calls the frozen five-update ES-ROM initializer. Both joint estimators use
    the same causal gate, memory geometry, MAP prior, clock, and filtered terminal.
    The returned history and forecast each contain all thirty physical coordinates.
    """
    _check_model(model, max_step)
    if (method not in ("gn", "l2", "huber") or type(count) is not int or count < 0
            or not math.isfinite(memory_radius) or memory_radius <= 0
            or not math.isfinite(huber_delta) or huber_delta <= 0
            or not math.isfinite(alpha) or alpha < 0 or not math.isfinite(gate_rad) or gate_rad < 0):
        raise ValueError("invalid MHE method, horizon, radius, Huber cutoff, MAP factor or gate")
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    if (q.ndim != 3 or not len(q) or q.shape[1:] != (11, 10) or v.shape != q.shape
            or power.shape != (len(q), 10) or equilibrium.shape != power.shape or scale.shape != (20,)
            or not all(torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale))
            or torch.any(scale <= 0) or power.sum(-1).abs().max() > 1e-10
            or (q @ model.mass).abs().max() > 1e-10 or (v @ model.mass).abs().max() > 1e-10):
        raise ValueError("eleven COI frames, balanced held input, finite equilibrium, positive scales required")
    defect, channel_noise = history_diagnostics(q, v, scale)
    sigma = channel_noise.square().mean(dim=(1, 2)).sqrt()
    joint = (defect > gate_rad) & (method != "gn")
    map_lambda = PRIOR_WEIGHT+torch.where(joint, alpha*sigma.square(), 0.)
    history = q.new_empty((len(q), 11, 30))
    origin = q.new_empty((len(q), 30))
    condition = q.new_empty((len(q),))
    if (~joint).any():
        use = ~joint
        baseline = predict_m3(model, q[use], v[use], power[use], equilibrium[use], scale, 0,
                              max_step=max_step, spend=spend, memory_radius=memory_radius, prior_weight=PRIOR_WEIGHT)
        singular = baseline["singular"]
        history[use], origin[use] = baseline["history"], baseline["forecast"][:, 0]
        condition[use] = singular[:, 0]/singular[:, -1]
    if joint.any():
        observed = torch.cat((q[joint], v[joint]), dim=-1)
        history[joint], origin[joint], condition[joint] = _fit_joint(
            model, observed, power[joint], equilibrium[joint], scale, channel_noise[joint], map_lambda[joint],
            max_step=max_step, spend=spend, memory_radius=memory_radius, method=method, huber_delta=huber_delta)
    forecast = (model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
                if count else origin[:, None])
    return {"forecast": forecast, "history": history, "hidden_start": history[:, 0, 20:],
            "origin": origin, "joint_active": joint, "kinematic_defect_rad": defect,
            "sigma_hat": sigma, "map_lambda": map_lambda, "condition": condition}
