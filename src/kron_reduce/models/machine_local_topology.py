"""Event-conditioned edge stiffness and reciprocal machine-local memory."""
from __future__ import annotations

import numpy as np
import torch

from kron_reduce.models.differentiable_history import solve_hidden_ball, solve_hidden_batch
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_history import history_with_tangent


class TopologyMemoryModel(DiagonalMemoryModel):
    """State ordering q, v, memory, and zero-sum coupling coordinates."""

    def __init__(self, backbone, *, topology, port, tau, train_scales=False):
        n = backbone.n_nodes
        has_memory = port is not None
        super().__init__(backbone, port_kind="local", port=port if has_memory else np.ones(n),
                         tau=tau if has_memory else np.ones(n), encoder=np.zeros((n, 8*n)),
                         feature_scale=np.ones(8*n), train_stiffness=train_scales)
        if backbone.backbone != "sine" or (tau is None) != (not has_memory):
            raise ValueError("sine backbone and paired memory coefficients required")
        del self.encoder
        if not has_memory:
            del self.port_parameter
            del self.rate_parameter
        self.memory_dimension = n if has_memory else 0
        self.topology_dimension = n-1 if topology else 0
        self.latent_dimension = self.memory_dimension+self.topology_dimension
        self.angle_damping_rho = 0.
        self.angle_damping_spatial_mix = 0.
        self.register_parameter("log_damping_factor",
                                torch.nn.Parameter(self.mass.new_zeros(())) if train_scales else None)
        self.register_buffer("edge_basis", self.incidence.abs().T @ self.basis[:, :self.topology_dimension])

    @property
    def coupling(self):
        if self.memory_dimension:
            return super().coupling
        return self.power_projection[:, :0]

    @property
    def tau(self):
        if self.memory_dimension:
            return super().tau
        return self.mass[:0]

    @property
    def effective_damping(self):
        if self.log_damping_factor is None:
            return self.damping
        return self.damping*self.log_damping_factor.exp()

    def set_angle_damping(self, rho, *, spatial_mix=0.):
        """Set the frozen strength and spatial mix of the passive angle correction."""
        rho, spatial_mix = float(rho), float(spatial_mix)
        if not 0. <= rho <= 2./3.:
            raise ValueError("angle damping rho must be between zero and two-thirds")
        if not 0. <= spatial_mix <= 1.:
            raise ValueError("spatial damping mix must be between zero and one")
        self.angle_damping_rho = rho
        self.angle_damping_spatial_mix = spatial_mix

    def _damping_root(self):
        values, modes = torch.linalg.eigh(self.damping)
        tolerance = 64*torch.finfo(values.dtype).eps*values.abs().max()
        if (not torch.isfinite(values).all() or torch.any(values < -tolerance)
                or (self.damping-self.damping.T).abs().max() > tolerance):
            raise ValueError("spatial damping requires a symmetric PSD baseline")
        root = (modes*values.clamp_min(0.).sqrt()) @ modes.T
        return root if self.log_damping_factor is None else root*(.5*self.log_damping_factor).exp()

    def node_correction(self, state):
        return state[..., 2*self.n_nodes+self.memory_dimension:] @ self.basis[:, :self.topology_dimension].T

    def edge_stiffness(self, state):
        return self.effective_stiffness*torch.exp(
            state[..., 2*self.n_nodes+self.memory_dimension:] @ self.edge_basis.T)

    def closure(self):
        raise ValueError("topology-conditioned states require the explicit model")

    def freeze_effective_stiffness(self):
        raise ValueError("freeze both topology model scales and retain the explicit state representation")

    def storage(self, state, equilibrium):
        n, r = self.n_nodes, self.memory_dimension
        angle = state[..., :n] @ self.incidence
        ref = equilibrium @ self.incidence
        potential = (self.edge_stiffness(state)*(ref.cos()-(ref+angle).cos()-angle*ref.sin())).sum(-1)
        v, z = state[..., n:2*n], state[..., 2*n:2*n+r]
        return potential+.5*((v.square()*self.mass).sum(-1)+z.square().sum(-1))

    def _rhs(self, power, equilibrium, *, tangent=False):
        n, r = self.n_nodes, self.memory_dimension
        edge = self.incidence
        ref = equilibrium @ edge
        inv_mass_projection = self.power_projection/self.mass[:, None]
        edge_acceleration = -(inv_mass_projection @ edge)
        coupling = self.coupling
        tau = self.tau
        damping = self.effective_damping
        visible_linear = -(inv_mass_projection @ torch.cat((damping, coupling), dim=1))
        memory_linear = torch.cat((coupling.T, -torch.diag(tau.reciprocal())), dim=1)
        applied = power @ inv_mass_projection.T
        local_rho = self.angle_damping_rho*self.angle_damping_spatial_mix
        rho = self.angle_damping_rho-local_rho
        damping_linear = -(inv_mass_projection @ damping) if rho else None
        if local_rho:
            root = self._damping_root()
            projected_root = inv_mass_projection @ root
            incident = edge.abs()
            incident = incident/incident.sum(dim=1, keepdim=True)

        def field(state, sensitivity=None):
            q, velocity_memory, a = state[..., :n], state[..., n:2*n+r], state[..., 2*n+r:]
            relative_angle = q @ edge
            angle = relative_angle+ref
            shifted_sine = angle.sin()-ref.sin()
            k = self.edge_stiffness(state)
            force = k*shifted_sine
            acceleration = velocity_memory @ visible_linear.T+applied+force @ edge_acceleration.T
            if rho:
                s = relative_angle.square().mean(dim=-1, keepdim=True)
                strength = rho*s/(1.+s)
                damping_acceleration = velocity_memory[..., :n] @ damping_linear.T
                acceleration = acceleration-strength*damping_acceleration
            if local_rho:
                local_s = relative_angle.square() @ incident.T
                local_strength = local_rho*local_s/(1.+local_s)
                root_velocity = velocity_memory[..., :n] @ root
                acceleration = acceleration+(local_strength*root_velocity) @ projected_root.T
            derivative = torch.cat((velocity_memory[..., :n], acceleration, velocity_memory @ memory_linear.T,
                                    torch.zeros_like(a)), dim=-1)
            if not tangent:
                return derivative
            sq, svm, sa = sensitivity[:, :n], sensitivity[:, n:2*n+r], sensitivity[:, 2*n+r:]
            edge_response = edge.T @ sq
            force_response = (k*angle.cos())[..., None]*edge_response+force[..., None]*(self.edge_basis @ sa)
            acceleration_response = visible_linear @ svm+edge_acceleration @ force_response
            if rho:
                ds = 2.*(relative_angle[..., None]*edge_response).mean(dim=1, keepdim=True)
                dstrength = rho*ds/(1.+s[..., None]).square()
                acceleration_response = (acceleration_response-strength[..., None]*(damping_linear @ svm[:, :n])
                                         -damping_acceleration[..., None]*dstrength)
            if local_rho:
                local_ds = 2.*(incident @ (relative_angle[..., None]*edge_response))
                local_dstrength = local_rho*local_ds/(1.+local_s[..., None]).square()
                acceleration_response = acceleration_response+projected_root @ (
                    local_strength[..., None]*(root @ svm[:, :n])+root_velocity[..., None]*local_dstrength)
            response = torch.cat((svm[:, :n], acceleration_response,
                                  memory_linear @ svm, torch.zeros_like(sa)), dim=1)
            return derivative, response

        return field


def _ball(x, radius):
    magnitude = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    return x*(radius/magnitude.clamp_min(radius))


@torch.no_grad()
def _memory_ball_update(model, hidden, design, residual, observed, power, equilibrium, scale, prior, *,
                        radius, max_step, spend):
    spend("svd", len(hidden))
    observed_singular = torch.linalg.svdvals(design)
    spend("matrix_products", len(hidden))
    rhs = residual+(design @ hidden.unsqueeze(-1)).squeeze(-1)
    cost = residual.square().sum(dim=1)
    if prior is not None:
        design = torch.cat((design, torch.diag(prior).expand(len(hidden), -1, -1)), dim=1)
        rhs = torch.cat((rhs, torch.zeros_like(hidden)), dim=1)
        cost = cost+(prior*hidden).square().sum(dim=1)
    fit = solve_hidden_ball(design, rhs, radius, spend=spend)
    chosen, chosen_cost = hidden, cost
    accepted_step = torch.zeros_like(cost)
    trial_failures = torch.zeros_like(cost, dtype=torch.int64)
    for halving in range(9):
        pending = accepted_step == 0
        step = 2.**(-halving)
        trial = torch.where(pending[:, None], torch.lerp(hidden, fit["hidden_start"], step), chosen)
        try:
            history = model.rollout(torch.cat((observed[:, 0], trial), dim=1), power, equilibrium, 10,
                                    max_step=max_step, spend=spend)
        except FloatingPointError:
            # The scorer isolates a failed batch; only the failing event backs off.
            if len(hidden) > 1:
                raise
            trial_failures += 1
            continue
        trial_cost = ((observed[:, 1:]-history[:, 1:, :2*model.n_nodes])/scale).square().sum(dim=(1, 2))
        if prior is not None:
            trial_cost = trial_cost+(prior*trial).square().sum(dim=1)
        accept = pending & (trial_cost <= cost)
        chosen = torch.where(accept[:, None], trial, chosen)
        chosen_cost = torch.where(accept, trial_cost, chosen_cost)
        accepted_step = torch.where(accept, step, accepted_step)
        if (accepted_step > 0).all():
            break
    diagnostics = {"observed_singular": observed_singular, "multipliers": fit["multipliers"], "kkt": fit["kkt"],
                   "before": cost, "after": chosen_cost, "step": accepted_step, "trial_failures": trial_failures}
    return chosen, fit["singular"], diagnostics


def predict_from_topology_history(model, q, v, power, equilibrium, scale, count, *, max_step, spend,
                                  memory_radius=1., prior_weight=0., exact_memory_ball=False):
    """Estimate both latent blocks from the same eleven observations."""
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    n, r = model.n_nodes, model.memory_dimension
    if (q.ndim != 3 or not len(q) or q.shape[1:] != (11, n) or v.shape != q.shape
            or power.shape != (len(q), n) or equilibrium.shape != power.shape or scale.shape != (2*n,)
            or max_step not in (.005, .0025, .00125) or type(count) is not int or count < 0
            or not np.isfinite(memory_radius) or memory_radius <= 0
            or not np.isfinite(prior_weight) or prior_weight < 0
            or not all(torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale))
            or torch.any(scale <= 0)):
        raise ValueError("eleven finite history frames and positive scales required")
    if max(power.sum(1).abs().max(), (q@model.mass).abs().max(), (v@model.mass).abs().max()) > 1e-10:
        raise ValueError("history needs balanced power and COI visible states")
    observed = torch.cat((q, v), dim=2)
    hidden = q.new_zeros((len(q), model.latent_dimension))
    singular = None
    prior, diagnostics = None, []
    if exact_memory_ball and (not r or model.topology_dimension):
        raise ValueError("the exact ball candidate requires memory with fixed topology")
    if prior_weight:
        radii = q.new_tensor([memory_radius]*r + [.6]*model.topology_dimension)
        prior = np.sqrt(prior_weight*10*2*n)/radii
    if model.latent_dimension:
        for _ in range(5):
            initial = torch.cat((observed[:, 0], hidden), dim=1)
            history, tangent = history_with_tangent(model, initial, power, equilibrium, max_step=max_step, spend=spend)
            design = (tangent[:, 1:, :2*n]/scale[None, None, :, None]).flatten(1, 2)
            residual = ((observed[:, 1:]-history[:, 1:, :2*n])/scale).flatten(1)
            if exact_memory_ball:
                hidden, singular, record = _memory_ball_update(
                    model, hidden, design, residual, observed, power, equilibrium, scale, prior,
                    radius=memory_radius, max_step=max_step, spend=spend)
                diagnostics.append(record)
                continue
            if prior_weight:
                # The linearized solve penalizes the new absolute hidden state.
                design = torch.cat((design, torch.diag(prior).expand(len(q), -1, -1)), dim=1)
                residual = torch.cat((residual, -hidden*prior), dim=1)
            result = solve_hidden_batch(design, residual, spend=spend)
            proposal = hidden+result["hidden_start"]
            hidden = torch.cat((_ball(proposal[:, :r], memory_radius), _ball(proposal[:, r:], .6)), dim=1)
            singular = result["singular"]
    history = model.rollout(torch.cat((observed[:, 0], hidden), dim=1), power, equilibrium, 10,
                            max_step=max_step, spend=spend)
    origin = torch.cat((observed[:, -1], history[:, -1, 2*n:]), dim=1)
    forecast = model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
    result = {"hidden_start": hidden, "history": history, "forecast": forecast, "singular": singular}
    if diagnostics:
        result["history_ball"] = {name: torch.stack([row[name] for row in diagnostics]) for name in diagnostics[0]}
    return result
