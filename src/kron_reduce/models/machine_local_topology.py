"""Event-conditioned edge stiffness and reciprocal machine-local memory."""
from __future__ import annotations

import numpy as np
import torch

from kron_reduce.models.differentiable_history import solve_hidden_batch
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
        visible_linear = -(inv_mass_projection @ torch.cat((self.effective_damping, coupling), dim=1))
        memory_linear = torch.cat((coupling.T, -torch.diag(tau.reciprocal())), dim=1)
        applied = power @ inv_mass_projection.T

        def field(state, sensitivity=None):
            q, velocity_memory, a = state[..., :n], state[..., n:2*n+r], state[..., 2*n+r:]
            angle = q @ edge+ref
            shifted_sine = angle.sin()-ref.sin()
            k = self.edge_stiffness(state)
            force = k*shifted_sine
            acceleration = velocity_memory @ visible_linear.T+applied+force @ edge_acceleration.T
            derivative = torch.cat((velocity_memory[..., :n], acceleration, velocity_memory @ memory_linear.T,
                                    torch.zeros_like(a)), dim=-1)
            if not tangent:
                return derivative
            sq, svm, sa = sensitivity[:, :n], sensitivity[:, n:2*n+r], sensitivity[:, 2*n+r:]
            force_response = (k*angle.cos())[..., None]*(edge.T @ sq)+force[..., None]*(self.edge_basis @ sa)
            response = torch.cat((svm[:, :n], visible_linear @ svm+edge_acceleration @ force_response,
                                  memory_linear @ svm, torch.zeros_like(sa)), dim=1)
            return derivative, response

        return field


def _ball(x, radius):
    magnitude = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    return x*(radius/magnitude.clamp_min(radius))


def predict_from_topology_history(model, q, v, power, equilibrium, scale, count, *, max_step, spend):
    """Estimate both latent blocks from the same eleven observations."""
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    n, r = model.n_nodes, model.memory_dimension
    if (q.ndim != 3 or not len(q) or q.shape[1:] != (11, n) or v.shape != q.shape
            or power.shape != (len(q), n) or equilibrium.shape != power.shape or scale.shape != (2*n,)
            or max_step not in (.005, .0025, .00125) or type(count) is not int or count < 0
            or not all(torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale))
            or torch.any(scale <= 0)):
        raise ValueError("eleven finite history frames and positive scales required")
    if max(power.sum(1).abs().max(), (q@model.mass).abs().max(), (v@model.mass).abs().max()) > 1e-10:
        raise ValueError("history needs balanced power and COI visible states")
    observed = torch.cat((q, v), dim=2)
    hidden = q.new_zeros((len(q), model.latent_dimension))
    singular = None
    if model.latent_dimension:
        for _ in range(5):
            initial = torch.cat((observed[:, 0], hidden), dim=1)
            history, tangent = history_with_tangent(model, initial, power, equilibrium, max_step=max_step, spend=spend)
            design = (tangent[:, 1:, :2*n]/scale[None, None, :, None]).flatten(1, 2)
            residual = ((observed[:, 1:]-history[:, 1:, :2*n])/scale).flatten(1)
            result = solve_hidden_batch(design, residual, spend=spend)
            proposal = hidden+result["hidden_start"]
            hidden = torch.cat((_ball(proposal[:, :r], 1.), _ball(proposal[:, r:], .6)), dim=1)
            singular = result["singular"]
    history = model.rollout(torch.cat((observed[:, 0], hidden), dim=1), power, equilibrium, 10,
                            max_step=max_step, spend=spend)
    origin = torch.cat((observed[:, -1], history[:, -1, 2*n:]), dim=1)
    forecast = model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
    return {"hidden_start": hidden, "history": history, "forecast": forecast, "singular": singular}
