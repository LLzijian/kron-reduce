"""Joint visible and hidden coefficients with optional physical function terms."""
from __future__ import annotations

import math

import numpy as np
import torch

from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.physics.coordinates import coi_basis


class JointMachineLocalModel(DiagonalMemoryModel):
    """Ten machine-local memories and the original sine backbone coefficient basis."""

    def __init__(self, backbone, *, port, tau, harmonic=False, modulated_damping=False):
        if backbone.n_nodes != 10 or backbone.backbone != "sine":
            raise ValueError("joint equations require a ten-machine sine backbone")
        super().__init__(backbone, port_kind="local", port=port, tau=tau,
                         encoder=np.zeros((10, 80)), feature_scale=np.ones(80))
        self.encoder.requires_grad_(False)
        basis = coi_basis(np.ones(10))
        d = np.diag(basis.T @ backbone.damping @ basis).copy()
        if (np.any(backbone.edge_stiffness <= 0) or np.any(d <= 0)
                or np.linalg.norm((basis * d) @ basis.T - backbone.damping, ord="fro") > 1e-12):
            raise ValueError("backbone must have positive edges and positive damping in its original basis")
        self.register_buffer("damping_basis", self._tensor(basis))
        self.register_buffer("damping_coefficients", self._tensor(d))
        self.edge_log_factor = torch.nn.Parameter(torch.zeros_like(self.stiffness))
        self.damping_log_factor = torch.nn.Parameter(torch.zeros_like(self.damping_coefficients))
        self.register_parameter("harmonic_strength", torch.nn.Parameter(self.mass.new_zeros(())) if harmonic else None)
        self.register_parameter("modulated_damping_strength",
                                torch.nn.Parameter(self.mass.new_zeros(())) if modulated_damping else None)

    @property
    def effective_stiffness(self):
        return self.stiffness * self.edge_log_factor.exp()

    @property
    def effective_damping(self):
        d = self.damping_coefficients * self.damping_log_factor.exp()
        return (self.damping_basis * d) @ self.damping_basis.T

    def project_dictionary_(self, shrink=0.):
        """Apply positivity, bounds and one-sided proximal shrinkage after an update."""
        if not math.isfinite(shrink) or shrink < 0:
            raise ValueError("shrink must be finite and nonnegative")
        with torch.no_grad():
            for value in (self.harmonic_strength, self.modulated_damping_strength):
                if value is not None:
                    value.sub_(shrink).clamp_(0., 2.)

    def potential(self, q, equilibrium):
        q, equilibrium = map(self._tensor, (q, equilibrium))
        edge, ref = q @ self.incidence, equilibrium @ self.incidence
        first = (torch.cos(ref) - torch.cos(ref + edge) - torch.sin(ref)*edge) * self.effective_stiffness
        result = first.sum(dim=-1)
        if self.harmonic_strength is not None:
            second = .5*(torch.cos(2*ref) - torch.cos(2*(ref+edge))) - torch.sin(2*ref)*edge
            result = result + (second*self.stiffness).sum(dim=-1)*self.harmonic_strength
        return result

    def damping_matrix(self, q):
        if self.modulated_damping_strength is None:
            return self.effective_damping
        edge = self._tensor(q) @ self.incidence
        weights = (8./self.incidence.shape[1])*self.modulated_damping_strength*edge.sin().square()
        return self.effective_damping + (self.incidence * weights[..., None, :]) @ self.incidence.T

    def closure(self):
        if self.harmonic_strength is not None or self.modulated_damping_strength is not None:
            raise ValueError("the fixed sine closure format cannot store dictionary terms")
        return super().closure()

    def freeze_effective_stiffness(self):
        raise ValueError("joint equations require their own frozen parameter representation")

    def _rhs(self, power, equilibrium, *, tangent=False):
        base = super()._rhs(power, equilibrium, tangent=tangent)
        harmonic, damping = self.harmonic_strength, self.modulated_damping_strength
        if harmonic is None and damping is None:
            return base
        n = self.n_nodes
        b = self.incidence
        edge_acceleration = -(self.power_projection @ b)/self.mass[:, None]
        ref = equilibrium @ b
        ref_harmonic = torch.sin(2*ref) if harmonic is not None else None
        h_weights = harmonic * self.stiffness if harmonic is not None else None
        d_weight = (8./b.shape[1])*damping if damping is not None else None

        def field(state, sensitivity=None):
            derivative = base(state, sensitivity)
            if tangent:
                derivative, response = derivative
                edge_response = b.T @ sensitivity[:, :n]
            edge = state[..., :n] @ b
            force = torch.zeros_like(edge)
            if harmonic is not None:
                angle = 2*(edge+ref)
                force = force + h_weights*(torch.sin(angle)-ref_harmonic)
                if tangent:
                    force_response = (2*h_weights*torch.cos(angle))[..., None]*edge_response
            else:
                force_response = 0.
            if damping is not None:
                speed = state[..., n:2*n] @ b
                weights = d_weight*edge.sin().square()
                force = force + weights*speed
                if tangent:
                    velocity_response = b.T @ sensitivity[:, n:2*n]
                    slope = d_weight*torch.sin(2*edge)*speed
                    force_response = force_response + weights[..., None]*velocity_response + slope[..., None]*edge_response
            derivative = torch.cat((derivative[..., :n], derivative[..., n:2*n]+force @ edge_acceleration.T,
                                    derivative[..., 2*n:]), dim=-1)
            if not tangent:
                return derivative
            response = torch.cat((response[:, :n], response[:, n:2*n]+edge_acceleration @ force_response,
                                  response[:, 2*n:]), dim=1)
            return derivative, response

        return field
