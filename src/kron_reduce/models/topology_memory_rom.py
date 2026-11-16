"""Explicit topology and memory dynamics derived from a shared storage function."""
from __future__ import annotations

from math import log

import torch

from kron_reduce.models.machine_local_topology import (
    TopologyMemoryModel,
    _edge_potential,
    _radial_coordinate,
)


class TopologyMemoryROM(TopologyMemoryModel):
    """State order: q, v, z, enabled common coordinate, enabled Helmert coordinates."""

    log_limit = log(2.)/2.

    def __init__(self, backbone, *, port, tau, common=False, spatial=False, saturation=False, train_scales=False):
        super().__init__(backbone, topology=False, port=port, tau=tau, train_scales=train_scales)
        if self.memory_dimension != self.n_nodes or (spatial and self.n_nodes < 3):
            raise ValueError("the ROM requires a local port per node and at least three nodes for spatial coupling")
        self.common = common
        self.spatial = spatial
        self.dynamic_dimension = int(common)+(self.n_nodes-1 if spatial else 0)
        self.latent_dimension += self.dynamic_dimension
        self.topology_stiffness = (1.,)*(int(common)+int(spatial))
        self.topology_rates = torch.nn.Parameter(self.mass.new_zeros(len(self.topology_stiffness)))
        self.register_parameter("saturation_strength", torch.nn.Parameter(self.mass.new_zeros(()))
                                if saturation else None)
        incident = self.incidence.abs()
        self.register_buffer("incident_weights", incident/incident.sum(dim=1, keepdim=True), persistent=False)
        self.register_buffer("spatial_basis", incident.T @ self.basis[:, :self.n_nodes-1] if spatial else None,
                             persistent=False)
        self.spatial_radius = (self.log_limit/float(self.spatial_basis.norm(dim=1).max()) if spatial else None)

    def _edge_modulation(self, coordinates):
        c, xi = coordinates[..., :int(self.common)], coordinates[..., int(self.common):]
        common = (2.*c/self.log_limit).tanh() if self.common else None
        radial, scale = _radial_coordinate(xi, self.spatial_radius) if self.spatial else (None, None)
        return common, radial, scale

    def edge_stiffness(self, state):
        common, radial, _ = self._edge_modulation(state[..., 3*self.n_nodes:])
        log_scale = self.log_limit*common if self.common else 0.
        if self.spatial:
            log_scale = log_scale+radial @ self.spatial_basis.T
        if self.dynamic_dimension:
            return self.effective_stiffness*log_scale.exp()
        return self.effective_stiffness

    def storage(self, state, equilibrium):
        coordinates = state[..., 3*self.n_nodes:]
        # Unit penalties fix the coordinate scale in both topology groups.
        return super().storage(state, equilibrium)+.5*coordinates.square().sum(dim=-1)

    def _rhs(self, power, equilibrium, *, tangent=False):
        if not self.dynamic_dimension and self.saturation_strength is None:
            return super()._rhs(power, equilibrium, tangent=tangent)

        n = self.n_nodes
        edge, port, incident = self.incidence, self.coupling, self.incident_weights
        ref = equilibrium @ edge
        inv_mass_projection = self.power_projection/self.mass[:, None]
        edge_acceleration = -(inv_mass_projection @ edge)
        port_acceleration = -(port/self.mass[:, None])
        damping_acceleration = -(inv_mass_projection @ self.effective_damping)
        applied = power @ inv_mass_projection.T
        rate = self.tau.reciprocal()
        mobility = self.topology_rates.clamp_min(0.)
        gamma = self.saturation_strength.clamp_min(0.) if self.saturation_strength is not None else 0.
        alpha = self.restoring_shape
        local_rho = self.angle_damping_rho*self.angle_damping_spatial_mix
        rho = self.angle_damping_rho-local_rho
        if local_rho:
            root = self._damping_root()
            projected_root = inv_mass_projection @ root

        def field(state, sensitivity=None):
            q, v, z, coordinates = state.split((n, n, n, self.dynamic_dimension), dim=-1)
            relative = q @ edge
            angle = relative+ref
            sine = angle.sin()-ref.sin()
            if alpha:
                first = 1.-2.*alpha*(2.*ref).cos()
                second = alpha*ref.cos()
                sine = first*sine+second*((2.*angle).sin()-(2.*ref).sin())
            common, radial, scale = self._edge_modulation(coordinates)
            log_scale = self.log_limit*common if self.common else 0.
            if self.spatial:
                log_scale = log_scale+radial @ self.spatial_basis.T
            k = self.effective_stiffness*log_scale.exp() if self.dynamic_dimension else self.effective_stiffness
            force = k*sine
            blocks = []
            if self.dynamic_dimension:
                edge_energy = k*_edge_potential(relative, ref, alpha)
            if self.common:
                log_slope = 2.*(1.-common.square())
                energy = edge_energy.sum(dim=-1, keepdim=True)
                blocks.append(-mobility[0]*(coordinates[..., :1]+log_slope*energy))
            if self.spatial:
                spatial_force = edge_energy @ self.spatial_basis
                inverse_radius2 = self.spatial_radius**-2
                dot = inverse_radius2*(radial*spatial_force).sum(dim=-1, keepdim=True)
                radial_gradient = spatial_force-radial*dot
                blocks.append(-mobility[-1]*(coordinates[..., int(self.common):]+scale*radial_gradient))

            local_s = relative.square() @ incident.T
            h = (1.+gamma*local_s).reciprocal()
            drive = v @ port
            damping = v @ damping_acceleration.T
            acceleration = applied+force @ edge_acceleration.T+damping+(h*z) @ port_acceleration.T
            if rho:
                s = relative.square().mean(dim=-1, keepdim=True)
                strength = rho*s/(1.+s)
                acceleration = acceleration-strength*damping
            if local_rho:
                local_strength = local_rho*local_s/(1.+local_s)
                root_velocity = v @ root
                acceleration = acceleration+(local_strength*root_velocity) @ projected_root.T
            derivative = torch.cat((v, acceleration, h*drive-rate*z, *blocks), dim=-1)
            if not tangent:
                return derivative

            sq, sv, sz, sa = sensitivity.split((n, n, n, self.dynamic_dimension), dim=1)
            sd = edge.T @ sq
            slope = angle.cos()
            if alpha:
                slope = first*slope+2.*second*(2.*angle).cos()
            log_response = log_slope[..., None]*sa[:, :1] if self.common else 0.
            if self.spatial:
                sxi = sa[:, int(self.common):]
                radial_dot = inverse_radius2*(radial[..., None]*sxi).sum(dim=1, keepdim=True)
                radial_response = scale[..., None]*(sxi-radial[..., None]*radial_dot)
                log_response = log_response+self.spatial_basis @ radial_response
            force_response = (k*slope)[..., None]*sd+force[..., None]*log_response
            if self.dynamic_dimension:
                energy_response = force[..., None]*sd+edge_energy[..., None]*log_response
            latent_response = []
            if self.common:
                curvature = -4./self.log_limit*common*log_slope
                latent_response.append(-mobility[0]*(
                    (1.+curvature*energy)[..., None]*sa[:, :1]
                    +log_slope[..., None]*energy_response.sum(dim=1, keepdim=True)))
            if self.spatial:
                sf = self.spatial_basis.T @ energy_response
                sdot = inverse_radius2*(radial_response*spatial_force[..., None]+radial[..., None]*sf).sum(
                    dim=1, keepdim=True)
                sscale = -scale[..., None].square()*radial_dot
                gradient_response = (scale[..., None]*(sf-radial_response*dot[..., None]-radial[..., None]*sdot)
                                     +sscale*radial_gradient[..., None])
                latent_response.append(-mobility[-1]*(sxi+gradient_response))
            local_ds = 2.*(incident @ (relative[..., None]*sd))
            sh = -gamma*h[..., None].square()*local_ds
            acceleration_response = (edge_acceleration @ force_response+damping_acceleration @ sv
                                     +port_acceleration @ (h[..., None]*sz+z[..., None]*sh))
            if rho:
                ds = 2.*(relative[..., None]*sd).mean(dim=1, keepdim=True)
                dstrength = rho*ds/(1.+s[..., None]).square()
                acceleration_response = (acceleration_response-strength[..., None]*(damping_acceleration @ sv)
                                         -damping[..., None]*dstrength)
            if local_rho:
                local_dstrength = local_rho*local_ds/(1.+local_s[..., None]).square()
                acceleration_response = acceleration_response+projected_root @ (
                    local_strength[..., None]*(root @ sv)+root_velocity[..., None]*local_dstrength)
            memory_response = h[..., None]*(port.T @ sv)+drive[..., None]*sh-rate[:, None]*sz
            return derivative, torch.cat((sv, acceleration_response, memory_response, *latent_response), dim=1)

        return field
