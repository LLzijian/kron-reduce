"""Jointly trainable energy structured SINDy with two reciprocal memory scales.

State order is q, v, z_fast, z_slow, common topology, nine Helmert coordinates.
The power balance follows directly from the shared potential and reciprocal ports;
regional positivity of the shifted sine potential still needs a separate audit.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.linalg import helmert

from .machine_local import DiagonalMemoryModel
from .nonlinear_latent import NonlinearPassiveClosure, complete_incidence

GROUPS = ("common", "spatial", "saturation", "harmonic2", "harmonic3",
          "angle_damping", "edge_damping", "speed_damping", "slow_memory",
          "topology_port", "coupled_decay")


class JointEnergySINDy(torch.nn.Module):
    """One shared 50-state equation for ten machines and every source/regime."""

    log_limit = math.log(2.) / 2.

    def __init__(self, backbone: NonlinearPassiveClosure, *, port, tau,
                 train_stiffness=True):
        super().__init__()
        if not isinstance(backbone, NonlinearPassiveClosure) or backbone.backbone != "sine":
            raise TypeError("a sine NonlinearPassiveClosure is required")
        n = backbone.mass.size
        port = np.asarray(port, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        if tau.shape == (n,):
            tau = np.stack((tau, np.clip(4*tau, .5, 2.9)))
        if (n != 10 or port.shape != (n,) or tau.shape != (2, n)
                or np.any(port <= 0) or np.any(tau <= .02) or np.any(tau >= 5.)
                or not np.isfinite(port).all() or not np.isfinite(tau).all()):
            raise ValueError("ten positive ports and two ten-machine decay scales required")
        self.n_nodes, self.latent_dimension, self.backbone_kind = n, 3*n, "sine"
        self.memory_dimension, self.topology_dimension = 2*n, n
        tensor = lambda x: torch.tensor(np.array(x, copy=True), dtype=torch.float64)
        incidence = complete_incidence(n)
        incident = np.abs(incidence)
        basis = helmert(n, full=False).T
        spatial = incident.T @ basis
        spatial -= spatial.mean(axis=0, keepdims=True)
        damping = np.asarray(backbone.damping)
        if damping.shape != (n, n) or np.min(np.linalg.eigvalsh(damping)) < -1e-9:
            raise ValueError("backbone damping must be positive semidefinite")
        eig, vec = np.linalg.eigh(damping)
        damping_root = vec @ np.diag(np.sqrt(np.maximum(eig, 0.)))
        buffers = {
            "mass": backbone.mass, "base_stiffness": backbone.edge_stiffness,
            "incidence": incidence, "power_projection": np.eye(n)-np.outer(backbone.mass, np.ones(n))/sum(backbone.mass),
            "incident_weights": incident/incident.sum(axis=1, keepdims=True),
            "spatial_basis": spatial, "machine_basis": basis, "damping_root": damping_root,
        }
        for name, value in buffers.items():
            self.register_buffer(name, tensor(value))
        self.register_buffer("spatial_radius", tensor(self.log_limit / float(np.linalg.norm(spatial, axis=1).max())))
        self.log_stiffness = torch.nn.Parameter(tensor(np.zeros(len(backbone.edge_stiffness))),
                                                requires_grad=train_stiffness)
        self.log_global_stiffness = torch.nn.Parameter(tensor(0.))
        self.log_damping = torch.nn.Parameter(tensor(np.zeros(n)))
        self.log_port = torch.nn.Parameter(tensor(np.stack((port, .5*port))).log())
        lower, upper = np.array([.04, .45])[:, None], np.array([.4, 3.])[:, None]
        if np.any(tau <= lower) or np.any(tau >= upper):
            raise ValueError("fast and slow decay times must occupy disjoint intervals")
        fraction = (tau-lower)/(upper-lower)
        self.rate_parameter = torch.nn.Parameter(tensor(np.log(fraction/(1-fraction))))
        self.topology_mobility = torch.nn.Parameter(tensor([0., 0.]))
        self.topology_storage = torch.nn.Parameter(tensor([0., 0.]))
        self.saturation_parameter = torch.nn.Parameter(tensor(0.))
        self.topology_port_parameter = torch.nn.Parameter(tensor(0.))
        self.harmonics = torch.nn.Parameter(tensor(np.zeros(2)))
        self.angle_damping_parameter = torch.nn.Parameter(tensor(0.))
        self.edge_damping_parameter = torch.nn.Parameter(tensor(0.))
        self.speed_damping_parameter = torch.nn.Parameter(tensor(0.))
        self.memory_decay_factor = torch.nn.Parameter(tensor(.01*np.tile([1., -1.], n)))
        self.register_buffer("group_gate", torch.ones(len(GROUPS), dtype=torch.float64))
        self.register_buffer("edge_gate", torch.ones(len(backbone.edge_stiffness), dtype=torch.float64))

    @classmethod
    def from_mass(cls, mass):
        """Analytic, untrained complete-graph seed; no fitted checkpoint is read."""
        mass = np.asarray(mass, dtype=np.float64)
        if mass.shape != (10,) or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("ten finite positive inertias required")
        n, e = len(mass), len(mass)*(len(mass)-1)//2
        projection = np.eye(n)-np.outer(mass, np.ones(n))/mass.sum()
        seed = NonlinearPassiveClosure(mass, np.ones(e),
                                       projection@np.diag(.1*mass)@projection.T,
                                       np.zeros((n, 0)), np.zeros((0, 0)), np.zeros((0, 0)))
        model = cls(seed, port=np.full(n, .1),
                    tau=np.stack((np.linspace(.12, .3, n), np.linspace(.65, 1.8, n))))
        with torch.no_grad():
            model.damping_root.copy_(model._tensor(projection@np.diag(np.sqrt(.1*mass))))
        return model

    def _tensor(self, value):
        if isinstance(value, torch.Tensor) and value.dtype != torch.float64:
            raise ValueError("joint model requires float64")
        return torch.as_tensor(value, dtype=torch.float64, device=self.mass.device)

    @property
    def tau(self):
        lower = self.rate_parameter.new_tensor([[.04], [.45]])
        upper = self.rate_parameter.new_tensor([[.4], [3.]])
        return lower+(upper-lower)*torch.sigmoid(self.rate_parameter)

    @property
    def effective_stiffness(self):
        # Centering fixes the geometric mean of edge correction factors.
        offset = self.log_stiffness-self.log_stiffness.mean()
        return self.edge_gate*self.base_stiffness*torch.exp(self.log_global_stiffness+offset)

    @property
    def effective_damping(self):
        root = self.damping_root*torch.exp(.5*self.log_damping)[None, :]
        return root@root.T

    def _gate(self, name):
        return self.group_gate[GROUPS.index(name)]

    @property
    def effective_decay_factor(self):
        mask = torch.cat((torch.ones(self.n_nodes, dtype=self.mass.dtype, device=self.mass.device),
                          self._gate("slow_memory").expand(self.n_nodes)))
        return self._gate("coupled_decay")*self.memory_decay_factor*mask

    @torch.no_grad()
    def prune_groups(self, names):
        for name in names:
            if name not in GROUPS:
                raise ValueError(f"unknown optional group: {name}")
            self.group_gate[GROUPS.index(name)] = 0.

    @torch.no_grad()
    def prune_edges(self, indices):
        gate = self.edge_gate.clone()
        for index in indices:
            if type(index) is not int or not 0 <= index < len(self.edge_gate):
                raise ValueError(f"invalid edge index: {index}")
            gate[index] = 0.
        active = gate.bool()
        if not active.any():
            raise ValueError("at least one edge must remain")
        spatial = self.incidence.abs().T@self.machine_basis
        spatial = spatial-spatial[active].mean(0, keepdim=True)
        spatial[~active] = 0.
        self.edge_gate.copy_(gate)
        self.spatial_basis.copy_(spatial)
        radius_denominator = torch.linalg.vector_norm(spatial, dim=1).max()
        self.spatial_radius.copy_(self.log_limit/radius_denominator.clamp_min(1e-12))

    @torch.no_grad()
    def set_visible_seed(self, stiffness, damping):
        """Initialize physical edge and node coefficients from train-only weak solve."""
        stiffness, damping = map(self._tensor, (stiffness, damping))
        if (stiffness.shape != self.base_stiffness.shape or damping.shape != self.mass.shape
                or not torch.isfinite(stiffness).all() or not torch.isfinite(damping).all()
                or torch.any(stiffness <= 0) or torch.any(damping <= 0)
                or torch.any(self.edge_gate == 0)):
            raise ValueError("positive finite complete edge and node seeds required")
        reference_root = self.power_projection @ torch.diag(torch.sqrt(.1*self.mass))
        if not torch.allclose(self.damping_root, reference_root, rtol=1e-12, atol=1e-12):
            raise ValueError("node damping seed requires the from_mass analytic root")
        log_edge = torch.log(stiffness/self.base_stiffness)
        self.log_global_stiffness.copy_(log_edge.mean())
        self.log_stiffness.copy_(log_edge-log_edge.mean())
        self.log_damping.copy_(torch.log(damping/(.1*self.mass)))

    def group_norms(self):
        return {
            "common": self._gate("common")*torch.nn.functional.softplus(self.topology_mobility[0]),
            "spatial": self._gate("spatial")*torch.nn.functional.softplus(self.topology_mobility[1]),
            "saturation": self._gate("saturation")*torch.nn.functional.softplus(self.saturation_parameter),
            "topology_port": self._gate("topology_port")*(.5*torch.tanh(self.topology_port_parameter)).abs(),
            "harmonic2": self._gate("harmonic2")*(.5*torch.tanh(self.harmonics[0])).abs(),
            "harmonic3": self._gate("harmonic3")*(.5*torch.tanh(self.harmonics[1])).abs(),
            "angle_damping": self._gate("angle_damping")*.95*torch.sigmoid(self.angle_damping_parameter),
            "edge_damping": self._gate("edge_damping")*torch.nn.functional.softplus(self.edge_damping_parameter),
            "speed_damping": self._gate("speed_damping")*torch.nn.functional.softplus(self.speed_damping_parameter),
            "slow_memory": self._gate("slow_memory")*self.log_port[1].exp().norm(),
            "coupled_decay": self.effective_decay_factor.norm(),
        }

    def regularization(self):
        return 1e-4 * sum(value for value in self.group_norms().values())

    def effective_coefficients(self):
        """Physical coefficients for frozen equation export and independent checks."""
        return {
            "mass": self.mass, "incidence": self.incidence,
            "projection": self.power_projection, "machine_basis": self.machine_basis,
            "spatial_basis": self.spatial_basis, "spatial_radius": self.spatial_radius,
            "damping_root":
                self.damping_root*torch.exp(.5*self.log_damping)[None, :],
            "stiffness": self.effective_stiffness, "port": self.log_port.exp(),
            "tau": self.tau,
            "mobility": torch.nn.functional.softplus(self.topology_mobility)+1e-6,
            "topology_storage": torch.nn.functional.softplus(self.topology_storage)+1e-6,
            "saturation": self.group_norms()["saturation"],
            "topology_port": .5*torch.tanh(self.topology_port_parameter)*self._gate("topology_port"),
            "harmonics": .5*torch.tanh(self.harmonics)*self.group_gate[3:5],
            "angle_damping": self.group_norms()["angle_damping"],
            "edge_damping": self.group_norms()["edge_damping"],
            "speed_damping": self.group_norms()["speed_damping"],
            "group_gate": self.group_gate,
            "edge_gate": self.edge_gate,
            "memory_decay_factor": self.effective_decay_factor,
        }

    def _potential(self, relative, reference):
        angle = relative+reference
        base = reference.cos()-angle.cos()-relative*reference.sin()
        force = angle.sin()-reference.sin()
        for index, order in enumerate((2, 3)):
            phase = order*reference
            harmonic = (phase.cos()-(order*angle).cos())/order-relative*phase.sin()
            slope = (order*angle).sin()-phase.sin()
            coefficient = .5*torch.tanh(self.harmonics[index])*self._gate(f"harmonic{order}")
            base = base+coefficient*(reference.cos()*harmonic-order*phase.cos()*(reference.cos()-angle.cos()-relative*reference.sin()))
            force = force+coefficient*(reference.cos()*slope-order*phase.cos()*(angle.sin()-reference.sin()))
        return base, force

    def _components(self, state, equilibrium):
        n = self.n_nodes
        q, v, z1, z2, common, xi = state.split((n, n, n, n, 1, n-1), dim=-1)
        relative = q@self.incidence
        reference = equilibrium@self.incidence
        potential, force = self._potential(relative, reference)
        c = self._gate("common")*common
        common_shape = torch.tanh(2*c/self.log_limit)
        spatial_gate = self._gate("spatial")
        xi_active = spatial_gate*xi
        scale = torch.rsqrt(1.+xi_active.square().sum(-1, keepdim=True)/self.spatial_radius**2)
        radial = xi_active*scale
        log_scale = self.log_limit*common_shape+radial@self.spatial_basis.T
        stiffness = self.effective_stiffness*torch.exp(log_scale)
        local_angle = relative.square()@self.incident_weights.T
        gamma = torch.nn.functional.softplus(self.saturation_parameter)*self._gate("saturation")
        saturation = torch.reciprocal(1.+gamma*local_angle)
        machine_topology = c+xi_active@self.machine_basis.T
        eta = .5*torch.tanh(self.topology_port_parameter)*self._gate("topology_port")
        saturation = saturation*torch.exp(eta*torch.tanh(machine_topology))
        port = self.log_port.exp()
        ports = port*saturation[..., None, :]
        ports = torch.stack((ports[..., 0, :], ports[..., 1, :]*self._gate("slow_memory")), dim=-2)
        return q, v, z1, z2, common, xi, relative, potential, force, stiffness, local_angle, ports, common_shape, radial, scale

    def storage(self, state, equilibrium):
        state, equilibrium = self._tensor(state), self._tensor(equilibrium)
        _, v, z1, z2, common, xi, _, potential, _, stiffness, *_ = self._components(state, equilibrium)
        kappa = torch.nn.functional.softplus(self.topology_storage)+1e-6
        return (.5*(v.square()*self.mass).sum(-1)+
                (stiffness*potential).sum(-1)+.5*(z1.square()+z2.square()).sum(-1)+
                .5*kappa[0]*common.squeeze(-1).square()+.5*kappa[1]*xi.square().sum(-1))

    def rhs(self, state, power, equilibrium):
        state, power, equilibrium = map(self._tensor, (state, power, equilibrium))
        n = self.n_nodes
        if (state.ndim not in (1, 2) or state.shape[-1] != 5*n or
                power.shape != state.shape[:-1]+(n,) or equilibrium.shape != power.shape or
                not all(torch.isfinite(x).all() for x in (state, power, equilibrium))):
            raise ValueError("finite state, power and equilibrium with matching dimensions required")
        return self._rhs(power, equilibrium)(state)

    def _rhs(self, power, equilibrium, *, tangent=False):
        n = self.n_nodes
        projection = self.power_projection
        incidence = self.incidence
        rate = self.tau.reciprocal()
        mobility = torch.nn.functional.softplus(self.topology_mobility)+1e-6
        kappa = torch.nn.functional.softplus(self.topology_storage)+1e-6
        root = self.damping_root*torch.exp(.5*self.log_damping)[None, :]
        rho = .95*torch.sigmoid(self.angle_damping_parameter)*self._gate("angle_damping")
        edge_weight = torch.nn.functional.softplus(self.edge_damping_parameter)*self._gate("edge_damping")
        speed_weight = torch.nn.functional.softplus(self.speed_damping_parameter)*self._gate("speed_damping")
        decay_factor = self.effective_decay_factor

        def field(state):
            (_, v, z1, z2, common, xi, relative, potential, shape_force,
             stiffness, local_angle, ports, common_shape, radial, scale) = self._components(state, equilibrium)
            edge_speed = v@incidence
            root_velocity = v@root
            damping = ((1.-rho*local_angle/(1.+local_angle))*root_velocity)@root.T
            damping = damping+(edge_weight*relative.square()/(1.+relative.square())*edge_speed)@incidence.T
            damping = damping+speed_weight*v.pow(3)/(1.+v.square())
            memory_force = ports[..., 0, :]*z1+ports[..., 1, :]*z2
            force = power-(stiffness*shape_force)@incidence.T-damping-memory_force@projection.T
            acceleration = (force@projection.T)/self.mass
            projected_speed = v@projection
            memory_drive = torch.stack((ports[..., 0, :]*projected_speed,
                                        ports[..., 1, :]*projected_speed), dim=-2)
            memory_coupling = (torch.cat((z1, z2), dim=-1)@decay_factor)[..., None]*decay_factor
            edge_energy = stiffness*potential
            common_gradient = (2.*(1.-common_shape.square())*edge_energy.sum(-1, keepdim=True)
                               *self._gate("common")+kappa[0]*common)
            spatial_force = edge_energy@self.spatial_basis
            spatial_gradient = (self._gate("spatial")*scale*(spatial_force-
                radial*(radial*spatial_force).sum(-1, keepdim=True)/self.spatial_radius**2)
                +kappa[1]*xi)
            return torch.cat((v, acceleration,
                              memory_drive[..., 0, :]-rate[0]*z1-memory_coupling[..., :n],
                              memory_drive[..., 1, :]-rate[1]*z2-memory_coupling[..., n:],
                              -mobility[0]*common_gradient,
                              -mobility[1]*spatial_gradient), dim=-1)
        if not tangent:
            return field

        def field_with_tangent(state, sensitivity):
            if sensitivity.shape[:2] != state.shape or sensitivity.ndim != 3:
                raise ValueError("state sensitivity dimensions differ")
            value = field(state)
            (_, v, z1, z2, common, xi, relative, potential, shape_force,
             stiffness, local_angle, ports, common_shape, radial, scale) = self._components(state, equilibrium)
            sq, sv, sz1, sz2, sc, sxi = sensitivity.split((n, n, n, n, 1, n-1), dim=1)
            edge_response = incidence.T@sq
            angle = relative+equilibrium@incidence
            reference = equilibrium@incidence
            slope = angle.cos()
            for index, order in enumerate((2, 3)):
                coefficient = .5*torch.tanh(self.harmonics[index])*self._gate(f"harmonic{order}")
                slope = slope+coefficient*(order*reference.cos()*(order*angle).cos()
                                           -order*(order*reference).cos()*angle.cos())
            local_response = 2.*self.incident_weights@(relative[..., None]*edge_response)
            active_xi = self._gate("spatial")*xi
            active_sxi = self._gate("spatial")*sxi
            radial_scale_response = -scale[..., None]**3*(active_xi[..., None]*active_sxi).sum(1, keepdim=True)/self.spatial_radius**2
            radial_response = scale[..., None]*active_sxi+active_xi[..., None]*radial_scale_response
            common_response = (2.*(1.-common_shape.square())*self._gate("common"))[..., None]*sc
            log_response = common_response+self.spatial_basis@radial_response
            stiffness_response = stiffness[..., None]*log_response
            force_response = (stiffness*slope)[..., None]*edge_response+shape_force[..., None]*stiffness_response
            edge_energy = stiffness*potential
            edge_energy_response = potential[..., None]*stiffness_response+(stiffness*shape_force)[..., None]*edge_response
            gamma = torch.nn.functional.softplus(self.saturation_parameter)*self._gate("saturation")
            sat = torch.reciprocal(1.+gamma*local_angle)
            machine_topology = self._gate("common")*common+active_xi@self.machine_basis.T
            machine_response = self._gate("common")*sc+self.machine_basis@active_sxi
            eta = .5*torch.tanh(self.topology_port_parameter)*self._gate("topology_port")
            port_log_response = (-gamma*sat[..., None]*local_response+
                                 eta*(1.-torch.tanh(machine_topology).square())[..., None]*machine_response)
            ports_response = ports[..., None]*port_log_response[:, None]
            root_velocity = v@root
            root_response = root.T@sv
            angle_factor = 1.-rho*local_angle/(1.+local_angle)
            angle_factor_response = -rho*local_response/(1.+local_angle[..., None]).square()
            damping_response = (angle_factor[..., None]*root_response+
                                root_velocity[..., None]*angle_factor_response)
            damping_response = root@damping_response
            edge_speed = v@incidence
            edge_speed_response = incidence.T@sv
            edge_factor = relative.square()/(1.+relative.square())
            edge_factor_response = 2.*relative[..., None]*edge_response/(1.+relative[..., None].square()).square()
            damping_response = damping_response+edge_weight*incidence@(
                edge_factor[..., None]*edge_speed_response+edge_speed[..., None]*edge_factor_response)
            damping_response = damping_response+speed_weight*(
                v.square()*(3.+v.square())/(1.+v.square()).square())[..., None]*sv
            memory_response = (ports[..., 0, :, None]*sz1+ports_response[..., 0, :, :]*z1[..., None]
                               +ports[..., 1, :, None]*sz2+ports_response[..., 1, :, :]*z2[..., None])
            acceleration_response = ((-incidence@force_response-damping_response-
                                      projection@memory_response).transpose(1, 2)@projection.T).transpose(1, 2)/self.mass[None, :, None]
            projected_speed = v@projection
            projected_response = projection.T@sv
            z1_response = (ports[..., 0, :, None]*projected_response+
                           ports_response[..., 0, :, :]*projected_speed[..., None]-rate[0, :, None]*sz1)
            z2_response = (ports[..., 1, :, None]*projected_response+
                           ports_response[..., 1, :, :]*projected_speed[..., None]-rate[1, :, None]*sz2)
            coupled_response = ((decay_factor@torch.cat((sz1, sz2), dim=1))[:, None, :]
                                *decay_factor[None, :, None])
            z1_response = z1_response-coupled_response[:, :n]
            z2_response = z2_response-coupled_response[:, n:]
            common_slope = 2.*(1.-common_shape.square())*self._gate("common")
            common_slope_response = (-4.*common_shape*self._gate("common")
                                     *(1.-common_shape.square())*2./self.log_limit)[..., None]*sc
            common_gradient_response = (common_slope_response*edge_energy.sum(-1, keepdim=True)[..., None]+
                                        common_slope[..., None]*edge_energy_response.sum(1, keepdim=True)+
                                        kappa[0]*sc)
            spatial_force = edge_energy@self.spatial_basis
            spatial_force_response = self.spatial_basis.T@edge_energy_response
            radial_dot = (radial*spatial_force).sum(-1, keepdim=True)/self.spatial_radius**2
            radial_dot_response = (radial_response*spatial_force[..., None]+
                                   radial[..., None]*spatial_force_response).sum(1, keepdim=True)/self.spatial_radius**2
            spatial_core = spatial_force-radial*radial_dot
            spatial_core_response = (spatial_force_response-radial_response*radial_dot[..., None]
                                     -radial[..., None]*radial_dot_response)
            spatial_gradient_response = self._gate("spatial")*(
                radial_scale_response*spatial_core[..., None]+scale[..., None]*spatial_core_response)+kappa[1]*sxi
            response = torch.cat((sv, acceleration_response, z1_response, z2_response,
                                  -mobility[0]*common_gradient_response,
                                  -mobility[1]*spatial_gradient_response), dim=1)
            return value, response

        return field_with_tangent

    def rollout(self, initial, power, equilibrium, count, *, max_step=.005, spend=None):
        initial, power, equilibrium = map(self._tensor, (initial, power, equilibrium))
        n = self.n_nodes
        if (type(count) is not int or count < 0 or initial.ndim != 2 or not len(initial)
                or initial.shape[1] != 5*n or power.shape != (len(initial), n)
                or equilibrium.shape != power.shape or max_step not in
                (.01, .005, .0025, .00125, .000625, .0003125)
                or not all(torch.isfinite(x).all() for x in (initial, power, equilibrium))):
            raise ValueError("invalid finite batch or 10 ms RK4 clock")
        if (torch.max(torch.abs(power.sum(-1))) > 1e-10 or
                torch.max(torch.abs(initial[:, :n]@self.mass)) > 1e-10 or
                torch.max(torch.abs(initial[:, n:2*n]@self.mass)) > 1e-10):
            raise ValueError("balanced power and COI visible initial state required")
        spend = spend or (lambda key, amount=1: None)
        spend("rollouts", 1)
        field = self._rhs(power, equilibrium)
        substeps = round(.01/max_step)
        h = .01/substeps
        def frame(state):
            for _ in range(substeps):
                state = self._rk4_step(state, field, h, self.mass.sum())
            return state
        if initial.is_cuda and not torch.is_grad_enabled() and count >= 100:
            result = DiagonalMemoryModel._cuda_rollout(initial, count, frame, spend, 4*substeps)
        else:
            state, frames = initial, [initial]
            for _ in range(count):
                state = frame(state)
                spend("torch_rhs", 4*substeps)
                frames.append(state)
            result = torch.stack(frames, dim=1)
        if not torch.isfinite(result).all() or torch.max(torch.abs(result)) > 1e12:
            raise FloatingPointError("nonfinite or divergent joint rollout")
        return result

    def _rk4_step(self, state, field, h, mass_sum):
        a = field(state)
        b = field(state+h*a/2)
        c = field(state+h*b/2)
        d = field(state+h*c)
        state = state+h*(a+2*b+2*c+d)/6
        n = self.n_nodes
        q, v = state[:, :n], state[:, n:2*n]
        return torch.cat((q-(q@self.mass)[:, None]/mass_sum,
                          v-(v@self.mass)[:, None]/mass_sum, state[:, 2*n:]), dim=-1)
