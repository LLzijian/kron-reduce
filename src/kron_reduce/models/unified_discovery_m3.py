"""Ten-machine structured discovery with reciprocal memory and passive topology dynamics."""

from __future__ import annotations

import math

import numpy as np
import torch

from kron_reduce.models.differentiable_history import solve_hidden_batch
from kron_reduce.models.first_swing_fit import predict_first_swing_fit
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.machine_local_topology import _radial_coordinate
from kron_reduce.models.nonlinear_history import history_with_tangent
from kron_reduce.physics.coordinates import coi_basis

MODULES = ("G1", "G2", "G3", "G4", "G5")
MODULE_PARAMETERS = {"G1": ("a",), "G2": ("rho",), "G3": ("mu_c",),
                     "G4": ("mu_xi", "kappa_xi"), "G5": ("gamma_s",)}
_DEFAULT = {"a": .05, "rho": 2/3, "mu_c": .05, "mu_xi": .05, "kappa_xi": .25, "gamma_s": .25}
_BOUNDS = {"a": (0., .6), "rho": (0., .85), "mu_c": (0., .5), "mu_xi": (0., .5),
           "kappa_xi": (.05, 1.), "gamma_s": (0., 8.)}
_LOG2 = math.log(2.)
_LOG4 = math.log(4.)


def _damping_coordinates(matrix):
    """Keep the registered A06 basis; handle other PSD known-truth backbones in the quotient."""
    basis = coi_basis(np.ones(len(matrix)))
    quotient = basis.T @ matrix @ basis
    diagonal = np.diag(quotient).copy()
    name = "uniform_coi_svd"
    if np.linalg.norm((basis*diagonal) @ basis.T-matrix, ord="fro") > 1e-12:
        diagonal, rotation = np.linalg.eigh(quotient)
        basis = basis @ rotation
        name = "uniform_coi_spectral"
    if np.any(diagonal <= 0.) or np.linalg.norm((basis*diagonal) @ basis.T-matrix, ord="fro") > 1e-12:
        raise ValueError("M3 requires a positive damping operator on the nine-dimensional zero-sum space")
    return basis, diagonal, name


def _edge_law(reference, a):
    """Shifted harmonic potential and its first two angle derivatives, sharing equilibrium factors."""
    sin_ref, cos_ref = reference.sin(), reference.cos()
    if a is not None:
        sin_ref2, cos_ref2 = (2*reference).sin(), (2*reference).cos()
        first, second = 1.-2.*a*cos_ref2, a*cos_ref

    def evaluate(relative, *, energy=False, slope=False):
        angle = reference+relative
        force = angle.sin()-sin_ref
        potential = cos_ref-angle.cos()-relative*sin_ref if energy else None
        curvature = angle.cos() if slope else None
        if a is not None:
            angle2 = 2.*angle
            force = first*force+second*(angle2.sin()-sin_ref2)
            if energy:
                potential = first*potential+second*(.5*(cos_ref2-angle2.cos())-relative*sin_ref2)
            if slope:
                curvature = first*curvature+2.*second*angle2.cos()
        return force, potential, curvature

    return evaluate


class UnifiedDiscoveryModel(DiagonalMemoryModel):
    """State order is q, v, ten memories, optional scalar c, optional nine Helmert coordinates."""

    def __init__(self, backbone, *, port, tau, support=(), module_values=None):
        if backbone.n_nodes != 10 or backbone.backbone != "sine" or np.any(backbone.edge_stiffness <= 0.):
            raise ValueError("M3 requires a ten-machine sine backbone with 45 positive edges")
        if np.any(np.asarray(port) < .01):
            raise ValueError("M3 local gains must be at least 0.01")
        support = tuple(support)
        if len(set(support)) != len(support) or set(support)-set(MODULES):
            raise ValueError("unknown or duplicated M3 support")
        super().__init__(backbone, port_kind="local", port=port, tau=tau,
                         encoder=np.zeros((10, 80)), feature_scale=np.ones(80))
        del self.encoder
        self._backbone = backbone
        self.support = tuple(group for group in MODULES if group in support)
        self.memory_dimension = self.n_nodes
        self.common = "G3" in self.support
        self.spatial = "G4" in self.support
        self.topology_dimension = int(self.common)+(self.n_nodes-1)*int(self.spatial)
        self.latent_dimension = self.memory_dimension+self.topology_dimension

        basis, d, self.damping_basis_kind = _damping_coordinates(backbone.damping)
        self.register_buffer("damping_basis", self._tensor(basis))
        self.register_buffer("damping_coefficients", self._tensor(d))
        self.edge_log_factor = torch.nn.Parameter(torch.zeros_like(self.stiffness))
        self.damping_log_factor = torch.nn.Parameter(torch.zeros_like(self.damping_coefficients))
        incident = self.incidence.abs()
        self.register_buffer("incident_weights", incident/incident.sum(dim=1, keepdim=True), persistent=False)
        columns = []
        if self.common:
            columns.append(torch.ones_like(self.stiffness)[:, None])
        if self.spatial:
            columns.append(incident.T @ self.basis)
        design = torch.cat(columns, dim=1) if columns else self.incidence.T[:, :0]
        self.register_buffer("topology_basis", design, persistent=False)

        active = {name for group in self.support for name in MODULE_PARAMETERS[group]}
        values = {} if module_values is None else dict(module_values)
        if values.keys()-active:
            raise ValueError("module values must belong to active groups")
        for name, (lower, upper) in _BOUNDS.items():
            parameter = None
            if name in active:
                value = float(values.get(name, _DEFAULT[name]))
                if not math.isfinite(value) or not lower <= value <= upper:
                    raise ValueError(f"{name} must be in [{lower}, {upper}]")
                parameter = torch.nn.Parameter(self.mass.new_tensor(value))
            self.register_parameter(name, parameter)

    @property
    def effective_stiffness(self):
        return self.stiffness*self.edge_log_factor.exp()

    @property
    def effective_damping_coefficients(self):
        return self.damping_coefficients*self.damping_log_factor.exp()

    @property
    def effective_damping(self):
        return (self.damping_basis*self.effective_damping_coefficients) @ self.damping_basis.T

    def _damping_root(self):
        return (self.damping_basis*self.effective_damping_coefficients.sqrt()) @ self.damping_basis.T

    @torch.no_grad()
    def project_parameters_(self):
        bounds = {**_BOUNDS, "edge_log_factor": (-_LOG2, _LOG2), "damping_log_factor": (-_LOG2, _LOG2),
                  "port_parameter": (math.log(.01), None)}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and name in bounds:
                parameter.clamp_(*bounds[name])
        return self

    @torch.no_grad()
    def with_support(self, support, module_values=None):
        """Copy shared coefficients exactly; optional values can explicitly replace active strengths."""
        copied = type(self)(self._backbone, port=np.ones(10), tau=np.ones(10), support=support,
                            module_values=module_values).to(device=self.mass.device)
        source = dict(self.named_parameters())
        overrides = set(module_values or {})
        for name, parameter in copied.named_parameters():
            if name in source and name not in overrides:
                parameter.copy_(source[name])
        return copied

    @torch.no_grad()
    def physical_parameters(self):
        return {"edge_stiffness": self.effective_stiffness.tolist(),
                "damping_coefficients": self.effective_damping_coefficients.tolist(),
                "port_gain": self.port_parameter.exp().tolist(), "tau": self.tau.tolist(),
                "damping_basis": self.damping_basis_kind,
                "modules": {group: {name: float(getattr(self, name)) for name in MODULE_PARAMETERS[group]}
                            for group in self.support}}

    def _topology_coefficients(self):
        penalty, mobility = [], []
        if self.common:
            penalty.append(self.mass.new_ones(1))
            mobility.append(self.mu_c.reshape(1))
        if self.spatial:
            penalty.append(self.kappa_xi.expand(self.n_nodes-1))
            mobility.append(self.mu_xi.expand(self.n_nodes-1))
        return torch.cat(penalty), torch.cat(mobility)

    def _modulation(self, coordinates):
        # Scale common and zero-sum spatial parts together; each edge respects the asymmetric bounds.
        u = coordinates @ self.topology_basis.T
        positive, negative = u.clamp_min(0.)/_LOG2, (-u).clamp_min(0.)/_LOG4
        denominator = 1.+(positive.pow(8)+negative.pow(8)).sum(dim=-1, keepdim=True)
        factor = denominator.pow(-1./8.)
        return u, factor, denominator, positive, negative

    def edge_stiffness(self, state):
        if not self.topology_dimension:
            return self.effective_stiffness
        u, factor, *_ = self._modulation(state[..., 3*self.n_nodes:])
        return self.effective_stiffness*(factor*u).exp()

    def damping_matrix(self, q):
        if self.rho is None:
            return self.effective_damping.expand(*q.shape[:-1], self.n_nodes, self.n_nodes)
        s = (q @ self.incidence).square() @ self.incident_weights.T
        root = self._damping_root()
        return self.effective_damping-(root*(self.rho*s/(1.+s))[..., None, :]) @ root.T

    def storage(self, state, equilibrium):
        state, equilibrium = map(self._tensor, (state, equilibrium))
        n = self.n_nodes
        q, v, z, coordinates = state.split((n, n, n, self.topology_dimension), dim=-1)
        _, potential, _ = _edge_law(equilibrium @ self.incidence, self.a)(q @ self.incidence, energy=True)
        value = (self.edge_stiffness(state)*potential).sum(dim=-1)
        value = value+.5*((self.mass*v.square()).sum(dim=-1)+z.square().sum(dim=-1))
        if self.topology_dimension:
            penalty, _ = self._topology_coefficients()
            value = value+.5*(penalty*coordinates.square()).sum(dim=-1)
        return value

    def dissipation(self, state, equilibrium):
        """Return vᵀD(q)v + zᵀRz + (∇χH)ᵀμ∇χH on each event."""
        state, equilibrium = map(self._tensor, (state, equilibrium))
        n = self.n_nodes
        q, v, z, coordinates = state.split((n, n, n, self.topology_dimension), dim=-1)
        damping = self.damping_matrix(q)
        value = (v*(damping @ v.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)+(z.square()/self.tau).sum(dim=-1)
        if self.topology_dimension:
            penalty, mobility = self._topology_coefficients()
            u, factor, denominator, positive, negative = self._modulation(coordinates)
            _, potential, _ = _edge_law(equilibrium @ self.incidence, self.a)(q @ self.incidence, energy=True)
            edge_energy = self.effective_stiffness*(u*factor).exp()*potential
            gradient, *_ = self._topology_gradient(coordinates, penalty, edge_energy, u, factor,
                                                   denominator, positive, negative)
            value = value+(mobility*gradient.square()).sum(dim=-1)
        return value

    def _topology_gradient(self, coordinates, penalty, edge_energy, u, factor, denominator, positive, negative):
        # The edge energy gradient in u is (D_u [factor*u])ᵀ edge_energy.
        f1 = -factor/(8.*denominator)
        grad_f = 8.*(positive.pow(7)/_LOG2-negative.pow(7)/_LOG4)
        dot = (u*edge_energy).sum(dim=-1, keepdim=True)
        gradient = penalty*coordinates+(factor*edge_energy+(f1*dot)*grad_f) @ self.topology_basis
        return gradient, f1, grad_f, dot

    def _rhs(self, power, equilibrium, *, tangent=False):
        n, width = self.n_nodes, self.topology_dimension
        edge, incident, basis = self.incidence, self.incident_weights, self.topology_basis
        stiffness, port, rate = self.effective_stiffness, self.coupling, self.tau.reciprocal()
        edge_law = _edge_law(equilibrium @ edge, self.a)
        inverse_mass_projection = self.power_projection/self.mass[:, None]
        applied = power @ inverse_mass_projection.T
        edge_acceleration = -(inverse_mass_projection @ edge)
        damping_acceleration = -(inverse_mass_projection @ self.effective_damping)
        port_acceleration = -(port/self.mass[:, None])
        if self.rho is not None:
            root = self._damping_root()
            projected_root = inverse_mass_projection @ root
        if width:
            penalty, mobility = self._topology_coefficients()

        def field(state, sensitivity=None):
            q, v, z, coordinates = state.split((n, n, n, width), dim=-1)
            relative = q @ edge
            sine, potential, slope = edge_law(relative, energy=bool(width), slope=tangent)
            if width:
                u, factor, denom, positive, negative = self._modulation(coordinates)
                k = stiffness*(factor*u).exp()
                energy = k*potential
                gradient, f1, grad_f, dot = self._topology_gradient(coordinates, penalty, energy, u, factor,
                                                                  denom, positive, negative)
                topology_rate = (-mobility*gradient,)
            else:
                k, topology_rate = stiffness, ()
            force = k*sine
            if self.rho is not None or self.gamma_s is not None:
                local_s = relative.square() @ incident.T
            drive = v @ port
            if self.gamma_s is not None:
                h = (1.+self.gamma_s*local_s).reciprocal()
                memory_port, velocity_port = h*z, h*drive
            else:
                memory_port, velocity_port = z, drive
            acceleration = (applied+force @ edge_acceleration.T+v @ damping_acceleration.T
                            +memory_port @ port_acceleration.T)
            if self.rho is not None:
                strength = self.rho*local_s/(1.+local_s)
                root_velocity = v @ root
                acceleration = acceleration+(strength*root_velocity) @ projected_root.T
            derivative = torch.cat((v, acceleration, velocity_port-rate*z, *topology_rate), dim=-1)
            if not tangent:
                return derivative

            sq, sv, sz, sc = sensitivity.split((n, n, n, width), dim=1)
            sd = edge.T @ sq
            force_response = (k*slope)[..., None]*sd
            if width:
                # Differentiate both the modulated restoring force and the gradient-flow torque.
                su = basis @ sc
                s_norm = (grad_f[..., None]*su).sum(dim=1, keepdim=True)
                sfactor = f1[..., None]*s_norm
                slog = factor[..., None]*su+u[..., None]*sfactor
                force_response = force_response+force[..., None]*slog
                se = force[..., None]*sd+energy[..., None]*slog
                sdot = (energy[..., None]*su+u[..., None]*se).sum(dim=1, keepdim=True)
                f2 = (9./64.)*factor/denom.square()
                hessian_diag = 56.*(positive.pow(6)/_LOG2**2+negative.pow(6)/_LOG4**2)
                s_grad_u = (factor[..., None]*se+energy[..., None]*sfactor
                            +grad_f[..., None]*((f2*dot)[..., None]*s_norm+f1[..., None]*sdot)
                            +(f1*dot*hessian_diag)[..., None]*su)
                topology_response = (-mobility[:, None]*(penalty[:, None]*sc+basis.T @ s_grad_u),)
            else:
                topology_response = ()
            if self.rho is not None or self.gamma_s is not None:
                slocal_s = 2.*(incident @ (relative[..., None]*sd))
            sdrive = port.T @ sv
            if self.gamma_s is not None:
                sh = -(self.gamma_s*h.square())[..., None]*slocal_s
                smemory_port = h[..., None]*sz+z[..., None]*sh
                svelocity_port = h[..., None]*sdrive+drive[..., None]*sh
            else:
                smemory_port, svelocity_port = sz, sdrive
            sacceleration = (edge_acceleration @ force_response+damping_acceleration @ sv
                             +port_acceleration @ smemory_port)
            if self.rho is not None:
                sstrength = self.rho*slocal_s/(1.+local_s[..., None]).square()
                sacceleration = sacceleration+projected_root @ (
                    strength[..., None]*(root @ sv)+root_velocity[..., None]*sstrength)
            response = torch.cat((sv, sacceleration, svelocity_port-rate[:, None]*sz, *topology_response), dim=1)
            return derivative, response

        return field

    def closure(self):
        if self.support:
            raise ValueError("M3 discovery groups require their explicit state and coefficient representation")
        return super().closure()


def predict_m3(model, q, v, power, equilibrium, scale, count, *, max_step, spend, memory_radius, prior_weight):
    """Infer ten memory coordinates from 100 ms, then freely evolve the augmented state."""
    if not model.topology_dimension:
        return predict_first_swing_fit(model, q, v, power, equilibrium, scale, count, max_step=max_step,
                                       spend=spend, memory_radius=memory_radius, prior_weight=prior_weight,
                                       soft_radius=True)
    q, v, power, equilibrium, scale = map(model._tensor, (q, v, power, equilibrium, scale))
    n, r, width = model.n_nodes, model.memory_dimension, model.topology_dimension
    if (q.ndim != 3 or not len(q) or q.shape[1:] != (11, n) or v.shape != q.shape
            or power.shape != (len(q), n) or equilibrium.shape != power.shape or scale.shape != (2*n,)
            or max_step not in (.005, .0025, .00125) or type(count) is not int or count < 0
            or not math.isfinite(memory_radius) or memory_radius <= 0.
            or not math.isfinite(prior_weight) or prior_weight <= 0.
            or not all(torch.isfinite(x).all() for x in (q, v, power, equilibrium, scale)) or torch.any(scale <= 0)):
        raise ValueError("M3 GN requires finite 100 ms histories and positive scale, prior and radius")
    if max(power.sum(1).abs().max(), (q@model.mass).abs().max(), (v@model.mass).abs().max()) > 1e-10:
        raise ValueError("history needs balanced power and COI visible states")

    observed = torch.cat((q, v), dim=2)
    radius = 2.*memory_radius
    prior = math.sqrt(prior_weight*10*2*n)/q.new_full((r,), memory_radius)
    prior_rows = torch.diag(prior).expand(len(q), -1, -1)
    coordinate = q.new_zeros((len(q), r))
    topology_start = q.new_zeros((len(q), width))
    for _ in range(5):
        hidden, radial_scale = _radial_coordinate(coordinate, radius)
        initial = torch.cat((observed[:, 0], hidden, topology_start), dim=1)
        history, tangent = history_with_tangent(model, initial, power, equilibrium, max_step=max_step, spend=spend)
        # Only memory columns enter the inverse problem; topology evolves from its known zero initial state.
        design = (tangent[:, 1:, :2*n, :r]/scale[None, None, :, None]).flatten(1, 2)
        design = radial_scale[..., None]*(design-(design @ hidden[..., None])*hidden[:, None, :]/radius**2)
        residual = ((observed[:, 1:]-history[:, 1:, :2*n])/scale).flatten(1)
        fit = solve_hidden_batch(torch.cat((design, prior_rows), dim=1),
                                 torch.cat((residual, -coordinate*prior), dim=1), spend=spend)
        coordinate = coordinate+fit["hidden_start"]

    hidden, _ = _radial_coordinate(coordinate, radius)
    history = model.rollout(torch.cat((observed[:, 0], hidden, topology_start), dim=1), power, equilibrium, 10,
                            max_step=max_step, spend=spend)
    origin = torch.cat((observed[:, -1], history[:, -1, 2*n:]), dim=1)
    forecast = model.rollout(origin, power, equilibrium, count, max_step=max_step, spend=spend)
    return {"forecast": forecast, "history": history, "hidden_start": hidden, "singular": fit["singular"]}
