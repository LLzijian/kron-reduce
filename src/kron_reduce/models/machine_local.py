"""Diagonal hidden memory with local or global reciprocal ports.

The visible backbone is frozen. This research model does not replace the
deployed initializer or the historical nonlinear trainer.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch
from scipy.linalg import helmert

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import (
    NonlinearPassiveClosure,
    _features,
    complete_incidence,
)


class DiagonalMemoryModel(torch.nn.Module):
    """One decay rate per hidden coordinate; ten local slots for ten machines."""

    def __init__(self, backbone, *, port_kind, port, tau, encoder, feature_scale, train_stiffness=False):
        super().__init__()
        if not isinstance(backbone, NonlinearPassiveClosure):
            raise TypeError("backbone must be a NonlinearPassiveClosure")
        if port_kind not in {"local", "global"}:
            raise ValueError("port_kind must be local or global")
        n = backbone.n_nodes
        port, tau, encoder, feature_scale = (
            np.asarray(value, dtype=np.float64) for value in (port, tau, encoder, feature_scale)
        )
        r = len(tau) if tau.ndim == 1 else 0
        if (r < 1 or encoder.shape != (r, 8*n) or feature_scale.shape != (8*n,)
                or np.any(feature_scale <= 0) or np.any(tau <= .02) or np.any(tau >= 5.)
                or not all(np.isfinite(v).all() for v in (port, tau, encoder, feature_scale))):
            raise ValueError("invalid hidden rates, encoder, or feature scales")
        if port_kind == "local":
            if r != n or port.shape != (n,) or np.any(port <= 0):
                raise ValueError("local ports need one positive amplitude per machine")
            raw_port = np.log(port)
        else:
            if port.shape != (n-1, r):
                raise ValueError("global port must have shape (nodes-1, hidden)")
            raw_port = port
        self.n_nodes, self.latent_dimension = n, r
        self.port_kind, self.backbone_kind = port_kind, backbone.backbone
        tensor = lambda value: torch.tensor(value, dtype=torch.float64)
        self.port_parameter = torch.nn.Parameter(tensor(raw_port))
        fraction = (np.log(tau) - np.log(.02)) / np.log(5./.02)
        self.rate_parameter = torch.nn.Parameter(tensor(np.log(fraction / (1-fraction))))
        self.encoder = torch.nn.Parameter(tensor(encoder))
        self.register_parameter("log_stiffness_factor",
                                torch.nn.Parameter(tensor(0.)) if train_stiffness else None)
        for name, value in {
            "mass": backbone.mass, "stiffness": backbone.edge_stiffness,
            "damping": backbone.damping, "feature_scale": feature_scale,
            "incidence": complete_incidence(n), "basis": helmert(n, full=False).T,
            "power_projection": np.eye(n)-np.outer(backbone.mass, np.ones(n))/backbone.mass.sum(),
        }.items():
            self.register_buffer(name, tensor(value))

    @property
    def effective_stiffness(self):
        if self.log_stiffness_factor is None:
            return self.stiffness
        return self.stiffness * self.log_stiffness_factor.exp()

    def freeze_effective_stiffness(self):
        """Export the active dynamics in the fixed-stiffness model format."""
        frozen = deepcopy(self)
        with torch.no_grad():
            frozen.stiffness.copy_(self.effective_stiffness)
        frozen.log_stiffness_factor = None
        return frozen.requires_grad_(False).eval()

    @property
    def tau(self):
        return torch.exp(math.log(.02) + math.log(5./.02)*torch.sigmoid(self.rate_parameter))

    @property
    def coupling(self):
        if self.port_kind == "local":
            return self.power_projection * torch.exp(self.port_parameter)[None, :]
        return self.basis @ self.port_parameter

    def regularization(self):
        # A fixed reference dimension keeps surviving rows' penalty unchanged.
        return 1e-4 * self.encoder.square().sum() / (2 * self.encoder.shape[1])

    def closure(self):
        array = lambda value: value.detach().cpu().numpy()
        return NonlinearPassiveClosure(
            array(self.mass), array(self.effective_stiffness), array(self.damping), array(self.coupling),
            np.zeros((self.latent_dimension, self.latent_dimension)),
            np.diag(1/array(self.tau)), self.backbone_kind,
        )

    def _tensor(self, value):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float64:
                raise ValueError("research model requires float64")
            return value.to(self.mass.device)
        return torch.as_tensor(value, dtype=torch.float64, device=self.mass.device)

    def rhs(self, state, power, equilibrium):
        state, power, equilibrium = map(self._tensor, (state, power, equilibrium))
        n, r = self.n_nodes, self.latent_dimension
        if (state.ndim not in (1, 2) or state.shape[-1] != 2*n+r
                or power.shape != state.shape[:-1]+(n,) or equilibrium.shape != power.shape):
            raise ValueError("state, power and equilibrium dimensions differ")
        if not all(torch.isfinite(v).all() for v in (state, power, equilibrium)):
            raise ValueError("model inputs must be finite")
        return self._rhs(power, equilibrium)(state)

    def _rhs(self, power, equilibrium, *, tangent=False):
        """Prepare time-invariant terms once, within the current autograd graph."""
        n = self.n_nodes
        incidence, mass, projection, damping = self.incidence, self.mass, self.power_projection, self.damping
        coupling, tau, stiffness = self.coupling, self.tau, self.effective_stiffness
        sine = self.backbone_kind == "sine"
        reference = equilibrium @ incidence if sine else None
        reference_force = torch.sin(reference) if sine else None
        # Fuse the fixed linear terms into one small state matrix on CUDA.
        if mass.is_cuda:
            visible_linear = -(projection @ torch.cat((damping, coupling), dim=1))/mass[:, None]
            hidden_linear = torch.cat((coupling.T, -torch.diag(tau.reciprocal())), dim=1)
            linear = torch.cat((visible_linear, hidden_linear), dim=0)
            edge_force = -(projection @ incidence)/mass[:, None]
            edge_force = (edge_force * stiffness).T
            applied = (power @ projection.T)/mass

            def cuda_field(state, sensitivity=None):
                q, vz = state[..., :n], state[..., n:]
                angle = q @ incidence
                force = torch.sin(angle + reference) - reference_force if sine else angle
                linear_value = vz @ linear.T
                acceleration = linear_value[..., :n] + applied + force @ edge_force
                derivative = torch.cat((vz[..., :n], acceleration, linear_value[..., n:]), dim=-1)
                if not tangent:
                    return derivative
                sq, svz = sensitivity[:, :n], sensitivity[:, n:]
                slope = torch.cos(angle + reference)[..., None] if sine else 1.
                linear_response = linear @ svz
                edge_response = edge_force.T @ (slope*(incidence.T @ sq))
                return derivative, torch.cat((svz[:, :n], linear_response[:, :n]+edge_response,
                                              linear_response[:, n:]), dim=1)

            return cuda_field

        def field(state, sensitivity=None):
            q, v, z = state[..., :n], state[..., n:2*n], state[..., 2*n:]
            edge = q @ incidence
            angle = edge + reference if sine else edge
            force = torch.sin(angle)-reference_force if sine else edge
            f = power - (force*stiffness) @ incidence.T - v @ damping.T - z @ coupling.T
            derivative = torch.cat((v, (f @ projection.T)/mass, v @ coupling-z/tau), dim=-1)
            if not tangent:
                return derivative
            sq, sv, sz = sensitivity[:, :n], sensitivity[:, n:2*n], sensitivity[:, 2*n:]
            slope = stiffness * torch.cos(angle) if sine else stiffness
            force_sensitivity = incidence @ (slope[..., None]*(incidence.T @ sq)) + damping @ sv + coupling @ sz
            return derivative, torch.cat((sv, -(projection @ force_sensitivity)/mass[None, :, None],
                                          coupling.T @ sv-sz/tau[None, :, None]), dim=1)

        return field

    def _linear_transition(self, max_step, spend):
        n, r = self.n_nodes, self.latent_dimension
        size = 3*n+r
        a = self.mass.new_zeros((size, size))
        p, c = self.power_projection, self.coupling
        inv_mass = 1/self.mass[:, None]
        a[:n, n:2*n] = torch.eye(n, dtype=a.dtype, device=a.device)
        a[n:2*n, :n] = -inv_mass*(p @ (self.incidence*self.effective_stiffness) @ self.incidence.T)
        a[n:2*n, n:2*n] = -inv_mass*(p @ self.damping)
        a[n:2*n, 2*n:2*n+r] = -inv_mass*c
        a[n:2*n, 2*n+r:] = inv_mass*p
        a[2*n:2*n+r, n:2*n] = c.T
        a[2*n:2*n+r, 2*n:2*n+r] = -torch.diag(1/self.tau)
        identity = torch.eye(size, dtype=a.dtype, device=a.device)
        projection = identity.clone()
        projection[:n, :n] = p.T
        projection[n:2*n, n:2*n] = p.T
        substeps = round(.01/max_step)
        h = .01/substeps
        ah = a*h
        spend("matrix_products", 3)
        a2 = ah @ ah
        a3 = a2 @ ah
        a4 = a3 @ ah
        spend("matrix_products", 1)
        step = projection @ (identity+ah+a2/2+a3/6+a4/24)
        output_step = identity
        for _ in range(substeps):
            spend("matrix_products", 1)
            output_step = output_step @ step
        return output_step

    def _rk4_step(self, state, field, h, mass_sum):
        a = field(state)
        b = field(state+h*a/2)
        c = field(state+h*b/2)
        d = field(state+h*c)
        state = state+h*(a+2*b+2*c+d)/6
        n = self.n_nodes
        q, v = state[:, :n], state[:, n:2*n]
        return torch.cat((q-(q@self.mass)[:, None]/mass_sum,
                          v-(v@self.mass)[:, None]/mass_sum, state[:, 2*n:]), dim=1)

    def rollout(self, initial, power, equilibrium, count, *, max_step=.005,
                linear_fast=True, spend=None):
        """Uniform 10 ms outputs, using projected RK4 in either implementation."""
        initial, power, equilibrium = map(self._tensor, (initial, power, equilibrium))
        n, r = self.n_nodes, self.latent_dimension
        if (type(count) is not int or count < 0 or initial.ndim != 2 or not len(initial)
                or initial.shape[1] != 2*n+r or power.shape != (len(initial), n)
                or equilibrium.shape != power.shape
                or max_step not in (.01, .005, .0025, .00125, .000625, .0003125)):
            raise ValueError("invalid batch, clock, or RK4 step")
        if not all(torch.isfinite(v).all() for v in (initial, power, equilibrium)):
            raise ValueError("rollout inputs must be finite")
        if (torch.max(torch.abs(power.sum(dim=1))) > 1e-10
                or torch.max(torch.abs(initial[:, :n] @ self.mass)) > 1e-10
                or torch.max(torch.abs(initial[:, n:2*n] @ self.mass)) > 1e-10):
            raise ValueError("rollout needs balanced power and COI visible state")
        spend = spend or (lambda key, amount=1: None)
        spend("rollouts", 1)
        if self.backbone_kind == "linear" and linear_fast:
            f = self._linear_transition(max_step, spend)
            powers = torch.eye(f.shape[0], dtype=f.dtype, device=f.device)[None]
            block = f
            while len(powers) < count+1:
                spend("matrix_products", len(powers))
                powers = torch.cat((powers, powers @ block), dim=0)
                if len(powers) < count+1:
                    spend("matrix_products", 1)
                    block = block @ block
            state = torch.cat((initial, power), dim=1)
            spend("matrix_products", count+1)
            result = torch.einsum("tij,bj->bti", powers[:count+1], state)[..., :2*n+r]
        else:
            field = self._rhs(power, equilibrium)
            mass_sum = self.mass.sum()
            substeps = round(.01/max_step)
            h = .01/substeps

            def frame(state):
                for _ in range(substeps):
                    state = self._rk4_step(state, field, h, mass_sum)
                return state

            if initial.is_cuda and not torch.is_grad_enabled() and count >= 100:
                result = self._cuda_rollout(initial, count, frame, spend, 4*substeps)
            else:
                state, outputs = initial, [initial]
                for _ in range(count):
                    spend("torch_rhs", 4*substeps)
                    state = frame(state)
                    outputs.append(state)
                result = torch.stack(outputs, dim=1)
        if not torch.isfinite(result).all() or torch.max(torch.abs(result)) > 1e12:
            raise FloatingPointError("nonfinite or divergent hidden-memory rollout")
        return result

    @staticmethod
    def _cuda_rollout(initial, count, frame, spend, calls_per_frame):
        # A short graph amortizes launches without capturing a full 8 s trajectory.
        block = 10
        graph = torch.cuda.CUDAGraph()
        state = initial.clone()
        output = initial.new_empty((len(initial), count+1, initial.shape[1]))
        output[:, 0] = initial
        stream = torch.cuda.Stream(device=initial.device)
        with torch.cuda.device(initial.device), torch.cuda.graph(graph, stream=stream):
            next_state, states = state, []
            for _ in range(block):
                next_state = frame(next_state)
                states.append(next_state)
            frames = torch.stack(states, dim=1)
        for start in range(1, count-block+2, block):
            spend("torch_rhs", calls_per_frame*block)
            graph.replay()
            output[:, start:start+block] = frames
            state.copy_(frames[:, -1])
        for index in range(count-count % block+1, count+1):
            spend("torch_rhs", calls_per_frame)
            state = frame(state)
            output[:, index] = state
        return output

    def causal_rollout(self, history_q, history_v, power, equilibrium, features, count,
                       *, max_step=.005, linear_fast=True, spend=None, return_history=False):
        q, v, power, equilibrium, features = map(
            self._tensor, (history_q, history_v, power, equilibrium, features)
        )
        n = self.n_nodes
        if (q.ndim != 3 or q.shape[1:] != (11, n) or v.shape != q.shape
                or features.shape != (len(q), 8*n)):
            raise ValueError("initializer requires eleven observed frames and matching features")
        zh = (features/self.feature_scale) @ self.encoder.T
        initial = torch.cat((q[:, 0], v[:, 0], zh), dim=1)
        history = self.rollout(initial, power, equilibrium, 10, max_step=max_step,
                               linear_fast=linear_fast, spend=spend)
        origin = torch.cat((q[:, -1], v[:, -1], history[:, -1, 2*n:]), dim=1)
        future = self.rollout(origin, power, equilibrium, count, max_step=max_step,
                              linear_fast=linear_fast, spend=spend)
        return (history, future) if return_history else future

    def predict(self, history: CausalStateHistory, equilibrium, count, *, max_step=.0025):
        """Prediction accepts measured history and a static prior, never targets."""
        if not isinstance(history, CausalStateHistory):
            raise TypeError("prediction requires CausalStateHistory")
        if len(history.time) != 11 or not np.allclose(np.diff(history.time), .01, atol=1e-12, rtol=0):
            raise ValueError("prediction requires the 100 ms/11-sample contract")
        features = _features(history, np.asarray(equilibrium), 3)
        with torch.no_grad():
            return self.causal_rollout(history.delta[None], history.omega[None],
                                      history.constant_power[None], np.asarray(equilibrium)[None],
                                      features[None], count, max_step=max_step)[0].cpu().numpy()
