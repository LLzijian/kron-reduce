"""Diagonal hidden memory with local or global reciprocal ports.

The visible backbone is frozen. This research model does not replace the
deployed initializer or the historical nonlinear trainer.
"""
from __future__ import annotations

import math

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

    def __init__(self, backbone, *, port_kind, port, tau, encoder, feature_scale):
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
        for name, value in {
            "mass": backbone.mass, "stiffness": backbone.edge_stiffness,
            "damping": backbone.damping, "feature_scale": feature_scale,
            "incidence": complete_incidence(n), "basis": helmert(n, full=False).T,
            "power_projection": np.eye(n)-np.outer(backbone.mass, np.ones(n))/backbone.mass.sum(),
        }.items():
            self.register_buffer(name, tensor(value))

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
            array(self.mass), array(self.stiffness), array(self.damping), array(self.coupling),
            np.zeros((self.latent_dimension, self.latent_dimension)),
            np.diag(1/array(self.tau)), self.backbone_kind,
        )

    @staticmethod
    def _tensor(value):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float64 or value.device.type != "cpu":
                raise ValueError("research model requires CPU float64")
            return value
        return torch.tensor(value, dtype=torch.float64)

    def rhs(self, state, power, equilibrium):
        state, power, equilibrium = map(self._tensor, (state, power, equilibrium))
        n, r = self.n_nodes, self.latent_dimension
        if (state.ndim not in (1, 2) or state.shape[-1] != 2*n+r
                or power.shape != state.shape[:-1]+(n,) or equilibrium.shape != power.shape):
            raise ValueError("state, power and equilibrium dimensions differ")
        if not all(torch.isfinite(v).all() for v in (state, power, equilibrium)):
            raise ValueError("model inputs must be finite")
        q, v, z = state[..., :n], state[..., n:2*n], state[..., 2*n:]
        edge, reference = q @ self.incidence, equilibrium @ self.incidence
        force = edge if self.backbone_kind == "linear" else torch.sin(edge+reference)-torch.sin(reference)
        coupling = self.coupling
        f = power - (force*self.stiffness) @ self.incidence.T - v @ self.damping.T - z @ coupling.T
        return torch.cat((v, (f @ self.power_projection.T)/self.mass,
                          v @ coupling-z/self.tau), dim=-1)

    def _linear_transition(self, max_step, spend):
        n, r = self.n_nodes, self.latent_dimension
        size = 3*n+r
        a = torch.zeros((size, size), dtype=torch.float64)
        p, c = self.power_projection, self.coupling
        inv_mass = 1/self.mass[:, None]
        a[:n, n:2*n] = torch.eye(n, dtype=torch.float64)
        a[n:2*n, :n] = -inv_mass*(p @ (self.incidence*self.stiffness) @ self.incidence.T)
        a[n:2*n, n:2*n] = -inv_mass*(p @ self.damping)
        a[n:2*n, 2*n:2*n+r] = -inv_mass*c
        a[n:2*n, 2*n+r:] = inv_mass*p
        a[2*n:2*n+r, n:2*n] = c.T
        a[2*n:2*n+r, 2*n:2*n+r] = -torch.diag(1/self.tau)
        projection = torch.eye(size, dtype=torch.float64)
        projection[:n, :n] = p.T
        projection[n:2*n, n:2*n] = p.T
        substeps = round(.01/max_step)
        h = .01/substeps
        identity = torch.eye(size, dtype=torch.float64)
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
            powers = torch.eye(f.shape[0], dtype=torch.float64)[None]
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
            state, outputs = initial, [initial]
            substeps = round(.01/max_step)
            h = .01/substeps
            for _ in range(count):
                for _ in range(substeps):
                    spend("torch_rhs", 4)
                    a = self.rhs(state, power, equilibrium)
                    b = self.rhs(state+h*a/2, power, equilibrium)
                    c = self.rhs(state+h*b/2, power, equilibrium)
                    d = self.rhs(state+h*c, power, equilibrium)
                    state = state+h*(a+2*b+2*c+d)/6
                    q, v = state[:, :n], state[:, n:2*n]
                    state = torch.cat((q-(q@self.mass)[:, None]/self.mass.sum(),
                                       v-(v@self.mass)[:, None]/self.mass.sum(), state[:, 2*n:]), dim=1)
                outputs.append(state)
            result = torch.stack(outputs, dim=1)
        if not torch.isfinite(result).all() or torch.max(torch.abs(result)) > 1e12:
            raise FloatingPointError("nonfinite or divergent hidden-memory rollout")
        return result

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
                                      features[None], count, max_step=max_step)[0].numpy()
