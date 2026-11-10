"""Small, isolated state-dependent port candidate for nonlinear closures."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import helmert

from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence


@dataclass(frozen=True, kw_only=True)
class StatePortClosure(NonlinearPassiveClosure):
    """Reciprocal port update with (n - 1) * r free physical coefficients."""

    delta: np.ndarray
    port_kind: str

    def __post_init__(self):
        super().__post_init__()
        delta = np.array(self.delta, dtype=float, copy=True)
        if self.port_kind not in {"constant", "state"}:
            raise ValueError("port_kind must be 'constant' or 'state'")
        if delta.ndim != 2 or delta.shape[0] != self.n_nodes - 1 or delta.shape[1] != self.latent_dimension:
            raise ValueError("delta must have shape (n_nodes - 1, latent_dimension)")
        if not np.isfinite(delta).all():
            raise ValueError("delta must be finite")
        delta.flags.writeable = False
        object.__setattr__(self, "delta", delta)

    @property
    def _basis(self):
        return helmert(self.n_nodes, full=False).T

    @property
    def _projections(self):
        m = self.mass
        q = np.eye(self.n_nodes) - np.ones((self.n_nodes, 1)) @ m[None, :] / m.sum()
        return q.T, q

    def _increment(self, q):
        value = self._basis @ self.delta
        if self.port_kind == "state":
            p, projection = self._projections
            value = p @ (np.sin(projection @ q)[:, None] * value)
        return value

    def rhs(self, state, power, equilibrium):
        base = super().rhs(state, power, equilibrium)
        n = self.n_nodes
        q, v, z = state[:n], state[n : 2 * n], state[2 * n :]
        increment = self._increment(q)
        p, projection = self._projections
        return base + np.r_[
            np.zeros(n),
            -(p @ (increment @ z)) / self.mass,
            increment.T @ (projection @ v),
        ]

    def torch_vector_field(self, delta=None):
        """Return a float64 Torch RHS retaining state and delta autograd paths."""
        import torch

        tensor = lambda value: torch.tensor(value, dtype=torch.float64)
        mass, stiffness = tensor(self.mass), tensor(self.edge_stiffness)
        damping, coupling = tensor(self.damping), tensor(self.coupling)
        skew, dissipation = tensor(self.latent_skew), tensor(self.latent_dissipation)
        basis = tensor(self._basis)
        p, projection = (tensor(value) for value in self._projections)
        incidence = tensor(complete_incidence(self.n_nodes))
        if delta is None:
            delta = tensor(self.delta)
        else:
            delta = torch.as_tensor(delta, dtype=torch.float64)
        if delta.shape != self.delta.shape or not torch.isfinite(delta).all():
            raise ValueError("delta must have the registered shape and finite values")
        n, r, kind = self.n_nodes, self.latent_dimension, self.port_kind

        def rhs(state, power, equilibrium):
            if state.shape != (2 * n + r,):
                raise ValueError("state has wrong dimension")
            power = torch.as_tensor(power, dtype=state.dtype, device=state.device)
            equilibrium = torch.as_tensor(equilibrium, dtype=state.dtype, device=state.device)
            if (state.dtype != torch.float64 or state.device.type != "cpu"
                    or power.shape != (n,) or equilibrium.shape != (n,)):
                raise ValueError("research RHS requires CPU float64 and node-aligned inputs")
            q, v, z = state[:n], state[n : 2 * n], state[2 * n :]
            edge, reference = incidence.T @ q, incidence.T @ equilibrium
            force = edge if self.backbone == "linear" else torch.sin(edge + reference) - torch.sin(reference)
            increment = basis @ delta
            if kind == "state":
                increment = p @ (torch.sin(projection @ q)[:, None] * increment)
            port = coupling + increment
            visible_force = power - incidence @ (stiffness * force) - damping @ v - port @ z
            visible_force = visible_force - visible_force.sum() * mass / mass.sum()
            return torch.cat(
                (v, visible_force / mass, coupling.T @ v + increment.T @ (projection @ v) + (skew - dissipation) @ z)
            )

        return rhs
