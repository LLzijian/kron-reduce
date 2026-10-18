"""Passive hidden dynamics and an observable-history initializer.

The initializer is for a *known* linear closure. It tests state observability;
it is not a replacement for learning closure parameters on train data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from kron_reduce.physics.coordinates import coi_project


def _array(value: np.ndarray, ndim: int, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"invalid {name}")
    result.flags.writeable = False
    return result


class UnobservableHistoryError(ValueError):
    """The allowed observation window cannot resolve the hidden initial state."""


@dataclass(frozen=True)
class CausalStateHistory:
    """Measured states ending at the forecast origin, on a relative clock.

    The constant input must be valid over the entire window. A window crossing
    a topology/input switch needs a separate, registered observation model.
    """

    time: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    constant_power: np.ndarray

    def __post_init__(self) -> None:
        for name, ndim in (("time", 1), ("delta", 2), ("omega", 2), ("constant_power", 1)):
            object.__setattr__(self, name, _array(getattr(self, name), ndim, name))
        if (
            len(self.time) < 2
            or self.time[-1] != 0.0
            or np.any(self.time > 0.0)
            or np.any(np.diff(self.time) <= 0.0)
        ):
            raise ValueError(
                "history must increase strictly and end at zero without future samples"
            )
        shape = (len(self.time), len(self.constant_power))
        if self.delta.shape != shape or self.omega.shape != shape:
            raise ValueError("history state arrays must match the registered clock and nodes")


@dataclass(frozen=True)
class LatentEstimate:
    latent_at_origin: np.ndarray
    observation_rank: int
    singular_values: tuple[float, ...]
    condition_number: float
    normalized_residual_rms: float
    history_sample_count: int
    history_duration_s: float


@dataclass(frozen=True)
class PassiveLatentClosure:
    r"""M v'=-K q-D v-C z+u; z'=C.T v+(J-R)z; q'=v.

    H=1/2(q.T K q+v.T M v+z.T z) and
    H'=v.T u-v.T D v-z.T R z. Balanced ports preserve the COI constraint.
    This identity is a structural check, not a region-of-attraction certificate.
    """

    mass: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    coupling: np.ndarray
    latent_skew: np.ndarray
    latent_dissipation: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "mass",
            "stiffness",
            "damping",
            "coupling",
            "latent_skew",
            "latent_dissipation",
        ):
            object.__setattr__(
                self, name, _array(getattr(self, name), 1 if name == "mass" else 2, name)
            )
        n, r = len(self.mass), self.coupling.shape[1]
        if n < 2 or np.any(self.mass <= 0.0):
            raise ValueError("positive mass and at least two nodes are required")
        if (
            self.stiffness.shape != (n, n)
            or self.damping.shape != (n, n)
            or self.coupling.shape != (n, r)
            or self.latent_skew.shape != (r, r)
            or self.latent_dissipation.shape != (r, r)
        ):
            raise ValueError("passive closure parameter dimensions differ")
        for matrix in (self.stiffness, self.damping, self.latent_dissipation):
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
                raise ValueError("storage and dissipation matrices must be symmetric")
            if matrix.size and float(np.linalg.eigvalsh(matrix).min()) < -1e-12:
                raise ValueError("storage and dissipation matrices must be positive semidefinite")
        if not np.allclose(self.latent_skew, -self.latent_skew.T, rtol=0.0, atol=1e-12):
            raise ValueError("latent interconnection must be skew symmetric")
        if any(
            not np.allclose(value, 0.0, rtol=0.0, atol=1e-12)
            for value in (
                self.stiffness.sum(axis=0),
                self.damping.sum(axis=0),
                self.coupling.sum(axis=0),
            )
        ):
            raise ValueError("ports and mechanical operators must be balanced")

    @property
    def n_nodes(self) -> int:
        return len(self.mass)

    @property
    def latent_dimension(self) -> int:
        return self.coupling.shape[1]

    def generator(self) -> np.ndarray:
        n, r = self.n_nodes, self.latent_dimension
        return np.block(
            [
                [np.zeros((n, n)), np.eye(n), np.zeros((n, r))],
                [
                    -self.stiffness / self.mass[:, None],
                    -self.damping / self.mass[:, None],
                    -self.coupling / self.mass[:, None],
                ],
                [np.zeros((r, n)), self.coupling.T, self.latent_skew - self.latent_dissipation],
            ]
        )

    def _power(self, power: np.ndarray) -> np.ndarray:
        value = np.asarray(power, dtype=np.float64)
        if (
            value.shape != (self.n_nodes,)
            or not np.isfinite(value).all()
            or abs(value.sum()) > 1e-10
        ):
            raise ValueError("closure needs finite balanced power")
        return np.concatenate(
            (np.zeros(self.n_nodes), value / self.mass, np.zeros(self.latent_dimension))
        )

    def _state(self, delta: np.ndarray, omega: np.ndarray, latent: np.ndarray) -> np.ndarray:
        delta, omega, latent = (
            np.asarray(value, dtype=np.float64) for value in (delta, omega, latent)
        )
        if (
            delta.shape != (self.n_nodes,)
            or omega.shape != delta.shape
            or latent.shape != (self.latent_dimension,)
        ):
            raise ValueError("initial state dimensions differ")
        state = np.concatenate((delta, omega, latent))
        if (
            not np.isfinite(state).all()
            or max(abs(delta @ self.mass), abs(omega @ self.mass)) > 1e-10
        ):
            raise ValueError("initial state must be finite and in mass-weighted COI coordinates")
        return state

    def rollout(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        latent: np.ndarray,
        time: np.ndarray,
        power: np.ndarray,
        *,
        max_step: float = 0.01,
    ) -> np.ndarray:
        """Free RK4 rollout. No measured state is injected after the origin."""
        clock = np.asarray(time, dtype=np.float64)
        if (
            clock.ndim != 1
            or not len(clock)
            or not np.isfinite(clock).all()
            or clock[0] != 0.0
            or np.any(np.diff(clock) <= 0.0)
        ):
            raise ValueError("forecast clock must start at zero and increase strictly")
        if not np.isfinite(max_step) or not 0.0 < max_step <= 0.01:
            raise ValueError("maximum RK4 step must lie in (0, 0.01]")
        state = self._state(delta, omega, latent)
        matrix, forcing = self.generator(), self._power(power)
        result = [state.copy()]
        n = self.n_nodes
        for interval in np.diff(clock):
            count = max(1, int(np.ceil(interval / max_step - 1e-12)))
            step = interval / count
            for _ in range(count):
                k1 = matrix @ state + forcing
                k2 = matrix @ (state + step * k1 / 2.0) + forcing
                k3 = matrix @ (state + step * k2 / 2.0) + forcing
                k4 = matrix @ (state + step * k3) + forcing
                state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                state[:n] = coi_project(state[:n], self.mass)
                state[n : 2 * n] = coi_project(state[n : 2 * n], self.mass)
                if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e12:
                    raise FloatingPointError("divergent augmented rollout")
            result.append(state.copy())
        return np.stack(result)

    def estimate_latent(
        self,
        history: CausalStateHistory,
        *,
        delta_scale: float = 1.0,
        omega_scale: float = 1.0,
        maximum_condition_number: float = 1e8,
        minimum_singular_value: float = 1e-8,
    ) -> LatentEstimate:
        """Infer hidden state using observable output equations, without derivatives.

        Only the hidden state at the start of the allowed window is unknown.
        Full rank is required; regularization must not hide missing information.
        """
        if not isinstance(history, CausalStateHistory):
            raise TypeError("initializer accepts only a causal state history")
        if not all(np.isfinite(value) and value > 0 for value in (delta_scale, omega_scale)):
            raise ValueError("observation scales must be finite and positive")
        if not np.isfinite(maximum_condition_number) or maximum_condition_number < 1.0:
            raise ValueError("maximum observation condition number must be finite and >= 1")
        if not np.isfinite(minimum_singular_value) or minimum_singular_value <= 0.0:
            raise ValueError("minimum observation singular value must be finite and positive")
        n, r = self.n_nodes, self.latent_dimension
        if r == 0:
            raise UnobservableHistoryError("the closure has no hidden coordinates")
        if history.delta.shape[1] != n:
            raise ValueError("history and closure node counts differ")
        zero_state = self._state(history.delta[0], history.omega[0], np.zeros(r))
        for values in (history.delta, history.omega):
            if np.max(np.abs(values @ self.mass)) > 1e-10:
                raise ValueError("history must use mass-weighted COI coordinates")
        matrix = self.generator()
        forcing = self._power(history.constant_power)
        size = len(zero_state)
        affine = np.zeros((size + 1, size + 1))
        affine[:size, :size], affine[:size, size] = matrix, forcing
        weights = np.concatenate((np.full(n, 1.0 / delta_scale), np.full(n, 1.0 / omega_scale)))
        equations, residuals = [], []
        last_transition = np.eye(size + 1)
        for index, elapsed in enumerate(history.time - history.time[0]):
            transition = expm(affine * elapsed)
            predicted = transition[:size, :size] @ zero_state + transition[:size, size]
            measured = np.concatenate((history.delta[index], history.omega[index]))
            equations.append(weights[:, None] * transition[: 2 * n, 2 * n : size])
            residuals.append(weights * (measured - predicted[: 2 * n]))
            last_transition = transition
        design, target = np.concatenate(equations), np.concatenate(residuals)
        initial, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
        condition = float(singular[0] / singular[-1]) if rank == r else float("inf")
        if (
            rank != r
            or condition > maximum_condition_number
            or singular[-1] < minimum_singular_value
        ):
            raise UnobservableHistoryError(
                f"hidden state unresolved: rank {rank}/{r}, condition {condition}, smallest singular value {singular[-1]}"
            )
        initial_state = zero_state.copy()
        initial_state[2 * n :] = initial
        origin = last_transition[:size, :size] @ initial_state + last_transition[:size, size]
        latent = origin[2 * n :].copy()
        latent.flags.writeable = False
        return LatentEstimate(
            latent,
            int(rank),
            tuple(singular.tolist()),
            condition,
            float(np.sqrt(np.mean((design @ initial - target) ** 2))),
            len(history.time),
            float(-history.time[0]),
        )

    def forecast_from_history(
        self,
        history: CausalStateHistory,
        time: np.ndarray,
        *,
        max_step: float = 0.01,
        **observer_options: float,
    ) -> tuple[np.ndarray, LatentEstimate]:
        estimate = self.estimate_latent(history, **observer_options)
        return self.rollout(
            history.delta[-1],
            history.omega[-1],
            estimate.latent_at_origin,
            time,
            history.constant_power,
            max_step=max_step,
        ), estimate

    def energy_identity(self, state: np.ndarray, power: np.ndarray) -> dict[str, float]:
        n = self.n_nodes
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (2 * n + self.latent_dimension,):
            raise ValueError("energy state dimensions differ")
        q, v, z = state[:n], state[n : 2 * n], state[2 * n :]
        self._state(q, v, z)
        gradient = np.concatenate((self.stiffness @ q, self.mass * v, z))
        supply = float(v @ np.asarray(power))
        dissipation = float(v @ self.damping @ v + z @ self.latent_dissipation @ z)
        rate = float(gradient @ (self.generator() @ state + self._power(power)))
        return {
            "storage": float(0.5 * (q @ self.stiffness @ q + (self.mass * v) @ v + z @ z)),
            "supply": supply,
            "dissipation": dissipation,
            "storage_rate": rate,
            "identity_error": abs(rate - supply + dissipation),
        }

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError("latent checkpoint already exists")
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps({"schema": "passive_latent_closure.v1"})),
                **{name: getattr(self, name) for name in self.__dataclass_fields__},
            )

    @classmethod
    def load(cls, path: Path) -> PassiveLatentClosure:
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["metadata"])) != {"schema": "passive_latent_closure.v1"}:
                raise ValueError("latent checkpoint schema mismatch")
            return cls(**{name: data[name] for name in cls.__dataclass_fields__})
