"""Versioned passive-network benchmarks for modal and stability correctness.

The registry is generated from explicit equations rather than fitted data.
Every reported pole is obtained from the exact quotient-space linearization
of the same nonlinear ODE used for simulation.  Synthetic evidence never
substitutes for IEEE39 evidence; it falsifies the discovery algorithm where
the physical truth is otherwise unavailable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from kron_reduce.evaluation.modal import DmdMode, linearized_passive_network_modes
from kron_reduce.physics.coordinates import coi_basis, coi_project, coi_project_power


@dataclass(frozen=True)
class SyntheticTrajectory:
    event_id: str
    fault_line: str
    regime: str
    time: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    pm: np.ndarray
    hidden_position: np.ndarray
    hidden_velocity: np.ndarray


@dataclass(frozen=True)
class PassiveNetworkBenchmark:
    benchmark_id: str
    mass: np.ndarray
    incidence: np.ndarray
    coupling: np.ndarray
    node_damping: np.ndarray
    edge_damping: np.ndarray
    modal_loadings: np.ndarray
    modal_frequency_hz: float | None = None
    modal_damping_ratio: float | None = None
    stability_barrier: float | None = None

    def __post_init__(self) -> None:
        mass = np.asarray(self.mass, dtype=float).reshape(-1)
        incidence = np.asarray(self.incidence, dtype=float)
        coupling = np.asarray(self.coupling, dtype=float).reshape(-1)
        node_damping = np.asarray(self.node_damping, dtype=float).reshape(-1)
        edge_damping = np.asarray(self.edge_damping, dtype=float).reshape(-1)
        loadings = np.asarray(self.modal_loadings, dtype=float)
        if (
            incidence.shape[0] != len(mass)
            or incidence.shape[1] != len(coupling)
            or node_damping.shape != mass.shape
            or edge_damping.shape != coupling.shape
            or loadings.ndim != 2
            or loadings.shape[0] != incidence.shape[1]
            or np.any(mass <= 0)
            or np.any(coupling <= 0)
            or np.any(node_damping < 0)
            or np.any(edge_damping < 0)
        ):
            raise ValueError("invalid passive benchmark dimensions or parameter domain")
        if not all(
            np.isfinite(value).all()
            for value in (mass, incidence, coupling, node_damping, edge_damping, loadings)
        ):
            raise ValueError("benchmark parameters must be finite")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "incidence", incidence)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "node_damping", node_damping)
        object.__setattr__(self, "edge_damping", edge_damping)
        object.__setattr__(self, "modal_loadings", loadings)

    @property
    def rank(self) -> int:
        return int(self.modal_loadings.shape[1])

    def ground_truth_modes(self) -> list[DmdMode]:
        return linearized_passive_network_modes(
            self.mass,
            self.incidence,
            np.zeros_like(self.mass),
            self.node_damping,
            self.coupling,
            edge_damping=self.edge_damping,
            modal_loadings=self.modal_loadings,
            modal_frequency_hz=self.modal_frequency_hz,
            modal_damping_ratio=self.modal_damping_ratio,
        )

    def _quotient_linearization(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact state matrix and raw-coordinate reconstruction."""
        basis = coi_basis(self.mass)
        reduced_mass = basis.T @ np.diag(self.mass) @ basis
        stiffness = self.incidence @ np.diag(self.coupling) @ self.incidence.T
        damping = (
            np.diag(self.node_damping)
            + self.incidence @ np.diag(self.edge_damping) @ self.incidence.T
        )
        inverse_mass = np.linalg.inv(reduced_mass)
        quotient, rank = basis.shape[1], self.rank
        rotor = 2 * quotient
        system = np.zeros((rotor + 2 * rank, rotor + 2 * rank), dtype=float)
        system[:quotient, quotient:rotor] = np.eye(quotient)
        system[quotient:rotor, :quotient] = -inverse_mass @ (basis.T @ stiffness @ basis)
        system[quotient:rotor, quotient:rotor] = -inverse_mass @ (basis.T @ damping @ basis)
        if rank:
            if self.modal_frequency_hz is None or self.modal_damping_ratio is None:
                raise ValueError("modal-port benchmark requires frequency and damping")
            port = basis.T @ self.incidence @ self.modal_loadings
            angular = 2.0 * np.pi * float(self.modal_frequency_hz)
            position, velocity = rotor, rotor + rank
            system[quotient:rotor, velocity:] = -inverse_mass @ port
            system[position:velocity, velocity:] = np.eye(rank)
            system[velocity:, quotient:rotor] = port.T
            system[velocity:, position:velocity] = -(angular**2) * np.eye(rank)
            system[velocity:, velocity:] = (
                -2.0 * float(self.modal_damping_ratio) * angular * np.eye(rank)
            )
        return system, basis

    def modal_initial_state(
        self,
        mode_indices: Iterable[int],
        *,
        amplitude: float,
        phase: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        system, basis = self._quotient_linearization()
        eigenvalues, eigenvectors = np.linalg.eig(system)
        positive = sorted(
            (index for index, value in enumerate(eigenvalues) if value.imag > 1e-10),
            key=lambda index: eigenvalues[index].imag,
        )
        selected = tuple(int(index) for index in mode_indices)
        if not selected or any(index < 0 or index >= len(positive) for index in selected):
            raise ValueError("mode_indices must select positive-frequency ground-truth modes")
        state = np.zeros(system.shape[0], dtype=float)
        for offset, mode_index in enumerate(selected):
            vector = eigenvectors[:, positive[mode_index]] * np.exp(1j * (phase + 0.41 * offset))
            vector = vector / max(float(np.linalg.norm(vector)), np.finfo(float).eps)
            state += np.real(vector)
        state *= float(amplitude) / max(float(np.max(np.abs(state))), np.finfo(float).eps)
        quotient = basis.shape[1]
        rotor = 2 * quotient
        delta = coi_project(basis @ state[:quotient], self.mass)
        omega = coi_project(basis @ state[quotient:rotor], self.mass)
        return delta, omega, state[rotor : rotor + self.rank], state[rotor + self.rank :]

    def _field(
        self,
        state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        pm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        delta, omega, position, velocity = state
        edge_angle = self.incidence.T @ delta
        edge_speed = self.incidence.T @ omega
        edge_force = self.coupling * np.sin(edge_angle) + self.edge_damping * edge_speed
        if self.rank:
            edge_force = edge_force + self.modal_loadings @ velocity
        force = coi_project_power(
            np.asarray(pm, dtype=float).reshape(-1)
            - self.node_damping * omega
            - self.incidence @ edge_force,
            self.mass,
        )
        if self.rank:
            angular = 2.0 * np.pi * float(self.modal_frequency_hz)
            modal_acceleration = (
                self.modal_loadings.T @ edge_speed
                - angular**2 * position
                - 2.0 * float(self.modal_damping_ratio) * angular * velocity
            )
        else:
            modal_acceleration = np.empty(0, dtype=float)
        return omega, force / self.mass, velocity, modal_acceleration

    def simulate(
        self,
        *,
        event_id: str,
        fault_line: str,
        regime: str,
        time: np.ndarray,
        delta0: np.ndarray,
        omega0: np.ndarray,
        pm: np.ndarray | None = None,
        hidden_position0: np.ndarray | None = None,
        hidden_velocity0: np.ndarray | None = None,
    ) -> SyntheticTrajectory:
        t = np.asarray(time, dtype=float).reshape(-1)
        if len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("synthetic time grid must be strictly increasing")
        forcing = (
            np.zeros((len(t), len(self.mass)), dtype=float)
            if pm is None
            else np.asarray(pm, dtype=float)
        )
        if forcing.shape != (len(t), len(self.mass)):
            raise ValueError("synthetic Pm must have shape (time, node)")
        forcing = coi_project_power(forcing, self.mass)
        state = (
            coi_project(np.asarray(delta0, dtype=float).reshape(-1), self.mass),
            coi_project(np.asarray(omega0, dtype=float).reshape(-1), self.mass),
            np.zeros(self.rank)
            if hidden_position0 is None
            else np.asarray(hidden_position0, dtype=float),
            np.zeros(self.rank)
            if hidden_velocity0 is None
            else np.asarray(hidden_velocity0, dtype=float),
        )
        if state[2].shape != (self.rank,) or state[3].shape != (self.rank,):
            raise ValueError("hidden initial state does not match benchmark rank")
        outputs = tuple([value.copy()] for value in state)
        for index, step in enumerate(np.diff(t)):
            u0, u1 = forcing[index], forcing[index + 1]
            uh = 0.5 * (u0 + u1)
            k1 = self._field(state, u0)
            k2 = self._field(
                tuple(value + 0.5 * step * slope for value, slope in zip(state, k1, strict=True)),
                uh,
            )
            k3 = self._field(
                tuple(value + 0.5 * step * slope for value, slope in zip(state, k2, strict=True)),
                uh,
            )
            k4 = self._field(
                tuple(value + step * slope for value, slope in zip(state, k3, strict=True)),
                u1,
            )
            state = tuple(
                value + step * (a + 2 * b + 2 * c + d) / 6.0
                for value, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
            )
            state = (
                coi_project(state[0], self.mass),
                coi_project(state[1], self.mass),
                state[2],
                state[3],
            )
            if not all(np.isfinite(value).all() for value in state):
                raise FloatingPointError("synthetic passive network left its numerical domain")
            for output, value in zip(outputs, state, strict=True):
                output.append(value.copy())
        return SyntheticTrajectory(
            event_id=event_id,
            fault_line=fault_line,
            regime=regime,
            time=t,
            delta=np.stack(outputs[0]),
            omega=np.stack(outputs[1]),
            pm=forcing,
            hidden_position=np.stack(outputs[2]),
            hidden_velocity=np.stack(outputs[3]),
        )

    def as_model_sample(self, trajectory: SyntheticTrajectory) -> dict:
        static = {
            "mass": self.mass.copy(),
            "incidence": self.incidence.copy(),
            "equilibrium_delta": np.zeros_like(self.mass),
            "Pm_equilibrium": np.zeros_like(self.mass),
            "Pm_clear": trajectory.pm[0].copy(),
            "Pm0": trajectory.pm[0].copy(),
            "delta0": trajectory.delta[0].copy(),
            "omega0": trajectory.omega[0].copy(),
        }
        return {
            "event_id": trajectory.event_id,
            "model_inputs": {
                "static": static,
                "exogenous": {
                    "t": trajectory.time.copy(),
                    "Pm": trajectory.pm.copy(),
                    "interpolation": "linear",
                },
            },
            "training_context": {
                "targets": {
                    "t": trajectory.time.copy(),
                    "delta": trajectory.delta.copy(),
                    "omega": trajectory.omega.copy(),
                },
                "Pm_observed": trajectory.pm.copy(),
                "claim_type": "conditional_rollout_not_causal_forecast",
                "metadata": {
                    "source_dataset": "synthetic_benchmark_registry_v1",
                    "fault_line": trajectory.fault_line,
                    "regime": trajectory.regime,
                },
            },
        }

    def energy(self, trajectory: SyntheticTrajectory) -> np.ndarray:
        kinetic = 0.5 * np.sum(self.mass * trajectory.omega**2, axis=1)
        potential = np.sum(
            self.coupling * (1.0 - np.cos(trajectory.delta @ self.incidence)), axis=1
        )
        if self.rank:
            angular = 2.0 * np.pi * float(self.modal_frequency_hz)
            storage = 0.5 * np.sum(
                trajectory.hidden_velocity**2 + angular**2 * trajectory.hidden_position**2,
                axis=1,
            )
        else:
            storage = np.zeros(len(trajectory.time))
        return kinetic + potential + storage


def _path_incidence(nodes: int) -> np.ndarray:
    matrix = np.zeros((nodes, nodes - 1), dtype=float)
    for edge in range(nodes - 1):
        matrix[edge, edge] = 1.0
        matrix[edge + 1, edge] = -1.0
    return matrix


def benchmark_registry_v1() -> dict[str, PassiveNetworkBenchmark]:
    four_incidence = _path_incidence(4)
    four_mass = np.asarray([1.0, 1.2, 0.9, 1.1])
    four_coupling = np.asarray([8.0, 5.0, 7.0])
    four_damping = 0.18 * four_mass
    b2 = PassiveNetworkBenchmark(
        benchmark_id="swing4_b2_modal",
        mass=four_mass,
        incidence=four_incidence,
        coupling=four_coupling,
        node_damping=four_damping,
        edge_damping=np.zeros(3),
        modal_loadings=np.empty((3, 0)),
    )
    port = PassiveNetworkBenchmark(
        benchmark_id="swing4_reciprocal_port_r1",
        mass=four_mass,
        incidence=four_incidence,
        coupling=four_coupling,
        node_damping=four_damping,
        edge_damping=np.zeros(3),
        modal_loadings=np.asarray([[0.8], [-0.35], [0.55]]),
        modal_frequency_hz=1.35,
        modal_damping_ratio=0.045,
    )
    two_incidence = np.asarray([[1.0], [-1.0]])
    pendulum = PassiveNetworkBenchmark(
        benchmark_id="swing2_known_separatrix",
        mass=np.ones(2),
        incidence=two_incidence,
        coupling=np.asarray([1.0]),
        node_damping=np.asarray([0.08, 0.08]),
        edge_damping=np.zeros(1),
        modal_loadings=np.empty((1, 0)),
        stability_barrier=2.0,
    )
    return {item.benchmark_id: item for item in (b2, port, pendulum)}
