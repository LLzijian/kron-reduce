"""Causal ten-machine baselines used only by the frozen P2 comparison.

Every model consumes :class:`P2PredictionInput`, which has no future input
schedule or routing metadata.  Fitting uses the target paths plus explicit
source/fault-line strata and applies source -> line -> event -> time-row equal
weights.  This module does not reuse legacy fit methods that read observed
mechanical-power trajectories.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import lsq_linear

from kron_reduce.data.p2_training import (
    P2PredictionInput,
    P2TrainingSample,
    hierarchical_event_weights,
    hierarchical_row_weights,
)
from kron_reduce.evaluation.p2_metrics import P2FitScales
from kron_reduce.physics.coordinates import coi_basis, coi_project, coi_project_power

P2_BASELINE_NAMES = (
    "B2-10-ZOH",
    "EGDF-10-ZOH",
    "GMDPM-10-CAUSAL",
    "SSDM-10-ZOH",
    "LIN-ODE-10",
    "SINDYc-ODE-10",
    "NODE-MLP-10",
)

_RIDGES = (1e-8, 1e-6, 1e-4)
_MEMORY_TAUS = (0.05, 0.2, 1.0)
_PM_CANDIDATES = ((0.0, 1.0), (0.01, 0.2), (0.01, 1.0), (0.1, 1.0), (0.1, 5.0))
_SINDY_THRESHOLDS = (0.0, 1e-4, 1e-3)
_NODE_HIDDEN = (8, 16)
_TRAINING_ROW_LIMIT = 64


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _check_training_roles(
    fit_samples: Sequence[P2TrainingSample], selection_samples: Sequence[P2TrainingSample]
) -> None:
    if not fit_samples or not selection_samples:
        raise ValueError("P2 fit and selection populations must be nonempty")
    if {sample.role for sample in fit_samples} != {"fit"}:
        raise ValueError("P2 optimizer may fit only the registered fit role")
    if {sample.role for sample in selection_samples} != {"selection"}:
        raise ValueError("P2 hyperparameters may use only the selection role")
    combined = tuple(fit_samples) + tuple(selection_samples)
    if len({sample.event_id for sample in combined}) != len(combined):
        raise ValueError("P2 fit and selection IDs must be disjoint")
    reference = fit_samples[0].prediction_input
    for sample in combined:
        request = sample.prediction_input
        if not np.array_equal(request.mass, reference.mass):
            raise ValueError("P2 baselines require one registered mass prior")
        if not np.array_equal(request.incidence, reference.incidence):
            raise ValueError("P2 baselines require one registered graph prior")


def _row_indices(length: int, limit: int = _TRAINING_ROW_LIMIT) -> np.ndarray:
    if length < 3:
        raise ValueError("P2 derivative fitting requires at least three time points")
    count = min(limit, length - 2)
    return np.unique(np.linspace(1, length - 2, count).round().astype(int))


def _derivative_rows(
    samples: Sequence[P2TrainingSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    deltas: list[np.ndarray] = []
    omegas: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    equilibria: list[np.ndarray] = []
    counts: list[int] = []
    for sample in samples:
        request = sample.prediction_input
        indices = _row_indices(len(request.time))
        denominator = request.time[indices + 1] - request.time[indices - 1]
        acceleration = (
            sample.target_omega[indices + 1] - sample.target_omega[indices - 1]
        ) / denominator[:, None]
        deltas.append(sample.target_delta[indices])
        omegas.append(sample.target_omega[indices])
        accelerations.append(acceleration)
        inputs.append(
            np.broadcast_to(request.pm_clear - request.pm_equilibrium, (len(indices), 10)).copy()
        )
        equilibria.append(np.broadcast_to(request.equilibrium_delta, (len(indices), 10)).copy())
        counts.append(len(indices))
    return (
        np.concatenate(deltas),
        np.concatenate(omegas),
        np.concatenate(accelerations),
        np.concatenate(inputs),
        np.concatenate(equilibria),
        hierarchical_row_weights(samples, counts),
    )


def _weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if design.ndim != 2 or target.ndim != 2 or len(design) != len(target):
        raise ValueError("P2 weighted ridge arrays are not row aligned")
    if weights.shape != (len(design),) or np.any(weights <= 0.0):
        raise ValueError("P2 weighted ridge requires positive row weights")
    root = np.sqrt(weights)[:, None]
    x = root * design
    y = root * target
    gram = x.T @ x + float(ridge) * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def _weighted_nonnegative(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    root = np.repeat(np.sqrt(weights), target.shape[1])
    matrix = design.reshape(-1, design.shape[-1]) * root[:, None]
    vector = target.reshape(-1) * root
    if ridge > 0.0:
        matrix = np.vstack((matrix, math.sqrt(ridge) * np.eye(matrix.shape[1])))
        vector = np.concatenate((vector, np.zeros(matrix.shape[1])))
    result = lsq_linear(matrix, vector, bounds=(0.0, np.inf), method="trf")
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"P2 nonnegative regression failed: {result.message}")
    return result.x


def _selection_score(
    model: P2Baseline,
    samples: Sequence[P2TrainingSample],
    scales: P2FitScales,
) -> dict[str, Any]:
    event_weights = hierarchical_event_weights(samples)
    losses: list[float] = []
    weighted = 0.0
    failures: list[str] = []
    delta_scale = np.asarray(scales.delta)[None, :]
    omega_scale = np.asarray(scales.omega)[None, :]
    for sample in samples:
        try:
            delta, omega = model.predict(sample.prediction_input)
            if (
                delta.shape != sample.target_delta.shape
                or omega.shape != sample.target_omega.shape
                or not np.isfinite(delta).all()
                or not np.isfinite(omega).all()
            ):
                raise FloatingPointError("invalid P2 rollout")
            loss = float(
                0.5
                * (
                    np.mean(((delta - sample.target_delta) / delta_scale) ** 2)
                    + np.mean(((omega - sample.target_omega) / omega_scale) ** 2)
                )
            )
            losses.append(loss)
            weighted += event_weights[sample.event_id] * loss
        except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
            failures.append(sample.event_id)
    return {
        "failure_count": len(failures),
        "failed_event_ids": failures,
        "normalized_joint_mse": weighted if not failures else None,
        "successful_event_losses": losses,
    }


def _candidate_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    loss = row["score"]["normalized_joint_mse"]
    return (
        float(row["score"]["failure_count"]),
        float("inf") if loss is None else float(loss),
        int(row["parameter_count"]),
    )


class P2Baseline(ABC):
    method_name: ClassVar[str] = "ABSTRACT"
    max_internal_step: float = 0.01

    def __init__(self) -> None:
        self.fit_record: dict[str, Any] = {}

    @abstractmethod
    def _candidate_specs(self) -> tuple[dict[str, Any], ...]: ...

    @classmethod
    @abstractmethod
    def _fit_spec(
        cls, samples: Sequence[P2TrainingSample], spec: Mapping[str, Any], seed: int
    ) -> P2Baseline: ...

    @abstractmethod
    def initial_state(self, request: P2PredictionInput) -> np.ndarray: ...

    @abstractmethod
    def rhs(self, state: np.ndarray, request: P2PredictionInput) -> np.ndarray: ...

    @abstractmethod
    def parameter_count(self) -> int: ...

    @abstractmethod
    def _arrays(self) -> dict[str, np.ndarray]: ...

    @classmethod
    @abstractmethod
    def _from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> P2Baseline: ...

    def fit(
        self,
        fit_samples: Sequence[P2TrainingSample],
        selection_samples: Sequence[P2TrainingSample],
        scales: P2FitScales,
        *,
        seed: int = 324,
    ) -> dict[str, Any]:
        _check_training_roles(fit_samples, selection_samples)
        candidates = []
        for spec in self._candidate_specs():
            candidate = type(self)._fit_spec(fit_samples, spec, seed)
            score = _selection_score(candidate, selection_samples, scales)
            candidates.append(
                {
                    "spec": dict(spec),
                    "score": score,
                    "parameter_count": candidate.parameter_count(),
                }
            )
        selected = min(candidates, key=_candidate_key)
        final_samples = tuple(fit_samples) + tuple(selection_samples)
        final = type(self)._fit_spec(final_samples, selected["spec"], seed)
        self.__dict__.update(final.__dict__)
        self.fit_record = {
            "schema": "kron_reduce.p2_baseline_fit.v1",
            "method": self.method_name,
            "seed": int(seed),
            "candidate_table": candidates,
            "selected_spec": selected["spec"],
            "selection_priority": [
                "zero_rollout_failure",
                "normalized_joint_loss",
                "smaller_model",
            ],
            "fit_event_ids": sorted(sample.event_id for sample in fit_samples),
            "selection_event_ids": sorted(sample.event_id for sample in selection_samples),
            "final_refit_event_ids": sorted(sample.event_id for sample in final_samples),
            "hierarchical_weighting": ["source", "fault_line", "event", "time_row"],
            "observed_pm_reads": [],
            "future_pm_reads": [],
            "source_routing": False,
            "parameter_count": self.parameter_count(),
        }
        return self.fit_record

    def predict(self, request: P2PredictionInput) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(request, P2PredictionInput):
            raise TypeError("P2 prediction accepts only P2PredictionInput")
        state = np.asarray(self.initial_state(request), dtype=np.float64)
        if state.ndim != 1 or len(state) < 20 or not np.isfinite(state).all():
            raise ValueError("invalid P2 model initial state")
        delta_out = [coi_project(state[:10], request.mass)]
        omega_out = [coi_project(state[10:20], request.mass)]
        state = state.copy()
        state[:10], state[10:20] = delta_out[0], omega_out[0]
        for interval in np.diff(request.time):
            substeps = max(1, int(np.ceil(float(interval) / self.max_internal_step)))
            step = float(interval) / substeps
            for _ in range(substeps):
                k1 = self.rhs(state, request)
                k2 = self.rhs(state + 0.5 * step * k1, request)
                k3 = self.rhs(state + 0.5 * step * k2, request)
                k4 = self.rhs(state + step * k3, request)
                state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                state[:10] = coi_project(state[:10], request.mass)
                state[10:20] = coi_project(state[10:20], request.mass)
                if not np.isfinite(state).all() or float(np.max(np.abs(state))) > 1e12:
                    raise FloatingPointError("nonfinite or divergent P2 rollout")
            delta_out.append(state[:10].copy())
            omega_out.append(state[10:20].copy())
        return np.stack(delta_out), np.stack(omega_out)

    def access_audit(self) -> dict[str, Any]:
        return {
            "schema": "kron_reduce.p2_baseline_access.v1",
            "method": self.method_name,
            "prediction_input_fields": [
                "delta0",
                "omega0",
                "Pm_equilibrium",
                "Pm_clear",
                "equilibrium_delta",
                "mass",
                "incidence",
                "time",
            ],
            "fit_target_fields": ["delta", "omega"],
            "stratification_only_fields": ["source_dataset", "fault_line", "event_id"],
            "observed_pm_reads": [],
            "future_pm_reads": [],
            "forbidden_field_reads": [],
            "source_routing": False,
            "continuous_integrator": "shared_mass_weighted_coi_rk4",
            "maximum_internal_step_s": self.max_internal_step,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("P2 checkpoints must use .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "kron_reduce.p2_baseline_checkpoint.v1",
            "method": self.method_name,
            "fit_record": self.fit_record,
            "access_audit": self.access_audit(),
        }
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
            **self._arrays(),
        )

    @classmethod
    def load(cls, path: str | Path) -> P2Baseline:
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            if metadata.get("method") != cls.method_name:
                raise ValueError("P2 checkpoint method mismatch")
            model = cls._from_arrays(
                {name: np.asarray(payload[name]) for name in payload.files if name != "metadata"}
            )
        model.fit_record = dict(metadata.get("fit_record", {}))
        if metadata.get("access_audit") != model.access_audit():
            raise ValueError("P2 checkpoint access contract mismatch")
        return model


def _power_projector(mass: np.ndarray) -> np.ndarray:
    return np.eye(10) - np.outer(mass, np.ones(10)) / float(np.sum(mass))


def _fit_b2_terms(
    samples: Sequence[P2TrainingSample], ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    delta, omega, acceleration, power, equilibrium, weights = _derivative_rows(samples)
    request = samples[0].prediction_input
    mass, incidence = request.mass, request.incidence
    projector = _power_projector(mass)
    designs = []
    targets = []
    for d, w, a, u, eq in zip(delta, omega, acceleration, power, equilibrium, strict=True):
        sine = np.sin(incidence.T @ d) - np.sin(incidence.T @ eq)
        design = np.concatenate(
            (projector @ np.diag(w), projector @ (incidence * sine[None, :])), axis=1
        )
        designs.append(design)
        targets.append(coi_project_power(u, mass) - mass * a)
    theta = _weighted_nonnegative(np.stack(designs), np.stack(targets), weights, float(ridge))
    return theta[:10], theta[10:]


def _filter_path(time: np.ndarray, values: np.ndarray, tau: float) -> np.ndarray:
    output = np.zeros_like(values)
    for index, step in enumerate(np.diff(time)):
        decay = math.exp(-float(step) / tau)
        output[index + 1] = decay * output[index] + tau * (1.0 - decay) * 0.5 * (
            values[index] + values[index + 1]
        )
    return output


def _node_gate(delta: np.ndarray, equilibrium: np.ndarray, incidence: np.ndarray) -> np.ndarray:
    edge_stress = np.abs(np.sin(delta @ incidence) - np.sin(equilibrium @ incidence))
    degree = np.maximum(np.sum(np.abs(incidence), axis=1), 1.0)
    node_stress = edge_stress @ np.abs(incidence).T / degree[None, :]
    return 0.5 + 0.5 * np.tanh((node_stress - 0.02) / 0.02)


class P2PhysicalBaseline(P2Baseline):
    use_node_storage: ClassVar[bool] = False
    use_edge_memory: ClassVar[bool] = False
    use_pm_closure: ClassVar[bool] = False

    def __init__(
        self,
        damping: np.ndarray | None = None,
        coupling: np.ndarray | None = None,
        *,
        node_alpha: np.ndarray | None = None,
        node_tau: float = 0.2,
        edge_damping: np.ndarray | None = None,
        edge_memory: np.ndarray | None = None,
        edge_tau: float = 0.2,
        pm_gain: float = 0.0,
        pm_decay: float = 1.0,
    ) -> None:
        super().__init__()
        self.damping = None if damping is None else np.asarray(damping, dtype=np.float64)
        self.coupling = None if coupling is None else np.asarray(coupling, dtype=np.float64)
        self.node_alpha = (
            np.zeros(10) if node_alpha is None else np.asarray(node_alpha, dtype=np.float64)
        )
        self.node_tau = float(node_tau)
        edge_count = 45 if coupling is None else len(coupling)
        self.edge_damping = (
            np.zeros(edge_count)
            if edge_damping is None
            else np.asarray(edge_damping, dtype=np.float64)
        )
        self.edge_memory = (
            np.zeros(edge_count)
            if edge_memory is None
            else np.asarray(edge_memory, dtype=np.float64)
        )
        self.edge_tau = float(edge_tau)
        self.pm_gain = float(pm_gain)
        self.pm_decay = float(pm_decay)

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.damping is None or self.coupling is None:
            raise RuntimeError(f"{self.method_name} is not fitted")
        arrays = [
            self.damping,
            self.coupling,
            self.node_alpha,
            self.edge_damping,
            self.edge_memory,
        ]
        if not all(np.isfinite(value).all() and np.all(value >= 0.0) for value in arrays):
            raise ValueError("P2 physical coefficients must be finite and nonnegative")
        if min(self.node_tau, self.edge_tau, self.pm_decay) <= 0.0 or self.pm_gain < 0.0:
            raise ValueError("P2 physical time constants and gains are invalid")
        return self.damping, self.coupling

    @classmethod
    def _fit_modules(
        cls,
        samples: Sequence[P2TrainingSample],
        *,
        ridge: float,
        node_tau: float,
        edge_tau: float,
        pm_gain: float,
        pm_decay: float,
    ) -> P2PhysicalBaseline:
        damping, coupling = _fit_b2_terms(samples, ridge)
        if cls.use_node_storage:
            positive = damping[damping > 0.0]
            floor = max(1e-8, 0.1 * float(np.percentile(positive, 10))) if len(positive) else 1e-8
            damping = np.maximum(damping, floor)
        model = cls(
            damping,
            coupling,
            node_tau=node_tau,
            edge_tau=edge_tau,
            pm_gain=pm_gain,
            pm_decay=pm_decay,
        )
        if cls.use_node_storage or cls.use_edge_memory:
            cls._fit_residual_modules(model, samples, ridge)
        model._require_fitted()
        return model

    @staticmethod
    def _fit_residual_modules(
        model: P2PhysicalBaseline, samples: Sequence[P2TrainingSample], ridge: float
    ) -> None:
        delta, omega, acceleration, power, equilibrium, weights = _derivative_rows(samples)
        request = samples[0].prediction_input
        mass, incidence = request.mass, request.incidence
        projector = _power_projector(mass)
        teacher_node: dict[str, np.ndarray] = {}
        teacher_edge: dict[str, np.ndarray] = {}
        if model.use_node_storage:
            for sample in samples:
                gate = _node_gate(
                    sample.target_delta,
                    sample.prediction_input.equilibrium_delta,
                    incidence,
                )
                teacher_node[sample.event_id] = _filter_path(
                    sample.prediction_input.time,
                    gate * sample.target_omega,
                    model.node_tau,
                )
        if model.use_edge_memory:
            for sample in samples:
                teacher_edge[sample.event_id] = _filter_path(
                    sample.prediction_input.time,
                    sample.target_omega @ incidence,
                    model.edge_tau,
                )
        designs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        cursor = 0
        for sample in samples:
            indices = _row_indices(len(sample.prediction_input.time))
            for local, index in enumerate(indices):
                d, w, a, u, eq = (
                    delta[cursor + local],
                    omega[cursor + local],
                    acceleration[cursor + local],
                    power[cursor + local],
                    equilibrium[cursor + local],
                )
                sine = np.sin(incidence.T @ d) - np.sin(incidence.T @ eq)
                base_force = coi_project_power(
                    u - model.damping * w - incidence @ (model.coupling * sine), mass
                )
                columns: list[np.ndarray] = []
                if model.use_node_storage:
                    gate = _node_gate(d[None, :], eq, incidence)[0]
                    columns.append(
                        -projector @ np.diag(gate * teacher_node[sample.event_id][index])
                    )
                if model.use_edge_memory:
                    edge_speed = incidence.T @ w
                    columns.append(-projector @ (incidence * edge_speed[None, :]))
                    columns.append(
                        -projector @ (incidence * teacher_edge[sample.event_id][index][None, :])
                    )
                designs.append(np.concatenate(columns, axis=1))
                targets.append(mass * a - base_force)
            cursor += len(indices)
        coefficients = _weighted_nonnegative(np.stack(designs), np.stack(targets), weights, ridge)
        cursor = 0
        if model.use_node_storage:
            model.node_alpha = coefficients[:10]
            cursor = 10
        if model.use_edge_memory:
            edge_count = request.incidence.shape[1]
            model.edge_damping = coefficients[cursor : cursor + edge_count]
            model.edge_memory = coefficients[cursor + edge_count : cursor + 2 * edge_count]

    def initial_state(self, request: P2PredictionInput) -> np.ndarray:
        self._require_fitted()
        parts = [request.delta0, request.omega0]
        if self.use_node_storage:
            parts.append(np.zeros(10))
        if self.use_edge_memory:
            parts.append(np.zeros(request.incidence.shape[1]))
        if self.use_pm_closure:
            parts.append(np.zeros(9))
        return np.concatenate(parts)

    def rhs(self, state: np.ndarray, request: P2PredictionInput) -> np.ndarray:
        damping, coupling = self._require_fitted()
        delta, omega = state[:10], state[10:20]
        incidence, mass = request.incidence, request.mass
        sine = np.sin(incidence.T @ delta) - np.sin(incidence.T @ request.equilibrium_delta)
        force = (
            request.pm_clear
            - request.pm_equilibrium
            - damping * omega
            - incidence @ (coupling * sine)
        )
        derivatives: list[np.ndarray] = [omega]
        cursor = 20
        if self.use_node_storage:
            q = state[cursor : cursor + 10]
            cursor += 10
            gate = _node_gate(delta[None, :], request.equilibrium_delta, incidence)[0]
            force -= self.node_alpha * gate * q
            derivatives.append(gate * omega - q / self.node_tau)
        if self.use_edge_memory:
            edge_count = incidence.shape[1]
            z = state[cursor : cursor + edge_count]
            cursor += edge_count
            edge_speed = incidence.T @ omega
            force -= incidence @ (self.edge_damping * edge_speed + self.edge_memory * z)
            derivatives.append(edge_speed - z / self.edge_tau)
        if self.use_pm_closure:
            zpm = state[cursor : cursor + 9]
            basis = coi_basis(np.ones(10))
            port = math.sqrt(self.pm_gain) * basis
            force += port @ zpm
            derivatives.append(-self.pm_decay * zpm - port.T @ omega)
        force = coi_project_power(force, mass)
        return np.concatenate((derivatives[0], force / mass, *derivatives[1:]))

    def parameter_count(self) -> int:
        damping, coupling = self._require_fitted()
        count = len(damping) + len(coupling)
        if self.use_node_storage:
            count += len(self.node_alpha) + 1
        if self.use_edge_memory:
            count += len(self.edge_damping) + len(self.edge_memory) + 1
        if self.use_pm_closure:
            count += 2
        return count

    def _arrays(self) -> dict[str, np.ndarray]:
        damping, coupling = self._require_fitted()
        return {
            "damping": damping,
            "coupling": coupling,
            "node_alpha": self.node_alpha,
            "node_tau": np.asarray(self.node_tau),
            "edge_damping": self.edge_damping,
            "edge_memory": self.edge_memory,
            "edge_tau": np.asarray(self.edge_tau),
            "pm_gain": np.asarray(self.pm_gain),
            "pm_decay": np.asarray(self.pm_decay),
        }

    @classmethod
    def _from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> P2PhysicalBaseline:
        return cls(
            arrays["damping"],
            arrays["coupling"],
            node_alpha=arrays["node_alpha"],
            node_tau=float(arrays["node_tau"]),
            edge_damping=arrays["edge_damping"],
            edge_memory=arrays["edge_memory"],
            edge_tau=float(arrays["edge_tau"]),
            pm_gain=float(arrays["pm_gain"]),
            pm_decay=float(arrays["pm_decay"]),
        )

    def physical_audit(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "status": "PASS",
            "finite_nonnegative_damping": bool(
                np.isfinite(self.damping).all() and np.all(self.damping >= 0.0)
            ),
            "finite_nonnegative_symmetric_edge_coupling": bool(
                np.isfinite(self.coupling).all() and np.all(self.coupling >= 0.0)
            ),
            "stable_node_storage": bool(not self.use_node_storage or self.node_tau > 0.0),
            "stable_edge_memory": bool(not self.use_edge_memory or self.edge_tau > 0.0),
            "hurwitz_pm_closure": bool(not self.use_pm_closure or self.pm_decay > 0.0),
            "source_routing": False,
            "level_6_claimed": False,
        }


class B2Baseline(P2PhysicalBaseline):
    method_name = "B2-10-ZOH"

    def _candidate_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple({"ridge": ridge} for ridge in _RIDGES)

    @classmethod
    def _fit_spec(
        cls, samples: Sequence[P2TrainingSample], spec: Mapping[str, Any], seed: int
    ) -> B2Baseline:
        del seed
        return cls._fit_modules(
            samples,
            ridge=float(spec["ridge"]),
            node_tau=0.2,
            edge_tau=0.2,
            pm_gain=0.0,
            pm_decay=1.0,
        )


class EGDFBaseline(P2PhysicalBaseline):
    method_name = "EGDF-10-ZOH"
    use_node_storage = True

    def _candidate_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"ridge": ridge, "node_tau": tau} for ridge in (1e-8, 1e-6) for tau in _MEMORY_TAUS
        )

    @classmethod
    def _fit_spec(
        cls, samples: Sequence[P2TrainingSample], spec: Mapping[str, Any], seed: int
    ) -> EGDFBaseline:
        del seed
        return cls._fit_modules(
            samples,
            ridge=float(spec["ridge"]),
            node_tau=float(spec["node_tau"]),
            edge_tau=0.2,
            pm_gain=0.0,
            pm_decay=1.0,
        )


class SSDMBaseline(P2PhysicalBaseline):
    method_name = "SSDM-10-ZOH"
    use_edge_memory = True

    def _candidate_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"ridge": ridge, "edge_tau": tau} for ridge in (1e-8, 1e-6) for tau in _MEMORY_TAUS
        )

    @classmethod
    def _fit_spec(
        cls, samples: Sequence[P2TrainingSample], spec: Mapping[str, Any], seed: int
    ) -> SSDMBaseline:
        del seed
        return cls._fit_modules(
            samples,
            ridge=float(spec["ridge"]),
            node_tau=0.2,
            edge_tau=float(spec["edge_tau"]),
            pm_gain=0.0,
            pm_decay=1.0,
        )


class GMDPMBaseline(P2PhysicalBaseline):
    method_name = "GMDPM-10-CAUSAL"
    use_node_storage = True
    use_edge_memory = True
    use_pm_closure = True

    def _candidate_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "ridge": 1e-6,
                "node_tau": tau,
                "edge_tau": tau,
                "pm_gain": gain,
                "pm_decay": decay,
            }
            for tau, (gain, decay) in zip(
                _MEMORY_TAUS + _MEMORY_TAUS[:2], _PM_CANDIDATES, strict=True
            )
        )

    @classmethod
    def _fit_spec(
        cls, samples: Sequence[P2TrainingSample], spec: Mapping[str, Any], seed: int
    ) -> GMDPMBaseline:
        del seed
        return cls._fit_modules(
            samples,
            ridge=float(spec["ridge"]),
            node_tau=float(spec["node_tau"]),
            edge_tau=float(spec["edge_tau"]),
            pm_gain=float(spec["pm_gain"]),
            pm_decay=float(spec["pm_decay"]),
        )


class LinearOdeBaseline(P2Baseline):
    method_name = "LIN-ODE-10"

    def __init__(
        self,
        state_matrix: np.ndarray | None = None,
        input_matrix: np.ndarray | None = None,
        mass: np.ndarray | None = None,
        *,
        pre_stability_abscissa: float = math.nan,
        post_stability_abscissa: float = math.nan,
        stability_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_matrix = None if state_matrix is None else np.asarray(state_matrix, dtype=float)
        self.input_matrix = None if input_matrix is None else np.asarray(input_matrix, dtype=float)
        self.mass = None if mass is None else np.asarray(mass, dtype=float)
        self.pre_stability_abscissa = float(pre_stability_abscissa)
        self.post_stability_abscissa = float(post_stability_abscissa)
        self.stability_shift = float(stability_shift)

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.state_matrix is None or self.input_matrix is None or self.mass is None:
            raise RuntimeError("LIN-ODE-10 is not fitted")
        if self.state_matrix.shape != (18, 18) or self.input_matrix.shape != (18, 9):
            raise ValueError("LIN-ODE-10 parameter shape mismatch")
        return self.state_matrix, self.input_matrix, self.mass

    def _candidate_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"ridge": ridge, "stability_margin": margin}
            for ridge in _RIDGES
            for margin in (0.01, 0.1)
        )

    @classmethod
    def _fit_spec(
        cls, samples: Sequence[P2TrainingSample], spec: Mapping[str, Any], seed: int
    ) -> LinearOdeBaseline:
        del seed
        delta, omega, acceleration, power, equilibrium, weights = _derivative_rows(samples)
        mass = samples[0].prediction_input.mass
        basis = coi_basis(mass)
        x = np.concatenate(((delta - equilibrium) @ basis, omega @ basis), axis=1)
        u = (power / mass[None, :]) @ basis
        target = np.concatenate((omega @ basis, acceleration @ basis), axis=1)
        coefficients = _weighted_ridge(
            np.concatenate((x, u), axis=1), target, weights, float(spec["ridge"])
        )
        state = coefficients[:18].T
        inputs = coefficients[18:].T
        pre = float(np.max(np.real(np.linalg.eigvals(state))))
        shift = max(0.0, pre + float(spec["stability_margin"]))
        state = state - shift * np.eye(18)
        post = float(np.max(np.real(np.linalg.eigvals(state))))
        return cls(
            state,
            inputs,
            mass,
            pre_stability_abscissa=pre,
            post_stability_abscissa=post,
            stability_shift=shift,
        )

    def initial_state(self, request: P2PredictionInput) -> np.ndarray:
        self._require_fitted()
        return np.concatenate((request.delta0, request.omega0))

    def rhs(self, state: np.ndarray, request: P2PredictionInput) -> np.ndarray:
        matrix, inputs, mass = self._require_fitted()
        if not np.array_equal(mass, request.mass):
            raise ValueError("LIN-ODE-10 mass prior changed")
        basis = coi_basis(mass)
        x = np.concatenate(
            (basis.T @ (state[:10] - request.equilibrium_delta), basis.T @ state[10:20])
        )
        u = basis.T @ ((request.pm_clear - request.pm_equilibrium) / mass)
        derivative = matrix @ x + inputs @ u
        return np.concatenate((basis @ derivative[:9], basis @ derivative[9:]))

    def parameter_count(self) -> int:
        matrix, inputs, _ = self._require_fitted()
        return int(matrix.size + inputs.size)

    def _arrays(self) -> dict[str, np.ndarray]:
        matrix, inputs, mass = self._require_fitted()
        return {
            "state_matrix": matrix,
            "input_matrix": inputs,
            "mass": mass,
            "pre_stability_abscissa": np.asarray(self.pre_stability_abscissa),
            "post_stability_abscissa": np.asarray(self.post_stability_abscissa),
            "stability_shift": np.asarray(self.stability_shift),
        }

    @classmethod
    def _from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> LinearOdeBaseline:
        return cls(
            arrays["state_matrix"],
            arrays["input_matrix"],
            arrays["mass"],
            pre_stability_abscissa=float(arrays["pre_stability_abscissa"]),
            post_stability_abscissa=float(arrays["post_stability_abscissa"]),
            stability_shift=float(arrays["stability_shift"]),
        )

    def spectral_audit(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "status": "PASS" if self.post_stability_abscissa < 0.0 else "FAIL",
            "pre_stability_abscissa": self.pre_stability_abscissa,
            "post_stability_abscissa": self.post_stability_abscissa,
            "stability_shift": self.stability_shift,
        }


def _sindy_library(delta_q: np.ndarray, omega_q: np.ndarray, input_q: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            delta_q,
            omega_q,
            input_q,
            np.sin(delta_q),
            delta_q * np.abs(delta_q),
            omega_q * np.abs(omega_q),
        ),
        axis=-1,
    )


class SindycOdeBaseline(P2Baseline):
    method_name = "SINDYc-ODE-10"

    def __init__(
        self,
        coefficients: np.ndarray | None = None,
        mass: np.ndarray | None = None,
        *,
        threshold: float = 0.0,
        ridge: float = 0.0,
    ) -> None:
        super().__init__()
        self.coefficients = None if coefficients is None else np.asarray(coefficients, dtype=float)
        self.mass = None if mass is None else np.asarray(mass, dtype=float)
        self.threshold = float(threshold)
        self.ridge = float(ridge)

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.coefficients is None or self.mass is None:
            raise RuntimeError("SINDYc-ODE-10 is not fitted")
        if self.coefficients.shape != (54, 9):
            raise ValueError("SINDYc-ODE-10 coefficient shape mismatch")
        return self.coefficients, self.mass

    def _candidate_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"ridge": ridge, "threshold": threshold}
            for ridge in (1e-8, 1e-6)
            for threshold in _SINDY_THRESHOLDS
        )

    @classmethod
    def _fit_spec(
        cls, samples: Sequence[P2TrainingSample], spec: Mapping[str, Any], seed: int
    ) -> SindycOdeBaseline:
        del seed
        delta, omega, acceleration, power, equilibrium, weights = _derivative_rows(samples)
        mass = samples[0].prediction_input.mass
        basis = coi_basis(mass)
        library = _sindy_library(
            (delta - equilibrium) @ basis,
            omega @ basis,
            (power / mass[None, :]) @ basis,
        )
        target = acceleration @ basis
        coefficients = _weighted_ridge(library, target, weights, float(spec["ridge"]))
        row_norm = np.linalg.norm(coefficients, axis=1)
        coefficients[row_norm < float(spec["threshold"])] = 0.0
        return cls(
            coefficients,
            mass,
            threshold=float(spec["threshold"]),
            ridge=float(spec["ridge"]),
        )

    def initial_state(self, request: P2PredictionInput) -> np.ndarray:
        self._require_fitted()
        return np.concatenate((request.delta0, request.omega0))

    def rhs(self, state: np.ndarray, request: P2PredictionInput) -> np.ndarray:
        coefficients, mass = self._require_fitted()
        if not np.array_equal(mass, request.mass):
            raise ValueError("SINDYc-ODE-10 mass prior changed")
        basis = coi_basis(mass)
        acceleration_q = (
            _sindy_library(
                basis.T @ (state[:10] - request.equilibrium_delta),
                basis.T @ state[10:20],
                basis.T @ ((request.pm_clear - request.pm_equilibrium) / mass),
            )
            @ coefficients
        )
        return np.concatenate((state[10:20], basis @ acceleration_q))

    def parameter_count(self) -> int:
        coefficients, _ = self._require_fitted()
        return int(np.count_nonzero(coefficients))

    def _arrays(self) -> dict[str, np.ndarray]:
        coefficients, mass = self._require_fitted()
        return {
            "coefficients": coefficients,
            "mass": mass,
            "threshold": np.asarray(self.threshold),
            "ridge": np.asarray(self.ridge),
        }

    @classmethod
    def _from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> SindycOdeBaseline:
        return cls(
            arrays["coefficients"],
            arrays["mass"],
            threshold=float(arrays["threshold"]),
            ridge=float(arrays["ridge"]),
        )

    def library_audit(self) -> dict[str, Any]:
        coefficients, _ = self._require_fitted()
        return {
            "status": "PASS",
            "library": [
                "delta_q",
                "omega_q",
                "Pm_clear_minus_equilibrium_q",
                "sin_delta_q",
                "delta_q_abs_delta_q",
                "omega_q_abs_omega_q",
            ],
            "group_sparsity_unit": "library_row_across_all_outputs",
            "active_rows": int(np.sum(np.linalg.norm(coefficients, axis=1) > 0.0)),
            "threshold": self.threshold,
            "ridge": self.ridge,
        }


class NodeMlpBaseline(P2Baseline):
    method_name = "NODE-MLP-10"

    def __init__(
        self,
        weight1: np.ndarray | None = None,
        bias1: np.ndarray | None = None,
        weight2: np.ndarray | None = None,
        bias2: np.ndarray | None = None,
        input_scale: np.ndarray | None = None,
        output_scale: np.ndarray | None = None,
        mass: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.weight1 = None if weight1 is None else np.asarray(weight1, dtype=float)
        self.bias1 = None if bias1 is None else np.asarray(bias1, dtype=float)
        self.weight2 = None if weight2 is None else np.asarray(weight2, dtype=float)
        self.bias2 = None if bias2 is None else np.asarray(bias2, dtype=float)
        self.input_scale = None if input_scale is None else np.asarray(input_scale, dtype=float)
        self.output_scale = None if output_scale is None else np.asarray(output_scale, dtype=float)
        self.mass = None if mass is None else np.asarray(mass, dtype=float)

    def _require_fitted(self) -> tuple[np.ndarray, ...]:
        values = (
            self.weight1,
            self.bias1,
            self.weight2,
            self.bias2,
            self.input_scale,
            self.output_scale,
            self.mass,
        )
        if any(value is None for value in values):
            raise RuntimeError("NODE-MLP-10 is not fitted")
        concrete = tuple(np.asarray(value) for value in values)
        w1, b1, w2, b2, input_scale, output_scale, mass = concrete
        if (
            w1.ndim != 2
            or w1.shape[1] != 27
            or b1.shape != (w1.shape[0],)
            or w2.shape != (9, w1.shape[0])
            or b2.shape != (9,)
            or input_scale.shape != (27,)
            or output_scale.shape != (9,)
            or mass.shape != (10,)
        ):
            raise ValueError("NODE-MLP-10 parameter shape mismatch")
        return concrete

    def _candidate_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"hidden": hidden, "epochs": 80, "learning_rate": 0.01} for hidden in _NODE_HIDDEN
        )

    @classmethod
    def _fit_spec(
        cls, samples: Sequence[P2TrainingSample], spec: Mapping[str, Any], seed: int
    ) -> NodeMlpBaseline:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        delta, omega, acceleration, power, equilibrium, weights = _derivative_rows(samples)
        mass = samples[0].prediction_input.mass
        basis = coi_basis(mass)
        x = np.concatenate(
            (
                (delta - equilibrium) @ basis,
                omega @ basis,
                (power / mass[None, :]) @ basis,
            ),
            axis=1,
        )
        y = acceleration @ basis
        input_scale = np.maximum(np.sqrt(np.sum(weights[:, None] * x**2, axis=0)), 1e-8)
        output_scale = np.maximum(np.sqrt(np.sum(weights[:, None] * y**2, axis=0)), 1e-8)
        xt = torch.as_tensor(x / input_scale, dtype=torch.float64)
        yt = torch.as_tensor(y / output_scale, dtype=torch.float64)
        wt = torch.as_tensor(weights, dtype=torch.float64)
        hidden = int(spec["hidden"])
        layer1 = torch.nn.Linear(27, hidden, dtype=torch.float64)
        layer2 = torch.nn.Linear(hidden, 9, dtype=torch.float64)
        optimizer = torch.optim.Adam(
            tuple(layer1.parameters()) + tuple(layer2.parameters()),
            lr=float(spec["learning_rate"]),
        )
        zero = torch.zeros((1, 27), dtype=torch.float64)
        for _ in range(int(spec["epochs"])):
            optimizer.zero_grad()
            prediction = layer2(torch.tanh(layer1(xt))) - layer2(torch.tanh(layer1(zero)))
            loss = torch.sum(wt[:, None] * (prediction - yt) ** 2)
            loss.backward()
            optimizer.step()
        return cls(
            layer1.weight.detach().numpy().copy(),
            layer1.bias.detach().numpy().copy(),
            layer2.weight.detach().numpy().copy(),
            layer2.bias.detach().numpy().copy(),
            input_scale,
            output_scale,
            mass,
        )

    def initial_state(self, request: P2PredictionInput) -> np.ndarray:
        self._require_fitted()
        return np.concatenate((request.delta0, request.omega0))

    def rhs(self, state: np.ndarray, request: P2PredictionInput) -> np.ndarray:
        w1, b1, w2, b2, input_scale, output_scale, mass = self._require_fitted()
        if not np.array_equal(mass, request.mass):
            raise ValueError("NODE-MLP-10 mass prior changed")
        basis = coi_basis(mass)
        inputs = (
            np.concatenate(
                (
                    basis.T @ (state[:10] - request.equilibrium_delta),
                    basis.T @ state[10:20],
                    basis.T @ ((request.pm_clear - request.pm_equilibrium) / mass),
                )
            )
            / input_scale
        )
        hidden = np.tanh(w1 @ inputs + b1)
        zero_hidden = np.tanh(b1)
        acceleration_q = (w2 @ hidden + b2 - (w2 @ zero_hidden + b2)) * output_scale
        return np.concatenate((state[10:20], basis @ acceleration_q))

    def parameter_count(self) -> int:
        w1, b1, w2, b2, *_ = self._require_fitted()
        return int(w1.size + b1.size + w2.size + b2.size)

    def _arrays(self) -> dict[str, np.ndarray]:
        w1, b1, w2, b2, input_scale, output_scale, mass = self._require_fitted()
        return {
            "weight1": w1,
            "bias1": b1,
            "weight2": w2,
            "bias2": b2,
            "input_scale": input_scale,
            "output_scale": output_scale,
            "mass": mass,
        }

    @classmethod
    def _from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> NodeMlpBaseline:
        return cls(
            arrays["weight1"],
            arrays["bias1"],
            arrays["weight2"],
            arrays["bias2"],
            arrays["input_scale"],
            arrays["output_scale"],
            arrays["mass"],
        )

    def deterministic_audit(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "status": "PASS",
            "deterministic_algorithms": True,
            "continuous_rhs": True,
            "shared_rk4": True,
            "hidden_width": int(self.weight1.shape[0]),
        }


BASELINE_CLASSES: dict[str, type[P2Baseline]] = {
    cls.method_name: cls
    for cls in (
        B2Baseline,
        EGDFBaseline,
        GMDPMBaseline,
        SSDMBaseline,
        LinearOdeBaseline,
        SindycOdeBaseline,
        NodeMlpBaseline,
    )
}


def new_p2_baseline(method: str) -> P2Baseline:
    try:
        return BASELINE_CLASSES[method]()
    except KeyError as exc:
        raise ValueError(f"unknown P2 baseline: {method}") from exc


def load_p2_baseline(method: str, path: str | Path) -> P2Baseline:
    try:
        return BASELINE_CLASSES[method].load(path)
    except KeyError as exc:
        raise ValueError(f"unknown P2 baseline: {method}") from exc
