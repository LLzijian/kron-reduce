from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .m0 import M0Model, _pm_equilibrium, _trapz


class TransientEmfM0Model(DynamicsModel):
    """Graph-consistent reduced third-order M0 with nodal transient EMF.

    The normalized hidden state ``q`` represents a clustered transient EMF::

        delta_dot = omega
        m omega_dot = Delta Pm - D omega
                      - B [K (q_tail q_head sin(eta) - sin(eta_eq))]
        tau q_dot = 1 - q - alpha/k_ref (i_d(q, eta) - i_d(1, eta_eq))

    ``i_d`` is assembled from the same incidence graph and coupling weights as
    active power.  The clearing value of ``q`` is the causal quasi-steady
    solution of the last equation using only ``delta0``.  At ``alpha=0`` this
    model is exactly the fitted second-order M0, which is always included as a
    validation candidate.

    The model is a reduced normalized flux-decay closure.  It does not claim
    that the canonical data expose physical generator terminal voltages or
    machine-level transient EMFs.
    """

    method_name = "M0-3E"
    DEFAULT_ALPHA_CANDIDATES = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    DEFAULT_TAU_CANDIDATES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

    def __init__(
        self,
        damping: np.ndarray | None = None,
        coupling: np.ndarray | None = None,
        alpha: float | None = None,
        tau: float | None = None,
        activation_threshold: float = 0.0,
        *,
        anchor: M0Model | None = None,
        alpha_candidates: tuple[float, ...] | None = None,
        tau_candidates: tuple[float, ...] | None = None,
        activation_threshold_candidates: tuple[float, ...] | None = None,
        activation_width: float = 0.05,
        activation_sparsity_tolerance: float | None = None,
        max_source_loss_degradation: float | None = None,
        max_internal_step: float = 0.01,
        q_floor: float = 0.05,
        q_ceiling: float = 3.0,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if anchor is not None and (damping is not None or coupling is not None):
            raise ValueError("provide either an M0 anchor or damping/coupling arrays")
        if anchor is None and damping is not None and coupling is not None:
            anchor = M0Model(damping, coupling, max_internal_step=max_internal_step)
        self.anchor = anchor
        self.alpha = None if alpha is None else float(alpha)
        self.tau = None if tau is None else float(tau)
        self.activation_threshold = float(activation_threshold)
        self.alpha_candidates = tuple(alpha_candidates or self.DEFAULT_ALPHA_CANDIDATES)
        self.tau_candidates = tuple(tau_candidates or self.DEFAULT_TAU_CANDIDATES)
        self.activation_threshold_candidates = tuple(
            activation_threshold_candidates or (self.activation_threshold,)
        )
        self.activation_width = float(activation_width)
        self.activation_sparsity_tolerance = (
            None if activation_sparsity_tolerance is None
            else float(activation_sparsity_tolerance)
        )
        self.max_source_loss_degradation = (
            None if max_source_loss_degradation is None
            else float(max_source_loss_degradation)
        )
        if any(not np.isfinite(value) or value < 0 for value in self.alpha_candidates):
            raise ValueError("transient-EMF alpha candidates must be finite and nonnegative")
        if 0.0 not in self.alpha_candidates:
            raise ValueError("alpha candidates must include zero as the exact M0 fallback")
        if any(not np.isfinite(value) or value <= 0 for value in self.tau_candidates):
            raise ValueError("transient-EMF tau candidates must be finite and positive")
        if any(
            not np.isfinite(value) or value < 0 or value >= 2
            for value in self.activation_threshold_candidates
        ):
            raise ValueError("activation thresholds must be finite and in [0, 2)")
        if not np.isfinite(self.activation_width) or self.activation_width <= 0:
            raise ValueError("activation width must be finite and positive")
        if (
            self.activation_sparsity_tolerance is not None
            and (
                not np.isfinite(self.activation_sparsity_tolerance)
                or self.activation_sparsity_tolerance < 0
            )
        ):
            raise ValueError("activation sparsity tolerance must be nonnegative")
        if (
            self.max_source_loss_degradation is not None
            and (
                not np.isfinite(self.max_source_loss_degradation)
                or self.max_source_loss_degradation < 0
            )
        ):
            raise ValueError("maximum source-loss degradation must be nonnegative")
        if not 0 < q_floor < 1 < q_ceiling:
            raise ValueError("q bounds must satisfy 0 < q_floor < 1 < q_ceiling")
        self.max_internal_step = float(max_internal_step)
        self.q_floor = float(q_floor)
        self.q_ceiling = float(q_ceiling)
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
        self.validation_scales_by_source: dict[str, dict[str, float]] = {}
        self.diagnostics = diagnostics or {}

    @property
    def damping(self) -> np.ndarray | None:
        return None if self.anchor is None else self.anchor.damping

    @property
    def coupling(self) -> np.ndarray | None:
        return None if self.anchor is None else self.anchor.coupling

    @staticmethod
    def _source(sample: dict[str, Any]) -> str:
        return str(
            sample.get("training_context", {})
            .get("metadata", {})
            .get("source_dataset", "all")
        )

    @classmethod
    def _source_groups(cls, samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            groups.setdefault(cls._source(sample), []).append(sample)
        return groups

    @staticmethod
    def _edge_endpoints(incidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        incidence = np.asarray(incidence, dtype=np.float64)
        tail = np.argmax(incidence, axis=0)
        head = np.argmin(incidence, axis=0)
        if not np.allclose(incidence[tail, np.arange(incidence.shape[1])], 1.0):
            raise ValueError("each incidence column must contain one +1 tail")
        if not np.allclose(incidence[head, np.arange(incidence.shape[1])], -1.0):
            raise ValueError("each incidence column must contain one -1 head")
        return tail, head

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        if (
            self.anchor is None
            or self.anchor.damping is None
            or self.anchor.coupling is None
            or self.alpha is None
            or self.tau is None
        ):
            raise RuntimeError("M0-3E is not fitted")
        return self.anchor.damping, self.anchor.coupling, self.alpha, self.tau

    def _graph_topology(self, static: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
        _, coupling, _, _ = self._require_fitted()
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        tail, head = self._edge_endpoints(incidence)
        weighted_degree = np.abs(incidence) @ coupling
        positive_degree = weighted_degree[weighted_degree > np.finfo(float).eps]
        k_ref = float(np.median(positive_degree)) if positive_degree.size else 1.0
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
        eta_eq = incidence.T @ equilibrium
        equilibrium_current = weighted_degree.copy()
        np.add.at(equilibrium_current, tail, -coupling*np.cos(eta_eq))
        np.add.at(equilibrium_current, head, -coupling*np.cos(eta_eq))
        return {
            "incidence": incidence, "tail": tail, "head": head,
            "weighted_degree": weighted_degree, "k_ref": k_ref,
            "equilibrium": equilibrium, "eta_eq": eta_eq,
            "equilibrium_current": equilibrium_current,
        }

    def _graph_current(
        self,
        q: np.ndarray,
        delta: np.ndarray,
        static: dict[str, np.ndarray],
        topology: dict[str, np.ndarray | float] | None = None,
    ) -> tuple[np.ndarray, float]:
        _, coupling, _, _ = self._require_fitted()
        topology = self._graph_topology(static) if topology is None else topology
        incidence = np.asarray(topology["incidence"])
        tail = np.asarray(topology["tail"], dtype=int)
        head = np.asarray(topology["head"], dtype=int)
        eta = incidence.T @ np.asarray(delta, dtype=np.float64).reshape(-1)
        cos_eta = np.cos(eta)
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        weighted_degree = np.asarray(topology["weighted_degree"])
        neighbor = np.zeros_like(q)
        np.add.at(neighbor, tail, coupling * cos_eta * q[head])
        np.add.at(neighbor, head, coupling * cos_eta * q[tail])
        return weighted_degree * q - neighbor, float(topology["k_ref"])

    def _large_signal_activation(
        self, delta: np.ndarray, topology: dict[str, np.ndarray | float]
    ) -> float:
        if self.activation_threshold <= 0:
            return 1.0
        incidence = np.asarray(topology["incidence"])
        eta_eq = np.asarray(topology["eta_eq"])
        stress = float(np.max(np.abs(np.sin(incidence.T @ delta)-np.sin(eta_eq))))
        coordinate = np.clip(
            (stress-self.activation_threshold)/self.activation_width, 0.0, 1.0
        )
        return float(coordinate**2*(3.0-2.0*coordinate))

    def initial_emf(
        self, delta: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Causal quasi-steady clearing EMF inferred from clearing angles."""
        _, coupling, alpha, _ = self._require_fitted()
        n_nodes = np.asarray(static["mass"]).size
        if alpha == 0.0:
            return np.ones(n_nodes, dtype=np.float64)
        topology = self._graph_topology(static)
        incidence = np.asarray(topology["incidence"])
        tail = np.asarray(topology["tail"], dtype=int)
        head = np.asarray(topology["head"], dtype=int)
        weighted_degree = np.asarray(topology["weighted_degree"])

        def current_matrix(angle: np.ndarray) -> np.ndarray:
            eta = incidence.T @ np.asarray(angle, dtype=np.float64).reshape(-1)
            matrix = np.diag(weighted_degree)
            edge_weight = coupling * np.cos(eta)
            matrix[tail, head] -= edge_weight
            matrix[head, tail] -= edge_weight
            return matrix

        equilibrium = np.asarray(topology["equilibrium"])
        clearing_matrix = current_matrix(delta)
        equilibrium_current = current_matrix(equilibrium) @ np.ones(n_nodes)
        effective_alpha = alpha*self._large_signal_activation(
            np.asarray(delta, dtype=np.float64).reshape(-1), topology
        )
        scale = effective_alpha / float(topology["k_ref"])
        q0 = np.linalg.solve(
            np.eye(n_nodes) + scale * clearing_matrix,
            np.ones(n_nodes) + scale * equilibrium_current,
        )
        if (
            not np.isfinite(q0).all()
            or np.min(q0) <= self.q_floor
            or np.max(q0) >= self.q_ceiling
        ):
            raise FloatingPointError("causal clearing EMF is outside the audited domain")
        return q0

    def vector_field_augmented(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        q: np.ndarray,
        pm: np.ndarray,
        static: dict[str, np.ndarray],
        topology: dict[str, np.ndarray | float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        damping, coupling, alpha, tau = self._require_fitted()
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        topology = self._graph_topology(static) if topology is None else topology
        incidence = np.asarray(topology["incidence"])
        equilibrium = np.asarray(topology["equilibrium"])
        tail = np.asarray(topology["tail"], dtype=int)
        head = np.asarray(topology["head"], dtype=int)
        eta = incidence.T @ delta
        eta_eq = incidence.T @ equilibrium
        flow = coupling * (q[tail] * q[head] * np.sin(eta) - np.sin(eta_eq))
        delta_pm = np.asarray(pm, dtype=np.float64).reshape(-1) - _pm_equilibrium(static)
        force = coi_project_power(delta_pm - damping * omega - incidence @ flow, mass)
        current, k_ref = self._graph_current(q, delta, static, topology)
        equilibrium_current = np.asarray(topology["equilibrium_current"])
        effective_alpha = alpha*self._large_signal_activation(delta, topology)
        q_dot = (
            1.0 - q - (effective_alpha/k_ref)*(current-equilibrium_current)
        )/tau
        return omega, force / mass, q_dot

    @staticmethod
    def _interpolation_mode(pm: np.ndarray, requested: str) -> str:
        if requested not in {"auto", "zoh", "linear"}:
            raise ValueError("Pm interpolation must be auto, zoh, or linear")
        if requested != "auto":
            return requested
        return "zoh" if np.allclose(pm, pm[[0]], rtol=0.0, atol=1e-14) else "linear"

    def rollout_augmented(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
        *,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_fitted()
        t = np.asarray(time_grid, dtype=np.float64)
        pm = np.asarray(exogenous["Pm"], dtype=np.float64)
        if len(t) != len(pm) or len(t) < 1:
            raise ValueError("time and Pm schedules must have the same positive length")
        step = self.max_internal_step if max_step is None else float(max_step)
        if step <= 0:
            raise ValueError("max_step must be positive")
        mode = self._interpolation_mode(pm, exogenous.get("interpolation", "auto"))
        mass = np.asarray(initial_state["mass"], dtype=np.float64).reshape(-1)
        delta = coi_project(np.asarray(initial_state["delta0"]).reshape(-1), mass)
        omega = coi_project(np.asarray(initial_state["omega0"]).reshape(-1), mass)
        q = self.initial_emf(delta, initial_state)
        topology = self._graph_topology(initial_state)
        output_delta, output_omega, output_q = [delta.copy()], [omega.copy()], [q.copy()]

        def input_at(index: int, fraction: float) -> np.ndarray:
            if mode == "zoh":
                return pm[index]
            return pm[index] + fraction * (pm[index + 1] - pm[index])

        for index in range(len(t) - 1):
            interval = float(t[index + 1] - t[index])
            if interval <= 0:
                raise ValueError("rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval / step)))
            h = interval / substeps
            for substep in range(substeps):
                f0 = substep / substeps
                fh = (substep + 0.5) / substeps
                f1 = (substep + 1.0) / substeps
                k1 = self.vector_field_augmented(delta, omega, q, input_at(index, f0), initial_state, topology)
                k2 = self.vector_field_augmented(
                    delta + 0.5*h*k1[0], omega + 0.5*h*k1[1], q + 0.5*h*k1[2],
                    input_at(index, fh), initial_state, topology,
                )
                k3 = self.vector_field_augmented(
                    delta + 0.5*h*k2[0], omega + 0.5*h*k2[1], q + 0.5*h*k2[2],
                    input_at(index, fh), initial_state, topology,
                )
                k4 = self.vector_field_augmented(
                    delta + h*k3[0], omega + h*k3[1], q + h*k3[2],
                    input_at(index, f1), initial_state, topology,
                )
                delta = coi_project(delta + h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0, mass)
                omega = coi_project(omega + h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0, mass)
                q = q + h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0
                if (
                    not np.isfinite(delta).all()
                    or not np.isfinite(omega).all()
                    or not np.isfinite(q).all()
                    or np.min(q) <= self.q_floor
                    or np.max(q) >= self.q_ceiling
                ):
                    raise FloatingPointError(f"M0-3E left audited state domain at step {index + 1}")
            output_delta.append(delta.copy())
            output_omega.append(omega.copy())
            output_q.append(q.copy())
        return np.stack(output_delta), np.stack(output_omega), np.stack(output_q)

    def rollout(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        delta, omega, _ = self.rollout_augmented(initial_state, time_grid, exogenous)
        return delta, omega

    def teacher_forced_emf(
        self,
        delta_path: np.ndarray,
        time_grid: np.ndarray,
        static: dict[str, np.ndarray],
        *,
        max_step: float | None = None,
    ) -> np.ndarray:
        """Integrate hidden EMF along an observed angle path for weak fitting.

        Only training/validation angle samples may be supplied here.  The
        method is never used at rollout time and does not read voltage,
        electrical power, labels, or a future-derived initial state.
        """
        self._require_fitted()
        delta_path = np.asarray(delta_path, dtype=np.float64)
        t = np.asarray(time_grid, dtype=np.float64)
        if delta_path.ndim != 2 or len(delta_path) != len(t) or len(t) < 1:
            raise ValueError("teacher-forced delta and time must have equal positive length")
        step = self.max_internal_step if max_step is None else float(max_step)
        if step <= 0:
            raise ValueError("max_step must be positive")
        topology = self._graph_topology(static)
        q = self.initial_emf(delta_path[0], static)
        output = [q.copy()]

        def derivative(q_value: np.ndarray, delta_value: np.ndarray) -> np.ndarray:
            _, _, alpha, tau = self._require_fitted()
            current, k_ref = self._graph_current(q_value, delta_value, static, topology)
            equilibrium_current = np.asarray(topology["equilibrium_current"])
            effective_alpha = alpha*self._large_signal_activation(delta_value, topology)
            return (
                1.0-q_value-(effective_alpha/k_ref)*(current-equilibrium_current)
            )/tau

        for index in range(len(t)-1):
            interval = float(t[index+1]-t[index])
            if interval <= 0:
                raise ValueError("teacher-forced time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval/step)))
            h = interval/substeps
            for substep in range(substeps):
                f0 = substep/substeps
                fh = (substep+0.5)/substeps
                f1 = (substep+1.0)/substeps

                def angle(fraction: float) -> np.ndarray:
                    return delta_path[index] + fraction*(
                        delta_path[index+1]-delta_path[index]
                    )

                k1 = derivative(q, angle(f0))
                k2 = derivative(q+0.5*h*k1, angle(fh))
                k3 = derivative(q+0.5*h*k2, angle(fh))
                k4 = derivative(q+h*k3, angle(f1))
                q = q+h*(k1+2*k2+2*k3+k4)/6.0
                if (
                    not np.isfinite(q).all()
                    or np.min(q) <= self.q_floor
                    or np.max(q) >= self.q_ceiling
                ):
                    raise FloatingPointError("teacher-forced EMF left audited domain")
            output.append(q.copy())
        return np.stack(output)

    def _validation_summary(
        self, samples: list[dict[str, Any]], context: RunContext
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        failures = 0
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            static, time_grid, exogenous = sample_rollout_inputs(sample)
            try:
                delta_p, omega_p, q_p = self.rollout_augmented(
                    static,
                    time_grid[:steps],
                    {"Pm": exogenous["Pm"][:steps], "interpolation": exogenous["interpolation"]},
                )
            except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
                failures += 1
                continue
            delta_t = np.asarray(target["delta"][:steps], dtype=np.float64)
            omega_t = np.asarray(target["omega"][:steps], dtype=np.float64)
            incidence = np.asarray(static["incidence"], dtype=np.float64)
            source = self._source(sample)
            scales = self.validation_scales_by_source.get(source, self.validation_scales)
            edge_error = (delta_p - delta_t) @ incidence
            omega_error = omega_p - omega_t
            mass = np.asarray(static["mass"], dtype=np.float64)
            rows.append({
                "source": source,
                "standardized_loss": float(
                    np.mean((edge_error / scales["edge_angle"])**2)
                    + np.mean((omega_error / scales["omega"])**2)
                ),
                "raw_rmse": float(np.sqrt(np.mean(np.concatenate((
                    (delta_p-delta_t).ravel(), omega_error.ravel(),
                ))**2))),
                "omega_rmse": float(np.sqrt(np.mean(omega_error**2))),
                "edge_rmse": float(np.sqrt(np.mean(edge_error**2))),
                "coi_drift": float(max(
                    np.max(np.abs(delta_p @ mass)), np.max(np.abs(omega_p @ mass)),
                )),
                "q_min": float(np.min(q_p)),
                "q_max": float(np.max(q_p)),
            })
        population = max(min(len(samples), context.max_cases), 1)
        if failures or not rows:
            return {
                "standardized_loss": float("inf"), "raw_mean_rmse": float("inf"),
                "raw_median_rmse": float("inf"), "omega_mean_rmse": float("inf"),
                "edge_angle_mean_rmse": float("inf"), "maximum_coi_drift": float("inf"),
                "minimum_q": float("nan"), "maximum_q": float("nan"),
                "failure_rate": failures/population, "source_standardized_loss": {},
            }
        sources = sorted({row["source"] for row in rows})
        source_losses = {
            source: float(np.mean([
                row["standardized_loss"] for row in rows if row["source"] == source
            ]))
            for source in sources
        }
        return {
            "standardized_loss": float(np.mean(list(source_losses.values()))),
            "raw_mean_rmse": float(np.mean([row["raw_rmse"] for row in rows])),
            "raw_median_rmse": float(np.median([row["raw_rmse"] for row in rows])),
            "omega_mean_rmse": float(np.mean([row["omega_rmse"] for row in rows])),
            "edge_angle_mean_rmse": float(np.mean([row["edge_rmse"] for row in rows])),
            "maximum_coi_drift": float(np.max([row["coi_drift"] for row in rows])),
            "minimum_q": float(np.min([row["q_min"] for row in rows])),
            "maximum_q": float(np.max([row["q_max"] for row in rows])),
            "failure_rate": 0.0,
            "source_standardized_loss": source_losses,
        }

    def fit(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-3E requires nonempty train and validation sets")
        selected_train = train[: run_context.max_cases]
        selected_val = val[: run_context.max_cases]
        self.max_internal_step = run_context.max_internal_step
        self.validation_scales = M0Model._training_scales(selected_train, len(selected_train))
        self.validation_scales_by_source = {
            source: M0Model._training_scales(group, len(group))
            for source, group in self._source_groups(selected_train).items()
        }

        anchor_fit: FitResult | None = None
        anchor_policy = "provided_frozen_anchor"
        if self.anchor is None or self.anchor.damping is None or self.anchor.coupling is None:
            train_groups = self._source_groups(selected_train)
            val_groups = self._source_groups(selected_val)
            preferred = "ieee39_highorder_dataset"
            anchor_train = train_groups.get(preferred, selected_train)
            anchor_val = val_groups.get(preferred, selected_val)
            self.anchor = M0Model(max_internal_step=run_context.max_internal_step)
            anchor_fit = self.anchor.fit(anchor_train, anchor_val, run_context)
            anchor_policy = (
                "source_homogeneous_highorder_weak_form"
                if preferred in train_groups else "all_available_weak_form"
            )
        self.anchor.max_internal_step = run_context.max_internal_step

        candidates: list[dict[str, Any]] = []
        for threshold in self.activation_threshold_candidates:
            self.activation_threshold = float(threshold)
            for alpha in self.alpha_candidates:
                if alpha == 0.0 and threshold != self.activation_threshold_candidates[0]:
                    continue
                for tau in self.tau_candidates:
                    self.alpha, self.tau = float(alpha), float(tau)
                    summary = self._validation_summary(selected_val, run_context)
                    candidates.append({
                        "activation_threshold": self.activation_threshold,
                        "alpha": self.alpha, "tau": self.tau, **summary,
                    })
        eligible = candidates
        fallback = min(
            (row for row in candidates if row["alpha"] == 0.0),
            key=lambda row: (row["standardized_loss"], row["tau"]),
        )
        if self.max_source_loss_degradation is not None:
            fallback_sources = fallback["source_standardized_loss"]
            eligible = [
                row for row in candidates
                if row["failure_rate"] == 0.0
                and all(
                    row["source_standardized_loss"].get(source, float("inf"))
                    <= baseline*(1.0+self.max_source_loss_degradation)+1e-12
                    for source, baseline in fallback_sources.items()
                )
            ]
        if self.activation_sparsity_tolerance is None:
            selection_pool = eligible
            selection_policy = "minimum_validation_loss"
            selection_key = lambda row: (
                row["standardized_loss"], row["raw_mean_rmse"],
                row["activation_threshold"], row["alpha"], row["tau"],
            )
        else:
            best_eligible_loss = min(row["standardized_loss"] for row in eligible)
            selection_pool = [
                row for row in eligible
                if row["standardized_loss"]
                <= best_eligible_loss*(1.0+self.activation_sparsity_tolerance)+1e-12
            ]
            selection_policy = "highest_activation_threshold_within_validation_tolerance"
            selection_key = lambda row: (
                -row["activation_threshold"], row["standardized_loss"],
                row["raw_mean_rmse"], row["alpha"], row["tau"],
            )
        selected = min(selection_pool, key=selection_key)
        self.activation_threshold = selected["activation_threshold"]
        self.alpha, self.tau = selected["alpha"], selected["tau"]
        equilibrium_residuals = [
            self.equilibrium_residual(sample["model_inputs"]["static"])
            for sample in selected_val
        ]
        self.diagnostics = {
            "optimizer": "source_homogeneous_M0_anchor_plus_validation_grid",
            "epochs": 0,
            "third_state": "normalized_cluster_transient_emf",
            "classical_machine_parameter_claim": False,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "initial_state_policy": "causal_quasi_steady_from_clearing_angle_only",
            "anchor_policy": anchor_policy,
            "anchor_train_case_ids": (
                list(anchor_fit.train_case_ids) if anchor_fit is not None else []
            ),
            "anchor_validation_case_ids": (
                list(anchor_fit.val_case_ids) if anchor_fit is not None else []
            ),
            "anchor_selected_ridge": self.anchor.selected_ridge,
            "anchor_damping": self.anchor.damping.tolist(),
            "anchor_coupling": self.anchor.coupling.tolist(),
            "selected_alpha": self.alpha,
            "selected_tau": self.tau,
            "selected_activation_threshold": self.activation_threshold,
            "activation_width": self.activation_width,
            "alpha_candidates": list(self.alpha_candidates),
            "tau_candidates": list(self.tau_candidates),
            "activation_threshold_candidates": list(self.activation_threshold_candidates),
            "selection_objective": "equal_source_mean_of_source_standardized_rollout_loss",
            "maximum_source_loss_degradation": self.max_source_loss_degradation,
            "eligible_candidate_count": len(eligible),
            "activation_sparsity_tolerance": self.activation_sparsity_tolerance,
            "sparsity_selection_pool_count": len(selection_pool),
            "selection_policy": selection_policy,
            "fallback_source_standardized_loss": fallback["source_standardized_loss"],
            "candidate_table": candidates,
            "selected_validation": selected,
            "strict_M0_fallback_included": True,
            "selected_is_M0_fallback": bool(self.alpha == 0.0),
            "maximum_equilibrium_residual": float(max(equilibrium_residuals)),
            "damping_frozen_after_anchor": True,
            "coupling_frozen_after_anchor": True,
        }
        return FitResult(
            method=self.method_name,
            status="completed",
            train_case_ids=tuple(sample["event_id"] for sample in selected_train),
            val_case_ids=tuple(sample["event_id"] for sample in selected_val),
            best_validation_loss=float(selected["standardized_loss"]),
            details=self.diagnostics,
        )

    def equilibrium_residual(self, static: dict[str, np.ndarray]) -> float:
        q = np.ones(np.asarray(static["mass"]).size, dtype=np.float64)
        delta = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
        omega = np.zeros_like(delta)
        pm = _pm_equilibrium(static)
        derivatives = self.vector_field_augmented(delta, omega, q, pm, static)
        return float(max(np.max(np.abs(value)) for value in derivatives))

    def convergence_audit(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> dict[str, float | bool]:
        trajectories = [
            self.rollout_augmented(
                initial_state, time_grid, exogenous,
                max_step=self.max_internal_step/divisor,
            )
            for divisor in (1, 2, 4)
        ]
        errors = []
        for coarse, fine in zip(trajectories[:-1], trajectories[1:]):
            errors.append(float(np.sqrt(sum(
                np.mean((coarse[index]-fine[index])**2) for index in range(3)
            ))))
        return {
            "dt_vs_dt2": errors[0],
            "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0]/max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0] + 1e-12),
        }

    def save(self, path: str | Path) -> None:
        damping, coupling, alpha, tau = self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            damping=damping,
            coupling=coupling,
            alpha=alpha,
            tau=tau,
            activation_threshold=self.activation_threshold,
            activation_width=self.activation_width,
            selected_ridge=(
                np.nan if self.anchor.selected_ridge is None else self.anchor.selected_ridge
            ),
            max_internal_step=self.max_internal_step,
            q_floor=self.q_floor,
            q_ceiling=self.q_ceiling,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransientEmfM0Model":
        path = Path(path)
        with np.load(path) as data:
            ridge = float(data["selected_ridge"])
            anchor = M0Model(
                data["damping"], data["coupling"],
                selected_ridge=None if np.isnan(ridge) else ridge,
                max_internal_step=float(data["max_internal_step"]),
            )
            model = cls(
                alpha=float(data["alpha"]), tau=float(data["tau"]),
                activation_threshold=(
                    float(data["activation_threshold"])
                    if "activation_threshold" in data else 0.0
                ),
                activation_width=(
                    float(data["activation_width"])
                    if "activation_width" in data else 0.05
                ),
                anchor=anchor,
                max_internal_step=float(data["max_internal_step"]),
                q_floor=float(data["q_floor"]), q_ceiling=float(data["q_ceiling"]),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get(
                "diagnostics", {}
            )
            if hasattr(model, "damping_mode"):
                model.damping_mode = str(
                    model.diagnostics.get("selected_damping_mode", "loaded")
                )
        return model

    def physics_audit(self) -> dict[str, float | str | bool]:
        if self.anchor is None or self.damping is None or self.coupling is None or self.alpha is None or self.tau is None:
            return {"status": "not_fitted"}
        nonnegative = bool(
            np.all(self.damping >= 0) and np.all(self.coupling >= 0)
            and self.alpha >= 0 and self.tau > 0
        )
        equilibrium_residual = float(
            self.diagnostics.get("maximum_equilibrium_residual", np.nan)
        )
        equilibrium_pass = bool(
            not np.isfinite(equilibrium_residual) or equilibrium_residual < 1e-10
        )
        return {
            "status": "pass" if nonnegative and equilibrium_pass else "fail",
            "method": self.method_name,
            "third_state": "normalized_cluster_transient_emf",
            "classical_machine_parameter_claim": False,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "initial_state_policy": "causal_quasi_steady_from_clearing_angle_only",
            "kirchhoff_graph_current_by_construction": True,
            "internal_force_balance_by_construction": True,
            "strict_M0_fallback_at_alpha_zero": True,
            "damping_frozen_after_anchor": True,
            "coupling_frozen_after_anchor": True,
            "nonnegative_parameters": nonnegative,
            "selected_alpha": float(self.alpha),
            "selected_tau": float(self.tau),
            "selected_activation_threshold": float(self.activation_threshold),
            "activation_width": float(self.activation_width),
            "maximum_equilibrium_residual": equilibrium_residual,
        }


class ReidentifiedTransientEmfM0Model(TransientEmfM0Model):
    """M0-3EG with alternating weak re-identification of ``D`` and ``K``.

    For fixed graph parameters, the hidden transient EMF is integrated along
    each observed *training* angle path.  Conditional on that hidden path, the
    swing equation is again linear in damping and coupling and is solved once
    by bounded integral ridge regression.  Repeating these two steps is a
    block-coordinate physics solve, not epoch training.

    Two damping structures are audited: free nonnegative nodal damping and a
    uniform damping-to-inertia ratio ``D_i = rho m_i``.  The latter is preferred
    whenever it lies within a declared validation tolerance of the best
    candidate, because it removes weak nodewise directions rather than filling
    them with arbitrary positive floors.
    """

    method_name = "M0-3ER"

    def __init__(
        self,
        *args: Any,
        damping_modes: tuple[str, ...] = ("free", "mass_proportional"),
        max_reidentification_iterations: int = 4,
        structural_validation_tolerance: float = 0.02,
        reidentification_source_guard: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        allowed = {"free", "mass_proportional"}
        self.damping_modes = tuple(damping_modes)
        if not self.damping_modes or any(mode not in allowed for mode in self.damping_modes):
            raise ValueError("damping modes must be free and/or mass_proportional")
        self.max_reidentification_iterations = int(max_reidentification_iterations)
        if self.max_reidentification_iterations < 1:
            raise ValueError("at least one re-identification iteration is required")
        self.structural_validation_tolerance = float(structural_validation_tolerance)
        self.reidentification_source_guard = float(reidentification_source_guard)
        if self.structural_validation_tolerance < 0 or self.reidentification_source_guard < 0:
            raise ValueError("re-identification tolerances must be nonnegative")
        self.damping_mode = "unfitted"

    def build_emf_conditioned_weak_system(
        self,
        samples: list[dict[str, Any]],
        *,
        windows: tuple[int, ...],
        max_cases: int | None,
        horizon_steps: int,
        max_fit_steps: int,
        equal_case_weight: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
        selected = samples if max_cases is None else samples[:max_cases]
        case_matrices: list[np.ndarray] = []
        case_vectors: list[np.ndarray] = []
        integration_windows = 0
        minimum_q, maximum_q = float("inf"), float("-inf")
        for sample in selected:
            target = sample["training_context"]["targets"]
            t = np.asarray(target["t"], dtype=np.float64)
            delta = np.asarray(target["delta"], dtype=np.float64)
            omega = np.asarray(target["omega"], dtype=np.float64)
            pm = np.asarray(sample["training_context"]["Pm_observed"], dtype=np.float64)
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
            incidence = np.asarray(static["incidence"], dtype=np.float64)
            equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
            limit = min(
                len(t),
                int(max_fit_steps) if int(max_fit_steps) > 0
                else max(horizon_steps*5, max(windows, default=1)+1),
            )
            if limit < 2:
                continue
            t, delta, omega, pm = t[:limit], delta[:limit], omega[:limit], pm[:limit]
            q = self.teacher_forced_emf(delta, t, static)
            minimum_q = min(minimum_q, float(np.min(q)))
            maximum_q = max(maximum_q, float(np.max(q)))
            projector = np.eye(mass.size)-np.outer(mass, np.ones(mass.size))/np.sum(mass)
            tail, head = self._edge_endpoints(incidence)
            eta_eq = incidence.T@equilibrium
            designs = []
            for index in range(limit):
                eta = incidence.T@delta[index]
                edge_feature = q[index, tail]*q[index, head]*np.sin(eta)-np.sin(eta_eq)
                designs.append(np.column_stack((
                    projector@np.diag(omega[index]),
                    incidence*edge_feature,
                )))
            designs = np.stack(designs)
            delta_pm = pm-_pm_equilibrium(static)
            rows, targets = [], []
            for width in sorted(set(int(value) for value in windows if int(value) > 0)):
                if width >= limit:
                    continue
                for start in range(limit-width):
                    stop = start+width
                    segment = slice(start, stop+1)
                    rows.append(_trapz(designs[segment], t[segment]))
                    targets.append(
                        _trapz(delta_pm[segment], t[segment])
                        -mass*(omega[stop]-omega[start])
                    )
            if not rows:
                continue
            case_matrix = np.concatenate(rows)
            case_vector = np.concatenate(targets)
            if equal_case_weight:
                weight = 1.0/np.sqrt(len(case_vector))
                case_matrix, case_vector = weight*case_matrix, weight*case_vector
            case_matrices.append(case_matrix)
            case_vectors.append(case_vector)
            integration_windows += len(rows)
        if not case_matrices:
            raise ValueError("no eligible EMF-conditioned integration windows")
        matrix, vector = np.concatenate(case_matrices), np.concatenate(case_vectors)
        return matrix, vector, {
            "n_scalar_equations": int(matrix.shape[0]),
            "n_free_node_and_edge_parameters": int(matrix.shape[1]),
            "n_integration_windows": int(integration_windows),
            "n_cases": int(len(case_matrices)),
            "case_weighting": (
                "equal_case_total_squared_row_weight" if equal_case_weight
                else "equal_scalar_rows"
            ),
            "teacher_forced_q_min": minimum_q,
            "teacher_forced_q_max": maximum_q,
        }

    @staticmethod
    def _mode_matrix(
        full_matrix: np.ndarray, mass: np.ndarray, n_nodes: int, mode: str
    ) -> np.ndarray:
        node_matrix = full_matrix[:, :n_nodes]
        edge_matrix = full_matrix[:, n_nodes:]
        if mode == "free":
            return full_matrix
        if mode == "mass_proportional":
            return np.column_stack((node_matrix@mass, edge_matrix))
        raise ValueError(f"unknown damping mode: {mode}")

    @staticmethod
    def _decode_mode(
        theta: np.ndarray, mass: np.ndarray, n_nodes: int, mode: str
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        if mode == "free":
            return theta[:n_nodes], theta[n_nodes:], None
        if mode == "mass_proportional":
            rho = float(theta[0])
            return rho*mass, theta[1:], rho
        raise ValueError(f"unknown damping mode: {mode}")

    def fit(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-3ER requires nonempty train and validation sets")
        damping0, coupling0, alpha, tau = self._require_fitted()
        selected_train = train[:run_context.max_cases]
        selected_val = val[:run_context.max_cases]
        self.max_internal_step = run_context.max_internal_step
        self.validation_scales = M0Model._training_scales(selected_train, len(selected_train))
        self.validation_scales_by_source = {
            source: M0Model._training_scales(group, len(group))
            for source, group in self._source_groups(selected_train).items()
        }
        reference_damping, reference_coupling = damping0.copy(), coupling0.copy()
        baseline = self._validation_summary(selected_val, run_context)
        candidates: list[dict[str, Any]] = [{
            "damping_mode": "frozen_M0_3EG", "iteration": 0, "ridge": None,
            "damping": reference_damping.copy(), "coupling": reference_coupling.copy(),
            "damping_to_inertia_ratio": None, "relative_coupling_change": 0.0,
            "solver": {}, "system": {}, "validation": baseline,
        }]
        mass = np.asarray(
            selected_train[0]["model_inputs"]["static"]["mass"], dtype=np.float64
        ).reshape(-1)
        for sample in selected_train[1:]:
            other = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=np.float64).reshape(-1)
            if not np.allclose(other, mass, rtol=0.0, atol=1e-14):
                raise ValueError("M0-3ER requires a common clustered mass prior")
        n_nodes = mass.size

        for mode in self.damping_modes:
            current_damping, current_coupling = reference_damping.copy(), reference_coupling.copy()
            for iteration in range(1, self.max_reidentification_iterations+1):
                self.anchor = M0Model(
                    current_damping, current_coupling,
                    max_internal_step=run_context.max_internal_step,
                )
                full_matrix, vector, system = self.build_emf_conditioned_weak_system(
                    selected_train,
                    windows=run_context.integral_windows,
                    max_cases=run_context.max_cases,
                    horizon_steps=run_context.horizon_steps,
                    max_fit_steps=run_context.fit_max_steps,
                    equal_case_weight=True,
                )
                matrix = self._mode_matrix(full_matrix, mass, n_nodes, mode)
                iteration_candidates = []
                for ridge in run_context.ridge_values:
                    theta, solver = M0Model._solve_nonnegative_ridge(matrix, vector, ridge)
                    damping, coupling, ratio = self._decode_mode(theta, mass, n_nodes, mode)
                    self.anchor = M0Model(
                        damping, coupling, selected_ridge=float(ridge),
                        max_internal_step=run_context.max_internal_step,
                    )
                    validation = self._validation_summary(selected_val, run_context)
                    relative_change = float(
                        np.linalg.norm(coupling-current_coupling)
                        / max(np.linalg.norm(current_coupling), np.finfo(float).eps)
                    )
                    row = {
                        "damping_mode": mode, "iteration": iteration,
                        "ridge": float(ridge), "damping": damping.copy(),
                        "coupling": coupling.copy(),
                        "damping_to_inertia_ratio": ratio,
                        "relative_coupling_change": relative_change,
                        "solver": solver, "system": system, "validation": validation,
                    }
                    candidates.append(row)
                    iteration_candidates.append(row)
                chosen = min(
                    iteration_candidates,
                    key=lambda row: (
                        row["validation"]["standardized_loss"],
                        row["validation"]["raw_mean_rmse"], row["ridge"],
                    ),
                )
                current_damping = chosen["damping"].copy()
                current_coupling = chosen["coupling"].copy()
                if chosen["relative_coupling_change"] < 1e-5:
                    break

        baseline_sources = baseline["source_standardized_loss"]
        eligible = [
            row for row in candidates
            if row["validation"]["failure_rate"] == 0.0
            and all(
                row["validation"]["source_standardized_loss"].get(source, float("inf"))
                <= loss*(1.0+self.reidentification_source_guard)+1e-12
                for source, loss in baseline_sources.items()
            )
        ]
        best_loss = min(row["validation"]["standardized_loss"] for row in eligible)
        near_best = [
            row for row in eligible
            if row["validation"]["standardized_loss"]
            <= best_loss*(1.0+self.structural_validation_tolerance)+1e-12
        ]
        preference = {"mass_proportional": 0, "free": 1, "frozen_M0_3EG": 2}
        selected = min(
            near_best,
            key=lambda row: (
                preference[row["damping_mode"]],
                row["validation"]["standardized_loss"],
                row["iteration"], float("inf") if row["ridge"] is None else row["ridge"],
            ),
        )
        self.anchor = M0Model(
            selected["damping"], selected["coupling"],
            selected_ridge=selected["ridge"], max_internal_step=run_context.max_internal_step,
        )
        self.alpha, self.tau = alpha, tau
        self.damping_mode = selected["damping_mode"]
        equilibrium_residuals = [
            self.equilibrium_residual(sample["model_inputs"]["static"])
            for sample in selected_val
        ]
        candidate_table = []
        for row in candidates:
            candidate_table.append({
                "damping_mode": row["damping_mode"],
                "iteration": row["iteration"], "ridge": row["ridge"],
                "standardized_loss": row["validation"]["standardized_loss"],
                "raw_mean_rmse": row["validation"]["raw_mean_rmse"],
                "raw_median_rmse": row["validation"]["raw_median_rmse"],
                "failure_rate": row["validation"]["failure_rate"],
                "source_standardized_loss": row["validation"]["source_standardized_loss"],
                "minimum_damping": float(np.min(row["damping"])),
                "median_damping": float(np.median(row["damping"])),
                "maximum_damping": float(np.max(row["damping"])),
                "damping_to_inertia_ratio": row["damping_to_inertia_ratio"],
                "relative_coupling_change": row["relative_coupling_change"],
                "condition_number": row["solver"].get("matrix_condition_number"),
                "effective_rank": row["solver"].get("effective_rank"),
                "weak_residual_rmse": row["solver"].get("weak_residual_rmse"),
                "teacher_forced_q_min": row["system"].get("teacher_forced_q_min"),
                "teacher_forced_q_max": row["system"].get("teacher_forced_q_max"),
            })
        self.diagnostics = {
            "optimizer": "alternating_teacher_forced_emf_plus_bounded_integral_ridge",
            "epochs": 0,
            "third_state": "normalized_cluster_transient_emf",
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "teacher_forcing_used_only_for_training_hidden_feature": True,
            "selected_damping_mode": self.damping_mode,
            "selected_iteration": selected["iteration"],
            "selected_ridge": selected["ridge"],
            "selected_damping_to_inertia_ratio": selected["damping_to_inertia_ratio"],
            "selected_alpha": self.alpha, "selected_tau": self.tau,
            "selected_activation_threshold": self.activation_threshold,
            "activation_width": self.activation_width,
            "damping_modes": list(self.damping_modes),
            "max_reidentification_iterations": self.max_reidentification_iterations,
            "structural_validation_tolerance": self.structural_validation_tolerance,
            "reidentification_source_guard": self.reidentification_source_guard,
            "eligible_candidate_count": len(eligible),
            "near_best_candidate_count": len(near_best),
            "baseline_validation": baseline,
            "selected_validation": selected["validation"],
            "candidate_table": candidate_table,
            "reference_damping": reference_damping.tolist(),
            "reference_coupling": reference_coupling.tolist(),
            "final_damping": self.damping.tolist(),
            "final_coupling": self.coupling.tolist(),
            "maximum_equilibrium_residual": float(max(equilibrium_residuals)),
            "physical_topology_claim": False,
        }
        return FitResult(
            method=self.method_name, status="completed",
            train_case_ids=tuple(sample["event_id"] for sample in selected_train),
            val_case_ids=tuple(sample["event_id"] for sample in selected_val),
            best_validation_loss=float(selected["validation"]["standardized_loss"]),
            details=self.diagnostics,
        )

    def physics_audit(self) -> dict[str, float | str | bool]:
        audit = super().physics_audit()
        if audit["status"] == "not_fitted":
            return audit
        selected_mode = str(self.diagnostics.get("selected_damping_mode", self.damping_mode))
        mass_ratio_consistent = True
        if selected_mode == "mass_proportional":
            ratio = self.diagnostics.get("selected_damping_to_inertia_ratio")
            mass_ratio_consistent = bool(ratio is not None and float(ratio) > 0)
        audit.update({
            "method": self.method_name,
            "alternating_integral_reidentification": True,
            "teacher_forcing_used_only_for_training_hidden_feature": True,
            "selected_damping_mode": selected_mode,
            "mass_proportional_ratio_positive": mass_ratio_consistent,
            "all_damping_strictly_positive": bool(np.all(self.damping > 0)),
            "status": "pass" if (
                audit["status"] == "pass" and mass_ratio_consistent
            ) else "fail",
        })
        return audit
