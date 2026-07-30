from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .m0 import M0Model, _pm_equilibrium


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
        selected = min(
            eligible,
            key=lambda row: (
                row["standardized_loss"], row["raw_mean_rmse"],
                row["activation_threshold"], row["alpha"], row["tau"],
            ),
        )
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
