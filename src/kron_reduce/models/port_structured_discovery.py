from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .m0 import M0Model
from .pm_modal_discovery import CausalPmGraphModalM0Model, GraphModalPmDynamics
from .transient_emf_m0 import TransientEmfM0Model


class PortStructuredDiscoveryM0Model(DynamicsModel):
    """Discover an interpretable port-structured closure around M0-3EG.

    The frozen M0-3EG supplies the swing balance, the mechanical-power port,
    the complete effective generator graph, and the normalized transient-EMF
    state.  This class discovers only an additional edge flow::

        m omega_dot = f_3EG - B r

        r_e = sum_f a[f,e] phi_f(eta_e, nu_e)
              + sum_j d[j,e] z[j,e]

        tau_j z_dot[j,e] = nu_e - z[j,e]

    where ``nu = B.T @ omega``.  The first three static features depend only
    on edge angle and are gradients of separable edge potentials.  The last
    two static features and every memory coefficient are constrained
    nonnegative, so they remove rather than inject mechanical energy.  In
    fact, a memory element has storage ``0.5*d*tau*z**2`` and contributes
    ``-d*z**2`` to the total supply balance.

    All coefficients are solved in one integral weak-form constrained ridge
    problem.  Group pruning, time constants and the exact zero-closure
    fallback are selected using validation rollouts only.  The model discovers
    effective generator interactions; it does not claim recovery of physical
    transmission lines or machine-level parameters.
    """

    method_name = "M0-PSD"
    STATIC_FEATURES = (
        "delta_sin",
        "delta_sin_abs",
        "delta_sin_cubic",
        "edge_speed",
        "speed_stress",
    )
    CONSERVATIVE_FEATURES = STATIC_FEATURES[:3]
    DISSIPATIVE_FEATURES = STATIC_FEATURES[3:]
    DEFAULT_TAU_MODE_CANDIDATES = (
        (),
        (0.05,),
        (0.2,),
        (1.0,),
        (0.05, 0.2),
        (0.2, 1.0),
    )

    def __init__(
        self,
        backbone: TransientEmfM0Model | None = None,
        *,
        static_coefficients: np.ndarray | None = None,
        taus: np.ndarray | tuple[float, ...] | None = None,
        memory_coefficients: np.ndarray | None = None,
        tau_mode_candidates: tuple[tuple[float, ...], ...] | None = None,
        ridge_values: tuple[float, ...] = (1e-6, 1e-4, 1e-2),
        group_thresholds: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1),
        coefficient_scales: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
        train_shortlist_per_order: int = 2,
        rollout_short_horizon: int = 20,
        rollout_long_shortlist: int = 4,
        complexity_penalty: float = 1e-4,
        minimum_validation_improvement: float = 0.02,
        maximum_source_loss_degradation: float = 0.02,
        max_internal_step: float = 0.01,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.backbone = backbone
        self.static_coefficients = (
            None
            if static_coefficients is None
            else np.asarray(static_coefficients, dtype=np.float64)
        )
        self.taus = (
            None if taus is None else np.asarray(taus, dtype=np.float64).reshape(-1)
        )
        self.memory_coefficients = (
            None
            if memory_coefficients is None
            else np.asarray(memory_coefficients, dtype=np.float64)
        )
        self.tau_mode_candidates = tuple(
            tuple(float(value) for value in row)
            for row in (tau_mode_candidates or self.DEFAULT_TAU_MODE_CANDIDATES)
        )
        for row in self.tau_mode_candidates:
            if (
                any(not np.isfinite(value) or value <= 0 for value in row)
                or tuple(sorted(set(row))) != row
            ):
                raise ValueError("memory time constants must be finite, positive and unique")
        if () not in self.tau_mode_candidates:
            raise ValueError("time-constant candidates must include static-only closure")
        self.ridge_values = tuple(float(value) for value in ridge_values)
        self.group_thresholds = tuple(float(value) for value in group_thresholds)
        self.coefficient_scales = tuple(float(value) for value in coefficient_scales)
        if any(not np.isfinite(value) or value < 0 for value in self.ridge_values):
            raise ValueError("ridge values must be finite and nonnegative")
        if any(not np.isfinite(value) or value < 0 for value in self.group_thresholds):
            raise ValueError("group thresholds must be finite and nonnegative")
        if any(
            not np.isfinite(value) or value <= 0 or value > 1
            for value in self.coefficient_scales
        ):
            raise ValueError("coefficient scales must lie in (0, 1]")
        self.train_shortlist_per_order = int(train_shortlist_per_order)
        self.rollout_short_horizon = int(rollout_short_horizon)
        self.rollout_long_shortlist = int(rollout_long_shortlist)
        if (
            self.train_shortlist_per_order < 1
            or self.rollout_short_horizon < 2
            or self.rollout_long_shortlist < 1
        ):
            raise ValueError("shortlist sizes and rollout screening horizon must be positive")
        self.complexity_penalty = float(complexity_penalty)
        self.minimum_validation_improvement = float(minimum_validation_improvement)
        self.maximum_source_loss_degradation = float(maximum_source_loss_degradation)
        if (
            self.complexity_penalty < 0
            or self.minimum_validation_improvement < 0
            or self.maximum_source_loss_degradation < 0
        ):
            raise ValueError("selection tolerances must be nonnegative")
        self.max_internal_step = float(max_internal_step)
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
        self.validation_scales_by_source: dict[str, dict[str, float]] = {}
        self.diagnostics = diagnostics or {}

    @staticmethod
    def _source(sample: dict[str, Any]) -> str:
        return str(
            sample.get("training_context", {})
            .get("metadata", {})
            .get("source_dataset", "all")
        )

    @classmethod
    def _source_groups(
        cls, samples: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            groups.setdefault(cls._source(sample), []).append(sample)
        return groups

    @property
    def damping(self) -> np.ndarray | None:
        return None if self.backbone is None else self.backbone.damping

    @property
    def coupling(self) -> np.ndarray | None:
        return None if self.backbone is None else self.backbone.coupling

    def _require_backbone(self) -> TransientEmfM0Model:
        if self.backbone is None:
            raise RuntimeError("M0-PSD requires a fitted M0-3EG backbone")
        self.backbone._require_fitted()
        return self.backbone

    def _require_fitted(
        self,
    ) -> tuple[TransientEmfM0Model, np.ndarray, np.ndarray, np.ndarray]:
        backbone = self._require_backbone()
        if (
            self.static_coefficients is None
            or self.taus is None
            or self.memory_coefficients is None
        ):
            raise RuntimeError("M0-PSD is not fitted")
        static = np.asarray(self.static_coefficients, dtype=np.float64)
        memory = np.asarray(self.memory_coefficients, dtype=np.float64)
        if static.ndim != 2 or static.shape[0] != len(self.STATIC_FEATURES):
            raise ValueError("static coefficients must have shape (feature, edge)")
        if memory.shape != (len(self.taus), static.shape[1]):
            raise ValueError("memory coefficients must have shape (mode, edge)")
        return backbone, self.taus, static, memory

    @staticmethod
    def _edge_coordinates(
        delta: np.ndarray, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        equilibrium = np.asarray(
            static["equilibrium_delta"], dtype=np.float64
        ).reshape(-1)
        eta = incidence.T @ np.asarray(delta, dtype=np.float64).reshape(-1)
        eta_eq = incidence.T @ equilibrium
        nu = incidence.T @ np.asarray(omega, dtype=np.float64).reshape(-1)
        return eta, eta_eq, nu

    def _static_features(
        self, delta: np.ndarray, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        eta, eta_eq, nu = self._edge_coordinates(delta, omega, static)
        delta_sin = np.sin(eta) - np.sin(eta_eq)
        return np.stack(
            (
                delta_sin,
                delta_sin * np.abs(delta_sin),
                delta_sin**3,
                nu,
                nu * np.abs(delta_sin),
            ),
            axis=0,
        )

    def initial_memory(
        self, delta: np.ndarray, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.taus is None:
            raise RuntimeError("M0-PSD memory order is not selected")
        _, _, nu = self._edge_coordinates(delta, omega, static)
        return np.repeat(nu[np.newaxis, :], len(self.taus), axis=0)

    def vector_field_augmented(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        q: np.ndarray,
        z: np.ndarray,
        pm: np.ndarray,
        static: dict[str, np.ndarray],
        topology: dict[str, np.ndarray | float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        backbone, taus, static_coefficients, memory_coefficients = self._require_fitted()
        dd, dw, dq = backbone.vector_field_augmented(
            delta, omega, q, pm, static, topology
        )
        if z.shape != memory_coefficients.shape:
            raise ValueError("memory state shape does not match discovered coefficients")
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        features = self._static_features(delta, omega, static)
        edge_flow = np.sum(static_coefficients * features, axis=0)
        if len(taus):
            edge_flow = edge_flow + np.sum(memory_coefficients * z, axis=0)
        correction = coi_project_power(-incidence @ edge_flow, mass)
        _, _, nu = self._edge_coordinates(delta, omega, static)
        dz = (
            (nu[np.newaxis, :] - z) / taus[:, np.newaxis]
            if len(taus)
            else np.empty_like(z)
        )
        return dd, dw + correction / mass, dq, dz

    @staticmethod
    def _input_mode(pm: np.ndarray, requested: str) -> str:
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        backbone, _, _, memory_coefficients = self._require_fitted()
        t = np.asarray(time_grid, dtype=np.float64)
        pm = np.asarray(exogenous["Pm"], dtype=np.float64)
        if len(t) != len(pm) or len(t) < 1:
            raise ValueError("time and Pm schedules must have the same positive length")
        step_limit = self.max_internal_step if max_step is None else float(max_step)
        if step_limit <= 0:
            raise ValueError("maximum integration step must be positive")
        mode = self._input_mode(pm, exogenous.get("interpolation", "auto"))
        mass = np.asarray(initial_state["mass"], dtype=np.float64).reshape(-1)
        delta = coi_project(np.asarray(initial_state["delta0"]).reshape(-1), mass)
        omega = coi_project(np.asarray(initial_state["omega0"]).reshape(-1), mass)
        q = backbone.initial_emf(delta, initial_state)
        z = self.initial_memory(delta, omega, initial_state)
        if z.shape != memory_coefficients.shape:
            raise ValueError("selected edge count does not match the rollout graph")
        topology = backbone._graph_topology(initial_state)
        outputs = ([delta.copy()], [omega.copy()], [q.copy()], [z.copy()])

        def input_at(index: int, fraction: float) -> np.ndarray:
            if mode == "zoh":
                return pm[index]
            return pm[index] + fraction * (pm[index + 1] - pm[index])

        def field(di, wi, qi, zi, pi):
            return self.vector_field_augmented(
                di, wi, qi, zi, pi, initial_state, topology
            )

        for index in range(len(t) - 1):
            interval = float(t[index + 1] - t[index])
            if interval <= 0:
                raise ValueError("rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval / step_limit)))
            step = interval / substeps
            for substep in range(substeps):
                f0 = substep / substeps
                fh = (substep + 0.5) / substeps
                f1 = (substep + 1.0) / substeps
                k1 = field(delta, omega, q, z, input_at(index, f0))
                k2 = field(
                    delta + 0.5 * step * k1[0],
                    omega + 0.5 * step * k1[1],
                    q + 0.5 * step * k1[2],
                    z + 0.5 * step * k1[3],
                    input_at(index, fh),
                )
                k3 = field(
                    delta + 0.5 * step * k2[0],
                    omega + 0.5 * step * k2[1],
                    q + 0.5 * step * k2[2],
                    z + 0.5 * step * k2[3],
                    input_at(index, fh),
                )
                k4 = field(
                    delta + step * k3[0],
                    omega + step * k3[1],
                    q + step * k3[2],
                    z + step * k3[3],
                    input_at(index, f1),
                )
                delta = coi_project(
                    delta + step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
                    mass,
                )
                omega = coi_project(
                    omega + step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
                    mass,
                )
                q = q + step * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
                z = z + step * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
                if not all(np.isfinite(value).all() for value in (delta, omega, q, z)):
                    raise FloatingPointError("nonfinite M0-PSD state")
                if np.min(q) <= backbone.q_floor or np.max(q) >= backbone.q_ceiling:
                    raise FloatingPointError("M0-PSD transient EMF left audited domain")
            for output, value in zip(outputs, (delta, omega, q, z), strict=True):
                output.append(value.copy())
        return tuple(np.stack(value) for value in outputs)  # type: ignore[return-value]

    def rollout(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        delta, omega, _, _ = self.rollout_augmented(
            initial_state, time_grid, exogenous
        )
        return delta, omega

    @staticmethod
    def _first_order_hold_speed(
        speed_path: np.ndarray, time_grid: np.ndarray, taus: np.ndarray
    ) -> np.ndarray:
        speed = np.asarray(speed_path, dtype=np.float64)
        t = np.asarray(time_grid, dtype=np.float64)
        taus = np.asarray(taus, dtype=np.float64).reshape(-1)
        if not len(taus):
            return np.empty((len(t), 0, speed.shape[1]), dtype=np.float64)
        z = np.repeat(speed[[0]], len(taus), axis=0)
        output = [z.copy()]
        for index, dt in enumerate(np.diff(t)):
            if dt <= 0:
                raise ValueError("memory-filter time must be strictly increasing")
            ratio = dt / taus
            decay = np.exp(-ratio)
            b0 = (1.0 - decay) / ratio - decay
            b1 = 1.0 - (1.0 - decay) / ratio
            z = (
                decay[:, None] * z
                + b0[:, None] * speed[index][None, :]
                + b1[:, None] * speed[index + 1][None, :]
            )
            output.append(z.copy())
        return np.stack(output)

    def _baseline_force_path(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        q: np.ndarray,
        pm: np.ndarray,
        static: dict[str, np.ndarray],
    ) -> np.ndarray:
        backbone = self._require_backbone()
        topology = backbone._graph_topology(static)
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        rows = []
        for index in range(len(delta)):
            _, acceleration, _ = backbone.vector_field_augmented(
                delta[index], omega[index], q[index], pm[index], static, topology
            )
            rows.append(mass * acceleration)
        return np.stack(rows)

    def _prepare_weak_paths(
        self,
        samples: list[dict[str, Any]],
        *,
        horizon_steps: int,
        max_fit_steps: int,
    ) -> list[dict[str, Any]]:
        backbone = self._require_backbone()
        prepared: list[dict[str, Any]] = []
        for sample in samples:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), horizon_steps)
            if max_fit_steps > 0:
                steps = min(steps, max_fit_steps)
            if steps < 2:
                continue
            t = np.asarray(target["t"][:steps], dtype=np.float64)
            delta = np.asarray(target["delta"][:steps], dtype=np.float64)
            omega = np.asarray(target["omega"][:steps], dtype=np.float64)
            pm = np.asarray(
                sample["training_context"]["Pm_observed"][:steps], dtype=np.float64
            )
            static = sample["model_inputs"]["static"]
            q = backbone.teacher_forced_emf(delta, t, static)
            features = np.stack(
                [self._static_features(delta[k], omega[k], static) for k in range(steps)]
            )
            prepared.append(
                {
                    "event_id": sample["event_id"],
                    "source": self._source(sample),
                    "t": t,
                    "delta": delta,
                    "omega": omega,
                    "static": static,
                    "features": features,
                    "baseline_force": self._baseline_force_path(
                        delta, omega, q, pm, static
                    ),
                }
            )
        if not prepared:
            raise ValueError("no M0-PSD weak paths could be prepared")
        return prepared

    def _build_normal_equations(
        self,
        prepared: list[dict[str, Any]],
        taus: tuple[float, ...],
        *,
        windows: tuple[int, ...],
    ) -> dict[str, Any]:
        taus_array = np.asarray(taus, dtype=np.float64)
        n_edges = np.asarray(prepared[0]["static"]["incidence"]).shape[1]
        n_static = len(self.STATIC_FEATURES) * n_edges
        n_parameters = n_static + len(taus_array) * n_edges
        gram = np.zeros((n_parameters, n_parameters), dtype=np.float64)
        response = np.zeros(n_parameters, dtype=np.float64)
        target_energy = 0.0
        effective_blocks = 0
        scalar_equations = 0
        horizon_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for item in prepared:
            t = item["t"]
            omega = item["omega"]
            static = item["static"]
            incidence = np.asarray(static["incidence"], dtype=np.float64)
            mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
            features = item["features"]
            speed = features[:, self.STATIC_FEATURES.index("edge_speed"), :]
            memory = self._first_order_hold_speed(speed, t, taus_array)
            static_design = -np.einsum(
                "ne,tfe->tnfe", incidence, features, optimize=True
            ).reshape(len(t), len(mass), n_static)
            if len(taus_array):
                memory_design = -np.einsum(
                    "ne,tme->tnme", incidence, memory, optimize=True
                ).reshape(len(t), len(mass), -1)
                design_path = np.concatenate((static_design, memory_design), axis=2)
            else:
                design_path = static_design
            dt = np.diff(t)
            design_prefix = np.concatenate(
                (
                    np.zeros((1, len(mass), n_parameters)),
                    np.cumsum(
                        0.5 * (design_path[:-1] + design_path[1:]) * dt[:, None, None],
                        axis=0,
                    ),
                ),
                axis=0,
            )
            force_path = item["baseline_force"]
            force_prefix = np.concatenate(
                (
                    np.zeros((1, len(mass))),
                    np.cumsum(
                        0.5 * (force_path[:-1] + force_path[1:]) * dt[:, None],
                        axis=0,
                    ),
                ),
                axis=0,
            )
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
            for width in sorted({int(value) for value in windows if int(value) > 0}):
                if width >= len(t):
                    continue
                starts = np.arange(len(t) - width)
                stops = starts + width
                matrix = (design_prefix[stops] - design_prefix[starts]).reshape(
                    -1, n_parameters
                )
                actual = mass * (omega[stops] - omega[starts])
                baseline = force_prefix[stops] - force_prefix[starts]
                target = (actual - baseline).reshape(-1)
                # Every case and every horizon receives unit total weight.
                weight = 1.0 / np.sqrt(max(len(target), 1))
                matrix = weight * matrix
                target = weight * target
                gram += matrix.T @ matrix
                response += matrix.T @ target
                target_energy += float(target @ target)
                effective_blocks += 1
                scalar_equations += len(target)
                horizon_counts[str(width)] = horizon_counts.get(str(width), 0) + len(starts)
        if effective_blocks == 0:
            raise ValueError("no M0-PSD integral windows could be constructed")
        return {
            "gram": gram,
            "response": response,
            "target_energy": target_energy,
            "effective_blocks": effective_blocks,
            "n_scalar_equations": scalar_equations,
            "n_parameters": n_parameters,
            "n_static_parameters": n_static,
            "n_memory_parameters": len(taus_array) * n_edges,
            "n_edges": n_edges,
            "n_memory_modes": len(taus_array),
            "horizon_window_counts": horizon_counts,
            "source_case_counts": source_counts,
            "weak_weighting": "equal_case_equal_horizon",
        }

    @staticmethod
    def _factorized_nonnegative_ridge(
        gram: np.ndarray,
        response: np.ndarray,
        target_energy: float,
        effective_blocks: int,
        ridge: float,
        active_columns: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        size = len(response)
        selected = (
            np.ones(size, dtype=bool)
            if active_columns is None
            else np.asarray(active_columns, dtype=bool)
        )
        result = np.zeros(size, dtype=np.float64)
        beta_full = np.zeros(size, dtype=np.float64)
        if not np.any(selected):
            residual = np.sqrt(max(target_energy, 0.0) / max(effective_blocks, 1))
            return result, beta_full, {
                "solver_success": True,
                "solver_iterations": 0,
                "matrix_condition_number": float("inf"),
                "effective_rank": 0,
                "weak_residual_rmse": float(residual),
            }
        sub_gram = np.asarray(gram[np.ix_(selected, selected)], dtype=np.float64)
        sub_response = np.asarray(response[selected], dtype=np.float64)
        diagonal = np.diag(sub_gram) / max(effective_blocks, 1)
        scale = np.sqrt(np.maximum(diagonal, 0.0))
        informative = scale > np.finfo(float).eps
        selected_indices = np.flatnonzero(selected)
        if not np.all(informative):
            selected[selected_indices[~informative]] = False
            return PortStructuredDiscoveryM0Model._factorized_nonnegative_ridge(
                gram,
                response,
                target_energy,
                effective_blocks,
                ridge,
                selected,
            )
        normalized_gram = sub_gram / np.outer(scale, scale)
        normalized_response = sub_response / scale
        regularized = 0.5 * (normalized_gram + normalized_gram.T)
        regularized += ridge * np.eye(len(scale))
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        clipped = np.maximum(eigenvalues, np.finfo(float).eps)
        factor = np.sqrt(clipped)[:, None] * eigenvectors.T
        rhs = np.linalg.lstsq(factor.T, normalized_response, rcond=None)[0]
        solve = lsq_linear(
            factor,
            rhs,
            bounds=(0.0, np.inf),
            max_iter=2000,
            lsmr_tol="auto",
        )
        beta = np.maximum(solve.x, 0.0)
        theta = beta / scale
        result[selected] = theta
        beta_full[selected] = beta
        sse = float(
            result @ gram @ result - 2.0 * result @ response + target_energy
        )
        positive = clipped[clipped > np.max(clipped) * 1e-12]
        condition = (
            float(np.max(positive) / np.min(positive)) if len(positive) else float("inf")
        )
        return result, beta_full, {
            "solver_success": bool(solve.success),
            "solver_iterations": int(solve.nit),
            "matrix_condition_number": condition,
            "effective_rank": len(positive),
            "weak_residual_rmse": float(
                np.sqrt(max(sse, 0.0) / max(effective_blocks, 1))
            ),
        }

    def _group_prune_and_refit(
        self, stats: dict[str, Any], ridge: float, relative_threshold: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        theta, beta, diagnostics = self._factorized_nonnegative_ridge(
            stats["gram"],
            stats["response"],
            stats["target_energy"],
            stats["effective_blocks"],
            ridge,
        )
        n_edges = int(stats["n_edges"])
        n_static = int(stats["n_static_parameters"])
        n_modes = int(stats["n_memory_modes"])
        static_beta = beta[:n_static].reshape(len(self.STATIC_FEATURES), n_edges)
        static_norms = static_beta
        if n_modes:
            memory_norms = np.sqrt(
                np.sum(beta[n_static:].reshape(n_modes, n_edges) ** 2, axis=0)
            )[None, :]
        else:
            memory_norms = np.empty((0, n_edges))
        group_norms = np.concatenate((static_norms, memory_norms), axis=0)
        maximum = float(np.max(group_norms)) if group_norms.size else 0.0
        cutoff = relative_threshold * maximum
        numerical_floor = max(maximum * 1e-12, np.finfo(float).eps)
        active_groups = group_norms > max(cutoff, numerical_floor)
        active_static = active_groups[: len(self.STATIC_FEATURES)].reshape(-1)
        if n_modes:
            active_memory = np.repeat(
                active_groups[-1][None, :], n_modes, axis=0
            ).reshape(-1)
            active_columns = np.concatenate((active_static, active_memory))
        else:
            active_columns = active_static
        theta, beta, refit = self._factorized_nonnegative_ridge(
            stats["gram"],
            stats["response"],
            stats["target_energy"],
            stats["effective_blocks"],
            ridge,
            active_columns,
        )
        del beta, diagnostics
        return theta, {
            **refit,
            "active_groups": int(np.sum(active_groups)),
            "active_edges": int(np.sum(np.any(active_groups, axis=0))),
            "active_coefficients": int(np.sum(theta > 0)),
            "relative_group_threshold": float(relative_threshold),
        }

    def _set_candidate(self, candidate: dict[str, Any]) -> None:
        n_edges = int(candidate["n_edges"])
        fit_taus = tuple(float(value) for value in candidate["fit_taus"])
        theta = np.asarray(candidate["theta"], dtype=np.float64)
        n_static = len(self.STATIC_FEATURES) * n_edges
        self.static_coefficients = theta[:n_static].reshape(
            len(self.STATIC_FEATURES), n_edges
        )
        if fit_taus:
            self.taus = np.asarray(fit_taus, dtype=np.float64)
            self.memory_coefficients = theta[n_static:].reshape(len(fit_taus), n_edges)
        else:
            # A dummy zero memory row keeps save/load and convergence arrays
            # well-defined while retaining an exact static-only realization.
            self.taus = np.asarray([1.0], dtype=np.float64)
            self.memory_coefficients = np.zeros((1, n_edges), dtype=np.float64)

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
                delta_p, omega_p = self.rollout(
                    static,
                    time_grid[:steps],
                    {
                        "Pm": exogenous["Pm"][:steps],
                        "interpolation": exogenous["interpolation"],
                    },
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
            rows.append(
                {
                    "source": source,
                    "standardized_loss": float(
                        np.mean((edge_error / scales["edge_angle"]) ** 2)
                        + np.mean((omega_error / scales["omega"]) ** 2)
                    ),
                    "raw_rmse": float(
                        np.sqrt(
                            np.mean(
                                np.concatenate(
                                    ((delta_p - delta_t).ravel(), omega_error.ravel())
                                )
                                ** 2
                            )
                        )
                    ),
                    "omega_rmse": float(np.sqrt(np.mean(omega_error**2))),
                    "edge_rmse": float(np.sqrt(np.mean(edge_error**2))),
                    "coi_drift": float(
                        max(
                            np.max(np.abs(delta_p @ mass)),
                            np.max(np.abs(omega_p @ mass)),
                        )
                    ),
                }
            )
        population = max(min(len(samples), context.max_cases), 1)
        if failures or not rows:
            return {
                "standardized_loss": float("inf"),
                "raw_mean_rmse": float("inf"),
                "raw_median_rmse": float("inf"),
                "omega_mean_rmse": float("inf"),
                "edge_angle_mean_rmse": float("inf"),
                "maximum_coi_drift": float("inf"),
                "failure_rate": failures / population,
                "source_standardized_loss": {},
            }
        sources = sorted({row["source"] for row in rows})
        source_losses = {
            source: float(
                np.mean(
                    [
                        row["standardized_loss"]
                        for row in rows
                        if row["source"] == source
                    ]
                )
            )
            for source in sources
        }
        return {
            "standardized_loss": float(np.mean(list(source_losses.values()))),
            "raw_mean_rmse": float(np.mean([row["raw_rmse"] for row in rows])),
            "raw_median_rmse": float(np.median([row["raw_rmse"] for row in rows])),
            "omega_mean_rmse": float(np.mean([row["omega_rmse"] for row in rows])),
            "edge_angle_mean_rmse": float(np.mean([row["edge_rmse"] for row in rows])),
            "maximum_coi_drift": float(np.max([row["coi_drift"] for row in rows])),
            "failure_rate": 0.0,
            "source_standardized_loss": source_losses,
        }

    @staticmethod
    def _serializable_candidate(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in row.items()
            if key not in {"theta", "gram", "response"}
        }

    @staticmethod
    def _support_from_theta(
        theta: np.ndarray, n_edges: int, n_modes: int
    ) -> dict[str, Any]:
        n_static = len(PortStructuredDiscoveryM0Model.STATIC_FEATURES) * n_edges
        static = np.asarray(theta[:n_static]).reshape(
            len(PortStructuredDiscoveryM0Model.STATIC_FEATURES), n_edges
        )
        memory = (
            np.asarray(theta[n_static:]).reshape(n_modes, n_edges)
            if n_modes
            else np.empty((0, n_edges))
        )
        static_support = static > 0
        memory_support = np.any(memory > 0, axis=0) if n_modes else np.zeros(n_edges, bool)
        edge_support = np.any(static_support, axis=0) | memory_support
        return {
            "edge_indices": np.flatnonzero(edge_support).tolist(),
            "active_edges": int(np.sum(edge_support)),
            "active_static_groups": int(np.sum(static_support)),
            "active_memory_edges": int(np.sum(memory_support)),
            "feature_names": [
                name
                for name, active in zip(
                    PortStructuredDiscoveryM0Model.STATIC_FEATURES,
                    np.any(static_support, axis=1),
                    strict=True,
                )
                if active
            ],
        }

    @staticmethod
    def _jaccard(left: list[int], right: list[int]) -> float:
        a, b = set(left), set(right)
        return float(len(a & b) / len(a | b)) if a or b else 1.0

    def _conservative_storage_audit(
        self, samples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        _, _, static_coefficients, memory_coefficients = self._require_fitted()
        nodes, weights = np.polynomial.legendre.leggauss(16)
        minimum_potential = float("inf")
        maximum_chain_error = 0.0
        minimum_cosine = float("inf")
        minimum_stiffness = float("inf")
        checked_states = 0
        for sample in samples:
            static = sample["model_inputs"]["static"]
            incidence = np.asarray(static["incidence"], dtype=np.float64)
            equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64)
            eta_eq = incidence.T @ equilibrium
            minimum_cosine = min(minimum_cosine, float(np.min(np.cos(eta_eq))))
            anchor = np.asarray(self.backbone.coupling, dtype=np.float64)
            stiffness = (anchor + static_coefficients[0]) * np.cos(eta_eq)
            laplacian = incidence @ np.diag(stiffness) @ incidence.T
            eigenvalues = np.linalg.eigvalsh(0.5 * (laplacian + laplacian.T))
            if len(eigenvalues) > 1:
                minimum_stiffness = min(minimum_stiffness, float(eigenvalues[1]))
            target = sample["training_context"]["targets"]
            delta_path = np.asarray(target["delta"], dtype=np.float64)
            stride = max(1, len(delta_path) // 20)
            for delta in delta_path[::stride]:
                eta = incidence.T @ delta
                midpoint = 0.5 * (eta + eta_eq)
                halfwidth = 0.5 * (eta - eta_eq)
                quadrature_eta = midpoint[:, None] + halfwidth[:, None] * nodes[None, :]
                u = np.sin(quadrature_eta) - np.sin(eta_eq)[:, None]
                integrands = np.stack((u, u * np.abs(u), u**3), axis=0)
                primitives = halfwidth[None, :] * np.einsum(
                    "fen,n->fe", integrands, weights, optimize=True
                )
                potential = float(np.sum(static_coefficients[:3] * primitives))
                minimum_potential = min(minimum_potential, potential)
                # Independent finite-difference chain-rule check on one edge.
                edge = checked_states % len(eta)
                epsilon = 1e-6
                edge_equilibrium = float(eta_eq[edge])
                edge_coefficients = static_coefficients[:3, edge].copy()

                def edge_potential(
                    value: float,
                    equilibrium: float = edge_equilibrium,
                    coefficients: np.ndarray = edge_coefficients,
                ) -> float:
                    mid = 0.5 * (value + equilibrium)
                    half = 0.5 * (value - equilibrium)
                    points = mid + half * nodes
                    du = np.sin(points) - np.sin(equilibrium)
                    primitive = half * np.array(
                        [
                            np.sum(weights * du),
                            np.sum(weights * du * np.abs(du)),
                            np.sum(weights * du**3),
                        ]
                    )
                    return float(coefficients @ primitive)

                numerical = (
                    edge_potential(float(eta[edge] + epsilon))
                    - edge_potential(float(eta[edge] - epsilon))
                ) / (2 * epsilon)
                delta_sin = np.sin(eta[edge]) - np.sin(eta_eq[edge])
                analytic = float(
                    static_coefficients[:3, edge]
                    @ np.array(
                        [delta_sin, delta_sin * abs(delta_sin), delta_sin**3]
                    )
                )
                maximum_chain_error = max(maximum_chain_error, abs(numerical - analytic))
                checked_states += 1
        if not np.isfinite(minimum_potential):
            minimum_potential = 0.0
        # Verify the passive memory identity independently of trajectory data.
        rng = np.random.default_rng(324)
        nu = rng.standard_normal(memory_coefficients.shape)
        z = rng.standard_normal(memory_coefficients.shape)
        mechanical_power = -np.sum(memory_coefficients * z * nu)
        storage_rate = np.sum(memory_coefficients * z * (nu - z))
        dissipation = np.sum(memory_coefficients * z**2)
        memory_identity_error = float(
            abs(mechanical_power + storage_rate + dissipation)
        )
        return {
            "minimum_conservative_potential_on_train_val_domain": minimum_potential,
            "maximum_conservative_chain_rule_error": maximum_chain_error,
            "minimum_equilibrium_edge_cosine": minimum_cosine,
            "minimum_quotient_stiffness_eigenvalue": minimum_stiffness,
            "memory_supply_identity_error": memory_identity_error,
            "checked_state_count": checked_states,
        }

    def fit(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-PSD requires nonempty train and validation sets")
        backbone = self._require_backbone()
        selected_train = train[: run_context.max_cases]
        selected_val = val[: run_context.max_cases]
        self.max_internal_step = run_context.max_internal_step
        backbone.max_internal_step = run_context.max_internal_step
        self.validation_scales = M0Model._training_scales(selected_train, len(selected_train))
        self.validation_scales_by_source = {
            source: M0Model._training_scales(group, len(group))
            for source, group in self._source_groups(selected_train).items()
        }
        prepared = self._prepare_weak_paths(
            selected_train,
            horizon_steps=run_context.horizon_steps,
            max_fit_steps=run_context.fit_max_steps,
        )
        train_candidates: list[dict[str, Any]] = []
        system_rows: list[dict[str, Any]] = []
        stats_cache: dict[tuple[float, ...], dict[str, Any]] = {}
        for tau_row in self.tau_mode_candidates:
            stats = self._build_normal_equations(
                prepared, tau_row, windows=run_context.integral_windows
            )
            stats_cache[tau_row] = stats
            system_rows.append(
                {
                    "fit_taus": list(tau_row),
                    **{
                        key: value
                        for key, value in stats.items()
                        if key not in {"gram", "response", "target_energy"}
                    },
                }
            )
            for ridge in self.ridge_values:
                for threshold in self.group_thresholds:
                    theta, solver = self._group_prune_and_refit(
                        stats, ridge, threshold
                    )
                    train_score = solver["weak_residual_rmse"] * (
                        1.0 + self.complexity_penalty * solver["active_groups"]
                    )
                    train_candidates.append(
                        {
                            "candidate": "port_structured_closure",
                            "fit_taus": list(tau_row),
                            "ridge": float(ridge),
                            "relative_group_threshold": float(threshold),
                            "coefficient_scale": 1.0,
                            "theta": theta,
                            "n_edges": stats["n_edges"],
                            "train_score": float(train_score),
                            **solver,
                        }
                    )
        shortlisted: list[dict[str, Any]] = []
        for order in sorted({len(row["fit_taus"]) for row in train_candidates}):
            group = sorted(
                (row for row in train_candidates if len(row["fit_taus"]) == order),
                key=lambda row: (
                    row["train_score"],
                    row["active_groups"],
                    row["ridge"],
                    row["relative_group_threshold"],
                ),
            )
            shortlisted.extend(group[: self.train_shortlist_per_order])

        n_edges = int(system_rows[0]["n_edges"])
        zero_theta = np.zeros(len(self.STATIC_FEATURES) * n_edges)
        candidate_specs: list[dict[str, Any]] = [
            {
                "candidate": "zero_closure",
                "fit_taus": [],
                "ridge": None,
                "relative_group_threshold": None,
                "coefficient_scale": 0.0,
                "theta": zero_theta,
                "n_edges": n_edges,
                "active_groups": 0,
                "active_edges": 0,
                "active_coefficients": 0,
                "weak_residual_rmse": None,
            }
        ]
        for row in shortlisted:
            for scale in self.coefficient_scales:
                scaled = np.asarray(row["theta"]) * scale
                candidate_specs.append(
                    {
                        **row,
                        "coefficient_scale": float(scale),
                        "theta": scaled,
                        "active_coefficients": int(np.sum(scaled > 0)),
                    }
                )

        short_context = replace(
            run_context,
            horizon_steps=min(run_context.horizon_steps, self.rollout_short_horizon),
        )
        short_candidates: list[dict[str, Any]] = []
        for row in candidate_specs:
            self._set_candidate(row)
            summary = self._validation_summary(selected_val, short_context)
            short_candidates.append(
                {**row, **{f"short_{key}": value for key, value in summary.items()}}
            )
        fallback_short = short_candidates[0]
        nonzero_shortlist = sorted(
            short_candidates[1:],
            key=lambda row: (
                row["short_standardized_loss"],
                row["short_raw_mean_rmse"],
                row["active_groups"],
                len(row["fit_taus"]),
            ),
        )[: self.rollout_long_shortlist]
        validation_candidates: list[dict[str, Any]] = []
        for row in [fallback_short, *nonzero_shortlist]:
            self._set_candidate(row)
            validation_candidates.append(
                {**row, **self._validation_summary(selected_val, run_context)}
            )
        fallback = validation_candidates[0]
        nonzero = [
            row
            for row in validation_candidates[1:]
            if row["failure_rate"] == 0.0
            and np.isfinite(row["standardized_loss"])
            and all(
                row["source_standardized_loss"].get(source, float("inf"))
                <= baseline * (1.0 + self.maximum_source_loss_degradation) + 1e-12
                for source, baseline in fallback["source_standardized_loss"].items()
            )
        ]
        best_nonzero = (
            min(
                nonzero,
                key=lambda row: (
                    row["standardized_loss"],
                    row["raw_mean_rmse"],
                    row["active_groups"],
                    len(row["fit_taus"]),
                ),
            )
            if nonzero
            else None
        )
        improvement = (
            0.0
            if best_nonzero is None
            else (
                fallback["standardized_loss"] - best_nonzero["standardized_loss"]
            )
            / max(fallback["standardized_loss"], np.finfo(float).eps)
        )
        promoted = bool(
            best_nonzero is not None
            and improvement >= self.minimum_validation_improvement
            and best_nonzero["raw_mean_rmse"] <= fallback["raw_mean_rmse"]
        )
        selected = best_nonzero if promoted else fallback
        self._set_candidate(selected)

        selected_fit_taus = tuple(float(value) for value in selected["fit_taus"])
        support = self._support_from_theta(
            np.asarray(selected["theta"]), n_edges, len(selected_fit_taus)
        )
        source_supports: dict[str, Any] = {}
        if promoted:
            for source in sorted({item["source"] for item in prepared}):
                source_prepared = [item for item in prepared if item["source"] == source]
                source_stats = self._build_normal_equations(
                    source_prepared,
                    selected_fit_taus,
                    windows=run_context.integral_windows,
                )
                source_theta, _ = self._group_prune_and_refit(
                    source_stats,
                    float(selected["ridge"]),
                    float(selected["relative_group_threshold"]),
                )
                source_supports[source] = self._support_from_theta(
                    source_theta, n_edges, len(selected_fit_taus)
                )
        source_names = sorted(source_supports)
        cross_source_jaccard = (
            self._jaccard(
                source_supports[source_names[0]]["edge_indices"],
                source_supports[source_names[1]]["edge_indices"],
            )
            if len(source_names) == 2
            else None
        )
        storage = self._conservative_storage_audit(selected_train + selected_val)
        self.diagnostics = {
            "optimizer": "nonnegative_port_structured_integral_ridge_plus_validation_rollout",
            "epochs": 0,
            "structure_discovery": True,
            "fixed_priors": [
                "swing_balance",
                "COI_quotient",
                "incidence_internal_force",
                "separable_conservative_edge_potential",
                "nonnegative_static_edge_dissipation",
                "passive_first_order_speed_memory",
                "causal_Pm_port",
            ],
            "discovered_quantities": [
                "effective_edge_support",
                "conservative_feature_support",
                "dissipative_feature_support",
                "passive_memory_support",
                "memory_order",
                "time_constants",
                "nonnegative_closure_coefficients",
            ],
            "reported_role": "port_structured_residual_closure_not_full_Lyapunov_certificate",
            "physical_line_recovery_claim": False,
            "machine_parameter_recovery_claim": False,
            "feature_names": list(self.STATIC_FEATURES),
            "conservative_features": list(self.CONSERVATIVE_FEATURES),
            "dissipative_features": list(self.DISSIPATIVE_FEATURES),
            "tau_mode_candidates": [list(row) for row in self.tau_mode_candidates],
            "selected_fit_taus": list(selected_fit_taus),
            "selected_runtime_taus": self.taus.tolist(),
            "selected_memory_order": int(
                np.sum(np.any(self.memory_coefficients > 0, axis=1))
            ),
            "selected_support": support,
            "source_supports": source_supports,
            "cross_source_edge_support_jaccard": cross_source_jaccard,
            "selected_validation": self._serializable_candidate(selected),
            "fallback_validation": self._serializable_candidate(fallback),
            "best_nonzero_before_gate": (
                None
                if best_nonzero is None
                else self._serializable_candidate(best_nonzero)
            ),
            "validation_improvement_vs_zero_closure": float(improvement),
            "minimum_validation_improvement": self.minimum_validation_improvement,
            "maximum_source_loss_degradation": self.maximum_source_loss_degradation,
            "promoted": promoted,
            "selected_is_zero_closure": not promoted,
            "candidate_table": [
                self._serializable_candidate(row) for row in short_candidates
            ],
            "long_horizon_candidate_table": [
                self._serializable_candidate(row) for row in validation_candidates
            ],
            "weak_systems": system_rows,
            "storage_audit": storage,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "initial_state_policy": "causal_quasi_steady_edge_speed_at_clearing",
            "zero_closure_backbone_fallback_included": True,
            "labels_used_for_fit_or_selection": False,
            "test_ids_read": [],
        }
        return FitResult(
            method=self.method_name,
            status="completed",
            train_case_ids=tuple(sample["event_id"] for sample in selected_train),
            val_case_ids=tuple(sample["event_id"] for sample in selected_val),
            best_validation_loss=float(selected["standardized_loss"]),
            details=self.diagnostics,
        )

    def convergence_audit(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> dict[str, float | bool]:
        trajectories = [
            self.rollout_augmented(
                initial_state,
                time_grid,
                exogenous,
                max_step=self.max_internal_step / divisor,
            )
            for divisor in (1, 2, 4)
        ]
        errors = []
        for coarse, fine in pairwise(trajectories):
            errors.append(
                float(
                    np.sqrt(
                        sum(np.mean((left - right) ** 2) for left, right in zip(coarse, fine))
                    )
                )
            )
        return {
            "dt_vs_dt2": errors[0],
            "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0] / max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0] + 1e-12),
        }

    def save(self, path: str | Path) -> None:
        backbone, taus, static, memory = self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backbone_path = path.with_name(path.stem + "_backbone.npz")
        backbone.save(backbone_path)
        np.savez_compressed(
            path,
            static_coefficients=static,
            taus=taus,
            memory_coefficients=memory,
            max_internal_step=self.max_internal_step,
            backbone_file=backbone_path.name,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {"method": self.method_name, "diagnostics": self.diagnostics}, indent=2
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> PortStructuredDiscoveryM0Model:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            backbone = TransientEmfM0Model.load(
                path.with_name(str(data["backbone_file"]))
            )
            model = cls(
                backbone=backbone,
                static_coefficients=data["static_coefficients"],
                taus=data["taus"],
                memory_coefficients=data["memory_coefficients"],
                max_internal_step=float(data["max_internal_step"]),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get(
                "diagnostics", {}
            )
        return model

    def physics_audit(self) -> dict[str, Any]:
        try:
            backbone, taus, static, memory = self._require_fitted()
        except (RuntimeError, ValueError):
            return {"status": "not_fitted"}
        storage = self.diagnostics.get("storage_audit", {})
        finite = bool(np.isfinite(static).all() and np.isfinite(memory).all())
        nonnegative = bool(np.all(static >= 0) and np.all(memory >= 0))
        stable_memory = bool(np.all(taus > 0) and np.isfinite(taus).all())
        local_stiffness = float(
            storage.get("minimum_quotient_stiffness_eigenvalue", float("nan"))
        )
        minimum_potential = float(
            storage.get(
                "minimum_conservative_potential_on_train_val_domain", float("nan")
            )
        )
        chain_error = float(
            storage.get("maximum_conservative_chain_rule_error", float("inf"))
        )
        identity_error = float(storage.get("memory_supply_identity_error", float("inf")))
        certificate = bool(
            finite
            and nonnegative
            and stable_memory
            and backbone.physics_audit().get("status") == "pass"
            and np.isfinite(local_stiffness)
            and local_stiffness > 0
            and np.isfinite(minimum_potential)
            and minimum_potential >= -1e-10
            and chain_error < 1e-6
            and identity_error < 1e-10
        )
        return {
            "status": "pass" if certificate else "fail",
            "method": self.method_name,
            "reported_role": "port-structured residual closure",
            "full_Lyapunov_certificate_claim": False,
            "physical_line_recovery_claim": False,
            "machine_parameter_recovery_claim": False,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "angle_shift_invariant_by_construction": True,
            "orientation_invariant_by_construction": True,
            "internal_force_balance_by_incidence": True,
            "conservative_edge_forces_are_potential_gradients": chain_error < 1e-6,
            "nonnegative_dissipative_coefficients": nonnegative,
            "passive_speed_memory_identity": identity_error < 1e-10,
            "stable_memory_poles": stable_memory,
            "minimum_quotient_stiffness_eigenvalue": local_stiffness,
            "minimum_conservative_potential_on_domain": minimum_potential,
            "maximum_conservative_chain_rule_error": chain_error,
            "memory_supply_identity_error": identity_error,
            "selected_active_edges": int(
                self.diagnostics.get("selected_support", {}).get("active_edges", 0)
            ),
            "selected_time_constants": taus.tolist(),
            "finite_coefficients": finite,
            "backbone_physics_status": backbone.physics_audit().get("status", "unknown"),
        }


class CausalPmPortStructuredM0Model(CausalPmGraphModalM0Model):
    """M0-PSD with the same train-discovered causal mechanical-power ODE as GMD."""

    method_name = "M0-PSD-PM"

    def __init__(
        self,
        backbone: PortStructuredDiscoveryM0Model | None = None,
        pm_dynamics: GraphModalPmDynamics | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(backbone=backbone, pm_dynamics=pm_dynamics, **kwargs)

    def _require_fitted(self):
        if self.backbone is None or not isinstance(
            self.backbone, PortStructuredDiscoveryM0Model
        ):
            raise RuntimeError("M0-PSD-PM requires a fitted M0-PSD backbone")
        return self.backbone._require_fitted(), self.pm_dynamics.require_fitted()

    def rollout_augmented(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray] | None = None,
        *,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (electrical, _, _, _), (pm_basis, _, _) = self._require_fitted()
        t = np.asarray(time_grid, dtype=np.float64)
        if len(t) < 1:
            raise ValueError("rollout requires a nonempty time grid")
        step_limit = self.max_internal_step if max_step is None else float(max_step)
        if step_limit <= 0:
            raise ValueError("maximum integration step must be positive")
        mass = np.asarray(initial_state["mass"], dtype=np.float64).reshape(-1)
        equilibrium_pm = np.asarray(initial_state["Pm_equilibrium"], dtype=np.float64)
        delta = coi_project(np.asarray(initial_state["delta0"]).reshape(-1), mass)
        omega = coi_project(np.asarray(initial_state["omega0"]).reshape(-1), mass)
        q = electrical.initial_emf(delta, initial_state)
        z = self.backbone.initial_memory(delta, omega, initial_state)
        modal_p = (
            np.asarray(initial_state["Pm_clear"], dtype=np.float64).reshape(-1)
            - equilibrium_pm
        ) @ pm_basis
        topology = electrical._graph_topology(initial_state)
        outputs = (
            [delta.copy()],
            [omega.copy()],
            [q.copy()],
            [z.copy()],
            [pm_basis @ modal_p + equilibrium_pm],
        )

        def field(di, wi, qi, zi, ai):
            absolute_pm = pm_basis @ ai + equilibrium_pm
            dd, dw, dq, dz = self.backbone.vector_field_augmented(
                di, wi, qi, zi, absolute_pm, initial_state, topology
            )
            da = self.pm_dynamics.modal_derivative(ai, wi, mass)
            return dd, dw, dq, dz, da

        for index in range(len(t) - 1):
            interval = float(t[index + 1] - t[index])
            if interval <= 0:
                raise ValueError("rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval / step_limit)))
            step = interval / substeps
            for _ in range(substeps):
                k1 = field(delta, omega, q, z, modal_p)
                k2 = field(
                    delta + 0.5 * step * k1[0],
                    omega + 0.5 * step * k1[1],
                    q + 0.5 * step * k1[2],
                    z + 0.5 * step * k1[3],
                    modal_p + 0.5 * step * k1[4],
                )
                k3 = field(
                    delta + 0.5 * step * k2[0],
                    omega + 0.5 * step * k2[1],
                    q + 0.5 * step * k2[2],
                    z + 0.5 * step * k2[3],
                    modal_p + 0.5 * step * k2[4],
                )
                k4 = field(
                    delta + step * k3[0],
                    omega + step * k3[1],
                    q + step * k3[2],
                    z + step * k3[3],
                    modal_p + step * k3[4],
                )
                delta = coi_project(
                    delta + step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
                    mass,
                )
                omega = coi_project(
                    omega + step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
                    mass,
                )
                q = q + step * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
                z = z + step * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
                modal_p = modal_p + step * (
                    k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]
                ) / 6
                if not all(
                    np.isfinite(value).all()
                    for value in (delta, omega, q, z, modal_p)
                ):
                    raise FloatingPointError("nonfinite M0-PSD-PM state")
            for output, value in zip(
                outputs,
                (delta, omega, q, z, pm_basis @ modal_p + equilibrium_pm),
                strict=True,
            ):
                output.append(value.copy())
        return tuple(np.stack(value) for value in outputs)  # type: ignore[return-value]

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backbone_path = path.with_name(path.stem + "_backbone.npz")
        self.backbone.save(backbone_path)
        basis, state, inputs = self.pm_dynamics.require_fitted()
        np.savez_compressed(
            path,
            basis=basis,
            state_matrix=state,
            input_matrix=inputs,
            max_internal_step=self.max_internal_step,
            maximum_effective_time_constant=self.maximum_effective_time_constant,
            backbone_file=backbone_path.name,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {"method": self.method_name, "diagnostics": self.diagnostics}, indent=2
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CausalPmPortStructuredM0Model:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            backbone = PortStructuredDiscoveryM0Model.load(
                path.with_name(str(data["backbone_file"]))
            )
            model = cls(
                backbone=backbone,
                pm_dynamics=GraphModalPmDynamics(
                    basis=data["basis"],
                    state_matrix=data["state_matrix"],
                    input_matrix=data["input_matrix"],
                ),
                max_internal_step=float(data["max_internal_step"]),
                maximum_effective_time_constant=float(
                    data["maximum_effective_time_constant"]
                ),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get(
                "diagnostics", {}
            )
            model.pm_dynamics.diagnostics = dict(model.diagnostics)
        return model

    def convergence_audit(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float | bool]:
        trajectories = [
            self.rollout_augmented(
                initial_state,
                time_grid,
                exogenous or {},
                max_step=self.max_internal_step / divisor,
            )
            for divisor in (1, 2, 4)
        ]
        errors = []
        for coarse, fine in pairwise(trajectories):
            errors.append(
                float(
                    np.sqrt(
                        sum(np.mean((left - right) ** 2) for left, right in zip(coarse, fine))
                    )
                )
            )
        return {
            "dt_vs_dt2": errors[0],
            "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0] / max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0] + 1e-12),
        }

    def physics_audit(self) -> dict[str, Any]:
        try:
            (_, _, _, _), (basis, state, inputs) = self._require_fitted()
        except (RuntimeError, ValueError):
            return {"status": "not_fitted"}
        maximum_sum = float(np.max(np.abs(np.sum(basis, axis=0))))
        orthogonality = float(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))))
        spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(state))))
        selected = self.diagnostics.get("selected_candidate", {})
        exact_zoh = bool(
            selected.get("candidate") == "Pm_clear_ZOH"
            and np.array_equal(state, np.zeros_like(state))
            and np.array_equal(inputs, np.zeros_like(inputs))
        )
        maximum_time_constant = (
            float("inf")
            if exact_zoh
            else 1.0 / max(abs(spectral_abscissa), np.finfo(float).eps)
        )
        finite = bool(np.isfinite(state).all() and np.isfinite(inputs).all())
        backbone_audit = self.backbone.physics_audit()
        hurwitz_and_bounded = bool(
            spectral_abscissa < 0
            and maximum_time_constant
            <= self.maximum_effective_time_constant + 1e-12
        )
        passed = bool(
            finite
            and maximum_sum < 1e-10
            and orthogonality < 1e-10
            and (exact_zoh or hurwitz_and_bounded)
            and backbone_audit.get("status") == "pass"
        )
        return {
            "status": "pass" if passed else "fail",
            "method": self.method_name,
            "causal_inputs_only": True,
            "future_observed_Pm_consumed_at_deployment": False,
            "Pm_clear_exact_initialization": True,
            "selected_is_exact_ZOH_fallback": exact_zoh,
            "stable_or_exact_ZOH": exact_zoh or hurwitz_and_bounded,
            "zero_total_power_invariant": maximum_sum < 1e-10,
            "maximum_basis_sum": maximum_sum,
            "basis_orthogonality_error": orthogonality,
            "Hurwitz_state_matrix": spectral_abscissa < 0,
            "spectral_abscissa": spectral_abscissa,
            "maximum_effective_time_constant": maximum_time_constant,
            "time_constant_cap": self.maximum_effective_time_constant,
            "time_constant_bound_pass": (
                exact_zoh or hurwitz_and_bounded
            ),
            "finite_coefficients": finite,
            "backbone_method": self.backbone.method_name,
            "backbone_physics_status": backbone_audit.get("status", "unknown"),
            "full_Lyapunov_certificate_claim": False,
            "physical_line_recovery_claim": False,
            "machine_level_governor_recovery_claim": False,
        }
