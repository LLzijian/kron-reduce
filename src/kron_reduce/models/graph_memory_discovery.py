from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .m0 import M0Model, _trapz
from .transient_emf_m0 import TransientEmfM0Model


class GraphMemoryDiscoveryM0Model(DynamicsModel):
    """Discover a stable graph-memory closure around an M0-3EG backbone.

    The model deliberately fixes only the swing balance, graph incidence and a
    stable first-order realization.  Edge support, nonlinear driving terms,
    memory order and time constants are selected from data::

        delta_dot = omega
        m omega_dot = f_3EG(delta, omega, q, Pm) - B r
        r_e = sum_(j,f) c[j,f,e] z[j,f,e]
        tau_j z_dot[j,f,e] = -z[j,f,e] + phi_f(eta_e, nu_e)

    ``phi`` contains orientation-odd, equilibrium-centred graph features.  A
    group sparse weak solve selects an edge/feature as a whole across memory
    modes.  The exact M0-3EG backbone (zero closure) is always a candidate.

    The hidden filters are initialized at their clearing-state quasi-steady
    values, so deployment reads only the registered clearing state.  Training
    uses delta/omega/Pm targets and never measured Pe, voltage or future-state
    initialization.
    """

    method_name = "M0-GSD"
    ALLOWED_FEATURES = (
        "delta_sin",
        "edge_speed",
        "delta_sin_abs",
        "delta_sin_cubic",
        "speed_stress",
    )
    DEFAULT_TAU_MODE_CANDIDATES = (
        (0.02,),
        (0.05,),
        (0.1,),
        (0.2,),
        (0.5,),
        (1.0,),
        (0.05, 0.2),
        (0.1, 0.5),
        (0.05, 0.2, 1.0),
    )

    def __init__(
        self,
        backbone: TransientEmfM0Model | None = None,
        *,
        taus: np.ndarray | tuple[float, ...] | None = None,
        coefficients: np.ndarray | None = None,
        feature_names: tuple[str, ...] = ALLOWED_FEATURES,
        tau_mode_candidates: tuple[tuple[float, ...], ...] | None = None,
        group_thresholds: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1),
        coefficient_scales: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
        ridge_values: tuple[float, ...] = (1e-6, 1e-4, 1e-2),
        train_shortlist_per_order: int = 2,
        rollout_short_horizon: int = 20,
        rollout_long_shortlist: int = 3,
        complexity_penalty: float = 1e-4,
        max_internal_step: float = 0.01,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.backbone = backbone
        self.taus = None if taus is None else np.asarray(taus, dtype=np.float64)
        self.coefficients = (
            None if coefficients is None
            else np.asarray(coefficients, dtype=np.float64)
        )
        self.feature_names = tuple(feature_names)
        if not self.feature_names or any(
            name not in self.ALLOWED_FEATURES for name in self.feature_names
        ):
            raise ValueError("unknown or empty graph-memory feature library")
        self.tau_mode_candidates = tuple(
            tuple(float(value) for value in row)
            for row in (tau_mode_candidates or self.DEFAULT_TAU_MODE_CANDIDATES)
        )
        if any(
            not row or any(not np.isfinite(value) or value <= 0 for value in row)
            or tuple(sorted(set(row))) != row
            for row in self.tau_mode_candidates
        ):
            raise ValueError("memory time-constant candidates must be positive and sorted")
        self.group_thresholds = tuple(float(value) for value in group_thresholds)
        self.coefficient_scales = tuple(float(value) for value in coefficient_scales)
        self.ridge_values = tuple(float(value) for value in ridge_values)
        if any(value < 0 or not np.isfinite(value) for value in self.group_thresholds):
            raise ValueError("group thresholds must be finite and nonnegative")
        if any(value <= 0 or value > 1 or not np.isfinite(value) for value in self.coefficient_scales):
            raise ValueError("coefficient scales must lie in (0, 1]")
        if any(value < 0 or not np.isfinite(value) for value in self.ridge_values):
            raise ValueError("ridge values must be finite and nonnegative")
        self.train_shortlist_per_order = int(train_shortlist_per_order)
        if self.train_shortlist_per_order < 1:
            raise ValueError("train shortlist must be positive")
        self.rollout_short_horizon = int(rollout_short_horizon)
        self.rollout_long_shortlist = int(rollout_long_shortlist)
        if self.rollout_short_horizon < 2 or self.rollout_long_shortlist < 1:
            raise ValueError("rollout screening horizon and shortlist must be positive")
        self.complexity_penalty = float(complexity_penalty)
        self.max_internal_step = float(max_internal_step)
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
        self.validation_scales_by_source: dict[str, dict[str, float]] = {}
        self.diagnostics = diagnostics or {}

    @property
    def damping(self) -> np.ndarray | None:
        return None if self.backbone is None else self.backbone.damping

    @property
    def coupling(self) -> np.ndarray | None:
        return None if self.backbone is None else self.backbone.coupling

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

    def _require_backbone(self) -> TransientEmfM0Model:
        if self.backbone is None:
            raise RuntimeError("M0-GSD requires a fitted M0-3EG backbone")
        self.backbone._require_fitted()
        return self.backbone

    def _require_fitted(self) -> tuple[TransientEmfM0Model, np.ndarray, np.ndarray]:
        backbone = self._require_backbone()
        if self.taus is None or self.coefficients is None:
            raise RuntimeError("M0-GSD is not fitted")
        incidence = np.asarray(backbone.diagnostics.get("incidence_shape", []))
        if self.coefficients.ndim != 3:
            raise ValueError("graph-memory coefficients must have shape (mode, feature, edge)")
        if self.coefficients.shape[:2] != (len(self.taus), len(self.feature_names)):
            raise ValueError("graph-memory coefficient shape does not match modes/features")
        del incidence
        return backbone, self.taus, self.coefficients

    def _edge_features(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        static: dict[str, np.ndarray],
    ) -> np.ndarray:
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        equilibrium = np.asarray(
            static["equilibrium_delta"], dtype=np.float64
        ).reshape(-1)
        eta = incidence.T @ np.asarray(delta, dtype=np.float64).reshape(-1)
        eta_eq = incidence.T @ equilibrium
        delta_sin = np.sin(eta) - np.sin(eta_eq)
        edge_speed = incidence.T @ np.asarray(omega, dtype=np.float64).reshape(-1)
        values = {
            "delta_sin": delta_sin,
            "edge_speed": edge_speed,
            "delta_sin_abs": delta_sin * np.abs(delta_sin),
            "delta_sin_cubic": delta_sin**3,
            "speed_stress": edge_speed * np.abs(delta_sin),
        }
        return np.stack([values[name] for name in self.feature_names], axis=0)

    def initial_memory(
        self, delta: np.ndarray, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.taus is None:
            raise RuntimeError("memory modes are not selected")
        features = self._edge_features(delta, omega, static)
        return np.repeat(features[np.newaxis, :, :], len(self.taus), axis=0)

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
        backbone, taus, coefficients = self._require_fitted()
        dd, dw, dq = backbone.vector_field_augmented(
            delta, omega, q, pm, static, topology
        )
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        if z.shape != coefficients.shape:
            raise ValueError("memory state shape does not match selected structure")
        edge_flow = np.sum(coefficients * z, axis=(0, 1))
        correction_force = coi_project_power(-incidence @ edge_flow, mass)
        features = self._edge_features(delta, omega, static)
        dz = (features[np.newaxis, :, :] - z) / taus[:, np.newaxis, np.newaxis]
        return dd, dw + correction_force / mass, dq, dz

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
        backbone, _, coefficients = self._require_fitted()
        t = np.asarray(time_grid, dtype=np.float64)
        pm = np.asarray(exogenous["Pm"], dtype=np.float64)
        if len(t) != len(pm) or len(t) < 1:
            raise ValueError("time and Pm schedules must have the same positive length")
        step = self.max_internal_step if max_step is None else float(max_step)
        if step <= 0:
            raise ValueError("max_step must be positive")
        mode = self._input_mode(pm, exogenous.get("interpolation", "auto"))
        mass = np.asarray(initial_state["mass"], dtype=np.float64).reshape(-1)
        delta = coi_project(np.asarray(initial_state["delta0"]).reshape(-1), mass)
        omega = coi_project(np.asarray(initial_state["omega0"]).reshape(-1), mass)
        q = backbone.initial_emf(delta, initial_state)
        z = self.initial_memory(delta, omega, initial_state)
        if z.shape != coefficients.shape:
            raise ValueError("selected graph-memory edge count does not match input graph")
        topology = backbone._graph_topology(initial_state)
        output_delta, output_omega = [delta.copy()], [omega.copy()]
        output_q, output_z = [q.copy()], [z.copy()]

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
                k1 = self.vector_field_augmented(
                    delta, omega, q, z, input_at(index, f0), initial_state, topology
                )
                k2 = self.vector_field_augmented(
                    delta + 0.5*h*k1[0], omega + 0.5*h*k1[1],
                    q + 0.5*h*k1[2], z + 0.5*h*k1[3],
                    input_at(index, fh), initial_state, topology,
                )
                k3 = self.vector_field_augmented(
                    delta + 0.5*h*k2[0], omega + 0.5*h*k2[1],
                    q + 0.5*h*k2[2], z + 0.5*h*k2[3],
                    input_at(index, fh), initial_state, topology,
                )
                k4 = self.vector_field_augmented(
                    delta + h*k3[0], omega + h*k3[1], q + h*k3[2], z + h*k3[3],
                    input_at(index, f1), initial_state, topology,
                )
                delta = coi_project(
                    delta + h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0, mass
                )
                omega = coi_project(
                    omega + h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0, mass
                )
                q = q + h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0
                z = z + h*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6.0
                if (
                    not np.isfinite(delta).all() or not np.isfinite(omega).all()
                    or not np.isfinite(q).all() or not np.isfinite(z).all()
                    or np.min(q) <= backbone.q_floor or np.max(q) >= backbone.q_ceiling
                ):
                    raise FloatingPointError(
                        f"M0-GSD left the audited state domain at step {index + 1}"
                    )
            output_delta.append(delta.copy())
            output_omega.append(omega.copy())
            output_q.append(q.copy())
            output_z.append(z.copy())
        return (
            np.stack(output_delta), np.stack(output_omega),
            np.stack(output_q), np.stack(output_z),
        )

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

    def teacher_forced_memory(
        self,
        delta_path: np.ndarray,
        omega_path: np.ndarray,
        time_grid: np.ndarray,
        static: dict[str, np.ndarray],
        taus: np.ndarray,
        *,
        max_step: float | None = None,
    ) -> np.ndarray:
        delta_path = np.asarray(delta_path, dtype=np.float64)
        omega_path = np.asarray(omega_path, dtype=np.float64)
        t = np.asarray(time_grid, dtype=np.float64)
        taus = np.asarray(taus, dtype=np.float64)
        if (
            delta_path.shape != omega_path.shape or delta_path.ndim != 2
            or len(delta_path) != len(t) or len(t) < 1
        ):
            raise ValueError("teacher-forced state paths must be shape aligned")
        step = self.max_internal_step if max_step is None else float(max_step)
        features = self._edge_features(delta_path[0], omega_path[0], static)
        z = np.repeat(features[np.newaxis, :, :], len(taus), axis=0)
        output = [z.copy()]

        for index in range(len(t)-1):
            interval = float(t[index+1]-t[index])
            if interval <= 0:
                raise ValueError("teacher-forced time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval/step)))
            h = interval/substeps

            def feature(fraction: float) -> np.ndarray:
                delta = delta_path[index] + fraction*(
                    delta_path[index+1]-delta_path[index]
                )
                omega = omega_path[index] + fraction*(
                    omega_path[index+1]-omega_path[index]
                )
                return self._edge_features(delta, omega, static)

            for substep in range(substeps):
                f0 = substep/substeps
                fh = (substep+0.5)/substeps
                f1 = (substep+1.0)/substeps
                k1 = (feature(f0)[np.newaxis, :, :]-z)/taus[:, None, None]
                k2 = (
                    feature(fh)[np.newaxis, :, :]-(z+0.5*h*k1)
                )/taus[:, None, None]
                k3 = (
                    feature(fh)[np.newaxis, :, :]-(z+0.5*h*k2)
                )/taus[:, None, None]
                k4 = (
                    feature(f1)[np.newaxis, :, :]-(z+h*k3)
                )/taus[:, None, None]
                z = z+h*(k1+2*k2+2*k3+k4)/6.0
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
            rows.append(mass*acceleration)
        return np.stack(rows)

    def _prepare_weak_paths(
        self,
        samples: list[dict[str, Any]],
        *,
        horizon_steps: int,
        max_fit_steps: int,
    ) -> list[dict[str, Any]]:
        """Cache backbone quantities shared by every memory candidate."""
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
            feature_path = np.stack([
                self._edge_features(delta[index], omega[index], static)
                for index in range(steps)
            ])
            prepared.append({
                "event_id": sample["event_id"], "t": t, "delta": delta,
                "omega": omega, "static": static, "feature_path": feature_path,
                "baseline_force": self._baseline_force_path(
                    delta, omega, q, pm, static
                ),
            })
        if not prepared:
            raise ValueError("no graph-memory weak paths could be prepared")
        return prepared

    @staticmethod
    def _first_order_hold_memory(
        feature_path: np.ndarray,
        time_grid: np.ndarray,
        taus: np.ndarray,
    ) -> np.ndarray:
        """Exact stable-filter update for linearly interpolated edge features."""
        feature_path = np.asarray(feature_path, dtype=np.float64)
        t = np.asarray(time_grid, dtype=np.float64)
        taus = np.asarray(taus, dtype=np.float64)
        z = np.repeat(feature_path[[0]], len(taus), axis=0)
        output = [z.copy()]
        for index, dt in enumerate(np.diff(t)):
            if dt <= 0:
                raise ValueError("memory-filter time must be strictly increasing")
            x = dt/taus
            decay = np.exp(-x)
            b0 = (1.0-decay)/x-decay
            b1 = 1.0-(1.0-decay)/x
            z = (
                decay[:, None, None]*z
                + b0[:, None, None]*feature_path[index][None, :, :]
                + b1[:, None, None]*feature_path[index+1][None, :, :]
            )
            output.append(z.copy())
        return np.stack(output)

    def _build_weak_memory_system_from_cache(
        self,
        prepared: list[dict[str, Any]],
        taus: tuple[float, ...] | np.ndarray,
        *,
        windows: tuple[int, ...],
        equal_case_weight: bool,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        taus_array = np.asarray(taus, dtype=np.float64)
        rows, targets = [], []
        n_windows = 0
        for item in prepared:
            t = item["t"]
            omega = item["omega"]
            static = item["static"]
            incidence = np.asarray(static["incidence"], dtype=np.float64)
            mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
            z = self._first_order_hold_memory(item["feature_path"], t, taus_array)
            baseline_force = item["baseline_force"]
            design_path = -np.einsum("ne,tmfe->tnmfe", incidence, z).reshape(
                len(t), len(mass), -1
            )
            dt = np.diff(t)
            design_prefix = np.concatenate((
                np.zeros((1, len(mass), design_path.shape[-1])),
                np.cumsum(
                    0.5*(design_path[:-1]+design_path[1:])*dt[:, None, None],
                    axis=0,
                ),
            ), axis=0)
            force_prefix = np.concatenate((
                np.zeros((1, len(mass))),
                np.cumsum(
                    0.5*(baseline_force[:-1]+baseline_force[1:])*dt[:, None],
                    axis=0,
                ),
            ), axis=0)
            case_rows, case_targets = [], []
            for width in windows:
                if width <= 0 or width >= len(t):
                    continue
                starts = np.arange(0, len(t)-width)
                stops = starts+width
                design_block = design_prefix[stops]-design_prefix[starts]
                actual_impulse = mass*(omega[stops]-omega[starts])
                baseline_impulse = force_prefix[stops]-force_prefix[starts]
                case_rows.append(design_block.reshape(-1, design_path.shape[-1]))
                case_targets.append((actual_impulse-baseline_impulse).reshape(-1))
                n_windows += len(starts)
            if not case_rows:
                continue
            matrix = np.vstack(case_rows)
            vector = np.concatenate(case_targets)
            if equal_case_weight:
                weight = 1.0/np.sqrt(max(len(vector), 1))
                matrix, vector = weight*matrix, weight*vector
            rows.append(matrix)
            targets.append(vector)
        if not rows:
            raise ValueError("no graph-memory weak windows could be constructed")
        matrix = np.vstack(rows)
        vector = np.concatenate(targets)
        return matrix, vector, {
            "n_scalar_equations": int(len(vector)),
            "n_parameters": int(matrix.shape[1]),
            "n_integration_windows": int(n_windows),
            "n_memory_modes": int(len(taus_array)),
            "n_features": int(len(self.feature_names)),
            "n_edges": int(matrix.shape[1]//(len(taus_array)*len(self.feature_names))),
        }

    def build_weak_memory_system(
        self,
        samples: list[dict[str, Any]],
        taus: tuple[float, ...] | np.ndarray,
        *,
        windows: tuple[int, ...] = (1, 5, 20),
        horizon_steps: int = 100,
        max_fit_steps: int = 0,
        equal_case_weight: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        prepared = self._prepare_weak_paths(
            samples, horizon_steps=horizon_steps, max_fit_steps=max_fit_steps
        )
        return self._build_weak_memory_system_from_cache(
            prepared, taus, windows=windows, equal_case_weight=equal_case_weight
        )

    @staticmethod
    def _scaled_ridge(
        matrix: np.ndarray, vector: np.ndarray, ridge: float
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        scale = np.sqrt(np.mean(matrix**2, axis=0))
        active = scale > np.finfo(float).eps
        scale = np.where(active, scale, 1.0)
        normalized = matrix/scale
        gram = normalized.T@normalized + ridge*np.eye(normalized.shape[1])
        beta = np.linalg.solve(gram, normalized.T@vector)
        beta[~active] = 0.0
        theta = beta/scale
        residual = matrix@theta-vector
        return theta, beta, {
            "matrix_condition_number": float(np.linalg.cond(normalized)),
            "effective_rank": int(np.linalg.matrix_rank(normalized)),
            "weak_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        }

    def _group_prune_and_refit(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        taus: tuple[float, ...],
        ridge: float,
        relative_threshold: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        theta, beta, diagnostics = self._scaled_ridge(matrix, vector, ridge)
        n_modes, n_features = len(taus), len(self.feature_names)
        n_edges = matrix.shape[1]//(n_modes*n_features)
        beta_group = beta.reshape(n_modes, n_features, n_edges)
        norms = np.sqrt(np.sum(beta_group**2, axis=0))
        cutoff = relative_threshold*float(np.max(norms)) if norms.size else 0.0
        active_groups = norms > cutoff if relative_threshold > 0 else norms > 0
        active_columns = np.repeat(active_groups[np.newaxis, :, :], n_modes, axis=0).reshape(-1)
        if not np.any(active_columns):
            return np.zeros(matrix.shape[1]), {
                **diagnostics, "active_groups": 0, "active_coefficients": 0,
                "relative_group_threshold": relative_threshold,
            }
        submatrix = matrix[:, active_columns]
        subtheta, _, subdiagnostics = self._scaled_ridge(submatrix, vector, ridge)
        result = np.zeros(matrix.shape[1])
        result[active_columns] = subtheta
        residual = matrix@result-vector
        return result, {
            **subdiagnostics,
            "weak_residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "active_groups": int(np.sum(active_groups)),
            "active_coefficients": int(np.sum(active_columns)),
            "relative_group_threshold": relative_threshold,
        }

    def _validation_summary(
        self, samples: list[dict[str, Any]], context: RunContext
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        failures = 0
        for sample in samples[:context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            static, time_grid, exogenous = sample_rollout_inputs(sample)
            try:
                delta_p, omega_p, _, _ = self.rollout_augmented(
                    static, time_grid[:steps],
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
            edge_error = (delta_p-delta_t)@incidence
            omega_error = omega_p-omega_t
            mass = np.asarray(static["mass"], dtype=np.float64)
            rows.append({
                "source": source,
                "standardized_loss": float(
                    np.mean((edge_error/scales["edge_angle"])**2)
                    + np.mean((omega_error/scales["omega"])**2)
                ),
                "raw_rmse": float(np.sqrt(np.mean(np.concatenate((
                    (delta_p-delta_t).ravel(), omega_error.ravel(),
                ))**2))),
                "omega_rmse": float(np.sqrt(np.mean(omega_error**2))),
                "edge_rmse": float(np.sqrt(np.mean(edge_error**2))),
                "coi_drift": float(max(
                    np.max(np.abs(delta_p@mass)), np.max(np.abs(omega_p@mass)),
                )),
            })
        population = max(min(len(samples), context.max_cases), 1)
        if failures or not rows:
            return {
                "standardized_loss": float("inf"), "raw_mean_rmse": float("inf"),
                "raw_median_rmse": float("inf"), "omega_mean_rmse": float("inf"),
                "edge_angle_mean_rmse": float("inf"), "maximum_coi_drift": float("inf"),
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
            "failure_rate": 0.0,
            "source_standardized_loss": source_losses,
        }

    def residual_hankel_diagnostic(
        self,
        samples: list[dict[str, Any]],
        *,
        horizon_steps: int = 100,
        block_rows: int = 12,
    ) -> dict[str, Any]:
        backbone = self._require_backbone()
        blocks = []
        for sample in samples:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), horizon_steps)
            if steps <= block_rows+1:
                continue
            t = np.asarray(target["t"][:steps], dtype=np.float64)
            delta = np.asarray(target["delta"][:steps], dtype=np.float64)
            omega = np.asarray(target["omega"][:steps], dtype=np.float64)
            pm = np.asarray(sample["training_context"]["Pm_observed"][:steps], dtype=np.float64)
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=np.float64)
            q = backbone.teacher_forced_emf(delta, t, static)
            force = self._baseline_force_path(delta, omega, q, pm, static)
            dt = np.diff(t)[:, None]
            residual = mass*(omega[1:]-omega[:-1])/dt - 0.5*(force[:-1]+force[1:])
            u, _, _ = np.linalg.svd(
                np.asarray(static["incidence"], dtype=np.float64), full_matrices=False
            )
            modal = residual@u[:, :len(mass)-1]
            width = len(modal)-block_rows+1
            block = np.stack([
                modal[start:start+block_rows].reshape(-1)
                for start in range(width)
            ], axis=1)
            scale = np.sqrt(np.mean(block**2, axis=1, keepdims=True))
            blocks.append(block/np.where(scale > np.finfo(float).eps, scale, 1.0))
        if not blocks:
            return {"status": "insufficient_data"}
        hankel = np.concatenate(blocks, axis=1)
        singular = np.linalg.svd(hankel, compute_uv=False)
        energy = np.cumsum(singular**2)/np.sum(singular**2)
        return {
            "status": "completed",
            "block_rows": block_rows,
            "hankel_shape": list(hankel.shape),
            "normalized_singular_values": (singular/singular[0]).tolist(),
            "rank_90pct": int(np.searchsorted(energy, 0.90)+1),
            "rank_95pct": int(np.searchsorted(energy, 0.95)+1),
            "rank_99pct": int(np.searchsorted(energy, 0.99)+1),
        }

    def fit(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-GSD requires nonempty train and validation sets")
        backbone = self._require_backbone()
        selected_train = train[:run_context.max_cases]
        selected_val = val[:run_context.max_cases]
        self.max_internal_step = run_context.max_internal_step
        backbone.max_internal_step = run_context.max_internal_step
        self.validation_scales = M0Model._training_scales(selected_train, len(selected_train))
        self.validation_scales_by_source = {
            source: M0Model._training_scales(group, len(group))
            for source, group in self._source_groups(selected_train).items()
        }
        hankel = self.residual_hankel_diagnostic(
            selected_train, horizon_steps=run_context.horizon_steps
        )
        prepared = self._prepare_weak_paths(
            selected_train,
            horizon_steps=run_context.horizon_steps,
            max_fit_steps=run_context.fit_max_steps,
        )
        train_candidates: list[dict[str, Any]] = []
        system_rows: list[dict[str, Any]] = []
        for tau_row in self.tau_mode_candidates:
            matrix, vector, system = self._build_weak_memory_system_from_cache(
                prepared, tau_row,
                windows=run_context.integral_windows,
                equal_case_weight=True,
            )
            system_rows.append({"taus": list(tau_row), **system})
            for ridge in self.ridge_values:
                for threshold in self.group_thresholds:
                    theta, diagnostics = self._group_prune_and_refit(
                        matrix, vector, tau_row, ridge, threshold
                    )
                    score = diagnostics["weak_residual_rmse"]*(
                        1.0+self.complexity_penalty*diagnostics["active_groups"]
                    )
                    train_candidates.append({
                        "taus": tau_row, "ridge": ridge,
                        "relative_group_threshold": threshold,
                        "theta": theta, "train_score": float(score),
                        **diagnostics, **system,
                    })
        shortlisted: list[dict[str, Any]] = []
        for order in sorted({len(row["taus"]) for row in train_candidates}):
            group = sorted(
                (row for row in train_candidates if len(row["taus"]) == order),
                key=lambda row: (
                    row["train_score"], row["active_groups"],
                    row["ridge"], row["relative_group_threshold"], row["taus"],
                ),
            )
            shortlisted.extend(group[:self.train_shortlist_per_order])

        n_edges = np.asarray(selected_train[0]["model_inputs"]["static"]["incidence"]).shape[1]
        candidate_specs: list[dict[str, Any]] = []
        # Exact frozen-backbone fallback is retained through both rollout stages.
        fallback_coefficients = np.zeros((1, len(self.feature_names), n_edges))
        candidate_specs.append({
            "candidate": "zero_closure", "taus": [1.0], "ridge": None,
            "relative_group_threshold": None, "coefficient_scale": 0.0,
            "active_groups": 0, "active_coefficients": 0,
            "weak_residual_rmse": None, "coefficients": fallback_coefficients,
        })
        for row in shortlisted:
            tau_array = np.asarray(row["taus"], dtype=np.float64)
            raw = row["theta"].reshape(
                len(tau_array), len(self.feature_names), n_edges
            )
            for coefficient_scale in self.coefficient_scales:
                coefficients = coefficient_scale*raw
                candidate_specs.append({
                    "candidate": "graph_memory", "taus": list(row["taus"]),
                    "ridge": row["ridge"],
                    "relative_group_threshold": row["relative_group_threshold"],
                    "coefficient_scale": coefficient_scale,
                    "active_groups": row["active_groups"],
                    "active_coefficients": int(np.sum(np.abs(coefficients) > 0)),
                    "weak_residual_rmse": row["weak_residual_rmse"],
                    "train_score": row["train_score"],
                    "matrix_condition_number": row["matrix_condition_number"],
                    "effective_rank": row["effective_rank"],
                    "coefficients": coefficients,
                })

        short_context = replace(
            run_context,
            horizon_steps=min(run_context.horizon_steps, self.rollout_short_horizon),
        )
        short_candidates: list[dict[str, Any]] = []
        for row in candidate_specs:
            self.taus = np.asarray(row["taus"], dtype=np.float64)
            self.coefficients = np.asarray(row["coefficients"], dtype=np.float64)
            summary = self._validation_summary(selected_val, short_context)
            short_candidates.append({
                **row,
                **{f"short_{key}": value for key, value in summary.items()},
            })
        fallback = short_candidates[0]
        nonzero_shortlist = sorted(
            short_candidates[1:],
            key=lambda row: (
                row["short_standardized_loss"], row["short_raw_mean_rmse"],
                row["active_groups"], len(row["taus"]),
            ),
        )[:self.rollout_long_shortlist]
        long_specs = [fallback, *nonzero_shortlist]
        validation_candidates: list[dict[str, Any]] = []
        for row in long_specs:
            self.taus = np.asarray(row["taus"], dtype=np.float64)
            self.coefficients = np.asarray(row["coefficients"], dtype=np.float64)
            summary = self._validation_summary(selected_val, run_context)
            validation_candidates.append({**row, **summary})

        eligible = [
            row for row in validation_candidates
            if row["failure_rate"] == 0.0 and np.isfinite(row["standardized_loss"])
        ]
        selected = min(
            eligible,
            key=lambda row: (
                row["standardized_loss"], row["raw_mean_rmse"],
                row["active_groups"], len(row["taus"]),
            ),
        )
        self.taus = np.asarray(selected["taus"], dtype=np.float64)
        self.coefficients = np.asarray(selected["coefficients"], dtype=np.float64)
        serializable_candidates = []
        for row in short_candidates:
            serializable_candidates.append({
                key: value for key, value in row.items() if key != "coefficients"
            })
        active = np.abs(self.coefficients) > 0
        selected_groups = np.any(active, axis=0)
        edge_support = np.any(selected_groups, axis=0)
        feature_support = np.any(selected_groups, axis=1)
        self.diagnostics = {
            "optimizer": "train_weak_group_sparse_graph_memory_plus_validation_rollout",
            "epochs": 0,
            "structure_discovery": True,
            "fixed_priors": [
                "swing_balance", "COI_quotient", "incidence_internal_force",
                "stable_first_order_memory", "causal_Pm_port",
            ],
            "discovered_quantities": [
                "memory_order", "time_constants", "edge_support",
                "feature_support", "closure_coefficients",
            ],
            "feature_names": list(self.feature_names),
            "tau_mode_candidates": [list(row) for row in self.tau_mode_candidates],
            "selected_taus": self.taus.tolist(),
            "selected_memory_order": int(len(self.taus)) if np.any(active) else 0,
            "selected_active_groups": int(np.sum(selected_groups)),
            "selected_active_edges": int(np.sum(edge_support)),
            "selected_edge_indices": np.flatnonzero(edge_support).tolist(),
            "selected_feature_names": [
                name for name, flag in zip(self.feature_names, feature_support) if flag
            ],
            "selected_validation": {
                key: value for key, value in selected.items() if key != "coefficients"
            },
            "fallback_validation": {
                key: value for key, value in validation_candidates[0].items()
                if key != "coefficients"
            },
            "candidate_table": serializable_candidates,
            "long_horizon_candidate_table": [
                {key: value for key, value in row.items() if key != "coefficients"}
                for row in validation_candidates
            ],
            "rollout_screening_horizon": int(short_context.horizon_steps),
            "rollout_long_shortlist": self.rollout_long_shortlist,
            "system_table": system_rows,
            "hankel_diagnostic": hankel,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "initial_state_policy": "causal_quasi_steady_edge_features_at_clearing",
            "zero_closure_backbone_fallback_included": True,
            "selected_is_zero_closure": bool(not np.any(active)),
            "labels_used_for_fit_or_selection": False,
        }
        return FitResult(
            method=self.method_name, status="completed",
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
                initial_state, time_grid, exogenous,
                max_step=self.max_internal_step/divisor,
            )
            for divisor in (1, 2, 4)
        ]
        errors = []
        for coarse, fine in zip(trajectories[:-1], trajectories[1:]):
            errors.append(float(np.sqrt(sum(
                np.mean((coarse[index]-fine[index])**2) for index in range(4)
            ))))
        return {
            "dt_vs_dt2": errors[0], "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0]/max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0]+1e-12),
        }

    def save(self, path: str | Path) -> None:
        backbone, taus, coefficients = self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backbone_path = path.with_name(path.stem+"_backbone.npz")
        backbone.save(backbone_path)
        np.savez_compressed(
            path, taus=taus, coefficients=coefficients,
            feature_names=np.asarray(self.feature_names),
            max_internal_step=self.max_internal_step,
            backbone_file=backbone_path.name,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphMemoryDiscoveryM0Model":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            backbone_name = str(data["backbone_file"])
            backbone = TransientEmfM0Model.load(path.with_name(backbone_name))
            model = cls(
                backbone=backbone, taus=data["taus"], coefficients=data["coefficients"],
                feature_names=tuple(str(value) for value in data["feature_names"]),
                max_internal_step=float(data["max_internal_step"]),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get(
                "diagnostics", {}
            )
        return model

    def physics_audit(self) -> dict[str, float | str | bool | list[Any]]:
        try:
            backbone, taus, coefficients = self._require_fitted()
        except (RuntimeError, ValueError):
            return {"status": "not_fitted"}
        finite = bool(np.isfinite(coefficients).all())
        stable = bool(np.all(taus > 0) and np.isfinite(taus).all())
        return {
            "status": "pass" if finite and stable else "fail",
            "method": self.method_name,
            "reported_role": "graph-structured stable-memory closure",
            "mechanical_damping_recovery_claim": False,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "initial_state_policy": "causal_quasi_steady_edge_features_at_clearing",
            "angle_shift_invariant_by_construction": True,
            "orientation_odd_feature_library": True,
            "internal_force_balance_by_incidence": True,
            "stable_memory_poles": stable,
            "selected_time_constants": taus.tolist(),
            "selected_memory_order": int(len(taus)) if np.any(coefficients) else 0,
            "selected_active_edges": int(np.sum(np.any(np.abs(coefficients) > 0, axis=(0, 1)))),
            "finite_coefficients": finite,
            "backbone_physics_status": backbone.physics_audit().get("status", "unknown"),
        }
