from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import FitResult, RunContext, sample_rollout_inputs
from .graph_memory_discovery import GraphMemoryDiscoveryM0Model
from .m0 import M0Model
from .transient_emf_m0 import TransientEmfM0Model


class GraphModalDiscoveryM0Model(GraphMemoryDiscoveryM0Model):
    """Discover a low-rank graph closure around a frozen M0-3EG model.

    The spatial closure basis is learned by a train-only POD of the weak force
    residual.  The equation library is then projected onto that basis and a
    group-sparse weak solve discovers the modal rank, nonlinear terms, static
    response and stable memory poles.  Validation is performed by closed-loop
    reintegration; the exact frozen backbone is always retained as a candidate.
    """

    method_name = "M0-GMD"
    DEFAULT_RANK_CANDIDATES = (1, 2, 4, 6, 9)
    DEFAULT_TAU_MODE_CANDIDATES = (
        (),
        (0.02,),
        (0.05,),
        (0.1,),
        (0.2,),
        (0.5,),
        (1.0,),
        (2.0,),
        (0.05, 0.2),
        (0.1, 0.5),
        (0.2, 1.0),
        (0.5, 2.0),
        (0.05, 0.2, 1.0),
        (0.1, 0.5, 2.0),
        (0.02, 0.1, 0.5, 2.0),
    )

    def __init__(
        self,
        backbone: TransientEmfM0Model | None = None,
        *,
        modal_basis: np.ndarray | None = None,
        taus: np.ndarray | tuple[float, ...] | None = None,
        static_coefficients: np.ndarray | None = None,
        memory_coefficients: np.ndarray | None = None,
        feature_names: tuple[str, ...] = GraphMemoryDiscoveryM0Model.ALLOWED_FEATURES,
        rank_candidates: tuple[int, ...] = DEFAULT_RANK_CANDIDATES,
        tau_mode_candidates: tuple[tuple[float, ...], ...] = DEFAULT_TAU_MODE_CANDIDATES,
        group_thresholds: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3),
        coefficient_scales: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.1, 0.15),
        ridge_values: tuple[float, ...] = (1e-4, 1e-2, 1e-1),
        weak_windows: tuple[int, ...] = (1, 5, 20, 40, 80),
        rollout_short_horizon: int = 20,
        rollout_long_shortlist: int = 30,
        train_shortlist: int = 20,
        complexity_penalty: float = 2e-3,
        max_internal_step: float = 0.01,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        # Do not call the edge-memory constructor: an empty tau tuple is a
        # meaningful static-only candidate in this model.
        self.backbone = backbone
        self.modal_basis = (
            None if modal_basis is None else np.asarray(modal_basis, dtype=np.float64)
        )
        self.taus = None if taus is None else np.asarray(taus, dtype=np.float64)
        self.static_coefficients = (
            None if static_coefficients is None
            else np.asarray(static_coefficients, dtype=np.float64)
        )
        self.memory_coefficients = (
            None if memory_coefficients is None
            else np.asarray(memory_coefficients, dtype=np.float64)
        )
        self.feature_names = tuple(feature_names)
        if not self.feature_names or any(
            name not in self.ALLOWED_FEATURES for name in self.feature_names
        ):
            raise ValueError("unknown or empty graph-modal feature library")
        self.rank_candidates = tuple(sorted(set(int(value) for value in rank_candidates)))
        if not self.rank_candidates or self.rank_candidates[0] < 1:
            raise ValueError("modal ranks must be positive")
        self.tau_mode_candidates = tuple(
            tuple(float(value) for value in row) for row in tau_mode_candidates
        )
        if any(
            any(not np.isfinite(value) or value <= 0 for value in row)
            or tuple(sorted(set(row))) != row
            for row in self.tau_mode_candidates
        ):
            raise ValueError("memory time constants must be positive and sorted")
        self.group_thresholds = tuple(float(value) for value in group_thresholds)
        self.coefficient_scales = tuple(float(value) for value in coefficient_scales)
        self.ridge_values = tuple(float(value) for value in ridge_values)
        self.weak_windows = tuple(sorted(set(int(value) for value in weak_windows)))
        if any(value < 0 or not np.isfinite(value) for value in self.group_thresholds):
            raise ValueError("group thresholds must be finite and nonnegative")
        if any(value <= 0 or value > 1 for value in self.coefficient_scales):
            raise ValueError("coefficient scales must lie in (0, 1]")
        if any(value < 0 or not np.isfinite(value) for value in self.ridge_values):
            raise ValueError("ridge values must be finite and nonnegative")
        if not self.weak_windows or self.weak_windows[0] < 1:
            raise ValueError("weak integration windows must be positive")
        self.rollout_short_horizon = int(rollout_short_horizon)
        self.rollout_long_shortlist = int(rollout_long_shortlist)
        self.train_shortlist = int(train_shortlist)
        self.complexity_penalty = float(complexity_penalty)
        self.max_internal_step = float(max_internal_step)
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
        self.validation_scales_by_source: dict[str, dict[str, float]] = {}
        self.diagnostics = diagnostics or {}

    def _require_fitted(
        self,
    ) -> tuple[TransientEmfM0Model, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        backbone = self._require_backbone()
        if (
            self.modal_basis is None or self.taus is None
            or self.static_coefficients is None or self.memory_coefficients is None
        ):
            raise RuntimeError("M0-GMD is not fitted")
        basis = np.asarray(self.modal_basis, dtype=np.float64)
        static = np.asarray(self.static_coefficients, dtype=np.float64)
        memory = np.asarray(self.memory_coefficients, dtype=np.float64)
        taus = np.asarray(self.taus, dtype=np.float64)
        if basis.ndim != 2 or static.shape != (len(self.feature_names), basis.shape[1]):
            raise ValueError("static graph-modal coefficient shape mismatch")
        if memory.shape != (len(taus), len(self.feature_names), basis.shape[1]):
            raise ValueError("memory graph-modal coefficient shape mismatch")
        if not np.isfinite(basis).all() or not np.isfinite(static).all() or not np.isfinite(memory).all():
            raise ValueError("graph-modal parameters must be finite")
        return backbone, basis, taus, static, memory

    def _modal_drivers(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        static: dict[str, np.ndarray],
        basis: np.ndarray | None = None,
    ) -> np.ndarray:
        backbone = self._require_backbone()
        use_basis = self.modal_basis if basis is None else basis
        if use_basis is None:
            raise RuntimeError("graph-modal basis has not been discovered")
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        coupling = np.asarray(backbone.coupling, dtype=np.float64).reshape(-1)
        edge_features = self._edge_features(delta, omega, static)
        node_drivers = np.einsum(
            "ne,fe,e->fn", incidence, edge_features, coupling, optimize=True
        )
        return node_drivers @ np.asarray(use_basis, dtype=np.float64)

    def initial_memory(
        self, delta: np.ndarray, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        _, _, taus, _, _ = self._require_fitted()
        drivers = self._modal_drivers(delta, omega, static)
        return np.repeat(drivers[np.newaxis, :, :], len(taus), axis=0)

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
        backbone, basis, taus, static_coef, memory_coef = self._require_fitted()
        dd, dw, dq = backbone.vector_field_augmented(
            delta, omega, q, pm, static, topology
        )
        drivers = self._modal_drivers(delta, omega, static, basis)
        if z.shape != memory_coef.shape:
            raise ValueError("graph-modal memory state shape mismatch")
        amplitude = np.sum(static_coef * drivers, axis=0)
        if len(taus):
            amplitude = amplitude + np.sum(
                memory_coef * (z - drivers[np.newaxis, :, :]), axis=(0, 1)
            )
            dz = (drivers[np.newaxis, :, :] - z) / taus[:, None, None]
        else:
            dz = np.empty_like(z)
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        correction_force = coi_project_power(-basis @ amplitude, mass)
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
        backbone, _, _, _, memory_coef = self._require_fitted()
        t = np.asarray(time_grid, dtype=np.float64)
        pm = np.asarray(exogenous["Pm"], dtype=np.float64)
        if len(t) != len(pm) or len(t) < 1:
            raise ValueError("time and Pm schedules must have the same positive length")
        step = self.max_internal_step if max_step is None else float(max_step)
        if step <= 0:
            raise ValueError("max_step must be positive")
        mode = self._input_mode(pm, str(exogenous.get("interpolation", "auto")))
        mass = np.asarray(initial_state["mass"], dtype=np.float64).reshape(-1)
        delta = coi_project(np.asarray(initial_state["delta0"]).reshape(-1), mass)
        omega = coi_project(np.asarray(initial_state["omega0"]).reshape(-1), mass)
        q = backbone.initial_emf(delta, initial_state)
        z = self.initial_memory(delta, omega, initial_state)
        if z.shape != memory_coef.shape:
            raise ValueError("selected modal structure does not match the input graph")
        topology = backbone._graph_topology(initial_state)
        out_delta, out_omega, out_q, out_z = [delta.copy()], [omega.copy()], [q.copy()], [z.copy()]

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
                f0, fh, f1 = (
                    substep / substeps,
                    (substep + 0.5) / substeps,
                    (substep + 1.0) / substeps,
                )
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
                    delta + h*k3[0], omega + h*k3[1],
                    q + h*k3[2], z + h*k3[3],
                    input_at(index, f1), initial_state, topology,
                )
                delta = coi_project(delta + h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0, mass)
                omega = coi_project(omega + h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0, mass)
                q = q + h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0
                z = z + h*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6.0
                if (
                    not np.isfinite(delta).all() or not np.isfinite(omega).all()
                    or not np.isfinite(q).all() or not np.isfinite(z).all()
                    or np.min(q) <= backbone.q_floor or np.max(q) >= backbone.q_ceiling
                ):
                    raise FloatingPointError(
                        f"M0-GMD left the audited state domain at step {index + 1}"
                    )
            out_delta.append(delta.copy())
            out_omega.append(omega.copy())
            out_q.append(q.copy())
            out_z.append(z.copy())
        return np.stack(out_delta), np.stack(out_omega), np.stack(out_q), np.stack(out_z)

    def rollout(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        delta, omega, _, _ = self.rollout_augmented(initial_state, time_grid, exogenous)
        return delta, omega

    def _prepare_raw_paths(
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
            pm = np.asarray(sample["training_context"]["Pm_observed"][:steps], dtype=np.float64)
            static = sample["model_inputs"]["static"]
            q = backbone.teacher_forced_emf(delta, t, static)
            baseline_force = self._baseline_force_path(delta, omega, q, pm, static)
            edge_features = np.stack([
                self._edge_features(delta[index], omega[index], static)
                for index in range(steps)
            ])
            mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
            dt = np.diff(t)[:, None]
            residual = mass*(omega[1:]-omega[:-1])/dt - 0.5*(
                baseline_force[:-1]+baseline_force[1:]
            )
            residual = coi_project_power(residual, mass)
            prepared.append({
                "event_id": sample["event_id"], "t": t, "delta": delta,
                "omega": omega, "static": static, "edge_features": edge_features,
                "baseline_force": baseline_force, "force_residual": residual,
            })
        if not prepared:
            raise ValueError("no graph-modal weak paths could be prepared")
        return prepared

    @staticmethod
    def _discover_modal_basis(
        prepared: list[dict[str, Any]], max_rank: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        blocks = []
        for item in prepared:
            residual = np.asarray(item["force_residual"], dtype=np.float64)
            scale = float(np.sqrt(np.mean(residual**2)))
            if scale > np.finfo(float).eps:
                blocks.append(residual / (scale*np.sqrt(len(residual))))
        if not blocks:
            raise ValueError("force residual has no discoverable spatial content")
        matrix = np.vstack(blocks)
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        n_nodes = matrix.shape[1]
        rank = min(max_rank, n_nodes-1, len(singular))
        raw = vh[:rank].T
        raw = raw - np.mean(raw, axis=0, keepdims=True)
        basis, _ = np.linalg.qr(raw)
        basis = basis[:, :rank]
        for column in range(rank):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0:
                basis[:, column] *= -1.0
        energy = np.cumsum(singular**2)/np.sum(singular**2)
        return basis, {
            "spatial_residual_shape": list(matrix.shape),
            "normalized_spatial_singular_values": (singular/singular[0]).tolist(),
            "spatial_rank_90pct": int(np.searchsorted(energy, 0.90)+1),
            "spatial_rank_95pct": int(np.searchsorted(energy, 0.95)+1),
            "maximum_rank_retained": int(rank),
        }

    def _attach_drivers(
        self, prepared: list[dict[str, Any]], basis: np.ndarray
    ) -> None:
        coupling = np.asarray(self._require_backbone().coupling, dtype=np.float64)
        for item in prepared:
            incidence = np.asarray(item["static"]["incidence"], dtype=np.float64)
            node = np.einsum(
                "ne,tfe,e->tfn", incidence, item["edge_features"], coupling,
                optimize=True,
            )
            item["driver_path"] = node @ basis

    def _build_normal_equations(
        self,
        prepared: list[dict[str, Any]],
        basis: np.ndarray,
        taus: tuple[float, ...],
        windows: tuple[int, ...],
    ) -> dict[str, Any]:
        rank = basis.shape[1]
        n_components = 1+len(taus)
        n_params = n_components*len(self.feature_names)*rank
        gram = np.zeros((n_params, n_params), dtype=np.float64)
        rhs = np.zeros(n_params, dtype=np.float64)
        yty = 0.0
        n_cases = 0
        n_scalar = 0
        n_windows = 0
        tau_array = np.asarray(taus, dtype=np.float64)
        for item in prepared:
            t = item["t"]
            omega = item["omega"]
            mass = np.asarray(item["static"]["mass"], dtype=np.float64).reshape(-1)
            driver = np.asarray(item["driver_path"], dtype=np.float64)[..., :rank]
            if len(tau_array):
                z = self._first_order_hold_memory(driver, t, tau_array)
                components = np.concatenate((
                    driver[:, None, :, :], z-driver[:, None, :, :]
                ), axis=1)
            else:
                components = driver[:, None, :, :]
            design_path = -np.einsum(
                "nr,tcfr->tncfr", basis, components, optimize=True
            ).reshape(len(t), len(mass), n_params)
            dt = np.diff(t)
            design_prefix = np.concatenate((
                np.zeros((1, len(mass), n_params)),
                np.cumsum(
                    0.5*(design_path[:-1]+design_path[1:])*dt[:, None, None],
                    axis=0,
                ),
            ))
            force = item["baseline_force"]
            force_prefix = np.concatenate((
                np.zeros((1, len(mass))),
                np.cumsum(0.5*(force[:-1]+force[1:])*dt[:, None], axis=0),
            ))
            case_gram = np.zeros_like(gram)
            case_rhs = np.zeros_like(rhs)
            case_yty = 0.0
            case_widths = 0
            for width in windows:
                if width <= 0 or width >= len(t):
                    continue
                starts = np.arange(len(t)-width)
                stops = starts+width
                block = design_prefix[stops]-design_prefix[starts]
                actual = mass*(omega[stops]-omega[starts])
                baseline = force_prefix[stops]-force_prefix[starts]
                x = block.reshape(-1, n_params)
                y = (actual-baseline).reshape(-1)
                width_weight = 1.0/max(len(y), 1)
                case_gram += width_weight*(x.T@x)
                case_rhs += width_weight*(x.T@y)
                case_yty += width_weight*float(y@y)
                case_widths += 1
                n_windows += len(starts)
                n_scalar += len(y)
            if case_widths == 0:
                continue
            gram += case_gram/case_widths
            rhs += case_rhs/case_widths
            yty += case_yty/case_widths
            n_cases += 1
        if n_cases == 0:
            raise ValueError("no graph-modal weak windows could be constructed")
        return {
            "gram": gram, "rhs": rhs, "yty": yty, "n_cases": n_cases,
            "n_scalar_equations": n_scalar, "n_integration_windows": n_windows,
            "n_parameters": n_params, "n_components": n_components,
            "n_features": len(self.feature_names), "modal_rank": rank,
            "window_weighting": "equal_case_equal_horizon",
        }

    @staticmethod
    def _solve_normal_equations(
        stats: dict[str, Any], ridge: float, active: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        gram_full = np.asarray(stats["gram"], dtype=np.float64)
        rhs_full = np.asarray(stats["rhs"], dtype=np.float64)
        if active is None:
            active = np.ones(len(rhs_full), dtype=bool)
        indices = np.flatnonzero(active)
        result = np.zeros_like(rhs_full)
        beta_full = np.zeros_like(rhs_full)
        if not len(indices):
            return result, beta_full, {
                "matrix_condition_number": float("inf"), "effective_rank": 0,
                "weak_residual_rmse": float(np.sqrt(stats["yty"]/stats["n_cases"])),
            }
        gram = gram_full[np.ix_(indices, indices)]
        rhs = rhs_full[indices]
        scale = np.sqrt(np.maximum(np.diag(gram)/stats["n_cases"], 0.0))
        nonzero = scale > np.finfo(float).eps
        safe_scale = np.where(nonzero, scale, 1.0)
        normalized_gram = gram/safe_scale[:, None]/safe_scale[None, :]
        normalized_rhs = rhs/safe_scale
        beta = np.linalg.solve(
            normalized_gram+ridge*np.eye(len(indices)), normalized_rhs
        )
        beta[~nonzero] = 0.0
        theta = beta/safe_scale
        result[indices] = theta
        beta_full[indices] = beta
        sse = float(result@gram_full@result-2*result@rhs_full+stats["yty"])
        return result, beta_full, {
            "matrix_condition_number": float(np.linalg.cond(normalized_gram)),
            "effective_rank": int(np.linalg.matrix_rank(normalized_gram)),
            "weak_residual_rmse": float(np.sqrt(max(sse, 0.0)/stats["n_cases"])),
        }

    def _group_prune_stats(
        self, stats: dict[str, Any], ridge: float, threshold: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        theta, beta, diagnostics = self._solve_normal_equations(stats, ridge)
        shape = (stats["n_components"], stats["n_features"], stats["modal_rank"])
        group_norm = np.sqrt(np.sum(beta.reshape(shape)**2, axis=0))
        cutoff = threshold*float(np.max(group_norm)) if group_norm.size else 0.0
        active_groups = group_norm > cutoff if threshold > 0 else group_norm > 0
        active = np.repeat(active_groups[None, :, :], shape[0], axis=0).reshape(-1)
        if np.any(active):
            theta, _, diagnostics = self._solve_normal_equations(stats, ridge, active)
        else:
            theta = np.zeros(stats["n_parameters"], dtype=np.float64)
        return theta, {
            **diagnostics,
            "active_groups": int(np.sum(active_groups)),
            "active_coefficients": int(np.sum(active)),
            "relative_group_threshold": float(threshold),
        }

    def _set_candidate(self, row: dict[str, Any], basis_full: np.ndarray) -> None:
        rank = int(row["modal_rank"])
        self.modal_basis = basis_full[:, :rank].copy()
        self.taus = np.asarray(row["taus"], dtype=np.float64)
        self.static_coefficients = np.asarray(row["static_coefficients"], dtype=np.float64)
        self.memory_coefficients = np.asarray(row["memory_coefficients"], dtype=np.float64)

    def fit(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-GMD requires nonempty train and validation sets")
        if self.backbone is None:
            self.backbone = TransientEmfM0Model()
        try:
            backbone = self._require_backbone()
            backbone_fit = None
        except RuntimeError:
            backbone_fit = self.backbone.fit(train, val, run_context)
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
        prepared = self._prepare_raw_paths(
            selected_train,
            horizon_steps=run_context.horizon_steps,
            max_fit_steps=run_context.fit_max_steps,
        )
        n_nodes = np.asarray(prepared[0]["static"]["mass"]).size
        ranks = tuple(value for value in self.rank_candidates if value <= n_nodes-1)
        basis_full, spatial = self._discover_modal_basis(prepared, max(ranks))
        self._attach_drivers(prepared, basis_full)

        train_candidates: list[dict[str, Any]] = []
        system_rows: list[dict[str, Any]] = []
        for rank in ranks:
            basis = basis_full[:, :rank]
            for taus in self.tau_mode_candidates:
                stats = self._build_normal_equations(
                    prepared, basis, taus, self.weak_windows
                )
                system_rows.append({
                    "modal_rank": rank, "taus": list(taus),
                    **{key: value for key, value in stats.items()
                       if key not in {"gram", "rhs", "yty"}},
                })
                for ridge in self.ridge_values:
                    for threshold in self.group_thresholds:
                        theta, diagnostics = self._group_prune_stats(
                            stats, ridge, threshold
                        )
                        score = diagnostics["weak_residual_rmse"]*(
                            1.0+self.complexity_penalty*diagnostics["active_groups"]
                        )
                        shape = (1+len(taus), len(self.feature_names), rank)
                        coefficients = theta.reshape(shape)
                        train_candidates.append({
                            "modal_rank": rank, "taus": tuple(taus), "ridge": ridge,
                            "relative_group_threshold": threshold,
                            "train_score": float(score),
                            "static_raw": coefficients[0],
                            "memory_raw": coefficients[1:],
                            **diagnostics,
                        })

        # Keep the strongest equation per rank/order, then a small global set.
        per_structure = []
        keys = sorted({(row["modal_rank"], row["taus"]) for row in train_candidates})
        for key in keys:
            group = [
                row for row in train_candidates
                if (row["modal_rank"], row["taus"]) == key
            ]
            per_structure.append(min(
                group,
                key=lambda row: (
                    row["train_score"], row["active_groups"], row["ridge"],
                    row["relative_group_threshold"], row["taus"],
                ),
            ))
        shortlisted = sorted(
            per_structure,
            key=lambda row: (
                row["train_score"], row["active_groups"],
                row["modal_rank"], len(row["taus"]),
            ),
        )[:self.train_shortlist]

        fallback_rank = 1
        fallback = {
            "candidate": "zero_closure", "modal_rank": fallback_rank, "taus": (),
            "ridge": None, "relative_group_threshold": None,
            "coefficient_scale": 0.0, "active_groups": 0,
            "active_coefficients": 0, "weak_residual_rmse": None,
            "static_coefficients": np.zeros((len(self.feature_names), fallback_rank)),
            "memory_coefficients": np.zeros((0, len(self.feature_names), fallback_rank)),
        }
        candidate_specs = [fallback]
        for row in shortlisted:
            for scale in self.coefficient_scales:
                candidate_specs.append({
                    "candidate": "graph_modal_closure",
                    "modal_rank": row["modal_rank"], "taus": row["taus"],
                    "ridge": row["ridge"],
                    "relative_group_threshold": row["relative_group_threshold"],
                    "coefficient_scale": scale,
                    "active_groups": row["active_groups"],
                    "active_coefficients": row["active_coefficients"],
                    "weak_residual_rmse": row["weak_residual_rmse"],
                    "train_score": row["train_score"],
                    "matrix_condition_number": row["matrix_condition_number"],
                    "effective_rank": row["effective_rank"],
                    "static_coefficients": scale*row["static_raw"],
                    "memory_coefficients": scale*row["memory_raw"],
                })

        short_context = replace(
            run_context,
            horizon_steps=min(run_context.horizon_steps, self.rollout_short_horizon),
        )
        short_rows = []
        for row in candidate_specs:
            self._set_candidate(row, basis_full)
            summary = self._validation_summary(selected_val, short_context)
            short_rows.append({
                **row, **{f"short_{key}": value for key, value in summary.items()}
            })
        nonzero = sorted(
            short_rows[1:],
            key=lambda row: (
                row["short_standardized_loss"], row["short_raw_mean_rmse"],
                row["active_groups"], row["modal_rank"], len(row["taus"]),
            ),
        )[:self.rollout_long_shortlist]
        long_rows = []
        for row in [short_rows[0], *nonzero]:
            self._set_candidate(row, basis_full)
            long_rows.append({**row, **self._validation_summary(selected_val, run_context)})
        eligible = [
            row for row in long_rows
            if row["failure_rate"] == 0.0 and np.isfinite(row["standardized_loss"])
        ]
        selected = min(
            eligible,
            key=lambda row: (
                row["standardized_loss"], row["raw_mean_rmse"],
                row["active_groups"], row["modal_rank"], len(row["taus"]),
            ),
        )
        self._set_candidate(selected, basis_full)

        def serializable(row: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value for key, value in row.items()
                if key not in {"static_coefficients", "memory_coefficients", "static_raw", "memory_raw"}
            }

        active_static = np.abs(self.static_coefficients) > 0
        active_memory = np.abs(self.memory_coefficients) > 0
        active_groups = active_static | (
            np.any(active_memory, axis=0) if len(self.taus) else False
        )
        active_features = np.any(active_groups, axis=1)
        active_modes = np.any(active_groups, axis=0)
        self.diagnostics = {
            "optimizer": "train_only_pod_plus_group_sparse_weak_solve_plus_validation_rollout",
            "epochs": 0,
            "weak_window_weighting": "equal_case_equal_horizon",
            "weak_windows": list(self.weak_windows),
            "backbone_fitted_inside_composite": backbone_fit is not None,
            "structure_discovery": True,
            "fixed_priors": [
                "swing_balance", "COI_quotient", "incidence_internal_force",
                "stable_first_order_memory", "causal_Pm_port",
            ],
            "discovered_quantities": [
                "spatial_residual_basis", "modal_rank", "memory_order",
                "time_constants", "feature_support", "modal_support",
                "static_vs_dynamic_support", "closure_coefficients",
            ],
            "feature_names": list(self.feature_names),
            "selected_modal_rank": int(self.modal_basis.shape[1]),
            "selected_taus": self.taus.tolist(),
            "selected_memory_order": int(len(self.taus)) if np.any(active_memory) else 0,
            "selected_active_groups": int(np.sum(active_groups)),
            "selected_active_modal_indices": np.flatnonzero(active_modes).tolist(),
            "selected_feature_names": [
                name for name, flag in zip(self.feature_names, active_features) if flag
            ],
            "selected_has_static_closure": bool(np.any(active_static)),
            "selected_has_dynamic_closure": bool(np.any(active_memory)),
            "selected_validation": serializable(selected),
            "fallback_validation": serializable(long_rows[0]),
            "candidate_table": [serializable(row) for row in short_rows],
            "long_horizon_candidate_table": [serializable(row) for row in long_rows],
            "system_table": system_rows,
            "spatial_discovery": spatial,
            "hankel_diagnostic": hankel,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "initial_state_policy": "causal_quasi_steady_modal_drivers_at_clearing",
            "zero_closure_backbone_fallback_included": True,
            "selected_is_zero_closure": bool(not np.any(active_groups)),
            "labels_used_for_fit_or_selection": False,
            "test_ids_read": [],
        }
        return FitResult(
            method=self.method_name, status="completed",
            train_case_ids=tuple(sample["event_id"] for sample in selected_train),
            val_case_ids=tuple(sample["event_id"] for sample in selected_val),
            best_validation_loss=float(selected["standardized_loss"]),
            details=self.diagnostics,
        )

    def save(self, path: str | Path) -> None:
        backbone, basis, taus, static, memory = self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backbone_path = path.with_name(path.stem+"_backbone.npz")
        backbone.save(backbone_path)
        np.savez_compressed(
            path, modal_basis=basis, taus=taus,
            static_coefficients=static, memory_coefficients=memory,
            feature_names=np.asarray(self.feature_names),
            max_internal_step=self.max_internal_step,
            backbone_file=backbone_path.name,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
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
            terms = [np.mean((coarse[index]-fine[index])**2) for index in range(3)]
            if coarse[3].size:
                terms.append(np.mean((coarse[3]-fine[3])**2))
            errors.append(float(np.sqrt(np.sum(terms))))
        return {
            "dt_vs_dt2": errors[0], "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0]/max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0]+1e-12),
        }

    @classmethod
    def load(cls, path: str | Path) -> "GraphModalDiscoveryM0Model":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            backbone = TransientEmfM0Model.load(path.with_name(str(data["backbone_file"])))
            model = cls(
                backbone=backbone, modal_basis=data["modal_basis"], taus=data["taus"],
                static_coefficients=data["static_coefficients"],
                memory_coefficients=data["memory_coefficients"],
                feature_names=tuple(str(value) for value in data["feature_names"]),
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
            backbone, basis, taus, static, memory = self._require_fitted()
        except (RuntimeError, ValueError):
            return {"status": "not_fitted"}
        stable = bool(np.all(taus > 0) and np.isfinite(taus).all())
        zero_sum = float(np.max(np.abs(np.sum(basis, axis=0))))
        orthogonality = float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1]))))
        finite = bool(np.isfinite(static).all() and np.isfinite(memory).all())
        passed = stable and finite and zero_sum < 1e-10 and orthogonality < 1e-10
        return {
            "status": "pass" if passed else "fail",
            "method": self.method_name,
            "reported_role": "train-discovered low-rank graph-modal closure",
            "mechanical_damping_recovery_claim": False,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "initial_state_policy": "causal_quasi_steady_modal_drivers_at_clearing",
            "angle_shift_invariant_by_construction": True,
            "orientation_odd_feature_library": True,
            "internal_force_balance_by_zero_sum_modal_basis": zero_sum < 1e-10,
            "maximum_modal_sum": zero_sum,
            "modal_orthogonality_error": orthogonality,
            "stable_memory_poles": stable,
            "selected_time_constants": taus.tolist(),
            "selected_modal_rank": int(basis.shape[1]),
            "finite_coefficients": finite,
            "backbone_physics_status": backbone.physics_audit().get("status", "unknown"),
        }
