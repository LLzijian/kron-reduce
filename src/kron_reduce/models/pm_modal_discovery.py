from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext
from .graph_modal_discovery import GraphModalDiscoveryM0Model
from .m0 import M0Model, _pm_equilibrium, _trapz


def power_balance_basis(n_nodes: int) -> np.ndarray:
    """Return a deterministic orthonormal basis for ``1.T @ p == 0``."""
    if n_nodes < 2:
        raise ValueError("at least two nodes are required for balanced power")
    direction = np.ones(n_nodes, dtype=np.float64) / np.sqrt(n_nodes)
    _, _, vh = np.linalg.svd(direction.reshape(1, -1), full_matrices=True)
    return vh[1:].T


class GraphModalPmDynamics:
    """Stable train-discovered ODE for the balanced mechanical-power port.

    The observed post-clear schedule is used only during fitting.  Deployment
    evolves the clearing mechanical power from predicted relative speed::

        a = U_p.T (Pm - Pm_equilibrium)
        v = U_p.T Pi_power(omega)
        a_dot = A a + B v

    ``U_p`` spans the zero-total-power subspace.  ``A`` and ``B`` are obtained
    from a multi-window integral solve; validation chooses ridge, sparsity and
    a Hurwitz spectral margin.  The exact ``A=B=0`` ZOH policy is included as
    a fallback.

    This is an identifiable reduced input-port model.  It is not claimed to
    recover the simulator's machine-level turbine-governor parameters because
    the canonical COI data omit the common-frequency state that drives those
    governors.
    """

    def __init__(
        self,
        *,
        basis: np.ndarray | None = None,
        state_matrix: np.ndarray | None = None,
        input_matrix: np.ndarray | None = None,
        weak_windows: tuple[int, ...] = (1, 5, 20, 40, 80),
        ridge_values: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0),
        relative_thresholds: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1),
        spectral_margins: tuple[float, ...] = (1e-4, 0.05, 0.2, 0.5),
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.basis = None if basis is None else np.asarray(basis, dtype=np.float64)
        self.state_matrix = (
            None if state_matrix is None
            else np.asarray(state_matrix, dtype=np.float64)
        )
        self.input_matrix = (
            None if input_matrix is None
            else np.asarray(input_matrix, dtype=np.float64)
        )
        self.weak_windows = tuple(sorted(set(int(value) for value in weak_windows)))
        self.ridge_values = tuple(float(value) for value in ridge_values)
        self.relative_thresholds = tuple(float(value) for value in relative_thresholds)
        self.spectral_margins = tuple(float(value) for value in spectral_margins)
        if not self.weak_windows or self.weak_windows[0] < 1:
            raise ValueError("Pm weak windows must be positive")
        if any(value < 0 or not np.isfinite(value) for value in self.ridge_values):
            raise ValueError("Pm ridge values must be finite and nonnegative")
        if any(value < 0 or value >= 1 for value in self.relative_thresholds):
            raise ValueError("Pm thresholds must lie in [0, 1)")
        if any(value <= 0 or not np.isfinite(value) for value in self.spectral_margins):
            raise ValueError("Pm spectral margins must be finite and positive")
        self.diagnostics = diagnostics or {}

    def require_fitted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.basis is None or self.state_matrix is None or self.input_matrix is None:
            raise RuntimeError("graph-modal Pm dynamics is not fitted")
        basis = np.asarray(self.basis, dtype=np.float64)
        state = np.asarray(self.state_matrix, dtype=np.float64)
        inputs = np.asarray(self.input_matrix, dtype=np.float64)
        rank = basis.shape[1]
        if basis.ndim != 2 or state.shape != (rank, rank) or inputs.shape != (rank, rank):
            raise ValueError("graph-modal Pm matrix shape mismatch")
        if not all(np.isfinite(value).all() for value in (basis, state, inputs)):
            raise ValueError("graph-modal Pm parameters must be finite")
        return basis, state, inputs

    @staticmethod
    def _source(sample: dict[str, Any]) -> str:
        return str(
            sample.get("training_context", {})
            .get("metadata", {})
            .get("source_dataset", "all")
        )

    @staticmethod
    def _spectral_shift(matrix: np.ndarray, margin: float) -> tuple[np.ndarray, float]:
        maximum_real = float(np.max(np.real(np.linalg.eigvals(matrix))))
        shift = max(0.0, maximum_real + margin)
        return matrix - shift*np.eye(matrix.shape[0]), shift

    def build_weak_system(
        self,
        samples: list[dict[str, Any]],
        *,
        horizon_steps: int,
        max_fit_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not samples:
            raise ValueError("Pm weak system requires training samples")
        n_nodes = np.asarray(samples[0]["model_inputs"]["static"]["mass"]).size
        basis = power_balance_basis(n_nodes)
        case_matrices: list[np.ndarray] = []
        case_targets: list[np.ndarray] = []
        horizon_counts = {str(width): 0 for width in self.weak_windows}
        source_counts: dict[str, int] = {}
        response_blocks: list[np.ndarray] = []
        for sample in samples:
            target = sample["training_context"]["targets"]
            t = np.asarray(target["t"], dtype=np.float64)
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
            if mass.size != n_nodes:
                raise ValueError("Pm discovery requires a common node representation")
            limit = min(
                len(t), int(max_fit_steps) if int(max_fit_steps) > 0
                else max(horizon_steps, max(self.weak_windows)+1),
            )
            if limit < 2:
                continue
            t = t[:limit]
            p = (
                np.asarray(sample["training_context"]["Pm_observed"][:limit])
                - _pm_equilibrium(static)
            )
            omega = coi_project_power(
                np.asarray(target["omega"][:limit], dtype=np.float64), mass
            )
            modal_p, modal_omega = p@basis, omega@basis
            response_blocks.append(p-p[[0]])
            horizon_blocks: list[tuple[np.ndarray, np.ndarray]] = []
            for width in self.weak_windows:
                if width >= limit:
                    continue
                rows, values = [], []
                for start in range(limit-width):
                    stop = start+width
                    segment = slice(start, stop+1)
                    rows.append(np.concatenate((
                        _trapz(modal_p[segment], t[segment]),
                        _trapz(modal_omega[segment], t[segment]),
                    )))
                    values.append(modal_p[stop]-modal_p[start])
                matrix, value = np.asarray(rows), np.asarray(values)
                scalar_count = max(value.size, 1)
                weight = 1.0/np.sqrt(scalar_count)
                horizon_blocks.append((weight*matrix, weight*value))
                horizon_counts[str(width)] += len(rows)
            if not horizon_blocks:
                continue
            case_weight = 1.0/np.sqrt(len(horizon_blocks))
            case_matrices.extend(case_weight*matrix for matrix, _ in horizon_blocks)
            case_targets.extend(case_weight*value for _, value in horizon_blocks)
            source = self._source(sample)
            source_counts[source] = source_counts.get(source, 0)+1
        if not case_matrices:
            raise ValueError("no eligible graph-modal Pm integration windows")
        matrix, targets = np.concatenate(case_matrices), np.concatenate(case_targets)
        response = np.concatenate(response_blocks)
        singular = np.linalg.svd(response, compute_uv=False)
        energy = singular**2
        cumulative = np.cumsum(energy)/max(np.sum(energy), np.finfo(float).eps)
        diagnostics = {
            "n_cases": int(sum(source_counts.values())),
            "source_counts": source_counts,
            "weak_windows": list(self.weak_windows),
            "horizon_window_counts": horizon_counts,
            "weak_weighting": "equal_case_equal_horizon",
            "matrix_shape": list(matrix.shape),
            "matrix_condition_number": float(np.linalg.cond(matrix)),
            "effective_rank": int(np.linalg.matrix_rank(matrix)),
            "response_singular_values": singular.tolist(),
            "response_rank_90pct": int(np.searchsorted(cumulative, 0.9)+1),
            "response_rank_95pct": int(np.searchsorted(cumulative, 0.95)+1),
            "response_rank_99pct": int(np.searchsorted(cumulative, 0.99)+1),
        }
        self.basis = basis
        return matrix, targets, diagnostics

    def train_candidates(
        self, matrix: np.ndarray, targets: np.ndarray
    ) -> list[dict[str, Any]]:
        rank = targets.shape[1]
        gram, cross = matrix.T@matrix, matrix.T@targets
        candidates: list[dict[str, Any]] = [{
            "candidate": "Pm_clear_ZOH", "ridge": None,
            "relative_threshold": None, "spectral_margin": None,
            "spectral_shift": 0.0, "spectral_abscissa": 0.0,
            "active_coefficients": 0,
            "state_matrix": np.zeros((rank, rank)),
            "input_matrix": np.zeros((rank, rank)),
            "weak_residual_rmse": float(np.sqrt(np.mean(targets**2))),
        }]
        for ridge in self.ridge_values:
            coefficient = np.linalg.solve(
                gram+ridge*np.eye(matrix.shape[1]), cross
            )
            raw_state, raw_input = coefficient[:rank].T, coefficient[rank:].T
            maximum = max(
                float(np.max(np.abs(raw_state))),
                float(np.max(np.abs(raw_input))),
                np.finfo(float).eps,
            )
            for threshold in self.relative_thresholds:
                state = raw_state.copy()
                inputs = raw_input.copy()
                state[np.abs(state) < threshold*maximum] = 0.0
                inputs[np.abs(inputs) < threshold*maximum] = 0.0
                for margin in self.spectral_margins:
                    stable_state, shift = self._spectral_shift(state, margin)
                    fitted = np.vstack((stable_state.T, inputs.T))
                    residual = matrix@fitted-targets
                    candidates.append({
                        "candidate": "stable_graph_modal_Pm_ODE",
                        "ridge": float(ridge),
                        "relative_threshold": float(threshold),
                        "spectral_margin": float(margin),
                        "spectral_shift": float(shift),
                        "spectral_abscissa": float(np.max(np.real(np.linalg.eigvals(stable_state)))),
                        "active_coefficients": int(np.count_nonzero(stable_state)+np.count_nonzero(inputs)),
                        "state_matrix": stable_state,
                        "input_matrix": inputs,
                        "weak_residual_rmse": float(np.sqrt(np.mean(residual**2))),
                    })
        return candidates

    def set_candidate(self, candidate: dict[str, Any]) -> None:
        self.state_matrix = np.asarray(candidate["state_matrix"], dtype=np.float64)
        self.input_matrix = np.asarray(candidate["input_matrix"], dtype=np.float64)

    def modal_derivative(self, modal_p: np.ndarray, omega: np.ndarray, mass: np.ndarray) -> np.ndarray:
        basis, state, inputs = self.require_fitted()
        modal_omega = coi_project_power(omega, mass)@basis
        return state@modal_p+inputs@modal_omega

    def teacher_forced_rollout(
        self, sample: dict[str, Any], *, horizon_steps: int, max_step: float = 0.01
    ) -> np.ndarray:
        basis, _, _ = self.require_fitted()
        static = sample["model_inputs"]["static"]
        target = sample["training_context"]["targets"]
        steps = min(len(target["t"]), horizon_steps)
        t = np.asarray(target["t"][:steps], dtype=np.float64)
        omega = np.asarray(target["omega"][:steps], dtype=np.float64)
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        equilibrium = _pm_equilibrium(static)
        modal_p = (
            np.asarray(static["Pm_clear"], dtype=np.float64).reshape(-1)-equilibrium
        )@basis
        output = [basis@modal_p+equilibrium]
        for index in range(len(t)-1):
            interval = float(t[index+1]-t[index])
            if interval <= 0:
                raise ValueError("Pm rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval/max_step)))
            step = interval/substeps
            for substep in range(substeps):
                fractions = (
                    substep/substeps, (substep+0.5)/substeps,
                    (substep+1.0)/substeps,
                )
                def omega_at(fraction: float) -> np.ndarray:
                    return omega[index]+fraction*(omega[index+1]-omega[index])
                k1 = self.modal_derivative(modal_p, omega_at(fractions[0]), mass)
                k2 = self.modal_derivative(modal_p+0.5*step*k1, omega_at(fractions[1]), mass)
                k3 = self.modal_derivative(modal_p+0.5*step*k2, omega_at(fractions[1]), mass)
                k4 = self.modal_derivative(modal_p+step*k3, omega_at(fractions[2]), mass)
                modal_p = modal_p+step*(k1+2*k2+2*k3+k4)/6.0
            output.append(basis@modal_p+equilibrium)
        return np.stack(output)


class CausalPmGraphModalM0Model(DynamicsModel):
    """M0-GMD closed with a causal train-discovered mechanical-power ODE."""

    method_name = "M0-GMD-PM"

    def __init__(
        self,
        backbone: GraphModalDiscoveryM0Model | None = None,
        pm_dynamics: GraphModalPmDynamics | None = None,
        *,
        max_internal_step: float = 0.01,
        standalone_shortlist: int = 16,
        minimum_validation_improvement: float = 0.02,
        maximum_effective_time_constant: float = 5.0,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.backbone = backbone
        self.pm_dynamics = pm_dynamics or GraphModalPmDynamics()
        self.max_internal_step = float(max_internal_step)
        self.standalone_shortlist = int(standalone_shortlist)
        self.minimum_validation_improvement = float(minimum_validation_improvement)
        self.maximum_effective_time_constant = float(maximum_effective_time_constant)
        if self.maximum_effective_time_constant <= 0:
            raise ValueError("maximum Pm time constant must be positive")
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
        self.validation_scales_by_source: dict[str, dict[str, float]] = {}
        self.diagnostics = diagnostics or {}

    @staticmethod
    def _source(sample: dict[str, Any]) -> str:
        return GraphModalPmDynamics._source(sample)

    def _require_fitted(self):
        if self.backbone is None:
            raise RuntimeError("M0-GMD-PM requires a fitted M0-GMD backbone")
        backbone_state = self.backbone._require_fitted()
        pm_state = self.pm_dynamics.require_fitted()
        return backbone_state, pm_state

    def rollout_augmented(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray] | None = None,
        *,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (electrical, _, _, _, memory), (pm_basis, _, _) = self._require_fitted()
        t = np.asarray(time_grid, dtype=np.float64)
        if len(t) < 1:
            raise ValueError("rollout requires a nonempty time grid")
        step_limit = self.max_internal_step if max_step is None else float(max_step)
        if step_limit <= 0:
            raise ValueError("max_step must be positive")
        mass = np.asarray(initial_state["mass"], dtype=np.float64).reshape(-1)
        equilibrium_pm = _pm_equilibrium(initial_state)
        delta = coi_project(np.asarray(initial_state["delta0"]).reshape(-1), mass)
        omega = coi_project(np.asarray(initial_state["omega0"]).reshape(-1), mass)
        q = electrical.initial_emf(delta, initial_state)
        z = self.backbone.initial_memory(delta, omega, initial_state)
        modal_p = (
            np.asarray(initial_state["Pm_clear"], dtype=np.float64).reshape(-1)
            - equilibrium_pm
        )@pm_basis
        topology = electrical._graph_topology(initial_state)
        outputs = ([delta.copy()], [omega.copy()], [q.copy()], [z.copy()], [pm_basis@modal_p+equilibrium_pm])

        def field(di, wi, qi, zi, ai):
            absolute_pm = pm_basis@ai+equilibrium_pm
            dd, dw, dq, dz = self.backbone.vector_field_augmented(
                di, wi, qi, zi, absolute_pm, initial_state, topology
            )
            da = self.pm_dynamics.modal_derivative(ai, wi, mass)
            return dd, dw, dq, dz, da

        for index in range(len(t)-1):
            interval = float(t[index+1]-t[index])
            if interval <= 0:
                raise ValueError("rollout time must be strictly increasing")
            substeps = max(1, int(np.ceil(interval/step_limit)))
            step = interval/substeps
            for _ in range(substeps):
                k1 = field(delta, omega, q, z, modal_p)
                k2 = field(
                    delta+0.5*step*k1[0], omega+0.5*step*k1[1],
                    q+0.5*step*k1[2], z+0.5*step*k1[3], modal_p+0.5*step*k1[4],
                )
                k3 = field(
                    delta+0.5*step*k2[0], omega+0.5*step*k2[1],
                    q+0.5*step*k2[2], z+0.5*step*k2[3], modal_p+0.5*step*k2[4],
                )
                k4 = field(
                    delta+step*k3[0], omega+step*k3[1], q+step*k3[2],
                    z+step*k3[3], modal_p+step*k3[4],
                )
                delta = coi_project(delta+step*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0, mass)
                omega = coi_project(omega+step*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0, mass)
                q = q+step*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0
                z = z+step*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6.0
                modal_p = modal_p+step*(k1[4]+2*k2[4]+2*k3[4]+k4[4])/6.0
                if not all(np.isfinite(value).all() for value in (delta, omega, q, z, modal_p)):
                    raise FloatingPointError("nonfinite M0-GMD-PM state")
            outputs[0].append(delta.copy())
            outputs[1].append(omega.copy())
            outputs[2].append(q.copy())
            outputs[3].append(z.copy())
            outputs[4].append(pm_basis@modal_p+equilibrium_pm)
        return tuple(np.stack(value) for value in outputs)  # type: ignore[return-value]

    def rollout(self, initial_state, time_grid, exogenous):
        delta, omega, _, _, _ = self.rollout_augmented(initial_state, time_grid, exogenous)
        return delta, omega

    def rollout_with_pm(self, initial_state, time_grid, exogenous=None):
        delta, omega, _, _, pm = self.rollout_augmented(initial_state, time_grid, exogenous)
        return delta, omega, pm

    def _validation_summary(
        self, samples: list[dict[str, Any]], context: RunContext
    ) -> dict[str, Any]:
        rows, failures = [], 0
        for sample in samples[:context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            static = sample["model_inputs"]["static"]
            try:
                delta_p, omega_p, pm_p = self.rollout_with_pm(
                    static, np.asarray(target["t"][:steps]), {}
                )
            except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
                failures += 1
                continue
            delta_t = np.asarray(target["delta"][:steps], dtype=np.float64)
            omega_t = np.asarray(target["omega"][:steps], dtype=np.float64)
            pm_t = np.asarray(sample["training_context"]["Pm_observed"][:steps])
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
                "pm_rmse": float(np.sqrt(np.mean((pm_p-pm_t)**2))),
                "coi_drift": float(max(
                    np.max(np.abs(delta_p@mass)), np.max(np.abs(omega_p@mass)),
                    np.max(np.abs(np.sum(pm_p-_pm_equilibrium(static), axis=1))),
                )),
            })
        population = max(min(len(samples), context.max_cases), 1)
        if failures or not rows:
            return {
                "standardized_loss": float("inf"), "raw_mean_rmse": float("inf"),
                "raw_median_rmse": float("inf"), "omega_mean_rmse": float("inf"),
                "edge_angle_mean_rmse": float("inf"), "pm_mean_rmse": float("inf"),
                "maximum_coi_drift": float("inf"), "failure_rate": failures/population,
                "source_standardized_loss": {},
            }
        sources = sorted({row["source"] for row in rows})
        source_losses = {
            source: float(np.mean([
                row["standardized_loss"] for row in rows if row["source"] == source
            ])) for source in sources
        }
        return {
            "standardized_loss": float(np.mean(list(source_losses.values()))),
            "raw_mean_rmse": float(np.mean([row["raw_rmse"] for row in rows])),
            "raw_median_rmse": float(np.median([row["raw_rmse"] for row in rows])),
            "omega_mean_rmse": float(np.mean([row["omega_rmse"] for row in rows])),
            "edge_angle_mean_rmse": float(np.mean([row["edge_rmse"] for row in rows])),
            "pm_mean_rmse": float(np.mean([row["pm_rmse"] for row in rows])),
            "maximum_coi_drift": float(np.max([row["coi_drift"] for row in rows])),
            "failure_rate": 0.0,
            "source_standardized_loss": source_losses,
        }

    def fit(self, train, val, run_context):
        if not train or not val:
            raise ValueError("M0-GMD-PM requires nonempty train and validation sets")
        if self.backbone is None:
            self.backbone = GraphModalDiscoveryM0Model()
        try:
            self.backbone._require_fitted()
            backbone_fit = None
        except (RuntimeError, ValueError):
            backbone_fit = self.backbone.fit(train, val, run_context)
        selected_train = train[:run_context.max_cases]
        selected_val = val[:run_context.max_cases]
        self.max_internal_step = run_context.max_internal_step
        self.backbone.max_internal_step = run_context.max_internal_step
        self.validation_scales = M0Model._training_scales(selected_train, len(selected_train))
        source_groups: dict[str, list[dict[str, Any]]] = {}
        for sample in selected_train:
            source_groups.setdefault(self._source(sample), []).append(sample)
        self.validation_scales_by_source = {
            source: M0Model._training_scales(group, len(group))
            for source, group in source_groups.items()
        }
        matrix, targets, weak = self.pm_dynamics.build_weak_system(
            selected_train,
            horizon_steps=run_context.horizon_steps,
            max_fit_steps=run_context.fit_max_steps,
        )
        candidates = self.pm_dynamics.train_candidates(matrix, targets)
        standalone_rows = []
        for candidate in candidates:
            self.pm_dynamics.set_candidate(candidate)
            values = []
            for sample in selected_val:
                predicted = self.pm_dynamics.teacher_forced_rollout(
                    sample, horizon_steps=run_context.horizon_steps,
                    max_step=run_context.max_internal_step,
                )
                observed = np.asarray(
                    sample["training_context"]["Pm_observed"][:len(predicted)]
                )
                values.append(float(np.sqrt(np.mean((predicted-observed)**2))))
            standalone_rows.append({
                **candidate,
                "standalone_pm_mean_rmse": float(np.mean(values)),
                "standalone_pm_median_rmse": float(np.median(values)),
                "standalone_pm_p90_rmse": float(np.quantile(values, 0.9)),
            })
        fallback = standalone_rows[0]
        shortlist = sorted(
            standalone_rows[1:],
            key=lambda row: (
                row["standalone_pm_mean_rmse"], row["standalone_pm_p90_rmse"],
                row["active_coefficients"], row["ridge"],
            ),
        )[:self.standalone_shortlist]
        validation_rows = []
        for candidate in [fallback, *shortlist]:
            self.pm_dynamics.set_candidate(candidate)
            validation_rows.append({**candidate, **self._validation_summary(selected_val, run_context)})
        fallback_validation = validation_rows[0]
        eligible = [
            row for row in validation_rows
            if row["failure_rate"] == 0.0 and np.isfinite(row["standardized_loss"])
            and (
                row["candidate"] == "Pm_clear_ZOH"
                or row["spectral_abscissa"]
                <= -1.0/self.maximum_effective_time_constant+1e-12
            )
        ]
        best = min(
            eligible,
            key=lambda row: (
                row["standardized_loss"], row["raw_mean_rmse"],
                row["active_coefficients"],
            ),
        )
        improvement = (
            fallback_validation["raw_mean_rmse"]-best["raw_mean_rmse"]
        )/max(fallback_validation["raw_mean_rmse"], np.finfo(float).eps)
        promoted = bool(
            best["candidate"] != "Pm_clear_ZOH"
            and improvement >= self.minimum_validation_improvement
            and best["failure_rate"] <= fallback_validation["failure_rate"]
        )
        selected = best if promoted else fallback_validation
        self.pm_dynamics.set_candidate(selected)

        def serializable(row: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value for key, value in row.items()
                if key not in {"state_matrix", "input_matrix"}
            }
        state = np.asarray(self.pm_dynamics.state_matrix)
        eigenvalues = np.linalg.eigvals(state)
        nonzero_real = np.abs(np.real(eigenvalues)) > np.finfo(float).eps
        time_constants = np.full(eigenvalues.shape, np.inf, dtype=np.float64)
        np.divide(
            1.0,
            np.abs(np.real(eigenvalues)),
            out=time_constants,
            where=nonzero_real,
        )
        self.pm_dynamics.diagnostics = {
            "optimizer": "equal_case_equal_horizon_integral_ridge_plus_Hurwitz_projection_plus_validation_rollout",
            "epochs": 0,
            "structure_discovery": True,
            "fixed_priors": [
                "zero_total_power_subspace", "causal_Pm_clear_initialization",
                "predicted_relative_speed_input", "Hurwitz_state_matrix",
            ],
            "discovered_quantities": [
                "cross_node_state_matrix", "cross_node_speed_response",
                "sparse_support", "modal_poles", "effective_time_constants",
            ],
            "weak_system": weak,
            "selected_candidate": serializable(selected),
            "best_candidate_before_gate": serializable(best),
            "fallback_candidate": serializable(fallback_validation),
            "validation_improvement_vs_ZOH": float(improvement),
            "promoted": promoted,
            "eigenvalues_real": np.real(eigenvalues).tolist(),
            "eigenvalues_imag": np.imag(eigenvalues).tolist(),
            "effective_time_constants": time_constants.tolist(),
            "candidate_table": [serializable(row) for row in standalone_rows],
            "validation_shortlist": [serializable(row) for row in validation_rows],
            "future_observed_Pm_consumed_at_deployment": False,
            "common_frequency_available": False,
            "maximum_effective_time_constant": self.maximum_effective_time_constant,
            "time_constant_bound_source": (
                "registered_ANDES_TGOV1N_T3_2.1s_with_"
                f"{self.maximum_effective_time_constant:g}s_conservative_cap"
            ),
            "machine_level_governor_recovery_claim": False,
            "labels_used_for_fit_or_selection": False,
            "test_ids_read": [],
        }
        self.diagnostics = {
            **self.pm_dynamics.diagnostics,
            "backbone_fitted_inside_composite": backbone_fit is not None,
            "backbone_method": self.backbone.method_name,
        }
        return FitResult(
            method=self.method_name, status="completed",
            train_case_ids=tuple(sample["event_id"] for sample in selected_train),
            val_case_ids=tuple(sample["event_id"] for sample in selected_val),
            best_validation_loss=float(selected["standardized_loss"]),
            details=self.diagnostics,
        )

    def save(self, path):
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backbone_path = path.with_name(path.stem+"_backbone.npz")
        self.backbone.save(backbone_path)
        basis, state, inputs = self.pm_dynamics.require_fitted()
        np.savez_compressed(
            path, basis=basis, state_matrix=state, input_matrix=inputs,
            max_internal_step=self.max_internal_step,
            maximum_effective_time_constant=self.maximum_effective_time_constant,
            backbone_file=backbone_path.name,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            backbone = GraphModalDiscoveryM0Model.load(
                path.with_name(str(data["backbone_file"]))
            )
            model = cls(
                backbone=backbone,
                pm_dynamics=GraphModalPmDynamics(
                    basis=data["basis"], state_matrix=data["state_matrix"],
                    input_matrix=data["input_matrix"],
                ),
                max_internal_step=float(data["max_internal_step"]),
                maximum_effective_time_constant=(
                    float(data["maximum_effective_time_constant"])
                    if "maximum_effective_time_constant" in data.files else 5.0
                ),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get(
                "diagnostics", {}
            )
            model.pm_dynamics.diagnostics = model.diagnostics.copy()
        return model

    def convergence_audit(self, initial_state, time_grid, exogenous=None):
        trajectories = [
            self.rollout_augmented(
                initial_state, time_grid, exogenous or {},
                max_step=self.max_internal_step/divisor,
            ) for divisor in (1, 2, 4)
        ]
        errors = []
        for coarse, fine in zip(trajectories[:-1], trajectories[1:]):
            errors.append(float(np.sqrt(sum(
                np.mean((coarse[index]-fine[index])**2) for index in range(5)
            ))))
        return {
            "dt_vs_dt2": errors[0], "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0]/max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0]+1e-12),
        }

    def physics_audit(self):
        try:
            (_, _, _, _, _), (basis, state, inputs) = self._require_fitted()
        except (RuntimeError, ValueError):
            return {"status": "not_fitted"}
        maximum_sum = float(np.max(np.abs(np.sum(basis, axis=0))))
        orthogonality = float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1]))))
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
            else 1.0/max(abs(spectral_abscissa), np.finfo(float).eps)
        )
        finite = bool(np.isfinite(state).all() and np.isfinite(inputs).all())
        hurwitz_and_bounded = bool(
            spectral_abscissa < 0
            and maximum_time_constant <= self.maximum_effective_time_constant+1e-12
        )
        passed = bool(
            finite and maximum_sum < 1e-10 and orthogonality < 1e-10
            and (exact_zoh or hurwitz_and_bounded)
            and self.backbone.physics_audit().get("status") == "pass"
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
            "time_constant_bound_pass": exact_zoh or hurwitz_and_bounded,
            "finite_coefficients": finite,
            "common_frequency_available": False,
            "machine_level_governor_recovery_claim": False,
        }
