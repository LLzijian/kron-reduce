from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .m0 import M0Model, _pm_equilibrium, _trapz


class ThirdOrderM0Model(DynamicsModel):
    """Causal third-order M0 with a dynamic electrical-power state.

    This is a reduced third-order closure, not a claim that the canonical data
    expose the classical transient-voltage state of a synchronous machine.  It
    evolves one balanced nodal electrical-power state in addition to angle and
    speed::

        delta_dot = omega
        m * omega_dot = Delta Pm - D * omega - pe
        tau * pe_dot = -pe + B @ (K * Delta sin(eta) + C * Delta omega_e)

    ``pe(0)`` is initialized from the clearing angle and speed using the same
    quasi-steady expression on the right-hand side.  No measured electrical
    power, voltage, extended state, or future target is used at rollout time.
    """

    method_name = "M0-3"
    DEFAULT_TAU_CANDIDATES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

    def __init__(
        self,
        damping: np.ndarray | None = None,
        coupling: np.ndarray | None = None,
        edge_damping: np.ndarray | None = None,
        tau: float | None = None,
        *,
        selected_ridge: float | None = None,
        max_internal_step: float = 0.01,
        tau_candidates: tuple[float, ...] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ):
        self.damping = None if damping is None else np.asarray(damping, dtype=np.float64)
        self.coupling = None if coupling is None else np.asarray(coupling, dtype=np.float64)
        self.edge_damping = (
            None if edge_damping is None else np.asarray(edge_damping, dtype=np.float64)
        )
        self.tau = None if tau is None else float(tau)
        self.selected_ridge = selected_ridge
        self.max_internal_step = float(max_internal_step)
        self.tau_candidates = tuple(tau_candidates or self.DEFAULT_TAU_CANDIDATES)
        if any(not np.isfinite(value) or value <= 0 for value in self.tau_candidates):
            raise ValueError("third-order tau candidates must be finite and positive")
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
        self.diagnostics = diagnostics or {}

    @staticmethod
    def _filtered_feature(t: np.ndarray, values: np.ndarray, tau: float) -> np.ndarray:
        """Stable exact-decay/trapezoidal-input first-order filter."""
        t = np.asarray(t, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        filtered = np.empty_like(values)
        filtered[0] = values[0]
        for index in range(len(t) - 1):
            dt = float(t[index + 1] - t[index])
            if dt <= 0:
                raise ValueError("filter time grid must be strictly increasing")
            decay = np.exp(-dt / tau)
            filtered[index + 1] = (
                decay * filtered[index]
                + (1.0 - decay) * 0.5 * (values[index] + values[index + 1])
            )
        return filtered

    @classmethod
    def build_weak_system(
        cls,
        samples: list[dict[str, Any]],
        *,
        tau: float,
        softening_beta: float = 0.0,
        windows: tuple[int, ...] = (1, 5, 20),
        max_cases: int | None = None,
        horizon_steps: int = 100,
        max_fit_steps: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        selected = samples if max_cases is None else samples[:max_cases]
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
            delta_pm = pm - _pm_equilibrium(static)
            limit = min(
                len(t),
                int(max_fit_steps) if int(max_fit_steps) > 0
                else max(horizon_steps * 5, max(windows, default=1) + 1),
            )
            eta = delta[:limit] @ incidence
            eta0 = incidence.T @ equilibrium
            delta_sin = np.sin(eta) - np.sin(eta0)
            edge_speed = omega[:limit] @ incidence
            angle_feature = delta_sin - softening_beta*delta_sin**3
            filtered_sin = cls._filtered_feature(t[:limit], angle_feature, tau)
            filtered_speed = cls._filtered_feature(t[:limit], edge_speed, tau)
            projector = np.eye(mass.size) - np.outer(mass, np.ones(mass.size)) / np.sum(mass)
            designs = np.stack([
                np.column_stack((
                    projector @ np.diag(omega[index]),
                    incidence * filtered_sin[index],
                    incidence * filtered_speed[index],
                ))
                for index in range(limit)
            ])
            for width in sorted(set(int(value) for value in windows if int(value) > 0)):
                if width >= limit:
                    continue
                for start in range(limit - width):
                    stop = start + width
                    segment = slice(start, stop + 1)
                    rows.append(_trapz(designs[segment], t[segment]))
                    targets.append(
                        _trapz(delta_pm[segment], t[segment])
                        - mass * (omega[stop] - omega[start])
                    )
        if not rows:
            raise ValueError("no eligible third-order weak-form windows")
        matrix = np.concatenate(rows)
        vector = np.concatenate(targets)
        return matrix, vector, {
            "n_scalar_equations": int(matrix.shape[0]),
            "n_parameters": int(matrix.shape[1]),
            "n_integration_windows": int(len(rows)),
            "candidate_tau": float(tau),
            "candidate_softening_beta": float(softening_beta),
            "maximum_abs_delta_sin": float(max(
                np.max(np.abs(
                    np.sin(np.asarray(sample["training_context"]["targets"]["delta"], dtype=np.float64) @ np.asarray(sample["model_inputs"]["static"]["incidence"], dtype=np.float64))
                    - np.sin(np.asarray(sample["model_inputs"]["static"]["incidence"], dtype=np.float64).T @ np.asarray(sample["model_inputs"]["static"]["equilibrium_delta"], dtype=np.float64).reshape(-1))
                )) for sample in selected
            )),
        }

    @staticmethod
    def _training_scales(samples: list[dict[str, Any]], max_cases: int) -> dict[str, float]:
        omega_values, edge_values = [], []
        for sample in samples[:max_cases]:
            target = sample["training_context"]["targets"]
            incidence = np.asarray(sample["model_inputs"]["static"]["incidence"])
            omega_values.append(np.asarray(target["omega"]))
            edge_values.append(np.asarray(target["delta"]) @ incidence)
        return {
            "omega": max(float(np.std(np.concatenate(omega_values))), 1e-8),
            "edge_angle": max(float(np.std(np.concatenate(edge_values))), 1e-8),
        }

    def _validation_summary(
        self, samples: list[dict[str, Any]], context: RunContext
    ) -> dict[str, float]:
        standardized, raw, omega_rmse, edge_rmse, drift, failures = [], [], [], [], [], 0
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            static, time_grid, exogenous = sample_rollout_inputs(sample)
            try:
                delta_p, omega_p = self.rollout(
                    static,
                    time_grid[:steps],
                    {"Pm": exogenous["Pm"][:steps], "interpolation": exogenous["interpolation"]},
                )
            except (FloatingPointError, ValueError, OverflowError):
                failures += 1
                continue
            delta_t = np.asarray(target["delta"][:steps])
            omega_t = np.asarray(target["omega"][:steps])
            incidence = np.asarray(static["incidence"])
            edge_error = (delta_p - delta_t) @ incidence
            omega_error = omega_p - omega_t
            standardized.append(
                np.mean((edge_error / self.validation_scales["edge_angle"]) ** 2)
                + np.mean((omega_error / self.validation_scales["omega"]) ** 2)
            )
            raw.append(float(np.sqrt(np.mean(np.concatenate((
                (delta_p - delta_t).ravel(), omega_error.ravel()
            )) ** 2))))
            omega_rmse.append(float(np.sqrt(np.mean(omega_error**2))))
            edge_rmse.append(float(np.sqrt(np.mean(edge_error**2))))
            mass = np.asarray(static["mass"])
            drift.append(float(max(np.max(np.abs(delta_p @ mass)), np.max(np.abs(omega_p @ mass)))))
        if failures or not standardized:
            return {
                "standardized_loss": float("inf"),
                "raw_mean_rmse": float("inf"),
                "raw_median_rmse": float("inf"),
                "omega_mean_rmse": float("inf"),
                "edge_angle_mean_rmse": float("inf"),
                "maximum_coi_drift": float("inf"),
                "failure_rate": failures / max(min(len(samples), context.max_cases), 1),
            }
        return {
            "standardized_loss": float(np.mean(standardized)),
            "raw_mean_rmse": float(np.mean(raw)),
            "raw_median_rmse": float(np.median(raw)),
            "omega_mean_rmse": float(np.mean(omega_rmse)),
            "edge_angle_mean_rmse": float(np.mean(edge_rmse)),
            "maximum_coi_drift": float(np.max(drift)),
            "failure_rate": 0.0,
        }

    def fit(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-3 requires nonempty train and validation sets")
        self.validation_scales = self._training_scales(train, run_context.max_cases)
        self.max_internal_step = run_context.max_internal_step
        n_nodes = np.asarray(train[0]["model_inputs"]["static"]["mass"]).size
        n_edges = np.asarray(train[0]["model_inputs"]["static"]["incidence"]).shape[1]
        candidates = []
        for tau in self.tau_candidates:
            matrix, vector, system_details = self.build_weak_system(
                train,
                tau=tau,
                windows=run_context.integral_windows,
                max_cases=run_context.max_cases,
                horizon_steps=run_context.horizon_steps,
                max_fit_steps=run_context.fit_max_steps,
            )
            for ridge in run_context.ridge_values:
                theta, solver_details = M0Model._solve_nonnegative_ridge(matrix, vector, ridge)
                self.damping = theta[:n_nodes]
                self.coupling = theta[n_nodes:n_nodes + n_edges]
                self.edge_damping = theta[n_nodes + n_edges:]
                self.tau = float(tau)
                summary = self._validation_summary(val, run_context)
                candidates.append((
                    summary["standardized_loss"], summary["raw_mean_rmse"],
                    float(tau), float(ridge), theta.copy(), system_details,
                    solver_details, summary,
                ))
        selected = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
        loss, _, tau, ridge, theta, system_details, solver_details, summary = selected
        self.damping = theta[:n_nodes]
        self.coupling = theta[n_nodes:n_nodes + n_edges]
        self.edge_damping = theta[n_nodes + n_edges:]
        self.tau = tau
        self.selected_ridge = ridge
        self.diagnostics = {
            **system_details,
            **solver_details,
            **{f"validation_{key}": value for key, value in summary.items()},
            "selected_tau": tau,
            "selected_ridge": ridge,
            "tau_candidates": list(self.tau_candidates),
            "epochs": 0,
            "optimizer": "single_bounded_third_order_weak_form_ridge",
            "third_state": "balanced_nodal_electrical_power_memory",
            "classical_transient_voltage_claim": False,
            "initial_state_policy": "clearing_quasi_steady_from_delta_omega_only",
            "uses_measured_Pe_or_voltage": False,
            "raw_parameters": theta.tolist(),
        }
        return FitResult(
            method=self.method_name,
            status="completed",
            train_case_ids=tuple(sample["event_id"] for sample in train[: run_context.max_cases]),
            val_case_ids=tuple(sample["event_id"] for sample in val[: run_context.max_cases]),
            best_validation_loss=float(loss),
            details=self.diagnostics,
        )

    def initial_electrical_power(
        self, delta: np.ndarray, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.coupling is None or self.edge_damping is None:
            raise RuntimeError("M0-3 is not fitted")
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
        delta_sin = np.sin(incidence.T @ delta) - np.sin(incidence.T @ equilibrium)
        edge_speed = incidence.T @ omega
        return incidence @ (self.coupling * delta_sin + self.edge_damping * edge_speed)

    def vector_field_augmented(
        self,
        delta: np.ndarray,
        omega: np.ndarray,
        electrical_power: np.ndarray,
        pm: np.ndarray,
        static: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.damping is None or self.tau is None:
            raise RuntimeError("M0-3 is not fitted")
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        delta_pm = np.asarray(pm, dtype=np.float64).reshape(-1) - _pm_equilibrium(static)
        target_electrical = self.initial_electrical_power(delta, omega, static)
        force = coi_project_power(
            delta_pm - self.damping * omega - electrical_power, mass
        )
        pe_dot = coi_project_power(
            (-electrical_power + target_electrical) / self.tau, mass
        )
        return omega, force / mass, pe_dot

    @staticmethod
    def _interpolation_mode(pm: np.ndarray, requested: str) -> str:
        if requested not in {"auto", "zoh", "linear"}:
            raise ValueError("Pm interpolation must be auto, zoh, or linear")
        if requested != "auto":
            return requested
        return "zoh" if np.allclose(pm, pm[[0]], rtol=0.0, atol=1e-14) else "linear"

    def rollout(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
        *,
        max_step: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        electrical = coi_project_power(
            self.initial_electrical_power(delta, omega, initial_state), mass
        )
        output_delta, output_omega = [delta.copy()], [omega.copy()]

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
                q0 = substep / substeps
                qh = (substep + 0.5) / substeps
                q1 = (substep + 1.0) / substeps
                k1 = self.vector_field_augmented(delta, omega, electrical, input_at(index, q0), initial_state)
                k2 = self.vector_field_augmented(
                    delta + 0.5*h*k1[0], omega + 0.5*h*k1[1], electrical + 0.5*h*k1[2],
                    input_at(index, qh), initial_state,
                )
                k3 = self.vector_field_augmented(
                    delta + 0.5*h*k2[0], omega + 0.5*h*k2[1], electrical + 0.5*h*k2[2],
                    input_at(index, qh), initial_state,
                )
                k4 = self.vector_field_augmented(
                    delta + h*k3[0], omega + h*k3[1], electrical + h*k3[2],
                    input_at(index, q1), initial_state,
                )
                delta = coi_project(delta + h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0, mass)
                omega = coi_project(omega + h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0, mass)
                electrical = coi_project_power(
                    electrical + h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0, mass
                )
            if not (
                np.isfinite(delta).all()
                and np.isfinite(omega).all()
                and np.isfinite(electrical).all()
            ):
                raise FloatingPointError(f"nonfinite M0-3 rollout at step {index + 1}")
            output_delta.append(delta.copy())
            output_omega.append(omega.copy())
        return np.stack(output_delta), np.stack(output_omega)

    def convergence_audit(
        self,
        initial_state: dict[str, np.ndarray],
        time_grid: np.ndarray,
        exogenous: dict[str, np.ndarray],
    ) -> dict[str, float | bool]:
        trajectories = [
            self.rollout(initial_state, time_grid, exogenous, max_step=self.max_internal_step/divisor)
            for divisor in (1, 2, 4)
        ]
        errors = []
        for coarse, fine in zip(trajectories[:-1], trajectories[1:]):
            errors.append(float(np.sqrt(
                np.mean((coarse[0]-fine[0])**2) + np.mean((coarse[1]-fine[1])**2)
            )))
        return {
            "dt_vs_dt2": errors[0],
            "dt2_vs_dt4": errors[1],
            "error_ratio": errors[0] / max(errors[1], np.finfo(float).eps),
            "convergent": bool(errors[1] <= errors[0] + 1e-12),
        }

    def save(self, path: str | Path) -> None:
        if self.damping is None or self.coupling is None or self.edge_damping is None or self.tau is None:
            raise RuntimeError("cannot save an unfitted M0-3 model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            damping=self.damping,
            coupling=self.coupling,
            edge_damping=self.edge_damping,
            tau=self.tau,
            selected_ridge=np.nan if self.selected_ridge is None else self.selected_ridge,
            max_internal_step=self.max_internal_step,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ThirdOrderM0Model":
        path = Path(path)
        with np.load(path) as data:
            ridge = float(data["selected_ridge"])
            model = cls(
                data["damping"], data["coupling"], data["edge_damping"], float(data["tau"]),
                selected_ridge=None if np.isnan(ridge) else ridge,
                max_internal_step=float(data["max_internal_step"]),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get("diagnostics", {})
        return model

    def physics_audit(self) -> dict[str, float | str | bool]:
        if self.damping is None or self.coupling is None or self.edge_damping is None or self.tau is None:
            return {"status": "not_fitted"}
        nonnegative = bool(
            np.all(self.damping >= 0)
            and np.all(self.coupling >= 0)
            and np.all(self.edge_damping >= 0)
        )
        return {
            "status": "pass" if nonnegative and self.tau > 0 else "fail",
            "third_state": "balanced_nodal_electrical_power_memory",
            "classical_transient_voltage_claim": False,
            "uses_measured_Pe_or_voltage": False,
            "future_state_initialization": False,
            "internal_force_balance_by_construction": True,
            "nonnegative_parameters": nonnegative,
            "tau_positive": bool(self.tau > 0),
            "selected_tau": float(self.tau),
        }


class StructuredThirdOrderM0Model(ThirdOrderM0Model):
    """Identifiable third-order M0 with graph-aligned edge dissipation.

    Free nodal damping and free complete-graph edge damping form a redundant
    parameterization on the COI quotient.  This variant removes that ambiguity
    by requiring ``C_e = gamma * K_e``.  For fixed ``tau`` and ``gamma`` the
    weak problem remains linear and has the same number of fitted coefficients
    as second-order M0.
    """

    method_name = "M0-3S"
    DEFAULT_GAMMA_CANDIDATES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)

    def __init__(
        self,
        damping: np.ndarray | None = None,
        coupling: np.ndarray | None = None,
        tau: float | None = None,
        gamma: float | None = None,
        softening_beta: float = 0.0,
        *,
        selected_ridge: float | None = None,
        max_internal_step: float = 0.01,
        tau_candidates: tuple[float, ...] | None = None,
        gamma_candidates: tuple[float, ...] | None = None,
        beta_candidates: tuple[float, ...] | None = None,
        case_weighting: str = "equal_scalar_rows",
        diagnostics: dict[str, Any] | None = None,
    ):
        self.gamma = None if gamma is None else float(gamma)
        self.gamma_candidates = tuple(
            gamma_candidates or self.DEFAULT_GAMMA_CANDIDATES
        )
        if any(not np.isfinite(value) or value < 0 for value in self.gamma_candidates):
            raise ValueError("structured M0-3 gamma candidates must be finite and nonnegative")
        self.softening_beta = float(softening_beta)
        self.beta_candidates = tuple(beta_candidates or (0.0,))
        if any(not np.isfinite(value) or value < 0 for value in self.beta_candidates):
            raise ValueError("structured M0-3 beta candidates must be finite and nonnegative")
        if case_weighting not in {"equal_scalar_rows", "equal_case"}:
            raise ValueError("case_weighting must be equal_scalar_rows or equal_case")
        self.case_weighting = case_weighting
        edge_damping = (
            None if coupling is None or gamma is None
            else float(gamma) * np.asarray(coupling, dtype=np.float64)
        )
        super().__init__(
            damping, coupling, edge_damping, tau,
            selected_ridge=selected_ridge,
            max_internal_step=max_internal_step,
            tau_candidates=tau_candidates,
            diagnostics=diagnostics,
        )

    def fit(
        self,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        run_context: RunContext,
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0-3S requires nonempty train and validation sets")
        self.validation_scales = self._training_scales(train, run_context.max_cases)
        self.max_internal_step = run_context.max_internal_step
        n_nodes = np.asarray(train[0]["model_inputs"]["static"]["mass"]).size
        n_edges = np.asarray(train[0]["model_inputs"]["static"]["incidence"]).shape[1]
        candidates = []
        validation_max_abs_delta_sin = max(
            float(np.max(np.abs(
                np.sin(np.asarray(sample["training_context"]["targets"]["delta"]) @ np.asarray(sample["model_inputs"]["static"]["incidence"]))
                - np.sin(np.asarray(sample["model_inputs"]["static"]["incidence"]).T @ np.asarray(sample["model_inputs"]["static"]["equilibrium_delta"]).reshape(-1))
            )))
            for sample in val[:run_context.max_cases]
        )
        for tau in self.tau_candidates:
            for beta in self.beta_candidates:
                if self.case_weighting == "equal_scalar_rows":
                    full_matrix, vector, system_details = self.build_weak_system(
                        train,
                        tau=tau,
                        softening_beta=beta,
                        windows=run_context.integral_windows,
                        max_cases=run_context.max_cases,
                        horizon_steps=run_context.horizon_steps,
                        max_fit_steps=run_context.fit_max_steps,
                    )
                else:
                    matrices, vectors = [], []
                    integration_windows = 0
                    maximum_abs_delta_sin = 0.0
                    for sample in train[:run_context.max_cases]:
                        case_matrix, case_vector, case_details = self.build_weak_system(
                            [sample], tau=tau,
                            softening_beta=beta,
                            windows=run_context.integral_windows,
                            max_cases=1,
                            horizon_steps=run_context.horizon_steps,
                            max_fit_steps=run_context.fit_max_steps,
                        )
                        weight = 1.0/np.sqrt(len(case_vector))
                        matrices.append(weight*case_matrix)
                        vectors.append(weight*case_vector)
                        integration_windows += int(case_details["n_integration_windows"])
                        maximum_abs_delta_sin = max(maximum_abs_delta_sin, float(case_details["maximum_abs_delta_sin"]))
                    full_matrix, vector = np.concatenate(matrices), np.concatenate(vectors)
                    system_details = {
                        "n_scalar_equations": int(full_matrix.shape[0]),
                        "n_parameters": int(full_matrix.shape[1]),
                        "n_integration_windows": int(integration_windows),
                        "candidate_tau": float(tau),
                        "candidate_softening_beta": float(beta),
                        "maximum_abs_delta_sin": maximum_abs_delta_sin,
                        "case_weighting": "equal_case_total_squared_row_weight",
                    }
                audited_max_abs_delta_sin = max(
                    float(system_details["maximum_abs_delta_sin"]),
                    validation_max_abs_delta_sin,
                )
                monotonic_margin = 1.0 - 3.0*float(beta)*audited_max_abs_delta_sin**2
                if monotonic_margin < -1e-12:
                    continue
                node_matrix = full_matrix[:, :n_nodes]
                angle_matrix = full_matrix[:, n_nodes:n_nodes+n_edges]
                speed_matrix = full_matrix[:, n_nodes+n_edges:]
                for gamma in self.gamma_candidates:
                    matrix = np.column_stack((node_matrix, angle_matrix + gamma*speed_matrix))
                    for ridge in run_context.ridge_values:
                        theta, solver_details = M0Model._solve_nonnegative_ridge(
                            matrix, vector, ridge
                        )
                        self.damping = theta[:n_nodes]
                        self.coupling = theta[n_nodes:]
                        self.gamma = float(gamma)
                        self.softening_beta = float(beta)
                        self.edge_damping = self.gamma*self.coupling
                        self.tau = float(tau)
                        summary = self._validation_summary(val, run_context)
                        candidates.append({
                            "standardized_loss": summary["standardized_loss"],
                            "raw_mean_rmse": summary["raw_mean_rmse"],
                            "tau": float(tau),
                            "gamma": float(gamma),
                            "beta": float(beta),
                            "monotonic_margin": float(monotonic_margin),
                            "ridge": float(ridge),
                            "theta": theta.copy(),
                            "system": system_details,
                            "solver": solver_details,
                            "validation": summary,
                        })
        if not candidates:
            raise ValueError("no structured M0-3 candidates pass the softening monotonicity audit")
        selected = min(
            candidates,
            key=lambda item: (
                item["standardized_loss"], item["raw_mean_rmse"],
                item["tau"], item["gamma"], item["beta"], item["ridge"],
            ),
        )
        theta = selected["theta"]
        self.damping = theta[:n_nodes]
        self.coupling = theta[n_nodes:]
        self.gamma = selected["gamma"]
        self.softening_beta = selected["beta"]
        self.edge_damping = self.gamma*self.coupling
        self.tau = selected["tau"]
        self.selected_ridge = selected["ridge"]
        candidate_table = [
            {
                "tau": item["tau"],
                "gamma": item["gamma"],
                "softening_beta": item["beta"],
                "softening_monotonic_margin": item["monotonic_margin"],
                "ridge": item["ridge"],
                "standardized_loss": item["standardized_loss"],
                "raw_mean_rmse": item["raw_mean_rmse"],
                "failure_rate": item["validation"]["failure_rate"],
                "weak_residual_rmse": item["solver"]["weak_residual_rmse"],
                "condition_number": item["solver"]["matrix_condition_number"],
                "effective_rank": item["solver"]["effective_rank"],
                "damping_median": float(np.median(item["theta"][:n_nodes])),
                "damping_min": float(np.min(item["theta"][:n_nodes])),
            }
            for item in candidates
        ]
        validation = selected["validation"]
        self.diagnostics = {
            **selected["system"],
            **selected["solver"],
            **{f"validation_{key}": value for key, value in validation.items()},
            "n_parameters": int(n_nodes+n_edges),
            "selected_tau": self.tau,
            "selected_gamma": self.gamma,
            "selected_softening_beta": self.softening_beta,
            "softening_monotonic_margin": selected["monotonic_margin"],
            "selected_ridge": self.selected_ridge,
            "tau_candidates": list(self.tau_candidates),
            "gamma_candidates": list(self.gamma_candidates),
            "beta_candidates": list(self.beta_candidates),
            "candidate_table": candidate_table,
            "epochs": 0,
            "optimizer": "single_bounded_graph_aligned_third_order_weak_form_ridge",
            "third_state": "balanced_nodal_electrical_power_memory",
            "edge_damping_constraint": "C_e_equals_gamma_times_K_e",
            "large_signal_edge_feature": "delta_sin_minus_beta_times_delta_sin_cubed",
            "same_support_stiffness_and_dissipation": True,
            "case_weighting": self.case_weighting,
            "classical_transient_voltage_claim": False,
            "initial_state_policy": "clearing_quasi_steady_from_delta_omega_only",
            "uses_measured_Pe_or_voltage": False,
            "raw_parameters": theta.tolist(),
        }
        return FitResult(
            method=self.method_name,
            status="completed",
            train_case_ids=tuple(sample["event_id"] for sample in train[:run_context.max_cases]),
            val_case_ids=tuple(sample["event_id"] for sample in val[:run_context.max_cases]),
            best_validation_loss=float(selected["standardized_loss"]),
            details=self.diagnostics,
        )

    def initial_electrical_power(
        self, delta: np.ndarray, omega: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.coupling is None or self.edge_damping is None:
            raise RuntimeError(f"{self.method_name} is not fitted")
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
        delta_sin = np.sin(incidence.T @ delta) - np.sin(incidence.T @ equilibrium)
        angle_feature = delta_sin - self.softening_beta*delta_sin**3
        edge_speed = incidence.T @ omega
        return incidence @ (
            self.coupling*angle_feature + self.edge_damping*edge_speed
        )

    def save(self, path: str | Path) -> None:
        if self.damping is None or self.coupling is None or self.tau is None or self.gamma is None:
            raise RuntimeError("cannot save an unfitted M0-3S model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            damping=self.damping,
            coupling=self.coupling,
            tau=self.tau,
            gamma=self.gamma,
            softening_beta=self.softening_beta,
            case_weighting=np.asarray(self.case_weighting),
            selected_ridge=np.nan if self.selected_ridge is None else self.selected_ridge,
            max_internal_step=self.max_internal_step,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "StructuredThirdOrderM0Model":
        path = Path(path)
        with np.load(path) as data:
            ridge = float(data["selected_ridge"])
            model = cls(
                data["damping"], data["coupling"], float(data["tau"]),
                float(data["gamma"]),
                float(data["softening_beta"]) if "softening_beta" in data else 0.0,
                selected_ridge=None if np.isnan(ridge) else ridge,
                max_internal_step=float(data["max_internal_step"]),
                case_weighting=(
                    str(data["case_weighting"])
                    if "case_weighting" in data else "equal_scalar_rows"
                ),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get("diagnostics", {})
        return model

    def physics_audit(self) -> dict[str, float | str | bool]:
        audit = super().physics_audit()
        if audit["status"] == "not_fitted":
            return audit
        support_match = bool(np.allclose(
            self.edge_damping, self.gamma*self.coupling, rtol=0.0, atol=1e-14
        ))
        audit.update({
            "method": self.method_name,
            "edge_damping_constraint": "C_e_equals_gamma_times_K_e",
            "same_support_stiffness_and_dissipation": support_match,
            "case_weighting": self.case_weighting,
            "gamma_nonnegative": bool(self.gamma >= 0),
            "selected_gamma": float(self.gamma),
            "softening_beta_nonnegative": bool(self.softening_beta >= 0),
            "selected_softening_beta": float(self.softening_beta),
            "softening_monotonic_margin": float(self.diagnostics.get("softening_monotonic_margin", 1.0)),
            "status": "pass" if (
                audit["status"] == "pass"
                and support_match
                and self.gamma >= 0
                and self.softening_beta >= 0
                and float(self.diagnostics.get("softening_monotonic_margin", 1.0)) >= -1e-12
            ) else "fail",
        })
        return audit


class LargeSignalThirdOrderM0Model(StructuredThirdOrderM0Model):
    """Third-order M0 with an audited large-signal coupling-softening scale."""

    method_name = "M0-3L"
    DEFAULT_BETA_CANDIDATES = (0.0, 0.05, 0.1, 0.2)

    def __init__(self, *args: Any, beta_candidates: tuple[float, ...] | None = None, **kwargs: Any):
        super().__init__(
            *args,
            beta_candidates=beta_candidates or self.DEFAULT_BETA_CANDIDATES,
            **kwargs,
        )
