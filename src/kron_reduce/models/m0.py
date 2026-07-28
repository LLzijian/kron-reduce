from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from kron_reduce.physics.coordinates import coi_project_power

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .integration import integration_convergence_audit, rk4_rollout


def _trapz(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = np.diff(np.asarray(t, dtype=np.float64))
    return np.sum(0.5 * (values[:-1] + values[1:]) * dt.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0)


def _pm_equilibrium(static: dict[str, np.ndarray]) -> np.ndarray:
    if "Pm_equilibrium" not in static:
        raise KeyError("Pm_equilibrium is required; Pm0 is the clearing value, not equilibrium")
    return np.asarray(static["Pm_equilibrium"], dtype=np.float64).reshape(-1)


class M0Model(DynamicsModel):
    method_name = "M0"

    def __init__(
        self,
        damping: np.ndarray | None = None,
        coupling: np.ndarray | None = None,
        *,
        selected_ridge: float | None = None,
        max_internal_step: float = 0.01,
        diagnostics: dict[str, Any] | None = None,
    ):
        self.damping = None if damping is None else np.asarray(damping, dtype=np.float64)
        self.coupling = None if coupling is None else np.asarray(coupling, dtype=np.float64)
        self.selected_ridge = selected_ridge
        self.max_internal_step = float(max_internal_step)
        self.diagnostics = diagnostics or {}
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}

    @staticmethod
    def _instantaneous_design(
        delta: np.ndarray, omega: np.ndarray, equilibrium: np.ndarray,
        incidence: np.ndarray, mass: np.ndarray,
    ) -> np.ndarray:
        n_nodes, n_edges = omega.size, incidence.shape[1]
        design = np.zeros((n_nodes, n_nodes + n_edges), dtype=np.float64)
        projector = np.eye(n_nodes) - np.outer(mass, np.ones(n_nodes)) / np.sum(mass)
        design[:, :n_nodes] = projector @ np.diag(omega)
        eta = delta @ incidence
        eta0 = incidence.T @ equilibrium
        design[:, n_nodes:] = incidence * (np.sin(eta) - np.sin(eta0))
        return design

    @classmethod
    def build_weak_system(
        cls,
        samples: list[dict[str, Any]],
        *,
        windows: tuple[int, ...] = (1, 5, 20),
        max_cases: int | None = None,
        horizon_steps: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
        rows, targets = [], []
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
            limit = min(len(t), max(horizon_steps * 5, max(windows, default=1) + 1))
            designs = np.stack([
                cls._instantaneous_design(delta[k], omega[k], equilibrium, incidence, mass)
                for k in range(limit)
            ])
            for width in sorted(set(int(w) for w in windows if int(w) > 0)):
                if width >= limit:
                    continue
                for start in range(limit - width):
                    stop = start + width
                    segment = slice(start, stop + 1)
                    integrated_design = _trapz(designs[segment], t[segment])
                    rhs = _trapz(delta_pm[segment], t[segment]) - mass * (
                        omega[stop] - omega[start]
                    )
                    rows.append(integrated_design)
                    targets.append(rhs)
        if not rows:
            raise ValueError("no eligible weak-form integration windows")
        matrix, vector = np.concatenate(rows), np.concatenate(targets)
        return matrix, vector, {
            "n_scalar_equations": int(matrix.shape[0]),
            "n_parameters": int(matrix.shape[1]),
            "n_integration_windows": int(len(rows)),
        }

    @staticmethod
    def _solve_nonnegative_ridge(matrix: np.ndarray, vector: np.ndarray, ridge: float):
        scale = np.sqrt(np.mean(matrix**2, axis=0))
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        normalized = matrix / scale
        augmented = np.vstack((normalized, np.sqrt(ridge) * np.eye(normalized.shape[1])))
        augmented_y = np.concatenate((vector, np.zeros(normalized.shape[1])))
        result = lsq_linear(augmented, augmented_y, bounds=(0.0, np.inf), max_iter=2000)
        theta = result.x / scale
        residual = matrix @ theta - vector
        return theta, {
            "solver_success": bool(result.success),
            "solver_iterations": int(result.nit),
            "matrix_condition_number": float(np.linalg.cond(normalized)),
            "effective_rank": int(np.linalg.matrix_rank(normalized)),
            "weak_residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "column_scale_min": float(np.min(scale)),
            "column_scale_max": float(np.max(scale)),
        }

    @staticmethod
    def _training_scales(samples: list[dict[str, Any]], max_cases: int) -> dict[str, float]:
        omega_values, edge_values = [], []
        for sample in samples[:max_cases]:
            target = sample["training_context"]["targets"]
            omega_values.append(np.asarray(target["omega"]))
            incidence = np.asarray(sample["model_inputs"]["static"]["incidence"])
            edge_values.append(np.asarray(target["delta"]) @ incidence)
        return {
            "omega": max(float(np.std(np.concatenate(omega_values))), 1e-8),
            "edge_angle": max(float(np.std(np.concatenate(edge_values))), 1e-8),
        }

    def validation_summary(self, samples: list[dict[str, Any]], context: RunContext) -> dict[str, float]:
        losses, raw_rmse, coi_drift = [], [], []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            static, sample_t, exogenous = sample_rollout_inputs(sample)
            dp, wp = self.rollout(
                static,
                sample_t[:steps],
                {
                    "Pm": exogenous["Pm"][:steps],
                    "interpolation": exogenous["interpolation"],
                },
            )
            incidence = np.asarray(static["incidence"])
            edge_error = (dp - target["delta"][:steps]) @ incidence
            omega_error = wp - target["omega"][:steps]
            losses.append(
                np.mean((edge_error / self.validation_scales["edge_angle"]) ** 2)
                + np.mean((omega_error / self.validation_scales["omega"]) ** 2)
            )
            raw_error = np.concatenate(((dp-target["delta"][:steps]).ravel(), (wp-target["omega"][:steps]).ravel()))
            raw_rmse.append(float(np.sqrt(np.mean(raw_error**2))))
            mass = np.asarray(static["mass"])
            coi_drift.append(float(max(np.max(np.abs(dp@mass)), np.max(np.abs(wp@mass)))))
        tail_count = max(
            1,
            int(np.ceil(float(context.m2a_validation_cvar_fraction) * len(losses))),
        )
        standardized_cvar = float(np.mean(np.sort(losses)[-tail_count:]))
        return {
            "standardized_loss": float(np.mean(losses)),
            "standardized_cvar": standardized_cvar,
            "raw_mean_rmse": float(np.mean(raw_rmse)),
            "raw_median_rmse": float(np.median(raw_rmse)),
            "maximum_coi_drift": float(np.max(coi_drift)),
        }

    def validation_loss(self, samples: list[dict[str, Any]], context: RunContext) -> float:
        return self.validation_summary(samples, context)["standardized_loss"]

    def fit(
        self, train: list[dict[str, Any]], val: list[dict[str, Any]], run_context: RunContext
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0 requires nonempty train and val sets")
        matrix, vector, system_details = self.build_weak_system(
            train,
            windows=run_context.integral_windows,
            max_cases=run_context.max_cases,
            horizon_steps=run_context.horizon_steps,
        )
        self.validation_scales = self._training_scales(train, run_context.max_cases)
        self.max_internal_step = run_context.max_internal_step
        n_nodes = train[0]["model_inputs"]["static"]["mass"].size
        candidates = []
        for ridge in run_context.ridge_values:
            theta, diagnostics = self._solve_nonnegative_ridge(matrix, vector, ridge)
            self.damping, self.coupling = theta[:n_nodes], theta[n_nodes:]
            loss = self.validation_loss(val, run_context)
            candidates.append((loss, float(ridge), theta.copy(), diagnostics))
        loss, ridge, theta, diagnostics = min(candidates, key=lambda item: item[0])
        self.damping, self.coupling = theta[:n_nodes], theta[n_nodes:]
        self.selected_ridge = ridge
        validation_summary = self.validation_summary(val, run_context)
        self.diagnostics = {
            **system_details,
            **diagnostics,
            "selected_ridge": ridge,
            "candidate_validation_losses": {str(item[1]): item[0] for item in candidates},
            "integral_windows": list(run_context.integral_windows),
            "epochs": 0,
            "raw_parameters": theta.tolist(),
            "final_parameters": theta.tolist(),
            "identifiable_quantity": "D and K under m=M/omega_s",
            "physical_topology_claim": False,
            "validation_raw_mean_rmse": validation_summary["raw_mean_rmse"],
            "validation_raw_median_rmse": validation_summary["raw_median_rmse"],
            "maximum_validation_coi_drift": validation_summary["maximum_coi_drift"],
        }
        return FitResult(
            method=self.method_name,
            status="completed",
            train_case_ids=tuple(s["event_id"] for s in train[: run_context.max_cases]),
            val_case_ids=tuple(s["event_id"] for s in val[: run_context.max_cases]),
            best_validation_loss=loss,
            details={"optimizer": "single_bounded_weak_form_ridge", **self.diagnostics},
        )

    def force(
        self, delta: np.ndarray, omega: np.ndarray, pm: np.ndarray, static: dict[str, np.ndarray]
    ) -> np.ndarray:
        if self.damping is None or self.coupling is None:
            raise RuntimeError("M0 is not fitted")
        incidence = np.asarray(static["incidence"])
        equilibrium = np.asarray(static["equilibrium_delta"])
        eta = incidence.T @ delta
        eta0 = incidence.T @ equilibrium
        delta_pm = np.asarray(pm).reshape(-1) - _pm_equilibrium(static)
        force = delta_pm - self.damping * omega - incidence @ (
            self.coupling * (np.sin(eta) - np.sin(eta0))
        )
        return coi_project_power(force, np.asarray(static["mass"]))

    def vector_field(self, delta, omega, pm, static):
        mass = np.asarray(static["mass"])
        return omega, self.force(delta, omega, pm, static) / mass

    def rollout(self, initial_state, time_grid, exogenous):
        return rk4_rollout(
            self.vector_field,
            np.asarray(initial_state["delta0"]).reshape(-1),
            np.asarray(initial_state["omega0"]).reshape(-1),
            time_grid,
            exogenous["Pm"],
            initial_state,
            interpolation=exogenous.get("interpolation", "auto"),
            max_step=self.max_internal_step,
        )

    def convergence_audit(self, initial_state, time_grid, exogenous):
        return integration_convergence_audit(
            self.vector_field,
            np.asarray(initial_state["delta0"]).reshape(-1),
            np.asarray(initial_state["omega0"]).reshape(-1),
            time_grid,
            exogenous["Pm"],
            initial_state,
            interpolation=exogenous.get("interpolation", "auto"),
            base_step=self.max_internal_step,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, damping=self.damping, coupling=self.coupling,
            selected_ridge=np.nan if self.selected_ridge is None else self.selected_ridge,
            max_internal_step=self.max_internal_step,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "unit_policy": "M_over_omega_s", "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "M0Model":
        path = Path(path)
        with np.load(path) as data:
            ridge = float(data["selected_ridge"]) if "selected_ridge" in data else np.nan
            max_step = float(data["max_internal_step"]) if "max_internal_step" in data else 0.01
            model = cls(data["damping"], data["coupling"], selected_ridge=None if np.isnan(ridge) else ridge, max_internal_step=max_step)
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get("diagnostics", {})
        return model

    def physics_audit(self) -> dict[str, float | str | bool]:
        if self.damping is None or self.coupling is None:
            return {"status": "not_fitted"}
        return {
            "status": "pass",
            "minimum_damping": float(np.min(self.damping)),
            "minimum_coupling": float(np.min(self.coupling)),
            "nonnegative_parameters": bool(np.all(self.damping >= 0) and np.all(self.coupling >= 0)),
            "internal_force_balance_by_construction": True,
            "pm_is_equilibrium_centered": True,
            "physical_topology_claim": False,
        }
