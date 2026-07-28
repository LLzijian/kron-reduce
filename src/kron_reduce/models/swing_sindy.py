from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project

from .base import DynamicsModel, FitResult, RunContext, sample_rollout_inputs
from .integration import rk4_rollout
from .m0 import M0Model, _pm_equilibrium


def _force_projector(mass: np.ndarray) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    return np.eye(mass.size) - np.outer(mass, np.ones(mass.size)) / np.sum(mass)


class SwingSINDyModel(DynamicsModel):
    """Direct second-order SINDy inside the exact swing kinematic shell.

    The sparse regression learns only the COI-balanced generalized force in
    ``m * omega_dot``.  The kinematic equation ``delta_dot = omega`` is exact,
    and the polynomial library is anchored at the declared equilibrium.
    """

    method_name = "Swing-SINDy"

    def __init__(
        self,
        coefficients: np.ndarray | None = None,
        n_nodes: int | None = None,
        *,
        selected_ridge: float | None = None,
        selected_threshold: float | None = None,
        max_internal_step: float = 0.01,
        diagnostics: dict[str, Any] | None = None,
    ):
        self.coefficients = (
            None
            if coefficients is None
            else np.asarray(coefficients, dtype=np.float64)
        )
        self.n_nodes = n_nodes
        self.selected_ridge = selected_ridge
        self.selected_threshold = selected_threshold
        self.max_internal_step = float(max_internal_step)
        self.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
        self.diagnostics = diagnostics or {}

    @staticmethod
    def _state_vector(delta, omega, pm, static):
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        equilibrium = np.asarray(
            static["equilibrium_delta"], dtype=np.float64
        ).reshape(-1)
        relative = coi_project(np.asarray(delta).reshape(-1) - equilibrium, mass)
        speed = coi_project(np.asarray(omega).reshape(-1), mass)
        delta_pm = np.asarray(pm).reshape(-1) - _pm_equilibrium(static)
        return np.concatenate((relative, speed, delta_pm))

    @classmethod
    def _library(cls, delta, omega, pm, static):
        state = cls._state_vector(delta, omega, pm, static)
        upper = np.triu_indices(state.size)
        with np.errstate(over="raise", invalid="raise"):
            quadratic = state[upper[0]] * state[upper[1]]
        return np.concatenate((state, quadratic))

    @staticmethod
    def _prepare_system(matrix, target):
        feature_scale = np.sqrt(np.mean(matrix**2, axis=0))
        feature_scale = np.where(
            feature_scale > np.finfo(float).eps, feature_scale, 1.0
        )
        target_scale = np.sqrt(np.mean(target**2, axis=0))
        target_scale = np.where(
            target_scale > np.finfo(float).eps, target_scale, 1.0
        )
        x = matrix / feature_scale
        y = target / target_scale
        return {
            "feature_scale": feature_scale,
            "target_scale": target_scale,
            "gram": x.T @ x,
            "rhs": x.T @ y,
            "normalized_design_rank": int(np.linalg.matrix_rank(x)),
            "normalized_design_condition_number": float(np.linalg.cond(x)),
        }

    @classmethod
    def _stridge(
        cls, matrix, target, ridge, threshold, max_iterations=25,
        prepared=None,
    ):
        prepared = cls._prepare_system(matrix, target) if prepared is None else prepared
        feature_scale = prepared["feature_scale"]
        target_scale = prepared["target_scale"]
        gram = prepared["gram"]
        rhs = prepared["rhs"]
        beta = np.zeros((matrix.shape[1], target.shape[1]), dtype=np.float64)
        iterations = 0
        for output in range(target.shape[1]):
            support = np.ones(matrix.shape[1], dtype=bool)
            for local_iteration in range(1, max_iterations + 1):
                active = np.flatnonzero(support)
                if not active.size:
                    break
                system = gram[np.ix_(active, active)] + ridge * np.eye(active.size)
                try:
                    fitted = np.linalg.solve(system, rhs[active, output])
                except np.linalg.LinAlgError:
                    fitted = np.linalg.lstsq(
                        system, rhs[active, output], rcond=None
                    )[0]
                beta[:, output] = 0.0
                beta[active, output] = fitted
                updated = np.abs(beta[:, output]) >= threshold
                iterations = max(iterations, local_iteration)
                if np.array_equal(updated, support):
                    break
                support = updated
        coefficients = beta * target_scale[None, :] / feature_scale[:, None]
        prediction = matrix @ coefficients
        return coefficients, {
            "solver_iterations": int(iterations),
            "active_parameters": int(np.count_nonzero(coefficients)),
            "derivative_force_rmse": float(
                np.sqrt(np.mean((prediction - target) ** 2))
            ),
            "normalized_design_rank": int(prepared["normalized_design_rank"]),
            "normalized_design_condition_number": float(
                prepared["normalized_design_condition_number"]
            ),
        }

    @classmethod
    def _derivative_system(cls, samples, context):
        features, forces = [], []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            t = np.asarray(target["t"], dtype=np.float64)
            if len(t) < 3:
                continue
            limit = min(len(t), context.horizon_steps)
            if limit < 3:
                continue
            delta = np.asarray(target["delta"][:limit], dtype=np.float64)
            omega = np.asarray(target["omega"][:limit], dtype=np.float64)
            pm = np.asarray(
                sample["model_inputs"]["exogenous"]["Pm"][:limit],
                dtype=np.float64,
            )
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
            domega = np.gradient(omega, t[:limit], axis=0, edge_order=2)
            projector = _force_projector(mass)
            for index in range(limit):
                features.append(
                    cls._library(delta[index], omega[index], pm[index], static)
                )
                forces.append(projector @ (mass * domega[index]))
        if not features:
            raise ValueError("Swing-SINDy requires trajectories with at least 3 samples")
        return np.asarray(features), np.asarray(forces)

    def _validation_summary(self, samples, context):
        losses, raw_rmse = [], []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            static, sample_t, exogenous = sample_rollout_inputs(sample)
            predicted_delta, predicted_omega = self.rollout(
                static,
                sample_t[:steps],
                {
                    "Pm": exogenous["Pm"][:steps],
                    "interpolation": exogenous["interpolation"],
                },
            )
            incidence = np.asarray(static["incidence"], dtype=np.float64)
            edge_error = (
                predicted_delta - np.asarray(target["delta"][:steps])
            ) @ incidence
            omega_error = predicted_omega - np.asarray(target["omega"][:steps])
            losses.append(
                np.mean((edge_error / self.validation_scales["edge_angle"]) ** 2)
                + np.mean((omega_error / self.validation_scales["omega"]) ** 2)
            )
            error = np.concatenate((edge_error.reshape(-1), omega_error.reshape(-1)))
            raw_rmse.append(float(np.sqrt(np.mean(error**2))))
        tail_count = max(
            1,
            int(np.ceil(context.m2a_validation_cvar_fraction * len(losses))),
        )
        return {
            "standardized_loss": float(np.mean(losses)),
            "standardized_cvar": float(np.mean(np.sort(losses)[-tail_count:])),
            "raw_mean_rmse": float(np.mean(raw_rmse)),
            "failure_count": 0,
        }

    def fit(self, train, val, run_context):
        if not train or not val:
            raise ValueError("Swing-SINDy requires nonempty train and validation sets")
        self.max_internal_step = run_context.max_internal_step
        self.validation_scales = M0Model._training_scales(
            train, run_context.max_cases
        )
        matrix, target = self._derivative_system(train, run_context)
        validation_matrix, validation_target = self._derivative_system(
            val, run_context
        )
        prepared = self._prepare_system(matrix, target)
        solved = []
        for ridge in run_context.ridge_values:
            for threshold in run_context.sindy_thresholds:
                coefficients, diagnostics = self._stridge(
                    matrix, target, ridge, threshold, prepared=prepared
                )
                derivative_validation_rmse = float(
                    np.sqrt(
                        np.mean(
                            (validation_matrix @ coefficients - validation_target) ** 2
                        )
                    )
                )
                solved.append(
                    (
                        derivative_validation_rmse,
                        float(ridge),
                        float(threshold),
                        coefficients,
                        diagnostics,
                    )
                )
        unique = {}
        for item in solved:
            key = np.ascontiguousarray(item[3]).tobytes()
            if key not in unique or item[0] < unique[key][0]:
                unique[key] = item
        ranked = sorted(unique.values(), key=lambda item: item[0])
        diverse = []
        seen = set()
        for key_index in (1, 2):  # retain the best candidate per ridge/threshold
            for value in sorted({item[key_index] for item in ranked}):
                candidate = next(item for item in ranked if item[key_index] == value)
                key = np.ascontiguousarray(candidate[3]).tobytes()
                if key not in seen:
                    diverse.append(candidate)
                    seen.add(key)
        diverse.extend(
            item for item in ranked
            if np.ascontiguousarray(item[3]).tobytes() not in seen
        )
        shortlist = diverse[: max(1, int(run_context.m2a_rollout_shortlist))]
        rollout_candidates = []
        for derivative_rmse, ridge, threshold, coefficients, diagnostics in shortlist:
            self.coefficients = coefficients
            self.n_nodes = target.shape[1]
            try:
                validation = self._validation_summary(val, run_context)
                rollout_loss = float(validation["standardized_loss"])
                rollout_cvar = float(validation["standardized_cvar"])
                failed = False
            except (FloatingPointError, ValueError, OverflowError):
                rollout_loss = rollout_cvar = float("inf")
                failed = True
            rollout_candidates.append(
                (
                    rollout_loss,
                    rollout_cvar,
                    failed,
                    derivative_rmse,
                    ridge,
                    threshold,
                    coefficients,
                    diagnostics,
                )
            )
        finite = [
            item for item in rollout_candidates
            if not item[2] and np.isfinite(item[0]) and np.isfinite(item[1])
        ]
        if not finite:
            raise FloatingPointError(
                "all direct Swing-SINDy candidates failed validation rollout"
            )
        selected = min(finite, key=lambda item: (item[0], item[1]))
        (
            loss,
            cvar,
            _,
            derivative_rmse,
            ridge,
            threshold,
            coefficients,
            solver_diagnostics,
        ) = selected
        self.coefficients = coefficients
        self.n_nodes = target.shape[1]
        self.selected_ridge = ridge
        self.selected_threshold = threshold
        n_state = 3 * self.n_nodes
        expected_features = n_state + n_state * (n_state + 1) // 2
        if self.coefficients.shape != (expected_features, self.n_nodes):
            raise RuntimeError("Swing-SINDy polynomial layout mismatch")
        self.diagnostics = {
            "objective": "direct_numerical_derivative_STRidge_then_closed_loop_selection",
            "solution_kind": "second_order_swing_sindy",
            "library": "equilibrium_anchored_full_polynomial_degree_2",
            "learned_equation": "m_omega_dot_only",
            "exact_kinematics": "delta_dot_equals_omega",
            "force_projection": "zero_total_force_COI",
            "exogenous_policy": (
                "registered_causal_zoh"
                if train[0]["training_context"]["claim_type"] == "causal_zoh_rollout"
                else "conditional_observed_pm_linear_interpolation"
            ),
            "epochs": 0,
            "n_training_rows": int(matrix.shape[0]),
            "n_library_features": int(matrix.shape[1]),
            "total_linear_candidates": len(solved),
            "unique_linear_candidates": len(unique),
            "closed_loop_shortlist_size": len(shortlist),
            "selected_ridge": ridge,
            "selected_threshold": threshold,
            "selected_derivative_validation_rmse": derivative_rmse,
            "selected_validation_cvar": cvar,
            "closed_loop_candidates": [
                {
                    "validation_loss": float(item[0]),
                    "validation_cvar": float(item[1]),
                    "failed": bool(item[2]),
                    "derivative_validation_rmse": float(item[3]),
                    "ridge": float(item[4]),
                    "threshold": float(item[5]),
                    "active_parameters": int(np.count_nonzero(item[6])),
                }
                for item in rollout_candidates
            ],
            **solver_diagnostics,
        }
        return FitResult(
            self.method_name,
            "completed",
            tuple(sample["event_id"] for sample in train[: run_context.max_cases]),
            tuple(sample["event_id"] for sample in val[: run_context.max_cases]),
            float(loss),
            self.diagnostics,
        )

    def force(self, delta, omega, pm, static):
        if self.coefficients is None:
            raise RuntimeError("Swing-SINDy is not fitted")
        with np.errstate(over="raise", invalid="raise"):
            raw_force = self._library(delta, omega, pm, static) @ self.coefficients
        return _force_projector(np.asarray(static["mass"])) @ raw_force

    def vector_field(self, delta, omega, pm, static):
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        speed = coi_project(np.asarray(omega).reshape(-1), mass)
        return speed, self.force(delta, speed, pm, static) / mass

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

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coefficients=self.coefficients,
            n_nodes=self.n_nodes,
            selected_ridge=(
                np.nan if self.selected_ridge is None else self.selected_ridge
            ),
            selected_threshold=(
                np.nan if self.selected_threshold is None else self.selected_threshold
            ),
            max_internal_step=self.max_internal_step,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {"method": self.method_name, "diagnostics": self.diagnostics},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = (
            json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if path.with_suffix(".json").exists()
            else {}
        )
        with np.load(path) as data:
            ridge = float(data["selected_ridge"])
            threshold = float(data["selected_threshold"])
            return cls(
                data["coefficients"],
                int(data["n_nodes"]),
                selected_ridge=None if np.isnan(ridge) else ridge,
                selected_threshold=None if np.isnan(threshold) else threshold,
                max_internal_step=float(data["max_internal_step"]),
                diagnostics=metadata.get("diagnostics", {}),
            )

    def physics_audit(self):
        if self.coefficients is None:
            return {"status": "not_fitted"}
        return {
            "status": "pass",
            "exact_second_order_kinematics": True,
            "equilibrium_anchored_library": True,
            "zero_total_internal_force_by_projection": True,
            "common_angle_shift_removed_by_coi_projection": True,
            "certificate_claim": False,
        }
