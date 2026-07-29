from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import Bounds, minimize

from kron_reduce.physics.coordinates import coi_basis, coi_project_power

from .base import FitResult, RunContext
from .m0 import M0Model, _pm_equilibrium


def _normal_equation_diagnostics(gram: np.ndarray, scalar_rows: int) -> dict[str, float | int]:
    scale = np.sqrt(np.maximum(np.diag(gram) / max(scalar_rows, 1), 0.0))
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    normalized = gram / np.outer(scale, scale)
    eigenvalues = np.linalg.eigvalsh(0.5 * (normalized + normalized.T))
    largest = max(float(eigenvalues[-1]), np.finfo(float).eps)
    tolerance = largest * max(normalized.shape) * np.finfo(float).eps
    positive = eigenvalues[eigenvalues > tolerance]
    return {
        "matrix_condition_number": (
            float(np.sqrt(largest / positive[0])) if positive.size else float("inf")
        ),
        "effective_rank": int(positive.size),
        "column_scale_min": float(np.min(scale)),
        "column_scale_max": float(np.max(scale)),
    }


def _solve_quadratic(
    gram: np.ndarray,
    rhs: np.ndarray,
    scalar_rows: int,
    ridge: float,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    inequalities: np.ndarray | None = None,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    scale = np.sqrt(np.maximum(np.diag(gram) / max(scalar_rows, 1), 0.0))
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    matrix = gram / np.outer(scale, scale)
    vector = rhs / scale
    matrix = 0.5 * (matrix + matrix.T) + float(ridge) * np.eye(len(rhs))

    def objective(x: np.ndarray) -> float:
        return float(0.5 * x @ matrix @ x - vector @ x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return matrix @ x - vector

    constraints = []
    if inequalities is not None and inequalities.size:
        normalized_constraints = inequalities / scale.reshape(1, -1)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, c=normalized_constraints: c @ x,
                "jac": lambda _x, c=normalized_constraints: c,
            }
        )
    x0 = np.zeros(len(rhs), dtype=np.float64) if initial is None else initial * scale
    x0 = np.minimum(np.maximum(x0, lower * scale), upper * scale)
    if constraints:
        method = "SLSQP"
        options = {"maxiter": 3000, "ftol": 1e-11}
    else:
        method = "L-BFGS-B"
        options = {"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 100}
    result = minimize(
        objective, x0, jac=gradient, method=method,
        bounds=Bounds(lower * scale, upper * scale),
        constraints=constraints, options=options,
    )
    theta = result.x / scale
    residual_sse = float(theta @ gram @ theta - 2.0 * rhs @ theta)
    return theta, {
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "weak_residual_sse_without_constant": residual_sse,
        **_normal_equation_diagnostics(gram, scalar_rows),
    }


def _prefix_trapezoid(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = np.diff(t).reshape((-1,) + (1,) * (values.ndim - 1))
    increments = 0.5 * (values[:-1] + values[1:]) * dt
    return np.concatenate((np.zeros_like(values[:1]), np.cumsum(increments, axis=0)))


class DerivativeM0Model(M0Model):
    """Old-style derivative regression under the current data and RK4 contract."""

    method_name = "M0-DD"

    def fit(self, train, val, run_context: RunContext) -> FitResult:
        if not train or not val:
            raise ValueError("M0-DD requires nonempty train and val sets")
        n_nodes = np.asarray(train[0]["model_inputs"]["static"]["mass"]).size
        n_edges = np.asarray(train[0]["model_inputs"]["static"]["incidence"]).shape[1]
        n_parameters = n_nodes + n_edges
        gram = np.zeros((n_parameters, n_parameters), dtype=np.float64)
        rhs = np.zeros(n_parameters, dtype=np.float64)
        scalar_rows = 0
        used_cases = train[: run_context.max_cases]
        for sample in used_cases:
            target = sample["training_context"]["targets"]
            fit_steps = run_context.fit_max_steps or run_context.horizon_steps
            t = np.asarray(target["t"], dtype=np.float64)[:fit_steps]
            delta = np.asarray(target["delta"], dtype=np.float64)[: len(t)]
            omega = np.asarray(target["omega"], dtype=np.float64)[: len(t)]
            pm = np.asarray(sample["training_context"]["Pm_observed"], dtype=np.float64)[: len(t)]
            if len(t) < 3:
                continue
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
            incidence = np.asarray(static["incidence"], dtype=np.float64)
            equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
            acceleration = np.gradient(omega, t, axis=0, edge_order=2)
            design = np.stack([
                self._instantaneous_design(delta[k], omega[k], equilibrium, incidence, mass)
                for k in range(len(t))
            ])
            target_force = pm - _pm_equilibrium(static) - acceleration * mass
            flat_design = design.reshape(-1, n_parameters)
            flat_target = target_force.reshape(-1)
            weight = 1.0 / flat_target.size
            gram += weight * (flat_design.T @ flat_design)
            rhs += weight * (flat_design.T @ flat_target)
            scalar_rows += flat_target.size
        if scalar_rows == 0:
            raise ValueError("no eligible derivative-regression samples")
        self.validation_scales = self._training_scales(train, run_context.max_cases)
        self.max_internal_step = run_context.max_internal_step
        candidates = []
        lower = np.zeros(n_parameters)
        upper = np.full(n_parameters, np.inf)
        for ridge in run_context.ridge_values:
            theta, diagnostics = _solve_quadratic(
                gram, rhs, scalar_rows, ridge, lower=lower, upper=upper
            )
            self.damping, self.coupling = theta[:n_nodes], theta[n_nodes:]
            summary = self.validation_summary(val, run_context)
            candidates.append((summary["standardized_loss"], float(ridge), theta.copy(), diagnostics, summary))
        loss, ridge, theta, diagnostics, summary = min(candidates, key=lambda item: item[0])
        self.damping, self.coupling = theta[:n_nodes], theta[n_nodes:]
        self.selected_ridge = ridge
        self.diagnostics = {
            **diagnostics,
            "n_scalar_equations": scalar_rows,
            "n_parameters": n_parameters,
            "selected_ridge": ridge,
            "epochs": 0,
            "derivative_scheme": "numpy_gradient_edge_order_2",
            "case_weighting": "equal_per_training_case",
            "candidate_validation_losses": {str(row[1]): row[0] for row in candidates},
            "raw_parameters": theta.tolist(),
            "final_parameters": theta.tolist(),
            "validation_raw_mean_rmse": summary["raw_mean_rmse"],
            "validation_raw_median_rmse": summary["raw_median_rmse"],
            "maximum_validation_coi_drift": summary["maximum_coi_drift"],
        }
        return FitResult(
            self.method_name, "completed",
            tuple(s["event_id"] for s in used_cases),
            tuple(s["event_id"] for s in val[: run_context.max_cases]),
            loss,
            {"optimizer": "single_bounded_derivative_ridge", **self.diagnostics},
        )

    @classmethod
    def load(cls, path: str | Path) -> "DerivativeM0Model":
        loaded = super().load(path)
        return cls(
            loaded.damping, loaded.coupling,
            selected_ridge=loaded.selected_ridge,
            max_internal_step=loaded.max_internal_step,
            diagnostics=loaded.diagnostics,
        )


class GraphM0Model(M0Model):
    """Equilibrium-anchored graph potential with node and edge damping."""

    method_name = "M0-GRAPH"

    def __init__(
        self,
        node_damping: np.ndarray | None = None,
        sine_coupling: np.ndarray | None = None,
        cosine_coupling: np.ndarray | None = None,
        edge_damping: np.ndarray | None = None,
        *,
        selected_ridge: float | None = None,
        selected_prune_threshold: float = 0.0,
        max_internal_step: float = 0.01,
        diagnostics: dict[str, Any] | None = None,
    ):
        super().__init__(
            damping=node_damping, coupling=sine_coupling,
            selected_ridge=selected_ridge, max_internal_step=max_internal_step,
            diagnostics=diagnostics,
        )
        self.cosine_coupling = None if cosine_coupling is None else np.asarray(cosine_coupling, dtype=np.float64)
        self.edge_damping = None if edge_damping is None else np.asarray(edge_damping, dtype=np.float64)
        self.selected_prune_threshold = float(selected_prune_threshold)

    @staticmethod
    def _instantaneous_graph_design(delta, omega, equilibrium, incidence, mass):
        n_nodes, n_edges = omega.size, incidence.shape[1]
        design = np.zeros((n_nodes, n_nodes + 3 * n_edges), dtype=np.float64)
        projector = np.eye(n_nodes) - np.outer(mass, np.ones(n_nodes)) / np.sum(mass)
        design[:, :n_nodes] = projector @ np.diag(omega)
        eta = incidence.T @ delta
        eta0 = incidence.T @ equilibrium
        design[:, n_nodes:n_nodes+n_edges] = incidence * (np.sin(eta) - np.sin(eta0))
        design[:, n_nodes+n_edges:n_nodes+2*n_edges] = incidence * (np.cos(eta) - np.cos(eta0))
        return design

    @classmethod
    def build_normal_system(cls, samples, *, windows, max_cases, horizon_steps, max_fit_steps=0, nonoverlap=True):
        selected = samples[:max_cases]
        first_static = selected[0]["model_inputs"]["static"]
        n_nodes = np.asarray(first_static["mass"]).size
        n_edges = np.asarray(first_static["incidence"]).shape[1]
        n_parameters = n_nodes + 3 * n_edges
        gram = np.zeros((n_parameters, n_parameters), dtype=np.float64)
        rhs = np.zeros(n_parameters, dtype=np.float64)
        target_square = 0.0
        scalar_rows = 0
        total_windows = 0
        strata = 0
        for sample in selected:
            target = sample["training_context"]["targets"]
            limit = min(
                len(target["t"]),
                int(max_fit_steps) if int(max_fit_steps) > 0 else int(horizon_steps),
            )
            if limit < 2:
                continue
            t = np.asarray(target["t"], dtype=np.float64)[:limit]
            delta = np.asarray(target["delta"], dtype=np.float64)[:limit]
            omega = np.asarray(target["omega"], dtype=np.float64)[:limit]
            pm = np.asarray(sample["training_context"]["Pm_observed"], dtype=np.float64)[:limit]
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
            incidence = np.asarray(static["incidence"], dtype=np.float64)
            equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
            delta_pm = pm - _pm_equilibrium(static)
            instantaneous = np.stack([
                cls._instantaneous_graph_design(delta[k], omega[k], equilibrium, incidence, mass)
                for k in range(limit)
            ])
            design_prefix = _prefix_trapezoid(instantaneous, t)
            pm_prefix = _prefix_trapezoid(delta_pm, t)
            eta = delta @ incidence
            for width in sorted(set(int(w) for w in windows if 0 < int(w) < limit)):
                starts = np.arange(0, limit - width, width if nonoverlap else 1)
                if starts.size == 0:
                    continue
                stops = starts + width
                count = len(starts)
                block = design_prefix[stops] - design_prefix[starts]
                block[:, :, n_nodes+2*n_edges:] = (
                    incidence[None, :, :] * (eta[stops] - eta[starts])[:, None, :]
                )
                target_block = (
                    pm_prefix[stops] - pm_prefix[starts]
                    - mass[None, :] * (omega[stops] - omega[starts])
                )
                flat_design = block.reshape(-1, n_parameters)
                flat_target = target_block.reshape(-1)
                weight = 1.0 / flat_target.size
                gram += weight * (flat_design.T @ flat_design)
                rhs += weight * (flat_design.T @ flat_target)
                target_square += weight * float(flat_target @ flat_target)
                scalar_rows += flat_target.size
                total_windows += count
                strata += 1
        if scalar_rows == 0:
            raise ValueError("no eligible graph weak-form windows")
        return gram, rhs, target_square, {
            "n_scalar_equations": scalar_rows,
            "n_parameters": n_parameters,
            "n_integration_windows": total_windows,
            "n_case_window_strata": strata,
        }

    @staticmethod
    def _curvature_inequalities(static: dict[str, np.ndarray]) -> np.ndarray:
        mass = np.asarray(static["mass"]).reshape(-1)
        incidence = np.asarray(static["incidence"])
        equilibrium = np.asarray(static["equilibrium_delta"]).reshape(-1)
        n_nodes, n_edges = len(mass), incidence.shape[1]
        matrix = np.zeros((n_edges, n_nodes + 3*n_edges), dtype=np.float64)
        eta0 = incidence.T @ equilibrium
        indices = np.arange(n_edges)
        matrix[indices, n_nodes + indices] = np.cos(eta0)
        matrix[indices, n_nodes + n_edges + indices] = -np.sin(eta0)
        return matrix

    @staticmethod
    def _potential_cone_transform(static: dict[str, np.ndarray]):
        """Map simple bounds to Ks>=0 and local-curvature>=0 exactly.

        For nonzero equilibrium sine, the feasible two-dimensional cone is
        represented by its two extreme rays.  This removes the linear
        inequality from the numerical optimizer and leaves a bounded convex
        quadratic problem.  Near zero sine, Ks is nonnegative and Kc remains
        an ordinary signed coordinate.
        """
        mass = np.asarray(static["mass"]).reshape(-1)
        incidence = np.asarray(static["incidence"])
        equilibrium = np.asarray(static["equilibrium_delta"]).reshape(-1)
        n_nodes, n_edges = len(mass), incidence.shape[1]
        n_parameters = n_nodes + 3*n_edges
        transform = np.eye(n_parameters, dtype=np.float64)
        lower = np.zeros(n_parameters, dtype=np.float64)
        upper = np.full(n_parameters, np.inf, dtype=np.float64)
        eta0 = incidence.T @ equilibrium
        for edge, angle in enumerate(eta0):
            sine, cosine = float(np.sin(angle)), float(np.cos(angle))
            ks_index = n_nodes + edge
            kc_index = n_nodes + n_edges + edge
            transform[np.ix_([ks_index, kc_index], [ks_index, kc_index])] = 0.0
            if abs(sine) <= 1e-6:
                if cosine < 0.0:
                    raise ValueError("equilibrium edge outside the convex local-potential domain")
                transform[ks_index, ks_index] = 1.0
                transform[kc_index, kc_index] = 1.0
                lower[kc_index] = -np.inf
            else:
                sign = float(np.sign(sine))
                # First ray: (Ks,Kc)=(0,-sign(sin)); second ray lies
                # exactly on Ks*cos(eta0)-Kc*sin(eta0)=0.
                transform[kc_index, ks_index] = -sign
                transform[ks_index, kc_index] = abs(sine)
                transform[kc_index, kc_index] = sign * cosine
        return transform, lower, upper

    @staticmethod
    def _parameter_diagnostics(theta, static):
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
        n_nodes, n_edges = len(mass), incidence.shape[1]
        dn = theta[:n_nodes]
        ks = theta[n_nodes:n_nodes+n_edges]
        kc = theta[n_nodes+n_edges:n_nodes+2*n_edges]
        de = theta[n_nodes+2*n_edges:]
        eta0 = incidence.T @ equilibrium
        curvature = ks * np.cos(eta0) - kc * np.sin(eta0)
        damping = np.diag(dn) + incidence @ np.diag(de) @ incidence.T
        stiffness = incidence @ np.diag(curvature) @ incidence.T
        basis = coi_basis(mass)
        reduced_mass = basis.T @ np.diag(mass) @ basis
        reduced_damping = basis.T @ damping @ basis
        reduced_stiffness = basis.T @ stiffness @ basis
        damping_eigenvalues = eigvalsh(reduced_damping, reduced_mass)
        stiffness_eigenvalues = eigvalsh(reduced_stiffness, reduced_mass)
        inverse_mass = np.linalg.solve(reduced_mass, np.eye(n_nodes - 1))
        system = np.block([
            [np.zeros((n_nodes-1, n_nodes-1)), np.eye(n_nodes-1)],
            [-inverse_mass @ reduced_stiffness, -inverse_mass @ reduced_damping],
        ])
        modes = np.linalg.eigvals(system)
        return {
            "node_damping": dn.tolist(),
            "sine_coupling": ks.tolist(),
            "cosine_coupling": kc.tolist(),
            "edge_damping": de.tolist(),
            "local_stiffness": curvature.tolist(),
            "minimum_node_damping": float(np.min(dn)),
            "minimum_edge_damping": float(np.min(de)),
            "minimum_local_stiffness": float(np.min(curvature)),
            "near_zero_node_damping_count": int(np.sum(dn <= 1e-8)),
            "near_zero_edge_damping_count": int(np.sum(de <= 1e-8)),
            "active_edge_count": int(np.sum(np.maximum.reduce((np.abs(ks), np.abs(kc), np.abs(de))) > 1e-8)),
            "quotient_damping_eigenvalues": damping_eigenvalues.tolist(),
            "quotient_stiffness_eigenvalues": stiffness_eigenvalues.tolist(),
            "minimum_quotient_damping_eigenvalue": float(np.min(damping_eigenvalues)),
            "minimum_quotient_stiffness_eigenvalue": float(np.min(stiffness_eigenvalues)),
            "linearized_mode_real_parts": np.real(modes).tolist(),
            "linearized_mode_imaginary_parts": np.imag(modes).tolist(),
            "maximum_linearized_mode_real_part": float(np.max(np.real(modes))),
        }

    def _assign(self, theta, n_nodes, n_edges):
        self.damping = theta[:n_nodes].copy()
        self.coupling = theta[n_nodes:n_nodes+n_edges].copy()
        self.cosine_coupling = theta[n_nodes+n_edges:n_nodes+2*n_edges].copy()
        self.edge_damping = theta[n_nodes+2*n_edges:].copy()

    def fit(self, train, val, run_context: RunContext) -> FitResult:
        if not train or not val:
            raise ValueError("M0-GRAPH requires nonempty train and val sets")
        static = train[0]["model_inputs"]["static"]
        n_nodes = np.asarray(static["mass"]).size
        n_edges = np.asarray(static["incidence"]).shape[1]
        n_parameters = n_nodes + 3*n_edges
        gram, rhs, target_square, system_details = self.build_normal_system(
            train,
            windows=run_context.integral_windows,
            max_cases=run_context.max_cases,
            horizon_steps=run_context.horizon_steps,
            max_fit_steps=run_context.fit_max_steps,
            nonoverlap=run_context.graph_nonoverlap_windows,
        )
        transform, lower, upper = self._potential_cone_transform(static)
        transformed_gram = transform.T @ gram @ transform
        transformed_rhs = transform.T @ rhs
        self.validation_scales = self._training_scales(train, run_context.max_cases)
        self.max_internal_step = run_context.max_internal_step
        candidates = []
        for ridge in run_context.ridge_values:
            cone_theta, solve_details = _solve_quadratic(
                transformed_gram, transformed_rhs,
                system_details["n_scalar_equations"], ridge,
                lower=lower, upper=upper,
            )
            full_theta = transform @ cone_theta
            if not solve_details["solver_success"]:
                continue
            for threshold in run_context.graph_prune_thresholds:
                theta = full_theta.copy()
                if threshold > 0:
                    scale = max(float(np.max(np.abs(theta))), np.finfo(float).eps)
                    theta[:n_nodes][np.abs(theta[:n_nodes]) / scale < threshold] = 0.0
                    edge_score = np.maximum.reduce((
                        np.abs(theta[n_nodes:n_nodes+n_edges]),
                        np.abs(theta[n_nodes+n_edges:n_nodes+2*n_edges]),
                        np.abs(theta[n_nodes+2*n_edges:]),
                    )) / scale
                    remove = edge_score < threshold
                    theta[n_nodes:n_nodes+n_edges][remove] = 0.0
                    theta[n_nodes+n_edges:n_nodes+2*n_edges][remove] = 0.0
                    theta[n_nodes+2*n_edges:][remove] = 0.0
                parameter_details = self._parameter_diagnostics(theta, static)
                if parameter_details["minimum_local_stiffness"] < -1e-8:
                    continue
                self._assign(theta, n_nodes, n_edges)
                summary = self.validation_summary(val, run_context)
                residual_sse = float(target_square + theta @ gram @ theta - 2.0 * rhs @ theta)
                candidates.append((
                    summary["standardized_loss"], float(ridge), float(threshold),
                    theta.copy(), {**solve_details, **parameter_details, "weighted_weak_residual_rmse": float(np.sqrt(max(residual_sse, 0.0)))},
                    summary,
                ))
        if not candidates:
            raise RuntimeError("all M0-GRAPH candidates failed physical constraints")
        loss, ridge, threshold, theta, details, summary = min(candidates, key=lambda item: item[0])
        self._assign(theta, n_nodes, n_edges)
        self.selected_ridge = ridge
        self.selected_prune_threshold = threshold
        self.diagnostics = {
            **system_details,
            **details,
            "selected_ridge": ridge,
            "selected_prune_threshold": threshold,
            "integral_windows": list(run_context.integral_windows),
            "window_sampling": "deterministic_nonoverlap" if run_context.graph_nonoverlap_windows else "all_overlapping",
            "case_window_weighting": "equal_per_case_by_window_length",
            "constraint_parameterization": "two_extreme_rays_per_edge_potential_cone",
            "epochs": 0,
            "candidate_validation_losses": {
                f"ridge={row[1]}|prune={row[2]}": row[0] for row in candidates
            },
            "raw_parameters": theta.tolist(),
            "final_parameters": theta.tolist(),
            "physical_topology_claim": False,
            "validation_raw_mean_rmse": summary["raw_mean_rmse"],
            "validation_raw_median_rmse": summary["raw_median_rmse"],
            "validation_standardized_cvar": summary["standardized_cvar"],
            "maximum_validation_coi_drift": summary["maximum_coi_drift"],
        }
        audit = self.physics_audit()
        if audit["status"] != "pass":
            raise RuntimeError(f"selected M0-GRAPH failed physics audit: {audit}")
        return FitResult(
            self.method_name, "completed",
            tuple(s["event_id"] for s in train[:run_context.max_cases]),
            tuple(s["event_id"] for s in val[:run_context.max_cases]),
            loss,
            {"optimizer": "single_constrained_graph_weak_form_ridge", **self.diagnostics},
        )

    def force(self, delta, omega, pm, static):
        if any(value is None for value in (self.damping, self.coupling, self.cosine_coupling, self.edge_damping)):
            raise RuntimeError("M0-GRAPH is not fitted")
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
        eta = incidence.T @ np.asarray(delta)
        eta0 = incidence.T @ equilibrium
        edge_speed = incidence.T @ np.asarray(omega)
        edge_force = (
            self.coupling * (np.sin(eta) - np.sin(eta0))
            + self.cosine_coupling * (np.cos(eta) - np.cos(eta0))
            + self.edge_damping * edge_speed
        )
        force = (
            np.asarray(pm).reshape(-1) - _pm_equilibrium(static)
            - self.damping * np.asarray(omega) - incidence @ edge_force
        )
        return coi_project_power(force, np.asarray(static["mass"]))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            node_damping=self.damping,
            sine_coupling=self.coupling,
            cosine_coupling=self.cosine_coupling,
            edge_damping=self.edge_damping,
            selected_ridge=np.nan if self.selected_ridge is None else self.selected_ridge,
            selected_prune_threshold=self.selected_prune_threshold,
            max_internal_step=self.max_internal_step,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "unit_policy": "M_over_omega_s", "diagnostics": self.diagnostics}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphM0Model":
        path = Path(path)
        with np.load(path) as data:
            ridge = float(data["selected_ridge"])
            model = cls(
                data["node_damping"], data["sine_coupling"],
                data["cosine_coupling"], data["edge_damping"],
                selected_ridge=None if np.isnan(ridge) else ridge,
                selected_prune_threshold=float(data["selected_prune_threshold"]),
                max_internal_step=float(data["max_internal_step"]),
            )
        metadata = path.with_suffix(".json")
        if metadata.exists():
            model.diagnostics = json.loads(metadata.read_text(encoding="utf-8")).get("diagnostics", {})
        return model

    def physics_audit(self) -> dict[str, float | str | bool]:
        if any(value is None for value in (self.damping, self.coupling, self.cosine_coupling, self.edge_damping)):
            return {"status": "not_fitted"}
        minimum_stiffness = float(self.diagnostics.get("minimum_local_stiffness", np.nan))
        maximum_mode_real = float(self.diagnostics.get("maximum_linearized_mode_real_part", np.nan))
        passed = bool(
            np.all(np.isfinite(np.concatenate((self.damping, self.coupling, self.cosine_coupling, self.edge_damping))))
            and np.all(self.damping >= -1e-10)
            and np.all(self.coupling >= -1e-10)
            and np.all(self.edge_damping >= -1e-10)
            and np.isfinite(minimum_stiffness) and minimum_stiffness >= -1e-8
            and np.isfinite(maximum_mode_real) and maximum_mode_real <= 1e-7
        )
        return {
            "status": "pass" if passed else "fail",
            "nonnegative_node_and_edge_damping": bool(np.all(self.damping >= -1e-10) and np.all(self.edge_damping >= -1e-10)),
            "nonnegative_sine_coupling": bool(np.all(self.coupling >= -1e-10)),
            "minimum_local_stiffness": minimum_stiffness,
            "maximum_linearized_mode_real_part": maximum_mode_real,
            "internal_force_balance_by_construction": True,
            "common_angle_shift_invariant": True,
            "equilibrium_anchored": True,
            "pm_is_equilibrium_centered": True,
            "physical_topology_claim": False,
        }
