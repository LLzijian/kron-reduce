from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import DynamicsModel, FitResult, RunContext
from .integration import rk4_rollout, torch_rk4_rollout
from .m0 import M0Model, _pm_equilibrium, _trapz


def _zero_sum_projector(mass: np.ndarray) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    return np.eye(mass.size) - np.outer(mass, np.ones(mass.size)) / np.sum(mass)


class M2AModel(DynamicsModel):
    """Static graph-SINDy residual with optional support-preserving refinement."""

    method_name = "M2A"

    def __init__(
        self,
        base: M0Model | None = None,
        node_theta: np.ndarray | None = None,
        edge_theta: np.ndarray | None = None,
        *,
        solution_kind: str = "one_step_solution",
        diagnostics: dict[str, Any] | None = None,
    ):
        self.base = base or M0Model()
        self.node_theta = None if node_theta is None else np.asarray(node_theta, dtype=np.float64)
        self.edge_theta = None if edge_theta is None else np.asarray(edge_theta, dtype=np.float64)
        self.solution_kind = solution_kind
        self.diagnostics = diagnostics or {}

    @staticmethod
    def _library(delta, omega, pm, static):
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
        mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
        eta = incidence.T @ delta
        eta0 = incidence.T @ equilibrium
        edge_speed = incidence.T @ omega
        delta_sin = np.sin(eta) - np.sin(eta0)
        delta_pm = np.asarray(pm).reshape(-1) - _pm_equilibrium(static)
        with np.errstate(over="raise", invalid="raise"):
            return (
                _zero_sum_projector(mass),
                omega * np.abs(omega),
                delta_pm * omega,
                delta_sin,
                edge_speed,
                edge_speed * delta_sin,
            )

    @classmethod
    def _instantaneous_design(cls, delta, omega, pm, static):
        projection, node_1, node_2, edge_1, edge_2, edge_3 = cls._library(
            delta, omega, pm, static
        )
        incidence = np.asarray(static["incidence"], dtype=np.float64)
        return np.column_stack((
            projection @ np.diag(node_1),
            projection @ np.diag(node_2),
            -incidence * edge_1,
            -incidence * edge_2,
            -incidence * edge_3,
        ))

    @classmethod
    def build_weak_residual_system(cls, base, samples, context):
        rows, targets = [], []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            t = np.asarray(target["t"], dtype=np.float64)
            delta = np.asarray(target["delta"], dtype=np.float64)
            omega = np.asarray(target["omega"], dtype=np.float64)
            pm = np.asarray(sample["training_context"]["Pm_observed"], dtype=np.float64)
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=np.float64)
            limit = min(len(t), max(context.horizon_steps * 5, max(context.integral_windows) + 1))
            designs = np.stack([
                cls._instantaneous_design(delta[k], omega[k], pm[k], static)
                for k in range(limit)
            ])
            base_force = np.stack([
                base.force(delta[k], omega[k], pm[k], static) for k in range(limit)
            ])
            for width in sorted(set(context.integral_windows)):
                if width <= 0 or width >= limit:
                    continue
                for start in range(limit - width):
                    stop = start + width
                    segment = slice(start, stop + 1)
                    rows.append(_trapz(designs[segment], t[segment]))
                    targets.append(
                        mass * (omega[stop] - omega[start])
                        - _trapz(base_force[segment], t[segment])
                    )
        if not rows:
            raise ValueError("no M2A weak residual windows")
        return np.concatenate(rows), np.concatenate(targets), len(rows)

    @staticmethod
    def _stridge(matrix, vector, ridge, threshold, max_iterations=25):
        scale = np.sqrt(np.mean(matrix**2, axis=0))
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        x = matrix / scale
        support = np.ones(x.shape[1], dtype=bool)
        beta = np.zeros(x.shape[1])
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            active = np.flatnonzero(support)
            if not active.size:
                break
            xa = x[:, active]
            augmented = np.vstack((xa, np.sqrt(ridge) * np.eye(active.size)))
            augmented_y = np.concatenate((vector, np.zeros(active.size)))
            fitted = np.linalg.lstsq(augmented, augmented_y, rcond=None)[0]
            beta[:] = 0.0
            beta[active] = fitted
            updated = np.abs(beta) >= threshold
            if np.array_equal(updated, support):
                break
            support = updated
        theta = beta / scale
        return theta, support, {
            "solver_iterations": iterations,
            "effective_rank": int(np.linalg.matrix_rank(x)),
            "matrix_condition_number": float(np.linalg.cond(x)),
            "weak_residual_rmse": float(np.sqrt(np.mean((matrix @ theta - vector) ** 2))),
        }

    @staticmethod
    def _unpack(theta, n_nodes, n_edges):
        node = np.column_stack((theta[:n_nodes], theta[n_nodes:2 * n_nodes]))
        offset = 2 * n_nodes
        edge = np.column_stack([
            theta[offset + j * n_edges: offset + (j + 1) * n_edges] for j in range(3)
        ])
        return node, edge

    @staticmethod
    def _pack(node, edge):
        return np.concatenate((node[:, 0], node[:, 1], edge[:, 0], edge[:, 1], edge[:, 2]))

    @staticmethod
    def _residual_force_numpy(node_theta, edge_theta, delta, omega, pm, static):
        projection, node_1, node_2, edge_1, edge_2, edge_3 = M2AModel._library(
            delta, omega, pm, static
        )
        incidence = np.asarray(static["incidence"])
        node = projection @ (node_theta[:, 0] * node_1 + node_theta[:, 1] * node_2)
        edge = edge_theta[:, 0] * edge_1 + edge_theta[:, 1] * edge_2 + edge_theta[:, 2] * edge_3
        return node - incidence @ edge

    def fit(self, train, val, run_context):
        if self.base.damping is None:
            self.base.fit(train, val, run_context)
        self.base.max_internal_step = run_context.max_internal_step
        matrix, vector, n_windows = self.build_weak_residual_system(self.base, train, run_context)
        n_nodes = train[0]["model_inputs"]["static"]["mass"].size
        n_edges = train[0]["model_inputs"]["static"]["incidence"].shape[1]
        # M2A is an optional correction to an already selected M0.  The exact
        # zero-residual model must therefore remain in the validation set: on
        # small or poorly conditioned folds every learned residual candidate
        # can be unstable even though the causal M0 rollout is finite.
        base_loss = self.base.validation_loss(val, run_context)
        zero_theta = np.zeros(matrix.shape[1], dtype=float)
        zero_support = np.zeros(matrix.shape[1], dtype=bool)
        fallback_diagnostics = {
            "candidate_kind": "exact_M0_fallback",
            "solver_iterations": 0,
            "effective_rank": int(np.linalg.matrix_rank(matrix)),
            "matrix_condition_number": float(np.linalg.cond(matrix)),
            "weak_residual_rmse": float(np.sqrt(np.mean(vector ** 2))),
        }
        candidates = []
        for ridge in run_context.ridge_values:
            for threshold in run_context.sindy_thresholds:
                theta, support, diagnostics = self._stridge(matrix, vector, ridge, threshold)
                self.node_theta, self.edge_theta = self._unpack(theta, n_nodes, n_edges)
                try:
                    loss = self._validation_loss(val, run_context)
                    failed = False
                except (FloatingPointError, ValueError):
                    loss, failed = float("inf"), True
                candidates.append((loss, failed, ridge, threshold, theta.copy(), support.copy(), diagnostics))
        candidates.append((
            base_loss, False, 0.0, 0.0, zero_theta, zero_support, fallback_diagnostics,
        ))
        selected = min(candidates, key=lambda item: item[0])
        loss, failed, ridge, threshold, theta, support, diagnostics = selected
        if failed or not np.isfinite(loss):
            raise FloatingPointError("all one-step M2A candidates failed validation rollout")
        self.node_theta, self.edge_theta = self._unpack(theta, n_nodes, n_edges)
        one_step_loss = loss
        one_step_improvement = (base_loss - one_step_loss) / max(base_loss, np.finfo(float).eps)
        candidate_theta = theta.copy()
        if one_step_improvement < run_context.refinement_min_improvement:
            theta = np.zeros_like(theta)
            support = np.zeros_like(support)
            self.node_theta, self.edge_theta = self._unpack(theta, n_nodes, n_edges)
            self.solution_kind = "one_step_rejected_to_M0"
            loss = base_loss
        refinement = {"attempted": False, "adopted": False}
        if run_context.rollout_refine and np.any(support):
            refinement = self._refine(train, val, run_context, theta, support)
            if refinement["adopted"]:
                theta = np.asarray(refinement.pop("theta"))
                self.node_theta, self.edge_theta = self._unpack(theta, n_nodes, n_edges)
                self.solution_kind = "rollout_refined"
                loss = float(refinement["validation_loss"])
            else:
                self.node_theta, self.edge_theta = self._unpack(theta, n_nodes, n_edges)
        self.diagnostics = {
            "objective": "single_integral_weak_form_STRidge",
            "epochs": 0,
            "n_integration_windows": n_windows,
            "selected_ridge": float(ridge),
            "selected_threshold": float(threshold),
            "active_parameters": int(np.sum(support)),
            "one_step_validation_loss": float(one_step_loss),
            "base_validation_loss": float(base_loss),
            "one_step_relative_improvement": float(one_step_improvement),
            "one_step_promoted": bool(one_step_improvement >= run_context.refinement_min_improvement),
            "raw_candidate_parameters": candidate_theta.tolist(),
            "solution_kind": self.solution_kind,
            "refinement": refinement,
            **diagnostics,
        }
        return FitResult(
            self.method_name, "completed",
            tuple(s["event_id"] for s in train[: run_context.max_cases]),
            tuple(s["event_id"] for s in val[: run_context.max_cases]),
            float(loss), self.diagnostics,
        )

    def _validation_loss(self, val, context):
        losses = []
        for sample in val[: context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            dp, wp = self.rollout(
                sample["model_inputs"]["static"], target["t"][:steps],
                {"Pm": sample["model_inputs"]["exogenous"]["Pm"][:steps], "interpolation": "zoh"},
            )
            incidence = np.asarray(sample["model_inputs"]["static"]["incidence"])
            losses.append(
                np.mean((((dp - target["delta"][:steps]) @ incidence) / self.base.validation_scales["edge_angle"]) ** 2)
                + np.mean(((wp - target["omega"][:steps]) / self.base.validation_scales["omega"]) ** 2)
            )
        return float(np.mean(losses))

    def _refine(self, train, val, context, theta0, support):
        """Optional closed-loop correction; sparse support is immutable."""
        torch.manual_seed(context.seed)
        parameter = torch.nn.Parameter(torch.as_tensor(theta0, dtype=torch.float64).clone())
        mask = torch.as_tensor(support, dtype=torch.float64)
        anchor = torch.as_tensor(theta0, dtype=torch.float64)
        optimizer = torch.optim.AdamW([parameter], lr=context.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        best_theta, best_val = anchor.clone(), self._validation_loss(val, context)
        patience = 0
        horizons = tuple(h for h in (5, 20, 40, 100) if h <= context.horizon_steps)
        for _ in range(max(1, context.epochs)):
            optimizer.zero_grad()
            losses = []
            active_theta = anchor + mask * (parameter - anchor)
            for sample in train[: context.max_cases]:
                target = sample["training_context"]["targets"]
                for horizon in horizons or (min(len(target["t"]), context.horizon_steps),):
                    steps = min(len(target["t"]), horizon)
                    if steps < 2:
                        continue
                    losses.append(self._torch_rollout_loss(sample, steps, active_theta, context))
            if not losses:
                break
            loss = torch.mean(torch.stack(losses)) + 1e-3 * torch.mean((active_theta - anchor) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter], 1.0)
            optimizer.step()
            with torch.no_grad():
                parameter.copy_(anchor + mask * (parameter - anchor))
                candidate = parameter.detach().cpu().numpy()
                self.node_theta, self.edge_theta = self._unpack(
                    candidate, self.node_theta.shape[0], self.edge_theta.shape[0]
                )
                try:
                    val_loss = self._validation_loss(val, context)
                except (FloatingPointError, ValueError):
                    val_loss = float("inf")
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val, best_theta, patience = val_loss, parameter.detach().clone(), 0
            else:
                patience += 1
                if patience >= 12:
                    break
        improvement = (self._validation_loss_with_theta(val, context, theta0) - best_val) / max(
            self._validation_loss_with_theta(val, context, theta0), np.finfo(float).eps
        )
        adopted = bool(np.isfinite(best_val) and improvement >= context.refinement_min_improvement)
        return {
            "attempted": True, "adopted": adopted, "validation_loss": float(best_val),
            "relative_improvement": float(improvement), "support_unchanged": True,
            "theta": best_theta.cpu().numpy() if adopted else theta0,
        }

    def _validation_loss_with_theta(self, val, context, theta):
        current = self._pack(self.node_theta, self.edge_theta)
        self.node_theta, self.edge_theta = self._unpack(theta, self.node_theta.shape[0], self.edge_theta.shape[0])
        try:
            return self._validation_loss(val, context)
        finally:
            self.node_theta, self.edge_theta = self._unpack(current, self.node_theta.shape[0], self.edge_theta.shape[0])

    def _torch_rollout_loss(self, sample, steps, theta, context):
        static_np = sample["model_inputs"]["static"]
        static = {key: torch.as_tensor(value, dtype=torch.float64) for key, value in static_np.items() if isinstance(value, np.ndarray)}
        target = sample["training_context"]["targets"]
        t = torch.as_tensor(target["t"][:steps], dtype=torch.float64)
        pm = torch.as_tensor(sample["model_inputs"]["exogenous"]["Pm"][:steps], dtype=torch.float64)
        n, e = static["mass"].numel(), static["incidence"].shape[1]
        node = torch.stack((theta[:n], theta[n:2*n]), dim=1)
        offset = 2*n
        edge = torch.stack(tuple(theta[offset+j*e:offset+(j+1)*e] for j in range(3)), dim=1)
        base_d = torch.as_tensor(self.base.damping, dtype=torch.float64)
        base_k = torch.as_tensor(self.base.coupling, dtype=torch.float64)

        def derivative(d, w, p, s):
            inc, mass = s["incidence"], s["mass"]
            eta, eta0 = inc.T @ d, inc.T @ s["equilibrium_delta"]
            dsin, ew = torch.sin(eta) - torch.sin(eta0), inc.T @ w
            dpm = p - s["Pm_equilibrium"]
            base_force = dpm - base_d*w - inc @ (base_k*dsin)
            raw_node = node[:, 0]*w*torch.abs(w) + node[:, 1]*dpm*w
            node_force = raw_node - mass*torch.sum(raw_node)/torch.sum(mass)
            edge_force = edge[:, 0]*dsin + edge[:, 1]*ew + edge[:, 2]*ew*dsin
            return w, (base_force + node_force - inc @ edge_force)/mass

        pd, pw = torch_rk4_rollout(
            derivative, static["delta0"].reshape(-1), static["omega0"].reshape(-1),
            t, pm, static, interpolation="zoh", max_step=context.max_internal_step,
        )
        td, tw = torch.as_tensor(target["delta"][:steps]), torch.as_tensor(target["omega"][:steps])
        return torch.mean((((pd-td) @ static["incidence"])/self.base.validation_scales["edge_angle"])**2) + torch.mean(((pw-tw)/self.base.validation_scales["omega"])**2)

    def force(self, delta, omega, pm, static):
        if self.node_theta is None or self.edge_theta is None:
            raise RuntimeError("M2A is not fitted")
        return self.base.force(delta, omega, pm, static) + self._residual_force_numpy(
            self.node_theta, self.edge_theta, delta, omega, pm, static
        )

    def vector_field(self, delta, omega, pm, static):
        return omega, self.force(delta, omega, pm, static) / np.asarray(static["mass"])

    def rollout(self, initial_state, time_grid, exogenous):
        return rk4_rollout(
            self.vector_field, np.asarray(initial_state["delta0"]).reshape(-1),
            np.asarray(initial_state["omega0"]).reshape(-1), time_grid, exogenous["Pm"], initial_state,
            interpolation=exogenous.get("interpolation", "auto"), max_step=self.base.max_internal_step,
        )

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, damping=self.base.damping, coupling=self.base.coupling,
            node_theta=self.node_theta, edge_theta=self.edge_theta,
            max_internal_step=self.base.max_internal_step,
        )
        path.with_suffix(".json").write_text(json.dumps({"method": self.method_name, "solution_kind": self.solution_kind, "diagnostics": self.diagnostics}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) if path.with_suffix(".json").exists() else {}
        with np.load(path) as data:
            base = M0Model(data["damping"], data["coupling"], max_internal_step=float(data["max_internal_step"]) if "max_internal_step" in data else 0.01)
            return cls(base, data["node_theta"], data["edge_theta"], solution_kind=meta.get("solution_kind", "one_step_solution"), diagnostics=meta.get("diagnostics", {}))

    def physics_audit(self):
        if self.node_theta is None:
            return {"status": "not_fitted"}
        return {
            "status": "pass", "edge_internal_force_balance_by_construction": True,
            "node_zero_total_force_by_construction": True,
            "residual_parameter_norm": float(np.linalg.norm(self.node_theta) + np.linalg.norm(self.edge_theta)),
            "certificate_claim": False, "solution_kind": self.solution_kind,
        }
