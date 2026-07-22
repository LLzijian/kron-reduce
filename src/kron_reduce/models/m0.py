from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from .base import DynamicsModel, FitResult, RunContext
from .integration import rk4_rollout


class M0Model(DynamicsModel):
    method_name = "M0"

    def __init__(self, damping: np.ndarray | None = None, coupling: np.ndarray | None = None):
        self.damping = None if damping is None else np.asarray(damping, dtype=np.float64)
        self.coupling = None if coupling is None else np.asarray(coupling, dtype=np.float64)

    @staticmethod
    def _features(
        delta: np.ndarray, omega: np.ndarray, equilibrium: np.ndarray, incidence: np.ndarray
    ) -> np.ndarray:
        n_nodes = omega.size
        n_edges = incidence.shape[1]
        design = np.zeros((n_nodes, n_nodes + n_edges), dtype=np.float64)
        design[np.arange(n_nodes), np.arange(n_nodes)] = -omega
        eta = incidence.T @ delta
        eta0 = incidence.T @ equilibrium
        design[:, n_nodes:] = -incidence * (np.sin(eta) - np.sin(eta0))
        return design

    def fit(
        self, train: list[dict[str, Any]], val: list[dict[str, Any]], run_context: RunContext
    ) -> FitResult:
        if not train or not val:
            raise ValueError("M0 requires nonempty train and val sets")
        designs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for sample in train[: run_context.max_cases]:
            context = sample["training_context"]
            target = context["targets"]
            t = np.asarray(target["t"])
            delta = np.asarray(target["delta"])
            omega = np.asarray(target["omega"])
            pm = np.asarray(context["Pm_observed"])
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"])
            incidence = np.asarray(static["incidence"])
            equilibrium = np.asarray(static["equilibrium_delta"])
            limit = min(len(t), max(run_context.horizon_steps * 5, 20))
            domega = np.gradient(omega[:limit], t[:limit], axis=0, edge_order=1)
            for k in range(limit):
                designs.append(self._features(delta[k], omega[k], equilibrium, incidence))
                targets.append(mass * domega[k] - pm[k])
        matrix = np.concatenate(designs, axis=0)
        vector = np.concatenate(targets, axis=0)
        ridge = 1e-6
        augmented = np.vstack([matrix, np.sqrt(ridge) * np.eye(matrix.shape[1])])
        augmented_y = np.concatenate([vector, np.zeros(matrix.shape[1])])
        solution = lsq_linear(augmented, augmented_y, bounds=(0.0, np.inf), max_iter=500)
        n_nodes = designs[0].shape[0]
        self.damping = solution.x[:n_nodes]
        self.coupling = solution.x[n_nodes:]
        losses = []
        for sample in val[: run_context.max_cases]:
            static = sample["model_inputs"]["static"]
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), run_context.horizon_steps)
            dp, wp = self.rollout(
                static,
                target["t"][:steps],
                {"Pm": sample["model_inputs"]["exogenous"]["Pm"][:steps]},
            )
            losses.append(float(np.mean((dp - target["delta"][:steps]) ** 2) + np.mean((wp - target["omega"][:steps]) ** 2)))
        return FitResult(
            method=self.method_name,
            status="completed",
            train_case_ids=tuple(s["event_id"] for s in train[: run_context.max_cases]),
            val_case_ids=tuple(s["event_id"] for s in val[: run_context.max_cases]),
            best_validation_loss=float(np.mean(losses)),
            details={"optimizer": "bounded_ridge", "identifiable_quantity": "D/M and K/M"},
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
        return pm - self.damping * omega - incidence @ (
            self.coupling * (np.sin(eta) - np.sin(eta0))
        )

    def vector_field(
        self, delta: np.ndarray, omega: np.ndarray, pm: np.ndarray, static: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
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
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, damping=self.damping, coupling=self.coupling)
        path.with_suffix(".json").write_text(
            json.dumps({"method": self.method_name, "unit_policy": "M_over_omega_s"}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "M0Model":
        with np.load(path) as data:
            return cls(data["damping"], data["coupling"])

    def physics_audit(self) -> dict[str, float | str | bool]:
        if self.damping is None or self.coupling is None:
            return {"status": "not_fitted"}
        return {
            "status": "pass",
            "minimum_damping": float(np.min(self.damping)),
            "minimum_coupling": float(np.min(self.coupling)),
            "nonnegative_parameters": bool(np.all(self.damping >= 0) and np.all(self.coupling >= 0)),
            "internal_force_balance_by_construction": True,
        }
