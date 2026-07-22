from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import DynamicsModel, FitResult, RunContext
from .integration import rk4_rollout


class SINDYcModel(DynamicsModel):
    method_name = "SINDYc"

    def __init__(self, coefficients=None, n_nodes: int | None = None):
        self.coefficients = None if coefficients is None else np.asarray(coefficients, dtype=np.float64)
        self.n_nodes = n_nodes

    @staticmethod
    def _library(delta, omega, pm, equilibrium):
        relative = delta - equilibrium
        return np.concatenate((
            np.ones(1), relative, omega, relative * np.abs(relative),
            omega * np.abs(omega), pm,
        ))

    def fit(self, train, val, run_context):
        features, derivatives = [], []
        for sample in train[: run_context.max_cases]:
            target = sample["training_context"]["targets"]
            static = sample["model_inputs"]["static"]
            pm = sample["training_context"]["Pm_observed"]
            limit = min(len(target["t"]), max(20, run_context.horizon_steps * 5))
            ddelta = np.gradient(target["delta"][:limit], target["t"][:limit], axis=0)
            domega = np.gradient(target["omega"][:limit], target["t"][:limit], axis=0)
            for k in range(limit):
                features.append(self._library(target["delta"][k], target["omega"][k], pm[k], static["equilibrium_delta"]))
                derivatives.append(np.concatenate((ddelta[k], domega[k])))
        theta = np.asarray(features)
        y = np.asarray(derivatives)
        ridge = 1e-5
        self.coefficients = np.linalg.solve(theta.T @ theta + ridge * np.eye(theta.shape[1]), theta.T @ y)
        self.coefficients[np.abs(self.coefficients) < 1e-6] = 0.0
        self.n_nodes = y.shape[1] // 2
        losses = []
        for sample in val[: run_context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), run_context.horizon_steps)
            dp, wp = self.rollout(sample["model_inputs"]["static"], target["t"][:steps], {"Pm": sample["model_inputs"]["exogenous"]["Pm"][:steps]})
            losses.append(np.mean((dp - target["delta"][:steps]) ** 2) + np.mean((wp - target["omega"][:steps]) ** 2))
        return FitResult(self.method_name, "completed", tuple(s["event_id"] for s in train[:run_context.max_cases]), tuple(s["event_id"] for s in val[:run_context.max_cases]), float(np.mean(losses)), {"library": "controlled_centered_quadratic"})

    def vector_field(self, delta, omega, pm, static):
        if self.coefficients is None or self.n_nodes is None:
            raise RuntimeError("SINDYc is not fitted")
        out = self._library(delta, omega, pm, static["equilibrium_delta"]) @ self.coefficients
        return out[: self.n_nodes], out[self.n_nodes :]

    def rollout(self, initial_state, time_grid, exogenous):
        return rk4_rollout(self.vector_field, np.asarray(initial_state["delta0"]).reshape(-1), np.asarray(initial_state["omega0"]).reshape(-1), time_grid, exogenous["Pm"], initial_state)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, coefficients=self.coefficients, n_nodes=self.n_nodes)
        path.with_suffix(".json").write_text(json.dumps({"method": self.method_name}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["coefficients"], int(data["n_nodes"]))

    def physics_audit(self):
        return {"status": "pass" if self.coefficients is not None else "not_fitted", "structure": "unconstrained_sparse_controlled", "certificate_claim": False}
