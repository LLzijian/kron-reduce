from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .base import DynamicsModel, FitResult, RunContext
from .integration import rk4_rollout
from .m0 import M0Model


class EnergyM2AModel(DynamicsModel):
    method_name = "M2A-energy"

    def __init__(self, base=None, residual_damping=None, residual_coupling=None, reported_name=None):
        self.base = base or M0Model()
        self.residual_damping = None if residual_damping is None else np.asarray(residual_damping, dtype=np.float64)
        self.residual_coupling = None if residual_coupling is None else np.asarray(residual_coupling, dtype=np.float64)
        self.reported_name = reported_name or "M2A-energy-regularized"

    def fit(self, train, val, run_context):
        if self.base.damping is None:
            self.base.fit(train, val, run_context)
        designs = []
        targets = []
        for sample in train[: run_context.max_cases]:
            context = sample["training_context"]
            target = context["targets"]
            t, delta, omega = target["t"], target["delta"], target["omega"]
            pm = context["Pm_observed"]
            static = sample["model_inputs"]["static"]
            mass = static["mass"]
            inc = static["incidence"]
            eq = static["equilibrium_delta"]
            limit = min(len(t), max(run_context.horizon_steps * 5, 20))
            domega = np.gradient(omega[:limit], t[:limit], axis=0, edge_order=1)
            for k in range(limit):
                residual = mass * domega[k] - self.base.force(delta[k], omega[k], pm[k], static)
                eta = inc.T @ delta[k]
                eta0 = inc.T @ eq
                design = np.zeros((len(mass), len(mass) + inc.shape[1]))
                design[np.arange(len(mass)), np.arange(len(mass))] = -omega[k]
                design[:, len(mass):] = -inc * (np.sin(eta) - np.sin(eta0))
                designs.append(design)
                targets.append(residual)
        matrix = np.concatenate(designs)
        vector = np.concatenate(targets)
        result = lsq_linear(matrix, vector, bounds=(0, np.inf), max_iter=500)
        n = designs[0].shape[0]
        self.residual_damping = result.x[:n]
        self.residual_coupling = result.x[n:]
        validation = []
        for sample in val[: run_context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), run_context.horizon_steps)
            dp, wp = self.rollout(sample["model_inputs"]["static"], target["t"][:steps], {"Pm": sample["model_inputs"]["exogenous"]["Pm"][:steps]})
            validation.append(np.mean((dp - target["delta"][:steps]) ** 2) + np.mean((wp - target["omega"][:steps]) ** 2))
        return FitResult(
            self.method_name,
            "completed",
            tuple(s["event_id"] for s in train[: run_context.max_cases]),
            tuple(s["event_id"] for s in val[: run_context.max_cases]),
            float(np.mean(validation)),
            {"reported_name": self.reported_name, "objective": "bounded_energy_structured_residual"},
        )

    def force(self, delta, omega, pm, static):
        if self.residual_damping is None or self.residual_coupling is None:
            raise RuntimeError("energy model is not fitted")
        inc = np.asarray(static["incidence"])
        eta = inc.T @ delta
        eta0 = inc.T @ np.asarray(static["equilibrium_delta"])
        residual = -self.residual_damping * omega - inc @ (
            self.residual_coupling * (np.sin(eta) - np.sin(eta0))
        )
        return self.base.force(delta, omega, pm, static) + residual

    def vector_field(self, delta, omega, pm, static):
        return omega, self.force(delta, omega, pm, static) / np.asarray(static["mass"])

    def rollout(self, initial_state, time_grid, exogenous):
        return rk4_rollout(self.vector_field, np.asarray(initial_state["delta0"]).reshape(-1), np.asarray(initial_state["omega0"]).reshape(-1), time_grid, exogenous["Pm"], initial_state)

    def storage(self, delta, omega, static):
        inc = np.asarray(static["incidence"])
        eq = np.asarray(static["equilibrium_delta"])
        eta = inc.T @ delta
        eta0 = inc.T @ eq
        coupling = self.base.coupling + self.residual_coupling
        shifted = np.cos(eta0) - np.cos(eta) - np.sin(eta0) * (eta - eta0)
        return 0.5 * np.sum(np.asarray(static["mass"]) * omega**2) + np.sum(coupling * shifted)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, damping=self.base.damping, coupling=self.base.coupling, residual_damping=self.residual_damping, residual_coupling=self.residual_coupling)
        path.with_suffix(".json").write_text(json.dumps({"method": self.method_name, "reported_name": self.reported_name}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            meta_path = Path(path).with_suffix(".json")
            reported = json.loads(meta_path.read_text(encoding="utf-8")).get("reported_name") if meta_path.exists() else None
            return cls(M0Model(data["damping"], data["coupling"]), data["residual_damping"], data["residual_coupling"], reported)

    def physics_audit(self):
        if self.residual_damping is None:
            return {"status": "not_fitted"}
        return {
            "status": "pending_certificate_gate",
            "minimum_total_damping": float(np.min(self.base.damping + self.residual_damping)),
            "minimum_total_coupling": float(np.min(self.base.coupling + self.residual_coupling)),
            "internal_force_balance_by_construction": True,
            "reported_name": self.reported_name,
        }
