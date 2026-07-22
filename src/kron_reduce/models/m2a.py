from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import DynamicsModel, FitResult, RunContext
from .integration import rk4_rollout
from .m0 import M0Model


class M2AModel(DynamicsModel):
    method_name = "M2A"

    def __init__(
        self,
        base: M0Model | None = None,
        node_theta: np.ndarray | None = None,
        edge_theta: np.ndarray | None = None,
    ):
        self.base = base or M0Model()
        self.node_theta = None if node_theta is None else np.asarray(node_theta, dtype=np.float64)
        self.edge_theta = None if edge_theta is None else np.asarray(edge_theta, dtype=np.float64)

    @staticmethod
    def _residual_force_numpy(node_theta, edge_theta, delta, omega, static):
        incidence = np.asarray(static["incidence"])
        equilibrium = np.asarray(static["equilibrium_delta"])
        eta = incidence.T @ delta
        eta0 = incidence.T @ equilibrium
        node = node_theta[:, 0] * omega + node_theta[:, 1] * omega * np.abs(omega)
        edge = edge_theta[:, 0] * (np.sin(2 * eta) - np.sin(2 * eta0))
        edge += edge_theta[:, 1] * (np.cos(eta) - np.cos(eta0))
        return node - incidence @ edge

    def fit(self, train, val, run_context):
        if self.base.damping is None:
            self.base.fit(train, val, run_context)
        n_nodes = train[0]["model_inputs"]["static"]["mass"].size
        n_edges = train[0]["model_inputs"]["static"]["incidence"].shape[1]
        torch.manual_seed(run_context.seed)
        node = torch.nn.Parameter(torch.zeros((n_nodes, 2), dtype=torch.float64))
        edge = torch.nn.Parameter(torch.zeros((n_edges, 2), dtype=torch.float64))
        optimizer = torch.optim.Adam([node, edge], lr=run_context.learning_rate)
        base_d = torch.as_tensor(self.base.damping, dtype=torch.float64)
        base_k = torch.as_tensor(self.base.coupling, dtype=torch.float64)
        best_loss = float("inf")
        best = None
        for _ in range(max(1, run_context.epochs)):
            optimizer.zero_grad()
            losses = []
            for sample in train[: run_context.max_cases]:
                target = sample["training_context"]["targets"]
                static = sample["model_inputs"]["static"]
                steps = min(len(target["t"]), run_context.horizon_steps)
                dt = torch.as_tensor(np.diff(target["t"][:steps]), dtype=torch.float64)
                truth_d = torch.as_tensor(target["delta"][:steps], dtype=torch.float64)
                truth_w = torch.as_tensor(target["omega"][:steps], dtype=torch.float64)
                pm = torch.as_tensor(sample["model_inputs"]["exogenous"]["Pm"][:steps], dtype=torch.float64)
                mass = torch.as_tensor(static["mass"], dtype=torch.float64)
                inc = torch.as_tensor(static["incidence"], dtype=torch.float64)
                eq = torch.as_tensor(static["equilibrium_delta"], dtype=torch.float64)
                d, w = truth_d[0], truth_w[0]
                predictions_d = [d]
                predictions_w = [w]
                eta0 = inc.T @ eq
                for k in range(steps - 1):
                    eta = inc.T @ d
                    base_force = pm[k] - base_d * w - inc @ (base_k * (torch.sin(eta) - torch.sin(eta0)))
                    node_force = node[:, 0] * w + node[:, 1] * w * torch.abs(w)
                    edge_force = edge[:, 0] * (torch.sin(2 * eta) - torch.sin(2 * eta0))
                    edge_force = edge_force + edge[:, 1] * (torch.cos(eta) - torch.cos(eta0))
                    acc = (base_force + node_force - inc @ edge_force) / mass
                    d = d + dt[k] * w
                    w = w + dt[k] * acc
                    predictions_d.append(d)
                    predictions_w.append(w)
                pd = torch.stack(predictions_d)
                pw = torch.stack(predictions_w)
                losses.append(torch.mean((pd - truth_d) ** 2) + torch.mean((pw - truth_w) ** 2))
            loss = torch.mean(torch.stack(losses)) + 1e-6 * (torch.sum(node**2) + torch.sum(edge**2))
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if value < best_loss:
                best_loss = value
                best = (node.detach().clone(), edge.detach().clone())
        assert best is not None
        self.node_theta = best[0].cpu().numpy()
        self.edge_theta = best[1].cpu().numpy()
        validation = self._validation_loss(val, run_context)
        return FitResult(
            self.method_name,
            "completed",
            tuple(s["event_id"] for s in train[: run_context.max_cases]),
            tuple(s["event_id"] for s in val[: run_context.max_cases]),
            validation,
            {"objective": "multi_horizon_closed_loop", "epochs": run_context.epochs},
        )

    def _validation_loss(self, val, context):
        losses = []
        for sample in val[: context.max_cases]:
            target = sample["training_context"]["targets"]
            steps = min(len(target["t"]), context.horizon_steps)
            dp, wp = self.rollout(
                sample["model_inputs"]["static"],
                target["t"][:steps],
                {"Pm": sample["model_inputs"]["exogenous"]["Pm"][:steps]},
            )
            losses.append(np.mean((dp - target["delta"][:steps]) ** 2) + np.mean((wp - target["omega"][:steps]) ** 2))
        return float(np.mean(losses))

    def force(self, delta, omega, pm, static):
        if self.node_theta is None or self.edge_theta is None:
            raise RuntimeError("M2A is not fitted")
        return self.base.force(delta, omega, pm, static) + self._residual_force_numpy(
            self.node_theta, self.edge_theta, delta, omega, static
        )

    def vector_field(self, delta, omega, pm, static):
        return omega, self.force(delta, omega, pm, static) / np.asarray(static["mass"])

    def rollout(self, initial_state, time_grid, exogenous):
        return rk4_rollout(
            self.vector_field,
            np.asarray(initial_state["delta0"]).reshape(-1),
            np.asarray(initial_state["omega0"]).reshape(-1),
            time_grid,
            exogenous["Pm"],
            initial_state,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            damping=self.base.damping,
            coupling=self.base.coupling,
            node_theta=self.node_theta,
            edge_theta=self.edge_theta,
        )
        path.with_suffix(".json").write_text(json.dumps({"method": self.method_name}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(M0Model(data["damping"], data["coupling"]), data["node_theta"], data["edge_theta"])

    def physics_audit(self):
        if self.node_theta is None:
            return {"status": "not_fitted"}
        return {
            "status": "pass",
            "edge_internal_force_balance_by_construction": True,
            "residual_parameter_norm": float(np.linalg.norm(self.node_theta) + np.linalg.norm(self.edge_theta)),
            "certificate_claim": False,
        }
