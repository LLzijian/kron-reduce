from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import DynamicsModel, FitResult, RunContext
from .integration import rk4_rollout
from .m0 import M0Model


class _PINNCore(torch.nn.Module):
    def __init__(self, n_nodes: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 * n_nodes, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_nodes),
        )

    def forward(self, delta_relative, omega, pm):
        inputs = torch.cat((delta_relative, omega, pm), dim=-1)
        zeros = torch.zeros_like(inputs)
        return self.net(inputs) - self.net(zeros)


class PINNModel(DynamicsModel):
    method_name = "PINN"

    def __init__(self, base=None, n_nodes: int | None = None, hidden: int = 24):
        self.base = base or M0Model()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.core = _PINNCore(n_nodes, hidden).double() if n_nodes is not None else None

    def _dataset(self, samples, context):
        xs, ps, ys = [], [], []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            static = sample["model_inputs"]["static"]
            pm = sample["training_context"]["Pm_observed"]
            limit = min(len(target["t"]), max(20, context.horizon_steps * 4))
            domega = np.gradient(target["omega"][:limit], target["t"][:limit], axis=0)
            for k in range(limit):
                residual = static["mass"] * domega[k] - self.base.force(target["delta"][k], target["omega"][k], pm[k], static)
                xs.append(np.concatenate((target["delta"][k] - static["equilibrium_delta"], target["omega"][k])))
                ps.append(pm[k]); ys.append(residual)
        return tuple(torch.as_tensor(np.asarray(v), dtype=torch.float64) for v in (xs, ps, ys))

    def fit(self, train, val, run_context):
        if self.base.damping is None:
            self.base.fit(train, val, run_context)
        self.n_nodes = train[0]["model_inputs"]["static"]["mass"].size
        torch.manual_seed(run_context.seed)
        self.core = _PINNCore(self.n_nodes, self.hidden).double()
        optimizer = torch.optim.Adam(self.core.parameters(), lr=run_context.learning_rate)
        train_x, train_p, train_y = self._dataset(train, run_context)
        val_x, val_p, val_y = self._dataset(val, run_context)
        best_loss, best_state = float("inf"), None
        for _ in range(max(1, run_context.epochs)):
            optimizer.zero_grad()
            prediction = self.core(train_x[:, :self.n_nodes], train_x[:, self.n_nodes:], train_p)
            equilibrium_penalty = torch.mean(self.core(torch.zeros_like(train_x[:1, :self.n_nodes]), torch.zeros_like(train_x[:1, self.n_nodes:]), torch.zeros_like(train_p[:1])) ** 2)
            loss = torch.mean((prediction - train_y) ** 2) + equilibrium_penalty
            loss.backward(); optimizer.step()
            with torch.no_grad():
                vp = self.core(val_x[:, :self.n_nodes], val_x[:, self.n_nodes:], val_p)
                validation = float(torch.mean((vp - val_y) ** 2))
            if validation < best_loss:
                best_loss = validation
                best_state = {key: value.detach().clone() for key, value in self.core.state_dict().items()}
        assert best_state is not None
        self.core.load_state_dict(best_state)
        return FitResult(self.method_name, "completed", tuple(s["event_id"] for s in train[:run_context.max_cases]), tuple(s["event_id"] for s in val[:run_context.max_cases]), best_loss, {"shared_across_cases": True, "checkpoint_restored": True, "loss": "derivative_plus_equilibrium_physics"})

    def vector_field(self, delta, omega, pm, static):
        if self.core is None:
            raise RuntimeError("PINN is not fitted")
        with torch.no_grad():
            residual = self.core(
                torch.as_tensor(delta - static["equilibrium_delta"], dtype=torch.float64).unsqueeze(0),
                torch.as_tensor(omega, dtype=torch.float64).unsqueeze(0),
                torch.as_tensor(pm, dtype=torch.float64).unsqueeze(0),
            ).squeeze(0).numpy()
        force = self.base.force(delta, omega, pm, static) + residual
        return omega, force / np.asarray(static["mass"])

    def rollout(self, initial_state, time_grid, exogenous):
        return rk4_rollout(self.vector_field, np.asarray(initial_state["delta0"]).reshape(-1), np.asarray(initial_state["omega0"]).reshape(-1), time_grid, exogenous["Pm"], initial_state)

    def save(self, path):
        if self.core is None:
            raise RuntimeError("PINN is not fitted")
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": self.method_name, "n_nodes": self.n_nodes, "hidden": self.hidden, "state_dict": self.core.state_dict(), "base_damping": self.base.damping, "base_coupling": self.base.coupling}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(M0Model(payload["base_damping"], payload["base_coupling"]), int(payload["n_nodes"]), int(payload["hidden"]))
        model.core.load_state_dict(payload["state_dict"])
        return model

    def physics_audit(self):
        return {"status": "pass" if self.core is not None else "not_fitted", "shared_model": True, "equilibrium_zero_by_construction": True, "certificate_claim": False}
