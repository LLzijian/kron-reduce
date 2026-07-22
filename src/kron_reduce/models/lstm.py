from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import DynamicsModel, FitResult, RunContext


class _AutoregressiveCore(torch.nn.Module):
    def __init__(self, n_nodes: int, hidden: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.init_h = torch.nn.Linear(2 * n_nodes, hidden)
        self.init_c = torch.nn.Linear(2 * n_nodes, hidden)
        self.cell = torch.nn.LSTMCell(3 * n_nodes, hidden)
        self.derivative = torch.nn.Linear(hidden, 2 * n_nodes)

    def initial_hidden(self, state):
        return torch.tanh(self.init_h(state)), torch.tanh(self.init_c(state))

    def step(self, state, pm, hidden):
        h, c = self.cell(torch.cat((state, pm), dim=-1), hidden)
        return self.derivative(h), (h, c)


class AutoregressiveLSTMModel(DynamicsModel):
    method_name = "LSTM"

    def __init__(self, n_nodes: int | None = None, hidden: int = 32):
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.core = _AutoregressiveCore(n_nodes, hidden).double() if n_nodes is not None else None

    def _loss(self, samples, context, train_mode):
        losses = []
        for sample in samples[: context.max_cases]:
            target = sample["training_context"]["targets"]
            pm = torch.as_tensor(sample["training_context"]["Pm_observed"], dtype=torch.float64)
            delta = torch.as_tensor(target["delta"], dtype=torch.float64)
            omega = torch.as_tensor(target["omega"], dtype=torch.float64)
            t = torch.as_tensor(target["t"], dtype=torch.float64)
            steps = min(len(t), context.horizon_steps)
            state0 = torch.cat((delta[0], omega[0])).unsqueeze(0)
            hidden = self.core.initial_hidden(state0)
            for k in range(steps - 1):
                # Teacher forcing is confined to optimizer/validation sequences.
                current = torch.cat((delta[k], omega[k])).unsqueeze(0)
                derivative, hidden = self.core.step(current, pm[k].unsqueeze(0), hidden)
                true_derivative = (torch.cat((delta[k + 1], omega[k + 1])) - current.squeeze(0)) / (t[k + 1] - t[k])
                losses.append(torch.mean((derivative.squeeze(0) - true_derivative) ** 2))
        return torch.mean(torch.stack(losses))

    def fit(self, train, val, run_context):
        self.n_nodes = train[0]["model_inputs"]["static"]["mass"].size
        torch.manual_seed(run_context.seed)
        self.core = _AutoregressiveCore(self.n_nodes, self.hidden).double()
        optimizer = torch.optim.Adam(self.core.parameters(), lr=run_context.learning_rate)
        best_loss, best_state = float("inf"), None
        for _ in range(max(1, run_context.epochs)):
            optimizer.zero_grad(); loss = self._loss(train, run_context, True); loss.backward(); optimizer.step()
            with torch.no_grad():
                validation = float(self._loss(val, run_context, False))
            if validation < best_loss:
                best_loss = validation
                best_state = {key: value.detach().clone() for key, value in self.core.state_dict().items()}
        assert best_state is not None
        self.core.load_state_dict(best_state)
        return FitResult(self.method_name, "completed", tuple(s["event_id"] for s in train[:run_context.max_cases]), tuple(s["event_id"] for s in val[:run_context.max_cases]), best_loss, {"test_teacher_forcing": False, "hidden_initialization": "initial_state", "checkpoint_restored": True})

    def rollout(self, initial_state, time_grid, exogenous):
        if self.core is None:
            raise RuntimeError("LSTM is not fitted")
        t = np.asarray(time_grid, dtype=np.float64); pm = np.asarray(exogenous["Pm"], dtype=np.float64)
        delta = np.empty((len(t), self.n_nodes)); omega = np.empty_like(delta)
        delta[0] = np.asarray(initial_state["delta0"]).reshape(-1); omega[0] = np.asarray(initial_state["omega0"]).reshape(-1)
        with torch.no_grad():
            state = torch.as_tensor(np.concatenate((delta[0], omega[0])), dtype=torch.float64).unsqueeze(0)
            hidden = self.core.initial_hidden(state)
            for k in range(len(t) - 1):
                derivative, hidden = self.core.step(state, torch.as_tensor(pm[k], dtype=torch.float64).unsqueeze(0), hidden)
                dt = float(t[k + 1] - t[k])
                state = state + dt * derivative
                values = state.squeeze(0).numpy()
                delta[k + 1], omega[k + 1] = values[:self.n_nodes], values[self.n_nodes:]
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"nonfinite LSTM rollout at step {k + 1}")
        return delta, omega

    def save(self, path):
        if self.core is None:
            raise RuntimeError("LSTM is not fitted")
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"method": self.method_name, "n_nodes": self.n_nodes, "hidden": self.hidden, "state_dict": self.core.state_dict()}, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(int(payload["n_nodes"]), int(payload["hidden"])); model.core.load_state_dict(payload["state_dict"]); return model

    def physics_audit(self):
        return {"status": "pass" if self.core is not None else "not_fitted", "autoregressive_inference": True, "test_teacher_forcing": False, "certificate_claim": False}
