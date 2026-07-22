from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import DynamicsModel, FitResult, RunContext


class ControlledOperatorModel(DynamicsModel):
    """Supplementary DMD/EDMD operator with an explicit prefix information budget."""

    method_name = "DMD"

    def __init__(self, lifting: str = "linear", matrix=None, n_nodes: int | None = None):
        self.lifting = lifting
        self.matrix = None if matrix is None else np.asarray(matrix)
        self.n_nodes = n_nodes

    def _lift(self, state, pm):
        features = [state, pm]
        if self.lifting == "quadratic":
            features.append(state * state)
        return np.concatenate(features)

    def fit(self, train, val, run_context):
        xs, ys = [], []
        for sample in train[:run_context.max_cases]:
            target = sample["training_context"]["targets"]
            pm = sample["training_context"]["Pm_observed"]
            state = np.concatenate((target["delta"], target["omega"]), axis=1)
            limit = min(len(state) - 1, run_context.horizon_steps * 4)
            for k in range(limit): xs.append(self._lift(state[k], pm[k])); ys.append(state[k + 1])
        x, y = np.asarray(xs), np.asarray(ys)
        self.matrix = np.linalg.solve(x.T @ x + 1e-5*np.eye(x.shape[1]), x.T @ y)
        self.n_nodes = y.shape[1] // 2
        return FitResult(self.method_name, "completed", tuple(s["event_id"] for s in train[:run_context.max_cases]), tuple(s["event_id"] for s in val[:run_context.max_cases]), float("nan"), {"supplementary": True, "lifting": self.lifting})

    def rollout(self, initial_state, time_grid, exogenous):
        state = np.concatenate((np.asarray(initial_state["delta0"]).reshape(-1), np.asarray(initial_state["omega0"]).reshape(-1)))
        out = np.empty((len(time_grid), len(state))); out[0] = state
        for k in range(len(time_grid)-1): out[k+1] = self._lift(out[k], exogenous["Pm"][k]) @ self.matrix
        return out[:, :self.n_nodes], out[:, self.n_nodes:]

    def save(self, path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,matrix=self.matrix,n_nodes=self.n_nodes,lifting=self.lifting)

    @classmethod
    def load(cls,path):
        with np.load(path) as d: return cls(str(d["lifting"]),d["matrix"],int(d["n_nodes"]))

    def physics_audit(self): return {"status":"not_applicable","supplementary":True,"certificate_claim":False}


class DMDModel(ControlledOperatorModel):
    method_name = "DMD"

    def __init__(self, matrix=None, n_nodes: int | None = None):
        super().__init__("linear", matrix, n_nodes)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["matrix"], int(data["n_nodes"]))


class EDMDKoopmanModel(ControlledOperatorModel):
    method_name = "EDMD-Koopman"

    def __init__(self, matrix=None, n_nodes: int | None = None):
        super().__init__("quadratic", matrix, n_nodes)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data["matrix"], int(data["n_nodes"]))


class HODMDPrefixModel:
    """Delay operator that requires a fixed observed prefix at inference.

    This intentionally does not implement ``DynamicsModel``: its delay state
    gives it a larger information budget than initial-state rollout methods.
    """

    method_name = "HODMD"

    def __init__(self, order: int = 4, matrix=None, n_nodes: int | None = None):
        if order < 2:
            raise ValueError("HODMD order must be at least two")
        self.order = order
        self.matrix = None if matrix is None else np.asarray(matrix, dtype=np.float64)
        self.n_nodes = n_nodes

    def fit(self, train, val, run_context):
        xs, ys = [], []
        for sample in train[: run_context.max_cases]:
            observed = sample["model_inputs"].get("observed_context")
            if observed is None or len(observed["t"]) <= self.order:
                raise ValueError("HODMD requires a fixed prefix longer than its delay order")
            state = np.concatenate((observed["delta"], observed["omega"]), axis=1)
            for index in range(self.order - 1, len(state) - 1):
                delayed = state[index - self.order + 1 : index + 1].reshape(-1)
                xs.append(np.concatenate((delayed, observed["Pm"][index])))
                ys.append(state[index + 1])
        design = np.asarray(xs)
        target = np.asarray(ys)
        self.matrix = np.linalg.solve(
            design.T @ design + 1e-5 * np.eye(design.shape[1]), design.T @ target
        )
        self.n_nodes = target.shape[1] // 2
        return FitResult(
            self.method_name,
            "completed",
            tuple(sample["event_id"] for sample in train[: run_context.max_cases]),
            tuple(sample["event_id"] for sample in val[: run_context.max_cases]),
            float("nan"),
            {"supplementary": True, "required_prefix_order": self.order},
        )

    def forecast(self, sample, time_grid, exogenous):
        if self.matrix is None or self.n_nodes is None:
            raise RuntimeError("HODMD is not fitted")
        observed = sample["model_inputs"].get("observed_context")
        if observed is None or len(observed["t"]) < self.order:
            raise ValueError("HODMD forecast requires the registered fixed prefix")
        history = list(
            np.concatenate((observed["delta"], observed["omega"]), axis=1)[-self.order :]
        )
        output = np.empty((len(time_grid), 2 * self.n_nodes), dtype=np.float64)
        output[0] = history[-1]
        pm = np.asarray(exogenous["Pm"], dtype=np.float64)
        for index in range(len(time_grid) - 1):
            feature = np.concatenate((np.asarray(history[-self.order :]).reshape(-1), pm[index]))
            state = feature @ self.matrix
            if not np.isfinite(state).all():
                raise FloatingPointError(f"nonfinite HODMD forecast at step {index + 1}")
            history.append(state)
            output[index + 1] = state
        return output[:, : self.n_nodes], output[:, self.n_nodes :]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            order=self.order,
            matrix=self.matrix,
            n_nodes=self.n_nodes,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(int(data["order"]), data["matrix"], int(data["n_nodes"]))
