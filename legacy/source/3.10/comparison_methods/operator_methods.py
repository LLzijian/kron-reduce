"""Controlled DMD/HODMD/EDMD forecast baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from support.common import dpm_array, state_array


@dataclass
class OperatorModel:
    method: str
    coef: np.ndarray
    n_node: int
    n_state: int
    delay: int
    lifting: str


def _lift_current(x: np.ndarray, u: np.ndarray, n_node: int, lifting: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if u.ndim == 1:
        u = u.reshape(1, -1)
    base = [np.ones((len(x), 1)), x, u]
    if lifting == "trig":
        delta = x[:, :n_node]
        base.extend([np.clip(x, -50, 50) ** 2, np.sin(delta), np.cos(delta)])
    return np.column_stack(base)


def _make_feature(history: np.ndarray, u: np.ndarray, n_node: int, method: str) -> np.ndarray:
    if method == "HODMD":
        return np.concatenate([[1.0], history.reshape(-1), np.asarray(u, dtype=np.float64).reshape(-1)])
    lifting = "trig" if method == "EDMD-Koopman" else "linear"
    return _lift_current(history[-1], u, n_node, lifting).ravel()


def fit_operator(train: list[dict[str, Any]], method: str, window_len: int, seed: int, sample_cap: int = 8000, ridge: float = 1e-5) -> OperatorModel:
    rng = np.random.default_rng(seed)
    n_node = int(train[0]["delta"].shape[1])
    n_state = 2 * n_node
    delay = min(max(1, int(window_len)), 12 if method == "HODMD" else 1)
    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []
    for seq in train:
        x = state_array(seq)
        u = dpm_array(seq)
        if len(x) <= delay + 1:
            continue
        for k in range(delay - 1, len(x) - 1):
            hist = x[k - delay + 1:k + 1]
            X_rows.append(_make_feature(hist, u[k], n_node, method))
            Y_rows.append(x[k + 1])
    X = np.vstack(X_rows)
    Y = np.vstack(Y_rows)
    if len(X) > sample_cap:
        idx = rng.choice(len(X), size=sample_cap, replace=False)
        X, Y = X[idx], Y[idx]
    coef = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)
    return OperatorModel(method=method, coef=coef, n_node=n_node, n_state=n_state, delay=delay, lifting="trig" if method == "EDMD-Koopman" else "linear")


def rollout_operator(model: OperatorModel, seq: dict[str, Any], window_len: int) -> tuple[np.ndarray, np.ndarray]:
    x_true = state_array(seq)
    u = dpm_array(seq)
    start = max(model.delay - 1, min(int(window_len) - 1, len(x_true) - 2))
    pred = np.zeros_like(x_true)
    pred[: start + 1] = x_true[: start + 1]
    history = [x_true[i].copy() for i in range(start - model.delay + 1, start + 1)]
    for k in range(start, len(x_true) - 1):
        feat = _make_feature(np.asarray(history[-model.delay:]), u[k], model.n_node, model.method)
        x_next = feat.reshape(1, -1) @ model.coef
        x_next = np.asarray(x_next).ravel()
        if not np.all(np.isfinite(x_next)) or np.max(np.abs(x_next)) > 1e6:
            raise RuntimeError(f"{model.method} rollout diverged")
        pred[k + 1] = x_next
        history.append(x_next)
    return pred[:, : model.n_node], pred[:, model.n_node:]
