"""SINDYc-style controlled sparse polynomial baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from support.common import dpm_array, r2_score_np, rmse_np, state_array


@dataclass
class SindycModel:
    coef: np.ndarray
    threshold: float
    ridge: float
    n_state: int
    n_node: int


def _features(x: np.ndarray, u: np.ndarray, n_node: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if u.ndim == 1:
        u = u.reshape(1, -1)
    delta = x[:, :n_node]
    omega = x[:, n_node:]
    return np.column_stack([
        np.ones(len(x)),
        x,
        np.clip(x, -50.0, 50.0) ** 2,
        np.sin(delta),
        np.cos(delta),
        u,
        u * omega,
    ])


def fit_sindyc(train: list[dict[str, Any]], seed: int, sample_cap: int = 6000, ridge: float = 1e-4, threshold: float = 1e-5) -> SindycModel:
    rng = np.random.default_rng(seed)
    Xs: list[np.ndarray] = []
    dXs: list[np.ndarray] = []
    Us: list[np.ndarray] = []
    n_node = int(train[0]["delta"].shape[1])
    for seq in train:
        x = state_array(seq)
        u = dpm_array(seq)
        dt = float(seq.get("dt_actual", np.median(np.diff(seq["t"]))))
        dx = np.gradient(x, dt, axis=0, edge_order=2)
        Xs.append(x)
        dXs.append(dx)
        Us.append(u)
    X = np.vstack(Xs)
    dX = np.vstack(dXs)
    U = np.vstack(Us)
    if len(X) > sample_cap:
        idx = rng.choice(len(X), size=sample_cap, replace=False)
        X, dX, U = X[idx], dX[idx], U[idx]
    Phi = _features(X, U, n_node)
    coef = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ dX)
    coef[np.abs(coef) < threshold] = 0.0
    return SindycModel(coef=coef, threshold=threshold, ridge=ridge, n_state=X.shape[1], n_node=n_node)


def rollout_sindyc(model: SindycModel, seq: dict[str, Any], start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(seq["t"], dtype=np.float64)
    x_true = state_array(seq)
    u = dpm_array(seq)
    pred = np.zeros_like(x_true)
    pred[: start + 1] = x_true[: start + 1]
    x = x_true[start].copy()
    for k in range(start, len(t) - 1):
        dt = float(t[k + 1] - t[k])
        dx = _features(x, u[k], model.n_node).reshape(1, -1) @ model.coef
        x = x + dt * np.asarray(dx).ravel()
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
            raise RuntimeError("SINDYc rollout diverged")
        pred[k + 1] = x
    return pred[:, : model.n_node], pred[:, model.n_node:]


def fit_score_sindyc(model: SindycModel, seqs: list[dict[str, Any]]) -> dict[str, float]:
    vals = []
    rmses = []
    for seq in seqs:
        x = state_array(seq)
        u = dpm_array(seq)
        dt = float(seq.get("dt_actual", np.median(np.diff(seq["t"]))))
        dx_true = np.gradient(x, dt, axis=0, edge_order=2)
        dx_pred = _features(x, u, model.n_node) @ model.coef
        vals.append(r2_score_np(dx_true, dx_pred))
        rmses.append(rmse_np(dx_true, dx_pred))
    return {
        "fit_r2": float(np.nanmean(vals)) if vals else float("nan"),
        "fit_rmse": float(np.nanmean(rmses)) if rmses else float("nan"),
        "active_terms": int(np.sum(np.abs(model.coef) > 0.0)),
    }
