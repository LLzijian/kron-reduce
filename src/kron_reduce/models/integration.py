from __future__ import annotations

from collections.abc import Callable

import numpy as np


def rk4_rollout(
    derivative: Callable[[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
    delta0: np.ndarray,
    omega0: np.ndarray,
    time_grid: np.ndarray,
    pm: np.ndarray,
    static: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_grid, dtype=np.float64)
    pm = np.asarray(pm, dtype=np.float64)
    if len(t) != len(pm) or len(t) < 1:
        raise ValueError("time and Pm schedules must have the same positive length")
    delta = np.empty((len(t), delta0.size), dtype=np.float64)
    omega = np.empty_like(delta)
    delta[0] = np.asarray(delta0).reshape(-1)
    omega[0] = np.asarray(omega0).reshape(-1)
    for index in range(len(t) - 1):
        dt = float(t[index + 1] - t[index])
        if dt <= 0:
            raise ValueError("rollout time must be strictly increasing")
        p0 = pm[index]
        p1 = pm[index + 1]
        ph = 0.5 * (p0 + p1)
        d0, w0 = delta[index], omega[index]
        k1d, k1w = derivative(d0, w0, p0, static)
        k2d, k2w = derivative(d0 + 0.5 * dt * k1d, w0 + 0.5 * dt * k1w, ph, static)
        k3d, k3w = derivative(d0 + 0.5 * dt * k2d, w0 + 0.5 * dt * k2w, ph, static)
        k4d, k4w = derivative(d0 + dt * k3d, w0 + dt * k3w, p1, static)
        delta[index + 1] = d0 + dt * (k1d + 2 * k2d + 2 * k3d + k4d) / 6
        omega[index + 1] = w0 + dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6
        if not np.isfinite(delta[index + 1]).all() or not np.isfinite(omega[index + 1]).all():
            raise FloatingPointError(f"nonfinite rollout at step {index + 1}")
    return delta, omega
