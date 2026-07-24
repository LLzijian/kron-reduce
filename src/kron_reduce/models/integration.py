from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_project


NumpyDerivative = Callable[
    [np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]],
    tuple[np.ndarray, np.ndarray],
]


def _interpolation_mode(pm: np.ndarray, requested: str) -> str:
    if requested not in {"auto", "zoh", "linear"}:
        raise ValueError("Pm interpolation must be auto, zoh, or linear")
    if requested != "auto":
        return requested
    return "zoh" if np.allclose(pm, pm[[0]], rtol=0.0, atol=1e-14) else "linear"


def _pm_at(p0: np.ndarray, p1: np.ndarray, fraction: float, mode: str) -> np.ndarray:
    return p0 if mode == "zoh" else p0 + fraction * (p1 - p0)


def rk4_rollout(
    derivative: NumpyDerivative,
    delta0: np.ndarray,
    omega0: np.ndarray,
    time_grid: np.ndarray,
    pm: np.ndarray,
    static: dict[str, np.ndarray],
    *,
    interpolation: str = "auto",
    max_step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """COI-preserving RK4 used by every NumPy dynamics model.

    Each output interval is internally subdivided.  ZOH inputs remain exactly
    constant throughout the interval; observed schedules are linearly sampled
    at RK stages.  Acceleration is projected at every stage and the accepted
    state is recentered to prevent accumulated common-mode round-off.
    """
    t = np.asarray(time_grid, dtype=np.float64)
    pm = np.asarray(pm, dtype=np.float64)
    if len(t) != len(pm) or len(t) < 1:
        raise ValueError("time and Pm schedules must have the same positive length")
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    mode = _interpolation_mode(pm, interpolation)
    mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
    delta = np.empty((len(t), mass.size), dtype=np.float64)
    omega = np.empty_like(delta)
    delta[0] = coi_project(np.asarray(delta0).reshape(-1), mass)
    omega[0] = coi_project(np.asarray(omega0).reshape(-1), mass)

    def field(d: np.ndarray, w: np.ndarray, p: np.ndarray):
        dd, dw = derivative(d, w, p, static)
        return coi_project(np.asarray(dd, dtype=np.float64), mass), coi_project(
            np.asarray(dw, dtype=np.float64), mass
        )

    for index in range(len(t) - 1):
        interval = float(t[index + 1] - t[index])
        if interval <= 0:
            raise ValueError("rollout time must be strictly increasing")
        substeps = max(1, int(np.ceil(interval / max_step)))
        h = interval / substeps
        d, w = delta[index].copy(), omega[index].copy()
        for substep in range(substeps):
            q0 = substep / substeps
            qh = (substep + 0.5) / substeps
            q1 = (substep + 1.0) / substeps
            p0 = _pm_at(pm[index], pm[index + 1], q0, mode)
            ph = _pm_at(pm[index], pm[index + 1], qh, mode)
            p1 = _pm_at(pm[index], pm[index + 1], q1, mode)
            k1d, k1w = field(d, w, p0)
            k2d, k2w = field(d + 0.5 * h * k1d, w + 0.5 * h * k1w, ph)
            k3d, k3w = field(d + 0.5 * h * k2d, w + 0.5 * h * k2w, ph)
            k4d, k4w = field(d + h * k3d, w + h * k3w, p1)
            d = coi_project(d + h * (k1d + 2 * k2d + 2 * k3d + k4d) / 6, mass)
            w = coi_project(w + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6, mass)
            if not np.isfinite(d).all() or not np.isfinite(w).all():
                raise FloatingPointError(f"nonfinite rollout at step {index + 1}")
        delta[index + 1], omega[index + 1] = d, w
    return delta, omega


def integration_convergence_audit(
    derivative: NumpyDerivative,
    delta0: np.ndarray,
    omega0: np.ndarray,
    time_grid: np.ndarray,
    pm: np.ndarray,
    static: dict[str, np.ndarray],
    *,
    interpolation: str = "auto",
    base_step: float = 0.01,
) -> dict[str, float | bool]:
    """Compare dt, dt/2, and dt/4 integrations without reading truth data."""
    trajectories = [
        rk4_rollout(
            derivative, delta0, omega0, time_grid, pm, static,
            interpolation=interpolation, max_step=base_step / divisor,
        )
        for divisor in (1, 2, 4)
    ]
    errors = []
    for coarse, fine in zip(trajectories[:-1], trajectories[1:]):
        error = np.sqrt(
            np.mean((coarse[0] - fine[0]) ** 2) + np.mean((coarse[1] - fine[1]) ** 2)
        )
        errors.append(float(error))
    ratio = errors[0] / max(errors[1], np.finfo(float).eps)
    return {
        "dt_vs_dt2": errors[0],
        "dt2_vs_dt4": errors[1],
        "error_ratio": float(ratio),
        "convergent": bool(errors[1] <= errors[0] + 1e-12),
    }


def torch_rk4_rollout(
    derivative: Callable[[Any, Any, Any, dict[str, Any]], tuple[Any, Any]],
    delta0: Any,
    omega0: Any,
    time_grid: Any,
    pm: Any,
    static: dict[str, Any],
    *,
    interpolation: str = "zoh",
    max_step: float = 0.01,
):
    """Differentiable counterpart of :func:`rk4_rollout`.

    Imported lazily so linear-only users do not pay a Torch import cost.
    """
    import torch

    mass = static["mass"].reshape(-1)

    def project(x):
        return x - torch.sum(x * mass) / torch.sum(mass)

    d, w = project(delta0.reshape(-1)), project(omega0.reshape(-1))
    output_d, output_w = [d], [w]
    for index in range(len(time_grid) - 1):
        interval = float((time_grid[index + 1] - time_grid[index]).detach().cpu())
        if interval <= 0:
            raise ValueError("rollout time must be strictly increasing")
        substeps = max(1, int(np.ceil(interval / max_step)))
        h = interval / substeps
        for substep in range(substeps):
            q0, qh, q1 = substep / substeps, (substep + 0.5) / substeps, (substep + 1) / substeps

            def p(frac):
                return pm[index] if interpolation == "zoh" else pm[index] + frac * (pm[index + 1] - pm[index])

            def f(di, wi, pi):
                dd, dw = derivative(di, wi, pi, static)
                return project(dd), project(dw)

            k1d, k1w = f(d, w, p(q0))
            k2d, k2w = f(d + 0.5 * h * k1d, w + 0.5 * h * k1w, p(qh))
            k3d, k3w = f(d + 0.5 * h * k2d, w + 0.5 * h * k2w, p(qh))
            k4d, k4w = f(d + h * k3d, w + h * k3w, p(q1))
            d = project(d + h * (k1d + 2 * k2d + 2 * k3d + k4d) / 6)
            w = project(w + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6)
        output_d.append(d)
        output_w.append(w)
    return torch.stack(output_d), torch.stack(output_w)
