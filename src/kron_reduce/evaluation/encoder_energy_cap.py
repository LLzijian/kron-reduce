"""Pure-array encoder-norm energy cap and single-path trajectory metrics."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .energy_constrained_initializer import solve_energy_ball
from .history_metrics import weighted_cvar


def _finite_array(value, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    source = np.asarray(value)
    if np.iscomplexobj(source):
        suffix = f" with shape {shape}" if shape is not None else ""
        raise ValueError(f"{name} must be finite{suffix}")
    array = np.asarray(source, dtype=np.float64)
    if ((shape is not None and array.shape != shape) or not np.isfinite(array).all()):
        suffix = f" with shape {shape}" if shape is not None else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return array


def encoder_energy_cap(integral: Mapping[str, object], encoder_z_h) -> dict[str, object]:
    """Solve the saved integral system inside the encoder hidden-state norm ball.

    The caller supplies an A14 integral record with ``Ltilde``, ``btilde4``,
    ``svd`` and ``zh4``. The supplied factorization is passed through unchanged;
    no decomposition or trajectory operation is performed here.
    """
    if not isinstance(integral, Mapping):
        raise TypeError("integral must be a mapping")
    required = {"Ltilde", "btilde4", "svd", "zh4"}
    if not required.issubset(integral):
        raise ValueError("integral record is missing saved A14 fields")

    matrix = _finite_array(integral["Ltilde"], "Ltilde")
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    right = _finite_array(integral["btilde4"], "btilde4", (matrix.shape[0],))
    encoder = _finite_array(encoder_z_h, "encoder_z_h", (matrix.shape[1],))
    saved_svd = integral["svd"]
    if not isinstance(saved_svd, Mapping):
        raise TypeError("saved svd must be a mapping")
    svd_fields = {"U", "s", "Vh", "rank", "tol"}
    if not svd_fields.issubset(saved_svd):
        raise ValueError("saved A14 SVD is missing U, s, Vh, rank or tol")
    svd = {
        "u": saved_svd["U"], "s": saved_svd["s"], "vh": saved_svd["Vh"],
        "rank": saved_svd["rank"], "threshold": saved_svd["tol"],
    }
    encoder_norm = math.hypot(*encoder)
    rank = saved_svd["rank"]
    if type(rank) is not int or not 0 <= rank <= matrix.shape[1]:
        raise ValueError("saved SVD rank is invalid")
    saved_zh4 = integral["zh4"]
    if saved_zh4 is None:
        if encoder_norm != 0.0 and rank == matrix.shape[1]:
            raise ValueError("full-rank A14 record needs its saved zh4")
        unconstrained = None
    else:
        unconstrained = _finite_array(saved_zh4, "zh4", (matrix.shape[1],))

    radius = encoder_norm
    try:
        with np.errstate(over="raise", invalid="raise"):
            encoder_residual = matrix @ encoder - right
            encoder_objective = 0.5 * float(encoder_residual @ encoder_residual)
    except FloatingPointError as error:
        raise ValueError("encoder residual is not representable") from error
    if not np.isfinite(encoder_residual).all() or not math.isfinite(encoder_objective):
        raise ValueError("encoder residual objective must be finite")

    solution = solve_energy_ball(
        matrix, right, svd, radius, unconstrained=unconstrained,
    )
    candidate_residual = None
    candidate_objective = None
    candidate_not_worse = None
    if solution["z"] is not None:
        try:
            with np.errstate(over="raise", invalid="raise"):
                candidate_residual = matrix @ solution["z"] - right
                candidate_objective = 0.5 * float(candidate_residual @ candidate_residual)
        except FloatingPointError as error:
            raise ValueError("candidate residual is not representable") from error
        if (not np.isfinite(candidate_residual).all()
                or not math.isfinite(candidate_objective)):
            raise ValueError("candidate residual objective must be finite")
        candidate_not_worse = candidate_objective <= encoder_objective

    return {
        "radius": radius,
        "encoder_z_h": encoder.copy(),
        "encoder_residual": encoder_residual,
        "encoder_residual_norm": math.hypot(*encoder_residual),
        "encoder_objective": encoder_objective,
        "candidate": solution,
        "candidate_residual": candidate_residual,
        "candidate_residual_norm": (None if candidate_residual is None
                                    else math.hypot(*candidate_residual)),
        "candidate_objective": candidate_objective,
        "candidate_objective_not_worse_than_encoder": candidate_not_worse,
    }


def _horizon_metrics(error: np.ndarray, scale: np.ndarray, complete: bool) -> dict[str, object]:
    normalized = error / scale[None, :]
    q_normalized, v_normalized = normalized[:, :10], normalized[:, 10:]
    q_physical, v_physical = error[:, :10], error[:, 10:]
    q_mse = float(np.mean(np.square(q_normalized)))
    v_mse = float(np.mean(np.square(v_normalized)))
    all_mse = 0.5 * (q_mse + v_mse)
    return {
        "point_count": int(error.shape[0]),
        "complete": complete,
        "normalized_mse": {"all": all_mse, "q": q_mse, "v": v_mse},
        "normalized_rmse": {
            "all": math.sqrt(all_mse), "q": math.sqrt(q_mse), "v": math.sqrt(v_mse),
        },
        "physical_rmse": {
            "q": math.sqrt(float(np.mean(np.square(q_physical)))),
            "v": math.sqrt(float(np.mean(np.square(v_physical)))),
        },
        "maximum_absolute_error": {
            "normalized": {
                "all": float(np.max(np.abs(normalized))),
                "q": float(np.max(np.abs(q_normalized))),
                "v": float(np.max(np.abs(v_normalized))),
            },
            "physical": {
                "q": float(np.max(np.abs(q_physical))),
                "v": float(np.max(np.abs(v_physical))),
            },
        },
    }


def trajectory_horizon_metrics(clock, truth, prediction, target_scale) -> dict[str, object]:
    """Measure one saved path on available future points at 1 s and 3 s.

    The initial point is excluded. Partial horizons use all points that actually
    exist within the horizon and are never padded or discarded.
    """
    time = _finite_array(clock, "clock")
    truth_array = _finite_array(truth, "truth")
    prediction_array = _finite_array(prediction, "prediction")
    if time.ndim != 1 or len(time) == 0:
        raise ValueError("clock must be a nonempty vector")
    if truth_array.shape != (len(time), 20) or prediction_array.shape != (len(time), 22):
        raise ValueError("truth must have shape (len(clock), 20) and prediction (len(clock), 22)")
    if len(time) > 1 and np.any(np.diff(time) <= 0):
        raise ValueError("clock must be strictly increasing")
    if abs(float(time[0])) > 1e-12:
        raise ValueError("clock origin must be zero")
    if len(time) > 1 and not np.allclose(np.diff(time), 0.01, rtol=0.0, atol=1e-12):
        raise ValueError("clock must use fixed 10 ms spacing")
    scale = _finite_array(target_scale, "target_scale", (20,))
    if np.any(scale <= 0):
        raise ValueError("target_scale must be positive")

    elapsed = time
    error = prediction_array[:, :20] - truth_array
    horizons = {}
    for horizon in (1, 3):
        selected = (elapsed > 0) & (elapsed <= horizon + 1e-12)
        points = error[selected]
        complete = bool(elapsed[-1] + 1e-12 >= horizon)
        if points.shape[0] == 0:
            horizons[str(horizon)] = {
                "point_count": 0, "complete": complete,
                "normalized_mse": {"all": None, "q": None, "v": None},
                "normalized_rmse": {"all": None, "q": None, "v": None},
                "physical_rmse": {"q": None, "v": None},
                "maximum_absolute_error": {
                    "normalized": {"all": None, "q": None, "v": None},
                    "physical": {"q": None, "v": None},
                },
            }
        else:
            horizons[str(horizon)] = _horizon_metrics(points, scale, complete)
    return {
        "initial_point_excluded": True,
        "available_future_point_count": int(np.count_nonzero(elapsed > 0)),
        "complete_3s": horizons["3"]["complete"],
        "horizons": horizons,
    }


__all__ = ["encoder_energy_cap", "trajectory_horizon_metrics", "weighted_cvar"]
