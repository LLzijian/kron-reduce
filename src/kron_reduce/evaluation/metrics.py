from __future__ import annotations

import numpy as np


def trajectory_metrics(
    true_delta: np.ndarray,
    true_omega: np.ndarray,
    pred_delta: np.ndarray,
    pred_omega: np.ndarray,
    *,
    metric_start_index: int = 0,
) -> dict[str, float | int]:
    if metric_start_index < 0:
        raise ValueError("metric_start_index must be nonnegative")
    arrays = [np.asarray(x)[metric_start_index:] for x in (true_delta, true_omega, pred_delta, pred_omega)]
    if not arrays[0].size or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("trajectory metric arrays must be nonempty and shape-aligned")
    td, tw, pd, pw = arrays
    error = np.concatenate(((pd - td).reshape(-1), (pw - tw).reshape(-1)))
    truth = np.concatenate((td.reshape(-1), tw.reshape(-1)))
    sse = float(error @ error)
    centered = truth - np.mean(truth)
    sst = float(centered @ centered)
    return {
        "n_scored_points": int(len(td)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "delta_rmse": float(np.sqrt(np.mean((pd - td) ** 2))),
        "omega_rmse": float(np.sqrt(np.mean((pw - tw) ** 2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "sse": sse,
        "sst": sst,
    }


def aggregate_trajectory_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    valid = [row for row in rows if np.isfinite(row.get("rmse", np.nan))]
    if not valid:
        return {"mean_rmse": float("nan"), "median_rmse": float("nan"), "pooled_r2": float("nan"), "failure_rate": 1.0}
    sse = sum(row["sse"] for row in valid); sst = sum(row["sst"] for row in valid)
    return {
        "mean_rmse": float(np.mean([row["rmse"] for row in valid])),
        "median_rmse": float(np.median([row["rmse"] for row in valid])),
        "pooled_r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "failure_rate": float(1.0 - len(valid) / len(rows)),
    }
