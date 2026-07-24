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
    delta_error = (pd - td).reshape(-1)
    omega_error = (pw - tw).reshape(-1)
    delta_truth = td.reshape(-1)
    omega_truth = tw.reshape(-1)
    delta_sse = float(delta_error @ delta_error)
    omega_sse = float(omega_error @ omega_error)
    delta_centered = delta_truth - np.mean(delta_truth)
    omega_centered = omega_truth - np.mean(omega_truth)
    delta_sst = float(delta_centered @ delta_centered)
    omega_sst = float(omega_centered @ omega_centered)
    return {
        "n_scored_points": int(len(td)),
        "n_scored_values": int(error.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "delta_rmse": float(np.sqrt(np.mean((pd - td) ** 2))),
        "omega_rmse": float(np.sqrt(np.mean((pw - tw) ** 2))),
        "delta_r2": float(1.0 - delta_sse / delta_sst) if delta_sst > 0 else float("nan"),
        "omega_r2": float(1.0 - omega_sse / omega_sst) if omega_sst > 0 else float("nan"),
        "delta_sse": delta_sse,
        "delta_sst": delta_sst,
        "omega_sse": omega_sse,
        "omega_sst": omega_sst,
        "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "sse": sse,
        "sst": sst,
    }


def aggregate_trajectory_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    valid = [row for row in rows if np.isfinite(row.get("rmse", np.nan))]
    if not valid:
        return {"mean_rmse": float("nan"), "median_rmse": float("nan"), "pooled_rmse": float("nan"), "worst_10pct_cvar_rmse": float("nan"), "pooled_r2": float("nan"), "pooled_delta_r2": float("nan"), "pooled_omega_r2": float("nan"), "failure_rate": 1.0}
    sse = sum(row["sse"] for row in valid); sst = sum(row["sst"] for row in valid)
    rmses = np.sort(np.asarray([row["rmse"] for row in valid], dtype=np.float64))
    tail_count = max(1, int(np.ceil(0.1 * len(rmses))))
    scored_values = sum(
        int(row.get("n_scored_values", round(row["sse"] / row["rmse"]**2) if row["rmse"] > 0 else 0))
        for row in valid
    )
    delta_sse = sum(float(row.get("delta_sse", np.nan)) for row in valid)
    delta_sst = sum(float(row.get("delta_sst", np.nan)) for row in valid)
    omega_sse = sum(float(row.get("omega_sse", np.nan)) for row in valid)
    omega_sst = sum(float(row.get("omega_sst", np.nan)) for row in valid)
    return {
        "mean_rmse": float(np.mean(rmses)),
        "median_rmse": float(np.median(rmses)),
        "pooled_rmse": float(np.sqrt(sse / scored_values)) if scored_values > 0 else float("nan"),
        "worst_10pct_cvar_rmse": float(np.mean(rmses[-tail_count:])),
        "pooled_r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "pooled_delta_r2": float(1.0 - delta_sse / delta_sst) if delta_sst > 0 and np.isfinite(delta_sse) else float("nan"),
        "pooled_omega_r2": float(1.0 - omega_sse / omega_sst) if omega_sst > 0 and np.isfinite(omega_sse) else float("nan"),
        "failure_rate": float(1.0 - len(valid) / len(rows)),
    }
