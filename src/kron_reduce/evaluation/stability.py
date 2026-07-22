from __future__ import annotations

import numpy as np


def classify_predicted_trajectory(
    delta: np.ndarray,
    omega: np.ndarray,
    *,
    angle_range_threshold: float = np.pi,
    speed_threshold_rad_s: float = 0.05 * 376.99111843077515,
) -> str:
    delta = np.asarray(delta); omega = np.asarray(omega)
    angle_range = np.max(delta, axis=1) - np.min(delta, axis=1)
    if np.any(angle_range >= angle_range_threshold) or np.any(np.abs(omega) >= speed_threshold_rad_s):
        return "unstable"
    return "stable"


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float | int]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("classification requires a nonempty aligned population")
    tp = sum(a == "unstable" and b == "unstable" for a, b in zip(y_true, y_pred))
    fn = sum(a == "unstable" and b == "stable" for a, b in zip(y_true, y_pred))
    fp = sum(a == "stable" and b == "unstable" for a, b in zip(y_true, y_pred))
    tn = sum(a == "stable" and b == "stable" for a, b in zip(y_true, y_pred))
    fsr = fn / (tp + fn) if tp + fn else float("nan")
    fur = fp / (tn + fp) if tn + fp else float("nan")
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "FSR": fsr, "FUR": fur,
        "Accuracy": (tp + tn) / len(y_true),
        "BalancedAccuracy": float(np.nanmean([tpr, tnr])),
    }
