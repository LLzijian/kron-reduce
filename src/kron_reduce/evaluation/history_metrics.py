"""Pure-array history metrics for the frozen A14 synthetic/integration tests.

This module only evaluates caller-supplied arrays. It performs no file access,
model construction, trajectory integration, or population selection.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np


def _finite_array(value, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite" + (f" with shape {shape}" if shape else ""))
    return array


def event_losses(paths, observed, target_scale) -> dict[str, np.ndarray]:
    """Return eventwise normalized MSE over the ten noninitial history points."""
    prediction = np.asarray(paths, dtype=np.float64)
    truth = np.asarray(observed, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[1:] != (11, 22):
        raise ValueError("paths must have shape (N, 11, 22)")
    if truth.shape != (prediction.shape[0], 11, 20):
        raise ValueError("observed must have shape (N, 11, 20) aligned with paths")
    scale = _finite_array(target_scale, "target_scale", (20,))
    if np.any(scale <= 0):
        raise ValueError("target_scale must be positive")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("paths and observed must be finite")
    error = (prediction[:, 1:, :20] - truth[:, 1:, :]) / scale[None, None, :]
    q_sq = np.square(error[:, :, :10])
    v_sq = np.square(error[:, :, 10:])
    q_mse = np.mean(q_sq, axis=(1, 2))
    v_mse = np.mean(v_sq, axis=(1, 2))
    all_mse = 0.5 * (q_mse + v_mse)
    return {"all": all_mse, "q": q_mse, "v": v_mse}


def weighted_cvar(losses, weights, tail_mass: float = 0.1) -> float:
    """Weighted worst-tail mean with stable ties and fractional boundary mass."""
    value = _finite_array(losses, "losses")
    weight = _finite_array(weights, "weights", value.shape)
    if value.ndim != 1 or len(value) == 0:
        raise ValueError("losses must be a nonempty vector")
    if np.any(value < 0) or np.any(weight <= 0):
        raise ValueError("losses must be nonnegative and weights strictly positive")
    if not np.isfinite(tail_mass) or not 0 < tail_mass <= 1:
        raise ValueError("tail_mass must be finite in (0, 1]")
    if not np.isclose(float(np.sum(weight)), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("weights must sum to one; subsets are never renormalized")
    remaining = float(tail_mass)
    total = 0.0
    for index in np.argsort(-value, kind="stable"):
        take = min(float(weight[index]), remaining)
        total += take * float(value[index])
        remaining -= take
        if remaining <= 8 * np.finfo(np.float64).eps * tail_mass:
            break
    if remaining > 8 * np.finfo(np.float64).eps * tail_mass:
        raise ValueError("weights do not cover the requested tail mass")
    return total / tail_mass


def hierarchy_weights(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Compute the frozen source→physical-line→event equal-mass weights.

    Output order exactly follows ``rows``; a physical line occurring in two
    source datasets remains a separate source-local line group.
    """
    if not rows:
        raise ValueError("at least one row is required")
    keys: list[tuple[str, str]] = []
    sources: set[str] = set()
    line_sets: dict[str, set[str]] = {}
    events: Counter[tuple[str, str]] = Counter()
    for row in rows:
        source = row.get("source_dataset")
        line = row.get("fault_line")
        if not isinstance(source, str) or not source or not isinstance(line, str) or not line:
            raise ValueError("each row needs nonempty source_dataset and fault_line strings")
        key = (source, line)
        keys.append(key)
        sources.add(source)
        line_sets.setdefault(source, set()).add(line)
        events[key] += 1
    n_source = len(sources)
    weights = np.array([
        1.0 / n_source / len(line_sets[key[0]]) / events[key]
        for key in keys
    ], dtype=np.float64)
    if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ArithmeticError("hierarchical weights failed to sum to one")
    return weights


def comparison(candidate, baseline, observed, target_scale, weights) -> dict[str, object]:
    """Compare complete aligned populations without dropping failed events."""
    candidate_loss = event_losses(candidate, observed, target_scale)
    baseline_loss = event_losses(baseline, observed, target_scale)
    weight = _finite_array(weights, "weights", candidate_loss["all"].shape)
    if np.any(weight <= 0) or not np.isclose(float(weight.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("weights must be positive and sum to one")
    weighted = {
        name: {
            "candidate": float(weight @ candidate_loss[name]),
            "baseline": float(weight @ baseline_loss[name]),
        }
        for name in ("all", "q", "v")
    }
    cvar = {
        "candidate": weighted_cvar(candidate_loss["all"], weight, 0.1),
        "baseline": weighted_cvar(baseline_loss["all"], weight, 0.1),
    }
    conditions = {
        "all_strictly_lower": weighted["all"]["candidate"] < weighted["all"]["baseline"],
        "q_not_higher": weighted["q"]["candidate"] <= weighted["q"]["baseline"],
        "v_not_higher": weighted["v"]["candidate"] <= weighted["v"]["baseline"],
        "worst10_cvar_not_higher": cvar["candidate"] <= cvar["baseline"],
    }
    return {
        "candidate_event_losses": candidate_loss,
        "baseline_event_losses": baseline_loss,
        "weighted_mse": weighted,
        "worst10_cvar": cvar,
        "conditions": conditions,
        "I_pass": all(conditions.values()),
    }
