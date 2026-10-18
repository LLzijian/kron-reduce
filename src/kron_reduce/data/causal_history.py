"""Past-only history reads, after the caller has authorized the event role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.physics.coordinates import coi_project


@dataclass(frozen=True)
class MeasuredHistoryWindow:
    history: CausalStateHistory
    forecast_origin_after_clearing_s: float
    source_sample_count: int
    access_log: tuple[dict[str, Any], ...]


def read_measured_history(
    event_group: Any, *, sample_count: int, mass: np.ndarray, constant_power: np.ndarray
) -> MeasuredHistoryWindow:
    """Read exactly a registered count of real samples, without interpolation.

    Event identity/role authorization belongs to the experiment's allowlist
    reader. This helper never enumerates event metadata or other channels.
    ``constant_power`` is the declared balanced power deviation, not a future
    observed input schedule. The forecast origin is the last real timestamp.
    """
    if type(sample_count) is not int or sample_count not in {11, 21, 51}:
        raise ValueError("history needs a registered real-sample count: 11, 21, or 51")
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 1 or len(mass) < 2 or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("history needs positive finite mass")
    access = []

    def read(channel: str) -> np.ndarray:
        access.append(
            {"dataset": f"postclear/{channel}", "start": 0, "stop_exclusive": sample_count}
        )
        return np.asarray(event_group[f"postclear/{channel}"][:sample_count], dtype=np.float64)

    clock = read("t")
    if (
        clock.shape != (sample_count,)
        or not np.isfinite(clock).all()
        or clock[0] != 0.0
        or np.any(np.diff(clock) <= 0.0)
        or np.max(np.diff(clock)) > 0.0100000001
    ):
        raise ValueError("history clock must cover the real registered samples from clearing")
    delta, omega = read("delta"), read("omega")
    if delta.shape != (sample_count, len(mass)) or omega.shape != delta.shape:
        raise ValueError("history channels must cover the registered real window")
    origin = float(clock[-1])
    history = CausalStateHistory(
        clock - origin, coi_project(delta, mass), coi_project(omega, mass), constant_power
    )
    return MeasuredHistoryWindow(history, origin, sample_count, tuple(access))
