"""Allowlisted measured states available before one forecast origin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalStateHistory:
    """Real measured states, with a relative clock ending at the forecast origin.

    The declared constant input must hold over the observation window. Crossing
    a topology/input change requires a separately registered observation model.
    """

    time: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    constant_power: np.ndarray

    def __post_init__(self) -> None:
        for name, ndim in (("time", 1), ("delta", 2), ("omega", 2), ("constant_power", 1)):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if array.ndim != ndim or not np.isfinite(array).all():
                raise ValueError(f"invalid {name}")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if (
            len(self.time) < 2
            or self.time[-1] != 0.0
            or np.any(self.time > 0.0)
            or np.any(np.diff(self.time) <= 0.0)
        ):
            raise ValueError(
                "history must increase strictly and end at zero without future samples"
            )
        shape = (len(self.time), len(self.constant_power))
        if self.delta.shape != shape or self.omega.shape != shape:
            raise ValueError("history state arrays must match the registered clock and nodes")
