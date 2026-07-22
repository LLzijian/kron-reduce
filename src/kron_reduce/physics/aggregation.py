from __future__ import annotations

import numpy as np


def _weighted(values: np.ndarray, mass: np.ndarray, members: tuple[int, ...]) -> np.ndarray:
    weights = mass[list(members)]
    return np.sum(values[..., list(members)] * weights, axis=-1) / np.sum(weights)


def aggregate_event_arrays(
    delta: np.ndarray,
    omega: np.ndarray,
    pm: np.ndarray,
    equilibrium_delta: np.ndarray,
    mass: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
) -> dict[str, np.ndarray]:
    return {
        "delta": np.stack([_weighted(delta, mass, group) for group in partition], axis=-1),
        "omega": np.stack([_weighted(omega, mass, group) for group in partition], axis=-1),
        "Pm": np.stack([np.sum(pm[..., list(group)], axis=-1) for group in partition], axis=-1),
        "equilibrium_delta": np.asarray([
            _weighted(equilibrium_delta, mass, group) for group in partition
        ]),
        "mass": np.asarray([np.sum(mass[list(group)]) for group in partition]),
    }
