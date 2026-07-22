from __future__ import annotations

import numpy as np

from .constants import OMEGA_S


def effective_mass(inertia_prior: np.ndarray, omega_s: float = OMEGA_S) -> np.ndarray:
    """Return M/omega_s for rad/s COI speed coordinates."""
    inertia_prior = np.asarray(inertia_prior, dtype=np.float64)
    if np.any(inertia_prior <= 0) or omega_s <= 0:
        raise ValueError("positive inertia and synchronous speed are required")
    return inertia_prior / omega_s
