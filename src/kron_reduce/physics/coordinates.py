from __future__ import annotations

import numpy as np


def coi_project(values: np.ndarray, mass: np.ndarray) -> np.ndarray:
    center = np.sum(values * mass, axis=-1, keepdims=True) / np.sum(mass)
    return values - center


def coi_basis(mass: np.ndarray) -> np.ndarray:
    vector = np.sqrt(mass) / np.linalg.norm(np.sqrt(mass))
    _, _, vh = np.linalg.svd(vector.reshape(1, -1), full_matrices=True)
    return vh[1:].T
