from __future__ import annotations

import numpy as np


def coi_project(values: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Project angle/speed coordinates onto ``mass.T @ x == 0``.

    The operation is idempotent and works for both vectors and batched arrays.
    """
    values = np.asarray(values, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if values.shape[-1] != mass.size or np.any(mass <= 0):
        raise ValueError("COI projection requires shape-aligned positive mass")
    center = np.sum(values * mass, axis=-1, keepdims=True) / np.sum(mass)
    return values - center


def coi_project_power(values: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Remove net accelerating power without changing relative injections."""
    values = np.asarray(values, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if values.shape[-1] != mass.size or np.any(mass <= 0):
        raise ValueError("power projection requires shape-aligned positive mass")
    imbalance = np.sum(values, axis=-1, keepdims=True)
    return values - imbalance * mass / np.sum(mass)


def coi_basis(mass: np.ndarray) -> np.ndarray:
    """Euclidean orthonormal basis for the raw-coordinate COI quotient.

    Raw angle/speed coordinates obey ``mass.T @ x = 0``.  Using ``sqrt(mass)``
    here mixes this constraint with mass-normalised coordinates and leaves a
    component of the common-angle mode in certificate Hessians.
    """
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if np.any(mass <= 0):
        raise ValueError("positive mass is required")
    vector = mass / np.linalg.norm(mass)
    _, _, vh = np.linalg.svd(vector.reshape(1, -1), full_matrices=True)
    return vh[1:].T
