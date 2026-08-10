from __future__ import annotations

import numpy as np


def validate_partition(
    partition: tuple[tuple[int, ...], ...],
    n_nodes: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate and canonicalize a disjoint, exhaustive node partition."""
    canonical = tuple(tuple(int(index) for index in group) for group in partition)
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if not canonical or any(not group for group in canonical):
        raise ValueError("partition must contain non-empty groups")
    flattened = tuple(index for group in canonical for index in group)
    if any(index < 0 or index >= n_nodes for index in flattened):
        raise ValueError("partition index is outside the node range")
    if len(set(flattened)) != len(flattened):
        raise ValueError("partition groups must be disjoint")
    if set(flattened) != set(range(n_nodes)):
        raise ValueError("partition must cover every node exactly once")
    return canonical


def cluster_aggregation_matrices(
    mass: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aggregation, reconstruction and mass-orthogonal projection matrices.

    For a column state ``x``, cluster coordinates are ``A @ x`` and the
    cluster-consensus reconstruction is ``R @ A @ x``.  With diagonal mass
    matrix ``M``, ``P = R @ A`` is idempotent and M-self-adjoint.
    """
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("mass must be a finite, strictly positive vector")
    canonical = validate_partition(partition, mass.size)
    aggregate = np.zeros((len(canonical), mass.size), dtype=float)
    reconstruct = np.zeros((mass.size, len(canonical)), dtype=float)
    for cluster, group in enumerate(canonical):
        members = np.asarray(group, dtype=int)
        cluster_mass = float(np.sum(mass[members]))
        aggregate[cluster, members] = mass[members] / cluster_mass
        reconstruct[members, cluster] = 1.0
    projection = reconstruct @ aggregate
    return aggregate, reconstruct, projection


def aggregate_state(
    values: np.ndarray,
    mass: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    """Mass-average node states along their final axis."""
    values = np.asarray(values, dtype=float)
    aggregate, _, _ = cluster_aggregation_matrices(mass, partition)
    if values.shape[-1] != aggregate.shape[1]:
        raise ValueError("state final axis does not match the mass vector")
    return values @ aggregate.T


def reconstruct_cluster_state(
    values: np.ndarray,
    mass: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    """Broadcast cluster coordinates back to their member nodes."""
    values = np.asarray(values, dtype=float)
    _, reconstruct, _ = cluster_aggregation_matrices(mass, partition)
    if values.shape[-1] != reconstruct.shape[1]:
        raise ValueError("state final axis does not match the partition")
    return values @ reconstruct.T


def project_cluster_consensus(
    values: np.ndarray,
    mass: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    """Project node states onto the cluster-consensus subspace."""
    values = np.asarray(values, dtype=float)
    _, _, projection = cluster_aggregation_matrices(mass, partition)
    if values.shape[-1] != projection.shape[0]:
        raise ValueError("state final axis does not match the mass vector")
    return values @ projection.T


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
    mass = np.asarray(mass, dtype=float)
    partition = validate_partition(partition, mass.size)
    return {
        "delta": aggregate_state(delta, mass, partition),
        "omega": aggregate_state(omega, mass, partition),
        "Pm": np.stack([np.sum(pm[..., list(group)], axis=-1) for group in partition], axis=-1),
        "equilibrium_delta": np.asarray([
            _weighted(equilibrium_delta, mass, group) for group in partition
        ]),
        "mass": np.asarray([np.sum(mass[list(group)]) for group in partition]),
    }
