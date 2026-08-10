from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation import (
    dominant_frequency,
    exact_dmd_modes,
    match_dmd_modes,
    spectral_overlap,
    spectral_signature,
)
from kron_reduce.physics import (
    aggregate_state,
    cluster_aggregation_matrices,
    project_cluster_consensus,
    reconstruct_cluster_state,
    validate_partition,
)
from kron_reduce.workflows.physical_cluster_audit import (
    DENSE_SOURCE,
    HIGHORDER_SOURCE,
    _discover_train_partition,
)


def test_cluster_projection_is_mass_orthogonal_and_energy_decomposes() -> None:
    mass = np.asarray([2.0, 3.0, 5.0, 7.0])
    partition = ((0, 2), (1, 3))
    aggregate, reconstruct, projection = cluster_aggregation_matrices(mass, partition)
    mass_matrix = np.diag(mass)
    values = np.asarray([[0.4, -0.3, 0.8, 0.2], [-0.1, 0.6, 0.5, -0.9]])
    low = project_cluster_consensus(values, mass, partition)
    high = values - low

    assert np.allclose(aggregate @ reconstruct, np.eye(2), atol=1e-14)
    assert np.allclose(projection @ projection, projection, atol=1e-14)
    assert np.allclose(projection.T @ mass_matrix, mass_matrix @ projection, atol=1e-14)
    assert np.allclose(aggregate_state(values, mass, partition) @ reconstruct.T, low)
    assert np.allclose(
        np.einsum("ti,ij,tj->t", values, mass_matrix, values),
        np.einsum("ti,ij,tj->t", low, mass_matrix, low)
        + np.einsum("ti,ij,tj->t", high, mass_matrix, high),
        atol=1e-14,
    )
    assert np.allclose(
        reconstruct_cluster_state(aggregate_state(values, mass, partition), mass, partition),
        low,
    )


@pytest.mark.parametrize(
    "partition",
    [((0,), (0, 1)), ((0,),), ((0, 2), (1,)), ((), (0, 1))],
)
def test_invalid_partition_is_rejected(partition: tuple[tuple[int, ...], ...]) -> None:
    with pytest.raises(ValueError):
        validate_partition(partition, 2)


def test_spectral_and_dmd_metrics_recover_a_damped_mode() -> None:
    dt = 0.01
    time = np.arange(0.0, 10.0, dt)
    frequency_hz = 0.7
    decay_rate = -0.1
    phase = 2.0 * np.pi * frequency_hz * time
    response = np.exp(decay_rate * time)[:, None] * np.column_stack([np.cos(phase), np.sin(phase)])
    perturbed = response @ np.asarray([[1.0, 0.1], [-0.1, 1.0]])

    frequency, power = spectral_signature(response, dt, max_frequency_hz=2.0)
    _, perturbed_power = spectral_signature(perturbed, dt, max_frequency_hz=2.0)
    modes = exact_dmd_modes(response, dt, rank=2, max_frequency_hz=2.0)
    matches = match_dmd_modes(modes, exact_dmd_modes(perturbed, dt, rank=2, max_frequency_hz=2.0))

    assert dominant_frequency(frequency, power) == pytest.approx(frequency_hz, abs=0.11)
    assert spectral_overlap(power, perturbed_power) > 0.99
    assert modes[0].frequency_hz == pytest.approx(frequency_hz, rel=1e-3)
    expected_damping = -decay_rate / np.hypot(decay_rate, 2.0 * np.pi * frequency_hz)
    assert modes[0].damping_ratio == pytest.approx(expected_damping, rel=1e-3)
    assert matches[0]["relative_frequency_error"] < 1e-3
    assert matches[0]["mac"] > 0.95


def test_train_only_partition_discovery_recovers_three_exact_coherent_pairs() -> None:
    time = np.arange(0.0, 1.01, 0.01)
    basis = np.column_stack(
        [
            np.sin(2.0 * np.pi * frequency * time + phase)
            for frequency, phase in zip(
                (0.31, 0.47, 0.73, 0.91, 1.13, 1.37, 1.61),
                (0.0, 0.2, 0.4, 0.7, 1.1, 1.5, 1.9),
                strict=True,
            )
        ]
    )
    omega = np.column_stack(
        [
            basis[:, 0],
            basis[:, 0],
            basis[:, 1],
            basis[:, 1],
            basis[:, 2],
            basis[:, 2],
            *basis[:, 3:].T,
        ]
    )
    delta = np.cumsum(omega, axis=0) * 0.01
    samples = []
    for source in (HIGHORDER_SOURCE, DENSE_SOURCE):
        samples.append(
            {
                "training_context": {
                    "targets": {"t": time, "delta": delta, "omega": omega},
                    "metadata": {"source_dataset": source},
                }
            }
        )

    partition, score_rows = _discover_train_partition(samples, maximum_time=1.0)

    assert {(0, 1), (2, 3), (4, 5)}.issubset(set(partition))
    assert {node for group in partition for node in group} == set(range(10))
    assert sum(bool(row["selected"]) for row in score_rows) == 3
