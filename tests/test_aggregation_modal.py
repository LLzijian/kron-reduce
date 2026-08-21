from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation import (
    DmdMode,
    cross_cluster_modal_consensus,
    dominant_frequency,
    exact_dmd_modes,
    input_aware_delay_modes,
    input_aware_modal_consensus,
    match_dmd_modes,
    pooled_input_aware_modal_consensus,
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


def test_input_aware_delay_modes_separates_forcing_from_autonomous_pole() -> None:
    dt = 0.01
    time = np.arange(0.0, 20.0, dt)
    frequency_hz = 0.8
    damping_ratio = 0.06
    natural = 2.0 * np.pi * frequency_hz
    state = np.zeros((len(time), 2))
    forcing = (
        0.8 * np.sin(2.0 * np.pi * 0.23 * time)
        + 0.35 * np.sin(2.0 * np.pi * 1.57 * time + 0.4)
    )
    state[0] = (0.25, -0.1)
    matrix = np.asarray(
        [[0.0, 1.0], [-natural**2, -2.0 * damping_ratio * natural]]
    )
    input_vector = np.asarray([0.0, 1.0])
    for index in range(len(time) - 1):
        x = state[index]
        u0 = forcing[index]
        u1 = forcing[index + 1]

        def field(value: np.ndarray, input_value: float) -> np.ndarray:
            return matrix @ value + input_vector * input_value

        k1 = field(x, u0)
        k2 = field(x + 0.5 * dt * k1, 0.5 * (u0 + u1))
        k3 = field(x + 0.5 * dt * k2, 0.5 * (u0 + u1))
        k4 = field(x + dt * k3, u1)
        state[index + 1] = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    modes = input_aware_delay_modes(
        state,
        forcing,
        dt,
        delays=2,
        ridge=1e-8,
        max_frequency_hz=2.0,
    )
    recovered = min(modes, key=lambda mode: abs(mode.frequency_hz - frequency_hz))
    expected_damped_frequency = frequency_hz * np.sqrt(1.0 - damping_ratio**2)
    assert recovered.frequency_hz == pytest.approx(expected_damped_frequency, rel=0.02)
    assert recovered.damping_ratio == pytest.approx(damping_ratio, abs=0.015)


def test_input_aware_modal_consensus_rejects_unstable_numerical_poles() -> None:
    dt = 0.01
    time = np.arange(0.0, 20.0, dt)
    frequency_hz = 0.8
    damping_ratio = 0.06
    natural = 2.0*np.pi*frequency_hz
    state = np.zeros((len(time), 2))
    forcing = np.sin(2.0*np.pi*0.23*time)+0.4*np.sin(2.0*np.pi*1.57*time)
    state[0] = (0.2, -0.08)
    matrix = np.asarray([[0.0, 1.0], [-natural**2, -2*damping_ratio*natural]])
    for index in range(len(time)-1):
        midpoint = 0.5*(forcing[index]+forcing[index+1])
        k1 = matrix@state[index]+np.asarray([0.0, forcing[index]])
        k2 = matrix@(state[index]+0.5*dt*k1)+np.asarray([0.0, midpoint])
        k3 = matrix@(state[index]+0.5*dt*k2)+np.asarray([0.0, midpoint])
        k4 = matrix@(state[index]+dt*k3)+np.asarray([0.0, forcing[index+1]])
        state[index+1] = state[index]+dt*(k1+2*k2+2*k3+k4)/6
    rng = np.random.default_rng(324)
    noisy = state+2e-4*rng.standard_normal(state.shape)
    consensus = input_aware_modal_consensus(
        noisy, forcing, dt,
        delay_orders=(2, 3, 4, 5),
        ridge_values=(1e-8, 1e-6, 1e-4),
        minimum_support_fraction=0.5,
        max_frequency_hz=2.0,
    )
    recovered = min(consensus, key=lambda item: abs(item.mode.frequency_hz-frequency_hz))
    assert recovered.support_fraction >= 0.5
    assert recovered.mode.frequency_hz == pytest.approx(
        frequency_hz*np.sqrt(1-damping_ratio**2), rel=0.03
    )
    assert recovered.mode.damping_ratio == pytest.approx(damping_ratio, abs=0.025)


def test_input_aware_modal_consensus_returns_empty_for_unexcited_constant_path() -> None:
    values = np.ones((200, 2))
    inputs = np.zeros((200, 1))
    consensus = input_aware_modal_consensus(
        values, inputs, 0.01,
        delay_orders=(2, 3), ridge_values=(1e-4, 1e-2),
    )
    assert consensus == []


def test_pooled_input_aware_consensus_recovers_shared_mode_across_events() -> None:
    dt = 0.01
    time = np.arange(0.0, 12.0, dt)
    frequency_hz, damping_ratio = 0.72, 0.045
    natural = 2*np.pi*frequency_hz
    matrix = np.asarray([[0.0, 1.0], [-natural**2, -2*damping_ratio*natural]])
    paths, inputs = [], []
    for event in range(4):
        forcing = np.sin(2*np.pi*(0.17+0.04*event)*time+0.3*event)
        state = np.zeros((len(time), 2))
        state[0] = (0.1+0.04*event, -0.03*event)
        for index in range(len(time)-1):
            midpoint = 0.5*(forcing[index]+forcing[index+1])
            k1 = matrix@state[index]+np.asarray([0.0, forcing[index]])
            k2 = matrix@(state[index]+0.5*dt*k1)+np.asarray([0.0, midpoint])
            k3 = matrix@(state[index]+0.5*dt*k2)+np.asarray([0.0, midpoint])
            k4 = matrix@(state[index]+dt*k3)+np.asarray([0.0, forcing[index+1]])
            state[index+1] = state[index]+dt*(k1+2*k2+2*k3+k4)/6
        paths.append(state)
        inputs.append(forcing)
    consensus = pooled_input_aware_modal_consensus(
        paths, inputs, dt,
        delay_orders=(2, 3, 4), ridge_values=(1e-8, 1e-6, 1e-4),
        minimum_support_fraction=2/3, max_frequency_hz=2.0,
    )
    recovered = min(consensus, key=lambda item: abs(item.mode.frequency_hz-frequency_hz))
    assert recovered.support_fraction >= 2/3
    assert recovered.mode.frequency_hz == pytest.approx(
        frequency_hz*np.sqrt(1-damping_ratio**2), rel=0.03
    )
    assert recovered.mode.damping_ratio == pytest.approx(damping_ratio, abs=0.02)


def _mode(frequency: float, damping: float, shape: np.ndarray, strength: float = 1.0) -> DmdMode:
    shape = np.asarray(shape, dtype=complex)
    shape = shape/np.linalg.norm(shape)
    magnitude = 2*np.pi*frequency/np.sqrt(max(1-damping**2, 1e-12))
    return DmdMode(
        frequency_hz=frequency, damping_ratio=damping,
        decay_rate=-damping*magnitude, strength=strength,
        shape_real=tuple(shape.real), shape_imag=tuple(shape.imag),
    )


def test_cross_cluster_modal_consensus_recovers_shared_shape_order_independently() -> None:
    rng = np.random.default_rng(324)
    base = np.asarray([1.0, -0.6, 0.25, -0.15], dtype=complex)
    modes = {}
    for index in range(10):
        phase = np.exp(1j*(0.17*index))
        shared = phase*base+0.015*(rng.standard_normal(4)+1j*rng.standard_normal(4))
        modes[f"line-{index}"] = [
            _mode(0.9*(1+0.012*rng.standard_normal()), 0.05+0.004*rng.standard_normal(), shared, 2.0),
            _mode(1.8+0.05*rng.standard_normal(), 0.15, rng.standard_normal(4)+1j*rng.standard_normal(4), 0.2),
        ]
    consensus = cross_cluster_modal_consensus(
        modes, frequency_tolerance=0.1, minimum_mac=0.8, minimum_support_fraction=0.8
    )
    reversed_consensus = cross_cluster_modal_consensus(
        dict(reversed(list(modes.items()))),
        frequency_tolerance=0.1, minimum_mac=0.8, minimum_support_fraction=0.8,
    )
    recovered = min(consensus, key=lambda item: abs(item.mode.frequency_hz-0.9))
    assert recovered.support_fraction == 1.0
    assert recovered.median_shape_mac > 0.99
    assert recovered.mode.frequency_hz == pytest.approx(0.9, rel=0.02)
    assert [item.as_dict() for item in consensus] == [item.as_dict() for item in reversed_consensus]


def test_cross_cluster_modal_consensus_rejects_heterogeneous_modes() -> None:
    modes = {
        f"line-{index}": [
            _mode(
                0.35+0.17*index,
                0.03+0.01*index,
                np.eye(8, dtype=complex)[index],
            )
        ]
        for index in range(8)
    }
    assert cross_cluster_modal_consensus(
        modes, frequency_tolerance=0.2, minimum_mac=0.5, minimum_support_fraction=0.8
    ) == []


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
