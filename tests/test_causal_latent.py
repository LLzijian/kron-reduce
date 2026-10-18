from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from kron_reduce.models.causal_latent import (
    CausalStateHistory,
    PassiveLatentClosure,
    UnobservableHistoryError,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project


def _model() -> PassiveLatentClosure:
    laplacian = np.eye(10) - np.ones((10, 10)) / 10.0
    return PassiveLatentClosure(
        np.linspace(1.0, 2.0, 10),
        2.0 * laplacian,
        0.15 * laplacian,
        0.7 * coi_basis(np.ones(10))[:, :2],
        np.array([[0.0, -2.0], [2.0, 0.0]]),
        np.diag([0.8, 0.5]),
    )


def _history(model: PassiveLatentClosure) -> tuple[CausalStateHistory, np.ndarray]:
    state = np.concatenate(
        (coi_project(np.linspace(-0.03, 0.03, 10), model.mass), np.zeros(10), [0.4, -0.25])
    )
    time = np.linspace(0.0, 0.2, 21)
    truth = np.stack([expm(model.generator() * t) @ state for t in time])
    return CausalStateHistory(time - time[-1], truth[:, :10], truth[:, 10:20], np.zeros(10)), truth[
        -1
    ]


def test_identical_retained_state_has_distinct_future_for_distinct_hidden_state() -> None:
    model = _model()
    time = np.linspace(0.0, 1.0, 101)
    zero = np.zeros(10)
    positive = model.rollout(zero, zero, np.array([0.4, -0.25]), time, zero)
    negative = model.rollout(zero, zero, np.array([-0.4, 0.25]), time, zero)
    assert np.array_equal(positive[0, :20], negative[0, :20])
    assert np.linalg.norm(positive[1, 10:20] - negative[1, 10:20]) > 1e-3
    midpoint = (positive + negative) / 2.0
    arbitrary = np.full_like(midpoint[:, :20], 0.01)
    error = 0.5 * (
        np.mean((arbitrary - positive[:, :20]) ** 2) + np.mean((arbitrary - negative[:, :20]) ** 2)
    )
    lower_bound = 0.25 * np.mean((positive[:, :20] - negative[:, :20]) ** 2)
    assert error >= lower_bound
    assert lower_bound > 0.0


def test_history_recovers_hidden_initial_condition_without_future_oracle() -> None:
    model = _model()
    history, truth = _history(model)
    future = np.linspace(0.0, 1.0, 101)
    prediction, estimate = model.forecast_from_history(history, future)
    assert estimate.observation_rank == model.latent_dimension
    assert np.allclose(estimate.latent_at_origin, truth[20:], atol=1e-12)
    exact = np.stack([expm(model.generator() * t) @ truth for t in future])
    assert np.max(np.abs(prediction - exact)) < 1e-8
    assert np.max(np.abs(prediction[:, :10] @ model.mass)) < 1e-10
    assert np.max(np.abs(prediction[:, 10:20] @ model.mass)) < 1e-10


def test_history_refuses_future_samples_and_preserves_clock() -> None:
    model = _model()
    history, _ = _history(model)
    with pytest.raises(ValueError, match="without future samples"):
        CausalStateHistory(
            history.time + 0.01, history.delta, history.omega, history.constant_power
        )
    with pytest.raises(ValueError, match="without future samples"):
        CausalStateHistory(
            history.time[:-1], history.delta[:-1], history.omega[:-1], history.constant_power
        )
    with pytest.raises(ValueError, match="scales"):
        model.estimate_latent(history, omega_scale=np.nan)


def test_unobservable_history_refuses_instead_of_assigning_zero() -> None:
    original = _model()
    coupling = original.coupling.copy()
    coupling[:, 1] = 0.0
    model = PassiveLatentClosure(
        original.mass, original.stiffness, original.damping, coupling, np.zeros((2, 2)), np.eye(2)
    )
    history, _ = _history(model)
    with pytest.raises(UnobservableHistoryError, match="rank 1/2"):
        model.estimate_latent(history)


def test_weakly_observed_state_refuses_even_with_full_rank() -> None:
    original = _model()
    model = PassiveLatentClosure(
        original.mass,
        original.stiffness,
        original.damping,
        original.coupling * 1e-10,
        original.latent_skew,
        original.latent_dissipation,
    )
    history, _ = _history(model)
    with pytest.raises(UnobservableHistoryError, match="smallest singular value"):
        model.estimate_latent(history)


def test_badly_conditioned_history_refuses_with_individually_nonzero_ports() -> None:
    original = _model()
    coupling = original.coupling.copy()
    coupling[:, 1] *= 1e-5
    model = PassiveLatentClosure(
        original.mass,
        original.stiffness,
        original.damping,
        coupling,
        np.zeros((2, 2)),
        np.eye(2),
    )
    history, _ = _history(model)
    with pytest.raises(UnobservableHistoryError, match="condition"):
        model.estimate_latent(history, maximum_condition_number=100.0)


def test_hidden_storage_identity_and_checkpoint_replay(tmp_path: Path) -> None:
    model = _model()
    rng = np.random.default_rng(324)
    for _ in range(30):
        state = np.concatenate(
            (
                coi_project(rng.normal(size=10), model.mass),
                coi_project(rng.normal(size=10), model.mass),
                rng.normal(size=2),
            )
        )
        power = rng.normal(size=10)
        power -= power.mean()
        audit = model.energy_identity(state, power)
        assert audit["identity_error"] < 1e-12
        assert audit["dissipation"] >= 0.0
    path = tmp_path / "closure.npz"
    model.save(path)
    restored = PassiveLatentClosure.load(path)
    history, _ = _history(model)
    time = np.linspace(0.0, 0.3, 31)
    assert np.array_equal(
        model.forecast_from_history(history, time)[0],
        restored.forecast_from_history(history, time)[0],
    )
    with pytest.raises(FileExistsError):
        model.save(path)


def test_history_with_nonzero_registered_constant_input() -> None:
    model = _model()
    state = np.zeros(22)
    state[20:] = [0.2, -0.3]
    power = np.linspace(-0.1, 0.1, 10)
    affine = np.zeros((23, 23))
    affine[:22, :22] = model.generator()
    affine[10:20, 22] = power / model.mass
    time = np.linspace(0.0, 0.2, 21)
    truth = np.stack([(expm(affine * t) @ np.r_[state, 1.0])[:22] for t in time])
    history = CausalStateHistory(time - time[-1], truth[:, :10], truth[:, 10:20], power)
    estimate = model.estimate_latent(history)
    assert np.allclose(estimate.latent_at_origin, truth[-1, 20:], atol=1e-12)
