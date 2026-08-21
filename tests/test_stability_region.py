from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation.stability_region import (
    certify_sublevel_region,
    check_psd,
    check_storage_curvature,
    integration_free_stability,
    missed_instability_rate,
    region_fidelity,
    stability_margin,
    supply_identity_error,
)


def _pendulum_grid():
    """Damped pendulum with an analytically known separatrix energy of 2."""
    delta = np.linspace(-2.8, 2.8, 200)
    omega = np.linspace(-2.5, 2.5, 201)
    dd, ww = np.meshgrid(delta, omega, indexing="ij")
    values = (0.5 * ww**2 + (1.0 - np.cos(dd))).ravel()
    decrease = (-0.1 * ww**2).ravel()
    return values, decrease


def test_certify_sublevel_recovers_pendulum_separatrix() -> None:
    values, decrease = _pendulum_grid()
    certificate = certify_sublevel_region(values, decrease, barrier_level=2.0, tolerance=0.0)
    assert certificate.certified
    assert certificate.sample_based
    assert certificate.level == pytest.approx(2.0)
    assert certificate.worst_decrease_inside <= 0.0
    certified_mask = values <= certificate.level
    truth = values < 2.0 + 1e-9
    fidelity = region_fidelity(certified_mask, truth)
    assert fidelity["conservative"]
    assert fidelity["symmetric_difference_ratio"] < 0.02


def test_certify_sublevel_stops_below_dissipation_violation() -> None:
    values, decrease = _pendulum_grid()
    delta = np.linspace(-2.8, 2.8, 200)
    omega = np.linspace(-2.5, 2.5, 201)
    dd, _ = np.meshgrid(delta, omega, indexing="ij")
    broken = np.where(dd.ravel() > 1.0, 0.5, decrease)
    certificate = certify_sublevel_region(values, broken, barrier_level=2.0, tolerance=0.0)
    assert certificate.certified
    # The certified level must sit just below the smallest storage among
    # violating samples (grid point just past delta = 1 at omega = 0).
    smallest_violating = float(np.min(values[dd.ravel() > 1.0]))
    assert 0.4 < certificate.level < smallest_violating


def test_certify_without_barrier_uses_sample_coverage_only() -> None:
    values, decrease = _pendulum_grid()
    certificate = certify_sublevel_region(values, decrease, tolerance=0.0)
    assert certificate.certified
    assert certificate.barrier_level is None
    assert certificate.level == pytest.approx(float(np.max(values)))


def test_certify_refuses_when_samples_are_too_sparse() -> None:
    certificate = certify_sublevel_region(
        np.array([0.1, 0.2, 0.3]), np.array([-0.1, -0.1, -0.1]), min_samples=8
    )
    assert not certificate.certified
    assert np.isnan(certificate.level)


def test_supply_identity_error_is_exact_for_passive_candidate() -> None:
    rng = np.linspace(-1.0, 1.0, 50)
    assert supply_identity_error(rng, rng, np.zeros(50)) == pytest.approx(0.0)
    assert supply_identity_error(rng, rng, np.full(50, 0.25)) == pytest.approx(0.25)


def test_psd_and_curvature_checks() -> None:
    loading = np.array([[1.0, 0.5], [0.0, 2.0], [0.3, 0.1]])
    assert check_psd(loading @ loading.T)["passed"]
    indefinite = check_psd(np.diag([1.0, -1.0]))
    assert not indefinite["passed"]
    assert indefinite["min_eigenvalue"] == pytest.approx(-1.0)
    assert check_storage_curvature(2.0 * np.eye(2))["passed"]
    assert not check_storage_curvature(np.diag([1.0, 0.0]))["passed"]


def test_margin_and_integration_free_judgement() -> None:
    assert stability_margin(2.0, 0.5) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        stability_margin(float("nan"), 0.5)
    margins = np.array([1.5, -0.5, 0.0])
    assert integration_free_stability(margins).tolist() == [True, False, False]


def test_missed_instability_rate_matches_fsr_convention() -> None:
    true_stable = np.array([True, True, False, False])
    predicted_stable = np.array([True, False, True, False])
    metrics = missed_instability_rate(true_stable, predicted_stable)
    assert metrics["FSR"] == pytest.approx(0.5)
    assert metrics["FUR"] == pytest.approx(0.5)
    perfect = missed_instability_rate(true_stable, true_stable)
    assert perfect["FSR"] == pytest.approx(0.0)
    assert perfect["FUR"] == pytest.approx(0.0)
