from __future__ import annotations

import numpy as np
import pytest

import kron_reduce.evaluation.encoder_energy_cap as encoder_energy_cap_module
from kron_reduce.evaluation.encoder_energy_cap import (
    encoder_energy_cap,
    trajectory_horizon_metrics,
    weighted_cvar,
)
from kron_reduce.evaluation.history_metrics import weighted_cvar as history_weighted_cvar


def _svd_for_diagonal(singular_values, rows=None):
    singular = np.asarray(singular_values, dtype=float)
    columns = singular.size
    rows = columns if rows is None else rows
    u = np.zeros((rows, columns))
    u[:columns, :] = np.eye(columns)
    return {
        "u": u,
        "s": singular,
        "vh": np.eye(columns),
        "rank": int(np.count_nonzero(singular > 1e-12)),
        "threshold": 1e-12,
    }


def _integral(matrix, right, z4, svd):
    return {
        "Ltilde": matrix, "btilde4": right, "zh4": z4,
        "svd": {"U": svd["u"], "s": svd["s"], "Vh": svd["vh"],
                "rank": svd["rank"], "tol": svd["threshold"]},
    }


def test_zero_encoder_radius_precedes_saved_rank_deficiency():
    matrix = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    right = np.array([1.0, -2.0, 3.0])
    record = _integral(matrix, right, np.zeros(2), _svd_for_diagonal([2.0, 0.0], rows=3))

    result = encoder_energy_cap(record, np.zeros(2))

    assert result["radius"] == 0.0
    assert result["candidate"]["status"] == "zero_baseline"
    np.testing.assert_array_equal(result["candidate"]["z"], np.zeros(2))
    assert result["candidate_objective"] == pytest.approx(result["encoder_objective"])
    assert result["candidate_objective_not_worse_than_encoder"] is True


def test_positive_radius_rank_deficiency_is_preserved_as_unresolved():
    matrix = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    right = np.array([1.0, 2.0, 3.0])
    record = _integral(matrix, right, np.zeros(2), _svd_for_diagonal([2.0, 0.0], rows=3))

    result = encoder_energy_cap(record, np.array([0.25, -0.5]))

    assert result["radius"] == pytest.approx(np.hypot(0.25, -0.5))
    assert result["candidate"]["status"] == "unresolved"
    assert result["candidate"]["z"] is None
    assert result["candidate_residual"] is None
    assert result["candidate_objective_not_worse_than_encoder"] is None
    assert np.isfinite(result["encoder_residual_norm"])


def test_active_candidate_reports_encoder_and_candidate_objectives_without_a_gate():
    matrix = np.eye(2)
    right = np.array([3.0, 1.0])
    record = _integral(matrix, right, right.copy(), _svd_for_diagonal([1.0, 1.0]))

    result = encoder_energy_cap(record, np.array([0.0, 1.0]))

    candidate = result["candidate"]
    assert candidate["status"] == "active"
    assert result["radius"] == 1.0
    assert result["encoder_residual_norm"] == pytest.approx(3.0)
    assert result["candidate_objective"] <= result["encoder_objective"]
    assert result["candidate_objective_not_worse_than_encoder"] is True
    assert result["candidate_residual"].shape == (2,)


def test_candidate_objective_comparison_is_reported_not_used_as_a_success_gate(monkeypatch):
    matrix = np.eye(2)
    right = np.array([1.0, 0.0])
    record = _integral(matrix, right, right.copy(), _svd_for_diagonal([1.0, 1.0]))
    monkeypatch.setattr(encoder_energy_cap_module, "solve_energy_ball", lambda *_args, **_kwargs: {
        "z": np.array([0.0, 1.0]), "status": "active",
    })

    result = encoder_energy_cap(record, np.array([1.0, 0.0]))

    assert result["encoder_objective"] == 0.0
    assert result["candidate_objective"] == pytest.approx(1.0)
    assert result["candidate_objective_not_worse_than_encoder"] is False


def test_inactive_candidate_preserves_saved_z4_bits(monkeypatch):
    monkeypatch.setattr(np.linalg, "svd", lambda *_args, **_kwargs: pytest.fail("must reuse saved SVD"))
    z4 = np.array([0.25, -0.0], dtype=np.float64)
    matrix = np.eye(2)
    record = _integral(matrix, z4.copy(), z4.copy(), _svd_for_diagonal([1.0, 1.0]))

    result = encoder_energy_cap(record, z4)

    assert result["candidate"]["status"] == "inactive"
    assert result["candidate"]["z"].tobytes() == z4.tobytes()
    assert result["candidate_objective_not_worse_than_encoder"] is True


def test_a14_integral_adapter_rejects_missing_fields_and_incompatible_shapes():
    with pytest.raises(ValueError, match="missing saved A14 fields"):
        encoder_energy_cap({"Ltilde": np.eye(2)}, np.ones(2))

    record = _integral(np.eye(2), np.ones(2), np.ones(2), _svd_for_diagonal([1.0, 1.0]))
    with pytest.raises(ValueError, match="encoder_z_h must be finite with shape"):
        encoder_energy_cap(record, np.ones(3))
    with pytest.raises(ValueError, match="btilde4 must be finite with shape"):
        encoder_energy_cap({**record, "btilde4": np.ones(3)}, np.ones(2))
    with pytest.raises(ValueError, match="zh4 must be finite with shape"):
        encoder_energy_cap({**record, "zh4": np.ones(1)}, np.ones(2))


def test_adapter_rejects_complex_encoder_without_dropping_imaginary_part():
    record = _integral(np.eye(2), np.ones(2), np.ones(2), _svd_for_diagonal([1.0, 1.0]))
    with pytest.raises(ValueError, match="encoder_z_h must be finite with shape"):
        encoder_energy_cap(record, np.array([1.0 + 2.0j, 0.0]))


def _trajectory_fixture(relative_times):
    clock = np.asarray(relative_times, dtype=float)
    truth = np.zeros((clock.size, 20))
    prediction = np.zeros((clock.size, 22))
    visible_values = np.zeros(clock.size)
    visible_values[0] = 999.0
    visible_values[1:] = np.arange(1, clock.size, dtype=float) / 100
    prediction[:, :10] = (2.0 * visible_values)[:, None]
    prediction[:, 10:20] = (4.0 * visible_values)[:, None]
    scales = np.r_[np.full(10, 2.0), np.full(10, 4.0)]
    return clock, truth, prediction, scales


def test_horizon_metrics_use_actual_future_points_exclude_t0_and_separate_units():
    clock, truth, prediction, scale = _trajectory_fixture(np.arange(0.0, 3.01, 0.01))

    result = trajectory_horizon_metrics(clock, truth, prediction, scale)

    one = result["horizons"]["1"]
    three = result["horizons"]["3"]
    assert result["initial_point_excluded"] is True
    assert result["available_future_point_count"] == 300
    assert one["point_count"] == 100 and one["complete"] is True
    one_values = np.arange(1, 101, dtype=float) / 100
    one_mse = float(np.mean(one_values**2))
    assert one["normalized_mse"]["all"] == pytest.approx(one_mse)
    assert one["normalized_rmse"]["all"] == pytest.approx(np.sqrt(one_mse))
    assert one["physical_rmse"]["q"] == pytest.approx(2 * np.sqrt(one_mse))
    assert one["physical_rmse"]["v"] == pytest.approx(4 * np.sqrt(one_mse))
    assert one["maximum_absolute_error"]["normalized"]["all"] == pytest.approx(1.0)
    assert one["maximum_absolute_error"]["physical"]["v"] == pytest.approx(4.0)
    assert three["point_count"] == 300
    assert three["complete"] is True and result["complete_3s"] is True
    three_values = np.arange(1, 301, dtype=float) / 100
    assert three["normalized_mse"]["all"] == pytest.approx(float(np.mean(three_values**2)))


def test_partial_three_second_horizon_uses_all_available_points_without_padding():
    clock, truth, prediction, scale = _trajectory_fixture(np.arange(0.0, 2.51, 0.01))

    result = trajectory_horizon_metrics(clock, truth, prediction, scale)

    one = result["horizons"]["1"]
    three = result["horizons"]["3"]
    assert one["point_count"] == 100
    assert one["normalized_mse"]["all"] == pytest.approx(float(np.mean((np.arange(1, 101)/100)**2)))
    assert three["point_count"] == 250
    assert three["complete"] is False
    assert result["complete_3s"] is False
    assert three["normalized_mse"]["all"] == pytest.approx(float(np.mean((np.arange(1, 251)/100)**2)))


@pytest.mark.parametrize("scale", [np.zeros(20), np.r_[np.ones(19), -1.0], np.full(20, np.nan)])
def test_horizon_metrics_reject_nonpositive_or_nonfinite_scales(scale):
    clock, truth, prediction, _ = _trajectory_fixture([0.0, 0.01])
    with pytest.raises(ValueError, match="target_scale"):
        trajectory_horizon_metrics(clock, truth, prediction, scale)


def test_horizon_metrics_rejects_bad_clock_and_state_shapes():
    clock, truth, prediction, scale = _trajectory_fixture([0.0, 0.01])
    with pytest.raises(ValueError, match="strictly increasing"):
        trajectory_horizon_metrics(np.array([0.0, 0.01, 0.01]),
                                   np.zeros((3, 20)), np.zeros((3, 22)), scale)
    with pytest.raises(ValueError, match="fixed 10 ms spacing"):
        trajectory_horizon_metrics(np.array([0.0, 0.02]),
                                   np.zeros((2, 20)), np.zeros((2, 22)), scale)
    with pytest.raises(ValueError, match="origin must be zero"):
        trajectory_horizon_metrics(np.array([0.01, 0.02]), truth, prediction, scale)
    with pytest.raises(ValueError, match=r"truth must have shape"):
        trajectory_horizon_metrics(clock, truth[:, :19], prediction, scale)


def test_weighted_cvar_is_reused_from_shared_history_metrics():
    losses = np.array([1.0, 4.0, 2.0])
    weights = np.array([0.2, 0.5, 0.3])
    assert weighted_cvar(losses, weights, 0.4) == history_weighted_cvar(losses, weights, 0.4)
