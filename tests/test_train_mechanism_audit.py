from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation.train_mechanism_audit import audit_event

MASS = np.array([1.0, 2.0, 3.0, 4.0, 1.3, 2.7, 5.1, 0.8, 3.2, 6.0])
EQ = np.linspace(-0.2, 0.2, 10)
OMEGA = 2.0 * np.pi * 60.0
SCALE = np.linspace(0.8, 1.4, 20)


def _paths(nfuture=16):
    history = np.zeros((11, 20))
    true = np.zeros((nfuture + 1, 20))
    pred = np.zeros_like(true)
    # Six positive future points make the earliest best truth window end at 15.
    true[1:7, 1] = 1.2 * np.pi
    relative = np.zeros(10)
    relative[0] = 0.03 * OMEGA
    relative[1] = -relative[0] * MASS[0] / MASS[1]
    true[1:7, 10:] = relative
    pred[:] = true
    # Later, larger prediction excursion must not replace the truth-selected window.
    if nfuture >= 14:
        pred[8:14, 2] = 1.6 * np.pi
        pred[8:14, 10:] = 1.4 * relative
    # Nonzero forecast error across scored points, including the representative window.
    pred[2:, 3] += 0.2
    return history, true, pred


def _run(history, truth, pred):
    return audit_event(history=history, true_future=truth, pred_future=pred,
                       equilibrium_delta=EQ, mass=MASS, omega_s=OMEGA,
                       target_scale=SCALE)


def test_origin_is_not_scored_twice_and_prediction_uses_observed_history():
    history, truth, pred = _paths()
    baseline = _run(history, truth, pred)
    shifted_history, shifted_truth, shifted_pred = history.copy(), truth.copy(), pred.copy()
    shifted_history[-1, :10] += 1000.0
    shifted_history[-1, 10:] += 20.0
    shifted_truth[0] = shifted_history[-1]
    shifted_pred[0] = shifted_history[-1]
    changed = _run(shifted_history, shifted_truth, shifted_pred)
    assert changed["normalized_point_e2"] == baseline["normalized_point_e2"]
    assert changed["event_mse"] == baseline["event_mse"]
    assert baseline["future_points"] == len(truth) - 1
    assert baseline["history_rho"]["truth"]["rho"] == pytest.approx(-1.0)
    assert baseline["history_rho"]["truth"]["triggered"] is False
    assert baseline["future_rho"]["truth"]["triggered"] is True


def test_truth_selects_fixed_window_and_original_time_coefficients():
    history, truth, pred = _paths()
    result = _run(history, truth, pred)
    representative = result["representative_truth_window"]
    assert representative["start_index"] == 11
    assert representative["end_index"] == 15
    fixed = representative["fixed_five_point_audit"]
    assert fixed["truth"]["eligible_windows"] == 1
    assert fixed["prediction"]["eligible_windows"] == 1
    assert fixed["truth"]["rho"] == pytest.approx(representative["truth_margin"])
    assert fixed["prediction"]["rho"] == pytest.approx(
        representative["prediction_margin_same_window"]
    )
    assert fixed["bound_verified"] is True
    assert result["future_rho"]["truth"]["first_trigger_record_s"] == pytest.approx(.15)
    assert result["future_rho"]["truth"]["first_trigger_forecast_s"] == pytest.approx(.05)
    assert fixed["truth"]["first_trigger_end_index"] == 4
    assert fixed["truth"]["index_offset_to_record_grid"] == 11
    assert fixed["truth"]["first_trigger_record_s"] == pytest.approx(.15)
    assert abs(fixed["prediction"]["rho"] - fixed["truth"]["rho"]) <= fixed["E_atom"] + 1e-12
    assert result["future_rho"]["prediction"]["best_window_end_index"] > 15
    pred_margin = result["future_rho"]["prediction"]["rho"]
    same_window_margin = representative["prediction_margin_same_window"]
    assert same_window_margin < pred_margin

    e2 = np.asarray(result["normalized_point_e2"])
    n1, n3 = min(len(e2), 100), min(len(e2), 300)
    expected = 0.5 * (e2[:n1].mean() + e2[:n3].mean())
    assert result["one_second_mse"] == pytest.approx(e2[:n1].mean())
    assert result["three_second_mse"] == pytest.approx(e2[:n3].mean())
    assert result["event_mse"] == pytest.approx(expected)
    for horizon in result["horizon_scores"].values():
        assert horizon["normalized_delta_mse_contribution"] + horizon[
            "normalized_omega_mse_contribution"
        ] == pytest.approx(horizon["normalized_mse"])
        assert horizon["physical_delta_mse_rad2"] >= 0
        assert horizon["physical_omega_mse_rad2_per_s2"] >= 0
    assert result["time_coefficients_sum"] == pytest.approx(1.0)
    assert result["representative_window_original_weight_mass_p"] > 0
    assert result["representative_window_error_contribution_C"] >= 0
    assert result["representative_window_error_density_C_over_p"] == pytest.approx(
        result["representative_window_error_contribution_C"]
        / result["representative_window_original_weight_mass_p"]
    )


def test_margin_bounds_dominate_fixed_window_change_with_nonuniform_mass():
    rng = np.random.default_rng(20261024)
    for _ in range(30):
        history = rng.normal(0, 0.1, size=(11, 20))
        truth = rng.normal(0, 0.2, size=(22, 20))
        truth[0] = history[-1]
        pred = truth + rng.normal(0, 0.03, size=truth.shape)
        pred[0] = history[-1]
        result = _run(history, truth, pred)
        for name in ("history_rho", "future_rho", "full_rho"):
            section = result[name]
            if section["E_atom"] is not None:
                difference = abs(section["prediction"]["rho"] - section["truth"]["rho"])
                tolerance = 1e-12 * max(1.0, abs(section["truth"]["rho"]),
                                        abs(section["prediction"]["rho"]), section["E_vector"])
                assert difference <= section["E_atom"] + tolerance
                assert section["E_atom"] <= section["E_vector"] + tolerance


def test_empty_tail_and_zero_mean_are_reported_as_none():
    history = np.zeros((11, 20))
    origin_only = np.zeros((1, 20))
    no_tail = _run(history, origin_only, origin_only)
    assert no_tail["future_rho"]["truth"]["rho"] is None
    assert no_tail["event_mse"] is None
    assert no_tail["representative_truth_window"] is None
    assert no_tail["maximum_to_mean_e2_ratio"] is None

    history, truth, _pred = _paths()
    exact = _run(history, truth, truth.copy())
    assert exact["maximum_e2"] == 0
    assert exact["ordinary_available_point_mean_e2"] == 0
    assert exact["maximum_to_mean_e2_ratio"] is None


def test_after_one_second_mean_is_independently_reconstructed():
    history, truth, pred = _paths(nfuture=120)
    pred[101:, 4] += np.linspace(0.05, 0.4, len(pred) - 101)
    result = _run(history, truth, pred)
    e2 = np.asarray(result["normalized_point_e2"])
    assert result["after_one_second_available_point_mean_e2"] == pytest.approx(e2[100:].mean())
    assert result["maximum_angle_error_rad"] == pytest.approx(
        np.max(np.abs((pred[1:] - truth[1:])[:, :10]))
    )
    point_angle = np.max(np.abs((pred[1:] - truth[1:])[:, :10]), axis=1)
    np.testing.assert_array_equal(result["point_max_angle_error_rad"], point_angle)
    assert result["maximum_angle_error_future_index"] == int(np.argmax(point_angle))+1
    mass_center = np.sum((pred[1:, 10:] - truth[1:, 10:]) * MASS, axis=1) / MASS.sum()
    coi_delta = (pred[1:, 10:] - truth[1:, 10:]) - mass_center[:, None]
    expected_speed_error = np.max(np.abs(coi_delta))
    assert result["maximum_coi_omega_error_rad_per_s"] == pytest.approx(expected_speed_error)
    assert result["maximum_coi_speed_error_pu"] == pytest.approx(expected_speed_error / OMEGA)
    np.testing.assert_allclose(result["point_max_coi_speed_error_pu"], np.max(np.abs(coi_delta), axis=1)/OMEGA,
                               rtol=0, atol=1e-15)


def test_nonfinite_and_mismatched_input_arrays_are_rejected():
    history, truth, pred = _paths()
    bad = truth.copy()
    bad[2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _run(history, bad, pred)
    with pytest.raises(ValueError, match="equal shape"):
        _run(history, truth, pred[:-1])
    bad_origin = pred.copy()
    bad_origin[0, 0] = 1.0
    with pytest.raises(ValueError, match="exactly equal"):
        _run(history, truth, bad_origin)


