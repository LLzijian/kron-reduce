from dataclasses import replace

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.learned_latent import history_features
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    LearnedNonlinearModel,
    NonlinearFitSettings,
    _features,
    fit_nonlinear_model,
    fixed_quadratic_basis,
)
from kron_reduce.physics.coordinates import coi_project
from kron_reduce.workflows.kiki_short_history import (
    CANDIDATES,
    SEEDS,
    ShortHistoryContract,
    candidate_settings,
    classify_los,
    real_evaluation_prerequisite,
    require_real_evaluation,
    score_fixed_predictions,
    select_structure,
    stability_refinement,
    synthetic_observability,
    synthetic_truth,
    trajectory_refinement,
)


def samples():
    closure = synthetic_truth(backbone="sine")
    rng = np.random.default_rng(12)
    output = []
    for i in range(2):
        q, v = [coi_project(rng.normal(0, 0.1, 10), closure.mass) for _ in range(2)]
        path = closure.rollout(q, v, np.array([0.3, -0.2]), np.arange(16) * 0.01,
                               np.zeros(10), np.zeros(10))
        history = CausalStateHistory(np.arange(11)*0.01 - 0.1, path[:11, :10],
                                     path[:11, 10:20], np.zeros(10))
        output.append(EquilibriumLearningSample(
            f"s{i}", "synthetic", f"line{i}", "train", history, np.arange(6)*0.01,
            path[10:, :10], path[10:, 10:20], np.zeros(10)))
    return closure, tuple(output)


def test_fixed_basis_is_data_independent_and_signed():
    basis = fixed_quadratic_basis()
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    assert all(col[np.flatnonzero(abs(col) > 1e-14)[0]] > 0 for col in basis.T)
    assert not basis.flags.writeable
    s = (np.arange(11) - 5)/5
    for col in (np.ones(11), s, s*s):
        np.testing.assert_allclose(basis @ basis.T @ col, col, atol=1e-14)
    _, data = samples()
    row = data[0]
    features = _features(row.history, row.equilibrium_delta, 3, "fixed_quadratic")
    np.testing.assert_allclose(features[:30], (basis.T @ row.history.delta).reshape(-1))
    assert len(features) == 80
    with pytest.raises(ValueError, match="three"):
        NonlinearFitSettings(frames=11, encoder_mode="fixed_quadratic")


def test_legacy_feature_path_and_new_contract():
    _, data = samples()
    s = data[0]
    assert np.array_equal(_features(s.history, s.equilibrium_delta, 3),
                          np.r_[history_features(s.history, 3), s.equilibrium_delta])
    contract = ShortHistoryContract()
    assert (contract.origin_s, contract.history_count, contract.splice_offset) == (0.1, 11, 10)
    contract.validate_history(s.history)
    with pytest.raises(ValueError):
        ShortHistoryContract(power_index=0)


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_actual_trainable_parameter_counts_and_checkpoint(candidate, tmp_path):
    closure, data = samples()
    settings = replace(candidate_settings(candidate, 324), epochs=1, horizons_s=(0.05,))
    model = fit_nonlinear_model(data, closure.mass, settings, allow_partial_horizons=True)
    assert model.fit_record["parameter_count"] == CANDIDATES[candidate]["parameter_count"]
    assert model.fit_record["actual_optimizer_steps"] == 1
    assert model.fit_record["training_rhs_event_evaluations"] == 4*2*(10+5)
    model.save(tmp_path / "model.npz")
    restored = LearnedNonlinearModel.load(tmp_path / "model.npz")
    row = data[0]
    assert np.array_equal(model.predict(row.history, row.forecast_time, row.equilibrium_delta),
                          restored.predict(row.history, row.forecast_time, row.equilibrium_delta))
    checks = trajectory_refinement(model, row, ShortHistoryContract(), model.target_scale)
    assert checks["qualified"]
    assert checks["history_initialization_refined_too"]


def records():
    return [{"candidate": c, "seed": s, "status": "completed", "actual_optimizer_steps": 80,
             "parameter_count": CANDIDATES[c]["parameter_count"], "val_main_nrmse": 1.0}
            for c in CANDIDATES for s in SEEDS]


def test_selection_median_ties_and_no_classification_routing():
    rows = records()
    assert select_structure(rows)["selected"] == "A"
    for r in rows:
        if r["candidate"] in {"C", "D"}:
            r["val_main_nrmse"] = 0.8
        r["fn"] = 0 if r["candidate"] == "C" else 100
    assert select_structure(rows)["selected"] == "D"
    for r in rows:
        if r["candidate"] == "D" and r["seed"] == 325:
            r["val_main_nrmse"] = 100
    assert select_structure(rows)["selected"] == "D"  # median, not mean or best seed


def test_failed_seed_is_retained_and_baseline_failure_incomplete():
    rows = records()
    rows[-1]["val_main_nrmse"] = float("nan")
    rows[0]["status"] = "failed"
    result = select_structure(rows)
    assert not result["candidate_table"][-1]["eligible"]
    assert len(result["candidate_table"][-1]["seeds"]) == 3
    assert not result["baseline_comparison_complete"]
    assert result["selected"] == "B"
    with pytest.raises(ValueError, match="12"):
        select_structure(rows[:-1])
    with pytest.raises(ValueError, match="12"):
        select_structure(rows + [rows[0]])


def test_los_and_fifth_point_and_contract_splice():
    contract = ShortHistoryContract()
    mass, equilibrium = np.ones(10), np.zeros(10)
    history = CausalStateHistory(np.arange(11)*0.01-0.1, np.zeros((11, 10)),
                                 np.zeros((11, 10)), np.zeros(10))
    prediction = np.zeros((801, 20))
    prediction[1:6, :2] = [2.0, -2.0]
    angle_only = classify_los(history, prediction, equilibrium, mass, 1.0, contract)
    assert not angle_only["predicted_los"]
    prediction[1:6, 10:12] = [0.03, -0.03]
    both = classify_los(history, prediction, equilibrium, mass, 1.0, contract)
    assert both["trigger_index"] == 15
    assert both["predicted_los_time_forecast_s"] == 0.05
    assert both["predicted_los_time_record_s"] == 0.15
    prediction[:, :10] = 0
    assert not classify_los(history, prediction, equilibrium, mass, 1.0, contract)["predicted_los"]


def test_refinement_subtracts_10_and_checks_finite_after_trigger():
    fine = np.zeros((801, 20))
    coarse = fine.copy()
    fc = {"predicted_los": True, "trigger_index": 15}
    cc = {"predicted_los": True, "trigger_index": 17}
    check = stability_refinement(fine, coarse, fc, cc, ShortHistoryContract())
    assert check["qualified"]
    assert check["comparison_future_points_including_origin"] == 8
    fine[-1, 0] = np.nan
    assert not stability_refinement(fine, coarse, fc, cc, ShortHistoryContract())["qualified"]


def test_short_target_weights_and_nonfinite_score():
    _, data = samples()
    short = replace(data[1], forecast_time=data[1].forecast_time[:3],
                    target_delta=data[1].target_delta[:3], target_omega=data[1].target_omega[:3])
    population = (data[0], short)
    predictions = {s.event_id: np.c_[s.target_delta, s.target_omega] + (i+1)
                   for i, s in enumerate(population)}
    result = score_fixed_predictions(predictions, population, np.ones(20))
    assert result["available"]["multi_horizon_normalized_rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["complete3_common"]["multi_horizon_normalized_rmse"] is None
    predictions[short.event_id][-1, 0] = np.nan
    result = score_fixed_predictions(predictions, population, np.ones(20))
    assert result["available"]["multi_horizon_normalized_rmse"] is None
    assert result["available"]["horizons"]["3.0"]["failed_event_count"] == 1


def test_known_system_observability_and_unobservable_counterexample():
    coupled, _ = synthetic_observability(observable=True)
    disconnected, _ = synthetic_observability(observable=False)
    assert all(r["rank"] == 2 and r["reconstruction_max_error"] < 1e-10 for r in coupled)
    assert all(r["rank"] == 0 and r["status"] == "REFUSE_HIDDEN_STATE_IDENTIFICATION"
               for r in disconnected)
    assert all(r["noise_gain_l2"] > 0 for r in coupled)
    assert all(r["worst_noise_identity_error"] < 1e-10 for r in coupled)


def test_no_envelope_means_no_real_evaluation_even_for_diagnostics():
    gate = real_evaluation_prerequisite()
    assert gate["status"] == "BLOCKED_NO_CERTIFIED_ENVELOPE"
    assert gate["certified_envelopes"] == []
    assert not gate["real_data_evaluation_allowed"]
    assert not gate["out_of_envelope_diagnostics_allowed"]
    with pytest.raises(PermissionError, match="100ms"):
        require_real_evaluation()
