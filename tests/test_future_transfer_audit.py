from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation.future_transfer_audit import (
    PARAMETERS,
    SEEDS,
    STEPS,
    compare_rule_counts,
    compare_rule_pair,
    control_key,
    preflight_prerequisite,
    preflight_truth,
    refinement,
    rule_audit,
    rule_counts,
)

OMEGA = 2.0 * np.pi * 60.0
SCALE = np.linspace(0.8, 1.4, 20)
MASS = np.array([1.0, 2.0, 3.0, 4.0, 1.3, 2.7, 5.1, 0.8, 3.2, 6.0])
EQ = np.zeros(10)


def _rule_arrays(nfuture=16):
    history = np.zeros((11, 20), dtype=np.float64)
    truth = np.zeros((nfuture + 1, 20), dtype=np.float64)
    pred = np.zeros_like(truth)
    relative_velocity = np.zeros(10)
    relative_velocity[0] = 0.04 * OMEGA
    relative_velocity[1] = -relative_velocity[0] * MASS[0] / MASS[1]
    truth[1:7, 0] = 1.2 * np.pi
    truth[1:7, 10:] = relative_velocity
    pred[:] = truth
    pred[8:14, 2] = 1.5 * np.pi
    pred[8:14, 10:] = 1.2 * relative_velocity
    path = np.zeros((nfuture + 1, 22), dtype=np.float64)
    path[:, :20] = pred
    path[:, 20:] = np.linspace(0.0, 0.1, nfuture + 1)[:, None]
    forecast = np.arange(nfuture + 1, dtype=np.float64) * 0.01
    history_time = np.arange(-10, 1, dtype=np.float64) * 0.01
    return history, truth, path, forecast, history_time


def _rule(history, truth, path, forecast, history_time):
    return rule_audit(history, truth, path, EQ, MASS, OMEGA, SCALE,
                      history_time, forecast)


def test_rule_audit_rebuilds_all_four_margins_and_first_point_indices():
    history, truth, path, forecast, history_time = _rule_arrays()
    diagnostic = _rule(history, truth, path, forecast, history_time)
    for key in ("history_rho", "future_rho", "full_rho"):
        assert diagnostic[key]["bound_verified"] is True
        assert diagnostic[key]["E_atom"] <= diagnostic[key]["E_vector"] + 1e-12
    representative = diagnostic["representative_truth_window"]
    assert representative["fixed_five_point_audit"]["bound_verified"] is True
    assert representative["start_index"] == 11
    assert representative["end_index"] == 15
    assert diagnostic["margin_validity"]["qualified"] is True
    assert diagnostic["truth_first_point_violation"]["first_future_record_index"] == 11
    assert diagnostic["truth_first_point_violation"]["first_future_forecast_s"] == pytest.approx(0.01)
    assert diagnostic["prediction_first_point_violation"]["first_future_record_index"] == 11


def test_rule_pair_uses_truth_selected_window_and_enforces_trigger_timing():
    history, truth, coarse, forecast, history_time = _rule_arrays()
    coarse_diag = _rule(history, truth, coarse, forecast, history_time)
    fine = coarse.copy()
    # Delay both positive components by two future samples while retaining a valid fixed-window bound.
    fine[1:3, 0] = 0.0
    fine[1:3, 10:] = 0.0
    fine[7:9, 0] = 1.2 * np.pi
    fine[7:9, 10:] = coarse[6, 10:]
    fine_diag = _rule(history, truth, fine[:, :20].copy(), forecast, history_time)
    fine_22 = fine.copy()
    fine_22[:, :20] = fine[:, :20]
    comparison = compare_rule_pair(
        history, coarse, fine_22, coarse_diag, EQ, MASS, OMEGA, SCALE,
        coarse_diag, fine_diag,
    )
    fixed = comparison["coarse_fine_rule_margin_comparison"]["representative_truth_window"]
    truth_fixed = coarse_diag["representative_truth_window"]
    assert (fixed["start_index"], fixed["end_index"]) == (
        truth_fixed["start_index"], truth_fixed["end_index"]
    )
    assert fixed["fixed_five_point_audit"]["bound_verified"] is True
    assert comparison["same_rule_decision"] is True
    assert comparison["trigger_within_two_points"] is True
    assert comparison["qualified"] is True

    too_late = fine_22.copy()
    too_late[1:4, :20] = 0.0
    too_late[8:11, 0] = 1.2 * np.pi
    too_late[8:11, 10:] = coarse[6, 10:]
    late_diag = _rule(history, truth, too_late, forecast, history_time)
    late = compare_rule_pair(history, coarse, too_late, coarse_diag, EQ, MASS, OMEGA,
                             SCALE, coarse_diag, late_diag)
    assert late["same_rule_decision"] is True
    assert late["trigger_within_two_points"] is False
    assert late["qualified"] is False


def test_rule_counts_preserve_unavailable_and_counter_protection():
    baseline = [
        {"observed_rule_hit": True, "prediction_rule_hit": False},
        {"observed_rule_hit": True, "prediction_rule_hit": True},
        {"observed_rule_hit": False, "prediction_rule_hit": True},
    ] + [{"observed_rule_hit": False, "prediction_rule_hit": False}] * 255
    candidate = [dict(row) for row in baseline]
    candidate[0]["prediction_rule_hit"] = True
    candidate[2]["prediction_rule_hit"] = False
    result = compare_rule_counts(candidate, baseline)
    assert result["D_pass"] is True
    assert result["candidate"]["retained"] == 2
    assert result["candidate"]["positive_not_retained"] == 0
    assert result["candidate"]["truth_not_triggered_prediction_triggered"] == 0
    candidate_with_none = [dict(row) for row in candidate]
    candidate_with_none[-1]["observed_rule_hit"] = None
    baseline_with_none = [dict(row) for row in baseline]
    baseline_with_none[-1]["observed_rule_hit"] = None
    counts = rule_counts(candidate_with_none)
    assert counts["unavailable_windows"] == 1
    assert counts["truth_positive"] == 2
    invalid = compare_rule_counts(candidate_with_none, baseline_with_none)
    assert invalid["D_pass"] is False
    with pytest.raises(ValueError, match="common truth window"):
        compare_rule_counts(candidate, [dict(row, observed_rule_hit=not row["observed_rule_hit"])
                                        for row in baseline])


def test_refinement_rejects_hidden_nonfinite_and_mismatched_paths():
    coarse = np.zeros((8, 22), dtype=np.float64)
    fine = np.zeros_like(coarse)
    scale = np.ones(20)
    assert refinement(coarse, fine, scale)["qualified"] is True
    bad_hidden = coarse.copy()
    bad_hidden[2, 21] = np.nan
    with pytest.raises(ValueError, match="finite"):
        refinement(bad_hidden, fine, scale)
    with pytest.raises(ValueError, match="equal 22-channel"):
        refinement(coarse, fine[:, :21], scale)


def test_finite_huge_inputs_cannot_pass_with_infinite_hidden_norm_bound():
    coarse = np.zeros((8, 22), dtype=np.float64)
    fine = coarse.copy()
    coarse[:, 20:] = -1e308
    fine[:, 20:] = 1e308
    with pytest.raises(FloatingPointError):
        refinement(coarse, fine, np.ones(20))


def test_rule_overflow_is_a_numerical_error_after_array_identity_validation():
    history, truth, path, forecast, history_time = _rule_arrays()
    path[1:, :20] = 1e308
    with pytest.raises(FloatingPointError):
        _rule(history, truth, path, forecast, history_time)


def _full_preflight_fixture():
    counts = np.full(258, 280, dtype=np.int64)
    counts[-3:] = 249
    offsets = np.r_[0, np.cumsum(counts + 1)].astype(np.int64)
    assert offsets[-1] == 72405
    rows = [{"event_id": f"e{i:03d}", "source_dataset": f"source-{i % 4}",
             "fault_line": f"line-{i % 7}", "cache_index": i} for i in range(258)]
    from kron_reduce.evaluation.history_metrics import hierarchy_weights
    weights = hierarchy_weights(rows)
    population = {
        "cache_rows": rows, "weights": weights,
        "allocated_slots": [{}] * 268,
        "seed_event_slots": [{}] * 774,
    }
    mass = np.ones(10, dtype=np.float64)
    history = np.zeros((258, 11, 20), dtype=np.float64)
    history_time = np.tile(np.arange(-10, 1, dtype=np.float64) * 0.01, (258, 1))
    inputs = {
        "history": history,
        "history_time": history_time,
        "constant_power": np.zeros((258, 10), dtype=np.float64),
        "equilibrium": np.zeros((258, 10), dtype=np.float64),
        "mass": mass,
        "target_scale": np.ones(20, dtype=np.float64),
    }
    future = np.zeros((72405, 22), dtype=np.float64)
    decoded = {"A14/permitted_inputs.npz": inputs}
    for seed in SEEDS:
        decoded[f"A14/static-{seed}.npz"] = {
            "parameters__" + name: np.ones(shape, dtype=np.float64)
            for name, shape in PARAMETERS.items()
        }
        decoded[f"A14/static-{seed}.npz"]["parameters__mass"] = mass.copy()
        for step in STEPS:
            decoded[control_key(seed, step)] = {"future": future, "offsets": offsets.copy()}
            for index in range(258):
                decoded[f"A14/seed-{seed}-event-{index:03d}/history-{step}.npz"] = {
                    "reset": np.zeros(22, dtype=np.float64),
                    "baseline_reset": np.zeros(22, dtype=np.float64),
                }
    return population, decoded, offsets, counts


def test_preflight_truth_accepts_synthetic_full_population_and_exact_substep_counts():
    population, decoded, offsets, counts = _full_preflight_fixture()
    pre = preflight_prerequisite(decoded, population)
    truth_times = np.concatenate([
        np.arange(int(count) + 1, dtype=np.float64) * 0.01 for count in counts
    ])
    truth = {
        "future": np.zeros((72405, 20), dtype=np.float64),
        "forecast_time": truth_times,
        "offsets": offsets.copy(),
        "omega_s": np.full(258, OMEGA, dtype=np.float64),
    }
    result = preflight_truth(truth, pre)
    assert result["qualified"] is True
    assert result["scored_points"] == 72147
    assert len(result["event_clock_counts"]) == 258
    assert result["event_clock_counts"][0]["rk4_updates"] == {"0.005": 560, "0.0025": 1120}


def test_preflight_truth_rejects_changed_offsets_and_bad_clocks():
    population, decoded, offsets, counts = _full_preflight_fixture()
    pre = preflight_prerequisite(decoded, population)
    clock = np.concatenate([np.arange(int(count) + 1) * 0.01 for count in counts])
    truth = {"future": np.zeros((72405, 20), dtype=np.float64),
             "forecast_time": clock.astype(np.float64), "offsets": offsets.copy(),
             "omega_s": np.full(258, OMEGA, dtype=np.float64)}
    mismatch = {**truth, "offsets": offsets.copy()}
    mismatch["offsets"][10] += 1
    with pytest.raises(ValueError, match="offsets"):
        preflight_truth(mismatch, pre)
    bad_clock = clock.astype(np.float64)
    bad_clock[1] = -0.01
    with pytest.raises(ValueError, match="clock"):
        preflight_truth({**truth, "forecast_time": bad_clock}, pre)
