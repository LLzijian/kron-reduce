"""Independent A17 replay checks with hand-computed synthetic expectations only."""

from copy import deepcopy

import numpy as np
import pytest

from kron_reduce.evaluation.input_consistency_result_audit import (
    replay_campaign_verdict,
    replay_event_scores,
    replay_paired_group,
    replay_population_scores,
)

FIELDS = ("primary", "q", "v", "horizon1", "horizon3")
CONDITIONS = (
    "primary_strictly_lower", "horizon1_not_higher", "horizon3_not_higher",
    "q_not_higher", "v_not_higher", "worst10_cvar_not_higher",
    "complete_3s_horizon3_conditional_mse_not_higher",
)


def _event(value, count, *, horizon1=None, horizon3=None):
    row = dict.fromkeys(FIELDS, float(value))
    row["future_points"] = count
    if horizon1 is not None:
        row["horizon1"] = float(horizon1)
    if horizon3 is not None:
        row["horizon3"] = float(horizon3)
    return row


def _complete_population(value):
    return {
        "complete": True, "metrics": dict.fromkeys(FIELDS, float(value)),
        "worst10_cvar": float(value),
        "complete_3s_horizon3_conditional_mse": float(value),
    }


def _paired_record(passed=True):
    conditions = dict.fromkeys(CONDITIONS, True)
    conditions["primary_strictly_lower"] = passed
    return {
        "complete": True, "pass": passed, "conditions": conditions,
        "differences": dict.fromkeys(FIELDS, -1.0 if passed else 0.0),
        "comparison_atol": 1e-12,
    }


def _all_groups():
    return {
        (reference, seed, step): _paired_record()
        for reference in ("I", "E")
        for seed in (324, 325, 326)
        for step in (.005, .0025)
    }


def _small_path():
    return np.zeros((3, 22)), np.zeros((3, 20)), np.array([0., .5, 2.]), np.ones(20)


def test_replay_horizons_exclude_origin_use_actual_clock_and_separate_q_v():
    clock = np.array([0., .2, 1., 1. + 2e-12, 2., 3.])
    truth = np.full((6, 20), 123.)
    prediction = np.zeros((6, 22))
    prediction[:, :20] = truth
    prediction[1:, :10] += 2. * np.array([1., 3., 5., 7., 9.])[:, None]
    prediction[1:, 10:20] += 4. * np.array([2., 4., 6., 8., 10.])[:, None]
    prediction[:, 20:] = 1e100  # Finite hidden values never enter visible MSE.
    result = replay_event_scores(prediction, truth, clock, np.r_[np.full(10, 2.),
                                                               np.full(10, 4.)])
    assert result["horizons"] == {
        "horizon1": {"q": 5., "v": 10., "all": 7.5, "scored_points": 2},
        "horizon3": {"q": 33., "v": 44., "all": 38.5, "scored_points": 5},
    }
    assert {key: result[key] for key in FIELDS} == {
        "primary": 23., "q": 19., "v": 27., "horizon1": 7.5, "horizon3": 38.5,
    }
    assert result["future_points"] == 5


def test_replay_horizon_tolerance_includes_the_registered_endpoint():
    prediction, truth, _, scale = _small_path()
    prediction[1, :20], prediction[2, :20] = 1., 3.
    clock = np.array([0., .5, 1. + 5e-13])
    result = replay_event_scores(prediction, truth, clock, scale)
    assert result["horizon1"] == result["horizon3"] == result["primary"] == 5.
    assert result["horizons"]["horizon1"]["scored_points"] == 2


@pytest.mark.parametrize("which", ["prediction", "truth", "clock", "scale", "hidden"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_replay_event_rejects_every_nonfinite_input(which, bad):
    prediction, truth, clock, scale = _small_path()
    values = {"prediction": prediction, "truth": truth, "clock": clock, "scale": scale}
    if which == "hidden":
        prediction[1, 20] = bad
    else:
        values[which].flat[-1] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        replay_event_scores(prediction, truth, clock, scale)


@pytest.mark.parametrize("damage", ["origin", "duplicate_clock", "past_clock", "no_1s", "scale"])
def test_replay_event_rejects_invalid_origin_clock_or_normalizer(damage):
    prediction, truth, clock, scale = _small_path()
    if damage == "origin":
        prediction[0, 0] = 1.
    elif damage == "duplicate_clock":
        clock[2] = clock[1]
    elif damage == "past_clock":
        clock[2] = 3. + 2e-12
    elif damage == "no_1s":
        clock[1] = 1.5
    else:
        scale[0] = 0.
    with pytest.raises(ValueError):
        replay_event_scores(prediction, truth, clock, scale)


def test_replay_event_finite_inputs_cannot_hide_squared_error_overflow():
    prediction, truth, clock, scale = _small_path()
    prediction[1:, :20] = 1e200
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(FloatingPointError):
        replay_event_scores(prediction, truth, clock, scale)


def test_population_uses_original_weighted_mean_and_pure_full3s_horizon():
    records = [_event(2, 50), _event(4, 100), _event(10, 300, horizon1=6, horizon3=14)]
    result = replay_population_scores(records, np.array([.25, .65, .1]),
                                      np.array([50, 100, 300]))
    assert result["complete"] is True
    assert result["metrics"] == pytest.approx({
        "primary": 4.1, "q": 4.1, "v": 4.1, "horizon1": 3.7, "horizon3": 4.5,
    })
    assert result["worst10_cvar"] == pytest.approx(10.)
    assert result["complete_3s_horizon3_conditional_mse"] == pytest.approx(14.)
    groups = result["duration_groups"]
    assert groups["under_1s"]["original_weight"] == .25
    assert groups["from_1s_under_3s"]["original_weight"] == .65
    assert groups["full_3s"]["partial_primary_contribution_same_denominator"] == 1.


def test_fractional_tail_uses_original_mass_and_handles_tied_losses_without_mutation():
    records = [_event(10, 50), _event(10, 100), _event(6, 300), _event(1, 300)]
    original = deepcopy(records)
    weights = np.array([.04, .03, .13, .8])
    result = replay_population_scores(records, weights, np.array([50, 100, 300, 300]))
    # The first two tied events use .07; only .03 of the third event's .13 is used.
    assert result["worst10_cvar"] == pytest.approx((.07 * 10 + .03 * 6) / .1)
    assert result["metrics"]["primary"] == pytest.approx(2.28)
    assert records == original
    assert np.array_equal(weights, np.array([.04, .03, .13, .8]))


def test_tail_can_stop_inside_a_single_event_larger_than_ten_percent():
    result = replay_population_scores([_event(9, 300), _event(1, 300)],
                                      np.array([.4, .6]), np.array([300, 300]))
    assert result["worst10_cvar"] == pytest.approx(9.)


def test_missing_event_preserves_weight_and_never_renormalizes_remaining_events():
    result = replay_population_scores([_event(2, 50), None, _event(10, 300, horizon3=14)],
                                      np.array([.25, .65, .1]), np.array([50, 100, 300]))
    assert result["complete"] is False
    assert result["allocated_eligible_events"] == 3
    assert result["completed_events"] == 2
    assert result["missing_weight"] == .65
    assert result["metrics"] is result["worst10_cvar"] is None
    assert result["partial_contributions_same_denominator"]["primary"] == 1.5
    assert result["duration_groups"]["from_1s_under_3s"]["missing_weight"] == .65
    assert result["complete_3s_horizon3_conditional_mse"] == pytest.approx(14.)


@pytest.mark.parametrize("records, counts", [
    ([None, None], [50, 300]),
    ([_event(2, 50), None], [50, 300]),
    ([_event(2, 50), _event(4, 100)], [50, 100]),
])
def test_full3s_conditional_is_absent_if_missing_or_no_full3s_events(records, counts):
    result = replay_population_scores(records, np.array([.4, .6]), np.array(counts))
    assert result["complete_3s_horizon3_conditional_mse"] is None


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
def test_population_rejects_nonfinite_or_negative_event_metrics(field, bad):
    record = _event(2, 300)
    record[field] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        replay_population_scores([record], np.ones(1), np.array([300]))


@pytest.mark.parametrize("weights, counts", [
    ([.2, .7], [50, 300]), ([0., 1.], [50, 300]),
    ([np.nan, .5], [50, 300]), ([.5, .5], [50., 300.]),
    ([.5, .5], [50, 299]),
])
def test_population_rejects_changed_weights_or_original_counts(weights, counts):
    with pytest.raises((ValueError, FloatingPointError)):
        replay_population_scores([_event(2, 50), _event(4, 300)],
                                 np.array(weights), np.array(counts))


def test_strict_primary_improvement_must_exceed_not_equal_tolerance():
    reference = _complete_population(2e-12)
    candidate = _complete_population(1e-12)
    boundary = replay_paired_group(candidate, reference)
    assert boundary["conditions"]["primary_strictly_lower"] is False
    assert boundary["pass"] is False
    candidate["metrics"]["primary"] = float(np.nextafter(1e-12, -np.inf))
    assert replay_paired_group(candidate, reference)["pass"] is True


@pytest.mark.parametrize("field", ["horizon1", "horizon3", "q", "v",
                                   "worst10_cvar", "complete_3s_horizon3_conditional_mse"])
def test_each_protection_accepts_equality_at_tolerance_and_rejects_above(field):
    candidate, reference = _complete_population(1.), _complete_population(2.)
    target_c = candidate["metrics"] if field in FIELDS else candidate
    target_r = reference["metrics"] if field in FIELDS else reference
    target_r[field] = 0.
    target_c[field] = 1e-12
    assert replay_paired_group(candidate, reference)["pass"] is True
    target_c[field] = float(np.nextafter(1e-12, np.inf))
    result = replay_paired_group(candidate, reference)
    assert result["pass"] is False
    assert result["conditions"][field + "_not_higher"] is False


def test_pair_rejects_missing_full3s_metric_and_marks_incomplete_populations():
    candidate, reference = _complete_population(1.), _complete_population(2.)
    candidate["complete_3s_horizon3_conditional_mse"] = None
    assert replay_paired_group(candidate, reference)["pass"] is False
    candidate["complete"] = False
    assert replay_paired_group(candidate, reference) == {
        "complete": False, "pass": False, "reason": "original_population_incomplete",
        "conditions": None, "differences": None,
    }


@pytest.mark.parametrize("reference", ["I", "E"])
@pytest.mark.parametrize("seed", [324, 325, 326])
@pytest.mark.parametrize("step", [.005, .0025])
def test_every_one_of_six_groups_is_required_separately_for_each_reference(reference, seed, step):
    groups = _all_groups()
    groups[(reference, seed, step)] = _paired_record(False)
    result = replay_campaign_verdict(True, groups)
    assert result["G_" + reference] is False
    assert result["G_" + ("E" if reference == "I" else "I")] is True
    assert result["scientific_verdict"] == "REJECT"


def test_campaign_acceptance_and_invalidity_keep_baseline_without_instability_claim():
    groups = _all_groups()
    result = replay_campaign_verdict(True, groups)
    assert result == {
        "V_pass": True, "G_I": True, "G_E": True, "scientific_verdict": "ACCEPT",
        "promotion_decision": "KEEP_BASELINE", "new_missed_instability_result_established": False,
    }
    invalid = replay_campaign_verdict(False, groups)
    assert invalid["G_I"] is invalid["G_E"] is False
    assert invalid["scientific_verdict"] == "REJECT"


@pytest.mark.parametrize("bad", [0, 1, "true", np.bool_(True), np.nan])
def test_campaign_requires_a_python_boolean_validity_gate(bad):
    with pytest.raises(ValueError):
        replay_campaign_verdict(bad, _all_groups())


@pytest.mark.parametrize("field", ["complete", "pass", "condition"])
@pytest.mark.parametrize("bad", [1, "true", np.bool_(True), np.nan])
def test_campaign_requires_explicit_booleans_in_every_paired_record(field, bad):
    groups = _all_groups()
    row = groups[("I", 324, .005)]
    if field == "condition":
        row["conditions"]["q_not_higher"] = bad
    else:
        row[field] = bad
    # NaN is both nonfinite and non-boolean; validation precedence is not a gate.
    if field == "condition" and isinstance(bad, float) and np.isnan(bad):
        expected_errors = (ValueError, FloatingPointError)
    else:
        expected_errors = (ValueError,)
    with pytest.raises(expected_errors):
        replay_campaign_verdict(True, groups)


@pytest.mark.parametrize("damage", ["missing", "extra", "inconsistent", "incomplete_pass"])
def test_campaign_rejects_missing_extra_or_inconsistent_groups(damage):
    groups = _all_groups()
    key = ("E", 326, .0025)
    if damage == "missing":
        del groups[key]
    elif damage == "extra":
        groups[("E", 999, .0025)] = _paired_record()
    elif damage == "inconsistent":
        groups[key]["conditions"]["v_not_higher"] = False
    else:
        groups[key]["complete"] = False
    with pytest.raises(ValueError):
        replay_campaign_verdict(True, groups)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_campaign_rejects_nonfinite_saved_differences(bad):
    groups = _all_groups()
    groups[("E", 326, .0025)]["differences"]["primary"] = bad
    with pytest.raises(FloatingPointError):
        replay_campaign_verdict(True, groups)


@pytest.mark.parametrize("bad", [1, "true", np.bool_(True), np.nan])
def test_pair_requires_explicit_boolean_completeness(bad):
    candidate, reference = _complete_population(1.), _complete_population(2.)
    candidate["complete"] = bad
    with pytest.raises(ValueError):
        replay_paired_group(candidate, reference)


def test_pair_rejects_nonfinite_nested_fields_even_for_incomplete_population():
    candidate, reference = _complete_population(1.), _complete_population(2.)
    candidate["complete"] = False
    candidate["metrics"]["q"] = np.nan
    with pytest.raises(FloatingPointError):
        replay_paired_group(candidate, reference)
