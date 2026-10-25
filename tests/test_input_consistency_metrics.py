"""Known-array comparisons, with no real data or instability classification."""
import copy

import numpy as np
import pytest

from kron_reduce.evaluation import input_consistency_metrics as metrics_module
from kron_reduce.evaluation.input_consistency_metrics import (
    campaign_verdict,
    future_event_scores,
    paired_group,
    population_scores,
)


def scores(amplitude, count):
    clock = np.arange(count + 1, dtype=float) / 100
    truth = np.zeros((count + 1, 20))
    pred = np.zeros((count + 1, 22))
    pred[1:, :10] = amplitude
    pred[1:, 10:20] = 2 * amplitude
    return future_event_scores(pred, truth, clock, np.ones(20))


def test_nonorigin_horizons_and_channel_contributions_have_known_solution():
    result = scores(2., 300)
    assert result["q"] == 4.
    assert result["v"] == 16.
    assert result["primary"] == result["horizon1"] == result["horizon3"] == 10.
    assert result["horizons"]["horizon1"]["scored_points"] == 100
    assert result["horizons"]["horizon3"]["scored_points"] == 300


def test_time_rule_uses_real_clock_and_original_available_targets():
    clock = np.array([0., .25, .75, 1.2, 2.8])
    truth = np.zeros((5, 20))
    pred = np.zeros((5, 22))
    pred[1:] = np.arange(1, 5)[:, None]
    result = future_event_scores(pred, truth, clock, np.ones(20))
    assert result["horizon1"] == 2.5
    assert result["horizon3"] == 7.5
    assert result["primary"] == 5.
    assert result["horizons"]["horizon1"]["scored_points"] == 2


def test_original_weight_contributions_and_fractional_tail_are_independent():
    records = [scores(1., 50), scores(2., 150), scores(4., 300)]
    weights, counts = np.array([.2, .75, .05]), np.array([50, 150, 300])
    out = population_scores(records, weights, counts)
    assert out["metrics"]["primary"] == pytest.approx(.2 * 2.5 + .75 * 10 + .05 * 40)
    assert out["worst10_cvar"] == pytest.approx((.05 * 40 + .05 * 10) / .1)
    assert out["complete_3s_horizon3_conditional_mse"] == pytest.approx(40)
    groups = out["duration_groups"]
    assert groups["under_1s"]["partial_primary_contribution_same_denominator"] == pytest.approx(.5)
    assert groups["full_3s"]["original_weight"] == .05


def test_failed_event_keeps_original_mass_and_prevents_complete_population_gate():
    out = population_scores([scores(1., 50), None, scores(4., 300)],
                            np.array([.2, .75, .05]), np.array([50, 150, 300]))
    assert not out["complete"]
    assert out["metrics"] is out["worst10_cvar"] is None
    assert out["missing_weight"] == .75
    assert out["partial_contributions_same_denominator"]["primary"] == pytest.approx(2.5)
    assert paired_group(out, out)["pass"] is False


def test_improving_rejected_reference_is_not_enough_for_campaign_acceptance():
    counts, weights = np.array([50, 150, 300]), np.array([.2, .75, .05])

    def group(a):
        return population_scores([scores(a, int(n)) for n in counts], weights, counts)

    C, I, E = group(2.), group(4.), group(1.)
    comparisons = {(ref, seed, step): paired_group(C, source)
                   for ref, source in (("I", I), ("E", E))
                   for seed in (324, 325, 326) for step in (.005, .0025)}
    out = campaign_verdict(True, comparisons)
    assert out["G_I"] is True and out["G_E"] is False
    assert out["scientific_verdict"] == "REJECT"
    assert out["promotion_decision"] == "KEEP_BASELINE"
    with pytest.raises(ValueError, match="twelve"):
        campaign_verdict(True, {k: v for k, v in comparisons.items() if k[1] != 326})


def test_all_protection_conditions_apply_and_validity_gate_cannot_be_ignored():
    counts, weights = np.array([300]), np.ones(1)
    C = population_scores([scores(1., 300)], weights, counts)
    E = population_scores([scores(2., 300)], weights, counts)
    good = paired_group(C, E)
    assert good["pass"]
    damaged = copy.deepcopy(C)
    damaged["worst10_cvar"] = E["worst10_cvar"] + 1.
    assert not paired_group(damaged, E)["pass"]
    all_good = {(ref, seed, step): good for ref in ("I", "E")
                for seed in (324, 325, 326) for step in (.005, .0025)}
    assert campaign_verdict(True, all_good)["scientific_verdict"] == "ACCEPT"
    assert campaign_verdict(False, all_good)["scientific_verdict"] == "REJECT"


def test_origin_mismatch_or_nonfinite_metric_is_preserved_as_failure():
    clock, truth, pred = np.array([0., .01]), np.zeros((2, 20)), np.zeros((2, 22))
    pred[0, 0] = .1
    with pytest.raises(ValueError, match="origin"):
        future_event_scores(pred, truth, clock, np.ones(20))
    pred[0, 0] = 0
    pred[1] = 1e200
    with np.errstate(over="ignore"), pytest.raises(FloatingPointError):
        future_event_scores(pred, truth, clock, np.ones(20))


def _valid_all_twelve_groups():
    counts, weights = np.array([300]), np.ones(1)
    candidate = population_scores([scores(1., 300)], weights, counts)
    reference = population_scores([scores(2., 300)], weights, counts)
    paired = paired_group(candidate, reference)
    return {
        (ref, seed, step): paired
        for ref in ("I", "E")
        for seed in (324, 325, 326)
        for step in (.005, .0025)
    }


def test_valid_paired_groups_cannot_turn_v_failure_into_acceptance():
    groups = _valid_all_twelve_groups()
    result = campaign_verdict(False, groups)
    assert result["V_pass"] is False
    assert result["G_I"] is result["G_E"] is False
    assert result["scientific_verdict"] == "REJECT"


@pytest.mark.parametrize("bad_pass", [1, "true", np.nan])
def test_campaign_rejects_non_boolean_pass_fields(bad_pass):
    groups = _valid_all_twelve_groups()
    key = next(iter(groups))
    groups[key] = copy.deepcopy(groups[key])
    groups[key]["pass"] = bad_pass
    with pytest.raises(ValueError, match="explicit complete/pass booleans"):
        campaign_verdict(True, groups)


def test_campaign_rejects_non_boolean_condition_or_nonfinite_difference():
    groups = _valid_all_twelve_groups()
    key = next(iter(groups))
    bad_condition = copy.deepcopy(groups)
    bad_condition[key]["conditions"]["q_not_higher"] = 1
    with pytest.raises(ValueError, match="seven explicit conditions"):
        campaign_verdict(True, bad_condition)

    bad_difference = copy.deepcopy(groups)
    bad_difference[key]["differences"]["primary"] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        campaign_verdict(True, bad_difference)


def test_finite_population_inputs_that_overflow_derived_weighted_sum_fail():
    maximum = np.finfo(np.float64).max
    row = {name: maximum for name in ("primary", "q", "v", "horizon1", "horizon3")}
    row["future_points"] = 1
    # The frozen validator intentionally tolerates tiny floating summation error;
    # these finite weights make the derived weighted sum overflow that allowance.
    weights = np.array([0.6000000000005, 0.4])
    assert abs(float(weights.sum()) - 1.) <= 1e-12
    with pytest.raises(FloatingPointError, match="nonfinite derived"):
        population_scores([row, row.copy()], weights, np.array([1, 1]))


def test_nonfinite_derived_cvar_is_rejected_even_when_all_inputs_are_finite(monkeypatch):
    monkeypatch.setattr(metrics_module, "weighted_cvar", lambda *args, **kwargs: float("inf"))
    with pytest.raises(FloatingPointError, match="nonfinite derived"):
        population_scores([scores(1., 1)], np.ones(1), np.array([1]))
