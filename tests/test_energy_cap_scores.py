"""Hand-built arrays only: no scientific input files, models, solvers or paths."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import kron_reduce.evaluation.energy_cap_scores as scores
from kron_reduce.evaluation.train_mechanism_audit import audit_event


def pair(n=300, loss=0.0):
    truth = np.zeros((n + 1, 20))
    prediction = np.zeros((n + 1, 22))
    prediction[1:, :20] = np.sqrt(loss)
    return {
        "history": np.zeros((11, 20)), "truth": truth,
        "coarse": prediction.copy(), "fine": prediction.copy(),
        "equilibrium": np.zeros(10), "mass": np.ones(10), "omega_s": 100.0,
        "scale": np.ones(20), "clock": np.arange(n + 1) * .01,
    }


def group_record(n=300, loss=0.0):
    return scores.score_pair(**pair(n, loss))["steps"]["0.0025"]


def groups(e=1.0, i=10.0, c=.98):
    result = {}
    for seed in (324, 325, 326):
        result[str(seed)] = {}
        for step in scores.STEPS:
            result[str(seed)][step] = {}
            for method, value in (("E", e), ("I", i), ("C", c)):
                result[str(seed)][step][method] = {
                    "M": value, **{key: value for key in scores.PROTECTIONS},
                    "event_ids": ["a"], "weights": [1.0], "counts": [300],
                    "truth_triggered": [False], "prediction_triggered": [False],
                    "miss": 0, "extra": 0, "unknown": 0, "rule_complete": True,
                    "array_qualified": True,
                }
    return result


def test_horizons_use_the_original_equal_window_mse_and_channel_definition():
    data = pair()
    for name in ("coarse", "fine"):
        data[name][1:101, :10] = 1
        data[name][1:101, 10:20] = 3
        data[name][101:, :10] = 2
    result = scores.score_pair(**data)
    record = result["steps"]["0.0025"]
    aggregate = scores.aggregate_group([record], [1.0], [300], event_ids=["a"], expected_count=1)
    assert aggregate["horizon1"] == pytest.approx(5)
    assert aggregate["horizon3"] == pytest.approx(3)
    assert aggregate["M"] == pytest.approx(4)
    assert aggregate["q"] == pytest.approx(2)
    assert aggregate["v"] == pytest.approx(6)
    assert aggregate["nrmse"] == pytest.approx(2)
    assert result["refinement"]["qualified"] is True


def test_partial_horizons_and_fractional_cvar_keep_original_population_weights():
    records = [group_record(n, loss) for n, loss in ((3, 9), (100, 4), (300, 1))]
    result = scores.aggregate_group(
        records, [.04, .06, .9], [3, 100, 300], event_ids=["a", "b", "c"], expected_count=3,
    )
    assert result["M"] == pytest.approx(1.5)
    assert result["cvar10"] == pytest.approx(6.0)
    assert result["complete3"] == pytest.approx(1.0)
    assert result["complete3_weight_mass"] == pytest.approx(.9)
    assert result["duration_groups"]["lt1s"]["main_mse_contribution"] == pytest.approx(.36)
    assert result["duration_groups"]["1s_to_lt3s"]["main_mse_contribution"] == pytest.approx(.24)
    assert result["duration_groups"]["complete3s"]["main_mse_contribution"] == pytest.approx(.9)
    assert result["counts"] == [3, 100, 300]


def test_history_excludes_initial_point_and_uses_ten_channels_per_component():
    truth = np.zeros((11, 20))
    prediction = np.zeros((11, 22))
    prediction[0, :20] = 100
    prediction[1:, :10] = 2
    prediction[1:, 10:20] = 4
    result = scores.score_history(truth, prediction, np.ones(20))
    assert result == {"all": 10.0, "q": 4.0, "v": 16.0}
    row = group_record()
    row["history"] = result
    summary = scores.aggregate_group([row], [1.0], [300], event_ids=["a"], expected_count=1)
    assert summary["history"] == {**result, "cvar10": 10.0}


def test_coarse_fine_fixed_window_is_selected_from_original_truth():
    data = pair(20)
    data["truth"][6:11, 0] = 2 * np.pi
    data["truth"][6:11, 10] = 3
    data["truth"][6:11, 11] = -3
    for name in ("coarse", "fine"):
        data[name][:, :20] = data["truth"]
        data[name][1:6, 0] = 20
        data[name][1:6, 10] = 9
        data[name][1:6, 11] = -9
    result = scores.score_pair(**data)
    reference = result["steps"]["0.005"]["audit"]["representative_truth_window"]
    comparison = result["refinement"]["coarse_fine_rule_margin_comparison"]
    assert reference["start_index"] == 16
    assert comparison["representative_truth_window"]["start_index"] == 16
    wrong_reference = audit_event(
        history=data["history"], true_future=data["coarse"][:, :20],
        pred_future=data["fine"][:, :20], equilibrium_delta=data["equilibrium"],
        mass=data["mass"], omega_s=data["omega_s"], target_scale=data["scale"],
    )
    assert wrong_reference["representative_truth_window"]["start_index"] == 11
    assert result["refinement"]["qualified"] is True


def test_small_path_difference_cannot_hide_a_trigger_disagreement():
    data = pair(5)
    for name, sign in (("coarse", -1), ("fine", 1)):
        data[name][1:, 0] = 2 * np.pi
        data[name][1:, 10] = 2 * (1 + sign * 1e-8)
        data[name][1:, 11] = -2 * (1 + sign * 1e-8)
    result = scores.score_pair(**data)
    assert result["refinement"]["visible_qualified"] is True
    assert result["refinement"]["trigger_consistent"] is False
    assert result["refinement"]["qualified"] is False


def test_hidden_refinement_has_its_own_gate():
    data = pair(3)
    data["coarse"][:, 20] = .001
    result = scores.score_pair(**data)
    assert result["refinement"]["visible_qualified"] is True
    assert result["refinement"]["hidden_qualified"] is False
    assert result["refinement"]["qualified"] is False


def test_actual_margin_values_are_checked_not_only_bound_flags(monkeypatch):
    original = scores.audit_event

    def broken(**kwargs):
        record = original(**kwargs)
        record["future_rho"]["E_atom"] = -1.0
        return record

    monkeypatch.setattr(scores, "audit_event", broken)
    result = scores.score_pair(**pair(3))
    assert result["refinement"]["qualified"] is False
    assert any("negative_error_bound" in value for value in result["refinement"]["margin_failures"])


@pytest.mark.parametrize("mutation", ["clock", "origin", "nonfinite", "shape"])
def test_bad_array_contracts_are_rejected(mutation):
    data = pair(3)
    if mutation == "clock":
        data["clock"][1] += 1e-15
    elif mutation == "origin":
        data["fine"][0, 0] = 1
    elif mutation == "nonfinite":
        data["coarse"][1, 20] = np.nan
    else:
        data["fine"] = data["fine"][:, :20]
    with pytest.raises(ValueError):
        scores.score_pair(**data)


def test_aggregation_rejects_deleted_cases_changed_counts_and_weights():
    record = group_record(3)
    with pytest.raises(ValueError, match="complete expected population"):
        scores.aggregate_group([record], [1.0], [3], event_ids=["a"])
    with pytest.raises(ValueError, match="count disagree"):
        scores.aggregate_group([record], [1.0], [4], event_ids=["a"], expected_count=1)
    with pytest.raises(ValueError, match="sum to one"):
        scores.aggregate_group([record], [.5], [3], event_ids=["a"], expected_count=1)


def test_unknown_rules_remain_unknown_and_empty_complete3_is_not_zero():
    record = group_record(3)
    record["audit"]["future_rho"]["prediction"]["triggered"] = None
    result = scores.aggregate_group([record], [1.0], [3], event_ids=["a"], expected_count=1)
    assert result["unknown"] == 1
    assert result["rule_complete"] is False
    assert result["complete3"] is None


def test_acceptance_is_against_encoder_and_does_not_require_gi():
    result = scores.verdict(groups(i=.5), valid=True, mechanism_exercised=True)
    assert result["G_E"] is True
    assert result["G_I"] is False
    assert result["D"] is True
    assert result["scientific_verdict"] == "ACCEPT"
    assert result["promotion_decision"] == "KEEP_BASELINE"
    assert result["new_training_authorized"] is False


def test_only_beating_rejected_integral_reference_cannot_pass():
    result = scores.verdict(groups(c=1.1), valid=True, mechanism_exercised=True)
    assert result["G_E"] is False
    assert result["G_I"] is True
    assert result["scientific_verdict"] == "REJECT"


def test_one_percent_is_not_replaced_by_any_strict_improvement():
    result = scores.verdict(groups(c=.995), valid=True, mechanism_exercised=True)
    assert result["G_E"] is False
    assert result["scientific_verdict"] == "REJECT"


def test_numerical_effect_proxy_can_fail_despite_one_percent_at_both_steps():
    data = groups()
    data["324"]["0.005"]["C"]["M"] = .97
    result = scores.verdict(data, valid=True, mechanism_exercised=True)
    assert all(row["one_percent"] for row in result["comparisons"]["E"]["324"]["steps"].values())
    assert result["comparisons"]["E"]["324"]["effect_resolved"] is False
    assert result["G_E"] is False


@pytest.mark.parametrize("key", scores.PROTECTIONS)
def test_each_protection_can_independently_reject(key):
    data = groups()
    data["325"]["0.0025"]["C"][key] = 1.01
    result = scores.verdict(data, valid=True, mechanism_exercised=True)
    assert result["G_E"] is False
    assert result["scientific_verdict"] == "REJECT"


@pytest.mark.parametrize("valid,exercised", [(False, True), (True, False)])
def test_no_numerical_validity_or_no_exercised_constraint_cannot_accept(valid, exercised):
    result = scores.verdict(groups(), valid=valid, mechanism_exercised=exercised)
    assert result["G_E"] is True
    assert result["scientific_verdict"] == "REJECT"


def test_new_false_trigger_rejects_even_with_future_mse_gain():
    data = groups()
    c = data["326"]["0.005"]["C"]
    c["extra"] = 1
    c["prediction_triggered"] = [True]
    result = scores.verdict(data, valid=True, mechanism_exercised=True)
    assert result["G_E"] is True
    assert result["D"] is False
    assert result["scientific_verdict"] == "REJECT"


def test_rule_counts_cannot_disagree_with_saved_event_triggers():
    data = groups()
    data["324"]["0.005"]["C"]["prediction_triggered"] = [True]
    with pytest.raises(ValueError, match="rule counts disagree"):
        scores.verdict(data, valid=True, mechanism_exercised=True)


def test_zero_reference_does_not_manufacture_relative_improvement():
    result = scores.verdict(groups(e=0, c=0), valid=True, mechanism_exercised=True)
    assert result["G_E"] is False
    assert result["comparisons"]["E"]["324"]["steps"]["0.005"]["relative_improvement"] is None


def test_verdict_requires_all_seeds_both_steps_and_matching_population():
    original = groups()
    data = deepcopy(original)
    del data["326"]
    with pytest.raises(ValueError, match="fixed seeds"):
        scores.verdict(data, valid=True, mechanism_exercised=True)
    data = deepcopy(original)
    del data["324"]["0.005"]
    with pytest.raises(ValueError, match="both fixed steps"):
        scores.verdict(data, valid=True, mechanism_exercised=True)
    data = deepcopy(original)
    data["324"]["0.005"]["C"]["weights"] = [.99]
    with pytest.raises(ValueError, match="population/weights/order"):
        scores.verdict(data, valid=True, mechanism_exercised=True)


def test_array_refinement_failure_cannot_be_hidden_by_caller_valid_flag():
    data = groups()
    data["324"]["0.005"]["C"]["array_qualified"] = False
    result = scores.verdict(data, valid=True, mechanism_exercised=True)
    assert result["V"] is False
    assert result["scientific_verdict"] == "REJECT"
