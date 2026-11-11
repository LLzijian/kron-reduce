
import numpy as np
import pytest

from kron_reduce.evaluation.state_port_train_scores import descriptive, paired_line_summary, verdict

SEEDS = (324, 325, 326)
STEPS = ("0.005", "0.0025")
METHODS = ("baseline", "constant", "state")


def _group(mse):
    truth = [True, False]
    return {
        "event_ids": ["event-a", "event-b"],
        "weights": [0.6, 0.4],
        "counts": [300, 180],
        "M": mse,
        "horizon1": mse,
        "horizon3": mse,
        "q": mse,
        "v": mse,
        "cvar10": mse,
        "complete3": mse,
        "miss": 0,
        "extra": 0,
        "unknown": 0,
        "rule_complete": True,
        "truth_triggered": truth,
        "prediction_triggered": truth.copy(),
        "array_qualified": True,
    }


def _groups():
    return {
        str(seed): {
            step: {
                "baseline": _group(1.0),
                "constant": _group(0.9),
                "state": _group(0.8),
            }
            for step in STEPS
        }
        for seed in SEEDS
    }


def test_train_verdict_accepts_all_five_gates():
    result = verdict(_groups(), valid=True)
    assert result["scientific_verdict"] == "ACCEPT"
    assert all(result[key] for key in (
        "V", "G_baseline", "G_constant", "D_baseline", "D_constant",
    ))
    assert result["promotion_decision"] == "KEEP_BASELINE"
    assert result["new_training_authorized"] is False


def test_constant_trajectory_protection_failure_rejects():
    groups = _groups()
    groups["324"]["0.005"]["constant"]["horizon3"] = 0.7
    result = verdict(groups, valid=True)
    assert result["G_constant"] is False
    assert result["scientific_verdict"] == "REJECT"


def test_state_rule_degradation_against_constant_rejects():
    groups = _groups()
    groups["325"]["0.0025"]["state"]["extra"] = 1
    groups["325"]["0.0025"]["state"]["prediction_triggered"] = [True, True]
    result = verdict(groups, valid=True)
    assert result["D_constant"] is False
    assert result["scientific_verdict"] == "REJECT"


def test_group_population_mismatch_is_rejected():
    groups = _groups()
    groups["326"]["0.005"]["state"]["event_ids"] = ["event-b", "event-a"]
    with pytest.raises(ValueError, match="population, weights or order mismatch"):
        verdict(groups, valid=True)


def test_descriptive_metrics_pool_only_true_future_points():
    truth = np.zeros((2, 3, 20))
    path = np.zeros((2, 3, 22))
    path[0, 1:, :20] = 1
    path[1, 1, :20] = 3
    paths = {
        "324": {
            step: {method: path.copy() for method in METHODS}
            for step in STEPS
        }
    }
    rows = [{"event_id": "a"}, {"event_id": "b"}]
    result = descriptive(paths, truth, np.array([2, 1]), np.ones(20), rows)
    summary = result["324"]["0.005"]["state"]
    assert summary["event_ell"] == [1.0, 9.0]
    assert summary["event_ell_mean"] == 5.0
    assert summary["event_ell_median"] == 5.0
    assert summary["pooled_normalized_mse"] == pytest.approx(11 / 3)
    assert summary["q_rmse_rad"] == pytest.approx((11 / 3) ** 0.5)
    assert summary["v_rmse_rad_per_s"] == pytest.approx((11 / 3) ** 0.5)


def test_paired_line_summary_uses_fixed_source_stratified_se():
    rows = []
    for source, line in (("A", "a1"), ("A", "a2"), ("B", "b1"), ("B", "b2")):
        for repeat in (0, 1):
            rows.append({"source_dataset": source, "fault_line": line,
                         "event_id": f"{source}-{line}-{repeat}"})
    baseline_diff = [1, 1, 3, 3, 4, 4, 8, 8]
    groups = {}
    for seed in SEEDS:
        groups[str(seed)] = {}
        for step in STEPS:
            groups[str(seed)][step] = {
                method: {"event_main_mse": ([0.0] * len(rows) if method != "baseline"
                                             else baseline_diff)}
                for method in METHODS
            }
    summary = paired_line_summary(groups, rows)["324"]["0.005"]["baseline"]
    assert summary["line_mean_ell_reference_minus_state"] == {
        "A/a1": 1.0, "A/a2": 3.0, "B/b1": 4.0, "B/b2": 8.0,
    }
    assert summary["source_mean"] == {"A": 2.0, "B": 6.0}
    assert summary["balanced_mean"] == 4.0
    assert summary["se"] == pytest.approx((1.25) ** 0.5)
