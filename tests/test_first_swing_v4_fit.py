import pytest
import torch

from kron_reduce.evaluation.first_swing_balanced_fit import (
    checkpoint_metrics,
    robust_trajectory_loss,
    trajectory_groups,
)


def test_huber_only_applies_to_stable_events_and_available_future():
    truth = torch.zeros((2, 3, 1), dtype=torch.float64)
    prediction = torch.tensor([[[99.], [4.], [100.]], [[-99.], [4.], [2.]]], dtype=torch.float64)
    errors = robust_trajectory_loss(prediction, truth, torch.ones(1), torch.tensor([1, 2]),
                                    torch.tensor([True, False]), 1.)
    torch.testing.assert_close(errors, torch.tensor([7., 10.], dtype=torch.float64))


def test_tn_precision_and_full_stable_precision_are_both_retained():
    records = [{"event_id": eid, "label": "stable", "horizon": "8", "rmse": value, "qualified": True}
               for eid, value in (("a", .2), ("b", .18), ("c", 2.9))]
    stability = [{"id": eid, "truth_label": "stable", "qualified": True,
                  "predicted_label": "unstable" if eid == "c" else "stable"} for eid in "abc"]
    groups = trajectory_groups({"records": records, "stability_events": stability})
    assert groups["allocated_stable"] == 3 and groups["tn_count"] == 2
    assert groups["median_8s"] == .2
    assert groups["mean_8s"] == pytest.approx((.2+.18+2.9)/3)
    assert groups["tn_mean_8s"] == pytest.approx(.19)
    records[-1]["qualified"] = False
    groups = trajectory_groups({"records": records, "stability_events": stability})
    assert groups["mean_8s"] is groups["median_8s"] is None
    assert groups["tn_mean_8s"] == pytest.approx(.19)


def test_joint_gate_requires_both_steps_of_train_witness_and_full_val():
    steps = ("0.0025", "0.00125")
    result = {"stability": {"allocated": 107, "uncovered": 0, "numerical_failures": 0,
                            "tp": 12, "fp": 2, "unsettled_predictions": 0},
              "stable_errors": {"qualified_stable_8s": 95, "median_8s": .21, "tn_mean_8s": .24, "mean_8s": .28},
              "first_500ms_peaks": {dt: {"positive_median": 1., "qualified": 12} for dt in steps},
              "task3_six_term_score": .3}
    witness = {dt: {"qualified": 33, "signed_median": 2.} for dt in steps}
    reference = dict.fromkeys(steps, 2.5)
    settings = {"max_val_fp": 2, "target_val_fp": 1, "max_val_stable_median": .21, "max_val_tn_mean": .24,
                "witness_relative_limit": .8}
    passed = checkpoint_metrics(result, witness, reference, settings, 3)
    assert passed["joint_pass"] and not passed["target_fp_pass"]
    witness[steps[1]]["signed_median"] = 2.01
    assert not checkpoint_metrics(result, witness, reference, settings, 3)["joint_pass"]
    witness[steps[1]]["qualified"] = 32
    assert checkpoint_metrics(result, witness, reference, settings, 3)["rank"][0] == 3
