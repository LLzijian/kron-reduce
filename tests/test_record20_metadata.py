from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.evaluation.record20_input import audit_record20_rule
from kron_reduce.evaluation.record20_metadata import audit_postlabel_metadata

SIGNATURE = "source-postfault-signature"
ALLOCATION = {"declared_length": 5, "fault_line": "Line_7"}


def _unstable_case():
    t = np.arange(5, dtype=np.float64) * 0.01
    delta = np.zeros((5, 2), dtype=np.float64)
    omega = np.ones((5, 2), dtype=np.float64)
    delta[:] = (0.0, np.pi + 0.1)
    omega[:] = (0.97, 1.03)
    rule = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    attrs = {
        "label": "unstable", "event_time_s": float(t[-1]),
        "termination_reason": "physical_loss_of_synchronism",
        "trajectory_end_s": float(t[-1]), "valid_length": np.int64(5),
        "clearing_time_s": np.float64(0.08), "record_start_delay_s": 0.003,
        "solver_success": np.bool_(True), "failed_steps": np.int32(0),
        "postfault_signature": SIGNATURE, "fault_line": "Line_7",
    }
    return attrs, dict(ALLOCATION), t, rule


def _stable_case():
    t = np.arange(1501, dtype=np.float64) * 0.01
    delta = np.zeros((1501, 2), dtype=np.float64)
    omega = np.ones((1501, 2), dtype=np.float64)
    rule = audit_record20_rule(t, delta, omega, np.array([1.0, 2.0]))
    attrs = {
        "label": b"stable", "event_time_s": b"",
        "termination_reason": b"stable_horizon_reached",
        "trajectory_end_s": float(t[-1]), "valid_length": 1501,
        "clearing_time_s": 0.05, "record_start_delay_s": 0.004,
        "solver_success": True, "failed_steps": 0,
        "postfault_signature": SIGNATURE.encode("utf-8"), "fault_line": "Line_7",
    }
    return attrs, {"declared_length": 1501, "fault_line": "Line_7"}, t, rule


def test_shortest_unstable_record_must_trigger_at_last_sample_and_match_metadata():
    attrs, allocation, t, rule = _unstable_case()
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["checkable"] is True
    assert result["consistent"] is True
    assert result["label"] == "unstable"
    assert result["checks"]["unstable_five_point_trigger_ends_at_record_end"] is True
    assert result["checks"]["event_time_matches_rule"] is True
    assert result["checks"]["trajectory_end_matches_record"] is True


def test_stable_1501_point_source_empty_event_time_is_accepted():
    attrs, allocation, t, rule = _stable_case()
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["checkable"] is True
    assert result["consistent"] is True
    assert result["label"] == "stable"
    assert result["checks"]["stable_record_meets_source_acceptance"] is True
    assert result["checks"]["event_time_matches_rule"] is True


def test_stable_record_with_wrong_end_time_is_inconsistent():
    attrs, allocation, t, rule = _stable_case()
    attrs["trajectory_end_s"] = float(t[-1] + 1e-6)
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["checkable"] is True
    assert result["consistent"] is False
    assert "trajectory_end_mismatch" in result["reasons"]


def test_stable_shorter_than_1501_points_cannot_satisfy_source_acceptance():
    t = np.arange(1500, dtype=np.float64) * 0.01
    delta = np.zeros((1500, 2), dtype=np.float64)
    omega = np.ones((1500, 2), dtype=np.float64)
    rule = audit_record20_rule(t, delta, omega, np.array([1.0, 1.0]))
    attrs, allocation, _, _ = _stable_case()
    allocation["declared_length"] = 1500
    attrs["valid_length"] = 1500
    attrs["trajectory_end_s"] = float(t[-1])
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["checks"]["stable_record_meets_source_acceptance"] is False
    assert "stable_rule_acceptance_mismatch" in result["reasons"]


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("solver_success", "true", "solver_success_invalid_type"),
        ("valid_length", "5", "valid_length_invalid_type_or_value"),
        ("failed_steps", "0", "failed_steps_invalid_type_or_value"),
        ("clearing_time_s", float("nan"), "clearing_time_s_invalid_type_or_nonfinite"),
        ("record_start_delay_s", 0.0, "record_start_delay_s_not_positive"),
        ("event_time_s", float("nan"), "unstable_event_time_invalid_type_or_nonfinite"),
    ],
)
def test_invalid_native_attribute_types_values_and_nan_are_rejected(key, value, reason):
    attrs, allocation, t, rule = _unstable_case()
    attrs[key] = value
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["consistent"] is False
    assert reason in result["reasons"]


def test_missing_attributes_and_non_utf8_bytes_are_not_defaulted():
    attrs, allocation, t, rule = _unstable_case()
    del attrs["failed_steps"]
    attrs["termination_reason"] = b"\xff"
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["label"] == "unstable"
    assert "missing_failed_steps" in result["reasons"]
    assert "termination_reason_invalid_string" in result["reasons"]
    assert result["consistent"] is False


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (lambda a, _r: a.update(label="maybe"), "unknown_label"),
        (lambda a, _r: a.update(event_time_s=0.0), "unstable_event_time_mismatch"),
        (lambda a, _r: a.update(termination_reason="stable_horizon_reached"),
         "unstable_termination_reason_mismatch"),
        (lambda a, _r: a.update(postfault_signature="other"), "postfault_signature_mismatch"),
        (lambda a, _r: a.update(fault_line="Line_8"), "fault_line_mismatch"),
        (lambda _a, r: r.update(first_five_point_window={"start": {}, "end": {"index": 3}}),
         "unstable_rule_trigger_mismatch"),
    ],
)
def test_wrong_label_trigger_time_reason_signature_or_line_are_reported(mutate, expected_reason):
    attrs, allocation, t, rule = _unstable_case()
    mutate(attrs, rule)
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert expected_reason in result["reasons"]
    assert result["consistent"] is False


def test_invalid_prelabel_rule_is_unverifiable_not_reclassified_as_stable():
    attrs, allocation, t, rule = _unstable_case()
    rule["status"] = "invalid"
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["checkable"] is False
    assert result["consistent"] is False
    assert result["label"] == "unstable"
    assert result["checks"]["label_rule_matches"] is None
    assert "record20_rule_unavailable" in result["reasons"]


def test_stable_label_cannot_pass_on_a_short_record_with_a_five_point_trigger():
    attrs, allocation, t, rule = _unstable_case()
    attrs["label"] = "stable"
    attrs["event_time_s"] = ""
    attrs["termination_reason"] = "stable_horizon_reached"
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["label"] == "stable"
    assert result["consistent"] is False
    assert "stable_rule_acceptance_mismatch" in result["reasons"]
    assert "stable_event_time_not_source_empty_string" not in result["reasons"]


def test_malformed_rule_trigger_structure_is_reported_not_treated_as_no_hit():
    attrs, allocation, t, rule = _stable_case()
    rule["first_five_point_window"] = None
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["consistent"] is False
    assert "record20_trigger_structure_invalid" in result["reasons"]
    assert "stable_rule_acceptance_mismatch" in result["reasons"]


def test_checker_does_not_mutate_attrs_allocation_clock_or_rule():
    attrs, allocation, t, rule = _unstable_case()
    before = (deepcopy(attrs), deepcopy(allocation), t.copy(), deepcopy(rule))
    audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert attrs == before[0]
    assert allocation == before[1]
    np.testing.assert_array_equal(t, before[2])
    assert rule == before[3]


@pytest.mark.parametrize(
    "bad_line",
    [7, True, 7.0, "7", b"7", bytearray(b"Line_7"), " Line_7", "Line_7.0", "Line_+7", "Line_é7", b"Line_\xff"],
)
def test_fault_line_accepts_only_exact_line_identity_text(bad_line):
    attrs, allocation, t, rule = _unstable_case()
    attrs["fault_line"] = bad_line
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["consistent"] is False
    assert any(reason.startswith("fault_line_invalid_") for reason in result["reasons"])


def test_plan_line_6_and_utf8_bytes_match_without_numeric_coercion():
    plan_path = Path(__file__).resolve().parents[1] / "configs/proposals/kiki_a12_record20_reads.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    line = plan["slots"][0]["fault_line"]
    assert line == "Line_6"
    attrs, allocation, t, rule = _unstable_case()
    attrs["fault_line"] = b"Line_6"
    allocation["fault_line"] = line
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["consistent"] is True
    assert result["checks"]["fault_line_matches_allocation"] is True


def test_fault_line_comparison_preserves_the_complete_identifier():
    attrs, allocation, t, rule = _unstable_case()
    attrs["fault_line"] = "Line_06"
    allocation["fault_line"] = "Line_6"
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["consistent"] is False
    assert result["checks"]["fault_line_matches_allocation"] is False
    assert "fault_line_mismatch" in result["reasons"]


@pytest.mark.parametrize("bad_line", [7, True, 7.0, "7", "Line_7.0"])
def test_allocation_fault_line_must_also_be_a_source_identity(bad_line):
    attrs, allocation, t, rule = _unstable_case()
    allocation["fault_line"] = bad_line
    result = audit_postlabel_metadata(attrs, allocation, t, rule, SIGNATURE)
    assert result["consistent"] is False
    assert "allocation_fault_line_invalid" in result["reasons"]
