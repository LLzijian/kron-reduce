"""Pure metadata checks for A12, including exact source fault-line identity.

The archived generator reads ``event["line"]`` against ``ss.Line.idx.v`` and
writes that value unchanged as ``fault_line`` (generate_private_dataset.py,
lines 56 and 187). Its pilot builder uses ``str(line)`` (line 214 onward), and
the frozen plan contains identities such as ``Line_6``. Therefore this field
is validated as text and compared without numeric normalization; other count
fields continue to require native integer values.
"""
from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any

import numpy as np

TIME_TOLERANCE_S = 1e-10
STABLE_REQUIRED_POINTS = 1501
_FAULT_LINE_IDENTITY = re.compile(r"Line_[0-9]+\Z", flags=re.ASCII)


def _text(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, str):
        return value, None
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8", errors="strict"), None
        except UnicodeDecodeError:
            return None, "invalid_utf8"
    return None, "not_string"


def _fault_line_identity(value: Any) -> tuple[str | None, str | None]:
    text, error = _text(value)
    if error is not None:
        return None, error
    if _FAULT_LINE_IDENTITY.fullmatch(text) is None:
        return None, "invalid_fault_line_identity"
    return text, None


def _strict_integer(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _trigger_end(record20_rule: dict[str, Any]) -> tuple[int | None, bool]:
    window = record20_rule.get("first_five_point_window")
    if not isinstance(window, dict):
        return None, False
    end = window.get("end")
    if not isinstance(end, dict) or "index" not in end:
        return None, False
    raw_index = end["index"]
    if raw_index is None:
        return None, True
    index = _strict_integer(raw_index)
    return index, index is not None


def _add(reasons: list[str], code: str) -> None:
    if code not in reasons:
        reasons.append(code)


def audit_postlabel_metadata(
    attrs: dict[str, Any],
    allocation_row: dict[str, Any],
    t,
    record20_rule: dict[str, Any],
    source_signature: str,
) -> dict[str, Any]:
    """Compare native event metadata against the frozen allocation and rule.

    The function never mutates its inputs, converts unknown labels into a
    default, or computes classification metrics. Raw attribute preservation
    remains the caller's responsibility.
    """
    if not isinstance(attrs, dict):
        raise TypeError("attrs must be the native attribute mapping")
    if not isinstance(allocation_row, dict):
        raise TypeError("allocation_row must be the frozen whitelist mapping")
    times = np.asarray(t, dtype=np.float64)
    reasons: list[str] = []
    checks: dict[str, bool | None] = {}
    required = (
        "label", "event_time_s", "termination_reason", "trajectory_end_s",
        "valid_length", "clearing_time_s", "record_start_delay_s",
        "solver_success", "failed_steps", "postfault_signature", "fault_line",
    )
    present = {key: key in attrs for key in required}
    for key, exists in present.items():
        checks[f"{key}_present"] = exists
        if not exists:
            _add(reasons, f"missing_{key}")

    label: str | None = None
    if present["label"]:
        label, error = _text(attrs["label"])
        if error is not None:
            _add(reasons, "invalid_label_type")
        elif label not in {"stable", "unstable"}:
            _add(reasons, "unknown_label")
            label = None
    checks["label_known"] = label in {"stable", "unstable"} if present["label"] else None

    n = int(times.size) if times.ndim == 1 else None
    clock_usable = bool(
        times.ndim == 1 and times.size > 0 and np.isfinite(times).all()
    )
    if not clock_usable:
        _add(reasons, "raw_clock_unavailable")
    declared_length = _strict_integer(allocation_row.get("declared_length"))
    checks["allocation_length_valid"] = declared_length is not None
    if declared_length is None:
        _add(reasons, "allocation_length_invalid")
    elif n is not None and declared_length != n:
        _add(reasons, "raw_length_differs_from_allocation")

    valid_length = _strict_integer(attrs.get("valid_length")) if present["valid_length"] else None
    checks["valid_length_native_integer"] = valid_length is not None if present["valid_length"] else None
    if present["valid_length"] and valid_length is None:
        _add(reasons, "valid_length_invalid_type_or_value")
    elif valid_length is not None and (valid_length != declared_length or valid_length != n):
        _add(reasons, "valid_length_mismatch")

    failed_steps = _strict_integer(attrs.get("failed_steps")) if present["failed_steps"] else None
    checks["failed_steps_native_integer_zero"] = failed_steps == 0 if present["failed_steps"] else None
    if present["failed_steps"] and failed_steps is None:
        _add(reasons, "failed_steps_invalid_type_or_value")
    elif failed_steps != 0:
        _add(reasons, "failed_steps_nonzero")

    solver_success = attrs.get("solver_success") if present["solver_success"] else None
    is_native_bool = isinstance(solver_success, (bool, np.bool_))
    checks["solver_success_native_true"] = bool(solver_success) if is_native_bool else False if present["solver_success"] else None
    if present["solver_success"] and not is_native_bool:
        _add(reasons, "solver_success_invalid_type")
    elif is_native_bool and not bool(solver_success):
        _add(reasons, "solver_success_false")

    for key in ("clearing_time_s", "record_start_delay_s"):
        value = _finite_number(attrs.get(key)) if present[key] else None
        checks[f"{key}_finite_positive"] = value is not None and value > 0 if present[key] else None
        if present[key] and value is None:
            _add(reasons, f"{key}_invalid_type_or_nonfinite")
        elif value is not None and value <= 0:
            _add(reasons, f"{key}_not_positive")

    term_reason = None
    if present["termination_reason"]:
        term_reason, error = _text(attrs["termination_reason"])
        if error is not None:
            _add(reasons, "termination_reason_invalid_string")
    signature = None
    if present["postfault_signature"]:
        signature, error = _text(attrs["postfault_signature"])
        if error is not None:
            _add(reasons, "postfault_signature_invalid_string")
        elif not isinstance(source_signature, str) or signature != source_signature:
            _add(reasons, "postfault_signature_mismatch")
    checks["postfault_signature_matches_source"] = (
        signature == source_signature if signature is not None and isinstance(source_signature, str)
        else False if present["postfault_signature"] else None
    )

    attr_fault_line, attr_fault_line_error = (
        _fault_line_identity(attrs.get("fault_line")) if present["fault_line"] else (None, None)
    )
    expected_fault_line, _ = _fault_line_identity(allocation_row.get("fault_line"))
    checks["fault_line_matches_allocation"] = (
        attr_fault_line == expected_fault_line
        if attr_fault_line is not None and expected_fault_line is not None
        else False if present["fault_line"] else None
    )
    if present["fault_line"] and attr_fault_line is None:
        _add(reasons, "fault_line_invalid_identity" if attr_fault_line_error == "invalid_fault_line_identity" else "fault_line_invalid_type")
    elif expected_fault_line is None:
        _add(reasons, "allocation_fault_line_invalid")
    elif attr_fault_line != expected_fault_line:
        _add(reasons, "fault_line_mismatch")

    trajectory_end = _finite_number(attrs.get("trajectory_end_s")) if present["trajectory_end_s"] else None
    checks["trajectory_end_matches_record"] = (
        abs(trajectory_end - float(times[-1])) <= TIME_TOLERANCE_S
        if trajectory_end is not None and clock_usable else False if present["trajectory_end_s"] else None
    )
    if present["trajectory_end_s"] and trajectory_end is None:
        _add(reasons, "trajectory_end_invalid_type_or_nonfinite")
    elif trajectory_end is not None and clock_usable and abs(trajectory_end - float(times[-1])) > TIME_TOLERANCE_S:
        _add(reasons, "trajectory_end_mismatch")

    rule_status = record20_rule.get("status") if isinstance(record20_rule, dict) else None
    rule_valid = rule_status == "valid"
    checks["record20_rule_valid"] = rule_valid
    if not rule_valid:
        _add(reasons, "record20_rule_unavailable")

    event_time_match: bool | None = None
    rule_label_match: bool | None = None
    if label == "unstable" and rule_valid:
        trigger_index, trigger_structure_valid = _trigger_end(record20_rule)
        expected_unstable = bool(
            trigger_structure_valid and n is not None and n > 0 and trigger_index == n - 1
        )
        rule_label_match = expected_unstable
        checks["unstable_five_point_trigger_ends_at_record_end"] = expected_unstable
        if not trigger_structure_valid:
            _add(reasons, "record20_trigger_structure_invalid")
        if not expected_unstable:
            _add(reasons, "unstable_rule_trigger_mismatch")
        if present["termination_reason"] and term_reason != "physical_loss_of_synchronism":
            _add(reasons, "unstable_termination_reason_mismatch")
        if present["event_time_s"]:
            event_time = _finite_number(attrs["event_time_s"])
            event_time_match = bool(
                event_time is not None and clock_usable and expected_unstable
                and abs(event_time - float(times[-1])) <= TIME_TOLERANCE_S
            )
            if event_time is None:
                _add(reasons, "unstable_event_time_invalid_type_or_nonfinite")
            elif not event_time_match:
                _add(reasons, "unstable_event_time_mismatch")
    elif label == "stable" and rule_valid:
        trigger_index, trigger_structure_valid = _trigger_end(record20_rule)
        stable_rule = bool(
            n == STABLE_REQUIRED_POINTS
            and record20_rule.get("stable_source_acceptance") is True
            and trigger_structure_valid and trigger_index is None
        )
        rule_label_match = stable_rule
        checks["stable_record_meets_source_acceptance"] = stable_rule
        if not trigger_structure_valid:
            _add(reasons, "record20_trigger_structure_invalid")
        if not stable_rule:
            _add(reasons, "stable_rule_acceptance_mismatch")
        if present["termination_reason"] and term_reason != "stable_horizon_reached":
            _add(reasons, "stable_termination_reason_mismatch")
        if present["event_time_s"]:
            event_text, error = _text(attrs["event_time_s"])
            event_time_match = error is None and event_text == ""
            if error is not None:
                _add(reasons, "stable_event_time_invalid_string")
            elif not event_time_match:
                _add(reasons, "stable_event_time_not_source_empty_string")
    elif label is None:
        checks["label_rule_matches"] = None

    checks["event_time_matches_rule"] = event_time_match if present["event_time_s"] else None
    checks["label_rule_matches"] = rule_label_match
    checkable = rule_valid and clock_usable and label is not None
    return {
        "checkable": checkable,
        "consistent": checkable and not reasons,
        "label": label,
        "record_points": n,
        "checks": checks,
        "reasons": reasons,
    }
