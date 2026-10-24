from __future__ import annotations

import json

import pytest

from kron_reduce.data.audit_access_ledger import AccessLedger, AccessLedgerError


def _plan():
    return [
        {"key": "phase1:shape", "phase": 1, "kind": "countdescriptor", "expected_elements": 1},
        {"key": "phase1:data", "phase": 1, "kind": "array", "expected_elements": 3},
        {"key": "phase2:label", "phase": 2, "kind": "attr", "expected_elements": 1},
    ]


def test_call_marks_attempt_before_callback_then_returns_once():
    ledger = AccessLedger(_plan(), logical_element_limit=3)
    seen = []

    def read_once():
        snapshot = ledger.snapshot()
        slot = next(item for item in snapshot["selectors"] if item["key"] == "phase1:data")
        seen.append((slot["status"], slot["sequence"], snapshot["actual_array_elements"]["attempted"]))
        return [1, 2, 3]

    result = ledger.call("phase1:data", read_once, element_count=len)
    assert result == [1, 2, 3]
    assert seen == [("attempted", 1, 3)]
    with pytest.raises(AccessLedgerError, match="already spent"):
        ledger.call("phase1:data", read_once, element_count=len)
    assert len(seen) == 1
    snapshot = ledger.snapshot()
    assert snapshot["actual_counts"]["array"] == {"attempted": 1, "returned": 1, "failed": 0}
    assert snapshot["slot_status_counts"]["not_attempted"] == 2
    assert len(json.dumps(snapshot, allow_nan=False)) > 0


def test_source_exception_is_rethrown_and_slot_is_spent_failed():
    ledger = AccessLedger(_plan(), logical_element_limit=3)

    def source_failure():
        raise OSError("source read failed")

    with pytest.raises(OSError, match="source read failed"):
        ledger.call("phase1:data", source_failure)
    with pytest.raises(AccessLedgerError, match="already spent"):
        ledger.call("phase1:data", lambda: [1, 2, 3])
    slot = next(item for item in ledger.snapshot()["selectors"] if item["key"] == "phase1:data")
    assert slot["status"] == "failed"
    assert slot["sequence"] == 1
    assert slot["exception_type"] == "OSError"
    assert slot["exception_message"] == "source read failed"
    assert ledger.snapshot()["actual_array_elements"] == {"attempted": 3, "returned": 0, "failed": 3}


def test_source_event_failure_does_not_poison_other_registered_reads():
    ledger = AccessLedger(_plan(), logical_element_limit=3)

    def event_read_failure():
        raise OSError("one event selection failed")

    with pytest.raises(OSError, match="one event selection failed"):
        ledger.call("phase1:data", event_read_failure)
    assert ledger.snapshot()["poisoned"] is False
    assert ledger.call("phase1:shape", lambda: {"shape": [3]}) == {"shape": [3]}
    snapshot = ledger.snapshot()
    assert snapshot["slot_status_counts"]["failed"] == 1
    assert snapshot["slot_status_counts"]["returned"] == 1
    assert snapshot["poisoned"] is False


def test_phase1_failure_and_explicit_skips_are_terminal_but_preserved_for_phase2():
    plan = [
        {"key": "phase1:bad_descriptor", "phase": 1, "kind": "countdescriptor", "expected_elements": 1},
        {"key": "phase1:event_delta", "phase": 1, "kind": "array", "expected_elements": 3},
        {"key": "phase1:event_omega", "phase": 1, "kind": "array", "expected_elements": 3},
        {"key": "phase1:static_omega_s", "phase": 1, "kind": "attr", "expected_elements": 1},
        {"key": "phase2:event_label", "phase": 2, "kind": "attr", "expected_elements": 1},
    ]
    ledger = AccessLedger(plan, logical_element_limit=6)
    with pytest.raises(OSError, match="descriptor unavailable"):
        ledger.call("phase1:bad_descriptor", lambda: (_ for _ in ()).throw(OSError("descriptor unavailable")))
    ledger.skip("phase1:event_delta", "dataset_missing", prerequisite_key="phase1:bad_descriptor")
    ledger.skip("phase1:event_omega", "dataset_missing", prerequisite_key="phase1:bad_descriptor")
    assert ledger.call("phase1:static_omega_s", lambda: 377.0) == 377.0
    ledger.seal_prelabel_freeze("a" * 64)
    assert ledger.call("phase2:event_label", lambda: "unstable") == "unstable"
    snapshot = ledger.snapshot()
    assert snapshot["phase"] == 2
    assert snapshot["slot_status_counts"] == {
        "not_attempted": 0, "skipped": 2, "attempted": 0, "returned": 2, "failed": 1,
    }
    slots = {slot["key"]: slot for slot in snapshot["selectors"]}
    assert slots["phase1:bad_descriptor"]["status"] == "failed"
    assert slots["phase1:event_delta"]["status"] == "not_attempted"
    assert slots["phase1:event_delta"]["skip_reason"] == "dataset_missing"
    assert slots["phase1:event_delta"]["prerequisite_key"] == "phase1:bad_descriptor"
    assert slots["phase1:event_omega"]["status"] == "not_attempted"


def test_explicitly_skipped_key_cannot_later_be_read():
    plan = [
        {"key": "phase1:descriptor", "phase": 1, "kind": "countdescriptor", "expected_elements": 1},
        {"key": "phase1:array", "phase": 1, "kind": "array", "expected_elements": 2},
    ]
    ledger = AccessLedger(plan, logical_element_limit=2)
    with pytest.raises(OSError):
        ledger.call("phase1:descriptor", lambda: (_ for _ in ()).throw(OSError("missing")))
    ledger.skip("phase1:array", "dataset_missing", prerequisite_key="phase1:descriptor")
    called = []
    with pytest.raises(AccessLedgerError, match="already spent"):
        ledger.call("phase1:array", lambda: called.append("read"))
    assert called == []
    assert ledger.snapshot()["poisoned"] is True


def test_phase2_is_denied_before_freeze_and_denial_poisons_all_later_calls():
    ledger = AccessLedger(_plan(), logical_element_limit=3)
    calls = []
    with pytest.raises(AccessLedgerError, match="belongs to phase 2"):
        ledger.call("phase2:label", lambda: calls.append("read"))
    assert calls == []
    assert ledger.snapshot()["poisoned"] is True
    with pytest.raises(AccessLedgerError, match="poisoned"):
        ledger.call("phase1:shape", lambda: calls.append("shape"))
    with pytest.raises(AccessLedgerError, match="poisoned"):
        ledger.seal_prelabel_freeze("a" * 64)
    assert ledger.snapshot()["phase"] == 1


def test_unresolved_phase1_slot_blocks_freeze_and_poisons_stage_transition():
    ledger = AccessLedger(_plan(), logical_element_limit=3)
    ledger.call("phase1:shape", lambda: {"shape": [3]})
    with pytest.raises(AccessLedgerError, match="terminal"):
        ledger.seal_prelabel_freeze("a" * 64)
    assert ledger.snapshot()["phase"] == 1
    assert ledger.snapshot()["poisoned"] is True
    with pytest.raises(AccessLedgerError, match="poisoned"):
        ledger.call("phase1:data", lambda: [1, 2, 3])


def test_invalid_descriptor_can_justify_skip_only_after_returned_validation():
    plan = [
        {"key": "phase1:shape", "phase": 1, "kind": "countdescriptor", "expected_elements": 1},
        {"key": "phase1:data", "phase": 1, "kind": "array", "expected_elements": 4},
        {"key": "phase2:attr", "phase": 2, "kind": "attr", "expected_elements": 1},
    ]
    ledger = AccessLedger(plan, logical_element_limit=4)
    ledger.call("phase1:shape", lambda: {"shape": [2]})
    ledger.declare_invalid_descriptor("phase1:shape", "declared_length_mismatch")
    ledger.skip("phase1:data", "shape_mismatch", prerequisite_key="phase1:shape")
    ledger.seal_prelabel_freeze("b" * 64)
    assert ledger.call("phase2:attr", lambda: "ok") == "ok"
    slot = next(s for s in ledger.snapshot()["selectors"] if s["key"] == "phase1:shape")
    assert slot["status"] == "returned"
    assert slot["validation_status"] == "invalid"
    assert slot["validation_reason"] == "declared_length_mismatch"


def test_invalid_freeze_sha_poisons_ledger():
    ledger = AccessLedger(_plan(), logical_element_limit=3)
    ledger.call("phase1:shape", lambda: {"shape": [3]})
    ledger.call("phase1:data", lambda: [1, 2, 3], element_count=len)
    with pytest.raises(AccessLedgerError, match="64 hexadecimal"):
        ledger.seal_prelabel_freeze("not-a-sha")
    assert ledger.snapshot()["phase"] == 1
    assert ledger.snapshot()["poisoned"] is True


def test_unknown_key_is_denied_before_callback_and_does_not_spend_slot():
    ledger = AccessLedger(_plan(), logical_element_limit=3)
    calls = []
    with pytest.raises(AccessLedgerError, match="unknown selector"):
        ledger.call("future:Pm[10]", lambda: calls.append("forbidden"))
    assert calls == []
    snapshot = ledger.snapshot()
    assert snapshot["sequence_count"] == 0
    assert snapshot["slot_status_counts"]["not_attempted"] == 3
    assert snapshot["denied_attempts"] == [{
        "key": "future:Pm[10]", "phase": 1, "reason": "unknown selector",
    }]


def test_phase1_registered_static_attr_is_allowed():
    plan = [
        {"key": "phase1:done", "phase": 1, "kind": "countdescriptor", "expected_elements": 1},
        {"key": "phase1:attr", "phase": 1, "kind": "attr", "expected_elements": 1},
    ]
    ledger = AccessLedger(plan, logical_element_limit=0)
    ledger.call("phase1:done", lambda: None)
    assert ledger.call("phase1:attr", lambda: "omega_s") == "omega_s"
    assert next(s for s in ledger.snapshot()["selectors"] if s["key"] == "phase1:attr")["status"] == "returned"


def test_returned_element_mismatch_is_a_failed_nonretryable_slot():
    ledger = AccessLedger(_plan(), logical_element_limit=3)
    with pytest.raises(AccessLedgerError, match="returned 2 elements; expected 3"):
        ledger.call("phase1:data", lambda: [1, 2], element_count=len)
    slot = next(s for s in ledger.snapshot()["selectors"] if s["key"] == "phase1:data")
    assert slot["status"] == "failed"
    assert slot["source_returned"] is True
    assert slot["returned_elements"] == 2
    assert ledger.snapshot()["actual_array_elements"] == {"attempted": 3, "returned": 2, "failed": 0}
    assert ledger.snapshot()["poisoned"] is True
    calls = []
    with pytest.raises(AccessLedgerError, match="poisoned"):
        ledger.call("phase1:shape", lambda: calls.append("shape"))
    assert calls == []
    with pytest.raises(AccessLedgerError, match="poisoned"):
        ledger.call("phase1:data", lambda: [1, 2, 3])


def test_element_counter_failure_preserves_source_return_and_poisons_ledger():
    ledger = AccessLedger(_plan(), logical_element_limit=3)
    source_returns = []

    def source_read():
        source_returns.append("returned")
        return [1, 2, 3]

    def bad_counter(_value):
        raise ValueError("cannot inspect returned selection")

    with pytest.raises(ValueError, match="cannot inspect returned selection"):
        ledger.call("phase1:data", source_read, element_count=bad_counter)
    assert source_returns == ["returned"]
    snapshot = ledger.snapshot()
    slot = next(item for item in snapshot["selectors"] if item["key"] == "phase1:data")
    assert slot["status"] == "failed"
    assert slot["source_returned"] is True
    assert slot["returned_elements"] is None
    assert slot["exception_type"] == "ValueError"
    assert slot["failure_reason"] == "element_count_failed"
    assert snapshot["actual_counts"]["array"] == {"attempted": 1, "returned": 1, "failed": 1}
    assert snapshot["unmeasured_returned_array_expected_elements"] == 3
    assert snapshot["poisoned"] is True
    calls = []
    with pytest.raises(AccessLedgerError, match="poisoned"):
        ledger.call("phase1:shape", lambda: calls.append("shape"))
    assert calls == []


def test_planned_kind_and_logical_element_caps_are_independent():
    with pytest.raises(ValueError, match="descriptor count exceeds"):
        AccessLedger([
            {"key": f"d{i}", "phase": 1, "kind": "countdescriptor", "expected_elements": 1}
            for i in range(1079)
        ], logical_element_limit=0)
    with pytest.raises(ValueError, match="array selections exceed"):
        AccessLedger([
            {"key": f"a{i}", "phase": 1, "kind": "array", "expected_elements": 0}
            for i in range(1079)
        ], logical_element_limit=0)
    with pytest.raises(ValueError, match="attribute reads exceed"):
        AccessLedger([
            {"key": f"x{i}", "phase": 2, "kind": "attr", "expected_elements": 1}
            for i in range(2967)
        ], logical_element_limit=0)
    with pytest.raises(ValueError, match="logical_element_limit"):
        AccessLedger([
            {"key": "a", "phase": 1, "kind": "array", "expected_elements": 2},
            {"key": "b", "phase": 1, "kind": "array", "expected_elements": 2},
        ], logical_element_limit=3)


def test_duplicate_and_malformed_plan_keys_are_rejected():
    one = {"key": "same", "phase": 1, "kind": "countdescriptor", "expected_elements": 1}
    with pytest.raises(ValueError, match="duplicate"):
        AccessLedger([one, dict(one)], logical_element_limit=0)
    with pytest.raises(ValueError, match="exactly key/phase/kind/expected_elements"):
        AccessLedger([{**one, "selector": "extra"}], logical_element_limit=0)


def test_unattempted_slots_remain_explicit_after_partial_failure():
    ledger = AccessLedger(_plan(), logical_element_limit=3)
    with pytest.raises(OSError):
        ledger.call("phase1:shape", lambda: (_ for _ in ()).throw(OSError("descriptor failed")))
    snapshot = ledger.snapshot()
    statuses = {slot["key"]: slot["status"] for slot in snapshot["selectors"]}
    assert statuses == {
        "phase1:shape": "failed",
        "phase1:data": "not_attempted",
        "phase2:label": "not_attempted",
    }
    assert snapshot["actual_counts"]["attr"]["attempted"] == 0


