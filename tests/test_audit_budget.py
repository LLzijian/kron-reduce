from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation.audit_budget import AuditBudgetExceeded, AuditRhsBudget


def test_segment_budget_blocks_before_callback_and_records_counts():
    budget = AuditRhsBudget(integration_limit=20, physical_limit=20, total_limit=20)
    calls = []
    with pytest.raises(AuditBudgetExceeded, match="segment RHS budget"), budget.segment(
        {"event": "e1"}, "history_integration", limit=1,
    ):
        assert budget.invoke(lambda: calls.append("called") or 5) == 5
        budget.invoke(lambda: calls.append("should-not-run"))
    assert calls == ["called"]
    row = budget.records[0]
    assert (row["attempted"], row["returned"], row["blocked"]) == (1, 1, 1)
    assert row["status"] == "failed"
    assert row["successful_rk4_updates"] is None
    assert budget.snapshot()["blocked"] == 1


def test_phase_and_global_caps_block_callbacks_before_execution():
    budget = AuditRhsBudget(integration_limit=1, physical_limit=5, total_limit=5)
    with budget.segment({}, "history_integration", 4):
        budget.invoke(lambda: "first")
    phase_calls = []
    with pytest.raises(AuditBudgetExceeded, match="phase RHS budget"), budget.segment(
        {}, "future_integration", 4,
    ):
        budget.invoke(lambda: phase_calls.append(1))
    assert phase_calls == []

    total = AuditRhsBudget(integration_limit=5, physical_limit=5, total_limit=1)
    with total.segment({}, "history_integration", 4):
        total.invoke(lambda: "first")
    global_calls = []
    with pytest.raises(AuditBudgetExceeded, match="total RHS budget"), total.segment(
        {}, "physical_checks", 4,
    ):
        total.invoke(lambda: global_calls.append(1))
    assert global_calls == []
    snap = total.snapshot()
    assert snap["attempted"] == snap["returned"] == 1
    assert snap["blocked"] == 1
    assert snap["physical_rhs"]["blocked"] == 1


@pytest.mark.parametrize("failure_call", [2, 4])
def test_callback_exception_counts_only_successful_returns(failure_call):
    budget = AuditRhsBudget()
    calls = 0

    def callback():
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise LookupError("synthetic callback failure")
        return calls

    with pytest.raises(LookupError, match="synthetic callback failure"), budget.segment(
        {"case": failure_call}, "future_integration", 10,
    ):
        for _ in range(5):
            budget.invoke(callback)
    row = budget.records[0]
    assert row["status"] == "failed"
    assert row["exception_type"] == "LookupError"
    assert row["attempted"] == failure_call
    assert row["returned"] == failure_call - 1
    assert row["successful_rk4_updates"] is None
    assert budget._active is None


def test_failed_segment_remainders_are_never_combined():
    budget = AuditRhsBudget()
    for event in ("first", "second"):
        with pytest.raises(RuntimeError, match="incomplete synthetic rollout"), budget.segment(
            {"event": event}, "history_integration", 10,
        ):
            for _ in range(3):
                budget.invoke(lambda: None)
            raise RuntimeError("incomplete synthetic rollout")
    assert [r["returned"] for r in budget.records] == [3, 3]
    assert [r["four_stage_returned_groups"] for r in budget.records] == [0, 0]
    assert [r["partial_returned_rhs"] for r in budget.records] == [3, 3]
    assert [r["successful_rk4_updates"] for r in budget.records] == [None, None]


def test_four_returns_then_rollout_error_does_not_claim_successful_update():
    budget = AuditRhsBudget()
    with pytest.raises(FloatingPointError, match="synthetic COI divergence"), budget.segment(
        {}, "future_integration", 10,
    ):
        for _ in range(4):
            budget.invoke(lambda: 0.0)
        raise FloatingPointError("synthetic COI divergence")
    row = budget.records[0]
    assert row["returned"] == 4
    assert row["four_stage_returned_groups"] == 1
    assert row["partial_returned_rhs"] == 0
    assert row["status"] == "failed"
    assert row["successful_rk4_updates"] is None


def test_successful_integration_segment_and_physical_segment_account_separately():
    budget = AuditRhsBudget()
    with budget.segment({"event": "ok"}, "history_integration", 4):
        values = [budget.invoke(lambda value=i: value) for i in range(4)]
    with budget.segment({"event": "physics"}, "physical_checks", 2):
        budget.invoke(lambda: "physical")
    assert values == [0, 1, 2, 3]
    integration, physical = budget.records
    assert integration["successful_rk4_updates"] == 1
    assert integration["status"] == "complete"
    assert physical["successful_rk4_updates"] is None
    assert physical["returned"] == 1
    assert physical["four_stage_returned_groups"] is None
    assert physical["partial_returned_rhs"] is None
    snapshot = budget.snapshot()
    assert snapshot["integration_rhs"]["returned"] == 4
    assert snapshot["physical_rhs"]["returned"] == 1


def test_result_returned_by_invoke_is_byteexact():
    budget = AuditRhsBudget()
    expected = np.array([[0.0, -1.25], [np.pi, 4.5]], dtype="<f8")
    with budget.segment({}, "physical_checks", 1):
        actual = budget.invoke(lambda: expected)
    assert actual is expected
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_nested_segment_no_active_invoke_and_finally_cleanup():
    budget = AuditRhsBudget()
    with pytest.raises(RuntimeError, match="active segment"):
        budget.invoke(lambda: None)
    with budget.segment({"outer": True}, "physical_checks", 2):
        with pytest.raises(RuntimeError, match="may not be nested"), budget.segment(
            {"nested": True}, "physical_checks", 1,
        ):
            pass
        budget.invoke(lambda: 1)
    assert len(budget.records) == 1
    assert budget._active is None
    with pytest.raises(RuntimeError, match="active segment"):
        budget.invoke(lambda: None)


def test_rollout_start_counts_only_after_explicit_successful_start_note():
    budget = AuditRhsBudget(rollout_limit=1)
    with pytest.raises(ValueError, match="synthetic initialization failure"), budget.segment(
        {"event": "init-fails"}, "history_integration", 8,
    ):
        raise ValueError("synthetic initialization failure")
    assert budget.records[0]["rollout_calls_started"] == 0

    with budget.segment({"event": "rollout"}, "future_integration", 8):
        budget.note_rollout_start()
        with pytest.raises(RuntimeError, match="at most one rollout"):
            budget.note_rollout_start()
    assert budget.records[1]["rollout_calls_started"] == 1
    assert budget.records[1]["rollout_calls_blocked"] == 0

    with pytest.raises(AuditBudgetExceeded, match="rollout budget"), budget.segment(
        {"event": "budgeted"}, "history_integration", 8,
    ):
        budget.note_rollout_start()
    assert budget.records[2]["rollout_calls_started"] == 0
    assert budget.records[2]["rollout_calls_blocked"] == 1
    assert budget.snapshot()["rollout_calls_started"] == 1
    assert budget.snapshot()["rollout_calls_blocked"] == 1


def test_rollout_start_rejects_no_segment_or_physical_phase_without_counting():
    budget = AuditRhsBudget()
    with pytest.raises(RuntimeError, match="active integration segment"):
        budget.note_rollout_start()
    with budget.segment({}, "physical_checks", 1), pytest.raises(RuntimeError, match="physical checks"):
        budget.note_rollout_start()
    assert budget.snapshot()["rollout_calls_started"] == 0
    assert budget.snapshot()["rollout_calls_blocked"] == 0


@pytest.mark.parametrize("phase,limit", [("unknown", 1), ("physical_checks", -1)])
def test_invalid_phase_or_limit_is_rejected(phase, limit):
    budget = AuditRhsBudget()
    with pytest.raises((ValueError, RuntimeError)), budget.segment({}, phase, limit):
        pass


@pytest.mark.parametrize("kwargs", [
    {"integration_limit": -1}, {"physical_limit": 1.5}, {"total_limit": True},
])
def test_constructor_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        AuditRhsBudget(**kwargs)
