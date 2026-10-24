from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.evaluation.event_rhs_budget import EventRhsBudget, EventRhsBudgetExceeded


def test_vectorized_callbacks_count_event_work_and_callback_calls_separately():
    budget = EventRhsBudget(integration_limit=20, physical_limit=5, total_limit=25,
                            rollout_limit=4, prediction_path_limit=4,
                            training_integration_limit=12, evaluation_integration_limit=8)
    calls = []
    state = np.ones((3, 2))
    with budget.segment({"cohort": "synthetic"}, "history_integration", 8) as meter:
        meter.note_rollout_start(3)
        for _ in range(2):
            value = meter.invoke(lambda x: calls.append(len(x)) or x + 1, state)
            assert value.shape == state.shape
    with budget.segment({"cohort": "synthetic"}, "future_integration", 6) as meter:
        meter.note_rollout_start(1)
        meter.note_prediction_paths(1)
        meter.invoke(lambda x: x * 2, np.ones(1))
    snap = budget.snapshot()
    assert calls == [3, 3]
    assert snap["integration"] == {
        "attempted": 7, "returned": 7, "blocked": 0,
        "callback_calls": 3, "callback_failures": 0,
    }
    assert snap["rollouts_started"] == 4
    assert snap["prediction_paths"] == 1
    assert snap["segments"][0]["successful_rk4_updates"] == 1
    assert snap["segments"][0]["four_stage_returned_groups"] == 1
    assert snap["limits"]["training_integration"] == 12


def test_budget_checks_before_callback_and_failed_segment_does_not_claim_rk4():
    budget = EventRhsBudget(integration_limit=4, physical_limit=0, total_limit=4,
                            rollout_limit=1, prediction_path_limit=0)
    calls = 0

    def rhs(state):
        nonlocal calls
        calls += 1
        return state

    with (pytest.raises(EventRhsBudgetExceeded),
          budget.segment({"event_id": "synthetic"}, "training_integration", 3) as meter):
        meter.invoke(rhs, np.ones((2, 3)))
        meter.invoke(rhs, np.ones((2, 3)))
    assert calls == 1
    row = budget.snapshot()["segments"][0]
    assert row["attempted"] == 2
    assert row["returned"] == 2
    assert row["blocked"] == 2
    assert row["status"] == "failed"
    assert row["successful_rk4_updates"] is None


def test_callback_failure_is_attempted_but_not_returned_or_retried():
    budget = EventRhsBudget(integration_limit=8, physical_limit=0, total_limit=8,
                            rollout_limit=1, prediction_path_limit=0)
    attempts = 0

    def raises(state):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("synthetic callback error")

    with (pytest.raises(RuntimeError, match="synthetic callback error"),
          budget.segment({"event_id": "fail"}, "future_integration", 8) as meter):
        meter.invoke(raises, np.ones(1))
    snap = budget.snapshot()
    row = snap["segments"][0]
    assert attempts == row["callback_calls"] == 1
    assert row["attempted"] == 1 and row["returned"] == 0
    assert row["callback_failures"] == 1
    assert row["successful_rk4_updates"] is None


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_budget_rejects_noninteger_or_negative_limits(bad):
    with pytest.raises(ValueError):
        EventRhsBudget(integration_limit=bad, physical_limit=0, total_limit=0,
                       rollout_limit=0, prediction_path_limit=0)
