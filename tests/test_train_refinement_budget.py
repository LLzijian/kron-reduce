from __future__ import annotations

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.evaluation.audit_budget import AuditBudgetExceeded, AuditRhsBudget
from kron_reduce.models.nonlinear_latent import (
    LearnedNonlinearModel,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
)
from kron_reduce.physics.coordinates import coi_project
from kron_reduce.workflows.kiki_train_mechanism import (
    counted_model,
    integration_substeps,
)

EVENTS = 258
SEEDS = 3
TOTAL_FUTURE_POINTS = 72_147
EXPECTED_RK4_UPDATES = 896_724
EXPECTED_INTEGRATION_RHS = 3_586_896
EXPECTED_PHYSICAL_RHS = 225_729
EXPECTED_TOTAL_RHS = 3_812_625
EXPECTED_ROLLOUTS = 1_548


def _synthetic_model_and_history():
    rng = np.random.default_rng(20261024)
    mass = np.linspace(1.0, 3.7, 10)
    incidence = np.zeros((10, 2))
    incidence[[0, 1], 0] = (1.0, -1.0)
    incidence[[3, 8], 1] = (0.7, -0.7)
    coupling = incidence @ np.diag([0.13, 0.08])
    projection = np.eye(10) - np.ones((10, 10)) / 10
    closure = NonlinearPassiveClosure(
        mass=mass,
        edge_stiffness=np.linspace(0.01, 0.05, 45),
        damping=0.04 * projection,
        coupling=coupling,
        latent_skew=np.array([[0.0, 0.03], [-0.03, 0.0]]),
        latent_dissipation=np.diag([0.08, 0.11]),
    )
    power = np.linspace(-0.2, 0.2, 10)
    power -= np.mean(power)
    equilibrium = coi_project(np.linspace(-0.15, 0.12, 10), mass)
    delta = np.stack([coi_project(rng.normal(0.0, 0.01, 10), mass) for _ in range(11)])
    omega = np.stack([coi_project(rng.normal(0.0, 0.004, 10), mass) for _ in range(11)])
    history = CausalStateHistory(
        time=np.arange(11, dtype=float) * 0.01 - 0.1,
        delta=delta,
        omega=omega,
        constant_power=power,
    )
    model = LearnedNonlinearModel(
        closure=closure,
        encoder=rng.normal(0.0, 0.02, size=(2, 80)),
        feature_scale=np.ones(80),
        target_scale=np.linspace(0.8, 1.2, 20),
        settings=NonlinearFitSettings(latent_dimension=2, frames=3),
        fit_record={"synthetic_only": True},
    )
    return model, history, equilibrium


def _assert_byteexact(left, right):
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def _a10_synthetic_counts():
    # An exact, synthetic 258-event population whose capped horizon counts sum
    # to the frozen A10 total; no data files or event metadata are read.
    counts = [280] * 165 + [279] * (EVENTS - 165)
    assert len(counts) == EVENTS
    assert sum(counts) == TOTAL_FUTURE_POINTS
    assert all(1 <= count <= 300 for count in counts)
    return counts


def test_a10_0025_original_clock_rule_and_fixed_total_budget_arithmetic():
    history_clock = np.arange(11, dtype=float) * 0.01 - 0.1
    assert integration_substeps(history_clock, 0.0025) == 40
    counts = _a10_synthetic_counts()
    future_updates_per_seed = sum(
        integration_substeps(np.arange(count + 1, dtype=float) * 0.01, 0.0025)
        for count in counts
    )
    assert future_updates_per_seed == 4 * TOTAL_FUTURE_POINTS

    rollouts = SEEDS * EVENTS * 2
    history_updates = SEEDS * EVENTS * 40
    future_updates = SEEDS * future_updates_per_seed
    rk4_updates = history_updates + future_updates
    integration_rhs = 4 * rk4_updates
    physical_rhs = SEEDS * (TOTAL_FUTURE_POINTS + EVENTS * 12)
    assert rollouts == EXPECTED_ROLLOUTS
    assert rk4_updates == EXPECTED_RK4_UPDATES
    assert integration_rhs == EXPECTED_INTEGRATION_RHS
    assert physical_rhs == EXPECTED_PHYSICAL_RHS
    assert integration_rhs + physical_rhs == EXPECTED_TOTAL_RHS

    budget = AuditRhsBudget(
        integration_limit=EXPECTED_INTEGRATION_RHS,
        physical_limit=EXPECTED_PHYSICAL_RHS,
        total_limit=EXPECTED_TOTAL_RHS,
        rollout_limit=EXPECTED_ROLLOUTS,
    )
    assert budget.snapshot()["limits"] == {
        "integration": EXPECTED_INTEGRATION_RHS,
        "physical": EXPECTED_PHYSICAL_RHS,
        "total": EXPECTED_TOTAL_RHS,
        "rollout": EXPECTED_ROLLOUTS,
    }


def test_counted_model_keeps_original_path_at_a10_step_for_one_59_and_300_points():
    for future_count in (1, 59, 300):
        reference, history, equilibrium = _synthetic_model_and_history()
        reference_z, reference_history = reference.initialize(history, equilibrium, max_step=0.0025)
        clock = np.arange(future_count + 1, dtype=float) * 0.01
        reference_future = reference.closure.rollout(
            history.delta[-1], history.omega[-1], reference_z, clock,
            history.constant_power, equilibrium, max_step=0.0025,
        )

        counted, counted_history, counted_equilibrium = _synthetic_model_and_history()
        budget = AuditRhsBudget(integration_limit=100_000, physical_limit=10_000,
                                total_limit=110_000, rollout_limit=2)
        counted_model(counted, budget)
        history_updates = integration_substeps(counted_history.time, 0.0025)
        assert history_updates == 40
        with budget.segment({"synthetic": "history"}, "history_integration", 4 * history_updates):
            counted_z, counted_history_path = counted.initialize(
                counted_history, counted_equilibrium, max_step=0.0025)

        future_updates = integration_substeps(clock, 0.0025)
        assert future_updates == 4 * future_count
        with budget.segment({"synthetic": f"future-{future_count}"},
                            "future_integration", 4 * future_updates):
            counted_future = counted.closure.rollout(
                counted_history.delta[-1], counted_history.omega[-1], counted_z, clock,
                counted_history.constant_power, counted_equilibrium, max_step=0.0025,
            )

        _assert_byteexact(counted_z, reference_z)
        _assert_byteexact(counted_history_path, reference_history)
        _assert_byteexact(counted_future, reference_future)
        assert [(r["attempted"], r["returned"], r["rollout_calls_started"],
                 r["successful_rk4_updates"]) for r in budget.records] == [
            (4 * history_updates, 4 * history_updates, 1, history_updates),
            (4 * future_updates, 4 * future_updates, 1, future_updates),
        ]


def test_budget_overflow_blocks_before_callback_and_does_not_retry_slot():
    budget = AuditRhsBudget(integration_limit=4, physical_limit=0, total_limit=4, rollout_limit=1)
    calls = []
    with budget.segment({"slot": "reserved-once"}, "history_integration", limit=4):
        budget.note_rollout_start()
        for _ in range(4):
            budget.invoke(lambda: calls.append("rhs") or None)
    with budget.segment({"slot": "reserved-once"}, "future_integration", limit=4):
        with pytest.raises(AuditBudgetExceeded):
            budget.note_rollout_start()
        with pytest.raises(AuditBudgetExceeded):
            budget.invoke(lambda: calls.append("must-not-run"))
    assert calls == ["rhs"] * 4
    assert [row["rollout_calls_started"] for row in budget.records] == [1, 0]
    assert [row["rollout_calls_blocked"] for row in budget.records] == [0, 1]
    assert [row["attempted"] for row in budget.records] == [4, 0]
    assert [row["returned"] for row in budget.records] == [4, 0]
    assert [row["blocked"] for row in budget.records] == [0, 1]
    assert budget.snapshot()["rollout_calls_started"] == 1


def test_failed_a10_segment_never_claims_rk4_update_or_retries():
    budget = AuditRhsBudget()
    calls = 0

    def rhs():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise FloatingPointError("synthetic integration failure")
        return calls

    with pytest.raises(FloatingPointError, match="synthetic integration failure"), budget.segment(
        {"slot": "failed-once"}, "future_integration", limit=20,
    ):
        for _ in range(4):
            budget.invoke(rhs)
    record = budget.records[0]
    assert (record["attempted"], record["returned"], record["blocked"]) == (4, 3, 0)
    assert record["status"] == "failed"
    assert record["successful_rk4_updates"] is None
    assert budget._active is None


