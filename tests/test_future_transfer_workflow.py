from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.future_transfer_reader import fixed_file_specs
from kron_reduce.evaluation.audit_budget import AuditBudgetExceeded, AuditRhsBudget
from kron_reduce.evaluation.scoped_audit_journal import ScopedAuditJournal
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.workflows.kiki_future_transfer import audited_future, counted_structure
from project_harness import runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _closure():
    n = 10
    projection = np.eye(n) - np.ones((n, n)) / n
    coupling = np.zeros((n, 2), dtype=np.float64)
    coupling[[0, 1], 0] = (0.1, -0.1)
    coupling[[3, 4], 1] = (0.07, -0.07)
    return NonlinearPassiveClosure(
        mass=np.linspace(1.0, 2.0, n),
        edge_stiffness=np.full(45, 0.01),
        damping=0.04 * projection,
        coupling=coupling,
        latent_skew=np.array([[0.0, 0.02], [-0.02, 0.0]]),
        latent_dissipation=np.diag([0.05, 0.08]),
    )


def _identity():
    return {"seed": 324, "event_id": "synthetic/event-000", "step": 0.01}


def _construction_journal(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "scopes")
    counters = {"model_constructions": 0, "eigvalsh_attempted": 0, "eigvalsh_returned": 0}
    journal.begin("construction-324", {"model_constructions": 1, "eigvalsh": 3})
    journal.append({"category": "closure_construction", "seed": 324, "transition": "attempted"})
    counters["model_constructions"] += 1
    with counted_structure(journal, counters):
        closure = _closure()
    journal.append({"category": "closure_construction", "seed": 324, "transition": "returned"})
    journal.finish({"status": "complete"})
    return closure, journal, counters


def test_fixed_reader_order_matches_frozen_metadata_plan_without_payload_access():
    plan = json.loads(
        (PROJECT_ROOT / "docs/research/kiki_a15_future_transfer/explicit_read_plan.json").read_text(
            encoding="utf-8"
        )
    )
    expected = [row["file_id"] for row in plan["npz_files"]]
    actual = [spec.key for spec in fixed_file_specs()]
    assert len(expected) == 1559
    assert actual == expected
    assert [spec.key for spec in fixed_file_specs() if spec.phase == "truth"] == [expected[-1]]


def test_bound_registration_metadata_matches_reader_without_opening_source_arrays():
    from kron_reduce.workflows.kiki_future_transfer import bound_metadata
    config = tomllib.loads((PROJECT_ROOT / "configs/experiments/paper_kiki_a15_future_transfer.toml").read_text(encoding="utf-8"))
    plan, inventory, indexed = bound_metadata(PROJECT_ROOT, config)
    assert len(indexed) == 1559
    assert inventory["totals"]["numeric_scalars"] == 11213485
    assert sum(plan["population"]["future_point_counts"]) == 72147


def test_synthetic_closure_eig_budget_and_audited_rollout_call_counts(tmp_path):
    closure, journal, counters = _construction_journal(tmp_path)
    assert counters == {"model_constructions": 1, "eigvalsh_attempted": 3, "eigvalsh_returned": 3}

    budget = AuditRhsBudget(integration_limit=8, physical_limit=3, total_limit=11, rollout_limit=1)
    saved = []
    identity = _identity()
    reservation = {"rollouts": 1, "integration_rhs": 8, "physical_rhs": 3, "eigvalsh": 3}
    journal.begin("future-324-event-000", reservation)
    with counted_structure(journal, counters):
        path, physical = audited_future(
            closure,
            np.zeros(22, dtype=np.float64),
            np.array([0.0, 0.01, 0.02]),
            np.zeros(10),
            np.zeros(10),
            0.01,
            budget,
            journal,
            identity,
            lambda value: saved.append(value.copy()),
        )
    journal.finish({"status": "complete"})
    assert path.shape == (3, 22) and np.array_equal(path, np.zeros_like(path))
    assert len(saved) == 1 and np.array_equal(saved[0], path)
    assert physical["qualified"] is True and len(physical["points"]) == 3
    assert budget.snapshot()["phases"]["future_integration"] == {
        "attempted": 8,
        "returned": 8,
        "blocked": 0,
    }
    assert budget.snapshot()["phases"]["physical_checks"] == {
        "attempted": 3,
        "returned": 3,
        "blocked": 0,
    }
    assert budget.snapshot()["rollout_calls_started"] == 1
    assert counters["eigvalsh_attempted"] == counters["eigvalsh_returned"] == 6
    journal.close()


def test_complete_three_second_pair_matches_frozen_original_rk4_arithmetic(tmp_path):
    closure, journal, counters = _construction_journal(tmp_path)
    budget = AuditRhsBudget(integration_limit=7200, physical_limit=602, total_limit=7802, rollout_limit=2)
    journal.begin("full-three-second-pair", {
        "rollouts": 2, "integration_rhs": 7200, "physical_rhs": 602, "eigvalsh": 6,
    })
    saved = []
    for step in (.005, .0025):
        with counted_structure(journal, counters):
            path, physical = audited_future(
                closure, np.zeros(22), np.arange(301) * .01, np.zeros(10), np.zeros(10),
                step, budget, journal, {**_identity(), "step": step},
                lambda value: saved.append(value.copy()),
            )
        assert path.shape == (301, 22) and physical["qualified"]
    receipt = journal.finish({"status": "complete"})
    journal.close()
    assert receipt["actual_counters"]["returned"] == {
        "rollouts": 2, "eigvalsh": 6, "integration_rhs": 7200, "physical_rhs": 602,
    }
    assert sum(r["successful_rk4_updates"] or 0 for r in budget.records) == 1800
    assert counters["eigvalsh_attempted"] == counters["eigvalsh_returned"] == 9
    assert len(saved) == 2 and np.array_equal(saved[0], saved[1])


@pytest.mark.parametrize("failure_kind", ["exception", "nan"])
@pytest.mark.parametrize("failure_point", [1, 2, 3])
def test_path_is_saved_before_physical_failure_and_slot_is_not_retried(tmp_path, failure_kind, failure_point):
    closure, journal, _ = _construction_journal(tmp_path)
    original = closure.energy_identity
    calls = []

    def bad_physics(*args):
        calls.append(1)
        if len(calls) < failure_point:
            return original(*args)
        if failure_kind == "exception":
            raise ArithmeticError("synthetic physical failure")
        return {"energy_rate": 1.0, "supply_minus_dissipation": 1.0, "identity_error": float("nan")}

    object.__setattr__(closure, "energy_identity", bad_physics)
    budget = AuditRhsBudget(integration_limit=8, physical_limit=3, total_limit=11, rollout_limit=1)
    saved = []
    journal.begin(
        "failed-future-324-event-000",
        {
            "rollouts": 1,
            "integration_rhs": 8,
            "physical_rhs": 3,
        },
    )
    with pytest.raises((ArithmeticError, FloatingPointError)) as caught:
        audited_future(
            closure,
            np.zeros(22),
            np.array([0.0, 0.01, 0.02]),
            np.zeros(10),
            np.zeros(10),
            0.01,
            budget,
            journal,
            _identity(),
            lambda value: saved.append(value.copy()),
        )
    journal.finish({"status": "failed", "automatic_retry": False})
    assert len(saved) == 1  # completed path persisted before physical checks began
    assert budget.snapshot()["rollout_calls_started"] == 1
    assert len(calls) == failure_point
    points = caught.value.partial_record["physical_points"]
    assert len(points) == (failure_point if failure_kind == "nan" else failure_point - 1)
    assert all(row["identity_error"] == 0 for row in points[:failure_point - 1])
    assert budget.records[-1]["phase"] == "physical_checks"
    assert budget.records[-1]["status"] == "failed"
    assert budget.records[-1]["successful_rk4_updates"] is None
    object.__setattr__(closure, "energy_identity", original)
    journal.close()


def test_global_rhs_budget_blocks_before_original_rhs_and_rollout_is_single_use(tmp_path):
    closure, journal, _ = _construction_journal(tmp_path)
    calls = []
    original_rhs = closure.rhs
    object.__setattr__(closure, "rhs", lambda *args: (calls.append(1), original_rhs(*args))[1])
    budget = AuditRhsBudget(integration_limit=8, physical_limit=3, total_limit=0, rollout_limit=1)
    journal.begin(
        "blocked-future-324-event-000",
        {
            "rollouts": 1,
            "integration_rhs": 8,
            "physical_rhs": 3,
        },
    )
    with pytest.raises(AuditBudgetExceeded, match="total"):
        audited_future(
            closure,
            np.zeros(22),
            np.array([0.0, 0.01, 0.02]),
            np.zeros(10),
            np.zeros(10),
            0.01,
            budget,
            journal,
            _identity(),
            lambda _value: None,
        )
    journal.finish({"status": "failed", "automatic_retry": False})
    snapshot = budget.snapshot()
    assert calls == []
    assert snapshot["attempted"] == snapshot["returned"] == 0
    assert snapshot["blocked"] == 1 and snapshot["rollout_calls_started"] == 1
    assert len(budget.records) == 1 and budget.records[0]["status"] == "failed"
    object.__setattr__(closure, "rhs", original_rhs)
    journal.close()


def test_nonfinite_eigvalsh_is_charged_as_returned_then_rejected(tmp_path, monkeypatch):
    journal = ScopedAuditJournal(tmp_path / "nonfinite-eigenvalue")
    journal.begin("nonfinite-eig", {"eigvalsh": 1})
    counters = {"eigvalsh_attempted": 0, "eigvalsh_returned": 0}
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda value: np.array([np.nan]))
    with (pytest.raises(FloatingPointError, match="nonfinite structural") as caught,
          counted_structure(journal, counters)):
        np.linalg.eigvalsh(np.eye(2))
    assert np.isnan(caught.value.partial_record["nonfinite_eigvalsh_return"][0])
    assert counters == {"eigvalsh_attempted": 1, "eigvalsh_returned": 1}
    receipt = journal.finish({"status": "failed"})
    journal.close()
    assert receipt["actual_counters"]["returned"] == {"eigvalsh": 1}


def test_completed_metric_group_persists_before_later_numerical_failure(tmp_path, monkeypatch):
    from kron_reduce.workflows import kiki_future_transfer as workflow
    calls = []

    def synthetic_compare(*args):
        calls.append(args)
        if len(calls) == 2:
            error = FloatingPointError("synthetic later group overflow")
            error.partial_record = {"candidate_events": [{"primary": 1.0}]}
            raise error
        return {"I_pass": True, "group": len(calls)}

    monkeypatch.setattr(workflow, "compare_future", synthetic_compare)
    first, valid = workflow.save_future_comparison(tmp_path, "first")
    assert valid and first["I_pass"] and (tmp_path / "comparison-first.json").is_file()
    failed, valid = workflow.save_future_comparison(tmp_path, "failed")
    assert not valid and failed["I_pass"] is False and (tmp_path / "comparison-failed.json").is_file()
    third, valid = workflow.save_future_comparison(tmp_path, "third")
    assert valid and third["group"] == 3 and len(calls) == 3
    assert json.loads((tmp_path / "comparison-first.json").read_text())["group"] == 1

    def invalid_source(*args):
        raise ValueError("source alignment mismatch")

    monkeypatch.setattr(workflow, "compare_future", invalid_source)
    with pytest.raises(ValueError, match="source alignment"):
        workflow.save_future_comparison(tmp_path, "source-invalid")
    assert not (tmp_path / "comparison-source-invalid.json").exists()


def test_generic_runner_refuses_bounded_future_mode_before_data_verification(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "load_manifest",
        lambda _root: {
            "experiments": {"bounded": {"execution_mode": "bounded_future_transfer_cli_only"}}
        },
    )
    monkeypatch.setattr(
        runner,
        "verify_data",
        lambda *args, **kwargs: pytest.fail("verify_data ran before mode guard"),
    )
    with pytest.raises(RuntimeError, match="BOUNDED_FUTURE_TRANSFER_CLI_ONLY"):
        runner.run_experiment(tmp_path, "bounded", "future_transfer")
