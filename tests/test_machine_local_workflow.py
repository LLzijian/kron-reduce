import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_synthetic import Budget, score_one
from kron_reduce.workflows import machine_local_synthetic as workflow


def _registration(stage):
    name, _ = workflow.STAGES[stage]
    return {
        "module": workflow.MODULE,
        "workflow": workflow.WORKFLOW,
        "stage": stage,
        "config": name + ".toml",
        "execution_mode": "bounded_synthetic_cli_only",
        "dataset": "synthetic_benchmark_registry_v1",
        "data_policy": "analytic_synthetic_only",
        "reads_validation_performance": False,
        "reads_test_performance": False,
    }


def _mock_authorization_prefix(monkeypatch, stage):
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    monkeypatch.setattr(workflow, "load_manifest", lambda _root: {"experiments": {
        workflow.STAGES[stage][0]: _registration(stage),
    }})
    monkeypatch.setattr(workflow, "load_state", lambda *_args: {})


def _policy_auth(stage, *, bindings=None, status="active"):
    return {
        **workflow.scope(stage),
        "status": status,
        "plan_path": workflow.PLAN,
        "protocol_path": workflow.PROTOCOL,
        "review_path": workflow.REVIEW,
        "implementation_bindings": {} if bindings is None else bindings,
    }


def test_generic_runner_refuses_bounded_synthetic_before_any_data_check(tmp_path, monkeypatch):
    from project_harness import runner

    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        "machine_local_s0_v1": _registration("s0"),
    }})
    monkeypatch.setattr(
        runner, "verify_data", lambda *_args, **_kwargs: pytest.fail("data check must not run")
    )
    with pytest.raises(RuntimeError, match="BOUNDED_SYNTHETIC_CLI_ONLY"):
        runner.run_experiment(tmp_path, "machine_local_s0_v1", "s0")


def test_stage_scopes_bind_fit_and_update_budgets():
    assert workflow.scope("s0")["fit_attempts"] == 0
    assert workflow.scope("s0")["optimizer_updates"] == 0
    assert workflow.scope("s1_base")["fit_attempts"] == 12
    assert workflow.scope("s1_base")["optimizer_updates"] == 2880


def test_authorization_rejects_unregistered_experiment_before_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    monkeypatch.setattr(workflow, "load_manifest", lambda _root: {"experiments": {}})
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("unregistered run read policy"))
    with pytest.raises(RuntimeError, match="MACHINE_LOCAL_REGISTRATION_MISMATCH"):
        workflow.authorize(tmp_path, "s0")


def test_authorization_requires_stage_approval_before_policy(monkeypatch, tmp_path):
    _mock_authorization_prefix(monkeypatch, "s0")
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: False)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_: pytest.fail("policy read before approval"))
    monkeypatch.setattr(workflow, "load_policy", lambda *_: pytest.fail("policy read before approval"))
    with pytest.raises(RuntimeError, match="MACHINE_LOCAL_STAGE_APPROVAL_REQUIRED"):
        workflow.authorize(tmp_path, "s0")


def test_authorization_rejects_inactive_one_use_scope(monkeypatch, tmp_path):
    _mock_authorization_prefix(monkeypatch, "s0")
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: True)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "digest")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    auth = _policy_auth("s0", status="consumed")
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {
        "bounded_synthetic_evaluations": {auth["authorization"]: auth},
    })
    with pytest.raises(RuntimeError, match="MACHINE_LOCAL_AUTHORIZATION_NOT_ACTIVE"):
        workflow.authorize(tmp_path, "s0")


def test_authorization_rejects_missing_exact_implementation_bindings(monkeypatch, tmp_path):
    _mock_authorization_prefix(monkeypatch, "s0")
    monkeypatch.setattr(workflow, "approval_recorded", lambda *_args: True)
    monkeypatch.setattr(workflow, "sha256_file", lambda *_args: "digest")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    auth = _policy_auth("s0")
    monkeypatch.setattr(workflow, "load_policy", lambda *_args: {
        "bounded_synthetic_evaluations": {auth["authorization"]: auth},
    })
    with pytest.raises(RuntimeError, match="MACHINE_LOCAL_BINDINGS_INCOMPLETE"):
        workflow.authorize(tmp_path, "s0")


def test_operation_budget_rejects_unregistered_and_over_limit_before_spending(tmp_path):
    budget = Budget(tmp_path / "operations.jsonl", {"probe": 1})
    try:
        with pytest.raises(RuntimeError, match="unregistered or exhausted operation budget: other"):
            budget.spend("other")
        assert budget.counts["other"] == 0
        budget.spend("probe")
        with pytest.raises(RuntimeError, match="unregistered or exhausted operation budget: probe"):
            budget.spend("probe")
        assert budget.counts["probe"] == 1
        budget.stream.flush()
        assert len((tmp_path / "operations.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    finally:
        budget.stream.close()


def test_budget_scope_records_returned_and_failed_allocations(tmp_path):
    budget = Budget(tmp_path / "scoped-operations.jsonl", {"probe": 3})
    try:
        with budget.scope("ok"):
            budget.spend("probe", 2)
        assert budget.scopes[-1] == {
            "scope": "ok",
            "transition": "returned",
            "returned_allocations": {"probe": 2},
        }
        assert budget.returned["probe"] == 2

        with pytest.raises(ValueError, match="synthetic scope failure"), budget.scope("partial"):
            budget.spend("probe")
            raise ValueError("synthetic scope failure")
        failed = budget.scopes[-1]
        assert failed["transition"] == "failed"
        assert failed["attempted_allocations"] == {"probe": 1}
        assert failed["returns_within_failed_scope"] == "unknown; all allocations remain spent"
        assert budget.counts["probe"] == 3
        assert budget.returned["probe"] == 2
    finally:
        budget.stream.close()


def test_budget_scope_cannot_borrow_another_scopes_unused_allocation(tmp_path):
    budget = Budget(
        tmp_path / "scope-limits.jsonl",
        {"probe": 4},
        scope_limits={"first": {"probe": 3}, "second": {"probe": 1}},
    )
    try:
        with budget.scope("first"):
            budget.spend("probe")
        with budget.scope("second"):
            budget.spend("probe")
            with pytest.raises(RuntimeError, match="scope budget: second/probe"):
                budget.spend("probe")
        assert budget.counts["probe"] == 2
        assert budget.returned["probe"] == 2
        assert budget.scopes[-1]["returned_allocations"] == {"probe": 1}
    finally:
        budget.stream.close()


def test_score_one_writes_g2_role_metrics_and_energy_json(tmp_path):
    class MockModel:
        tau = torch.tensor([0.3, 0.7], dtype=torch.float64)

        def __init__(self):
            self.calls = []

        def state_dict(self):
            return {"encoder": torch.zeros((2, 80), dtype=torch.float64)}

        def causal_rollout(self, q, v, power, equilibrium, features, count, **kwargs):
            self.calls.append((count, kwargs["max_step"], kwargs.get("return_history", False)))
            future = torch.zeros((64, count + 1, 22), dtype=torch.float64)
            if kwargs.get("return_history"):
                history = torch.zeros((64, 11, 22), dtype=torch.float64)
                return history, future
            return future

        def closure(self):
            class MockClosure:
                damping = np.zeros((10, 10))
                latent_dissipation = np.zeros((2, 2))

                @staticmethod
                def energy_identity(_state, _power, _equilibrium):
                    return {
                        "energy_rate": 0.0,
                        "supply_minus_dissipation": 0.0,
                        "identity_error": 0.0,
                    }

            return MockClosure()

    model = MockModel()
    truth = SimpleNamespace(mass=np.ones(10), edge_stiffness=np.ones(45))
    paths = np.zeros((64, 811, 40))
    scale = np.ones(20)
    q = np.zeros((64, 11, 10))
    v = np.zeros_like(q)
    power = np.zeros((64, 10))
    equilibrium = np.zeros_like(power)
    features = np.zeros((64, 80))
    budget = Budget(tmp_path / "score-operations.jsonl", {"numpy_rhs": 3})
    try:
        record = score_one(
            model, np.zeros(1), "mock_g2", "local", False, "G2", truth, paths, scale,
            q, v, power, equilibrium, features, tmp_path, budget,
        )
    finally:
        budget.stream.close()

    assert model.calls == [(800, 0.0025, True), (800, 0.00125, False)]
    assert set(record["metrics"]) == {"train", "val", "test"}
    assert record["model"] == "G2"
    assert record["energy_identity_error"] == 0.0
    assert len(record["energy_identity_samples"]) == 3
    assert all(
        sample["normalizer"] == 1.0
        and sample["energy_rate"] == 0.0
        and sample["supply_minus_dissipation"] == 0.0
        and sample["identity_error"] == 0.0
        for sample in record["energy_identity_samples"]
    )
    assert record["success"] is True and record["updates"] == 240
    for role in ("train", "val", "test"):
        metrics = record["metrics"][role]
        assert metrics["nrmse_3s_all"] == 0.0
        assert metrics["rmse_3s_all"] == 0.0
        assert metrics["step_difference"] == 0.0
    persisted = json.loads((tmp_path / "mock_g2.json").read_text(encoding="utf-8"))
    assert persisted["metrics"]["test"]["nrmse_8s_all"] == 0.0
    assert persisted["energy_identity_error"] == 0.0
