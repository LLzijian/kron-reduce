from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from kron_reduce.evaluation.audit_budget import AuditRhsBudget
from kron_reduce.evaluation.scoped_audit_journal import ScopedAuditJournal
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from project_harness.evidence import sha256_file


def _workflow():
    from kron_reduce.workflows import kiki_integral_history_audit as workflow
    return workflow


def test_both_step_paths_have_distinct_durable_rhs_contexts(tmp_path):
    workflow = _workflow()
    closure, _ = _closure()
    q, v, z, clock, power, eq = _inputs()
    budget = AuditRhsBudget(240, 22, 262, 2)
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("pair", {"integration_rhs": 240, "physical_rhs": 22, "rollouts": 2})
    for step in (.005, .0025):
        workflow.audited_path(closure, q, v, z, clock, power, eq, step, budget, journal,
                             {"seed": 324, "event_id": "synthetic-pair", "step": step})
    closed = journal.finish({"status": "complete"})
    journal.close()
    assert closed["actual_counters"]["returned"] == {
        "rollouts": 2, "integration_rhs": 240, "physical_rhs": 22}
    rows = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
    rhs = [r for r in rows if r["category"] == "rhs"]
    assert {r["step"] for r in rhs} == {.005, .0025}
    assert all(r["seed"] == 324 and r["event_id"] == "synthetic-pair" for r in rhs)


def test_failed_path_scope_closes_and_independent_event_continues(tmp_path):
    workflow = _workflow()
    q, v, z, clock, power, eq = _inputs()
    journal = ScopedAuditJournal(tmp_path / "journal")
    budget = AuditRhsBudget(160, 22, 182, 2)
    closure, counter = _closure(fail_at=5)
    reservation = {"integration_rhs": 80, "physical_rhs": 11, "rollouts": 1}
    journal.begin("failed", reservation)
    with pytest.raises(FloatingPointError, match="synthetic RHS failure"):
        workflow.audited_path(closure, q, v, z, clock, power, eq, .005, budget, journal,
                             {"seed": 324, "event_id": "failed", "step": .005})
    failed = journal.finish({"status": "failed"})
    assert failed["actual_counters"]["attempted"]["integration_rhs"] == 5
    assert failed["actual_counters"]["returned"]["integration_rhs"] == 4
    assert counter["n"] == 5
    journal.begin("independent", reservation)
    other, _ = _closure()
    workflow.audited_path(other, q, v, z, clock, power, eq, .005, budget, journal,
                         {"seed": 324, "event_id": "other", "step": .005})
    journal.finish({"status": "complete"})
    journal.close()
    assert budget.records[0]["successful_rk4_updates"] is None
    assert budget.records[1]["successful_rk4_updates"] == 20


def test_completed_path_is_preserved_before_physical_failure(monkeypatch):
    workflow = _workflow()
    closure, _ = _closure()
    q, v, z, clock, power, eq = _inputs()
    budget = AuditRhsBudget(80, 11, 91, 1)
    preserved = []

    class Sink:
        def append(self, row):
            pass

    def fail_physical(*args):
        raise FloatingPointError("synthetic physical failure")

    monkeypatch.setattr(NonlinearPassiveClosure, "energy_identity", fail_physical)
    with pytest.raises(FloatingPointError, match="synthetic physical failure"):
        workflow.audited_path(closure, q, v, z, clock, power, eq, .005, budget, Sink(),
                             {"seed": 324, "event_id": "physical-fail", "step": .005},
                             path_sink=lambda p: preserved.append(p.copy()))
    assert len(preserved) == 1 and preserved[0].shape == (11, 22)
    assert budget.records[0]["successful_rk4_updates"] == 20
    assert budget.records[1]["status"] == "failed"


def test_static_numerical_failure_preserves_seed_slots_and_next_seed_runs(tmp_path, monkeypatch):
    workflow = _workflow()
    original = workflow.prepare_static
    calls = []

    def fail_first_seed(closure, scale, *, owner_key, ledger):
        calls.append(owner_key)
        if owner_key.endswith("324"):
            returned = ledger.call("svd", lambda: np.full((2,), np.nan), kind="synthetic", shape=(10, 2))
            error = FloatingPointError("synthetic static failure")
            error.partial_record = {"returned_bad_svd": returned}
            raise error
        return original(closure, scale, owner_key=owner_key, ledger=ledger)

    monkeypatch.setattr(workflow, "prepare_static", fail_first_seed)
    slots = [{"seed": seed, "cache_index": index, "status": "not_attempted"}
             for seed in (324, 325, 326) for index in range(258)]
    closure, _ = _closure()
    model = SimpleNamespace(closure=closure, target_scale=np.ones(20))
    journal = ScopedAuditJournal(tmp_path / "journal")
    failed, first = workflow.prepare_seed_scope(model, 324, "synthetic", slots, tmp_path, journal)
    assert failed is None and first["returned"] == {"svd": 1}
    assert all(r["status"] == "prerequisite_failed" for r in slots[:258])
    assert all(r["status"] == "not_attempted" for r in slots[258:])
    assert (tmp_path / "static-324-failed.npz").is_file()
    next_static, second = workflow.prepare_seed_scope(model, 325, "synthetic", slots, tmp_path, journal)
    assert next_static is not None and second["returned"] == {"svd": 2}
    assert calls == ["A06-step240-seed324", "A06-step240-seed325"]
    journal.close()


def test_static_identity_error_stops_instead_of_becoming_local(tmp_path, monkeypatch):
    workflow = _workflow()
    slots = [{"seed": 324, "status": "not_attempted"}]

    def invalid(*args, **kwargs):
        raise ValueError("model fingerprint mismatch")

    monkeypatch.setattr(workflow, "prepare_static", invalid)
    closure, _ = _closure()
    journal = ScopedAuditJournal(tmp_path / "journal")
    with pytest.raises(ValueError, match="model fingerprint"):
        workflow.prepare_seed_scope(SimpleNamespace(closure=closure, target_scale=np.ones(20)),
                                    324, "synthetic", slots, tmp_path, journal)
    assert slots[0]["status"] == "not_attempted"
    assert not (tmp_path / "journal/static-324-closed.json").exists()
    journal.close()


def _closure(fail_at: int | None = None):
    n = 10
    mass = np.linspace(1.0, 2.0, n, dtype=np.float64)
    mass /= mass.sum()
    projection = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    coupling = projection @ np.column_stack((np.linspace(-.2, .2, n), np.cos(np.arange(n))))
    closure = NonlinearPassiveClosure(
        mass=mass,
        edge_stiffness=np.full(n * (n - 1) // 2, 0.02),
        damping=0.01 * projection,
        coupling=coupling,
        latent_skew=np.array([[0.0, -0.1], [0.1, 0.0]]),
        latent_dissipation=np.diag([0.02, 0.03]),
    )
    if fail_at is not None:
        original = closure.rhs
        calls = {"n": 0}

        def fail_after_one_rk_stage(state, power, equilibrium):
            calls["n"] += 1
            if calls["n"] == fail_at:
                raise FloatingPointError("synthetic RHS failure")
            return original(state, power, equilibrium)

        object.__setattr__(closure, "rhs", fail_after_one_rk_stage)
        return closure, calls
    return closure, None


def _inputs():
    n = 10
    mass = np.linspace(1.0, 2.0, n)
    mass /= mass.sum()
    q = np.linspace(-0.01, 0.01, n)
    q -= mass @ q
    v = np.linspace(0.005, -0.005, n)
    v -= mass @ v
    z = np.array([0.01, -0.02])
    power = np.linspace(-0.01, 0.01, n)
    power -= power.mean()
    eq = np.linspace(-0.1, 0.1, n)
    clock = np.arange(11, dtype=np.float64) * 0.01
    return q, v, z, clock, power, eq


def test_audited_path_counts_original_rollout_and_physical_checks_and_resets_rhs():
    workflow = _workflow()
    closure, _ = _closure()
    q, v, z, clock, power, eq = _inputs()
    budget = AuditRhsBudget(80, 11, 91, 1)

    class Sink:
        def __init__(self):
            self.rows = []

        def append(self, row):
            self.rows.append(dict(row))

    sink = Sink()
    path, physical = workflow.audited_path(
        closure, q, v, z, clock, power, eq, .005, budget, sink, {"synthetic": True}
    )
    assert path.shape == (11, 22)
    np.testing.assert_array_equal(path[0, :10], q)
    np.testing.assert_array_equal(path[0, 10:20], v)
    assert physical["qualified"] is True
    observed = np.zeros((11, 20), dtype=np.float64)
    observed[-1, :10] = np.linspace(-.02, .02, 10)
    observed[-1, 10:] = np.linspace(.01, -.01, 10)
    reset = np.r_[observed[-1], path[-1, 20:]]
    np.testing.assert_array_equal(reset[:20], observed[-1])
    np.testing.assert_array_equal(reset[20:], path[-1, 20:])
    snapshot = budget.snapshot()
    assert snapshot["attempted"] == snapshot["returned"] == 91
    assert snapshot["rollout_calls_started"] == 1
    history, physical_segment = budget.records
    assert history["phase"] == "history_integration"
    assert history["attempted"] == history["returned"] == 80
    assert history["successful_rk4_updates"] == 20
    assert history["status"] == "complete"
    assert physical_segment["phase"] == "physical_checks"
    assert physical_segment["attempted"] == physical_segment["returned"] == 11
    assert physical_segment["successful_rk4_updates"] is None
    assert physical_segment["status"] == "complete"
    assert not "rhs" in vars(closure)
    assert sum(row.get("category") == "rhs" and row.get("transition") == "returned"
               for row in sink.rows) == 91


def test_failed_rollout_after_four_returned_stages_is_failed_and_not_retried():
    workflow = _workflow()
    closure, calls = _closure(fail_at=5)
    q, v, z, clock, power, eq = _inputs()
    budget = AuditRhsBudget(80, 11, 91, 1)

    class Sink:
        def __init__(self):
            self.rows = []

        def append(self, row):
            self.rows.append(dict(row))

    sink = Sink()
    with pytest.raises(FloatingPointError, match="synthetic RHS failure"):
        workflow.audited_path(
            closure, q, v, z, clock, power, eq, .005, budget, sink, {"synthetic": True}
        )
    assert calls["n"] == 5
    assert len(budget.records) == 1
    failed = budget.records[0]
    assert failed["status"] == "failed"
    assert failed["returned"] == 4
    assert failed["four_stage_returned_groups"] == 1
    assert failed["partial_returned_rhs"] == 0
    assert failed["successful_rk4_updates"] is None
    assert budget.snapshot()["rollout_calls_started"] == 1
    assert budget.snapshot()["attempted"] == 5
    assert budget.snapshot()["returned"] == 4
    assert sum(row.get("transition") == "attempted" and row.get("category") == "rhs"
               for row in sink.rows) == 5


def test_reject_stage_evidence_and_pack_record_keeps_nested_arrays_and_sanitizes_nan():
    workflow = _workflow()
    evidence = workflow.stage_evidence(
        {"source_commit": "abc"}, "REJECT", "synthetic only"
    )
    assert evidence["scientific_verdict"] == "REJECT"
    assert evidence["promotion_decision"] == "KEEP_BASELINE"
    assert evidence["model_promoted"] is False
    assert evidence["status"] == "PASS"  # the audit process succeeded; science may reject

    arrays = {}
    metadata = workflow.pack_record(
        {"outer": [{"matrix": np.arange(6, dtype=np.float64).reshape(2, 3),
                    "diagnostic": float("nan")}], "missing": None},
        arrays=arrays,
    )
    assert set(arrays) == {"outer__0__matrix"}
    assert arrays["outer__0__matrix"].shape == (2, 3)
    assert not arrays["outer__0__matrix"].dtype.hasobject
    assert metadata["outer"][0]["diagnostic"]["value"] is None
    assert metadata["outer"][0]["diagnostic"]["reason"] == "nonfinite_diagnostic"
    assert metadata["missing"] is None
    with pytest.raises(TypeError, match="object evidence is forbidden"):
        workflow.pack_record(np.array([object()], dtype=object))


def test_population_preserves_268_allocated_258_eligible_order_and_weights(tmp_path, monkeypatch):
    workflow = _workflow()
    rows = []
    for index in range(268):
        rows.append({
            "event_id": f"evt-{index:03d}", "raw_id": f"raw-{index:03d}",
            "source_dataset": "source-A", "fault_line": f"Line_{index % 5 + 1}",
            "source_role": "train", "role": "train",
            "declared_length": 12 if index < 258 else 11,
        })
    allocation = {"source_bindings": {"source-A": {}}, "rows": rows}
    population = workflow.population(allocation)
    assert len(population["allocated_slots"]) == 268
    assert len(population["cache_rows"]) == 258
    assert [row["event_id"] for row in population["cache_rows"]] == [f"evt-{i:03d}" for i in range(258)]
    assert len(population["seed_event_slots"]) == 3 * 258
    assert len(population["weights"]) == 258
    assert sum(population["weights"]) == pytest.approx(1.0)

    # bound_metadata compares the persisted population object as a whole. A
    # changed weight/order is rejected before inventory or NPZ access.
    allocation_path = tmp_path / "allocation.json"
    plan_path = tmp_path / "plan.json"
    inventory_path = tmp_path / "inventory.json"
    allocation_path.write_text(json.dumps(allocation), encoding="utf-8")
    tampered = json.loads(json.dumps(population))
    tampered["weights"][0] += 0.01
    plan_path.write_text(json.dumps({"population": tampered}), encoding="utf-8")
    inventory_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *args, **kwargs: None)
    with pytest.raises(PermissionError, match="population/order/weight plan changed"):
        workflow.bound_metadata(tmp_path, {
            "allocation_path": "allocation.json", "allocation_sha256": sha256_file(allocation_path),
            "read_plan_path": "plan.json", "read_plan_sha256": sha256_file(plan_path),
            "inventory_path": "inventory.json", "inventory_sha256": sha256_file(inventory_path),
        })


def test_generic_runner_refuses_bounded_history_mode_before_dataset_verification(tmp_path, monkeypatch):
    from project_harness import runner

    manifest = {"experiments": {"synthetic-a14": {
        "execution_mode": "bounded_history_cache_audit_cli_only",
        "dataset": "kiki_ieee39_v1",
    }}}
    calls = []
    monkeypatch.setattr(runner, "load_manifest", lambda _root: manifest)
    monkeypatch.setattr(
        runner, "verify_data",
        lambda *args, **kwargs: calls.append("verify_data")
        or pytest.fail("generic runner must guard before dataset verification"),
    )
    with pytest.raises(RuntimeError, match="BOUNDED_HISTORY_CACHE_AUDIT_CLI_ONLY"):
        runner.run_experiment(tmp_path, "synthetic-a14", "history_mechanism")
    assert calls == []
