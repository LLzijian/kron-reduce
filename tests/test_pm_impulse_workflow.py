from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.evaluation import pm_impulse_audit as kernel
from kron_reduce.workflows import kiki_pm_impulse_audit as workflow


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metadata_case():
    run = workflow.RUN
    population = {"cache_rows": [{"event_id": "source/event"}], "seed_event_slots": [1]}
    npz_path = f"results/{run}/sample.npz"
    descriptor_path = f"results/{run}/sample.json"
    npz_sha = "a" * 64
    descriptor_sha = "b" * 64
    plan = {
        "source_manifest_path": f"results/{run}/artifact_hashes.json",
        "source_manifest_sha256": "c" * 64,
        "population": population,
        "inherited_result_receipt_path": f"results/{run}/receipt.json",
        "npz_files": [{"path": npz_path, "sha256": npz_sha, "json_path": descriptor_path}],
        "metadata_files": [{"path": descriptor_path, "sha256": descriptor_sha}],
    }
    documents = {
        plan["source_manifest_path"]: {"sample.npz": npz_sha, "sample.json": descriptor_sha},
        f"results/{run}/allocated_population.json": population,
        "configs/proposals/kiki_a14_integral_history_reads.json": {"population": population},
        plan["inherited_result_receipt_path"]: {
            "run_id": run, "V_pass": True, "authorization_status": "consumed",
            "result_artifact_manifest_sha256": plan["source_manifest_sha256"],
        },
    }
    return plan, documents


def test_verify_metadata_accepts_bound_population_manifest_and_inherited_v():
    plan, documents = _metadata_case()
    assert workflow.verify_metadata(plan, documents) is None


@pytest.mark.parametrize("mutation", ["npz_manifest", "descriptor_manifest", "population", "a14_population", "v_false", "wrong_receipt_run"])
def test_verify_metadata_rejects_any_broken_source_or_v_inheritance(mutation):
    plan, documents = _metadata_case()
    if mutation == "npz_manifest":
        documents[plan["source_manifest_path"]]["sample.npz"] = "d" * 64
    elif mutation == "descriptor_manifest":
        documents[plan["source_manifest_path"]]["sample.json"] = "d" * 64
    elif mutation == "population":
        documents[f"results/{workflow.RUN}/allocated_population.json"] = {"cache_rows": []}
    elif mutation == "a14_population":
        documents["configs/proposals/kiki_a14_integral_history_reads.json"]["population"] = {"cache_rows": []}
    elif mutation == "v_false":
        documents[plan["inherited_result_receipt_path"]]["V_pass"] = False
    else:
        documents[plan["inherited_result_receipt_path"]]["run_id"] = "other-run"
    with pytest.raises(ValueError):
        workflow.verify_metadata(plan, documents)


def _array_descriptor(name: str, array: np.ndarray) -> dict:
    return {
        "array_member": name,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256_c_order_bytes": _sha(array.tobytes(order="C")),
    }


def test_verify_arrays_checks_selected_logical_members_and_ignores_other_members():
    q = np.arange(6, dtype=np.float64).reshape(2, 3)
    mass = np.ones(3, dtype=np.float64)
    document = {
        "q": _array_descriptor("q", q),
        "mass": _array_descriptor("mass", mass),
        "system__q": _array_descriptor("system__q", q + 100),
    }
    workflow.verify_arrays({"path": "synthetic.npz"}, {"q": q, "mass": mass}, document)


@pytest.mark.parametrize("mutation", ["missing", "shape", "dtype", "digest", "duplicate_conflict"])
def test_verify_arrays_rejects_missing_or_conflicting_selected_descriptor(mutation):
    array = np.arange(4, dtype=np.float64)
    descriptor = _array_descriptor("chosen", array)
    document = {"selected": descriptor}
    if mutation == "missing":
        document = {"unrelated": _array_descriptor("other", array)}
    elif mutation == "shape":
        descriptor["shape"] = [2, 2]
    elif mutation == "dtype":
        descriptor["dtype"] = "<f4"
    elif mutation == "digest":
        descriptor["sha256_c_order_bytes"] = "0" * 64
    elif mutation == "duplicate_conflict":
        document["duplicate"] = {**descriptor, "sha256_c_order_bytes": "0" * 64}
    with pytest.raises(ValueError):
        workflow.verify_arrays({"path": "synthetic.npz"}, {"chosen": array}, document)


def test_identity_clocks_derive_intervals_from_saved_origin_subtractions():
    raw = np.array([10000.0 + 0.01 * index for index in range(11)], dtype=np.float64)
    saved = np.array([float(raw[index]) - float(raw[10]) for index in range(11)])
    shifted = np.array([float(saved[index]) - float(saved[0]) for index in range(11)])
    expected_h = np.array([float(shifted[index + 1]) - float(shifted[index]) for index in range(10)])
    slots = [{"time": shifted.copy(), "h": expected_h.copy()} for _ in range(3)]
    arithmetic = workflow.IdentityArithmetic()
    result = arithmetic.clocks(raw, saved, slots)
    assert result.tobytes() == expected_h.tobytes()
    assert arithmetic.counts["sub"] == 11 + 11 + 10
    assert arithmetic.counts["finite_check"] == 32


def test_identity_clocks_reject_cross_seed_slot_mismatch_and_nonfinite_subtraction():
    raw = np.arange(11, dtype=np.float64) / 100.0
    saved = np.array([raw[index] - raw[10] for index in range(11)])
    shifted = np.array([saved[index] - saved[0] for index in range(11)])
    intervals = np.diff(shifted)
    slots = [{"time": shifted.copy(), "h": intervals.copy()} for _ in range(3)]
    slots[2]["h"][0] = np.nextafter(slots[2]["h"][0], np.inf)
    with pytest.raises(ValueError, match="saved time/interval"):
        workflow.IdentityArithmetic().clocks(raw, saved, slots)
    with pytest.raises(ValueError, match="nonfinite"):
        workflow.IdentityArithmetic().sub(float("inf"), 1.0)


def _approved_auth_fixture(tmp_path: Path, monkeypatch, *, status="user_approved", dirty=False, missing_review=False):
    root = tmp_path
    config_path = root / "configs/experiments/a16.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'input_contract = "measured_history_100ms_zoh_v1"\nread_plan_path = "read.json"\n'
        'read_plan_sha256 = "r"\ncompute_plan_path = "compute.json"\ncompute_plan_sha256 = "c"\n',
        encoding="utf-8",
    )
    experiment = {"config": "a16.toml", "execution_mode": workflow.MODE}
    approval = {
        "status": status, "authority": "explicit_user_reply", "scientific_design_approver": "Astra Ultra",
        "experiment": workflow.EXPERIMENT, "workflow": workflow.WORKFLOW, "stage": workflow.STAGE,
        "module": workflow.MODULE, "dataset": "kiki_ieee39_v1", "max_campaign_executions": 1,
        "retry_allowed": False, "max_fit_attempts": 0, "max_rhs": 0,
        "internal_val_payload_allowed": False, "source_val_payload_allowed": False,
        "source_test_payload_allowed": False, "model_promotion_allowed": False,
        "deployment_predictions_allowed": False,
        "read_plan_path": "read.json", "read_plan_sha256": "r",
        "compute_plan_path": "compute.json", "compute_plan_sha256": "c",
        "implementation_bindings": {"src/" + workflow.MODULE.replace(".", "/") + ".py": "i"},
    }
    if not missing_review:
        approval.update({"scientific_review_path": "review.md", "scientific_review_sha256": "s"})
    monkeypatch.setattr(workflow, "load_manifest", lambda _root: {"experiments": {workflow.EXPERIMENT: experiment}})
    monkeypatch.setattr(workflow, "load_policy", lambda _root: {"development_evaluations": {workflow.AUTH: approval}})
    monkeypatch.setattr(workflow, "validate_run_policy", lambda *_args: None)
    monkeypatch.setattr(workflow, "_verify_development_text_bindings", lambda *_args: None)
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_args: None)
    if dirty:
        monkeypatch.setattr(workflow, "assert_clean", lambda _root: (_ for _ in ()).throw(RuntimeError("dirty")))
    else:
        monkeypatch.setattr(workflow, "assert_clean", lambda _root: None)
    return root, experiment


@pytest.mark.parametrize("case", ["consumed", "missing_review", "dirty"])
def test_registered_execution_gates_source_runner_before_any_source_touch(tmp_path, monkeypatch, case):
    root, _experiment = _approved_auth_fixture(
        tmp_path,
        monkeypatch,
        status="consumed" if case == "consumed" else "user_approved",
        dirty=case == "dirty",
        missing_review=case == "missing_review",
    )
    calls = []
    monkeypatch.setattr(workflow, "_run_experiment", lambda *_args: calls.append("source-runner"))
    monkeypatch.setattr(workflow, "bound_metadata", lambda *_args: calls.append("metadata-source"))
    with pytest.raises((PermissionError, KeyError, RuntimeError)):
        workflow.execute_registered_audit(root)
    assert calls == []


def _scopes(tmp_path):
    return workflow.ScientificScopes(tmp_path, {"per_invocation_maxima": {
        kind: kernel.expected_budget(kind) for kind in ("source", "impulse", "slot", "aggregate")}})


def test_successful_scope_requires_exact_counts_and_cannot_repeat(tmp_path):
    scopes = _scopes(tmp_path)
    result = scopes.invoke("source-one", "source", lambda ledger:
        kernel.prepare_source(np.ones(10), 1.0, np.zeros(10), ledger=ledger))
    assert result.counts["attempted"] == kernel.expected_budget("source")
    assert scopes.rows[0]["execution_status"] == "computed"
    with pytest.raises(RuntimeError, match="already spent"):
        scopes.invoke("source-one", "source", lambda _: None)


def test_closed_but_truncated_success_is_rejected_and_preserved(tmp_path):
    scopes = _scopes(tmp_path)

    def omit_work(ledger):
        ledger.closed = True
        return {"misleading_status": "success"}

    with pytest.raises(RuntimeError, match="exact static schedule"):
        scopes.invoke("truncated", "source", omit_work)
    assert not scopes.rows
    assert (tmp_path / "truncated-count-mismatch.json").is_file()


def test_numerical_failure_and_prerequisite_skip_have_distinct_accounting(tmp_path):
    scopes = _scopes(tmp_path)

    def fail(ledger):
        ledger.call("point.add", lambda: 1.0)
        ledger.closed = True
        error = kernel.NumericalError("injected synthetic overflow")
        error.counts = ledger.snapshot()
        error.partial_record = {"last_numeric_return": float("inf")}
        raise error

    assert scopes.invoke("impulse-one", "impulse", fail, allow_numerical_failure=True) is None

    def skipped(ledger):
        ledger.closed = True
        return kernel.SlotResult("upstream_numerical_failure", False, {}, ledger.snapshot(), "impulse failed")

    result = scopes.invoke("slot-one", "slot", skipped, prerequisite="impulse-one")
    assert not result.complete
    assert scopes.rows[0]["execution_status"] == "numerical_failure"
    assert scopes.rows[1]["execution_status"] == "not_attempted_prerequisite_failed"
    assert scopes.rows[1]["counts"]["attempted"] == {}
    with pytest.raises(RuntimeError, match="failed prerequisite"):
        scopes.invoke("slot-bad", "slot", skipped, prerequisite="unknown")


def test_partial_population_aggregate_closes_at_its_actual_static_counts(tmp_path):
    scopes = _scopes(tmp_path)
    values = {key: kernel.Dual.exact(1.0) for key in kernel.SUMMARY_FIELDS}
    records = [kernel.SlotResult("resolved_nonzero", True, values, {}) for _ in range(257)]
    records.append(kernel.SlotResult("numerical_failure", False, {}, {}, "synthetic failure"))
    result = scopes.invoke("aggregate-partial", "aggregate", lambda ledger:
        kernel.aggregate_slots(records, np.full(258, 1.0 / 258), ledger=ledger))
    assert not result.complete and result.values["population_metrics"] is None
    assert result.values["missing_weight"].point == 1.0 / 258
    assert scopes.rows[0]["counts"]["attempted"] == kernel.expected_budget("aggregate", completed_slots=257)

