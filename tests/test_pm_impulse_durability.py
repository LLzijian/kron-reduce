from __future__ import annotations

import json
import os

import numpy as np
import pytest

from kron_reduce.data.pm_impulse_reader import FAMILIES, RUN, SEEDS, SOURCES, FiniteAccessLedger
from kron_reduce.evaluation.pm_impulse_audit import (
    Dual,
    HistoryImpulse,
    ScalarLedger,
    aggregate_slots,
    audit_slot,
    expected_budget,
    prepare_source,
)
from kron_reduce.workflows.kiki_pm_impulse_audit import ScientificScopes


def _fail_nth_fsync(monkeypatch, nth: int):
    native_fsync = os.fsync
    calls = 0

    def injected(fd):
        nonlocal calls
        calls += 1
        if calls == nth:
            raise OSError("injected fsync failure")
        return native_fsync(fd)

    monkeypatch.setattr(os, "fsync", injected)
    return lambda: calls


def _compute():
    return {"per_invocation_maxima": {"source": expected_budget("source")}}


def _source_fn(called):
    def run(ledger):
        called.append(True)
        return prepare_source(np.ones(10), 1.0, np.ones(10), ledger=ledger)
    return run


def _scope(tmp_path):
    directory = tmp_path / "scope"
    directory.mkdir()
    return ScientificScopes(directory, _compute())


def test_reservation_fsync_failure_poisoned_before_kernel(monkeypatch, tmp_path):
    calls = _fail_nth_fsync(monkeypatch, 1)
    scopes = _scope(tmp_path)
    kernel = []
    with pytest.raises(OSError, match="injected"):
        scopes.invoke("source-one", "source", _source_fn(kernel))
    assert kernel == []
    assert scopes.failed
    with pytest.raises(RuntimeError, match="poisoned"):
        scopes.invoke("source-two", "source", _source_fn(kernel))
    assert kernel == [] and calls() == 1


@pytest.mark.parametrize("failed_fsync", [2, 3], ids=["npz", "json"])
def test_saved_artifact_fsync_failure_blocks_close_and_next_scope(monkeypatch, tmp_path, failed_fsync):
    calls = _fail_nth_fsync(monkeypatch, failed_fsync)
    scopes = _scope(tmp_path)
    kernel = []
    with pytest.raises(OSError, match="injected"):
        scopes.invoke("source-one", "source", _source_fn(kernel))
    assert len(kernel) == 1
    assert scopes.failed and scopes.rows == []
    journal = (tmp_path / "scope" / "scientific_calls.jsonl").read_text()
    assert '"status":"closed"' not in journal
    with pytest.raises(RuntimeError, match="poisoned"):
        scopes.invoke("source-two", "source", _source_fn(kernel))
    assert len(kernel) == 1 and calls() == failed_fsync


def test_closed_journal_fsync_failure_does_not_publish_row(monkeypatch, tmp_path):
    calls = _fail_nth_fsync(monkeypatch, 4)
    scopes = _scope(tmp_path)
    kernel = []
    with pytest.raises(OSError, match="injected"):
        scopes.invoke("source-one", "source", _source_fn(kernel))
    assert len(kernel) == 1 and scopes.rows == [] and scopes.failed
    assert list((tmp_path / "scope").glob("source-one.*"))
    with pytest.raises(RuntimeError, match="poisoned"):
        scopes.invoke("source-two", "source", _source_fn(kernel))
    assert len(kernel) == 1 and calls() == 4


def _reader_slots():
    slots, metadata = [], {}

    def add(key, kind, count):
        slots.append({"key": key, "phase": 1, "kind": kind, "expected_elements": count})
        metadata[key] = {"source": "synthetic", "event_id": "synthetic/event", "role": "train",
                         "dataset_or_attr": "Pm", "selection": "synthetic"}

    # Exact A16 finite inventory shape, represented only by synthetic metadata.
    for path, family in [(f"results/{RUN}/permitted_inputs.npz", "inputs")]:
        for name, shape in FAMILIES[family].items():
            key = f"{path}::{name}"
            add(key + "::descriptor", "countdescriptor", 1)
            add(key + "::array", "array", int(np.prod(shape)))
    for seed in SEEDS:
        for path, family in [(f"results/{RUN}/static-{seed}.npz", "static")]:
            for name, shape in FAMILIES[family].items():
                key = f"{path}::{name}"
                add(key + "::descriptor", "countdescriptor", 1)
                add(key + "::array", "array", int(np.prod(shape)))
        for index in range(258):
            path = f"results/{RUN}/seed-{seed}-event-{index:03d}/integral.npz"
            for name, shape in FAMILIES["integral"].items():
                key = f"{path}::{name}"
                add(key + "::descriptor", "countdescriptor", 1)
                add(key + "::array", "array", int(np.prod(shape)))
    for source in SOURCES:
        for name in ("attr-a", "attr-b", "attr-c", "attr-d", "attr-e", "attr-f", "attr-g", "attr-h", "attr-i"):
            add(f"{source}::attr::{name}", "attr", 1)
        for name in ("inertia", "equilibrium/Pm"):
            key = f"{source}::{name}"
            add(key + "::descriptor", "countdescriptor", 1)
            add(key + "::array", "array", 10)
    # The production ledger requires 258 event pairs per source.
    for index in range(258):
        source = SOURCES[index // 129]
        event = f"{source}/event_{index:03d}"
        for name, count in (("Pm", 110), ("t", 11)):
            key = f"{event}::{name}"
            add(key + "::descriptor", "countdescriptor", 1)
            add(key + "::array", "array", count)
    return slots, metadata


@pytest.mark.parametrize("key", ["ieee39_expansion/event_000::Pm::descriptor",
                                 "ieee39_expansion/event_000::Pm::array"])
def test_reader_attempt_log_fsync_failure_prevents_source_callback(monkeypatch, tmp_path, key):
    calls = _fail_nth_fsync(monkeypatch, 1)
    slots, metadata = _reader_slots()
    ledger = FiniteAccessLedger(slots, metadata, tmp_path / "array_access.jsonl")
    payload = []
    with pytest.raises(OSError, match="injected"):
        ledger.call(key, lambda: payload.append("called"), element_count=lambda _: 1)
    assert payload == [] and ledger.failed
    with pytest.raises(PermissionError, match="poisoned"):
        ledger.call("ieee39_expansion/event_000::t::array", lambda: payload.append("called"), element_count=lambda _: 11)
    assert payload == [] and calls() == 1


def test_native_fsync_source_scope_closes_and_publishes(tmp_path):
    scopes = _scope(tmp_path)
    kernel = []
    scopes.invoke("source-one", "source", _source_fn(kernel))
    assert kernel == [True] and scopes.rows[0]["scope"] == "source-one"
    assert scopes.rows[0]["counts"]["closed"] is True
    assert not scopes.failed


def test_kernel_overflow_is_persisted_as_closed_failure_and_aggregation_keeps_missing_weight(tmp_path):
    zero = Dual.exact(0.0)
    impulse = HistoryImpulse(
        u11=tuple(tuple(zero for _ in range(10)) for _ in range(11)),
        d=tuple(tuple(zero for _ in range(10)) for _ in range(10)),
        origin_difference=tuple(zero for _ in range(10)), anchor={}, counts={},
    )
    matrix = np.zeros((100, 2)); matrix[0, 0] = 1e308; matrix[3, 1] = 1.0
    common = {
        "c": np.ones(10), "S": np.ones(10), "g4": np.zeros(100),
        "btilde4": np.zeros(100), "dynamic_residual_scaled4": np.zeros(100),
        "force_residual4": np.zeros((10, 10)), "force_residual_projector": np.eye(10),
    }
    compute = {"per_invocation_maxima": {"slot": expected_budget("slot"), "source": expected_budget("source"),
                                                   "aggregate": expected_budget("aggregate")}}
    scopes = ScientificScopes(tmp_path / "scopes", compute)

    def overflow(ledger):
        return audit_slot(impulse, Ltilde=matrix, ledger=ledger, **common)

    result = scopes.invoke("slot-overflow", "slot", overflow, allow_numerical_failure=True)
    assert result.status == "numerical_failure" and result.complete is False
    assert result.counts["last_numeric_return"]["value"] == float("inf")
    packed = json.loads((tmp_path / "scopes" / "slot-overflow.json").read_text())
    raw_last = packed["counts"]["last_numeric_return"]
    assert raw_last["key"] == "point.mul"
    assert raw_last["value"] == {"value": None, "reason": "nonfinite_return", "representation": "inf"}
    assert scopes.rows[0]["counts"]["last_numeric_return"] == {
        "key": "point.mul", "value": None, "representation": "inf",
        "reason": "nonfinite_numeric_return"}
    assert scopes.rows[0]["scope"] == "slot-overflow" and not scopes.failed

    finite_matrix = np.zeros((100, 2)); finite_matrix[0, 0] = 1.0; finite_matrix[3, 1] = 1.0

    def finite_slot(ledger):
        return audit_slot(impulse, Ltilde=finite_matrix, ledger=ledger, **common)

    successful = scopes.invoke("slot-success", "slot", finite_slot)
    assert successful.complete and scopes.rows[-1]["counts"]["closed"]
    scopes.invoke("source-after-overflow", "source", _source_fn([]))
    records = [successful] * 257 + [result]
    weights = np.full(258, 1 / 258)
    aggregate_max = expected_budget("aggregate")
    aggregated = aggregate_slots(records, weights, ledger=ScalarLedger(aggregate_max))
    assert aggregated.complete is False
    assert aggregated.values["total_weight"].point == pytest.approx(1.0)
    assert aggregated.values["missing_weight"].point == pytest.approx(1 / 258)
    assert aggregated.values["population_metrics"] is None
