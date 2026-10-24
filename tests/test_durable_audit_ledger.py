from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import kron_reduce.data.durable_audit_ledger as durable
from kron_reduce.data.audit_access_ledger import AccessLedgerError
from kron_reduce.data.durable_audit_ledger import (
    AppendOnlyAuditJournal,
    DurableAccessLedger,
    DurableAuditJournalError,
)


def _plan():
    return [
        {"key": "static:shape", "phase": 1, "kind": "countdescriptor", "expected_elements": 1},
        {"key": "train:event:delta", "phase": 1, "kind": "array", "expected_elements": 2},
        {"key": "val:event:label", "phase": 2, "kind": "attr", "expected_elements": 1},
    ]


def _metadata():
    return {
        "static:shape": {
            "source": "synthetic", "event_id": "static", "role": "train",
            "dataset_or_attr": "events/event-a/t", "selection": "shape_dtype",
        },
        "train:event:delta": {
            "source": "synthetic", "event_id": "event-a", "role": "train",
            "dataset_or_attr": "events/event-a/delta", "selection": [0, 2, 0, 10],
        },
        "val:event:label": {
            "source": "synthetic", "event_id": "event-b", "role": "val",
            "dataset_or_attr": "events/event-b/label", "selection": "scalar",
        },
    }


def _ledger(tmp_path):
    return DurableAccessLedger(
        _plan(), logical_element_limit=2, metadata_by_key=_metadata(),
        journal_path=tmp_path / "phase1.jsonl",
    )


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_attempt_is_fsynced_before_source_callback_and_result_is_not_logged(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    fsync_calls = []
    original_fsync = durable.os.fsync

    def track_fsync(fd):
        fsync_calls.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(durable.os, "fsync", track_fsync)

    def read_array():
        assert [row["transition"] for row in _records(tmp_path / "phase1.jsonl")] == ["attempted"]
        assert fsync_calls
        return np.array([123456.789, 42.0])

    result = ledger.call("train:event:delta", read_array, element_count=lambda array: int(array.size))
    np.testing.assert_array_equal(result, [123456.789, 42.0])
    records = _records(tmp_path / "phase1.jsonl")
    assert [row["transition"] for row in records] == ["attempted", "returned"]
    assert records[0]["metadata"]["event_id"] == "event-a"
    serialized = (tmp_path / "phase1.jsonl").read_text(encoding="utf-8")
    assert "123456.789" not in serialized and "42.0" not in serialized
    assert "unstable" not in serialized


def test_source_exception_is_rethrown_and_failed_transition_is_durable(tmp_path):
    ledger = _ledger(tmp_path)

    def fail_source():
        raise OSError("synthetic read failure")

    with pytest.raises(OSError, match="synthetic read failure"):
        ledger.call("train:event:delta", fail_source, element_count=lambda _value: 0)
    rows = _records(tmp_path / "phase1.jsonl")
    assert [row["transition"] for row in rows] == ["attempted", "failed"]
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["exception_type"] == "OSError"
    assert "synthetic read failure" not in (tmp_path / "phase1.jsonl").read_text(encoding="utf-8")


def test_journal_failure_is_access_error_failclosed_and_never_retries_source(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    calls = []

    def fail_fsync(_fd):
        raise OSError("disk sync failed")

    monkeypatch.setattr(durable.os, "fsync", fail_fsync)
    with pytest.raises(DurableAuditJournalError) as first:
        ledger.call("train:event:delta", lambda: calls.append("source"))
    assert isinstance(first.value, AccessLedgerError)
    assert isinstance(first.value.__cause__, OSError)
    assert calls == []
    assert ledger.snapshot()["poisoned"] is True
    slot = next(row for row in ledger.snapshot()["selectors"] if row["key"] == "train:event:delta")
    assert slot["status"] == "failed"
    with pytest.raises(DurableAuditJournalError):
        ledger.call("static:shape", lambda: calls.append("second source"))
    assert calls == []


def test_phase1_can_be_hashed_after_close_then_phase2_uses_new_journal(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.call("static:shape", lambda: (3, "float64"))
    ledger.call("train:event:delta", lambda: np.ones((2,)), element_count=lambda array: int(array.size))
    phase1_path = tmp_path / "phase1.jsonl"
    old_journal = ledger._journal
    ledger.close_phase1_journal()
    ledger.close_phase1_journal()  # idempotent
    phase1_sha = hashlib.sha256(phase1_path.read_bytes()).hexdigest()
    assert old_journal._closed

    ledger.rotate_to_phase2_journal(tmp_path / "phase2.jsonl")
    with pytest.raises(OSError, match="closed or failed"):
        old_journal.append({"transition": "forbidden"})
    ledger.seal_prelabel_freeze(phase1_sha)
    assert ledger.call("val:event:label", lambda: "unstable") == "unstable"
    ledger.close_journal()
    assert hashlib.sha256(phase1_path.read_bytes()).hexdigest() == phase1_sha
    phase2 = _records(tmp_path / "phase2.jsonl")
    assert [row["transition"] for row in phase2] == [
        "prelabel_freeze_sealed", "attempted", "returned",
    ]
    assert phase2[0]["prelabel_freeze_sha256"] == phase1_sha
    assert "unstable" not in (tmp_path / "phase2.jsonl").read_text(encoding="utf-8")


def test_close_failure_poisons_and_raises_access_error(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)

    def fail_fsync(_fd):
        raise OSError("close sync failed")

    monkeypatch.setattr(durable.os, "fsync", fail_fsync)
    with pytest.raises(DurableAuditJournalError) as caught:
        ledger.close_phase1_journal()
    assert isinstance(caught.value.__cause__, OSError)
    assert ledger.snapshot()["poisoned"]


def test_journal_exclusive_create_refuses_overwrite(tmp_path):
    path = tmp_path / "exists.jsonl"
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        AppendOnlyAuditJournal(path)
    assert path.read_text(encoding="utf-8") == "preserve"

