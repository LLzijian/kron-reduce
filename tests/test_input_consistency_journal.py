"""A17 reservations and failure closure on explicitly synthetic operations."""
import json

import pytest

from kron_reduce.evaluation.input_consistency_journal import InputConsistencyJournal
from kron_reduce.evaluation.scoped_audit_journal import ScopeJournalError


def test_exact_reservation_closes_only_after_evidence_is_saved(tmp_path):
    journal = InputConsistencyJournal(tmp_path / "operations")
    journal.begin("synthetic-solve", {"ls": 1, "ls_diagnostics": 1})
    assert journal.call("ls", lambda: 12) == 12
    assert journal.call("ls_diagnostics", lambda: True)
    order = []
    closed = journal.persist_and_finish(lambda: order.append("saved") or {"result.json": "0" * 64},
                                       status="complete")
    assert order == ["saved"]
    assert closed["durably_closed"] is True
    assert closed["actual_counters"]["returned"] == {"ls": 1, "ls_diagnostics": 1}
    assert len(closed["range_sha256"]) == 64
    journal.close()


def test_local_numerical_failure_keeps_failed_slot_then_next_independent_scope(tmp_path):
    journal = InputConsistencyJournal(tmp_path / "operations")
    journal.begin("synthetic-failed", {"ls": 1, "ls_diagnostics": 1})

    def fail():
        raise FloatingPointError("synthetic overflow")

    with pytest.raises(FloatingPointError):
        journal.call("ls", fail)
    first = journal.persist_and_finish(lambda: {"failed.json": "a" * 64}, status="numerical_failure")
    assert first["actual_counters"]["failed"] == {"ls": 1}
    journal.begin("synthetic-next", {"ls": 1})
    journal.call("ls", lambda: 3)
    journal.persist_and_finish(lambda: {"ok.json": "b" * 64}, status="complete")
    journal.close()
    assert len(journal.closed_scopes) == 2


def test_retry_after_local_failure_is_rejected_and_globally_stops(tmp_path):
    journal = InputConsistencyJournal(tmp_path / "failure-lock")
    journal.begin("synthetic-failed-slot", {"ls": 2})
    dispatched = []

    def numerical_failure():
        dispatched.append("first")
        raise FloatingPointError("synthetic numerical failure")

    with pytest.raises(FloatingPointError):
        journal.call("ls", numerical_failure)
    with pytest.raises(ScopeJournalError, match="failed A17 scope"):
        journal.call("ls", lambda: dispatched.append("retry"))
    assert dispatched == ["first"]
    assert journal.poisoned
    assert journal.attempted == journal.failed == {"ls": 1}
    assert journal.next_index == {"ls": 1}
    with pytest.raises(ScopeJournalError):
        journal.persist_and_finish(lambda: pytest.fail("normal close forbidden"), status="numerical_failure")
    with pytest.raises(ScopeJournalError):
        journal.begin("synthetic-independent-slot", {"ls": 1})
    assert not (journal.directory / "synthetic-failed-slot-closed.json").exists()
    journal.close()


def test_returned_value_survives_a_failed_terminal_record(tmp_path, monkeypatch):
    journal = InputConsistencyJournal(tmp_path / "terminal-failure")
    journal.begin("synthetic-returned", {"ls": 1})
    original = journal.append

    def fail_terminal(entry):
        if entry["transition"] == "returned":
            raise OSError("terminal write failed")
        return original(entry)

    monkeypatch.setattr(journal, "append", fail_terminal)
    with pytest.raises(OSError) as caught:
        journal.call("ls", lambda: {"answer": [2., 3.]})
    partial = caught.value.input_consistency_operation_partial
    assert partial["function_started"] and partial["function_returned"]
    assert partial["returned_value"] == {"answer": [2., 3.]}
    assert journal.poisoned and not (journal.directory / "synthetic-returned-closed.json").exists()
    journal.close()


def test_original_numerical_error_is_kept_when_failed_record_also_fails(tmp_path, monkeypatch):
    journal = InputConsistencyJournal(tmp_path / "failed-terminal")
    journal.begin("synthetic-numerical", {"ls": 1})
    original = journal.append

    def fail_terminal(entry):
        if entry["transition"] == "failed":
            raise OSError("failure record write failed")
        return original(entry)

    def fail_science():
        error = FloatingPointError("numeric failure")
        error.partial_record = {"original": [4., 5.]}
        raise error

    monkeypatch.setattr(journal, "append", fail_terminal)
    with pytest.raises(OSError) as caught:
        journal.call("ls", fail_science)
    partial = caught.value.input_consistency_operation_partial
    assert partial["function_started"] and not partial["function_returned"]
    assert partial["function_failure"]["type"] == "FloatingPointError"
    assert partial["function_failure"]["partial_record"] == {"original": [4., 5.]}
    assert journal.poisoned
    journal.close()


@pytest.mark.parametrize(
    ("status", "evidence"),
    [
        ("success", None),
        ("complete", {"artifact_hashes": {"forged.json": "f" * 64}}),
        ("complete", {"status": "numerical_failure"}),
    ],
)
def test_close_status_and_artifact_evidence_cannot_be_overridden(
    tmp_path, status, evidence
):
    journal = InputConsistencyJournal(tmp_path / f"invalid-close-{status}-{bool(evidence)}")
    journal.begin("synthetic-close", {"ls": 0})
    saved = []
    with pytest.raises(ScopeJournalError):
        journal.persist_and_finish(
            lambda: saved.append(True) or {"real.json": "a" * 64},
            status=status,
            evidence=evidence,
        )
    assert saved == []
    assert journal.poisoned
    assert not (journal.directory / "synthetic-close-closed.json").exists()
    journal.close()


@pytest.mark.parametrize("phase", ["reservation", "attempt", "save", "close"])
def test_durability_or_save_error_permanently_poisons_the_object(tmp_path, monkeypatch, phase):
    journal = InputConsistencyJournal(tmp_path / "operations")

    def fail(*args, **kwargs):
        raise OSError("synthetic disk failure")

    if phase == "reservation":
        monkeypatch.setattr(journal, "_durable_json", fail)
    with pytest.raises((OSError, ScopeJournalError)):
        journal.begin("synthetic-durable", {"ls": 1})
        if phase == "attempt":
            monkeypatch.setattr(journal.stream, "write", fail)
        journal.call("ls", lambda: 1)
        if phase == "close":
            monkeypatch.setattr(journal, "_durable_json", fail)
        journal.persist_and_finish(fail if phase == "save" else dict, status="complete")
    assert journal.poisoned
    with pytest.raises(ScopeJournalError):
        journal.call("ls", lambda: pytest.fail("must stop before another operation"))
    with pytest.raises(ScopeJournalError):
        journal.begin("synthetic-retry", {"ls": 1})
    journal.close()


def test_no_extra_call_or_incomplete_success_marker(tmp_path):
    journal = InputConsistencyJournal(tmp_path / "operations")
    journal.begin("synthetic-cap", {"ls": 1})
    journal.call("ls", lambda: 2)
    with pytest.raises(ScopeJournalError, match="limit"):
        journal.call("ls", lambda: pytest.fail("over-budget call dispatched"))
    journal.close()
    assert not (tmp_path / "operations/synthetic-cap-closed.json").exists()
    reserved = json.loads((tmp_path / "operations/synthetic-cap-reserved.json").read_text())
    assert reserved["spent"] is True and reserved["interrupted_status"] == "unknown_spent"
