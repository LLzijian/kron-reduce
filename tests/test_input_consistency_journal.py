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
