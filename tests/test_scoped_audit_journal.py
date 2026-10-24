from __future__ import annotations

import hashlib
import json

import pytest

from kron_reduce.evaluation.scoped_audit_journal import ScopedAuditJournal, ScopeJournalError


def _pair(journal, entry):
    journal.append({**entry, "transition": "attempted"})
    journal.append({**entry, "transition": "returned"})


def test_successful_scope_binds_reserved_operations_to_exact_byte_range(tmp_path):
    directory = tmp_path / "journal"
    journal = ScopedAuditJournal(directory)
    reservation = {"probe": 1, "owner": "seed-324", "rk4_updates": 1}
    journal.begin("scope-1", reservation)
    reserved = json.loads((directory / "scope-1-reserved.json").read_text(encoding="utf-8"))
    assert reserved["spent"] is True
    assert reserved["retry_allowed"] is False

    _pair(journal, {"category": "probe", "sequence": 1, "owner": "seed-324"})
    closed = journal.finish({"status": "complete", "evidence": "synthetic"})
    payload = journal.path.read_bytes()[closed["start_byte_offset"]:closed["end_byte_offset"]]
    assert closed["durably_closed"] is True
    assert closed["range_sha256"] == hashlib.sha256(payload).hexdigest()
    assert closed["actual_counters"] == {
        "attempted": {"probe": 1}, "returned": {"probe": 1}, "failed": {}
    }
    saved = json.loads((directory / "scope-1-closed.json").read_text(encoding="utf-8"))
    assert saved == closed
    journal.close()


def test_all_registered_categories_pair_and_consume_their_limits(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("scope-all", {
        "expm": 1, "svd": 1, "force": 1, "ls": 1, "features": 1,
        "encoder": 1, "model_loads": 1, "model_constructions": 1,
        "rollouts": 1, "integration_rhs": 2, "physical_rhs": 1,
    })
    for category in ("expm", "svd", "force", "ls"):
        _pair(journal, {"category": category, "index": 0, "context": "scope-all"})
    _pair(journal, {"operation": "features", "seed": 324, "event_index": 0})
    _pair(journal, {"operation": "encoder", "seed": 324, "event_index": 0})
    _pair(journal, {"category": "model_load_and_construction", "seed": 324})
    rollout = {"category": "rollout", "seed": 324, "event_id": "event-a", "step": 0.005}
    journal.append({**rollout, "transition": "attempted"})
    rhs = {
        "category": "rhs", "seed": 324, "event_id": "event-a", "step": 0.005,
        "phase": "history_integration", "sequence": 0,
    }
    _pair(journal, rhs)
    journal.append({**rollout, "transition": "returned"})
    # A new step context may start its own sequence at zero.
    _pair(journal, {**rhs, "step": 0.0025})
    _pair(journal, {
        "category": "rhs", "seed": 324, "event_id": "event-a", "step": 0.005,
        "phase": "physical_checks", "sequence": 0,
    })
    result = journal.finish({"status": "complete"})
    assert result["actual_counters"]["attempted"] == {
        "expm": 1, "svd": 1, "force": 1, "ls": 1, "features": 1,
        "encoder": 1, "model_loads": 1, "model_constructions": 1,
        "rollouts": 1, "integration_rhs": 2, "physical_rhs": 1,
    }
    journal.close()


def test_rhs_sequence_reuse_is_bound_to_distinct_seed_event_step_contexts(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("rhs-steps", {"integration_rhs": 2})
    for step in (0.005, 0.0025):
        row = {
            "category": "rhs", "seed": 324, "event_id": "event-a", "step": step,
            "phase": "future_integration", "sequence": 1,
        }
        _pair(journal, row)
    result = journal.finish({"status": "complete"})
    assert result["actual_counters"]["attempted"] == {"integration_rhs": 2}
    journal.close()


def test_rhs_terminal_with_wrong_step_and_duplicate_same_step_sequence_are_rejected(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "wrong-step")
    journal.begin("rhs-wrong-step", {"integration_rhs": 1})
    row = {
        "category": "rhs", "seed": 324, "event_id": "event-a", "step": 0.005,
        "phase": "future_integration", "sequence": 1,
    }
    journal.append({**row, "transition": "attempted"})
    with pytest.raises(ScopeJournalError, match="no matching attempted"):
        journal.append({**row, "step": 0.0025, "transition": "returned"})
    journal.close()

    journal = ScopedAuditJournal(tmp_path / "duplicate")
    journal.begin("rhs-duplicate", {"integration_rhs": 2})
    _pair(journal, row)
    with pytest.raises(ScopeJournalError, match="duplicate or overlapping"):
        journal.append({**row, "transition": "attempted"})
    journal.close()


def test_failed_partial_scope_is_legal_and_keeps_actual_counts(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("scope-failed", {"probe": 3})
    row = {"category": "probe", "sequence": 7}
    journal.append({**row, "transition": "attempted"})
    journal.append({**row, "transition": "failed", "error_type": "OSError"})
    closed = journal.finish({"status": "failed"})
    assert closed["actual_counters"] == {
        "attempted": {"probe": 1}, "returned": {}, "failed": {"probe": 1}
    }
    assert closed["durably_closed"] is True
    journal.close()


def test_scope_attempt_without_terminal_cannot_be_closed(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("scope-open", {"probe": 1})
    journal.append({"category": "probe", "sequence": 1, "transition": "attempted"})
    with pytest.raises(ScopeJournalError, match="without returned/failed"):
        journal.finish({"status": "failed"})
    assert journal.poisoned is True
    assert not (journal.directory / "scope-open-closed.json").exists()
    journal.close()


def test_missing_failed_transition_and_unmatched_terminal_are_rejected(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("scope-unmatched", {"probe": 1})
    with pytest.raises(ScopeJournalError, match="no matching attempted"):
        journal.append({"category": "probe", "sequence": 1, "transition": "failed"})
    assert journal.poisoned is True
    journal.close()


def test_over_budget_and_scope_id_override_poison_before_more_log_writes(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("scope-budget", {"probe": 1})
    _pair(journal, {"category": "probe", "sequence": 1})
    offset = journal.path.stat().st_size
    with pytest.raises(ScopeJournalError, match="limit exceeded"):
        journal.append({"category": "probe", "sequence": 2, "transition": "attempted"})
    assert journal.poisoned is True
    assert journal.path.stat().st_size == offset
    journal.close()

    journal = ScopedAuditJournal(tmp_path / "journal-override")
    journal.begin("scope-override", {"probe": 1})
    with pytest.raises(ScopeJournalError, match="cannot override scope_id"):
        journal.append({"scope_id": "forged", "category": "probe",
                        "sequence": 1, "transition": "attempted"})
    assert journal.poisoned is True
    assert journal.path.stat().st_size == 0
    journal.close()


def test_scope_range_tampering_is_rejected_during_finish(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("scope-tamper", {"probe": 1})
    _pair(journal, {"category": "probe", "sequence": 1})
    payload = journal.path.read_bytes()
    assert b'"scope_id":"scope-tamper"' in payload
    journal.stream.flush()
    changed = payload.replace(b"scope-tamper", b"scope-forged")
    journal.path.write_bytes(changed)
    with pytest.raises(ScopeJournalError, match="invalid records"):
        journal.finish({"status": "complete"})
    assert journal.poisoned is True
    assert not (journal.directory / "scope-tamper-closed.json").exists()
    journal.close()


def test_complete_status_requires_every_reserved_operation_returned(tmp_path):
    journal = ScopedAuditJournal(tmp_path / "journal")
    journal.begin("scope-incomplete", {"probe": 2})
    _pair(journal, {"category": "probe", "sequence": 1})
    with pytest.raises(ScopeJournalError, match="full reservation"):
        journal.finish({"status": "complete"})
    assert journal.poisoned is True
    journal.close()


def test_interrupted_scope_remains_spent_and_journal_failure_poisons(tmp_path):
    directory = tmp_path / "interrupted"
    journal = ScopedAuditJournal(directory)
    journal.begin("scope-interrupted", {"probe": 1})
    journal.append({"category": "probe", "sequence": 1, "transition": "attempted"})
    journal.close()
    reserved = json.loads((directory / "scope-interrupted-reserved.json").read_text(encoding="utf-8"))
    assert reserved["interrupted_status"] == "unknown_spent"
    assert not (directory / "scope-interrupted-closed.json").exists()
    with pytest.raises(FileExistsError):
        ScopedAuditJournal(directory)

    journal = ScopedAuditJournal(tmp_path / "write-failure")
    journal.begin("scope-write", {"probe": 1})

    def fail_write(_value):
        raise OSError("disk unavailable")

    journal.stream.write = fail_write
    with pytest.raises(ScopeJournalError, match="per-call write/flush failed"):
        journal.append({"category": "probe", "sequence": 1, "transition": "attempted"})
    assert journal.poisoned is True
    with pytest.raises(ScopeJournalError, match="poisoned"):
        journal.append({"category": "probe", "sequence": 1, "transition": "returned"})
    journal.close()
