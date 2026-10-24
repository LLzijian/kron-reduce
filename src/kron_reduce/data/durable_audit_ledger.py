"""Durable, append-only transition journal for the canonical access ledger."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kron_reduce.data.audit_access_ledger import AccessLedger, AccessLedgerError

_METADATA_KEYS = {"source", "event_id", "role", "dataset_or_attr", "selection"}


class DurableAuditJournalError(AccessLedgerError):
    """A journal operation failed; the associated ledger is fail-closed."""


class AppendOnlyAuditJournal:
    """Exclusive-create UTF-8 JSONL journal, fsyncing every transition."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._closed = False
        self._failed = False

    def append(self, entry: dict[str, Any]) -> None:
        if self._closed or self._failed:
            raise OSError("audit journal is closed or failed")
        line = json.dumps(
            entry, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ) + "\n"
        try:
            self._stream.write(line)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except Exception:
            self._failed = True
            raise

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            self._failed = True
            failure = error
        try:
            self._stream.close()
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            self._failed = True
            if failure is None:
                failure = error
        finally:
            self._closed = True
        if failure is not None:
            raise failure


class DurableAccessLedger(AccessLedger):
    """Canonical selector accounting with durable per-transition records."""

    def __init__(
        self,
        plan,
        *,
        logical_element_limit: int,
        metadata_by_key: dict[str, dict[str, Any]],
        journal_path: str | Path,
    ) -> None:
        super().__init__(plan, logical_element_limit=logical_element_limit)
        if not isinstance(metadata_by_key, dict) or set(metadata_by_key) != set(self._slots):
            raise ValueError("metadata_by_key must cover exactly the registered selectors")
        self._metadata: dict[str, dict[str, Any]] = {}
        for key, metadata in metadata_by_key.items():
            if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
                raise ValueError(f"metadata for {key} must contain exactly {_METADATA_KEYS}")
            encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
            self._metadata[key] = json.loads(encoded)
        self._journal = AppendOnlyAuditJournal(journal_path)
        self._journal_phase = 1
        self._journal_failure: str | None = None

    def _raise_journal_error(self, operation: str, error: BaseException) -> None:
        self._poisoned = True
        self._journal_failure = type(error).__name__
        raise DurableAuditJournalError(f"durable audit journal {operation} failed") from error

    def _transition(
        self, key: str, transition: str, *, extra: dict[str, Any] | None = None
    ) -> None:
        slot = self._slots.get(key)
        entry: dict[str, Any] = {
            "transition": transition,
            "key": key,
            "phase": slot["phase"] if slot is not None else self.phase,
            "metadata": self._metadata.get(key),
            "status": slot["status"] if slot is not None else None,
            "sequence": slot["sequence"] if slot is not None else None,
            "decision_sequence": slot["decision_sequence"] if slot is not None else None,
            "expected_elements": slot["expected_elements"] if slot is not None else None,
            "attempted_elements": slot["attempted_elements"] if slot is not None else None,
            "returned_elements": slot["returned_elements"] if slot is not None else None,
            "source_returned": slot["source_returned"] if slot is not None else False,
            "failure_reason": slot["failure_reason"] if slot is not None else None,
            "exception_type": slot["exception_type"] if slot is not None else None,
            "skip_reason": slot["skip_reason"] if slot is not None else None,
            "prerequisite_key": slot["prerequisite_key"] if slot is not None else None,
            "prelabel_freeze_sha256": self.prelabel_freeze_sha256,
        }
        if extra:
            entry.update(extra)
        try:
            self._journal.append(entry)
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            self._raise_journal_error(transition, error)

    def _denial_transition(self, key: str, error: BaseException) -> None:
        self._transition(
            str(key),
            "denied",
            extra={"denial_type": type(error).__name__},
        )

    def call(self, key, function, *args, element_count=None, **kwargs):
        def durable_callback(*call_args, **call_kwargs):
            # AccessLedger has already charged the slot. Persist before touching the source.
            self._transition(key, "attempted")
            return function(*call_args, **call_kwargs)

        try:
            result = super().call(
                key, durable_callback, *args, element_count=element_count, **kwargs
            )
        except BaseException as source_or_ledger_error:
            if isinstance(source_or_ledger_error, DurableAuditJournalError):
                # The attempted record itself failed; another append cannot recover it.
                raise
            slot = self._slots.get(key) if isinstance(key, str) else None
            if slot is not None and slot["status"] == "failed":
                try:
                    self._transition(key, "failed")
                except DurableAuditJournalError as journal_error:
                    # A journal failure must not be swallowed by upstream OSError handlers.
                    raise journal_error from source_or_ledger_error
            else:
                self._denial_transition(key, source_or_ledger_error)
            raise
        self._transition(key, "returned")
        return result

    def skip(self, key: str, reason: str, *, prerequisite_key: str) -> None:
        try:
            super().skip(key, reason, prerequisite_key=prerequisite_key)
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            self._denial_transition(key, error)
            raise
        self._transition(key, "skipped")

    def declare_invalid_descriptor(self, key: str, reason: str) -> None:
        try:
            super().declare_invalid_descriptor(key, reason)
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            self._denial_transition(key, error)
            raise
        self._transition(key, "descriptor_invalid")

    def close_phase1_journal(self) -> None:
        """Close phase 1 so its bytes can be hashed before phase 2 is opened."""
        if self._journal_phase != 1:
            raise AccessLedgerError("phase-1 journal is no longer active")
        if self._journal._closed:
            return
        try:
            self._journal.close()
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            self._raise_journal_error("phase-1 close", error)

    def rotate_to_phase2_journal(self, journal_path: str | Path) -> None:
        """Open phase 2 only after phase 1 has been closed and externally hashed."""
        if self._poisoned:
            raise AccessLedgerError("cannot rotate a poisoned access ledger")
        if self.phase != 1 or self._journal_phase != 1 or not self._journal._closed:
            self._poisoned = True
            raise AccessLedgerError("close the phase-1 journal before phase-2 rotation")
        try:
            self._journal = AppendOnlyAuditJournal(journal_path)
            self._journal_phase = 2
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            self._raise_journal_error("phase-2 rotation", error)

    def seal_prelabel_freeze(self, sha256: str) -> None:
        if self._poisoned:
            raise AccessLedgerError("cannot seal a poisoned access ledger")
        if self._journal_phase != 2:
            self._poisoned = True
            raise AccessLedgerError("phase-2 journal must be open before sealing the freeze")
        try:
            super().seal_prelabel_freeze(sha256)
        except AccessLedgerError as error:
            self._denial_transition("<prelabel-freeze>", error)
            raise
        self._transition("<prelabel-freeze>", "prelabel_freeze_sealed")

    def close_journal(self) -> None:
        try:
            self._journal.close()
        except OSError as error:
            self._raise_journal_error("close", error)

    def snapshot(self) -> dict[str, Any]:
        result = super().snapshot()
        result["durable_journal"] = {
            "path": str(self._journal.path),
            "journal_phase": self._journal_phase,
            "closed": self._journal._closed,
            "failed": self._journal._failed,
            "failure": self._journal_failure,
        }
        return result
