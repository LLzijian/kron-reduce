"""Durable, byte-range-bound scope reservations and per-call transitions."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path


class ScopeJournalError(RuntimeError):
    """Persistence or transition validation failed; the scope must stop."""


_BUDGET_KEYS = frozenset({
    "expm", "svd", "force", "ls", "features", "encoder", "model_loads",
    "model_constructions", "rollouts", "integration_rhs", "physical_rhs", "probe", "eigvalsh",
})
_IGNORED_RESERVATION_FIELDS = frozenset({
    "slot", "owner", "model_sha256", "input_sha256", "model_sources",
    "read_plan_sha256", "rk4_updates",
})
_TERMINAL = frozenset({"returned", "failed"})


class ScopedAuditJournal:
    """Spend a scope before calls, then bind its closure to a hashed byte range."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.path = self.directory / "calls.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.active = None
        self.active_start = None
        self.reservation = None
        self.pending = {}
        self.seen = set()
        self.attempted = Counter()
        self.returned = Counter()
        self.failed = Counter()
        self.poisoned = False
        self.closed_scopes = []

    def _poison(self, message, cause=None):
        self.poisoned = True
        error = ScopeJournalError(message)
        if cause is None:
            raise error
        raise error from cause

    def _check(self):
        if self.poisoned or self.stream.closed:
            raise ScopeJournalError("journal is closed or poisoned")

    @staticmethod
    def _budget_keys(value):
        return value if isinstance(value, tuple) else (value,)

    @staticmethod
    def _validate_reservation(reservation):
        if not isinstance(reservation, dict):
            raise ScopeJournalError("scope reservation must be a mapping")
        budgets = {}
        for key, value in reservation.items():
            if key in _IGNORED_RESERVATION_FIELDS:
                continue
            if key in _BUDGET_KEYS:
                if type(value) is not int or value < 0:
                    raise ScopeJournalError(f"invalid reserved operation limit: {key}")
                budgets[key] = value
            elif type(value) is int:
                raise ScopeJournalError(f"unknown reserved operation category: {key}")
        return budgets

    def _durable_json(self, path, value):
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as error:
            self.poisoned = True
            raise ScopeJournalError("durable scope marker failed") from error

    def begin(self, scope_id, reservation):
        self._check()
        if (self.active is not None or not isinstance(scope_id, str) or not scope_id
                or not scope_id.replace("-", "").isalnum()):
            self._poison("nested or invalid audit scope")
        try:
            budgets = self._validate_reservation(reservation)
        except ScopeJournalError as error:
            self._poison(str(error), error)
        start = self.stream.buffer.tell()
        value = {
            "scope_id": scope_id,
            "reservation": reservation,
            "reserved_operation_limits": budgets,
            "start_byte_offset": start,
            "spent": True,
            "retry_allowed": False,
            "interrupted_status": "unknown_spent",
        }
        self._durable_json(self.directory / f"{scope_id}-reserved.json", value)
        self.active = scope_id
        self.active_start = start
        self.reservation = budgets
        self.pending = {}
        self.seen = set()
        self.attempted = Counter()
        self.returned = Counter()
        self.failed = Counter()

    @staticmethod
    def _key(entry):
        category = entry.get("category")
        if category in {"expm", "svd", "force", "ls", "eigvalsh"}:
            index = entry.get("index")
            if type(index) is not int or index < 0:
                raise ValueError("operator transition needs a nonnegative index")
            return (category, index), category, False
        if category == "model_load_and_construction":
            seed = entry.get("seed")
            if type(seed) is not int:
                raise ValueError("model load transition needs an integer seed")
            return (category, seed), ("model_loads", "model_constructions"), False
        if category == "closure_construction":
            seed = entry.get("seed")
            if type(seed) is not int:
                raise ValueError("closure construction needs an integer seed")
            return (category, seed), "model_constructions", False
        if category == "rhs":
            seed, event_id, step = entry.get("seed"), entry.get("event_id"), entry.get("step")
            phase, sequence = entry.get("phase"), entry.get("sequence")
            if type(seed) is not int or not isinstance(event_id, str) or not event_id:
                raise ValueError("RHS transition needs seed and event_id")
            if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step):
                raise ValueError("RHS transition needs a finite numeric step")
            if phase not in {"history_integration", "future_integration", "physical_checks"}:
                raise ValueError("RHS transition has an unknown phase")
            if type(sequence) is not int or sequence < 0:
                raise ValueError("RHS transition needs a nonnegative sequence")
            budget = "physical_rhs" if phase == "physical_checks" else "integration_rhs"
            return (category, seed, event_id, float(step), phase, sequence), budget, False
        if category == "rollout":
            seed, event_id, step = entry.get("seed"), entry.get("event_id"), entry.get("step")
            if type(seed) is not int or not isinstance(event_id, str) or not event_id:
                raise ValueError("rollout transition needs seed and event_id")
            if isinstance(step, bool) or not isinstance(step, (int, float)):
                raise ValueError("rollout transition needs a numeric step")
            return (category, seed, event_id, float(step)), "rollouts", False
        operation = entry.get("operation")
        if operation in {"features", "encoder"}:
            seed, event_index = entry.get("seed"), entry.get("event_index")
            if type(seed) is not int or type(event_index) is not int or event_index < 0:
                raise ValueError("feature/encoder transition needs seed and event_index")
            return (operation, seed, event_index), operation, False
        if category == "probe":
            token = entry.get("sequence", entry.get("index"))
            if token is None:
                raise ValueError("probe transition needs a sequence or index")
            return (category, token), category, False
        raise ValueError(f"unknown transition category or operation: {category!r}/{operation!r}")

    def append(self, entry):
        self._check()
        if self.active is None:
            self._poison("call transition requires a reserved scope")
        if not isinstance(entry, dict) or "scope_id" in entry:
            self._poison("transition entry must be a mapping and cannot override scope_id")
        transition = entry.get("transition")
        if transition not in {"attempted", "returned", "failed"}:
            self._poison("unknown call transition")
        try:
            key, budget_key, rhs_reusable = self._key(entry)
        except (TypeError, ValueError) as error:
            self._poison("invalid call transition identity", error)

        if transition == "attempted":
            if key in self.pending or (key in self.seen and not rhs_reusable):
                self._poison("duplicate or overlapping attempted transition")
            for key_name in self._budget_keys(budget_key):
                limit = self.reservation.get(key_name)
                if limit is None:
                    self._poison(f"operation was not reserved: {key_name}")
                if self.attempted[key_name] >= limit:
                    self._poison(f"reserved operation limit exceeded: {key_name}")
            self.pending[key] = budget_key
            self.seen.add(key)
            for key_name in self._budget_keys(budget_key):
                self.attempted[key_name] += 1
        else:
            if key not in self.pending:
                self._poison("terminal transition has no matching attempted transition")
            budget_key = self.pending.pop(key)
            target = self.returned if transition == "returned" else self.failed
            for key_name in self._budget_keys(budget_key):
                target[key_name] += 1

        record = {"scope_id": self.active, **entry}
        try:
            self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                         separators=(",", ":")) + "\n")
            self.stream.flush()
        except BaseException as error:
            self.poisoned = True
            raise ScopeJournalError("per-call write/flush failed") from error

    def sync(self):
        self._check()
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException as error:
            self.poisoned = True
            raise ScopeJournalError("scope log sync failed") from error

    def _read_active_range(self, end):
        start = self.active_start
        try:
            with self.path.open("rb") as stream:
                stream.seek(start)
                payload = stream.read(end - start)
        except OSError as error:
            self._poison("scope range read failed", error)
        if len(payload) != end - start or (payload and not payload.endswith(b"\n")):
            self._poison("scope byte range is incomplete")
        rows = []
        try:
            for line in payload.splitlines():
                item = json.loads(line.decode("utf-8"))
                if not isinstance(item, dict) or item.get("scope_id") != self.active:
                    raise ValueError("range contains a mismatched scope identity")
                rows.append(item)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._poison("scope byte range contains invalid records", error)
        return payload, rows

    def _validate_complete_range(self, status):
        if self.pending:
            self._poison("scope has attempted calls without returned/failed terminals")
        if any(self.attempted[key] != self.returned[key] + self.failed[key]
               for key in self.attempted):
            self._poison("scope transition counters do not balance")
        for key, count in self.attempted.items():
            if count > self.reservation.get(key, 0):
                self._poison(f"scope operation exceeded its reservation: {key}")
        if status == "complete":
            for key, limit in self.reservation.items():
                if self.attempted[key] != limit or self.returned[key] != limit or self.failed[key] != 0:
                    self._poison(f"complete scope did not consume and return its full reservation: {key}")

    def finish(self, evidence):
        self._check()
        if self.active is None:
            self._poison("no active scope to close")
        if not isinstance(evidence, dict) or set(evidence) & {
            "scope_id", "start_byte_offset", "end_byte_offset", "range_sha256",
            "actual_counters", "durably_closed",
        }:
            self._poison("scope close evidence is invalid or overrides journal fields")
        status = evidence.get("status")
        if not isinstance(status, str) or not status:
            self._poison("scope close evidence needs a status")
        self.sync()
        end = self.stream.buffer.tell()
        payload, rows = self._read_active_range(end)
        # Rebuild counts from the durable bytes; in-memory counters are only a
        # streaming guard and never the source of the close receipt.
        self.pending, self.attempted, self.returned, self.failed = {}, Counter(), Counter(), Counter()
        self.seen = set()
        try:
            for row in rows:
                entry = {key: value for key, value in row.items() if key != "scope_id"}
                transition = entry.get("transition")
                key, budget_key, rhs_reusable = self._key(entry)
                if transition == "attempted":
                    if key in self.pending or (key in self.seen and not rhs_reusable):
                        raise ValueError("duplicate attempted transition in saved range")
                    for key_name in self._budget_keys(budget_key):
                        if (key_name not in self.reservation
                                or self.attempted[key_name] >= self.reservation[key_name]):
                            raise ValueError("saved range exceeds reserved budget")
                    self.pending[key] = budget_key
                    self.seen.add(key)
                    for key_name in self._budget_keys(budget_key):
                        self.attempted[key_name] += 1
                elif transition in _TERMINAL:
                    if key not in self.pending:
                        raise ValueError("saved terminal transition is unmatched")
                    budget_key = self.pending.pop(key)
                    target = self.returned if transition == "returned" else self.failed
                    for key_name in self._budget_keys(budget_key):
                        target[key_name] += 1
                else:
                    raise ValueError("saved range has an unknown transition")
            self._validate_complete_range(status)
        except ScopeJournalError:
            self.poisoned = True
            raise
        except ValueError as error:
            self.poisoned = True
            raise ScopeJournalError("saved scope transitions or operation budget are invalid") from error
        actual_counters = {
            "attempted": dict(self.attempted),
            "returned": dict(self.returned),
            "failed": dict(self.failed),
        }
        value = {
            "scope_id": self.active,
            "start_byte_offset": self.active_start,
            "end_byte_offset": end,
            "range_sha256": hashlib.sha256(payload).hexdigest(),
            "actual_counters": actual_counters,
            "durably_closed": True,
            **evidence,
        }
        self._durable_json(self.directory / f"{self.active}-closed.json", value)
        self.closed_scopes.append(value)
        self.active = None
        self.active_start = None
        self.reservation = None
        self.pending = {}
        self.seen = set()
        return value

    def close(self):
        if self.stream.closed:
            return
        try:
            if not self.poisoned:
                self.sync()
            self.stream.close()
        except BaseException as error:
            self.poisoned = True
            raise ScopeJournalError("journal close failed") from error
