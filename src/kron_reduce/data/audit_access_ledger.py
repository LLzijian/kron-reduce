"""Small, I/O-free access ledger for a pre-registered two-phase audit plan."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

MAX_DESCRIPTORS = 1078
MAX_ARRAY_SELECTIONS = 1078
MAX_ATTRIBUTE_READS = 2966
_KINDS = {"countdescriptor", "array", "attr"}
_PHASES = {1, 2}


class AccessLedgerError(RuntimeError):
    """Raised before access or when a returned selection violates its plan."""


class AccessLedger:
    """Enforce unique planned accesses and preserve attempted/returned states.

    The caller supplies all selectors and logical array sizes up front. The
    ledger neither opens files nor interprets selector names.
    """

    def __init__(self, plan: Sequence[Mapping[str, Any]], *, logical_element_limit: int):
        if type(logical_element_limit) is not int or logical_element_limit < 0:
            raise ValueError("logical_element_limit must be a nonnegative integer")
        self.logical_element_limit = logical_element_limit
        self.phase = 1
        self.prelabel_freeze_sha256: str | None = None
        self._sequence = 0
        self._source_call_count = 0
        self._poisoned = False
        self._denied: list[dict[str, Any]] = []
        self._violations: list[dict[str, Any]] = []
        self._unmeasured_array_element_returns = 0
        self._counts = {
            "countdescriptor": {"attempted": 0, "returned": 0, "failed": 0},
            "array": {"attempted": 0, "returned": 0, "failed": 0},
            "attr": {"attempted": 0, "returned": 0, "failed": 0},
        }
        self._elements = {"attempted": 0, "returned": 0, "failed": 0}
        self._slots: dict[str, dict[str, Any]] = {}
        array_element_plan = 0
        for raw in plan:
            if not isinstance(raw, Mapping):
                raise TypeError("each planned selector must be a mapping")
            if set(raw) != {"key", "phase", "kind", "expected_elements"}:
                raise ValueError("planned selector must contain exactly key/phase/kind/expected_elements")
            key = raw["key"]
            phase = raw["phase"]
            kind = raw["kind"]
            elements = raw["expected_elements"]
            if not isinstance(key, str) or not key:
                raise ValueError("selector key must be a nonempty string")
            if key in self._slots:
                raise ValueError(f"duplicate planned selector: {key}")
            if phase not in _PHASES or type(phase) is not int:
                raise ValueError(f"invalid selector phase for {key}")
            if kind not in _KINDS:
                raise ValueError(f"invalid selector kind for {key}")
            if type(elements) is not int or elements < 0:
                raise ValueError(f"expected_elements must be a nonnegative integer for {key}")
            if phase == 2 and kind != "attr":
                raise ValueError("phase 2 may only contain pre-registered attr selectors")
            self._slots[key] = {
                "key": key,
                "phase": phase,
                "kind": kind,
                "expected_elements": elements,
                "status": "not_attempted",
                "sequence": None,
                "decision_sequence": None,
                "attempted_elements": 0,
                "returned_elements": None,
                "source_returned": False,
                "returned_unmeasured_expected_elements": 0,
                "exception_type": None,
                "exception_message": None,
                "failure_reason": None,
                "skip_reason": None,
                "prerequisite_key": None,
                "validation_status": None,
                "validation_reason": None,
            }
            if kind == "array":
                array_element_plan += elements
        if not self._slots:
            raise ValueError("access plan must contain at least one selector")
        counts = {kind: sum(slot["kind"] == kind for slot in self._slots.values()) for kind in _KINDS}
        if counts["countdescriptor"] > MAX_DESCRIPTORS:
            raise ValueError("planned descriptor count exceeds 1078")
        if counts["array"] > MAX_ARRAY_SELECTIONS:
            raise ValueError("planned array selections exceed 1078")
        if counts["attr"] > MAX_ATTRIBUTE_READS:
            raise ValueError("planned attribute reads exceed 2966")
        if array_element_plan > self.logical_element_limit:
            raise ValueError("planned array elements exceed logical_element_limit")
        self._planned_counts = counts
        self._planned_array_elements = array_element_plan

    def _deny(self, key: str, reason: str) -> None:
        self._poisoned = True
        self._denied.append({"key": key, "phase": self.phase, "reason": reason})
        raise AccessLedgerError(reason)

    @staticmethod
    def _machine_reason(reason: str) -> bool:
        return isinstance(reason, str) and re.fullmatch(r"[a-z][a-z0-9_]*", reason) is not None

    def declare_invalid_descriptor(self, key: str, reason: str) -> None:
        """Mark a successfully-read phase-1 descriptor invalid for dependent skips."""
        if self._poisoned:
            self._deny(key, "ledger is poisoned after a forbidden access attempt")
        if not isinstance(key, str) or key not in self._slots:
            self._deny(str(key), "unknown selector")
        slot = self._slots[key]
        if self.phase != 1 or slot["phase"] != 1 or slot["kind"] != "countdescriptor":
            self._deny(key, "only phase-1 count descriptors can be declared invalid")
        if slot["status"] != "returned" or slot["validation_status"] is not None:
            self._deny(key, "invalid descriptor must be returned and not previously classified")
        if not self._machine_reason(reason):
            self._deny(key, "invalid descriptor reason must be a machine-readable code")
        self._sequence += 1
        slot["decision_sequence"] = self._sequence
        slot["validation_status"] = "invalid"
        slot["validation_reason"] = reason

    def skip(self, key: str, reason: str, *, prerequisite_key: str) -> None:
        """Terminally skip a planned selector because a prior prerequisite failed."""
        if self._poisoned:
            self._deny(key, "ledger is poisoned after a forbidden access attempt")
        if not isinstance(key, str) or key not in self._slots:
            self._deny(str(key), "unknown selector")
        slot = self._slots[key]
        if slot["phase"] != self.phase:
            self._deny(key, f"selector belongs to phase {slot['phase']}, current phase is {self.phase}")
        if slot["status"] != "not_attempted" or slot["skip_reason"] is not None:
            self._deny(key, "selector is already attempted or skipped")
        if not self._machine_reason(reason):
            self._deny(key, "skip reason must be a nonempty machine-readable code")
        if not isinstance(prerequisite_key, str) or prerequisite_key not in self._slots:
            self._deny(key, "skip prerequisite must name a registered selector")
        prerequisite = self._slots[prerequisite_key]
        prerequisite_order = prerequisite["decision_sequence"] or prerequisite["sequence"]
        if (prerequisite_order is None or prerequisite_order >= self._sequence + 1
                or not (prerequisite["status"] == "failed"
                        or (prerequisite["kind"] == "countdescriptor"
                            and prerequisite["validation_status"] == "invalid"))):
            self._deny(key, "skip prerequisite must be an earlier failed or invalid descriptor")
        self._sequence += 1
        slot["decision_sequence"] = self._sequence
        slot["skip_reason"] = reason
        slot["prerequisite_key"] = prerequisite_key

    def seal_prelabel_freeze(self, sha256: str) -> None:
        """Open phase 2 only after all phase-1 slots returned and were frozen."""
        if self._poisoned:
            self._deny("<prelabel-freeze>", "ledger is poisoned after a forbidden access attempt")
        if self.phase != 1:
            self._deny("<prelabel-freeze>", "prelabel freeze can only transition phase 1 to phase 2")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            self._deny("<prelabel-freeze>", "prelabel freeze SHA must be 64 hexadecimal characters")
        incomplete = [key for key, slot in self._slots.items()
                      if slot["phase"] == 1 and slot["status"] == "not_attempted"
                      and slot["skip_reason"] is None]
        if incomplete:
            self._deny("<prelabel-freeze>", "all phase-1 selectors must be terminal before phase-2 transition")
        self.prelabel_freeze_sha256 = sha256.lower()
        self.phase = 2

    def call(self, key: str, function: Callable[..., Any], *args,
             element_count: Callable[[Any], int] | None = None, **kwargs) -> Any:
        """Attempt one registered selection exactly once, preserving failures."""
        if self._poisoned:
            self._deny(key, "ledger is poisoned after a forbidden access attempt")
        if not isinstance(key, str) or key not in self._slots:
            self._deny(str(key), "unknown selector")
        slot = self._slots[key]
        if slot["phase"] != self.phase:
            self._deny(key, f"selector belongs to phase {slot['phase']}, current phase is {self.phase}")
        if slot["status"] != "not_attempted" or slot["skip_reason"] is not None:
            self._deny(key, f"selector already spent with status {slot['status']}")
        if not callable(function):
            self._deny(key, "selection callback is not callable")
        kind = slot["kind"]
        planned_count = self._planned_counts[kind]
        if self._counts[kind]["attempted"] + 1 > planned_count:
            self._deny(key, f"{kind} selection budget exhausted")
        if kind == "array" and self._elements["attempted"] + slot["expected_elements"] > self.logical_element_limit:
            self._deny(key, "array logical-element budget exhausted")

        self._sequence += 1
        self._source_call_count += 1
        slot["status"] = "attempted"
        slot["sequence"] = self._sequence
        slot["attempted_elements"] = slot["expected_elements"]
        self._counts[kind]["attempted"] += 1
        if kind == "array":
            self._elements["attempted"] += slot["expected_elements"]
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            slot["status"] = "failed"
            slot["exception_type"] = type(error).__name__
            slot["exception_message"] = str(error)
            self._counts[kind]["failed"] += 1
            if kind == "array":
                self._elements["failed"] += slot["expected_elements"]
            raise
        slot["source_returned"] = True
        self._counts[kind]["returned"] += 1
        try:
            actual = slot["expected_elements"] if element_count is None else element_count(result)
        except BaseException as error:
            slot["status"] = "failed"
            slot["failure_reason"] = "element_count_failed"
            slot["exception_type"] = type(error).__name__
            slot["exception_message"] = str(error)
            slot["returned_unmeasured_expected_elements"] = slot["expected_elements"]
            self._counts[kind]["failed"] += 1
            if kind == "array":
                self._unmeasured_array_element_returns += slot["expected_elements"]
            self._poisoned = True
            self._violations.append({"key": key, "phase": self.phase,
                                     "reason": "element_count_failed",
                                     "exception_type": type(error).__name__,
                                     "exception_message": str(error),
                                     "source_returned": True})
            raise
        if type(actual) is not int or actual < 0:
            actual = None
        slot["returned_elements"] = actual
        if actual is not None and kind == "array":
            self._elements["returned"] += actual
        if actual != slot["expected_elements"]:
            slot["status"] = "failed"
            slot["failure_reason"] = "returned_elements_mismatch"
            self._counts[kind]["failed"] += 1
            if actual is None:
                slot["returned_unmeasured_expected_elements"] = slot["expected_elements"]
                if kind == "array":
                    self._unmeasured_array_element_returns += slot["expected_elements"]
                message = f"selector {key} returned an invalid element count; expected {slot['expected_elements']}"
            else:
                message = f"selector {key} returned {actual} elements; expected {slot['expected_elements']}"
            self._poisoned = True
            self._violations.append({"key": key, "phase": self.phase,
                                     "reason": "returned_elements_mismatch",
                                     "actual_elements": actual,
                                     "expected_elements": slot["expected_elements"],
                                     "source_returned": True})
            raise AccessLedgerError(message)
        slot["status"] = "returned"
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe copy of selector and budget accounting."""
        statuses = {
            "not_attempted": sum(slot["status"] == "not_attempted" and slot["skip_reason"] is None
                                 for slot in self._slots.values()),
            "skipped": sum(slot["skip_reason"] is not None for slot in self._slots.values()),
            "attempted": sum(slot["status"] == "attempted" for slot in self._slots.values()),
            "returned": sum(slot["status"] == "returned" for slot in self._slots.values()),
            "failed": sum(slot["status"] == "failed" for slot in self._slots.values()),
        }
        return deepcopy({
            "phase": self.phase,
            "poisoned": self._poisoned,
            "prelabel_freeze_sha256": self.prelabel_freeze_sha256,
            "sequence_count": self._sequence,
            "source_call_count": self._source_call_count,
            "slot_status_counts": statuses,
            "planned_counts": self._planned_counts,
            "actual_counts": self._counts,
            "planned_array_elements": self._planned_array_elements,
            "logical_element_limit": self.logical_element_limit,
            "actual_array_elements": self._elements,
            "unmeasured_returned_array_expected_elements": self._unmeasured_array_element_returns,
            "limits": {
                "countdescriptor": MAX_DESCRIPTORS,
                "array": MAX_ARRAY_SELECTIONS,
                "attr": MAX_ATTRIBUTE_READS,
                "array_logical_elements": self.logical_element_limit,
            },
            "selectors": list(self._slots.values()),
            "denied_attempts": self._denied,
            "violations": self._violations,
        })
