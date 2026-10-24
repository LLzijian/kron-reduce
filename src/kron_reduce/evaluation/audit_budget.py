"""Strict accounting for bounded, post-fit numerical-audit callbacks."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any


class AuditBudgetExceeded(RuntimeError):
    """Raised before a callback when an audit RHS budget is exhausted."""


class AuditRhsBudget:
    """Count independent integration and physical-check RHS audit calls."""

    _PHASES = ("history_integration", "future_integration", "physical_checks")

    def __init__(
        self,
        integration_limit: int = 2_690_172,
        physical_limit: int = 451_458,
        total_limit: int = 3_141_630,
        rollout_limit: int = 3_096,
    ) -> None:
        self.integration_limit = self._validate_limit(integration_limit, "integration_limit")
        self.physical_limit = self._validate_limit(physical_limit, "physical_limit")
        self.total_limit = self._validate_limit(total_limit, "total_limit")
        self.rollout_limit = self._validate_limit(rollout_limit, "rollout_limit")
        self.records: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._attempted = {phase: 0 for phase in self._PHASES}
        self._returned = {phase: 0 for phase in self._PHASES}
        self._blocked = {phase: 0 for phase in self._PHASES}
        self._total_attempted = 0
        self._total_returned = 0
        self._total_blocked = 0
        self._rollout_calls_started = 0
        self._rollout_calls_blocked = 0

    @staticmethod
    def _validate_limit(value: int, name: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    @contextmanager
    def segment(self, identity: dict, phase: str, limit: int) -> Iterator[AuditRhsBudget]:
        """Account one independent history, future, or physical-check segment."""
        if self._active is not None:
            raise RuntimeError("audit budget segments may not be nested")
        if phase not in self._PHASES:
            raise ValueError(f"invalid audit phase: {phase!r}")
        if not isinstance(identity, dict):
            raise TypeError("segment identity must be a dict")
        limit = self._validate_limit(limit, "segment limit")
        record: dict[str, Any] = {
            "identity": deepcopy(identity), "phase": phase, "limit": limit,
            "attempted": 0, "returned": 0, "blocked": 0,
            "four_stage_returned_groups": None, "partial_returned_rhs": None,
            "successful_rk4_updates": None,
            "rollout_calls_started": 0, "rollout_calls_blocked": 0,
        }
        self._active = record
        completed = False
        try:
            yield self
            completed = True
        except BaseException as error:
            record["status"] = "failed"
            record["exception_type"] = type(error).__name__
            record["exception"] = str(error)
            raise
        finally:
            returned = record["returned"]
            if phase in ("history_integration", "future_integration"):
                record["four_stage_returned_groups"], record["partial_returned_rhs"] = divmod(returned, 4)
            if completed:
                record["status"] = "complete"
                if phase in ("history_integration", "future_integration"):
                    record["successful_rk4_updates"] = returned // 4
            elif "status" not in record:
                record["status"] = "failed"
                record["exception_type"] = "UnknownError"
                record["exception"] = "segment exited without normal completion"
            self.records.append(deepcopy(record))
            self._active = None

    def invoke(self, function: Callable, *args, **kwargs):
        """Call once if all active segment, phase, and total caps permit it."""
        record = self._active
        if record is None:
            raise RuntimeError("audit RHS invoke requires an active segment")
        phase = record["phase"]
        phase_limit = (self.physical_limit if phase == "physical_checks"
                       else self.integration_limit)
        exceeded = None
        if record["attempted"] >= record["limit"]:
            exceeded = "segment"
        elif ((self._attempted["history_integration"] + self._attempted["future_integration"]
               if phase != "physical_checks" else self._attempted[phase]) >= phase_limit):
            exceeded = "phase"
        elif self._total_attempted >= self.total_limit:
            exceeded = "total"
        if exceeded is not None:
            record["blocked"] += 1
            self._blocked[phase] += 1
            self._total_blocked += 1
            raise AuditBudgetExceeded(f"audit {exceeded} RHS budget exhausted for {phase}")

        record["attempted"] += 1
        self._attempted[phase] += 1
        self._total_attempted += 1
        result = function(*args, **kwargs)
        record["returned"] += 1
        self._returned[phase] += 1
        self._total_returned += 1
        return result

    def note_rollout_start(self) -> None:
        """Record a rollout invocation after all initialization has succeeded."""
        record = self._active
        if record is None:
            raise RuntimeError("rollout start requires an active integration segment")
        if record["phase"] not in ("history_integration", "future_integration"):
            raise RuntimeError("rollout start is invalid during physical checks")
        if record["rollout_calls_started"] != 0:
            raise RuntimeError("each integration segment may start at most one rollout")
        if self._rollout_calls_started >= self.rollout_limit:
            record["rollout_calls_blocked"] += 1
            self._rollout_calls_blocked += 1
            raise AuditBudgetExceeded("audit rollout budget exhausted")
        record["rollout_calls_started"] += 1
        self._rollout_calls_started += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a detached summary of global and per-phase call accounting."""
        integration_phases = ("history_integration", "future_integration")
        integration = {
            key: sum(counter[phase] for phase in integration_phases)
            for key, counter in (("attempted", self._attempted),
                                 ("returned", self._returned),
                                 ("blocked", self._blocked))
        }
        physical = {key: counter["physical_checks"] for key, counter in (
            ("attempted", self._attempted), ("returned", self._returned),
            ("blocked", self._blocked))}
        return {
            "attempted": self._total_attempted, "returned": self._total_returned,
            "blocked": self._total_blocked,
            "rollout_calls_started": self._rollout_calls_started,
            "rollout_calls_blocked": self._rollout_calls_blocked,
            "integration_rhs": integration, "physical_rhs": physical,
            "phases": {
                phase: {"attempted": self._attempted[phase],
                        "returned": self._returned[phase],
                        "blocked": self._blocked[phase]}
                for phase in self._PHASES
            },
            "limits": {"integration": self.integration_limit,
                       "physical": self.physical_limit, "total": self.total_limit,
                       "rollout": self.rollout_limit},
        }
