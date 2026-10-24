"""Logical event-RHS accounting for bounded vectorized forecast work."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any


class EventRhsBudgetExceeded(RuntimeError):
    """Raised before a callback would exceed a registered event-RHS budget."""


class EventRhsBudget:
    """Account event work separately from vectorized callback invocations.

    ``invoke(function, state, ...)`` charges one logical RHS evaluation per row
    when ``state`` is rank two and one when it is rank one. The callback count
    is tracked independently, so batching does not hide event-level cost.
    """

    _PHASES = ("training_integration", "history_integration",
               "future_integration", "physical_checks")
    _INTEGRATION = frozenset(_PHASES[:3])

    def __init__(self, *, integration_limit: int, physical_limit: int,
                 total_limit: int, rollout_limit: int,
                 prediction_path_limit: int, training_integration_limit: int | None = None,
                 evaluation_integration_limit: int | None = None) -> None:
        self.integration_limit = self._limit(integration_limit, "integration_limit")
        self.physical_limit = self._limit(physical_limit, "physical_limit")
        self.total_limit = self._limit(total_limit, "total_limit")
        self.rollout_limit = self._limit(rollout_limit, "rollout_limit")
        self.prediction_path_limit = self._limit(prediction_path_limit, "prediction_path_limit")
        self.training_integration_limit = (None if training_integration_limit is None else
            self._limit(training_integration_limit, "training_integration_limit"))
        self.evaluation_integration_limit = (None if evaluation_integration_limit is None else
            self._limit(evaluation_integration_limit, "evaluation_integration_limit"))
        self.records: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._attempted = {phase: 0 for phase in self._PHASES}
        self._returned = {phase: 0 for phase in self._PHASES}
        self._blocked = {phase: 0 for phase in self._PHASES}
        self._callback_calls = {phase: 0 for phase in self._PHASES}
        self._callback_failures = {phase: 0 for phase in self._PHASES}
        self._total_attempted = self._total_returned = self._total_blocked = 0
        self._total_callback_calls = self._total_callback_failures = 0
        self._rollouts_started = self._rollouts_blocked = 0
        self._prediction_paths = self._prediction_paths_blocked = 0

    @staticmethod
    def _limit(value: int, name: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    @contextmanager
    def segment(self, identity: dict, phase: str, limit: int) -> Iterator[EventRhsBudget]:
        if self._active is not None:
            raise RuntimeError("event-RHS budget segments may not be nested")
        if phase not in self._PHASES:
            raise ValueError(f"invalid event-RHS phase: {phase!r}")
        if not isinstance(identity, dict):
            raise TypeError("segment identity must be a dict")
        limit = self._limit(limit, "segment limit")
        record = {"identity": deepcopy(identity), "phase": phase, "limit": limit,
                  "attempted": 0, "returned": 0, "blocked": 0,
                  "callback_calls": 0, "callback_failures": 0,
                  "rollouts_started": 0, "prediction_paths": 0,
                  "four_stage_returned_groups": None, "partial_returned_rhs": None,
                  "successful_rk4_updates": None}
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
            if phase in self._INTEGRATION:
                record["four_stage_returned_groups"], record["partial_returned_rhs"] = divmod(
                    record["returned"], 4)
            if completed:
                record["status"] = "complete"
                if phase in self._INTEGRATION:
                    record["successful_rk4_updates"] = record["returned"] // 4
            elif "status" not in record:
                record.update(status="failed", exception_type="UnknownError",
                              exception="segment exited without normal completion")
            self.records.append(deepcopy(record))
            self._active = None

    def invoke(self, function: Callable, state, *args, **kwargs):
        record = self._active
        if record is None:
            raise RuntimeError("event-RHS invoke requires an active segment")
        shape = getattr(state, "shape", None)
        if shape is None or len(shape) not in (1, 2):
            raise ValueError("RHS state must be a rank-one or rank-two array/tensor")
        cost = int(shape[0]) if len(shape) == 2 else 1
        if cost < 1:
            raise ValueError("RHS batch must contain at least one event")
        phase = record["phase"]
        phase_limit = self.physical_limit if phase == "physical_checks" else self.integration_limit
        phase_attempted = self._attempted[phase] if phase == "physical_checks" else sum(
            self._attempted[p] for p in self._INTEGRATION)
        partition_limit = (self.training_integration_limit if phase == "training_integration" else
                           self.evaluation_integration_limit if phase in
                           ("history_integration", "future_integration") else None)
        partition_attempted = (self._attempted["training_integration"] if phase == "training_integration"
                               else sum(self._attempted[p] for p in ("history_integration", "future_integration")))
        exceeded = ("segment" if record["attempted"] + cost > record["limit"] else
                    "partition" if partition_limit is not None and partition_attempted + cost > partition_limit else
                    "phase" if phase_attempted + cost > phase_limit else
                    "total" if self._total_attempted + cost > self.total_limit else None)
        if exceeded:
            record["blocked"] += cost
            self._blocked[phase] += cost
            self._total_blocked += cost
            raise EventRhsBudgetExceeded(f"event-RHS {exceeded} budget exhausted in {phase}")
        record["attempted"] += cost
        self._attempted[phase] += cost
        self._total_attempted += cost
        record["callback_calls"] += 1
        self._callback_calls[phase] += 1
        self._total_callback_calls += 1
        try:
            result = function(state, *args, **kwargs)
        except BaseException:
            record["callback_failures"] += 1
            self._callback_failures[phase] += 1
            self._total_callback_failures += 1
            raise
        record["returned"] += cost
        self._returned[phase] += cost
        self._total_returned += cost
        return result

    def note_prediction_paths(self, count: int) -> None:
        record = self._active
        if record is None:
            raise RuntimeError("prediction-path accounting requires an active segment")
        if record["phase"] not in ("history_integration", "future_integration"):
            raise RuntimeError("prediction paths may only be recorded in an integration segment")
        count = self._limit(count, "prediction path count")
        if count < 1:
            raise ValueError("prediction path count must be positive")
        if self._prediction_paths + count > self.prediction_path_limit:
            record["prediction_paths_blocked"] = record.get("prediction_paths_blocked", 0) + count
            self._prediction_paths_blocked += count
            raise EventRhsBudgetExceeded("prediction-path budget exhausted")
        record["prediction_paths"] += count
        self._prediction_paths += count

    def note_rollout_start(self, count: int = 1) -> None:
        record = self._active
        if record is None:
            raise RuntimeError("rollout accounting requires an active segment")
        if record["phase"] not in self._INTEGRATION:
            raise RuntimeError("rollout start is invalid during physical checks")
        count = self._limit(count, "rollout count")
        if count < 1:
            raise ValueError("rollout count must be positive")
        if record["rollouts_started"]:
            raise RuntimeError("each segment may record rollout starts only once")
        if self._rollouts_started + count > self.rollout_limit:
            record["rollouts_blocked"] = record.get("rollouts_blocked", 0) + count
            self._rollouts_blocked += count
            raise EventRhsBudgetExceeded("rollout budget exhausted")
        record["rollouts_started"] += count
        self._rollouts_started += count

    def snapshot(self) -> dict[str, Any]:
        integration_phases = tuple(p for p in self._PHASES if p != "physical_checks")
        def totals(counter, phases):
            return {"attempted": sum(counter[p] for p in phases),
                    "returned": sum(self._returned[p] for p in phases),
                    "blocked": sum(self._blocked[p] for p in phases),
                    "callback_calls": sum(self._callback_calls[p] for p in phases),
                    "callback_failures": sum(self._callback_failures[p] for p in phases)}
        return {"attempted": self._total_attempted, "returned": self._total_returned,
                "blocked": self._total_blocked, "callback_calls": self._total_callback_calls,
                "callback_failures": self._total_callback_failures,
                "rollouts_started": self._rollouts_started, "rollouts_blocked": self._rollouts_blocked,
                "prediction_paths": self._prediction_paths,
                "prediction_paths_blocked": self._prediction_paths_blocked,
                "integration": totals(self._attempted, integration_phases),
                "physical_checks": totals(self._attempted, ("physical_checks",)),
                "phases": {p: {"attempted": self._attempted[p], "returned": self._returned[p],
                               "blocked": self._blocked[p], "callback_calls": self._callback_calls[p],
                               "callback_failures": self._callback_failures[p]} for p in self._PHASES},
                "limits": {"integration": self.integration_limit, "physical": self.physical_limit,
                           "total": self.total_limit, "rollout": self.rollout_limit,
                           "prediction_paths": self.prediction_path_limit,
                           "training_integration": self.training_integration_limit,
                           "evaluation_integration": self.evaluation_integration_limit},
                "segments": deepcopy(self.records)}
