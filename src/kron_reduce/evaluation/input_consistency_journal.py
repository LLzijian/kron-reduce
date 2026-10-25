"""A17 operation reservations using the accepted durable range journal.

New categories are isolated here; historical campaign budgets are unchanged.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from kron_reduce.evaluation.scoped_audit_journal import ScopedAuditJournal, ScopeJournalError

CATEGORIES = frozenset({
    "closure_constructions", "structural_eigvalsh", "parameter_identity",
    "ls", "decomposition_check", "ls_diagnostics", "quadrature_proxy",
    "history_clock_checks", "future_clock_checks", "input_anchor_checks",
    "rollouts", "path_preflight", "input_eval", "integration_rhs", "physical_rhs",
    "successful_rk4_updates", "visible_resets", "energy_values", "path_refinement", "physical_judgement",
    "event_scores", "population_scores", "paired_groups", "reference_identity_checks",
})
LOCAL_NUMERICAL_ERRORS = (FloatingPointError, OverflowError, np.linalg.LinAlgError)


class InputConsistencyJournal(ScopedAuditJournal):
    """Count frozen high-level calls, not every BLAS scalar instruction."""

    def __init__(self, directory):
        super().__init__(directory)
        self.next_index = Counter()

    @staticmethod
    def _validate_reservation(reservation):
        if (not isinstance(reservation, dict) or set(reservation) - CATEGORIES
                or any(type(value) is not int or value < 0 for value in reservation.values())):
            raise ScopeJournalError("invalid A17 exact operation reservation")
        return dict(reservation)

    @staticmethod
    def _key(entry):
        category, index = entry.get("category"), entry.get("index")
        if category not in CATEGORIES or type(index) is not int or index < 0:
            raise ValueError("unknown A17 operation category or identity")
        return (category, index), category, False

    def begin(self, scope_id, reservation):
        try:
            super().begin(scope_id, reservation)
            self.next_index = Counter()
        except BaseException:
            self.poisoned = True
            raise

    def call(self, category, function, *, details=None):
        self._check()
        index = self.next_index[category]
        self.next_index[category] += 1
        entry = {"category": category, "index": index, "details": details}
        try:
            self.append({**entry, "transition": "attempted"})
            try:
                value = function()
            except BaseException as error:
                self.append({**entry, "transition": "failed", "error_type": type(error).__name__,
                             "message": str(error)})
                if not isinstance(error, LOCAL_NUMERICAL_ERRORS):
                    self.poisoned = True
                raise
            self.append({**entry, "transition": "returned"})
            return value
        except BaseException as error:
            if not isinstance(error, LOCAL_NUMERICAL_ERRORS):
                self.poisoned = True
            raise

    def append(self, entry):
        try:
            return super().append(entry)
        except BaseException:
            self.poisoned = True
            raise

    def persist_and_finish(self, save_artifacts, *, status, evidence=None):
        """Never close or continue a scope if its evidence could not be saved."""
        self._check()
        try:
            artifacts = save_artifacts()
            return self.finish({"status": status, "artifact_hashes": artifacts, **(evidence or {})})
        except BaseException:
            self.poisoned = True
            raise

    def finish(self, evidence):
        try:
            return super().finish(evidence)
        except BaseException:
            self.poisoned = True
            raise
