"""Pure-array verification helpers for the fixed-model A10 train audit.

This module accepts already reconstructed diagnostics and metadata. It never
opens data, loads a model, or performs a prediction or integration.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

ATOL = 1e-12
SEEDS = (324, 325, 326)
STEPS = (0.005, 0.0025)
REFINEMENT_LIMIT = 1e-5
MARGIN_ATOL = 1e-12


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _margin_bound_failures(bound: Any, path: str) -> list[str]:
    """Validate a recorded rho Lipschitz bound rather than trusting its flag."""
    if not isinstance(bound, Mapping):
        return [f"{path}:missing_bound_record"]
    failures = []
    if bound.get("bound_verified") is not True:
        failures.append(f"{path}:bound_not_verified")
    truth = bound.get("truth")
    prediction = bound.get("prediction")
    rho_t = truth.get("rho") if isinstance(truth, Mapping) else None
    rho_p = prediction.get("rho") if isinstance(prediction, Mapping) else None
    atom = bound.get("E_atom")
    vector = bound.get("E_vector")
    if not all(_finite_number(value) for value in (rho_t, rho_p, atom, vector)):
        failures.append(f"{path}:nonfinite_or_missing_margin_value")
        return failures
    if atom < 0 or vector < 0:
        failures.append(f"{path}:negative_error_bound")
        return failures
    tolerance = MARGIN_ATOL * max(1.0, abs(rho_t), abs(rho_p), vector)
    if abs(rho_p - rho_t) > atom + tolerance:
        failures.append(f"{path}:rho_difference_exceeds_E_atom")
    if atom > vector + tolerance:
        failures.append(f"{path}:E_atom_exceeds_E_vector")
    return failures


def _diagnostic_margin_failures(diagnostic: Any, prefix: str = "diagnostic") -> list[str]:
    if not isinstance(diagnostic, Mapping):
        return [f"{prefix}:missing_diagnostic"]
    failures = []
    for key in ("history_rho", "future_rho", "full_rho"):
        failures.extend(_margin_bound_failures(diagnostic.get(key), f"{prefix}.{key}"))
    representative = diagnostic.get("representative_truth_window")
    fixed = (representative.get("fixed_five_point_audit")
             if isinstance(representative, Mapping) else None)
    failures.extend(_margin_bound_failures(
        fixed, f"{prefix}.representative_truth_window.fixed_five_point_audit"))
    saved_margin = representative.get("prediction_margin_same_window") if isinstance(representative,Mapping) else None
    fixed_margin = fixed.get("prediction",{}).get("rho") if isinstance(fixed,Mapping) else None
    if not _finite_number(saved_margin) or not _finite_number(fixed_margin):
        failures.append(f"{prefix}.representative_truth_window:missing_prediction_margin")
    elif abs(saved_margin-fixed_margin) > MARGIN_ATOL*max(1.0,abs(saved_margin),abs(fixed_margin)):
        failures.append(f"{prefix}.representative_truth_window:prediction_margin_mismatch")
    return failures


def _coarse_fine_margin_failures(
    item: Mapping[str, Any], coarse_diagnostic: Any, fine_diagnostic: Any,
) -> list[str]:
    comparison = item.get("coarse_fine_rule_margin_comparison")
    if not isinstance(comparison, Mapping):
        return ["coarse_fine_rule_margin_comparison:missing"]
    failures = []
    for key in ("history_rho", "future_rho", "full_rho"):
        failures.extend(_margin_bound_failures(comparison.get(key), f"coarse_fine.{key}"))

    representative = comparison.get("representative_truth_window")
    fixed = (representative.get("fixed_five_point_audit")
             if isinstance(representative, Mapping) else None)
    failures.extend(_margin_bound_failures(
        fixed, "coarse_fine.representative_truth_window.fixed_five_point_audit"))
    if isinstance(representative, Mapping) and isinstance(fixed, Mapping):
        truth = fixed.get("truth")
        prediction = fixed.get("prediction")
        coarse_rho = truth.get("rho") if isinstance(truth, Mapping) else None
        fine_rho = prediction.get("rho") if isinstance(prediction, Mapping) else None
        saved_coarse = representative.get("coarse_margin_same_window")
        saved_fine = representative.get("fine_margin_same_window")
        if not all(_finite_number(value) for value in
                   (coarse_rho, fine_rho, saved_coarse, saved_fine)):
            failures.append("coarse_fine.representative_truth_window:missing_same_window_margins")
        else:
            tolerance = MARGIN_ATOL * max(1.0, abs(coarse_rho), abs(fine_rho))
            if abs(saved_coarse - coarse_rho) > tolerance:
                failures.append("coarse_fine.representative_truth_window:coarse_margin_mismatch")
            if abs(saved_fine - fine_rho) > tolerance:
                failures.append("coarse_fine.representative_truth_window:fine_margin_mismatch")

    def window_indices(diagnostic: Any) -> tuple[Any, Any] | None:
        window = (diagnostic.get("representative_truth_window")
                  if isinstance(diagnostic, Mapping) else None)
        if not isinstance(window, Mapping):
            return None
        start, end = window.get("start_index"), window.get("end_index")
        if type(start) is not int or type(end) is not int or start < 0 or end != start+4 or end < 11:
            return None
        count = diagnostic.get("future_points")
        if type(count) is int and end >= 11+count:
            return None
        return start, end

    expected = window_indices(coarse_diagnostic)
    fine = window_indices(fine_diagnostic)
    saved = window_indices({"representative_truth_window": representative})
    if expected is None or fine is None or saved is None:
        failures.append("coarse_fine.representative_truth_window:missing_window_indices")
    elif expected != fine or saved != expected:
        failures.append("coarse_fine.representative_truth_window:window_indices_mismatch")
    for side, diagnostic in (("coarse",coarse_diagnostic),("fine",fine_diagnostic)):
        actual_window = diagnostic.get("representative_truth_window", {}) if isinstance(diagnostic,Mapping) else {}
        actual_margin = actual_window.get("prediction_margin_same_window") if isinstance(actual_window,Mapping) else None
        compared_margin = representative.get(f"{side}_margin_same_window") if isinstance(representative,Mapping) else None
        if not _finite_number(actual_margin) or not _finite_number(compared_margin):
            failures.append(f"coarse_fine.representative_truth_window:{side}_path_margin_missing")
        elif abs(actual_margin-compared_margin) > MARGIN_ATOL*max(1.0,abs(actual_margin),abs(compared_margin)):
            failures.append(f"coarse_fine.representative_truth_window:{side}_path_margin_mismatch")
    return failures


def compare_recursive(saved: Any, rebuilt: Any, *, atol: float = ATOL) -> list[dict[str, Any]]:
    """Compare complete JSON-like trees; numeric leaves use absolute tolerance."""
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    mismatches: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                mismatches.append({"path": path, "reason": "type_mismatch"})
                return
            left_keys, right_keys = set(left), set(right)
            for key in sorted(left_keys - right_keys, key=str):
                mismatches.append({"path": f"{path}.{key}", "reason": "missing_rebuilt_field"})
            for key in sorted(right_keys - left_keys, key=str):
                mismatches.append({"path": f"{path}.{key}", "reason": "unexpected_rebuilt_field"})
            for key in sorted(left_keys & right_keys, key=str):
                visit(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
                mismatches.append({"path": path, "reason": "type_mismatch"})
                return
            if len(left) != len(right):
                mismatches.append({"path": path, "reason": "length_mismatch",
                                   "saved": len(left), "rebuilt": len(right)})
                return
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, f"{path}[{index}]")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not bool or type(right) is not bool or left != right:
                mismatches.append({"path": path, "reason": "boolean_mismatch",
                                   "saved": left, "rebuilt": right})
            return
        if isinstance(left, int) or isinstance(right, int):
            if type(left) is not int or type(right) is not int or left != right:
                mismatches.append({"path": path, "reason": "integer_mismatch",
                                   "saved": left, "rebuilt": right})
            return
        if isinstance(left, float) or isinstance(right, float):
            if (isinstance(left, bool) or isinstance(right, bool)
                    or not isinstance(left, (int, float)) or not isinstance(right, (int, float))
                    or not math.isfinite(float(left)) or not math.isfinite(float(right))):
                mismatches.append({"path": path, "reason": "nonfinite_or_non_numeric"})
                return
            difference = abs(float(left) - float(right))
            if difference > atol:
                mismatches.append({"path": path, "reason": "numeric_mismatch",
                                   "absolute_difference": difference})
            return
        if type(left) is not type(right) or left != right:
            mismatches.append({"path": path, "reason": "value_mismatch",
                               "saved": left, "rebuilt": right})

    visit(saved, rebuilt, "$" )
    return mismatches


def compare_saved_replay(actual: Any, expected: Any, *, atol: float = ATOL) -> dict[str, Any]:
    """Compare a rebuilt old replay with its saved counterpart recursively."""
    mismatches = compare_recursive(expected, actual, atol=atol)
    return {"qualified": not mismatches, "absolute_tolerance": atol,
            "mismatch_count": len(mismatches), "mismatches": mismatches}


def verify_saved_replay(
    *, event_ids: Sequence[str], event_weights: Sequence[float], counts: Sequence[int],
    rebuilt_diagnostics: Mapping[str, Mapping[str, Any]],
    saved_diagnostics: Mapping[str, Mapping[str, Any]],
    rebuilt_summary: Mapping[str, Any], saved_summary: Mapping[str, Any],
    atol: float = ATOL,
) -> dict[str, Any]:
    """Verify all old .005 event diagnostics and the complete weighted summary."""
    ids = tuple(event_ids)
    weights = tuple(float(w) for w in event_weights)
    lengths = tuple(counts)
    if len(ids) != len(set(ids)) or len(ids) != len(weights) or len(ids) != len(lengths):
        raise ValueError("event identities, weights, and point counts must be unique and aligned")
    if any(not math.isfinite(w) or w < 0 for w in weights) or abs(math.fsum(weights) - 1.0) > 1e-12:
        raise ValueError("event weights must be finite, nonnegative, and sum to one")
    expected_ids = set(ids)
    per_event: list[dict[str, Any]] = []
    for event_id in ids:
        if event_id not in rebuilt_diagnostics or event_id not in saved_diagnostics:
            per_event.append({"event_id": event_id, "mismatches": [
                {"path": "$", "reason": "missing_event_diagnostic"}]})
            continue
        rebuilt = rebuilt_diagnostics[event_id]
        saved = saved_diagnostics[event_id]
        per_event.append({"event_id": event_id,
                          "mismatches": compare_recursive(saved, rebuilt, atol=atol)})
    extras = sorted((set(rebuilt_diagnostics) | set(saved_diagnostics)) - expected_ids)
    summary_mismatches = compare_recursive(saved_summary, rebuilt_summary, atol=atol)
    qualified = (not extras and not summary_mismatches
                 and all(not item["mismatches"] for item in per_event)
                 and set(rebuilt_diagnostics) == expected_ids
                 and set(saved_diagnostics) == expected_ids)
    return {"qualified": qualified, "absolute_tolerance": atol,
            "event_count": len(ids), "event_mismatch_count": sum(bool(x["mismatches"]) for x in per_event),
            "extra_event_ids": extras, "per_event": per_event,
            "summary_mismatches": summary_mismatches}


def _weighted_summary(event_ids: Sequence[str], weights: Sequence[float], counts: Sequence[int],
                      rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    failures = [event_id for event_id in event_ids if rows[event_id].get("status") != "ok"]
    if failures:
        return {"complete_population": False, "failed_event_ids": failures,
                "main_nrmse": None, "duration_buckets": None}
    main_mse = math.fsum(w * rows[e]["diagnostic"]["event_mse"]
                         for e, w in zip(event_ids, weights, strict=True))
    horizon = {h: math.sqrt(math.fsum(w * rows[e]["diagnostic"]["horizon_scores"][h]["normalized_mse"]
                                      for e, w in zip(event_ids, weights, strict=True)))
               for h in ("1.0", "3.0")}
    buckets = {}
    predicates = (("through_1s", lambda n: n <= 100),
                  ("between_1s_and_3s", lambda n: 100 < n < 300),
                  ("scored_at_3s", lambda n: n == 300))
    for name, predicate in predicates:
        indexes = [i for i, n in enumerate(counts) if predicate(n)]
        mass = math.fsum(weights[i] for i in indexes)
        contribution = math.fsum(weights[i] * rows[event_ids[i]]["diagnostic"]["event_mse"]
                                 for i in indexes)
        buckets[name] = {"event_count": len(indexes), "original_weight_mass": mass,
                         "same_denominator_normalized_mse_contribution": contribution,
                         "conditional_normalized_mse": contribution / mass if mass else None}
    if abs(math.fsum(x["same_denominator_normalized_mse_contribution"] for x in buckets.values())
           - main_mse) > ATOL:
        raise ArithmeticError("duration contributions do not reconstruct the population MSE")
    return {"complete_population": True, "event_count": len(event_ids), "main_mse": main_mse,
            "main_nrmse": math.sqrt(main_mse), "horizon_nrmse": horizon,
            "duration_buckets": buckets}


def summarize_refinement_audit(
    *, event_ids: Sequence[str], event_weights: Sequence[float], counts: Sequence[int],
    metadata: Mapping[str, Mapping[str, Any]], rows: Mapping[tuple[int, float, str], Mapping[str, Any]],
    refinements: Mapping[tuple[int, str], Mapping[str, Any]],
    source_checks: Mapping[str, Mapping[str, Any]], old_replay_checks: Mapping[str, Mapping[str, Any]],
    budget_complete: bool,
) -> dict[str, Any]:
    """Rebuild the complete A10 verdict and event/line support gate.

    ``rows`` contains all (seed, step, event) records for steps .005 and .0025.
    Each refinement record supplies the independently measured visible-path
    difference, rule/time checks, and fixed-window margin comparison.
    """
    ids = tuple(event_ids)
    weights = tuple(float(w) for w in event_weights)
    lengths = tuple(counts)
    if len(ids) != 258 or len(set(ids)) != len(ids):
        raise ValueError("A10 requires all 258 distinct eligible train event IDs")
    if len(weights) != len(ids) or len(lengths) != len(ids):
        raise ValueError("event weights and future counts must align with all event IDs")
    if any(not math.isfinite(w) or w < 0 for w in weights) or abs(math.fsum(weights) - 1.0) > ATOL:
        raise ValueError("event weights must be finite, nonnegative, and sum to one")
    expected_rows = {(seed, step, event_id) for seed in SEEDS for step in STEPS for event_id in ids}
    expected_refinements = {(seed, event_id) for seed in SEEDS for event_id in ids}
    missing_rows = sorted(expected_rows - set(rows))
    extra_rows = sorted(set(rows) - expected_rows)
    missing_refinements = sorted(expected_refinements - set(refinements))
    extra_refinements = sorted(set(refinements) - expected_refinements)
    groups: dict[str, Any] = {}
    row_checks: list[dict[str, Any]] = []
    all_rows_pass = not missing_rows and not extra_rows
    for seed in SEEDS:
        for step in STEPS:
            population = {}
            for event_id in ids:
                key = (seed, step, event_id)
                row = rows.get(key)
                if row is None:
                    all_rows_pass = False
                    continue
                if row.get("status") != "ok" or row.get("physical", {}).get("qualified") is not True:
                    all_rows_pass = False
                    row_checks.append({"seed": seed, "step": step, "event_id": event_id,
                                       "reason": "path_or_physical_check_failed"})
                bound_failures = _diagnostic_margin_failures(row.get("diagnostic"))
                if bound_failures:
                    all_rows_pass = False
                    row_checks.append({"seed": seed, "step": step, "event_id": event_id,
                                       "reason": "invalid_true_path_margin_bounds",
                                       "details": bound_failures})
                population[event_id] = row
            if len(population) == len(ids):
                groups[f"{seed}-{step:g}"] = _weighted_summary(ids, weights, lengths, population)
            else:
                groups[f"{seed}-{step:g}"] = {"complete_population": False,
                                                "failed_event_count": len(ids) - len(population)}

    refinement_failures = []
    all_refinements_pass = not missing_refinements and not extra_refinements
    for key in sorted(expected_refinements):
        item = refinements.get(key)
        if item is None:
            continue
        seed, event_id = key
        difference = item.get("max_normalized_difference")
        numeric_ok = (isinstance(difference, (int, float)) and not isinstance(difference, bool)
                      and math.isfinite(float(difference)) and difference <= REFINEMENT_LIMIT)
        same_rule = item.get("same_rule_decision") is True
        time_ok = item.get("trigger_within_two_points") is True
        coarse_row = rows.get((seed, 0.005, event_id), {})
        fine_row = rows.get((seed, 0.0025, event_id), {})
        margin_failures = _coarse_fine_margin_failures(
            item, coarse_row.get("diagnostic"), fine_row.get("diagnostic"))
        margin_ok = not margin_failures
        ok = numeric_ok and same_rule and time_ok and margin_ok
        if not ok:
            all_refinements_pass = False
            refinement_failures.append({"seed": seed, "event_id": event_id,
                                        "max_normalized_difference": difference,
                                        "numeric_ok": numeric_ok, "same_rule_decision": same_rule,
                                        "trigger_within_two_points": time_ok,
                                        "margin_bounds_verified": margin_ok,
                                        "margin_bound_failures": margin_failures})

    complete_source = bool(source_checks) and all(v.get("qualified") is True for v in source_checks.values())
    complete_old_replay = bool(old_replay_checks) and all(v.get("qualified") is True
                                                          for v in old_replay_checks.values())
    complete_population_physical = all_rows_pass
    audit_pass = bool(budget_complete and complete_source and complete_old_replay
                      and complete_population_physical and all_refinements_pass)

    supported_events = []
    for event_id in ids:
        meta = metadata.get(event_id)
        if not isinstance(meta, Mapping) or not isinstance(meta.get("fault_line"), str):
            continue
        qualifies = True
        for seed in SEEDS:
            for step in STEPS:
                row = rows.get((seed, step, event_id))
                if row is None or row.get("status") != "ok":
                    qualifies = False
                    continue
                diagnostic = row.get("diagnostic", {})
                history_truth = diagnostic.get("history_rho", {}).get("truth", {})
                future_truth = diagnostic.get("future_rho", {}).get("truth", {})
                future_pred = diagnostic.get("future_rho", {}).get("prediction", {})
                qualifies &= (history_truth.get("triggered") is False
                              and isinstance(future_truth.get("rho"), (int, float))
                              and not isinstance(future_truth.get("rho"), bool)
                              and future_truth["rho"] > 1e-12
                              and future_pred.get("triggered") is False)
            fine = rows.get((seed, 0.0025, event_id))
            excess = (fine or {}).get("diagnostic", {}).get(
                "representative_window_excess_over_event_mean_C_minus_pM")
            qualifies &= (isinstance(excess, (int, float)) and not isinstance(excess, bool)
                          and math.isfinite(float(excess)) and excess > 1e-12)
        if qualifies:
            supported_events.append({"event_id": event_id, **meta})
    lines = sorted({row["fault_line"] for row in supported_events})
    support_pass = bool(audit_pass and len(supported_events) >= 2 and len(lines) >= 2)
    per_seed = {}
    for seed in SEEDS:
        fine_rows = [rows.get((seed, 0.0025, event_id), {}) for event_id in ids]
        observed = [row for row in fine_rows if row.get("status") == "ok"
                    and row.get("diagnostic", {}).get("observed_rule_hit") is True]
        per_seed[str(seed)] = {
            "observed_positive_new_rule_count": len(observed),
            "observed_positive_not_preserved_event_ids": [
                row["event_id"] for row in observed
                if row.get("diagnostic", {}).get("prediction_rule_hit") is not True
            ],
            "predicted_trigger_without_observed_trigger_count": sum(
                row.get("status") == "ok"
                and row.get("diagnostic", {}).get("prediction_rule_hit") is True
                and row.get("diagnostic", {}).get("observed_rule_hit") is False
                for row in fine_rows
            ),
        }
    return {
        "audit_all_checks_passed": audit_pass,
        "scientific_verdict": "ACCEPT" if audit_pass else "REJECT",
        "promotion_decision": "KEEP_BASELINE", "model_promoted": False,
        "new_training_authorized": False, "nonlinear_memory_supported_by_this_audit": False,
        "selected_candidate": None,
        "time_aggregation_proposal_supported": support_pass,
        "per_seed_fine_rule_diagnostics": per_seed,
        "time_local_error_evidence_events": supported_events,
        "supporting_events": supported_events, "distinct_physical_fault_lines": lines,
        "classification_fsr_computed": False,
        "weighted_trajectory_summaries": groups,
        "coverage": {"expected_rows": len(expected_rows), "present_rows": len(rows),
                     "missing_rows": len(missing_rows), "extra_rows": len(extra_rows),
                     "expected_refinements": len(expected_refinements),
                     "present_refinements": len(refinements),
                     "missing_refinements": len(missing_refinements),
                     "extra_refinements": len(extra_refinements)},
        "gates": {"budget_complete": bool(budget_complete), "source_checks_pass": complete_source,
                  "old_replay_checks_pass": complete_old_replay,
                  "all_774_paths_physical_and_status_pass": complete_population_physical,
                  "all_path_and_refinement_margin_bounds_pass": (
                      all_rows_pass and all_refinements_pass),
                  "all_774_refinements_pass": all_refinements_pass,
                  "support_event_count": len(supported_events),
                  "support_line_count": len(lines)},
        "row_failures": row_checks, "refinement_failures": refinement_failures,
    }


def summarize_refinement(
    event_ids: Sequence[str], metadata: Mapping[str, Mapping[str, Any]],
    rows: Mapping[tuple[int, float, str], Mapping[str, Any]],
    refinements: Mapping[tuple[int, str], Mapping[str, Any]],
    source_checks: Mapping[str, Mapping[str, Any]], *, budget_complete: bool,
    event_weights: Sequence[float] | None = None, counts: Sequence[int] | None = None,
    old_replay_checks: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Driver-facing API; source_checks may include each required old replay check."""
    ids = tuple(event_ids)
    if event_weights is None:
        event_weights = [1.0 / len(ids)] * len(ids) if ids else []
    if counts is None:
        counts = [int(rows[(SEEDS[0], STEPS[0], event_id)]["diagnostic"]["future_points"])
                  if (SEEDS[0], STEPS[0], event_id) in rows else 0 for event_id in ids]
    checks = old_replay_checks if old_replay_checks is not None else source_checks
    return summarize_refinement_audit(
        event_ids=ids, event_weights=event_weights, counts=counts, metadata=metadata,
        rows=rows, refinements=refinements, source_checks=source_checks,
        old_replay_checks=checks, budget_complete=budget_complete,
    )
