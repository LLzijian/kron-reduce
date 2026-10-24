"""Mechanical A11 validity, utility, mechanism, and campaign decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

SEEDS = (324, 325, 326)
GROUPS = ("mean", "time")
STEPS = ("0.005", "0.0025")
LOCAL_METRICS = (
    "weighted_window_error_density",
    "weighted_E_atom",
    "weighted_absolute_rho_difference",
)
_RECORD_NUMBERS = (
    "val_main_nrmse", "val_complete3_nrmse", "fn", "fp", "tp", "tn",
    "unresolved", "development_evaluation_refusals", "eligible_events",
)
_BASELINE_NUMBERS = (
    "original_main_nrmse", "original_complete3_nrmse", "main_nrmse",
    "complete3_nrmse", "fn", "fp",
)


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _nonnegative(value: Any) -> bool:
    return _number(value) and value >= 0


def _count(value: Any) -> bool:
    return _nonnegative(value) and int(value) == value


def _record_ok(row: Any) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not isinstance(row, Mapping):
        return False, ["record is missing or not a mapping"]
    for key in _RECORD_NUMBERS:
        value = row.get(key)
        if not _number(value):
            reasons.append(f"{key} is missing or non-finite")
        elif key in {"fn", "fp", "tp", "tn", "unresolved", "development_evaluation_refusals", "eligible_events"} and not _count(value):
            reasons.append(f"{key} is not a nonnegative integer")
        elif key in {"val_main_nrmse", "val_complete3_nrmse"} and value < 0:
            reasons.append(f"{key} is negative")
    if reasons:
        return False, reasons
    if row["eligible_events"] != 88:
        reasons.append("eligible_events is not 88")
    if row["tp"] + row["fn"] != 14:
        reasons.append("val confusion counts do not contain 14 unstable events")
    if row["tn"] + row["fp"] != 74:
        reasons.append("val confusion counts do not contain 74 stable events")
    if row["unresolved"] != 0:
        reasons.append("unresolved events are nonzero")
    if row["development_evaluation_refusals"] != 0:
        reasons.append("development evaluation refusals are nonzero")
    for container_key in ("train_main_mse_by_step", "local_metrics_by_step"):
        container = row.get(container_key)
        if not isinstance(container, Mapping):
            reasons.append(f"{container_key} is missing or not a mapping")
            continue
        for step in STEPS:
            step_values = container.get(step)
            if container_key == "train_main_mse_by_step":
                if not _nonnegative(step_values):
                    reasons.append(f"{container_key}[{step}] is missing, negative, or non-finite")
            elif not isinstance(step_values, Mapping):
                reasons.append(f"{container_key}[{step}] is missing or not a mapping")
            else:
                for metric in LOCAL_METRICS:
                    if not _nonnegative(step_values.get(metric)):
                        reasons.append(f"{container_key}[{step}].{metric} is missing, negative, or non-finite")
    return not reasons, reasons


def _baseline_ok(row: Any) -> tuple[bool, list[str]]:
    if not isinstance(row, Mapping):
        return False, ["baseline is missing or not a mapping"]
    reasons = [f"{key} is missing, negative, or non-finite" for key in _BASELINE_NUMBERS if not _nonnegative(row.get(key))]
    return not reasons, reasons


def _utility(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return (
        row["val_main_nrmse"] <= baseline["original_main_nrmse"]
        and row["val_main_nrmse"] <= baseline["main_nrmse"]
        and row["val_complete3_nrmse"] <= 1.05 * baseline["original_complete3_nrmse"]
        and row["val_complete3_nrmse"] <= 1.05 * baseline["complete3_nrmse"]
        and row["fn"] <= baseline["fn"]
        and row["fn"] <= 1
        and row["fp"] == 0
    )


def _mechanism_seed(time: Mapping[str, Any], mean: Mapping[str, Any]) -> tuple[bool, bool, list[str]]:
    reasons: list[str] = []
    detection_improved = time["fn"] < mean["fn"]
    for step in STEPS:
        for metric in LOCAL_METRICS:
            if mean["local_metrics_by_step"][step][metric] - time["local_metrics_by_step"][step][metric] <= 1e-12:
                reasons.append(f"{step} {metric} improvement is not greater than 1e-12")
    if time["train_main_mse_by_step"]["0.005"] > mean["train_main_mse_by_step"]["0.005"]:
        reasons.append("0.005 train main MSE increased")
    if time["val_main_nrmse"] > mean["val_main_nrmse"]:
        reasons.append("val main NRMSE increased against mean")
    if time["val_complete3_nrmse"] > 1.05 * mean["val_complete3_nrmse"]:
        reasons.append("complete3 NRMSE exceeds 1.05 times mean")
    if time["fn"] > mean["fn"]:
        reasons.append("FN increased against mean")
    if time["fp"] > mean["fp"]:
        reasons.append("FP increased against mean")
    return not reasons, detection_improved, reasons


def outcome(
    records: Mapping[tuple[str, int], Mapping[str, Any]],
    baselines: Mapping[int, Mapping[str, Any]],
    *,
    validity_gates: Mapping[str, bool],
) -> dict[str, Any]:
    """Return the frozen campaign decision; malformed evidence fails closed."""
    reasons: list[str] = []
    expected_records = {(group, seed) for group in GROUPS for seed in SEEDS}
    if not isinstance(records, Mapping):
        records = {}
        reasons.append("records is not a mapping")
    if set(records) != expected_records:
        reasons.append("records do not contain exactly the six required group/seed rows")
    if not isinstance(baselines, Mapping):
        baselines = {}
        reasons.append("baselines is not a mapping")
    if set(baselines) != set(SEEDS):
        reasons.append("baselines do not contain exactly the three required seeds")
    if not isinstance(validity_gates, Mapping) or not validity_gates:
        gates_ok = False
        reasons.append("validity_gates must be a nonempty mapping")
    else:
        gates_ok = all(type(value) is bool and value is True for value in validity_gates.values())
        if not gates_ok:
            reasons.append("one or more validity gates are false or not boolean true")

    checked_records: dict[tuple[str, int], Mapping[str, Any]] = {}
    checked_baselines: dict[int, Mapping[str, Any]] = {}
    structural_ok = set(records) == expected_records and set(baselines) == set(SEEDS)
    for key in expected_records:
        ok, row_reasons = _record_ok(records.get(key))
        if not ok:
            structural_ok = False
            reasons.extend([f"{key}: {reason}" for reason in row_reasons])
        else:
            checked_records[key] = records[key]
    for seed in SEEDS:
        ok, row_reasons = _baseline_ok(baselines.get(seed))
        if not ok:
            structural_ok = False
            reasons.extend([f"baseline {seed}: {reason}" for reason in row_reasons])
        else:
            checked_baselines[seed] = baselines[seed]

    validity_passed = gates_ok and structural_ok
    group_results: dict[str, Any] = {}
    for group in GROUPS:
        local_utility = validity_passed and all(
            _utility(checked_records[(group, seed)], checked_baselines[seed])
            for seed in SEEDS
        )
        per_seed = {}
        if validity_passed:
            for seed in SEEDS:
                row = checked_records[(group, seed)]
                base = checked_baselines[seed]
                per_seed[seed] = {
                    "utility_passed": _utility(row, base),
                    "zero_missed_instability": row["fn"] == 0 and row["fp"] == 0,
                    "fn_below_source": row["fn"] < base["fn"],
                }
        group_results[group] = {
            "utility_passed": local_utility,
            "zero": local_utility and all(
                checked_records[(group, seed)]["fn"] == 0 and checked_records[(group, seed)]["fp"] == 0
                for seed in SEEDS
            ) if validity_passed else False,
            "per_seed": per_seed,
        }

    mechanism_reasons: dict[int, list[str]] = {}
    mechanism_per_seed: dict[int, bool] = {}
    detection_improved = False
    if validity_passed:
        for seed in SEEDS:
            ok, detected, why = _mechanism_seed(
                checked_records[("time", seed)], checked_records[("mean", seed)]
            )
            mechanism_per_seed[seed] = ok
            mechanism_reasons[seed] = why
            detection_improved = detection_improved or detected
    mechanism_supported = validity_passed and group_results["time"]["utility_passed"] and all(mechanism_per_seed.values())

    zero_groups = [group for group in GROUPS if group_results[group]["zero"]]
    same_group_retry = validity_passed and any(
        group_results[group]["utility_passed"]
        and any(checked_records[(group, seed)]["fn"] < checked_baselines[seed]["fn"] for seed in SEEDS)
        for group in GROUPS
    )
    if zero_groups:
        campaign_verdict = "ACCEPT"
    elif same_group_retry:
        campaign_verdict = "RETRY"
    else:
        campaign_verdict = "REJECT"

    return {
        "validity_passed": validity_passed,
        "validity_reasons": reasons,
        "group_results": group_results,
        "mechanism_supported": mechanism_supported,
        "mechanism_per_seed": mechanism_per_seed,
        "mechanism_reasons": mechanism_reasons,
        "raw_fn_improved_against_mean": detection_improved,
        "detection_improved": bool(mechanism_supported and detection_improved),
        "zero_groups": zero_groups,
        "campaign_verdict": campaign_verdict,
        "selected_candidate": None,
        "promotion_decision": "KEEP_BASELINE",
        "model_promoted": False,
        "new_training_authorized": False,
    }
