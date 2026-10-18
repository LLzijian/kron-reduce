"""Matched P2 trajectory scales and aggregation primitives."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class P2FitScales:
    delta: tuple[float, ...]
    omega: tuple[float, ...]
    event_ids_sha256: str
    algorithm: str = "source_line_event_equal_rms_change_from_clearing_v1"

    def __post_init__(self) -> None:
        if len(self.delta) != 10 or len(self.omega) != 10:
            raise ValueError("P2 scales require ten-machine coordinates")
        values = np.asarray((*self.delta, *self.omega), dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("P2 scales must be finite and positive")
        if len(self.event_ids_sha256) != 64:
            raise ValueError("P2 scales require a fit-event hash")

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _sample_fields(sample: Any) -> tuple[str, str, str, np.ndarray, np.ndarray]:
    if hasattr(sample, "prediction_input"):
        event_id = str(sample.event_id)
        source = str(sample.source_dataset)
        line = str(sample.fault_line)
        delta = np.asarray(sample.target_delta, dtype=float)
        omega = np.asarray(sample.target_omega, dtype=float)
    else:
        event_id = str(sample["event_id"])
        context = sample["training_context"]
        metadata = context["metadata"]
        source = str(metadata["source_dataset"])
        line = str(metadata["fault_line"])
        targets = context["targets"]
        delta = np.asarray(targets["delta"], dtype=float)
        omega = np.asarray(targets["omega"], dtype=float)
    if not event_id or not source or not line:
        raise ValueError("P2 scale sample lacks event/source/line identity")
    if delta.shape != (1001, 10) or omega.shape != delta.shape:
        raise ValueError("P2 scale sample must contain complete ten-machine trajectories")
    if not np.isfinite(delta).all() or not np.isfinite(omega).all():
        raise ValueError("P2 scale sample is nonfinite")
    return event_id, source, line, delta, omega


def fit_scales(
    samples: Sequence[Any],
    *,
    delta_floor: float = 1e-6,
    omega_floor: float = 1e-6,
) -> P2FitScales:
    """Compute source→line→event equal RMS dynamic ranges from fit only."""
    if not samples or delta_floor <= 0.0 or omega_floor <= 0.0:
        raise ValueError("P2 scale calculation needs samples and positive floors")
    by_line: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    ids: list[str] = []
    for sample in samples:
        event_id, source, line, delta, omega = _sample_fields(sample)
        ids.append(event_id)
        by_line[(source, line)].append(
            (
                np.mean((delta - delta[[0]]) ** 2, axis=0),
                np.mean((omega - omega[[0]]) ** 2, axis=0),
            )
        )
    if len(ids) != len(set(ids)):
        raise ValueError("P2 scale fit events must be unique")
    by_source: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for (source, _line), rows in by_line.items():
        by_source[source].append(
            (
                np.mean([row[0] for row in rows], axis=0),
                np.mean([row[1] for row in rows], axis=0),
            )
        )
    if len(by_source) < 2:
        raise ValueError("P2 scales require both registered sources")
    source_moments = [
        (
            np.mean([row[0] for row in lines], axis=0),
            np.mean([row[1] for row in lines], axis=0),
        )
        for lines in by_source.values()
    ]
    delta = np.maximum(np.sqrt(np.mean([row[0] for row in source_moments], axis=0)), delta_floor)
    omega = np.maximum(np.sqrt(np.mean([row[1] for row in source_moments], axis=0)), omega_floor)
    ids_hash = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
    return P2FitScales(tuple(delta.tolist()), tuple(omega.tolist()), ids_hash)


def event_metrics(
    *,
    event_id: str,
    source_dataset: str,
    fault_line: str,
    horizon_s: int,
    true_delta: np.ndarray,
    true_omega: np.ndarray,
    pred_delta: np.ndarray,
    pred_omega: np.ndarray,
    scales: P2FitScales,
) -> dict[str, Any]:
    prefix = {1: 101, 5: 501, 10: 1001}.get(horizon_s)
    if prefix is None:
        raise ValueError("P2 reports only 1, 5 and 10 seconds")
    arrays = [
        np.asarray(value, dtype=float)[:prefix]
        for value in (true_delta, true_omega, pred_delta, pred_omega)
    ]
    if any(value.shape != (prefix, 10) for value in arrays):
        raise ValueError("P2 metric arrays must cover the registered horizon")
    if not all(np.isfinite(value).all() for value in arrays):
        return {
            "event_id": event_id,
            "source_dataset": source_dataset,
            "fault_line": fault_line,
            "horizon_s": horizon_s,
            "status": "failed",
            "failure_reason": "nonfinite_trajectory",
        }
    td, tw, pd, pw = arrays
    delta_error, omega_error = pd - td, pw - tw
    delta_scale = np.asarray(scales.delta)[None, :]
    omega_scale = np.asarray(scales.omega)[None, :]
    delta_sse = float(np.sum(delta_error**2))
    omega_sse = float(np.sum(omega_error**2))
    delta_sst = float(np.sum((td - np.mean(td, axis=0, keepdims=True)) ** 2))
    omega_sst = float(np.sum((tw - np.mean(tw, axis=0, keepdims=True)) ** 2))
    return {
        "event_id": event_id,
        "source_dataset": source_dataset,
        "fault_line": fault_line,
        "horizon_s": horizon_s,
        "status": "ok",
        "delta_rmse": float(np.sqrt(np.mean(delta_error**2))),
        "omega_rmse": float(np.sqrt(np.mean(omega_error**2))),
        "normalized_joint_rmse": float(
            np.sqrt(
                0.5
                * (
                    np.mean((delta_error / delta_scale) ** 2)
                    + np.mean((omega_error / omega_scale) ** 2)
                )
            )
        ),
        "delta_sse": delta_sse,
        "delta_sst": delta_sst,
        "omega_sse": omega_sse,
        "omega_sst": omega_sst,
        "scored_values_per_channel": prefix * 10,
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("P2 aggregation requires registered events")
    ids = [str(row.get("event_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("P2 aggregate rows require unique event IDs")
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    success_scores = np.asarray(
        [float(row["normalized_joint_rmse"]) for row in successful], dtype=float
    )
    if not np.isfinite(success_scores).all():
        raise ValueError("P2 successful rows contain nonfinite scores")
    primary_available = not failed
    tail_count = max(1, math.ceil(0.10 * len(successful))) if successful else 0
    tail = (
        sorted(successful, key=lambda row: float(row["normalized_joint_rmse"]))[-tail_count:]
        if tail_count
        else []
    )
    weights = _hierarchical_row_map(successful) if successful else {}
    delta_sse = sum(weights[str(row["event_id"])] * float(row["delta_sse"]) for row in successful)
    delta_sst = sum(weights[str(row["event_id"])] * float(row["delta_sst"]) for row in successful)
    omega_sse = sum(weights[str(row["event_id"])] * float(row["omega_sse"]) for row in successful)
    omega_sst = sum(weights[str(row["event_id"])] * float(row["omega_sst"]) for row in successful)
    scored = sum(
        weights[str(row["event_id"])] * int(row["scored_values_per_channel"]) for row in successful
    )
    weighted_mean = sum(
        weights[str(row["event_id"])] * float(row["normalized_joint_rmse"]) for row in successful
    )
    weighted_median = (
        _weighted_median(
            success_scores, np.asarray([weights[str(row["event_id"])] for row in successful])
        )
        if successful
        else None
    )
    return {
        "event_count": len(rows),
        "successful_event_count": len(successful),
        "failed_event_count": len(failed),
        "failure_rate": len(failed) / len(rows),
        "primary_metrics_available": primary_available,
        "mean_normalized_joint_rmse": float(weighted_mean) if primary_available else None,
        "median_normalized_joint_rmse": float(weighted_median) if primary_available else None,
        "pooled_delta_rmse": float(np.sqrt(delta_sse / scored))
        if primary_available and scored
        else None,
        "pooled_omega_rmse": float(np.sqrt(omega_sse / scored))
        if primary_available and scored
        else None,
        "pooled_delta_r2": float(1.0 - delta_sse / delta_sst)
        if primary_available and delta_sst > 0
        else None,
        "pooled_omega_r2": float(1.0 - omega_sse / omega_sst)
        if primary_available and omega_sst > 0
        else None,
        "worst_10pct_mean_normalized_joint_rmse": (
            float(np.mean([float(row["normalized_joint_rmse"]) for row in tail]))
            if primary_available and tail
            else None
        ),
        "worst_10pct_event_ids": [str(row["event_id"]) for row in tail]
        if primary_available
        else [],
        "failed_event_ids": [str(row["event_id"]) for row in failed],
        "pooled_r2_centering": "within_event_within_node_over_time",
        "population_weighting": "source_line_event_equal",
        "failures_imputed_as_zero": False,
    }


def _hierarchical_row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        event_id = str(row.get("event_id", ""))
        source = str(row.get("source_dataset", ""))
        line = str(row.get("fault_line", ""))
        if not event_id or not source or not line:
            raise ValueError("P2 hierarchical metrics require event/source/fault-line keys")
        groups[source][line].append(event_id)
    return {
        event_id: 1.0 / len(groups) / len(lines) / len(event_ids)
        for lines in groups.values()
        for event_ids in lines.values()
        for event_id in event_ids
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(ordered_values[int(np.searchsorted(cumulative, 0.5, side="left"))])


def stratified_fault_line_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    draws: int = 5000,
    seed: int = 324,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Source-stratified bootstrap that resamples complete fault-line clusters."""

    if draws <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("P2 bootstrap settings are invalid")
    failed_ids = [str(row.get("event_id", "")) for row in rows if row.get("status") != "ok"]
    if failed_ids:
        return {
            "status": "UNDETERMINED",
            "reason": "failed_rollouts_are_not_imputed",
            "failed_event_ids": failed_ids,
        }
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(metric)
        if row.get("status") != "ok" or value is None or not np.isfinite(float(value)):
            continue
        by_source[str(row["source_dataset"])][str(row["fault_line"])].append(float(value))
    if not by_source:
        return {"status": "UNDETERMINED", "reason": "no_finite_complete_rows"}
    line_counts = {source: len(lines) for source, lines in by_source.items()}
    if any(count < 3 for count in line_counts.values()):
        return {
            "status": "UNDETERMINED",
            "reason": "fewer_than_three_fault_lines_in_a_source",
            "fault_line_counts_by_source": line_counts,
        }
    line_means = {
        source: np.asarray([np.mean(lines[line]) for line in sorted(lines)], dtype=np.float64)
        for source, lines in by_source.items()
    }
    observed = float(np.mean([np.mean(values) for values in line_means.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        source_values = []
        for values in line_means.values():
            selected = rng.integers(0, len(values), size=len(values))
            source_values.append(float(np.mean(values[selected])))
        samples[draw] = float(np.mean(source_values))
    tail = (1.0 - confidence) / 2.0
    return {
        "status": "completed",
        "estimand": "source_equal_mean_of_fault_line_equal_event_means",
        "metric": metric,
        "estimate": observed,
        "ci_lower": float(np.quantile(samples, tail)),
        "ci_upper": float(np.quantile(samples, 1.0 - tail)),
        "confidence": confidence,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "fault_line_counts_by_source": line_counts,
        "resampling_unit": "fault_line_with_all_events_preserved",
    }


def paired_stratified_fault_line_bootstrap(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    draws: int = 5000,
    seed: int = 324,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap candidate-minus-reference paired differences by complete fault line."""

    if draws <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("P2 paired-bootstrap settings are invalid")
    reference = {str(row.get("event_id", "")): row for row in reference_rows}
    candidate = {str(row.get("event_id", "")): row for row in candidate_rows}
    if (
        not reference
        or set(reference) != set(candidate)
        or len(reference) != len(reference_rows)
        or len(candidate) != len(candidate_rows)
    ):
        raise ValueError("P2 paired bootstrap requires identical unique event populations")
    failed_ids = sorted(
        event_id
        for event_id in reference
        if reference[event_id].get("status") != "ok" or candidate[event_id].get("status") != "ok"
    )
    if failed_ids:
        return {
            "status": "UNDETERMINED",
            "reason": "failed_rollouts_are_not_imputed",
            "failed_event_ids": failed_ids,
        }
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for event_id, reference_row in reference.items():
        candidate_row = candidate[event_id]
        reference_stratum = (
            str(reference_row.get("source_dataset", "")),
            str(reference_row.get("fault_line", "")),
        )
        candidate_stratum = (
            str(candidate_row.get("source_dataset", "")),
            str(candidate_row.get("fault_line", "")),
        )
        if reference_stratum != candidate_stratum or not all(reference_stratum):
            raise ValueError("P2 paired bootstrap strata differ between methods")
        left, right = reference_row.get(metric), candidate_row.get(metric)
        if left is None or right is None or not np.isfinite([float(left), float(right)]).all():
            return {
                "status": "UNDETERMINED",
                "reason": "paired_metric_is_missing_or_nonfinite",
                "event_id": event_id,
            }
        source, line = reference_stratum
        by_source[source][line].append(float(right) - float(left))
    line_counts = {source: len(lines) for source, lines in by_source.items()}
    if any(count < 3 for count in line_counts.values()):
        return {
            "status": "UNDETERMINED",
            "reason": "fewer_than_three_fault_lines_in_a_source",
            "fault_line_counts_by_source": line_counts,
        }
    line_differences = {
        source: np.asarray([np.mean(lines[line]) for line in sorted(lines)], dtype=np.float64)
        for source, lines in by_source.items()
    }
    estimate = float(np.mean([np.mean(values) for values in line_differences.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        source_values = []
        for values in line_differences.values():
            selected = rng.integers(0, len(values), size=len(values))
            source_values.append(float(np.mean(values[selected])))
        samples[draw] = float(np.mean(source_values))
    tail = (1.0 - confidence) / 2.0
    return {
        "status": "completed",
        "estimand": "candidate_minus_reference_source_equal_fault_line_equal_event_mean",
        "metric": metric,
        "estimate": estimate,
        "ci_lower": float(np.quantile(samples, tail)),
        "ci_upper": float(np.quantile(samples, 1.0 - tail)),
        "confidence": confidence,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "fault_line_counts_by_source": line_counts,
        "resampling_unit": "paired_fault_line_with_all_events_preserved",
    }
