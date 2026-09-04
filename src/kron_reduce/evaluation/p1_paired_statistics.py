"""Preregistered P1 paired fault-line summaries (descriptive only)."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from kron_reduce.data.p1_causal import P1RoleManifest

_FIELDS = {"event_id", "source_dataset", "fault_line", "horizon_seconds", "output", "status", "reference_rmse", "candidate_rmse"}
_HORIZONS, _OUTPUTS = {1, 5, 10}, {"delta", "omega"}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()).hexdigest()


def view_key(source: str | None, horizon: int, output: str) -> str:
    return json.dumps([source, horizon, output], separators=(",", ":"))


def _mean_pair(pairs: Sequence[tuple[float, float]]) -> tuple[float, float]:
    try:
        with np.errstate(over="raise", invalid="raise"):
            reference, candidate = np.mean(pairs, axis=0)
    except FloatingPointError as error:
        raise ValueError("nonfinite paired mean") from error
    if not np.isfinite([reference, candidate]).all():
        raise ValueError("nonfinite paired mean")
    return float(reference), float(candidate)


def _point(pairs: Sequence[tuple[float, float]]) -> dict[str, float | None]:
    reference, candidate = _mean_pair(pairs)
    difference = candidate - reference
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative = difference / reference if reference != 0 else None
    return {"reference_mean": float(reference), "candidate_mean": float(candidate), "absolute_difference": float(difference), "relative_degradation": None if relative is None or not np.isfinite(relative) else float(relative), "nonfinite_relative_reason": None if relative is None or np.isfinite(relative) else "nonfinite_relative_point"}


def _tails(events: list[dict[str, Any]]) -> dict[str, Any]:
    count = math.ceil(.1 * len(events))
    result = {}
    for metric in ("reference_rmse", "candidate_rmse"):
        selected = sorted(events, key=lambda row: (-row[metric], row["event_id"]))[:count]
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean = float(np.mean([row[metric] for row in selected]))
        except FloatingPointError as error:
            raise ValueError("nonfinite tail mean") from error
        if not np.isfinite(mean):
            raise ValueError("nonfinite tail mean")
        result[metric] = {"event_ids": [row["event_id"] for row in selected], "mean": mean}
    return result


def _ci(samples: np.ndarray, point: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not np.isfinite(samples).all(): raise ValueError("nonfinite bootstrap mean")
    try:
        with np.errstate(over="raise", invalid="raise"):
            absolute = samples[:, 1] - samples[:, 0]
    except FloatingPointError as error:
        raise ValueError("nonfinite bootstrap difference") from error
    if not np.isfinite(absolute).all(): raise ValueError("nonfinite bootstrap difference")
    absolute_ci = np.quantile(absolute, [.025, .975])
    if not np.isfinite(absolute_ci).all(): raise ValueError("nonfinite bootstrap CI")
    ci = {"absolute_CI": [float(x) for x in absolute_ci]}
    zero = samples[:, 0] == 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"): relative = absolute / samples[:, 0]
    if point["relative_degradation"] is None or np.any(zero) or not np.isfinite(relative).all():
        ci["relative_CI"] = None
        ci["relative_CI_reason"] = point["nonfinite_relative_reason"] or ("zero_reference_point" if point["reference_mean"] == 0 else "zero_reference_bootstrap_denominator" if np.any(zero) else "nonfinite_relative_bootstrap")
        ci["relative_CI_zero_denominator_count"] = int(np.sum(zero))
    else:
        relative_ci = np.quantile(relative, [.025, .975])
        if not np.isfinite(relative_ci).all(): raise ValueError("nonfinite relative CI")
        ci["relative_CI"] = [float(x) for x in relative_ci]
        ci["relative_CI_reason"] = None
        ci["relative_CI_zero_denominator_count"] = 0
    return ci, {"sampled_means_sha256": _hash(samples.tolist())}


def summarize_paired_rows(rows: Sequence[dict], manifest: P1RoleManifest, role: str) -> dict:
    """Summarize a complete registered role without making an acceptance claim."""
    expected = {r.event_id: r for r in manifest.authorize(role, manifest.ids(role))}
    keyed: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        if set(row) != _FIELDS or row["event_id"] not in expected or isinstance(row["horizon_seconds"], bool) or row["horizon_seconds"] not in _HORIZONS or row["output"] not in _OUTPUTS or row["status"] not in {"completed", "failed"}:
            raise ValueError("invalid paired statistics row")
        meta = expected[row["event_id"]]
        if (row["source_dataset"], row["fault_line"]) != (meta.source_dataset, meta.fault_line):
            raise ValueError("paired row metadata mismatch")
        key = (row["event_id"], row["horizon_seconds"], row["output"])
        if key in keyed:
            raise ValueError("duplicate paired event row")
        if row["status"] == "completed":
            for name in ("reference_rmse", "candidate_rmse"):
                if not isinstance(row[name], (int, float)) or isinstance(row[name], bool) or not np.isfinite(row[name]) or row[name] < 0:
                    raise ValueError("completed paired rows require finite nonnegative RMSE")
        keyed[key] = dict(row)
    required = {(event, horizon, output) for event in expected for horizon in _HORIZONS for output in _OUTPUTS}
    if set(keyed) != required:
        raise ValueError("paired rows must cover the complete authorized role")
    sources = tuple(sorted({r.source_dataset for r in expected.values()}))
    views: dict[str, Any] = {}
    for horizon in sorted(_HORIZONS):
        for output in sorted(_OUTPUTS):
            matching = [keyed[event, horizon, output] for event in expected]
            for source in (*sources, None):
                events = [r for r in matching if source is None or r["source_dataset"] == source]
                name = view_key(source, horizon, output)
                failed = [r["event_id"] for r in events if r["status"] == "failed"]
                view: dict[str, Any] = {"source_dataset": source, "horizon_seconds": horizon, "output": output, "event_ids": sorted(r["event_id"] for r in events), "failed_event_ids": sorted(failed), "scientific_acceptance": False}
                if failed:
                    view.update(status="UNAVAILABLE_FAILED", reference_mean=None, candidate_mean=None, absolute_difference=None, relative_degradation=None, absolute_CI=None, relative_CI=None, bootstrap_executed=False)
                    views[name] = view; continue
                by_source_line: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
                event_pairs = []
                for r in events:
                    by_source_line[r["source_dataset"]][r["fault_line"]].append((r["reference_rmse"], r["candidate_rmse"]))
                    event_pairs.append(r)
                line_pairs = {s: {line: _mean_pair(p) for line, p in sorted(lines.items())} for s, lines in by_source_line.items()}
                source_points = {s: _point(list(lines.values())) for s, lines in line_pairs.items()}
                point = source_points[source] if source is not None else _point([(p["reference_mean"], p["candidate_mean"]) for p in source_points.values()])
                view.update(point, line_event_pairs={s: {line: {"event_ids": sorted(r["event_id"] for r in events if r["source_dataset"] == s and r["fault_line"] == line), "event_pairs": [list(x) for x in by_source_line[s][line]], "reference_mean": pair[0], "candidate_mean": pair[1]} for line, pair in lines.items()} for s, lines in line_pairs.items()}, tails=_tails(event_pairs))
                eligible = all(len(line_pairs[s]) >= 3 for s in line_pairs)
                if source is None:
                    physical = defaultdict(set)
                    for s, lines in line_pairs.items():
                        for line in lines: physical[line].add(s)
                    strata = defaultdict(list)
                    for line, owners in physical.items(): strata[tuple(sorted(owners))].append(line)
                    strata = {owners: sorted(lines) for owners, lines in strata.items()}
                    eligible &= all(sum(len(lines) for owners, lines in strata.items() if s in owners and len(lines) >= 2) >= 3 for s in line_pairs)
                if not eligible:
                    view.update(status="UNDETERMINED", absolute_CI=None, relative_CI=None, bootstrap_executed=False, reason="insufficient_resampleable_fault_lines")
                    views[name] = view; continue
                rng = np.random.default_rng(324); samples = []
                if source is not None:
                    order = sorted(line_pairs[source]); counts = rng.multinomial(len(order), [1 / len(order)] * len(order), size=2000)
                    for count in counts: samples.append(np.average(np.array([line_pairs[source][x] for x in order]), axis=0, weights=count))
                    order = {str((source,)): order}; stratification = order
                else:
                    counts = []
                    ordered_strata = sorted(strata.items())
                    for owners, lines in ordered_strata: counts.append(rng.multinomial(len(lines), [1 / len(lines)] * len(lines), size=2000))
                    for draw in range(2000):
                        weights = defaultdict(float)
                        for (_, lines), count in zip(ordered_strata, counts, strict=True):
                            for line, n in zip(lines, count[draw], strict=True): weights[line] += n
                        per_source = [np.average(np.array([line_pairs[s][line] for line in line_pairs[s]]), axis=0, weights=[weights[line] for line in line_pairs[s]]) for s in sorted(line_pairs)]
                        samples.append(np.mean(per_source, axis=0))
                    order = {str(k): v for k, v in ordered_strata}; stratification = order
                ci, audit = _ci(np.asarray(samples), point)
                view.update(status="COMPLETED", bootstrap_executed=True, bootstrap={"draws": 2000, "seed": 324, "line_order": order, "strata": stratification, "sampling_counts_sha256": _hash([x.tolist() for x in counts])}, **ci, **audit)
                views[name] = view
    return {"schema": "kron_reduce.p1_paired_statistics.v1", "role": role, "role_manifest_sha256": manifest.source_sha256, "bootstrap_draws": 2000, "bootstrap_seed": 324, "views": views}
