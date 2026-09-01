"""Preregistered nonnegative P1 effect summaries; descriptive only."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from kron_reduce.data.p1_causal import P1RoleManifest

_FIELDS = {"event_id", "source_dataset", "fault_line", "horizon_seconds", "status", "value"}


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def view_key(source: str | None, horizon: int) -> str:
    return json.dumps([source, horizon], separators=(",", ":"))


def _mean(values) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            value = float(np.mean(values))
    except FloatingPointError as error:
        raise ValueError("nonfinite effect mean") from error
    if not np.isfinite(value) or value < 0:
        raise ValueError("nonfinite effect mean")
    return value


def _weighted_mean(values, weights) -> float:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value = float(np.average(values, weights=weights))
    except (FloatingPointError, ZeroDivisionError) as error:
        raise ValueError("nonfinite bootstrap effect") from error
    if not np.isfinite(value) or value < 0:
        raise ValueError("nonfinite bootstrap effect")
    return value


def summarize_nonnegative_effect_rows(
    rows: Sequence[dict], manifest: P1RoleManifest, role: str, metric_name: str
) -> dict:
    if not isinstance(metric_name, str) or not metric_name:
        raise ValueError("metric name required")
    expected = {x.event_id: x for x in manifest.authorize(role, manifest.ids(role))}
    keyed = {}
    for row in rows:
        if (
            set(row) != _FIELDS
            or row["event_id"] not in expected
            or isinstance(row["horizon_seconds"], bool)
            or row["horizon_seconds"] not in {1, 5, 10}
            or row["status"] not in {"completed", "failed"}
        ):
            raise ValueError("invalid effect row")
        meta = expected[row["event_id"]]
        if (row["source_dataset"], row["fault_line"]) != (meta.source_dataset, meta.fault_line):
            raise ValueError("effect metadata mismatch")
        key = (row["event_id"], row["horizon_seconds"])
        if key in keyed:
            raise ValueError("duplicate effect row")
        value = row["value"]
        if row["status"] == "completed" and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("completed effect requires finite nonnegative value")
        if row["status"] == "failed" and value is not None:
            raise ValueError("failed effect value must be None")
        keyed[key] = dict(row)
    required = {(event, horizon) for event in expected for horizon in (1, 5, 10)}
    if set(keyed) != required:
        raise ValueError("effect rows must cover complete authorized role")
    sources, views = sorted({x.source_dataset for x in expected.values()}), {}
    for horizon in (1, 5, 10):
        matched = [keyed[event, horizon] for event in expected]
        for source in (*sources, None):
            events = [x for x in matched if source is None or x["source_dataset"] == source]
            failed = sorted(x["event_id"] for x in events if x["status"] == "failed")
            view = {
                "source_dataset": source,
                "horizon_seconds": horizon,
                "event_ids": sorted(x["event_id"] for x in events),
                "failed_event_ids": failed,
                "metric_name": metric_name,
                "scientific_acceptance": False,
            }
            if failed:
                view.update(
                    status="UNAVAILABLE_FAILED",
                    point_estimate=None,
                    ci95=None,
                    bootstrap_executed=False,
                )
                views[view_key(source, horizon)] = view
                continue
            by_source = defaultdict(lambda: defaultdict(list))
            for x in events:
                by_source[x["source_dataset"]][x["fault_line"]].append(x)
            lines = {
                s: {
                    line: _mean([x["value"] for x in values])
                    for line, values in sorted(group.items())
                }
                for s, group in by_source.items()
            }
            source_means = {s: _mean(list(group.values())) for s, group in lines.items()}
            point = (
                source_means[source] if source is not None else _mean(list(source_means.values()))
            )
            event_values = {
                s: {
                    line: {
                        "event_ids": sorted(x["event_id"] for x in values),
                        "values": [x["value"] for x in values],
                        "line_mean": lines[s][line],
                    }
                    for line, values in group.items()
                }
                for s, group in by_source.items()
            }
            view.update(
                point_estimate=point,
                line_event_values=event_values,
                tails={
                    "event_ids": [
                        x["event_id"]
                        for x in sorted(events, key=lambda x: (-x["value"], x["event_id"]))[
                            : math.ceil(0.1 * len(events))
                        ]
                    ],
                    "mean": _mean(
                        [
                            x["value"]
                            for x in sorted(events, key=lambda x: (-x["value"], x["event_id"]))[
                                : math.ceil(0.1 * len(events))
                            ]
                        ]
                    ),
                },
            )
            eligible = all(len(group) >= 3 for group in lines.values())
            if source is None:
                owners = defaultdict(set)
                for s, group in lines.items():
                    for line in group:
                        owners[line].add(s)
                strata = {
                    key: sorted(
                        line for line, owner in owners.items() if tuple(sorted(owner)) == key
                    )
                    for key in sorted({tuple(sorted(x)) for x in owners.values()})
                }
                eligible &= all(
                    sum(
                        len(group)
                        for owner, group in strata.items()
                        if s in owner and len(group) >= 2
                    )
                    >= 3
                    for s in lines
                )
            if not eligible:
                view.update(
                    status="UNDETERMINED",
                    ci95=None,
                    bootstrap_executed=False,
                    reason="insufficient_resampleable_fault_lines",
                )
                views[view_key(source, horizon)] = view
                continue
            rng = np.random.default_rng(324)
            samples = []
            if source is not None:
                order = {str((source,)): sorted(lines[source])}
                count_list = [
                    rng.multinomial(
                        len(next(iter(order.values()))),
                        [1 / len(next(iter(order.values())))] * len(next(iter(order.values()))),
                        size=2000,
                    )
                ]
                for count in count_list[0]:
                    samples.append(
                        _weighted_mean(
                            [lines[source][x] for x in next(iter(order.values()))], count
                        )
                    )
            else:
                order = {str(k): v for k, v in strata.items()}
                count_list = [
                    rng.multinomial(len(group), [1 / len(group)] * len(group), size=2000)
                    for group in strata.values()
                ]
                for draw in range(2000):
                    counts = defaultdict(float)
                    for group, count in zip(strata.values(), count_list, strict=True):
                        for line, n in zip(group, count[draw], strict=True):
                            counts[line] += n
                    samples.append(
                        _mean(
                            [
                                _weighted_mean(
                                    [lines[s][x] for x in lines[s]], [counts[x] for x in lines[s]]
                                )
                                for s in sorted(lines)
                            ]
                        )
                    )
            array = np.asarray(samples)
            if not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError("nonfinite bootstrap effect")
            ci = np.quantile(array, [0.025, 0.975])
            if not np.isfinite(ci).all() or ci[0] > ci[1] or ci[0] < 0:
                raise ValueError("invalid effect CI")
            view.update(
                status="COMPLETED",
                ci95=[float(x) for x in ci],
                bootstrap_executed=True,
                bootstrap={
                    "draws": 2000,
                    "seed": 324,
                    "line_order": order,
                    "strata": order,
                    "sampling_counts_sha256": _hash([x.tolist() for x in count_list]),
                    "sampled_values_sha256": _hash(array.tolist()),
                },
            )
            views[view_key(source, horizon)] = view
    return {
        "schema": "kron_reduce.p1_effect_statistics.v1",
        "metric_name": metric_name,
        "role": role,
        "role_manifest_sha256": manifest.source_sha256,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 324,
        "views": views,
    }
