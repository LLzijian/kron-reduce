from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pytest

from kron_reduce.data.p1_causal import P1EventRole, P1RoleManifest
from kron_reduce.evaluation.p1_effect_statistics import summarize_nonnegative_effect_rows, view_key


def _manifest(extra=0):
    rows = [P1EventRole(f"a{i}", "A", f"l{i}", "fit", 1001) for i in range(3)]
    rows += [P1EventRole(f"b{i}", "B", f"l{i}", "fit", 1001) for i in range(3)]
    rows += [P1EventRole(f"x{i}", "A", "l0", "fit", 1001) for i in range(extra)]
    rows += [
        P1EventRole("s", "s", "s", "selection", 1001),
        P1EventRole("i", "i", "i", "internal_audit", 1001),
    ]
    return P1RoleManifest(tuple(rows), "F1", hashlib.sha256(repr(rows).encode()).hexdigest())


def _rows(manifest, value=lambda s, l, e: {"l0": 1.0, "l1": 3.0, "l2": 5.0}[l]):
    return [
        {
            "event_id": r.event_id,
            "source_dataset": r.source_dataset,
            "fault_line": r.fault_line,
            "horizon_seconds": h,
            "status": "completed",
            "value": value(r.source_dataset, r.fault_line, r.event_id),
        }
        for r in manifest.records
        if r.role == "fit"
        for h in (1, 5, 10)
    ]


def test_line_equal_weighting_shared_pairing_and_json_safe_output():
    manifest = _manifest()
    result = summarize_nonnegative_effect_rows(
        _rows(
            manifest,
            lambda s, l, e: (
                {"l0": 1.0, "l1": 2.0, "l2": 4.0}[l]
                if s == "A"
                else {"l0": 4.0, "l1": 3.0, "l2": 1.0}[l]
            ),
        ),
        manifest,
        "fit",
        "energy",
    )
    pooled = result["views"][view_key(None, 1)]
    assert pooled["point_estimate"] == pytest.approx(2.5) and pooled["ci95"] == pytest.approx(
        [2.5, 2.5]
    )
    json.dumps(result, allow_nan=False)
    duplicated = _manifest(20)
    left = summarize_nonnegative_effect_rows(_rows(manifest), manifest, "fit", "effect")["views"][
        view_key("A", 1)
    ]
    right = summarize_nonnegative_effect_rows(_rows(duplicated), duplicated, "fit", "effect")[
        "views"
    ][view_key("A", 1)]
    assert (left["point_estimate"], left["ci95"]) == (right["point_estimate"], right["ci95"])


def test_failures_validation_collision_small_and_extreme():
    manifest = _manifest()
    rows = _rows(manifest)
    rows[0].update(status="failed", value=None)
    failed = summarize_nonnegative_effect_rows(rows, manifest, "fit", "port")["views"][
        view_key("A", 1)
    ]
    assert failed["status"] == "UNAVAILABLE_FAILED" and failed["failed_event_ids"]
    bad = _rows(manifest)
    bad[0]["horizon_seconds"] = True
    with pytest.raises(ValueError):
        summarize_nonnegative_effect_rows(bad, manifest, "fit", "x")
    bad = _rows(manifest)
    bad[0]["value"] = True
    with pytest.raises(ValueError):
        summarize_nonnegative_effect_rows(bad, manifest, "fit", "x")
    small = P1RoleManifest(
        (
            P1EventRole("p", "pooled", "q", "fit", 1001),
            P1EventRole("s", "s", "s", "selection", 1001),
            P1EventRole("i", "i", "i", "internal_audit", 1001),
        ),
        "F1",
        hashlib.sha256(b"s").hexdigest(),
    )
    out = summarize_nonnegative_effect_rows(_rows(small, lambda *_: 1.0), small, "fit", "x")[
        "views"
    ]
    assert (
        view_key("pooled", 1) != view_key(None, 1)
        and out[view_key(None, 1)]["status"] == "UNDETERMINED"
    )
    with pytest.raises(ValueError):
        summarize_nonnegative_effect_rows(_rows(manifest, lambda *_: 1e308), manifest, "fit", "x")
    bootstrap_overflow = _rows(
        manifest,
        lambda source, line, event: 1e308 if line == "l0" else 0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="bootstrap effect"):
            summarize_nonnegative_effect_rows(bootstrap_overflow, manifest, "fit", "x")


def test_percentile_interval_is_not_moved_to_contain_point(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(np, "quantile", lambda values, probabilities: np.array([10.0, 11.0]))
    view = summarize_nonnegative_effect_rows(_rows(manifest), manifest, "fit", "effect")["views"][
        view_key("A", 1)
    ]
    assert view["point_estimate"] == pytest.approx(3.0)
    assert view["ci95"] == [10.0, 11.0]


@pytest.mark.parametrize(
    "attack",
    ["missing", "duplicate", "metadata", "failed_value", "negative", "nan"],
)
def test_effect_rows_reject_incomplete_duplicate_or_invalid_cells(attack):
    manifest = _manifest()
    rows = _rows(manifest)
    if attack == "missing":
        rows.pop()
    elif attack == "duplicate":
        rows.append(dict(rows[0]))
    elif attack == "metadata":
        rows[0]["fault_line"] = "attack"
    elif attack == "failed_value":
        rows[0].update(status="failed", value=0.0)
    elif attack == "negative":
        rows[0]["value"] = -1.0
    else:
        rows[0]["value"] = np.nan
    with pytest.raises(ValueError):
        summarize_nonnegative_effect_rows(rows, manifest, "fit", "effect")
