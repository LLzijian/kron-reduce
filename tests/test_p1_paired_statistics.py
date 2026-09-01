from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from kron_reduce.data.p1_causal import P1EventRole, P1RoleManifest
from kron_reduce.evaluation.p1_paired_statistics import summarize_paired_rows, view_key


def _manifest() -> P1RoleManifest:
    records = []
    for source in ("a", "b"):
        for line in ("l1", "l2", "l3"):
            for index in range(2): records.append(P1EventRole(f"{source}{line}{index}", source, line, "fit", 1001))
    records += [P1EventRole("s", "s", "s", "selection", 1001), P1EventRole("i", "i", "i", "internal_audit", 1001)]
    return P1RoleManifest(tuple(records), "F1", hashlib.sha256(b"paired").hexdigest())


def _rows(manifest, *, candidate=lambda s, line, event: 2.0, reference=lambda s, line, event: 1.0):
    return [{"event_id": r.event_id, "source_dataset": r.source_dataset, "fault_line": r.fault_line, "horizon_seconds": h, "output": o, "status": "completed", "reference_rmse": reference(r.source_dataset, r.fault_line, r.event_id), "candidate_rmse": candidate(r.source_dataset, r.fault_line, r.event_id)} for r in manifest.records if r.role == "fit" for h in (1, 5, 10) for o in ("delta", "omega")]


def test_line_then_source_equal_weighting_reproducibility_and_constant_ci() -> None:
    manifest = _manifest()
    rows = _rows(manifest, reference=lambda s, l, e: {"l1": 1., "l2": 2., "l3": 3.}[l], candidate=lambda s, l, e: 2 * {"l1": 1., "l2": 2., "l3": 3.}[l])
    first, second = summarize_paired_rows(rows, manifest, "fit"), summarize_paired_rows(list(reversed(rows)), manifest, "fit")
    view = first["views"][view_key("a", 1, "delta")]
    assert view["reference_mean"] == pytest.approx(2.) and view["relative_degradation"] == pytest.approx(1.)
    assert view["relative_CI"] == pytest.approx([1., 1.])
    assert first == second and view["bootstrap"]["sampling_counts_sha256"]


def test_pooled_shared_physical_line_preserves_pairing_and_failure_is_not_dropped() -> None:
    manifest = _manifest()
    effect = {"l1": .2, "l2": .7, "l3": 1.4}
    rows = _rows(manifest, reference=lambda s, l, e: 2., candidate=lambda s, l, e: 2. + effect[l] if s == "a" else 2. - effect[l])
    result = summarize_paired_rows(rows, manifest, "fit")
    pooled = result["views"][view_key(None, 1, "delta")]
    assert pooled["absolute_difference"] == pytest.approx(0.)
    assert pooled["absolute_CI"] == pytest.approx([0., 0.])
    rows[0]["status"], rows[0]["reference_rmse"], rows[0]["candidate_rmse"] = "failed", None, None
    failed = summarize_paired_rows(rows, manifest, "fit")["views"][view_key("a", 1, "delta")]
    assert failed["status"] == "UNAVAILABLE_FAILED" and failed["failed_event_ids"]


def test_validation_zero_denominator_undetermined_and_tails() -> None:
    manifest = _manifest(); rows = _rows(manifest)
    for row in rows:
        if row["source_dataset"] == "a" and row["fault_line"] == "l1":
            row["reference_rmse"], row["candidate_rmse"] = 0., 1.
    result = summarize_paired_rows(rows, manifest, "fit")
    assert result["views"][view_key("a", 1, "delta")]["relative_CI"] is None
    assert result["views"][view_key("a", 1, "delta")]["tails"]["candidate_rmse"]["event_ids"]
    tail_rows = _rows(manifest, candidate=lambda s, line, e: {"l1": .1, "l2": .1, "l3": 2.5}[line])
    tail_view = summarize_paired_rows(tail_rows, manifest, "fit")["views"][view_key("a", 1, "delta")]
    assert tail_view["candidate_mean"] < tail_view["reference_mean"]
    assert tail_view["tails"]["candidate_rmse"]["mean"] > tail_view["reference_mean"]
    with pytest.raises(ValueError): summarize_paired_rows(rows[:-1], manifest, "fit")
    bad = _rows(manifest); bad[0]["source_dataset"] = "attack"
    with pytest.raises(ValueError): summarize_paired_rows(bad, manifest, "fit")
    small = P1RoleManifest((P1EventRole("f", "a", "l", "fit", 1001), P1EventRole("s", "s", "s", "selection", 1001), P1EventRole("i", "i", "i", "internal_audit", 1001)), "F1", hashlib.sha256(b"small").hexdigest())
    assert summarize_paired_rows(_rows(small), small, "fit")["views"][view_key("a", 1, "delta")]["status"] == "UNDETERMINED"


def _custom(records):
    return P1RoleManifest(tuple(records + [P1EventRole("s", "s", "s", "selection", 1001), P1EventRole("i", "i", "i", "internal_audit", 1001)]), "F1", hashlib.sha256(repr(records).encode()).hexdigest())


def test_real_pooled_source_key_does_not_collide_and_both_views_have_ci() -> None:
    manifest = _custom([P1EventRole(f"p{x}", "pooled", f"l{x}", "fit", 1001) for x in range(3)])
    result = summarize_paired_rows(_rows(manifest), manifest, "fit")["views"]
    assert view_key("pooled", 1, "delta") != view_key(None, 1, "delta")
    assert result[view_key("pooled", 1, "delta")]["status"] == result[view_key(None, 1, "delta")]["status"] == "COMPLETED"


def test_pooled_strata_eligibility_and_event_duplication_line_weights() -> None:
    records = [P1EventRole("a1", "A", "solo", "fit", 1001), P1EventRole("a2", "A", "ab", "fit", 1001), P1EventRole("b1", "B", "ab", "fit", 1001), P1EventRole("a3", "A", "ac", "fit", 1001), P1EventRole("c1", "C", "ac", "fit", 1001)]
    records += [P1EventRole(f"b{x}", "B", f"b{x}", "fit", 1001) for x in range(2, 5)] + [P1EventRole(f"c{x}", "C", f"c{x}", "fit", 1001) for x in range(2, 5)]
    mixed = _custom(records); result = summarize_paired_rows(_rows(mixed), mixed, "fit")["views"]
    assert result[view_key("A", 1, "delta")]["status"] == "COMPLETED"
    assert result[view_key(None, 1, "delta")]["status"] == "UNDETERMINED"
    base = _custom([P1EventRole(f"x{x}", "A", f"l{x}", "fit", 1001) for x in range(3)])
    duplicated = _custom([P1EventRole(f"x{x}", "A", f"l{x}", "fit", 1001) for x in range(3)] + [P1EventRole(f"many{x}", "A", "l0", "fit", 1001) for x in range(20)])
    values = {"l0": (1., 1.5), "l1": (3., 2.), "l2": (5., 7.)}
    left = summarize_paired_rows(_rows(base, reference=lambda s, l, e: values[l][0], candidate=lambda s, l, e: values[l][1]), base, "fit")["views"][view_key("A", 1, "delta")]
    right = summarize_paired_rows(_rows(duplicated, reference=lambda s, l, e: values[l][0], candidate=lambda s, l, e: values[l][1]), duplicated, "fit")["views"][view_key("A", 1, "delta")]
    assert (left["reference_mean"], left["candidate_mean"], left["absolute_CI"]) == (right["reference_mean"], right["candidate_mean"], right["absolute_CI"])


def test_sorted_sampling_hash_and_extreme_values() -> None:
    manifest = _custom([P1EventRole("z", "A", "a", "fit", 1001), P1EventRole("y", "A", "b", "fit", 1001), P1EventRole("x", "A", "c", "fit", 1001)])
    rows = _rows(manifest); view = summarize_paired_rows(rows, manifest, "fit")["views"][view_key("A", 1, "delta")]
    order = view["bootstrap"]["line_order"][str(("A",))]
    counts = np.random.default_rng(324).multinomial(3, [1 / 3] * 3, size=2000)
    digest = hashlib.sha256(json.dumps(counts.tolist(), separators=(",", ":"), sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert order == ["a", "b", "c"] and view["bootstrap"]["sampling_counts_sha256"] == digest
    tiny = _rows(manifest, reference=lambda *_: 1e-300, candidate=lambda *_: 1e100)
    assert summarize_paired_rows(tiny, manifest, "fit")["views"][view_key("A", 1, "delta")]["relative_degradation"] is None
    huge = _rows(manifest, reference=lambda *_: 1e308, candidate=lambda *_: 1e308)
    with pytest.raises(ValueError, match="nonfinite paired mean"):
        summarize_paired_rows(huge, manifest, "fit")
    boolean = _rows(manifest); boolean[0]["horizon_seconds"] = True
    with pytest.raises(ValueError):
        summarize_paired_rows(boolean, manifest, "fit")
