from __future__ import annotations

import numpy as np

from kron_reduce.evaluation.p2_metrics import (
    P2FitScales,
    aggregate_metrics,
    event_metrics,
    fit_scales,
    paired_stratified_fault_line_bootstrap,
    stratified_fault_line_bootstrap,
)


def _sample(event_id: str, source: str, line: str, amplitude: float) -> dict:
    time = np.arange(1001)[:, None] * 0.01
    node = np.arange(1, 11)[None, :]
    delta = amplitude * time * node
    omega = 0.1 * amplitude * time * node
    return {
        "event_id": event_id,
        "training_context": {
            "metadata": {"source_dataset": source, "fault_line": line},
            "targets": {"delta": delta, "omega": omega},
        },
    }


def test_p2_fit_scales_are_source_line_event_equal_and_hashed() -> None:
    samples = [
        _sample("a1", "A", "L1", 1.0),
        _sample("a2", "A", "L1", 3.0),
        _sample("a3", "A", "L2", 2.0),
        _sample("b1", "B", "L3", 4.0),
    ]
    scales = fit_scales(samples)
    assert len(scales.delta) == len(scales.omega) == 10
    assert len(scales.sha256) == 64
    reordered = fit_scales(list(reversed(samples)))
    assert scales == reordered


def test_p2_pooled_r2_centers_each_event_and_node_over_time() -> None:
    scales = P2FitScales((1.0,) * 10, (1.0,) * 10, "a" * 64)
    time = np.arange(101)[:, None]
    node_offset = 1000.0 * np.arange(10)[None, :]
    truth = np.broadcast_to(time + node_offset, (101, 10)).copy()
    pred = truth + 1.0
    row = event_metrics(
        event_id="event",
        source_dataset="source",
        fault_line="line",
        horizon_s=1,
        true_delta=truth,
        true_omega=truth,
        pred_delta=pred,
        pred_omega=pred,
        scales=scales,
    )
    expected_sst = 2 * float(np.sum((time - np.mean(time)) ** 2)) * 10
    assert np.isclose(row["delta_sst"] + row["omega_sst"], expected_sst)
    aggregate = aggregate_metrics([row])
    assert aggregate["pooled_r2_centering"] == "within_event_within_node_over_time"
    assert aggregate["pooled_delta_r2"] < 1.0


def test_p2_failure_is_not_silently_dropped_or_imputed() -> None:
    scales = P2FitScales((1.0,) * 10, (1.0,) * 10, "b" * 64)
    truth = np.zeros((101, 10))
    ok = event_metrics(
        event_id="ok",
        source_dataset="A",
        fault_line="L1",
        horizon_s=1,
        true_delta=truth,
        true_omega=truth,
        pred_delta=truth,
        pred_omega=truth,
        scales=scales,
    )
    bad_pred = truth.copy()
    bad_pred[3, 2] = np.nan
    failed = event_metrics(
        event_id="failed",
        source_dataset="B",
        fault_line="L2",
        horizon_s=1,
        true_delta=truth,
        true_omega=truth,
        pred_delta=bad_pred,
        pred_omega=truth,
        scales=scales,
    )
    aggregate = aggregate_metrics([ok, failed])
    assert aggregate["failure_rate"] == 0.5
    assert aggregate["primary_metrics_available"] is False
    assert aggregate["mean_normalized_joint_rmse"] is None
    assert aggregate["failed_event_ids"] == ["failed"]
    assert aggregate["failures_imputed_as_zero"] is False


def test_p2_tail_keeps_exact_event_ids() -> None:
    rows = []
    for index in range(11):
        rows.append(
            {
                "event_id": f"e{index:02d}",
                "source_dataset": "A" if index < 6 else "B",
                "fault_line": f"L{index:02d}",
                "status": "ok",
                "normalized_joint_rmse": float(index),
                "delta_sse": 1.0,
                "delta_sst": 2.0,
                "omega_sse": 1.0,
                "omega_sst": 2.0,
                "scored_values_per_channel": 10,
            }
        )
    result = aggregate_metrics(rows)
    assert result["worst_10pct_event_ids"] == ["e09", "e10"]
    assert result["worst_10pct_mean_normalized_joint_rmse"] == 9.5


def test_p2_bootstrap_resamples_lines_within_source() -> None:
    rows = [
        {
            "event_id": f"{source}-{line}-{event}",
            "source_dataset": source,
            "fault_line": line,
            "status": "ok",
            "effect": float(index + event),
        }
        for source in ("A", "B")
        for index, line in enumerate(("L1", "L2", "L3"))
        for event in range(2)
    ]
    result = stratified_fault_line_bootstrap(rows, "effect", draws=200, seed=324)
    assert result["status"] == "completed"
    assert result["fault_line_counts_by_source"] == {"A": 3, "B": 3}
    assert result["resampling_unit"] == "fault_line_with_all_events_preserved"


def test_p2_paired_bootstrap_preserves_population_and_failures() -> None:
    reference = [
        {
            "event_id": f"{source}-{line}-{event}",
            "source_dataset": source,
            "fault_line": line,
            "status": "ok",
            "effect": 2.0,
        }
        for source in ("A", "B")
        for line in ("L1", "L2", "L3")
        for event in range(2)
    ]
    candidate = [{**row, "effect": 1.5} for row in reference]
    result = paired_stratified_fault_line_bootstrap(reference, candidate, "effect", draws=100)
    assert result["estimate"] == result["ci_lower"] == result["ci_upper"] == -0.5
    candidate[0]["status"] = "failed"
    result = paired_stratified_fault_line_bootstrap(reference, candidate, "effect", draws=100)
    assert result["status"] == "UNDETERMINED"
    assert result["reason"] == "failed_rollouts_are_not_imputed"
    single = stratified_fault_line_bootstrap(candidate, "effect", draws=100)
    assert single["status"] == "UNDETERMINED"
