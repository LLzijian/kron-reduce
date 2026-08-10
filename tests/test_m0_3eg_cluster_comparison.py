from __future__ import annotations

import numpy as np

from kron_reduce.workflows.m0_3eg_cluster_comparison import (
    _cluster_bootstrap_difference,
    _summary,
)


def _row(event: str, line: str, route: str, rmse: float) -> dict[str, object]:
    return {
        "representation": "clustered_7_train_discovered",
        "event_id": event,
        "fault_line": line,
        "source_dataset": "source",
        "model_route": route,
        "status": "completed",
        "rmse": rmse,
        "omega_rmse": rmse / 2,
        "edge_angle_rmse": rmse * 2,
    }


def test_fault_line_bootstrap_preserves_paired_difference() -> None:
    rows = []
    for event, line, native, projected in (
        ("e1", "L1", 1.0, 2.0),
        ("e2", "L1", 2.0, 3.0),
        ("e3", "L2", 3.0, 4.0),
    ):
        rows.extend(
            [
                _row(event, line, "native_7", native),
                _row(event, line, "generator_10_then_project", projected),
            ]
        )
    result = _cluster_bootstrap_difference(rows, draws=200, seed=324)
    assert result["n_paired_cases"] == 3
    assert result["n_fault_line_clusters"] == 2
    assert np.isclose(result["mean_difference_native_minus_projected"], -1.0)
    assert result["ci95_upper"] < 0.0


def test_summary_keeps_routes_separate() -> None:
    rows = [
        _row("e1", "L1", "native_7", 1.0),
        _row("e1", "L1", "generator_10_then_project", 2.0),
        _row("e2", "L2", "native_7", 3.0),
        _row("e2", "L2", "generator_10_then_project", 4.0),
    ]
    summary = _summary(rows)
    by_route = {row["model_route"]: row for row in summary}
    assert np.isclose(by_route["native_7"]["mean_rmse"], 2.0)
    assert np.isclose(by_route["generator_10_then_project"]["mean_rmse"], 3.0)
    assert all(row["failure_rate"] == 0.0 for row in summary)
