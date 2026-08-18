from __future__ import annotations

import json

import numpy as np

from kron_reduce.physics.global_quotient import (
    global_quotient_gap,
    mass_orthonormal_quotient_coordinates,
    unified_quotient_agglomeration,
)


def _local_excitation_paths(*, hidden_on_one_locality: bool = False):
    time = np.arange(240, dtype=float) * 0.01
    paths = []
    for index in range(12):
        phase = 0.17 * index
        first = (1.0 + 0.03 * index) * np.sin(2.0 * time + phase)
        second = (0.7 + 0.02 * index) * np.sin(1.1 * time - 0.5 * phase)
        inner = np.zeros_like(time)
        if hidden_on_one_locality and index >= 8:
            inner = 0.35 * np.sin(3.7 * time + phase)
        delta = np.column_stack((first + inner, first - inner, second, second))
        omega = np.gradient(delta, time, axis=0)
        paths.append(
            {
                "event_id": f"E{index:02d}",
                "fault_line": f"L{index:02d}",
                "source_dataset": "left-only" if index < 6 else "right-only",
                "delta": delta,
                "omega": omega,
                "u": np.zeros_like(delta),
            }
        )
    return paths


def test_mass_orthonormal_quotient_basis_is_complete():
    mass = np.array([2.0, 3.0, 5.0, 7.0])
    basis = mass_orthonormal_quotient_coordinates(mass, ((0, 1), (2, 3)))
    complete = basis.full
    assert np.allclose(complete.T @ (mass[:, None] * complete), np.eye(4), atol=1e-12)
    assert basis.retained.shape == (4, 2)
    assert basis.discarded.shape == (4, 2)


def test_unified_quotient_gap_accepts_union_of_disjoint_local_excitations():
    result = global_quotient_gap(
        _local_excitation_paths(),
        np.ones(4),
        ((0, 1), (2, 3)),
        horizons=(5, 20),
        folds=4,
        max_rows_per_fault_line=40,
        bootstrap_draws=100,
        tolerance=0.02,
    )
    assert result["status"] == "EMPIRICALLY_SUFFICIENT"
    assert result["uqg_ci_upper"] < 0.02
    assert result["source_labels_used"] is False


def test_one_locally_excited_internal_mode_rejects_global_hard_cluster():
    result = global_quotient_gap(
        _local_excitation_paths(hidden_on_one_locality=True),
        np.ones(4),
        ((0, 1), (2, 3)),
        horizons=(5, 20),
        folds=4,
        max_rows_per_fault_line=40,
        bootstrap_draws=100,
        tolerance=0.02,
    )
    assert result["status"] == "PARTITION_REJECTED"
    drivers = [
        row
        for horizon in result["horizons"]
        for row in horizon["tail_fault_lines"]
    ]
    assert any(row["fault_line"] in {"L08", "L09", "L10", "L11"} for row in drivers)


def test_source_labels_cannot_change_the_partition_statistic():
    paths = _local_excitation_paths()
    first = global_quotient_gap(
        paths,
        np.ones(4),
        ((0, 1), (2, 3)),
        horizons=(5,),
        folds=4,
        max_rows_per_fault_line=30,
        bootstrap_draws=50,
    )
    relabelled = [{**path, "source_dataset": f"arbitrary-{index % 3}"} for index, path in enumerate(paths)]
    second = global_quotient_gap(
        relabelled,
        np.ones(4),
        ((0, 1), (2, 3)),
        horizons=(5,),
        folds=4,
        max_rows_per_fault_line=30,
        bootstrap_draws=50,
    )
    assert first["uqg"] == second["uqg"]
    assert first["uqg_ci_upper"] == second["uqg_ci_upper"]


def test_certificate_guided_agglomeration_recovers_the_two_pairs():
    result = unified_quotient_agglomeration(
        _local_excitation_paths(),
        np.ones(4),
        horizons=(5, 20),
        screen_horizon=5,
        minimum_clusters=1,
        folds=4,
        max_rows_per_fault_line=30,
        tolerance=0.02,
        screen_bootstrap_draws=50,
        final_bootstrap_draws=100,
    )
    assert result["status"] == "HARD_CLUSTER_ADMISSIBLE"
    assert result["selected_partition"] == [[0, 1], [2, 3]]
    assert result["source_overlap_required"] is False


def test_v5_workflow_keeps_causal_overlap_audit_out_of_selection(root):
    workflow = json.loads(
        (root / "configs" / "workflows" / "m0_afsd_v5_unified_quotient_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert workflow["approval_required"] == ["merge_gate"]
    assert workflow["causal_overlap_role"] == "post_selection_attribution_audit_only"
    assert workflow["selection_population"] == "union_of_all_QA_train_fault_lines"
