from __future__ import annotations

import json
import tomllib

import numpy as np

from kron_reduce.evaluation.selection_validity import (
    complexity_key,
    complexity_record,
    equal_fault_line_paired_inference,
    holm_adjust,
    nested_fault_line_folds,
    profile_joint_inertia,
    source_excitation_overlap_audit,
    structure_selection_ids,
)
from kron_reduce.models import M0Model
from kron_reduce.workflows.m0_afsd_v3_joint_selection_validity import (
    candidate_lattice,
    hypothesis_registry,
)


def _row(
    event: str,
    line: str,
    source: str,
    amplitude: str,
    richness: str,
    *,
    quality: str = "Q-A",
    role: str = "train",
) -> dict:
    return {
        "event_id": event,
        "fault_line": line,
        "source_dataset": source,
        "amplitude_level": amplitude,
        "richness_level": richness,
        "quality_grade": quality,
        "role": role,
        "structure_selection_eligible": quality == "Q-A",
    }


def _config(root) -> dict:
    return tomllib.loads(
        (
            root
            / "configs/experiments/paper_m0_afsd_v3_joint_selection_validity_f1_seed324.toml"
        ).read_text(encoding="utf-8")
    )


def test_v3_is_independent_registered_and_merge_locked(root) -> None:
    config = _config(root)
    workflow = json.loads(
        (
            root / "configs/workflows/m0_afsd_v3_joint_selection_validity.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "configs/project_manifest.json").read_text(encoding="utf-8")
    )
    assert config["fold"] == "F1" and config["seed"] == 324
    assert config["test_access"] == "forbidden"
    assert config["other_folds_access"] == "forbidden"
    assert config["pm_mode"] == "observed"
    assert workflow["approval_required"] == ["merge_gate"]
    assert workflow["stages"][5:11] == [
        "hypothesis_registration",
        "eligibility_audit",
        "controlled_experiment",
        "residual_diagnosis",
        "adversarial_review",
        "acceptance",
    ]
    names = {
        "paper_m0_afsd_v3_hypothesis_f1_seed324",
        "paper_m0_afsd_v3_eligibility_f1_seed324",
        "paper_m0_afsd_v3_controlled_f1_seed324",
        "paper_m0_afsd_v3_residual_f1_seed324",
        "paper_m0_afsd_v3_adversarial_f1_seed324",
        "paper_m0_afsd_v3_acceptance_f1_seed324",
    }
    assert names <= set(manifest["experiments"])
    assert all(
        manifest["experiments"][name]["workflow"]
        == "m0_afsd_v3_joint_selection_validity"
        for name in names
    )
    assert all(
        manifest["experiments"][name]["reads_test_performance"] is False
        for name in names
    )


def test_candidate_lattice_contains_full_b3_memory_space_and_module_grids(root) -> None:
    config = _config(root)
    lattice = candidate_lattice(config)
    assert {row["spatial_coordinates"] for row in lattice["spatial_candidates"]} == {
        5,
        6,
        7,
        8,
        9,
        10,
    }
    assert len(lattice["spatial_candidates"]) == 7  # adaptive k7 plus fixed k7
    assert len(lattice["b3_grid"]) == 7 * 8
    assert sum(row["exact_B2_fallback"] for row in lattice["b3_grid"]) == 8
    assert len(lattice["memory_grid"]) == 4 * 5
    modules = lattice["same_support_module_lattice"]
    assert len(modules) == 8
    assert any(row["code"] == "SS-zero" for row in modules)
    assert any(row["code"] == "SS-CDM" for row in modules)
    assert lattice["intra_cluster_modes_first_round"] is False
    registry = hypothesis_registry(config, "run")
    assert registry["registered_before_data_qualification"] is True
    assert registry["statistics"]["cluster_bootstrap_draws"] == 5000


def test_source_excitation_audit_fails_on_source_exclusive_cells_and_ignores_qb() -> None:
    rows = []
    for source in ("A", "B"):
        for index in range(5):
            rows.append(
                _row(
                    f"low-{source}-{index}",
                    f"L-{source}-{index}",
                    source,
                    "A-low",
                    "R-low",
                )
            )
    for index in range(5):
        rows.append(
            _row(
                f"high-A-{index}",
                f"HA-{index}",
                "A",
                "A-high",
                "R-high",
            )
        )
        rows.append(
            _row(
                f"high-B-QB-{index}",
                f"HB-{index}",
                "B",
                "A-high",
                "R-high",
                quality="Q-B",
            )
        )
    result = source_excitation_overlap_audit(rows, minimum_fault_line_clusters=5)
    assert result["status"] == "SOURCE_EXCITATION_CONFOUNDED"
    assert result["claim_allowed"] is False
    missing = {
        (row["source_dataset"], row["amplitude_level"], row["richness_level"])
        for row in result["failing_source_cells"]
    }
    assert ("B", "A-high", "R-high") in missing


def test_source_excitation_audit_passes_only_with_fault_line_overlap() -> None:
    rows = []
    for source in ("A", "B"):
        for amplitude in ("A-low", "A-high"):
            for index in range(5):
                rows.append(
                    _row(
                        f"{source}-{amplitude}-{index}",
                        f"{source}-{amplitude}-{index}",
                        source,
                        amplitude,
                        "R-low",
                    )
                )
    result = source_excitation_overlap_audit(rows, minimum_fault_line_clusters=5)
    assert result["status"] == "PASS"
    assert len(result["comparable_cells"]) == 2


def test_nested_fault_line_cv_is_deterministic_and_has_no_group_leakage() -> None:
    rows = [
        _row(
            f"event-{line}-{event}",
            f"L{line}",
            "A" if line % 2 else "B",
            ("A-low", "A-mid", "A-high")[line % 3],
            "R-high" if line % 2 else "R-low",
        )
        for line in range(20)
        for event in range(2)
    ]
    first = nested_fault_line_folds(rows, outer_folds=5, inner_folds=4)
    second = nested_fault_line_folds(rows, outer_folds=5, inner_folds=4)
    assert first == second
    for outer in range(5):
        subset = [row for row in first if row["outer_fold"] == outer]
        evaluation = {
            row["fault_line"]
            for row in subset
            if row["outer_role"] == "evaluation"
        }
        development = {
            row["fault_line"]
            for row in subset
            if row["outer_role"] == "development"
        }
        assert evaluation
        assert not evaluation & development
        for line in development:
            assert len(
                {
                    row["inner_fold"]
                    for row in subset
                    if row["fault_line"] == line
                }
            ) == 1


def test_fault_line_equal_inference_and_exact_sign_flip_are_not_event_weighted() -> None:
    rows = [
        {"fault_line": "L1", "comparator_minus_reference": 10.0}
        for _ in range(10)
    ] + [{"fault_line": "L2", "comparator_minus_reference": -1.0}]
    result = equal_fault_line_paired_inference(rows, bootstrap_draws=500, seed=324)
    assert np.isclose(result["mean_difference"], 4.5)
    assert not np.isclose(result["mean_difference"], 9.0)
    assert result["sign_flip_method"] == "exact"
    assert result["sign_flip_draws"] == 4


def test_holm_adjustment_is_monotone_and_complexity_uses_actual_states() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}
    simple = complexity_record(
        spatial_coordinates=7,
        active_dynamic_states=0,
        active_parameters=20,
        active_edges=6,
    )
    memory = complexity_record(
        spatial_coordinates=7,
        active_dynamic_states=2,
        active_parameters=22,
        active_edges=6,
    )
    assert simple["total_state_count"] == 14
    assert memory["total_state_count"] == 16
    assert complexity_key(simple) < complexity_key(memory)


def test_only_qa_train_ids_are_admissible_for_structure_selection() -> None:
    rows = [
        _row("qa-train", "L1", "A", "A-low", "R-low"),
        _row("qb-train", "L2", "A", "A-low", "R-low", quality="Q-B"),
        _row(
            "qa-validation",
            "L3",
            "A",
            "A-low",
            "R-low",
            role="validation",
        ),
    ]
    assert structure_selection_ids(rows) == {"qa-train"}


def test_joint_inertia_audit_recovers_excited_synthetic_parameters() -> None:
    true_mass = np.asarray([1.0, 1.5, 0.8])
    true_damping = np.asarray([0.3, 0.45, 0.22])
    true_coupling = np.asarray([1.1, 0.8, 0.95])
    incidence = np.asarray(
        [[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, -1.0]]
    )
    truth = M0Model(true_damping, true_coupling, max_internal_step=0.0025)
    t = np.linspace(0.0, 1.0, 101)
    samples = []
    for index, frequency in enumerate((1.3, 1.7, 2.1, 2.6, 3.1, 3.7)):
        delta0 = np.asarray(
            [0.03 + 0.003 * index, -0.015 + 0.002 * index, 0.01]
        )
        delta0 -= np.sum(true_mass * delta0) / np.sum(true_mass)
        omega0 = np.asarray(
            [0.012 - 0.001 * index, -0.007 + 0.0005 * index, 0.004]
        )
        omega0 -= np.sum(true_mass * omega0) / np.sum(true_mass)
        static = {
            "delta0": delta0[None, :],
            "omega0": omega0[None, :],
            "Pm_equilibrium": np.zeros(3),
            "Pm_clear": np.zeros((1, 3)),
            "Pm0": np.zeros((1, 3)),
            "equilibrium_delta": np.zeros(3),
            "mass": true_mass.copy(),
            "incidence": incidence,
        }
        first = (0.05 + 0.006 * index) * np.sin(frequency * t + 0.2 * index)
        second = (0.04 + 0.004 * index) * np.cos(
            (0.7 * frequency + 0.4) * t - 0.1 * index
        )
        pm = np.column_stack((first, second, -first - second))
        delta, omega = truth.rollout(
            static, t, {"Pm": pm, "interpolation": "linear"}
        )
        # The declared prior initializes the audit but is not the generating
        # mass.  This is synthetic correctness evidence only.
        audit_static = {**static, "mass": np.asarray([1.1, 1.1, 1.1])}
        samples.append(
            {
                "event_id": f"synthetic-{index}",
                "model_inputs": {
                    "static": audit_static,
                    "exogenous": {"t": t, "Pm": pm},
                },
                "training_context": {
                    "targets": {"t": t, "delta": delta, "omega": omega},
                    "Pm_observed": pm,
                    "claim_type": "conditional_rollout_not_causal_forecast",
                },
            }
        )
    result = profile_joint_inertia(
        samples,
        windows=(1, 5, 10, 20),
        max_windows_per_case_horizon=12,
        ridge=1e-10,
        maximum_condition_number=1e8,
    )
    assert result["status"] == "IDENTIFIABLE"
    assert np.allclose(result["effective_mass"], true_mass, rtol=0.03, atol=0.03)
    assert np.allclose(
        result["effective_damping"], true_damping, rtol=0.05, atol=0.03
    ), result
    assert np.allclose(
        result["effective_coupling"], true_coupling, rtol=0.05, atol=0.03
    )


def test_joint_inertia_audit_rejects_unexcited_parameters() -> None:
    t = np.linspace(0.0, 0.2, 21)
    static = {
        "delta0": np.zeros((1, 2)),
        "omega0": np.zeros((1, 2)),
        "Pm_equilibrium": np.zeros(2),
        "Pm_clear": np.zeros((1, 2)),
        "Pm0": np.zeros((1, 2)),
        "equilibrium_delta": np.zeros(2),
        "mass": np.ones(2),
        "incidence": np.asarray([[1.0], [-1.0]]),
    }
    zero = np.zeros((len(t), 2))
    sample = {
        "event_id": "unexcited",
        "model_inputs": {"static": static, "exogenous": {"t": t, "Pm": zero}},
        "training_context": {
            "targets": {"t": t, "delta": zero, "omega": zero},
            "Pm_observed": zero,
            "claim_type": "conditional_rollout_not_causal_forecast",
        },
    }
    result = profile_joint_inertia(
        [sample], windows=(1, 5), max_windows_per_case_horizon=4
    )
    assert result["status"] == "PARAMETER_NOT_IDENTIFIABLE"
    assert result["effective_rank"] < result["n_parameters"]
