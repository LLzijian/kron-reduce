from __future__ import annotations

import json
import tomllib

from kron_reduce.workflows.m0_afsd_v6_full_model_subset_validation import (
    _select_minimum_branch,
    _subset_eligible,
    deterministic_fault_line_roles,
)


def test_v6_workflow_is_independent_and_merge_locked(root) -> None:
    workflow = json.loads(
        (
            root
            / "configs/workflows/m0_afsd_v6_full_model_subset_validation.json"
        ).read_text(encoding="utf-8")
    )
    assert workflow["initial_stage"] == "scaffold"
    assert workflow["approval_required"] == ["merge_gate"]
    assert workflow["global_spatial_result"] == "FULL_RESOLUTION_REQUIRED"
    assert workflow["stages"][5:11] == [
        "hypothesis_registration",
        "source_recommit",
        "controlled_experiment",
        "residual_diagnosis",
        "adversarial_review",
        "acceptance",
    ]


def test_v6_config_freezes_observed_pm_scope_and_complete_b3_grid(root) -> None:
    config = tomllib.loads(
        (
            root
            / "configs/experiments/paper_m0_afsd_v6_full_model_subset_validation_f1_seed324.toml"
        ).read_text(encoding="utf-8")
    )
    assert config["fold"] == "F1"
    assert config["seed"] == 324
    assert config["pm_mode"] == "observed"
    assert config["test_access"] == "forbidden"
    assert config["other_folds_access"] == "forbidden"
    assert config["b3_alpha_candidates"] == [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    assert config["b3_tau_candidates"] == [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5]
    assert config["claim_boundaries"]["validation_used_for_frozen_confirmation_only"]


def test_fault_line_roles_are_deterministic_disjoint_and_source_balanced() -> None:
    rows = []
    for source, count in (("a", 10), ("b", 6)):
        for index in range(count):
            rows.append({"fault_line": f"{source}-{index}", "source_dataset": source})
    left = deterministic_fault_line_roles(rows, seed=324, fit_fraction=0.6, selection_fraction=0.2)
    right = deterministic_fault_line_roles(rows, seed=324, fit_fraction=0.6, selection_fraction=0.2)
    assert left == right
    assert set(left) == {row["fault_line"] for row in rows}
    for source in ("a", "b"):
        roles = {left[row["fault_line"]] for row in rows if row["source_dataset"] == source}
        assert roles == {"fit", "selection", "audit"}


def test_minimum_branch_uses_two_percent_equivalence_and_complexity() -> None:
    rows = [
        {"branch": "B2", "mean_rmse": 1.01, "failure_rate": 0.0},
        {"branch": "B3", "mean_rmse": 1.00, "failure_rate": 0.0},
        {"branch": "SS-DM", "mean_rmse": 0.995, "failure_rate": 0.0},
    ]
    result = _select_minimum_branch(rows, {"B2": 20, "B3": 30, "SS-DM": 50}, 0.02)
    assert result["route"] == "B2"
    assert result["best_mean_rmse"] == 0.995


def test_subset_rule_is_quality_and_dynamics_only() -> None:
    config = {"subset_amplitude_level": "A-low", "subset_richness_level": "R-low"}
    base = {
        "quality_grade": "Q-A",
        "amplitude_level": "A-low",
        "richness_level": "R-low",
        "structure_selection_eligible": True,
        "eligible_h1000": True,
    }
    assert _subset_eligible({**base, "source_dataset": "source-a"}, config)
    assert _subset_eligible({**base, "source_dataset": "source-b"}, config)
    assert not _subset_eligible({**base, "richness_level": "R-high"}, config)
    assert not _subset_eligible({**base, "quality_grade": "Q-B"}, config)
