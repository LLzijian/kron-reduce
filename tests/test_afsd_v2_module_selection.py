from __future__ import annotations

import json
import tomllib

import numpy as np

from kron_reduce.workflows.m0_afsd_v2_module_selection import (
    MODULES,
    _combine_stats,
    _hypothesis_registry,
    _module_code,
    _module_mask,
    _paired_cluster_bootstrap,
    _select_same_support,
)


def test_v2_workflow_is_independent_and_merge_locked(root) -> None:
    config = tomllib.loads(
        (
            root
            / "configs/experiments/paper_m0_afsd_v2_module_selection_f1_seed324.toml"
        ).read_text(encoding="utf-8")
    )
    workflow = json.loads(
        (
            root / "configs/workflows/m0_afsd_v2_module_selection.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "configs/project_manifest.json").read_text(encoding="utf-8")
    )
    assert config["fold"] == "F1" and config["seed"] == 324
    assert config["pm_mode"] == "observed" and config["test_access"] == "forbidden"
    assert workflow["approval_required"] == ["merge_gate"]
    assert workflow["stages"][5:10] == [
        "hypothesis_registration",
        "controlled_experiment",
        "residual_diagnosis",
        "adversarial_review",
        "acceptance",
    ]
    names = {
        "paper_m0_afsd_v2_hypothesis_f1_seed324",
        "paper_m0_afsd_v2_controlled_f1_seed324",
        "paper_m0_afsd_v2_residual_f1_seed324",
        "paper_m0_afsd_v2_adversarial_f1_seed324",
        "paper_m0_afsd_v2_acceptance_f1_seed324",
    }
    assert names <= set(manifest["experiments"])
    assert all(
        manifest["experiments"][name]["workflow"]
        == "m0_afsd_v2_module_selection"
        for name in names
    )
    assert all(
        manifest["experiments"][name]["reads_test_performance"] is False
        for name in names
    )


def test_hypothesis_registration_contains_complete_module_lattice() -> None:
    registry = _hypothesis_registry(
        {
            "report_horizons": [100, 500, 1000],
            "minimum_model_improvement": 0.02,
            "bootstrap_draws": 200,
            "bootstrap_confidence": 0.95,
            "minimum_module_bootstrap_support": 0.8,
            "minimum_edge_bootstrap_support": 0.8,
            "maximum_coi_drift": 1e-10,
            "claim_boundaries": {"test_performance": False},
        },
        "run",
    )
    lattice = registry["controlled_factors"]["module_lattice"]
    assert len(lattice) == 8
    assert [] in lattice and list(MODULES) in lattice
    assert registry["registered_before_response_metrics"] is True


def test_module_masks_are_disjoint_and_cover_full_design() -> None:
    n_edges, n_modes = 4, 2
    masks = {
        module: _module_mask(n_edges, n_modes, {module}) for module in MODULES
    }
    assert not np.any(masks["conservative"] & masks["dissipative"])
    assert not np.any(masks["conservative"] & masks["memory"])
    assert not np.any(masks["dissipative"] & masks["memory"])
    assert np.all(
        masks["conservative"] | masks["dissipative"] | masks["memory"]
    )
    assert _module_code(set()) == "SS-zero"
    assert _module_code(set(MODULES)) == "SS-CDM"


def test_normal_equation_cluster_combination_preserves_multiplicity() -> None:
    base = {
        "gram": np.eye(2),
        "response": np.asarray([1.0, 2.0]),
        "target_energy": 3.0,
        "effective_blocks": 4,
        "n_scalar_equations": 8,
        "n_parameters": 2,
        "n_static_parameters": 1,
        "n_memory_parameters": 1,
        "n_edges": 1,
        "n_memory_modes": 1,
    }
    combined = _combine_stats([(base, 2), (base, 3)])
    assert np.array_equal(combined["gram"], 5 * np.eye(2))
    assert np.array_equal(combined["response"], np.asarray([5.0, 10.0]))
    assert combined["target_energy"] == 15.0
    assert combined["effective_blocks"] == 20


def test_fault_line_cluster_bootstrap_keeps_paired_events() -> None:
    reference = []
    comparator = []
    for event, line, value in (("a", "L1", 1.0), ("b", "L1", 2.0), ("c", "L2", 3.0)):
        common = {
            "event_id": event,
            "fault_line": line,
            "status": "completed",
        }
        reference.append({**common, "rmse": value})
        comparator.append({**common, "rmse": value + 0.5})
    result = _paired_cluster_bootstrap(
        reference,
        comparator,
        draws=200,
        confidence=0.95,
        seed=324,
    )
    assert result["cluster_unit"] == "fault_line"
    assert result["n_paired_cases"] == 3
    assert result["n_fault_line_clusters"] == 2
    assert np.isclose(result["mean_difference_comparator_minus_reference"], 0.5)
    assert result["ci_lower"] > 0.0


def test_minimum_same_support_selector_prefers_simpler_within_two_percent() -> None:
    design = [
        {"route": "SS-zero", "module_count": 0, "enabled_modules": [], "physics_status": "pass"},
        {"route": "SS-C", "module_count": 1, "enabled_modules": ["conservative"], "physics_status": "pass"},
        {"route": "SS-CDM", "module_count": 3, "enabled_modules": list(MODULES), "physics_status": "pass"},
    ]
    aggregate = [
        {
            "cohort": "C3->C3",
            "horizon_steps": 1000,
            "branch": route,
            "failure_rate": 0.0,
            "mean_rmse": value,
        }
        for route, value in (("SS-zero", 1.5), ("SS-C", 1.01), ("SS-CDM", 1.0))
    ]
    selected = _select_same_support(aggregate, design, 0.02)
    assert selected["route"] == "SS-C"
    assert selected["enabled_modules"] == ["conservative"]
