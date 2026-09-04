from __future__ import annotations

import hashlib

import numpy as np
import pytest
from test_p1_causal_calibration import _sample

from kron_reduce.data.p1_causal import P1EventRole, P1RoleManifest
from kron_reduce.evaluation.p1_module_identifiability import (
    audit_saved_model_identifiability,
    audit_weak_module_blocks,
)
from kron_reduce.models import EDGE_MEMORY, CandidateBlockInstance, PrunableEnergyModel
from kron_reduce.models.base import RunContext, sample_rollout_inputs
from kron_reduce.models.p1_calibration import core_fingerprint


def _layout(module_parameters: int = 1) -> dict[str, slice]:
    return {
        "node_damping": slice(0, 1),
        "coupling": slice(1, 2),
        "module": slice(2, 2 + module_parameters),
        "all": slice(0, 2 + module_parameters),
    }


def _audit(matrix: np.ndarray, vector: np.ndarray, *, module_parameters: int = 1):
    return audit_weak_module_blocks(
        matrix,
        vector,
        _layout(module_parameters),
        ({"instance_id": "module", "family": "KNOWN"},),
        selected_ridge=1e-8,
    )


def test_known_independent_positive_block_passes_rank_and_two_sided_profile() -> None:
    rng = np.random.default_rng(324)
    matrix = rng.normal(size=(600, 3))
    vector = matrix @ np.array([1.0, 2.0, 3.0])
    audit = _audit(matrix, vector)
    row = audit["module_results"][0]
    assert row["all_module_columns_active"] is True
    assert row["global_rank"] == row["global_parameter_count"] == 3
    assert row["conditional_rank"] == row["module_parameter_count"] == 1
    assert row["sigma_ratio"] > 0.5
    assert row["condition_number"] < 2
    assert [item["multiplier"] for item in row["profile_rows"]] == [0.5, 2.0]
    assert min(item["relative_increase"] for item in row["profile_rows"]) > 0.02
    assert row["parameter_identifiability_passed"] is True
    assert audit["ridge_rows_counted_as_observations"] is False


def test_scalar_module_collinear_with_nuisance_cannot_pass_as_condition_one() -> None:
    rng = np.random.default_rng(11)
    first = rng.normal(size=500)
    second = rng.normal(size=500)
    matrix = np.column_stack((first, second, first))
    vector = matrix @ np.array([1.0, 1.0, 1.0])
    row = _audit(matrix, vector)["module_results"][0]
    assert row["conditional_rank"] == 0
    assert row["sigma_ratio"] == 0
    assert row["condition_number"] is None
    assert row["conditional_rank_gate_passed"] is False
    assert row["parameter_identifiability_passed"] is False


def test_multicolumn_module_fails_when_only_one_direction_is_conditionally_visible() -> None:
    rng = np.random.default_rng(12)
    nuisance = rng.normal(size=(500, 2))
    independent = rng.normal(size=500)
    matrix = np.column_stack((nuisance, independent, nuisance[:, 0]))
    vector = matrix @ np.array([1.0, 1.0, 2.0, 1.0])
    row = _audit(matrix, vector, module_parameters=2)["module_results"][0]
    assert row["module_parameter_count"] == 2
    assert row["conditional_rank"] == 1
    assert row["sigma_ratio"] == 0
    assert row["parameter_identifiability_passed"] is False


def test_nonnegative_nuisance_cannot_be_removed_from_structural_rank() -> None:
    rng = np.random.default_rng(121)
    backbone = rng.normal(size=(500, 2))
    target = rng.normal(size=500)
    matrix = np.column_stack((backbone, target, -target))
    vector = matrix @ np.array([1.0, 1.0, 2.0, 0.0])
    layout = {
        "node_damping": slice(0, 1),
        "coupling": slice(1, 2),
        "target": slice(2, 3),
        "inactive_nuisance": slice(3, 4),
        "all": slice(0, 4),
    }
    result = audit_weak_module_blocks(
        matrix,
        vector,
        layout,
        (
            {"instance_id": "target", "family": "KNOWN"},
            {"instance_id": "inactive_nuisance", "family": "KNOWN"},
        ),
        selected_ridge=1e-8,
    )
    target_row = result["module_results"][0]
    assert target_row["all_module_columns_active"] is True
    assert target_row["global_parameter_count"] == 4
    assert target_row["global_rank"] == 3
    assert target_row["conditional_rank_gate_passed"] is False
    assert all(row["parameter_identifiability_passed"] is False for row in result["module_results"])


def test_profile_rejects_a_ranked_but_noise_dominated_module() -> None:
    rng = np.random.default_rng(13)
    q, _ = np.linalg.qr(rng.normal(size=(500, 4)))
    matrix = q[:, :3]
    vector = matrix @ np.array([1.0, 2.0, 0.2]) + 100.0 * q[:, 3]
    row = _audit(matrix, vector)["module_results"][0]
    assert row["conditional_rank_gate_passed"] is True
    assert row["profile_loss_increase"] < 0.02
    assert row["profile_gate_passed"] is False
    assert row["parameter_identifiability_passed"] is False


def _manifest() -> P1RoleManifest:
    records = tuple(
        P1EventRole(f"{role}_{index}", "synthetic", f"{role}_line_{index}", role, 1001)
        for role, count in (("fit", 3), ("selection", 1), ("internal_audit", 1))
        for index in range(count)
    )
    return P1RoleManifest(
        records,
        "F1",
        hashlib.sha256(b"saved identifiability fixture").hexdigest(),
    )


def _saved_fixture():
    manifest = _manifest()
    instance = CandidateBlockInstance(
        "memory:known",
        EDGE_MEMORY,
        time_constant=0.2,
        factor=((0.5,), (0.2,), (-0.3,)),
        port_rank=1,
    )
    model = PrunableEnergyModel(block_instances=(instance,))
    model._assign_parameters(np.array([0.2, 0.25, 0.3, 1.0, 0.8, 0.9, 1.2]), 3, 3)
    model.diagnostics = {"selected_ridge": 1e-8}
    samples = []
    for index, role in enumerate(row for row in manifest.records if row.role == "fit"):
        sample = _sample(role, manifest)
        static = sample["model_inputs"]["static"]
        static["delta0"] = np.array([[0.1 + 0.01 * index, -0.01 * index, -0.1]])
        static["omega0"] = np.array([[0.01 * index, 0.04, -0.04 - 0.01 * index]])
        delta, omega = model.rollout(*sample_rollout_inputs(sample))
        sample["training_context"]["targets"].update(delta=delta, omega=omega)
        samples.append(sample)
    context = RunContext(
        seed=324,
        horizon_steps=1001,
        fit_max_steps=1001,
        max_cases=3,
        max_internal_step=0.01,
        integral_windows=(1, 5, 20, 40, 80),
        ridge_values=(1e-8,),
        weak_max_windows_per_case_horizon=8,
    )
    return model, samples, manifest, context


def test_saved_model_wrapper_binds_complete_fit_role_and_never_mutates_core() -> None:
    model, samples, manifest, context = _saved_fixture()
    before = core_fingerprint(model)
    result = audit_saved_model_identifiability(
        model,
        samples,
        manifest,
        context,
        selected_ridge=1e-8,
    )
    assert core_fingerprint(model) == before
    assert result["role"] == "fit"
    assert result["fit_event_ids"] == list(manifest.ids("fit"))
    assert result["selection_data_reopened"] is False
    assert result["internal_audit_data_used"] is False
    assert result["matrix_shape"][1] == 7
    assert [row["instance_id"] for row in result["module_results"]] == ["memory:known"]


@pytest.mark.parametrize(
    ("attack", "match", "exception"),
    [
        ("nonfinite", "finite nonempty", ValueError),
        ("bad_layout", "partition", ValueError),
        ("bool_ridge", "selected ridge", TypeError),
        ("duplicate_module", "exactly name", ValueError),
    ],
)
def test_weak_identifiability_rejects_malformed_or_ambiguous_inputs(
    attack, match, exception
) -> None:
    matrix = np.eye(3)
    vector = np.ones(3)
    layout = _layout()
    modules = ({"instance_id": "module", "family": "KNOWN"},)
    ridge = 1e-8
    if attack == "nonfinite":
        matrix[0, 0] = np.nan
    elif attack == "bad_layout":
        layout["module"] = slice(1, 3)
    elif attack == "bool_ridge":
        ridge = True
    else:
        modules = (*modules, modules[0])
    with pytest.raises(exception, match=match):
        audit_weak_module_blocks(
            matrix,
            vector,
            layout,
            modules,
            selected_ridge=ridge,
        )
