from hashlib import sha256

import pytest

from kron_reduce.evaluation.modal_observability import (
    InternalPoleObservability,
    ModalBootstrapEvidence,
    ModalCompositeEvidence,
    ModalSupportStatus,
    ModalViewEvidence,
    TrajectoryMatchEvidence,
    internal_pole_reportable,
    modal_composite_status,
    modal_view_status,
)
from kron_reduce.evaluation.module_contribution import (
    DegeneracyStatus,
    DeletionCell,
    DeletionDegradation,
    EffectCI,
    JointB2Evidence,
    ModuleContributionEvidence,
    ModuleContributionStatus,
    aggregate_degeneracy,
    classify_module_contribution,
)
from kron_reduce.evaluation.pm_closure_diagnostics import (
    CONTROLLED_ARMS,
    FinalLossCellWeight,
    FiveArmRecord,
    GradientAudit,
    GradientAuditStatus,
    GradientMeasurement,
    ObservedGapRecoveryStatus,
    PMClosureArm,
    SharedF1EvaluationContract,
    observed_gap_recovery_from_rmse,
    validate_balanced_loss_weights,
    validate_five_arm_contract,
)


def _effect(lower: float = 0.02, upper: float = 0.04) -> EffectCI:
    return EffectCI((lower + upper) / 2, lower, upper)


def _evidence(**overrides: object) -> ModuleContributionEvidence:
    values: dict[str, object] = {
        "direct_prediction_effect_ci": _effect(),
        "storage_effect_ci": _effect(),
        "port_effect_ci": _effect(),
        "deletion_cells": (
            DeletionCell("primary_h10", DeletionDegradation(0.03, 0.020001, 0.04), primary=True),
            DeletionCell("source_a_h1", DeletionDegradation(0.01, 0.0, 0.01)),
        ),
        "sigma_ratio": 1e-3,
        "condition_number": 1e3,
        "profile_loss_increase": 0.03,
        "numerical_q95": 1e-5,
        "supply_identity_error": 1e-8,
    }
    values.update(overrides)
    return ModuleContributionEvidence(**values)  # type: ignore[arg-type]


def test_module_zero_effect_uses_three_inclusive_upper_bounds() -> None:
    zero = _evidence(
        direct_prediction_effect_ci=EffectCI(0.001, 0.0, 0.001),
        storage_effect_ci=EffectCI(0.0001, 0.0, 0.0001),
        port_effect_ci=EffectCI(0.0001, 0.0, 0.0001),
        sigma_ratio=0.0,
    )
    assert classify_module_contribution(zero).status is ModuleContributionStatus.ZERO_EFFECT
    assert classify_module_contribution(_evidence()).status is ModuleContributionStatus.INDEPENDENTLY_SUPPORTED


def test_module_identifiability_replaceable_and_cross_boundary() -> None:
    equality = _evidence(direct_prediction_effect_ci=EffectCI(0.01, 0.001, 0.02), sigma_ratio=0.0)
    assert classify_module_contribution(equality).status is ModuleContributionStatus.UNIDENTIFIABLE
    replaceable = _evidence(
        deletion_cells=(
            DeletionCell("primary", DeletionDegradation(0.02, 0.02, 0.02), primary=True),
            DeletionCell("other", DeletionDegradation(0.02, 0.0, 0.02)),
        )
    )
    assert classify_module_contribution(replaceable).status is ModuleContributionStatus.REPLACEABLE
    unresolved = _evidence(deletion_cells=(DeletionCell("primary", DeletionDegradation(0.02, 0.02, 0.03), primary=True),))
    assert classify_module_contribution(unresolved).status is ModuleContributionStatus.UNRESOLVED
    high_but_singular = _evidence(sigma_ratio=0.0)
    assert classify_module_contribution(high_but_singular).status is ModuleContributionStatus.UNIDENTIFIABLE


def test_unshifted_percentile_intervals_need_not_contain_the_point_estimate() -> None:
    effect = EffectCI(0.01, 0.02, 0.03)
    deletion = DeletionDegradation(0.04, -0.01, 0.02)
    assert effect.as_dict() == {"point": 0.01, "lower": 0.02, "upper": 0.03}
    assert deletion.as_dict() == {"point": 0.04, "lower": -0.01, "upper": 0.02}
    with pytest.raises(ValueError, match="lower <= upper"):
        EffectCI(0.01, 0.03, 0.02)
    with pytest.raises(ValueError, match="lower <= upper"):
        DeletionDegradation(0.01, 0.03, 0.02)


def _joint(*, effect: EffectCI | None = None, deletion: DeletionDegradation | None = None) -> JointB2Evidence:
    effect = effect or EffectCI(0.0005, 0.0, 0.001)
    deletion = deletion or DeletionDegradation(0.01, 0.0, 0.02)
    return JointB2Evidence(
        effect,
        effect,
        effect,
        (
            DeletionCell("s1-1-delta", deletion, source="s1", horizon=1, output="delta"),
            DeletionCell("s1-1-omega", deletion, source="s1", horizon=1, output="omega"),
        ),
        ("s1",),
        (1,),
        ("delta", "omega"),
    )


def test_joint_b2_requires_three_effects_and_complete_registered_cells() -> None:
    result = aggregate_degeneracy(tau_y=0.001, tau_phys=0.001, joint_evidence=_joint())
    assert result.status is DegeneracyStatus.PROVEN
    with pytest.raises(ValueError, match="exactly cover"):
        JointB2Evidence(_effect(), _effect(), _effect(), (), ("s1",), (1,), ("delta",))


def test_joint_b2_refuted_only_by_strict_lower_otherwise_unresolved() -> None:
    refuted = aggregate_degeneracy(
        tau_y=0.001,
        tau_phys=0.001,
        joint_evidence=_joint(effect=EffectCI(0.02, 0.0011, 0.03)),
    )
    assert refuted.status is DegeneracyStatus.REFUTED
    unresolved = aggregate_degeneracy(
        tau_y=0.001,
        tau_phys=0.001,
        joint_evidence=_joint(effect=EffectCI(0.001, 0.0, 0.002)),
    )
    assert unresolved.status is DegeneracyStatus.UNRESOLVED
    assert aggregate_degeneracy(EffectCI(0.0, 0.0, 0.0), 0.001).status is DegeneracyStatus.UNRESOLVED


def _modal_view(**overrides: object) -> ModalViewEvidence:
    values: dict[str, object] = {
        "view_name": "registered_view",
        "preregistered": True,
        "equilibrium_residual": 1e-7,
        "io_rank": 1,
        "visibility": 1e-3,
        "finite_window_energy_share": 0.01,
        "independent_lines": 3,
        "bootstrap_support": 0.8,
        "bootstrap": ModalBootstrapEvidence(2000, 324, "fault_line"),
        "trajectory_match": TrajectoryMatchEvidence(0.05, 0.05, 0.8),
    }
    values.update(overrides)
    return ModalViewEvidence(**values)  # type: ignore[arg-type]


def test_internal_pole_uses_rank_and_legacy_count_alias_without_visibility_cap() -> None:
    assert internal_pole_reportable(InternalPoleObservability(port_rank=1, visibility=2.0, finite_window_energy_share=0.01))
    legacy = InternalPoleObservability(port_count=1, visibility=1e-3, finite_window_energy_share=0.01)
    assert legacy.port_rank == legacy.port_count == 1


def test_modal_composite_strict_bootstrap_and_line_count_statuses() -> None:
    assert modal_composite_status(ModalCompositeEvidence((_modal_view(),))) is ModalSupportStatus.SUPPORTED
    assert modal_composite_status(ModalCompositeEvidence((_modal_view(independent_lines=2),))) is ModalSupportStatus.UNDETERMINED
    invalid = _modal_view(bootstrap=ModalBootstrapEvidence(1999, 324, "fault_line"))
    assert modal_composite_status(ModalCompositeEvidence((invalid,))) is ModalSupportStatus.INVALID
    assert modal_composite_status(ModalCompositeEvidence((_modal_view(io_rank=0),))) is ModalSupportStatus.UNSUPPORTED
    assert modal_composite_status(ModalCompositeEvidence((_modal_view(preregistered=False),))) is ModalSupportStatus.UNSUPPORTED
    assert modal_view_status(_modal_view(preregistered=False)) is ModalSupportStatus.UNSUPPORTED
    with pytest.raises(ValueError):
        ModalBootstrapEvidence(True, 324, "fault_line")


def _hash(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def test_shared_contract_requires_role_id_hashes() -> None:
    with pytest.raises(TypeError):
        SharedF1EvaluationContract()
    with pytest.raises(ValueError, match="role ID hashes"):
        SharedF1EvaluationContract(role_id_hashes={"fit": _hash("fit")})


CONTRACT = SharedF1EvaluationContract(
    role_id_hashes={role: _hash(f"role:{role}") for role in ("fit", "selection", "internal_audit")}
)


def _arms() -> list[FiveArmRecord]:
    h = CONTRACT.contract_hash
    arm3_checkpoint = _hash("arm3-core-checkpoint")
    return [
        FiveArmRecord(PMClosureArm.B2_ZOH_ORIGINAL_LOSS, CONTROLLED_ARMS[PMClosureArm.B2_ZOH_ORIGINAL_LOSS], None, h, PMClosureArm.B2_ZOH_ORIGINAL_LOSS, _hash("arm1"), False, False, True, False),
        FiveArmRecord(PMClosureArm.CURRENT_ZOH_ORIGINAL_LOSS, CONTROLLED_ARMS[PMClosureArm.CURRENT_ZOH_ORIGINAL_LOSS], PMClosureArm.B2_ZOH_ORIGINAL_LOSS, h, PMClosureArm.CURRENT_ZOH_ORIGINAL_LOSS, _hash("arm2"), False, False, True, False),
        FiveArmRecord(PMClosureArm.CURRENT_ZOH_BALANCED_LOSS, CONTROLLED_ARMS[PMClosureArm.CURRENT_ZOH_BALANCED_LOSS], PMClosureArm.CURRENT_ZOH_ORIGINAL_LOSS, h, PMClosureArm.CURRENT_ZOH_BALANCED_LOSS, arm3_checkpoint, False, False, True, False),
        FiveArmRecord(PMClosureArm.CURRENT_OBSERVED_PM_BALANCED_LOSS_CONDITIONAL_UPPER_BOUND, CONTROLLED_ARMS[PMClosureArm.CURRENT_OBSERVED_PM_BALANCED_LOSS_CONDITIONAL_UPPER_BOUND], PMClosureArm.CURRENT_ZOH_BALANCED_LOSS, h, PMClosureArm.CURRENT_ZOH_BALANCED_LOSS, arm3_checkpoint, True, True, False, False),
        FiveArmRecord(PMClosureArm.CURRENT_CAUSAL_PM_CLOSURE_BALANCED_LOSS, CONTROLLED_ARMS[PMClosureArm.CURRENT_CAUSAL_PM_CLOSURE_BALANCED_LOSS], PMClosureArm.CURRENT_ZOH_BALANCED_LOSS, h, PMClosureArm.CURRENT_ZOH_BALANCED_LOSS, arm3_checkpoint, False, False, True, False),
    ]


def test_five_arm_contract_locks_names_flags_and_checkpoint_hashes() -> None:
    assert validate_five_arm_contract(_arms(), contract=CONTRACT).valid
    wrong = _arms()
    wrong[3] = FiveArmRecord(PMClosureArm.CURRENT_OBSERVED_PM_BALANCED_LOSS_CONDITIONAL_UPPER_BOUND, "bad", PMClosureArm.CURRENT_ZOH_BALANCED_LOSS, wrong[3].common_contract_hash, PMClosureArm.CURRENT_ZOH_BALANCED_LOSS, _hash("not-arm3"), True, True, True, False)
    violations = validate_five_arm_contract(wrong, contract=CONTRACT).violations
    assert {"wrong_arm_name:4", "wrong_flags:4", "arm4_must_reuse_arm3_checkpoint_hash"} <= set(violations)
    assert not validate_five_arm_contract(_arms()[:-1], contract=CONTRACT).valid


def test_observed_gap_is_computed_from_three_rmse_arms() -> None:
    assert observed_gap_recovery_from_rmse(1.0, 0.6, 0.8).status is ObservedGapRecoveryStatus.PASS
    assert observed_gap_recovery_from_rmse(1.0, 1.0, 0.8).status is ObservedGapRecoveryStatus.UNDETERMINED
    assert observed_gap_recovery_from_rmse(1.0, 0.6, 0.9).status is ObservedGapRecoveryStatus.FAIL
    assert observed_gap_recovery_from_rmse(2e-13, 1e-13, 2e-13).status is ObservedGapRecoveryStatus.FAIL


def _final_cells() -> tuple[FinalLossCellWeight, ...]:
    return (
        FinalLossCellWeight("a-delta", "a", "la", "ea", "1", "delta", 0.25),
        FinalLossCellWeight("a-omega", "a", "la", "ea", "1", "omega", 0.25),
        FinalLossCellWeight("b-delta", "b", "lb", "eb", "1", "delta", 0.25),
        FinalLossCellWeight("b-omega", "b", "lb", "eb", "1", "omega", 0.25),
    )


def _gradient_audit() -> GradientAudit:
    return GradientAudit(
        expected_cell_ids=tuple(cell.cell_id for cell in _final_cells()),
        expected_parameter_group_ids=("core",),
        measurements=tuple(GradientMeasurement(cell.cell_id, "core", (1.0, 0.0), 0.25) for cell in _final_cells()),
        total_gradients={"core": (4.0, 0.0)},
    )


def test_final_hierarchical_cell_weights_sum_and_conditionally_balance() -> None:
    registered = tuple(cell.registered_key for cell in _final_cells())
    audit = validate_balanced_loss_weights(final_cell_weights=_final_cells(), registered_cell_keys=registered, gradient_audit=_gradient_audit())
    assert audit.balanced and audit.gradient_status is GradientAuditStatus.PASS
    missing_registered = validate_balanced_loss_weights(
        final_cell_weights=_final_cells(),
        registered_cell_keys=(*registered, ("missing", "line", "event", "1", "delta")),
        gradient_audit=_gradient_audit(),
    )
    assert "final_cells_do_not_match_registered_cell_keys" in missing_registered.violations
    legacy = validate_balanced_loss_weights({"delta": 0.5, "omega": 0.5}, {"source": {"a": 1.0}}, registered_cell_keys=registered, gradient_audit=GradientAudit.not_applicable("frozen"))
    assert not legacy.balanced and "legacy_simplified_weights_are_incomplete" in legacy.violations


def test_gradient_requires_complete_cell_group_matrix_and_explicit_na() -> None:
    registered = tuple(cell.registered_key for cell in _final_cells())
    incomplete = GradientAudit(("a", "b"), ("core",), (GradientMeasurement("a", "core", (1.0,), 1.0),), {"core": (1.0,)})
    audit = validate_balanced_loss_weights(final_cell_weights=_final_cells(), registered_cell_keys=registered, gradient_audit=incomplete)
    assert not audit.balanced and "incomplete_gradient_cell_group_matrix" in audit.violations
    cancelling = GradientAudit(
        tuple(cell.cell_id for cell in _final_cells()),
        ("core",),
        (
            GradientMeasurement("a-delta", "core", (1.0, -1.0), 0.25),
            GradientMeasurement("a-omega", "core", (-1.0, 1.0), 0.25),
            GradientMeasurement("b-delta", "core", (1.0, -1.0), 0.25),
            GradientMeasurement("b-omega", "core", (-1.0, 1.0), 0.25),
        ),
        {"core": (0.0, 0.0)},
    )
    cancellation_audit = validate_balanced_loss_weights(final_cell_weights=_final_cells(), registered_cell_keys=registered, gradient_audit=cancelling)
    assert cancellation_audit.balanced and cancellation_audit.gradient_status is GradientAuditStatus.PASS
    na = validate_balanced_loss_weights(final_cell_weights=_final_cells(), registered_cell_keys=registered, gradient_audit=GradientAudit.not_applicable("no trainable parameters"))
    assert na.balanced and na.gradient_status is GradientAuditStatus.NOT_APPLICABLE
    with pytest.raises(ValueError, match="explicit N/A"):
        GradientAudit()


def test_small_individual_gradients_cannot_hide_source_domination() -> None:
    cells = _final_cells()
    gradients = (7.0, 7.0, 1.0, 1.0)
    audit = GradientAudit(
        expected_cell_ids=tuple(cell.cell_id for cell in cells),
        expected_parameter_group_ids=("core",),
        measurements=tuple(
            GradientMeasurement(cell.cell_id, "core", (gradient,), gradient**2 / 100.0)
            for cell, gradient in zip(cells, gradients, strict=True)
        ),
        total_gradients={"core": (16.0,)},
    )
    result = validate_balanced_loss_weights(
        final_cell_weights=cells,
        registered_cell_keys=tuple(cell.registered_key for cell in cells),
        gradient_audit=audit,
    )
    assert not result.balanced
    assert "aggregate_gradient_scale_imbalance:source" in result.violations
