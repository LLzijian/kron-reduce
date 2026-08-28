import pytest

from kron_reduce.evaluation.modal_observability import (
    InternalPoleObservability,
    ModalObservabilityThresholds,
    ModalSupportStatus,
    TrajectoryMatchEvidence,
    equilibrium_residual_passes,
    internal_pole_reportable,
    modal_support_status,
    trajectory_match_passes,
)
from kron_reduce.evaluation.module_contribution import (
    DeletionDegradation,
    EffectCI,
    ModuleContributionEvidence,
    ModuleContributionStatus,
    aggregate_degeneracy,
    classify_module_contribution,
)
from kron_reduce.evaluation.pm_closure_diagnostics import (
    ArmEffect,
    PMClosureRootCause,
    classify_root_cause,
    validate_balanced_loss_weights,
)


def _evidence(**overrides):
    values = {
        "direct_prediction_effect_ci": EffectCI(0.03, 0.02, 0.04),
        "physical_effect_ci": EffectCI(0.03, 0.02, 0.04),
        "deletion_primary_degradation": DeletionDegradation(0.02, 0.001, 0.02),
        "other_horizon_source_max_degradation_upper": 0.02,
        "sigma_ratio": 1e-3,
        "condition_number": 1e3,
        "profile_loss_increase": 0.03,
        "numerical_q95": 1e-5,
        "supply_identity_error": 1e-8,
    }
    values.update(overrides)
    return ModuleContributionEvidence(**values)


def test_module_boundaries_zero_effect_is_first_and_independent_support():
    zero = _evidence(
        direct_prediction_effect_ci=EffectCI(0.0009, 0.0, 0.0009),
        physical_effect_ci=EffectCI(0.00009, 0.0, 0.00009),
        sigma_ratio=0.0,
        condition_number=1e99,
        profile_loss_increase=0.0,
    )
    assert classify_module_contribution(zero).status is ModuleContributionStatus.ZERO_EFFECT
    equality = _evidence(
        direct_prediction_effect_ci=EffectCI(0.001, 0.0, 0.001),
        physical_effect_ci=EffectCI(0.0001, 0.0, 0.0001),
        sigma_ratio=0.0,
    )
    assert classify_module_contribution(equality).status is ModuleContributionStatus.UNIDENTIFIABLE
    assert classify_module_contribution(_evidence()).status is ModuleContributionStatus.INDEPENDENTLY_SUPPORTED


def test_module_replaceable_and_ambiguous_results_do_not_force_b2():
    replaceable = _evidence(
        deletion_primary_degradation=DeletionDegradation(0.01, 0.0, 0.02),
    )
    assert classify_module_contribution(replaceable).status is ModuleContributionStatus.REPLACEABLE
    unresolved = aggregate_degeneracy(
        EffectCI(0.1, 0.09, 0.1), 0.001, [ModuleContributionStatus.UNIDENTIFIABLE]
    )
    assert unresolved.degenerate_to_b2 is False
    assert not aggregate_degeneracy(
        EffectCI(0.1, 0.09, 0.1), 0.001, [ModuleContributionStatus.UNRESOLVED]
    ).degenerate_to_b2
    assert aggregate_degeneracy(EffectCI(0.0009, 0.0, 0.0009), 0.001, []).degenerate_to_b2
    assert aggregate_degeneracy(
        EffectCI(0.1, 0.09, 0.1),
        0.001,
        [ModuleContributionStatus.ZERO_EFFECT, ModuleContributionStatus.REPLACEABLE],
    ).degenerate_to_b2


def test_deletion_degradation_two_percent_boundary_is_independently_supported():
    assert classify_module_contribution(
        _evidence(deletion_primary_degradation=DeletionDegradation(0.02, 1e-6, 0.02))
    ).status is ModuleContributionStatus.INDEPENDENTLY_SUPPORTED


def test_pm_observed_arm_is_conditional_and_causal_gate_needs_half_gap_recovery():
    passing = ArmEffect(0.05, 1e-6, 0.02)
    diagnosis = classify_root_cause(
        balanced_loss=ArmEffect(0.0, 0.0, 0.0),
        observed_pm=passing,
        causal_closure=passing,
        observed_gap_recovery_ratio=0.49,
        best_causal_omega_pooled_r2=0.2,
        new_residual_heldout_improvement=0.0,
        support=0.0,
    )
    assert diagnosis.conditional_only is True
    assert diagnosis.eligible_for_promotion is False
    assert PMClosureRootCause.CLOSURE_FAMILY_INSUFFICIENT in diagnosis.reasons
    recovered = classify_root_cause(
        balanced_loss=ArmEffect(0.0, 0.0, 0.0),
        observed_pm=passing,
        causal_closure=passing,
        observed_gap_recovery_ratio=0.50,
        best_causal_omega_pooled_r2=0.2,
        new_residual_heldout_improvement=0.0,
        support=0.0,
    )
    assert PMClosureRootCause.CAUSAL_CLOSURE in recovered.reasons
    not_passing = classify_root_cause(
        balanced_loss=ArmEffect(0.0, 0.0, 0.0),
        observed_pm=ArmEffect(0.0, 0.0, 0.0),
        causal_closure=ArmEffect(0.0, 0.0, 0.0),
        observed_gap_recovery_ratio=0.0,
        best_causal_omega_pooled_r2=0.2,
        new_residual_heldout_improvement=0.0,
        support=0.0,
    )
    assert not_passing.conditional_only is True
    assert not_passing.eligible_for_promotion is False


def test_pm_mixed_and_loss_weight_gradient_attack():
    diagnosis = classify_root_cause(
        balanced_loss=ArmEffect(0.05, 0.001, 0.02),
        observed_pm=ArmEffect(0.0, 0.0, 0.0),
        causal_closure=ArmEffect(0.0, 0.0, 0.0),
        observed_gap_recovery_ratio=0.0,
        best_causal_omega_pooled_r2=0.0,
        new_residual_heldout_improvement=0.05,
        support=0.8,
    )
    assert diagnosis.classification is PMClosureRootCause.MIXED
    assert set(diagnosis.reasons) == {
        PMClosureRootCause.LOSS_SCALING,
        PMClosureRootCause.MISSING_MODULE,
    }
    stratum_weights = {
        "source": {"a": 1.0, "b": 1.0},
        "fault_line": {"a": 2.0, "b": 2.0},
        "event": {"a": 3.0, "b": 3.0},
        "horizon": {"a": 4.0, "b": 4.0},
    }
    assert validate_balanced_loss_weights(
        {"delta": 0.5, "omega": 0.5}, stratum_weights, {"source_a": 0.8}
    ).balanced
    audit = validate_balanced_loss_weights(
        {"delta": 0.5, "omega": 0.5},
        stratum_weights,
        {"source_a": 0.81},
    )
    assert audit.balanced is False
    assert "scale_imbalance" in audit.violations
    assert "missing_stratum:horizon" in validate_balanced_loss_weights(
        {"delta": 0.5, "omega": 0.5},
        {key: value for key, value in stratum_weights.items() if key != "horizon"},
    ).violations
    unequal = {**stratum_weights, "event": {"a": 3.0, "b": 3.1}}
    assert "stratum_weights_not_equal:event" in validate_balanced_loss_weights(
        {"delta": 0.5, "omega": 0.5}, unequal
    ).violations


def test_modal_observability_and_matching_gates():
    assert modal_support_status(1, 1) is ModalSupportStatus.UNDETERMINED
    assert modal_support_status(3, 3) is ModalSupportStatus.SUPPORTED
    assert modal_support_status(3, 2) is ModalSupportStatus.UNSUPPORTED
    assert not internal_pole_reportable(InternalPoleObservability(0, 1.0, 1.0))
    assert not internal_pole_reportable(InternalPoleObservability(1, 1e-3, 0.009))
    assert trajectory_match_passes(TrajectoryMatchEvidence(0.05, 0.05, 0.8))
    assert not trajectory_match_passes(TrajectoryMatchEvidence(0.051, 0.05, 0.8))
    assert equilibrium_residual_passes(1e-7)
    assert not equilibrium_residual_passes(1.01e-7)
    with pytest.raises(ValueError):
        InternalPoleObservability(1, 1.01, 0.1)
    with pytest.raises(ValueError):
        ModalObservabilityThresholds(finite_window_energy_share=1.01)
