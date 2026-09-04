"""Frozen typed P1 five-arm, loss-balance, and closure diagnostics."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from hashlib import sha256
from math import isclose, isfinite


class PMClosureRootCause(StrEnum):
    LOSS_SCALING = "LOSS_SCALING"
    MECHANICAL_POWER_INFORMATION = "MECHANICAL_POWER_INFORMATION"
    CAUSAL_CLOSURE = "CAUSAL_CLOSURE"
    CLOSURE_FAMILY_INSUFFICIENT = "CLOSURE_FAMILY_INSUFFICIENT"
    MISSING_MODULE = "MISSING_MODULE"
    MIXED = "MIXED"
    UNRESOLVED = "UNRESOLVED"


class PMClosureArm(IntEnum):
    B2_ZOH_ORIGINAL_LOSS = 1
    CURRENT_ZOH_ORIGINAL_LOSS = 2
    CURRENT_ZOH_BALANCED_LOSS = 3
    CURRENT_OBSERVED_PM_BALANCED_LOSS_CONDITIONAL_UPPER_BOUND = 4
    CURRENT_CAUSAL_PM_CLOSURE_BALANCED_LOSS = 5


CONTROLLED_ARMS: dict[PMClosureArm, str] = {
    PMClosureArm.B2_ZOH_ORIGINAL_LOSS: "b2_zoh_original_loss",
    PMClosureArm.CURRENT_ZOH_ORIGINAL_LOSS: "current_zoh_original_loss",
    PMClosureArm.CURRENT_ZOH_BALANCED_LOSS: "current_zoh_balanced_loss",
    PMClosureArm.CURRENT_OBSERVED_PM_BALANCED_LOSS_CONDITIONAL_UPPER_BOUND: "current_observed_pm_balanced_loss_conditional_upper_bound",
    PMClosureArm.CURRENT_CAUSAL_PM_CLOSURE_BALANCED_LOSS: "current_causal_pm_closure_balanced_loss",
}


class ObservedGapRecoveryStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNDETERMINED = "UNDETERMINED"


class GradientAuditStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLE_NAMES = ("fit", "selection", "internal_audit")
_WEIGHT_DIMENSIONS = ("source", "fault_line", "event", "horizon", "output")


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


@dataclass(frozen=True)
class SharedF1EvaluationContract:
    """A real F1 role snapshot is mandatory; no placeholder contract exists."""

    role_id_hashes: Mapping[str, str]
    fold: str = "F1"
    data_roles: tuple[str, ...] = _ROLE_NAMES
    horizons_seconds: tuple[int, ...] = (1, 5, 10)
    integrator: str = "RK4"
    max_step: float = 0.01
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 324
    bootstrap_cluster_unit: str = "fault_line"
    output_metric_schema: str = "delta_omega_by_event_horizon_source_fault_line"

    def __post_init__(self) -> None:
        if self.fold != "F1" or self.data_roles != _ROLE_NAMES or self.horizons_seconds != (1, 5, 10):
            raise ValueError("P1 uses F1, exact roles, and horizons (1, 5, 10)")
        if set(self.role_id_hashes) != set(_ROLE_NAMES) or not all(_is_sha256(value) for value in self.role_id_hashes.values()):
            raise ValueError("fit/selection/internal_audit role ID hashes must be non-empty SHA-256 values")
        if self.integrator != "RK4" or self.max_step != 0.01:
            raise ValueError("P1 uses RK4 with max_step 0.01")
        if (self.bootstrap_draws, self.bootstrap_seed, self.bootstrap_cluster_unit) != (2000, 324, "fault_line"):
            raise ValueError("P1 bootstrap is exactly 2000/324/fault_line")
        if not self.output_metric_schema:
            raise ValueError("output metric schema is required")

    @property
    def contract_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FiveArmRecord:
    arm: PMClosureArm
    arm_name: str
    base_arm: PMClosureArm | None
    common_contract_hash: str
    checkpoint_arm: PMClosureArm
    checkpoint_hash: str
    future_pm: bool
    conditional_only: bool
    main_table: bool
    eligible_for_promotion: bool

    def __post_init__(self) -> None:
        if not isinstance(self.arm, PMClosureArm) or not isinstance(self.checkpoint_arm, PMClosureArm):
            raise TypeError("arm and checkpoint_arm must be PMClosureArm values")
        if self.base_arm is not None and not isinstance(self.base_arm, PMClosureArm):
            raise TypeError("base_arm must be a PMClosureArm or None")
        if not all(type(value) is bool for value in (self.future_pm, self.conditional_only, self.main_table, self.eligible_for_promotion)):
            raise TypeError("arm flags must be bool")
        if not _is_sha256(self.common_contract_hash) or not _is_sha256(self.checkpoint_hash):
            raise ValueError("common_contract_hash and checkpoint_hash must be SHA-256 values")


PMClosureArmRecord = FiveArmRecord


@dataclass(frozen=True)
class FiveArmContractAudit:
    valid: bool
    violations: tuple[str, ...]


def validate_five_arm_contract(records: Iterable[FiveArmRecord], *, contract: SharedF1EvaluationContract) -> FiveArmContractAudit:
    """Reject every order, name, role-hash, checkpoint, or flag deviation."""
    ordered = tuple(records)
    violations: list[str] = []
    required = tuple(PMClosureArm)
    if tuple(record.arm for record in ordered) != required:
        violations.append("arms_must_be_exactly_ordered_1_to_5")
    expected_bases = {required[0]: None, required[1]: required[0], required[2]: required[1], required[3]: required[2], required[4]: required[2]}
    expected_flags = {required[0]: (False, False, True, False), required[1]: (False, False, True, False), required[2]: (False, False, True, False), required[3]: (True, True, False, False), required[4]: (False, False, True, False)}
    arm3 = next((record for record in ordered if record.arm is required[2]), None)
    for record in ordered:
        if record.arm_name != CONTROLLED_ARMS.get(record.arm):
            violations.append(f"wrong_arm_name:{int(record.arm)}")
        if record.base_arm != expected_bases.get(record.arm):
            violations.append(f"wrong_base:{int(record.arm)}")
        if record.common_contract_hash != contract.contract_hash:
            violations.append(f"wrong_contract_hash:{int(record.arm)}")
        if (record.future_pm, record.conditional_only, record.main_table, record.eligible_for_promotion) != expected_flags.get(record.arm):
            violations.append(f"wrong_flags:{int(record.arm)}")
        expected_checkpoint_arm = required[2] if record.arm in (required[3], required[4]) else record.arm
        if record.checkpoint_arm is not expected_checkpoint_arm:
            violations.append(f"wrong_checkpoint_arm:{int(record.arm)}")
        if record.arm in (required[3], required[4]) and (arm3 is None or record.checkpoint_hash != arm3.checkpoint_hash):
            violations.append(f"arm{int(record.arm)}_must_reuse_arm3_checkpoint_hash")
    return FiveArmContractAudit(not violations, tuple(violations))


@dataclass(frozen=True)
class ArmEffect:
    omega_improvement_point: float
    omega_improvement_lower: float
    non_target_max_degradation_upper: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(name, value)
        if self.omega_improvement_lower > self.omega_improvement_point:
            raise ValueError("omega_improvement_lower cannot exceed its point estimate")

    @property
    def passes(self) -> bool:
        return self.omega_improvement_point >= 0.05 and self.omega_improvement_lower > 0.0 and self.non_target_max_degradation_upper <= 0.02


@dataclass(frozen=True)
class ObservedGapRecovery:
    numerator: float
    denominator: float

    def __post_init__(self) -> None:
        _finite("numerator", self.numerator)
        _finite("denominator", self.denominator)

    @classmethod
    def from_rmse(cls, rmse3: float, rmse4: float, rmse5: float) -> ObservedGapRecovery:
        for name, value in (("rmse3", rmse3), ("rmse4", rmse4), ("rmse5", rmse5)):
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        return cls(rmse3 - rmse5, rmse3 - rmse4)

    @property
    def status(self) -> ObservedGapRecoveryStatus:
        if self.denominator <= 0.0:
            return ObservedGapRecoveryStatus.UNDETERMINED
        ratio = self.numerator / self.denominator
        tolerance = 8.0 * sys.float_info.epsilon * max(1.0, abs(ratio))
        return ObservedGapRecoveryStatus.PASS if ratio >= 0.50 - tolerance else ObservedGapRecoveryStatus.FAIL

    @property
    def ratio(self) -> float | None:
        return None if self.denominator <= 0.0 else self.numerator / self.denominator


def observed_gap_recovery(numerator: float, denominator: float) -> ObservedGapRecovery:
    return ObservedGapRecovery(numerator, denominator)


def observed_gap_recovery_from_rmse(rmse3: float, rmse4: float, rmse5: float) -> ObservedGapRecovery:
    return ObservedGapRecovery.from_rmse(rmse3, rmse4, rmse5)


@dataclass(frozen=True)
class PMClosureDiagnosis:
    classification: PMClosureRootCause
    reasons: tuple[PMClosureRootCause, ...]
    conditional_only: bool
    eligible_for_promotion: bool


def classify_root_cause(*, balanced_loss: ArmEffect, observed_pm: ArmEffect, causal_closure: ArmEffect, observed_gap_recovery_ratio: float | ObservedGapRecovery, best_causal_omega_pooled_r2: float, new_residual_heldout_improvement: float, support: float) -> PMClosureDiagnosis:
    for name, value in (("best_causal_omega_pooled_r2", best_causal_omega_pooled_r2), ("new_residual_heldout_improvement", new_residual_heldout_improvement), ("support", support)):
        _finite(name, value)
    if not 0.0 <= support <= 1.0:
        raise ValueError("support must be in [0, 1]")
    if isinstance(observed_gap_recovery_ratio, ObservedGapRecovery):
        recovered = observed_gap_recovery_ratio.status is ObservedGapRecoveryStatus.PASS
    else:
        _finite("observed_gap_recovery_ratio", observed_gap_recovery_ratio)
        if not 0.0 <= observed_gap_recovery_ratio <= 1.0:
            raise ValueError("observed_gap_recovery_ratio must be in [0, 1]")
        recovered = observed_gap_recovery_ratio >= 0.50
    reasons: list[PMClosureRootCause] = []
    if balanced_loss.passes:
        reasons.append(PMClosureRootCause.LOSS_SCALING)
    if observed_pm.passes:
        reasons.extend((PMClosureRootCause.MECHANICAL_POWER_INFORMATION, PMClosureRootCause.CAUSAL_CLOSURE if causal_closure.passes and recovered else PMClosureRootCause.CLOSURE_FAMILY_INSUFFICIENT))
    if best_causal_omega_pooled_r2 <= 0.0 and new_residual_heldout_improvement >= 0.05 and support >= 0.80:
        reasons.append(PMClosureRootCause.MISSING_MODULE)
    classification = PMClosureRootCause.UNRESOLVED if not reasons else reasons[0] if len(reasons) == 1 else PMClosureRootCause.MIXED
    return PMClosureDiagnosis(classification, tuple(reasons), conditional_only=True, eligible_for_promotion=False)


@dataclass(frozen=True)
class FinalLossCellWeight:
    cell_id: str
    source: str
    fault_line: str
    event: str
    horizon: str
    output: str
    weight: float

    def __post_init__(self) -> None:
        if not self.cell_id or not all(getattr(self, dimension) for dimension in _WEIGHT_DIMENSIONS):
            raise ValueError("a final loss cell needs all hierarchical identifiers")
        _finite("weight", self.weight)

    @property
    def registered_key(self) -> tuple[str, str, str, str, str]:
        return (self.source, self.fault_line, self.event, self.horizon, self.output)


@dataclass(frozen=True)
class GradientMeasurement:
    cell_id: str
    parameter_group_id: str
    gradient: tuple[float, ...]
    share: float

    def __post_init__(self) -> None:
        if not self.cell_id or not self.parameter_group_id or not isinstance(self.gradient, tuple) or not self.gradient:
            raise ValueError("gradient cell/group IDs and a non-empty vector tuple are required")
        if not all(isfinite(value) for value in self.gradient):
            raise ValueError("gradient vector must be finite")
        _finite("share", self.share)


@dataclass(frozen=True)
class GradientAudit:
    expected_cell_ids: tuple[str, ...] = ()
    expected_parameter_group_ids: tuple[str, ...] = ()
    measurements: tuple[GradientMeasurement, ...] = ()
    total_gradients: Mapping[str, tuple[float, ...]] | None = None
    not_applicable_reason: str | None = None

    def __post_init__(self) -> None:
        measured = bool(self.expected_cell_ids or self.expected_parameter_group_ids or self.measurements or self.total_gradients is not None)
        if measured and self.not_applicable_reason is not None:
            raise ValueError("gradient audit cannot be both measured and N/A")
        if not measured and not self.not_applicable_reason:
            raise ValueError("no gradients requires an explicit N/A reason")
        if measured and (not self.expected_cell_ids or not self.expected_parameter_group_ids or self.total_gradients is None):
            raise ValueError("measured gradients require expected cells, parameter groups, and total vectors")

    @classmethod
    def not_applicable(cls, reason: str) -> GradientAudit:
        return cls(not_applicable_reason=reason)


def _gradient_audit_violations(audit: GradientAudit) -> tuple[GradientAuditStatus, list[str]]:
    if audit.not_applicable_reason is not None:
        return GradientAuditStatus.NOT_APPLICABLE, []
    violations: list[str] = []
    if len(set(audit.expected_cell_ids)) != len(audit.expected_cell_ids) or len(set(audit.expected_parameter_group_ids)) != len(audit.expected_parameter_group_ids):
        violations.append("duplicate_expected_gradient_id")
    expected = {(cell, group) for cell in audit.expected_cell_ids for group in audit.expected_parameter_group_ids}
    actual = {(measurement.cell_id, measurement.parameter_group_id) for measurement in audit.measurements}
    if len(actual) != len(audit.measurements) or actual != expected:
        violations.append("incomplete_gradient_cell_group_matrix")
    totals = audit.total_gradients or {}
    if set(totals) != set(audit.expected_parameter_group_ids):
        violations.append("missing_total_gradient_group")
    for group, vector in totals.items():
        if not isinstance(vector, tuple) or not vector or not all(isfinite(value) for value in vector):
            violations.append(f"invalid_total_gradient_vector:{group}")
    by_group: dict[str, list[GradientMeasurement]] = defaultdict(list)
    for measurement in audit.measurements:
        by_group[measurement.parameter_group_id].append(measurement)
    for group in audit.expected_parameter_group_ids:
        total = totals.get(group)
        measurements = by_group[group]
        if not isinstance(total, tuple) or not total or any(len(item.gradient) != len(total) for item in measurements):
            violations.append(f"gradient_dimension_mismatch:{group}")
            continue
        summed = tuple(sum(item.gradient[index] for item in measurements) for index in range(len(total)))
        if any(not isclose(left, right, abs_tol=1e-12) for left, right in zip(summed, total, strict=True)):
            violations.append(f"gradient_vector_not_closed:{group}")
    squared = [sum(value * value for value in measurement.gradient) for measurement in audit.measurements]
    squared_total = sum(squared)
    if any(measurement.share < 0.0 or measurement.share > 0.80 for measurement in audit.measurements):
        violations.append("scale_imbalance")
    expected_share_sum = 0.0 if squared_total == 0.0 else 1.0
    if not isclose(sum(item.share for item in audit.measurements), expected_share_sum, abs_tol=1e-12):
        violations.append("gradient_share_not_closed")
    elif squared_total > 0.0 and any(not isclose(item.share, squared[index] / squared_total, abs_tol=1e-12) for index, item in enumerate(audit.measurements)):
        violations.append("gradient_share_not_matched_to_l2_contribution")
    return (GradientAuditStatus.PASS if not violations else GradientAuditStatus.FAIL), violations


@dataclass(frozen=True)
class BalancedLossWeightAudit:
    balanced: bool
    violations: tuple[str, ...]
    final_cell_weights: tuple[FinalLossCellWeight, ...] = ()
    gradient_status: GradientAuditStatus = GradientAuditStatus.FAIL

    def as_dict(self) -> dict[str, object]:
        return {"balanced": self.balanced, "violations": list(self.violations), "final_cell_weights": [asdict(cell) for cell in self.final_cell_weights], "gradient_status": self.gradient_status.value}


def validate_balanced_loss_weights(output_weights: Mapping[str, float] | None = None, stratum_weights: Mapping[str, Mapping[str, float]] | None = None, gradient_shares: Mapping[str, float] | None = None, *, final_cell_weights: Iterable[FinalLossCellWeight] | None = None, registered_cell_keys: Iterable[tuple[str, str, str, str, str]] | None = None, gradient_audit: GradientAudit | None = None) -> BalancedLossWeightAudit:
    """Validate registered final cells, hierarchical conditional marginals, and vector gradients."""
    cells = tuple(final_cell_weights or ())
    registered = set(registered_cell_keys or ())
    violations: list[str] = []
    if output_weights is not None or stratum_weights is not None or gradient_shares is not None:
        violations.append("legacy_simplified_weights_are_incomplete")
    if not registered:
        violations.append("missing_registered_cell_keys")
    if not cells:
        violations.append("missing_final_cell_weights")
    else:
        if len({cell.cell_id for cell in cells}) != len(cells):
            violations.append("duplicate_final_cell_id")
        if len({cell.registered_key for cell in cells}) != len(cells):
            violations.append("duplicate_final_cell_key")
        if {cell.registered_key for cell in cells} != registered:
            violations.append("final_cells_do_not_match_registered_cell_keys")
        if any(cell.weight <= 0.0 for cell in cells):
            violations.append("invalid_final_cell_weight")
        if not isclose(sum(cell.weight for cell in cells), 1.0, abs_tol=1e-12):
            violations.append("final_cell_weights_do_not_sum_to_one")
        for index, dimension in enumerate(_WEIGHT_DIMENSIONS):
            grouped: dict[tuple[str, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
            for cell in cells:
                values = cell.registered_key
                grouped[values[:index]][values[index]] += cell.weight
            if any(len(values) > 1 and max(values.values()) - min(values.values()) > 1e-12 for values in grouped.values()):
                violations.append(f"conditional_marginal_not_equal:{dimension}")
    if gradient_audit is None:
        violations.append("missing_gradient_audit")
        gradient_status = GradientAuditStatus.FAIL
    else:
        gradient_status, gradient_violations = _gradient_audit_violations(gradient_audit)
        violations.extend(gradient_violations)
        if gradient_audit.not_applicable_reason is None and {cell.cell_id for cell in cells} != set(gradient_audit.expected_cell_ids):
            violations.append("gradient_expected_cell_ids_do_not_match_final_cells")
        if gradient_status is GradientAuditStatus.PASS:
            # Many small cells must not hide domination by one source or horizon.
            by_cell = {cell.cell_id: cell for cell in cells}
            for dimension in ("source", "horizon", "output"):
                shares: dict[str, float] = defaultdict(float)
                for measurement in gradient_audit.measurements:
                    cell = by_cell.get(measurement.cell_id)
                    if cell is not None:
                        shares[getattr(cell, dimension)] += measurement.share
                if len(shares) > 1 and max(shares.values()) > 0.80:
                    violations.append(f"aggregate_gradient_scale_imbalance:{dimension}")
                    gradient_status = GradientAuditStatus.FAIL
    return BalancedLossWeightAudit(not violations, tuple(violations), cells, gradient_status)
