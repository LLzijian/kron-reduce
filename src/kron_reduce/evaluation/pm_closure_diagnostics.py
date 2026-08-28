"""Pure P1 gates for separating loss, power-information, and closure causes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite


class PMClosureRootCause(StrEnum):
    LOSS_SCALING = "LOSS_SCALING"
    MECHANICAL_POWER_INFORMATION = "MECHANICAL_POWER_INFORMATION"
    CAUSAL_CLOSURE = "CAUSAL_CLOSURE"
    CLOSURE_FAMILY_INSUFFICIENT = "CLOSURE_FAMILY_INSUFFICIENT"
    MISSING_MODULE = "MISSING_MODULE"
    MIXED = "MIXED"
    UNRESOLVED = "UNRESOLVED"


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ArmEffect:
    """A 10-s omega improvement and the worst non-target degradation bound."""

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
        return (
            self.omega_improvement_point >= 0.05
            and self.omega_improvement_lower > 0.0
            and self.non_target_max_degradation_upper <= 0.02
        )

    def as_dict(self) -> dict[str, float | bool]:
        return {**asdict(self), "passes": self.passes}


@dataclass(frozen=True)
class PMClosureDiagnosis:
    classification: PMClosureRootCause
    reasons: tuple[PMClosureRootCause, ...]
    conditional_only: bool
    eligible_for_promotion: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification.value,
            "reasons": [reason.value for reason in self.reasons],
            "conditional_only": self.conditional_only,
            "eligible_for_promotion": self.eligible_for_promotion,
        }


def classify_root_cause(
    *,
    balanced_loss: ArmEffect,
    observed_pm: ArmEffect,
    causal_closure: ArmEffect,
    observed_gap_recovery_ratio: float,
    best_causal_omega_pooled_r2: float,
    new_residual_heldout_improvement: float,
    support: float,
) -> PMClosureDiagnosis:
    """Classify precomputed factorial-arm evidence without fitting anything.

    Observed ``Pm`` is deliberately kept as a conditional upper bound.  It
    never makes a result eligible for promotion, even when its arm passes.
    """
    for name, value in (
        ("observed_gap_recovery_ratio", observed_gap_recovery_ratio),
        ("best_causal_omega_pooled_r2", best_causal_omega_pooled_r2),
        ("new_residual_heldout_improvement", new_residual_heldout_improvement),
        ("support", support),
    ):
        _finite(name, value)
    if not 0.0 <= observed_gap_recovery_ratio <= 1.0:
        raise ValueError("observed_gap_recovery_ratio must be in [0, 1]")
    if not 0.0 <= support <= 1.0:
        raise ValueError("support must be in [0, 1]")

    reasons: list[PMClosureRootCause] = []
    if balanced_loss.passes:
        reasons.append(PMClosureRootCause.LOSS_SCALING)
    if observed_pm.passes:
        reasons.append(PMClosureRootCause.MECHANICAL_POWER_INFORMATION)
        if causal_closure.passes and observed_gap_recovery_ratio >= 0.50:
            reasons.append(PMClosureRootCause.CAUSAL_CLOSURE)
        else:
            reasons.append(PMClosureRootCause.CLOSURE_FAMILY_INSUFFICIENT)
    if (
        best_causal_omega_pooled_r2 <= 0.0
        and new_residual_heldout_improvement >= 0.05
        and support >= 0.80
    ):
        reasons.append(PMClosureRootCause.MISSING_MODULE)

    if not reasons:
        classification = PMClosureRootCause.UNRESOLVED
    elif len(reasons) == 1:
        classification = reasons[0]
    else:
        classification = PMClosureRootCause.MIXED
    return PMClosureDiagnosis(
        classification=classification,
        reasons=tuple(reasons),
        conditional_only=True,
        eligible_for_promotion=False,
    )


@dataclass(frozen=True)
class BalancedLossWeightAudit:
    balanced: bool
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"balanced": self.balanced, "violations": list(self.violations)}


def validate_balanced_loss_weights(
    output_weights: Mapping[str, float],
    stratum_weights: Mapping[str, Mapping[str, float]],
    gradient_shares: Mapping[str, float] | None = None,
) -> BalancedLossWeightAudit:
    """Validate 50/50 outputs and equal weights within every required stratum."""
    violations: list[str] = []
    for key in ("delta", "omega"):
        value = output_weights.get(key)
        if value is None or not isfinite(value) or abs(value - 0.5) > 1e-12:
            violations.append(f"{key}_weight")

    for dimension in ("source", "fault_line", "event", "horizon"):
        weights = stratum_weights.get(dimension)
        if not weights:
            violations.append(f"missing_stratum:{dimension}")
            continue
        values = tuple(weights.values())
        if any(not isfinite(value) or value <= 0.0 for value in values):
            violations.append(f"invalid_stratum_weight:{dimension}")
        elif max(values) - min(values) > 1e-12:
            violations.append(f"stratum_weights_not_equal:{dimension}")

    for name, share in (gradient_shares or {}).items():
        if not isfinite(share) or share < 0.0 or share > 1.0:
            violations.append(f"invalid_gradient_share:{name}")
        elif share > 0.80:
            violations.append("scale_imbalance")
    return BalancedLossWeightAudit(not violations, tuple(violations))
