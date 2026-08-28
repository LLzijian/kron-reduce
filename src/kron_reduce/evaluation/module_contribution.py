"""Frozen, inference-only gates for P1 module-contribution evidence.

The routines here deliberately consume already-computed counterfactual
statistics.  They do not fit or select a model, so callers can use them on a
properly frozen evaluation split without accidentally changing the evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite


class ModuleContributionStatus(StrEnum):
    INDEPENDENTLY_SUPPORTED = "INDEPENDENTLY_SUPPORTED"
    REPLACEABLE = "REPLACEABLE"
    ZERO_EFFECT = "ZERO_EFFECT"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"
    UNRESOLVED = "UNRESOLVED"


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ModuleContributionThresholds:
    prediction_floor: float = 1e-3
    prediction_numerical_multiplier: float = 10.0
    physical_floor: float = 1e-4
    physical_numerical_multiplier: float = 100.0
    deletion_degradation: float = 0.02
    sigma_ratio: float = 1e-6
    condition_number: float = 1e6
    profile_loss_increase: float = 0.02

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.condition_number <= 0.0:
            raise ValueError("condition_number must be positive")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS = ModuleContributionThresholds()


@dataclass(frozen=True)
class EffectCI:
    """A non-negative effect estimate with an ordered two-sided CI."""

    point: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not self.lower <= self.point <= self.upper:
            raise ValueError("effect CI must satisfy lower <= point <= upper")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DeletionDegradation:
    """Change in primary metric after deleting a module and refitting the rest."""

    point: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(name, value)
        if not self.lower <= self.point <= self.upper:
            raise ValueError("deletion degradation must satisfy lower <= point <= upper")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleContributionEvidence:
    direct_prediction_effect_ci: EffectCI
    physical_effect_ci: EffectCI
    deletion_primary_degradation: DeletionDegradation
    other_horizon_source_max_degradation_upper: float
    sigma_ratio: float
    condition_number: float
    profile_loss_increase: float
    numerical_q95: float
    supply_identity_error: float

    def __post_init__(self) -> None:
        for name in (
            "other_horizon_source_max_degradation_upper",
            "sigma_ratio",
            "condition_number",
            "profile_loss_increase",
            "numerical_q95",
            "supply_identity_error",
        ):
            value = getattr(self, name)
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.condition_number <= 0.0:
            raise ValueError("condition_number must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "direct_prediction_effect_ci": self.direct_prediction_effect_ci.as_dict(),
            "physical_effect_ci": self.physical_effect_ci.as_dict(),
            "deletion_primary_degradation": self.deletion_primary_degradation.as_dict(),
            "other_horizon_source_max_degradation_upper": self.other_horizon_source_max_degradation_upper,
            "sigma_ratio": self.sigma_ratio,
            "condition_number": self.condition_number,
            "profile_loss_increase": self.profile_loss_increase,
            "numerical_q95": self.numerical_q95,
            "supply_identity_error": self.supply_identity_error,
        }


@dataclass(frozen=True)
class ModuleContributionResult:
    status: ModuleContributionStatus
    tau_y: float
    tau_phys: float

    def __post_init__(self) -> None:
        _finite("tau_y", self.tau_y)
        _finite("tau_phys", self.tau_phys)
        if self.tau_y < 0.0 or self.tau_phys < 0.0:
            raise ValueError("tolerances must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "tau_y": self.tau_y, "tau_phys": self.tau_phys}


def contribution_tolerances(
    numerical_q95: float,
    supply_identity_error: float,
    thresholds: ModuleContributionThresholds = DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS,
) -> tuple[float, float]:
    """Return the frozen prediction and physical numerical floors."""
    _finite("numerical_q95", numerical_q95)
    _finite("supply_identity_error", supply_identity_error)
    if numerical_q95 < 0.0 or supply_identity_error < 0.0:
        raise ValueError("numerical errors must be non-negative")
    return (
        max(thresholds.prediction_numerical_multiplier * numerical_q95, thresholds.prediction_floor),
        max(thresholds.physical_numerical_multiplier * supply_identity_error, thresholds.physical_floor),
    )


def classify_module_contribution(
    evidence: ModuleContributionEvidence,
    thresholds: ModuleContributionThresholds = DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS,
) -> ModuleContributionResult:
    """Classify a module using the P1 priority order and strict boundaries."""
    tau_y, tau_phys = contribution_tolerances(
        evidence.numerical_q95, evidence.supply_identity_error, thresholds
    )
    prediction_zero = evidence.direct_prediction_effect_ci.upper < tau_y
    physical_zero = evidence.physical_effect_ci.upper < tau_phys
    if prediction_zero and physical_zero:
        return ModuleContributionResult(ModuleContributionStatus.ZERO_EFFECT, tau_y, tau_phys)

    unidentifiable = (
        evidence.sigma_ratio < thresholds.sigma_ratio
        or evidence.condition_number > thresholds.condition_number
        or evidence.profile_loss_increase <= thresholds.profile_loss_increase
    )
    if unidentifiable:
        return ModuleContributionResult(ModuleContributionStatus.UNIDENTIFIABLE, tau_y, tau_phys)

    direct_and_physical = (
        evidence.direct_prediction_effect_ci.lower > tau_y
        and evidence.physical_effect_ci.lower > tau_phys
    )
    deletion = evidence.deletion_primary_degradation
    independently_supported = (
        direct_and_physical
        and deletion.point >= thresholds.deletion_degradation
        and deletion.lower > 0.0
        and evidence.other_horizon_source_max_degradation_upper <= thresholds.deletion_degradation
    )
    if independently_supported:
        return ModuleContributionResult(
            ModuleContributionStatus.INDEPENDENTLY_SUPPORTED, tau_y, tau_phys
        )
    if direct_and_physical and deletion.upper <= thresholds.deletion_degradation:
        return ModuleContributionResult(ModuleContributionStatus.REPLACEABLE, tau_y, tau_phys)
    return ModuleContributionResult(ModuleContributionStatus.UNRESOLVED, tau_y, tau_phys)


@dataclass(frozen=True)
class DegeneracyAssessment:
    degenerate_to_b2: bool
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def aggregate_degeneracy(
    full_vs_b2_effect_ci: EffectCI,
    tau_y: float,
    optional_module_statuses: Iterable[ModuleContributionStatus | str],
) -> DegeneracyAssessment:
    """Decide whether the full candidate has evidence to collapse to B2.

    Ambiguous or unidentifiable module-level results are intentionally not
    evidence of degeneracy.
    """
    _finite("tau_y", tau_y)
    if tau_y < 0.0:
        raise ValueError("tau_y must be non-negative")
    statuses = tuple(ModuleContributionStatus(status) for status in optional_module_statuses)
    if full_vs_b2_effect_ci.upper < tau_y:
        return DegeneracyAssessment(True, "full_vs_b2_effect_below_prediction_floor")
    if statuses and all(
        status in {ModuleContributionStatus.ZERO_EFFECT, ModuleContributionStatus.REPLACEABLE}
        for status in statuses
    ):
        return DegeneracyAssessment(True, "all_optional_modules_zero_effect_or_replaceable")
    return DegeneracyAssessment(False, None)
