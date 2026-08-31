"""Frozen, inference-only P1 module-contribution evidence gates.

The gate distinguishes zeroing a fitted module from deleting it and refitting
the remainder.  Replaceability is never evidence that all modules collapse to
B2; that needs an explicit joint counterfactual.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite


class ModuleContributionStatus(StrEnum):
    INDEPENDENTLY_SUPPORTED = "INDEPENDENTLY_SUPPORTED"
    REPLACEABLE = "REPLACEABLE"
    ZERO_EFFECT = "ZERO_EFFECT"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"
    UNRESOLVED = "UNRESOLVED"


class DegeneracyStatus(StrEnum):
    PROVEN = "PROVEN"
    DEGENERATE_TO_B2 = "PROVEN"  # Backward-compatible spelling.
    REFUTED = "REFUTED"
    NOT_DEGENERATE = "REFUTED"  # Backward-compatible spelling.
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
    """Primary-metric change after deleting a module and refitting the rest."""

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
class DeletionCell:
    """A preregistered delete/refit cell; primary cells decide independence."""

    name: str
    degradation: DeletionDegradation
    primary: bool = False
    source: str | None = None
    horizon: float | None = None
    output: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("deletion cell name is required")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "degradation": self.degradation.as_dict(),
            "primary": self.primary,
            "source": self.source,
            "horizon": self.horizon,
            "output": self.output,
        }


@dataclass(frozen=True, init=False)
class ModuleContributionEvidence:
    """Three zeroing effects and every preregistered delete/refit cell.

    The legacy physical/deletion arguments are read compatibility only.  They
    are expanded conservatively and cannot restore the old B2 inference.
    """

    direct_prediction_effect_ci: EffectCI
    storage_effect_ci: EffectCI
    port_effect_ci: EffectCI
    deletion_cells: tuple[DeletionCell, ...]
    sigma_ratio: float
    condition_number: float
    profile_loss_increase: float
    numerical_q95: float
    supply_identity_error: float

    def __init__(
        self,
        *,
        direct_prediction_effect_ci: EffectCI,
        storage_effect_ci: EffectCI | None = None,
        port_effect_ci: EffectCI | None = None,
        deletion_cells: Iterable[DeletionCell] | Mapping[str, DeletionDegradation] | None = None,
        sigma_ratio: float,
        condition_number: float,
        profile_loss_increase: float,
        numerical_q95: float,
        supply_identity_error: float,
        physical_effect_ci: EffectCI | None = None,
        deletion_primary_degradation: DeletionDegradation | None = None,
        other_horizon_source_max_degradation_upper: float | None = None,
    ) -> None:
        if storage_effect_ci is None:
            storage_effect_ci = physical_effect_ci
        if port_effect_ci is None:
            port_effect_ci = physical_effect_ci
        if storage_effect_ci is None or port_effect_ci is None:
            raise ValueError("storage_effect_ci and port_effect_ci are required")
        if deletion_cells is None:
            cells: tuple[DeletionCell, ...] = ()
            if deletion_primary_degradation is not None:
                cells = (DeletionCell("legacy_primary", deletion_primary_degradation, primary=True),)
            if other_horizon_source_max_degradation_upper is not None:
                other = DeletionDegradation(
                    other_horizon_source_max_degradation_upper,
                    0.0,
                    other_horizon_source_max_degradation_upper,
                )
                cells += (DeletionCell("legacy_other_horizon_source", other),)
        elif isinstance(deletion_cells, Mapping):
            cells = tuple(DeletionCell(str(name), value) for name, value in deletion_cells.items())
        else:
            cells = tuple(deletion_cells)
        if not cells:
            raise ValueError("at least one registered deletion cell is required")
        if len({cell.name for cell in cells}) != len(cells):
            raise ValueError("deletion cell names must be unique")
        for name, value in (
            ("sigma_ratio", sigma_ratio),
            ("condition_number", condition_number),
            ("profile_loss_increase", profile_loss_increase),
            ("numerical_q95", numerical_q95),
            ("supply_identity_error", supply_identity_error),
        ):
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if condition_number <= 0.0:
            raise ValueError("condition_number must be positive")
        for field, value in (
            ("direct_prediction_effect_ci", direct_prediction_effect_ci),
            ("storage_effect_ci", storage_effect_ci),
            ("port_effect_ci", port_effect_ci),
            ("deletion_cells", cells),
            ("sigma_ratio", sigma_ratio),
            ("condition_number", condition_number),
            ("profile_loss_increase", profile_loss_increase),
            ("numerical_q95", numerical_q95),
            ("supply_identity_error", supply_identity_error),
        ):
            object.__setattr__(self, field, value)

    @property
    def physical_effect_ci(self) -> EffectCI:
        if self.storage_effect_ci != self.port_effect_ci:
            raise AttributeError("physical_effect_ci is ambiguous; use storage_effect_ci/port_effect_ci")
        return self.storage_effect_ci

    @property
    def deletion_primary_degradation(self) -> DeletionDegradation:
        primary = next((cell.degradation for cell in self.deletion_cells if cell.primary), None)
        if primary is None:
            raise AttributeError("no primary deletion cell is registered")
        return primary

    @property
    def other_horizon_source_max_degradation_upper(self) -> float:
        return max((cell.degradation.upper for cell in self.deletion_cells if not cell.primary), default=0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "direct_prediction_effect_ci": self.direct_prediction_effect_ci.as_dict(),
            "storage_effect_ci": self.storage_effect_ci.as_dict(),
            "port_effect_ci": self.port_effect_ci.as_dict(),
            "deletion_cells": [cell.as_dict() for cell in self.deletion_cells],
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
        for name, value in (("tau_y", self.tau_y), ("tau_phys", self.tau_phys)):
            _finite(name, value)
            if value < 0.0:
                raise ValueError("tolerances must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "tau_y": self.tau_y, "tau_phys": self.tau_phys}


def contribution_tolerances(
    numerical_q95: float,
    supply_identity_error: float,
    thresholds: ModuleContributionThresholds = DEFAULT_MODULE_CONTRIBUTION_THRESHOLDS,
) -> tuple[float, float]:
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
    """Apply frozen strict boundaries.  All cross-boundary cases are unresolved."""
    tau_y, tau_phys = contribution_tolerances(
        evidence.numerical_q95, evidence.supply_identity_error, thresholds
    )
    effects = (
        (evidence.direct_prediction_effect_ci, tau_y),
        (evidence.storage_effect_ci, tau_phys),
        (evidence.port_effect_ci, tau_phys),
    )
    if all(effect.upper <= tolerance for effect, tolerance in effects):
        return ModuleContributionResult(ModuleContributionStatus.ZERO_EFFECT, tau_y, tau_phys)
    if (
        evidence.sigma_ratio < thresholds.sigma_ratio
        or evidence.condition_number > thresholds.condition_number
        or evidence.profile_loss_increase <= thresholds.profile_loss_increase
    ):
        return ModuleContributionResult(ModuleContributionStatus.UNIDENTIFIABLE, tau_y, tau_phys)
    nonzero_established = all(effect.lower > tolerance for effect, tolerance in effects)
    primary_cells = tuple(cell for cell in evidence.deletion_cells if cell.primary)
    if nonzero_established and any(
        cell.degradation.lower > thresholds.deletion_degradation for cell in primary_cells
    ):
        return ModuleContributionResult(ModuleContributionStatus.INDEPENDENTLY_SUPPORTED, tau_y, tau_phys)
    if nonzero_established and all(
        cell.degradation.upper <= thresholds.deletion_degradation for cell in evidence.deletion_cells
    ):
        return ModuleContributionResult(ModuleContributionStatus.REPLACEABLE, tau_y, tau_phys)
    return ModuleContributionResult(ModuleContributionStatus.UNRESOLVED, tau_y, tau_phys)


@dataclass(frozen=True)
class JointB2Evidence:
    """The only admissible evidence for a full-candidate B2 collapse."""

    joint_prediction_effect_ci: EffectCI
    joint_storage_effect_ci: EffectCI
    joint_port_effect_ci: EffectCI
    joint_delete_refit_b2_cells: tuple[DeletionCell, ...]
    registered_sources: tuple[str, ...]
    registered_horizons: tuple[float, ...]
    registered_outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.registered_sources or not self.registered_horizons or not self.registered_outputs:
            raise ValueError("registered source, horizon, and output cells are required")
        expected = {
            (source, horizon, output)
            for source in self.registered_sources
            for horizon in self.registered_horizons
            for output in self.registered_outputs
        }
        actual = {(cell.source, cell.horizon, cell.output) for cell in self.joint_delete_refit_b2_cells}
        if None in {value for cell in actual for value in cell}:
            raise ValueError("joint deletion cells require source, horizon, and output")
        if actual != expected or len(actual) != len(self.joint_delete_refit_b2_cells):
            raise ValueError("joint deletion cells must exactly cover registered source×horizon×output")

    @property
    def joint_zero_effect_ci(self) -> EffectCI:
        """Removed ambiguous legacy aggregate effect alias."""
        raise AttributeError("use joint_prediction_effect_ci/storage_effect_ci/port_effect_ci")

    @property
    def joint_delete_refit_b2_degradation(self) -> DeletionDegradation:
        """Removed ambiguous legacy aggregate delete/refit alias."""
        raise AttributeError("use joint_delete_refit_b2_cells")


@dataclass(frozen=True)
class DegeneracyAssessment:
    status: DegeneracyStatus
    reason: str | None = None

    @property
    def degenerate_to_b2(self) -> bool:
        return self.status is DegeneracyStatus.PROVEN

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "degenerate_to_b2": self.degenerate_to_b2, "reason": self.reason}


def aggregate_degeneracy(
    full_vs_b2_effect_ci: EffectCI | None = None,
    tau_y: float | None = None,
    optional_module_statuses: Iterable[ModuleContributionStatus | str] = (),
    *,
    joint_evidence: JointB2Evidence | None = None,
    joint_delete_refit_b2_degradation: DeletionDegradation | None = None,
    deletion_threshold: float = 0.02,
    tau_phys: float | None = None,
) -> DegeneracyAssessment:
    """Use only explicit joint-zero and joint-delete/refit evidence for B2."""
    _ = tuple(optional_module_statuses)  # Former per-module inference is intentionally ignored.
    if tau_y is None:
        raise ValueError("tau_y is required")
    _finite("tau_y", tau_y)
    _finite("deletion_threshold", deletion_threshold)
    if tau_y < 0.0 or deletion_threshold < 0.0:
        raise ValueError("thresholds must be non-negative")
    if tau_phys is None:
        tau_phys = tau_y
    _finite("tau_phys", tau_phys)
    if tau_phys < 0.0:
        raise ValueError("tau_phys must be non-negative")
    if joint_evidence is not None and (full_vs_b2_effect_ci is not None or joint_delete_refit_b2_degradation is not None):
        raise ValueError("joint_evidence cannot be mixed with legacy aggregate inputs")
    if joint_evidence is None:
        return DegeneracyAssessment(DegeneracyStatus.UNRESOLVED, "missing_joint_delete_refit_b2_evidence")
    effects = (
        (joint_evidence.joint_prediction_effect_ci, tau_y),
        (joint_evidence.joint_storage_effect_ci, tau_phys),
        (joint_evidence.joint_port_effect_ci, tau_phys),
    )
    if all(effect.upper <= tolerance for effect, tolerance in effects) and all(
        cell.degradation.upper <= deletion_threshold for cell in joint_evidence.joint_delete_refit_b2_cells
    ):
        return DegeneracyAssessment(DegeneracyStatus.PROVEN, "joint_three_effects_and_delete_refit_cells")
    if any(effect.lower > tolerance for effect, tolerance in effects) or any(
        cell.degradation.lower > deletion_threshold for cell in joint_evidence.joint_delete_refit_b2_cells
    ):
        return DegeneracyAssessment(DegeneracyStatus.REFUTED, "joint_effect_or_delete_refit_cell_exceeds_threshold")
    return DegeneracyAssessment(DegeneracyStatus.UNRESOLVED, "joint_counterfactual_crosses_threshold")
