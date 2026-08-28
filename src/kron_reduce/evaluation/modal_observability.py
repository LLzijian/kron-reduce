"""Frozen, non-solving observability gates for already identified shared poles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite


class ModalSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNDETERMINED = "UNDETERMINED"


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ModalObservabilityThresholds:
    visibility: float = 1e-3
    finite_window_energy_share: float = 0.01
    min_independent_lines: int = 3
    support: float = 0.80
    frequency_relative: float = 0.05
    damping_absolute: float = 0.05
    mac: float = 0.80
    equilibrium_residual: float = 1e-7

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(name, float(value))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_independent_lines < 1:
            raise ValueError("min_independent_lines must be positive")
        for name in ("visibility", "finite_window_energy_share", "support", "mac"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be at most one")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS = ModalObservabilityThresholds()


@dataclass(frozen=True)
class InternalPoleObservability:
    port_count: int
    visibility: float
    finite_window_energy_share: float

    def __post_init__(self) -> None:
        if self.port_count < 0:
            raise ValueError("port_count must be non-negative")
        for name in ("visibility", "finite_window_energy_share"):
            value = getattr(self, name)
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if value > 1.0:
                raise ValueError(f"{name} must be at most one")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def internal_pole_reportable(
    evidence: InternalPoleObservability,
    thresholds: ModalObservabilityThresholds = DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS,
) -> bool:
    """A pole needs a non-zero port and local visibility/energy to be reported."""
    return bool(
        evidence.port_count > 0
        and evidence.visibility >= thresholds.visibility
        and evidence.finite_window_energy_share >= thresholds.finite_window_energy_share
    )


def modal_support_status(
    independent_lines: int,
    supporting_lines: int,
    thresholds: ModalObservabilityThresholds = DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS,
) -> ModalSupportStatus:
    """Return UNDETERMINED before support fractions are interpreted."""
    if independent_lines < 0 or supporting_lines < 0:
        raise ValueError("line counts must be non-negative")
    if supporting_lines > independent_lines:
        raise ValueError("supporting_lines cannot exceed independent_lines")
    if independent_lines < thresholds.min_independent_lines:
        return ModalSupportStatus.UNDETERMINED
    return (
        ModalSupportStatus.SUPPORTED
        if supporting_lines / independent_lines >= thresholds.support
        else ModalSupportStatus.UNSUPPORTED
    )


@dataclass(frozen=True)
class TrajectoryMatchEvidence:
    frequency_relative_error: float
    damping_absolute_error: float
    mac: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.mac > 1.0:
            raise ValueError("mac must be at most one")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def trajectory_match_passes(
    evidence: TrajectoryMatchEvidence,
    thresholds: ModalObservabilityThresholds = DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS,
) -> bool:
    return bool(
        evidence.frequency_relative_error <= thresholds.frequency_relative
        and evidence.damping_absolute_error <= thresholds.damping_absolute
        and evidence.mac >= thresholds.mac
    )


def equilibrium_residual_passes(
    residual: float,
    thresholds: ModalObservabilityThresholds = DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS,
) -> bool:
    _finite("residual", residual)
    if residual < 0.0:
        raise ValueError("residual must be non-negative")
    return residual <= thresholds.equilibrium_residual
