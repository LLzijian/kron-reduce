"""Frozen, non-solving observability gates for already identified shared poles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite


class ModalSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNDETERMINED = "UNDETERMINED"
    INVALID = "INVALID"


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


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
        for name in ("finite_window_energy_share", "support", "mac"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be at most one")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS = ModalObservabilityThresholds()


@dataclass(frozen=True, init=False)
class InternalPoleObservability:
    port_rank: int
    visibility: float
    finite_window_energy_share: float

    def __init__(
        self,
        port_rank: int | None = None,
        visibility: float = 0.0,
        finite_window_energy_share: float = 0.0,
        *,
        port_count: int | None = None,
    ) -> None:
        if port_rank is None:
            port_rank = port_count
        elif port_count is not None and port_rank != port_count:
            raise ValueError("port_rank and legacy port_count disagree")
        if port_rank is None:
            raise ValueError("port_rank must be provided")
        _nonnegative_int("port_rank", port_rank)
        for name in ("visibility", "finite_window_energy_share"):
            value = visibility if name == "visibility" else finite_window_energy_share
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if finite_window_energy_share > 1.0:
            raise ValueError("finite_window_energy_share must be at most one")
        object.__setattr__(self, "port_rank", port_rank)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "finite_window_energy_share", finite_window_energy_share)

    @property
    def port_count(self) -> int:
        """Read-only compatibility alias for pre-P1 rank naming."""
        return self.port_rank

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def internal_pole_reportable(
    evidence: InternalPoleObservability,
    thresholds: ModalObservabilityThresholds = DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS,
) -> bool:
    """A pole needs a non-zero port and local visibility/energy to be reported."""
    return bool(
        evidence.port_rank > 0
        and evidence.visibility >= thresholds.visibility
        and evidence.finite_window_energy_share >= thresholds.finite_window_energy_share
    )


def modal_support_status(
    independent_lines: int,
    supporting_lines: int,
    thresholds: ModalObservabilityThresholds = DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS,
) -> ModalSupportStatus:
    """Return UNDETERMINED before support fractions are interpreted."""
    _nonnegative_int("independent_lines", independent_lines)
    _nonnegative_int("supporting_lines", supporting_lines)
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


@dataclass(frozen=True)
class ModalBootstrapEvidence:
    """The registered resampling contract for one local observability view."""

    draws: int
    seed: int
    cluster_unit: str

    def __post_init__(self) -> None:
        _nonnegative_int("bootstrap draws", self.draws)
        _nonnegative_int("bootstrap seed", self.seed)

    @property
    def registered(self) -> bool:
        return self.draws == 2000 and self.seed == 324 and self.cluster_unit == "fault_line"


@dataclass(frozen=True)
class ModalViewEvidence:
    """Typed composite evidence for one preregistered excitation view."""

    view_name: str
    preregistered: bool
    equilibrium_residual: float
    io_rank: int
    visibility: float
    finite_window_energy_share: float
    independent_lines: int
    bootstrap_support: float
    bootstrap: ModalBootstrapEvidence
    trajectory_match: TrajectoryMatchEvidence

    def __post_init__(self) -> None:
        if not self.view_name:
            raise ValueError("view_name is required")
        if type(self.preregistered) is not bool:
            raise ValueError("preregistered must be bool")
        _nonnegative_int("io_rank", self.io_rank)
        _nonnegative_int("independent_lines", self.independent_lines)
        for name in (
            "equilibrium_residual",
            "visibility",
            "finite_window_energy_share",
            "bootstrap_support",
        ):
            value = getattr(self, name)
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("finite_window_energy_share", "bootstrap_support"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be at most one")


def modal_view_status(
    evidence: ModalViewEvidence,
    thresholds: ModalObservabilityThresholds = DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS,
) -> ModalSupportStatus:
    """Assess a view without re-solving modes or pooling excitation labels."""
    if not evidence.preregistered:
        return ModalSupportStatus.UNSUPPORTED
    if evidence.independent_lines < thresholds.min_independent_lines:
        return ModalSupportStatus.UNDETERMINED
    if not evidence.bootstrap.registered:
        return ModalSupportStatus.INVALID
    passes = (
        equilibrium_residual_passes(evidence.equilibrium_residual, thresholds)
        and evidence.io_rank > 0
        and evidence.visibility >= thresholds.visibility
        and evidence.finite_window_energy_share >= thresholds.finite_window_energy_share
        and evidence.bootstrap_support >= thresholds.support
        and trajectory_match_passes(evidence.trajectory_match, thresholds)
    )
    return ModalSupportStatus.SUPPORTED if passes else ModalSupportStatus.UNSUPPORTED


@dataclass(frozen=True)
class ModalCompositeEvidence:
    """Local observability audits for one shared-equation pole."""

    views: tuple[ModalViewEvidence, ...]

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("at least one modal view is required")
        names = [view.view_name for view in self.views]
        if len(names) != len(set(names)):
            raise ValueError("modal view names must be unique")


def modal_composite_status(
    evidence: ModalCompositeEvidence,
    thresholds: ModalObservabilityThresholds = DEFAULT_MODAL_OBSERVABILITY_THRESHOLDS,
) -> ModalSupportStatus:
    """Support needs one fully passing preregistered, adequately sampled view."""
    registered = tuple(view for view in evidence.views if view.preregistered)
    if not registered:
        return ModalSupportStatus.UNSUPPORTED
    statuses = tuple(modal_view_status(view, thresholds) for view in registered)
    if ModalSupportStatus.INVALID in statuses:
        return ModalSupportStatus.INVALID
    if ModalSupportStatus.SUPPORTED in statuses:
        return ModalSupportStatus.SUPPORTED
    if all(status is ModalSupportStatus.UNDETERMINED for status in statuses):
        return ModalSupportStatus.UNDETERMINED
    return ModalSupportStatus.UNSUPPORTED


# More explicit spelling for external call sites.
evaluate_modal_composite = modal_composite_status
