"""Sample-based stability-region certification and margin metrics (Level 6).

These functions implement the Level 6 evaluation axis of
``docs/structure_discovery_roadmap.md``: certify the largest *sampled*
sublevel set of a candidate storage/Lyapunov function on which a decrease
condition holds, compare the certified region against a numerical
ground-truth region of attraction, and derive integration-free stability
judgements with their missed-instability rate (FSR).

Honesty boundary: a certificate here is sample-based evidence, not a formal
proof.  ``certify_sublevel_region`` certifies the largest level whose sampled
sublevel meets the decrease condition; dense sampling and a barrier
(controlling-UEP) energy strengthen it, but the certificate never outruns the
samples.  This is why the roadmap develops the Level 6 axis synthetic-first:
only against known ground truth can the fidelity metrics below be falsified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegionCertificate:
    """Outcome of a sampled sublevel-set certification attempt."""

    certified: bool
    level: float
    n_samples_inside: int
    worst_decrease_inside: float
    barrier_level: float | None
    tolerance: float
    sample_based: bool = True


@dataclass(frozen=True)
class StabilityCertificate:
    """Complete, claim-gating Level 6 certificate record.

    The record deliberately keeps statistical safety, region fidelity, margin
    quality, sample independence, and physical/numerical checks separate. A
    good classifier score alone can therefore never earn the Level 6 name.
    """

    region_conservative: bool
    stable_envelope_coverage: float
    region_symmetric_difference_ratio: float
    normalized_margin_mae: float
    margin_rank_correlation: float
    independent_unstable_groups: int
    fsr_95_upper: float
    false_stable_certificates: int
    physics_gates_passed: bool
    numerical_gates_passed: bool
    claim_boundary: str
    sample_based: bool = True

    def __post_init__(self) -> None:
        for name in (
            "region_conservative", "physics_gates_passed", "numerical_gates_passed", "sample_based"
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an explicit boolean")
        for name in ("independent_unstable_groups", "false_stable_certificates"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer count, not a boolean or fraction")
        unit_interval_fields = {
            "stable_envelope_coverage": self.stable_envelope_coverage,
            "region_symmetric_difference_ratio": self.region_symmetric_difference_ratio,
            "normalized_margin_mae": self.normalized_margin_mae,
            "fsr_95_upper": self.fsr_95_upper,
        }
        for name, value in unit_interval_fields.items():
            if isinstance(value, bool) or not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if isinstance(self.margin_rank_correlation, bool) or not np.isfinite(self.margin_rank_correlation) or not -1.0 <= self.margin_rank_correlation <= 1.0:
            raise ValueError("margin_rank_correlation must be finite and lie in [-1, 1]")
        if self.independent_unstable_groups < 0 or self.false_stable_certificates < 0:
            raise ValueError("sample counts must be nonnegative")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be explicit")

    @property
    def gates(self) -> dict[str, bool]:
        return {
            "region_conservative": self.region_conservative,
            "stable_envelope_coverage_at_least_50pct": self.stable_envelope_coverage >= 0.50,
            "region_symmetric_difference_at_most_20pct": (
                self.region_symmetric_difference_ratio <= 0.20
            ),
            "normalized_margin_mae_at_most_10pct": self.normalized_margin_mae <= 0.10,
            "margin_rank_correlation_at_least_0_8": self.margin_rank_correlation >= 0.80,
            "at_least_80_independent_unstable_groups": self.independent_unstable_groups >= 80,
            "fsr_95_upper_at_most_5pct": self.fsr_95_upper <= 0.05,
            "no_false_stable_certificates": self.false_stable_certificates == 0,
            "physics_gates_passed": self.physics_gates_passed,
            "numerical_gates_passed": self.numerical_gates_passed,
        }

    @property
    def level_6_eligible(self) -> bool:
        return all(self.gates.values())

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["gates"] = self.gates
        result["level_6_eligible"] = self.level_6_eligible
        result["status"] = "pass" if self.level_6_eligible else "fail"
        return result


def supply_identity_error(
    energy_rate: np.ndarray,
    input_power: np.ndarray,
    dissipation_rate: np.ndarray,
) -> float:
    """Maximum absolute residual of ``Hdot = input - dissipation`` over samples.

    A passive-by-construction candidate must return values at solver
    tolerance; anything larger voids the supply identity and therefore any
    downstream stability-margin claim.
    """
    rate = np.asarray(energy_rate, dtype=float).ravel()
    supply = np.asarray(input_power, dtype=float).ravel()
    dissipation = np.asarray(dissipation_rate, dtype=float).ravel()
    if not (rate.shape == supply.shape == dissipation.shape):
        raise ValueError("energy_rate, input_power and dissipation_rate must share a shape")
    if rate.size == 0:
        raise ValueError("supply identity requires at least one sample")
    residual = rate - supply + dissipation
    return float(np.max(np.abs(residual)))


def check_psd(matrix: np.ndarray, *, tolerance: float = 1e-10) -> dict:
    """Positive-semidefinite check for dissipation/transfer matrices."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("PSD check requires a square matrix")
    symmetric = 0.5 * (array + array.T)
    minimum = float(np.min(np.linalg.eigvalsh(symmetric)))
    return {"passed": bool(minimum >= -float(tolerance)), "min_eigenvalue": minimum}


def check_storage_curvature(hessian: np.ndarray, *, min_eigenvalue: float = 1e-12) -> dict:
    """Positive curvature of the storage Hessian at the certified equilibrium."""
    array = np.asarray(hessian, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("curvature check requires a square Hessian")
    symmetric = 0.5 * (array + array.T)
    minimum = float(np.min(np.linalg.eigvalsh(symmetric)))
    return {"passed": bool(minimum >= float(min_eigenvalue)), "min_eigenvalue": minimum}


def certify_sublevel_region(
    values: np.ndarray,
    decrease: np.ndarray,
    *,
    barrier_level: float | None = None,
    tolerance: float = 1e-12,
    max_levels: int = 512,
    min_samples: int = 8,
) -> RegionCertificate:
    """Certify the largest sampled sublevel ``{V <= c}`` meeting a decrease condition.

    ``values`` are storage evaluations ``V(x)`` and ``decrease`` the matching
    ``Vdot(x)`` (already including any equality-handling the caller requires)
    at the same sample points.  Candidate levels sweep upward; because
    sublevel sets are nested, the first violated level ends the search.  When
    ``barrier_level`` is given (e.g. a controlling-UEP energy), no level above
    it is considered; without it the certificate is only as strong as the
    sample coverage and must be reported as such.
    """
    v = np.asarray(values, dtype=float).ravel()
    d = np.asarray(decrease, dtype=float).ravel()
    if v.shape != d.shape:
        raise ValueError("values and decrease must share a shape")
    if v.size == 0 or not (np.isfinite(v).all() and np.isfinite(d).all()):
        raise ValueError("certification requires nonempty finite samples")
    upper = float(np.max(v)) if barrier_level is None else min(float(barrier_level), float(np.max(v)))
    unique = np.unique(v[v <= upper])
    if unique.size == 0:
        return RegionCertificate(
            certified=False, level=float("nan"), n_samples_inside=0,
            worst_decrease_inside=float("nan"),
            barrier_level=barrier_level, tolerance=float(tolerance),
        )
    levels = (
        unique if unique.size <= max_levels
        else np.linspace(float(unique[0]), upper, int(max_levels))
    )
    best: tuple[float, int, float] | None = None
    for level in levels:
        inside = v <= level
        count = int(np.count_nonzero(inside))
        if count < min_samples:
            continue
        worst = float(np.max(d[inside]))
        if worst <= float(tolerance):
            best = (float(level), count, worst)
        else:
            break
    if best is None:
        return RegionCertificate(
            certified=False, level=float("nan"), n_samples_inside=0,
            worst_decrease_inside=float("nan"),
            barrier_level=barrier_level, tolerance=float(tolerance),
        )
    return RegionCertificate(
        certified=True, level=best[0], n_samples_inside=best[1],
        worst_decrease_inside=best[2],
        barrier_level=barrier_level, tolerance=float(tolerance),
    )


def region_fidelity(
    certified_mask: np.ndarray,
    true_stable_mask: np.ndarray,
) -> dict:
    """Compare a certified region against the (numerical) true region of attraction.

    Conservativeness — certified ⊆ true — is mandatory for an honest
    stability certificate: a single falsely certified sample fails it.
    """
    certified = np.asarray(certified_mask, dtype=bool).ravel()
    truth = np.asarray(true_stable_mask, dtype=bool).ravel()
    if certified.shape != truth.shape or certified.size == 0:
        raise ValueError("region fidelity requires nonempty masks of equal shape")
    false_certified = int(np.count_nonzero(certified & ~truth))
    missed_stable = int(np.count_nonzero(truth & ~certified))
    return {
        "n_domain": int(certified.size),
        "conservative": false_certified == 0,
        "false_certified": false_certified,
        "missed_stable": missed_stable,
        "symmetric_difference_ratio": (false_certified + missed_stable) / float(certified.size),
    }


def stability_margin(certificate_level: float, storage_at_clearing: float) -> float:
    """Integration-free margin ``c - V(x_clear)``; positive means certified stable."""
    level = float(certificate_level)
    value = float(storage_at_clearing)
    if not (np.isfinite(level) and np.isfinite(value)):
        raise ValueError("margin requires finite certificate level and storage value")
    return level - value


def integration_free_stability(margins: np.ndarray, *, threshold: float = 0.0) -> np.ndarray:
    """Judge stability from margins alone: stable iff margin exceeds the threshold."""
    values = np.asarray(margins, dtype=float).ravel()
    if not np.isfinite(values).all():
        raise ValueError("integration-free judgement requires finite margins")
    return values > float(threshold)


def missed_instability_rate(
    true_stable: np.ndarray,
    predicted_stable: np.ndarray,
) -> dict:
    """FSR/FUR of a stability judgement, matching ``evaluation.stability`` conventions.

    The positive class is *unstable*: FSR = FN/(TP+FN) is the fraction of
    truly unstable cases judged stable (the dangerous error); FUR = FP/(FP+TN)
    is the fraction of truly stable cases judged unstable.
    """
    truth = np.asarray(true_stable, dtype=bool).ravel()
    predicted = np.asarray(predicted_stable, dtype=bool).ravel()
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("FSR requires nonempty masks of equal shape")
    tp = int(np.count_nonzero(~truth & ~predicted))
    fn = int(np.count_nonzero(~truth & predicted))
    fp = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(truth & predicted))
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "FSR": fn / (tp + fn) if tp + fn else float("nan"),
        "FUR": fp / (fp + tn) if fp + tn else float("nan"),
    }
