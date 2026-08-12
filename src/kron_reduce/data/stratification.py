from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from kron_reduce.physics.coordinates import coi_basis


def _trapz(values: np.ndarray, time: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(time)))


def _safe_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    if total <= np.finfo(float).eps or values.size <= 1:
        return 0.0
    probability = values / total
    probability = probability[probability > 0.0]
    return float(-np.sum(probability * np.log(probability)) / np.log(values.size))


def _block_hankel(values: np.ndarray, delays: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 * delays + 2:
        raise ValueError("trajectory is too short for the requested block Hankel matrix")
    columns = len(values) - delays + 1
    return np.concatenate(
        [values[offset : offset + columns].T for offset in range(delays)], axis=0
    )


def _rank_summary(values: np.ndarray, delays: int, relative_floor: float) -> dict[str, Any]:
    hankel = _block_hankel(values, delays)
    singular = np.linalg.svd(hankel, compute_uv=False)
    if not len(singular) or singular[0] <= np.finfo(float).eps:
        return {
            "rank": 0,
            "stable_rank": 0.0,
            "condition_number": float("inf"),
            "singular_values": [],
        }
    relative = singular / singular[0]
    rank = int(np.sum(relative >= relative_floor))
    stable_rank = float(np.sum(singular**2) / singular[0] ** 2)
    retained = singular[: max(rank, 1)]
    return {
        "rank": rank,
        "stable_rank": stable_rank,
        "condition_number": float(retained[0] / max(retained[-1], np.finfo(float).eps)),
        "singular_values": singular[: min(20, len(singular))].tolist(),
    }


def _spectral_summary(values: np.ndarray, dt: float) -> dict[str, float | int]:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0, keepdims=True)
    spectrum = np.sum(np.abs(np.fft.rfft(centered * np.hanning(len(centered))[:, None], axis=0)) ** 2, axis=1)
    frequency = np.fft.rfftfreq(len(centered), d=dt)
    keep = (frequency >= 0.05) & (frequency <= 5.0)
    spectrum = spectrum[keep]
    if not len(spectrum) or float(np.sum(spectrum)) <= np.finfo(float).eps:
        return {"spectral_entropy": 0.0, "repeatable_peak_count": 0}
    normalized = spectrum / np.sum(spectrum)
    peak = normalized[1:-1] >= normalized[:-2]
    peak &= normalized[1:-1] >= normalized[2:]
    peak &= normalized[1:-1] >= max(0.01, 0.05 * float(np.max(normalized)))
    return {
        "spectral_entropy": _safe_entropy(normalized),
        "repeatable_peak_count": int(np.sum(peak)),
    }


@dataclass(frozen=True)
class StratificationPolicy:
    kinetic_scale: float
    angle_scale: float
    amplitude_edges: tuple[float, float]
    richness_edge: float
    activation_edges: tuple[float, float]
    minimum_structure_steps: int
    report_horizons: tuple[int, ...]
    hankel_delays: int
    hankel_relative_floor: float
    maximum_hankel_condition: float
    created_from_case_ids: tuple[str, ...]
    labels_used: bool = False
    validation_or_test_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def trajectory_information(
    sample: dict[str, Any],
    *,
    report_horizons: tuple[int, ...] = (100, 500, 1000),
    maximum_analysis_steps: int = 1001,
    hankel_delays: int = 12,
    hankel_relative_floor: float = 1e-3,
) -> dict[str, Any]:
    """Compute label-free excitation and observability evidence for one trajectory.

    Only the registered angle, speed and mechanical-power paths are consumed.
    Metadata is deliberately not read here, so event labels, valid-length fields,
    fault identity and source identity cannot affect the numerical stratum.
    """

    target = sample["training_context"]["targets"]
    static = sample["model_inputs"]["static"]
    time = np.asarray(target["t"], dtype=np.float64)
    delta = np.asarray(target["delta"], dtype=np.float64)
    omega = np.asarray(target["omega"], dtype=np.float64)
    pm = np.asarray(sample["training_context"]["Pm_observed"], dtype=np.float64)
    mass = np.asarray(static["mass"], dtype=np.float64).reshape(-1)
    incidence = np.asarray(static["incidence"], dtype=np.float64)
    equilibrium = np.asarray(static["equilibrium_delta"], dtype=np.float64).reshape(-1)
    steps = min(len(time), int(maximum_analysis_steps))
    time, delta, omega, pm = time[:steps], delta[:steps], omega[:steps], pm[:steps]
    finite = bool(
        np.isfinite(time).all()
        and np.isfinite(delta).all()
        and np.isfinite(omega).all()
        and np.isfinite(pm).all()
    )
    increments = np.diff(time)
    strictly_increasing = bool(len(increments) and np.all(increments > 0.0))
    dt = float(np.median(increments)) if strictly_increasing else float("nan")
    uniform_sampling = bool(
        strictly_increasing
        and np.max(np.abs(increments - dt)) <= max(1e-10, 1e-4 * dt)
    )
    coi_delta = float(np.max(np.abs(delta @ mass))) if finite else float("inf")
    coi_omega = float(np.max(np.abs(omega @ mass))) if finite else float("inf")
    power_balance = float(np.max(np.abs(np.sum(pm, axis=1)))) if finite else float("inf")
    eligibility = {
        f"eligible_h{horizon}": bool(
            finite and strictly_increasing and len(time) >= int(horizon)
        )
        for horizon in report_horizons
    }

    if not finite or not strictly_increasing or len(time) < max(8, hankel_delays * 2 + 2):
        return {
            "event_id": sample["event_id"],
            "n_real_steps": len(time),
            "finite": finite,
            "strictly_increasing": strictly_increasing,
            "uniform_sampling": uniform_sampling,
            "coi_delta_drift": coi_delta,
            "coi_omega_drift": coi_omega,
            "pm_balance_error": power_balance,
            "kinetic_integral": float("nan"),
            "angle_integral": float("nan"),
            "nonlinearity_ratio": float("nan"),
            "hankel_rank": 0,
            "hankel_half_rank": 0,
            "hankel_rank_stable": False,
            "hankel_stable_rank": 0.0,
            "hankel_condition_number": float("inf"),
            "spectral_entropy": 0.0,
            "repeatable_peak_count": 0,
            **eligibility,
        }

    edge_delta = (delta - equilibrium) @ incidence
    eta = delta @ incidence
    eta_eq = equilibrium @ incidence
    nonlinear = np.sin(eta) - np.sin(eta_eq)
    linearized = np.cos(eta_eq) * edge_delta
    kinetic = _trapz(np.sum(omega**2 * mass[None, :], axis=1), time)
    angle = _trapz(np.sum(edge_delta**2, axis=1), time)
    nonlinearity = float(
        np.linalg.norm(nonlinear - linearized)
        / max(np.linalg.norm(nonlinear), np.finfo(float).eps)
    )

    basis = coi_basis(mass)
    quotient = np.concatenate((delta @ basis, omega @ basis), axis=1)
    if len(quotient) > 256:
        indices = np.linspace(0, len(quotient) - 1, 256).round().astype(int)
        quotient = quotient[indices]
    delays = min(hankel_delays, max(2, (len(quotient) - 2) // 2))
    full = _rank_summary(quotient, delays, hankel_relative_floor)
    half_length = max(2 * delays + 2, len(quotient) // 2)
    first = _rank_summary(quotient[:half_length], delays, hankel_relative_floor)
    second = _rank_summary(quotient[-half_length:], delays, hankel_relative_floor)
    # Numerical rank grows with the number and duration of snapshots, so the
    # full-window integer rank is not directly comparable with a half window.
    # Repeatability is instead judged between two equal-length, independent
    # halves using stable rank, while the full-window rank remains a richness
    # descriptor.
    half_stable_difference = abs(first["stable_rank"] - second["stable_rank"])
    stable_tolerance = max(
        0.25,
        0.5 * min(first["stable_rank"], second["stable_rank"]),
    )
    spectrum = _spectral_summary(omega, dt)
    return {
        "event_id": sample["event_id"],
        "n_real_steps": len(time),
        "finite": finite,
        "strictly_increasing": strictly_increasing,
        "uniform_sampling": uniform_sampling,
        "coi_delta_drift": coi_delta,
        "coi_omega_drift": coi_omega,
        "pm_balance_error": power_balance,
        "kinetic_integral": kinetic,
        "angle_integral": angle,
        "nonlinearity_ratio": nonlinearity,
        "hankel_rank": int(full["rank"]),
        "hankel_half_rank": int(min(first["rank"], second["rank"])),
        "hankel_rank_stable": bool(half_stable_difference <= stable_tolerance),
        "hankel_half_stable_rank_difference": float(half_stable_difference),
        "hankel_stable_rank": float(full["stable_rank"]),
        "hankel_condition_number": float(full["condition_number"]),
        "spectral_entropy": float(spectrum["spectral_entropy"]),
        "repeatable_peak_count": int(spectrum["repeatable_peak_count"]),
        **eligibility,
    }


def fit_stratification_policy(
    train_samples: list[dict[str, Any]],
    *,
    report_horizons: tuple[int, ...] = (100, 500, 1000),
    minimum_structure_steps: int = 100,
    hankel_delays: int = 12,
    hankel_relative_floor: float = 1e-3,
    maximum_hankel_condition: float = 1e8,
) -> tuple[StratificationPolicy, list[dict[str, Any]]]:
    if not train_samples:
        raise ValueError("stratification requires nonempty train samples")
    raw = [
        trajectory_information(
            sample,
            report_horizons=report_horizons,
            hankel_delays=hankel_delays,
            hankel_relative_floor=hankel_relative_floor,
        )
        for sample in train_samples
    ]
    valid = [
        row
        for row in raw
        if row["finite"]
        and row["strictly_increasing"]
        and row["n_real_steps"] >= minimum_structure_steps
        and np.isfinite(row["kinetic_integral"])
        and np.isfinite(row["angle_integral"])
    ]
    if len(valid) < 3:
        raise ValueError("fewer than three train trajectories support stratification")
    kinetic_scale = max(float(np.median([row["kinetic_integral"] for row in valid])), 1e-12)
    angle_scale = max(float(np.median([row["angle_integral"] for row in valid])), 1e-12)
    amplitudes = np.asarray(
        [
            row["kinetic_integral"] / kinetic_scale
            + row["angle_integral"] / angle_scale
            for row in valid
        ],
        dtype=np.float64,
    )
    richness = np.asarray(
        [
            np.log1p(row["hankel_stable_rank"])
            + row["spectral_entropy"]
            + 0.25 * np.log1p(row["repeatable_peak_count"])
            for row in valid
        ],
        dtype=np.float64,
    )
    policy = StratificationPolicy(
        kinetic_scale=kinetic_scale,
        angle_scale=angle_scale,
        amplitude_edges=tuple(float(value) for value in np.quantile(amplitudes, (1 / 3, 2 / 3))),
        richness_edge=float(np.median(richness)),
        activation_edges=tuple(
            float(value)
            for value in np.quantile(amplitudes + richness, (1 / 3, 2 / 3))
        ),
        minimum_structure_steps=int(minimum_structure_steps),
        report_horizons=tuple(int(value) for value in report_horizons),
        hankel_delays=int(hankel_delays),
        hankel_relative_floor=float(hankel_relative_floor),
        maximum_hankel_condition=float(maximum_hankel_condition),
        created_from_case_ids=tuple(sample["event_id"] for sample in train_samples),
    )
    return policy, [apply_stratification(row, policy) for row in raw]


def apply_stratification(
    information: dict[str, Any], policy: StratificationPolicy
) -> dict[str, Any]:
    row = dict(information)
    kinetic = float(row.get("kinetic_integral", float("nan")))
    angle = float(row.get("angle_integral", float("nan")))
    amplitude = kinetic / policy.kinetic_scale + angle / policy.angle_scale
    richness = (
        np.log1p(float(row.get("hankel_stable_rank", 0.0)))
        + float(row.get("spectral_entropy", 0.0))
        + 0.25 * np.log1p(float(row.get("repeatable_peak_count", 0)))
    )
    if not np.isfinite(amplitude):
        amplitude_level = "A-uncertain"
    elif amplitude <= policy.amplitude_edges[0]:
        amplitude_level = "A-low"
    elif amplitude <= policy.amplitude_edges[1]:
        amplitude_level = "A-mid"
    else:
        amplitude_level = "A-high"
    rank_stable = bool(row.get("hankel_rank_stable", False))
    condition = float(row.get("hankel_condition_number", float("inf")))
    if not rank_stable or not np.isfinite(richness):
        richness_level = "R-uncertain"
    else:
        richness_level = "R-low" if richness <= policy.richness_edge else "R-high"
    basic = bool(
        row.get("finite", False)
        and row.get("strictly_increasing", False)
        and int(row.get("n_real_steps", 0)) >= policy.minimum_structure_steps
        and float(row.get("coi_delta_drift", float("inf"))) <= 1e-10
        and float(row.get("coi_omega_drift", float("inf"))) <= 1e-10
        and float(row.get("pm_balance_error", float("inf"))) <= 1e-10
    )
    if not basic:
        quality = "Q-C"
    elif rank_stable and int(row.get("hankel_rank", 0)) >= 2 and condition <= policy.maximum_hankel_condition:
        quality = "Q-A"
    else:
        quality = "Q-B"
    row.update(
        {
            "excitation_amplitude": float(amplitude),
            "richness_score": float(richness),
            "amplitude_level": amplitude_level,
            "richness_level": richness_level,
            "quality_grade": quality,
            "structure_selection_eligible": quality == "Q-A",
            "coefficient_fit_eligible": quality in {"Q-A", "Q-B"},
        }
    )
    return row


def stratify_samples(
    samples: list[dict[str, Any]], policy: StratificationPolicy
) -> list[dict[str, Any]]:
    return [
        apply_stratification(
            trajectory_information(
                sample,
                report_horizons=policy.report_horizons,
                hankel_delays=policy.hankel_delays,
                hankel_relative_floor=policy.hankel_relative_floor,
            ),
            policy,
        )
        for sample in samples
    ]


def cumulative_cohorts(
    samples: list[dict[str, Any]],
    rows: Iterable[dict[str, Any]],
    *,
    activation_edges: tuple[float, float] | None = None,
    minimum_cases: int = 3,
    allow_train_fallback: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return nested C1/C2/C3 corpora without using labels or validation truth."""

    by_id = {str(row["event_id"]): row for row in rows}
    eligible = [
        sample
        for sample in samples
        if by_id[sample["event_id"]]["coefficient_fit_eligible"]
    ]
    if len(eligible) < minimum_cases:
        raise ValueError("too few coefficient-fit-eligible trajectories")
    structure = [
        sample
        for sample in eligible
        if by_id[sample["event_id"]]["structure_selection_eligible"]
    ]
    ordered = sorted(
        structure,
        key=lambda sample: (
            by_id[sample["event_id"]]["excitation_amplitude"]
            + by_id[sample["event_id"]]["richness_score"],
            sample["event_id"],
        ),
    )
    fallback_used = False
    if activation_edges is None:
        if not allow_train_fallback:
            raise ValueError("frozen train activation edges are required")
        population = ordered if len(ordered) >= minimum_cases else sorted(
            eligible,
            key=lambda sample: (
                by_id[sample["event_id"]]["excitation_amplitude"]
                + by_id[sample["event_id"]]["richness_score"],
                sample["event_id"],
            ),
        )
        fallback_used = len(ordered) < minimum_cases
        scores = [
            by_id[sample["event_id"]]["excitation_amplitude"]
            + by_id[sample["event_id"]]["richness_score"]
            for sample in population
        ]
        activation_edges = tuple(float(value) for value in np.quantile(scores, (1 / 3, 2 / 3)))
    c1 = [
        sample
        for sample in ordered
        if by_id[sample["event_id"]]["excitation_amplitude"]
        + by_id[sample["event_id"]]["richness_score"]
        <= activation_edges[0]
    ]
    c2 = [
        sample
        for sample in ordered
        if by_id[sample["event_id"]]["excitation_amplitude"]
        + by_id[sample["event_id"]]["richness_score"]
        <= activation_edges[1]
    ]
    if allow_train_fallback and len(c1) < minimum_cases:
        c1 = ordered[: min(len(ordered), minimum_cases)]
        fallback_used = True
    if allow_train_fallback and len(c2) < len(c1):
        c2 = list(c1)
        fallback_used = True
    c3_ids = {sample["event_id"] for sample in eligible}
    cohorts = {
        "C1": c1,
        "C2": c2,
        "C3": [sample for sample in samples if sample["event_id"] in c3_ids],
    }
    return cohorts, {
        "nested": {sample["event_id"] for sample in c1}
        <= {sample["event_id"] for sample in c2}
        <= c3_ids,
        "minimum_cases": minimum_cases,
        "quality_fallback_used": fallback_used,
        "definitions": {
            "C1": "Q-A below the frozen first train activation-score tercile",
            "C2": "Q-A below the frozen second train activation-score tercile",
            "C3": "all Q-A and Q-B coefficient-fit-eligible train trajectories",
        },
        "activation_edges": list(activation_edges),
        "counts": {name: len(group) for name, group in cohorts.items()},
    }
