from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class DmdMode:
    frequency_hz: float
    damping_ratio: float
    decay_rate: float
    strength: float
    shape_real: tuple[float, ...]
    shape_imag: tuple[float, ...]

    def as_dict(self) -> dict[str, float | tuple[float, ...]]:
        return asdict(self)


def spectral_signature(
    values: np.ndarray,
    dt: float,
    *,
    channel_weights: np.ndarray | None = None,
    max_frequency_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a normalized, channel-aggregated one-sided power spectrum."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 4:
        raise ValueError("values must have shape (time, channels) with at least four samples")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    centered = values - np.mean(values, axis=0, keepdims=True)
    window = np.hanning(values.shape[0])[:, None]
    spectrum = np.abs(np.fft.rfft(centered * window, axis=0)) ** 2
    weights = np.ones(values.shape[1], dtype=float) if channel_weights is None else np.asarray(
        channel_weights, dtype=float
    )
    if weights.shape != (values.shape[1],) or np.any(weights < 0.0):
        raise ValueError("channel_weights must be a non-negative channel vector")
    power = spectrum @ weights
    frequency = np.fft.rfftfreq(values.shape[0], d=dt)
    keep = frequency > 0.0
    if max_frequency_hz is not None:
        keep &= frequency <= float(max_frequency_hz)
    frequency = frequency[keep]
    power = power[keep]
    total = float(np.sum(power))
    if total > 0.0:
        power = power / total
    return frequency, power


def spectral_overlap(reference_power: np.ndarray, candidate_power: np.ndarray) -> float:
    """Histogram-intersection overlap in [0, 1] for spectra on the same grid."""
    reference = np.asarray(reference_power, dtype=float)
    candidate = np.asarray(candidate_power, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("spectra must be one-dimensional arrays on the same grid")
    reference_sum = float(np.sum(reference))
    candidate_sum = float(np.sum(candidate))
    if reference_sum <= 0.0 or candidate_sum <= 0.0:
        return float(reference_sum == candidate_sum)
    return float(np.sum(np.minimum(reference / reference_sum, candidate / candidate_sum)))


def dominant_frequency(frequency: np.ndarray, power: np.ndarray) -> float:
    frequency = np.asarray(frequency, dtype=float)
    power = np.asarray(power, dtype=float)
    if frequency.shape != power.shape or frequency.ndim != 1 or frequency.size == 0:
        raise ValueError("frequency and power must be non-empty vectors on the same grid")
    return float(frequency[int(np.argmax(power))])


def modal_assurance_criterion(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Complex modal assurance criterion (MAC) in [0, 1]."""
    reference = np.asarray(reference, dtype=complex).reshape(-1)
    candidate = np.asarray(candidate, dtype=complex).reshape(-1)
    denominator = float(np.vdot(reference, reference).real * np.vdot(candidate, candidate).real)
    if denominator <= 0.0:
        return 0.0
    return float(np.abs(np.vdot(reference, candidate)) ** 2 / denominator)


def exact_dmd_modes(
    values: np.ndarray,
    dt: float,
    *,
    rank: int | None = None,
    min_frequency_hz: float = 0.05,
    max_frequency_hz: float = 5.0,
) -> list[DmdMode]:
    """Estimate effective oscillatory ringdown modes with exact DMD.

    These descriptors are empirical poles of the supplied response.  They are
    not asserted to be eigenvalues of a non-autonomous or nonlinear system.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 6:
        raise ValueError("values must have shape (time, channels) with at least six samples")
    centered = values - np.mean(values, axis=0, keepdims=True)
    snapshots = centered.T
    left, singular, right_h = np.linalg.svd(snapshots[:, :-1], full_matrices=False)
    tolerance = np.finfo(float).eps * max(snapshots[:, :-1].shape) * singular[0]
    numerical_rank = int(np.sum(singular > tolerance))
    retained = numerical_rank if rank is None else min(int(rank), numerical_rank)
    if retained < 1:
        return []
    left_r = left[:, :retained]
    singular_r = singular[:retained]
    right_r = right_h[:retained, :].conj().T
    reduced = left_r.conj().T @ snapshots[:, 1:] @ right_r @ np.diag(1.0 / singular_r)
    eigenvalues, eigenvectors = np.linalg.eig(reduced)
    modes = snapshots[:, 1:] @ right_r @ np.diag(1.0 / singular_r) @ eigenvectors
    amplitudes, *_ = np.linalg.lstsq(modes, snapshots[:, 0], rcond=None)
    continuous = np.log(eigenvalues.astype(complex)) / float(dt)
    output: list[DmdMode] = []
    for index, pole in enumerate(continuous):
        frequency = float(abs(pole.imag) / (2.0 * np.pi))
        if pole.imag <= 0.0 or frequency < min_frequency_hz or frequency > max_frequency_hz:
            continue
        magnitude = float(abs(pole))
        shape = modes[:, index]
        norm = float(np.linalg.norm(shape))
        if not np.isfinite(magnitude) or norm <= 0.0:
            continue
        shape = shape / norm
        output.append(
            DmdMode(
                frequency_hz=frequency,
                damping_ratio=float(-pole.real / magnitude),
                decay_rate=float(pole.real),
                strength=float(abs(amplitudes[index]) * norm),
                shape_real=tuple(float(value) for value in shape.real),
                shape_imag=tuple(float(value) for value in shape.imag),
            )
        )
    return sorted(output, key=lambda mode: mode.strength, reverse=True)


def match_dmd_modes(
    reference: list[DmdMode],
    candidate: list[DmdMode],
    *,
    max_relative_frequency_error: float = 0.5,
) -> list[dict[str, float]]:
    """Match empirical modes by frequency and shape using a global assignment."""
    if not reference or not candidate:
        return []
    cost = np.empty((len(reference), len(candidate)), dtype=float)
    mac = np.empty_like(cost)
    relative_frequency = np.empty_like(cost)
    for row, ref in enumerate(reference):
        ref_shape = np.asarray(ref.shape_real) + 1j * np.asarray(ref.shape_imag)
        for column, cand in enumerate(candidate):
            cand_shape = np.asarray(cand.shape_real) + 1j * np.asarray(cand.shape_imag)
            mac[row, column] = modal_assurance_criterion(ref_shape, cand_shape)
            relative_frequency[row, column] = abs(
                cand.frequency_hz - ref.frequency_hz
            ) / max(ref.frequency_hz, np.finfo(float).eps)
            cost[row, column] = relative_frequency[row, column] + 1.0 - mac[row, column]
    rows, columns = linear_sum_assignment(cost)
    matches: list[dict[str, float]] = []
    for row, column in zip(rows, columns, strict=True):
        if relative_frequency[row, column] > max_relative_frequency_error:
            continue
        ref = reference[row]
        cand = candidate[column]
        matches.append(
            {
                "reference_frequency_hz": ref.frequency_hz,
                "candidate_frequency_hz": cand.frequency_hz,
                "relative_frequency_error": float(relative_frequency[row, column]),
                "reference_damping_ratio": ref.damping_ratio,
                "candidate_damping_ratio": cand.damping_ratio,
                "absolute_damping_ratio_error": abs(cand.damping_ratio - ref.damping_ratio),
                "mac": float(mac[row, column]),
                "reference_strength": ref.strength,
                "candidate_strength": cand.strength,
            }
        )
    return matches
