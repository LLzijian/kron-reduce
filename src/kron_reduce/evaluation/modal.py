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


@dataclass(frozen=True)
class ConsensusDmdMode:
    """A mode retained by an input-aware order/regularization stability map."""

    mode: DmdMode
    support_fraction: float
    supporting_fits: int
    total_fits: int
    frequency_mad_hz: float
    damping_mad: float

    def as_dict(self) -> dict[str, object]:
        return {
            **self.mode.as_dict(),
            "support_fraction": self.support_fraction,
            "supporting_fits": self.supporting_fits,
            "total_fits": self.total_fits,
            "frequency_mad_hz": self.frequency_mad_hz,
            "damping_mad": self.damping_mad,
        }


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
    with np.errstate(divide="ignore", invalid="ignore"):
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


def input_aware_delay_modes(
    values: np.ndarray,
    inputs: np.ndarray,
    dt: float,
    *,
    delays: int = 8,
    ridge: float = 1e-6,
    min_frequency_hz: float = 0.05,
    max_frequency_hz: float = 5.0,
    maximum_discrete_radius: float = 1.05,
) -> list[DmdMode]:
    """Estimate autonomous modes of a forced response with a delay DMDc model.

    The fitted predictor is a multivariable ARX realization::

        y[k+1] = sum_j A[j] y[k-j] + sum_j B[j] u[k-j].

    Its companion matrix contains only the autonomous ``A`` blocks; the known
    input is therefore not folded into the reported poles.  Delay coordinates
    permit observable hidden states to appear without claiming that they are
    measured physical device states.  The routine is intended as a modal
    diagnostic/teacher, not as the final nonlinear rollout model.
    """
    values = np.asarray(values, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    if values.ndim != 2 or inputs.ndim != 2 or len(values) != len(inputs):
        raise ValueError("values and inputs must be aligned two-dimensional paths")
    if not np.isfinite(values).all() or not np.isfinite(inputs).all():
        raise ValueError("values and inputs must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    delays = int(delays)
    if delays < 1 or len(values) <= delays + 2:
        raise ValueError("delays must leave at least three regression rows")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    if maximum_discrete_radius <= 0.0:
        raise ValueError("maximum_discrete_radius must be positive")

    # Equilibrium/operating-point centering is essential: an affine offset is
    # not an oscillatory state, and input centering prevents the mean schedule
    # from acting as a spurious autonomous mode.
    output = values - np.mean(values, axis=0, keepdims=True)
    forcing = inputs - np.mean(inputs, axis=0, keepdims=True)
    n_output = output.shape[1]
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    initial_delays: list[np.ndarray] = []
    for index in range(delays - 1, len(output) - 1):
        output_delay = np.concatenate(
            [output[index - lag] for lag in range(delays)]
        )
        input_delay = np.concatenate(
            [forcing[index - lag] for lag in range(delays)]
        )
        rows.append(np.concatenate((output_delay, input_delay)))
        targets.append(output[index + 1])
        initial_delays.append(output_delay)
    design = np.asarray(rows)
    target = np.asarray(targets)
    scale = np.sqrt(np.mean(design**2, axis=0))
    informative = scale > max(float(np.max(scale)) * 1e-12, np.finfo(float).eps)
    scale = np.where(informative, scale, 1.0)
    normalized = design / scale
    gram = normalized.T @ normalized
    regularizer = ridge * np.eye(gram.shape[0])
    coefficients = np.linalg.solve(gram + regularizer, normalized.T @ target)
    coefficients = coefficients / scale[:, None]

    state_size = delays * n_output
    top = coefficients[:state_size].T
    companion = np.zeros((state_size, state_size), dtype=float)
    companion[:n_output] = top
    if delays > 1:
        companion[n_output:, :-n_output] = np.eye(state_size - n_output)
    eigenvalues, eigenvectors = np.linalg.eig(companion)

    # Estimate modal amplitudes over all delay states rather than from a single
    # clearing sample, which is particularly fragile when a mode starts near a
    # node of its oscillation.
    delay_states = np.asarray(initial_delays).T
    amplitudes, *_ = np.linalg.lstsq(eigenvectors, delay_states, rcond=None)
    amplitude = np.sqrt(np.mean(np.abs(amplitudes) ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        continuous = np.log(eigenvalues.astype(complex)) / float(dt)
    modes: list[DmdMode] = []
    for index, pole in enumerate(continuous):
        discrete = eigenvalues[index]
        frequency = float(abs(pole.imag) / (2.0 * np.pi))
        if (
            pole.imag <= 0.0
            or frequency < min_frequency_hz
            or frequency > max_frequency_hz
            or abs(discrete) > maximum_discrete_radius
        ):
            continue
        shape = eigenvectors[:n_output, index]
        norm = float(np.linalg.norm(shape))
        magnitude = float(abs(pole))
        if not np.isfinite(magnitude) or norm <= np.finfo(float).eps:
            continue
        shape = shape / norm
        modes.append(
            DmdMode(
                frequency_hz=frequency,
                damping_ratio=float(-pole.real / magnitude),
                decay_rate=float(pole.real),
                strength=float(amplitude[index] * norm),
                shape_real=tuple(float(value) for value in shape.real),
                shape_imag=tuple(float(value) for value in shape.imag),
            )
        )
    return sorted(modes, key=lambda mode: mode.strength, reverse=True)


def input_aware_modal_consensus(
    values: np.ndarray,
    inputs: np.ndarray,
    dt: float,
    *,
    delay_orders: tuple[int, ...] = (4, 6, 8, 10),
    ridge_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
    minimum_support_fraction: float = 0.5,
    frequency_tolerance: float = 0.08,
    minimum_mac: float = 0.6,
    minimum_damping_ratio: float = 0.0,
    maximum_damping_ratio: float = 0.5,
    min_frequency_hz: float = 0.05,
    max_frequency_hz: float = 5.0,
) -> list[ConsensusDmdMode]:
    """Retain poles repeated across a delay-DMDc stabilization diagram.

    Ordinary DMD can fold a measured forcing schedule into its autonomous
    operator.  A single high-order DMDc/ARX fit, however, can replace that bias
    with poorly conditioned spurious poles.  This routine therefore varies
    only two pre-registered numerical choices (delay order and ridge), clusters
    modes by frequency and complex MAC, and keeps modes supported by a stated
    fraction of fits.  It is deliberately a low-capacity diagnostic and does
    not train the final nonlinear rollout model.
    """
    if not 0 < minimum_support_fraction <= 1:
        raise ValueError("minimum_support_fraction must lie in (0, 1]")
    if not 0 < frequency_tolerance < 1:
        raise ValueError("frequency_tolerance must lie in (0, 1)")
    if not 0 <= minimum_mac <= 1:
        raise ValueError("minimum_mac must lie in [0, 1]")
    orders = tuple(sorted({int(value) for value in delay_orders if int(value) > 0}))
    ridges = tuple(sorted({float(value) for value in ridge_values}))
    if not orders or not ridges or any(value < 0 or not np.isfinite(value) for value in ridges):
        raise ValueError("delay orders and finite nonnegative ridge values are required")

    fits: list[list[DmdMode]] = []
    for order in orders:
        for ridge in ridges:
            modes = input_aware_delay_modes(
                values,
                inputs,
                dt,
                delays=order,
                ridge=ridge,
                min_frequency_hz=min_frequency_hz,
                max_frequency_hz=max_frequency_hz,
            )
            fits.append(
                [
                    mode
                    for mode in modes
                    if minimum_damping_ratio <= mode.damping_ratio <= maximum_damping_ratio
                ]
            )
    total_fits = len(fits)
    observations = sorted(
        (
            (mode.frequency_hz, fit_index, mode)
            for fit_index, modes in enumerate(fits)
            for mode in modes
        ),
        key=lambda item: (item[0], item[1], -item[2].strength),
    )
    clusters: list[list[tuple[int, DmdMode]]] = []
    for _, fit_index, mode in observations:
        shape = np.asarray(mode.shape_real)+1j*np.asarray(mode.shape_imag)
        best_cluster = None
        best_cost = float("inf")
        for cluster_index, cluster in enumerate(clusters):
            frequencies = np.asarray([member.frequency_hz for _, member in cluster])
            center = float(np.median(frequencies))
            relative = abs(mode.frequency_hz-center)/max(center, np.finfo(float).eps)
            representative = max(cluster, key=lambda item: item[1].strength)[1]
            representative_shape = np.asarray(representative.shape_real)+1j*np.asarray(representative.shape_imag)
            mac = modal_assurance_criterion(representative_shape, shape)
            cost = relative+1.0-mac
            if relative <= frequency_tolerance and mac >= minimum_mac and cost < best_cost:
                best_cluster, best_cost = cluster_index, cost
        if best_cluster is None:
            clusters.append([(fit_index, mode)])
        else:
            # At most one pole from each numerical fit contributes to a
            # cluster; keep its strongest instance to prevent repeated roots
            # from inflating support.
            cluster = clusters[best_cluster]
            existing = next((index for index, item in enumerate(cluster) if item[0] == fit_index), None)
            if existing is None:
                cluster.append((fit_index, mode))
            elif mode.strength > cluster[existing][1].strength:
                cluster[existing] = (fit_index, mode)

    retained: list[ConsensusDmdMode] = []
    for cluster in clusters:
        support = len({fit_index for fit_index, _ in cluster})/total_fits
        if support+1e-15 < minimum_support_fraction:
            continue
        members = [mode for _, mode in cluster]
        frequencies = np.asarray([mode.frequency_hz for mode in members])
        dampings = np.asarray([mode.damping_ratio for mode in members])
        strengths = np.asarray([mode.strength for mode in members])
        reference = max(members, key=lambda mode: mode.strength)
        reference_shape = np.asarray(reference.shape_real)+1j*np.asarray(reference.shape_imag)
        aligned = []
        for member in members:
            shape = np.asarray(member.shape_real)+1j*np.asarray(member.shape_imag)
            phase = np.vdot(reference_shape, shape)
            if abs(phase) > np.finfo(float).eps:
                shape = shape*np.exp(-1j*np.angle(phase))
            aligned.append(shape)
        consensus_shape = np.mean(aligned, axis=0)
        norm = np.linalg.norm(consensus_shape)
        if norm <= np.finfo(float).eps:
            continue
        consensus_shape = consensus_shape/norm
        frequency = float(np.median(frequencies))
        damping = float(np.median(dampings))
        magnitude = 2*np.pi*frequency/max(np.sqrt(max(1.0-damping**2, 1e-12)), 1e-12)
        retained.append(
            ConsensusDmdMode(
                mode=DmdMode(
                    frequency_hz=frequency,
                    damping_ratio=damping,
                    decay_rate=float(-damping*magnitude),
                    strength=float(np.median(strengths)),
                    shape_real=tuple(float(value) for value in consensus_shape.real),
                    shape_imag=tuple(float(value) for value in consensus_shape.imag),
                ),
                support_fraction=float(support),
                supporting_fits=len(cluster),
                total_fits=total_fits,
                frequency_mad_hz=float(np.median(np.abs(frequencies-frequency))),
                damping_mad=float(np.median(np.abs(dampings-damping))),
            )
        )
    return sorted(retained, key=lambda item: (-item.support_fraction, -item.mode.strength))


def pooled_input_aware_delay_modes(
    value_paths: list[np.ndarray] | tuple[np.ndarray, ...],
    input_paths: list[np.ndarray] | tuple[np.ndarray, ...],
    dt: float,
    *,
    delays: int = 8,
    ridge: float = 1e-4,
    min_frequency_hz: float = 0.05,
    max_frequency_hz: float = 5.0,
    maximum_discrete_radius: float = 1.05,
) -> list[DmdMode]:
    """Estimate a common forced realization from multiple independent paths.

    Rows are pooled only after centering each trajectory independently; no
    artificial transition is introduced between fault events.  The operation
    expresses the scientific hypothesis that the events share one local
    state matrix while their initial conditions and forcing schedules differ.
    """
    if len(value_paths) != len(input_paths) or not value_paths:
        raise ValueError("equal nonempty value and input path collections are required")
    delays = int(delays)
    if delays < 1:
        raise ValueError("delays must be positive")
    outputs, forcings = [], []
    n_output = n_input = None
    for raw_values, raw_inputs in zip(value_paths, input_paths, strict=True):
        values = np.asarray(raw_values, dtype=float)
        inputs = np.asarray(raw_inputs, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        if values.ndim != 2 or inputs.ndim != 2 or len(values) != len(inputs):
            raise ValueError("each value/input path must be aligned and two-dimensional")
        if len(values) <= delays+2 or not np.isfinite(values).all() or not np.isfinite(inputs).all():
            raise ValueError("each path must be finite and longer than the delay order")
        if n_output is None:
            n_output, n_input = values.shape[1], inputs.shape[1]
        if values.shape[1] != n_output or inputs.shape[1] != n_input:
            raise ValueError("all paths must share output and input dimensions")
        outputs.append(values-np.mean(values, axis=0, keepdims=True))
        forcings.append(inputs-np.mean(inputs, axis=0, keepdims=True))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    assert n_output is not None
    rows, targets, initial_delays = [], [], []
    for output, forcing in zip(outputs, forcings, strict=True):
        for index in range(delays-1, len(output)-1):
            output_delay = np.concatenate([output[index-lag] for lag in range(delays)])
            input_delay = np.concatenate([forcing[index-lag] for lag in range(delays)])
            rows.append(np.concatenate((output_delay, input_delay)))
            targets.append(output[index+1])
            initial_delays.append(output_delay)
    design, target = np.asarray(rows), np.asarray(targets)
    scale = np.sqrt(np.mean(design**2, axis=0))
    informative = scale > max(float(np.max(scale))*1e-12, np.finfo(float).eps)
    scale = np.where(informative, scale, 1.0)
    normalized = design/scale
    gram = normalized.T@normalized+float(ridge)*np.eye(normalized.shape[1])
    coefficients = np.linalg.solve(gram, normalized.T@target)/scale[:, None]
    state_size = delays*n_output
    companion = np.zeros((state_size, state_size), dtype=float)
    companion[:n_output] = coefficients[:state_size].T
    if delays > 1:
        companion[n_output:, :-n_output] = np.eye(state_size-n_output)
    eigenvalues, eigenvectors = np.linalg.eig(companion)
    delay_states = np.asarray(initial_delays).T
    amplitudes, *_ = np.linalg.lstsq(eigenvectors, delay_states, rcond=None)
    amplitude = np.sqrt(np.mean(np.abs(amplitudes)**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        continuous = np.log(eigenvalues.astype(complex))/float(dt)
    modes: list[DmdMode] = []
    for index, pole in enumerate(continuous):
        frequency = float(abs(pole.imag)/(2*np.pi))
        if (
            pole.imag <= 0
            or frequency < min_frequency_hz
            or frequency > max_frequency_hz
            or abs(eigenvalues[index]) > maximum_discrete_radius
        ):
            continue
        shape = eigenvectors[:n_output, index]
        norm, magnitude = float(np.linalg.norm(shape)), float(abs(pole))
        if not np.isfinite(magnitude) or norm <= np.finfo(float).eps:
            continue
        shape = shape/norm
        modes.append(DmdMode(
            frequency_hz=frequency,
            damping_ratio=float(-pole.real/magnitude),
            decay_rate=float(pole.real),
            strength=float(amplitude[index]*norm),
            shape_real=tuple(float(value) for value in shape.real),
            shape_imag=tuple(float(value) for value in shape.imag),
        ))
    return sorted(modes, key=lambda mode: mode.strength, reverse=True)


def pooled_input_aware_modal_consensus(
    value_paths: list[np.ndarray] | tuple[np.ndarray, ...],
    input_paths: list[np.ndarray] | tuple[np.ndarray, ...],
    dt: float,
    *,
    delay_orders: tuple[int, ...] = (4, 6, 8, 10),
    ridge_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
    minimum_support_fraction: float = 0.5,
    frequency_tolerance: float = 0.08,
    minimum_mac: float = 0.6,
    minimum_damping_ratio: float = 0.0,
    maximum_damping_ratio: float = 0.5,
    min_frequency_hz: float = 0.05,
    max_frequency_hz: float = 5.0,
) -> list[ConsensusDmdMode]:
    """Stabilization consensus for a shared multi-trajectory DMDc model."""
    if not 0 < minimum_support_fraction <= 1:
        raise ValueError("minimum_support_fraction must lie in (0, 1]")
    orders = tuple(sorted({int(value) for value in delay_orders if int(value) > 0}))
    ridges = tuple(sorted({float(value) for value in ridge_values}))
    fits: list[list[DmdMode]] = []
    for order in orders:
        for ridge in ridges:
            modes = pooled_input_aware_delay_modes(
                value_paths, input_paths, dt,
                delays=order, ridge=ridge,
                min_frequency_hz=min_frequency_hz,
                max_frequency_hz=max_frequency_hz,
            )
            fits.append([
                mode for mode in modes
                if minimum_damping_ratio <= mode.damping_ratio <= maximum_damping_ratio
            ])
    # Reuse the deterministic stability-map clustering by temporarily
    # providing the precomputed fits to the same local algorithm.  Keeping it
    # explicit here avoids silently concatenating independent trajectories.
    total_fits = len(fits)
    observations = sorted(
        ((mode.frequency_hz, fit_index, mode) for fit_index, modes in enumerate(fits) for mode in modes),
        key=lambda item: (item[0], item[1], -item[2].strength),
    )
    clusters: list[list[tuple[int, DmdMode]]] = []
    for _, fit_index, mode in observations:
        shape = np.asarray(mode.shape_real)+1j*np.asarray(mode.shape_imag)
        choice, best_cost = None, float("inf")
        for cluster_index, cluster in enumerate(clusters):
            center = float(np.median([member.frequency_hz for _, member in cluster]))
            relative = abs(mode.frequency_hz-center)/max(center, np.finfo(float).eps)
            representative = max(cluster, key=lambda item: item[1].strength)[1]
            representative_shape = np.asarray(representative.shape_real)+1j*np.asarray(representative.shape_imag)
            mac = modal_assurance_criterion(representative_shape, shape)
            cost = relative+1-mac
            if relative <= frequency_tolerance and mac >= minimum_mac and cost < best_cost:
                choice, best_cost = cluster_index, cost
        if choice is None:
            clusters.append([(fit_index, mode)])
        else:
            cluster = clusters[choice]
            existing = next((index for index, item in enumerate(cluster) if item[0] == fit_index), None)
            if existing is None:
                cluster.append((fit_index, mode))
            elif mode.strength > cluster[existing][1].strength:
                cluster[existing] = (fit_index, mode)
    retained: list[ConsensusDmdMode] = []
    for cluster in clusters:
        support = len({fit_index for fit_index, _ in cluster})/max(total_fits, 1)
        if support+1e-15 < minimum_support_fraction:
            continue
        members = [mode for _, mode in cluster]
        frequencies = np.asarray([mode.frequency_hz for mode in members])
        dampings = np.asarray([mode.damping_ratio for mode in members])
        strengths = np.asarray([mode.strength for mode in members])
        reference = max(members, key=lambda mode: mode.strength)
        reference_shape = np.asarray(reference.shape_real)+1j*np.asarray(reference.shape_imag)
        aligned = []
        for member in members:
            shape = np.asarray(member.shape_real)+1j*np.asarray(member.shape_imag)
            phase = np.vdot(reference_shape, shape)
            aligned.append(shape*np.exp(-1j*np.angle(phase)) if abs(phase) > np.finfo(float).eps else shape)
        shape = np.mean(aligned, axis=0)
        shape = shape/max(np.linalg.norm(shape), np.finfo(float).eps)
        frequency, damping = float(np.median(frequencies)), float(np.median(dampings))
        magnitude = 2*np.pi*frequency/max(np.sqrt(max(1-damping**2, 1e-12)), 1e-12)
        retained.append(ConsensusDmdMode(
            mode=DmdMode(
                frequency_hz=frequency, damping_ratio=damping,
                decay_rate=float(-damping*magnitude), strength=float(np.median(strengths)),
                shape_real=tuple(float(value) for value in shape.real),
                shape_imag=tuple(float(value) for value in shape.imag),
            ),
            support_fraction=float(support), supporting_fits=len(cluster), total_fits=total_fits,
            frequency_mad_hz=float(np.median(np.abs(frequencies-frequency))),
            damping_mad=float(np.median(np.abs(dampings-damping))),
        ))
    return sorted(retained, key=lambda item: (-item.support_fraction, -item.mode.strength))


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
