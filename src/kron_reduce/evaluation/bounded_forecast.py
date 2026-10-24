"""Budgeted, ragged causal-history rollout without padding forecast targets."""
from __future__ import annotations

import math

import numpy as np

from kron_reduce.models.nonlinear_latent import _features, complete_incidence


def _clock(sample, horizon_s):
    if horizon_s is None:
        clock = np.asarray(sample.forecast_time, dtype=np.float64)
        if (clock.ndim != 1 or len(clock) < 2 or len(clock) > 301
                or not np.isfinite(clock).all() or clock[0] != 0.0
                or np.any(np.diff(clock) <= 0)):
            raise ValueError("sample forecast clock must contain its real 1..300 future points")
        return clock.copy()
    if horizon_s not in (3.0, 8.0):
        raise ValueError("fixed forecast horizon must be 3 or 8 seconds")
    count = round(horizon_s / 0.01)
    if abs(count * 0.01 - horizon_s) > 1e-12:
        raise ValueError("fixed horizon must lie on the 10 ms output clock")
    return np.arange(count + 1, dtype=np.float64) * 0.01


def _validate_history(history, n):
    if (history.delta.shape != (11, n) or history.omega.shape != (11, n)
            or history.time.shape != (11,) or history.time[0] != -0.1 or history.time[-1] != 0.0
            or np.any(np.diff(history.time) <= 0)
            or not np.isfinite(history.delta).all() or not np.isfinite(history.omega).all()
            or history.constant_power.shape != (n,) or not np.isfinite(history.constant_power).all()):
        raise ValueError("history must be finite, real 11-point/10 ms data")


def forecast_bounded(model, samples, *, max_step, horizon_s=None, budget, identity):
    """Return augmented history and ragged augmented futures under an RHS budget.

    Each future starts with the exact measured endpoint. With no fixed horizon,
    only the sample's available forecast clock is used, capped at 300 points.
    """
    samples = tuple(samples)
    if not samples or len({s.event_id for s in samples}) != len(samples):
        raise ValueError("forecast cohort must be nonempty and have unique event IDs")
    if max_step not in (0.005, 0.0025):
        raise ValueError("bounded refinement supports only 5 ms or 2.5 ms")
    if not isinstance(identity, dict):
        raise TypeError("budget identity must be a mapping")
    closure, n = model.closure, model.closure.n_nodes
    for sample in samples:
        _validate_history(sample.history, n)
        if np.asarray(sample.equilibrium_delta).shape != (n,) or not np.isfinite(sample.equilibrium_delta).all():
            raise ValueError("sample equilibrium must be finite and node aligned")
    clocks = [_clock(sample, horizon_s) for sample in samples]
    if any(not np.array_equal(clock, clocks[0]) for clock in clocks[1:]) and horizon_s is not None:
        raise AssertionError("fixed-horizon clocks should be identical")
    count_events = len(samples)
    history_substeps = [
        [math.ceil(float(dt) / max_step - 1e-12) for dt in np.diff(sample.history.time)]
        for sample in samples
    ]
    if any(count not in (2, 4) for counts in history_substeps for count in counts):
        raise ValueError("history clock intervals must require exactly two or four registered substeps")
    history_limit = sum(sum(counts) for counts in history_substeps) * 4
    future_limit = sum(
        sum(math.ceil(float(dt) / max_step - 1e-12) for dt in np.diff(clock)) * 4
        for clock in clocks
    )
    mass = np.asarray(closure.mass)
    incidence = complete_incidence(n)
    eq = np.stack([s.equilibrium_delta for s in samples])
    power = np.stack([s.history.constant_power for s in samples])

    def rhs(value, row_indices):
        q, v, z = value[:, :n], value[:, n:2*n], value[:, 2*n:]
        bq, beq = q @ incidence, eq[row_indices] @ incidence
        force = bq if closure.backbone == "linear" else np.sin(bq + beq) - np.sin(beq)
        force_vector = (force * closure.edge_stiffness) @ incidence.T
        f = power[row_indices] - force_vector - v @ closure.damping.T - z @ closure.coupling.T
        f -= f.sum(axis=1, keepdims=True) * mass[None, :] / float(np.sum(mass))
        return np.concatenate((v, f / mass[None, :],
                               v @ closure.coupling
                               + z @ (closure.latent_skew - closure.latent_dissipation).T), axis=1)

    def rk4(value, h, row_indices, meter):
        k1 = meter.invoke(rhs, value, row_indices)
        k2 = meter.invoke(rhs, value + h * k1 / 2, row_indices)
        k3 = meter.invoke(rhs, value + h * k2 / 2, row_indices)
        k4 = meter.invoke(rhs, value + h * k3, row_indices)
        output = value + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        output[:, :n] -= (output[:, :n] @ mass)[:, None] / float(np.sum(mass))
        output[:, n:2*n] -= (output[:, n:2*n] @ mass)[:, None] / float(np.sum(mass))
        if not np.isfinite(output).all() or np.any(np.max(np.abs(output), axis=1) > 1e12):
            raise FloatingPointError("nonfinite or divergent bounded forecast")
        return output

    history_paths = np.empty((count_events, 11, 2*n + closure.latent_dimension), dtype=np.float64)
    features = np.stack([_features(s.history, s.equilibrium_delta,
                                   model.settings.frames, model.settings.encoder_mode) for s in samples])
    if model.feature_scale.shape != (features.shape[1],) or np.any(model.feature_scale <= 0):
        raise ValueError("model feature scales are invalid")
    z0 = (features / model.feature_scale[None, :]) @ model.encoder.T
    initial = np.stack([np.r_[s.history.delta[0], s.history.omega[0]] for s in samples])
    state = np.concatenate((initial, z0), axis=1)
    with budget.segment({**identity, "population": "history", "event_ids": [s.event_id for s in samples]},
                        "history_integration", history_limit) as meter:
        meter.note_prediction_paths(count_events)
        meter.note_rollout_start(count_events)
        history_paths[:, 0] = state
        for interval in range(10):
            intervals = np.asarray([
                float(np.diff(sample.history.time)[interval]) for sample in samples
            ])
            counts = [history_substeps[i][interval] for i in range(count_events)]
            for substep in range(max(counts)):
                rows = [i for i, count in enumerate(counts) if substep < count]
                for h in np.unique(intervals[rows] / np.asarray(counts)[rows]):
                    indices = np.asarray([i for i in rows
                                          if intervals[i] / counts[i] == h], dtype=np.int64)
                    state[indices] = rk4(state[indices], float(h), indices, meter)
            history_paths[:, interval + 1] = state

    # Match the established history-to-future splice: retain the estimated
    # latent endpoint, then reset visible q/v to the measured final state.
    visible_endpoint = np.stack([np.r_[s.history.delta[-1], s.history.omega[-1]] for s in samples])
    state = np.concatenate((visible_endpoint, history_paths[:, -1, 2*n:]), axis=1)
    futures: list[list[np.ndarray]] = [[state[i].copy()] for i in range(count_events)]
    maximum_intervals = max(len(clock) - 1 for clock in clocks)
    with budget.segment({**identity, "population": "future", "event_ids": [s.event_id for s in samples]},
                        "future_integration", future_limit) as meter:
        meter.note_rollout_start(count_events)
        for interval in range(maximum_intervals):
            active = [i for i, clock in enumerate(clocks) if interval + 1 < len(clock)]
            if not active:
                break
            delta_t = [float(clocks[i][interval + 1] - clocks[i][interval]) for i in active]
            substeps = [math.ceil(dt / max_step - 1e-12) for dt in delta_t]
            for substep in range(max(substeps)):
                rows = [j for j, total in enumerate(substeps) if substep < total]
                indices = np.asarray([active[j] for j in rows], dtype=np.int64)
                hs = np.asarray([delta_t[j] / substeps[j] for j in rows], dtype=np.float64)
                # Registered sample clocks are usually uniform. For arbitrary
                # ragged intervals, integrate each distinct h cohort separately.
                for h in np.unique(hs):
                    local = np.flatnonzero(hs == h)
                    idx = indices[local]
                    state[idx] = rk4(state[idx], float(h), idx, meter)
            for i in active:
                futures[i].append(state[i].copy())

    history_result = np.array(history_paths, copy=True)
    future_result = [np.stack(path) for path in futures]
    for i, path in enumerate(future_result):
        expected_origin = np.r_[samples[i].history.delta[-1], samples[i].history.omega[-1]]
        if (path.shape != (len(clocks[i]), 2*n + closure.latent_dimension)
                or not np.array_equal(path[0, :2*n], expected_origin)
                or not np.isfinite(path).all()):
            raise FloatingPointError("bounded forecast has invalid origin, shape, or nonfinite values")
    return history_result, future_result
