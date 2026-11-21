"""Frozen reference-equilibrium spectra and measured val ringdown modes.

The two windows start 0.5 and 1.5 s after forecast origin and end at 8 s.
Use 20 delays spaced by 20 ms, a 10 ms Hankel shift and ranks 6/10/14.
Empirical poles in 0.35–3 Hz require two cycles, damping ratio in [0,.4]
and a modal speed contribution of at least 5% of the window signal norm.
Stabilization uses 8% frequency, .05 damping and .80 MAC in both windows.
Model assignment uses 25% frequency and .50 MAC. Strong correspondence
uses 10% frequency, .05 damping and .80 MAC. All denominators are retained.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment

from kron_reduce.evaluation.modal import DmdMode, modal_assurance_criterion, state_space_modes
from kron_reduce.physics.coordinates import coi_basis, coi_project

_RANKS = (6, 10, 14)
_WINDOWS = (.5, 1.5)
_DT = .01
_BAND = (.35, 3.)
_NAMES = ("hard_r", "m3")


def _shape(mode):
    return np.array(mode.shape_real)+1j*np.array(mode.shape_imag)


def _report_mode(mode):
    vector = _shape(mode)
    pivot = int(np.argmax(np.abs(vector)))
    aligned = vector*np.exp(-1j*np.angle(vector[pivot]))
    amplitude = np.abs(aligned)
    substantial = amplitude >= .2*amplitude.max()
    return {**mode.as_dict(), "shape_real": aligned.real.tolist(), "shape_imag": aligned.imag.tolist(),
            "largest_speed_components_zero_based": np.argsort(-amplitude, kind="stable")[:3].tolist(),
            "in_phase_with_pivot": np.flatnonzero(substantial & (aligned.real >= .5*amplitude)).tolist(),
            "opposite_phase_to_pivot": np.flatnonzero(substantial & (aligned.real <= -.5*amplitude)).tolist()}


def _spectrum(model, equilibrium, mass):
    """Linearize the frozen shifted-sine, reciprocal local-memory field at zero balanced input."""
    n = len(mass)
    if (n != 10 or model.n_nodes != n or model.memory_dimension != n or model.latent_dimension != n
            or model.mass.device.type != "cpu" or model.mass.dtype != torch.float64):
        raise ValueError("M4 expects CPU float64 ten-machine models with ten memories and fixed topology")
    if not np.allclose(model.mass.detach().numpy(), mass, rtol=0., atol=1e-12):
        raise ValueError("event mass and frozen model mass differ")

    with torch.no_grad():
        edge, k, damping, coupling, tau = (v.detach().numpy() for v in
            (model.incidence, model.effective_stiffness, model.effective_damping, model.coupling, model.tau))
        stiffness = (edge*(k*np.cos(equilibrium @ edge))) @ edge.T
        inverse_mass_projection = model.power_projection.detach().numpy()/mass[:, None]
        full = np.zeros((3*n, 3*n))
        full[:n, n:2*n] = np.eye(n)
        full[n:2*n, :n] = -inverse_mass_projection @ stiffness
        full[n:2*n, n:2*n] = -inverse_mass_projection @ damping
        full[n:2*n, 2*n:] = -inverse_mass_projection @ coupling
        full[2*n:, n:2*n] = coupling.T
        full[2*n:, 2*n:] = -np.diag(1./tau)
        basis = coi_basis(mass)
        lift = block_diag(basis, basis, np.eye(n))
        matrix = lift.T @ full @ lift
        tensor = lambda x: torch.tensor(x, dtype=torch.float64)
        zero = torch.zeros((1, n), dtype=torch.float64)
        origin = torch.zeros((1, 3*n), dtype=torch.float64)
        eq = tensor(equilibrium[None, :])
        field = model._rhs(zero, eq)
        residual = float(field(origin).abs().max())
        _, action = model._rhs(zero, eq, tangent=True)(origin, tensor(lift[None]))
        tangent_error = float(np.max(np.abs(matrix-lift.T @ action[0].numpy())))
    with torch.enable_grad():
        ad = torch.autograd.functional.jacobian(lambda x: field(x[None])[0], origin[0]).detach().numpy()
    ad_error = float(np.max(np.abs(matrix-lift.T @ ad @ lift)))
    if not (residual <= 1e-12 and ad_error <= 1e-9 and tangent_error <= 1e-9):
        raise ValueError(f"analytical linearization disagrees with the field: {residual=}, {ad_error=}, {tangent_error=}")
    observation = np.zeros((n, len(matrix)))
    observation[:, n-1:2*(n-1)] = np.sqrt(mass)[:, None]*basis
    modes = state_space_modes(matrix, observation, min_frequency_hz=_BAND[0], max_frequency_hz=_BAND[1])
    poles = np.linalg.eigvals(matrix)
    return modes, {"quotient_dimension": len(matrix), "mass": mass.tolist(), "equilibrium": equilibrium.tolist(),
                   "equilibrium_residual_inf": residual, "tangent_max_abs_error": tangent_error,
                   "autodiff_max_abs_error": ad_error, "matrix": matrix.tolist(),
                   "all_quotient_poles_per_s": [[float(p.real), float(p.imag)] for p in poles],
                   "max_pole_real_per_s": float(poles.real.max()), "modes": [_report_mode(m) for m in modes]}


def _hankel_fits(weighted_speed):
    """One block-Hankel factorization per window, with physical shapes from its first ten rows."""
    signal = weighted_speed-weighted_speed.mean(axis=0)
    length = len(signal)-38
    stack = np.concatenate([signal[lag:lag+length].T for lag in range(0, 40, 2)])
    left, right = stack[:, :-1], stack[:, 1:]
    u, singular, vt = np.linalg.svd(left, full_matrices=False)
    numerical_rank = int(np.count_nonzero(singular > 1e-9*singular[0]))
    length_s = (len(signal)-1)*_DT
    norm_signal = np.linalg.norm(left[:10])
    fits, records, cached = {}, {}, {}
    for rank in _RANKS:
        usable = min(rank, numerical_rank)
        if usable not in cached:
            modes = []
            if usable and norm_signal > 1e-12:
                mapped = (right @ vt[:usable].T)/singular[:usable]
                eigenvalues, vectors = np.linalg.eig(u[:, :usable].T @ mapped)
                hankel_shapes = mapped @ vectors
                # Modal coordinates at every shifted snapshot give a scale-invariant RMS contribution.
                coefficients = np.linalg.pinv(hankel_shapes, rcond=1e-9) @ left
                for i, value in enumerate(eigenvalues):
                    if value.imag <= 0. or abs(value) < np.finfo(float).eps:
                        continue
                    pole = np.log(value)/_DT
                    frequency = float(pole.imag/(2*np.pi))
                    damping = float(-pole.real/abs(pole))
                    if not (_BAND[0] <= frequency <= _BAND[1] and frequency*length_s >= 2.
                            and 0. <= damping <= .4):
                        continue
                    shape = hankel_shapes[:10, i]
                    norm = np.linalg.norm(shape)
                    if norm <= np.finfo(float).eps:
                        continue
                    contribution = float(np.linalg.norm(2.*(shape[:, None]*coefficients[i]).real)/norm_signal)
                    if contribution < .05:
                        continue
                    shape = shape/norm
                    modes.append(DmdMode(frequency, damping, float(pole.real), contribution,
                                         tuple(shape.real.tolist()), tuple(shape.imag.tolist())))
            cached[usable] = sorted(modes, key=lambda m: (-m.strength, m.frequency_hz))
        fits[rank] = cached[usable]
        records[rank] = {"requested_rank": rank, "effective_rank": usable,
                         "modes": [_report_mode(m) for m in fits[rank]]}
    return fits, {"numerical_rank": numerical_rank, "samples": len(signal), "rank_fits": records}


def _assign(reference, candidates, *, frequency_limit, mac_limit, damping_limit=None):
    """Maximize admissible one-to-one matches, then minimize relative frequency error + 1 − MAC."""
    if not reference or not candidates:
        return {}
    count = len(reference)
    cost = np.full((count, len(candidates)+count), float(2*(count+1)))
    cost[:, len(candidates):] = count+1.
    pairs = {}
    for i, observed in enumerate(reference):
        for j, candidate in enumerate(candidates):
            error = candidate.frequency_hz-observed.frequency_hz
            relative = abs(error)/observed.frequency_hz
            mac = float(np.clip(modal_assurance_criterion(_shape(observed), _shape(candidate)), 0., 1.))
            damping_error = candidate.damping_ratio-observed.damping_ratio
            if (relative <= frequency_limit and mac >= mac_limit
                    and (damping_limit is None or abs(damping_error) <= damping_limit)):
                cost[i, j] = relative+1.-mac
                pairs[i, j] = {"reference_index": i, "candidate_index": j,
                               "frequency_error_hz": error, "relative_frequency_error": relative,
                               "damping_ratio_error": damping_error, "mac": mac}
    rows, columns = linear_sum_assignment(cost)
    return {int(i): pairs[int(i), int(j)] for i, j in zip(rows, columns, strict=True) if (i, j) in pairs}


def _ringdown(speed, mass):
    weighted = np.sqrt(mass)*coi_project(speed, mass)
    fits, windows = {}, {}
    for start in _WINDOWS:
        fits[start], windows[start] = _hankel_fits(weighted[round(start/_DT):801])
    primary = fits[.5][10]
    matches = {(start, rank): _assign(primary, fits[start][rank], frequency_limit=.08,
                                      mac_limit=.8, damping_limit=.05)
               for start in _WINDOWS for rank in _RANKS}
    retained, records = [], []
    for i, mode in enumerate(primary):
        windows_support = {}
        for start in _WINDOWS:
            ranks = [rank for rank in _RANKS if i in matches[start, rank]]
            actual = {windows[start]["rank_fits"][rank]["effective_rank"] for rank in ranks}
            distinct = {windows[start]["rank_fits"][rank]["effective_rank"] for rank in _RANKS}
            windows_support[str(start)] = {"supporting_ranks": ranks, "distinct_supporting_ranks": len(actual),
                "distinct_available_ranks": len(distinct), "passed": len(ranks) >= 2 and len(actual) >= min(2, len(distinct))}
        accepted = all(item["passed"] for item in windows_support.values())
        records.append({"primary_rank10_index": i, "support_by_window_start_s": windows_support, "accepted": accepted})
        if accepted:
            retained.append(mode)
    return retained, {"windows": {str(k): v for k, v in windows.items()}, "stabilization": records,
                      "retained": [_report_mode(m) for m in retained]}


def _medians(matches):
    if not matches:
        return {"pairs": 0, "median_absolute_frequency_error_hz": None, "median_relative_frequency_error": None,
                "median_absolute_damping_ratio_error": None, "median_mac": None}
    return {"pairs": len(matches),
            "median_absolute_frequency_error_hz": float(np.median([abs(p["frequency_error_hz"]) for p in matches])),
            "median_relative_frequency_error": float(np.median([p["relative_frequency_error"] for p in matches])),
            "median_absolute_damping_ratio_error": float(np.median([abs(p["damping_ratio_error"]) for p in matches])),
            "median_mac": float(np.median([p["mac"] for p in matches]))}


def evaluate_modes(models, stable_events):
    """Compare two frozen models to independently identified rotor-speed modes on the same events."""
    if set(models) != set(_NAMES) or not len(stable_events):
        raise ValueError("expected the hard_r and m3 model pair and a nonempty stable val roster")
    if getattr(models["m3"], "support", None) != ("G2",):
        raise ValueError("the frozen M3 diagnostic requires its registered G2 support")
    if any(event.get("role") != "val" or event["label"] != "stable" for event in stable_events):
        raise ValueError("M4 ringdown observations must belong to the stable val roster")
    if len({event["id"] for event in stable_events}) != len(stable_events):
        raise ValueError("duplicated event in the stable val roster")

    references, by_static, spectra, events = {}, {}, {}, {}
    paired_dominant = {name: [] for name in _NAMES}
    for event in stable_events:
        mass, equilibrium = (np.asarray(event[key], dtype=np.float64) for key in ("mass", "equilibrium"))
        if (mass.shape != (10,) or equilibrium.shape != (10,) or not np.isfinite(equilibrium).all()
                or not np.isfinite(mass).all() or np.any(mass <= 0.)):
            raise ValueError("a reference equilibrium and ten finite positive masses are required")
        identity = (mass.tobytes(), equilibrium.tobytes())
        if identity not in by_static:
            index = str(len(by_static))
            by_static[identity] = index
            spectra[index], references[index] = {}, {}
            for name in _NAMES:
                spectra[index][name], references[index][name] = _spectrum(models[name], equilibrium, mass)
        index = by_static[identity]
        row = {"reference_index": index, "status": None, "retained_modes": 0, "comparisons": {}}
        events[event["id"]] = row
        if "target_omega" in event:
            speed = np.asarray(event["target_omega"], dtype=np.float64)
        else:
            raw = np.asarray(event["x"], dtype=np.float64)
            if raw.ndim != 2 or raw.shape[1] != 20:
                raise ValueError("M4 requires the ten speed columns of the TASK3 observation")
            speed = raw[10:, 10:]
        if speed.ndim != 2 or speed.shape[1] != 10:
            raise ValueError("target_omega must contain ten rotor speeds")
        if len(speed) < 801:
            row.update(status="incomplete_window", available_forecast_frames=len(speed))
            continue
        if not np.isfinite(speed[50:801]).all():
            row.update(status="nonfinite_window")
            continue
        if "forecast_time" in event:
            clock = np.asarray(event["forecast_time"])[:801]
            if clock.shape != (801,) or not np.allclose(clock, np.arange(801)*_DT, atol=1e-10, rtol=0.):
                raise ValueError("measured ringdown clock differs from the registered TASK3 clock")
        try:
            measured, identification = _ringdown(speed[:801], mass)
        except np.linalg.LinAlgError as failure:
            row.update(status="identification_failed", error=str(failure))
            continue
        row.update(status="identified" if measured else "no_stabilized_mode", retained_modes=len(measured),
                   identification=identification)
        for name in _NAMES:
            modes = spectra[index][name]
            matches = _assign(measured, modes, frequency_limit=.25, mac_limit=.5)
            for pair in matches.values():
                j = pair["candidate_index"]
                pair["strong"] = bool(pair["relative_frequency_error"] <= .10 and pair["mac"] >= .80
                                      and abs(pair["damping_ratio_error"]) <= .05 and modes[j].decay_rate <= 0.)
            row["comparisons"][name] = {"matches": list(matches.values()), "dominant": matches.get(0),
                "unmatched_empirical_indices": sorted(set(range(len(measured)))-matches.keys()),
                "unmatched_analytic_indices": sorted(set(range(len(modes)))-{p["candidate_index"] for p in matches.values()})}
        if all(row["comparisons"][name]["dominant"] is not None for name in _NAMES):
            for name in _NAMES:
                paired_dominant[name].append(row["comparisons"][name]["dominant"])

    n_events = len(events)
    n_modes = sum(row["retained_modes"] for row in events.values())
    summary = {"allocated_stable_val": n_events, "distinct_static_references": len(references),
               "events_with_stabilized_mode": sum(row["status"] == "identified" for row in events.values()),
               "retained_event_modes": n_modes,
               "status_counts": {status: sum(row["status"] == status for row in events.values())
                                 for status in ("identified", "no_stabilized_mode", "incomplete_window",
                                                "nonfinite_window", "identification_failed")},
               "models": {}}
    for name in _NAMES:
        compared = [row["comparisons"][name] for row in events.values() if name in row["comparisons"]]
        matched = [pair for item in compared for pair in item["matches"]]
        dominant = [item["dominant"] for item in compared if item["dominant"] is not None]
        strong = sum(pair["strong"] for pair in dominant)
        summary["models"][name] = {
            "all_matched_event_modes": _medians(matched), "unmatched_empirical_modes": n_modes-len(matched),
            "dominant_matched_events": _medians(dominant),
            "dominant_match_fraction_all_events": len(dominant)/n_events, "dominant_strong_events": strong,
            "dominant_strong_fraction_all_events": strong/n_events,
            "same_empirical_dominant_matched_by_both": _medians(paired_dominant[name])}
    return {"summary": summary, "references": references, "events": events,
            "contract": {"reference": "shifted q=v=z=0, projected power p=0, frozen static delta_star",
                "observation": "sqrt(mass) times COI rotor speed; machine indices start at zero",
                "analytic_strength": "observed speed norm of a unit-norm quotient eigenvector",
                "empirical_strength": "norm of reconstructed conjugate pair divided by observed speed norm",
                "dominant": "largest primary-window strength among stabilized empirical modes",
                "windows_after_forecast_origin_s": [[start, 8.] for start in _WINDOWS],
                "frequency_band_hz": list(_BAND), "hankel_delay_rows": 20, "row_spacing_s": .02,
                "hankel_shift_s": _DT, "requested_ranks": list(_RANKS), "primary_rank": 10,
                "numerical_singular_value_fraction": 1e-9, "minimum_cycles": 2.,
                "empirical_damping_range": [0., .4], "minimum_rms_contribution": .05,
                "stabilize": {"relative_frequency": .08, "absolute_damping": .05, "mac": .80,
                              "min_requested_ranks_per_window": 2, "min_distinct_effective_ranks_if_available": 2},
                "assign": {"relative_frequency": .25, "mac": .50},
                "strong": {"relative_frequency": .10, "absolute_damping": .05, "mac": .80,
                           "nonpositive_analytic_decay_rate_required": True}}}
