"""Leakage-locked local observability audit for frozen shared equation poles."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from kron_reduce.models.prunable_energy_model import PrunableEnergyModel

from .modal import DmdMode, exact_dmd_modes, modal_assurance_criterion
from .modal_observability import ModalSupportStatus

STATIC_SPECTRUM_FIELDS = (
    "mass",
    "incidence",
    "equilibrium_delta",
    "Pm_equilibrium",
)
DMD_RANKS = (4, 6, 8)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 324
MINIMUM_INDEPENDENT_LINES = 3
MINIMUM_LINE_EVENT_SUPPORT = 0.50
MINIMUM_LINE_SUPPORT_FRACTION = 0.80
MINIMUM_BOOTSTRAP_SUPPORT_PROBABILITY = 0.80
MINIMUM_FINITE_WINDOW_ENERGY = 0.01
MAXIMUM_FREQUENCY_RELATIVE_ERROR = 0.05
MAXIMUM_DAMPING_ABSOLUTE_ERROR = 0.05
MINIMUM_MAC = 0.80
MINIMUM_RANK_MATCHES = 2
MINIMUM_FREQUENCY_HZ = 0.05
MAXIMUM_FREQUENCY_HZ = 5.0


class StaticFamilyError(ValueError):
    """The registered events do not share one equation linearization point."""


@dataclass(frozen=True)
class SharedEquationPole:
    pole_id: str
    checkpoint_sha256: str
    static_signature_sha256: str
    eigenvalue_real: float
    eigenvalue_imag: float
    frequency_hz: float
    damping_ratio: float
    speed_visibility: float
    input_output_residue_rank: int
    equilibrium_residual_inf_norm: float
    shape_real: tuple[float, ...]
    shape_imag: tuple[float, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.pole_id):
            raise ValueError("pole_id must be a SHA-256 value")
        for name in ("checkpoint_sha256", "static_signature_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 value")
        scalars = (
            self.eigenvalue_real,
            self.eigenvalue_imag,
            self.frequency_hz,
            self.damping_ratio,
            self.speed_visibility,
            self.equilibrium_residual_inf_norm,
        )
        if not np.isfinite(scalars).all() or self.frequency_hz <= 0.0:
            raise ValueError("shared equation pole scalars must be finite")
        if (
            self.eigenvalue_imag <= 0.0
            or not MINIMUM_FREQUENCY_HZ <= self.frequency_hz <= MAXIMUM_FREQUENCY_HZ
            or self.speed_visibility < 1e-3
            or self.equilibrium_residual_inf_norm > 1e-7
        ):
            raise ValueError("shared equation pole does not satisfy the frozen equation gates")
        if type(self.input_output_residue_rank) is not int or self.input_output_residue_rank < 1:
            raise ValueError("a shared equation pole needs positive input-output residue rank")
        real = np.asarray(self.shape_real, dtype=float)
        imag = np.asarray(self.shape_imag, dtype=float)
        if real.ndim != 1 or real.shape != imag.shape or not real.size:
            raise ValueError("pole shapes must be nonempty aligned vectors")
        if not np.isfinite(real).all() or not np.isfinite(imag).all():
            raise ValueError("pole shapes must be finite")

    @property
    def shape(self) -> np.ndarray:
        return np.asarray(self.shape_real) + 1j * np.asarray(self.shape_imag)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_signature(array: Any) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    if not value.size or not np.isfinite(value).all():
        raise StaticFamilyError("shared-spectrum static arrays must be finite and nonempty")
    return {
        "dtype": "float64-le",
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def shared_static_signature(samples: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Require one exact equilibrium/static family before any target is inspected."""

    if not samples:
        raise StaticFamilyError("at least one registered sample is required")
    by_event: dict[str, dict[str, Any]] = {}
    for sample in samples:
        event_id = str(sample.get("event_id", ""))
        if not event_id or event_id in by_event:
            raise StaticFamilyError("event IDs must be nonempty and unique")
        try:
            static = sample["model_inputs"]["static"]
            mass = np.asarray(static["mass"], dtype=float)
            incidence = np.asarray(static["incidence"], dtype=float)
            equilibrium = np.asarray(static["equilibrium_delta"], dtype=float)
            equilibrium_pm = np.asarray(static["Pm_equilibrium"], dtype=float)
            if (
                mass.ndim != 1
                or mass.size < 2
                or np.any(mass <= 0.0)
                or incidence.ndim != 2
                or incidence.shape[0] != mass.size
                or equilibrium.shape != mass.shape
                or equilibrium_pm.shape != mass.shape
            ):
                raise StaticFamilyError("shared-spectrum static arrays are not shape aligned")
            signatures = {name: _array_signature(static[name]) for name in STATIC_SPECTRUM_FIELDS}
        except (KeyError, TypeError) as error:
            raise StaticFamilyError("missing shared-spectrum static field") from error
        by_event[event_id] = signatures
    ordered = sorted(by_event)
    reference = by_event[ordered[0]]
    changed = [event_id for event_id in ordered[1:] if by_event[event_id] != reference]
    if changed:
        raise StaticFamilyError(
            "INVALID_STATIC_FAMILY: registered events have different equation statics: "
            + ",".join(changed)
        )
    document = {
        "schema": "kron_reduce.p1_shared_static_signature.v1",
        "fields": list(STATIC_SPECTRUM_FIELDS),
        "reference_event_id": ordered[0],
        "event_count": len(ordered),
        "event_ids_sha256": _sha256_payload(ordered),
        "arrays": reference,
    }
    return _sha256_payload(document), document


def extract_shared_equation_poles(
    model: PrunableEnergyModel,
    samples: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
) -> tuple[tuple[SharedEquationPole, ...], dict[str, Any]]:
    """Extract one equation spectrum without reading trajectory targets or view labels."""

    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a SHA-256 value")
    static_signature, static_document = shared_static_signature(samples)
    reference_id = str(static_document["reference_event_id"])
    reference = next(sample for sample in samples if str(sample["event_id"]) == reference_id)
    spectrum = model.modal_spectrum(
        reference["model_inputs"]["static"],
        min_frequency_hz=MINIMUM_FREQUENCY_HZ,
        max_frequency_hz=MAXIMUM_FREQUENCY_HZ,
        minimum_speed_visibility=1e-3,
    )
    diagnostics = spectrum.get("diagnostics", {})
    if diagnostics.get("status") == "refused" or not diagnostics.get(
        "modal_diagnostics_complete", False
    ):
        raise ValueError(
            "shared equation spectrum was refused: "
            + str(diagnostics.get("refusal_reason", "incomplete_modal_diagnostics"))
        )
    if not diagnostics.get("energy_hessian_positive_definite", False):
        raise ValueError("shared equation spectrum has no positive-definite storage Hessian")
    ordered = sorted(
        spectrum.get("equation_reportable_modes", ()),
        key=lambda row: (
            float(row["frequency_hz"]),
            float(row["damping_ratio"]),
            float(row["eigenvalue_real"]),
            float(row["eigenvalue_imag"]),
        ),
    )
    poles = []
    for index, row in enumerate(ordered):
        identity = {
            "checkpoint_sha256": checkpoint_sha256,
            "static_signature_sha256": static_signature,
            "index": index,
            "eigenvalue_real": float(row["eigenvalue_real"]),
            "eigenvalue_imag": float(row["eigenvalue_imag"]),
        }
        linearization = row.get("linearization", diagnostics)
        poles.append(
            SharedEquationPole(
                pole_id=_sha256_payload(identity),
                checkpoint_sha256=checkpoint_sha256,
                static_signature_sha256=static_signature,
                eigenvalue_real=float(row["eigenvalue_real"]),
                eigenvalue_imag=float(row["eigenvalue_imag"]),
                frequency_hz=float(row["frequency_hz"]),
                damping_ratio=float(row["damping_ratio"]),
                speed_visibility=float(row["speed_visibility"]),
                input_output_residue_rank=int(row["input_output_residue_rank"]),
                equilibrium_residual_inf_norm=float(
                    linearization["equilibrium_residual_inf_norm"]
                ),
                shape_real=tuple(float(value) for value in row["shape_real"]),
                shape_imag=tuple(float(value) for value in row["shape_imag"]),
            )
        )
    return tuple(poles), {
        "schema": "kron_reduce.p1_shared_equation_poles.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "static_signature_sha256": static_signature,
        "static_signature": static_document,
        "frequency_source": "frozen_prunable_energy_model_equation_linearization",
        "trajectory_modes_used_to_define_poles": False,
        "internal_poles": spectrum.get("internal_poles", []),
        "equation_reportable_modes": [pole.as_dict() for pole in poles],
        "diagnostics": diagnostics,
    }


def mass_weighted_pole_shape(pole: SharedEquationPole, mass: np.ndarray) -> np.ndarray:
    mass = np.asarray(mass, dtype=float).reshape(-1)
    shape = pole.shape
    if shape.shape != mass.shape or np.any(mass <= 0.0) or not np.isfinite(mass).all():
        raise ValueError("mass and pole shape must be finite, positive and aligned")
    weighted = np.sqrt(mass) * shape
    norm = float(np.linalg.norm(weighted))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("mass-weighted pole shape has zero norm")
    return weighted / norm


def finite_window_energy_share(values: np.ndarray, complex_shape: np.ndarray) -> float:
    """Energy in the real subspace spanned by a complex speed mode."""

    values = np.asarray(values, dtype=float)
    shape = np.asarray(complex_shape, dtype=complex).reshape(-1)
    if values.ndim != 2 or values.shape[1] != shape.size or values.shape[0] < 2:
        raise ValueError("finite-window path and complex mode shape are not aligned")
    if not np.isfinite(values).all() or not np.isfinite(shape).all():
        raise ValueError("finite-window inputs must be finite")
    centered = values - np.mean(values, axis=0, keepdims=True)
    total = float(np.sum(centered**2))
    if total <= np.finfo(float).eps:
        return 0.0
    span = np.column_stack((shape.real, shape.imag))
    left, singular, _ = np.linalg.svd(span, full_matrices=False)
    tolerance = max(span.shape) * np.finfo(float).eps * float(singular[0])
    basis = left[:, singular > tolerance]
    if basis.shape[1] == 0:
        return 0.0
    projection = centered @ basis @ basis.T
    share = float(np.sum(projection**2) / total)
    return float(np.clip(share, 0.0, 1.0))


def _candidate_shape(mode: DmdMode) -> np.ndarray:
    return np.asarray(mode.shape_real) + 1j * np.asarray(mode.shape_imag)


def match_poles_one_to_one(
    poles: Sequence[SharedEquationPole],
    candidates: Sequence[DmdMode],
    mass: np.ndarray,
) -> list[dict[str, Any]]:
    """Globally assign empirical modes, then apply the frozen hard gates."""

    poles = tuple(poles)
    candidates = tuple(candidates)
    weighted_shapes = [mass_weighted_pole_shape(pole, mass) for pole in poles]
    for mode in candidates:
        shape = _candidate_shape(mode)
        if weighted_shapes and shape.shape != weighted_shapes[0].shape:
            raise ValueError("candidate and shared-pole shapes are not aligned")
        if not np.isfinite(
            [mode.frequency_hz, mode.damping_ratio, mode.decay_rate, mode.strength]
        ).all() or not np.isfinite(shape).all():
            raise ValueError("candidate modes must be finite")
    assignments: dict[int, int] = {}
    metrics: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    if poles and candidates:
        cost = np.empty((len(poles), len(candidates)))
        for row, pole in enumerate(poles):
            for column, candidate in enumerate(candidates):
                frequency = abs(candidate.frequency_hz - pole.frequency_hz) / max(
                    pole.frequency_hz, np.finfo(float).eps
                )
                damping = abs(candidate.damping_ratio - pole.damping_ratio)
                mac = float(
                    np.clip(
                        modal_assurance_criterion(
                            weighted_shapes[row], _candidate_shape(candidate)
                        ),
                        0.0,
                        1.0,
                    )
                )
                value = (
                    frequency / MAXIMUM_FREQUENCY_RELATIVE_ERROR
                    + damping / MAXIMUM_DAMPING_ABSOLUTE_ERROR
                    + (1.0 - mac) / (1.0 - MINIMUM_MAC)
                )
                metrics[(row, column)] = (frequency, damping, mac, value)
                cost[row, column] = value
        rows, columns = linear_sum_assignment(cost)
        assignments = dict(zip((int(row) for row in rows), (int(col) for col in columns)))
    output = []
    for row, pole in enumerate(poles):
        column = assignments.get(row)
        if column is None:
            output.append(
                {
                    "pole_id": pole.pole_id,
                    "candidate_index": None,
                    "candidate_frequency_hz": None,
                    "candidate_damping_ratio": None,
                    "frequency_relative_error": None,
                    "damping_absolute_error": None,
                    "mac": None,
                    "assignment_cost": None,
                    "match_passed": False,
                    "status": "UNMATCHED",
                }
            )
            continue
        candidate = candidates[column]
        frequency, damping, mac, cost_value = metrics[(row, column)]
        passes = bool(
            frequency <= MAXIMUM_FREQUENCY_RELATIVE_ERROR
            and damping <= MAXIMUM_DAMPING_ABSOLUTE_ERROR
            and mac >= MINIMUM_MAC
        )
        output.append(
            {
                "pole_id": pole.pole_id,
                "candidate_index": column,
                "candidate_frequency_hz": float(candidate.frequency_hz),
                "candidate_damping_ratio": float(candidate.damping_ratio),
                "frequency_relative_error": float(frequency),
                "damping_absolute_error": float(damping),
                "mac": float(mac),
                "assignment_cost": float(cost_value),
                "match_passed": passes,
                "status": "MATCHED_PASS" if passes else "MATCHED_FAIL",
            }
        )
    return output


def _sample_metadata(sample: Mapping[str, Any]) -> tuple[str, str, str]:
    event_id = str(sample.get("event_id", ""))
    try:
        context = sample["training_context"]
        metadata = context["metadata"]
        view = str(metadata["source_dataset"])
        line = str(metadata["fault_line"])
    except (KeyError, TypeError) as error:
        raise ValueError("registered event metadata are required") from error
    if not event_id or not view or not line:
        raise ValueError("event, view and fault line must be nonempty")
    return event_id, view, line


def audit_local_observability_events(
    samples: Sequence[Mapping[str, Any]],
    poles: Sequence[SharedEquationPole],
    *,
    expected_static_signature_sha256: str,
    dmd_ranks: tuple[int, ...] = DMD_RANKS,
) -> dict[str, Any]:
    """Audit all event/pole cells without dropping DMD failures or weak events."""

    if tuple(dmd_ranks) != DMD_RANKS:
        raise ValueError(f"DMD ranks are frozen to {DMD_RANKS}")
    actual_static_signature, _ = shared_static_signature(samples)
    if actual_static_signature != expected_static_signature_sha256:
        raise StaticFamilyError("event statics do not match the frozen shared spectrum")
    poles = tuple(poles)
    if any(pole.static_signature_sha256 != actual_static_signature for pole in poles):
        raise StaticFamilyError("a pole belongs to another static family")
    event_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda row: str(row["event_id"])):
        event_id, view, line = _sample_metadata(sample)
        try:
            targets = sample["training_context"]["targets"]
            time = np.asarray(targets["t"], dtype=float)
            omega = np.asarray(targets["omega"], dtype=float)
            mass = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=float).reshape(-1)
        except (KeyError, TypeError) as error:
            raise ValueError("registered modal target arrays are required") from error
        if (
            time.shape != (1001,)
            or omega.shape != (1001, len(mass))
            or not np.isfinite(time).all()
            or not np.isfinite(omega).all()
            or not np.isclose(time[0], 0.0, rtol=0.0, atol=1e-12)
            or not np.isclose(time[-1], 10.0, rtol=0.0, atol=1e-12)
            or not np.allclose(np.diff(time), 0.01, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("modal audit requires the complete finite 0--10 s target window")
        if np.max(np.abs(omega @ mass)) > 1e-10:
            raise ValueError("modal audit requires COI-projected target speed")
        weighted = omega * np.sqrt(mass)[None, :]
        energy = {
            pole.pole_id: finite_window_energy_share(
                weighted, mass_weighted_pole_shape(pole, mass)
            )
            for pole in poles
        }
        rank_passes: dict[str, int] = {pole.pole_id: 0 for pole in poles}
        for rank in DMD_RANKS:
            dmd_error = None
            try:
                candidates = exact_dmd_modes(
                    weighted,
                    0.01,
                    rank=rank,
                    min_frequency_hz=MINIMUM_FREQUENCY_HZ,
                    max_frequency_hz=MAXIMUM_FREQUENCY_HZ,
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                candidates = []
                dmd_error = f"{type(error).__name__}: {error}"
            for index, candidate in enumerate(candidates):
                candidate_rows.append(
                    {
                        "event_id": event_id,
                        "view_name": view,
                        "fault_line": line,
                        "dmd_rank": rank,
                        "candidate_index": index,
                        **candidate.as_dict(),
                    }
                )
            matches = match_poles_one_to_one(poles, candidates, mass)
            for match in matches:
                rank_passes[match["pole_id"]] += int(match["match_passed"])
                rank_rows.append(
                    {
                        "event_id": event_id,
                        "view_name": view,
                        "fault_line": line,
                        "dmd_rank": rank,
                        "candidate_count": len(candidates),
                        "dmd_error": dmd_error,
                        **match,
                    }
                )
        for pole in poles:
            event_supported = bool(
                energy[pole.pole_id] >= MINIMUM_FINITE_WINDOW_ENERGY
                and rank_passes[pole.pole_id] >= MINIMUM_RANK_MATCHES
            )
            event_rows.append(
                {
                    "event_id": event_id,
                    "view_name": view,
                    "fault_line": line,
                    "pole_id": pole.pole_id,
                    "finite_window_energy_share": energy[pole.pole_id],
                    "dmd_rank_passes": rank_passes[pole.pole_id],
                    "registered_dmd_ranks": len(DMD_RANKS),
                    "event_supported": event_supported,
                }
            )
    return {
        "schema": "kron_reduce.p1_modal_event_observability.v1",
        "dmd_ranks": list(DMD_RANKS),
        "trajectory_dmd_interpretation": "finite_window_empirical_descriptor_only",
        "event_rows": event_rows,
        "rank_rows": rank_rows,
        "candidate_rows": candidate_rows,
    }


def summarize_local_observability(
    event_rows: Sequence[Mapping[str, Any]],
    expected_events: Sequence[Mapping[str, str]],
    pole_ids: Sequence[str],
) -> dict[str, Any]:
    """Aggregate complete event cells into line bootstrap and composite pole states."""

    pole_ids = tuple(sorted(str(value) for value in pole_ids))
    if len(pole_ids) != len(set(pole_ids)):
        raise ValueError("pole IDs must be unique")
    expected = {
        (str(row["event_id"]), str(row["view_name"]), str(row["fault_line"]))
        for row in expected_events
    }
    expected_event_ids = [str(row["event_id"]) for row in expected_events]
    if (
        len(expected) != len(expected_events)
        or len(expected_event_ids) != len(set(expected_event_ids))
        or not expected
    ):
        raise ValueError("expected event metadata must be complete and unique")
    cells: dict[tuple[str, str], dict[tuple[str, str, str], bool]] = defaultdict(dict)
    for row in event_rows:
        key = (str(row["event_id"]), str(row["view_name"]), str(row["fault_line"]))
        pole_id = str(row["pole_id"])
        if key not in expected or pole_id not in pole_ids:
            raise ValueError("event observability row is outside the registered cells")
        cell = (pole_id, key[0])
        if key in cells[cell]:
            raise ValueError("duplicate event/pole observability row")
        if type(row["event_supported"]) is not bool:
            raise ValueError("event_supported must be bool")
        cells[cell][key] = bool(row["event_supported"])
    required = {(pole_id, event_id) for pole_id in pole_ids for event_id, _, _ in expected}
    if set(cells) != required or any(len(values) != 1 for values in cells.values()):
        raise ValueError("event observability rows do not cover the complete registered role")

    event_lookup = {event_id: (view, line) for event_id, view, line in expected}
    grouped: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for pole_id, event_id in required:
        view, line = event_lookup[event_id]
        key = (event_id, view, line)
        grouped[(pole_id, view, line)].append(cells[(pole_id, event_id)][key])
    line_rows = []
    for (pole_id, view, line), flags in sorted(grouped.items()):
        fraction = float(np.mean(flags))
        line_rows.append(
            {
                "pole_id": pole_id,
                "view_name": view,
                "fault_line": line,
                "event_count": len(flags),
                "supporting_event_count": int(np.sum(flags)),
                "event_support_fraction": fraction,
                "line_supported": bool(fraction >= MINIMUM_LINE_EVENT_SUPPORT),
            }
        )

    by_view: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in line_rows:
        by_view[(row["pole_id"], row["view_name"])].append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    view_rows = []
    bootstrap_cells = []
    for pole_id, view in sorted(by_view):
        rows = by_view[(pole_id, view)]
        flags = np.asarray([row["line_supported"] for row in rows], dtype=float)
        draws = rng.choice(flags, size=(BOOTSTRAP_DRAWS, len(flags)), replace=True).mean(axis=1)
        point = float(np.mean(flags))
        lower, upper = (float(value) for value in np.quantile(draws, (0.025, 0.975)))
        probability = float(np.mean(draws >= MINIMUM_LINE_SUPPORT_FRACTION))
        if len(flags) < MINIMUM_INDEPENDENT_LINES:
            status = ModalSupportStatus.UNDETERMINED
        elif probability >= MINIMUM_BOOTSTRAP_SUPPORT_PROBABILITY:
            status = ModalSupportStatus.SUPPORTED
        else:
            status = ModalSupportStatus.UNSUPPORTED
        view_rows.append(
            {
                "pole_id": pole_id,
                "view_name": view,
                "independent_lines": len(flags),
                "supporting_lines": int(np.sum(flags)),
                "line_support_fraction": point,
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
                "bootstrap_support_probability": probability,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "cluster_unit": "source_dataset_x_fault_line",
                "status": status.value,
            }
        )
        bootstrap_cells.append(
            {
                "pole_id": pole_id,
                "view_name": view,
                "draw_support_fractions": draws.tolist(),
            }
        )

    pole_statuses = {}
    for pole_id in pole_ids:
        statuses = [row["status"] for row in view_rows if row["pole_id"] == pole_id]
        if ModalSupportStatus.SUPPORTED.value in statuses:
            pole_statuses[pole_id] = "LOCALLY_OBSERVED"
        elif ModalSupportStatus.UNSUPPORTED.value in statuses:
            pole_statuses[pole_id] = "NOT_OBSERVED"
        else:
            pole_statuses[pole_id] = "UNDETERMINED"
    return {
        "schema": "kron_reduce.p1_shared_pole_local_observability.v1",
        "line_rows": line_rows,
        "view_rows": view_rows,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "source_dataset_x_fault_line",
            "cells": bootstrap_cells,
        },
        "pole_statuses": pole_statuses,
    }
