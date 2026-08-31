"""Causal quotient-port mechanical-power ablation for a frozen energy core.

This restricted first-order family is not a stability certificate or a claim
that every mechanical-power closure has been searched.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data.p1_causal import P1RoleManifest, validate_p1_samples
from kron_reduce.data.p1_pm_targets import PmObservationBatch, PmReadPurpose
from kron_reduce.models.prunable_energy_model import PrunableEnergyModel
from kron_reduce.physics import coi_project

CLOSURE_DECAY_CANDIDATES = (0.2, 1.0, 5.0)


def mass_quotient_basis(mass: np.ndarray) -> np.ndarray:
    """Deterministic Householder complement in the fixed generator ordering."""
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or len(mass) < 2 or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("finite positive mass is required")
    direction = np.sqrt(mass) / np.linalg.norm(np.sqrt(mass))
    reflector = -direction
    reflector[0] += 1
    denominator = float(reflector @ reflector)
    if denominator <= np.finfo(float).eps:
        raise ValueError("mass quotient is numerically ill-conditioned")
    orthogonal = np.eye(len(mass)) - 2 * np.outer(reflector, reflector) / denominator
    return orthogonal[:, 1:] / np.sqrt(mass)[:, None]


class PassivePmClosure:
    def __init__(self, mass: np.ndarray, kappa: np.ndarray, decay: float, *, basis: np.ndarray | None = None):
        self.mass = np.asarray(mass, dtype=float).copy()
        default_basis = mass_quotient_basis(self.mass)
        self.basis = default_basis if basis is None else np.asarray(basis, dtype=float).copy()
        self.kappa = np.asarray(kappa, dtype=float).copy()
        self.decay = float(decay)
        n = len(self.mass)
        if self.basis.shape != (n, n - 1) or not np.isfinite(self.basis).all():
            raise ValueError("invalid quotient basis")
        if not np.allclose(self.basis.T @ (self.mass[:, None] * self.basis), np.eye(n - 1), rtol=0, atol=1e-10) or not np.allclose(self.basis.T @ self.mass, 0, rtol=0, atol=1e-10):
            raise ValueError("basis must be mass-orthonormal and COI orthogonal")
        if self.kappa.shape != (n - 1,) or not np.isfinite(self.kappa).all() or np.any(self.kappa < 0) or not np.isfinite(self.decay) or self.decay <= 0:
            raise ValueError("closure gains must be nonnegative and decay positive")
        self.active = np.flatnonzero(self.kappa > 0)
        self.port = self.mass[:, None] * self.basis[:, self.active] * np.sqrt(self.kappa[self.active])[None, :]
        for value in (self.mass, self.basis, self.kappa, self.active, self.port):
            value.flags.writeable = False

    @property
    def n_states(self) -> int:
        return len(self.active)

    def field(self, omega: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega, state = np.asarray(omega), np.asarray(state)
        if omega.shape != self.mass.shape or state.shape != (self.n_states,):
            raise ValueError("closure state or speed shape mismatch")
        return self.port @ state, -self.decay * state - self.port.T @ omega

    @staticmethod
    def storage(state: np.ndarray) -> float:
        return 0.5 * float(state @ state)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "kron_reduce.passive_pm_closure.v1", "mass": self.mass.tolist(),
            "basis": self.basis.tolist(), "kappa": self.kappa.tolist(), "decay": self.decay,
            "active_coordinates": self.active.tolist(), "initialization": "zero_increment_from_Pm_clear",
            "model_input": "predicted_speed_only", "level_6_claimed": False,
        }

    def save(self, path: Path) -> None:
        with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(self.as_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> PassivePmClosure:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(value["mass"], value["kappa"], value["decay"], basis=value["basis"])
        if value != model.as_dict():
            raise ValueError("closure file contradicts its equation/input contract")
        return model


def exponential_linear_filter(time: np.ndarray, values: np.ndarray, decay: float) -> np.ndarray:
    """Exact exponential convolution of linearly interpolated observations."""
    time, values = np.asarray(time, dtype=float), np.asarray(values, dtype=float)
    if time.ndim != 1 or values.ndim != 2 or len(time) != len(values) or not len(time) or time[0] != 0 or not np.isfinite(time).all() or not np.isfinite(values).all() or np.any(np.diff(time) <= 0) or not np.isfinite(decay) or decay <= 0:
        raise ValueError("invalid filter time, signal, or decay")
    filtered = np.zeros_like(values)
    for index, step in enumerate(np.diff(time)):
        x = decay * step
        integral = -np.expm1(-x) / decay
        linear = step * (0.5 - x / 6 + x*x / 24 - x*x*x / 120) if x < 1e-4 else (step - integral) / x
        filtered[index + 1] = np.exp(-x) * filtered[index] + integral * values[index] + linear * (values[index + 1] - values[index])
    if not np.isfinite(filtered).all():
        raise FloatingPointError("nonfinite filtered speed")
    return filtered


def fit_passive_pm_candidates(
    fit_samples: Sequence[dict[str, Any]],
    observations: PmObservationBatch,
    manifest: P1RoleManifest,
) -> tuple[tuple[PassivePmClosure, dict[str, Any]], ...]:
    validate_p1_samples(fit_samples, manifest, "fit")
    if observations.purpose != PmReadPurpose.FIT_CLOSURE_TARGET or observations.role != "fit" or observations.role_manifest_sha256 != manifest.source_sha256 or observations.event_ids != manifest.ids("fit") or set(observations.pm_by_event) != set(manifest.ids("fit")):
        raise ValueError("closure parameters require complete fit-only Pm targets")
    samples = sorted(fit_samples, key=lambda row: row["event_id"])
    mass = np.asarray(samples[0]["model_inputs"]["static"]["mass"])
    basis = mass_quotient_basis(mass)
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        metadata = sample["training_context"]["metadata"]
        groups[metadata["source_dataset"]][metadata["fault_line"]].append(sample["event_id"])
    weights = {event: 1 / len(groups) / len(lines) / len(events) for lines in groups.values() for events in lines.values() for event in events}
    candidates = []
    for decay in CLOSURE_DECAY_CANDIDATES:
        numerator, denominator = np.zeros(len(mass) - 1), np.zeros(len(mass) - 1)
        for sample in samples:
            static, target = sample["model_inputs"]["static"], sample["training_context"]["targets"]
            if not np.array_equal(static["mass"], mass):
                raise ValueError("one closure requires the same registered mass prior")
            pm = np.asarray(observations.pm_by_event[sample["event_id"]])
            clear = np.asarray(static["Pm_clear"]).reshape(-1)
            if pm.shape != np.shape(target["omega"]) or not np.isfinite(pm).all() or not np.allclose(pm[0], clear, rtol=0, atol=1e-10) or np.max(np.abs(pm.sum(axis=1))) > 1e-10:
                raise ValueError("invalid fit Pm target")
            velocity = (target["omega"] * mass) @ basis
            force = (pm - clear) @ basis
            filtered = exponential_linear_filter(target["t"], velocity, decay)
            weight = weights[sample["event_id"]]
            numerator += weight * np.mean(filtered * force, axis=0)
            denominator += weight * np.mean(filtered**2, axis=0)
        if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
            raise FloatingPointError("nonfinite closure regression")
        excited = denominator > 1e-14
        kappa = np.zeros(len(mass) - 1)
        kappa[excited] = np.maximum(0, -numerator[excited] / denominator[excited])
        closure = PassivePmClosure(mass, kappa, decay, basis=basis)
        candidates.append((closure, {
            "decay": decay, "fit_ids": list(manifest.ids("fit")), "role_manifest_sha256": manifest.source_sha256,
            "event_weights": weights, "excitation_denominator": denominator.tolist(),
            "unexcited_coordinates": np.flatnonzero(~excited).tolist(),
            "nonpassive_unconstrained_coordinates": np.flatnonzero(excited & (numerator > 0)).tolist(),
            "kappa": kappa.tolist(), "n_states": closure.n_states,
            "future_pm_used_as_training_target_only": True,
            "identifiability_claimed": False, "scientific_acceptance": False,
        }))
    return tuple(candidates)


class ClosureAugmentedModel:
    """The entire existing core and reciprocal closure evolve in one RK4 step."""

    def __init__(self, core: PrunableEnergyModel, closure: PassivePmClosure):
        core._require_fitted()
        self.core, self.closure = copy.deepcopy(core), closure
        self.max_internal_step = self.core.max_internal_step

    def _field(self, state: tuple[np.ndarray, ...], clear: np.ndarray, static: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
        delta, omega, hidden, z = state
        increment, zdot = self.closure.field(omega, z)
        dd, dw, dh = self.core._flat_field(delta, omega, hidden, clear + increment, static)
        return dd, dw, dh, zdot

    def rollout_augmented(self, static: dict[str, np.ndarray], time_grid: np.ndarray, exogenous: dict[str, Any], *, max_step: float | None = None) -> tuple[np.ndarray, ...]:
        allowed = {"delta0", "omega0", "Pm_clear", "Pm0", "equilibrium_delta", "Pm_equilibrium", "mass", "incidence"}
        if set(static) - allowed or set(exogenous) - {"Pm", "interpolation"} or exogenous.get("interpolation", "zoh") != "zoh":
            raise ValueError("closure rollout accepts only the causal clearing-state contract")
        mass = np.asarray(static["mass"])
        time, pm = np.asarray(time_grid), np.asarray(exogenous["Pm"])
        clear = np.asarray(static["Pm_clear"]).reshape(-1)
        step_limit = self.max_internal_step if max_step is None else float(max_step)
        if not np.array_equal(mass, self.closure.mass) or clear.shape != mass.shape or not np.isfinite(clear).all() or abs(float(clear.sum())) > 1e-10:
            raise ValueError("closure mass or clearing power mismatch")
        if time.ndim != 1 or not len(time) or time[0] != 0 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0) or not np.isfinite(step_limit) or step_limit <= 0:
            raise ValueError("invalid integration grid or step")
        if pm.shape != (len(time), len(mass)) or not np.array_equal(pm, np.broadcast_to(clear, pm.shape)):
            raise ValueError("future observed Pm is forbidden in the causal closure rollout")
        delta = coi_project(np.asarray(static["delta0"]).reshape(-1), mass)
        omega = coi_project(np.asarray(static["omega0"]).reshape(-1), mass)
        hidden = self.core._pack_hidden(self.core.initial_hidden(omega, static))
        state = (delta, omega, hidden, np.zeros(self.closure.n_states))
        if not all(np.isfinite(value).all() for value in state):
            raise ValueError("nonfinite clearing state")
        outputs = [[value.copy()] for value in state]
        for interval in np.diff(time):
            substeps = max(1, int(np.ceil(interval / step_limit)))
            step = interval / substeps
            for _ in range(substeps):
                k1 = self._field(state, clear, static)
                k2 = self._field(tuple(value + step * slope / 2 for value, slope in zip(state, k1, strict=True)), clear, static)
                k3 = self._field(tuple(value + step * slope / 2 for value, slope in zip(state, k2, strict=True)), clear, static)
                k4 = self._field(tuple(value + step * slope for value, slope in zip(state, k3, strict=True)), clear, static)
                state = tuple(value + step * (a + 2*b + 2*c + d) / 6 for value, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True))
                state = (coi_project(state[0], mass), coi_project(state[1], mass), state[2], state[3])
                if not all(np.isfinite(value).all() for value in state):
                    raise FloatingPointError("nonfinite coupled core/closure rollout")
            for output, value in zip(outputs, state, strict=True):
                output.append(value.copy())
        arrays = tuple(np.stack(output) for output in outputs)
        return (*arrays, clear[None, :] + arrays[3] @ self.closure.port.T)

    def rollout(self, static: dict[str, np.ndarray], time_grid: np.ndarray, exogenous: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        delta, omega, _hidden, _z, _pm = self.rollout_augmented(static, time_grid, exogenous)
        return delta, omega

    def storage(self, delta: np.ndarray, omega: np.ndarray, hidden: np.ndarray, z: np.ndarray, static: dict[str, np.ndarray]) -> float:
        return self.core.storage(delta, omega, static, hidden, pm_clear=np.asarray(static["Pm_clear"]).reshape(-1)) + self.closure.storage(z)

    def supply_identity_error(self, static: dict[str, np.ndarray], state: tuple[np.ndarray, ...], *, step: float = 1e-6) -> float:
        clear = np.asarray(static["Pm_clear"]).reshape(-1)
        derivative = self._field(state, clear, static)
        plus = tuple(value + step * slope for value, slope in zip(state, derivative, strict=True))
        minus = tuple(value - step * slope for value, slope in zip(state, derivative, strict=True))
        numerical = (self.storage(*plus, static) - self.storage(*minus, static)) / (2 * step)
        delta, omega, hidden, z = state
        expected = -self.core.dissipation_rate(delta, omega, hidden, static) - self.closure.decay * float(z @ z)
        return abs(numerical - expected)
